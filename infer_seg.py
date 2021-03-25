import numpy as np
import torch
from torch.backends import cudnn

cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import tool.voc_cmap as voc_cmap
import torch.nn.functional as F
import os.path
import cv2


def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = crf_inference_inf(ori_img, bgcam_score, labels=21)

    return crf_score


def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83 / scale_factor, srgb=5, rgbim=np.copy(img_c), compat=4)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_seg", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--database', default='voc', type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--output_crf", default=None, type=str)

    args = parser.parse_args()
    model = getattr(importlib.import_module(args.network), 'SegNet')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    if args.database == 'voc':
        infer_dataset = voc12.data.SegmentationDatasetMSF(args.infer_list, image_dir=args.image_dir,
                                                          scales=[1, 0.5, 0.75, 1.25, 1.5],
                                                          inter_transform=torchvision.transforms.Compose(
                                                              [np.asarray,
                                                               model.normalize,
                                                               imutils.HWC_to_CHW]))

        infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        n_gpus = torch.cuda.device_count()
        model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
        cmap = voc_cmap.get_cmap()
        pred_softmax = torch.nn.Softmax(dim=0)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]
        label = label[0]

        img_path = os.path.join(args.image_dir, img_name + '.jpg')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    seg = model_replicas[i % n_gpus](img.cuda())
                    # seg = F.relu(seg, inplace=True)
                    seg = F.interpolate(seg, orig_img_size, mode='bilinear', align_corners=False)[0]
                    # seg = seg.cpu().numpy() * label.clone().view(21, 1, 1).numpy()
                    seg = seg.cpu().numpy()
                    if i % 2 == 1:
                        seg = np.flip(seg, axis=-1)
                    return seg


        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        seg_list = thread_pool.pop_results()

        sum_seg = np.sum(seg_list, axis=0)

        norm_seg = sum_seg * 0.1

        if args.output is not None:
            if not os.path.exists(args.mask_dir):
                os.makedirs(args.mask_dir)
            pred = np.argmax(norm_seg, 0)
            out_img = np.uint8(pred)
            out_img = Image.fromarray(out_img)
            out_img.putpalette(cmap)
            out_img.save(os.path.join(args.mask_dir, img_name + '.png'))

        if args.output_crf is not None:
            if not os.path.exists(args.output_crf):
                os.makedirs(args.output_crf)

            pred = torch.from_numpy(norm_seg)
            pred_prob = pred_softmax(pred)

            img_temp = cv2.imread(os.path.join(args.image_dir, img_name + '.jpg'))
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
            img_original = img_temp.astype(np.uint8)

            crf_la = _crf_with_alpha(pred_prob, img_original)

            crf_img = np.argmax(crf_la, 0)

            out_img = np.uint8(crf_img)

            out_img = Image.fromarray(out_img)
            out_img.putpalette(cmap)
            out_img.save(os.path.join(args.output_crf, img_name + '.png'))

        print(iter)
