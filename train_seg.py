import numpy as np
import torch
from torch.backends import cudnn
import torch.nn as nn

cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib
import torch.nn.functional as F

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--network", default="network.resnet38_seg", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--mask_dir", required=True, type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_seg", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument('--database', default='voc', type=str)
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'SegNet')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    if args.database == 'voc':
        train_dataset = voc12.data.SegmentationDataset(img_name_list_path=args.train_list, image_dir=args.image_dir,
                                                       mask_dir=args.mask_dir, rescale=[448, 768], flip=True, cropsize=448)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_seg"
        import network.resnet38d

        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):

        for iter, (name, image, mask) in enumerate(train_data_loader):

            inputs = image.cuda()
            target = mask.cuda()
            outputs = model(inputs)
            outputs = F.interpolate(outputs, (args.crop_size, args.crop_size), mode='bilinear', align_corners=False)
            loss = criterion(outputs, target)
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)

        else:
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')
