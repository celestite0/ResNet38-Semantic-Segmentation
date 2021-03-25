import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image
import os.path
import scipy.misc
import cv2
from PIL import Image
import math

from tool import imutils
import random

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('voc12/cls_labels.npy').item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

class SegmentationDataset(Dataset):
    def __init__(self, img_name_list_path, image_dir, mask_dir, rescale=None, flip=False, cropsize=None,
                 img_transform=None, mask_transform=None):
        self.img_name_list_path = img_name_list_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.img_name_list = load_img_name_list(self.img_name_list_path)
        self.rescale = rescale
        self.flip = flip
        self.cropsize = cropsize

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        image = Image.open(os.path.join(self.image_dir, name + '.jpg')).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + '.png'))

        if self.flip is True and bool(random.getrandbits(1)):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # rescale
        tol = imutils.RandomResizeImageAndMask(self.rescale[0], self.rescale[1])
        image, mask = tol(image, mask)

        # transform
        data_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = data_transforms(image)
        mask = torch.LongTensor(np.array(mask).astype(np.int64))

        # crop
        crop = [self.cropsize, self.cropsize]
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, crop[0] - h)
        pad_lr = max(0, crop[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image = image[:, i:i + crop[0], j:j + crop[1]]
        mask = mask[i:i + crop[0], j:j + crop[1]]

        return name, image, mask


class SegmentationDatasetMSF(Dataset):

    def __init__(self, img_name_list_path, image_dir, scales=None, inter_transform=None, unit=1):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.image_dir = image_dir
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = Image.open(os.path.join(self.image_dir, name + '.jpg')).convert("RGB")

        label = torch.from_numpy(self.label_list[idx])
        bg = torch.from_numpy(np.array([1.0]).astype(np.float32))
        label = torch.cat((bg, label))
        #label = torch.from_numpy(np.array([1.0]).astype(np.float32))

        rounded_size = (
            int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

