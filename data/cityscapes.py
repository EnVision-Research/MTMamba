import os
import json
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation
import torch.utils.data as data
from PIL import Image, ImageDraw, ImageFont
import imageio, cv2
import torch
import copy
import torchvision.transforms as t_transforms
import pickle

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def imresize(img, size, mode, resample):
    size = (size[1], size[0]) # width, height
    _img = Image.fromarray(img)#, mode=mode)
    _img = _img.resize(size, resample)
    _img = np.array(_img)
    return _img

class CITYSCAPES(data.Dataset):
    def __init__(self, p, root, split=["train"], is_transform=False,
                 img_size=[1024, 2048], augmentations=None, 
                 task_list=['semseg', 'depth'], ignore_index=255):

        if isinstance(split, str):
            split = [split]
        else:
            split.sort()
            split = split

        self.split = split
        self.root = root
        self.split_text = '+'.join(split)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = img_size 

        self.task_flags = {'semseg': True, 'insseg': False, 'depth': True}
        self.task_list = task_list
        self.files = {}

        self.files[self.split_text] = []
        for _split in self.split:
            self.images_base = os.path.join(self.root, 'leftImg8bit', _split)
            self.annotations_base = os.path.join(self.root, 'gtFine', _split)
            self.files[self.split_text] += recursive_glob(rootdir=self.images_base, suffix='.png')
            self.depth_base = os.path.join(self.root, 'disparity',  _split)
        ori_img_no = len(self.files[self.split_text])

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = ignore_index
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.ori_img_size = [1024, 2048]
        self.label_dw_ratio = img_size[0] / self.ori_img_size[0] # hacking

        if len(self.files[self.split_text]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split_text, self.images_base))

        # image to tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_transform = t_transforms.Compose([t_transforms.ToTensor(), t_transforms.Normalize(mean, std)])


    def __len__(self):
        return len(self.files[self.split_text])

    def __getitem__(self, index):
        
        img_path = self.files[self.split_text][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        depth_path = os.path.join(self.depth_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'disparity.png')  
                            

        img = cv2.imread(img_path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = {'image': img}
        sample['meta'] = {'img_name': img_path.split('.')[0].split('/')[-1],
                        'img_size': (img.shape[0], img.shape[1]),
                        'scale_factor': np.array([self.img_size[1]/img.shape[1], self.img_size[0]/img.shape[0]]), # in xy order
                        }

        if 'semseg' in self.task_list:
            lbl = imageio.imread(lbl_path)
            sample['semseg'] = lbl

        if 'depth' in self.task_list:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) # disparity

            depth[depth>0] = (depth[depth>0] - 1) / 256 # disparity values

            # make the invalid idx to -1
            depth[depth==0] = -1

            # assign the disparity of sky to zero
            sky_mask = lbl == 10
            depth[sky_mask] = 0

            sample['depth'] = depth

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        if 'semseg' in self.task_list:
            sample['semseg'] = self.encode_segmap(sample['semseg'])
        
        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def transform(self, sample):
        img = sample['image']
        if 'semseg' in self.task_list:
            lbl = sample['semseg']
        if 'depth' in self.task_list:
            depth = sample['depth']

        img_ori_shape = img.shape[:2]
        img = img.astype(np.uint8)

        if self.img_size != self.ori_img_size:
            img = imresize(img, (self.img_size[0], self.img_size[1]), 'RGB', Image.BILINEAR)

        if 'semseg' in self.task_list:
            classes = np.unique(lbl)
            lbl = lbl.astype(int)

        if 'depth' in self.task_list:
            depth = np.expand_dims(depth, axis=0)
            depth = torch.from_numpy(depth).float()
            sample['depth'] = depth

        if 'semseg' in self.task_list:
            if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
                print('after det', classes,  np.unique(lbl))
                raise ValueError("Segmentation map contained invalid class values")
            lbl = torch.from_numpy(lbl).long()
            sample['semseg'] = lbl

        img = self.img_transform(img)
        sample['image'] = img

        return sample

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        old_mask = mask.copy()
        for _validc in self.valid_classes:
            mask[old_mask==_validc] = self.class_map[_validc] 
        return mask

class ComposeAug(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        sample['image'], sample['semseg'], sample['depth'] =  np.array(sample['image'], dtype=np.uint8), np.array(sample['semseg'], dtype=np.uint8), np.array(sample['depth'], dtype=np.float32)
        sample['image'], sample['semseg'], sample['depth'] = Image.fromarray(sample['image'], mode='RGB'), Image.fromarray(sample['semseg'], mode='L'), Image.fromarray(sample['depth'], mode='F')
        if 'insseg' in sample.keys():
            sample['insseg'] = np.array(sample['insseg'], dtype=np.int32)
            sample['insseg'] = Image.fromarray(sample['insseg'], mode='I')

        assert sample['image'].size == sample['semseg'].size
        assert sample['image'].size == sample['depth'].size
        if 'insseg' in sample.keys():
            assert sample['image'].size == sample['insseg'].size

        for a in self.augmentations:
            sample = a(sample)

        sample['image'] = np.array(sample['image'])
        sample['semseg'] = np.array(sample['semseg'], dtype=np.uint8)
        sample['depth'] = np.array(sample['depth'], dtype=np.float32)
        if 'insseg' in sample.keys():
            sample['insseg'] = np.array(sample['insseg'], dtype=np.uint64)

        return sample