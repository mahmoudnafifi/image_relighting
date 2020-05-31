from os.path import join
from os import listdir
from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageOps
import random


class DataLoading(Dataset):
    def __init__(self, imgs_dir, target_dir=None, gt_dir=None):
        self.patch_size = 256
        self.imgs_dir = imgs_dir
        self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir)
                         if not file.startswith('.')]
        self.target_dir = target_dir
        self.gt_dir = gt_dir
        if self.target_dir is not None:
            if self.gt_dir is None:
                raise Exception('No ground-truth dir is given')

            self.targetimgfiles = [join(target_dir, file) for file in listdir(target_dir)
                                   if not file.startswith('.')]
            #self.targetimgfiles = random.shuffle(self.targetimgfiles)
        if self.gt_dir is not None:
            if self.target_dir is not None:
                self.gtimgfiles = self.targetimgfiles
                # for i in range(len(self.imgfiles)):
                #     self.gtimgfiles[i] = self.targetimgfiles[i].replace(self.target_dir, self.gt_dir)
                #     _, tail = path.split(self.gtimgfiles[i])
                #     parts = tail.split('_')
                #     tg_lighting = parts[1] + '_' + parts[2]
                #     _, tail = path.split(self.imgfiles[i])
                #     parts = tail.split('_')
                #     base = parts[0]
                #     self.gtimgfiles[i] = join(self.gt_dir, base + '_' + tg_lighting)
            else:
                self.gtimgfiles = [join(gt_dir, file) for file in listdir(gt_dir)
                                   if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.imgfiles)} input examples')

    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, pil_img, patch_size=256, aug_op=0, patch_coords=None):
        if aug_op is 1:
            pil_img = ImageOps.mirror(pil_img)
        elif aug_op is 2:
            pil_img = ImageOps.flip(pil_img)
        elif aug_op is 3:
            pil_img.rotate(0, translate=(50, 50), expand=False)


        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        if patch_coords is not None:
            img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_nd = img_nd[:, :, :3]
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):

        img_file = self.imgfiles[i]
        in_img = Image.open(img_file)
        # get image size
        w, h = in_img.size
        if w > 512 or h > 512:
            in_img = in_img.resize((512, 512), Image.ANTIALIAS)
        w, h = in_img.size
        # get flipping option
        aug_op = np.random.randint(4)
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        in_img_patch = self.preprocess(in_img, self.patch_size, aug_op, patch_coords=(patch_x, patch_y))

        if self.target_dir is not None:
            j = random.randint(0, len(self.targetimgfiles)-1)
            target_imgfile = self.targetimgfiles[j]
            target_img = Image.open(target_imgfile)
            w_, h_ = target_img.size
            if w_ > 512 or h_ > 512:
                target_img = target_img.resize((512, 512), Image.ANTIALIAS)
            target_img_patch = self.preprocess(target_img, self.patch_size, aug_op, patch_coords=(patch_x, patch_y))
            if self.gt_dir is None:
                return {'input': torch.from_numpy(in_img_patch), 'target': torch.from_numpy(target_img_patch)}
            else:
                _, tail = path.split(target_imgfile)
                parts = tail.split('_')
                tg_lighting = parts[1] + '_' + parts[2]
                _, tail = path.split(img_file)
                parts = tail.split('_')
                base = parts[0]
                gt_imgfile = join(self.gt_dir, base + '_' + tg_lighting)
                #print(f'Input: {img_file}, target: {target_imgfile}, gt: {gt_imgfile}\n')
                gt_img = Image.open(gt_imgfile)
                w_, h_ = gt_img.size
                if w_ > 512 or h_ > 512:
                    gt_img = gt_img.resize((512, 512), Image.ANTIALIAS)
                gt_img_patch = self.preprocess(gt_img, self.patch_size, aug_op, patch_coords=(patch_x, patch_y))
                return {'input': torch.from_numpy(in_img_patch), 'target': torch.from_numpy(target_img_patch),
                        'gt': torch.from_numpy(gt_img_patch)}

        elif self.gt_dir is not None:
            gt_imgfile = self.gtimgfiles[i]
            gt_img = Image.open(gt_imgfile)
            w_, h_ = gt_img.size
            if w_ > 512 or h_ > 512:
                gt_img = gt_img.resize((512, 512), Image.ANTIALIAS)
            gt_img_patch = self.preprocess(gt_img, self.patch_size, aug_op, patch_coords=(patch_x, patch_y))
            return {'input': torch.from_numpy(in_img_patch), 'gt': torch.from_numpy(gt_img_patch)}
        else:
            return {'input': torch.from_numpy(in_img_patch)}


