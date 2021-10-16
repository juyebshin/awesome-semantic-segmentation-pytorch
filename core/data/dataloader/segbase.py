"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask): # transform for train data
        # random mirror
        # if random.random() < 0.5: # randomly flip image?
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        # default base_size: 520
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # random int in range [260, 1040]
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
            # resize the image keeping the ratio h:w 
        else: # this is the usual case
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        # default crop_size: 480
        if short_size < crop_size:
            # pad if rescaled size is smaller than crop_size
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            # print('img size: {sz}'.format(sz=img.size))
            # print('mask size: {sz}'.format(sz=mask.size))
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=20, fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size) # w - 480
        y1 = random.randint(0, h - crop_size) # h - 480
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform: from PIL Image to numpy array
        img, mask = self._img_transform(img), self._mask_transform(mask)
        # print('_sync_transform mask size: {sz}'.format(sz=mask.size))
        # print('_sync_transform img size: {sz}'.format(sz=img.size))
        # print(np.unique(mask, return_counts= True))
        # here img, mask becomes 1d array?
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int64')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
