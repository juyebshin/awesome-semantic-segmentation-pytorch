import os
import numpy as np
import cv2
import torch

from PIL import Image
from collections import namedtuple
from .segbase import SegmentationDataset
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label.
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #           name     id trainId      category  catId hasInstances ignoreInEval            color rgb
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ), # skyblue
    Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ), # purple
    Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255,   0,   0) ), # red
    Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ), # blue
    Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ), # u-turn
    Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ), # thru & u-turn
    Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ), # purple
    Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
# color to label
color2label     = { label.color   : label for label in labels}


from torchvision.transforms import Compose, ToTensor
class LabelDistanceTransform:
    def __init__(self, num_classes, bins=(4, 16, 64, 128), alphas=(8., 6., 4., 2., 1.), reduce=False,
                 ignore_id=19):
        self.num_classes = num_classes
        self.reduce = reduce
        self.bins = bins
        self.alphas = alphas
        self.ignore_id = ignore_id

    def __call__(self, example):
        labels = example.astype('uint8')
        # for k in mapId2TrainId:
        #   labels[example == k] = mapId2TrainId[k]
        present_classes = np.unique(labels)
        # print('labels size: {}'.format(labels.shape))
        # print('present_classes: {}'.format(present_classes))
        # distances: [num_class, batch_size, h, w]
        distances = np.zeros([self.num_classes] + list(labels.shape), dtype=np.float32) - 1.
        # print('distances shape: {}'.format(distances.shape))
        for i in range(self.num_classes):
            if i not in present_classes:
                continue
            class_mask = labels == i
            # print('class mask size: {}'.format(class_mask.shape))
            # print('distances[i][class_mask]: {}'.format(distances[i][class_mask]))
            distances[i][class_mask] = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
            
        if self.reduce:
            ignore_mask = labels == self.ignore_id
            distances[distances < 0] = 0
            distances = distances.sum(axis=0)
            label_distance_bins = np.digitize(distances, self.bins)
            label_distance_alphas = np.zeros(label_distance_bins.shape, dtype=np.float32)
            for idx, alpha in enumerate(self.alphas):
                label_distance_alphas[label_distance_bins == idx] = alpha
            label_distance_alphas[ignore_mask] = 0
            distance_alphas = label_distance_alphas
        else:
            distance_alphas = distances
        return distance_alphas

class ApolloSegmentation(SegmentationDataset):
    """ApolloScape Semantic Road Marking Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './list/' './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = '/home/ubuntu/VDC/Dataset/ApolloScape/LaneSegmentation'
    NUM_CLASS = 36



    def __init__(self, root='./list', split='train', mode=None, transform=None, **kwargs):
        super(ApolloSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        with open(os.path.join(root, split + '.txt'), "r", encoding='utf-8-sig') as f:
            self.img_list = []
            self.label_list = []
            # for overfitting with 100 images
            iter = 0
            for line in f:
                line = line.replace('/home/houyuenan/ApolloScapes', self.BASE_DIR)
                _img = []
                _img.append(line.split()[0])
                _img.append(line.split()[1])
                _img = ' '.join(_img)
                _label = []
                _label.append(line.split()[2])
                _label.append(line.split()[3])
                _label = ' '.join(_label)
                # print(_img)
                assert os.path.isfile(_img)
                assert os.path.isfile(_label)
                self.img_list.append(_img)
                self.label_list.append(_label)
                # iter += 1
                # if iter == 10:
                #     break
        print('Found {} images for {}'.format(len(self.img_list), split))
        self.cnt = 1

        # dir to save input and target image
        self.color_dir = './runs/pred_pic/bisenet_best_resnet18_apollos/color/'
        self.label_dir = './runs/pred_pic/bisenet_best_resnet18_apollos/label/'
        if not os.path.exists(self.color_dir):
                os.makedirs(self.color_dir)
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

        # save image list as csv for evaluation
        # if self.mode == 'testval':
        #     print('saving image list...')
        #     import csv
        #     with open('img_list.csv', 'w') as f:
        #         write = csv.writer(f)
        #         for img in self.img_list:
        #             img = os.path.splitext(os.path.split(img)[-1])[0] + '.png'
        #             # print(img)
        #             write.writerow([img])
            
        #     with open('label_list.csv', 'w') as f:
        #         write = csv.writer(f)
        #         for label in self.label_list:
        #             # save relative path
        #             label = os.path.relpath(label, start=self.BASE_DIR)
        #             print(label)
        #             write.writerow([label])

        # for boundary aware loss
        self.dist_transform = Compose(
        [LabelDistanceTransform(num_classes=self.NUM_CLASS, bins=(8, 16, 32), alphas=(8., 4., 2., 1.), reduce=True, ignore_id=255),
        ToTensor(),])

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        img = img.crop((0, 1700, img.size[0], img.size[1]))
        # img.save(os.path.join(self.color_dir, os.path.split(self.img_list[index])[-1]))
        # print('__getitem__ mode: {mode}'.format(mode=self.mode))
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.img_list[index])
        mask = Image.open(self.label_list[index])
        mask = mask.crop((0, 1700, mask.size[0], mask.size[1]))
        # mask_save = mask.convert('RGB')
        # mask_save.save(os.path.join(self.label_dir, os.path.split(self.label_list[index])[-1]))
        # print('cropped mask size: {sz}'.format(sz=mask.size))
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        # mask: type int64
        if self.dilate:
            org = mask.astype('uint8')
            mask = mask.astype('uint8')
            mask = cv2.dilate(mask, kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 3)), iterations=1)
            mask = mask.astype('int64')
            if self.cnt == 0:
                print('saving dilated sample')
                cv2.imwrite('original.png', org)
                cv2.imwrite('dilated.png', mask)
                self.cnt = 1
        
        # 2021-12-03 todo: compute distance weights alpha here
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        if self.bloss:
            dist_alphas = self.dist_transform(mask)
            return img, mask, dist_alphas, os.path.basename(self.img_list[index])
        # histo = np.unique(mask)
        # # print('unique mask: {un}'.format(un=histo))
        # for x in histo:
        #     mask[mask == x] = id2label[x].trainId
        # input_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        # ])
        # print('__getitem__ mask size: {sz}'.format(sz=mask.shape))
        # print('__getitem__ img size: {sz}'.format(sz=img.cpu().shape))
        # assert img.size == mask.size
        return img, mask, os.path.basename(self.img_list[index])

    def __len__(self):
        return len(self.img_list)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64')
        histo = np.unique(target)
        for x in histo:
            target[target == x] = id2label[x].trainId
        return target

    @property
    def pred_offset(self):
        return 0


if __name__ == '__main__':
    dataset = ApolloSegmentation()