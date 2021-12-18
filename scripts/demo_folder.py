import os
import sys
import time
import glob
import argparse
import torch
import cv2
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc, pascal_aug, ade20k, citys', 'apollos'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./demo_pic', type=str,
                    help='path to save the predict result')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--img-glob', type=str, default='*.jpg',
                    help='Glob match if directory of images is specified (default: \'*.png\').')
parser.add_argument('--img-format', type=str, default='RGB',
                    choices=('RGB', 'RGGB'), help='image format')
parser.add_argument('--crop-size', type=int, default=1700,
                    help='x coordinate to crop (only bottom part is used)')
args = parser.parse_args()

class VideoStreamer(object):
    def __init__(self, basedir, img_glob, transform=None):
        self.listing = []
        self.i = 0
        self.skip = 1
        self.maxlen = 1000000
        self.transform = transform

        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')
    
    def read_image(self, impath, format='RGB'):
        if format == 'RGB':
            im = Image.open(impath).convert('RGB')
        elif format == 'RGGB':
            npimg = np.array(Image.open(impath).convert('L'))
            # Bayer BGGR to RGB
            npimg = cv2.cvtColor(npimg, code=cv2.COLOR_BAYER_RG2RGB)
            # npimg = cv2.resize(npimg, (1608, 480), interpolation=cv2.INTER_AREA)
            # h, w = npimg.shape
            # oh, ow = h//2, w//2

            # # raw uint8 samples Bayer RGGB
            # R  = npimg[0::2, 0::2] # row 1, 3, ... col 0, 2, ...
            # B  = npimg[1::2, 1::2] # row 0, 2, ... col 1, 3, ...
            # G0 = npimg[0::2, 1::2] # row 0, 2, ... col 0, 2, ...
            # G1 = npimg[1::2, 0::2] # row 1, 3, ... col 1, 3, ...

            # R = R[:oh, :ow]
            # B = B[:oh, :ow]
            # G = G0[:oh, :ow]//2 + G1[:oh, :ow]//2

            # npimg = np.dstack((B, G, R))
            im = Image.fromarray(npimg, 'RGB')
        if self.transform is not None:
            im = self.transform(im).unsqueeze(0)
        
        return im
    
    def next_frame(self, format='RGB'):
        if self.i == self.maxlen:
            return (None, None, False)
        
        image_file = self.listing[self.i]
        input_image = self.read_image(image_file, format=format)
        self.i = self.i + 1

        return (input_image, image_file, True)


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # image = Image.open(config.input_pic).convert('RGB')
    # image = image.crop((0, 1700, image.size[0], image.size[1]))
    # images = transform(image).unsqueeze(0).to(device)

    vs = VideoStreamer(config.input_pic, img_glob=config.img_glob)

    model = get_model(args.model, local_rank=args.local_rank, pretrained=True, root=args.save_folder).to(device)
    print('Finished loading model!')

    while True:
        image, file_name, status = vs.next_frame(format=args.img_format)
        if status is False:
            break
        
        image = image.crop((0, args.crop_size, image.size[0], image.size[1]))
        # if args.img_format == 'RGGB':
        #     image = image.resize((1608, 480))
        images = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(images)

        print(output[0].shape)
        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        print(np.unique(pred, return_counts=True))
        mask = get_color_pallete(pred, args.dataset)
        outname = os.path.splitext(os.path.split(file_name)[-1])[0] + '.png'
        # mask.save(os.path.join(args.outdir, outname))
        inname = os.path.splitext(os.path.split(file_name)[-1])[0] + '.jpg'
        mask_gray = mask.convert('L') # to grayscale mask
        image.paste(mask, (0, 0), mask=mask_gray)
        image.save(os.path.join(args.outdir, outname))
        image.show()


if __name__ == '__main__':
    demo(args)
