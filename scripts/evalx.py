from __future__ import print_function

import os
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=True,
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d, root=args.save_dir).to(self.device)
        # model = get_model(args.model, local_rank=args.local_rank, pretrained=True, root=args.save_folder).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        avg_mIoU = 0
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            num_valid = len(torch.unique(target))
            # print('input image shape: {}'.format(image.shape))

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU, IoU = self.metric.get(num_valid)
            avg_mIoU += mIoU
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
            np.set_printoptions(precision=2, suppress=True)
            logger.info("Class wise IoU: {}".format((IoU*100)))

            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                # print(np.unique(pred, return_counts=True))
                # print(np.unique(target.cpu().data.numpy(), return_counts=True))

                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
        avg_mIoU /= 1000
        print(avg_mIoU*100, "%")
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    args.model = 'bisenet_best'
    args.backbone = 'resnet18'
    args.dataset = 'apollos'
    args.log_dir = '~/runs/logs/dilate_bloss/'
    args.save_dir = '~/.torch/models/bloss/'
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    if not args.distributed or (args.distributed and args.local_rank % num_gpus == 0):
        args.save_pred = True
        if args.save_pred:
            outdir = args.log_dir
            outdir = outdir.replace('logs', 'pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset))
            # outdir = os.path.join(outdir, '{}_{}_{}'.format(args.model, args.backbone, args.dataset))
            print('predicted images are saved to: {}'.format(outdir))
            # outdir = './runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

    print('log file is saved to: {}'.format(args.log_dir))
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
