"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Juyeb Shin boundary loss 2021-12-02
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor

from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']



# TODO: optim function
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs) # preds: 1d tuple
        print('in loss.py preds shape: {len}'.format(len=[len(a) for a in preds]))
        # print('size preds: {}'.format(preds.size()))
        # print('in loss.py target shape: {len}'.format(len=[len(a) for a in target]))
        # print([len(a) for a in preds])
        # print([len(a) for a in target])
        inputs = tuple(list(preds) + [target])
        print('in crossentropy inputs length: {}'.format([len(a) for a in inputs]))
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            # same as nn.CrossEntropyLoss.forward
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)) 


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, nclass, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))


def get_segmentation_loss(model, use_ohem=False, bloss=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)
    
    if bloss:
        return BoundaryAwareFocalLoss(**kwargs)

    model = model.lower()
    if model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'icnet':
        return ICNetLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)




class BoundaryAwareFocalLoss(nn.Module):
    def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_each = print_each
        self.step_counter = 0
        self.gamma = gamma

    def forward(self, input, target, dist, **kwargs):
        input = torch.stack(list(input), dim=0)
        input = input.squeeze(0)
        # if input.shape[-2:] != target.shape[-2:]:
        #     input = upsample(input, target.shape[-2:])
        target[target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
        label_distance_alphas = dist.to(input.device)
        # print('label_distance size: {}'.format(label_distance_alphas.size()))
        N = (label_distance_alphas.data > 0.).sum()
        # print('input size: {}'.format(input.size()))
        # print('target size: {}'.format(target.size()))
        # print('dist size: {}'.format(dist.size()))
        if N.le(0):
            return torch.zeros(size=(0,), device=label_distance_alphas.device, requires_grad=True).sum()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        alphas = label_distance_alphas.view(-1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        loss = -1 * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
        loss = loss.sum() / N

        if (self.step_counter % self.print_each) == 0:
            print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1

        return loss


# class LabelDistanceTransform:
#     def __init__(self, num_classes, bins=(4, 16, 64, 128), alphas=(8., 6., 4., 2., 1.), reduce=False,
#                  ignore_id=19):
#         self.num_classes = num_classes
#         self.reduce = reduce
#         self.bins = bins
#         self.alphas = alphas
#         self.ignore_id = ignore_id

#     def __call__(self, example):
#         labels = np.array(example.cpu())
#         # for k in mapId2TrainId:
#         #   labels[example == k] = mapId2TrainId[k]
#         present_classes = np.unique(labels)
#         print('labels size: {}'.format(labels.shape))
#         print('present_classes: {}'.format(present_classes))
#         # distances: [num_class, batch_size, h, w]
#         distances = np.zeros([self.num_classes] + list(labels.shape), dtype=np.float32) - 1.
#         print('distances shape: {}'.format(distances.shape))
#         for i in range(self.num_classes):
#             if i not in present_classes:
#                 continue
#             class_mask = labels == i
#             print('class mask size: {}'.format(class_mask.shape))
#             # print('distances[i][class_mask]: {}'.format(distances[i][class_mask]))
#             distances[i][class_mask] = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
            
#         if self.reduce:
#             ignore_mask = labels == self.ignore_id
#             distances[distances < 0] = 0
#             distances = distances.sum(axis=0)
#             label_distance_bins = np.digitize(distances, self.bins)
#             label_distance_alphas = np.zeros(label_distance_bins.shape, dtype=np.float32)
#             for idx, alpha in enumerate(self.alphas):
#                 label_distance_alphas[label_distance_bins == idx] = alpha
#             label_distance_alphas[ignore_mask] = 0
#             distance_alphas = label_distance_alphas
#         else:
#             distance_alphas = distances
#         return distance_alphas

# class BoundaryAwareFocalLoss(nn.Module):
#     def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20):
#         super(BoundaryAwareFocalLoss, self).__init__()
#         self.num_classes = num_classes
#         self.ignore_id = ignore_id
#         self.print_each = print_each
#         self.step_counter = 0
#         self.gamma = gamma

#         self.transform = Compose(
#             [LabelDistanceTransform(num_classes=self.num_classes, bins=(8, 16, 32), alphas=(8., 4., 2., 1.), reduce=True, ignore_id=0),
#             ToTensor(),])

#     def forward(self, input, target, **kwargs):
#         # input: tuple (batch_size) of Tensor (image)
#         # if input.shape[-2:] != target.shape[-2:]:
#         #     input = upsample(input, target.shape[-2:])
#         _input = input[0].clone().detach()
#         losses = 0
#         dist_alphas = torch.zeros(input[0].size(),)
#         for b in range(target.shape[0]): # for batch_size
#             _target = torch.tensor(target[b].clone().detach())
#             _target[_target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
#             dist_alphas[b] = self.transform(_target)
#         # dist_alphas = self.transform(target)
#         label_distance_alphas = dist_alphas.to(_input.device)
#         print('label_distance size: {}'.format(label_distance_alphas.size()))
#         N = (label_distance_alphas.data > 0.).sum()
#         if N.le(0):
#             return torch.zeros(size=(0,), device=label_distance_alphas.device, requires_grad=True).sum()
#         if _input.dim() > 2:
#             _input = _input.view(_input.size(0), _input.size(1), -1)  # N,C,H,W => N,C,H*W
#             _input = _input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             _input = _input.contiguous().view(-1, _input.size(2))  # N,H*W,C => N*H*W,C
#         # one hot encoding
#         target = target.unsqueeze(1)
#         one_hot = torch.LongTensor(target.size(0), self.num_classes, target.size(2), target.size(3)).to(target.device).zero_()
#         target = one_hot.scatter_(1, target.data, 1)
#         print('target scatter_ size: {}'.format(target.size()))
#         if target.dim() > 2:
#             target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
#             target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C
#         # target = target.view(-1, 1)
#         alphas = label_distance_alphas.view(-1)

#         print('_input size: {}'.format(_input.size()))
#         print('target size: {}'.format(target.size()))
#         logpt = F.log_softmax(_input, dim=-1)
#         print('logpt size: {}'.format(logpt.size()))
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = logpt.detach().exp()

#         print('alphas size: {}'.format(alphas.size()))
#         loss = -1 * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
#         loss = loss.sum() / N
#             # losses += loss

#         if (self.step_counter % self.print_each) == 0:
#             print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
#         self.step_counter += 1

#         return loss