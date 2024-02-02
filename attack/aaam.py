import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor
import random
from utils import Normalize

__all__ = ['AAAM']


class aaam:
    def __init__(
            self,
            eps: float = 16 / 255,
            eta: float = 7,
            alpha: float = 1.6 / 255,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
    ):
        self.eps = eps
        self.alpha = alpha
        self.device = device
        self.eta = eta
        self.criterion = nn.MSELoss(reduction='mean')
        self.seed_torch(1024)
        self.trans = Normalize(mean, std)

    def seed_torch(self, seed):
        """Set a random seed to ensure that the results are reproducible"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def LossCosine(self, model, images, frist, second) -> tensor:
        """
        args:
        I_y1: cam1 x, ytrue
        I_y2: cam2 x, ysec
        """
        cam1, cam2 = self.calculate_cam(model, images, frist, second)  # 不处理batchsize
        temp = cam1 * cam2
        norm = torch.norm(cam1) * torch.norm(cam2)
        L_cosine = torch.norm(temp / norm, p=1)
        L = -  torch.log((1 - L_cosine) / 2)  # normalize
        return L

    def Loss(self, model: nn.Module, x: tensor, y: tensor, frist, second) -> tensor:
        pred = model(self.trans(x))
        loss = F.cross_entropy(pred, y)  # PGD loss
        gamma = 1000
        loss = self.LossCosine(model, x, frist, second) - gamma * loss
        return loss

    def calculate_cam(self, model, images, frist, second):
        from saliency_maps.gradients import GradCamplusplus
        cam = GradCamplusplus(model)
        cam1 = cam.get_gradient(images, "layer4", frist)[0]
        cam2 = cam.get_gradient(images, "layer4", second)[0]
        # cam1 = F.interpolate(cam1,size=(images.shape[2], images.shape[3]),mode='bilinear', align_corners=False)
        # cam2 = F.interpolate(cam1,size=(images.shape[2], images.shape[3]),mode='bilinear', align_corners=False)
        return cam1, cam2

    def __call__(self, model, images, labels, clip_min=None, clip_max=None):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        n = 0
        b, c, h, w = images.size()
        if b != 1:
            assert "batchsize must be == 1!"

        first = labels
        logits = model(self.TNormalize(images))
        temp = logits.argmax(dim=1)
        logits[:, temp] = float("-inf")
        second = logits.argmax(dim=1)
        rmse = float("-inf")
        # 对于cam eta取多少合适？
        while rmse < self.eta:
            adv.requires_grad = True
            N = c * h * w
            loss = self.Loss(model, adv, labels, first, second)
            gt1 = torch.autograd.grad(loss, adv, retain_graph=False)[0]
            gt1 = (N * gt1) / torch.norm(gt1, p=1) + gt1 / torch.norm(gt1, p=2)
            gt1 = gt1 / 2
            # for i in range(4): SI 操作
            #     adv = adv / 2**i
            n = n + 1
            adv = adv - self.alpha * gt1
            delta = torch.clip(adv - images, -self.eps, self.eps)
            adv = torch.clip(images + delta, 0, 1).detach_()
            rmse = torch.sqrt(self.criterion(images, adv))
        return adv
