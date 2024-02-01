import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
from utils import Normalize


# TIDIM
class tifgsm:
    r"""
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
    Examples::
        >>> attack = tifgsm(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7)
    """

    def __init__(
            self,
            model,
            device=None,
            eps=16 / 255,
            alpha=1.6 / 255,
            steps=10,
            decay=1.0,
            kernel_name='gaussian',
            len_kernel=7,
            nsig=3,
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
    ):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.model = model
        self.device = device
        self.seed_torch(1234)
        self.trans = Normalize(mean=mean, std=std)

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

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.trans(adv_images))
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(np.linspace((-kernlen + 1) / 2,
                                        (kernlen - 1) / 2, kernlen) / (kernlen + 1) * 2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
