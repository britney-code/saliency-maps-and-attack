import numpy as np
import torch
from Normalize import Normalize
from torch.nn import functional as F
__all__ = ['MIG']


class MIG:
    def __init__(
            self,
            model,
            eps: float = 16 / 255,
            ens: int = 20,
            alpha: float = 1.6 / 255,
            iters: int = 10,
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            decay=1,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        """MIG
        >>> attack = MIG(...)
        >>> adv = attack(...)
        """
        self.model = model
        self.decay = decay
        self.eps = eps
        self.ens = ens
        self.alpha = alpha
        self.iters = iters
        self.device = device
        self.trans = Normalize(mean=mean, std=std)

    def clip_by_tensor(self, t, t_min, t_max):
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    
    def transform(self, data, **kwargs):
        x_base = torch.zeros_like(data).to(self.device)
        return torch.cat([x_base + i/self.ens * (data - x_base) for i in range(0, self.ens + 1)], dim=0)
    
    def get_loss(self, logits, label):
        loss = torch.mean(logits.gather(1, label.view(-1, 1)))   # torch.gather(input, dim, index,)
        return loss 
    
    def get_grad(self, loss, adv, **kwargs):
        return torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
    
    def get_momentum(self, grad, momentum, **kwargs):
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))
    
    def get_logits(self, x, **kwargs):
        return self.model(self.trans(x))

    def __call__(self, images, labels, *args, **kwargs):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        x_base = torch.zeros_like(adv).to(self.device)
        momentum = torch.zeros_like(images)
        images_min = self.clip_by_tensor(images - self.eps, t_min=0, t_max=1)
        images_max = self.clip_by_tensor(images + self.eps, t_min=0, t_max=1)
        for i in range(self.iters):
            adv.requires_grad_(True)
            logits = self.get_logits(self.transform(adv))
            logits = F.softmax(logits, dim = 1)
            loss = self.get_loss(logits, labels.repeat(self.ens + 1))
            grad = self.get_grad(loss, adv)
            IG = (adv - x_base) * grad / self.ens 
            momentum = self.get_momentum(IG, momentum=momentum)
            adv = adv + self.alpha * torch.sign(momentum)
            adv = self.clip_by_tensor(adv, images_min, images_max)
        return adv
