import numpy as np
import torch
import torch.nn.functional as F
from utils import Normalize
from fft.dct import dct_2d,idct_2d


class SSA:
    "Spectrum Simulation attack (ECCV'2022 ORAL)"

    def __init__(
            self,
            eps=16 / 255,
            iters=10,
            ens=20,
            sigma=16 / 255,
            u=1.0,
            rho=0.5,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        self.eps = eps
        self.iters = iters
        self.ens = ens
        self.sigma = sigma
        self.u = u
        self.rho = rho
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = eps / iters
        self.trans = Normalize(mean=mean, std=std)

    def clip_by_tensor(self, t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def maps(self):
        """saliency maps in frencymaps"""
        """grad_all = 0
        for images, images_ID, gt_cpu in tqdm(data_loader):
            gt = gt_cpu.cuda()
            images = images.cuda()
            img_dct = dct.dct_2d(images)
            img_dct = V(img_dct, requires_grad=True)
            img_idct = dct.idct_2d(img_dct)

            output_ = model(img_idct)
            loss = F.cross_entropy(output, gt)
            loss.backward()
            grad = img_dct.grad.data
            grad = grad.mean(dim=1).abs().sum(dim=0).cpu().numpy()
            grad_all = grad_all + grad

        x = grad_all / 1000.0
        x = (x - x.min()) / (x.max() - x.min())
        g1 = sns.heatmap(x, cmap="rainbow")
        g1.set(yticklabels=[])  # remove the tick labels
        g1.set(ylabel=None)  # remove the axis label
        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel=None)  # remove the axis label
        g1.tick_params(left=False)
        g1.tick_params(bottom=False)
        sns.despine(left=True, bottom=True)
        plt.show()
        plt.savefig("fig.png")"""

    def __call__(self, model, inputs, labels, *args, **kwargs):
        adv = inputs.clone().detach().to(self.device)
        inputs_min = self.clip_by_tensor(inputs - self.eps, 0.0, 1.0)
        inputs_max = self.clip_by_tensor(inputs + self.eps, 0.0, 1.0)
        for l in range(self.iters):
            noise = 0
            for x in range(self.ens):
                gauss = torch.randn(inputs.shape) * self.sigma
                gauss = gauss.to(self.device)
                dct = dct_2d(adv + gauss).to(self.device)
                mask = (torch.rand_like(adv) * 2 * self.rho + 1 - self.rho).to(self.device)
                idct = idct_2d(dct * mask)
                idct.requires_grad = True
                logits = model(self.trans(idct))
                model.zero_grad()
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                noise += idct.grad.data
            noise = noise / self.ens
            adv = adv + self.alpha * noise.sign()
            adv = self.clip_by_tensor(adv, inputs_min, inputs_max)
        return adv.clone().detach()
