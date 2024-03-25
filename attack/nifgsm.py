import torch
import torch.nn as nn
from Normalize import Normalize


class nifgsm:
    def __init__(
            self,
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
            decay=1.0,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.trans = Normalize(mean=mean, std=std)

    def forward(self, model, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum
            outputs = model(self.trans(nes_images))
            # Calculate loss
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad = self.decay * momentum + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images
