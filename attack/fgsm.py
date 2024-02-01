import torch
import torch.nn as nn
from utils import Normalize


class fgsm:
    def __init__(
            self, model, eps=8 / 255, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ):
        self.eps = eps
        self.model = model
        self.device = device
        self.trans = Normalize(mean=mean, std=std)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.model(self.trans(images))
        cost = loss(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
