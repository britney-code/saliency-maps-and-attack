import torch
import torch.nn as nn
from Normalize import Normalize
import torchvision

class ifgsm:
    def __init__(self,
                 steps,
                 eps,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]
    ):
        self.steps = steps
        self.device = device
        self.eps = eps
        self.trans = Normalize(mean, std)
        self.alpha = self.eps / self.steps 

  

    def attack(self, model, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()

        for i in range(self.steps):
            adv.requires_grad = True
            logits = model(self.trans(adv))
            ce_loss = nn.CrossEntropyLoss()  # loss = F.nll_loss(logits, labels)
            loss = ce_loss(logits, labels)
            loss.backward()
            adv = adv + self.alpha * torch.sign(adv.grad)
            model.zero_grad()
            diff = adv - images
            delta = torch.clamp(diff, -self.eps, self.eps)
            adv = torch.clip(delta + images, 0, 1).detach_()
        return adv
