
import torch


# PSNR  # 上升
def PSNR(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=False)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# MSE # 下降
def MSE(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=False)


if __name__ == '__main__':
    from pprint import pprint
    from piq import psnr, ssim, vif_p
    from sewar.full_ref import uqi

    # PSNR反映了原始图像和对抗图像之间像素值的差异。PSNR值越高，图像质量越好,  PSNR # 上升
    original_images = torch.rand(10, 3, 299, 299)
    adversarial_images = torch.rand(10, 3, 299, 299)
    output = psnr(original_images, adversarial_images, data_range=1.0, reduction="none")
    pprint(output)
    # VIF 量化了与相应的干净图像相比，对抗性示例保留重要视觉信息的程度, VIF值越高，图像质量越好。  # VIF # 上升
    output = vif_p(original_images, adversarial_images,data_range=1.0, reduction="none")
    pprint(output)
    # SSIM 量化了对抗性示例与相应的干净图像之间的结构相似性，SSIM值越高，图像质量越好。 # SSIM # 上升
    output = ssim(original_images, adversarial_images,data_range=1.0,reduction="none")
    pprint(output)
    # MSE 揭露了对抗扰动的大小
    mse = MSE(original_images, adversarial_images)  # MSE # 下降
    pprint(mse)
    # UQI是一种通用的图像质量衡量标准，可从多个角度提供更全面的评估。 # 不支持batchsize
    output = uqi(original_images.numpy().transpose(0, 2, 3, 1), adversarial_images.numpy().transpose(0, 2, 3, 1))
    pprint(output)

