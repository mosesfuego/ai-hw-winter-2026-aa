import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    ori_images = images.clone().detach()
    images = ori_images.clone()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        images = images + alpha * grad.sign()
        eta = torch.clamp(images - ori_images, -epsilon, epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images