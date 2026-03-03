import torch
import torch.nn.functional as F

def mifgsm_attack(model, images, labels, epsilon, alpha, iters, mu=1.0):
    ori_images = images.clone().detach()
    images = ori_images.clone()
    g = torch.zeros_like(images)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        g = mu * g + grad
        images = images + alpha * g.sign()
        eta = torch.clamp(images - ori_images, -epsilon, epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images