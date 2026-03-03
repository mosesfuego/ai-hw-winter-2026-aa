import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    perturbed = images + epsilon * grad.sign()
    return torch.clamp(perturbed, 0, 1).detach()