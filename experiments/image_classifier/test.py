import torch
import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(models.LeNet5()))
print(count_parameters(models.AlexNet()))
print(count_parameters(models.VGG16()))