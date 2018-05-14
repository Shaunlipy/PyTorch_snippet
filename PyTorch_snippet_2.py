import torch
import torchvision.models as models

# For vgg

model = models.vgg19(pretrained = True)
model.classifier._modules['6'] = torch.nn.Linear(4096, 4)
