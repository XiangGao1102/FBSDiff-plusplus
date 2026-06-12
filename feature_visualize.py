from PIL import Image
import torch
import numpy as np

feature = torch.load('noise.pt').cuda()
feature = torch.mean(feature.squeeze().permute(1, 2, 0), dim=-1, keepdim=True)
feature = feature.expand(-1, -1, 3).view(-1, 3)
print(feature.shape)
feature = (feature - torch.min(feature, dim=0)[0]) / (torch.max(feature, dim=0)[0] - torch.min(feature, dim=0)[0])
feature = feature.view(64, 64, 3) * 255
Image.fromarray(feature.cpu().numpy().astype(np.uint8)).show()




