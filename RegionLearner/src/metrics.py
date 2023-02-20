import torch
import numpy as np
import torchvision
from torchvision.io import read_image

def top_N_accuracy(x, y, model, n=3, reduce="mean", image_height=224, image_width=224, device="cuda"):
    if type(n) is list or type(n) is tuple:
        top_n_accuracy = np.zeros(len(n))
    else:
        top_n_accuracy = np.zeros(1)
    for i in range(len(x)):
        image_tensor = torch.unsqueeze((torchvision.transforms.Resize((image_height, image_width))(read_image(x[i])) / 255).to(device), 0)
        output = model(image_tensor).squeeze().detach().cpu().numpy()
        sorted_indices = np.argsort(-output)
        
        for j in range(len(top_n_accuracy)):
            if y[i] in sorted_indices[:n[j]]:
                top_n_accuracy[j] += 1
      
    if reduce == "mean":
        top_n_accuracy = top_n_accuracy / len(x)      
    return top_n_accuracy

""" # x = (N, C, H, W), y = (N, Z) 
def top_N_error_batched(x, y, model, n=3, reduce="mean", image_height=224, image_width=224, device="cuda"):
    if type(n) is list or type(n) is tuple:
        top_n_error = np.zeros(len(n))
    else:
        top_n_error = np.zeros(1)
        
    output = model(x).squeeze(0).detach().cpu().numpy()
    sorted_indices = np.argsort(-output)
    for j in range(len(top_n_error)):
        if np.isin(sorted_indices[..., :n[j]]):
            top_n_error[j] += 1 """