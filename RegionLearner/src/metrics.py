import torch
import numpy as np
import torchvision
from torchvision.io import read_image
import copy

def exponential_moving_average(xt, prev_ema, alpha=0.9):
    """Function to calculate exponential moving average (EMA).
    
    EMAₜ = α ⋅ xₜ + (1 - α) ⋅ EMAₜ₋₁
    
    Args:
        xt (ndarray): the current value.
        prev_ema (ndarray): the EMA at the previous step.
        alpha (float, optional): parameter that defines how much the current value weighs against the previous EMA. Defaults to 0.9.

    Returns:
        ndarray: the array computed with the exponential moving average.
    """    
    return alpha * xt + (1-alpha) * prev_ema

def top_N_accuracy(x, y, model, n=1, image_height=224, image_width=224, device="cuda", use_exponential_moving_average=False, alpha=0.9):
    """Computes the top N accuracies.

    Args:
        x (list(str)): the paths of the images.
        y (list(int)): the corresponding labels for the regions.
        model (torch.tensor): the deep neural network.
        n (int, list(int), tuple(int), optional): the top N accuracies to compute (e.g. [1, 3] means top-1 and top-3 accuracies). Defaults to 1.
        image_height (int, optional): the height of the images. Defaults to 224.
        image_width (int, optional): the width of the images. Defaults to 224.
        device (str, optional): the device to use for the inference. Defaults to "cuda".
        use_exponential_moving_average (bool, optional): whether use the exponential moving average to compute the top-N accuracies. Defaults to False.
        alpha (float, optional): parameter used in the exponential moving average. Defaults to 0.9.

    Returns:
        ndarray: the top-N accuracies.
    """    
    if type(n) is list or type(n) is tuple:
        top_n_accuracy = np.zeros(len(n))
    else:
        top_n_accuracy = np.zeros(1)
    for i in range(len(x)):
        image_tensor = (torchvision.transforms.Resize((image_height, image_width))(read_image(x[i])) / 255).unsqueeze(0).to(device)
        output = model(image_tensor).squeeze().detach().cpu().numpy()
        if use_exponential_moving_average:
            if i != 0:
                output = exponential_moving_average(output, prev_ema, alpha)
            prev_ema = copy.deepcopy(output) 
        sorted_indices = np.argsort(-output)
        
        for j in range(len(top_n_accuracy)):
            if y[i] in sorted_indices[:n[j]]:
                top_n_accuracy[j] += 1
    
    top_n_accuracy = top_n_accuracy / len(x)
    return top_n_accuracy