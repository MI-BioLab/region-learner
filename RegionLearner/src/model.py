from torch import nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_model(total_regions):
    """Function to build the model.

    Args:
        total_regions (int): the total number of regions.

    Returns:
        torch.tensor: the deep neural network.
    """    
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, total_regions),
            nn.Sigmoid())
        
    return model