import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import configparser
from torch.utils.tensorboard import SummaryWriter
import os

from dataset import RegionDataset
from model import get_model
from loss import FocalLoss, compute_weights
from utils import read_txt

def train(train_dataloader, model, loss_fn, optimizer):
    train_loss = 0.0
    model.train()
    for batch, X in enumerate(train_dataloader):
        x, y = X
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= (batch + 1)
    return train_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    config_path = "config/config.cfg"

    #dataset
    path_to_dataset = "datasets/KITTI/09/train/"
    images_folder = "images/"
    dataset_file = "dataset.txt"
    centroids_file = "centroids.txt"
    graph_file = "graph.txt"

    #nn
    batch_size = 128
    image_width = 224
    image_height = 224
    beta = 0.999
    gamma = 2
    loss_reduction = "mean"
    epochs=50
    learning_rate = 1e-3
    use_augmentation = False
    brightness = (0.7, 1.3)
    contrast = (0.7, 1.3)
    saturation = (0.7, 1.3)
    hue = (-0.1, 0.1)
    random_perspective_distortion = 0.3
    random_perspective_p = 0.5
    random_rotation_degrees = (-10, 10)
    save_best_model= True
    save_model_path = "datasets/KITTI/09/kitti_09.pt"
    save_model_serialized = True
    serialize_on_gpu = False
    save_model_path_serialized = "datasets/KITTI/09/kitti_09_serialized.pt"

    #visualizer
    position_color="blue"
    position_dim=80
    position_annotation="robot"
    test_colors="bisque"
    circle_dim=10
    annotate_regions=False

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)
    parser.add_argument('--path-to-dataset', type=str)
    parser.add_argument('--images-folder', type=str)
    parser.add_argument('--dataset-file', type=str)
    parser.add_argument('--centroids-file', type=str)
    parser.add_argument('--graph-file', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--image-width', type=int)
    parser.add_argument('--image-height', type=int)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--loss-reduction', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--use-augmentation', type=int)
    parser.add_argument('--brightness', type=float)
    parser.add_argument('--contrast', type=float)
    parser.add_argument('--saturation', type=float)
    parser.add_argument('--hue', type=float)
    parser.add_argument('--random-perspective-distortion', type=float)
    parser.add_argument('--random-perspective-p', type=float)
    parser.add_argument('--random-rotation-degrees', type=tuple)
    parser.add_argument('--save-best-model', type=int)
    parser.add_argument('--save-model-path', type=str)
    parser.add_argument('--save-model-serialized', type=int)
    parser.add_argument('--serialize-on-gpu', type=int)
    parser.add_argument('--save-model-path-serialized', type=str)
    parser.add_argument('--position-color', type=str)
    parser.add_argument('--position-dim', type=int)
    parser.add_argument('--position-annotation', type=str)
    parser.add_argument('--test-colors', type=str)
    parser.add_argument('--circle-dim', type=int)
    parser.add_argument('--annotate-regions', type=int)

    args = parser.parse_args()

    config_path = args.config_path if args.config_path else config_path

    config = configparser.ConfigParser()
    config.read(config_path)

    if "dataset" in config:
        path_to_dataset = config["dataset"]["path_to_dataset"] if "path_to_dataset" in config["dataset"] else path_to_dataset
        images_folder = config["dataset"]["images_folder"] if "images_folder" in config["dataset"] else images_folder
        dataset_file = config["dataset"]["dataset_file"] if "dataset_file" in config["dataset"] else dataset_file
        centroids_file = config["dataset"]["centroids_file"] if "centroids_file" in config["dataset"] else centroids_file
        graph_file = config["dataset"]["graph_file"] if "graph_file" in config["dataset"] else graph_file

    if "nn" in config:
        batch_size = int(config["nn"]["batch_size"]) if "batch_size" in config["nn"] else batch_size
        image_width = int(config["nn"]["image_width"]) if "image_width" in config["nn"] else image_width
        image_height = int(config["nn"]["image_height"]) if "image_height" in config["nn"] else image_height
        beta = float(config["nn"]["beta"]) if "beta" in config["nn"] else beta
        gamma = int(config["nn"]["gamma"]) if "gamma" in config["nn"] else gamma
        loss_reduction = config["nn"]["loss_reduction"] if "loss_reduction" in config["nn"] else loss_reduction
        epochs = int(config["nn"]["epochs"]) if "epochs" in config["nn"] else epochs
        learning_rate = float(config["nn"]["learning_rate"]) if "learning_rate" in config["nn"] else learning_rate
        use_augmentation = config["nn"]["use_augmentation"] == "true" if "use_augmentation" in config["nn"] else use_augmentation
        brightness = eval(config["nn"]["brightness"]) if "brightness" in config["nn"] else brightness
        contrast = eval(config["nn"]["contrast"]) if "contrast" in config["nn"] else contrast
        saturation = eval(config["nn"]["saturation"]) if "saturation" in config["nn"] else saturation
        hue = eval(config["nn"]["hue"]) if "hue" in config["nn"] else hue
        random_perspective_distortion = float(config["nn"]["random_perspective_distortion"]) if "random_perspective_distortion" in config["nn"] else random_perspective_distortion
        random_perspective_p = float(config["nn"]["random_perspective_p"]) if "random_perspective_p" in config["nn"] else random_perspective_p
        random_rotation_degrees = eval(config["nn"]["random_rotation_degrees"]) if "random_rotation_degrees" in config["nn"] else random_rotation_degrees
        
        save_best_model = config["nn"]["save_best_model"] == "true" if "save_best_model" in config["nn"] else save_best_model
        save_model_path = config["nn"]["save_model_path"] if "save_model_path" in config["nn"] else save_model_path
        save_model_serialized = config["nn"]["save_model_serialized"] == "true" if "save_model_serialized" in config["nn"] else save_model_serialized
        serialize_on_gpu = config["nn"]["serialize_on_gpu"] == "true" if "serialize_on_gpu" in config["nn"] else serialize_on_gpu
        save_model_path_serialized = config["nn"]["save_model_path_serialized"] if "save_model_path_serialized" in config["nn"] else save_model_path_serialized

    if "visualizer" in config:
        position_color = config["visualizer"]["position_color"] if "position_color" in config["visualizer"] else position_color
        position_dim = int(config["visualizer"]["position_dim"]) if "position_dim" in config["visualizer"] else position_dim
        position_annotation = config["visualizer"]["position_annotation"] if "position_annotation" in config["visualizer"] else position_annotation
        test_colors = config["visualizer"]["test_colors"] if "test_colors" in config["visualizer"] else test_colors
        circle_dim = int(config["visualizer"]["circle_dim"]) if "circle_dim" in config["visualizer"] else circle_dim 
        annotate_regions = config["visualizer"]["annotate_regions"] == "true" if "annotate_regions" in config["visualizer"] else annotate_regions

    path_to_dataset = args.path_to_dataset if args.path_to_dataset else path_to_dataset
    images_folder = args.images_folder if args.images_folder else images_folder
    dataset_file = args.dataset_file if args.dataset_file else dataset_file
    centroids_file = args.centroids_file if args.centroids_file else centroids_file
    graph_file = args.graph_file if args.graph_file else graph_file

    batch_size = args.batch_size if args.batch_size else batch_size
    image_width = args.image_width if args.image_width else image_width
    image_height = args.image_height if args.image_height else image_height
    beta = args.beta if args.beta else beta
    gamma = args.gamma if args.gamma else gamma
    loss_reduction = args.loss_reduction if args.loss_reduction else loss_reduction
    epochs = args.epochs if args.epochs else epochs
    learning_rate = args.learning_rate if args.learning_rate else learning_rate
    use_augmentation = args.use_augmentation if args.use_augmentation else use_augmentation
    brightness = args.brightness if args.brightness else brightness
    contrast = args.contrast if args.contrast else contrast
    saturation = args.saturation if args.saturation else saturation
    hue = args.hue if args.hue else hue
    random_perspective_distortion = args.random_perspective_distortion if args.random_perspective_distortion else random_perspective_distortion
    random_perspective_p = args.random_perspective_p if args.random_perspective_p else random_perspective_p
    random_rotation_degrees = args.random_rotation_degrees if args.random_rotation_degrees else random_rotation_degrees
    save_best_model = args.save_best_model if args.save_best_model else save_best_model
    save_model_path = args.save_model_path if args.save_model_path else save_model_path
    save_model_serialized = args.save_model_serialized if args.save_model_serialized else save_model_serialized
    serialize_on_gpu = args.serialize_on_gpu if args.serialize_on_gpu else serialize_on_gpu
    save_model_path_serialized = args.save_model_path_serialized if args.save_model_path_serialized else save_model_path_serialized

    position_color = args.position_color if args.position_color else position_color
    position_dim = args.position_dim if args.position_dim else position_dim
    position_annotation = args.position_annotation if args.position_annotation else position_annotation
    test_colors = args.test_colors if args.test_colors else test_colors
    circle_dim = args.circle_dim if args.circle_dim else circle_dim
    annotate_regions = args.annotate_regions if args.annotate_regions else annotate_regions


    images_regions = read_txt(path_to_dataset + dataset_file, "\t")
    centroids = read_txt(path_to_dataset + centroids_file, "\t")
    total_regions = len(centroids)

    x = images_regions[:, 0]
    y = images_regions[:, 1].astype(int)
    x = [path_to_dataset + images_folder + name for name in x]    

    train_dataset = RegionDataset(x, y, total_regions, height=image_height, width=image_width, device=device, 
                                use_augmentation=use_augmentation, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                                random_perspective_distortion=random_perspective_distortion, random_perspective_p=random_perspective_p, random_rotation_degrees=random_rotation_degrees)
    train_weights = torch.tensor(compute_weights(y, total_regions, beta=beta)).to(device).float()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_loss_fn = FocalLoss(gamma, train_weights, reduction=loss_reduction)

    model = get_model(total_regions)
    model.to(device)

    
    lr = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    min_loss = np.inf
    n_epoch_since_last_best_loss = 0
    writer = SummaryWriter()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if(n_epoch_since_last_best_loss == 5):
            n_epoch_since_last_best_loss = 0
            for g in optimizer.param_groups:
                lr *= 0.3333
                g['lr'] = lr
                print(f"Learning rate decreased. New learning rate: {lr:.6f}")  
        if lr < 1e-5:
            break

        train_loss = train(train_dataloader, model, train_loss_fn, optimizer)

        if min_loss > train_loss:
                print(f'Loss Decreased({min_loss:.6f}--->{train_loss:.6f}) \t Saving The Model')
                min_loss = train_loss
                if save_best_model:
                    torch.save(model.state_dict(), save_model_path)
                n_epoch_since_last_best_loss = 0
        
        
        else:
            n_epoch_since_last_best_loss += 1
        
        

        print("Train loss: " + str(train_loss))
        writer.add_scalar('Loss/train', train_loss, t)
        writer.add_scalar('Learning Rate', lr, t)
    print("Done!") 
    
    if save_model_serialized:
        model_name = save_model_path_serialized.split("/")[-1].split(".")[0]
        model_path_serialized = os.path.join(*save_model_path_serialized.split("/")[:-1]) 
        if serialize_on_gpu:
            if torch.cuda.is_available():
                model.to("cuda")
                model_serialized = torch.jit.script(model)
                model_serialized.save(save_model_path_serialized)
                
                best_model = get_model(total_regions)
                best_model.load_state_dict(torch.load(save_model_path))
                best_model_serialized = torch.jit.script(best_model)                
            else: 
                model_serialized = torch.jit.script(model)
                
                best_model = get_model(total_regions)
                best_model.load_state_dict(torch.load(save_model_path))
                best_model_serialized = torch.jit.script(best_model)  
        else:
            model.to("cpu")
            model_serialized = torch.jit.script(model)
            
            best_model = get_model(total_regions)
            best_model.load_state_dict(torch.load(save_model_path))
            best_model_serialized = torch.jit.script(best_model)              
                
        model_serialized.save(save_model_path_serialized)
        best_model_serialized.save(os.path.join(model_path_serialized, "best_" + model_name + ".pt"))





    
    
    