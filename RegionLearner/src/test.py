import torch
import argparse
import configparser
import os

from glob import glob

from model import get_model
from utils import read_txt
from metrics import top_N_accuracy     

def parse_parameters():
    """Function to parse parameters.
    
    Each parameter has a default value, then it can be set from a config file or/and the command line.

    Returns:
        list: the list of parameters parsed.
    """    
    config_path = "config/config.cfg"

    path_to_dataset = "datasets/KITTI/09/train/"
    images_folder = "images/"
    dataset_file = "dataset.txt"
    centroids_file = "centroids.txt"
    graph_file = "graph.txt"

    model_path = "datasets/KITTI/09/kitti_09.pt"
    model_path_serialized = "datasets/KITTI/09/kitti_09_serialized.pt"
    image_width = 224
    image_height = 224
    
    path_to_test_dataset="D:/UNI/datasets/OpenLoris/corridor1-1_5-package/corridor1-2/color/"
    path_to_test="datasets/OpenLoris/corridor/test/"
    train_correspondences_file="corridor2_train_correspondences.txt"
    test_on_training_set = False
    use_best_model = True
    top_accuracy = [1, 3]
    use_exponential_moving_average = True
    alpha = 0.9

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)
    parser.add_argument('--path-to-dataset', type=str)
    parser.add_argument('--images-folder', type=str)
    parser.add_argument('--dataset-file', type=str)
    parser.add_argument('--centroids-file', type=str)
    parser.add_argument('--graph-file', type=str)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--model-path-serialized', type=str)
    parser.add_argument('--image-width', type=int)
    parser.add_argument('--image-height', type=int)
    parser.add_argument('--path-to-test-dataset', type=str)
    parser.add_argument('--path-to-test', type=str)
    parser.add_argument('--train-correspondences-file', type=str)
    parser.add_argument('--test-on-training-set', type=int)
    parser.add_argument('--use-best-model', type=int)
    parser.add_argument('--top-accuracy', type=list)
    parser.add_argument('--use-exponential-moving-average', type=int)
    parser.add_argument('--alpha', type=float)

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
        model_path = config["nn"]["save_model_path"] if "save_model_path" in config["nn"] else model_path
        model_path_serialized = config["nn"]["save_model_path_serialized"] if "save_model_path_serialized" in config["nn"] else model_path_serialized
        image_width = int(config["nn"]["image_width"]) if "image_width" in config["nn"] else image_width
        image_height = int(config["nn"]["image_height"]) if "image_height" in config["nn"] else image_height
        
    if "test" in config:
        test_on_training_set = config["test"]["test_on_training_set"] =="true" if "test_on_training_set" in config["test"] else test_on_training_set
        if not test_on_training_set:
            path_to_test_dataset = config["test"]["path_to_test_dataset"] if "path_to_test_dataset" in config["test"] else path_to_test_dataset
            path_to_test = config["test"]["path_to_test"] if "path_to_test" in config["test"] else path_to_test
            train_correspondences_file = config["test"]["train_correspondences_file"] if "train_correspondences_file" in config["test"] else train_correspondences_file
        use_best_model = config["test"]["use_best_model"] == "true" if "use_best_model" in config["test"] else use_best_model
        top_accuracy = config["test"]["top_accuracy"].split(",") if "top_accuracy" in config["test"] else top_accuracy
        use_exponential_moving_average = config["test"]["use_exponential_moving_average"] == "true" if "use_exponential_moving_average" in config["test"] else use_exponential_moving_average
        alpha = config["test"]["alpha"] == "true" if "alpha" in config["test"] else alpha

    path_to_dataset = args.path_to_dataset if args.path_to_dataset else path_to_dataset
    images_folder = args.images_folder if args.images_folder else images_folder
    dataset_file = args.dataset_file if args.dataset_file else dataset_file
    centroids_file = args.centroids_file if args.centroids_file else centroids_file
    graph_file = args.graph_file if args.graph_file else graph_file

    model_path = args.model_path if args.model_path else model_path
    model_path_serialized = args.model_path_serialized if args.model_path_serialized else model_path_serialized
    image_width = args.image_width if args.image_width else image_width
    image_height = args.image_height if args.image_height else image_height
    
    path_to_test_dataset = args.path_to_test_dataset if args.path_to_test_dataset else path_to_test_dataset
    path_to_test = args.path_to_test if args.path_to_test else path_to_test
    train_correspondences_file = args.train_correspondences_file if args.train_correspondences_file else train_correspondences_file
    test_on_training_set = args.test_on_training_set if args.test_on_training_set else test_on_training_set
    use_best_model = args.use_best_model if args.use_best_model else use_best_model
    top_accuracy = args.top_accuracy if args.top_accuracy else top_accuracy
    use_exponential_moving_average = args.use_exponential_moving_average if args.use_exponential_moving_average else use_exponential_moving_average
    alpha = args.alpha if args.alpha else alpha   
    
    return path_to_dataset, images_folder, dataset_file, centroids_file, graph_file, model_path, model_path_serialized, \
        image_width, image_height, path_to_test_dataset, path_to_test, train_correspondences_file, test_on_training_set, \
        use_best_model ,top_accuracy, use_exponential_moving_average, alpha   
        
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    path_to_dataset, images_folder, dataset_file, centroids_file, graph_file, model_path, model_path_serialized, \
    image_width, image_height, path_to_test_dataset, path_to_test, train_correspondences_file, test_on_training_set, \
    use_best_model ,top_accuracy, use_exponential_moving_average, alpha = parse_parameters()

    images_paths = [os.path.normpath(path) for path in glob(path_to_test_dataset + "*")]
    train_images_regions = read_txt(path_to_dataset + dataset_file, "\t")
    train_centroids = read_txt(path_to_dataset + centroids_file, "\t")
    
    if test_on_training_set:
        x = train_images_regions[:, 0]
        y = train_images_regions[:, 1].astype(int)
        x = [path_to_dataset + images_folder + name for name in x] 
    
    else:
        train_correspondences = read_txt(path_to_test + train_correspondences_file, "\t")
        y = []
        x = []
        for i in range(len(train_correspondences)):
            start = images_paths.index(os.path.normpath(path_to_test_dataset + train_correspondences[i][2]))
            end = images_paths.index(os.path.normpath(path_to_test_dataset + train_correspondences[i][3]))
            for j in range(start, end):
                y.append(train_correspondences[i][1])
                x.append(images_paths[j])  
        
    total_regions = len(train_centroids)
    
    if use_best_model:
        model_name = model_path_serialized.split("/")[-1].split(".")[0]
        model_path = os.path.join(*model_path_serialized.split("/")[:-1]) 
        model_path_serialized = os.path.join(model_path, "best_" + model_name + ".pt")
    model = torch.jit.load(model_path_serialized).to(device)
    
    model.eval()
 
    accuracy = top_N_accuracy(x, y, model, [1, 3], image_height, image_width, device, use_exponential_moving_average, alpha)
    print(accuracy)


    