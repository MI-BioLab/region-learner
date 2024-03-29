import matplotlib.pyplot as plt
import numpy as np
import argparse
import configparser

from utils import read_txt

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
    position_color="blue"
    position_dim=80
    position_annotation="robot"
    circle_dim=10
    annotate_regions=False
    x_label = "x (m)"
    y_label = "y (m)"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)
    parser.add_argument('--path-to-dataset', type=str)
    parser.add_argument('--images-folder', type=str)
    parser.add_argument('--dataset-file', type=str)
    parser.add_argument('--centroids-file', type=str)
    parser.add_argument('--graph-file', type=str)
    parser.add_argument('--position-color', type=str)
    parser.add_argument('--position-dim', type=int)
    parser.add_argument('--position-annotation', type=str)
    parser.add_argument('--circle-dim', type=int)
    parser.add_argument('--annotate-regions', type=int)
    parser.add_argument('--x-label', type=str)
    parser.add_argument('--y-label', type=str)

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

    if "visualizer" in config:
        position_color = config["visualizer"]["position_color"] if "position_color" in config["visualizer"] else position_color
        position_dim = int(config["visualizer"]["position_dim"]) if "position_dim" in config["visualizer"] else position_dim
        position_annotation = config["visualizer"]["position_annotation"] if "position_annotation" in config["visualizer"] else position_annotation
        circle_dim = int(config["visualizer"]["circle_dim"]) if "circle_dim" in config["visualizer"] else circle_dim 
        annotate_regions = config["visualizer"]["annotate_regions"] == "true" if "annotate_regions" in config["visualizer"] else annotate_regions 
        x_label = config["visualizer"]["x_label"] if "x_label" in config["visualizer"] else x_label 
        y_label = config["visualizer"]["y_label"] if "y_label" in config["visualizer"] else y_label 

    path_to_dataset = args.path_to_dataset if args.path_to_dataset else path_to_dataset
    images_folder = args.images_folder if args.images_folder else images_folder
    dataset_file = args.dataset_file if args.dataset_file else dataset_file
    centroids_file = args.centroids_file if args.centroids_file else centroids_file
    graph_file = args.graph_file if args.graph_file else graph_file

    position_color = args.position_color if args.position_color else position_color
    position_dim = args.position_dim if args.position_dim else position_dim
    position_annotation = args.position_annotation if args.position_annotation else position_annotation
    circle_dim = args.circle_dim if args.circle_dim else circle_dim
    annotate_regions = args.annotate_regions if args.annotate_regions else annotate_regions
    x_label = args.x_label if args.x_label else x_label
    y_label = args.y_label if args.y_label else y_label
    
    return path_to_dataset, images_folder, dataset_file, centroids_file, graph_file, position_color, position_dim, \
        position_annotation, circle_dim, annotate_regions, x_label, y_label

def draw_centroids(x, y, regions, colors=[], 
                    position=None, position_color="blue", position_dim=80, position_annotation="robot", x_label="x (m)", y_label="y (m)"):
    """Function to draw the centroids.

    Args:
        x (ndarray): the x coordinates of the centroids of the training sequence.
        y (ndarray): the y coordinates of the centroids of the training sequence.
        regions (ndarray): the corresponding regions.
        colors (list, optional): colors to use to draw the centroids. Defaults to [].
        position (tuple or list, optional): the position of the robot. Defaults to None.
        position_color (str, optional): the color for the position. Defaults to "blue".
        position_dim (int, optional): the radius of the circle representing the position. Defaults to 80.
        position_annotation (str, optional): the annotation to be written next to the position. Defaults to "robot".
        x_label (str, optional): the label to display for axis x. Defaults to "x (m)".
        y_label (str, optional): the label to display for axis y. Defaults to "y (m)".
        
    Raises:
        Exception: regions and x must be the same length.
        Exception: regions and y must be the same length.
        Exception: x and y must be the same length.
        Exception: x_test and y_test must be the same length.
    """
    if len(regions) != len(x):
        raise Exception(f"draw_centroids: regions and x must be the same length! Got {len(regions)} and {len(x)}")
    if len(regions) != len(y):
        raise Exception(f"draw_centroids: regions and y must be the same length! Got {len(regions)} and {len(y)}")
    if len(x) != len(y):
        raise Exception(f"draw_centroids: x and y must be the same length! Got {len(x)} and {len(y)}")
    fig, ax = plt.subplots()
    
    if len(colors) == 0:
        colors = "red"
    elif len(colors) < len(regions):
        diff = len(regions) - len(colors)
        for i in range(diff):
            colors.append(colors[-1])
    ax.scatter(x, y, c = colors)

    if position:
        ax.scatter(position[0], position[1], c = position_color, s = position_dim)
        ax.annotate(position_annotation, (position[0], position[1]))

    ax.plot(x, y, color = "black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for i in regions:
        ax.annotate(str(i), (x[i], y[i]))
        
    plt.show()

def draw_regions(x, y, regions, total_regions, colors=[], circle_dim=10, annotate_regions=False, x_label="x (m)", y_label="y (m)"):
    """Function to draw the graph colored with a different color for of each region.

    Args:
       x (ndarray): the x coordinates of the nodes of the training sequence.
        y (ndarray): the y coordinates of the nodes of the training sequence.
        regions (ndarray): the corresponding regions.
        total_regions (int): the total number of regions.
        colors (list, optional): colors to use to draw the nodes. Defaults to [].
        circle_dim (int, optional): the radius of the circle representing the nodes. Defaults to 10.
        annotate_regions (bool, optional): whether annotate the regions. Defaults to False.
        x_label (str, optional): the label to display for axis x. Defaults to "x (m)".
        y_label (str, optional): the label to display for axis y. Defaults to "y (m)".

    Raises:
        Exception: colors must be a list.
        Exception: regions and x must be the same length.
        Exception: regions and y must be the same length.
        Exception: x and y must be the same length.
        Exception: colors, x and y must be the same length, or colors length must be equal to total_regions.
    """    
    total_colors = []
    fig, ax = plt.subplots()
    if type(colors) is not list:
        raise Exception(f"draw_regions: colors must be a list! Got {type(colors)}")
    if len(regions) != len(x):
        raise Exception(f"draw_regions: regions and x must be the same length! Got {len(regions)} and {len(x)}")
    if len(regions) != len(y):
        raise Exception(f"draw_regions: regions and y must be the same length! Got {len(regions)} and {len(y)}")
    if len(x) != len(y):
        raise Exception(f"draw_regions: x and y must be the same length! Got {len(x)} and {len(y)}")
    if len(colors) > 0 and len(colors) != len(x) and len(colors) != total_regions:
        raise Exception(f"draw_regions: colors, x and y must be the same length, or colors length must be equal to total_regions! Got len(x)={len(x)}, total_regions={total_regions}, and len(colors)={len(colors)}")
    
    if len(colors) == total_regions:
        for j in regions:
            total_colors.append(colors[j])
    elif len(colors) == 0:
        for i in range(total_regions):
            color = np.random.uniform(0.4, 1.0, 3)
            for j in regions:
                if j == i:
                    total_colors.append(color)
    elif len(colors) == regions:
        total_colors = colors

    if annotate_regions:
        for i in range(total_regions):
            for j in range(len(regions)):
                if i == regions[j]:
                    ax.annotate(str(i), (x[j], y[j]))
                    break

    ax.scatter(x, y, c = total_colors, s = circle_dim)    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__ == "__main__":
    path_to_dataset, images_folder, dataset_file, centroids_file, graph_file, position_color, position_dim, \
    position_annotation, circle_dim, annotate_regions, x_label, y_label = parse_parameters()

    images_regions = read_txt(path_to_dataset + dataset_file, "\t")
    centroids = read_txt(path_to_dataset + centroids_file, "\t")
    graph = read_txt(path_to_dataset + graph_file, "\t")

    total_regions = len(centroids)

    draw_centroids(centroids[:, 0].astype(float), centroids[:, 1].astype(float), centroids[:, 2].astype(int), 
                    position_color=position_color, position_annotation=position_annotation, 
                    position_dim=position_dim, x_label=x_label, y_label=y_label) 

    draw_regions(graph[:, 0].astype(float), graph[:, 1].astype(float), graph[:, 2].astype(int), total_regions, 
                circle_dim=int(circle_dim), annotate_regions=annotate_regions, x_label=x_label, y_label=y_label)