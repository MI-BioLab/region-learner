import matplotlib.pyplot as plt
import numpy as np
import argparse
import configparser

from utils import read_txt

def draw_centroids(x, y, regions, colors=[], 
                    position=None, position_color="blue", position_dim=80, position_annotation="robot",
                    x_test=None, y_test=None, test_colors="bisque"):
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

    if x_test and y_test:
        if len(x_test) != len(y_test):
            raise Exception(f"draw_centroids: x_test and y_test must be the same length! Got {len(x_test)} and {len(y_test)}")
        if type(test_colors) is list:
            if len(test_colors) > 0 and len(test_colors) < len(x_test):
                diff = len(x_test) - len(test_colors)
                for i in range(diff):
                    test_colors.append(test_colors[-1])

        ax.scatter(x_test, y_test, c = test_colors)
        ax.plot(x_test, y_test, color = test_colors)

    if position:
        ax.scatter(position[0], position[1], c = position_color, s = position_dim)
        ax.annotate(position_annotation, (position[0], position[1]))

    ax.plot(x, y, color = "black")

    for i in regions:
        ax.annotate(str(i), (x[i], y[i]))
        
    plt.show()

def draw_regions(x, y, regions, total_regions, colors=[], circle_dim=10, annotate_regions=False):
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
            color = np.random.uniform(0.4, 1.0, 3,)
            print(color)
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
    plt.show()

if __name__ == "__main__":
    config_path = "config/config.cfg"

    #dataset
    path_to_dataset = "datasets/KITTI/09/train/"
    images_folder = "images/"
    dataset_file = "dataset.txt"
    centroids_file = "centroids.txt"
    graph_file = "graph.txt"

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

    position_color = args.position_color if args.position_color else position_color
    position_dim = args.position_dim if args.position_dim else position_dim
    position_annotation = args.position_annotation if args.position_annotation else position_annotation
    test_colors = args.test_colors if args.test_colors else test_colors
    circle_dim = args.circle_dim if args.circle_dim else circle_dim
    annotate_regions = args.annotate_regions if args.annotate_regions else annotate_regions

    images_regions = read_txt(path_to_dataset + dataset_file, "\t")
    centroids = read_txt(path_to_dataset + centroids_file, "\t")
    graph = read_txt(path_to_dataset + graph_file, "\t")

    total_regions = len(centroids)

    draw_centroids(centroids[:, 0].astype(float), centroids[:, 1].astype(float), centroids[:, 2].astype(int), 
                    position_color=position_color, position_annotation=position_annotation, 
                    position_dim=position_dim, test_colors=test_colors) 

    draw_regions(graph[:, 0].astype(float), graph[:, 1].astype(float), graph[:, 2].astype(int), total_regions, 
                circle_dim=int(circle_dim), annotate_regions=annotate_regions)