import argparse
import configparser
import sys

from datasets.KITTI_dataset import KITTIDataset
from datasets.OpenLoris_dataset import OpenLorisDataset
from datasets.TUM_dataset import TUMDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        prog='LoopClosureInspector', 
        description='Python tool for loop closure ground truth labeling.')

    # Config File
    parser.add_argument(
        '--cfg', help='Configuration file path', required=True)

    args = parser.parse_args()
    
    cfg = configparser.ConfigParser()
    cfg.read(args.cfg)
    return cfg

def dataset_factory(use):
    if use == "KITTI":
        dataset = KITTIDataset()
      
    elif use == "OPENLORIS":
        dataset = OpenLorisDataset()

    elif use == "TUM":
        dataset = TUMDataset()
    
    return dataset

if __name__ == '__main__':
    cfg = parse_arguments(sys.argv)

    #read dataset
    dataset = dataset_factory(cfg["settings"]["use"])
    input = eval(cfg["settings"]["input_poses"])
    labels = eval(cfg["settings"]["labels"])
    sequences_len = []
    if type(input) == list:
        poses = pd.DataFrame()
        for path in input:
            seq = dataset.read_file(path)
            sequences_len.append(len(seq))
            poses = pd.concat([poses, seq])
    else:
        poses = dataset.read_file(input)
        
    print(labels)

    #get translations to draw
    translation_axis = cfg[cfg["settings"]["use"]]["translation_axis"].split(",")
    translations = dataset.get_translations(poses, translation_axis)

    #read pairs
    pairs = pd.read_csv(cfg["settings"]["input_pairs"], sep=",").to_numpy()
    

    #set indices to draw
    size = len(poses)
    loop_gt = np.zeros((size, size))

    for pair in pairs:
        if pair[0] != pair[1]:
            loop_gt[pair[0], pair[1]] = 1

    x = translations[:, [0]]
    y = translations[:, [1]]
    
    colors = [None, "tomato"]

    #plot the poses
    start = 0
    for i in range(len(sequences_len)):
        plt.plot(x[start:start+sequences_len[i]], y[start:start+sequences_len[i]], c=colors[i], label=labels[i])
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        start += sequences_len[i]

    #plot the correspondences
    indices, _ = np.where(loop_gt > 0)
    x_coord = x[indices]
    y_coord = y[indices]
    plt.scatter(x_coord, y_coord, c="#51f00e", label="matches")
    plt.legend()
    plt.savefig("img.png")