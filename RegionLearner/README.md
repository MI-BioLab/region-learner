RegionLearner
====

RegionLearner represents the deep learning based part of the project. Here the deep neural network is trained to predict the regions probabilities.

# Requirements
This package is realized using python and PyTorch. We recommend to install conda environment and then install PyTorch inside the conda environment.
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Install [PyTorch](https://pytorch.org/get-started/locally/) in conda environment. We used PyTorch with cuda 11.6 for our experiments.
3. Install the additional requirements using the command ```pip install -r path/to/RegionLearner/requirements.txt```.

# Parameters
Inside the configuration file ```RegionLearner/config/config.cfg``` a lot of parameters can be specified in order to customize the algorithm.
The parameters logic works as follows: each parameter has a default value defined in the source files (where it is used). You can specify a different value either by using the configuration file or by passing arguments to the console when launching python files. Console arguments have the maximum "priority", followed by the configuration file and the default value at the end. This means that console arguments overwrite both the configuration file and the default value, while configuration file overwrites the default value.

Now take a look at the configuration file. It is divided into four sections, according to the cfg sintax: **dataset**, **nn**, **test** and **visualizer**.

## Dataset
The parameters for the dataset are specified in this section. As you read in rtabmap README.md, the dataset is created during the exploration phase. Once the images and files (dataset file, centroids file and graph file) from the exploration are saved, you should create a folder hierarchy as follows:
```
path_to_dataset
│   dataset.txt  
│   graph.txt
│   centroids.txt  
│
└───images
│    │   000001.png
│    │   000002.png
│    │   ...

```
So you should specify the path to the dataset, the name of the images directory and the names of the files in the corresponding parameters.

**Attention**: The slash at the end of the path for *path_to_dataset* and *images_folder* is necessary.

## NN
The parameters for the neural network are specified in this section. Most of them are self-explanatory, so they are ignored in the following. The noteworthy parameters are:
- *beta*, a parameter to weight imbalanced classes. This weighting mechanism is described in the paper [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/pdf/1901.05555.pdf). Default value is 0.999.
- *gamma*, a parameter to tune the Focal Loss. Default value is 2;
- *loss_reduction*, to define how to compute the reduction of the loss. It can be *mean*, or *sum*. Default value is *mean*;
- *use_augmentation*, to define whether to use data augmentation. Parameters below in the configuration file are related to this one. Default value is false.  
- *save_best_model*, to define whether to save the best model (the model at the epoch with the lowest loss value). If it's true, *save_model_path* must be specified.
- *save_model_serialized*, to define whether to save the model serialized (to embed it in rtabmap using libtorch). Default value is true. If it's true, *save_model_path_serialized* must be specified.
- *serialize_on_gpu*, to define whether the model should be serialized on gpu. Default value is false.

Our experiments were conducted without data augmentation.

## Test
The parameters for the test are defined here. The test is intended as the test on the network predictions (top-N accuracy), not as the test on the loop closure detection (which can only be performed using rtabmap). Parameters are: 
- *test_on_training_set* defines whether to test the algorithm on the training set, using the parameters specified in section *dataset*. This is for those datasets (e.g. KITTI) for which a test set does not exist. Default value is false.
- *path_to_test_images* in case a test set exists, this parameter specifies the path to the images of the test set. 
- *path_to_test* in case a test set exists, this parameter specifies the path to the test folder. 
- *train_correspondences_file*, a set of correspondences manually identified between training images and test images.
- *use_best_model* specifies whether to use the best model specified in *save_model_path* of section **nn** or to use the model saved at the last epoch.
- *top_n_accuracy* the top-n accuracy to use for the test. You can specify a single value (e.g. 1 means top-1 accuracy) or a list of values (e.g. [1, 3] means top-1 and top-3 accuracy), so you can compute more top-n accuracy in a single shot.
- *use_exponential_moving_average* specifies whether to use the exponential moving average during test. If true, you should specify the *alpha* value.

The folder structure for the test set should be as follows:
```
path_to_test
│   train_correspondences_file.txt  
│
└───images
│    │   000001.png
│    │   000002.png
│    │   ...

```

Parameter *path_to_test_images* was voluntarily written separately to the test path because for datasets where images are already present even outside the rosbags (e.g., OpenLoris) one can directly specify the path to those images, without the need to save them from the bags (e.g., path/to/OpenLoris/market1-1_3-package/market1-1/color/).

**Attention**: the slash at the end of the path is necessary.

## Visualizer 
The parameters for the visualizer are specified in this section. The parameters are:
- *position_color*, *position_dim* and *position_annotation*  allow to specificy how to draw the circle that represents the robot position on the map. The last one (*position_annotation*) is the annotation to write along the circle.
- *circle_dim* to specify the dimension of the circles that represent the nodes of the graph.
- *annotate_zones* to define whether the zones in the graph draw should be annotated.

**Attention**: graph file, centroids file and the other stuffs must be configured as described above.

If you desire to visualize the graph and the centroids of the test sequence, you need to put them inside the *path_to_test* folder and edit *path_to_dataset* in the **dataset** section.

# Noteworthy things
Noteworthy things are describer below, to provide a better understanding of some important features.

## Learning rate decay
Epochs and learning rate can be specified as parameters. The learning rate decay policy works as follow: the specified learning rate is the initial learning rate. If after 5 epochs the value returned by the loss function does not decrease, the learning rate is divided by a factor of 3. This continues until the learning rate reaches the minimum of $10^{-5}$, when the training stops. On the other side, if the number of epochs exceed that defined in the parameter, training is stopped. In our experiments we used $10^{-3}$ as initial learning rate and 100 as maximum epochs, but training always stopped before 100 epochs because of the decrease in the learning rate.

## Train-test correspondences files
Inside the *path_to_test* folder, you should have a training-test correspondences file (*train_correspondences_file*). It correlates the images between the training set and the test. Below an example of the content of the file is shown:

```
000001.png	0	000001.png	000037.png
000040.png	1	000037.png	000111.png
000122.png	2	000111.png	000160.png
000157.png	3	000160.png	000200.png
000211.png	4	000200.png	000245.png
```

The first column is the name of the image in the training set. The second one is the corresponding region, the third and fourth are the range of images in the test set corresponding to those in the same region of the training set.

In the example above, region 0 starts at image 000001.png and ends at image 000039.png in the training set (since region 1 starts at image 000040.png). In the test set the corresponding images manually identified starts at 000001.png and ends at 000037.png (excluded). So region 1 starts at image 000040.png and ends at image 000121.png in the training set (since region 2 starts at image 000122.png), while it starts at image 000037.png and ends at image 000111.png (excluded) and so on.