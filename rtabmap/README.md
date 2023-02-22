rtabmap
=======

Modified version of rtabmap. Please, refer to the original [rtabmap](https://github.com/introlab/rtabmap) for the installation.

# Requirements
In addition to the requirements in the original version, the following requirements must be satisfied:
- [LibTorch](https://pytorch.org/cppdocs/installing.html) (C++ PyTorch library);
- [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (personally I followed the point 1.3.2 Debian Local Installation of this [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)). Versions should be chosen according to your GPU (CUDA 11.6 and cuDNN 8.6.0 were used in the development of this package).

# Modifications
Several changes have been made w.r.t the original rtabmap package. They are described in detail below.

 ## Parameters
 In order to customize the modified version of rtabmap, several parameters have been added (see ```rtabmap/corelib/include/rtabmap/core/Parameters.h```). These new parameters have **Regions** as prefix and they are:
 - **ModelPath** (*string*), the path to the pre-trained model;
 - **UseGPU** (*bool*), whether use GPU for the model inference (default to false);
 - **TargetHeight** (*int*), target height for images to feed the neural net (default to 224);
 - **TargetWidth** (*int*), target width for images to feed the neural net (default to 224);
 - **UseExponentialMovingAverage** (*bool*), whether use exponential moving average for weighting regions (default to true);
 - **Alpha** (*float*), the weight for the exponential moving average (default to 0.9);
 - **InferenceMode** (*bool*), whether rtabmap must be launched in inference mode (default to false). If false, rtabmap will be in exploration mode, so the graph is clustered, and the files saved for the training;
 - **UseXY** (*bool*), whether the 2D pose is computed using (x,y) coordinates. Coordinates (x,z) are used if false (default to false). This parameter must be set depending on the frame of reference of the device or the dataset used (e.g. KITTI uses (x,z) coordinates, OpenLoris uses (x,y) coordinates);
 - **KRegionsRetrieved*** (*int*), the top-k regions to retrieve (default to 0);
 - **MaxNodesRetrieved*** (*int*), the max number of nodes to retrieve (default to 0);
 - **ImagesSaveFolder** (*string*), the folder in which save the images;
 - **DatasetSaveFile** (*string*), the file in which save the images paths and the regions assigned;
 - **GraphSaveFile** (*string*), the file in which save the poses and the regions assigned for valid signatures;
 - **CentroidsSaveFile** (*string*), the file in which save the centroids and the regions assigned;
 - **RadiusUpperBound** (*float*), the radius upper bound for a region (default to 80);
 - **DesiredAverageCardinality** (*int*), the desired average cardinality for a region (default to 100);
 - **MeshShapeFactor** (*float*), the mesh shape factor to tune clustering (default to 50);
 - **KeepPrefixPath** (*bool*), whether the entire path or just the name should be kept when saving the images for the dataset (default to false);
 - **NameTotalLength** (*int*), the total name length for the images. 0s are used to fill the name (i.e. the name of the images saved are 000001, 000002...) (default to 6). 
 
\* Parameters not used in the experiments and intended for future works.

To correctly launch rtabmap with this parameters you should write ```--Regions/{parameter name} {value}```. How to use them is explained in the following section.


## General operation
The modified version of rtabmap can be launched in two different modalities: **exploration mode** or **inference mode**.

### Exploration mode
It is intended as the exploration phase, in which the robot navigates a new environment, the graph is constructed and clustered and some important files are saved for the training. If rtabmap is launched with the parameter ```--Regions/InferenceMode false```, then it starts in exploration mode. Here, you should specify the parameters:
- **UseXY**, because the graph clustering algorithm must know which coordinates to use;
- **ImagesSaveFolder**, to save the images acquired during the exploration, that are necessary for training;
- **DatasetSaveFile**, to save the dataset. The dataset is a txt file that links the names of the images (or the entire path if **KeepPrefixPath** is true) and the corresponding regions. <br> For example, the content of the dataset file can be <br> 000001.png&emsp;0 <br>000002.png&emsp;0 <br>... <br>000200.png&emsp;5<br>one line for each image acquired;
- **GraphSaveFile**, to save the 2D positions (according to the parameter **UseXY**) of the nodes of the graph constructed by rtabmap with the corresponding regions in a txt file. <br> For example, the content of the graph file can be <br> 0&emsp;0&emsp;0 <br>0.2011&emsp;0.1187&emsp;0 <br>... <br>63.0569&emsp;98.2839&emsp;5<br>one line for each node in the graph;
- **CentroidsSaveFile**, to save the 2D positions (according to the parameter **UseXY**) of the centroids of the clustered graph and the corresponding regions in a txt file. <br> For example, the content of the centroids file can be <br>-1.22641&emsp;23.1104&emsp;0<br>0.227025&emsp;79.8067&emsp;1<br>...<br>69.0922&emsp;217.902&emsp;5<br>one line for each cluster;
- **RadiusUpperBound**, **DesiredAverageCardinality** and **MeshShapeFactor**, to tune the clustering algorithm;
- **KeepPrefixPath**, **NameTotalLength**, to define how to save the images names in the dataset file.

### Inference mode
This is the inference phase; after training on a particular environment the robot can predict the regions probabilities by observing the images it is acquiring. Here, you should specify the parameters:
- **ModelPath** because the inference mode requires the trained model;
 - **UseGPU** whether use GPU for the model inference. Since the model is very cheap, in our experiments the inference was always performed using CPU;
 - **TargetHeight** and **TargetWidth** to define the image size to feed the model;
 - **UseExponentialMovingAverage** and **Alpha** if you want to use the exponential moving average;


## Code
The classes for the graph clustering have been added to the code. Since for now the algorithm isn't able to reassign a node to a different cluster (because the neural network isn't trained in a continual learning fashion), the code is simplified to perform a static clustering. You can find the implementation in the folders ```rtabmap/corelib/include/rtabmap/core/graph_clustering``` and ```rtabmap/corelib/src/graph_clustering```, for headers and cpp respectively.

Code changes were made in the following files (both header and cpp):
- *Rtabmap*, which includes the main cycle of rtabmap. Here the graph is clustered and the file saved or the inference is performed, according to the current mode of rtabmap. Methods ```loadModel```, ```imageToTensor```, ```predict```, ```computeWeightedExponentialMovingAverage```, ```sortRegionsProbabilities```, ```saveRegionsDatasetAndGraph```, ```saveRegionsDataset```, ```saveRegionsGraph```, ```computeImageName``` were added;
- *Memory*, where methods ```forgetByRegion``` and ```reactivateSignaturesByRegion``` were added;
- *Signature*, where methods ```setRegion``` and ```getRegion``` were added;
- *DBDriver*, where the methods ```countRegions``` and ```loadSignaturesByRegion``` were added to load information from the DB, and methods ```countRegionsQuery``` and ```loadSignaturesByRegionQuery``` to specialize them inside *DBDriverSqlite3*.


To store the regions inside the database, the column ```region_id``` has been added to table ```Node``` in the database schemas in the ```rtabmap/corelib/src/resources``` (also in those contained in ```rtabmap/corelib/src/resources/backward_compatibility```).

The file ```CMakeLists.txt``` was modified to include LibTorch.
 
 
 
