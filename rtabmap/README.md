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
 - **InferenceMode** (*bool*), whether rtabmap must be launched in inference mode (default to false);
 - **UseXY** (*bool*), whether the 2D pose is computed using (x,y) coordinates. Coordinates (x,z) are used if false (default to false);
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
 
 
 
