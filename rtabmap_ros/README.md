rtabmap_ros
=======

Modified version of the RTAB-Map's ROS package. Please, refer to the original version of [rtabmap_ros](https://github.com/introlab/rtabmap_ros) for the installation. For our experiments, ROS Melodic and Ubuntu 18.04 were used.

# Requirements
In addition to the requirements in the original version, the following requirements must be satisfied:
- [LibTorch](https://pytorch.org/cppdocs/installing.html) (C++ PyTorch library);
- [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (personally I followed the point 1.3.2 Debian Local Installation of this [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)). Versions should be chosen according to your GPU (CUDA 11.6 and cuDNN 8.6.0 were used in the development of this package).

# Modifications
Compared to the original rtabmap_ros, only the CMakeLists.txt file has been modified to include LibTorch and three launch files have been added:
- ```openloris.launch``` to launch rtabmap_ros with the appropriate parameters for the bags of the [OpenLoris-Scene dataset](https://lifelong-robotic-vision.github.io/dataset/scene.html);
- ```kitti.launch``` to launch rtabmap_ros with the appropriate parameters for the bags of the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (without velodyne);
- ```campus.launch``` to launch rtabmap_ros with the appropriate parameters for the bags of the Campus dataset (acquired in our Campus corridors).

# Explanation of the launch files
In the following, a part of the parameters used in ```rtabmap_ros/launch/kitti.launch``` is reported.
```
--Mem/IncrementalMemory true 
--Mem/InitWMWithAllNodes false
--RGBD/OptimizeMaxError 10 
--Rtabmap/MemoryThr 50 
--Rtabmap/LoopThr 0.05 
--Rtabmap/CreateIntermediateNodes true
--Rtabmap/DetectionRate 1
--Regions/ModelPath /home/matteo/models/best_kitti_00_serialized.pt 
--Regions/InferenceMode false
--Regions/UseXY false
--Regions/KRegionsRetrieved 0
--Regions/MaxNodesRetrieved 0
--Regions/ImagesSaveFolder /home/matteo/rtabmap_exploration/images/
--Regions/DatasetSaveFile /home/matteo/rtabmap_exploration/dataset.txt
--Regions/GraphSaveFile /home/matteo/rtabmap_exploration/graph.txt
--Regions/CentroidsSaveFile /home/matteo/rtabmap_exploration/centroids.txt
--Regions/RadiusUpperBound 40
--Regions/DesiredAverageCardinality 30
--Regions/MeshShapeFactor 1
--Regions/KeepPrefixPath false
--Regions/NameTotalLength 6"
```

Parameters flagged with the prefix **Regions** have already been explained in the README.md file of rtabmap directory. 
In this case, since the parameters ```Regions/InferenceMode``` is false, rtabmap is in exploration mode. You should change the path of the files to save the dataset and all the other stuffs.

In exploration mode, parameters  ```--Rtabmap/DetectionRate 1``` and ```--Rtabmap/CreateIntermediateNodes true``` are very important. The former refers to the frequency with which the main loop of rtabmap works. Default value is 1, and it means that it works at 1Hz, so one time per second. This value should not be too high because the optimization process in the backend takes time and otherwise too much work accumulates, slowing down the whole process. But in exploration mode, the images necessary to train the neural network are captured and saved, so they should be as much as possible. The latter parameter allows to create intermediate nodes ignoring the frequency, but they are considered as "invalid", so they are not used for graph optimization, avoiding slowing down rtabmap. In this way, at each second only an optimization cycle occurs, but all the nodes enters in the rtabmap cycle, so the images can be saved. <br>
In inference mode ```--Rtabmap/CreateIntermediateNodes``` should be false, to avoid the creation of unnecessary nodes.

Parameter ```--Rtabmap/MemoryThr 50``` refers to the maximum size of the Working Memory, limited to 50 nodes.

Parameter ```--Mem/IncrementalMemory true``` refers to the creation of new nodes of the map. If you want to perform relocalization after an exploration, you should set this to false (e.g in OpenLoris you can use market1 and market2 as sequences with incremental memory enabled to construct the graph and the database and then run market3 without incremental memory, just to see if the robot is able to perform relocalization when navigates known part of the map.)

Parameter ```--Mem/InitWMWithAllNodes false``` refers to the initialization of the Working Memory. In our experiments was always set to false.

Parameter ```--Rtabmap/LoopThr 0.05``` is threshold that defines whether a signature in the WM should be considered as a loop closure hypothesis. It should be set according to the dataset used. Values that are too low are prone to false positives, while values that are too high are prone to missed detection.

Parameters ```--Regions/RadiusUpperBound 40```, ```--Regions/DesiredAverageCardinality 30``` and ```--Regions/MeshShapeFactor 1``` are used only in exploration mode and they tune the clustering algorithm. They should be set accordingly to the dataset used. Values that are too low will create very small clusters, values that are too high will create very large clusters. More in depth, ```--Regions/RadiusUpperBound 40``` refers to the maximum radius of the nodes of the cluster w.r.t. the centroid. Depending on the speed of the robot, this value can change significantly. For example, KITTI is acquired with cameras on a car, so the velocity is very high and for this reason the radius upper bound is set to 40 meters (because they are traveled in a relatively short time and the clusters will therefore have the right size), on the contrary Campus is acquired with a mobile robot that is quite slow, so the radius upper bound is set to 3 meters, to avoid clusters with hundreds of nodes. The other two parameters and more in general the scattering-based algorithm are useful to avoid precisely this problem.

