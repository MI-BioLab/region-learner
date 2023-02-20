rtabmap_ros
=======

Modified version of the RTAB-Map's ROS package. Please, refer to the original version of [rtabmap_ros](https://github.com/introlab/rtabmap_ros) for the installation.

# Requirements
In addition to the requirements in the original version, the following requirements must be satisfied:
- [LibTorch](https://pytorch.org/cppdocs/installing.html) (C++ PyTorch library);
- [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (personally I followed the point 1.3.2 Debian Local Installation of this [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)). Versions should be chosen according to your GPU (CUDA 11.6 and cuDNN 8.6.0 were used in the development of this package).

# Modifications
Compared to the original rtabmap_ros, only the CMakeLists.txt file has been modified to include libtorch and three launch files have been added:
- openloris.launch to launch rtabmap_ros with the appropriate parameters for the bags of the [OpenLoris-Scene dataset](https://lifelong-robotic-vision.github.io/dataset/scene.html);
- kitti.launch to launch rtabmap_ros with the appropriate parameters for the bags of the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (without velodyne);
- campus.launch to launch rtabmap_ros with the appropriate parameters for the bags of the Campus dataset (acquired in our Campus corridors).
