# loop-closure-inspector

Package to automatically detect loop closures or correspondences between poses in one or more path, given a ground truth of the poses.

# How it works
The tool performs 2D (currently) match detection or loop closure detection by evaluating a set of ground truth poses. It assesses each pose by determining if there exist any poses within a specified radius. If such poses exist, the tool then determines if the difference between the 1D orientations relative to the 2D plane falls within a defined angular tolerance and if the temporal separation between the two poses meets a minimum threshold. 

If these criteria are met, it indicates that two or more poses have identical coordinates and orientation, thereby implying that the robot or camera has traced the same path while moving in the same direction, effectively forming a loop closure.

The parameters for the **distance_upper_bound** (radius), **max_angular_difference** (angle), and **n_frame_since_last** (time span) must be set based on the data used. For the KITTI dataset, which is an outdoor dataset recorded at a 10 fps rate from a car, the radius can be set to 2 meters, the angle to 20 degrees, and the time span to 100. This is because 100 frames at 10 fps equates to 10 seconds, which is the minimum amount of time that must pass between two poses in order to be considered a possible closure. This avoids the possibility of considering poses that are too close in both space and time as matches, such as when the camera is stationary in one location.

## Example

To a better understanding, consider the following example.

A ground truth for the poses is given, corresponding to the path below (the blue line). The arrow shows the direction of the robot while the red point is the pose to which the algorithm is searching a possible loop closure.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/loop.png?raw=true)

First, a search on a radius distance is done. So every point outside the red circle can be discarded.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/loop_radius.png?raw=true)

Second, a search on the orientation is done. Only the points inside the circle with the same orientation of the point are kept.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/loop_angle.png?raw=true)

Lastly, only the poses occurred after o before n frame are kept, in order to discard points that are close in time but not loop closures. The green poses are the candidates that can be considered loop closures for the red point.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/loop_final.png?raw=true)

## Negative example

Contrary to the previous example, in this case the robot returns to a place but it moves in the opposite direction (purple point). In this case, the loop closure is not detected.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/no_loop.png?raw=true)

# Requirements
To install the requirements use the following command:
```
pip install -r requirements.txt
```

This package uses **GriSPy** (https://github.com/mchalela/GriSPy) to perform an efficient search.

# How to use it
You can simply clone the repository and launch the *loop_inspector.py* inside the folder */src/* with the command
```
python src/loop_inspector.py --cfg /config/config.cfg
```

## Configurations
Inside the folder */config/* there are two files:
- **config.cfg** to set the configurations for the ground truth to create.
- **test.cfg** to display the ground truth created.

### config.cfg
This file contains the default settings for different datasets. 

In the config.cfg you can set the dataset to use, the input file of the ground truth of the poses and the output folder.

Six parameters must be set, according to the dataset used:
- **distance_lower_bound** the lower bound for radius search in meters
- **distance_upper_bound** the upper bound for radius search in meters
- **n_frame_since_last** the number of frame that must occur between two loop closures
- **translation_axis** the axis to consider for the 2D loop closure
- **rotation_axis** the axis to consider for the angle
- **max_angular_difference** the maximum angle difference

### test.cfg
In this file, you can set the dataset, the path to the file that contains the pairs created and the path to the file of the ground truth poses.

Here, only translation_axis are set, depending on the dataset, in order to display the results.

## Input
The input is the path to the ground truth of the poses.

## Outputs
It outputs three files:
- **matrix.npy** a matrix of shape NxN where N is the number of poses. Each cell contains 0 if the poses i,j are not considered loop closure, 1 if they are
- **matrix.txt** same as above
- **pairs.txt** a set of tuple with the indexes of the poses that are loop closures

## Test
To run the test, after the creation of the three outputs, the following command can be used
```
python src/test.py --cfg /config/test.cfg
```

### Test with KITTI sequence 00
Example with the sequence 00 of the KITTI dataset. The red points are the poses in which the camera pass a second time, in the same direction. So they are the loop closures matches.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/kitti_00_loop.png?raw=true)

### Test with KITTI sequence 08
Example with the sequence 08 of the KITTI dataset. In this case the camera returns more times to the same places, but it moves with a different orientation, therefore no loop closure is detected.

![alt text](https://github.com/scumatteo/region-learner/blob/main/LoopClosureInspector/images/kitti_08_loop.png?raw=true)

## Datasets available
The tool is able to handle the following datasets:
- [X] KITTI odometry dataset
- [X] OpenLoris-Scene dataset
- [X] TUM RGB-D dataset

In the future new datasets will be added.




