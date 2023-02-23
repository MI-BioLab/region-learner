# Region Prediction for Efficient Robot Localization on Large Maps

Official repository of the paper **Region Prediction for Efficient Robot Localization on Large Maps** (available here [?]).

The paper has been submitted to [IROS 2023](https://ieee-iros.org/). <with the following clearifier video.>

<!--- https://user-images.githubusercontent.com/41426942/220202864-8da8bff3-fd33-4902-8a96-14e2577a1376.mp4 -->

The idea is explained in detail in the paper, therefore the focus below will be on how to install everything to easily reproduce experiments (or perform new ones). 

Requirements and details for each part are outlined in the README files in the corresponding directory. 

Please, when prompted to read a file, read it before continuing with the next steps. 

# Setting the environment
1. Clone the repository.
2. Read the README.md file in the rtabmap directory and install rtabmap.
3. Read the README.md file in the rtabmap_ros directory and install rtabmap_ros. For our experiments, ROS Melodic and Ubuntu 18.04 were used, as the Campus rosbags were captured in the same environment.
4. Read the README.md file in the RegionLearner directory.
5. Execute the script datasets.sh to download the rosbags.

# General operation

# Run the experiments
1. Launch rtabmap_ros with the appropriate launch file in exploration mode using the command ```roslaunch rtabmap_ros {lauchername}.launch```. Then, run the bag using the command ```rosbag play --clock path/to/{bagname}.bag``` to acquire the dataset.
2. Set the configurations of RegionLearner in ```RegionLearner/config/config.cfg```.
3. Launch the train of the deep neural network using the command ```python path/to/RegionLearner/src/train.py```.
4. Launch rtabmap_ros with the appropriate launch file in inference mode to predict the regions probabilities, enabling large scale loop closure detection. The commands to use are the same as 6.
5. Optionally launch the test with the command ```python path/to/RegionLearner/src/test.py``` to see the top-N accuracy of the network predictions.


If you want to use the loop closure inspector tool, you should do:
1. Read the README.md file in the LoopClosureInspector directory.
2. Set the configurations of LoopClosureInspector in ```LoopClosureInspector/config/config.cfg```. 
3. Run LoopClosureInspector using the command ```python path/to/LoopClosureInspector/src/loop_inspector.py```. 
4. Set the configurations of the test in ```LoopClosureInspector/config/test.cfg```. 
5. Run the test using the command ```python path/to/LoopClosureInspector/src/test.py```. 
