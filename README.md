# Region Prediction for Efficient Robot Localization on Large Maps

Official repository of the paper **Region Prediction for Efficient Robot Localization on Large Maps** (available [here](https://arxiv.org/abs/2303.00295)).

The paper has been submitted to [IROS 2023](https://ieee-iros.org/) with the following clearifier video.

https://user-images.githubusercontent.com/41426942/220202864-8da8bff3-fd33-4902-8a96-14e2577a1376.mp4

<br>
<br>

The idea is explained in detail in the paper, therefore the focus below will be on how to install everything to easily reproduce experiments (or perform new ones). 

Requirements and details for each part are outlined in the README files in the corresponding directory. 

Please, when prompted to read a file, read it before continuing with the next steps. 

# Setting the environment
1. Clone the repository.
2. Read the README.md file in the [rtabmap folder](https://github.com/MI-BioLab/region-learner/tree/main/rtabmap) and install rtabmap.
3. Read the README.md file in the [rtabmap_ros folder](https://github.com/MI-BioLab/region-learner/tree/main/rtabmap_ros) and install rtabmap_ros.
4. Read the README.md file in the [RegionLearner folder](https://github.com/MI-BioLab/region-learner/tree/main/RegionLearner).
5. Execute the script datasets.sh to download the rosbags.

# Run the experiments
Once you read all the README files as listed above, you should be able to understand the general operation of the system.

1. To acquire the dataset, launch rtabmap_ros with the appropriate launch file in exploration mode using the command ```roslaunch rtabmap_ros {lauchername}.launch```. Then, run the bag using the command ```rosbag play --clock path/to/{bagname}.bag```. Here the images and the files required for training are saved.
2. Create the dataset folder structure as explained in RegionLearner README and set the configurations of RegionLearner in ```RegionLearner/config/config.cfg```. 
3. Launch the training of the deep neural network using the command ```python path/to/RegionLearner/src/train.py```.
4. At this point, the trained neural network is ready to be used for the inference. Launch rtabmap_ros with the appropriate launch file in inference mode to predict the regions probabilities, enabling large scale loop closure detection. The commands to use are the same as 1.

## Testing the prediction accuracy
To test the accuracy of the network predictions, you should have the training-test images correspondences. If the dataset contains the raw images (e.g., OpenLoris), you can directly create the training-test file by manually aligning the images, as explained in RegionLearner README file. If the dataset is provided only as bagfiles (e.g., Campus), you need to run rtabmap in exploration mode on the test sequence, in order to acquire the raw images and then create the file as above.
Once you have the trained model, the test images and the training-test correspondences file, you can run the test with the command ```python path/to/RegionLearner/src/test.py``` to see the top-N accuracy of the network predictions.

## Automatic tool for loop detection
When the dataset is provided with a ground-truth of the poses, you can use LoopClosureInspector to automatically find matching poses.

If you want to use it, please read the README.md inside the [LoopClosureInspector folder](https://github.com/MI-BioLab/region-learner/tree/main/LoopClosureInspector) before proceeding.

If you want to use the loop closure inspector tool, you should do:
1. Set the configurations of LoopClosureInspector in ```LoopClosureInspector/config/config.cfg```. 
2. Run LoopClosureInspector using the command ```python path/to/LoopClosureInspector/src/loop_inspector.py```. Depending on the configurations, different files will be saved (as explained in the README file).
4. Set the configurations of the test in ```LoopClosureInspector/config/test.cfg```. 
5. Run the test using the command ```python path/to/LoopClosureInspector/src/test.py```. An image will be saved where the loop closures are highlighed. 



# TODO
- load the material for the experiments