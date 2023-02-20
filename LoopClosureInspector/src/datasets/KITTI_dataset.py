import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from datasets.base_dataset import BaseDataset

class KITTIDataset(BaseDataset):
    """
    Pose Ground-Truth reader that conforms to the KITTI Odometry format
    http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    """
    
    # KITTI represents odometry in 4x4 homogeneous pose matrices
    # and only writes the 3x3 rotation and 3x1 translation.
    # The missing row would be [0, 0, 0, 1]

    def __init__(self):
        self.COL_NAMES = self.KITTI_LIKE
        self.ROT_COL_NAMES = self.ROT_KITTI_LIKE

    def get_angles(self, poses, rotation_axis):
        return self._get_angles_KITTI_like(poses, rotation_axis)

    