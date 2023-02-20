import numpy as np

def _angle_difference(a1, a2):
    """
    Returns the minimum angular distance between two angles in the
    range (-180, 180] as a positive angle in the range [0, 180]
    
    Args:
        a1, a2 (num): two numbers representing the angles to compare

    Returns:
        (float): the angular distance
    """
    a1 = a1 + 360 if a1 < 0 else a1  # in [0,360) now
    a2 = a2 + 360 if a2 < 0 else a2  # in [0,360) now
    difference = (a2 - a1 + 180) % 360 - 180
    return abs(difference)

def angular_distance_1D(c0, centres, dim):
    distances = np.empty(len(centres))
    for idx, c1 in enumerate(centres):
        # calculate the distance
        dis = _angle_difference(c0, c1)

        # store the distance
        distances[idx] = dis

    return distances