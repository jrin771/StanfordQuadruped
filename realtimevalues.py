import cv2
import sys
import numpy as np
import argparse
from scipy import signal

sys.path.append("../..")
from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT


def get_yaw_rate(yaw_values):
    """
    This function calculates the yaw rate, which is the derivative of the yaw values over time.
    Arguments:
    yaw_values -- list or numpy array of yaw values
    Returns:
    yaw_rate -- list of yaw rate values
    """
    time = np.arange(len(yaw_values))  # Time intervals
    yaw_rate = np.gradient(yaw_values, time)  # Compute derivative using numpy gradient function
    return yaw_rate


def get_pose(b, keypoint_dict):
    """
    This function calculates the rotation matrix based on the given body and keypoint dictionary.
    The rotation matrix is then used to calculate the yaw, pitch, and roll values.
    Arguments:
    b -- the body object from BlazePose
    keypoint_dict -- dictionary containing keypoints indices
    Returns:
    yaw, pitch, and roll values 
    """
    # Get the landmarks for the left shoulder, right shoulder, and mid hip
    left_shoulder = np.array(b.landmarks[keypoint_dict['left_shoulder'], :3]).astype(float)
    right_shoulder = np.array(b.landmarks[keypoint_dict['right_shoulder'], :3]).astype(float)
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    left_hip = np.array(b.landmarks[keypoint_dict['left_hip'], :3]).astype(float)
    right_hip = np.array(b.landmarks[keypoint_dict['right_hip'], :3]).astype(float)
    mid_hip = (left_hip + right_hip) / 2 
    
    # Calculate the base vectors in the body coordinate system
    v1 = right_shoulder - left_shoulder
    v2 = mid_hip - mid_shoulder
    v3 = np.cross(v1, v2)

    # Normalize the base vectors
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)

    # Construct the rotation matrix
    rotation_matrix = np.array([v1, v2, v3]).T

    # Compute yaw, pitch, and roll based on the provided formulas
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    return yaw, pitch, roll


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
args = parser.parse_args()            

pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model)
renderer = BlazeposeRenderer(pose, output=args.output)

yaw_values = []  # Store yaw values for yaw rate calculation

while True:
    # Run BlazePose on the next frame
    frame, body = pose.next_frame()
    if frame is None:
        break
    
    # Draw 2D skeleton
    frame = renderer.draw(frame, body)
    
    # Pose estimation
    if body:
        yaw, pitch, roll = get_pose(body, KEYPOINT_DICT)
        yaw_values.append(yaw)  # Store yaw value
        if len(yaw_values) > 1:  # Check if there are at least two yaw values for yaw rate calculation
            yaw_rate = get_yaw_rate(yaw_values)  # Calculate the yaw rate
            print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}, Yaw Rate: {yaw_rate[-1]}")  # Print the current yaw rate
    
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

renderer.exit()
pose.exit()
