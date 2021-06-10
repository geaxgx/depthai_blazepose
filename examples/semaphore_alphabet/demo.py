import cv2
from math import atan2, degrees
import sys
sys.path.append("../..")
from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
import argparse


# For gesture demo
semaphore_flag = {
        (3,4):'A', (2,4):'B', (1,4):'C', (0,4):'D',
        (4,7):'E', (4,6):'F', (4,5):'G', (2,3):'H',
        (0,3):'I', (0,6):'J', (3,0):'K', (3,7):'L',
        (3,6):'M', (3,5):'N', (2,1):'O', (2,0):'P',
        (2,7):'Q', (2,6):'R', (2,5):'S', (1,0):'T',
        (1,7):'U', (0,5):'V', (7,6):'W', (7,5):'X',
        (1,6):'Y', (5,6):'Z',
}

def recognize_gesture(b):  
    # b: body         

    def angle_with_y(v):
        # v: 2d vector (x,y)
        # Returns angle in degree of v with y-axis of image plane
        if v[1] == 0:
            return 90
        angle = atan2(v[0], v[1])
        return degrees(angle)

    # For the demo, we want to recognize the flag semaphore alphabet
    # For this task, we just need to measure the angles of both arms with vertical
    right_arm_angle = angle_with_y(b.landmarks[KEYPOINT_DICT['right_elbow'],:2] - b.landmarks[KEYPOINT_DICT['right_shoulder'],:2])
    left_arm_angle = angle_with_y(b.landmarks[KEYPOINT_DICT['left_elbow'],:2] - b.landmarks[KEYPOINT_DICT['left_shoulder'],:2])
    right_pose = int((right_arm_angle +202.5) / 45) % 8 
    left_pose = int((left_arm_angle +202.5) / 45) % 8
    letter = semaphore_flag.get((right_pose, left_pose), None)
    return letter

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

while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    # Gesture recognition
    if body: 
        letter = recognize_gesture(body)
        if letter:
            cv2.putText(frame, letter, (frame.shape[1] // 2, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0,190,255), 3)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()

