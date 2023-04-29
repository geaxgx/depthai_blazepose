#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import numpy as np
from djitellopy import Tello
import time

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")                 
parser_tracker.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default="world",
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai
tracker = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,            
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=True,
            trace=args.trace)   

renderer = BlazeposeRenderer(
                tracker, 
                show_3d=False, 
                output=args.output)

ref_frame = None
is_start = False
ref_hand_vec = None
VERTICAL_THRESHOLD_UP    = 0.5  ## radians 
VERTICAL_THRESHOLD_DOWN  = -0.5  ## radians 


tello = Tello()

tello.connect()

tello.send_keepalive()

tello.streamon()
# frame_read = tello.get_frame_read()

tello.takeoff()

def dot_prd(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

while True:
    # Run blazepose on next frame
    frame, body = tracker.next_frame()
    if frame is None: break


    # Draw 2d skeleton
    if is_start:
        frame = renderer.draw(frame, body, angle)
    else:
        frame = renderer.draw(frame, body)

    key = renderer.waitKey(delay=1)
    
    if(key == 27):
        break

    # Get direction: Up/Down
    if body is not None:
        right_shoulder = body.landmarks[12]
        right_wrist = body.landmarks[16]
        if (key==ord('r') or (key==ord('s') and is_start is False)):    
            ref_frame = frame
            is_start = True
            ref_right_wrist, ref_right_shoulder = right_wrist, right_shoulder
            ref_hand_vec = ref_right_wrist - ref_right_shoulder
        if (key==ord('q') and is_start is True):
            ref_frame = None
            is_start = False
            continue

        if is_start and ref_hand_vec is not None:
            hand_vec = right_wrist - right_shoulder
            if hand_vec is not None:
                sign = 1 if (right_wrist - ref_right_wrist)[1] < 0 else -1
                angle = sign * np.arccos(dot_prd(hand_vec, ref_hand_vec))
            else:
                print('Right arm not visible completely. Please Align.')

            if angle>VERTICAL_THRESHOLD_UP:
                print('UP! UP! Away!')
                tello.send_rc_control(0,0,1,0)
                time.sleep(1)
                tello.send_rc_control(0,0,0,0)
            if angle<VERTICAL_THRESHOLD_DOWN:
                print('Shawty get low, low, low!')
                tello.send_rc_control(0,0,-1,0)

            print(angle)

    ## stop
    if key==ord('p') and is_start is True:
        ref_frame = None
        is_start = False
        ref_hand_vec = None
    
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()

tello.land()