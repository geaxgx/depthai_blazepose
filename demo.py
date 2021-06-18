#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or '831') or path to an .blob file (default=%(default)s)")
parser.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser.add_argument('--filter_window_size', type=int, default=5,
                    help="Smoothing filter window size. Higher value adds to lag and to stability (default=%(default)i)")                    
parser.add_argument('--filter_velocity_scale', type=float, default=10,
                    help="Smoothing filter velocity scale. Lower value adds to lag and to stability (default=%(default)s)")                    
parser.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser.add_argument('-3', '--show_3d', action="store_true", 
                    help="Display skeleton in 3d in a separate window (valid only for full body landmark model)")
parser.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai
pose = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,
            filter_window_size=args.filter_window_size,
            filter_velocity_scale=args.filter_velocity_scale,                
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=args.stats,
            trace=args.trace)   

renderer = BlazeposeRenderer(
                pose, 
                show_3d=args.show_3d, 
                output=args.output)

while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()