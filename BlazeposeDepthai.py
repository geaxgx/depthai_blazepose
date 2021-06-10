import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
import depthai as dai

import open3d as o3d
from o3d_utils import create_segment, create_grid
import time

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = SCRIPT_DIR / "models/pose_detection_sh4.blob"
LANDMARK_MODEL_FULL = SCRIPT_DIR / "models/pose_landmark_full_sh4.blob"
LANDMARK_MODEL_FULL_0831 = SCRIPT_DIR / "models/pose_landmark_full_0831_sh4.blob"
LANDMARK_MODEL_LITE = SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob"


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

class BlazeposeDepthai:
    """
    Blazepose body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
    - pd_model: Blazepose detection model blob file (if None, takes the default value POSE_DETECTION_MODEL),
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - lm_model: Blazepose landmark model blob file
                    - None or "full": the default blob file LANDMARK_MODEL_FULL,
                    - "lite": the default blob file LANDMARK_MODEL_LITE,
                    - "831": the full model from previous version of mediapipe (0.8.3.1) LANDMARK_MODEL_FULL_0831,
                    - a path of a blob file. 
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - crop : boolean which indicates if square cropping is done or not
    - smoothing: boolean which indicates if smoothing filtering is applied
    - filter_window_size and filter_velocity_scale:
            The filter keeps track (on a window of specified size) of
            value changes over time, which as result gives velocity of how value
            changes over time. With higher velocity it weights new values higher.
            - higher filter_window_size adds to lag and to stability
            - lower filter_velocity_scale adds to lag and to stability

    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                                The width is calculated accordingly to height and depends on value of 'crop'
    - stats : boolean, when True, display some statistics when exiting.   
    - trace: boolean, when True print some debug messages   
    - force_detection:     boolean, force person detection on every frame (never use landmarks from previous frame to determine ROI)           
    """
    def __init__(self, input_src="rgb",
                pd_model=None, 
                pd_score_thresh=0.5,
                lm_model=None,
                lm_score_thresh=0.7,
                crop=False,
                smoothing= True,
                filter_window_size=5,
                filter_velocity_scale=10,
                internal_fps=None,
                internal_frame_height=1080,
                stats=False,
                trace=False,
                force_detection=False
                ):
        
        self.pd_model = pd_model if pd_model else POSE_DETECTION_MODEL
        print(f"Pose detection blob file : {self.pd_model}")
        self.rect_transf_scale = 1.25
        if lm_model is None or lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "831":
            self.lm_model = LANDMARK_MODEL_FULL_0831
            self.rect_transf_scale = 1.5
        else:
            self.lm_model = lm_model
        print(f"Landmarks using blob file : {self.lm_model}")
        
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh
        self.smoothing = smoothing
        self.crop = crop 
        self.internal_fps = internal_fps     
        self.stats = stats
        self.force_detection = force_detection
        
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            # Note that here (in Host mode), specifying "rgb_laconic" has no effect
            # Color camera frame is systematically transferred to the host
            self.input_type = "rgb" # OAK* internal color camera
            if internal_fps is None:
                if "831" in str(lm_model):
                    self.internal_fps = 15
                elif "full" in str(lm_model):
                    self.internal_fps = 12
                else: 
                    self.internal_fps = 20
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = internal_fps # Used when saving the output in a video file. Should be close to the real fps

            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
                self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)

        if self.input_type != "rgb":
            print(f"Original frame size: {self.img_w}x{self.img_h}")
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h) # // 16 * 16
            else:
                self.frame_size = max(self.img_w, self.img_h) #// 16 * 16
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)
            
            
            print(f"Frame working size: {self.img_w}x{self.img_h}")

        self.nb_kps = 33 # Number of "viewable" keypoints

        if self.smoothing:
            self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_kps, 3))
    
        # Create SSD anchors 
        self.anchors = mpu.generate_blazepose_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        self.pd_input_length = 224
        self.lm_input_length = 256
        self.device = dai.Device(self.create_pipeline())
        print("Pipeline started")

        # Define data queues 
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pre_pd_manip_cfg = self.device.getInputQueue(name="pre_pd_manip_cfg")
        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
        self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        self.q_lm_in = self.device.getInputQueue(name="lm_in")
        self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
            

        self.fps = FPS()

        self.nb_frames = 0
        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_lm_inferences_after_landmarks_ROI = 0
        self.nb_frames_no_body = 0

        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0

        self.use_previous_landmarks = False



        self.cfg_pre_pd = dai.ImageManipConfig()
        self.cfg_pre_pd.setResizeThumbnail(self.pd_input_length, self.pd_input_length)

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_3)
        

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)

            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)
            
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam.video.link(cam_out.input)

            # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
            print("Creating Pose Detection pre processing image manip...")
            pre_pd_manip = pipeline.create(dai.node.ImageManip)
            pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
            pre_pd_manip.setWaitForConfigInput(True)
            pre_pd_manip.inputImage.setQueueSize(1)
            pre_pd_manip.inputImage.setBlocking(False)
            cam.preview.link(pre_pd_manip.inputImage)

            pre_pd_manip_cfg_in = pipeline.create(dai.node.XLinkIn)
            pre_pd_manip_cfg_in.setStreamName("pre_pd_manip_cfg")
            pre_pd_manip_cfg_in.out.link(pre_pd_manip.inputConfig)   

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_model).resolve().absolute()))
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Pose detection input                 
        if self.input_type == "rgb":
            pre_pd_manip.out.link(pd_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("pd_in")
            pd_in.out.link(pd_nn.input)

        # Pose detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        

        # Define landmark model
        print("Creating Landmark Neural Network...")          
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_model).resolve().absolute()))
        lm_nn.setNumInferenceThreads(1)
        # Landmark input
        # if self.input_type == "rgb":
        #     pre_lm_manip.out.link(lm_nn.input)
        # else:
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        # Landmark output
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)
            
        print("Pipeline created.")
        return pipeline        

        
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("Identity_1"), dtype=np.float16) # 2254
        bboxes = np.array(inference.getLayerFp16("Identity"), dtype=np.float16).reshape((self.nb_anchors,12)) # 2254x12
        # Decode bboxes
        bodies = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        if bodies:
            body = bodies[0]
        else:
            return None
        mpu.detections_to_rect(body)
        mpu.rect_transformation(body, self.frame_size, self.frame_size, self.rect_transf_scale)
        return body
   
    def lm_postprocess(self, body, inference):
        body.lm_score = inference.getLayerFp16("output_poseflag")[0]
        if body.lm_score > self.lm_score_thresh:  

            lm_raw = np.array(inference.getLayerFp16("ld_3d")).reshape(-1,5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the body of
            # interest and range from [0.0, 255.0].
            # - Z coordinate is measured in "image pixels" like
            # the X and Y coordinates and represents the
            # distance relative to the plane of the subject's
            # hips, which is the origin of the Z axis. Negative
            # values are between the hips and the camera;
            # positive values are behind the hips. Z coordinate
            # scale is similar with X, Y scales but has different
            # nature as obtained not via human annotation, by
            # fitting synthetic data (GHUM model) to the 2D
            # annotation.
            # - Visibility, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame and not occluded by another bigger body
            # part or another object.
            # - Presence, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame.

            # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
            lm_raw[:,:3] /= self.lm_input_length
            # Apply sigmoid on visibility and presence (if used later)
            # lm_raw[:,3:5] = 1 / (1 + np.exp(-lm_raw[:,3:5]))
            
            # body.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
            body.norm_landmarks = lm_raw[:,:3]
            # Now calculate body.landmarks = the landmarks in the image coordinate system (in pixel) (body.landmarks)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in body.rect_points[1:]], dtype=np.float32) # body.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps+2,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
            # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
            lm_z = body.norm_landmarks[:self.nb_kps+2,2:3] * body.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            if self.smoothing:
                lm_xyz = self.filter.apply(lm_xyz)
            body.landmarks = lm_xyz.astype(np.int)

            # body_from_landmarks will be used to initialize the bounding rotated rectangle in the next frame
            # The only information we need are the 2 landmarks 33 and 34
            self.body_from_landmarks = mpu.Body(pd_kps=body.landmarks[self.nb_kps:self.nb_kps+2,:2]/self.frame_size)

            # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
            if self.pad_h > 0:
                body.landmarks[:,1] -= self.pad_h
                for i in range(len(body.rect_points)):
                    body.rect_points[i][1] -= self.pad_h
            if self.pad_w > 0:
                body.landmarks[:,0] -= self.pad_w
                for i in range(len(body.rect_points)):
                    body.rect_points[i][0] -= self.pad_w
                
                
    def next_frame(self):

        self.fps.update()
           
        if self.input_type == "rgb":
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
            if self.pad_h:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame
            # For debugging
            # if not self.crop:
            #     lb = self.q_lb_out.get()
            #     if lb:
            #         lb = lb.getCvFrame()
            #         cv2.imshow("letterbox", lb)
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
            if self.pad_h or self.pad_w:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame

        if self.force_detection or not self.use_previous_landmarks:
            if self.input_type == "rgb":
                self.q_pre_pd_manip_cfg.send(self.cfg_pre_pd)
            else:
                frame_nn = dai.ImgFrame()
                frame_nn.setTimestamp(time.monotonic())
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(square_frame, (self.pd_input_length, self.pd_input_length)))
                pd_rtrip_time = now()
                self.q_pd_in.send(frame_nn)

            # Get pose detection
            inference = self.q_pd_out.get()
            if self.input_type != "rgb": 
                pd_rtrip_time = now() - pd_rtrip_time
                self.glob_pd_rtrip_time += pd_rtrip_time
            body = self.pd_postprocess(inference)
            self.nb_pd_inferences += 1
        else:
            body = self.body_from_landmarks
            mpu.detections_to_rect(body) # self.regions.pd_kps are initialized from landmarks on previous frame
            mpu.rect_transformation(body, self.frame_size, self.frame_size, self.rect_transf_scale)


        # Landmarks
        if body:
            frame_nn = mpu.warp_rect_img(body.rect_points, square_frame, self.lm_input_length, self.lm_input_length)
            frame_nn = frame_nn / 255.
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(frame_nn, (self.lm_input_length, self.lm_input_length)))
            lm_rtrip_time = now()
            self.q_lm_in.send(nn_data)
            
            # Get landmarks
            inference = self.q_lm_out.get()
            lm_rtrip_time = now() - lm_rtrip_time
            self.glob_lm_rtrip_time += lm_rtrip_time
            self.nb_lm_inferences += 1
            self.lm_postprocess(body, inference)
            if body.lm_score < self.lm_score_thresh:
                body = None
                self.use_previous_landmarks = False
                if self.smoothing: self.filter.reset()
            else:
                self.use_previous_landmarks = True
            
        else:
            self.use_previous_landmarks = False
            if self.smoothing: self.filter.reset()

        return video_frame, body


    def exit(self):
        self.device.close()
        # Print some stats
        print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.nb_frames})")
        print(f"# pose detection inferences : {self.nb_pd_inferences}")
        print(f"# landmark inferences       : {self.nb_lm_inferences}")
        if self.input_type != "rgb" and self.nb_pd_inferences != 0: print(f"Pose detection round trip   : {self.glob_pd_rtrip_time/self.nb_pd_inferences*1000:.1f} ms")
        if self.nb_lm_inferences != 0:  print(f"Landmark round trip         : {self.glob_lm_rtrip_time/self.nb_lm_inferences*1000:.1f} ms")

           
