import numpy as np
import cv2
from numpy.core.fromnumeric import trace
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS
import depthai as dai
import time
import marshal
import sys
from string import Template

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = SCRIPT_DIR / "models/pose_detection_sh4.blob"
LANDMARK_MODEL_FULL = SCRIPT_DIR / "models/pose_landmark_full_sh4.blob"
LANDMARK_MODEL_LITE = SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob"
LANDMARK_MODEL_FULL_0831 = SCRIPT_DIR / "models/pose_landmark_full_0831_sh4.blob"
DETECTION_POSTPROCESSING_MODEL = SCRIPT_DIR / "custom_models/DetectionBestCandidate_sh1.blob"
DIVIDE_BY_255_MODEL = SCRIPT_DIR / "custom_models/DivideBy255_sh1.blob"


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

class BlazeposeDepthai:
    """
    Blazepose body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
                    Note that as we are in Edge mode, input sources coming from the host like a image or a video is not supported 
    - pd_model: Blazepose detection model blob file (if None, takes the default value POSE_DETECTION_MODEL),
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - pp_model: detection postprocessing model blob file  (if None, takes the default value DETECTION_POSTPROCESSING_MODEL),,
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
                pp_model=None,
                lm_model=None,
                lm_score_thresh=0.7,
                crop=False,
                smoothing= True,
                filter_window_size=5,
                filter_velocity_scale=10,
                stats=False,               
                internal_fps=None,
                internal_frame_height=1080,
                trace=False,
                force_detection=False):

        self.pd_model = pd_model if pd_model else POSE_DETECTION_MODEL
        self.pp_model = pp_model if pd_model else DETECTION_POSTPROCESSING_MODEL
        self.divide_by_255_model = DIVIDE_BY_255_MODEL
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

        self.trace = trace
        self.force_detection = force_detection
        
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = "laconic" in input_src # Camera frames are not sent to the host
            if internal_fps is None:
                if "831" in str(lm_model):
                    self.internal_fps = 18
                elif "full" in str(self.lm_model):
                    self.internal_fps = 14
                else:
                    self.internal_fps = 27
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

        else:
            print("Invalid input source:", input_src)
            sys.exit()

        self.nb_kps = 33

        if self.smoothing:
            self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_kps, 3))

        # Define and start pipeline
        self.device = dai.Device(self.create_pipeline())
        print("Pipeline started")

        # Define data queues 
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)
        # For debugging
        # self.q_pre_pd_manip_out = self.device.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
        # self.q_pre_lm_manip_out = self.device.getOutputQueue(name="pre_lm_manip_out", maxSize=1, blocking=False)

        self.fps = FPS()

        self.nb_frames = 0
        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_lm_inferences_after_landmarks_ROI = 0
        self.nb_frames_no_body = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)
        self.pd_input_length = 224
        self.lm_input_length = 256

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera) 
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

        if not self.laconic:
            cam_out = pipeline.create(dai.node.XLinkOut)
            cam_out.setStreamName("cam_out")
            cam.video.link(cam_out.input)


        # Define manager script node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScriptData(self.build_manager_script())

        # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
        print("Creating Pose Detection pre processing image manip...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        # For debugging
        # pre_pd_manip_out = pipeline.createXLinkOut()
        # pre_pd_manip_out.setStreamName("pre_pd_manip_out")
        # pre_pd_manip.out.link(pre_pd_manip_out.input)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(str(Path(self.pd_model).resolve().absolute()))
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        pre_pd_manip.out.link(pd_nn.input)
       
        # Define pose detection post processing "model"
        print("Creating Pose Detection post processing Neural Network...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(str(Path(self.pp_model).resolve().absolute()))
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])

        # Define link to send pd result to host 
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)

        # Define landmark pre processing image manip
        print("Creating Landmark pre processing image manip...") 
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length*self.lm_input_length*3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)

        # For debugging
        # pre_lm_manip_out = pipeline.createXLinkOut()
        # pre_lm_manip_out.setStreamName("pre_lm_manip_out")
        # pre_lm_manip.out.link(pre_lm_manip_out.input)
    
        # manager_script.outputs['to_lm'].link(pre_lm_manip.inputImage)
        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        # Define normalization model between ImageManip and landmark model
        # This is a temporary step. Could be removed when support of setFrameType(RGBF16F16F16p) in ImageManip node
        print("Creating DiveideBy255 Neural Network...") 
        divide_nn = pipeline.create(dai.node.NeuralNetwork)
        divide_nn.setBlobPath(str(Path(self.divide_by_255_model).resolve().absolute()))
        pre_lm_manip.out.link(divide_nn.input) 

        # Define landmark model
        print("Creating Landmark Neural Network...") 
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(str(Path(self.lm_model).resolve().absolute()))
        # lm_nn.setNumInferenceThreads(1)

        # pre_lm_manip.out.link(lm_nn.input)   
        divide_nn.out.link(lm_nn.input)       
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])

        print("Pipeline created.")

        return pipeline        

    def build_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the NN model (thunder or lightning),
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script.py which is a python template
        '''
        # Read the template
        with open('template_manager_script.py', 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE = "node.warn" if self.trace else "#",
                    _pd_score_thresh = self.pd_score_thresh,
                    _lm_score_thresh = self.lm_score_thresh,
                    _force_detection = self.force_detection,
                    _pad_h_norm = self.pad_h / self.img_h,
                    _height_ratio = self.frame_size / self.img_h,
                    _rect_transf_scale = self.rect_transf_scale
        )
        # For debugging
        # with open("tmp_code.py", "w") as file:
        #     file.write(code)

        return code

   
    def lm_postprocess(self, body, lms):
        # lms : landmarks sent by Manager script node to host (list of 39*5 elements for full body or 31*5 for upper body)
        lm_raw = np.array(lms).reshape(-1,5)
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
        lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps,:2], axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  

        # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
        # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
        lm_z = body.norm_landmarks[:self.nb_kps,2:3] * body.rect_w_a / 4
        lm_xyz = np.hstack((lm_xy, lm_z))
        if self.smoothing:
            lm_xyz = self.filter.apply(lm_xyz)
        body.landmarks = lm_xyz.astype(np.int)
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            body.landmarks[:,1] -= self.pad_h
            for i in range(len(body.rect_points)):
                body.rect_points[i][1] -= self.pad_h
        # if self.pad_w > 0:
        #     body.landmarks[:,0] -= self.pad_w
        #     for i in range(len(body.rect_points)):
        #         body.rect_points[i][0] -= self.pad_w  
    
                
    def next_frame(self):

        self.fps.update()
            
        if self.laconic:
            video_frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()       

        # For debugging
        # pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
        # if pre_pd_manip:
        #     pre_pd_manip = pre_pd_manip.getCvFrame()
        #     cv2.imshow("pre_pd_manip", pre_pd_manip)
        # pre_lm_manip = self.q_pre_lm_manip_out.tryGet()
        # if pre_lm_manip:
        #     pre_lm_manip = pre_lm_manip.getCvFrame()
        #     cv2.imshow("pre_lm_manip", pre_lm_manip)
                                
        # Get result from device
        res = marshal.loads(self.q_manager_out.get().getData())
        if res["type"] != 0 and res["lm_score"] > self.lm_score_thresh:
            body = mpu.Body()
            body.rect_x_center_a = res["rect_center_x"] * self.frame_size
            body.rect_y_center_a = res["rect_center_y"] * self.frame_size
            body.rect_w_a = body.rect_h_a = res["rect_size"] * self.frame_size
            body.rotation = res["rotation"] 
            body.rect_points = mpu.rotated_rect_to_points(body.rect_x_center_a, body.rect_y_center_a, body.rect_w_a, body.rect_h_a, body.rotation)
            body.lm_score = res["lm_score"]
            self.lm_postprocess(body, res['lms'])

        else:
            body = None

        # Statistics
        if self.stats:
            self.nb_frames += 1
            if res["type"] == 0:
                self.nb_pd_inferences += 1
                self.nb_frames_no_body += 1
            else:  
                self.nb_lm_inferences += 1
                if res["type"] == 1:
                    self.nb_pd_inferences += 1
                else: # res["type"] == 2
                    self.nb_lm_inferences_after_landmarks_ROI += 1
                if res["lm_score"] < self.lm_score_thresh: self.nb_frames_no_body += 1

        return video_frame, body


    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nbf})")
            print(f"# frames without body       : {self.nb_frames_no_body}")
            print(f"# pose detection inferences : {self.nb_pd_inferences}")
            print(f"# landmark inferences       : {self.nb_lm_inferences} - # after pose detection: {self.nb_lm_inferences - self.nb_lm_inferences_after_landmarks_ROI} - # after landmarks ROI prediction: {self.nb_lm_inferences_after_landmarks_ROI}")
        
        
           

