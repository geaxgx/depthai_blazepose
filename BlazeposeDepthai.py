import numpy as np
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
from math import sin, cos
import depthai as dai
import time, sys

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = str(SCRIPT_DIR / "models/pose_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/pose_landmark_full_sh4.blob")
LANDMARK_MODEL_HEAVY = str(SCRIPT_DIR / "models/pose_landmark_heavy_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob")


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
                    - "heavy": default blob file LANDMARK_MODEL_HEAVY,
                    - a path of a blob file. 
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - xyz: boolean, when True get the (x, y, z) coords of the reference point (center of the hips) (if the device supports depth measures).
    - crop : boolean which indicates if square cropping is done or not
    - smoothing: boolean which indicates if smoothing filtering is applied
    - filter_window_size and filter_velocity_scale:
            The filter keeps track (on a window of specified size) of
            value changes over time, which as result gives velocity of how value
            changes over time. With higher velocity it weights new values higher.
            - higher filter_window_size adds to lag and to stability
            - lower filter_velocity_scale adds to lag and to stability

    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
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
                xyz=False,
                crop=False,
                smoothing= True,
                internal_fps=None,
                resolution="full",
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
        elif lm_model == "heavy":
            self.lm_model = LANDMARK_MODEL_HEAVY
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
        self.presence_threshold = 0.5
        self.visibility_threshold = 0.5

        self.device = dai.Device()
        self.xyz = False
        
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            # Note that here (in Host mode), specifying "rgb_laconic" has no effect
            # Color camera frame is systematically transferred to the host
            self.input_type = "rgb" # OAK* internal color camera
            if internal_fps is None:
                if "heavy" in str(lm_model):
                    self.internal_fps = 10
                elif "full" in str(lm_model):
                    self.internal_fps = 8
                else: # Light
                    self.internal_fps = 13
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2

            else:
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")

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
            
            self.filter_landmarks = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.05,
                beta=80,
                derivate_cutoff=1
            )
            # landmarks_aux corresponds to the 2 landmarks used to compute the ROI in next frame
            self.filter_landmarks_aux = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.01,
                beta=10,
                derivate_cutoff=1
            )
            self.filter_landmarks_world = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.1,
                beta=40,
                derivate_cutoff=1,
                disable_value_scaling=True
            )
            if self.xyz:
                self.filter_xyz = mpu.LowPassFilter(alpha=0.25)
    
        # Create SSD anchors 
        self.anchors = mpu.generate_blazepose_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        self.pd_input_length = 224
        self.lm_input_length = 256
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pre_pd_manip_cfg = self.device.getInputQueue(name="pre_pd_manip_cfg")
            if self.xyz:
                self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=1, blocking=False)
                self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")

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
        # pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            if self.resolution[0] == 1920:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
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
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
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

            if self.xyz:

                # For now, RGB needs fixed focus to properly align with depth.
                # The value used during calibration should be used here
                calib_data = self.device.readCalibration()
                calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
                print(f"RGB calibration lens position: {calib_lens_pos}")
                cam.initialControl.setManualFocus(calib_lens_pos)

                mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
                left = pipeline.createMonoCamera()
                left.setBoardSocket(dai.CameraBoardSocket.LEFT)
                left.setResolution(mono_resolution)
                left.setFps(self.internal_fps)

                right = pipeline.createMonoCamera()
                right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                right.setResolution(mono_resolution)
                right.setFps(self.internal_fps)

                stereo = pipeline.createStereoDepth()
                stereo.setConfidenceThreshold(230)
                # LR-check is required for depth alignment
                stereo.setLeftRightCheck(True)
                stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
                stereo.setSubpixel(False)  # subpixel True -> latency

                spatial_location_calculator = pipeline.createSpatialLocationCalculator()
                spatial_location_calculator.setWaitForConfigInput(True)
                spatial_location_calculator.inputDepth.setBlocking(False)
                spatial_location_calculator.inputDepth.setQueueSize(1)

                spatial_data_out = pipeline.createXLinkOut()
                spatial_data_out.setStreamName("spatial_data_out")
                spatial_data_out.input.setQueueSize(1)
                spatial_data_out.input.setBlocking(False)

                spatial_calc_config_in = pipeline.createXLinkIn()
                spatial_calc_config_in.setStreamName("spatial_calc_config_in")

                left.out.link(stereo.left)
                right.out.link(stereo.right)    

                stereo.depth.link(spatial_location_calculator.inputDepth)

                spatial_location_calculator.out.link(spatial_data_out.input)
                spatial_calc_config_in.out.link(spatial_location_calculator.inputConfig)


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
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        # Landmark output
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)
            
        print("Pipeline created.")
        return pipeline        

    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.presence_threshold
    
    def is_visible(self, body, lm_id):
        if body.visibility[lm_id] > self.visibility_threshold and \
            0 <= body.landmarks[lm_id][0] < self.img_w and \
            0 <= body.landmarks[lm_id][1] < self.img_h :
            return True
        else:
            return False

    def query_body_xyz(self, body):
        # We want the 3d position (x,y,z) in meters of the body reference keypoint
        # in the camera coord system.
        # The reference point is either :
        # - the middle of the hips if both hips are present (presence of rght and left hips > threshold),
        # - the middle of the shoulders in case at leats one hip is not present and
        #   both shoulders are present,
        # - None otherwise
        if self.is_visible(body, mpu.KEYPOINT_DICT['right_hip']) and self.is_visible(body, mpu.KEYPOINT_DICT['left_hip']):
            body.xyz_ref = "mid_hips"
            body.xyz_ref_coords_pixel = np.mean([
                body.landmarks[mpu.KEYPOINT_DICT['right_hip']][:2],
                body.landmarks[mpu.KEYPOINT_DICT['left_hip']][:2]], 
                axis=0)
        elif self.is_visible(body, mpu.KEYPOINT_DICT['right_shoulder']) and self.is_visible(body, mpu.KEYPOINT_DICT['left_shoulder']):
            body.xyz_ref = "mid_shoulders"
            body.xyz_ref_coords_pixel = np.mean([
                body.landmarks[mpu.KEYPOINT_DICT['right_shoulder']][:2],
                body.landmarks[mpu.KEYPOINT_DICT['left_shoulder']][:2]],
                axis=0) 
        else:
            body.xyz_ref = None
            return
        # Prepare the request to SpatialLocationCalculator
        # ROI : small rectangular zone around the reference keypoint
        half_zone_size = int(max(body.rect_w_a / 90, 4))
        xc = int(body.xyz_ref_coords_pixel[0] + self.crop_w)
        yc = int(body.xyz_ref_coords_pixel[1])
        roi_left = max(0, xc - half_zone_size)
        roi_right = min(self.img_w-1, xc + half_zone_size)
        roi_top = max(0, yc - half_zone_size)
        roi_bottom = min(self.img_h-1, yc + half_zone_size)
        roi_topleft = dai.Point2f(roi_left, roi_top)
        roi_bottomright = dai.Point2f(roi_right, roi_bottom)
        # Config
        conf_data = dai.SpatialLocationCalculatorConfigData()
        conf_data.depthThresholds.lowerThreshold = 100
        conf_data.depthThresholds.upperThreshold = 10000
        # conf_data.roi = dai.Rect(roi_center, roi_size)
        conf_data.roi = dai.Rect(roi_topleft, roi_bottomright)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs([conf_data])
        # spatial_rtrip_time = now()
        self.q_spatial_config.send(cfg)

        # Receives spatial locations
        spatial_data = self.q_spatial_data.get().getSpatialLocations()
        # self.glob_spatial_rtrip_time += now() - spatial_rtrip_time
        # self.nb_spatial_requests += 1
        sd = spatial_data[0]
        body.xyz_zone =  [
            int(sd.config.roi.topLeft().x) - self.crop_w,
            int(sd.config.roi.topLeft().y),
            int(sd.config.roi.bottomRight().x) - self.crop_w,
            int(sd.config.roi.bottomRight().y)
            ]
        body.xyz = np.array([
            sd.spatialCoordinates.x,
            sd.spatialCoordinates.y,
            sd.spatialCoordinates.z
            ])
        if self.smoothing:
            body.xyz = self.filter_xyz.apply(body.xyz)
        
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
        # The output names of the landmarks model are :
        # Identity_1 (1x1) : score (previously output_poseflag)
        # Identity_2 (1x128x128x1) (previously output_segmentation)
        # Identity_3 (1x64x64x39) (previously output_heatmap)
        # Identity_4 (1x117) world 3D landmarks (previously world_3d)
        # Identity (1x195) image 3D landmarks (previously ld_3d)
        body.lm_score = inference.getLayerFp16("Identity_1")[0]
        if body.lm_score > self.lm_score_thresh:  

            lm_raw = np.array(inference.getLayerFp16("Identity")).reshape(-1,5)
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
            body.visibility = 1 / (1 + np.exp(-lm_raw[:,3]))
            body.presence = 1 / (1 + np.exp(-lm_raw[:,4]))
            
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

            # World landmarks are predicted in meters rather than in pixels of the image
            # and have origin in the middle of the hips rather than in the corner of the
            # pose image (cropped with given rectangle). Thus only rotation (but not scale
            # and translation) is applied to the landmarks to transform them back to
            # original  coordinates.
            body.landmarks_world = np.array(inference.getLayerFp16("Identity_4")).reshape(-1,3)[:self.nb_kps]
            sin_rot = sin(body.rotation)
            cos_rot = cos(body.rotation)
            rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
            body.landmarks_world[:,:2] = np.dot(body.landmarks_world[:,:2], rot_m)
            
            if self.smoothing:
                timestamp = now()
                object_scale = body.rect_w_a
                lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(lm_xyz[:self.nb_kps], timestamp, object_scale)
                lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(lm_xyz[self.nb_kps:], timestamp, object_scale)
                body.landmarks_world = self.filter_landmarks_world.apply(body.landmarks_world, timestamp)

            body.landmarks = lm_xyz.astype(np.int32)

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
            if self.input_type != "rgb" and (self.force_detection or not self.use_previous_landmarks): 
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
                if self.smoothing: 
                    self.filter_landmarks.reset()
                    self.filter_landmarks_aux.reset()
                    self.filter_landmarks_world.reset()
            else:
                self.use_previous_landmarks = True
                if self.xyz:
                    self.query_body_xyz(body)
            
        else:
            self.use_previous_landmarks = False
            if self.smoothing: 
                self.filter_landmarks.reset()
                self.filter_landmarks_aux.reset()
                self.filter_landmarks_world.reset()
                if self.xyz: self.filter_xyz.reset()
                
        return video_frame, body


    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nbf})")
            print(f"# pose detection inferences : {self.nb_pd_inferences}")
            print(f"# landmark inferences       : {self.nb_lm_inferences}")
            if self.input_type != "rgb" and self.nb_pd_inferences != 0: print(f"Pose detection round trip   : {self.glob_pd_rtrip_time/self.nb_pd_inferences*1000:.1f} ms")
            if self.nb_lm_inferences != 0:  print(f"Landmark round trip         : {self.glob_lm_rtrip_time/self.nb_lm_inferences*1000:.1f} ms")

           
