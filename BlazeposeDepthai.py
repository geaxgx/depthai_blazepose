import math
import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
import depthai as dai
from math import atan2

import open3d as o3d
from o3d_utils import create_segment, create_grid
import time

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = SCRIPT_DIR / "models/pose_detection.blob"
FULL_BODY_LANDMARK_MODEL = SCRIPT_DIR / "models/pose_landmark_full_body.blob"
UPPER_BODY_LANDMARK_MODEL = SCRIPT_DIR / "models/pose_landmark_upper_body.blob"

import csv
import numpy as np
import os

class EMADictSmoothing(object):
  """Smoothes pose classification."""

  def __init__(self, window_size=10, alpha=0.2):
    self._window_size = window_size
    self._alpha = alpha

    self._data_in_window = []

  def __call__(self, data):
    """Smoothes given pose classification.

    Smoothing is done by computing Exponential Moving Average for every pose
    class observed in the given time window. Missed pose classes arre replaced
    with 0.
    
    Args:
      data: Dictionary with pose classification. Sample:
          {
            'pushups_down': 8,
            'pushups_up': 2,
          }

    Result:
      Dictionary in the same format but with smoothed and float instead of
      integer values. Sample:
        {
          'pushups_down': 8.3,
          'pushups_up': 1.7,
        }
    """
    # Add new data to the beginning of the window for simpler code.
    self._data_in_window.insert(0, data)
    self._data_in_window = self._data_in_window[:self._window_size]

    # Get all keys.
    keys = set([key for data in self._data_in_window for key, _ in data.items()])

    # Get smoothed values.
    smoothed_data = dict()
    for key in keys:
      factor = 1.0
      top_sum = 0.0
      bottom_sum = 0.0
      for data in self._data_in_window:
        value = data[key] if key in data else 0.0

        top_sum += factor * value
        bottom_sum += factor

        # Update factor.
        factor *= (1.0 - self._alpha)

      smoothed_data[key] = top_sum / bottom_sum

    return smoothed_data

class PoseSample(object):

  def __init__(self, name, landmarks, class_name, embedding):
    self.name = name
    self.landmarks = landmarks
    self.class_name = class_name
    
    self.embedding = embedding


class PoseSampleOutlier(object):

  def __init__(self, sample, detected_class, all_classes):
    self.sample = sample
    self.detected_class = detected_class
    self.all_classes = all_classes

class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
    assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    embedding = self._get_pose_distance_embedding(landmarks)

    return embedding

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]

    # Hips center.
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

  def _get_pose_distance_embedding(self, landmarks):
    """Converts pose landmarks into 3D embedding.

    We use several pairwise 3D distances to form pose embedding. All distances
    include X and Y components with sign. We differnt types of pairs to cover
    different pose classes. Feel free to remove some or add new.
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances.
    """
    embedding = np.array([
        # One joint.

        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

        self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

        # Two joints.

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

        # Four joints.

        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # Five joints.

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),
        
        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # Cross body.

        self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        # Body bent direction.

        # self._get_distance(
        #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
        #     landmarks[self._landmark_names.index('left_hip')]),
        # self._get_distance(
        #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
        #     landmarks[self._landmark_names.index('right_hip')]),
    ])

    return embedding

  def _get_average_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return self._get_distance(lmk_from, lmk_to)

  def _get_distance(self, lmk_from, lmk_to):
    return lmk_to - lmk_from

class PoseClassifier(object):
  """Classifies pose landmarks."""

  def __init__(self,
               pose_samples_folder,
               pose_embedder,
               file_extension='csv',
               file_separator=',',
               n_landmarks=33,
               n_dimensions=3,
               top_n_by_max_distance=30,
               top_n_by_mean_distance=10,
               axes_weights=(1., 1., 0.2)):
    self._pose_embedder = pose_embedder
    self._n_landmarks = n_landmarks
    self._n_dimensions = n_dimensions
    self._top_n_by_max_distance = top_n_by_max_distance
    self._top_n_by_mean_distance = top_n_by_mean_distance
    self._axes_weights = axes_weights

    self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                 file_extension,
                                                 file_separator,
                                                 n_landmarks,
                                                 n_dimensions,
                                                 pose_embedder)

  def _load_pose_samples(self,
                         pose_samples_folder,
                         file_extension,
                         file_separator,
                         n_landmarks,
                         n_dimensions,
                         pose_embedder):
    """Loads pose samples from a given folder.
    
    Required folder structure:
      neutral_standing.csv
      pushups_down.csv
      pushups_up.csv
      squats_down.csv
      ...

    Required CSV structure:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
      ...
    """
    # Each file in the folder represents one pose class.
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    pose_samples = []
    for file_name in file_names:
      # Use file name as pose class name.
      class_name = file_name[:-(len(file_extension) + 1)]
      
      # Parse CSV.
      with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
          assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
          landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
          pose_samples.append(PoseSample(
              name=row[0],
              landmarks=landmarks,
              class_name=class_name,
              embedding=pose_embedder(landmarks),
          ))

    return pose_samples

  def find_pose_sample_outliers(self):
    """Classifies each sample against the entire database."""
    # Find outliers in target poses
    outliers = []
    for sample in self._pose_samples:
      # Find nearest poses for the target one.
      pose_landmarks = sample.landmarks.copy()
      pose_classification = self.__call__(pose_landmarks)
      class_names = [class_name for class_name, count in pose_classification.items() if count == max(pose_classification.values())]

      # Sample is an outlier if nearest poses have different class or more than
      # one pose class is detected as nearest.
      if sample.class_name not in class_names or len(class_names) != 1:
        outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

    return outliers

  def __call__(self, pose_landmarks):
    """Classifies given pose.

    Classification is done in two stages:
      * First we pick top-N samples by MAX distance. It allows to remove samples
        that are almost the same as given pose, but has few joints bent in the
        other direction.
      * Then we pick top-N samples by MEAN distance. After outliers are removed
        on a previous step, we can pick samples that are closes on average.
    
    Args:
      pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

    Returns:
      Dictionary with count of nearest pose samples from the database. Sample:
        {
          'pushups_down': 8,
          'pushups_up': 2,
        }
    """
    # Check that provided and target poses have the same shape.
    assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

    # Get given pose embedding.
    pose_embedding = self._pose_embedder(pose_landmarks)
    flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

    # Filter by max distance.
    #
    # That helps to remove outliers - poses that are almost the same as the
    # given one, but has one joint bent into another direction and actually
    # represnt a different pose class.
    max_dist_heap = []
    for sample_idx, sample in enumerate(self._pose_samples):
      max_dist = min(
          np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      max_dist_heap.append([max_dist, sample_idx])

    max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
    max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

    # Filter by mean distance.
    #
    # After removing outliers we can find the nearest pose by mean distance.
    mean_dist_heap = []
    for _, sample_idx in max_dist_heap:
      sample = self._pose_samples[sample_idx]
      mean_dist = min(
          np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      mean_dist_heap.append([mean_dist, sample_idx])

    mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
    mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

    # Collect results into map: (class_name -> n_samples)
    class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
    result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

    print(result)

    return result


# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_FULL_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], 
                    [23,24],
                    [22,16,18,20,16,14,12], 
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]
LINES_UPPER_BODY = [[12,11,23,24,12], 
                    [22,16,18,20,16,14,12], 
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]
# LINE_MESH_*_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINE_MESH_FULL_BODY = [ [9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15],
                        [24,26],[26,28],[32,30],
                        [23,25],[25,27],[29,31]]
LINE_TEST = [ [12,11],[11,23],[23,24],[24,12]]

COLORS_FULL_BODY = ["middle","right","left",
                    "right","right","right","right","right",
                    "middle","middle","middle","middle",
                    "left","left","left","left","left",
                    "right","right","right","left","left","left"]
COLORS_FULL_BODY = [rgb[x] for x in COLORS_FULL_BODY]
LINE_MESH_UPPER_BODY = [[9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15]
                        ]

# For gesture demo
semaphore_flag = {
        (3,4):'A', (2,4):'B', (1,4):'C', (0,4):'D',
        (4,7):'E', (4,6):'F', (4,5):'G', (2,3):'H',
        (0,3):'I', (0,6):'J', (3,0):'K', (3,7):'L',
        (3,6):'M', (3,5):'N', (2,1):'O', (2,0):'P',
        (2,7):'Q', (2,6):'R', (2,5):'S', (1,0):'T',
        (1,7):'U', (0,5):'V', (7,6):'W', (7,5):'X',
        (1,6):'Y', (5,6):'Z'
}

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)

class BlazeposeDepthai:
    def __init__(self, input_src=None,
                pd_path=POSE_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                lm_path=FULL_BODY_LANDMARK_MODEL,
                lm_score_threshold=0.7,
                full_body=True,
                use_gesture=False,
                use_pose=False,
                smoothing= True,
                filter_window_size=5,
                filter_velocity_scale=10,
                show_3d=False,
                crop=False,
                multi_detection=False,
                output=None,
                internal_fps=15):
        
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.full_body = full_body
        self.use_gesture = use_gesture
        self.use_pose = use_pose
        self.smoothing = smoothing
        self.show_3d = show_3d
        self.crop = crop
        self.multi_detection = multi_detection
        if self.multi_detection:
            print("With multi-detection, smoothing filter is disabled.")
            self.smoothing = False
        self.internal_fps = internal_fps
        
        if input_src == None:
            self.input_type = "internal" # OAK* internal color camera
            self.video_fps = internal_fps # Used when saving the output in a video file. Should be close to the real fps
            video_height = video_width = 1080 # Depends on cam.setResolution() in create_pipeline()
        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            video_height, video_width = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)

        self.nb_kps = 33 if self.full_body else 25

        if self.smoothing:
            self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_kps, 3))
    
        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Rendering flags
        self.show_pd_box = False
        self.show_pd_kps = False
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_scores = False
        self.show_gesture = self.use_gesture
        self.show_pose = self.use_pose
        self.show_fps = True

        if self.show_3d:
            self.vis3d = o3d.visualization.Visualizer()
            self.vis3d.create_window() 
            opt = self.vis3d.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            z = min(video_height, video_width)/3
            self.grid_floor = create_grid([0,video_height,-z],[video_width,video_height,-z],[video_width,video_height,z],[0,video_height,z],5,2, color=(1,1,1))
            self.grid_wall = create_grid([0,0,z],[video_width,0,z],[video_width,video_height,z],[0,video_height,z],5,2, color=(1,1,1))
            self.vis3d.add_geometry(self.grid_floor)
            self.vis3d.add_geometry(self.grid_wall)
            view_control = self.vis3d.get_view_control()
            view_control.set_up(np.array([0,-1,0]))
            view_control.set_front(np.array([0,0,-1]))

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(video_width, video_height)) 

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        self.pd_input_length = 128

        if self.input_type == "internal":
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            # Crop video to square shape (palm detection takes square image as input)
            self.video_size = min(cam.getVideoSize())
            cam.setVideoSize(self.video_size, self.video_size)
            # 
            cam.setFps(self.internal_fps)
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            # Link video output to host for higher resolution
            cam.video.link(cam_out.input)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Pose detection input                 
        if self.input_type == "internal":
            pd_nn.input.setQueueSize(1)
            pd_nn.input.setBlocking(False)
            cam.preview.link(pd_nn.input)
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
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        lm_nn.setNumInferenceThreads(1)
        # Landmark input
        self.lm_input_length = 256
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
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,12)) # 896x12

        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=not self.multi_detection)
        # Non maximum suppression (not needed if best_only is True)
        if self.multi_detection: 
            self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        
        mpu.detections_to_rect(self.regions, kp_pair=[0,1] if self.full_body else [2,3])
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.frame_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                # Key point 0 - mid hip center
                # Key point 1 - point that encodes size & rotation (for full body)
                # Key point 2 - mid shoulder center
                # Key point 3 - point that encodes size & rotation (for upper body)
                if self.full_body:
                    # Only kp 0 and 1 used
                    list_kps = [0, 1]
                else:
                    # Only kp 2 and 3 used for upper body
                    list_kps = [2, 3]
                for kp in list_kps:
                    x = int(r.pd_kps[kp][0] * self.frame_size)
                    y = int(r.pd_kps[kp][1] * self.frame_size)
                    cv2.circle(frame, (x, y), 3, (0,0,255), -1)
                    cv2.putText(frame, str(kp), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Pose score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.frame_size+10), int((r.pd_box[1]+r.pd_box[3])*self.frame_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

   
    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("output_poseflag")[0]
        if region.lm_score > self.lm_score_threshold:  
            self.nb_active_regions += 1

            lm_raw = np.array(inference.getLayerFp16("ld_3d")).reshape(-1,5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the region of
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
            
            # region.landmarks contains the landmarks normalized 3D coordinates in the relative oriented body bounding box
            region.landmarks = lm_raw[:,:3]
            # Calculate the landmark coordinate in square padded image (region.landmarks_padded)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(region.landmarks[:self.nb_kps,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            # A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
            # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
            lm_z = region.landmarks[:self.nb_kps,2:3] * region.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            if self.smoothing:
                lm_xyz = self.filter.apply(lm_xyz)
            region.landmarks_padded = lm_xyz.astype(np.int)
            # If we added padding to make the image square, we need to remove this padding from landmark coordinates
            # region.landmarks_abs contains absolute landmark coordinates in the original image (padding removed))
            region.landmarks_abs = region.landmarks_padded.copy()
            if self.pad_h > 0:
                region.landmarks_abs[:,1] -= self.pad_h
            if self.pad_w > 0:
                region.landmarks_abs[:,0] -= self.pad_w

            if self.use_gesture: self.recognize_gesture(region)

            if self.use_pose: self.recognize_pose(region)


    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:
                
                list_connections = LINES_FULL_BODY if self.full_body else LINES_UPPER_BODY
                lines = [np.array([region.landmarks_padded[point,:2] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
                
                for i,x_y in enumerate(region.landmarks_padded[:,:2]):
                    if i > 10:
                        color = (0,255,0) if i%2==0 else (0,0,255)
                    elif i == 0:
                        color = (0,255,255)
                    elif i in [4,5,6,8,10]:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)
                    cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

                if self.show_3d:
                    points = region.landmarks_abs
                    lines = LINE_MESH_FULL_BODY if self.full_body else LINE_MESH_UPPER_BODY
                    colors = COLORS_FULL_BODY
                    for i,a_b in enumerate(lines):
                        a, b = a_b
                        line = create_segment(points[a], points[b], radius=5, color=colors[i])
                        if line: self.vis3d.add_geometry(line, reset_bounding_box=False)
                    
                    

            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
                        (int(region.pd_box[0] * self.frame_size+10), int((region.pd_box[1]+region.pd_box[3])*self.frame_size+90)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            if self.use_gesture and self.show_gesture:
                cv2.putText(frame, region.gesture, (int(region.pd_box[0]*self.frame_size+10), int(region.pd_box[1]*self.frame_size-50)), 
                        cv2.FONT_HERSHEY_PLAIN, 5, (0,1190,255), 3)
            if self.use_pose and self.show_pose:
                cv2.putText(frame, region.pose, (int(region.pd_box[0]*self.frame_size+10), int(region.pd_box[1]*self.frame_size-50)), 
                        cv2.FONT_HERSHEY_PLAIN, 5, (0,1190,255), 3)
            


          
    def recognize_gesture(self, r):           

        def angle_with_y(v):
            # v: 2d vector (x,y)
            # Returns angle in degree ofv with y-axis of image plane
            if v[1] == 0:
                return 90
            angle = atan2(v[0], v[1])
            return np.degrees(angle)

        # For the demo, we want to recognize the flag semaphore alphabet
        # For this task, we just need to measure the angles of both arms with vertical
        right_arm_angle = angle_with_y(r.landmarks_abs[14,:2] - r.landmarks_abs[12,:2])
        left_arm_angle = angle_with_y(r.landmarks_abs[13,:2] - r.landmarks_abs[11,:2])
        right_pose = int((right_arm_angle +202.5) / 45) 
        left_pose = int((left_arm_angle +202.5) / 45) 
        r.gesture = semaphore_flag.get((right_pose, left_pose), None)

    def recognize_pose(self, r):           

        r.pose = "Pose not detected"


    #################################################################################

        

        pose_embedder = FullBodyPoseEmbedder()

        pose_classifier = PoseClassifier(
                pose_samples_folder='./fitness_poses_csvs_out',
                pose_embedder=pose_embedder,
                top_n_by_max_distance=30,
                top_n_by_mean_distance=10)

        assert r.landmarks_abs.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(r.landmarks_abs.shape)

        print(r.landmarks_abs)
        print(type(r.landmarks_abs))

        r.landmarks_abs = r.landmarks_abs.astype('float32')

        pose_classification = pose_classifier(r.landmarks_abs)

        pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

            # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_filter(pose_classification)

        max_sample = 0
        pose = 0

        for i in pose_classification_filtered.keys():
            if pose_classification_filtered[i]>max_sample:
                pose = i
                max_sample = pose_classification_filtered[i]

        r.pose = pose
                    
    def run(self):

        device = dai.Device(self.create_pipeline())
        device.startPipeline()

        # Define data queues 
        if self.input_type == "internal":
            q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            q_lm_out = device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
            q_lm_in = device.getInputQueue(name="lm_in")
        else:
            q_pd_in = device.getInputQueue(name="pd_in")
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
            q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
            q_lm_in = device.getInputQueue(name="lm_in")

        self.fps = FPS(mean_nb_frames=20)

        seq_num = 0
        nb_pd_inferences = 0
        nb_lm_inferences = 0
        glob_pd_rtrip_time = 0
        glob_lm_rtrip_time = 0
        while True:
            self.fps.update()
             
            if self.input_type == "internal":
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()
                self.frame_size = video_frame.shape[0] # The image is square cropped on the device
                self.pad_w = self.pad_h = 0
            else:
                if self.input_type == "image":
                    vid_frame = self.img
                else:
                    ok, vid_frame = self.cap.read()
                    if not ok:
                        break
                    
                h, w = vid_frame.shape[:2]
                if self.crop:
                    # Cropping the long side to get a square shape
                    self.frame_size = min(h, w)
                    dx = (w - self.frame_size) // 2
                    dy = (h - self.frame_size) // 2
                    video_frame = vid_frame[dy:dy+self.frame_size, dx:dx+self.frame_size]
                else:
                    # Padding on the small side to get a square shape
                    self.frame_size = max(h, w)
                    self.pad_h = int((self.frame_size - h)/2)
                    self.pad_w = int((self.frame_size - w)/2)
                    video_frame = cv2.copyMakeBorder(vid_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(seq_num)
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
                pd_rtrip_time = now()
                q_pd_in.send(frame_nn)
                

                seq_num += 1

            annotated_frame = video_frame.copy()

            # Get pose detection
            inference = q_pd_out.get()
            if self.input_type != "internal": 
                pd_rtrip_time = now() - pd_rtrip_time
                glob_pd_rtrip_time += pd_rtrip_time
            self.pd_postprocess(inference)
            self.pd_render(annotated_frame)
            nb_pd_inferences += 1

            # Landmarks
            self.nb_active_regions = 0
            if self.show_3d:
                self.vis3d.clear_geometries()
                self.vis3d.add_geometry(self.grid_floor, reset_bounding_box=False)
                self.vis3d.add_geometry(self.grid_wall, reset_bounding_box=False)
            for i,r in enumerate(self.regions):
                frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(frame_nn, (self.lm_input_length, self.lm_input_length)))
                if i == 0: lm_rtrip_time = now() # We measure only for the first region
                q_lm_in.send(nn_data)
                
                # Get landmarks
                inference = q_lm_out.get()
                if i == 0: 
                    lm_rtrip_time = now() - lm_rtrip_time
                    glob_lm_rtrip_time += lm_rtrip_time
                    nb_lm_inferences += 1
                self.lm_postprocess(r, inference)
                self.lm_render(annotated_frame, r)
            if self.show_3d:
                self.vis3d.poll_events()
                self.vis3d.update_renderer()
            if self.smoothing and self.nb_active_regions == 0:
                self.filter.reset()

            if self.input_type != "internal" and not self.crop:
                annotated_frame = annotated_frame[self.pad_h:self.pad_h+h, self.pad_w:self.pad_w+w]

            if self.show_fps:
                self.fps.display(annotated_frame, orig=(50,50), size=1, color=(240,180,100))
            cv2.imshow("Blazepose", annotated_frame)

            if self.output:
                self.output.write(annotated_frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('1'):
                self.show_pd_box = not self.show_pd_box
            elif key == ord('2'):
                self.show_pd_kps = not self.show_pd_kps
            elif key == ord('3'):
                self.show_rot_rect = not self.show_rot_rect
            elif key == ord('4'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('5'):
                self.show_scores = not self.show_scores
            elif key == ord('6'):
                self.show_gesture = not self.show_gesture
            elif key == ord('f'):
                self.show_fps = not self.show_fps

        # Print some stats
        print(f"# pose detection inferences : {nb_pd_inferences}")
        print(f"# landmark inferences       : {nb_lm_inferences}")
        if self.input_type != "internal" and nb_pd_inferences != 0: print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
        if nb_lm_inferences != 0:  print(f"Landmark round trip         : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")

        if self.output:
            self.output.release()
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,  
                        help="Path to video or image file to use as input (default: internal camera")
    parser.add_argument('-g', '--gesture', action="store_true", 
                        help="enable gesture recognition")
    parser.add_argument('-ps', '--pose', action="store_true", 
                        help="enable pose recognition")
    parser.add_argument("--pd_m", type=str,
                        help="Path to an .blob file for pose detection model")
    parser.add_argument("--lm_m", type=str,
                        help="Path to an .blob file for landmark model")
    parser.add_argument('-c', '--crop', action="store_true", 
                        help="Center crop frames to a square shape before feeding pose detection model")
    parser.add_argument('-u', '--upper_body', action="store_true", 
                        help="Use an upper body model")
    parser.add_argument('--no_smoothing', action="store_true", 
                        help="Disable smoothing filter")
    parser.add_argument('--filter_window_size', type=int, default=5,
                        help="Smoothing filter window size. Higher value adds to lag and to stability (default=%(default)i)")                    
    parser.add_argument('--filter_velocity_scale', type=float, default=10,
                        help="Smoothing filter velocity scale. Lower value adds to lag and to stability (default=%(default)s)")                    
    parser.add_argument('-3', '--show_3d', action="store_true", 
                        help="Display skeleton in 3d in a separate window (valid only for full body landmark model)")
    parser.add_argument("-o","--output",
                        help="Path to output video file")
    parser.add_argument('--multi_detection', action="store_true", 
                        help="Force multiple person detection (at your own risk)")
    parser.add_argument('--internal_fps', type=int, default=15,
                        help="Fps of internal color camera. Too high value lower NN fps (default=%(default)i)")                    


        

    args = parser.parse_args()

    if not args.pd_m:
        args.pd_m = POSE_DETECTION_MODEL
    if not args.lm_m:
        if args.upper_body:
            args.lm_m = UPPER_BODY_LANDMARK_MODEL
        else:
            args.lm_m = FULL_BODY_LANDMARK_MODEL
    ht = BlazeposeDepthai(input_src=args.input, 
                    pd_path=args.pd_m,
                    lm_path=args.lm_m,
                    full_body=not args.upper_body,
                    smoothing=not args.no_smoothing,
                    filter_window_size=args.filter_window_size,
                    filter_velocity_scale=args.filter_velocity_scale,
                    use_gesture=args.gesture,
                    use_pose=args.pose,
                    show_3d=args.show_3d,
                    crop=args.crop,
                    multi_detection=args.multi_detection,
                    output=args.output,
                    internal_fps=args.internal_fps)
    ht.run()