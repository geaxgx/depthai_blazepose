# Blazepose tracking with DepthAI

Running Google Mediapipe body pose tracking models on [DepthAI](https://docs.luxonis.com/en/gen2/) hardware (OAK-1, OAK-D, ...)

For OpenVINO version, please visit : [openvino_blazepose](https://github.com/geaxgx/openvino_blazepose)

![Demo](img/taichi.gif)
## Install

Install the python packages DepthAI, Opencv, open3d with the following command:

```python3 -m pip install -r requirements.txt```

## Run

**Usage:**

```
> python BlazeposeDepthai.py -h
usage: BlazeposeDepthai.py [-h] [-i INPUT] [-g] [--pd_m PD_M] [--lm_m LM_M]
                           [-c] [-u] [--no_smoothing]
                           [--filter_window_size FILTER_WINDOW_SIZE]
                           [--filter_velocity_scale FILTER_VELOCITY_SCALE]
                           [-3] [-o OUTPUT] [--multi_detection]
                           [--internal_fps INTERNAL_FPS]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to video or image file to use as input (default:
                        internal camera
  -g, --gesture         enable gesture recognition
  --pd_m PD_M           Path to an .blob file for pose detection model
  --lm_m LM_M           Path to an .blob file for landmark model
  -c, --crop            Center crop frames to a square shape before feeding
                        pose detection model
  -u, --upper_body      Use an upper body model
  --no_smoothing        Disable smoothing filter
  --filter_window_size FILTER_WINDOW_SIZE
                        Smoothing filter window size. Higher value adds to lag
                        and to stability (default=5)
  --filter_velocity_scale FILTER_VELOCITY_SCALE
                        Smoothing filter velocity scale. Lower value adds to
                        lag and to stability (default=10)
  -3, --show_3d         Display skeleton in 3d in a separate window (valid
                        only for full body landmark model)
  -o OUTPUT, --output OUTPUT
                        Path to output video file
  --multi_detection     Force multiple person detection (at your own risk)
  --internal_fps INTERNAL_FPS
                        Fps of internal color camera. Too high value lower NN
                        fps (default=15)

```
**Examples :**

- To use default internal color camera as input :

    ```python3 BlazeposeDepthai.py```

- To use a file (video or image) as input :

    ```python3 BlazeposeDepthai.py -i filename```

- To show the skeleton in 3D (note that it will lower the FPS):

    ```python3 BlazeposeDepthai.py -3```

- To demo gesture recognition :

    ```python3 BlazeposeDepthai.py -g```

    This is a very basic demo that can read semaphore alphabet by measuring arm angles.

![Gesture recognition](img/semaphore.gif)

- By default, a temporal filter smoothes the landmark positions. You can tune the smoothing with the arguments *--filter_window_size* and *--filter_velocity_scale*. Use *--no_smoothing* to disable the filter.

Use keypress between 1 and 6 to enable/disable the display of body features (bounding box, landmarks, scores, gesture,...), 'f' to show/hide FPS, spacebar to pause, Esc to exit.



## The models 
You can directly find the model files (.xml and .bin) under the 'models' directory. Below I describe how to get the files in case you need to regenerate the models.

1) Clone this github repository in a local directory (DEST_DIR)
2) In DEST_DIR/models directory, download the source tflite models from Mediapipe:
* [Pose detection model](https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection.tflite)
* [Full-body pose landmark model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_full_body.tflite)
* [Upper-body pose landmark model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body.tflite)
3) Install the amazing [PINTO's tflite2tensorflow tool](https://github.com/PINTO0309/tflite2tensorflow). Use the docker installation which includes many packages including a recent version of Openvino.
3) From DEST_DIR, run the tflite2tensorflow container:  ```./docker_tflite2tensorflow.sh```
4) From the running container: 
```
cd resources/models
./convert_models.sh
```
The *convert_models.sh* converts the tflite models in tensorflow (.pb), then converts the pb file into Openvino IR format (.xml and .bin), and finally converts the IR files in MyriadX format (.blob).

By default, the number of SHAVES associated with the blob files is 4. In case you want to generate new blobs with different number of shaves, you can use the script *gen_blob_shave.sh*:
```
# Example: to generate blobs for 6 shaves
./gen_blob_shave.sh -m pd -n 6        # will generate pose_detection_sh6.blob
./gen_blob_shave.sh -m lm_full -n 6   # will generate pose_landmark_full_body_sh6.blob
./gen_blob_shave.sh -m lm_up -n 6     # will generate pose_landmark_upper_body_sh6.blob
```


**Explanation about the Model Optimizer params :**
- The preview of the OAK-* color camera outputs BGR [0, 255] frames . The original tflite pose detection model is expecting RGB [-1, 1] frames. ```--reverse_input_channels``` converts BGR to RGB. ```--mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]``` normalizes the frames between [-1, 1].
- The images which are fed to the landmark model are built on the host in a format similar to the OAK-* cameras (BGR [0, 255]). The original landmark models are expecting RGB [0, 1] frames. Therefore, the following arguments are used ```--reverse_input_channels --scale_values [255.0, 255.0, 255.0]```


## Credits
* [Google Mediapipe](https://github.com/google/mediapipe)
* Katsuya Hyodo a.k.a [Pinto](https://github.com/PINTO0309), the Wizard of Model Conversion !
* [Tai Chi Step by Step For Beginners Training Session 4](https://www.youtube.com/watch?v=oawZ_7wNWrU&ab_channel=MasterSongKungFu)
* [Semaphore with The RCR Museum](https://www.youtube.com/watch?v=DezaTjQYPh0&ab_channel=TheRoyalCanadianRegimentMuseum)