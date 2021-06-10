"""
This file is the template of the scripting node source code in edge mode
Substitution is made in BlazeposeDepthaiEdge.py
"""

import marshal
from math import sin, cos, atan2, pi, hypot, degrees, floor

${_TRACE} ("Starting manager script node")

def send_result(buf, type, lm_score=0, rect_center_x=0, rect_center_y=0, rect_size=0, rotation=0, lms=0):
    global marshal
    # type : 0, 1 or 2
    #   0 : pose detection only (detection score < threshold)
    #   1 : pose detection + landmark regression
    #   2 : landmark regression only (ROI computed from previous landmarks)
    result = dict([("type", type), ("lm_score", lm_score), ("rotation", rotation),
            ("rect_center_x", rect_center_x), ("rect_center_y", rect_center_y), ("rect_size", rect_size), ("lms", lms)])
    result_serial = marshal.dumps(result)
    ${_TRACE} ("len result:"+str(len(result_serial)))
    
    buf.getData()[:] = result_serial  
    node.io['host'].send(buf)
    ${_TRACE} ("Manager sent result to host")

def rr2img(rrn_x, rrn_y, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, sin_rot, cos_rot):
    # Convert a point (rrn_x, rrn_y) expressed in normalized rotated rectangle (rrn)
    # into (X, Y) expressed in normalized image (sqn)
    # global rect_center_x, rect_center_y, rect_size, cos_rot, sin_rot
    # cos_rot = cos(rotation)
    # sin_rot = sin(rotation)
    X = sqn_rr_center_x + sqn_rr_size * ((rrn_x - 0.5) * cos_rot + (0.5 - rrn_y) * sin_rot)
    Y = sqn_rr_center_y + sqn_rr_size * ((rrn_y - 0.5) * cos_rot + (rrn_x - 0.5) * sin_rot)
    return X, Y

# send_new_frame_to_branch defines on which branch new incoming frames are sent
# 1 = pose detection branch 
# 2 = landmark branch
send_new_frame_to_branch = 1

# Predefined buffer variables used for sending result to host
buf1 = Buffer(109)
buf2 = Buffer(1884)
buf3 = Buffer(113)

next_roi_lm_idx = 33*5

cfg_pre_pd = ImageManipConfig()
cfg_pre_pd.setResizeThumbnail(224, 224, 0, 0, 0)

while True:
    if send_new_frame_to_branch == 1: # Routing frame to pd
        node.io['pre_pd_manip_cfg'].send(cfg_pre_pd)
        ${_TRACE} ("Manager sent thumbnail config to pre_pd manip")
        # Wait for pd post processing's result 
        detection = node.io['from_post_pd_nn'].get().getLayerFp16("result")
        ${_TRACE} ("Manager received pd result: "+str(detection))
        pd_score, sqn_rr_center_x, sqn_rr_center_y, sqn_scale_x, sqn_scale_y = detection
        scale_center_x = sqn_scale_x - sqn_rr_center_x
        scale_center_y = sqn_scale_y - sqn_rr_center_y
        sqn_rr_size = 2 * ${_rect_transf_scale} * hypot(scale_center_x, scale_center_y)
        rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
        rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))
        if pd_score < ${_pd_score_thresh}:
            send_result(buf1, 0)
            continue

    # Routing frame to lm

    # Tell pre_lm_manip how to crop body region 
    rr = RotatedRect()
    rr.center.x    = sqn_rr_center_x
    rr.center.y    = sqn_rr_center_y * ${_height_ratio} - ${_pad_h_norm}
    rr.size.width  = sqn_rr_size
    rr.size.height = sqn_rr_size * ${_height_ratio}
    rr.angle       = degrees(rotation)
    cfg = ImageManipConfig()
    cfg.setCropRotatedRect(rr, True)
    cfg.setResize(256, 256)
    node.io['pre_lm_manip_cfg'].send(cfg)
    ${_TRACE} ("Manager sent config to pre_lm manip")

    # Wait for lm's result
    lm_result = node.io['from_lm_nn'].get()
    ${_TRACE} ("Manager received result from lm nn")
    lm_score = lm_result.getLayerFp16("output_poseflag")[0]
    if lm_score > ${_lm_score_thresh}:
        lms = lm_result.getLayerFp16("ld_3d")
        send_result(buf2, send_new_frame_to_branch, lm_score, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation, lms)
        if not ${_force_detection}:
            send_new_frame_to_branch = 2 
            # Calculate the ROI for next frame
            # rrn_ : normalized [0:1] coordinates in rotated rectangle coordinate systems 
            rrn_rr_center_x = lms[next_roi_lm_idx] / 256
            rrn_rr_center_y = lms[next_roi_lm_idx+1] / 256
            rrn_scale_x = lms[next_roi_lm_idx+5] / 256
            rrn_scale_y = lms[next_roi_lm_idx+6] / 256
            sin_rot = sin(rotation)
            cos_rot = cos(rotation)
            sqn_scale_x, sqn_scale_y = rr2img(rrn_scale_x, rrn_scale_y, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, sin_rot, cos_rot)
            sqn_rr_center_x, sqn_rr_center_y = rr2img(rrn_rr_center_x, rrn_rr_center_y, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, sin_rot, cos_rot)
            scale_center_x = sqn_scale_x - sqn_rr_center_x
            scale_center_y = sqn_scale_y - sqn_rr_center_y
            sqn_rr_size = 2 * ${_rect_transf_scale} * hypot(scale_center_x, scale_center_y) 
            rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
            rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))          
    else:
        send_result(buf3, send_new_frame_to_branch, lm_score)
        send_new_frame_to_branch = 1