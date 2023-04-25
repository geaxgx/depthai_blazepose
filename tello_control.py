from djitellopy import Tello
import cv2
import time 

tello = Tello()
# print(tello)
tello.connect()

# tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()

while True:
    img = frame_read.frame
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break
    elif key == ord('a'):
        tello.send_rc_control(1,0,0,0)
        time.sleep(1)
        tello.send_rc_control(0,0,0,0)
    elif key == ord('a'):
        tello.send_rc_control(-1,0,0,0)
        time.sleep(1)
        tello.send_rc_control(0,0,0,0)
    elif key == ord('s'):
        tello.send_rc_control(0,-1,0,0)
        time.sleep(1)
        tello.send_rc_control(0,0,0,0)
    elif key == ord('w'):
        tello.send_rc_control(0,1,0,0)
        time.sleep(1)
        tello.send_rc_control(0,0,0,0)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.send_rc_control(0,0,1,0)
        time.sleep(1)
        tello.send_rc_control(0,0,0,0)
    elif key == ord('f'):
        tello.send_rc_control(0,0,-1,0)
        time.sleep(1)
        tello.send_rc_control(0,0,0,0)
        
tello.land()
