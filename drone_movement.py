#!/usr/bin/env python

from Socket import Socket
import threading
from djitellopy import tello
from time import sleep

def get_command(command, val=None):
    command_dict = {
        'LEFT-RIGHT': f"rc {val} 0 0 0",    #-1 if left
        'FRONT-BACK': f"rc 0 {val} 0 0",    #-1 if back
        'UP': f"rc 0 0 {val} 0",            #-1 if down
        'YAW-UP-DOWN': f"rc 0 0 0 {val}",   #-1 if yaw down
        'CLOCKWISE': f"cw {val}",
        'COUNTER-CLOCKWISE': f"ccw {val}",
        'BATTERY': "battery?",
        'POS-LEFT': f"rc {val} 0 0 0",
        'GO': "go "+' '.join(map(str, val)) if type(val)==list else "",\
        'TAKEOFF': "takeoff ",
    }

    if command in command_dict.keys():
        return command_dict[command]
    
    return command

def relay_command_to_drone(drone=None, is_remote=False):

    takeoff_message = get_command("TAKEOFF")
    move_message = get_command("GO", [50,50,0,25])

    if is_remote:
        sock = Socket(client_ip = '169.254.222.143', server_ip = '169.254.176.231', \
                      port = 4000)
        print("Sending Message: ", 'command')
        sock.send_socket_message(takeoff_message)
        sock.send_socket_message(move_message)

        sock.close()
    else:
        if drone:
            drone.takeoff()
            drone.send_control_command(move_message)


def Main():
    print("Client Started")

    drone = tello.Tello()
    drone.connect()
    
    # battery_level = drone.get_battery()
    # if battery_level<50:
    #     print(f'not enough battery, {battery_level}')
    #     exit(1)

    # drone.takeoff()
    # tello.send_keepalive()


    thread1 = threading.Thread(target=relay_command_to_drone, kwargs={'drone':None, 'is_remote':True})
    thread2 = threading.Thread(target=relay_command_to_drone, kwargs={'drone':drone, 'is_remote':False})

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
    
    
    drone.land()



if __name__=='__main__':
    Main()