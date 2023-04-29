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
        'GO': "go "+' '.join(map(str, val)) if type(val)==list else "",
        'TAKEOFF': "takeoff ",
    }

    if command in command_dict.keys():
        return command_dict[command]
    
    return command

def relay_command_to_drone(command_list=[], drone=None, socket=None, is_remote=False):

    takeoff_message = get_command("TAKEOFF")
    # move_message = get_command("GO", [50,50,0,25])

    command_list = [get_command("GO", command) for command in command_list]


    if is_remote:
        if socket:
            socket.send_socket_message(takeoff_message)
            for command in command_list:
                socket.send_socket_message(command)
                print("Sending Message: ", command)

            socket.close()
        else:
            print("Unable to find server")
            
    else:
        if drone:
            drone.takeoff()
            for command in command_list:
                drone.send_control_command(command)
            
            drone.land()

def parallelize(command_list, drone, sock):
    '''
    send commands to drones in parallel threads
    Currently works only for one wifi connected drone and one server connected drone
    '''
    thread1 = threading.Thread(target=relay_command_to_drone, kwargs={'command_list': command_list, 'drone':None, 'socket': sock, 'is_remote':True})
    thread2 = threading.Thread(target=relay_command_to_drone, kwargs={'command_list': command_list, 'drone':drone, 'socket': None, 'is_remote':False})

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    
    

def Main():
    print("Client Started")

    drone = tello.Tello()
    drone.connect()

    sock = Socket(client_ip = '169.254.222.143', \
                  server_ip = '169.254.176.231', port = 4000)
            
    # command_list = [[50,50,0,25], [10,10,0,25]]
    import numpy as np
    command_list = np.load('command_list.npy').astype(np.int32)

    # noise = np.random.randint(1,5,command_list.shape)
    command_list[:,:3,] = np.abs(command_list[:,:3,]*4)
    # command_list = command_list+noise


    # parallelize(command_list, drone, sock)
    parallelize(command_list.tolist(), drone, sock)
    
    
    # battery_level = drone.get_battery()
    # if battery_level<50:
    #     print(f'not enough battery, {battery_level}')
    #     exit(1)

    # drone.takeoff()
    # tello.send_keepalive()




if __name__=='__main__':
    Main()
    # pass