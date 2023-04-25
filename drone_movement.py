#!/usr/bin/env python

import socket
import threading
from djitellopy import tello
from time import sleep

def get_command(command, val=None):
    command_dict = {
        'LEFT-RIGHT': f"rc {val} 0 0 0",
        # 'RIGHT': "rc 0 1 0 0",
        'FRONT-BACK': f"rc 0 {val} 0 0",
        # 'BACK': "rc -1 0 0 0",
        'UP': f"rc 0 0 {val} 0",
        # 'DOWN': "rc 0 0 -1 0",
        'YAW-UP-DOWN': f"rc 0 0 0 {val}",
        # 'YAW_DOWN': "rc 0 0 0 -1",
        'CLOCKWISE': f"cw {val}",
        'COUNTER-CLOCKWISE': f"ccw {val}",
        'BATTERY': "battery?",
        'POS-LEFT': f"rc {val} 0 0 0",
        'GO': "go "+' '.join(map(str, val)) if type(val)==list else "",
    }

    if command in command_dict.keys():
        return command_dict[command]
    
    return command

def send_socket_message(message):
    host='169.254.222.143' #client ip
    port = 4000
    
    server = ('169.254.176.231', 4000)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host,port))

    s.sendto(message.encode('utf-8'), server)
    data, addr = s.recvfrom(1024)
    data = data.decode('utf-8')
    print("Received from server: " + data)
    s.close()
    return data

def relay_command_to_drone(command, drone=None, is_remote=False):
    if is_remote:
        print("Sending Message: ", command)
        response = send_socket_message(command)
    else:
        if drone:
            drone.send_control_command(command)


def Main():
    print("Client Started")

    # sleep(2)    
    
    drone = tello.Tello()
    drone.connect()
    
    # battery_level = drone.get_battery()
    # if battery_level<50:
    #     print(f'not enough battery, {battery_level}')
    #     exit(1)

    drone.takeoff()
    # tello.send_keepalive()

    message = get_command("GO", [50,50,0,25])

    thread1 = threading.Thread(target=relay_command_to_drone, args=[message], kwargs={'drone':None, 'is_remote':True})
    thread2 = threading.Thread(target=relay_command_to_drone, args=[message], kwargs={'drone':drone, 'is_remote':False})

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
    
    
    drone.land()



if __name__=='__main__':
    Main()