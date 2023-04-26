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
        'GO': "go "+' '.join(map(str, val)) if type(val)==list else "",\
        'TAKEOFF': "takeoff ",
    }

    if command in command_dict.keys():
        return command_dict[command]
    
    return command

def send_socket_message(sock, message):
    server = ('169.254.176.231', 4000)
    sock.sendto(message.encode('utf-8'), server)
    data, addr = sock.recvfrom(1024)
    data = data.decode('utf-8')
    print("Received from server: " + data)
    sock.close()
    return data


def bind_socket():
    host='169.254.222.143' #client ip
    port = 4000
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host,port))
    print('binding')
    return s

def relay_command_to_drone(drone=None, is_remote=False):

    takeoff_message = get_command("TAKEOFF")
    move_message = get_command("GO", [50,50,0,25])

    if is_remote:
        s = bind_socket()
        print("Sending Message: ", 'command')
        send_socket_message(s, takeoff_message)
        send_socket_message(s, move_message)
    else:
        if drone:
            drone.takeoff()
            drone.send_control_command(move_message)


def Main():
    print("Client Started")

    # sleep(2)    
    
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