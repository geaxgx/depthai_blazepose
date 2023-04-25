#!/usr/bin/env python

import socket
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
        
    }
    if command in command_dict.keys():
        return command_dict[command]
    return command

def Send_Message(message):
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

    message = get_command("LEFT-RIGHT", 25)

    print("Sending Message: ", message)
    response = Send_Message(message)
    
    sleep(2)
    
    drone.move_right(100)
    

    drone.land()



if __name__=='__main__':
    Main()