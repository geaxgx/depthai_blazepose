import socket

class Socket():

    # host='169.254.222.143' #client ip
    socket = None

    def __init__(self, client_ip='169.254.222.143', server_ip='169.254.176.231', port=4000) -> None:
        self.client = client_ip
        self.server = server_ip
        self.port = port
        self.socket = self.bind_socket()

    def bind_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.client,self.port))
        print('binding')
        return s


    def send_socket_message(self, message):
        if self.socket is None:
            print('No Socket found')
            return None
        
        self.socket.sendto(message.encode('utf-8'), (self.server, self.port))
        
        data, addr = self.socket.recvfrom(1024)
        data = data.decode('utf-8')
        
        print("Received from server: " + data)
        
        return data
    
    def close(self):
        if self.socket is None:
            print('No Socket found')
            return
        self.socket.close()