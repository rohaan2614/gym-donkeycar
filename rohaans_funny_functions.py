import socket

def is_port_in_use(port=9091, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    
# print(is_port_in_use())