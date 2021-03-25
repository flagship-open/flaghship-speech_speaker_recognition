import requests

# Set IP address of server
ip_address = 'http://165.132.56.182:8888/Identification_Request'
sess = requests.Session()

# Inputs
re_register = 'False'
identify = 'False'
file_path = './test_data/aaa/aaa@yonsei.ac.kr-0.mp4'

# Perform registration or identification
r = sess.post(ip_address, data={'re_register': re_register, \
                                'identify': identify, 'file_path': file_path})
if r.status_code == 200:
    print(r.content)
else:
    print('fail :(')
