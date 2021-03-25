import requests
from glob import glob
import os
import json

ip_address = 'http://165.132.56.182:8888/Identification_Request'
sess = requests.Session()

# Enrollment
speaker_list = glob('./test_data/*/*-0.mp4')
print(speaker_list)
for spk in speaker_list: 
    r = sess.post(ip_address, data={'re_register': 'False', 'identify': 'False', 'file_path': spk})
    if r.status_code == 200:
        print(r.content)
    else:
        print('fail :(')

# Test
hit_acc = 0
with open(test_list, 'r') as fp:
    file_list = fp.readlines()
for i in range(len(file_list)):
    file_path = file_list[i].replace('\n', '')
    r = sess.post(ip_address, data={'re_register': 'False', 'identify': 'True', 'file_path': file_path})
    spk_name = os.path.basename(file_list[i]).split('-')[0]

    est_ID = json.loads(r.content.decode('utf-8'))['100001'].split(': ')[-1]
    print(est_ID, spk_name)

    if est_ID == spk_name:
        hit_acc += 1

print('Accuracy: ', hit_acc / len(file_list) * 100)
