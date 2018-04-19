import random
import os
import pathlib

import numpy as np
import cv2
from skimage.color import gray2rgb

random = random.SystemRandom()
_dirs = ['datasets', 'media']
serv_path = os.getcwd()

def _get_random_string(length=12,
                      allowed_chars='abcdefghijklmnopqrstuvwxyz'
                                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    return ''.join(random.choice(allowed_chars) for i in range(length))
 
def get_secret_key():
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'
    return _get_random_string(50, chars)


def decode_image(file):
    return gray2rgb(cv2.imdecode(np.fromstring(file, np.uint8), 0))
    

def check_file(file):
    if file.filename == "":
        return False
    return True


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4".format(input = avi_file_path, output = output_name)).close()
    return True

def mkdir(folder_name):
    user_path = os.path.join(serv_path, folder_name)
    pathlib.Path(user_path).mkdir(parents=True, exist_ok=True)
    for d in _dirs:
        pathlib.Path(os.path.join(user_path, d)).mkdir(parents=True, exist_ok=True)
    return True


def isExist(username, filename, folder=_dirs[1]):
    return os.path.isfile(os.path.join(serv_path, os.path.join(username, os.path.join(folder, filename))))


def getFile(username, filename, folder=_dirs[1]):
    return os.path.join(serv_path, os.path.join(username, os.path.join(folder, filename)))


def createFile(username, file, folder=_dirs[1]):
    pt = os.path.join(serv_path, os.path.join(username, folder))
    cv2.imwrite(os.path.join(pt, file.filename+'.png'), cv2.imdecode(np.fromstring(file.read(), np.uint8), 1))
    return os.path.join(pt, file.filename+'.png')

