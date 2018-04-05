from json import dumps, loads
import sys

import dlib
from requests import get as rget
from skimage import io, color


def updateDescriptors(desc, name):
    global savedDescriptors
    desc = list(desc)
    if len(desc) == 128:
        record = {
            "name": name,
            "descriptor": desc
        }
        try:
            savedDescriptors.append(record)
            saveDescriptors(savedDescriptors)
        except AttributeError:
            print("error")


def loadDescriptors(name):
    with open("runet.json", 'r') as file:
        savedDescriptors = loads(file.read())
        file.close()
    return savedDescriptors


def saveDescriptors(savedDescriptors):
    with open("runet.json", 'w') as file:
        file.write(dumps(savedDescriptors, indent=4))
        file.close()


def img_parse(start=100, end=1024840):
    global savedDescriptors
    print("Starting")
    count = 0
    for i in range(start, end):
        for desc in savedDescriptors:
            if i == int(desc['name']):
                break
        else:
            if i < 10000:
                pn = '0'
            else:
                t = str(i)
                pn = t[:len(t)-4]
            page = 'https://runet-id.com/files/photo/{0}/{1}_200.jpg'.format(pn, i)
            r = rget(page, stream=True)
            if r.status_code == 200:
                count+=1
                f = open('img.jpg', 'wb')
                f.write(r.content)
                f.close()
                img = color.gray2rgb(io.imread(f.name))
                updateDescriptors(extract_descriptor(img), str(i))
                sys.stdout.write("\r%d%%" % i)
                sys.stdout.flush()
    #savedDescriptors = sorted(savedDescriptors, key=lambda k: k['name'])
    print("Finished")


def extract_descriptor(img):
    try:
        face_face = []
        dets_webcam = detector(img, 1)
        for k, d in enumerate(dets_webcam):
            shape = sp(img, d)
            face_face = facerec.compute_face_descriptor(img, shape)
        return face_face
    except RuntimeError:
        print('error')


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('files\\shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('files\\dlib_face_recognition_resnet_model_v1.dat')
try:
    savedDescriptors = loadDescriptors()
    savedDescriptors = sorted(savedDescriptors, key=lambda k: k['name'])
except:
    savedDescriptors = list()
