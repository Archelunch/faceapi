import os
import datetime
from json import dumps, loads

from tools import decode_image, convert_avi_to_mp4, createFile

import cv2
import dlib
import imutils
import vptree
import numpy as np

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

COLOUR_CORRECT_BLUR_FRAC = 0.6

face_cascade = cv2.CascadeClassifier('files\\haarcascade_frontalface_alt.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('files\\shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('files\\dlib_face_recognition_resnet_model_v1.dat')
predictor = dlib.shape_predictor('files\\shape_predictor_68_face_landmarks.dat')


user_tree = None


def get_user(img):
    return user_tree.get_n_nearest_neighbors({'descriptor':extract_descriptor(img)}, 1)


def extract_descriptor(img):
    face_face = []
    dets_webcam = detector(img, 1)
    for k, d in enumerate(dets_webcam):
        shape = sp(img, d)
        face_face = facerec.compute_face_descriptor(img, shape)
    return face_face


def detect_faces(frame):
    frame = imutils.resize(frame)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x*4, y*4), (x*4 + w*4, y*4 + h*4), (0, 255, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [frame, faces]


def euclide(x, y):

    return np.linalg.norm(np.asarray(x['descriptor'][0]) - np.asarray(y['descriptor'][0]))


def cosine(x, y):
    try:
        t1 = np.asarray(x['descriptor'])
        t2 = np.asarray(y['descriptor'])
        return 1 - (np.dot(t1, t2) / (np.sqrt(np.dot(t1, t1)) * np.sqrt(np.dot(t2, t2))))
    except:
        return 100

def sorter(inp):
    return inp["mn"]


def compare_photos(x, y, alg):
    if not alg in algs:
        raise ValueError('Такого алгоритма нет')
    return algs[alg](
        {'descriptor':extract_descriptor(decode_image(x))},
        {'descriptor':extract_descriptor(decode_image(y))}
    )


def get_landmarks(im, alone):
    rects = detector(im, 1)
    if len(rects) > 1 and alone==True:
        raise ValueError('Many faces')
    if len(rects) == 0:
        raise ValueError('No faces')
    if alone==True:
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    else:
        ans = []
        for i in range(len(rects)):
            t = np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
            ans.append(t)
        return ans

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float)
    points2 = points2.astype(np.float)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname, alone):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im, alone)

    return im, s


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float) * im1_blur.astype(np.float) /
                                                im2_blur.astype(np.float))


def swap_face(p1, p2, username):
    im1, landmarks123 = read_im_and_landmarks(p1, False)
    im2, landmarks2 = read_im_and_landmarks(p2, True)
    for landmarks1 in landmarks123:
        M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                       landmarks2[ALIGN_POINTS])

        mask = get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)
        combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)

        warped_im2 = warp_im(im2, M, im1.shape)
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
        im1 = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    name = 'out{0}.png'.format(datetime.datetime.now().strftime("%H%M%S%B%d%Y"))
    print(os.path.join(os.getcwd(), os.path.join(os.path.join(username, 'media'), name)))
    cv2.imwrite(os.path.join(os.getcwd(), os.path.join(os.path.join(username, 'media'), name)), output_im)
    return name

def swap_video(p1, p2, id):
    cap = cv2.VideoCapture(p1)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = cap.read()
    cap.release()
    cap = cv2.VideoCapture(p1)
    out = cv2.VideoWriter('archive\\output{0}.avi'.format(id),fourcc, 20, frame.shape[:2])
    while(cap.isOpened()):
         ret, frame = cap.read()
         cv2.imwrite('archive\\mishaoutput{0}.png'.format(id), frame)
         if ret:
             try:
                 swap_face('archive\\mishaoutput{0}.png'.format(id), p2, id)
                 frame = cv2.imread('archive\\output{0}.png'.format(id))
                 out.write(frame)
             except Exception as e:
                 frame = cv2.imread('archive\\mishaoutput{0}.png'.format(id))
                 out.write(frame)
             #cv2.imshow('Frame',frame)
         else:
             break
    convert_avi_to_mp4('archive\\output{0}.avi'.format(id), 'archive\\output{0}'.format(id))
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def loadDescriptors(name):
    with open(name, 'r') as file:
        savedDescriptors = loads(file.read())
        file.close()
    return savedDescriptors


algs = {'cosine':cosine, 'euclide':euclide}