from __future__ import print_function
import cv2 as cv
import argparse
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt
from pylab import rcParams


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Face detection
    faces = face_cascade.detectMultiScale(frame_gray)
    print(faces)
    for (x, y, w, h) in faces:

        frame = cv.rectangle(frame, (x, y), (x+w, y+h),
                             (255, 0, 255), 1)

    cv.imshow("Capture Face Detection", frame)


parser = argparse.ArgumentParser(description="Code for facial recognition")
parser.add_argument("--face_cascade", help="Path to face cascade",
                    default="data/lbpcascade_frontalface.xml")
parser.add_argument(
    '--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    print(frame)
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
