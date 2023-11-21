# Detect facial landmarks to calcualte facial shape
# to determine which haircut fits you best!

from __future__ import print_function
import cv2 as cv
import argparse


def detectAndDisplay(frame, landmark_detector):
    """
    Detect and plot points on face, when recongnized
    """

    # Make grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Rectangle face detection
    faces = face_cascade.detectMultiScale(frame_gray)

    # Hold point location for facial point landmarks
    landmark_points = []

    for (x, y, w, h) in faces:

        # Plot rectangle around face on image
        frame = cv.rectangle(frame, (x, y), (x+w, y+h),
                             (128, 128, 128), 1)

        # Fit landmark model with gray scale image and faces detection
        _, landmarks = landmark_detector.fit(frame_gray, faces)
        for landmark in landmarks:

            # Plot x/y coords of facial landmarks
            for x, y in landmark[0]:

                # Apppend x,y pair in tuple format
                landmark_points.append((int(x), int(y)))
                cv.circle(frame, (int(x), int(y)), 1, (255, 255, 255), 1)

        # Check if atleast all points have been plotted
        if len(landmark_points) >= 68:

            """
            Need to be added

            Eye brow line
            eyebrow text
            lips line
            lips text 
            """

            # Points to plot lines
            eyes_left = landmark_points[0]
            eyes_right = landmark_points[16]
            chin_left = landmark_points[6]
            chin_right = landmark_points[10]
            nose_left = landmark_points[2]
            nose_right = landmark_points[14]

            # Plots lines
            cv.line(frame, eyes_left, eyes_right, (0, 0, 0), 2)
            cv.putText(frame, "Eye Line", (eyes_left[0], (eyes_left[1] - 10)),
                       cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
            cv.line(frame, chin_left, chin_right, (0, 0, 0), 2)
            cv.putText(frame, "Chin Line", (chin_left[0], (chin_left[1] - 10)),
                       cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
            cv.line(frame, nose_left, nose_right, (0, 0, 0), 2)
            cv.putText(frame, "Nose Line", (nose_left[0], (nose_left[1] - 10)),
                       cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
    cv.imshow("Capture Face Detection", frame)


# Landmark detection data
LBFmodel = "data/lbfmodel.yaml"
# Face detection data
haarcascade = "data/haarcascade_frontalface_alt2.xml"


# Expected options
parser = argparse.ArgumentParser(description="Code for facial recognition")
parser.add_argument(
    '--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

face_cascade = cv.CascadeClassifier(haarcascade)

camera_device = args.camera

# Land mark detector model
landmark_detector = cv.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Load camera
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# Loop passing in frame
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    # Passed in a frame flipped on the x axis to be more 'natrual'
    detectAndDisplay(cv.flip(frame, 1), landmark_detector)
    if cv.waitKey(10) == 27:
        break
