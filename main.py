from __future__ import print_function
import cv2 as cv
import argparse
print("OpenCV version:", cv.__version__)


def detectAndDisplay(frame, landmark_detector):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Face detection
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x, y, w, h) in faces:

        frame = cv.rectangle(frame, (x, y), (x+w, y+h),
                             (255, 0, 255), 1)
        _, landmarks = landmark_detector.fit(frame_gray, faces)
        for landmark in landmarks:
            for x, y in landmark[0]:
                # display landmarks on "image_cropped"
                # with white colour in BGR and thickness 1
                cv.circle(frame, (int(x), int(y)), 1, (255, 255, 255), 1)

    cv.imshow("Capture Face Detection", frame)


# save facial landmark detection model's name as LBFmodel
LBFmodel = "data/lbfmodel.yaml"
# save face detection algorithm's name as haarcascade
haarcascade = "data/haarcascade_frontalface_alt2.xml"


parser = argparse.ArgumentParser(description="Code for facial recognition")
parser.add_argument("--face_cascade", help="Path to face cascade",
                    default="data/haarcascade_frontalface_alt2.xml")
parser.add_argument(
    '--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

face_cascade = cv.CascadeClassifier(haarcascade)

camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

landmark_detector = cv.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

while True:
    ret, frame = cap.read()
    print(frame)
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame, landmark_detector)
    if cv.waitKey(10) == 27:
        break
