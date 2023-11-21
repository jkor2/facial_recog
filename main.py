# Detect facial landmarks to calcualte facial shape
# to determine which haircut fits you best!

from __future__ import print_function
import cv2 as cv
import argparse


class Main:
    def __init__(self) -> None:
        self._LBFModel = "data/lbfmodel.yaml"  # LBF Modek
        self._haarcascade = "data/haarcascade_frontalface_alt2.xml"  # Training Data
        self._parser = argparse.ArgumentParser(
            description="Code for facial recognition")
        self._args = None
        self._face_cascade = None
        self._landmark_detector = None

    def run_dectection(self):
        """
        Begin facial detection
        """

        # Args
        self._parser.add_argument(
            '--camera', help='Camera device number.', type=int, default=0)
        self._args = self._parser.parse_args()

        # Detection Data
        self._face_cascade = cv.CascadeClassifier(self._haarcascade)

        # Camera with args
        init_camera = self._args.camera

        # Facial Landmark Model
        self._landmark_detector = cv.face.createFacemarkLBF()
        self._landmark_detector.loadModel(self._LBFModel)

        # Capture
        capture_user = cv.VideoCapture(init_camera)

        # If camera fails attempt to open
        if not capture_user.isOpened():
            print("Error capturing user")
            exit(0)

        while True:
            ret, frame = capture_user.read()
            # Confrim frame is valid
            if frame is None:
                print("Error - No frames captured")
                break

            # Call detect and display method
            self.detect_and_display(cv.flip(frame, 1), self._landmark_detector)
            if cv.waitKey(10) == 27:
                break

    def detect_and_display(self, frame, landmark_detector):
        """
        Dection of face and its landmarks, visual plots
        """

        # Make grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # Rectangle face detection
        faces = self._face_cascade.detectMultiScale(frame_gray)

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


run = Main()
run.run_dectection()
