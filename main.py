# Detect facial landmarks to calcualte facial shape
# to determine which haircut fits you best!
# Need to come up with ratio to determine face shape

"""
Need to further distinguish rectangle and square, jaw angle would be benefical
Need to look further into round and oval classification
Toruble detecting faceshape with beard
"""

from __future__ import print_function
import cv2 as cv
import argparse


class Main:
    """
    Main class, handles all operations of facial detection
    and face shape calcaultions
    """

    def __init__(self) -> None:
        self._LBFModel = "data/lbfmodel.yaml"  # LBF Model
        self._haarcascade = "data/lbpcascade_frontalface.xml"  # Training Data
        self._parser = argparse.ArgumentParser(
            description="Code for facial recognition")
        self._args = None
        self._face_cascade = None
        self._landmark_detector = None
        self._image = "faces/round/round-test.png"

    def run_detection_stillshot(self):
        """
        Begin face detecion on single image
        """

        # Creaste cascade and landmark detector
        self.create_face_cascade()
        self.create_LM_detector()

        if self._image is None:
            return False

        image = cv.imread(self._image)

        # Resize the image to a specific width (e.g., 800 pixels)
        target_width = 800
        ratio = target_width / image.shape[1]
        image = cv.resize(image, (target_width, int(image.shape[0] * ratio)))

        self.detect_and_display(image, self._landmark_detector)

        cv.waitKey(0)

    def run_dectection_live(self):
        """
        Begin face detection on live camera
        """

        # Create cascade and landmark detector
        self.create_face_cascade()
        self.create_LM_detector()

        # Args
        self._parser.add_argument(
            '--camera', help='Camera device number.', type=int, default=0)
        self._args = self._parser.parse_args()

        # Camera with args
        init_camera = self._args.camera

        # Capture
        capture_user = cv.VideoCapture(init_camera)

        # If camera fails attempt to open
        if not capture_user.isOpened():
            print("Error capturing user")
            exit(0)

        while True:
            _, frame = capture_user.read()
            # Confrim frame is valid
            if frame is None:
                print("Error - No frames captured")
                break

            # Call detect and display method
            self.detect_and_display(cv.flip(frame, 1), self._landmark_detector)
            # Exit program on press of exit key
            if cv.waitKey(10) == 27:
                break
            # cv.waitKey(0)
            # To run live capture frame by frame, comment out cv.waitKey
            # section and repalce with just cv.waitKey(0)

    def detect_and_display(self, frame, landmark_detector):
        """
        Dection of face and its landmarks, visual plots
        """

        # Make grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # Rectangle face detection
        faces = self._face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.2, minNeighbors=5)

        # Hold point location for facial point landmarks
        landmark_points = []

        for (x, y, w, h) in faces:

            # Plot rectangle around face on image
            # frame = cv.rectangle(frame, (x, y), (x+w, y+h),
            #                     (128, 128, 128), 1)

            # Top of forehead area, y
            forehead = y

            # Calculate the middle of the between x1 and x2 (x + w)
            forhead_mid = (x + (x+w)) // 2

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

                # Needed Facial Landmark (x,y) pairs
                cheek_left = landmark_points[1]
                cheek_right = landmark_points[15]
                chin_left = landmark_points[6]
                chin_right = landmark_points[10]
                nose_left = landmark_points[3]
                nose_right = landmark_points[13]
                eye_brow_left = landmark_points[17]
                eye_brow_right = landmark_points[26]
                bottom_chin = landmark_points[8]
                # For jaw angle calculation
                cheek_bone_right_down_one = landmark_points[12]

                # Calcaulte face landmark distances
                cheek_distance = cheek_right[0] - cheek_left[0]
                top_jaw_distance = nose_right[0] - nose_left[0]
                forehead_distance = eye_brow_right[0] - eye_brow_left[0]
                chin_distance = chin_right[0] - chin_left[0]
                head_lenghth = bottom_chin[1] - forehead

                # Plots lines
                cv.line(frame, cheek_left, cheek_right, (128, 128, 128), 2)
                cv.putText(frame, str(cheek_distance), (cheek_left[0], (cheek_left[1] - 5)),
                           cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
                cv.line(frame, chin_left, chin_right, (128, 128, 128), 2)
                cv.putText(frame, str(chin_distance), (chin_left[0], (chin_left[1] - 10)),
                           cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
                cv.line(frame, nose_left, nose_right, (128, 128, 128), 2)
                cv.putText(frame, str(top_jaw_distance), (nose_left[0], (nose_left[1] - 10)),
                           cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
                # Forehead calculated based on farthest out eyebrow point and plotted slightly below the top of forehead
                cv.line(frame, (eye_brow_left[0], int(forehead * 1.10)),
                        (eye_brow_right[0], int(forehead * 1.10)), (128, 128, 128), 2)
                cv.putText(frame, str(forehead_distance),
                           (eye_brow_left[0], int(forehead * 1.05)), cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)
                cv.line(frame, bottom_chin, (int(forhead_mid),
                        int(forehead)), (128, 128, 128), 2)
                cv.putText(frame, str(head_lenghth), ((int(forhead_mid) + 10), int(bottom_chin[1])),
                           cv.FONT_HERSHEY_DUPLEX, .35, (0, 0, 0), 0)

                # Plot lines jaw angle
                cv.line(frame, nose_left, cheek_bone_right_down_one,
                        (0, 0, 0), 2)
                cv.line(frame, nose_left, nose_right, (0, 0, 0), 2)
                cv.line(frame, nose_right, cheek_bone_right_down_one,
                        (0, 0, 0), 2)

                self.calculate_face_shape(
                    cheek_distance, top_jaw_distance, forehead_distance, chin_distance, head_lenghth, frame)

        cv.imshow("Face Shape", frame)

    def calculate_face_shape(self, cheek, jaw, forehead, chin, head_length, frame):
        """
        Takes 4 parameters
        cheek, jaw, forehead, chin - distances
        calculates facial shape 
        """

        # Ratios
        cheek_ratio = cheek / head_length
        jaw_ratio = jaw / head_length
        forehead_ratio = forehead / head_length
        chin_ratio = chin / head_length
        head_ratio = head_length / cheek

        print(cheek_ratio)
        print(jaw_ratio)
        print(forehead_ratio)
        print(chin_ratio)
        print(head_ratio)

        result = "Loading..."

        # Round Face
        # Cheek needs to be widest point
        if (
            0.8 <= cheek_ratio <= 1.0 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            head_ratio <= 1.25
        ):
            result = "Face Shape: Round Face"

        # Oval Face
        # Cheek needs to be widest point
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.6 <= jaw_ratio <= 0.7 and
            0.5 <= forehead_ratio <= 0.7 and
            0.2 <= chin_ratio <= 0.4 and
            1.25 <= head_ratio <= 1.5
        ):
            result = "Face Shape: Oval Face"

        # Rectangle Face
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            head_ratio >= 1.30
        ):
            result = "Face Shape: Rectangle Face"

        # Square Face
        elif (
            0.7 <= cheek_ratio <= 0.99 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.99 and
            0.3 <= chin_ratio <= 0.5 and
            head_ratio <= 1.29
        ):
            result = "Face Shape: Square Face"

        # Heart-Shaped Face
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.5 <= forehead_ratio <= 0.7 and
            0.3 <= chin_ratio <= 0.4 and
            1.2 <= head_ratio <= 1.4
        ):
            result = "Face Shape: Heart-Shaped Face"

        # Diamond Shaped Face
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            1.2 <= head_ratio <= 1.4
        ):
            result = "Face Shape: Diamond Shaped Face"

        # If none of the conditions match
        else:
            result = "Face Shape: Please adjust distance from camera"

        cv.putText(frame, str(result), (10, 50),
                   cv.FONT_HERSHEY_DUPLEX, .5, (0, 0, 0), 1)

    # -------------Create-Methods-----------------------

    def create_LM_detector(self):
        """
        Creates landmark detector 
        loading lbf model
        """
        self._landmark_detector = cv.face.createFacemarkLBF()
        self._landmark_detector.loadModel(self._LBFModel)

    def create_face_cascade(self):
        """
        Creates face cascade
        """
        self._face_cascade = cv.CascadeClassifier(self._haarcascade)


run = Main()
run.run_detection_stillshot()
