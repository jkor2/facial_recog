# Facial Shape Detection

Welcome to Facial Shape Detection! This project utilizes facial landmarks to calculate facial shapes and helps you determine the best-suited haircut for your face.

## Features

- Detection of facial landmarks using the LBF model.
- Identification of facial shapes based on ratios of distinct facial features.
- Support for both still shot detection and live camera detection.

## LBF Model and Haar Cascade Classifier

- **LBF Model:** The project employs the LBF model, a part of the OpenCV library, for facial landmark detection. LBF (Local Binary Feature) is a powerful and robust algorithm for facial landmark localization. OpenCV is an open-source computer vision library widely used for image and video processing tasks.

- **Haar Cascade Classifier:** The Haar Cascade classifier, also a component of OpenCV, is used for detecting faces in images. The project utilizes a pre-trained Haar Cascade classifier for accurate face detection.

- **OpenCV:** By leveraging OpenCV's functionalities, this project benefits from a well-established and versatile computer vision library.

## Example
<div>
<img src="faces/rectangle/rectangle.png" alt="Example Image" width="300" height="350">
<img src="faces/rectangle/positive-test.png" alt="Example Image" width="300" height="350">
</div>

## Facial Shapes

Discover the ideal haircut for your face shape. Supported facial shapes include:

- Round Face
- Oval Face
- Rectangle Face
- Square Face
- Heart-Shaped Face
- Diamond Shaped Face

## Future Improvements

- Chin angle for further accuracy 
- Investigate and enhance heart and diamond face shape 
- Face shape detection with a beard.
