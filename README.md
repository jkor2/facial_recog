# Facial Shape Detection for Optimal Haircuts

This Python script utilizes OpenCV to detect facial landmarks and determine facial shape, providing users with insights to optimize their haircut based on their face shape.

## How It Works

1. **Face Detection:** The script employs Haar cascades for face detection in a video stream or image. Detected faces are enclosed with rectangles.

2. **Facial Landmark Detection:** The program uses the LBF (Local Binary Feature) model to identify facial landmarks. These landmarks are crucial points on the face, allowing for detailed analysis.

3. **Landmark Visualization:** The code plots the facial landmarks on the image, forming a comprehensive map of the face.

4. **Facial Shape Lines:** Essential lines indicating specific facial features such as eyes, chin, and nose are drawn. These lines aid in determining the user's facial shape.

5. **User Interface:** The resulting image is displayed in a window, providing a visual representation of the detected facial landmarks and shape.

## Additional Features (To Be Implemented)

The code provides a foundation for further enhancements:

- **Eye Brow Line Detection:** Detect and display the eyebrow line on the image.
- **Lips Line Detection:** Implement lips line detection and visualization.
- **Text Annotations:** Add text annotations for each detected feature (e.g., "Eye Line," "Chin Line").

## How to Use

1. Ensure you have the required dependencies installed.
2. Adjust the paths for the LBF model and Haar cascade files in the script.
3. Run the script and capture a video stream or provide an image.
4. View the displayed window with facial landmarks and shape lines.
5. Analyze the visual representation to determine your facial shape.

Feel free to contribute to the project and add more features to enhance the user experience!

**Note:** Press 'ESC' to exit the program.

## Acknowledgments

- This project is a learning exercise for utilizing OpenCV for facial landmark detection.
- Haar cascade and LBF model files are essential for accurate face and landmark detection.
