# Drowsiness Detection using Spatiotemporal Autoencoder
This Python project focuses on building a drowsiness detection system utilizing a Spatiotemporal Autoencoder. The system processes webcam images, applies a Deep Learning model to classify whether the person's eyes are open or closed, and alerts the user when signs of drowsiness are detected.

# Project Structure
train.py: Script for training the Spatiotemporal Autoencoder model. It loads video frames, preprocesses images, constructs the model architecture, and trains the model to detect anomalies.

test.py: Script for real-time drowsiness detection. It uses the pre-trained model to classify eye states in a live video feed, detecting and notifying abnormal events.

# Instructions
## Training the Model
Execute train.py to capture frames from a video, preprocess the images, and train the Spatiotemporal Autoencoder model.

bash
Copy code
python train.py
The trained model will be saved as "saved_model.h5".

## Real-time Drowsiness Detection
Replace __path_to_custom_test_video in test.py with the path to your custom test video.

Execute test.py to initiate real-time drowsiness detection using the pre-trained model.

bash
Copy code
python test.py
The script will display the video feed with detected abnormal events highlighted.

## Dependencies
Ensure you have the required libraries installed:

bash
Copy code
pip install opencv-python numpy keras imutils
Feel free to explore, experiment, and contribute to enhance the effectiveness of this drowsiness detection system. Safe driving!
