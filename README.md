# Eye State Detection for Drowsiness Monitoring

## Overview
This project implements a deep learning model to detect the state of a person's eyes (open or closed) using images captured from a webcam. The primary goal is to monitor drowsiness in real-time, which can be particularly useful for applications in automotive safety, fatigue detection, and health monitoring.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Real-time Detection](#real-time-detection)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Image Preprocessing**: The project includes image loading, resizing, and normalization.
- **Transfer Learning**: Utilizes MobileNet for efficient feature extraction and classification.
- **Binary Classification**: Classifies eye states as "open" or "closed".
- **Real-time Detection**: Monitors eye states using a webcam feed.
- **Visual Feedback**: Displays the detected eye state on the video feed.

## Technologies Used
- **Python**: The primary programming language used for implementation.
- **OpenCV**: For image processing and real-time video capture.
- **TensorFlow**: For building and training the deep learning model.
- **Keras**: High-level API for TensorFlow to simplify model building.
- **NumPy**: For numerical operations and array manipulations.
- **Matplotlib**: For visualizing images and results.
- **Pickle**: For saving and loading model data.

## Dataset
The dataset consists of images of eyes in two states: "closed" and "opened". The images are organized in separate directories for each class. The dataset is loaded, preprocessed, and split into training and validation sets. 

## Usage

### Training the Model
- Run the training script to preprocess the images and train the model. The model will be saved as `my_model.h5`.
- Adjust the number of epochs in the training script as needed.

### Real-time Detection
- Run the real-time detection script to start monitoring eye states using your webcam.
- Press 'q' to exit the webcam feed.

## Model Training
The model is built using transfer learning with MobileNet. The last layers are modified to suit binary classification (open vs. closed eyes). The model is compiled with binary cross-entropy loss and the Adam optimizer.

### Training Steps
1. Load and preprocess the dataset.
2. Shuffle and split the data into training and validation sets.
3. Define the model architecture using MobileNet.
4. Compile the model with appropriate loss and optimizer.
5. Train the model on the dataset.

## Real-time Detection
The real-time detection script captures video from the webcam, detects faces and eyes, and predicts the state of the eyes using the trained model. The results are displayed on the video feed.

### Detection Steps
1. Capture video from the webcam.
2. Detect faces and eyes using Haar cascades.
3. Preprocess the detected eye region.
4. Use the trained model to predict the eye state.
5. Display the prediction on the video feed.
