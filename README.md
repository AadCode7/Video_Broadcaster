# Video_Broadcaster

This project is a real-time video streaming and segmentation application built with FastAPI, OpenCV, and Ultralytics YOLOv8. The application supports background blurring, masking, and custom background replacement using semantic segmentation powered by YOLO.

## Features

Web-based UI hosted via FastAPI

Real-time video capture from webcam or video source

Person segmentation using YOLOv8m-seg.pt

Apply background blur, black background, or custom background

Adjustable FPS and blur intensity

Start/Stop streaming via REST API

Enumerate available video devices

## Technologies Used

Python (3.8+)

FastAPI: for the backend REST API

Uvicorn: ASGI server for running FastAPI

OpenCV: for video frame processing and masking

Ultralytics YOLOv8: pre-trained segmentation model

Torch (PyTorch): deep learning framework

Threading: to run video streaming without blocking the API

## API Endpoints

GET / — serves index.html

GET /start — starts streaming (query params: source, fps, blur, background, preview)

GET /stop — stops the video stream

GET /devices — lists available video input devices
