#!/usr/bin/env python3

import cv2
import jetson.inference
import jetson.utils
import numpy as np
import time
import boto3
from botocore.exceptions import NoCredentialsError
import os
from datetime import datetime
from dotenv import load_dotenv
import queue
import threading

load_dotenv(verbose=True)

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_FOLDER = os.getenv('S3_FOLDER')
S3_REGION = os.getenv('S3_REGION')

# Detection settings
PERSON_CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 5

def resize_for_display(image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    scale = min(max_width/w, max_height/h)
    return cv2.resize(image, (int(w*scale), int(h*scale))) if scale < 1 else image

def upload_worker(q, s3_client):
    while True:
        task = q.get()
        if task is None: break
        try:
            image_data, s3_key = task
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=image_data)
            print(f"Uploaded: s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"Upload failed: {str(e)}")
        q.task_done()

def detect_objects():
    # Initialize S3 client and upload queue
    s3_client = boto3.client('s3',
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    upload_queue = queue.Queue()
    thread = threading.Thread(target=upload_worker, args=(upload_queue, s3_client))
    thread.daemon = True
    thread.start()

    # Initialize detection network
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
    
    # Optimized video capture with MJPEG encoding
    cap = cv2.VideoCapture(10)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    if not cap.isOpened():
        print("Error opening video device")
        return

    last_upload_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        # Convert frame to CUDA (optimized single-step conversion)
        cuda_img = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform detection
        detections = net.Detect(cuda_img)
        
        person_detected = False
        current_time = time.time()

        # Process detections
        for det in detections:
            if det.ClassID == PERSON_CLASS_ID and det.Confidence >= CONFIDENCE_THRESHOLD:
                person_detected = True
                left, top, right, bottom = map(int, [det.Left, det.Top, det.Right, det.Bottom])
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {det.Confidence:.2f}", 
                           (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Async upload handling
        if person_detected and (current_time - last_upload_time) > COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{S3_FOLDER}person_{timestamp}.jpg"
            _, buffer = cv2.imencode('.jpg', frame)
            upload_queue.put((buffer.tobytes(), s3_key))
            last_upload_time = current_time

        # Display handling
        cv2.imshow("Detection", resize_for_display(frame, 800, 600))
        
        # Key input handling
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{S3_FOLDER}manual_{timestamp}.jpg"
            _, buffer = cv2.imencode('.jpg', frame)
            upload_queue.put((buffer.tobytes(), s3_key))

    # Cleanup
    upload_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print('Program terminated')

if __name__ == "__main__":
    detect_objects()
