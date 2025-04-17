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

# AWS S3 Configuration - Set to your specific bucket
S3_BUCKET = 'nick-ap-s3'
S3_FOLDER = 'images/'
S3_REGION = 'ap-southeast-2'  # Change this if your bucket is in a different region

# Detection settings
PERSON_CLASS_ID = 1  # In COCO dataset, person is class 1
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 5  # Time between captures to avoid multiple uploads of the same person

def resize_for_display(image, max_width=1280, max_height=720):
    """Resize image for display purposes while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    # Calculate the resize factor
    scale = min(max_width / w, max_height / h)
    
    # Only resize if the image is larger than the max dimensions
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        return resized
    return image

def upload_to_s3(local_file, s3_key):
    """Upload a file to the S3 bucket"""
    s3_client = boto3.client('s3', region_name=S3_REGION)
    try:
        s3_client.upload_file(local_file, S3_BUCKET, s3_key)
        print(f"Upload Successful: s3://{S3_BUCKET}/{s3_key}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return False

def detect_objects():
    # Load the detection network
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
    
    # Open the camera using OpenCV with /dev/video10
    cap = cv2.VideoCapture(10)  # Use 10 for /dev/video10
    
    # Set camera properties if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    
    if not cap.isOpened():
        print("Error: Could not open video device /dev/video10")
        return
    
    last_upload_time = 0
    
    while cap.isOpened():
        # Capture frame from OpenCV
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from /dev/video10")
            time.sleep(0.2)
            continue
            
        # Convert OpenCV BGR image to CUDA format for Jetson Inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cuda_img = jetson.utils.cudaFromNumpy(frame_rgb)
        
        # Perform detection
        detections = net.Detect(cuda_img)
        
        person_detected = False
        current_time = time.time()
        
        # Process and display detections
        for detection in detections:
            class_id = detection.ClassID
            confidence = detection.Confidence
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add class label and confidence
            class_name = net.GetClassDesc(class_id)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if a person is detected with sufficient confidence
            if class_id == PERSON_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
                person_detected = True
        
        # If a person is detected and cooldown period has passed, capture and upload
        if person_detected and (current_time - last_upload_time) > COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_filename = f"person_detected_{timestamp}.jpg"
            s3_key = f"{S3_FOLDER}person_detected_{timestamp}.jpg"
            
            # Save the image locally
            cv2.imwrite(local_filename, frame)
            print(f"Person detected! Image saved as {local_filename}")
            
            # Upload to S3
            if upload_to_s3(local_filename, s3_key):
                last_upload_time = current_time
                print(f"Image uploaded to s3://{S3_BUCKET}/{s3_key}")
            
            # Optional: Delete local file after upload
            os.remove(local_filename)
            print(f"Local file {local_filename} deleted")
        
        # Resize frame for display (without affecting processing or saved image quality)
        display_frame = resize_for_display(frame, max_width=800, max_height=600)
        
        # Display the resized frame
        cv2.imshow("Object Detection", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == ord('s'):  # Press 's' to save image manually
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            manual_filename = f"manual_capture_{timestamp}.jpg"
            s3_manual_key = f"{S3_FOLDER}manual_capture_{timestamp}.jpg"
            
            cv2.imwrite(manual_filename, frame)
            print(f"Manually saved image to {manual_filename}")
            
            # Upload manual capture to S3
            if upload_to_s3(manual_filename, s3_manual_key):
                print(f"Manual capture uploaded to s3://{S3_BUCKET}/{s3_manual_key}")
            
            # Optional: Delete local file after upload
            os.remove(manual_filename)
            print(f"Local file {manual_filename} deleted")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print('end program')

if __name__ == "__main__":
    detect_objects()