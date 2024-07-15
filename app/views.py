from django.conf import settings
from django.utils import timezone
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import StreamingHttpResponse,HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CarNumberPlate
from .serializers import NumberPlateSerializer
import cv2
import easyocr
import numpy as np
import os
import threading
from datetime import datetime
from django.utils import timezone
from django.views.decorators import gzip
from django.utils.decorators import method_decorator
from PIL import Image
import tempfile
import re
import pytz
from google.cloud import vision
import io
import geocoder
from ultralytics import YOLO
import cvzone
import math

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed
# class ANPRView(APIView):
#     def post(self, request, *args, **kwargs):
#         file = request.FILES['image']
#         temp_file_path = default_storage.save('temp.jpg', ContentFile(file.read()))
#         image_path = os.path.join(default_storage.location, temp_file_path)

#         image = cv2.imread(image_path)
#         edged = self.preprocess_image(image)
#         plate = self.detect_plate(image, edged)
#         if plate is not None:
#             text = self.recognize_plate(image, plate)
#             response_data = {'plate': text}
#         else:
#             response_data = {'plate': 'No plate detected'}

#         os.remove(image_path)
#         return Response(response_data, status=status.HTTP_200_OK)

#     def preprocess_image(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edged = cv2.Canny(blurred, 75, 200)
#         return edged

#     def detect_plate(self, image, edged):
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         plate = None
#         for contour in contours:
#             if cv2.contourArea(contour) < 1000:
#                 continue
#             peri = cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
#             if len(approx) == 4:
#                 plate = approx
#                 break
#         return plate 

#     def recognize_plate(self, image, plate):
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask, [plate], -1, 255, -1)
#         (x, y) = np.where(mask == 255)
#         (topx, topy) = (np.min(x), np.min(y))
#         (bottomx, bottomy) = (np.max(x), np.max(y))
#         cropped = image[topx:bottomx+1, topy:bottomy+1]
#         gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('window_title', gray_cropped) 
#         cv2.waitKey(0)
#         text = pytesseract.image_to_string(gray_cropped, config='--psm 8')
#         text = text.strip()  # Remove leading/trailing whitespace
#         text = text.replace(']', '')  # Remove specific unwanted character
#         text = text.replace('\n', '')  # Remove newline characters
#         return text

class ANPRViewcc(APIView):
    # camera_url = 'rtsp://admin:abe%40123456@192.168.5.105/Streaming/Channels/401'
    camera_url = 'rtsp://admin:abe@123456@192.168.5.103:554/stream1'
    # camera_url ='DS-2CD043G2-I(U)'
    # camera_index = 0 

    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                           "scissors", "teddy bear", "hair drier", "toothbrush"]
        self.reader = easyocr.Reader(['en'])  # Initi

    def post(self, request, *args, **kwargs):
        print("Received POST request")
        image, image_path = self.capture_frame()
        if image is None:
            return Response({'error': 'Failed to capture frame from camera'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Detect objects using YOLO model
        objects_detected = self.detect_objects(image)
        if not objects_detected:
            return Response({'error': 'No relevant objects detected'}, status=status.HTTP_200_OK)

        edged = self.preprocess_image(image)
        if edged is None:
            return Response({'error': 'Failed to preprocess image'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        plate = self.detect_plate(image, edged)
        if plate is not None:
            text = self.recognize_plate(image, plate)
            print(text,'this is text')
            self.save_plate_to_database(text)
        else:
            text = 'No plate detected'

        response_data = {
            'plate': text,
        }

        return Response(response_data, status=status.HTTP_200_OK)

    

    def capture_frame(self):
        print("Attempting to capture frame from RTSP stream")
        try:
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                print("Failed to open video capture")
                return None, None

            ret, frame = cap.read()
            cap.release()  # Release the capture object

            if not ret:
                print("Failed to capture frame")
                return None, None

            scale_percent =60  # Percent of original size (adjust as needed)
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Save the captured frame to a file
            image_path = 'captured_frame.jpg'
            cv2.imwrite(image_path, resized_frame)
            print(f"Frame saved as {image_path}")

            cv2.imshow('Captured Frame', resized_frame)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

            return resized_frame, image_path

        except Exception as e:
            print(f"Exception in capture_frame: {e}")
            return None, None

    def detect_objects(self, image):
        print("Detecting objects in the image")
        results = self.model(image)
        detected_objects = []
        FOCAL_LENGTH = 1271  # You need to calibrate this for your specific camera setup
        REAL_WIDTH = 1.8  # Real width of a car in meters (average)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if self.classNames[cls] in ["bicycle", "car","person", "motorbike","truck"]:
                    detected_objects.append(self.classNames[cls])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    distance = (REAL_WIDTH * FOCAL_LENGTH) / w  # Distance estimation
                    if distance >= 3.048:  # 10 feet in meters
                        detected_objects.append(self.classNames[cls])
                        cvzone.cornerRect(image, (x1, y1, w, h), l=9)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        if conf > 0.4:
                            cvzone.putTextRect(image, f'{self.classNames[cls]} {conf} {distance:.2f}m', (max(0, x1), max(35, y1)),
                                           scale=0.6, thickness=1, offset=3)
        cv2.imshow('Detected Objects', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detected_objects

    def preprocess_image(self, image):
        print("Preprocessing image")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            cv2.imshow('Processing Frame', edged)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

            return edged
        except Exception as e:
            print(f"Exception in preprocess_image: {e}")
            return None

    def detect_plate(self, image, edged):
        print("Detecting plate")
        try:
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            plate = None

            for contour in contours:
                # Filter out small or large contours
                if cv2.contourArea(contour) < 1000 or cv2.contourArea(contour) > 15000:
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Look for rectangular contours
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)

                    # Assuming aspect ratio of license plates is typically between 2 and 5
                    if 0 < aspect_ratio < 5:
                        plate = approx
                        break

            if plate is not None:
                print("Plate detected")
                cv2.drawContours(image, [plate], -1, (0, 250, 0), 2)
                cv2.imshow('Detecting Plate', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No plate detected")
                return None

            return plate

        except Exception as e:
            print(f"Exception in detect_plate: {e}")
            return None

    def recognize_plate(self, image, plate):
        print("Recognizing plate")
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [plate], -1, 255, -1)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            cropped = image[topx:bottomx + 1, topy:bottomy + 1]
            result = self.reader.readtext(cropped)
            if result:
                text = result[0][-2]
                # Filter the recognized text to include only alphanumeric characters
                text = re.sub(r'[^A-Za-z0-9]', '', text)
                text = text.strip()
                # Apply additional logic to verify and correct the text
                if len(text) < 4 or len(text) > 10:
                    text = 'Invalid plate detected'
                return text
            else:
                return 'No plate detected'
         

        except Exception as e:
            print(f"Exception in recognize_plate: {e}")
            return None

    def save_plate_to_database(self, text):
        if text and text != 'No plate detected':
            kolkata_tz = pytz.timezone('Asia/Kolkata')
            detected_at = timezone.localtime(timezone.now(), kolkata_tz)
            car_plate = CarNumberPlate(number_plate=text, detected_at=detected_at)
            car_plate.save()
            print(f"Saved {text} to database at {detected_at}")


class CarNumberPlateListView(APIView):
    def get(self, request, *args, **kwargs):
        plates = CarNumberPlate.objects.all()
        serializer = NumberPlateSerializer(plates, many=True)
        return Response(serializer.data)



class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture('rtsp://admin:abe@123456@192.168.5.104:554/stream1')
        if not self.video.isOpened():
            raise ValueError("Unable to open video source")
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        if self.video.isOpened(): 
            self.video.release()

    def get_frame(self):
        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@gzip.gzip_page
def video_feed(request):
    try:
        return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(f"Error: {e}")
        return HttpResponse(f"Error: {e}", status=500, content_type="text/plain")
    

