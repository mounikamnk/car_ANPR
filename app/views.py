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
import numpy as np
import pytesseract
import os
import threading
import time
from django.views.decorators import gzip
from django.utils.decorators import method_decorator
from PIL import Image
import tempfile
import re
import pytz
from google.cloud import vision
import io
import geocoder

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed
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
    # camera_url = 'rtsp://admin:abe%40123456@192.168.1.105/Streaming/Channels/401'
    camera_url='rtsp://admin:abe@123456@192.168.1.105:554/stream1'
    # camera_url ='DS-2CD043G2-I(U)'
    def post(self, request, *args, **kwargs):
        print("Received POST request")
        image, image_path = self.capture_frame()
        if image is None:
            return Response({'error': 'Failed to capture frame from camera'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        edged = self.preprocess_image(image)
        if edged is None:
            return Response({'error': 'Failed to preprocess image'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        plate = self.detect_plate(image, edged)
        if plate is not None:
            text = self.recognize_plate(image, plate)
            location = request.data.get('location') or self.get_location_from_ip(request)
            print(text,location,'this is text')
            self.save_plate_to_database(text,location)
        else:  
            text = 'No plate detected'

        # Call extract_text_from_image using the saved image path
        extracted_text = self.extract_text_from_image(image_path)

        response_data = {
            'plate': text,
            # 'extracted_text': extracted_text,
        }

        return Response(response_data, status=status.HTTP_200_OK)
    def get_location_from_ip(self, request):
        try:
            ip = request.META.get('REMOTE_ADDR', None)
            if ip:
                g = geocoder.ip(ip)
                if g.ok:
                    return f"{g.city}, {g.state}, {g.country}"
        except Exception as e:
            print(f"Exception in get_location_from_ip: {e}")
        return "Unknown Location"

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

            scale_percent = 50  # Percent of original size (adjust as needed)
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
    
    def preprocess_image(self, image):
        print("Preprocessing image")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            cv2.imshow('processing Frame', edged)
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
                    if 2 < aspect_ratio < 5:
                        plate = approx
                        break

            if plate is not None:
                print("Plate detected")
                cv2.drawContours(image, [plate], -1, (0, 250, 0), 2)
                cv2.imshow('detecting plate', image)
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

            cropped = image[topx:bottomx+1, topy:bottomy+1]
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray_cropped, config='--psm 8')

            # Filter the recognized text to include only alphanumeric characters
            text = re.sub(r'[^A-Za-z0-9]', '', text)
            text = text.strip()
            return text

        except Exception as e:
            print(f"Exception in recognize_plate: {e}")
            return None

    def extract_text_from_image(self, image_path):
        print("Extracting text from image")
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            text = re.sub(r'[^A-Za-z0-9]', '', text)
            # print(text,'texting message')
            return text
        except Exception as e:
            print(f"Exception in extract_text_from_image: {e}")
            return None
    def save_plate_to_database(self, text,location):
        if text and text != 'No plate detected':
            kolkata_tz = pytz.timezone('Asia/Kolkata')
            detected_at = timezone.localtime(timezone.now(), kolkata_tz)
            car_plate = CarNumberPlate(number_plate=text, detected_at=detected_at,location=location)
            car_plate.save()
            print(f"Saved {text} to database at {detected_at} with location {location}")
           
class CarNumberPlateListView(APIView):
    def get(self, request, *args, **kwargs):
        plates = CarNumberPlate.objects.all()
        serializer = CarNumberPlateSerializer(plates, many=True)
        return Response(serializer.data)



class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture('rtsp://admin:abe@123456@192.168.1.105:554/stream1')
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
    

