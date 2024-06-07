from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import StreamingHttpResponse,HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import NumberPlateSerializer
import cv2
import numpy as np
import pytesseract
import os
import threading
import time
from django.views.decorators import gzip
from django.utils.decorators import method_decorator


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed
class ANPRView(APIView):
    def post(self, request, *args, **kwargs):
        file = request.FILES['image']
        temp_file_path = default_storage.save('temp.jpg', ContentFile(file.read()))
        image_path = os.path.join(default_storage.location, temp_file_path)

        image = cv2.imread(image_path)
        edged = self.preprocess_image(image)
        plate = self.detect_plate(image, edged)
        if plate is not None:
            text = self.recognize_plate(image, plate)
            response_data = {'plate': text}
        else:
            response_data = {'plate': 'No plate detected'}

        os.remove(image_path)
        return Response(response_data, status=status.HTTP_200_OK)

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        return edged

    def detect_plate(self, image, edged):
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        plate = None
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                plate = approx
                break
        return plate

    def recognize_plate(self, image, plate):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [plate], -1, 255, -1)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = image[topx:bottomx+1, topy:bottomy+1]
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_cropped, config='--psm 8')
        text = text.strip()  # Remove leading/trailing whitespace
        text = text.replace(']', '')  # Remove specific unwanted character
        text = text.replace('\n', '')  # Remove newline characters
        return text

class ANPRViewcc(APIView):
    camera_url = 'rtsp://admin:abe%40123456@192.168.1.105/Streaming/Channels/401'

    def post(self, request, *args, **kwargs):
        image = self.capture_frame()
        if image is None:
            return Response({'error': 'Failed to capture frame from camera'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        edged = self.preprocess_image(image)
        plate = self.detect_plate(image, edged)
        if plate is not None:
            text = self.recognize_plate(image, plate)
            response_data = {'plate': text}
        else:
            response_data = {'plate': 'No plate detected'}

        return Response(response_data, status=status.HTTP_200_OK)

    def capture_frame(self):
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None
        
        return frame

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        return edged

    def detect_plate(self, image, edged):
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        plate = None
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                plate = approx
                break
        return plate

    def recognize_plate(self, image, plate):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [plate], -1, 255, -1)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = image[topx:bottomx+1, topy:bottomy+1]
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_cropped, config='--psm 8')
        text = text.strip()  # Remove leading/trailing whitespace
        text = text.replace(']', '')  # Remove specific unwanted characters
        text = text.replace('\n', '')  # Remove newline characters
        return text
    



class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture('rtsp://admin:abe%40123456@192.168.1.105/Streaming/Channels/401')
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