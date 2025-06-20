import face_recognition

from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
# Загрузка модели
model = load_model("models/face_detection_model.keras")

# def detect_face(img):
#     # img должен быть в формате RGB
#     face_locations = face_recognition.face_locations(img)
#     if not face_locations:
#         return None

#     top, right, bottom, left = face_locations[0]
#     face = img[top:bottom, left:right]
#     return face

# def detect_face_location(image):
#     locations = face_recognition.face_locations(image)
#     if not locations:
#         return None
#     return locations[0]  # (top, right, bottom, left)

def detect_face(img):
    if img.shape[2] == 3 and img.dtype == 'uint8':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = img
    
    orig_h, orig_w = rgb_img.shape[:2]
    
    # Предобработка изображения для модели
    resized = cv2.resize(rgb_img, (224, 224))
    input_tensor = resized.astype('float32') / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Предсказание модели
    bbox_pred, class_pred = model.predict(input_tensor)
    
    # Обработка результатов
    class_score = class_pred[0][0]
    bbox = bbox_pred[0]
    
    if class_score > 0.2:  # Порог уверенности
        x1 = int(bbox[0] * orig_w)
        y1 = int(bbox[1] * orig_h)
        x2 = int(bbox[2] * orig_w)
        y2 = int(bbox[3] * orig_h)
        
        face = rgb_img[y1:y2, x1:x2]
        return face
    return None

def detect_face_location(img):
    if img.shape[2] == 3 and img.dtype == 'uint8':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = img
    
    orig_h, orig_w = rgb_img.shape[:2]
    
    # Предобработка изображения для модели
    resized = cv2.resize(rgb_img, (224, 224))
    input_tensor = resized.astype('float32') / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Предсказание модели
    bbox_pred, class_pred = model.predict(input_tensor)
    
    # Обработка результатов
    class_score = class_pred[0][0]
    bbox = bbox_pred[0]
    
    if class_score > 0.2:  # Порог уверенности
        x1 = int(bbox[0] * orig_w)
        y1 = int(bbox[1] * orig_h)
        x2 = int(bbox[2] * orig_w)
        y2 = int(bbox[3] * orig_h)
        
        # Формат (top, right, bottom, left)
        return (y1, x2, y2, x1)
    return None
