import cv2
import os
import pandas as pd
import mediapipe as mp
import numpy as np
from utils.data_loader_utils import load_images, create_directories, draw_landmarks, show_channels


def create_directories(output_dir):
    for region in ['face', 'right_eye', 'left_eye', 'mouth']:
        for label in ['makeup', 'no_makeup']:
            os.makedirs(f"{output_dir}/{label}/{region}", exist_ok=True)
        

def extract_roi(image, coordinates, size, margin):
    x_coords = [x for x, y in coordinates]
    y_coords = [y for x, y in coordinates]
    x1, y1 = max(0, min(x_coords) - margin), max(0, min(y_coords) - margin)
    x2, y2 = min(image.shape[1], max(x_coords) + margin), min(image.shape[0], max(y_coords) + margin)
    
    if x1 >= x2 or y1 >= y2:
        print(f"Errore: ROI non valida con coordinate ({x1}, {y1}, {x2}, {y2})")
        return None
    
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        print(f"Errore: ROI vuota per coordinate ({x1}, {y1}, {x2}, {y2})")
        return None

    roi = cv2.resize(roi, size)
    
    return roi

def getLabel(file_path):

    with open(file_path, 'r') as file:

        first_line = file.readline().strip()

        first_value = first_line.split()[0] if first_line else None
        return first_value


def process_images(raw_dir, label_dir, output_dir):

    
    create_directories(output_dir)


    for file_name in os.listdir(raw_dir):
        image_path = os.path.join(raw_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        image = cv2.imread(image_path)
        label = None
        if os.path.exists(f"{output_dir}/no_makeup/face/{file_name}") or os.path.exists(f"{output_dir}/makeup/face/{file_name}"):
            continue
        for label_file in os.listdir(label_dir):
            if os.path.splitext(label_file)[0] == base_name:
                label_path = os.path.join(label_dir, label_file)
                label = getLabel(label_path)
        roi_landmarks = detect_landmarks(image)
        if roi_landmarks is None:
            continue

        left_eye = extract_roi(image, roi_landmarks["left_eye"], (52, 52), margin=10)
        right_eye = extract_roi(image, roi_landmarks["right_eye"], (52, 52), margin=10)
        mouth = extract_roi(image, roi_landmarks["mouth"], (56, 62), margin=10)
        face = extract_roi(image, roi_landmarks["face"], (150, 130), margin=10)

        label_name = "makeup" if label == "0" else "no_makeup"

        if mouth is not None:
            cv2.imwrite(f"{output_dir}/{label_name}/mouth/{file_name}", mouth)
        if left_eye is not None:
            cv2.imwrite(f"{output_dir}/{label_name}/left_eye/{file_name}", left_eye)
        if right_eye is not None:
            cv2.imwrite(f"{output_dir}/{label_name}/right_eye/{file_name}", right_eye)
        if face is not None:
            cv2.imwrite(f"{output_dir}/{label_name}/face/{file_name}", face)

    


def get_landmark_coordinates(landmarks, image_shape, indices):
    h, w = image_shape[:2]
    coordinates = []
    for idx in indices:
        x = int(landmarks[idx][0] * w)  # Converti x normalizzato in pixel
        y = int(landmarks[idx][1] * h)  # Converti y normalizzato in pixel
        coordinates.append((x, y))
    return coordinates

def detect_landmarks(image):
    # Inizializza Face Mesh di MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Carica immagine
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Esegui il rilevamento dei landmark
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("Nessun volto rilevato.")
        return None

    # Disegna i landmark sull'immagine
    landmarks = [
        (lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark
    ]

    # Ottieni dimensioni immagine
    image_shape = image.shape

    # Landmark di interesse
    roi_landmarks = {
        'left_eye': get_landmark_coordinates(landmarks, image_shape, [33, 133, 246, 161, 160, 159, 158, 157, 173]),
        'right_eye': get_landmark_coordinates(landmarks, image_shape, [362, 398, 384, 385, 386, 387, 388, 466, 263]),
        'mouth': get_landmark_coordinates(landmarks, image_shape, [61, 291, 78, 308, 191, 80, 81, 82, 13, 312, 311, 310]),
        'face': get_landmark_coordinates(landmarks, image_shape, [10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                                                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                                                  172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109])
    }

    return roi_landmarks



"""process_images("data/test/images", "data/test/labels", "data/test/processed")"""

