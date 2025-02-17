import cv2
import os
import pandas as pd
import mediapipe as mp
import numpy as np
from utils.data_loader_utils import load_images, create_directories, draw_landmarks, show_channels
from src.preprocessing.normalizer import normalize_roi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directories(output_dir, isTestImage=False):
    regions = ['face', 'right_eye', 'left_eye', 'mouth']
    labels = ['makeup', 'no_makeup']
    for region in regions:
        if not isTestImage:
            for label in labels:
                os.makedirs(os.path.join(output_dir, label, region), exist_ok=True)
        else:
            os.makedirs(os.path.join(output_dir, region), exist_ok=True)
        

def extract_roi(image, coordinates, size, margin):
    if not coordinates:
        logging.warning("Lista di coordinate vuota, impossibile estrarre ROI.")
        return None
    
    x_coords, y_coords = zip(*coordinates)
    x1, y1 = max(0, min(x_coords) - margin), max(0, min(y_coords) - margin)
    x2, y2 = min(image.shape[1], max(x_coords) + margin), min(image.shape[0], max(y_coords) + margin)
    
    if x1 >= x2 or y1 >= y2:
        logging.error(f"ROI non valida con coordinate ({x1}, {y1}, {x2}, {y2})")
        return None
    
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        logging.error(f"ROI vuota per coordinate ({x1}, {y1}, {x2}, {y2})")
        return None
    
    return cv2.resize(roi, size)

def get_label(file_path):

    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip() 
            return first_line.split()[0] if first_line else None  
    except FileNotFoundError:
        logging.warning(f"File etichetta non trovato: {file_path}")
        return None
    except IndexError:
        logging.error(f"Formato errato nel file etichetta: {file_path}")
        return None

def detect_landmarks(image):
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            logging.info("Nessun volto rilevato.")
            return None
        
        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        image_shape = image.shape

        return {
            'left_eye': get_landmark_coordinates(landmarks, image_shape, [145, 124, 189]),
            'right_eye': get_landmark_coordinates(landmarks, image_shape, [374, 413, 353]),
            'mouth': get_landmark_coordinates(landmarks, image_shape, [61, 291, 17]),
            'face': get_landmark_coordinates(landmarks, image_shape, [33, 263, 152])
        }

def get_landmark_coordinates(landmarks, image_shape, indices):
    h, w = image_shape[:2]
    return [(int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)) for idx in indices]

def process_images(raw_dir, label_dir, output_dir):
    create_directories(output_dir)
    
    for file_name in os.listdir(raw_dir):
        image_path = os.path.join(raw_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        image = cv2.imread(image_path)
        
        if image is None:
            logging.warning(f"Impossibile leggere immagine: {image_path}")
            continue
        
        label = None
        for label_file in os.listdir(label_dir):
            if os.path.splitext(label_file)[0] == base_name:
                label = get_label(os.path.join(label_dir, label_file))
                break
        
        if label is None:
            logging.warning(f"Etichetta non trovata per immagine: {file_name}")
            continue
        
        roi_landmarks = detect_landmarks(image)
        if roi_landmarks is None:
            continue
        
        normalized_rois = normalize_roi(image, roi_landmarks)
        label_name = "makeup" if label == "0" else "no_makeup"
        
        for region, roi in normalized_rois.items():
            if roi is not None:
                cv2.imwrite(os.path.join(output_dir, label_name, region, file_name), roi)
                logging.info(f"Salvata ROI {region} per {file_name}")




