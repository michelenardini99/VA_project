import cv2
import numpy as np
import logging
import os
import joblib
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from src.preprocessing.data_loader import detect_landmarks, create_directories
from src.preprocessing.normalizer import normalize_roi
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from src.feature_extraction.csv_creator import extract_features

def load_features(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[0, 1:-1].values
    return X

def get_features():
    path = '{ "face": ["data/test/features/face_features.csv"], "left_eye": ["data/test/features/left_eye_features.csv"], "right_eye": ["data/test/features/right_eye_features.csv"], "mouth": ["data/test/features/mouth_features.csv"], "eyes+mouth": ["data/test/features/left_eye_features.csv", "data/test/features/right_eye_features.csv", "data/test/features/mouth_features.csv"], "all": ["data/test/features/left_eye_features.csv", "data/test/features/right_eye_features.csv", "data/test/features/mouth_features.csv", "data/test/features/face_features.csv"] }'
    roi = json.loads(path)
    extract_features(folders_to_visit=["data/test/processed/mouth"], csv_file_path=roi["mouth"][0])
    extract_features(folders_to_visit=["data/test/processed/left_eye"], csv_file_path=roi["left_eye"][0])
    extract_features(folders_to_visit=["data/test/processed/right_eye"], csv_file_path=roi["right_eye"][0])
    extract_features(folders_to_visit=["data/test/processed/face"], csv_file_path=roi["face"][0])


def predict_from_image(image_path, m = 'all'):

    path = '{ "face": ["data/test/features/face_features.csv"], "left_eye": ["data/test/features/left_eye_features.csv"], "right_eye": ["data/test/features/right_eye_features.csv"], "mouth": ["data/test/features/mouth_features.csv"], "eyes+mouth": ["data/test/features/left_eye_features.csv", "data/test/features/right_eye_features.csv", "data/test/features/mouth_features.csv"], "all": ["data/test/features/left_eye_features.csv", "data/test/features/right_eye_features.csv", "data/test/features/mouth_features.csv", "data/test/features/face_features.csv"] }'
    
    model = json.loads(path)

    create_directories("data/test/processed", isTestImage=True)
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Impossibile leggere l'immagine: {image_path}")
        return None
    
    roi_landmarks = detect_landmarks(image)
    if roi_landmarks is None:
        logging.error(f"Nessun landmark indivuato: {image_path}")
        return None
        
    normalized_rois = normalize_roi(image, roi_landmarks)
    print(normalized_rois)
    for region, roi in normalized_rois.items():
        if roi is not None:
            cv2.imwrite(os.path.join("data/test/processed", region, "test.jpg"), roi)
            logging.info(f"Salvata ROI {region} per test.jpg")
    
    get_features()
    scaler = StandardScaler()
    X_tot = None
    if m is "all" or m is "eyes+mouth":
        for roi in model[m]:
            X = load_features(roi)
            X = X.reshape(1, -1)
            features_scaled = scaler.fit_transform(X)
            if X_tot is None:
                X_tot = features_scaled
            else:
                X_tot = np.hstack((X_tot, features_scaled))
    else:
        X = load_features(model[m][0])
        features_scaled = scaler.fit_transform(X)
        X_tot = selector.fit_transform(features_scaled)

    model = joblib.load(f"models/{m}.pkl")
    
    
    prediction = model.predict(X_tot)
    probability = model.predict_proba(X_tot)
    
    class_names = ["Truccata", "Non Truccata"]
    predicted_class = class_names[int(prediction[0])]
    confidence = np.max(probability) * 100
    
    logging.info(f"Predizione: {predicted_class} con confidenza {confidence:.2f}%")
    return predicted_class, confidence