import os
from sklearn.preprocessing import StandardScaler
import csv
import re
import cv2
from src.feature_extraction.feature_extractor import *


def extract_features(folders_to_visit, csv_file_path, isTestImage=False):
    data = []
    labels = [0, 1]
    
    for i, fold in enumerate(folders_to_visit):
        count = 0
        for file_name in os.listdir(fold):
            count += 1
            image_path = os.path.join(fold, file_name)
            
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Impossibile leggere immagine: {image_path}")
                continue
            base_name = os.path.splitext(file_name)[0]
            id = int(base_name) if isTestImage else f"m_{count}" if i == 0 else f"n_{count}"
            if fold.endswith('face'):
                color_feature = extract_color_features(image, (3, 3)).reshape(-1)
                gabor_feature = extract_gabor_features(image).reshape(-1)
                gist_feature = extract_gist_feature(image)
                eoh_feature = extract_eoh_features(image)
                lbp_feature = extract_lbp_features(image)
                feature = np.hstack([id, color_feature, gabor_feature, gist_feature, eoh_feature, lbp_feature, labels[i]])
            else:
                color_feature = extract_color_features(image, (5, 5)).reshape(-1)
                feature = np.hstack([id, color_feature, labels[i]])
            data.append(feature)
    
    if len(data) == 0:
        logging.error("Nessun dato disponibile per la creazione del CSV.")
        return
    data = np.array(data, dtype=object)
    scaler = StandardScaler()
    if len(data) > 1:
        data[:, 1:-1] = scaler.fit_transform(data[:, 1:-1].astype(float))
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    
    logging.info(f"CSV salvato con successo in {csv_file_path}")
        