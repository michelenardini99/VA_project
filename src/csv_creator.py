import os
from sklearn.preprocessing import StandardScaler
import csv
import re
import cv2
from feature_extractor import *

def create_csv(folders_to_visit, csv_file_path):


    data = []
    image_names = []
    labels = [0, 1]
    for i, fold in enumerate(folders_to_visit):
        count = 0
        for file_name in os.listdir(fold):
            count+=1
            image_path = os.path.join(fold, file_name)
            if not image_path.endswith('.jpg'):
                continue
            image = cv2.imread(image_path) 
            id = f"m_{count}" if i == 0 else f"n_{count}"
            print(id)
            if fold.endswith('face'):
                color_feature = extract_color_features(image, (3, 3)).reshape(-1)
                gabor_feature = extract_gabor_features(image).reshape(-1)
                gist_feature = extract_gist_feature(image)
                eoh_feature = extract_eoh_features(image)
                lbp_feature = extract_lbp_features(image)
                feature = np.hstack([id, count, color_feature, gabor_feature, eoh_feature, lbp_feature, labels[i]])
            else:
                color_feature = extract_color_features(image, (5, 5)).reshape(-1)
                feature = np.hstack([id, count, color_feature, labels[i]])
                
            

            data.append(feature)

    data = np.array(data, dtype=object)
    scaler = StandardScaler()
    data[:, 1:-1] = scaler.fit_transform(data[:, 1:-1].astype(float))
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        
"""create_csv(folders_to_visit=["data/test/processed/makeup/mouth","data/test/processed/no_makeup/mouth"], csv_file_path="data/test/features/mouth_features.csv")
create_csv(folders_to_visit=["data/test/processed/makeup/left_eye","data/test/processed/no_makeup/left_eye"], csv_file_path="data/test/features/left_eye_features.csv")
create_csv(folders_to_visit=["data/test/processed/makeup/right_eye","data/test/processed/no_makeup/right_eye"], csv_file_path="data/test/features/right_eye_features.csv")
create_csv(folders_to_visit=["data/test/processed/makeup/face","data/test/processed/no_makeup/face"], csv_file_path="data/test/features/faces_features.csv")"""
