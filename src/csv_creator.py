from feature_extractor import *
import os
from sklearn.preprocessing import StandardScaler
import csv

def create_csv(folders_to_visit):

    data = []
    image_names = []
    labels = [0, 1]
    for i, fold in enumerate(folders_to_visit):
        for file_name in os.listdir(fold):
            image_path = os.path.join(fold, file_name)
            image = cv2.imread(image_path) 

            color_feature = extract_color_features(image, (3, 3)).reshape(-1)
            gabor_feature = extract_gabor_features(image).reshape(-1)
            eoh_feature = extract_eoh_features(image)
            lbp_feature = extract_lbp_features(image)

            feature = np.hstack([color_feature, gabor_feature, eoh_feature, lbp_feature, labels[i]])

            data.append(feature)
            #labels.append(label)

    data = np.array(data)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    with open('data/features/faces_features.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

            


create_csv(folders_to_visit=["data/processed/makeup/face","data/processed/no_makeup/face"])