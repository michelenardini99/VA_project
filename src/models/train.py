import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import os
import logging
import joblib
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_features(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    return X, y


    
def train_and_evaluate(model_name, 
                        roi_path="all", 
                        svm_params=(8, 'rbf', 0.001), 
                        base_classifier=DecisionTreeClassifier(max_depth=1), 
                        X_test_pers=None, 
                        y_test_pers=None):

    path = '{ "face": ["data/features/face_features.csv"], "left_eye": ["data/features/left_eye_features.csv"], "right_eye": ["data/features/right_eye_features.csv"], "mouth": ["data/features/mouth_features.csv"], "eyes+mouth": ["data/features/left_eye_features.csv", "data/features/right_eye_features.csv", "data/features/mouth_features.csv"], "all": ["data/features/left_eye_features.csv", "data/features/right_eye_features.csv", "data/features/mouth_features.csv", "data/features/face_features.csv"] }'
    os.makedirs("models", exist_ok=True)
    X_train_total, X_test_total, y_train, y_test = None, None, None, None
    classifiers = []
    roi = json.loads(path)
    for i, roi in enumerate(roi[roi_path]):
        X, y = load_features(roi)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logging.info(f"Dimensioni X_train: {X_train.shape}, y_train: {y_train.shape}")
        logging.info(f"Dimensioni X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        if model_name == 'svm':
            C, kernel, gamma = svm_params
            clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, class_weight='balanced')
        elif model_name == 'adaboost':
            clf = AdaBoostClassifier(estimator=base_classifier, n_estimators=1000)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=int(X_train.shape[0] * 0.1), random_state=42))),
            ('classifier', clf)
        ])
        
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            logging.info(f"Train idx: {len(train_idx)}, Validation idx: {len(val_idx)}")
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            pipeline.fit(X_fold_train, y_fold_train)
            score = pipeline.score(X_fold_val, y_fold_val)
            cv_scores.append(score)
        
        pipeline.fit(X_train, y_train)
        classifiers.append((f"clf_{i}", pipeline))
        
        if X_train_total is None:
            X_train_total = X_train
            X_test_total = X_test
        else:
            X_train_total = np.hstack((X_train_total, X_train))
            X_test_total = np.hstack((X_test_total, X_test))
    
    if len(classifiers) == 1:
        logging.info("Training completato")
        accuracy = classifiers[0][1].score(X_test_total, y_test)
        logging.info(f"Accuratezza: {accuracy:.2f}")
        joblib.dump(classifiers[0][1], f"models/{roi_path}.pkl")
        logging.info(f"Modello salvato: models/{roi_path}.pkl")
    else:
        ensemble_model = VotingClassifier(estimators=classifiers, voting='soft')
        ensemble_model.fit(X_train_total, y_train)
        
        if X_test_pers is None:
            accuracy = ensemble_model.score(X_test_total, y_test)
        else:
            accuracy = ensemble_model.score(X_test_pers, y_test_pers)
        
        joblib.dump(ensemble_model, f"models/{roi_path}.pkl")
        logging.info("Training completato con ensemble learning su più ROI")
        logging.info(f"Accuratezza del modello ensemble: {accuracy:.2f}")
        logging.info(f"Modello ensemble salvato: models/{roi_path}.pkl")
        
        return ensemble_model