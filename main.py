import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from src.preprocessing.data_loader import process_images
from src.feature_extraction.csv_creator import extract_features
from src.models.train import train_and_evaluate
from src.models.image_prediction import predict_from_image
import joblib
import json


def main():

    path = '{ "face": ["data/features/face_features.csv"], "left_eye": ["data/features/left_eye_features.csv"], "right_eye": ["data/features/right_eye_features.csv"], "mouth": ["data/features/mouth_features.csv"], "eyes+mouth": ["data/features/left_eye_features.csv", "data/features/right_eye_features.csv", "data/features/mouth_features.csv"], "all": ["data/features/left_eye_features.csv", "data/features/right_eye_features.csv", "data/features/mouth_features.csv", "data/features/face_features.csv"] }'

    roi = json.loads(path)


    if not os.path.exists("data/processed/makeup") or not os.path.exists("data/processed/no_makeup"):
        process_images("data/raw/images", "data/raw/labels", "data/processed")

    if not os.path.exists("data/features/mouth_features.csv"):
        extract_features(folders_to_visit=["data/processed/makeup/mouth", "data/processed/no_makeup/mouth"], csv_file_path=roi["mouth"][0])
    if not os.path.exists("data/features/left_eye_features.csv"):
        extract_features(folders_to_visit=["data/processed/makeup/left_eye", "data/processed/no_makeup/left_eye"], csv_file_path=roi["left_eye"][0])
    if not os.path.exists("data/features/right_eye_features.csv"):
        extract_features(folders_to_visit=["data/processed/makeup/right_eye", "data/processed/no_makeup/right_eye"], csv_file_path=roi["right_eye"][0])
    if not os.path.exists("data/features/face_features.csv"):
        extract_features(folders_to_visit=["data/processed/makeup/face", "data/processed/no_makeup/face"], csv_file_path=roi["face"][0])

    train_and_evaluate("svm")

    predict_from_image("data/test/test.jpg")

if __name__ == "__main__":
    main()