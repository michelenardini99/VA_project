import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from src.data_loader import process_images
from src.train import train_and_evaluate
from utils.data_loader_utils import load_image
from src.csv_creator import create_csv
import joblib


LATEST_MODEL_PATH = "models/latest_model.joblib"

def process_im():

    print(f"Processando le immagini")
    
    process_images(raw_dir="data/raw/images", label_dir="data/raw/labels", output_dir="data/processed")
    

    messagebox.showinfo("Successo", "Immagini processate con successo!")


def extract_features():

    print(f"Estraendo le feature")
    
    create_csv(folders_to_visit=["data/processed/makeup/mouth","data/processed/no_makeup/mouth"], csv_file_path="data/features/mouth_features.csv")
    create_csv(folders_to_visit=["data/processed/makeup/left_eye","data/processed/no_makeup/left_eye"], csv_file_path="data/features/left_eye_features.csv")
    create_csv(folders_to_visit=["data/processed/makeup/right_eye","data/processed/no_makeup/right_eye"], csv_file_path="data/features/right_eye_features.csv")
    create_csv(folders_to_visit=["data/processed/makeup/face","data/processed/no_makeup/face"], csv_file_path="data/features/faces_features.csv")

    messagebox.showinfo("Successo", "Feature estratte con successo!")


def train_model():

    model_type = tk.StringVar(value="svm")
    features_type = tk.StringVar(value="face")
    progress = None


    def start_training():

        file_paths = {
                "face": ["data/features/faces_features.csv"],
                "left_eye": ["data/features/left_eye_features.csv"],
                "right_eye": ["data/features/right_eye_features.csv"],
                "mouth": ["data/features/mouth_features.csv"],
                "eyes+mouth": ["data/features/left_eye_features.csv", "data/features/right_eye_features.csv", "data/features/mouth_features.csv"],
                "all": ["data/features/faces_features.csv", "data/features/left_eye_features.csv", "data/features/right_eye_features.csv", "data/features/mouth_features.csv"]
            }

        selected_features = features_type.get()
        selected_model = model_type.get()

        if selected_features not in file_paths:
            messagebox.showerror("Errore", "Selezione delle feature non valida.")
            return

        try:
            progress["value"] = 10
            root.update_idletasks()

            progress["value"] = 40
            root.update_idletasks()

            # Addestra il modello
            messagebox.showinfo("Training", f"Addestramento modello {selected_model} con feature: {selected_features}")
            model_path = f"models/{selected_model}_{selected_features}_model.joblib"
            pipeline= train_and_evaluate(roi_path=file_paths[selected_features], model_name=selected_model)
            save_model(pipeline)

            progress["value"] = 100
            root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Errore", str(e))
            progress["value"] = 0



    popup = tk.Toplevel()
    popup.title("Seleziona il modello e le feature")
    tk.Label(popup, text="Seleziona il modello:").pack()
    tk.Radiobutton(popup, text="SVM", variable=model_type, value="svm").pack()
    tk.Radiobutton(popup, text="AdaBoost", variable=model_type, value="adaboost").pack()
    tk.Label(popup, text="Seleziona le feature:").pack()
    tk.Radiobutton(popup, text="Occhio destro", variable=features_type, value="right_eye").pack()
    tk.Radiobutton(popup, text="Occhio sinistro", variable=features_type, value="left_eye").pack()
    tk.Radiobutton(popup, text="Bocca", variable=features_type, value="mouth").pack()
    tk.Radiobutton(popup, text="Viso", variable=features_type, value="face").pack()
    tk.Radiobutton(popup, text="Occhi e Bocca", variable=features_type, value="eyes+mouth").pack()
    tk.Radiobutton(popup, text="Tutte", variable=features_type, value="all").pack()

    progress = ttk.Progressbar(popup, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=20)
    tk.Button(popup, text="Inizia addestramento", command=start_training).pack()


def save_model(pipeline):
    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": pipeline}, LATEST_MODEL_PATH)
    print(f"Modello salvato in: {LATEST_MODEL_PATH}")

def load_latest_model():
    if not os.path.exists(LATEST_MODEL_PATH):
        raise FileNotFoundError("Nessun modello salvato trovato.")
    return joblib.load(LATEST_MODEL_PATH)

def elaborate_image():
    file_path = filedialog.askopenfilename(title="Seleziona l'immagine da predire", filetypes=[("Image Files", "*.jpg *.png")])
    if not file_path:
        messagebox.showinfo("Info", "Nessuna immagine selezionata.")
        return
    
    image = load_image(file_path)

    detect_and_preprocess_faces([image], save_dir="data/predict_image")


def main():
    global root
    root = tk.Tk()
    root.title("Rilevamento Viso Truccato")
    root.geometry("400x300")

    tk.Button(root, text="Carica e Processa Immagini", command=process_im, width=30).pack(pady=10)
    tk.Button(root, text="Estrai Feature", command=extract_features, width=30).pack(pady=10)
    tk.Button(root, text="Addestra Modello", command=train_model, width=30).pack(pady=50)
    tk.Button(root, text="Elabora Immagine da predire", command=elaborate_image, width=30).pack(pady=10)
    tk.Button(root, text="Predict Immagine", command=predict_image, width=30).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()