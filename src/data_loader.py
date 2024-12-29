import cv2
import os

haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haarcascadeeye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
mouth_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"




def detect_and_preprocess_faces(images, save_dir, face_size=(150, 130), eye_size=(52,52), mouth_size=(56, 62)):
    
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    eye_cascade = cv2.CascadeClassifier(haarcascadeeye_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

    faces_count = 0
    eyes_count = 0
    mouths_count = 0

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        

        for i, (x, y, w, h) in enumerate(faces):

            faces_count += 1
            
            face = image[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]

            face_normalized = cv2.resize(face, face_size)

            cv2.imwrite(os.path.join(save_dir + "/face/", f"face_{faces_count}.jpg"), face_normalized)

            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(0, 0))

            for j, (xe, ye, we, he) in enumerate(eyes):
                eyes_count += 1
                eye = face[ye:ye+he, xe:xe+we]

                eye_normalized = cv2.resize(eye, eye_size)
                cv2.imwrite(os.path.join(save_dir + "/eyes/", f"eyes{eyes_count}.jpg"), eye_normalized)

            mouths = mouth_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(0, 0))

            for j, (xm, ym, wm, hm) in enumerate(mouths):
                mouths_count += 1
                mouth = face[ym:ym+hm, xm:xm+wm]

                mouth_normalized = cv2.resize(mouth, mouth_size)
                mouth_normalized = cv2.resize(mouth, mouth_size)

                cv2.imwrite(os.path.join(save_dir + "/mouth/", f"mouth_{mouths_count}.jpg"), mouth_normalized)

def load_images_with_pattern(folder_path, pattern):
    images = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(pattern):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Errore: Impossibile leggere l'immagine {image_path}")
    
    return images
        

images = load_images_with_pattern(folder_path = "data/raw/", pattern = "n.jpg")

detect_and_preprocess_faces(images, save_dir="data/processed/no_makeup")

images = load_images_with_pattern(folder_path = "data/raw/", pattern = "m.jpg")

detect_and_preprocess_faces(images, save_dir="data/processed/makeup")