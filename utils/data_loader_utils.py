import cv2
import os
import matplotlib.pyplot as plt


def load_images(folder_path):
    images = []

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
        else:
            print(f"Errore: Impossibile leggere l'immagine {image_path}")

    return images


def load_image(path):
    image = cv2.imread(path)
    return image

def create_directories(base_path):
    """
    Crea le directory per i dati processati.
    """
    dirs = [
        f"{base_path}/makeup/face",
        f"{base_path}/makeup/right_eye",
        f"{base_path}/makeup/left_eye",
        f"{base_path}/makeup/mouth",
        f"{base_path}/no_makeup/face",
        f"{base_path}/no_makeup/right_eye",
        f"{base_path}/no_makeup/left_eye",
        f"{base_path}/no_makeup/mouth",
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def draw_landmarks(image, landmarks_dict):
    annotated_image = image.copy()
    colors = {
        'left_eye': (0, 255, 0),  # Verde
        'right_eye': (255, 0, 0),  # Blu
        'mouth': (0, 0, 255),  # Rosso
        'face_contour': (255, 255, 0)  # Giallo
    }

    for region, points in landmarks_dict.items():
        for (x, y) in points:
            cv2.circle(annotated_image, (x, y), 2, colors[region], -1)

    return annotated_image

def show_channels(image_path, output_folder):
    image = cv2.imread(image_path)

    if image is None:
        print("Errore: immagine non trovata.")
        return

    # Converti da BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Separazione dei canali R, G, B
    R, G, B = cv2.split(image_rgb)

    # Converti da BGR a HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Separazione dei canali H, S, V
    H, S, V = cv2.split(image_hsv)

    # Crea un layout per visualizzare i canali
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(R, cmap='gray')
    axes[0, 0].set_title("Canale R")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(G, cmap='gray')
    axes[0, 1].set_title("Canale G")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(B, cmap='gray')
    axes[0, 2].set_title("Canale B")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(H, cmap='gray')
    axes[1, 0].set_title("Canale H")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(S, cmap='gray')
    axes[1, 1].set_title("Canale S")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(V, cmap='gray')
    axes[1, 2].set_title("Canale V")
    axes[1, 2].axis("off")

    plt.tight_layout()

    # Salva l'immagine combinata in una cartella
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, "channels_visualization.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Immagine salvata in {output_path}")