import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.stats import skew
from skimage.feature import local_binary_pattern

def color_descriptor(image_trucco, image_no_trucco, output_path):
    """
    Calcola e confronta le statistiche (media, varianza, skewness) del canale Saturation tra un occhio truccato e non truccato.
    Salva un'immagine che mostra i risultati.

    Parameters:
        image_trucco (str): Path dell'immagine dell'occhio truccato.
        image_no_trucco (str): Path dell'immagine dell'occhio non truccato.
        output_path (str): Path dove salvare l'immagine risultante.
    """
    def calculate_statistics(channel):
        """Calcola media, varianza e skewness per il canale di saturazione."""
        data = channel.flatten()
        stats = {
            'mean': np.mean(data),
            'variance': np.var(data),
            'skewness': skew(data)
        }
        return stats

    def plot_statistics(ax, stats_trucco, stats_no_trucco, title):
        """Crea un grafico comparativo delle statistiche."""
        labels = ['Saturation']
        trucco_values = [stats_trucco[title]]
        no_trucco_values = [stats_no_trucco[title]]

        x = np.arange(len(labels))
        ax.bar(x - 0.2, trucco_values, 0.4, label='Trucco', color='blue')
        ax.bar(x + 0.2, no_trucco_values, 0.4, label='No Trucco', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    # Carica e converte le immagini in HSV
    img_trucco = cv2.imread(image_trucco)
    img_no_trucco = cv2.imread(image_no_trucco)

    hsv_trucco = cv2.cvtColor(img_trucco, cv2.COLOR_BGR2HSV)
    hsv_no_trucco = cv2.cvtColor(img_no_trucco, cv2.COLOR_BGR2HSV)

    # Estrai il canale Saturation
    saturation_trucco = hsv_trucco[:, :, 1]
    saturation_no_trucco = hsv_no_trucco[:, :, 1]

    # Calcola statistiche
    stats_trucco = calculate_statistics(saturation_trucco)
    stats_no_trucco = calculate_statistics(saturation_no_trucco)

    # Crea la figura
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    # Immagini originali
    axs[0, 0].imshow(cv2.cvtColor(img_trucco, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Occhio con Trucco")
    axs[0, 0].axis('off')

    axs[1, 0].imshow(cv2.cvtColor(img_no_trucco, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Occhio senza Trucco")
    axs[1, 0].axis('off')

    # Statistiche: Media
    plot_statistics(axs[0, 1], stats_trucco, stats_no_trucco, 'mean')
    axs[0, 1].set_ylabel("Media")

    # Statistiche: Varianza
    plot_statistics(axs[0, 2], stats_trucco, stats_no_trucco, 'variance')
    axs[0, 2].set_ylabel("Varianza")

    # Statistiche: Skewness
    plot_statistics(axs[0, 3], stats_trucco, stats_no_trucco, 'skewness')
    axs[0, 3].set_ylabel("Skewness")

    # Salva la figura
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_lambda(scale, k_max=np.pi / 2, f=np.sqrt(2)):
    return 2 * np.pi / (k_max / (f ** scale))


def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def apply_gabor_filter(image, output_path):
    gabor_responses = []
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    frequencies = [0.1, 0.2]

    for theta in thetas:
        for freq in frequencies:
            response, _ = gabor(image, frequency=freq, theta=theta)
            gabor_responses.append(response)

    combined_response = np.sum(gabor_responses, axis=0)

    return combined_response

def compute_gabor_responses(image, frequencies=[0.1, 0.2, 0.3]):
    responses = []
    for frequency in frequencies:
        for theta in range(0, 180, 45):  # Orientamenti a 45 gradi
            theta_rad = np.deg2rad(theta)
            real, _ = gabor(image, frequency=frequency, theta=theta_rad)
            responses.append(real)
    return responses

def compute_gist_descriptor(image, n_components=16):
    responses = compute_gabor_responses(image)
    flattened_responses = [r.flatten() for r in responses]
    gist_vector = np.concatenate(flattened_responses)

    # Riduzione dimensionale con PCA
    pca = PCA(n_components=0.95, svd_solver='full')
    gist_descriptor = pca.fit_transform(gist_vector.reshape(1, -1))
    return gist_descriptor.flatten()

def plot_gabor_and_gist(image_truccato, image_non_truccato, output_path):
    # Converti in scala di grigi
    gray_truccato = cv2.cvtColor(image_truccato, cv2.COLOR_BGR2GRAY)
    gray_non_truccato = cv2.cvtColor(image_non_truccato, cv2.COLOR_BGR2GRAY)

    # Risposte Gabor per viso truccato
    gabor_responses_truccato = compute_gabor_responses(gray_truccato)

    # Risposte Gabor per viso non truccato
    gabor_responses_non_truccato = compute_gabor_responses(gray_non_truccato)

    # Descrittore Gist per entrambi
    gist_truccato = compute_gist_descriptor(gray_truccato)
    gist_non_truccato = compute_gist_descriptor(gray_non_truccato)

    # Visualizzazione
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Immagini originali
    axes[0, 0].imshow(cv2.cvtColor(image_truccato, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Immagine truccata")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(image_non_truccato, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Immagine non truccata")
    axes[0, 1].axis("off")

    # Griglia delle risposte Gabor per viso truccato
    axes[1, 0].imshow(gabor_responses_truccato[0], cmap="gray")
    axes[1, 0].set_title("Risposte Gabor (Truccato)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gabor_responses_non_truccato[0], cmap="gray")
    axes[1, 1].set_title("Risposte Gabor (Non truccato)")
    axes[1, 1].axis("off")

    # Descrittori Gist
    axes[2, 0].plot(gist_truccato, label="Gist (Truccato)")
    axes[2, 0].set_title("Descrittore Gist (Truccato)")
    axes[2, 0].legend()

    axes[2, 1].plot(gist_non_truccato, label="Gist (Non truccato)")
    axes[2, 1].set_title("Descrittore Gist (Non truccato)")
    axes[2, 1].legend()

    # Spazio vuoto
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def calculate_eoh(image, bins=39):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients along x and y
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude and orientation of gradients
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # Quantize angles into bins
    bin_width = 360 / bins
    quantized_angles = (angle // bin_width).astype(int)

    # Create the histogram
    eoh = np.zeros(bins)
    for bin_idx in range(bins):
        eoh[bin_idx] = np.sum(magnitude[quantized_angles == bin_idx])

    return eoh, magnitude

def generate_comparison_figures(image_with_makeup, image_without_makeup, output_dir):
    create_output_directory(output_dir)

    # Convert images to grayscale
    gray_with_makeup = cv2.cvtColor(image_with_makeup, cv2.COLOR_BGR2GRAY)
    gray_without_makeup = cv2.cvtColor(image_without_makeup, cv2.COLOR_BGR2GRAY)

    # Generate Gabor filter responses
    gabor_m = apply_gabor_filter(gray_with_makeup, os.path.join(output_dir, "gabor_with_makeup.png"))
    gabor_n = apply_gabor_filter(gray_without_makeup, os.path.join(output_dir, "gabor_without_makeup.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(gabor_m, cmap='gray')
    axes[0].set_title("Combinazioni di Gabor filter in viso truccato")
    axes[0].axis("off")

    axes[1].imshow(gabor_n, cmap='gray')
    axes[1].set_title("Combinazione di Gabor filter in viso non truccato")
    axes[1].axis("off")

    plt.savefig(os.path.join(output_dir, "gabor_application.png"))
    plt.close()

    plot_gabor_and_gist(image_with_makeup, image_without_makeup, "reports/figures/gist_comparison.png")


    m_eoh, m_edge_image = calculate_eoh(image_with_makeup)
    n_eoh, n_edge_image = calculate_eoh(image_without_makeup)

    # Generate edge image using Canny
    n_edges = cv2.Canny(cv2.cvtColor(image_without_makeup, cv2.COLOR_BGR2GRAY), 100, 200)
    m_edges = cv2.Canny(cv2.cvtColor(image_with_makeup, cv2.COLOR_BGR2GRAY), 100, 200)


    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    axes[0, 0].imshow(cv2.cvtColor(image_without_makeup, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Immagine originale senza trucco")
    axes[0, 0].axis("off")

    # Edge image
    axes[0, 1].imshow(n_edges, cmap="gray")
    axes[0, 1].set_title("Immagine con i contorni rilevati")
    axes[0, 1].axis("off")

    axes[0, 2].plot(n_eoh, color="blue")
    axes[0, 2].set_title("Istogramma")
    axes[0, 2].set_xlabel("Bins")

    # EOH histogram
    axes[1, 0].imshow(cv2.cvtColor(image_with_makeup, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Immagine originale con trucco")
    axes[1, 0].axis("off")

    # Edge image
    axes[1, 1].imshow(m_edges, cmap="gray")
    axes[1, 1].set_title("Immagine con i contorni rilevati")
    axes[1, 1].axis("off")

    # EOH histogram
    axes[1, 2].plot(m_eoh, color="blue")
    axes[1, 2].set_title("Istogramma")
    axes[1, 2].set_xlabel("Bins")

    plt.savefig(os.path.join(output_dir, "eoh_application.png"))
    plt.close()

def calculate_lbp_histogram(image, radius=1, n_points=8, method='uniform'):
    """
    Calcola la mappa LBP e genera l'istogramma dei pattern locali.

    Args:
        image (numpy.ndarray): immagine in input in scala di grigi.
        radius (int): raggio per il calcolo di LBP.
        n_points (int): numero di punti circostanti per il calcolo di LBP.

    Returns:
        lbp (numpy.ndarray): immagine LBP.
        histogram (numpy.ndarray): istogramma dei pattern locali.
    """
    lbp = local_binary_pattern(image, n_points, radius, method)

    # Calcolo dell'istogramma dei pattern uniformi (59 bin)
    n_bins = int(n_points*(n_points-1)+3)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 59))
    hist = hist.astype('float') / hist.sum()  # Normalizzazione
    print(np.array(hist))
    return lbp, hist

def visualize_lbp_and_histograms(image_trucco, image_no_trucco, save_path):
    """
    Visualizza e confronta le mappe LBP e gli istogrammi tra un'immagine truccata e una non truccata.

    Args:
        image_trucco (numpy.ndarray): immagine con trucco (in scala di grigi).
        image_no_trucco (numpy.ndarray): immagine senza trucco (in scala di grigi).
        save_path (str): percorso per salvare il risultato.
    """
    # Calcola LBP e istogrammi per entrambe le immagini
    lbp_trucco, hist_trucco = calculate_lbp_histogram(image_trucco)
    lbp_no_trucco, hist_no_trucco = calculate_lbp_histogram(image_no_trucco)

    # Crea una figura per visualizzare i risultati
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Immagine originale con trucco
    axs[0, 0].imshow(image_trucco, cmap="gray")
    axs[0, 0].set_title("Immagine originale con trucco")
    axs[0, 0].axis("off")

    # Mappa LBP con trucco
    axs[0, 1].imshow(lbp_trucco, cmap="gray")
    axs[0, 1].set_title("Mappa LBP con trucco")
    axs[0, 1].axis("off")

    # Istogramma con trucco
    axs[0, 2].bar(range(len(hist_trucco)), hist_trucco, color="blue")
    axs[0, 2].set_title("Istogramma con trucco")
    axs[0, 2].set_xlabel("Pattern LBP")
    axs[0, 2].set_ylabel("Frequenza")

    # Immagine originale senza trucco
    axs[1, 0].imshow(image_no_trucco, cmap="gray")
    axs[1, 0].set_title("Immagine originale senza trucco")
    axs[1, 0].axis("off")

    # Mappa LBP senza trucco
    axs[1, 1].imshow(lbp_no_trucco, cmap="gray")
    axs[1, 1].set_title("Mappa LBP senza trucco")
    axs[1, 1].axis("off")

    # Istogramma senza trucco
    axs[1, 2].bar(range(len(hist_no_trucco)), hist_no_trucco, color="orange")
    axs[1, 2].set_title("Istogramma senza trucco")
    axs[1, 2].set_xlabel("Pattern LBP")
    axs[1, 2].set_ylabel("Frequenza")

    # Salva la figura
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Example usage
image_trucco = cv2.imread("data/test/test2.jpg", cv2.IMREAD_GRAYSCALE)
image_no_trucco = cv2.imread("data/test/test.jpg", cv2.IMREAD_GRAYSCALE)
visualize_lbp_and_histograms(image_trucco, image_no_trucco, "reports/figures/lbp_comparison.png")