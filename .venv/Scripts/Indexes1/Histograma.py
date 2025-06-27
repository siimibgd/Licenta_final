import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

def image_histogram(image):
    try:
        plant_seedling = iio.imread(image)
    except FileNotFoundError:
        print(f"Error: The image file '{image}' was not found.")
        print("Please make sure the file path is correct and the image exists.")
        exit() 
    
    # Verifică dacă imaginea e într-adevăr color
    if len(plant_seedling.shape) != 3 or plant_seedling.shape[2] != 3:
        raise ValueError("Imaginea nu este color (RGB).")
    
    # Afișează imaginea RGB
    fig_rgb, ax_rgb = plt.subplots()
    ax_rgb.imshow(plant_seedling)
    ax_rgb.set_title("Imagine RGB")
    ax_rgb.axis("off")
    plt.show()
    
    # Culori pentru canalele R, G, B
    colors = ("red", "green", "blue")
    
    # Histograma RGB ajustată
    fig_hist, ax_hist = plt.subplots()
    ax_hist.set_xlim([0, 256])
    
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            plant_seedling[:, :, channel_id], bins=256, range=(0, 256)
        )
    
        adjusted_histogram = np.where(histogram > 80000, histogram - 80000, histogram)
    
        ax_hist.plot(bin_edges[:-1], adjusted_histogram, color=color)
    
    ax_hist.set_title("Histograma RGB")
    ax_hist.set_xlabel("Valoare culoare (0-255)")
    ax_hist.set_ylabel("Număr pixeli")
    plt.show()