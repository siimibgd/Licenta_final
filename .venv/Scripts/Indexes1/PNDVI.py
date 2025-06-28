import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import os
import tkinter as tk
from tkinter import filedialog, messagebox


def calculate_and_visualize_rgb_indices(image_path, output_dir):
    try:
        # Creează directorul de ieșire dacă nu există
        os.makedirs(output_dir, exist_ok=True)

        # Deschide imaginea și asigură că este în format RGB
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=float)

        # Separă canalele R, G, B și normalizează la 0-1
        R = img_array[:, :, 0] / 255.0
        G = img_array[:, :, 1] / 255.0
        B = img_array[:, :, 2] / 255.0

        # Creează o mască pentru a exclude pixelii negri (R=0, G=0, B=0)
        non_black_mask = ~((R == 0) & (G == 0) & (B == 0))

        # Adaugă un epsilon mic la numitori pentru a preveni împărțirea la zero
        epsilon = 1e-8

        def calculate_masked_index(calculation_array):
            index = np.full_like(R, np.nan)
            if isinstance(calculation_array, tuple):
                # Gestionează cazurile ca (numerator, denominator)
                numerator = calculation_array[0]
                denominator = calculation_array[1]
                # Asigură că împărțirea se face doar pe non_black_mask și unde numitorul nu e zero
                valid_mask = non_black_mask & (denominator != 0)
                index[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            else:
                index[non_black_mask] = calculation_array[non_black_mask]
            return index

        # Dicționar pentru a stoca indicii calculați și colormaps-urile recomandate
        indices = {}
        colormaps = {}

        # Pseudo-NDVI (Blue-Red): (B - R) / (B + R)
        numerator = B - R
        denominator = B + R + epsilon
        indices['Pseudo-NDVI (Blue-Red)'] = calculate_masked_index((numerator, denominator))
        colormaps['Pseudo-NDVI (Blue-Red)'] = 'Greens'

        # --- Procesează și salvează fiecare indice ---
        processed_count = 0
        for name, index_array in indices.items():
            # Verifică dacă toți pixelii valizi (non-negri) au rezultat în NaN
            if np.all(np.isnan(index_array[non_black_mask])) and np.any(non_black_mask):
                print(
                    f"Se omite '{name}' deoarece nu conține date valide pentru pixelii non-negri.")
                continue

            # Dacă întreaga imagine (inclusiv fundalul negru) este NaN
            if np.all(np.isnan(index_array)):
                print(
                    f"Se omite '{name}' deoarece întregul array al indicelui este NaN.")
                continue

            # Setează intervalul de normalizare pentru indicii care variază de obicei între -1 și 1
            vmin, vmax = -1, 1

            # Obține colormap-ul și setează valorile NaN să fie transparente 
            cmap = plt.colormaps[colormaps[name]].copy()
            cmap.set_bad(color='black', alpha=0)  

            # Creează figura cu un fundal transparent
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

            # Afișează datele indicelui
            im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=14)
            ax.axis('off')

            # Adaugă o bară de culori
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label("Valoare Index")

            # Salvează figura cu un fundal transparent
            output_filename = os.path.join(output_dir,
                                           f"{name.replace(' ', '_').replace('/', '_')}.png")
            plt.savefig(output_filename, bbox_inches='tight', dpi=300, transparent=True)
            print(f"Index '{name}' salvat cu scala de culori la: {output_filename}")

            im = Image.open(output_filename)

            im.show()

            plt.close(fig) 
            processed_count += 1

        if processed_count == 0:
            messagebox.showinfo("Procesare Imagini",
                                "Niciun indice nu a putut fi calculat. Asigură-te că imaginea conține pixeli non-negri valizi.")
        else:
            messagebox.showinfo("Procesare Imagini",
                                f"Procesare completă! Rezultatele au fost salvate în: {output_dir}")

    except FileNotFoundError:
        messagebox.showerror("Eroare", f"Eroare: Fișierul imagine nu a fost găsit la calea specificată: {image_path}")
    except Exception as e:
        messagebox.showerror("Eroare", f"A apărut o eroare: {e}")


def run_image_processor_with_gui_pndvi(image_path=None):
    """
    Rulează logica de procesare a imaginii, preluând opțional o cale de imagine
    și solicitând utilizatorului un director de ieșire printr-o interfață grafică.
    """
    if image_path is None or not os.path.isfile(image_path):
        messagebox.showerror("Eroare", "Calea imaginii nu este validă. Asigură-te că imaginea există.")
        return

    # Ascunde fereastra principală Tkinter
    root = tk.Tk()
    root.withdraw()

    # Deschide fereastra de dialog pentru a selecta directorul de ieșire
    output_directory = filedialog.askdirectory(
        title="Selectați Directorul de Salvare a Rezultatelor"
    )

    if output_directory:
        # Extrage numele fișierului imagine fără extensie
        image_filename_base = os.path.splitext(os.path.basename(image_path))[0]
        # Creează un subdirector specific pentru imaginea procesată în directorul de ieșire ales
        final_output_subdir = os.path.join(output_directory, f"results_for_{image_filename_base}")

        print(f"Imagine de procesat: {image_path}")
        print(f"Director de ieșire ales: {final_output_subdir}")
        # Apelăm calculate_and_visualize_rgb_indices
        calculate_and_visualize_rgb_indices(image_path, final_output_subdir)
    else:
        messagebox.showwarning("Anulat", "Selectarea directorului de ieșire a fost anulată.")


# --- Exemplu de utilizare (dacă scriptul este rulat direct) ---
if __name__ == "__main__":
    # fișier dummy pentru demonstrație dacă nu există
    dummy_image_path = "Timp/B/test_image.png"
    if not os.path.exists(dummy_image_path):
        os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
        try:
            # Creează o imagine de test simplă cu un fundal negru
            array = np.zeros((100, 100, 3), dtype=np.uint8)
            # Adaugă un cerc verde
            y, x = np.ogrid[-50:50, -50:50]
            mask = x * x + y * y <= 40 * 40
            array[mask] = [0, 255, 0]  # Verde
            # Adaugă un pătrat roșu
            array[10:30, 10:30] = [255, 0, 0]
            img = Image.fromarray(array)
            img.save(dummy_image_path)
            print(f"S-a creat o imagine dummy de test '{dummy_image_path}' pentru demonstrație.")
        except Exception as e:
            print(f"Nu s-a putut crea o imagine dummy: {e}")

    # Apelăm direct funcția GUI
    print(f"Rulând procesorul de imagine cu GUI pentru imaginea: {dummy_image_path}")
    run_image_processor_with_gui(dummy_image_path)