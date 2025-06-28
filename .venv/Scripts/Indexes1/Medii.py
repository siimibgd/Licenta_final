import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image
import cv2
import numpy as np

class ImageCalculatorApp:
    def __init__(self, master,image_path=None):
        self.master = master
        master.title("Calculator Imagine")
        master.geometry("500x350") # Setează dimensiunea inițială a ferestrei

        self.image_path = image_path # Variabilă pentru a stoca calea imaginii

        # --- Dropdown pentru opțiunile de calcul ---
        self.options = [
            "ExG",
            "ExGR",
            "GLI",
            "Pseudo-NDVI",  
            "VARI",
            "NGRDI",
            "TGI"
        ]
        self.selected_option = tk.StringVar(master)
        self.selected_option.set(self.options[0]) # Setează opțiunea implicită

        self.option_menu_label = tk.Label(master, text="Alege un calcul:")
        self.option_menu_label.pack(pady=5)

        # ttk.OptionMenu necesită primul argument să fie master, apoi StringVar și prima opțiune implicită
        self.option_menu = ttk.OptionMenu(master, self.selected_option, self.options[0], *self.options, command=self.on_option_select)
        self.option_menu.pack(pady=5)

        # --- Etichetă pentru afișarea rezultatului ---
        self.result_label = tk.Label(master, text="Rezultatul calculului va apărea aici.", wraplength=400)
        self.result_label.pack(pady=20)


    def on_option_select(self, selected_value):
        """
        Funcția apelată automat când se selectează o opțiune din dropdown.
        """
        if self.image_path:
            # Apelăm funcția de procesare internă
            result = self._process_image(self.image_path, selected_value)
            self.result_label.config(text=result)
        else:
            self.result_label.config(text="Te rog să selectezi o imagine mai întâi.")

    def _process_image(self, image_path, calculation_type):
        """
        Funcție internă pentru procesarea imaginii.
        """
        try:
            img = Image.open(image_path)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            R = image_rgb[:, :, 0].astype(float)
            G = image_rgb[:, :, 1].astype(float)
            B = image_rgb[:, :, 2].astype(float)
            R = image_rgb[:, :, 0] / 255.0
            G = image_rgb[:, :, 1] / 255.0
            B = image_rgb[:, :, 2] / 255.0
            epsilon = 1e-8
            # Crează mască pentru pixeli negri (R=G=B=0)
            non_black_mask = ~((R == 0) & (G == 0) & (B == 0))

            valid_mask = non_black_mask
            if calculation_type == "ExG":
                exg = 2 * G - R - B
                exg_valid= exg[valid_mask]
                Exg_mediu=np.mean(exg_valid)
                return f"Medie ExG: {Exg_mediu:.2f}"
            elif calculation_type == "ExGR":
                # Formula: 2 * G - R - B
                exg = 2 * G - R - B
                # --- ExGR (Excess Green minus Excess Red Index) ---
                exr = 1.4 * R - G
                ExGR = exg - exr
                # Aplică masca pentru a extrage valorile relevante
                ExGR_valid = ExGR[valid_mask]
                # Calculează media valorilor ExGR valide
                ExGR_mediu = np.mean(ExGR_valid)
                return f"Medie ExGR: {ExGR_mediu:.2f}"
            elif calculation_type == "GLI":
                numerator_gli = 2 * G - R - B
                denominator_gli = 2 * G + R + B + epsilon
                gli = numerator_gli / denominator_gli
                gli_valid = gli[valid_mask]
                gli_mediu = np.mean(gli_valid)
                return f"Medie GLI: {gli_mediu:.2f}"
            elif calculation_type == "Pseudo-NDVI":
                numerator = B - R
                denominator = B + R + epsilon
                pseudo_ndvi = numerator / denominator
                pseudo_ndvi_valid = pseudo_ndvi[valid_mask]
                pseudo_ndvi_mediu = np.mean(pseudo_ndvi_valid)
                return f"Medie Pseudo-NDVI: {pseudo_ndvi_mediu:.2f}"
            elif calculation_type == "VARI":
                numerator_vari = G - R
                denominator_vari = G + R - B + epsilon
                vari = numerator_vari / denominator_vari
                vari_valid = vari[valid_mask]
                vari_mediu = np.mean(vari_valid)
                return f"Medie VARI: {vari_mediu:.2f}"
            elif calculation_type == "NGRDI":
                numerator_ngrdi = G - R
                denominator_ngrdi = G + R + epsilon
                ngrdi = numerator_ngrdi / denominator_ngrdi
                ngrdi_valid = ngrdi[valid_mask]
                ngrdi_mediu = np.mean(ngrdi_valid)
                return f"Medie NGRDI: {ngrdi_mediu:.2f}"
            elif calculation_type == "TGI":
                tgi = G - 0.37 * R - 0.63 * B
                tgi_valid = tgi[valid_mask]
                tgi_mediu = np.mean(tgi_valid)
                return f"Medie TGI: {tgi_mediu:.2f}"
            else:
                return "Tip de calcul necunoscut."

        except FileNotFoundError:
            return f"Eroare: Imaginea '{image_path}' nu a fost găsită."
        except Exception as e:
            return f"A apărut o eroare la procesarea imaginii: {e}"

# Această secțiune face ca scriptul să ruleze aplicația doar când este executat direct.
# Când este importat de un alt script, această parte nu se va executa.
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCalculatorApp(root)
    root.mainloop()