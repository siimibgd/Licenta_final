import sys
import subprocess
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QMainWindow,
    QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QStandardPaths
from Indexes1 import (ExG, ExGR, GLI, PNDVI, VARI,NGRDI, TGI, CLASIFICARE, Medii, Histograma)
#from Scripts.Indexes1 import ExG
#from Scripts.Indexes1 import ExGR
#from Scripts.Indexes1 import GLI
#from Scripts.Indexes1 import PNDVI
#from Scripts.Indexes1 import VARI
#from Scripts.Indexes1 import NGRDI
#from Scripts.Indexes1 import TGI
#from Scripts.Indexes1 import CLASIFICARE
import tkinter as tk
#from Scripts.Indexes1 import Medii
#from Scripts.Indexes1 import Histograma

class ImageWindow(QWidget):
    """
    A window to display a selected image with  buttons on the left side.
    """
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Imagine selectată")
        self.image_path = image_path
        main_layout = QHBoxLayout()

        # --- Left Panel: Button ---
        button_panel_widget = QWidget()
        button_panel_layout = QVBoxLayout(button_panel_widget)
        button_panel_layout.setAlignment(Qt.AlignTop)

        # Create buttons
        self.buttons = []
        Button_names=["ExG","ExGR","GLI","Pseudo-NDVI","VARI","NGRDI","TGI","CLASIFICARE","MEDII","Histograma"]
        for i in Button_names:
            btn = QPushButton(f"Buton {i}")
            btn.clicked.connect(lambda checked, b=i: self.on_button_clicked(b))
            button_panel_layout.addWidget(btn)
            self.buttons.append(btn)

        button_panel_layout.addStretch(1)

        # --- Right Panel: Image Display ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Load and scale the pixmap
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            # Handle case where image loading fails
            QMessageBox.warning(self, "Eroare Imagine", f"Nu s-a putut încărca imaginea: {image_path}. Asigură-te că fișierul există și este un format valid.")
            self.image_label.setText("Imaginea nu a putut fi încărcată.")
            self.image_label.setStyleSheet("color: red;")
        else:
            self.image_label.setPixmap(pixmap.scaled(780, 580, Qt.KeepAspectRatio, Qt.FastTransformation))

        # Add the button panel and the image label to the main horizontal layout
        main_layout.addWidget(button_panel_widget)
        main_layout.addWidget(self.image_label)

        # Set the main layout for the ImageWindow
        self.setLayout(main_layout)

        # Adjust window size to accommodate button and image
        self.resize(980, 600)

    def run_exg_function(imag):
        ExG.run_image_processor_with_gui_exg(imag)

    def run_exgr_function(imag):
        ExGR.run_image_processor_with_gui_exgr(imag)
    def run_gli_function(imag):
        GLI.run_image_processor_with_gui_gli(imag)
    def run_pndvi_function(imag):
        PNDVI.run_image_processor_with_gui_pndvi(imag)
    def run_vari_function(imag):
        VARI.run_image_processor_with_gui_vari(imag)
    def run_ngrdi_function(imag):
        NGRDI.run_image_processor_with_gui_ngrdi(imag)
    def run_tgi_function(imag):
        TGI.run_image_processor_with_gui_tgi(imag)
    def run_clasificare_function(imag):
        prediction=CLASIFICARE.predict_image(imag)
        QMessageBox.critical(None, "Rezultat Clasificare", f"Clasificare: {prediction}")
    def run_medie_function(imag):
        root = tk.Tk()
        app = Medii.ImageCalculatorApp(root,imag)
        root.mainloop()
    def run_histograma(imag):
        Histograma.image_histogram(imag)
    function_map = {
        "ExG": run_exg_function,
        "ExGR": run_exgr_function,
        "GLI": run_gli_function,
        "Pseudo-NDVI": run_pndvi_function,
        "VARI": run_vari_function,
        "NGRDI": run_ngrdi_function,
        "TGI": run_tgi_function,
        "CLASIFICARE": run_clasificare_function,
        "MEDII": run_medie_function,
        "Histograma":run_histograma
    }

    def on_button_clicked(self, button_number):
        """
        Launches a specific script based on the button clicked.
        """

        func_to_launch = self.function_map.get(button_number)

        if func_to_launch:
            try:
                func_to_launch(self.image_path)
            except Exception as e:
                QMessageBox.critical(None, "Eroare Lansare Funcție",
                                     f"Nu s-a putut lansa funcția pentru Buton {button_number}:\n{e}")
        else:
            QMessageBox.warning(None, "Eroare Buton", f"Nicio funcție definită pentru Buton {button_number}.")


class MainApp(QMainWindow):
    """
    The main application window with buttons to open various sub-applications.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interfață Principală")
        self.resize(350, 250)

        layout = QVBoxLayout()

        self.btn_image_browser = QPushButton("Deschide browser de imagini")
        self.btn_image_browser.clicked.connect(self.open_image_browser)

        self.btn_copernicus = QPushButton("Deschide interfața Copernicus")
        self.btn_copernicus.clicked.connect(self.open_copernicus_app)

        self.btn_observation = QPushButton("Deschide interfața de observatii")
        self.btn_observation.clicked.connect(self.open_observation_app)

        layout.addWidget(self.btn_image_browser)
        layout.addWidget(self.btn_copernicus)
        layout.addWidget(self.btn_observation)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_window = None

    def open_image_browser(self):
        """
        Opens a file dialog to select an image and displays it in a new ImageWindow.
        """
        default_dir = "Copernicus_App_Downloads"
        if not os.path.exists(default_dir):
            default_dir = QStandardPaths.standardLocations(QStandardPaths.PicturesLocation)[0] if QStandardPaths.standardLocations(QStandardPaths.PicturesLocation) else ''
            if not default_dir:
                default_dir = os.getcwd()


        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selectează o imagine",
            default_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.image_window = ImageWindow(file_path)
            self.image_window.show()

    def open_copernicus_app(self):
        """
        Launches the external 'TestAPP2.py' script.
        """
        script_path = "TestAPP2.py"
        if os.path.exists(script_path):
            try:
                subprocess.Popen([sys.executable, script_path])
            except Exception as e:
                QMessageBox.critical(self, "Eroare Lansare Aplicație", f"Nu s-a putut lansa {script_path}:\n{e}")
        else:
            QMessageBox.warning(self, "Fișier Lipsă", f"Fișierul '{script_path}' nu a fost găsit. Asigură-te că este în același director cu această aplicație.")

    def open_observation_app(self):
        """
        Launches the external 'Point_on_img_2.py' script.
        """
        script_path = "Point_on_img_2.py"
        if os.path.exists(script_path):
            try:
                subprocess.Popen([sys.executable, script_path])
            except Exception as e:
                QMessageBox.critical(self, "Eroare Lansare Aplicație", f"Nu s-a putut lansa {script_path}:\n{e}")
        else:
            QMessageBox.warning(self, "Fișier Lipsă", f"Fișierul '{script_path}' nu a fost găsit. Asigură-te că este în același director cu această aplicație.")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred during application startup: {e}", file=sys.stderr)
        sys.exit(1)

