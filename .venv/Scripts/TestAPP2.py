import sys
import os
import re # Import regex for date extraction
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QScrollArea, QMessageBox, QInputDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile, QWebEngineDownloadItem, QWebEnginePage
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QUrl, Qt, QStandardPaths, QTimer, QSettings, pyqtSignal


APP_NAME = "CopernicusBrowserApp"
ORG_NAME = "Licenta"


default_download_dir = os.path.join(QStandardPaths.writableLocation(QStandardPaths.DownloadLocation), "Copernicus_App_Downloads")


os.makedirs(default_download_dir, exist_ok=True)

class ImageViewer(QWidget):
    """
    A simple widget to display an image with a scroll area.
    Used to show downloaded images.
    """
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle(f"Preview: {os.path.basename(image_path)}")

        # Load image and scale it
        label = QLabel()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Image Error", f"Could not load image: {image_path}")
            self.deleteLater() # Close the viewer if image loading fails
            return

        # Scale pixmap to fit initially, but keep original aspect ratio
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter) # Center image if smaller than view

        # Add scroll area for larger images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) # Allow the widget to resize with scroll area
        scroll.setWidget(label)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        self.resize(min(pixmap.width() + 40, 1000), min(pixmap.height() + 40, 800)) # Adjust initial window size
        self.show()

class BrowserApp(QMainWindow):
    """
    Main application window containing the QWebEngineView for the Copernicus Browser.
    Handles page loading, download requests, and status updates.
    """
    download_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Copernicus Browser App")
        self.resize(1200, 800)

        # --- Settings for persistent download path ---
        self.settings = QSettings(ORG_NAME, APP_NAME)
        # Retrieve stored download path or use the default
        self.current_download_dir = self.settings.value("download_path", default_download_dir, type=str)
        os.makedirs(self.current_download_dir, exist_ok=True) # Ensure it exists

        self.webview = QWebEngineView()

        # --- QWebEngineProfile Setup ---
        # A custom profile allows setting specific behaviors like download path
        self.profile = QWebEngineProfile("CopernicusProfile", self)
        self.profile.setDownloadPath(self.current_download_dir)
        self.profile.downloadRequested.connect(self.handle_download)

        
        self.page = QWebEnginePage(self.profile, self)
        self.webview.setPage(self.page)

        # --- UI Layout ---
        self.status_label = QLabel("Initializing browser...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(25)
        self.status_label.setStyleSheet("font-size: 11pt; color: #333;")

        layout = QVBoxLayout()
        layout.addWidget(self.webview)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # --- Initial Information Messages ---
        # Inform user where files will be saved
        QMessageBox.information(self, "Download Folder",
                                f"Downloaded files will be saved to:\n{self.current_download_dir}\n")

        # --- Loading Control ---
        self.url = QUrl("https://browser.dataspace.copernicus.eu/")
        #self.url = QUrl("https://google.com") # For testing general browser functionality
        self.load_attempts = 0
        self.max_attempts = 3 # Number of times to retry loading the URL

        # Connect signals for page loading
        self.webview.loadStarted.connect(self.on_load_started)
        self.webview.loadProgress.connect(self.on_load_progress)
        self.webview.loadFinished.connect(self.on_load_finished)

        self.download_status_signal.connect(self.update_status_label)

        self.load_url()

    def update_status_label(self, message):
        """Updates the status label text."""
        self.status_label.setText(message)

    def load_url(self):
        """Initiates loading of the target URL, with retry logic."""
        self.load_attempts += 1
        self.update_status_label(f"Loading {self.url.toString()} (Attempt {self.load_attempts}/{self.max_attempts})...")
        self.webview.load(self.url)

    def on_load_started(self):
        """Called when the page load begins."""
        self.update_status_label("Loading page started...")

    def on_load_progress(self, progress):
        """Called as the page loads, providing a progress percentage."""
        self.update_status_label(f"Loading page... {progress}%")

    def on_load_finished(self, ok):
        """Called when the page finishes loading."""
        if ok:
            self.update_status_label("Page loaded successfully!")
            # Reset attempts on successful load
            self.load_attempts = 0
        else:
            self.update_status_label(f"Failed to load page (Attempt {self.load_attempts}/{self.max_attempts})")
            if self.load_attempts < self.max_attempts:
                # Retry after a delay if not successful and within max attempts
                QTimer.singleShot(2000, self.load_url) # Retry after 2 seconds
            else:
                self.update_status_label("Failed to load page after multiple attempts. Please check internet connection.")
                QMessageBox.critical(self, "Loading Error",
                                     f"Could not load '{self.url.toString()}' after {self.max_attempts} attempts.\n"
                                     "Please check your internet connection and try again.")

    def get_user_location_input(self):
        """
        Opens a QInputDialog to get a location string from the user.
        Returns the entered string or None if cancelled.
        """
        
        default_location = "Timișoara" 

        location, ok = QInputDialog.getText(self, "Enter Location",
                                            "Please enter a location name for the image:",
                                            text=default_location)
        if ok and location:
            return "".join(c for c in location if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        return None

    def handle_download(self, item: QWebEngineDownloadItem):
        """
        Handles download requests from the QWebEngineView.
        Extracts date, gets user location, renames files, and tracks download status.
        """
        suggested_name = item.suggestedFileName() or "downloaded_file"

        # 1. Extract Date from suggested_name
        first_date = self.extract_date_from_filename(suggested_name)
        if first_date:
            date_part = first_date
        else:
            date_part = "unknown_date"
            print(f"Warning: Could not extract date from filename: {suggested_name}")


        # 2. Get Location from User
        user_location = self.get_user_location_input()
        if user_location is None: 
            QMessageBox.warning(self, "Download Cancelled", "Image download was cancelled because no location was provided.")
            item.cancel() # Cancel the download
            return

        base_name_without_date_and_ext = suggested_name
        if first_date:
             base_name_without_date_and_ext = re.sub(r'^\d{4}-\d{2}-\d{2}(?:-\d{2}_\d{2})?\S*', '', suggested_name).strip('-_')

        
        base_name_without_date_and_ext = re.sub(r'_+', '_', base_name_without_date_and_ext)
        if base_name_without_date_and_ext.startswith('_'):
            base_name_without_date_and_ext = base_name_without_date_and_ext[1:]
        if base_name_without_date_and_ext.endswith('_'):
            base_name_without_date_and_ext = base_name_without_date_and_ext[:-1]

        # Extract original extension
        original_base, original_ext = os.path.splitext(suggested_name)
        if not original_ext: 
            original_ext = ".zip" 

        new_filename_parts = [date_part, user_location]
        if base_name_without_date_and_ext:
            new_filename_parts.append(base_name_without_date_and_ext)

        new_filename = "_".join(part for part in new_filename_parts if part).replace("__", "_") # Remove any double underscores
        new_filename = new_filename.strip('_') + original_ext

        new_filename = "".join(c for c in new_filename if c.isalnum() or c in ('.', '_', '-')).strip()

        initial_path = os.path.join(self.current_download_dir, new_filename)
        final_path = self.get_unique_filename(initial_path)

        item.setPath(final_path)

        # Connect signals for download status
        item.downloadProgress.connect(lambda bytes_received, bytes_total:
                                     self.on_download_progress(item, bytes_received, bytes_total))
        item.finished.connect(lambda: self.on_download_finished(item, final_path))

        item.accept() # Accept the download
        self.download_status_signal.emit(f"Download started: {os.path.basename(final_path)}")
        QMessageBox.information(self, "Download Started",
                                f"Download of '{os.path.basename(final_path)}' has started.\n"
                                f"It will be saved to: {self.current_download_dir}",
                                QMessageBox.Ok)

    def extract_date_from_filename(self, filename):
        
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return match.group(1)
        return None


    def get_unique_filename(self, path):
        """Generates a unique filename by appending a counter if file exists."""
        base, ext = os.path.splitext(path)
        if not ext: 
            ext = ""
        i = 1
        unique_path = path
        while os.path.exists(unique_path):
            unique_path = f"{base}_{i}{ext}"
            i += 1
        return unique_path

    def on_download_progress(self, item: QWebEngineDownloadItem, bytes_received, bytes_total):
        """Updates the status label with download progress."""
        if bytes_total > 0:
            progress = (bytes_received / bytes_total) * 100
            self.download_status_signal.emit(f"Downloading {os.path.basename(item.path())}: {progress:.1f}%")
        else:
            self.download_status_signal.emit(f"Downloading {os.path.basename(item.path())}...")

    def on_download_finished(self, item: QWebEngineDownloadItem, path):
        """Called when a download finishes (successfully or not)."""
        if item.state() == QWebEngineDownloadItem.DownloadCompleted:
            self.download_status_signal.emit(f"Download completed: {os.path.basename(path)}")
            self.show_image_if_valid(path)
        elif item.state() == QWebEngineDownloadItem.DownloadCancelled:
            self.download_status_signal.emit(f"Download cancelled: {os.path.basename(path)}")
            QMessageBox.warning(self, "Download Cancelled",
                                f"Download of '{os.path.basename(path)}' was cancelled.")
        else:
            # Handle other states like DownloadInterrupted, DownloadFailed
            self.download_status_signal.emit(f"Download failed: {os.path.basename(path)} (Reason: {item.interruptReason()})")
            QMessageBox.critical(self, "Download Failed",
                                 f"Failed to download '{os.path.basename(path)}'.\n"
                                 f"Reason: {item.interruptReason()}")

    def show_image_if_valid(self, path):
        """
        Checks if the downloaded file is a valid image and opens it in ImageViewer.
        """
        # List of common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
        file_ext = os.path.splitext(path)[1].lower()

        if file_ext in image_extensions:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                QTimer.singleShot(100, lambda: ImageViewer(path))
            else:
                self.download_status_signal.emit(f"Downloaded image file invalid or empty: {os.path.basename(path)}")
        else:
            self.download_status_signal.emit(f"Downloaded file is not a recognized image type: {os.path.basename(path)}")

# --- Main Application Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)

    window = BrowserApp()
    window.show()

    sys.exit(app.exec_())