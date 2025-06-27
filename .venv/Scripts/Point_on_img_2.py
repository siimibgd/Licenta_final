import tkinter as tk
from tkinter import Toplevel, Label, Entry, Button, messagebox, filedialog, Text
from PIL import Image, ImageTk
import os
import json
import shutil

CONFIG_FILE = "config.json"  # Fișier pentru a memora ultimul director folosit

# Referințe globale pentru Tkinter
root_global = None
canvas_global = None
add_point_button = None
delete_point_button = None
main_img_tk_global = None

# Datele aplicației
current_points_data = {}
project_directory = ""
points_data_file = ""

# Starea aplicației
adding_point_mode = False
deleting_point_mode = False
main_img_original_dims = (0, 0)
main_img_current_dims = (0, 0)


# --- Funcții de Gestionare a Configurației ---
def save_config(last_dir):
    """Salvează ultimul director folosit în fișierul de configurare."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump({"last_project_dir": last_dir}, f)

def load_config():
    """Încarcă ultimul director folosit din fișierul de configurare, dacă există."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            try:
                config = json.load(f)
                return config.get("last_project_dir")
            except json.JSONDecodeError:
                return None
    return None

# --- Funcții de Gestionare a Proiectului și Datelor ---
def setup_project_paths(proj_dir, main_image_path):
    """Setează căile globale și numele unic al fișierului de date."""
    global project_directory, points_data_file
    project_directory = proj_dir
    image_filename = os.path.basename(main_image_path)
    filename_without_ext, _ = os.path.splitext(image_filename)
    json_filename = f"{filename_without_ext}_points.json"
    points_data_file = os.path.join(project_directory, json_filename)
    print(f"Fișierul de date pentru această sesiune este: {points_data_file}")

def load_points_from_file():
    """Încarcă punctele din fișierul JSON specific sesiunii curente."""
    global current_points_data
    current_points_data = {}  # Resetează datele înainte de încărcare
    if os.path.exists(points_data_file):
        with open(points_data_file, 'r', encoding='utf-8') as f:
            try:
                current_points_data = json.load(f)
                print(f"Puncte încărcate din {points_data_file}")
            except json.JSONDecodeError:
                messagebox.showerror("Eroare", f"Fișierul {points_data_file} este corupt.")
    else:
        print(f"Fișierul de date '{points_data_file}' nu există. Se începe cu 0 puncte.")

def save_points_to_file():
    """Salvează punctele curente în fișierul JSON specific sesiunii curente."""
    if not project_directory or not points_data_file:
        return
    with open(points_data_file, 'w', encoding='utf-8') as f:
        json.dump(current_points_data, f, indent=4, ensure_ascii=False)
    print(f"Puncte salvate în {points_data_file}")

# --- Funcții pentru Interfața Grafică ---
def show_associated_image(image_path, point_name, description, parent_window):
    """Afișează detaliile (imagine, descriere) pentru un punct."""
    if not os.path.isabs(image_path):
        full_image_path = os.path.join(project_directory, image_path)
    else:
        full_image_path = image_path

    new_window = Toplevel(parent_window)
    new_window.title(f"Detalii: {point_name}")
    new_window.transient(parent_window)
    new_window.grab_set()

    try:

        Label(new_window, text=point_name, font=("Arial", 14, "bold")).pack(pady=(10, 5))
        if description:
            Label(new_window, text=description, font=("Arial", 10), wraplength=500, justify=tk.LEFT).pack(padx=10,
                                                                                                          pady=(0, 10))
        img_pil = Image.open(full_image_path)
        img_pil.thumbnail((800, 600), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_pil)
        img_label = Label(new_window, image=tk_img)
        img_label.image = tk_img
        img_label.pack(padx=10, pady=10)
    except Exception as e:
        Label(new_window, text=f"Eroare: {e}", fg="red").pack(padx=20, pady=20)
    finally:
        new_window.protocol("WM_DELETE_WINDOW", lambda: (new_window.grab_release(), new_window.destroy()))
        parent_window.wait_window(new_window)

def redraw_all_points():
    """Curăță și redesenează toate punctele pe hartă."""
    if canvas_global:
        canvas_global.delete("hotspot_tag")
        for name, data in current_points_data.items():
            original_x, original_y, _, _ = data
            scale_x = main_img_current_dims[0] / main_img_original_dims[0]
            scale_y = main_img_current_dims[1] / main_img_original_dims[1]
            display_x, display_y, radius = original_x * scale_x, original_y * scale_y, 8
            canvas_global.create_oval(display_x - radius, display_y - radius, display_x + radius, display_y + radius,
                                      fill="lime", outline="darkgreen", width=3, tags=(name, "hotspot_tag"))
            canvas_global.create_text(display_x, display_y - radius - 8, text=name, fill="cyan",
                                      font=("Arial", 11, "bold"), anchor=tk.S, tags=(name, "hotspot_tag"))

def set_mode(mode):
    """Setează modul de operare: 'add', 'delete', sau 'browse'."""
    global adding_point_mode, deleting_point_mode
    adding_point_mode = (mode == 'add')
    deleting_point_mode = (mode == 'delete')

    # Actualizează aspectul butoanelor pentru feedback vizual
    add_point_button.config(relief=tk.SUNKEN if adding_point_mode else tk.RAISED,
                            bg='orange' if adding_point_mode else 'SystemButtonFace')
    delete_point_button.config(relief=tk.SUNKEN if deleting_point_mode else tk.RAISED,
                               bg='red' if deleting_point_mode else 'SystemButtonFace')

def toggle_add_point_mode():
    """Activează/dezactivează modul de adăugare."""
    if adding_point_mode:
        set_mode('browse')
    else:
        set_mode('add')
        messagebox.showinfo("Mod Adăugare", "Modul de adăugare este ACTIV. Click pe hartă pentru a plasa un punct.")

def toggle_delete_point_mode():
    """ Activează/dezactivează modul de ștergere."""
    if deleting_point_mode:
        set_mode('browse')
    else:
        set_mode('delete')
        messagebox.showinfo("Mod Ștergere", "Modul de ștergere este ACTIV. Click pe un punct pentru a-l șterge.")

def on_map_click(event):
    """Gestionează toate click-urile pe hartă, în funcție de modul activ."""
    # Găsește cel mai apropiat punct de click
    closest_item = canvas_global.find_closest(event.x, event.y, halo=15)
    point_name = None
    if closest_item:
        tags = canvas_global.gettags(closest_item[0])
        if "hotspot_tag" in tags:
            point_name = next((tag for tag in tags if tag != "hotspot_tag"), None)

    if adding_point_mode:
        # ... codul de adăugare punct ...
        scale_x = main_img_current_dims[0] / main_img_original_dims[0]
        scale_y = main_img_current_dims[1] / main_img_original_dims[1]
        original_x, original_y = int(event.x / scale_x), int(event.y / scale_y)
        set_mode('browse')  # Ieși din modul adăugare după click
        prompt_new_point_info(root_global, original_x, original_y)

    elif deleting_point_mode:
        if point_name:
            if messagebox.askyesno("Confirmare Ștergere",
                                   f"Sunteți sigur că doriți să ștergeți punctul '{point_name}'?"):
                del current_points_data[point_name]
                save_points_to_file()
                redraw_all_points()
                print(f"Punctul '{point_name}' a fost șters.")

    else:  # Mod 'browse'
        if point_name:
            _, _, img_path, desc = current_points_data[point_name]
            show_associated_image(img_path, point_name, desc, root_global)

def prompt_new_point_info(parent_window, x, y):
    """Dialog pentru a introduce detaliile noului punct."""
    # (adăugarea numelui, căii imaginii, descrierii, copierea fișierului, etc.)
    dialog = Toplevel(parent_window)
    dialog.title("Adaugă Punct Nou")
    dialog.transient(parent_window)
    dialog.grab_set()

    Label(dialog, text="Nume Punct:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    name_entry = Entry(dialog, width=50)
    name_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
    name_entry.focus_set()

    Label(dialog, text="Imagine Asociată:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    path_entry = Entry(dialog, width=50)
    path_entry.grid(row=1, column=1, padx=5, pady=5)

    def browse_image():
        file_path = filedialog.askopenfilename(title="Selectează Imagine",
                                               filetypes=[("Imagini", "*.png *.jpg *.jpeg")])
        if file_path:
            path_entry.delete(0, tk.END)
            path_entry.insert(0, file_path)

    Button(dialog, text="Răsfoiește...", command=browse_image).grid(row=1, column=2, padx=5, pady=5)

    Label(dialog, text="Scurtă Descriere:").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
    desc_text = Text(dialog, width=50, height=4, wrap=tk.WORD)
    desc_text.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

    def on_ok():
        point_name, original_img_path, description = name_entry.get().strip(), path_entry.get().strip(), desc_text.get(
            "1.0", tk.END).strip()
        if not all([point_name, original_img_path]):
            messagebox.showwarning("Date Incomplete", "Numele și calea imaginii sunt obligatorii.")
            return
        if point_name in current_points_data:
            messagebox.showwarning("Nume Existent", "Un punct cu acest nume există deja.")
            return
        if not os.path.exists(original_img_path):
            messagebox.showerror("Eroare", "Fișierul imagine nu a fost găsit.")
            return

        associated_img_dir = os.path.join(project_directory, "associated_images")
        os.makedirs(associated_img_dir, exist_ok=True)
        try:
            new_img_path = os.path.join(associated_img_dir, os.path.basename(original_img_path))
            shutil.copy(original_img_path, new_img_path)
            relative_img_path = os.path.relpath(new_img_path, project_directory)
            current_points_data[point_name] = [x, y, relative_img_path, description]
            save_points_to_file()
            redraw_all_points()
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Eroare la Salvare", f"Nu s-a putut copia fișierul imagine: {e}")

    Button(dialog, text="Salvează", command=on_ok).grid(row=3, column=1, pady=10)
    Button(dialog, text="Anulează", command=dialog.destroy).grid(row=3, column=2, pady=10)

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent_window.wait_window(dialog)

# --- Funcția Principală a Aplicației ---
def create_interactive_image_app(main_image_path):
    """Creează și configurează fereastra principală a aplicației."""
    global root_global, canvas_global, add_point_button, delete_point_button, main_img_tk_global
    global main_img_original_dims, main_img_current_dims

    root_global = tk.Tk()
    root_global.title(
        f"Analiză Interactivă - {os.path.basename(project_directory)} / {os.path.basename(main_image_path)}")

    load_points_from_file()

    try:
        main_img_pil = Image.open(main_image_path)
        main_img_original_dims = main_img_pil.size
        # Scalare imagine
        scale = min((root_global.winfo_screenwidth() * 0.9) / main_img_original_dims[0],
                    (root_global.winfo_screenheight() * 0.8) / main_img_original_dims[1])
        main_img_current_dims = (int(main_img_original_dims[0] * scale), int(main_img_original_dims[1] * scale))
        main_img_pil = main_img_pil.resize(main_img_current_dims, Image.LANCZOS)
        main_img_tk_global = ImageTk.PhotoImage(main_img_pil)
    except Exception as e:
        messagebox.showerror("Eroare la Pornire", f"Nu s-a putut încărca imaginea principală:\n{e}")
        root_global.destroy()
        return

    # --- Panoul de Control  ---
    control_frame = tk.Frame(root_global)
    control_frame.pack(pady=5)

    add_point_button = Button(control_frame, text="Adaugă Punct", command=toggle_add_point_mode, width=20)
    add_point_button.pack(side=tk.LEFT, padx=5)

    delete_point_button = Button(control_frame, text="Șterge Punct", command=toggle_delete_point_mode, width=20)
    delete_point_button.pack(side=tk.LEFT, padx=5)

    Button(control_frame, text="Încarcă Alt Proiect", command=change_project, width=20).pack(side=tk.LEFT, padx=5)

    # Canvas pentru imagine
    canvas_global = tk.Canvas(root_global, width=main_img_current_dims[0], height=main_img_current_dims[1])
    canvas_global.pack()
    canvas_global.create_image(0, 0, anchor=tk.NW, image=main_img_tk_global)
    canvas_global.bind('<Button-1>', on_map_click)

    redraw_all_points()

    root_global.protocol("WM_DELETE_WINDOW", lambda: (save_points_to_file(), root_global.destroy()))
    root_global.mainloop()

# --- Funcții pentru a gestiona ciclul de viață al aplicației ---
def change_project():
    """Închide fereastra curentă și reia procesul de selecție."""
    save_points_to_file()
    root_global.destroy()
    start_application()  # Reia de la început

def start_application():
    """Funcția care gestionează pornirea și selecția proiectului/imaginii."""
    root_temp = tk.Tk()
    root_temp.withdraw()

    last_dir = load_config()  # Încarcă ultimul director folosit

    # Pasul 1: Selectare director
    selected_dir = filedialog.askdirectory(title="Selectați Directorul Proiectului", initialdir=last_dir)
    if not selected_dir:
        root_temp.destroy()
        return

    save_config(selected_dir)  # Salvează directorul selectat pentru data viitoare

    # Pasul 2: Selectare imagine
    main_image_file = filedialog.askopenfilename(
        title="Selectați Imaginea Principală",
        initialdir=selected_dir,  # Deschide în directorul proiectului
        filetypes=[("Imagini", "*.png *.jpg *.jpeg"), ("Toate fișierele", "*.*")]
    )
    if not main_image_file:
        root_temp.destroy()
        return

    root_temp.destroy()

    # Pasul 3: Configurare căi și pornire aplicație
    setup_project_paths(selected_dir, main_image_file)
    create_interactive_image_app(main_image_file)


# --- Punctul de Start al Execuției ---
if __name__ == "__main__":
    start_application()