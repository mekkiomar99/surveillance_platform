"""Dialog d'enregistrement - Version SANS PIL."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import tkinter as tk
from tkinter import messagebox
import tempfile
from PIL import Image, ImageTk

from utils.logger import get_logger
logger = get_logger("EnrollmentDialog")


class EnrollmentDialog:
    """Dialog simplifié sans PIL."""
    
    def __init__(self, parent, db_manager, recognizer, detector):
        self.parent = parent
        self.db_manager = db_manager
        self.recognizer = recognizer
        self.detector = detector
        
        self.window = tk.Toplevel(parent)
        self.window.title("Enregistrer une Personne")
        self.window.geometry("700x550")
        self.window.configure(bg='#2b2b2b')
        
        self.cap = None
        self.is_capturing = False
        self.captured_images = []
        self.target_images = 10
        self.video_job = None
        self.temp_dir = tempfile.mkdtemp()
        self.frame_count = 0
        
        self._setup_ui()
    
    def _setup_ui(self):
        tk.Label(
            self.window,
            text="Enregistrement d'une Nouvelle Personne",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='white'
        ).pack(pady=10)
        
        form = tk.Frame(self.window, bg='#2b2b2b')
        form.pack(pady=5)
        
        tk.Label(form, text="Nom:", bg='#2b2b2b', fg='white').grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.name_entry = tk.Entry(form, width=25)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(form, text="Autorisé:", bg='#2b2b2b', fg='white').grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.authorized_var = tk.BooleanVar(value=False)
        tk.Checkbutton(form, variable=self.authorized_var, bg='#2b2b2b', fg='white', selectcolor='#2b2b2b').grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        self.capture_count = tk.Label(
            self.window,
            text=f"Captures: 0 / {self.target_images}",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 12)
        )
        self.capture_count.pack(pady=5)
        
        # Canvas pour afficher la vidéo
        self.canvas = tk.Canvas(self.window, width=480, height=360, bg='#000000')
        self.canvas.pack(pady=10)
        
        buttons = tk.Frame(self.window, bg='#2b2b2b')
        buttons.pack(pady=10)
        
        self.start_btn = tk.Button(buttons, text="Démarrer Webcam", command=self.start_capture,
                bg='#0078d7', fg='white', width=14)
        self.start_btn.grid(row=0, column=0, padx=3)
        
        self.capture_btn = tk.Button(buttons, text="Capturer", command=self.capture_face,
                bg='#107c10', fg='white', width=14, state='disabled')
        self.capture_btn.grid(row=0, column=1, padx=3)
        
        self.train_btn = tk.Button(buttons, text="Enregistrer", command=self.train_model,
                bg='#e81123', fg='white', width=14, state='disabled')
        self.train_btn.grid(row=0, column=2, padx=3)
        
        tk.Button(buttons, text="Fermer", command=self.close,
                bg='#666666', fg='white', width=14).grid(row=0, column=3, padx=3)
        
        self.status = tk.Label(self.window, text="Prêt", bg='#2b2b2b', fg='#00ff00')
        self.status.pack(pady=5)
    
    def _frame_to_photoimage(self, frame):
        """Convertit un frame OpenCV en PhotoImage via Pillow."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            return ImageTk.PhotoImage(image=image)
        except Exception as e:
            logger.error(f"Erreur conversion: {e}")
            return None
    
    def start_capture(self):
        if self.is_capturing:
            return
        
        # Fallback Windows: certains pilotes renvoient un flux noir sans CAP_DSHOW.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'ouvrir la webcam")
            return
        
        self.is_capturing = True
        self.captured_images = []
        self.status.config(text="Webcam active", fg='#00ff00')
        self.capture_btn.config(state='normal')
        self.start_btn.config(state='disabled')
        self._update_video()
    
    def _update_video(self):
        if not self.is_capturing or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_capture()
            return
        
        faces = self.detector.detect(frame)
        
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        self.current_frame = frame.copy()
        self.frame_count += 1
        
        try:
            display = cv2.resize(frame, (480, 360))
            photo = self._frame_to_photoimage(display)
            if photo:
                self.canvas.create_image(240, 180, image=photo)
                self.canvas.image = photo  # Garder référence
        except Exception as e:
            logger.error(f"Erreur affichage: {e}")
        
        self.video_job = self.window.after(66, self._update_video)
    
    def capture_face(self):
        if not self.is_capturing or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        faces = self.detector.detect(frame)
        if not faces:
            messagebox.showwarning("Attention", "Aucun visage détecté")
            return
        
        face = self.detector.crop_face(frame, faces[0])
        if face is not None:
            self.captured_images.append(face)
            self.capture_count.config(text=f"Captures: {len(self.captured_images)} / {self.target_images}")
            
            if len(self.captured_images) >= self.target_images:
                self._on_complete()
    
    def _on_complete(self):
        self.is_capturing = False
        if self.video_job:
            self.window.after_cancel(self.video_job)
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status.config(text=f"{len(self.captured_images)} captures terminées!", fg='#00ff00')
        self.train_btn.config(state='normal')
        self.capture_btn.config(state='disabled')
        self.start_btn.config(state='normal')
        messagebox.showinfo("Succès", "10 captures effectuées!\nCliquez sur 'Enregistrer'")
    
    def train_model(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Erreur", "Entrez un nom")
            return
        if len(self.captured_images) == 0:
            messagebox.showerror("Erreur", "Aucune capture")
            return
        
        self.status.config(text="Enregistrement...", fg='#ffff00')
        
        try:
            features = self.recognizer.enroll(name, self.captured_images, self.authorized_var.get())
            
            if features is not False:
                self.db_manager.add_person(name=name, authorized=self.authorized_var.get(), face_encoding=features)
                self.recognizer.train(self.db_manager)
                self.status.config(text=f"{name} enregistré!", fg='#00ff00')
                messagebox.showinfo("Succès", f"{name} enregistré!")
            else:
                messagebox.showerror("Erreur", "Échec extraction")
        except Exception as e:
            logger.error(f"Erreur: {e}")
            messagebox.showerror("Erreur", str(e))
    
    def stop_capture(self):
        self.is_capturing = False
        if self.video_job:
            self.window.after_cancel(self.video_job)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state='normal')
    
    def close(self):
        self.stop_capture()
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
        self.window.destroy()
    
    def show(self):
        self.window.transient(self.parent)
        self.window.grab_set()
        self.parent.wait_window(self.window)