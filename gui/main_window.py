"""Fenêtre principale de l'interface de surveillance."""

import os
import time
import tempfile
from datetime import datetime
import threading
import traceback

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk

import config
from utils.logger import get_logger
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.watermark import Watermark
from core.alert_manager import AlertManager
from gui.enrollment_dialog import EnrollmentDialog
from gui.verification_dialog import VerificationDialog

logger = get_logger("MainWindow")


class SurveillanceWindow:
    """Fenêtre principale de la plateforme de surveillance."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
        self.root = tk.Tk()
        self.root.title("Plateforme de Surveillance Vidéo Intelligente")
        self.root.geometry("1200x700")
        self.root.configure(bg='#1a1a1a')
        
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer(threshold=config.RECOGNITION_THRESHOLD)
        self.watermark = Watermark(method=config.WATERMARK_METHOD, jpeg_quality=config.JPEG_QUALITY)
        self.alert_manager = AlertManager(db_manager)
        
        self.cap = None
        self.is_streaming = False
        self.stream_thread = None
        
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.frames_since_update = 0
        self.current_frame = None
        
        self._setup_ui()
        self._setup_menu()
    
    def _setup_ui(self):
        top_frame = tk.Frame(self.root, bg='#2b2b2b', height=50)
        top_frame.pack(side='top', fill='x')
        top_frame.pack_propagate(False)
        
        tk.Label(top_frame, text="SURVEILLANCE VIDÉO", font=('Arial', 18, 'bold'),
                bg='#2b2b2b', fg='white').pack(side='left', padx=20)
        
        self.status_label = tk.Label(top_frame, text="Statut: Arrêté",
                font=('Arial', 12), bg='#2b2b2b', fg='#ff0000')
        self.status_label.pack(side='right', padx=20)
        
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True)
        
        left_panel = tk.Frame(main_frame, bg='#1a1a1a')
        left_panel.pack(side='left', fill='both', expand=True)
        
        self.canvas = tk.Canvas(left_panel, bg='#000000', width=640, height=480)
        self.canvas.pack(pady=10)
        
        control_frame = tk.Frame(left_panel, bg='#2b2b2b')
        control_frame.pack(pady=10)
        
        self.webcam_btn = tk.Button(control_frame, text="Webcam", command=self.start_webcam,
                bg='#107c10', fg='white', font=('Arial', 12), width=12)
        self.webcam_btn.grid(row=0, column=0, padx=5)
        
        self.file_btn = tk.Button(control_frame, text="Ouvrir Vidéo", command=self.open_video,
                bg='#0078d7', fg='white', font=('Arial', 12), width=12)
        self.file_btn.grid(row=0, column=1, padx=5)
        
        self.capture_btn = tk.Button(control_frame, text="Capturer", command=self.manual_capture,
                bg='#ff8c00', fg='white', font=('Arial', 12), width=12, state='disabled')
        self.capture_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Arrêter", command=self.stop_stream,
                bg='#e81123', fg='white', font=('Arial', 12), width=12, state='disabled')
        self.stop_btn.grid(row=0, column=3, padx=5)
        
        right_panel = tk.Frame(main_frame, bg='#2b2b2b', width=350)
        right_panel.pack(side='right', fill='both')
        right_panel.pack_propagate(False)
        
        stats_frame = tk.LabelFrame(right_panel, text="Statistiques",
                bg='#2b2b2b', fg='white', font=('Arial', 11, 'bold'))
        stats_frame.pack(pady=10, padx=10, fill='x')
        
        self.fps_label = tk.Label(stats_frame, text="FPS: 0", bg='#2b2b2b', fg='#00ff00')
        self.fps_label.pack(anchor='w', padx=10, pady=2)
        
        self.faces_label = tk.Label(stats_frame, text="Visages: 0", bg='#2b2b2b', fg='white')
        self.faces_label.pack(anchor='w', padx=10, pady=2)
        
        self.total_label = tk.Label(stats_frame, text="Total: 0", bg='#2b2b2b', fg='white')
        self.total_label.pack(anchor='w', padx=10, pady=2)
        
        logs_frame = tk.LabelFrame(right_panel, text="Derniers Accès",
                bg='#2b2b2b', fg='white', font=('Arial', 11, 'bold'))
        logs_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.logs_tree = ttk.Treeview(logs_frame, columns=('status', 'name', 'time'), show='headings', height=15)
        self.logs_tree.heading('status', text='Statut')
        self.logs_tree.heading('name', text='Nom')
        self.logs_tree.heading('time', text='Heure')
        self.logs_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        self._refresh_logs()
    
    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Ouvrir Vidéo...", command=self.open_video)
        file_menu.add_command(label="Exporter Logs...", command=self.export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.quit_app)
        
        db_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Base de données", menu=db_menu)
        db_menu.add_command(label="Enregistrer personne...", command=self.open_enrollment_dialog)
        db_menu.add_command(label="Voir personnes...", command=self.show_persons_list)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Outils", menu=tools_menu)
        tools_menu.add_command(label="Vérifier Tatouage...", command=self.open_verification_dialog)
    
    def start_webcam(self):
        if self.is_streaming:
            return
        
        logger.info("Tentative d'ouverture webcam...")
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logger.error("Impossible d'ouvrir la webcam")
            messagebox.showerror("Erreur", "Impossible d'ouvrir la webcam\nVérifiez qu'elle n'est pas utilisée par une autre application")
            return
        
        logger.info("Webcam ouverte avec succès")
        self._start_stream()
        self.status_label.config(text="Statut: Webcam Active", fg='#00ff00')
        self.webcam_btn.config(state='disabled')
        self.file_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.capture_btn.config(state='normal')
    
    def _start_stream(self):
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._video_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Thread vidéo démarré")
    
    def _video_loop(self):
        frame_skip = max(1, config.FRAME_SKIP)
        skip_counter = 0
        
        logger.info("Boucle vidéo démarrée")
        
        while self.is_streaming and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Frame non lu")
                    break
                
                self.current_frame = frame.copy()
                skip_counter += 1
                
                if skip_counter < frame_skip:
                    continue
                skip_counter = 0
                
                faces = self.detector.detect(frame)
                face_labels = ['unknown'] * len(faces)
                
                for i, bbox in enumerate(faces):
                    face = self.detector.crop_face(frame, bbox)
                    if face is not None:
                        result = self.recognizer.recognize(face, self.db_manager)
                        person_name = result.get('name', 'Inconnu') or 'Inconnu'
                        status = result.get('status', 'unknown')
                        confidence = result.get('confidence', 0.0)
                        
                        if result.get('name'):
                            face_labels[i] = status
                        
                        watermarked = self.watermark.embed(
                            frame.copy(), config.CAMERA_ID, person_name, status
                        )
                        
                        self.alert_manager.process_alert(
                            person_name, status, confidence, config.CAMERA_ID, frame, watermarked
                        )
                
                display_frame = self.detector.draw_faces(frame, faces, face_labels)
                self.frame_count += 1
                
                self.root.after(0, lambda: self._update_display(display_frame, len(faces)))
                self.root.after(0, self._update_fps)
                self.root.after(0, self._refresh_stats)
                
            except Exception as e:
                logger.error(f"Erreur boucle: {e}")
                traceback.print_exc()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Boucle vidéo terminée")
    
    def _update_display(self, frame, face_count):
        try:
            display = cv2.resize(frame, (640, 480))
            # OpenCV fournit du BGR; Tkinter/Pillow attend du RGB.
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            self.canvas.delete('all')
            self.canvas.create_image(320, 240, image=photo)
            self.canvas.photo = photo
            
            self.faces_label.config(text=f"Visages: {face_count}")
        except Exception as e:
            logger.error(f"Erreur affichage: {e}")
    
    def _update_fps(self):
        self.frames_since_update += 1
        elapsed = time.time() - self.last_fps_update
        if elapsed >= 1.0:
            self.fps = self.frames_since_update / elapsed
            self.frames_since_update = 0
            self.last_fps_update = time.time()
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
    
    def _refresh_stats(self):
        try:
            stats = self.alert_manager.get_alert_stats()
            self.total_label.config(text=f"Total: {stats['total_count']}")
            self._refresh_logs()
        except:
            pass
    
    def _refresh_logs(self):
        try:
            for item in self.logs_tree.get_children():
                self.logs_tree.delete(item)
            
            for log in self.db_manager.get_recent_logs(15):
                time_str = log['timestamp'].split(' ')[1] if log['timestamp'] else '-'
                self.logs_tree.insert('', 0, values=(log['status'].upper(), log['person_name'], time_str))
        except:
            pass
    
    def open_video(self):
        if self.is_streaming:
            self.stop_stream()
        
        path = filedialog.askopenfilename(title="Sélectionner une vidéo",
                filetypes=[("Vidéo", "*.mp4 *.avi *.mov"), ("Tous", "*.*")])
        
        if not path:
            return
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'ouvrir la vidéo")
            return
        
        self._start_stream()
        self.status_label.config(text=f"Statut: {os.path.basename(path)}", fg='#00ff00')
        self.webcam_btn.config(state='disabled')
        self.file_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
    
    def stop_stream(self):
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.webcam_btn.config(state='normal')
        self.file_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')
        self.status_label.config(text="Statut: Arrêté", fg='#ff0000')
    
    def manual_capture(self):
        if self.current_frame is None:
            return
        try:
            watermarked = self.watermark.embed(self.current_frame.copy(), config.CAMERA_ID, "MANUEL", "unknown")
            filename = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(config.CAPTURES_DIR, filename)
            os.makedirs(config.CAPTURES_DIR, exist_ok=True)
            cv2.imwrite(path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, 85])
            messagebox.showinfo("Succès", f"Capture: {filename}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
    
    def open_enrollment_dialog(self):
        EnrollmentDialog(self.root, self.db_manager, self.recognizer, self.detector).show()
        self.recognizer._try_load_model()
    
    def open_verification_dialog(self):
        VerificationDialog(self.root).show()
    
    def show_persons_list(self):
        persons = self.db_manager.get_all_persons()
        list_window = tk.Toplevel(self.root)
        list_window.title("Personnes")
        list_window.geometry("400x300")
        tk.Label(list_window, text=f"{len(persons)} personne(s)", font=('Arial', 14)).pack()
        for p in persons:
            status = "✓" if p['authorized'] else "✗"
            tk.Label(list_window, text=f"{status} {p['name']}").pack()
        tk.Button(list_window, text="Fermer", command=list_window.destroy).pack()
    
    def export_logs(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.db_manager.export_logs_csv(path)
            messagebox.showinfo("Succès", "Logs exportés")
    
    def quit_app(self):
        self.stop_stream()
        self.root.quit()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.mainloop()