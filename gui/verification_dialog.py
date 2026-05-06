"""Dialog de vérification du tatouage numérique."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
from utils.logger import get_logger
import config
from core.watermark import Watermark

logger = get_logger("VerificationDialog")


class VerificationDialog:
    """Dialog pour vérifier l'intégrité du tatouage dans une image."""
    
    def __init__(self, parent):
        """Initialise le dialog."""
        self.parent = parent
        self.watermark = Watermark(method=config.WATERMARK_METHOD, jpeg_quality=config.JPEG_QUALITY)
        self.image_path = None
        self.original_image = None
        
        self.window = tk.Toplevel(parent)
        self.window.title("Vérification du Tatouage")
        self.window.geometry("900x600")
        self.window.configure(bg='#2b2b2b')
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configure l'interface utilisateur."""
        title_label = tk.Label(
            self.window,
            text="Vérification de l'Intégrité du Tatouage",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        title_label.pack(pady=10)
        
        top_frame = tk.Frame(self.window, bg='#2b2b2b')
        top_frame.pack(pady=10)
        
        self.open_btn = tk.Button(
            top_frame,
            text="Ouvrir une Image",
            command=self.open_image,
            bg='#0078d7',
            fg='white',
            font=('Arial', 12),
            padx=20,
            pady=5
        )
        self.open_btn.pack(side='left', padx=5)
        
        self.test_robustness_btn = tk.Button(
            top_frame,
            text="Tester Robustesse (JPEG Q=75)",
            command=self.test_robustness,
            bg='#107c10',
            fg='white',
            font=('Arial', 12),
            padx=20,
            pady=5,
            state='disabled'
        )
        self.test_robustness_btn.pack(side='left', padx=5)
        
        content_frame = tk.Frame(self.window, bg='#2b2b2b')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        left_panel = tk.Frame(content_frame, bg='#1a1a1a')
        left_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(
            left_panel,
            text="Image",
            bg='#1a1a1a',
            fg='white',
            font=('Arial', 12, 'bold')
        ).pack(pady=5)
        
        self.image_label = tk.Label(left_panel, bg='#1a1a1a', width=400, height=300)
        self.image_label.pack(pady=10)
        
        right_panel = tk.Frame(content_frame, bg='#2b2b2b')
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        tk.Label(
            right_panel,
            text="Informations du Tatouage",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 14, 'bold')
        ).pack(pady=10)
        
        info_frame = tk.Frame(right_panel, bg='#2b2b2b')
        info_frame.pack(fill='x', pady=5)
        
        self.info_labels = {}
        fields = [
            ('status', 'Statut:'),
            ('timestamp', 'Horodatage:'),
            ('camera_id', 'Caméra:'),
            ('person', 'Personne:'),
            ('watermark_status', 'Tatouage:')
        ]
        
        for key, label_text in fields:
            row = tk.Frame(info_frame, bg='#2b2b2b')
            row.pack(fill='x', pady=3)
            
            tk.Label(
                row,
                text=label_text,
                bg='#2b2b2b',
                fg='#888888',
                font=('Arial', 11),
                width=15,
                anchor='w'
            ).pack(side='left')
            
            value_label = tk.Label(
                row,
                text="-",
                bg='#2b2b2b',
                fg='white',
                font=('Arial', 11),
                anchor='w'
            )
            value_label.pack(side='left')
            self.info_labels[key] = value_label
        
        results_frame = tk.LabelFrame(
            right_panel,
            text="Résultat de la Vérification",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 12, 'bold')
        )
        results_frame.pack(fill='x', pady=10, padx=5)
        
        self.result_text = tk.Text(
            results_frame,
            height=8,
            bg='#1a1a1a',
            fg='white',
            font=('Courier', 10),
            state='disabled'
        )
        self.result_text.pack(padx=10, pady=10, fill='x')
        
        button_frame = tk.Frame(self.window, bg='#2b2b2b')
        button_frame.pack(pady=10)
        
        self.close_btn = tk.Button(
            button_frame,
            text="Fermer",
            command=self.close,
            bg='#666666',
            fg='white',
            font=('Arial', 12),
            padx=20,
            pady=5
        )
        self.close_btn.pack()
    
    def open_image(self):
        """Ouvre une image depuis le disque."""
        path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[
                ("Images JPEG", "*.jpg *.jpeg"),
                ("Images PNG", "*.png"),
                ("Toutes les images", "*.*")
            ]
        )
        
        if not path:
            return
        
        self.image_path = path
        self.original_image = cv2.imread(path)
        
        if self.original_image is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image")
            return
        
        self._display_image(self.original_image)
        self._verify_watermark()
    
    def _display_image(self, image):
        """Affiche l'image avec PIL."""
        try:
            display = cv2.resize(image, (400, 300))
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img_pil)
            
            self.image_label.imgtk = photo
            self.image_label.configure(image=photo)
        except Exception as e:
            logger.error(f"Erreur affichage image: {e}")
    
    def _verify_watermark(self):
        """Vérifie le watermark dans l'image chargée."""
        if self.original_image is None:
            return
        
        result = self.watermark.verify_watermark(self.image_path)
        
        if result['success']:
            payload = result['payload']
            
            self.info_labels['status'].config(text="TROUVÉ", fg='#00ff00')
            self.info_labels['timestamp'].config(text=payload.get('timestamp', '-'))
            self.info_labels['camera_id'].config(text=payload.get('camera_id', '-'))
            self.info_labels['person'].config(text=payload.get('person', '-'))
            self.info_labels['watermark_status'].config(text="Valide", fg='#00ff00')
            
            self._update_result_text(
                f"✓ Tatouage détecté et validé\n"
                f"  Méthode: {config.WATERMARK_METHOD}\n"
                f"  Timestamp: {payload.get('timestamp', '-')}\n"
                f"  Caméra: {payload.get('camera_id', '-')}\n"
                f"  Personne: {payload.get('person', '-')}\n"
                f"  Statut: {payload.get('status', '-')}",
                'green'
            )
            
            self.test_robustness_btn.config(state='normal')
            
        else:
            self.info_labels['status'].config(text="NON TROUVÉ", fg='#ff0000')
            self.info_labels['timestamp'].config(text="-")
            self.info_labels['camera_id'].config(text="-")
            self.info_labels['person'].config(text="-")
            self.info_labels['watermark_status'].config(text="Invalide", fg='#ff0000')
            
            self._update_result_text(
                f"✗ Tatouage non détecté\n"
                f"  Erreur: {result.get('error', 'Inconnue')}\n"
                f"  L'image peut ne pas contenir de tatouage ou avoir été corrompue.",
                'red'
            )
            
            self.test_robustness_btn.config(state='disabled')
    
    def _update_result_text(self, text, color):
        """Met à jour le texte de résultat."""
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        color_map = {
            'green': '#00ff00',
            'red': '#ff0000',
            'yellow': '#ffff00',
            'white': '#ffffff'
        }
        
        self.result_text.insert(tk.END, text)
        self.result_text.tag_configure('color', foreground=color_map.get(color, 'white'))
        self.result_text.tag_add('color', 1.0, tk.END)
        
        self.result_text.config(state='disabled')
    
    def test_robustness(self):
        """Teste la robustesse du watermark à la compression JPEG."""
        if self.original_image is None:
            return
        
        self._update_result_text("Test de robustesse en cours...\nCompression JPEG Q=75...", 'yellow')
        
        try:
            watermarked_path = self.image_path
            result = self.watermark.verify_watermark(watermarked_path)
            
            if not result['success']:
                self._update_result_text(
                    "Échec: Impossible de détecter le watermark original",
                    'red'
                )
                return
            
            robustness = self.watermark.test_robustness(self.original_image, jpeg_quality=75)
            
            if 'ber' in robustness:
                ber = robustness['ber']
                success_rate = 1 - ber
                
                status_color = 'green' if ber < 0.1 else ('yellow' if ber < 0.2 else 'red')
                status_text = "EXCELLENT" if ber < 0.05 else ("BON" if ber < 0.1 else ("MOYEN" if ber < 0.2 else "FAIBLE"))
                
                self._update_result_text(
                    f"Test de Robustesse (JPEG Q=75)\n"
                    f"{'─' * 40}\n"
                    f"Taux d'erreur (BER): {ber:.2%}\n"
                    f"Taux de réussite: {success_rate:.2%}\n"
                    f"Bits comparés: {robustness.get('bits_compared', '-')}\n"
                    f"Erreurs détectées: {robustness.get('errors', '-')}\n"
                    f"{'─' * 40}\n"
                    f"Résultat: {status_text}",
                    status_color
                )
                
                self.info_labels['watermark_status'].config(
                    text=f"{status_text} (BER: {ber:.1%})",
                    fg=color_map.get(status_color, 'white') if 'color_map' in dir() else '#00ff00'
                )
                
                colors = {'green': '#00ff00', 'yellow': '#ffff00', 'red': '#ff0000'}
                self.info_labels['watermark_status'].config(
                    text=f"{status_text} (BER: {ber:.1%})",
                    fg=colors.get(status_color, '#ffffff')
                )
            else:
                self._update_result_text(
                    f"Erreur lors du test: {robustness.get('error', 'Inconnue')}",
                    'red'
                )
                
        except Exception as e:
            logger.error(f"Erreur test robustesse: {e}")
            self._update_result_text(f"Erreur: {str(e)}", 'red')
    
    def close(self):
        """Ferme le dialog."""
        self.window.destroy()
    
    def show(self):
        """Affiche le dialog."""
        self.window.transient(self.parent)
        self.window.grab_set()
        self.parent.wait_window(self.window)