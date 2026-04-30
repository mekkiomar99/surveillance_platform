"""Module de détection de visages avec Haar Cascade."""

import cv2
import numpy as np
import os
from utils.logger import get_logger

logger = get_logger("FaceDetector")

# MTCNN désactivé - utilisation de Haar Cascade uniquement pour éviter les problèmes TensorFlow
MTCNN_AVAILABLE = False
MTCNN = None


class FaceDetector:
    """Détecteur de visages multi-méthode."""
    
    def __init__(self, haarcascade_path=None):
        """
        Initialise le détecteur de visages.
        
        Args:
            haarcascade_path: Chemin vers le fichier Haar Cascade XML
        """
        self.haarcascade_path = haarcascade_path
        self.face_cascade = None
        self.mtcnn_detector = None
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialise les détecteurs disponibles."""
        self._init_haar()
        self._init_mtcnn()
    
    def _init_haar(self):
        """Initialise le détecteur Haar Cascade."""
        if self.haarcascade_path and os.path.exists(self.haarcascade_path):
            cascade_path = self.haarcascade_path
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                logger.warning("Haar Cascade par défaut non trouvé")
                return
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.error("Impossible de charger Haar Cascade")
                self.face_cascade = None
            else:
                logger.info(f"Haar Cascade chargé: {cascade_path}")
        except Exception as e:
            logger.error(f"Erreur chargement Haar: {e}")
            self.face_cascade = None
    
    def _init_mtcnn(self):
        """MTCNN désactivé - utilisation de Haar Cascade uniquement."""
        self.mtcnn_detector = None
    
    def detect_haar(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Détection via Haar Cascade.
        
        Args:
            frame: Image NumPy (BGR)
            scale_factor: Facteur de mise à l'échelle
            min_neighbors: Nombre minimum de voisins
            min_size: Taille minimale du visage
        
        Returns:
            Liste de bounding boxes [(x, y, w, h), ...]
        """
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            
        except Exception as e:
            logger.error(f"Erreur détection Haar: {e}")
            return []
    
    def detect_mtcnn(self, frame):
        """
        Détection via MTCNN.
        
        Args:
            frame: Image NumPy (BGR)
        
        Returns:
            Liste de bounding boxes [(x, y, w, h), ...]
        """
        if self.mtcnn_detector is None:
            return []
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.mtcnn_detector.detect_faces(rgb)
            
            boxes = []
            for det in detections:
                if det['confidence'] > 0.5:
                    x, y, w, h = det['box']
                    boxes.append((max(0, x), max(0, y), w, h))
            
            return boxes
            
        except Exception as e:
            logger.error(f"Erreur détection MTCNN: {e}")
            return []
    
    def detect(self, frame, use_haar=True):
        """
        Détection de visages avec fallback automatique.
        
        Args:
            frame: Image NumPy (BGR)
            use_haar: Si True, tente Haar d'abord puis MTCNN
        
        Returns:
            Liste de bounding boxes [(x, y, w, h), ...]
        """
        faces = []
        
        if use_haar and self.face_cascade is not None:
            faces = self.detect_haar(frame)
        
        if len(faces) == 0 and self.mtcnn_detector is not None:
            logger.debug("Fallback vers MTCNN")
            faces = self.detect_mtcnn(frame)
        
        return faces
    
    def draw_faces(self, frame, faces, labels=None):
        """
        Dessine les rectangles autour des visages détectés.
        
        Args:
            frame: Image NumPy (BGR) - modifiée en place
            faces: Liste de bounding boxes
            labels: Liste optionnelle de labels (status)
        
        Returns:
            Frame avec rectangles dessinés
        """
        colors = {
            'authorized': (0, 255, 0),      # Vert
            'unauthorized': (0, 165, 255),   # Orange
            'unknown': (0, 0, 255),           # Rouge
            'default': (255, 255, 255)        # Blanc
        }
        
        for idx, (x, y, w, h) in enumerate(faces):
            color = colors['default']
            label = ""
            
            if labels and idx < len(labels):
                label = labels[idx]
                color = colors.get(label, colors['default'])
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            if label:
                cv2.putText(
                    frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        return frame
    
    def crop_face(self, frame, bbox):
        """
        Découpe un visage depuis le frame.
        
        Args:
            frame: Image NumPy (BGR)
            bbox: Bounding box (x, y, w, h)
        
        Returns:
            Image du visage découpé ou None
        """
        x, y, w, h = bbox
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None
        
        face = frame[y:y+h, x:x+w]
        
        if face.size == 0:
            return None
        
        return face.copy()
    
    def preprocess_for_recognition(self, face_img, target_size=(128, 128)):
        """
        Prétraite l'image du visage pour la reconnaissance.
        
        Args:
            face_img: Image du visage
            target_size: Taille cible (width, height)
        
        Returns:
            Image prétraitée
        """
        if face_img is None:
            return None
        
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            resized = cv2.resize(gray, target_size)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(resized)
            
            normalized = equalized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Erreur prétraitement: {e}")
            return None
    
    def get_face_count(self, frame):
        """Retourne le nombre de visages détectés dans le frame."""
        faces = self.detect(frame)
        return len(faces)