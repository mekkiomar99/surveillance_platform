"""Module de reconnaissance faciale avec LBP et SVM."""

import cv2
import numpy as np
import pickle
import os
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
import config

logger = get_logger("FaceRecognizer")


class FaceRecognizer:
    """Reconnaissance faciale basée sur LBP + SVM."""
    
    def __init__(self, model_path=None, threshold=0.6):
        """
        Initialise le reconnaisseur facial.
        
        Args:
            model_path: Chemin vers le fichier modèle
            threshold: Seuil de confiance en dessous duquel -> inconnu
        """
        self.model_path = model_path or config.MODEL_PATH
        self.threshold = threshold
        
        self.svm_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.reverse_labels = {}
        self.is_trained = False
        
        self._try_load_model()
    
    def _try_load_model(self):
        """Tente de charger un modèle existant."""
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
                logger.info(f"Modèle chargé: {self.model_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle: {e}")
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """
        Extrait les caractéristiques LBP d'une image.
        
        Args:
            image: Image en niveaux de gris (normalisée 0-1)
            radius: Rayon du voisinage
            n_points: Nombre de points dans le cercle
        
        Returns:
            Histogramme LBP
        """
        if image is None:
            return None
        
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            
            n_bins = n_points + 2
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=n_bins,
                range=(0, n_bins),
                density=True
            )
            
            return hist
            
        except Exception as e:
            logger.error(f"Erreur extraction LBP: {e}")
            return None
    
    def extract_multi_scale_lbp(self, image):
        """
        Extrait des caractéristiques LBP multi-échelle.
        
        Args:
            image: Image normalisée
        
        Returns:
            Vecteur de caractéristiques concaténées
        """
        features = []
        
        scales = [(1, 8), (2, 16), (3, 24)]
        
        for radius, n_points in scales:
            hist = self.extract_lbp_features(image, radius, n_points)
            if hist is not None:
                features.extend(hist)
        
        if not features:
            return None
        
        return np.array(features)
    
    def preprocess_image(self, image, target_size=(128, 128)):
        """
        Prétraite une image pour la reconnaissance.
        
        Args:
            image: Image couleur (BGR)
            target_size: Taille cible
        
        Returns:
            Image normalisée
        """
        if image is None:
            return None
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            resized = cv2.resize(gray, target_size)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(resized)
            
            normalized = equalized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Erreur prétraitement: {e}")
            return None
    
    def enroll(self, name, images, authorized=False):
        """
        Enregistre une nouvelle personne dans le système.
        
        Args:
            name: Nom de la personne
            images: Liste d'images de visages
            authorized: True si la personne est autorisée
        
        Returns:
            True si succès
        """
        try:
            features_list = []
            
            for img in images:
                processed = self.preprocess_image(img)
                if processed is not None:
                    features = self.extract_multi_scale_lbp(processed)
                    if features is not None:
                        features_list.append(features)
            
            if len(features_list) == 0:
                logger.error("Aucune caractéristique extraite")
                return None
            
            features_array = np.array(features_list)
            mean_features = np.mean(features_array, axis=0)
            
            logger.info(f"{name}: {len(features_list)} images enrollées")
            return mean_features
            
        except Exception as e:
            logger.error(f"Erreur enrollment: {e}")
            return False
    
    def prepare_training_data(self, db_manager):
        """
        Prépare les données d'entraînement depuis la base.
        
        Args:
            db_manager: Gestionnaire de base de données
        
        Returns:
            (X_train, y_train) ou None
        """
        try:
            persons = db_manager.get_all_persons()
            
            if not persons:
                logger.warning("Aucune personne dans la base")
                return None
            
            X_list = []
            y_list = []
            
            for person in persons:
                encoding = db_manager.get_person_encoding(person['id'])
                if encoding is not None:
                    if isinstance(encoding, np.ndarray):
                        X_list.append(encoding)
                    else:
                        X_list.append(np.array(encoding, dtype=np.float32))
                    y_list.append(str(person['name']))
            
            if len(X_list) < 2:
                logger.warning("Pas assez de personnes pour entraîner")
                return None
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            unique_names = [str(name) for name in np.unique(y)]
            self.label_encoder = {name: idx for idx, name in enumerate(unique_names)}
            self.reverse_labels = {idx: name for name, idx in self.label_encoder.items()}
            
            y_encoded = np.array([self.label_encoder[str(name)] for name in y])
            
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y_encoded
            
        except Exception as e:
            logger.error(f"Erreur préparation données: {e}")
            return None
    
    def train(self, db_manager):
        """
        Entraîne le modèle SVM sur les données de la base.
        
        Args:
            db_manager: Gestionnaire de base de données
        
        Returns:
            True si succès
        """
        try:
            data = self.prepare_training_data(db_manager)
            
            if data is None:
                return False
            
            X_train, y_train = data
            
            self.svm_classifier = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            self.svm_classifier.fit(X_train, y_train)
            
            self.is_trained = True
            
            self.save_model(self.model_path)
            
            logger.info(f"Modèle entraîné sur {len(X_train)}样本")
            return True
            
        except Exception as e:
            logger.error(f"Erreur entraînement: {e}")
            return False
    
    def recognize(self, face_img, db_manager=None):
        """
        Reconnaît un visage.
        
        Args:
            face_img: Image du visage (BGR)
            db_manager: Gestionnaire de base de données (optionnel)
        
        Returns:
            Dict avec name, status, confidence
        """
        try:
            processed = self.preprocess_image(face_img)
            if processed is None:
                return {'name': None, 'status': 'unknown', 'confidence': 0.0}
            
            features = self.extract_multi_scale_lbp(processed)
            if features is None:
                return {'name': None, 'status': 'unknown', 'confidence': 0.0}
            
            if not self.is_trained or self.svm_classifier is None:
                if db_manager:
                    logger.debug("Tentative de chargement du modèle depuis la base")
                    self._load_from_db(db_manager)
                
                if not self.is_trained:
                    # Fallback: permet la reconnaissance même avec une seule personne enregistrée.
                    if db_manager:
                        return self._recognize_by_distance(features, db_manager)
                    return {'name': None, 'status': 'unknown', 'confidence': 0.0, 'reason': 'no_model'}
            
            if db_manager and not self.is_trained:
                self._load_from_db(db_manager)
            
            if not self.is_trained:
                if db_manager:
                    return self._recognize_by_distance(features, db_manager)
                return {'name': None, 'status': 'unknown', 'confidence': 0.0, 'reason': 'no_model'}
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            probabilities = self.svm_classifier.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            if confidence < self.threshold:
                return {
                    'name': None,
                    'status': 'unknown',
                    'confidence': float(confidence)
                }
            
            predicted_name = self.reverse_labels[predicted_idx]
            
            if db_manager:
                person = db_manager.get_person_by_name(predicted_name)
                if person:
                    status = 'authorized' if person['authorized'] else 'unauthorized'
                else:
                    status = 'unknown'
            else:
                status = 'unknown'
            
            return {
                'name': predicted_name,
                'status': status,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Erreur reconnaissance: {e}")
            return {'name': None, 'status': 'unknown', 'confidence': 0.0, 'error': str(e)}

    def _recognize_by_distance(self, features, db_manager):
        """Fallback de reconnaissance par distance si le SVM n'est pas disponible."""
        try:
            persons = db_manager.get_all_persons()
            if not persons:
                return {'name': None, 'status': 'unknown', 'confidence': 0.0, 'reason': 'no_person'}

            best_person = None
            best_distance = None

            for person in persons:
                encoding = db_manager.get_person_encoding(person['id'])
                if encoding is None:
                    continue

                ref = np.array(encoding, dtype=np.float32)
                if ref.shape != features.shape:
                    continue

                dist = float(np.linalg.norm(features - ref))
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_person = person

            if best_person is None:
                return {'name': None, 'status': 'unknown', 'confidence': 0.0, 'reason': 'no_encoding'}

            # Convertit une distance en pseudo-confiance (0..1), puis applique le seuil global.
            confidence = 1.0 / (1.0 + best_distance)
            if confidence < self.threshold:
                return {'name': None, 'status': 'unknown', 'confidence': float(confidence)}

            return {
                'name': best_person['name'],
                'status': 'authorized' if best_person['authorized'] else 'unauthorized',
                'confidence': float(confidence)
            }
        except Exception as e:
            logger.error(f"Erreur fallback distance: {e}")
            return {'name': None, 'status': 'unknown', 'confidence': 0.0, 'error': str(e)}
    
    def _load_from_db(self, db_manager):
        """Charge les données depuis la base et réentraîne si nécessaire."""
        try:
            data = self.prepare_training_data(db_manager)
            if data is not None:
                X_train, y_train = data
                if len(X_train) >= 2:
                    self.svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
                    self.svm_classifier.fit(X_train, y_train)
                    self.is_trained = True
                    logger.info("Modèle réentraîné depuis la base")
        except Exception as e:
            logger.error(f"Erreur chargement depuis DB: {e}")
    
    def save_model(self, path):
        """Sauvegarde le modèle sur disque."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_data = {
                'svm_classifier': self.svm_classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'reverse_labels': self.reverse_labels,
                'is_trained': self.is_trained,
                'threshold': self.threshold
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Modèle sauvegardé: {path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
    
    def load_model(self, path):
        """Charge le modèle depuis disque."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.svm_classifier = model_data['svm_classifier']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.reverse_labels = model_data['reverse_labels']
            self.is_trained = model_data['is_trained']
            # Conserve le seuil configuré au runtime (config.py) au lieu d'un seuil
            # potentiellement obsolète stocké dans un ancien modèle.
            
            logger.info("Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def get_feature_vector(self, face_img):
        """
        Retourne le vecteur de caractéristiques pour une image de visage.
        
        Args:
            face_img: Image du visage
        
        Returns:
            Vecteur numpy ou None
        """
        processed = self.preprocess_image(face_img)
        if processed is None:
            return None
        
        return self.extract_multi_scale_lbp(processed)