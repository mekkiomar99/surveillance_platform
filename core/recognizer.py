"""Module de reconnaissance faciale avec HOG, LBP et KNN."""

import cv2
import numpy as np
import pickle
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
import config

logger = get_logger("FaceRecognizer")


class FaceRecognizer:
    """Reconnaissance faciale hybride HOG + LBP + KNN avec Alignement."""
    
    def __init__(self, model_path=None, threshold=0.6):
        """
        Initialise le reconnaisseur facial.
        """
        self.model_path = model_path or config.MODEL_PATH
        self.threshold = threshold
        
        self.classifier = None # Renommé de svm_classifier à classifier
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.reverse_labels = {}
        self.is_trained = False
        
        # Filtrage temporel pour éviter les sauts de noms
        self.history = []
        self.history_size = 5
        
        # Détecteur d'yeux pour l'alignement
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        self._try_load_model()
    
    def _try_load_model(self):
        """Tente de charger un modèle existant."""
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
                logger.info(f"Modèle chargé: {self.model_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle: {e}")
    
    def align_face(self, gray_img):
        """
        Aligne le visage de manière robuste.
        """
        try:
            # Réduire la taille pour une détection plus rapide
            small = cv2.resize(gray_img, (0, 0), fx=0.5, fy=0.5)
            eyes = self.eye_cascade.detectMultiScale(small, 1.1, 5)
            
            if len(eyes) >= 2:
                # Remettre à l'échelle originale
                eyes = eyes * 2
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
                right_eye = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
                
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dy, dx))
                
                center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0]), flags=cv2.INTER_CUBIC)
                return aligned
        except:
            pass
        return gray_img

    def extract_features(self, image):
        """
        Extrait des caractéristiques hybrides robustes.
        """
        if image is None:
            return None
        
        try:
            img_float = image.astype(np.float32)
            
            # 1. HOG (Structure robuste)
            hog_feats = hog(
                img_float, 
                orientations=8, 
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), 
                block_norm='L2-Hys'
            )
            
            # 2. LBP (Texture)
            img_uint8 = (img_float * 255).astype(np.uint8)
            lbp = local_binary_pattern(img_uint8, 8, 1, method='uniform')
            (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            return np.hstack([hog_feats, lbp_hist])
        except Exception as e:
            logger.error(f"Erreur extraction: {e}")
            return None
    
    def preprocess_image(self, image, target_size=None):
        """
        Prétraitement complet avec normalisation de lumière.
        """
        if image is None:
            return None
        
        if target_size is None:
            target_size = config.FACE_SIZE
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Alignement
            aligned = self.align_face(gray)
            
            # 2. Redimensionnement
            resized = cv2.resize(aligned, target_size)
            
            # 3. Double Normalisation (CLAHE + HE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            equalized = clahe.apply(resized)
            equalized = cv2.equalizeHist(equalized) # Normalisation globale
            
            return equalized.astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"Erreur prétraitement: {e}")
            return None
    
    def enroll(self, name, images, authorized=False):
        """Enregistre une personne."""
        try:
            features_list = []
            for img in images:
                processed = self.preprocess_image(img)
                if processed is not None:
                    features = self.extract_features(processed)
                    if features is not None:
                        features_list.append(features)
            
            if not features_list:
                return None
            
            return np.mean(features_list, axis=0)
        except Exception as e:
            logger.error(f"Erreur enrollment: {e}")
            return False
    
    def prepare_training_data(self, db_manager):
        """Prépare les données."""
        try:
            persons = db_manager.get_all_persons()
            if not persons: return None
            
            X_list, y_list = [], []
            for person in persons:
                encoding = db_manager.get_person_encoding(person['id'])
                if encoding is not None:
                    X_list.append(np.array(encoding, dtype=np.float32))
                    y_list.append(str(person['name']))
            
            if not X_list: return None
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            unique_names = [str(name) for name in np.unique(y)]
            self.label_encoder = {name: idx for idx, name in enumerate(unique_names)}
            self.reverse_labels = {idx: name for name, idx in self.label_encoder.items()}
            
            y_encoded = np.array([self.label_encoder[str(name)] for name in y])
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y_encoded
        except Exception as e:
            logger.error(f"Erreur prépa: {e}")
            return None
    
    def train(self, db_manager):
        """Entraîne le modèle KNN (plus stable que SVM pour peu de données)."""
        try:
            data = self.prepare_training_data(db_manager)
            if data is None:
                self.classifier = None
                self.is_trained = False
                if os.path.exists(self.model_path): os.remove(self.model_path)
                return False
            
            X_train, y_train = data
            
            # KNN est plus intuitif : il cherche les voisins les plus proches.
            # On utilise 3 voisins pour une décision par vote.
            n_neighbors = min(3, len(X_train))
            self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
            self.classifier.fit(X_train, y_train)
            
            self.is_trained = True
            self.save_model(self.model_path)
            logger.info(f"Modèle KNN entraîné ({len(X_train)} personnes)")
            return True
        except Exception as e:
            logger.error(f"Erreur entraînement: {e}")
            return False
    
    def recognize(self, face_img, db_manager=None):
        """Reconnaissance avec filtrage temporel."""
        try:
            processed = self.preprocess_image(face_img)
            if processed is None: return {'name': None, 'status': 'unknown', 'confidence': 0.0}
            
            features = self.extract_features(processed)
            if features is None: return {'name': None, 'status': 'unknown', 'confidence': 0.0}
            
            if not self.is_trained or self.classifier is None:
                if db_manager: self._load_from_db(db_manager)
                if not self.is_trained:
                    if db_manager: return self._recognize_by_distance(features, db_manager)
                    return {'name': None, 'status': 'unknown', 'confidence': 0.0}

            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Calcul des probabilités
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            predicted_name = self.reverse_labels[predicted_idx]
            
            # FILTRAGE TEMPOREL : Vote majoritaire sur les dernières frames
            self.history.append(predicted_name)
            if len(self.history) > self.history_size: self.history.pop(0)
            
            # Le nom final est celui qui apparaît le plus dans l'historique
            final_name = max(set(self.history), key=self.history.count)
            
            if confidence < self.threshold:
                return {'name': None, 'status': 'unknown', 'confidence': float(confidence)}
            
            if db_manager:
                person = db_manager.get_person_by_name(final_name)
                status = 'authorized' if person and person['authorized'] else 'unauthorized'
            else:
                status = 'unknown'
            
            return {'name': final_name, 'status': status, 'confidence': float(confidence)}
            
        except Exception as e:
            logger.error(f"Erreur reconnaissance: {e}")
            return {'name': None, 'status': 'unknown', 'confidence': 0.0}

    def _recognize_by_distance(self, features, db_manager):
        """Fallback distance exponentielle."""
        try:
            persons = db_manager.get_all_persons()
            if not persons: return {'name': None, 'status': 'unknown', 'confidence': 0.0}

            best_person, best_distance = None, None
            for person in persons:
                encoding = db_manager.get_person_encoding(person['id'])
                if encoding is None: continue
                dist = float(np.linalg.norm(features - np.array(encoding, dtype=np.float32)))
                if best_distance is None or dist < best_distance:
                    best_distance, best_person = dist, person

            if best_person is None: return {'name': None, 'status': 'unknown', 'confidence': 0.0}
            confidence = float(np.exp(-best_distance))
            if confidence < self.threshold: return {'name': None, 'status': 'unknown', 'confidence': confidence}
            return {
                'name': best_person['name'],
                'status': 'authorized' if best_person['authorized'] else 'unauthorized',
                'confidence': confidence
            }
        except:
            return {'name': None, 'status': 'unknown', 'confidence': 0.0}
    
    def _load_from_db(self, db_manager):
        self.train(db_manager)
    
    def save_model(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'reverse_labels': self.reverse_labels,
                'is_trained': self.is_trained
            }
            with open(path, 'wb') as f: pickle.dump(model_data, f)
        except Exception as e: logger.error(f"Erreur save: {e}")
    
    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.classifier = model_data.get('classifier') or model_data.get('svm_classifier')
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.reverse_labels = model_data['reverse_labels']
            self.is_trained = model_data['is_trained']
        except Exception as e:
            logger.error(f"Erreur load: {e}")
            raise
    
    def get_feature_vector(self, face_img):
        processed = self.preprocess_image(face_img)
        return self.extract_features(processed) if processed is not None else None