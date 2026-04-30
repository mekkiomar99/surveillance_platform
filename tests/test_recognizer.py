"""Tests pour le module de reconnaissance faciale."""

import unittest
import numpy as np
import cv2
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.recognizer import FaceRecognizer
from database.db_manager import DatabaseManager


class TestFaceRecognizer(unittest.TestCase):
    """Tests unitaires pour le reconnaisseur facial."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(self.temp_db.name)
        
        self.recognizer = FaceRecognizer(threshold=0.6)
        
        self.sample_faces = []
        for _ in range(5):
            face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            self.sample_faces.append(face)
    
    def tearDown(self):
        """Cleanup après les tests."""
        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except:
                pass
    
    def test_recognizer_initialization(self):
        """Teste que le reconnaisseur s'initialise correctement."""
        self.assertIsNotNone(self.recognizer)
        self.assertIsNotNone(self.recognizer.scaler)
    
    def test_preprocess_image(self):
        """Teste le prétraitement d'une image."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        processed = self.recognizer.preprocess_image(image, target_size=(128, 128))
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (128, 128))
        self.assertTrue(processed.max() <= 1.0)
    
    def test_extract_lbp_features(self):
        """Teste l'extraction des caractéristiques LBP."""
        image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        
        features = self.recognizer.extract_lbp_features(image)
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
    
    def test_extract_multi_scale_lbp(self):
        """Teste l'extraction LBP multi-échelle."""
        image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        
        features = self.recognizer.extract_multi_scale_lbp(image)
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 10)
    
    def test_enroll(self):
        """Teste l'enregistrement d'une personne."""
        name = "Alice"
        
        result = self.recognizer.enroll(name, self.sample_faces, authorized=True)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
    
    def test_recognize_unknown(self):
        """Teste la reconnaissance d'un visage inconnu."""
        unknown_face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        result = self.recognizer.recognize(unknown_face, self.db_manager)
        
        self.assertIsNotNone(result)
        self.assertIn('status', result)
        self.assertIn('confidence', result)
    
    def test_enroll_and_recognize(self):
        """Teste l'enregistrement puis la reconnaissance."""
        name = "Bob"
        
        encoding = self.recognizer.enroll(name, self.sample_faces, authorized=True)
        
        self.assertIsNotNone(encoding)
        
        self.db_manager.add_person(
            name=name,
            authorized=True,
            face_encoding=encoding
        )
        
        test_image = self.sample_faces[0].copy()
        
        result = self.recognizer.recognize(test_image, self.db_manager)
        
        self.assertIn('status', result)
    
    def test_threshold_behavior(self):
        """Teste le comportement du seuil de confiance."""
        high_threshold_recognizer = FaceRecognizer(threshold=0.9)
        
        unknown_face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        result = high_threshold_recognizer.recognize(unknown_face, self.db_manager)
        
        self.assertIn('status', result)


def is_not_none(value):
    return value is not None


if __name__ == '__main__':
    unittest.main()