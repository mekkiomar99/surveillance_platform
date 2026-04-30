"""Tests pour le module de détection de visages."""

import unittest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    """Tests unitaires pour le détecteur de visages."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.detector = FaceDetector()
    
    def test_detector_initialization(self):
        """Teste que le détecteur s'initialise correctement."""
        self.assertIsNotNone(self.detector)
        self.assertTrue(self.detector.face_cascade is not None or self.detector.mtcnn_detector is not None)
    
    def test_detect_no_face(self):
        """Teste la détection sur une image sans visage."""
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        faces = self.detector.detect(black_image)
        
        self.assertEqual(len(faces), 0)
    
    def test_detect_frontal_face(self):
        """Teste la détection sur une image avec un visage simulé."""
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_face.png')
        
        if not os.path.exists(test_image_path):
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(canvas, (200, 100), (440, 400), (255, 200, 200), -1)
            cv2.imwrite(test_image_path, canvas)
        
        image = cv2.imread(test_image_path)
        self.assertIsNotNone(image)
        
        faces = self.detector.detect(image)
        
        self.assertIsInstance(faces, list)
    
    def test_draw_faces(self):
        """Teste le dessin des bounding boxes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [(100, 100, 200, 200), (300, 150, 150, 150)]
        
        result = self.detector.draw_faces(frame, faces, ['authorized', 'unknown'])
        
        self.assertEqual(result.shape, frame.shape)
    
    def test_crop_face(self):
        """Teste le découpage d'un visage."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)
        
        face = self.detector.crop_face(frame, bbox)
        
        self.assertIsNotNone(face)
        self.assertEqual(face.shape[0], 200)
        self.assertEqual(face.shape[1], 200)
    
    def test_preprocess_for_recognition(self):
        """Teste le prétraitement pour la reconnaissance."""
        face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        processed = self.detector.preprocess_for_recognition(face, target_size=(128, 128))
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (128, 128))
        self.assertTrue(processed.max() <= 1.0 and processed.min() >= 0.0)


if __name__ == '__main__':
    unittest.main()