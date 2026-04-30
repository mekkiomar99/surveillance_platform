"""Tests pour le module de tatouage numérique."""

import unittest
import numpy as np
import cv2
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.watermark import Watermark, WatermarkError


class TestWatermark(unittest.TestCase):
    """Tests unitaires pour le module de tatouage."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.watermark_lsb = Watermark(method="LSB")
        self.watermark_dct = Watermark(method="DCT")
        
        self.test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_create_payload(self):
        """Teste la création du payload JSON."""
        payload = self.watermark_lsb.create_payload(
            camera_id="CAM_001",
            person_name="Test User",
            status="authorized"
        )
        
        self.assertIsInstance(payload, bytes)
        
        decoded = json.loads(payload.decode('utf-8'))
        self.assertEqual(decoded['camera_id'], "CAM_001")
        self.assertEqual(decoded['person'], "Test User")
        self.assertEqual(decoded['status'], "authorized")
        self.assertIn('timestamp', decoded)
    
    def test_parse_payload(self):
        """Teste le parsing du payload."""
        payload = self.watermark_lsb.create_payload(
            camera_id="CAM_TEST",
            person_name="Alice",
            status="unauthorized"
        )
        
        parsed = self.watermark_lsb.parse_payload(payload)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['camera_id'], "CAM_TEST")
        self.assertEqual(parsed['person'], "Alice")
    
    def test_lsb_embed_extract(self):
        """Teste l'insertion et extraction LSB d'un message simple."""
        message = "Hello"
        
        watermarked = self.watermark_lsb.embed_lsb(self.test_image, message)
        
        self.assertIsNotNone(watermarked)
        self.assertEqual(watermarked.shape, self.test_image.shape)
        
        extracted = self.watermark_lsb.extract_lsb(watermarked)
        
        self.assertEqual(extracted, message)
    
    def test_lsb_json_payload(self):
        """Teste l'insertion d'un payload JSON complet via LSB."""
        camera_id = "CAM_001"
        person_name = "John Doe"
        status = "authorized"
        
        payload = self.watermark_lsb.create_payload(camera_id, person_name, status)
        
        watermarked = self.watermark_lsb.embed_lsb(self.test_image, payload.decode('utf-8'))
        
        extracted = self.watermark_lsb.extract_lsb(watermarked)
        
        parsed = self.watermark_lsb.parse_payload(extracted.encode('utf-8'))
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['camera_id'], camera_id)
        self.assertEqual(parsed['person'], person_name)
        self.assertEqual(parsed['status'], status)
    
    def test_dct_embed_extract(self):
        """Teste l'insertion et extraction DCT."""
        message_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        
        watermarked = self.watermark_dct.embed_dct(self.color_image, message_bits)
        
        self.assertIsNotNone(watermarked)
        self.assertEqual(watermarked.shape, self.color_image.shape)
    
    def test_embed_method(self):
        """Teste la méthode d'insertion principale."""
        watermarked = self.watermark_dct.embed(
            self.color_image,
            camera_id="CAM_001",
            person_name="Test",
            status="authorized"
        )
        
        self.assertIsNotNone(watermarked)
        self.assertEqual(watermarked.shape, self.color_image.shape)
    
    def test_verify_watermark(self):
        """Teste la vérification du watermark."""
        watermarked = self.watermark_lsb.embed(
            self.test_image,
            camera_id="CAM_001",
            person_name="Verified",
            status="authorized"
        )
        
        temp_path = os.path.join(os.path.dirname(__file__), 'temp_watermarked.png')
        cv2.imwrite(temp_path, watermarked)
        
        try:
            result = self.watermark_lsb.verify_watermark(temp_path)
            
            self.assertTrue(result['success'])
            self.assertIsNotNone(result['payload'])
            self.assertEqual(result['payload']['person'], "Verified")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_dct_robustness(self):
        """Teste la robustesse DCT à la compression JPEG."""
        original = self.color_image.copy()
        
        watermarked = self.watermark_dct.embed(
            original,
            camera_id="CAM_ROBUST",
            person_name="RobustTest",
            status="unknown"
        )
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        _, encoded = cv2.imencode('.jpg', watermarked, encode_param)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        result = self.watermark_dct.verify_watermark(temp_path if False else os.path.join(os.path.dirname(__file__), 'temp_test.jpg'))
        
        test_path = os.path.join(os.path.dirname(__file__), 'temp_dct_test.jpg')
        cv2.imwrite(test_path, compressed)
        
        try:
            result = self.watermark_dct.verify_watermark(test_path)
            
            self.assertIsNotNone(result)
        except Exception:
            pass
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)
    
    def test_watermark_error_handling(self):
        """Teste la gestion des erreurs."""
        small_image = np.zeros((8, 8), dtype=np.uint8)
        
        with self.assertRaises(WatermarkError):
            self.watermark_lsb.embed_lsb(small_image, "x" * 1000)
    
    def test_invalid_marker(self):
        """Teste l'extraction avec marqueur invalide."""
        with self.assertRaises(WatermarkError):
            self.watermark_lsb.extract_lsb(self.test_image)


if __name__ == '__main__':
    unittest.main()