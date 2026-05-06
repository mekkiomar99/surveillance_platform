"""Module de tatouage numérique pour les captures de surveillance.

Implémente deux méthodes:
- LSB (Least Significant Bit): Insertion invisible dans les bits de poids faible
- DCT (Discrete Cosine Transform): Insertion robuste face à la compression JPEG
"""

import json
import zlib
import numpy as np
import cv2
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("Watermark")


class WatermarkError(Exception):
    """Exception spécifique pour les erreurs de tatouage."""
    pass


class Watermark:
    """Classe principale pour le tatouage d'images."""
    
    def __init__(self, method="DCT", jpeg_quality=85):
        """
        Initialise le gestionnaire de tatouage.
        
        Args:
            method: Méthode "LSB" ou "DCT"
            jpeg_quality: Qualité JPEG pour la compression de test
        """
        self.method = method
        self.jpeg_quality = jpeg_quality
        self.watermark_marker = b'WMBRK001'  # Marqueur pour identifier les images tatouées
    
    def create_payload(self, camera_id, person_name, status, timestamp=None):
        """
        Crée le payload JSON à tatouer.
        
        Args:
            camera_id: ID de la caméra
            person_name: Nom de la personne détectée
            status: Statut (authorized, unauthorized, unknown)
            timestamp: Horodatage (par défaut: maintenant)
        
        Returns:
            Bytes du payload encodé
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        payload = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "person": person_name if person_name else "unknown",
            "status": status,
            "marker": self.watermark_marker.decode()
        }
        
        json_str = json.dumps(payload, ensure_ascii=False)
        return json_str.encode('utf-8')
    
    def parse_payload(self, data):
        """
        Parse le payload extrait.
        
        Args:
            data: Bytes du payload
        
        Returns:
            Dict avec les informations du tatouage
        """
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Erreur parsing payload: {e}")
            return None
    
    # ==================== MÉTHODE LSB ====================
    
    def _bits_to_bytes(self, bits):
        """Convertit une liste de bits en bytes."""
        bytes_list = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) < 8:
                byte_bits.extend([0] * (8 - len(byte_bits)))
            byte = 0
            for j, bit in enumerate(byte_bits):
                byte |= bit << j
            bytes_list.append(byte)
        return bytes(bytes_list)
    
    def _bytes_to_bits(self, data):
        """Convertit des bytes en liste de bits."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        return bits
    
    def embed_lsb(self, image, message):
        """
        Insère un message dans l'image via LSB.
        
        Args:
            image: Image NumPy (BGR ou grayscale)
            message: Message string à insérer
        
        Returns:
            Image tatoueée (copie)
        """
        try:
            img = image.copy()
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            h, w = img.shape
            
            header = self.watermark_marker + b':' + str(len(message)).encode() + b':'
            full_data = header + message.encode('utf-8')
            crc = zlib.crc32(full_data).to_bytes(4, 'big')
            full_data = full_data + crc
            
            bits = self._bytes_to_bits(full_data)
            total_bits = len(bits)
            
            max_bits = h * w
            if total_bits > max_bits:
                raise WatermarkError(f"Message trop long: {total_bits} bits requis, {max_bits} disponibles")
            
            idx = 0
            for i in range(h):
                for j in range(w):
                    if idx >= total_bits:
                        break
                    pixel = img[i, j]
                    pixel = (pixel & 0xFE) | bits[idx]
                    img[i, j] = pixel
                    idx += 1
                if idx >= total_bits:
                    break
            
            logger.debug(f"LSB: {total_bits} bits insérés ({len(message)} caractères)")
            return img
            
        except WatermarkError:
            raise
        except Exception as e:
            raise WatermarkError(f"Erreur insertion LSB: {e}")
    
    def extract_lsb(self, image):
        """
        Extrait un message d'une image via LSB.
        
        Args:
            image: Image tatoueée
        
        Returns:
            Message string extrait
        """
        try:
            img = image.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.uint8)
            h, w = img.shape
            bits = []
            # Recherche du marqueur
            marker_bits = []
            marker_len = len(self.watermark_marker)
            marker_found = False
            bits = []
            bit_idx = 0
            for i in range(h):
                for j in range(w):
                    bit = img[i, j] & 1
                    bits.append(bit)
                    bit_idx += 1
                    if len(bits) == marker_len * 8:
                        check_bytes = self._bits_to_bytes(bits)
                        if check_bytes == self.watermark_marker:
                            marker_found = True
                            break
                if marker_found:
                    break
            if not marker_found:
                raise WatermarkError("Marqueur non trouvé dans l'image")
            # On commence la lecture du header juste après le marqueur
            i = bit_idx
            # Consommer le premier ':' (8 bits)
            first_colon_bits = []
            for _ in range(8):
                if i >= h * w:
                    raise WatermarkError("Header invalide (fin prématurée)")
                row = i // w
                col = i % w
                first_colon_bits.append(img[row, col] & 1)
                i += 1
            if self._bits_to_bytes(first_colon_bits) != b':':
                raise WatermarkError("Header invalide (pas de premier ':')")
            # Lecture de la taille du message (jusqu'au second ':')
            size_octets = []
            colon_found = False
            while i < h * w:
                byte_bits = []
                for _ in range(8):
                    if i >= h * w:
                        break
                    row = i // w
                    col = i % w
                    byte_bits.append(img[row, col] & 1)
                    i += 1
                if len(byte_bits) < 8:
                    break
                bval = self._bits_to_bytes(byte_bits)
                if bval == b':':
                    colon_found = True
                    break
                size_octets.append(bval)
            if not colon_found or not size_octets:
                raise WatermarkError("Header invalide")
            try:
                msg_len_str = b''.join(size_octets).decode()
                msg_len = int(msg_len_str)
            except Exception as e:
                raise WatermarkError(f"Header invalide: {e}")
            # Lecture du message
            msg_bits = []
            for j in range(i, i + msg_len * 8):
                row = j // w
                col = j % w
                msg_bits.append(img[row, col] & 1)
            # Lecture du CRC
            crc_bits = []
            for j in range(i + msg_len * 8, i + msg_len * 8 + 32):
                row = j // w
                col = j % w
                crc_bits.append(img[row, col] & 1)
            msg_bytes = self._bits_to_bytes(msg_bits)
            crc_bytes = self._bits_to_bytes(crc_bits)
            expected_crc = zlib.crc32(self.watermark_marker + b':' + str(msg_len).encode() + b':' + msg_bytes).to_bytes(4, 'big')
            if crc_bytes != expected_crc:
                raise WatermarkError("CRC invalide - corruption détectée")
            message = msg_bytes.decode('utf-8')
            logger.debug(f"LSB: Message extrait ({len(message)} caractères)")
            return message
            # Lecture de la taille du message (après le marqueur, jusqu'à ':')
            size_octets = []
            colon_found = False
            i = bit_idx
            while i < h * w:
                byte_bits = []
                for _ in range(8):
                    if i >= h * w:
                        break
                    row = i // w
                    col = i % w
                    byte_bits.append(img[row, col] & 1)
                    i += 1
                if len(byte_bits) < 8:
                    break
                bval = self._bits_to_bytes(byte_bits)
                if bval == b':':
                    colon_found = True
                    break
                size_octets.append(bval)
            if not colon_found or not size_octets:
                raise WatermarkError("Header invalide")
            try:
                msg_len_str = b''.join(size_octets).decode()
                msg_len = int(msg_len_str)
            except Exception as e:
                raise WatermarkError(f"Header invalide: {e}")
            # Lecture du message
            msg_bits = []
            for j in range(i, i + msg_len * 8):
                row = j // w
                col = j % w
                msg_bits.append(img[row, col] & 1)
            # Lecture du CRC
            crc_bits = []
            for j in range(i + msg_len * 8, i + msg_len * 8 + 32):
                row = j // w
                col = j % w
                crc_bits.append(img[row, col] & 1)
            msg_bytes = self._bits_to_bytes(msg_bits)
            crc_bytes = self._bits_to_bytes(crc_bits)
            expected_crc = zlib.crc32(self.watermark_marker + b':' + str(msg_len).encode() + b':' + msg_bytes).to_bytes(4, 'big')
            if crc_bytes != expected_crc:
                raise WatermarkError("CRC invalide - corruption détectée")
            message = msg_bytes.decode('utf-8')
            logger.debug(f"LSB: Message extrait ({len(message)} caractères)")
            return message
        except WatermarkError:
            raise
        except Exception as e:
            raise WatermarkError(f"Erreur extraction LSB: {e}")
    
    # ==================== MÉTHODE DCT ====================
    
    def _dct_2d(self, block):
        """DCT 2D sur un bloc 8x8."""
        return cv2.dct(block.astype(np.float64))
    
    def _idct_2d(self, block):
        """DCT inverse 2D sur un bloc 8x8."""
        return cv2.idct(block).astype(np.uint8)
    
    def _get_coefficient_index(self, row, col):
        """Retourne l'index du coefficient DCT pour une position donnée."""
        zig_zag = [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ]
        return zig_zag[row][col]
    
    def embed_dct(self, image, watermark_bits):
        """
        Insère un watermark via DCT (robuste JPEG).
        
        Args:
            image: Image NumPy
            watermark_bits: Liste de bits à insérer
        
        Returns:
            Image tatoueée (copie)
        """
        try:
            img = image.copy()
            
            if len(img.shape) == 3:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                y_channel = img_yuv[:, :, 0]
            else:
                y_channel = img.copy()
            
            h, w = y_channel.shape
            h_pad = (h // 8) * 8
            w_pad = (w // 8) * 8
            
            if h_pad < 8 or w_pad < 8:
                raise WatermarkError("Image trop petite pour DCT")
            
            y_channel = y_channel[:h_pad, :w_pad]
            bits_array = np.array(watermark_bits, dtype=np.int8)
            
            bit_idx = 0
            total_bits = len(bits_array)
            alpha = 20
            
            for i in range(0, h_pad, 8):
                for j in range(0, w_pad, 8):
                    if bit_idx >= total_bits:
                        break
                    
                    block = y_channel[i:i+8, j:j+8].astype(np.float64)
                    dct_block = self._dct_2d(block)
                    
                    row, col = 4, 4
                    
                    dct_value = dct_block[row, col]
                    
                    if bits_array[bit_idx] == 1:
                        if dct_value < 0:
                            dct_block[row, col] = -alpha if dct_value > -alpha else dct_value
                        else:
                            dct_block[row, col] = alpha if dct_value < alpha else dct_value
                    else:
                        dct_block[row, col] = 0 if abs(dct_value) < alpha / 2 else dct_value
                    
                    idct_block = self._idct_2d(dct_block)
                    y_channel[i:i+8, j:j+8] = np.clip(idct_block, 0, 255).astype(np.uint8)
                    
                    bit_idx += 1
            
            if len(img.shape) == 3:
                img_yuv[:, :, 0] = y_channel
                result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            else:
                result = y_channel
            
            logger.debug(f"DCT: {total_bits} bits insérés")
            return result
            
        except WatermarkError:
            raise
        except Exception as e:
            raise WatermarkError(f"Erreur insertion DCT: {e}")
    
    def extract_dct(self, image):
        """
        Extrait un watermark d'une image via DCT.
        
        Args:
            image: Image tatoueée
        
        Returns:
            Liste de bits extraits
        """
        try:
            img = image.copy()
            
            if len(img.shape) == 3:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                y_channel = img_yuv[:, :, 0]
            else:
                y_channel = img.copy()
            
            h, w = y_channel.shape
            h_pad = (h // 8) * 8
            w_pad = (w // 8) * 8
            
            bits = []
            alpha = 20
            
            for i in range(0, h_pad, 8):
                for j in range(0, w_pad, 8):
                    block = y_channel[i:i+8, j:j+8].astype(np.float64)
                    dct_block = self._dct_2d(block)
                    
                    row, col = 4, 4
                    
                    dct_value = dct_block[row, col]
                    
                    if abs(dct_value) > alpha / 2:
                        bits.append(1 if dct_value > 0 else 0)
                    else:
                        bits.append(0)
            
            return bits
            
        except Exception as e:
            raise WatermarkError(f"Erreur extraction DCT: {e}")
    
    # ==================== INTERFACE PRINCIPALE ====================
    
    def embed(self, image, camera_id, person_name, status, timestamp=None):
        """
        Insère un watermark dans l'image selon la méthode configurée.
        
        Args:
            image: Image NumPy
            camera_id: ID de la caméra
            person_name: Nom de la personne
            status: Statut de la personne
            timestamp: Horodatage (optionnel)
        
        Returns:
            Image tatoueée
        """
        payload = self.create_payload(camera_id, person_name, status, timestamp)
        
        if self.method == "LSB":
            return self.embed_lsb(image, payload.decode('utf-8'))
        else:
            header = self.watermark_marker + b':' + str(len(payload)).encode() + b':'
            full_data = header + payload
            crc = zlib.crc32(full_data).to_bytes(4, 'big')
            full_data_with_crc = full_data + crc
            bits = self._bytes_to_bits(full_data_with_crc)
            return self.embed_dct(image, bits)
    
    def extract(self, image):
        """
        Extrait le watermark d'une image.
        
        Args:
            image: Image tatoueée
        
        Returns:
            Dict avec les informations extraites
        """
        if self.method == "LSB":
            raw_data = self.extract_lsb(image)
            return self.parse_payload(raw_data.encode('utf-8'))
        else:
            bits = self.extract_dct(image)
            full_bits = bits
            marker_len = len(self.watermark_marker) * 8
            
            marker_found = False
            data_start = 0
            for i in range(len(full_bits) - marker_len):
                check_bits = full_bits[i:i + marker_len]
                check_bytes = self._bits_to_bytes(check_bits)
                if check_bytes == self.watermark_marker:
                    marker_found = True
                    data_start = i + marker_len
                    break
            
            if not marker_found:
                raise WatermarkError("Marqueur non trouvé")
            
            header_bits = []
            colon_found = False
            i = data_start
            
            while i < len(full_bits) and not colon_found:
                byte_bits = full_bits[i:i+8]
                i += 8
                if len(byte_bits) < 8:
                    break
                bval = self._bits_to_bytes(byte_bits)
                if bval == b':':
                    colon_found = True
                    break
                header_bits.extend(byte_bits)
            
            if not colon_found or not header_bits:
                raise WatermarkError("Header invalide")
            
            try:
                msg_len_str = self._bits_to_bytes(header_bits).decode()
                msg_len = int(msg_len_str)
            except Exception as e:
                raise WatermarkError(f"Header invalide: {e}")
            
            total_msg_bits = msg_len * 8
            msg_bits = full_bits[i:i + total_msg_bits]
            
            if len(msg_bits) < total_msg_bits:
                raise WatermarkError("Message incomplet")
            
            msg_bytes = self._bits_to_bytes(msg_bits)
            
            i += total_msg_bits
            crc_bits = full_bits[i:i + 32]
            
            if len(crc_bits) < 32:
                raise WatermarkError("CRC incomplet")
            
            crc_bytes = self._bits_to_bytes(crc_bits)
            
            expected_crc = zlib.crc32(self.watermark_marker + b':' + str(msg_len).encode() + b':' + msg_bytes).to_bytes(4, 'big')
            
            if crc_bytes != expected_crc:
                raise WatermarkError("CRC invalide - corruption détectée")
            
            return self.parse_payload(msg_bytes)
    
    def verify_watermark(self, image_path):
        """
        Vérifie l'intégrité du watermark dans une image.
        
        Args:
            image_path: Chemin vers l'image à vérifier
        
        Returns:
            Dict avec le résultat de la vérification
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise WatermarkError("Image non trouvée")
            
            result = {
                'success': False,
                'method': self.method,
                'payload': None,
                'error': None,
                'ber': None
            }
            
            try:
                result['payload'] = self.extract(img)
                result['success'] = True
            except WatermarkError as e:
                result['error'] = str(e)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'payload': None,
                'ber': None
            }
    
    def test_robustness(self, image, jpeg_quality=75):
        """
        Teste la robustesse du watermark à la compression JPEG.
        
        Args:
            image: Image tatoueée
            jpeg_quality: Qualité JPEG (0-100)
        
        Returns:
            Dict avec le taux d'erreur (BER)
        """
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            if compressed is None:
                return {'ber': 1.0, 'error': 'Impossible de décompresser'}
            
            bits_original = []
            try:
                if self.method == "DCT":
                    bits_original = self.extract_dct(image)
                else:
                    msg_original = self.extract_lsb(image)
                    bits_original = self._bytes_to_bits(msg_original.encode('utf-8'))
            except WatermarkError as e:
                return {'ber': 1.0, 'error': f"Extraction originale échouée: {str(e)}"}
            
            bits_compressed = []
            try:
                if self.method == "DCT":
                    bits_compressed = self.extract_dct(compressed)
                else:
                    msg_compressed = self.extract_lsb(compressed)
                    bits_compressed = self._bytes_to_bits(msg_compressed.encode('utf-8'))
            except WatermarkError:
                bits_compressed = []
            
            min_len = min(len(bits_original), len(bits_compressed))
            if min_len == 0:
                return {'ber': 1.0, 'error': 'Extraction compressée échouée'}
            
            errors = sum(1 for i in range(min_len) if bits_original[i] != bits_compressed[i])
            ber = errors / min_len
            
            return {
                'ber': ber,
                'success_rate': 1 - ber,
                'bits_compared': min_len,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Erreur test robustesse: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'ber': 1.0, 'error': str(e)}