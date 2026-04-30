"""Fonctions utilitaires pour le traitement d'images."""

import cv2
import numpy as np
from PIL import Image
import os
from utils.logger import get_logger

logger = get_logger("ImageUtils")


def resize_with_aspect_ratio(image, max_width=640, max_height=480):
    """
    Redimensionne une image en préservant le ratio.
    
    Args:
        image: Image NumPy
        max_width: Largeur maximale
        max_height: Hauteur maximale
    
    Returns:
        Image redimensionnée
    """
    h, w = image.shape[:2]
    
    width_ratio = max_width / w
    height_ratio = max_height / h
    ratio = min(width_ratio, height_ratio, 1.0)
    
    new_width = int(w * ratio)
    new_height = int(h * ratio)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def convert_to_grayscale(image):
    """Convertit une image en niveaux de gris."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image, clip_limit=2.0, tile_size=(8, 8)):
    """Améliore le contraste via CLAHE."""
    gray = convert_to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applique un flou gaussien."""
    return cv2.GaussianBlur(image, kernel_size, 0)


def blur_region(image, bbox, kernel_size=(31, 31)):
    """
    Applique un flou sur une région de l'image (pour RGPD).
    
    Args:
        image: Image NumPy
        bbox: Bounding box (x, y, w, h)
        kernel_size: Taille du noyau de flou
    
    Returns:
        Image avec région floutée
    """
    x, y, w, h = bbox
    result = image.copy()
    
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w > 0 and h > 0:
        roi = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, kernel_size, 0)
        result[y:y+h, x:x+w] = blurred
    
    return result


def save_image(image, path, quality=95):
    """Sauvegarde une image sur disque."""
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            with open(path, 'wb') as f:
                f.write(encoded.tobytes())
        else:
            cv2.imwrite(path, image)
        
        return True
    except Exception as e:
        logger.error(f"Erreur sauvegarde image: {e}")
        return False


def load_image(path):
    """Charge une image depuis disque."""
    try:
        return cv2.imread(path)
    except Exception as e:
        logger.error(f"Erreur chargement image: {e}")
        return None


def create_thumbnail(image, size=(100, 100)):
    """Crée une miniature de l'image."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def numpy_to_pil(image):
    """Convertit NumPy array en PIL Image."""
    if image is None:
        return None
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_numpy(image):
    """Convertit PIL Image en NumPy array."""
    if image is None:
        return None
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def calculate_histogram(image, channels=1):
    """Calcule l'histogramme de l'image."""
    if channels == 1:
        gray = convert_to_grayscale(image)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist.flatten()
    else:
        histograms = []
        for i in range(channels):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms.append(hist.flatten())
        return histograms


def apply_blur_for_gdpr(image, faces, unknown_faces_idx):
    """
    Applique un flou sur les visages inconnus selon le mode RGPD.
    
    Args:
        image: Image NumPy
        faces: Liste de bounding boxes
        unknown_faces_idx: Indices des visages inconnus
    
    Returns:
        Image avec visages inconnus floutés
    """
    result = image.copy()
    
    for idx in unknown_faces_idx:
        if idx < len(faces):
            result = blur_region(result, faces[idx])
    
    return result


def overlay_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                 font_scale=1, color=(255, 255, 255), thickness=2):
    """Superpose du texte sur l'image."""
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2, label=None):
    """Dessine une bounding box sur l'image."""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    if label:
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, thickness)
    
    return image


def compress_image(image, quality=85):
    """Compresse l'image au format JPEG."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def calculate_similarity(img1, img2):
    """Calcule la similarité entre deux images (PSNR)."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    gray1 = convert_to_grayscale(img1)
    gray2 = convert_to_grayscale(img2)
    
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
    
    if mse == 0:
        return 100.0
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def create_grid(images, rows, cols, cell_size=(100, 100)):
    """Crée une grille d'images."""
    cell_h, cell_w = cell_size
    
    grid_h = rows * cell_h
    grid_w = cols * cell_w
    
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        if img is None:
            continue
        
        row = idx // cols
        col = idx % cols
        
        if row >= rows:
            break
        
        thumbnail = create_thumbnail(img, cell_size)
        
        y = row * cell_h
        x = col * cell_w
        grid[y:y+cell_h, x:x+cell_w] = thumbnail
    
    return grid