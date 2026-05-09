# Paramètres globaux de la plateforme de surveillance
import os

TARGET_FPS = 15
CAMERA_ID = "CAM_001"
CAMERA_SOURCE = 0  # 0 = webcam, ou chemin fichier vidéo

# Reconnaissance faciale
RECOGNITION_THRESHOLD = 0.35
FACE_SIZE = (160, 160) # Augmenté pour capturer plus de détails HOG

# Tatouage numérique
WATERMARK_METHOD = "DCT"  # "LSB" ou "DCT"
JPEG_QUALITY = 100

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "surveillance.db")
CAPTURES_DIR = os.path.join(BASE_DIR, "data", "captures")
FACES_DB_DIR = os.path.join(BASE_DIR, "data", "faces_db")
HAARCASCADES_DIR = os.path.join(BASE_DIR, "data", "haarcascades")
MODEL_PATH = os.path.join(FACES_DB_DIR, "face_model.pkl")

# RGPD - Si True, floute les visages non reconnus
GDPR_MODE = False

# Optimisation performances
FRAME_SKIP = 2  # Traiter 1 frame sur 2
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480

# Alertes
ENABLE_SOUND_ALERTS = True
ALERT_SOUND_LEVEL_2 = "data/alert.wav"
ALERT_SOUND_LEVEL_3 = "data/alarm.wav"