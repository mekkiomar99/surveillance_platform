"""Point d'entrée principal de la Plateforme de Surveillance Vidéo Intelligente."""

import sys
import os

# Désactiver les messages TensorFlow (sans bloquer les imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from database.db_manager import DatabaseManager
from gui.main_window import SurveillanceWindow
from utils.logger import setup_logger


def main():
    logger = setup_logger("Main")
    logger.info("=" * 60)
    logger.info("PLATEFORME DE SURVEILLANCE VIDÉO INTELLIGENTE")
    logger.info("=" * 60)
    
    logger.info(f"Configuration:")
    logger.info(f"  - Méthode tatouage: {config.WATERMARK_METHOD}")
    logger.info(f"  - Seuil reconnaissance: {config.RECOGNITION_THRESHOLD}")
    logger.info(f"  - FPS cible: {config.TARGET_FPS}")
    
    try:
        db_manager = DatabaseManager(config.DB_PATH)
        logger.info("Base de données initialisée")
        
        logger.info("Lancement de l'interface...")
        
        app = SurveillanceWindow(db_manager)
        app.run()
        
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Application fermée")


if __name__ == "__main__":
    main()