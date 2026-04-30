"""Journalisation structurée pour la plateforme de surveillance."""

import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configure et retourne un logger structuré.
    
    Args:
        name: Nom du logger
        log_file: Chemin du fichier de log (optionnel)
        level: Niveau de logging
    
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter les handlers dupliqués
    if logger.handlers:
        return logger
    
    # Format avec horodatage
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier si spécifié
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """Récupère ou crée un logger."""
    return logging.getLogger(name) if logging.getLogger(name).handlers else setup_logger(name)


class AccessLogger:
    """Logger spécialisé pour les accès."""
    
    def __init__(self, log_file=None):
        self.logger = setup_logger("AccessLogger", log_file)
    
    def log_access(self, person_name, status, camera_id, confidence):
        """Log un accès détecté."""
        self.logger.info(
            f"ACCÈS: {person_name} | Statut: {status} | "
            f"Caméra: {camera_id} | Confiance: {confidence:.2f}"
        )
    
    def log_alert(self, level, message):
        """Log une alerte."""
        levels = {1: "INFO", 2: "WARNING", 3: "CRITICAL"}
        getattr(self.logger, levels.get(level, "INFO").lower())(f"ALERTE NIVEAU {level}: {message}")
    
    def log_error(self, component, error):
        """Log une erreur."""
        self.logger.error(f"ERREUR [{component}]: {error}")