"""Module de gestion des alertes et notifications."""

import os
import cv2
from datetime import datetime
from utils.logger import get_logger
import config

logger = get_logger("AlertManager")

try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False
    logger.warning("pygame non disponible - sons désactivés")


class AlertLevel:
    """Niveaux d'alerte."""
    LEVEL_1_AUTHORIZED = 1
    LEVEL_2_UNAUTHORIZED = 2
    LEVEL_3_UNKNOWN = 3


class AlertManager:
    """Gestionnaire d'alertes pour la plateforme de surveillance."""
    
    def __init__(self, db_manager, captures_dir=None):
        """
        Initialise le gestionnaire d'alertes.
        
        Args:
            db_manager: Gestionnaire de base de données
            captures_dir: Répertoire de sauvegarde des captures
        """
        self.db_manager = db_manager
        self.captures_dir = captures_dir or config.CAPTURES_DIR
        os.makedirs(self.captures_dir, exist_ok=True)
        
        self.alert_callbacks = []
        self.sound_enabled = config.ENABLE_SOUND_ALERTS
        
        if self.sound_enabled and pygame_available:
            try:
                pygame.mixer.init()
                self.sound_level_2 = self._load_sound(config.ALERT_SOUND_LEVEL_2)
                self.sound_level_3 = self._load_sound(config.ALERT_SOUND_LEVEL_3)
            except Exception as e:
                logger.warning(f"Erreur initialisation sons: {e}")
                self.sound_enabled = False
    
    def _load_sound(self, path):
        """Charge un fichier audio."""
        if os.path.exists(path):
            try:
                return pygame.mixer.Sound(path)
            except Exception as e:
                logger.error(f"Erreur chargement son {path}: {e}")
        return None
    
    def register_alert_callback(self, callback):
        """Enregistre une fonction de callback pour les alertes."""
        if callable(callback):
            self.alert_callbacks.append(callback)
    
    def process_alert(self, person_name, status, confidence, camera_id, frame, watermarked_frame):
        """
        Traite une alerte selon le niveau de menace.
        
        Args:
            person_name: Nom de la personne
            status: Statut (authorized, unauthorized, unknown)
            confidence: Confiance de reconnaissance
            camera_id: ID de la caméra
            frame: Frame original
            watermarked_frame: Frame tatoué
        
        Returns:
            Dict avec les informations de l'alerte
        """
        alert_info = {
            'level': self._get_alert_level(status),
            'person_name': person_name,
            'status': status,
            'confidence': confidence,
            'camera_id': camera_id,
            'timestamp': datetime.now().isoformat(),
            'capture_path': None,
            'alert_sent': False
        }
        
        if alert_info['level'] == AlertLevel.LEVEL_1_AUTHORIZED:
            alert_info['description'] = "Accès autorisé enregistré"
        elif alert_info['level'] == AlertLevel.LEVEL_2_UNAUTHORIZED:
            alert_info['description'] = "ALERTE: Personne non autorisée détectée"
        else:
            alert_info['description'] = "ALARME: Personne inconnue détectée"
        
        capture_path = self._save_capture(person_name, status, watermarked_frame, camera_id)
        if capture_path:
            alert_info['capture_path'] = capture_path
        
        alert_info['alert_sent'] = self._send_alert(alert_info)
        
        self._log_to_database(alert_info)
        
        self._notify_callbacks(alert_info)
        
        return alert_info
    
    def _get_alert_level(self, status):
        """Détermine le niveau d'alerte selon le statut."""
        if status == 'authorized':
            return AlertLevel.LEVEL_1_AUTHORIZED
        elif status == 'unauthorized':
            return AlertLevel.LEVEL_2_UNAUTHORIZED
        else:
            return AlertLevel.LEVEL_3_UNKNOWN
    
    def _save_capture(self, person_name, status, watermarked_frame, camera_id):
        """Sauvegarde une capture tatouée."""
        try:
            if watermarked_frame is None:
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            person_safe = person_name.replace(' ', '_') if person_name else 'unknown'
            status_safe = status.replace(' ', '_')
            
            filename = f"{timestamp}_{camera_id}_{person_safe}_{status_safe}.jpg"
            filepath = os.path.join(self.captures_dir, filename)
            
            quality = config.JPEG_QUALITY
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', watermarked_frame, encode_param)
            
            with open(filepath, 'wb') as f:
                f.write(encoded.tobytes())
            
            logger.debug(f"Capture sauvegardée: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde capture: {e}")
            return None
    
    def _send_alert(self, alert_info):
        """Envoie l'alerte (sons, notifications)."""
        level = alert_info['level']
        
        if level == AlertLevel.LEVEL_2_UNAUTHORIZED:
            if self.sound_enabled and hasattr(self, 'sound_level_2') and self.sound_level_2:
                try:
                    self.sound_level_2.play()
                except Exception as e:
                    logger.error(f"Erreur lecture son: {e}")
            return True
        
        elif level == AlertLevel.LEVEL_3_UNKNOWN:
            if self.sound_enabled and hasattr(self, 'sound_level_3') and self.sound_level_3:
                try:
                    self.sound_level_3.play()
                except Exception as e:
                    logger.error(f"Erreur lecture son: {e}")
            return True
        
        return False
    
    def _log_to_database(self, alert_info):
        """Enregistre l'alerte dans la base de données."""
        try:
            person_id = None
            if alert_info['person_name']:
                person = self.db_manager.get_person_by_name(alert_info['person_name'])
                if person:
                    person_id = person['id']
            
            self.db_manager.log_access(
                person_id=person_id,
                person_name=alert_info['person_name'],
                status=alert_info['status'],
                confidence=alert_info['confidence'],
                camera_id=alert_info['camera_id'],
                capture_path=alert_info.get('capture_path'),
                alert_sent=alert_info['alert_sent']
            )
            
        except Exception as e:
            logger.error(f"Erreur logging base de données: {e}")
    
    def _notify_callbacks(self, alert_info):
        """Notifie les callbacks enregistrés."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                logger.error(f"Erreur callback: {e}")
    
    def send_email_notification(self, alert_info, smtp_config=None):
        """
        Structure pour envoi d'email (non implémenté - à brancher).
        
        Args:
            alert_info: Informations de l'alerte
            smtp_config: Configuration SMTP (host, port, user, password, to)
        """
        if smtp_config is None:
            return False
        
        logger.info(f"Email notification préparé: {alert_info['person_name']} - {alert_info['status']}")
        return True
    
    def get_recent_alerts(self, limit=10):
        """Récupère les alertes récentes."""
        return self.db_manager.get_recent_logs(limit)
    
    def get_alert_stats(self):
        """Récupère les statistiques d'alertes."""
        return self.db_manager.get_access_stats()
    
    def clear_old_captures(self, days=30):
        """Supprime les captures anciennes."""
        try:
            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            deleted = 0
            
            for filename in os.listdir(self.captures_dir):
                filepath = os.path.join(self.captures_dir, filename)
                if os.path.isfile(filepath):
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        deleted += 1
            
            logger.info(f"Captures supprimées: {deleted}")
            return deleted
            
        except Exception as e:
            logger.error(f"Erreur清理 captures: {e}")
            return 0


class AlertDialog:
    """Dialog pour afficher les alertes (utilisé par la GUI)."""
    
    COLORS = {
        'authorized': '#00FF00',
        'unauthorized': '#FFA500',
        'unknown': '#FF0000'
    }
    
    @staticmethod
    def get_color(status):
        """Retourne la couleur associée au statut."""
        return AlertDialog.COLORS.get(status, '#FFFFFF')
    
    @staticmethod
    def get_message(status, person_name, confidence):
        """Génère le message d'alerte."""
        messages = {
            'authorized': f"Accès autorisé\n{person_name}\nConfiance: {confidence:.1%}",
            'unauthorized': f"Attention!\n{person_name} non autorisé\nConfiance: {confidence:.1%}",
            'unknown': f"ALARME!\nPersonne inconnue\nConfiance: {confidence:.1%}"
        }
        return messages.get(status, "Statut inconnu")