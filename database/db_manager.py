"""Gestionnaire de base de données SQLite pour la plateforme de surveillance."""

import sqlite3
import pickle
import os
import csv
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("DatabaseManager")


class DatabaseManager:
    """Gestionnaire de la base de données SQLite."""
    
    def __init__(self, db_path):
        """
        Initialise le gestionnaire de base de données.
        
        Args:
            db_path: Chemin vers le fichier SQLite
        """
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self):
        """Retourne une connexion à la base de données."""
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialise la base de données avec le schéma si nécessaire."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='persons';")
            exists = cursor.fetchone()
            if not exists:
                logger.info("Initialisation du schéma de la base de données...")
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = f.read()
                cursor.executescript(schema)
                conn.commit()
                logger.info("Base de données initialisée avec succès.")
            conn.close()
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    def add_person(self, name, authorized=False, face_encoding=None):
        """Ajoute ou met à jour une personne dans la base de données."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, image_count FROM persons WHERE name = ?", (name,))
            existing = cursor.fetchone()
            
            if existing:
                person_id = existing[0]
                encoding_blob = pickle.dumps(face_encoding) if face_encoding is not None else None
                cursor.execute("""
                    UPDATE persons 
                    SET authorized = ?, enrolled_at = CURRENT_TIMESTAMP, 
                        face_encoding = ?, image_count = image_count + 1
                    WHERE id = ?
                """, (authorized, encoding_blob, person_id))
            else:
                encoding_blob = pickle.dumps(face_encoding) if face_encoding is not None else None
                cursor.execute("""
                    INSERT INTO persons (name, authorized, face_encoding)
                    VALUES (?, ?, ?)
                """, (name, authorized, encoding_blob))
                person_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.info(f"Personne ajoutée/mise à jour: {name} (ID: {person_id})")
            return person_id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la personne: {e}")
            raise
    
    def get_person_by_name(self, name):
        """Récupère une personne par son nom."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, authorized, enrolled_at, face_encoding
                FROM persons WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'authorized': bool(row[2]),
                    'enrolled_at': row[3],
                    'face_encoding': pickle.loads(row[4]) if row[4] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération: {e}")
            return None
    
    def get_all_persons(self):
        """Récupère toutes les personnes enregistrées."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, authorized, enrolled_at
                FROM persons ORDER BY enrolled_at DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'authorized': bool(row[2]),
                    'enrolled_at': row[3]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération: {e}")
            return []
    
    def get_person_encoding(self, person_id):
        """Récupère l'encodage facial d'une personne."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT face_encoding FROM persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            conn.close()
            
            return pickle.loads(row[0]) if row and row[0] else None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'encodage: {e}")
            return None
    
    def log_access(self, person_id, person_name, status, confidence, camera_id, capture_path=None, alert_sent=False):
        """
        Enregistre un accès dans la base.
        
        Args:
            person_id: ID de la personne (ou None si inconnue)
            person_name: Nom de la personne
            status: Statut (authorized, unauthorized, unknown)
            confidence: Confiance de la reconnaissance
            camera_id: ID de la caméra
            capture_path: Chemin de la capture (optionnel)
            alert_sent: True si une alerte a été envoyée
        
        Returns:
            ID du log créé
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO access_logs 
                (person_id, person_name, status, confidence, camera_id, capture_path, alert_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (person_id, person_name, status, confidence, camera_id, capture_path, alert_sent))
            
            log_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.debug(f"Accès enregistré: {person_name} - {status}")
            return log_id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'accès: {e}")
            return None
    
    def get_recent_logs(self, limit=10):
        """Récupère les derniers logs d'accès."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, person_name, status, confidence, camera_id, timestamp, capture_path, alert_sent
                FROM access_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': row[0],
                    'person_name': row[1] if row[1] else 'Inconnu',
                    'status': row[2],
                    'confidence': row[3],
                    'camera_id': row[4],
                    'timestamp': row[5],
                    'capture_path': row[6],
                    'alert_sent': bool(row[7])
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des logs: {e}")
            return []
    
    def export_logs_csv(self, output_path, start_date=None, end_date=None):
        """
        Exporte les logs en CSV.
        
        Args:
            output_path: Chemin du fichier CSV de sortie
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT id, person_name, status, confidence, camera_id, timestamp, capture_path, alert_sent
                FROM access_logs WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Nom', 'Statut', 'Confiance', 'Caméra', 'Horodatage', 'Capture', 'Alerte'])
                writer.writerows(rows)
            
            logger.info(f"Logs exportés: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export: {e}")
            return False
    
    def update_log_capture(self, log_id, capture_path):
        """Met à jour le chemin de capture pour un log."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE access_logs SET capture_path = ? WHERE id = ?
            """, (capture_path, log_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Erreur mise à jour capture: {e}")
    
    def get_access_stats(self):
        """Récupère les statistiques d'accès."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM access_logs 
                GROUP BY status
            """)
            by_status = dict(cursor.fetchall())
            
            cursor.execute("SELECT COUNT(*) FROM access_logs WHERE DATE(timestamp) = DATE('now')")
            today_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_logs")
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'by_status': by_status,
                'today_count': today_count,
                'total_count': total_count
            }
            
        except Exception as e:
            logger.error(f"Erreur statistiques: {e}")
            return {'by_status': {}, 'today_count': 0, 'total_count': 0}
    
    def delete_person(self, name):
        """Supprime une personne de la base."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM persons WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            logger.info(f"Personne supprimée: {name}")
            return True
        except Exception as e:
            logger.error(f"Erreur suppression: {e}")
            return False
    
    def delete_person_by_name(self, name):
        """Supprime une personne de la base de données par son nom."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM persons WHERE name = ?", (name,))
            conn.commit()
            deleted = cursor.rowcount
            conn.close()
            if deleted:
                logger.info(f"Personne supprimée: {name}")
            else:
                logger.warning(f"Aucune personne trouvée à supprimer: {name}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la personne: {e}")
            return False
    
    def camera_exists(self, camera_id):
        """Vérifie si une caméra existe."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM cameras WHERE id = ?", (camera_id,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            logger.error(f"Erreur vérification caméra: {e}")
            return False
    
    def add_camera(self, camera_id, location):
        """Ajoute une caméra."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO cameras (id, location) VALUES (?, ?)
            """, (camera_id, location))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Erreur ajout caméra: {e}")
            return False
