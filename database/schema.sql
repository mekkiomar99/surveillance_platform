-- Schéma de la base de données de surveillance
-- Plateforme de Surveillance Vidéo Intelligente avec Tatouage

-- Table des personnes enregistrées
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    authorized BOOLEAN DEFAULT FALSE,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_encoding BLOB,
    image_count INTEGER DEFAULT 0
);

-- Table des logs d'accès
CREATE TABLE IF NOT EXISTS access_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    person_name TEXT,
    status TEXT CHECK(status IN ('authorized', 'unauthorized', 'unknown')),
    confidence REAL DEFAULT 0.0,
    camera_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    capture_path TEXT,
    alert_sent BOOLEAN DEFAULT FALSE,
    watermark_verified BOOLEAN DEFAULT FALSE
);

-- Table des caméras
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    location TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des captures tatouéés
CREATE TABLE IF NOT EXISTS captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id INTEGER REFERENCES access_logs(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    watermark_method TEXT,
    timestamp_capture TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_id TEXT,
    hash_sha256 TEXT
);

-- Index pour optimiser les requêtes
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_status ON access_logs(status);
CREATE INDEX IF NOT EXISTS idx_logs_camera ON access_logs(camera_id);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

-- Vue pour les statistiques d'accès
CREATE VIEW IF NOT EXISTS access_stats AS
SELECT 
    status,
    COUNT(*) as count,
    DATE(timestamp) as date
FROM access_logs
GROUP BY status, DATE(timestamp);

-- Insertion caméra par défaut
INSERT OR IGNORE INTO cameras (id, location, active) VALUES ('CAM_001', 'Entrée Principale', TRUE);