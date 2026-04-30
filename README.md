# Plateforme de Surveillance Vidéo Intelligente avec Tatouage des Captures

## Description

Plateforme de surveillance complète pour la détection et reconnaissance faciale en temps réel, avec tatouage numérique des captures pour garantir l'authenticité des preuves.

## Fonctionnalités

- **Détection de visages en temps réel** (Haar Cascade + MTCNN fallback)
- **Reconnaissance faciale** (LBP + SVM)
- **Tatouage numérique** (LSB invisible + DCT robuste JPEG)
- **Gestion des alertes** selon le niveau de menace
- **Base de données SQLite** pour les logs et personnes
- **Interface GUI Tkinter** complète
- **Export CSV** des logs d'accès

## Structure du Projet

```
surveillance_platform/
├── main.py                    # Point d'entrée
├── config.py                  # Paramètres globaux
├── requirements.txt          # Dépendances
├── core/
│   ├── detector.py            # Détection visages
│   ├── recognizer.py          # Reconnaissance faciale
│   ├── watermark.py           # Tatouage LSB/DCT
│   └── alert_manager.py       # Gestion alertes
├── database/
│   ├── db_manager.py          # Interface SQLite
│   └── schema.sql             # Schéma BDD
├── gui/
│   ├── main_window.py         # Fenêtre principale
│   ├── enrollment_dialog.py   # Enregistrement personne
│   └── verification_dialog.py # Vérification tatouage
├── utils/
│   ├── logger.py              # Journalisation
│   └── image_utils.py         # Utilitaires image
├── data/                      # Données (BDD, captures)
└── tests/                    # Tests unitaires
```

## Installation

```bash
pip install -r requirements.txt
```

### Dépendances principales

- opencv-python >= 4.8.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scikit-image >= 0.21.0
- scipy >= 1.11.0
- Pillow >= 10.0.0
- mtcnn >= 0.1.1
- tensorflow >= 2.13.0
- pygame >= 2.5.0

## Lancement

```bash
python main.py
```

## Utilisation

### 1. Enregistrer une personne

1. Menu → Base de données → Enregistrer une personne
2. Entrez le nom et cochez "Autorisé" si nécessaire
3. Cliquez sur "Démarrer Webcam"
4. Cliquez sur "Capturer" 10 fois pour enregistrer le visage
5. Cliquez sur "Enregistrer & Entraîner"

### 2. Surveillance

1. Cliquez sur "Démarrer Webcam" ou "Ouvrir Vidéo"
2. Les visages sont automatiquement détectés et reconnus
3. Les captures sont tatouées et sauvegardées

### 3. Vérifier un tatouage

1. Menu → Outils → Vérifier Tatouage
2. Ouvrez une image capturée
3. Visualisez les informations extraites

## Tests

```bash
python -m pytest tests/ -v
```

## Paramètres

Modifiez `config.py` pour personnaliser :

- `TARGET_FPS`: FPS cible (défaut: 15)
- `RECOGNITION_THRESHOLD`: Seuil de reconnaissance (défaut: 0.6)
- `WATERMARK_METHOD`: "LSB" ou "DCT"
- `GDPR_MODE`: Si True, floute les visages inconnus

## Méthodes de Tatouage

### LSB (Least Significant Bit)

- Insertion invisible dans les bits de poids faible
- Capacité: ~25% de la taille de l'image
- Non robuste à la compression JPEG

### DCT (Discrete Cosine Transform)

- Insertion robuste dans les coefficients DCT
- Résiste à la compression JPEG (Q≥75)
- Taux d'erreur (BER) cible < 10%

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    GUI Tkinter                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐          │
│  │Detector │→ │Recognizer│→ │Watermark │→ Alert   │
│  └─────────┘  └──────────┘  └──────────┘          │
│      ↓            ↓            ↓                    │
│  ┌──────────────────────────────────���──┐          │
│  │          Database Manager           │          │
│  │          (SQLite)                   │          │
│  └─────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

## Conformité RGPD

Activez le mode RGPD dans `config.py` (`GDPR_MODE = True`) pour automatiquement flouter les visages non reconnus.

## License

Projet académique - Hamdi Chebbi - ING-4-SSIRF