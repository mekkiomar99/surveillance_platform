# Plateforme de Surveillance Vidéo Intelligente avec Tatouage des Captures

## Description

Plateforme de surveillance avancée combinant intelligence artificielle pour la reconnaissance faciale et cryptographie pour la protection de l'intégrité des preuves numériques. Le système permet de surveiller un flux vidéo, d'identifier les individus et de générer des captures d'écran certifiées par un tatouage numérique (watermarking) invisible.

## Fonctionnalités Clés

- **Détection de visages Multi-moteur** : Haar Cascade haute performance avec fallback MTCNN pour une précision maximale.
- **Reconnaissance Faciale Hybride (HOG + LBP + KNN)** :
    - Utilise les **HOG (Histogram of Oriented Gradients)** pour la structure géométrique.
    - Utilise les **LBP (Local Binary Patterns)** pour la texture de la peau.
    - **Alignement Facial Automatique** : Redresse le visage via la position des yeux pour une reconnaissance stable même si la tête est inclinée.
    - **Filtrage Temporel** : Vote majoritaire sur les 5 dernières frames pour éliminer les confusions fugaces.
- **Tatouage Numérique de Preuve (Watermarking)** :
    - **DCT (Discrete Cosine Transform)** : Robuste à la compression JPEG (Q≥75), idéal pour les fichiers légers.
    - **LSB (Least Significant Bit)** : Invisible et haute capacité, utilisant automatiquement le format PNG sans perte.
    - **Vérification Intégrée** : Outil d'extraction des métadonnées (Horodatage, Caméra, ID) pour prouver l'authenticité d'une image.
- **Gestion Intelligente des Alertes** : Différenciation entre accès autorisés (Vert) et non autorisés (Rouge).
- **Conformité RGPD** : Option de floutage automatique des visages inconnus.

## Structure du Projet

```
surveillance_platform/
├── main.py                    # Point d'entrée
├── config.py                  # Paramètres globaux (Seuils, méthodes, chemins)
├── requirements.txt          # Dépendances Python
├── core/
│   ├── detector.py            # Moteurs de détection visages
│   ├── recognizer.py          # Intelligence : HOG+LBP, KNN, Alignement
│   ├── watermark.py           # Sécurité : Algorithmes DCT et LSB (fenêtre glissante)
│   └── alert_manager.py       # Logique de sauvegarde et alertes
├── database/
│   ├── db_manager.py          # Interface SQLite
│   └── schema.sql             # Définition des tables (persons, logs)
├── gui/
│   ├── main_window.py         # Interface de surveillance principale
│   ├── enrollment_dialog.py   # Gestion de l'enregistrement et suppression
│   └── verification_dialog.py # Analyseur d'intégrité des captures
├── data/                      # Stockage (DB, modèles .pkl, captures)
└── utils/                     # Journalisation et utilitaires image
```

## Installation

```bash
pip install -r requirements.txt
```

### Dépendances principales
- `opencv-python` : Traitement d'image et vidéo.
- `scikit-image` : Extraction de caractéristiques HOG/LBP.
- `scikit-learn` : Classification KNN et SVM.
- `numpy` & `scipy` : Calculs matriciels et transformées DCT.

## Utilisation

### 1. Enregistrement Optimal
1. Menu → Base de données → Enregistrer une personne.
2. Capturez 10 photos en restant bien face à la caméra pour l'alignement initial.
3. Le système entraîne automatiquement un modèle KNN dès que 2 personnes sont présentes.

### 2. Surveillance et Preuve
1. Cliquez sur **Webcam**. Le système affiche le nom et le statut en temps réel.
2. Cliquez sur **Capturer** pour générer une preuve tatouée.
3. Le format de fichier (.jpg ou .png) est géré automatiquement selon la méthode de tatouage choisie.

### 3. Vérification de Preuve
1. Menu → Outils → Vérifier Tatouage.
2. Sélectionnez une image dans `data/captures`.
3. Le système extrait les données cachées et valide le CRC pour prouver qu'aucune modification n'a été faite.

## Paramètres (config.py)

- `RECOGNITION_THRESHOLD` : Ajusté à **0.35** pour l'algorithme hybride.
- `WATERMARK_METHOD` : "DCT" (robuste) ou "LSB" (invisible).
- `FACE_SIZE` : **160x160** pour une analyse HOG précise.

## Sécurité et Authentification

Le système garantit l'intégrité via :
- **CRC32** : Vérifie que le message caché n'a pas été corrompu.
- **Marqueur de Synchronisation** : Recherche par fenêtre glissante pour retrouver le tatouage même après un léger recadrage.
- **Noyau Linéaire/KNN** : Séparation stricte des profils pour éviter l'usurpation d'identité.

---
*Développé pour la surveillance intelligente et la certification de preuves numériques.*
