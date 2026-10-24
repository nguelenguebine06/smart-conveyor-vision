
# 🏭 Smart Conveyor Vision

**Système intelligent de détection des trous sur un tapis roulant industriel par vision par ordinateur et intelligence artificielle.**

Smart Conveyor Vision est un prototype de vision industrielle qui combine la simulation 3D avec Blender, la détection d'objets avec YOLO26n et le traitement vidéo avec OpenCV. L'application permet d'analyser des vidéos d'un convoyeur, de détecter les trous présents sur la bande transporteuse et de visualiser les résultats dans une interface interactive développée avec Streamlit.

## 🎯 Objectifs du projet

- Modéliser et simuler un convoyeur industriel en 3D avec Blender.
- Générer des données synthétiques pour entraîner un modèle de détection.
- Entraîner et évaluer un modèle YOLO26n pour détecter les trous.
- Analyser des vidéos et visualiser les détections.
- Présenter les résultats et les métriques du modèle dans une interface web.

## ✨ Fonctionnalités

- **Simulation 3D :** création d'un convoyeur et de scénarios comportant des anomalies avec Blender.
- **Détection par IA :** identification des trous grâce au modèle YOLO26n entraîné.
- **Analyse vidéo :** traitement des vidéos avec OpenCV et affichage des cadres de détection.
- **Interface interactive :** navigation entre les pages Dashboard, Modèle IA, Analyse vidéo et Entraînement du modèle.
- **Visualisation des performances :** consultation des courbes d'apprentissage, des métriques et de la matrice de confusion.
- **Export des résultats :** téléchargement des vidéos annotées après analyse.

## 🛠️ Technologies utilisées

| Technologie | Rôle |
|---|---|
| Python | Développement du système |
| Blender | Modélisation et simulation 3D du convoyeur |
| YOLO26n / Ultralytics | Détection d'objets |
| OpenCV | Traitement des images et des vidéos |
| Streamlit | Développement de l'interface interactive |
| Pandas | Lecture et affichage des métriques d'entraînement |
| CSS | Personnalisation de l'interface |

## 📁 Structure du projet

```text
Smart-Conveyor-Vision/
├── app.py
├── detect_video.py
├── style.css
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   ├── logo_blender.png
│   ├── Ultralytics_YOLO_logo.jpg
│   └── tapis_3D_blender.png
│
├── dataset/
│
├── videos/
│   └── convoyeur_video.mp4
│
├── train/
│   ├── test_dataset.py
│   ├── test_detection.py
│   ├── train.py
│   └── visualisation_annotation.py
│
└── runs/
    └── detect/
        └── runs/
            └── detection_trous/
                └── weights/
                    └── best.pt
```

*Cette arborescence est indicative. Adapte les noms et les chemins à l'organisation réelle de ton projet. Les dossiers de résultats et les poids du modèle peuvent être exclus du dépôt s'ils sont trop volumineux.*

## 🧪 Scripts de développement

Le dossier `train/` regroupe les scripts Python utilisés pendant le développement et la validation du système.

- **`test_dataset.py` :** vérification du dataset et de sa structure.
- **`train.py` :** entraînement du modèle YOLO26n.
- **`test_detection.py` :** tests de détection sur des images ou des données de test.
- **`visualisation_annotation.py` :** visualisation des annotations et des cadres autour des objets.

Les noms ci-dessus sont des exemples de présentation : conserve les noms exacts de tes fichiers.

## ⚙️ Installation et lancement

### Prérequis

- Python installé sur la machine.
- Git pour récupérer le dépôt.
- Les fichiers nécessaires au fonctionnement du modèle.
- Un environnement compatible avec les dépendances d'Ultralytics et d'OpenCV.

### 1. Cloner le dépôt

```bash
git clone URL_DU_DEPOT
cd Smart-Conveyor-Vision
```

Remplace `URL_DU_DEPOT` par l'adresse réelle de ton dépôt GitHub.

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre ensuite dans le navigateur à l'adresse locale indiquée par Streamlit.

## 🧠 Modèle d'intelligence artificielle

Le projet utilise **YOLO26n**, un modèle de détection d'objets de la famille Ultralytics, entraîné pour reconnaître la classe `trou` à partir de données synthétiques issues de la simulation du convoyeur.

Le modèle entraîné est chargé par l'application pour effectuer les prédictions sur les vidéos.

Le fichier de poids personnalisé, généralement nommé `best.pt`, doit être disponible au chemin attendu par le code. S'il n'est pas inclus dans le dépôt, il faut prévoir une méthode pour le récupérer.

## 📊 Évaluation du modèle

L'application présente plusieurs indicateurs et visualisations issus de l'entraînement et de la validation :

- Précision.
- Rappel.
- mAP@50.
- mAP@50-95.
- Courbes d'apprentissage et de performance.
- Matrice de confusion.

Les valeurs affichées doivent être interprétées en fonction du jeu de données utilisé. Les performances mesurées sur des données synthétiques ne garantissent pas les mêmes résultats sur un convoyeur industriel réel.

## 🎥 Démonstration

Le fichier `videos/convoyeur_video.mp4` peut servir de vidéo de démonstration du convoyeur simulé.

L'application permet de charger une vidéo compatible, de lancer la détection et de télécharger la vidéo annotée.

Une capture d'écran du Dashboard ou une courte vidéo de démonstration peut également être ajoutée ici pour illustrer le fonctionnement du système.

## ⚠️ Limites et perspectives

Smart Conveyor Vision est un prototype expérimental développé à partir d'un environnement simulé.

Les perspectives d'amélioration comprennent :

- La validation du modèle sur des images et vidéos industrielles réelles.
- L'enrichissement du dataset avec différents types d'anomalies.
- L'amélioration de la robustesse face aux variations d'éclairage, de perspective et de vitesse du convoyeur.
- L'optimisation du temps d'inférence pour une éventuelle utilisation en temps réel.
- L'intégration future à un système de surveillance industrielle.

## 🐳 Exécution avec Docker

L'application Smart Conveyor Vision peut être exécutée dans un conteneur Docker afin de faciliter son installation et son utilisation.

### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et démarré.
- Le code source du projet cloné localement.

### 1. Construire l'image Docker

À la racine du projet, exécuter :

```bash
docker build -t smart-conveyor-vision .
```

Cette commande construit une image contenant l'environnement Python, les dépendances et les fichiers nécessaires à l'application.

> La première construction peut prendre du temps, notamment lors de l'installation de PyTorch.

### 2. Lancer l'application

```bash
docker run --rm -p 8501:8501 smart-conveyor-vision
```

### 3. Accéder à l'interface

Ouvrir le navigateur à l'adresse suivante :

http://localhost:8501

### 4. Arrêter l'application

Dans le terminal où le conteneur s'exécute, appuyer sur `Ctrl + C`.

L'option `--rm` supprime automatiquement le conteneur à son arrêt. L'image Docker reste disponible pour les prochains lancements.

### Contenu Docker

L'image est conçue pour inclure :
- L'interface web développée avec Streamlit.
- Les dépendances nécessaires à la détection d'anomalies.
- Le modèle YOLO entraîné (`best.pt`).
- Les graphiques et résultats d'entraînement utilisés dans l'interface.

Le dataset complet et les vidéos de simulation ne sont pas inclus dans l'image.

### Technologies utilisées

Python · YOLO · OpenCV · Streamlit · Docker · Blender

## 👨‍💻 Auteur

**NGUELE NGEBINE Lumière**

Étudiant ingénieur en informatique, intéressé par l'intelligence artificielle, la vision par ordinateur et le développement d'applications.

## 📄 Licence

Ce projet est distribué sous licence MIT. Consulte le fichier
[LICENSE](LICENSE) pour connaître les conditions d'utilisation,
de modification et de redistribution.

## Licence

- Code source : [Licence MIT](LICENSE)
- Jeu de données : [Licence CC BY 4.0](DATASET_LICENSE.md)