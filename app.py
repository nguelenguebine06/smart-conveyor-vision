import tempfile
from pathlib import Path
import cv2
from detect_video import detecter_video
import streamlit as st

@st.cache_resource
def charger_modele():
    return YOLO(
        "runs/detect/runs/detection_trous/weights/best.pt"
    )

st.set_page_config(
    page_title="Smart Conveyor Vision",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = Path(__file__).parent / "style.css"

    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )

load_css()


if "dernier_rapport" not in st.session_state:
    st.session_state.dernier_rapport = None


# NAVIGATION


with st.sidebar:
    logo_blender = (
        Path(__file__).parent
        / "assets"
        / "logo_blender.png"
    )

    logo_yolo = (
        Path(__file__).parent
        / "assets"
        / "Ultralytics_YOLO_logo.jpg"
    )

    if logo_blender.exists():
        st.image(str(logo_blender), width=150)

    if logo_yolo.exists():
        st.image(str(logo_yolo), width=150)

    st.title("🏭 Smart Conveyor")
    st.caption("Vision industrielle par IA")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Modèle IA",
            "Analyse vidéo",
            "Entraînement du modèle"
        ]
    )

    st.divider()
    st.caption("YOLO26n • OpenCV • Streamlit • Blender ")


# VUE DASHBOARD


if page == "Dashboard":

        #st.title("Vision")
        col1, col2 = st.columns([1, 1.2], gap="medium")
        with col1:
            st.markdown('<div style="padding-top: 100px;">', unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center;">
                <h1>Smart Conveyor Vision</h1>
                <h3>Inspection intelligente</h3>
                <p>
                    Prototype de détection automatique des trous
                    sur un tapis roulant industriel grâce à YOLO26n.
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_modele, col_classe, col_statut = st.columns(3)

            with col_modele:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Modèle de détection</div>
                    <div class="metric-value">YOLO26n</div>
                </div>
                """, unsafe_allow_html=True)

            with col_classe:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Classe détectée</div>
                    <div class="metric-value">Trou</div>
                </div>
                """, unsafe_allow_html=True)

            with col_statut:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Statut</div>
                    <div class="metric-value">Prêt</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            image_convoyeur = (
                    Path(__file__).parent
                    / "assets"
                    / "tapis_3D_blender.png"
            )

            if image_convoyeur.exists():
                st.image(
                    str(image_convoyeur),
                    width=1000
                )
            st.markdown("""
            <div class="description-card">
                <h3>À propos du système</h3>
                <p>
                    Le convoyeur est modélisé et simulé en 3D avec Blender
                    afin de représenter une bande transporteuse comportant
                    des anomalies, notamment des trous. Les images générées
                    sont ensuite analysées par YOLO26n pour détecter ces
                    défauts. L'interface Streamlit permet de visualiser
                    le système et de présenter les résultats de détection.
                </p>
            </div>
            """, unsafe_allow_html=True)


# VUE MODELE IA

elif page == "Modèle IA":
    st.title("Modèle IA")
    st.write("Présentation du modèle de détection des trous.")

    st.subheader("Informations générales")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Architecture :** YOLO26n")
        st.write("**Tâche :** Détection d'objets")
        st.write("**Classe :** trou")

    with col2:
        st.write("**Environnement :** Python / Ultralytics")
        st.write("**Entraînement :** données synthétiques Blender")
        st.write("**Exécution :** CPU")

    st.divider()

    st.subheader("Métriques de validation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Précision", "99,5 %")
    col2.metric("Rappel", "100 %")
    col3.metric("mAP@50", "99,5 %")
    col4.metric("mAP@50-95", "94,4 %")

    st.caption(
        "Résultats obtenus sur les 80 images de validation "
        "du jeu de données synthétique."
    )

    st.divider()

    st.subheader("Temps d'inférence")

    st.metric("Inférence moyenne", "105,9 ms / image")

    st.write(
        "Le temps d'inférence a été mesuré sur CPU. "
        "Les performances peuvent varier selon le matériel "
        "et la résolution des images."
    )

    st.info(
        "Ces résultats concernent uniquement les données synthétiques "
        "utilisées pour la validation. Ils ne garantissent pas les "
        "mêmes performances sur un convoyeur industriel réel."
    )



# VUE ANALYSE VIDEO

elif page == "Analyse vidéo":
    st.title("Analyse vidéo")
    st.write("Charge une vidéo pour détecter les trous du convoyeur.")

    video_file = st.file_uploader(
        "Sélectionner une vidéo",
        type=["mp4", "avi", "mov"]
    )

    if video_file is not None:
        st.subheader("Vidéo originale")
        st.video(video_file)

        st.write(f"Fichier : {video_file.name}")

        seuil = st.slider(
            "Seuil de confiance YOLO",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05
        )

        if st.button("Lancer la prédiction"):
            chemin_video = None
            chemin_sortie = None

            try:
                # Créer les fichiers temporaires
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(video_file.name).suffix
                ) as fichier:
                    fichier.write(video_file.getvalue())
                    chemin_video = fichier.name

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as fichier:
                    chemin_sortie = fichier.name

                st.subheader("Prédiction en cours")

                affichage = st.empty()
                barre = st.progress(0)
                statut = st.empty()

                def afficher_image(image, traitees, total):
                    image_rgb = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

                    affichage.image(
                        image_rgb,
                        channels="RGB",
                        use_container_width=True
                    )

                    if total > 0:
                        barre.progress(
                            min(traitees / total, 1.0)
                        )

                    statut.write(
                        f"Images traitées : {traitees} / {total}"
                    )

                # Appel au module de détection
                resultats = detecter_video(
                    video_path=chemin_video,
                    output_path=chemin_sortie,
                    callback=afficher_image,
                    conf=seuil
                )

                st.success("Analyse terminée.")

                st.subheader("Résultats")
                col1, col2 = st.columns(2)

                col1.metric(
                    "Images traitées",
                    resultats["images_traitees"]
                )

                col2.metric(
                    "Détections cumulées",
                    resultats["total_detections"]
                )

                with open(chemin_sortie, "rb") as fichier:
                    video_annotee = fichier.read()

                st.download_button(
                    "Télécharger la vidéo annotée",
                    data=video_annotee,
                    file_name="convoyeur_detecte.mp4",
                    mime="video/mp4"
                )

            except Exception as erreur:
                st.error(f"Erreur pendant l'analyse : {erreur}")

            finally:
                # Nettoyer les fichiers temporaires
                for chemin in (chemin_video, chemin_sortie):
                    if chemin and Path(chemin).exists():
                        Path(chemin).unlink()
    else:
        st.info("Charge une vidéo pour commencer.")

# VUE ENTRAINNEMENT DU MODELE

elif page == "Entraînement du modèle":
    st.title("Entraînement du modèle")
    st.write(
        "Visualisation des performances et des résultats "
        "de l'entraînement de YOLO26n."
    )

    dossier = (
        Path(__file__).parent
        / "runs"
        / "detect"
        / "runs"
        / "detection_trous"
    )

    # 1. Courbes d'apprentissage
    st.subheader("1. Courbes d'apprentissage")

    fichier = dossier / "results.png"
    if fichier.exists():
        st.image(str(fichier), caption="Résultats de l'entraînement",
                 use_container_width=True)
    else:
        st.warning("Fichier results.png introuvable.")

    # 2. Performances de détection
    st.subheader("2. Courbes de performance")

    colonnes = st.columns(2)

    graphiques = [
        ("BoxP_curve.png", "Précision (BoxP)"),
        ("BoxR_curve.png", "Rappel (BoxR)"),
        ("BoxF1_curve.png", "Score F1 (BoxF1)"),
        ("BoxPR_curve.png", "Courbe précision-rappel"),
    ]

    for i, (nom, titre) in enumerate(graphiques):
        chemin = dossier / nom
        if chemin.exists():
            with colonnes[i % 2]:
                st.image(
                    str(chemin),
                    caption=titre,
                    use_container_width=True
                )

    # 3. Matrice de confusion
    st.subheader("3. Matrice de confusion")

    matrice = dossier / "confusion_matrix_normalized.png"
    if matrice.exists():
        st.image(
            str(matrice),
            caption="Matrice de confusion normalisée",
            use_container_width=True
        )

    # 4. Dataset et exemples
    st.subheader("4. Données d'entraînement")

    colonnes = st.columns(2)

    for nom, titre, colonne in [
        ("labels.jpg", "Répartition des annotations", 0),
        ("train_batch0.jpg", "Exemples d'entraînement", 1),
        ("val_batch2_labels.jpg", "Exemples de validation", 0),
    ]:
        chemin = dossier / nom
        if chemin.exists():
            with colonnes[colonne]:
                st.image(
                    str(chemin),
                    caption=titre,
                    use_container_width=True
                )
    # 5. Tableau des métriques par époque
    st.subheader("5. Métriques de chaque époque")

    fichier_csv = dossier / "results.csv"

    if fichier_csv.exists():
        import pandas as pd

        df = pd.read_csv(fichier_csv)
        df.columns = df.columns.str.strip()

        # Nettoyer les noms des colonnes
        df.columns = [
            colonne.replace("metrics/", "")
                   .replace("train/", "train_")
                   .replace("val/", "val_")
                   .replace("(", "")
                   .replace(")", "")
            for colonne in df.columns
        ]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Télécharger les métriques (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="metriques_entrainement.csv",
            mime="text/csv"
        )
    else:
        st.warning("Le fichier results.csv est introuvable.")