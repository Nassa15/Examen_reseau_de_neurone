import tensorflow as tf
import librosa
import numpy as np


# =========================================================
# PARAMÈTRES
# =========================================================

SAMPLE_RATE = 16000
DUREE_AUDIO = 3
TAILLE_MFCC = 40

MODEL_PATH = "../Modele/modele_locuteur.h5"

PARAMETRES_PATH = "../Modele/parametres.npz"

CLASSES_PATH = "../Modele/classes.txt"


# =========================================================
# FICHIER AUDIO À TESTER
# =========================================================

AUDIO_TEST = "../donnees/test/Nassa/28.wav"


# =========================================================
# 1. CHARGER LE MODÈLE
# =========================================================

print("\n======================================")
print("       CHARGEMENT DU MODÈLE")
print("======================================")

modele = tf.keras.models.load_model(
    MODEL_PATH
)

print("Modèle chargé avec succès.")


# =========================================================
# 2. CHARGER LES PARAMÈTRES
# =========================================================

parametres = np.load(
    PARAMETRES_PATH
)

mean = parametres["mean"]

std = parametres["std"]


# =========================================================
# 3. CHARGER LES CLASSES
# =========================================================

with open(
    CLASSES_PATH,
    "r"
) as fichier:

    classes = [
        ligne.strip()
        for ligne in fichier
        if ligne.strip()
    ]


print("\nClasses :")

for i, classe in enumerate(classes):

    print(
        i,
        "->",
        classe
    )


# =========================================================
# 4. EXTRAIRE LES MFCC
# =========================================================

def extraire_mfcc(fichier):

    print(
        "\nChargement du fichier :",
        fichier
    )

    # -----------------------------------------------------
    # Charger le son
    # -----------------------------------------------------

    audio, sr = librosa.load(
        fichier,
        sr=SAMPLE_RATE,
        duration=DUREE_AUDIO
    )

    # -----------------------------------------------------
    # Longueur attendue
    # -----------------------------------------------------

    longueur = SAMPLE_RATE * DUREE_AUDIO

    # -----------------------------------------------------
    # Audio trop court
    # -----------------------------------------------------

    if len(audio) < longueur:

        audio = np.pad(
            audio,
            (
                0,
                longueur - len(audio)
            )
        )

    # -----------------------------------------------------
    # Audio trop long
    # -----------------------------------------------------

    else:

        audio = audio[:longueur]

    # -----------------------------------------------------
    # MFCC
    # -----------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=TAILLE_MFCC
    )

    return mfcc.astype(
        np.float32
    )


# =========================================================
# 5. EXTRACTION
# =========================================================

mfcc = extraire_mfcc(
    AUDIO_TEST
)


print(
    "\nDimension MFCC :",
    mfcc.shape
)


# =========================================================
# 6. NORMALISATION
# =========================================================

mfcc = (
    mfcc - mean
) / (
    std + 1e-8
)


# =========================================================
# 7. AJOUTER LA DIMENSION BATCH
# =========================================================

# Avant :
#
# (40, nombre_frames)
#
# Après :
#
# (1, 40, nombre_frames)

mfcc = np.expand_dims(
    mfcc,
    axis=0
)


print(
    "Dimension entrée modèle :",
    mfcc.shape
)


# =========================================================
# 8. PRÉDICTION
# =========================================================

probabilites = modele.predict(
    mfcc,
    verbose=0
)


# =========================================================
# 9. RÉCUPÉRER LA CLASSE
# =========================================================

indice = np.argmax(
    probabilites[0]
)

classe_predite = classes[indice]

confiance = probabilites[0][indice] * 100


# =========================================================
# 10. AFFICHER LES PROBABILITÉS
# =========================================================

print("\n======================================")
print("          RÉSULTAT PRÉDICTION")
print("======================================")

for i, probabilite in enumerate(
    probabilites[0]
):

    print(
        f"{classes[i]:15s} : "
        f"{probabilite * 100:.2f}%"
    )


# =========================================================
# 11. RÉSULTAT FINAL
# =========================================================

print("\n--------------------------------------")

print(
    "🎤 Locuteur reconnu :",
    classe_predite
)

print(
    f"🎯 Confiance : {confiance:.2f}%"
)

print("--------------------------------------")
