import tensorflow as tf
from tensorflow.keras import layers
import librosa
import numpy as np
import os


# =========================================================
# PARAMÈTRES
# =========================================================

SAMPLE_RATE = 16000
DUREE_AUDIO = 3
TAILLE_MFCC = 40

BATCH_SIZE = 32
EPOCHS = 30

TRAIN_DIR = "../donnees/entrainement"

TEST_DIR = "../donnees/Test"

MODEL_PATH = "/Modele/modele_locuteur.h5"


# =========================================================
# 1. EXTRACTION DES MFCC
# =========================================================

def extraire_mfcc(fichier):

    # -----------------------------------------------------
    # Charger le fichier audio
    # -----------------------------------------------------

    audio, sr = librosa.load(
        fichier,
        sr=SAMPLE_RATE,
        duration=DUREE_AUDIO
    )

    # -----------------------------------------------------
    # Déterminer la longueur attendue
    # -----------------------------------------------------

    longueur = SAMPLE_RATE * DUREE_AUDIO

    # -----------------------------------------------------
    # Si le fichier est trop court
    # Ajouter des zéros
    # -----------------------------------------------------

    if len(audio) < longueur:

        audio = np.pad(
            audio,
            (0, longueur - len(audio))
        )

    # -----------------------------------------------------
    # Si le fichier est trop long
    # Couper à 3 secondes
    # -----------------------------------------------------

    else:

        audio = audio[:longueur]

    # -----------------------------------------------------
    # Extraction des MFCC
    # -----------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=TAILLE_MFCC
    )

    # -----------------------------------------------------
    # Conversion en float32
    # -----------------------------------------------------

    return mfcc.astype(np.float32)


# =========================================================
# 2. CHARGEMENT DU DATASET
# =========================================================

def charger_dataset(dossier):

    X = []
    y = []

    # -----------------------------------------------------
    # Récupérer les classes
    # -----------------------------------------------------

    classes = sorted([
        nom
        for nom in os.listdir(dossier)
        if os.path.isdir(
            os.path.join(dossier, nom)
        )
    ])

    print("\nClasses trouvées :")

    for i, classe in enumerate(classes):

        print(
            i,
            "->",
            classe
        )

    # -----------------------------------------------------
    # Association classe -> numéro
    # -----------------------------------------------------

    class_to_index = {
        classe: i
        for i, classe in enumerate(classes)
    }

    # -----------------------------------------------------
    # Parcourir chaque classe
    # -----------------------------------------------------

    for classe in classes:

        dossier_classe = os.path.join(
            dossier,
            classe
        )

        label = class_to_index[classe]

        print(
            f"\nChargement de la classe : {classe}"
        )

        # -------------------------------------------------
        # Parcourir les fichiers
        # -------------------------------------------------

        fichiers = os.listdir(
            dossier_classe
        )

        for fichier in fichiers:

            # -------------------------------------------------
            # Vérifier que c'est un fichier WAV
            # -------------------------------------------------

            if not fichier.lower().endswith(".wav"):

                continue

            chemin = os.path.join(
                dossier_classe,
                fichier
            )

            try:

                # ---------------------------------------------
                # Extraire MFCC
                # ---------------------------------------------

                mfcc = extraire_mfcc(
                    chemin
                )

                # ---------------------------------------------
                # Ajouter aux données
                # ---------------------------------------------

                X.append(mfcc)

                y.append(label)

            except Exception as erreur:

                print(
                    "Erreur :",
                    chemin
                )

                print(
                    erreur
                )

    # -----------------------------------------------------
    # Transformer en tableaux NumPy
    # -----------------------------------------------------

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    return X, y, classes


# =========================================================
# 3. CHARGER LES DONNÉES D'ENTRAÎNEMENT
# =========================================================

print("\n======================================")
print("      CHARGEMENT ENTRAÎNEMENT")
print("======================================")

X_train, y_train, classes_train = charger_dataset(
    TRAIN_DIR
)


# =========================================================
# 4. CHARGER LES DONNÉES DE TEST
# =========================================================

print("\n======================================")
print("          CHARGEMENT TEST")
print("======================================")

X_test, y_test, classes_test = charger_dataset(
    TEST_DIR
)


# =========================================================
# 5. VÉRIFICATION DES CLASSES
# =========================================================

print("\n======================================")
print("             CLASSES")
print("======================================")

print(
    classes_train
)


# =========================================================
# 6. NOMBRE DE CLASSES
# =========================================================

NB_CLASSES = len(classes_train)

print(
    "\nNombre de classes :",
    NB_CLASSES
)


# =========================================================
# 7. INFORMATIONS SUR LES DONNÉES
# =========================================================

print("\n======================================")
print("       INFORMATIONS DATASET")
print("======================================")

print(
    "X_train :",
    X_train.shape
)

print(
    "y_train :",
    y_train.shape
)

print(
    "X_test  :",
    X_test.shape
)

print(
    "y_test  :",
    y_test.shape
)


# =========================================================
# 8. NORMALISATION
# =========================================================

print("\n======================================")
print("          NORMALISATION")
print("======================================")


# Moyenne calculée uniquement sur TRAIN

mean = np.mean(X_train)

std = np.std(X_train)


# Normalisation TRAIN

X_train = (
    X_train - mean
) / (
    std + 1e-8
)


# Normalisation TEST
# IMPORTANT :
# On utilise la moyenne et l'écart-type
# du TRAIN.

X_test = (
    X_test - mean
) / (
    std + 1e-8
)


# =========================================================
# 9. CONSTRUCTION DU MODÈLE
# =========================================================

print("\n======================================")
print("          CONSTRUCTION RNA")
print("======================================")


modele = tf.keras.Sequential([

    # -----------------------------------------------------
    # Entrée MFCC
    # -----------------------------------------------------

    layers.Input(
        shape=X_train.shape[1:]
    ),

    # -----------------------------------------------------
    # Transformer la matrice MFCC en vecteur
    # -----------------------------------------------------

    layers.Flatten(),

    # -----------------------------------------------------
    # Première couche cachée
    # -----------------------------------------------------

    layers.Dense(
        128,
        activation="relu"
    ),

    # -----------------------------------------------------
    # Deuxième couche cachée
    # -----------------------------------------------------

    layers.Dense(
        64,
        activation="relu"
    ),

    # -----------------------------------------------------
    # Couche de sortie
    # -----------------------------------------------------

    layers.Dense(
        NB_CLASSES,
        activation="softmax"
    )
])


# =========================================================
# 10. AFFICHER LE MODÈLE
# =========================================================

print("\n======================================")
print("             RÉSUMÉ")
print("======================================")

modele.summary()


# =========================================================
# 11. COMPILATION
# =========================================================

modele.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =========================================================
# 12. ENTRAÎNEMENT
# =========================================================

print("\n======================================")
print("            ENTRAÎNEMENT")
print("======================================")


historique = modele.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=1
)


# =========================================================
# 13. ÉVALUATION
# =========================================================

print("\n======================================")
print("             ÉVALUATION")
print("======================================")


loss, accuracy = modele.evaluate(
    X_test,
    y_test,
    verbose=1
)


print(
    "\nLoss :",
    f"{loss:.4f}"
)

print(
    "Accuracy :",
    f"{accuracy:.4f}"
)


# =========================================================
# 14. SAUVEGARDE DU MODÈLE
# =========================================================

print("\n======================================")
print("            SAUVEGARDE")
print("======================================")


# Créer le dossier si nécessaire

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


modele.save(
    MODEL_PATH
)


print(
    "\nModèle sauvegardé :"
)

print(
    MODEL_PATH
)


# =========================================================
# 15. SAUVEGARDE DES INFORMATIONS IMPORTANTES
# =========================================================

# On sauvegarde aussi les classes,
# la moyenne et l'écart-type.

np.savez(
    "../Modele/parametres.npz",

    mean=mean,

    std=std
)


# Sauvegarder les noms des classes

with open(
    "../Modele/classes.txt",
    "w"
) as fichier:

    for classe in classes_train:

        fichier.write(
            classe + "\n"
        )


print(
    "\nParamètres sauvegardés."
)

print(
    "Classes sauvegardées."
)


print("\n======================================")
print("             TERMINÉ")
print("======================================")
