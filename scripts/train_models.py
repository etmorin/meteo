import pandas as pd
import numpy as np
from tensorflow import keras

biomes = ['oceanic', 'continental', 'mediterranean', 'mountain', 'desert', 'tropical', 'subarctic', 'equatorial']

def load_data(biome):
    file = f"data/data_{biome}.csv"
    df = pd.read_csv(file)

    # Variables météo pour l'entraînement
    X = df[['temp', 'windspeed', 'pressure']].values

    # Exemple de transformation du weathercode en 3 classes : pluie (0), nuage (1), soleil (2)
    y = df['weathercode'].apply(lambda x: 0 if x > 50 else 1 if x > 0 else 2).values
    y_one_hot = np.eye(3)[y]

    # Normalisation
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, y_one_hot

def train_and_save_model(biome):
    X, y = load_data(biome)

    model = keras.models.Sequential([
        keras.layers.Dense(16, activation='tanh', input_shape=(X.shape[1],)),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=1)

    # Sauvegarde du modèle
    model.save(f"models/model_{biome}.h5")
    print(f"Modèle pour {biome} sauvegardé.")

if __name__ == "__main__":
    for biome in biomes:
        print(f"🔧 Entraînement pour le biome: {biome}")
        train_and_save_model(biome)
