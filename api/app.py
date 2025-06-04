from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Liste des biomes pour charger le bon modèle
biome_models = {
    'oceanic': 'models/model_oceanic.h5',
    'continental': 'models/model_continental.h5',
    'mediterranean': 'models/model_mediterranean.h5',
    'mountain': 'models/model_mountain.h5',
    'desert': 'models/model_desert.h5',
    'tropical': 'models/model_tropical.h5',
    'subarctic': 'models/model_subarctic.h5',
    'equatorial': 'models/model_equatorial.h5'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    biome = data.get('biome')
    if biome not in biome_models:
        return jsonify({'error': 'Biome invalide'}), 400

    # Charger le modèle
    model = keras.models.load_model(biome_models[biome])

    # Extraire les features (dans l’ordre : temp, humidity, pressure, etc.)
    features = [
        data.get('temp'),
        data.get('humidity'),
        data.get('pressure'),
        data.get('windspeed', 0)  # Exemple si tu as le vent
    ]

    # Normaliser si besoin (ici, fictif - à adapter avec tes vraies moyennes et std)
    features = np.array(features).reshape(1, -1)
    # features = (features - moyenne) / ecart_type  # À faire si normalisation

    # Prédire
    predictions = model.predict(features)[0] * 100  # Pourcentages

    result = {
        'rain': round(predictions[0], 2),
        'cloudy': round(predictions[1], 2),
        'sunny': round(predictions[2], 2)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)