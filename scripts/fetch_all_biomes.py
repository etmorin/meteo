import requests
import pandas as pd
from datetime import datetime

# Liste des biomes et leurs coordonnées (exemple - à adapter si tu veux)
biomes = {
    'oceanic': (44.65, -1.17),          # Arcachon
    'continental': (50.8503, 4.3517),   # Bruxelles
    'mediterranean': (43.3, 5.4),       # Marseille
    'mountain': (45.9, 6.1),            # Chamonix
    'desert': (23.4, 25.6),             # Sahara
    'tropical': (-3.7, -38.5),          # Fortaleza
    'subarctic': (64.1, -21.9),         # Reykjavik
    'equatorial': (0.5, 101.4)          # Sumatra
}

def fetch_and_save(biome_name, lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    response = requests.get(url)
    data = response.json()

    current = data['current_weather']

    meteo_data = {
        'datetime': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'temp': [current['temperature']],
        'windspeed': [current['windspeed']],
        'pressure': [current.get('pressure', 1012)],  # Fallback si manquant
        'weathercode': [current['weathercode']],
        'biome': [biome_name]
    }

    df = pd.DataFrame(meteo_data)
    file_name = f"data/data_{biome_name}.csv"

    df.to_csv(file_name, mode='a', index=False, header=not pd.io.common.file_exists(file_name))
    print(f"[{biome_name}] Données météo sauvegardées dans {file_name}")

def main():
    for biome, coords in biomes.items():
        fetch_and_save(biome, coords[0], coords[1])

if __name__ == "__main__":
    main()
