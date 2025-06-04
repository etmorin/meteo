import pandas as pd
import numpy as np

# ================================
# Chargement des données météo
# ================================
# Exemple de CSV fictif (tu peux le remplacer par tes vraies données météo)
# Crée un fichier "weather_data.csv" avec :
# temp,humidity,pressure,weather
# 22,60,1012,sunny
# 19,70,1008,rain
# 25,50,1010,cloudy
# ...

df = pd.read_csv('weather_data.csv')

# Encodage des labels (pluie=0, nuage=1, soleil=2)
labels = df['weather'].map({'rain': 0, 'cloudy': 1, 'sunny': 2}).values
y_one_hot = np.eye(3)[labels]  # 3 classes ➜ one-hot

# Sélection des colonnes météo
X = df[['temp', 'humidity', 'pressure']].values

# Normalisation des données météo
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# ================================
# Réseau neuronal
# ================================
class NeuralNetworkMeteo:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.tanh(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.model_output = self.softmax(self.output_input)

    def backward(self, X, y, lr=0.01):
        output_error = self.model_output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (1 - self.hidden_output ** 2)

        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        d_weights_input_hidden = np.dot(X.T, hidden_error)

        self.weights_hidden_output -= lr * d_weights_hidden_output
        self.weights_input_hidden -= lr * d_weights_input_hidden

        return self.mse_loss(y, self.model_output)

    def train(self, X, y, epochs=200, lr=0.01):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y, lr)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.model_output, axis=1)

    def predict_probabilities(self, X):
        self.forward(X)
        return self.model_output

# ================================
# Entraînement
# ================================
nn = NeuralNetworkMeteo(input_size=X.shape[1], hidden_size=16, output_size=3)
nn.train(X, y_one_hot, epochs=200, lr=0.01)

# Exemple de prédiction (les 5 premières lignes)
predictions = nn.predict_probabilities(X[:5])
print("\nPrédictions des 5 premières lignes :")
for i, pred in enumerate(predictions):
    print(f"Exemple {i+1}: {pred * 100} %")

# Sauvegarde des poids (optionnel)
np.savez('model_meteo_weights.npz', 
         weights_input_hidden=nn.weights_input_hidden,
         weights_hidden_output=nn.weights_hidden_output)
print("\nModèle sauvegardé sous 'model_meteo_weights.npz'.")
