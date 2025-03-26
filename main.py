import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from gui import start_gui  # Import funkcji uruchamiającej interfejs użytkownika

# Wczytanie danych z pliku CSV
data = pd.read_csv('cars_data.csv')

# Przygotowanie danych wejściowych (cechy) oraz etykiet (klasy)
X = data[['Max speed', 'Weight', 'Acceleration', 'Seats']].values  # Cechy samochodu
y = data['Normal/Sport'].values  # Klasyfikacja: 0 - samochód normalny, 1 - sportowy

# Podział danych na zestaw treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja danych wejściowych (skalowanie cech do podobnych wartości)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Dopasowanie skalera i przekształcenie danych treningowych
X_test_scaled = scaler.transform(X_test)  # Skalowanie danych testowych

# Konwersja danych na tensory PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Definicja modelu neuronowego (Perceptron wielowarstwowy)
class PerceptronModel(nn.Module):
    def __init__(self):
        super(PerceptronModel, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Warstwa wejściowa (4 cechy → 10 neuronów)
        self.fc2 = nn.Linear(10, 2)  # Warstwa wyjściowa (10 neuronów → 2 klasy)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Funkcja aktywacji ReLU
        x = self.fc2(x)  # Wynik surowy przed softmax
        return x

# Inicjalizacja modelu
model = PerceptronModel()

# Definicja funkcji straty i optymalizatora
criterion = nn.CrossEntropyLoss()  # Funkcja kosztu dla klasyfikacji wieloklasowej
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastyczny gradient prosty (SGD)

# Trening modelu
num_epochs = 1000  # Liczba epok treningowych
for epoch in range(num_epochs):
    model.train()  # Tryb treningowy
    optimizer.zero_grad()  # Zerowanie gradientów
    outputs = model(X_train_tensor)  # Przewidywanie dla zbioru treningowego
    loss = criterion(outputs, y_train_tensor)  # Obliczenie błędu (straty)
    loss.backward()  # Obliczenie gradientów
    optimizer.step()  # Aktualizacja wag modelu

    # Wyświetlanie informacji co 100 epok
    if (epoch + 1) % 100 == 0:
        print(f'Epoka [{epoch + 1}/{num_epochs}], Strata: {loss.item():.4f}')

# Ocena modelu na danych testowych
model.eval()  # Ustawienie modelu w tryb ewaluacji
with torch.no_grad():
    outputs = model(X_test_tensor)  # Przewidywanie na zbiorze testowym
    _, predicted = torch.max(outputs, 1)  # Pobranie przewidywanych etykiet
    accuracy = accuracy_score(y_test_tensor, predicted)  # Obliczenie dokładności
    print(f'Dokładność na zbiorze testowym: {accuracy * 100:.2f}%')

# Uruchomienie interfejsu graficznego do klasyfikacji samochodów
start_gui(model, scaler)
