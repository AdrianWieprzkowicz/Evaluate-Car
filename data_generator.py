import random
import csv

# Funkcja do generowania pojedynczego wiersza danych
def generate_car_data():
    # Decydujemy, czy samochód jest sportowy, czy nie
    is_sport_car = random.choice([0, 1])

    if is_sport_car:
        # Dla samochodów sportowych
        speed = random.randint(220, 450)  # Prędkość maksymalna (km/h)
        weight = random.randint(800, 2500)  # Masa auta (kg)
        acceleration = round(random.uniform(2.0, 7.0), 1)  # Przyspieszenie do 100 km/h (s)
        seats = random.randint(1, 5)  # Liczba siedzeń (maksymalnie 4)
    else:
        # Dla samochodów niesportowych
        speed = random.randint(180, 250)  # Prędkość maksymalna (km/h)
        weight = random.randint(1200, 3000)  # Masa auta (kg)
        acceleration = round(random.uniform(6.0, 17.0), 1)  # Przyspieszenie do 100 km/h (s)
        seats = random.randint(4, 7)  # Liczba siedzeń (maksymalnie 7)

    return [speed, weight, acceleration, seats, is_sport_car]

# Generowanie 1000 rekordów
data = [generate_car_data() for _ in range(1000)]

# Zapisanie do pliku CSV
header = ['Max speed', 'Weight', 'Acceleration', 'Seats', 'Normal/Sport']

with open('cars_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Zapisz nagłówek
    writer.writerows(data)  # Zapisz dane

print("Baza danych została zapisana w pliku cars_data.csv")