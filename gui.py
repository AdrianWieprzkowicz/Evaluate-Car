import tkinter as tk
from tkinter import messagebox
import torch

def evaluate_car_gui(model, scaler):

    # Pobieranie danch wejściowych od użytkownika, przetwarzanie ich i używanie modelu do oceny samochodu.
    # Wyświetla komunikat z przewidywaną kategorią pojazdu.

    try:
        # Pobranie i konwersja danych z pól tekstowych
        max_speed = float(entry_max_speed.get())  # Prędkość maksymalna w km/h
        weight = float(entry_weight.get())  # Waga pojazdu w kg
        acceleration = float(entry_acceleration.get())  # Czas przyspieszenia 0-100 km/h w sekundach
        num_seats = int(entry_num_seats.get())  # Liczba miejsc w samochodzie

        # Przygotowanie danych do analizy przez model
        data = [[max_speed, weight, acceleration, num_seats]]

        # Skalowanie danych wejściowych przy użyciu dostarczonego skalera
        data_scaled = scaler.transform(data)

        # Konwersja danych do formatu tensorowego dla PyTorch
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

        # Przewidywanie klasy samochodu przez model
        model.eval()  # Ustawienie modelu w tryb ewaluacji (wyłączenie uczenia)
        with torch.no_grad():
            output = model(data_tensor)
            _, predicted = torch.max(output, 1)  # Pobranie indeksu klasy o najwyższym prawdopodobieństwie

            # Wyświetlenie odpowiedniego komunikatu w zależności od wyniku modelu
            if predicted.item() == 1:
                messagebox.showinfo("Result", "The model predicts that this is a sports car.")
            else:
                messagebox.showinfo("Result", "The model predicts that this is a regular car.")
    except ValueError:
        # Obsługa błędu w przypadku niepoprawnych danych wejściowych
        messagebox.showerror("Error", "Please enter valid numerical data.")

def start_gui(model, scaler):

    # Tworzenie i uruchamianie interfejsu użytkownika do wprowadzania danych oraz oceny samochodu.

    global entry_max_speed, entry_weight, entry_acceleration, entry_num_seats

    # Inicjalizacja głównego okna aplikacji
    root = tk.Tk()
    root.title("Car Evaluation")
    root.geometry("400x300")  # Ustawienie rozmiaru okna

    # Utworzenie ramki dla lepszego rozmieszczenia elementów w oknie
    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(expand=True)

    # Etykieta i pole tekstowe do wprowadzania prędkości maksymalnej
    tk.Label(frame, text="Max Speed (km/h):").pack(pady=5)
    entry_max_speed = tk.Entry(frame)
    entry_max_speed.pack()

    # Etykieta i pole tekstowe do wprowadzania wagi pojazdu
    tk.Label(frame, text="Weight (kg):").pack(pady=5)
    entry_weight = tk.Entry(frame)
    entry_weight.pack()

    # Etykieta i pole tekstowe do wprowadzania czasu przyspieszenia
    tk.Label(frame, text="Acceleration to 100 km/h (seconds):").pack(pady=5)
    entry_acceleration = tk.Entry(frame)
    entry_acceleration.pack()

    # Etykieta i pole tekstowe do wprowadzania liczby siedzeń
    tk.Label(frame, text="Number of Seats:").pack(pady=5)
    entry_num_seats = tk.Entry(frame)
    entry_num_seats.pack()

    # Przycisk uruchamiający ocenę samochodu
    button_evaluate = tk.Button(frame, text="Evaluate Car", command=lambda: evaluate_car_gui(model, scaler))
    button_evaluate.pack(pady=10)

    # Uruchomienie pętli głównej interfejsu graficznego
    root.mainloop()
