import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import re

def iterazionePuntoFisso():
    print("\n*** ITERAZIONE DEL PUNTO FISSO - Manuel Ganino ***")
    f = input("Inserisci la funzione del punto fisso. Attenzione: l'incognita rappresentala con x (ES. cos(x)): ")
    x0 = float(input("Inserisci il valore di partenza per l'iterazione (ES. 0.5): "))
    epsilon = input("Inserisci la tolleranza che determina la precisione dell'iterazione (ES. 10^-6): ")
    num_iterazioni = int(input("Inserisci il numero max di iterazioni (ES. 100): "))
    x_min = float(input("Inserisci l'intervallo minimo di rappresentazione del grafico: "))
    x_max = float(input("Inserisci l'intervallo massimo di rappresentazione del grafico: "))

    epsilon = converti_esponente(epsilon)

    print("Inizio del procedimento di risoluzione")

    x_sol = x0
    iterazioni = [x_sol]  # Lista per memorizzare le iterazioni

    if 'log' in f and x_sol <= 0:
        raise ValueError(f"Errore: log({x_sol}) non è definito.")

    for i in range(num_iterazioni):
        x_nuovo = calcola_nuovo_valore_funzione(f, x_sol)

        if x_nuovo is None:
            print(f"Iterazione interrotta all'iterazione {i+1}: valore non valido.")
            sys.exit(0)

        # Stampa i valori dell'iterazione corrente
        print(f"Iterazione {i+1}: x = {x_sol:.5f}, g(x) = {x_nuovo:.5f}")
        iterazioni.append(x_nuovo)
        # criterio di arresto
        if abs(x_nuovo - x_sol) < epsilon:
            print(f"Iterazione terminata \n la soluzione è {x_nuovo} con {i+1} iterazioni e un errore di {abs(x_nuovo - x_sol)}")
            visualizza_grafico(iterazioni, f, x0, x_min, x_max)
            return
        else:
            x_sol = x_nuovo
    print("Nessuna convergenza")
    visualizza_grafico(iterazioni, f, x0, x_min, x_max)
    return

def calcola_nuovo_valore_funzione(f, x_sol):
    try:
        espressione = preprocessa_espressione(f)
        funzioni_math = {nome: getattr(math, nome) for nome in dir(math) if callable(getattr(math, nome))}
        funzioni_math["x"] = x_sol
        return eval(espressione, {}, funzioni_math)
    except ValueError as e:
        print(f"Errore nel calcolo della funzione: {e}")
        return None
    except ZeroDivisionError as e:
        print(f"Errore nel calcolo della funzione: divisione per zero - {e}")
        return None
    except Exception as e:
        print(f"Errore nel calcolo della funzione: {e}")
        return None

def preprocessa_espressione(espressione):
    espressione = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', espressione)
    espressione = re.sub(r'(\d)\(', r'\1*(', espressione)
    return espressione

def converti_esponente(valore):
    if "^" in valore:
        base, esponente = valore.split("^")
        return float(base) ** float(esponente)
    return float(valore)


def visualizza_grafico(iterazioni, funzione, x0, x_min, x_max):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Iterazione del Punto Fisso\n$g(x) = {funzione}$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.grid(True)

    # Preparazione grafica
    x_vals = np.linspace(x_min, x_max, 300)
    y_vals = [calcola_nuovo_valore_funzione(funzione, x) for x in x_vals]
    line_gx, = ax.plot(x_vals, y_vals, 'b', label='g(x)')
    line_yx, = ax.plot(x_vals, x_vals, 'k', label='y = x')

    # Elementi animati con etichette
    path_line, = ax.plot([], [], 'r--', lw=1, alpha=0.7, label='Percorso iterazioni')
    start_dot = ax.scatter([x0], [x0], color='green', s=100, zorder=5, label='Punto iniziale (x0)')
    end_dot = ax.scatter([], [], color='purple', s=100, zorder=5, label='Punto corrente')

    # Creazione legenda personalizzata
    proxy_start = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)
    proxy_end = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10)
    proxy_path = plt.Line2D([], [], color='red', linestyle='--')

    ax.legend(
        handles=[line_gx, line_yx, proxy_path, proxy_start, proxy_end],
        labels=['g(x)', 'y = x', 'Percorso iterazioni', 'Punto iniziale (x0)', 'Punto corrente'],
        loc='upper right'
    )

    # Testo animato
    text_artist = ax.text(
        0.02, 0.95, '',
        transform=ax.transAxes,
        va='top',
        ha='left',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    def init():
        path_line.set_data([], [])
        end_dot.set_offsets(np.empty((0, 2)))
        text_artist.set_text('')
        return path_line, end_dot, text_artist

    def animate(frame):
        current_data = iterazioni[:frame + 1]

        # Costruisci percorso
        x_path, y_path = [], []
        for i in range(len(current_data)):
            if i == 0:
                x_path.append(current_data[i])
                y_path.append(current_data[i])
            else:
                # Verticale
                x_path.extend([current_data[i - 1], current_data[i - 1]])
                y_path.extend([current_data[i - 1], current_data[i]])
                # Orizzontale
                x_path.extend([current_data[i - 1], current_data[i]])
                y_path.extend([current_data[i], current_data[i]])

        path_line.set_data(x_path, y_path)

        # Aggiorna punto finale
        if current_data:
            end_dot.set_offsets(np.array([[current_data[-1], current_data[-1]]]))

        # Aggiorna testo con valori formattati
        if frame < len(iterazioni) - 1:
            text_str = (f"Iterazione: {frame + 1}\n"
                        f"xₙ = {iterazioni[frame]:.5f}\n"
                        f"g(xₙ) = {iterazioni[frame + 1]:.5f}")
        else:
            text_str = (f"Soluzione finale\n"
                        f"x = {iterazioni[-1]:.5f}\n"
                        f"Iterazioni: {len(iterazioni) - 1}")

        text_artist.set_text(text_str)

        return path_line, end_dot, text_artist

    ani = FuncAnimation(
        fig, animate,
        frames=len(iterazioni),
        init_func=init,
        blit=True
    )

    try:
        ani.save('punto_fisso.mp4', writer='ffmpeg', fps=2, extra_args=['-vcodec', 'libx264'])
        print("Animazione salvata correttamente come punto_fisso.mp4")
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")
        print("Assicurati che FFmpeg sia installato:")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Mac: brew install ffmpeg")
        print("  Windows: scarica da https://ffmpeg.org/download.html")

    plt.close()

iterazionePuntoFisso()