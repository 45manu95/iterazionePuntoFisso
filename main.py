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

    print("\n*** ANALISI INIZIALE ***")
    # Analisi dominio e continuità
    dominio_valido, messaggio_dominio = analizza_dominio(f, x_min, x_max)
    if not dominio_valido:
        print(f"Errore dominio: {messaggio_dominio}")
        sys.exit(1)

    # Calcolo derivata e contrattività
    derivata_iniziale, contrattivo = verifica_contrattivita(f, x0)
    print(f"Derivata stimata in x0: {derivata_iniziale:.2f}")
    print("Contrattività locale:", "✅" if contrattivo else "❌ (|g'(x)| ≥ 1)")

    if not contrattivo:
        print("Avvertenza: La funzione potrebbe non essere contrattiva intorno a x0!")
        scelta = input("Vuoi comunque procedere? (s/n): ")
        if scelta.lower() != 's':
            sys.exit(0)

    print("\n*** INIZIO ITERAZIONI ***")
    x_sol = x0
    iterazioni = [x_sol]
    derivate = [derivata_iniziale]

    for i in range(num_iterazioni):
        x_nuovo = calcola_nuovo_valore_funzione(f, x_sol)

        if x_nuovo is None:
            print(f"Iterazione interrotta all'iterazione {i + 1}: valore non valido.")
            sys.exit(0)

        # Calcola derivata corrente
        derivata_corrente = calcola_derivata(f, x_sol)
        derivate.append(derivata_corrente)

        print(f"Iterazione {i + 1}:")
        print(f"xₙ = {x_sol:.5f}, g(xₙ) = {x_nuovo:.5f}, |g'(xₙ)| = {abs(derivata_corrente):.5f}")

        iterazioni.append(x_nuovo)

        if abs(x_nuovo - x_sol) < epsilon:
            print(f"\n*** CONVERGENZA RAGGIUNTA ***")
            print(f"Soluzione: {x_nuovo:.8f}")
            print(f"Iterazioni: {i + 1}")
            print(f"Errore: {abs(x_nuovo - x_sol):.2e}")
            print(
                f"Derivata finale: {derivata_corrente:.5f} ({'contrattivo' if abs(derivata_corrente) < 1 else 'non contrattivo'})")
            visualizza_grafico(iterazioni, f, x0, x_min, x_max, derivate)
            return
        else:
            x_sol = x_nuovo

    print("\n*** NESSUNA CONVERGENZA ***")
    visualizza_grafico(iterazioni, f, x0, x_min, x_max, derivate)

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


def calcola_derivata(f, x, h=1e-5):
    f_plus = calcola_nuovo_valore_funzione(f, x + h)
    f_minus = calcola_nuovo_valore_funzione(f, x - h)
    return (f_plus - f_minus) / (2 * h)


def analizza_dominio(f, x_min, x_max):
    campioni = 100
    x_test = np.linspace(x_min, x_max, campioni)

    for x in x_test:
        try:
            calcola_nuovo_valore_funzione(f, x)
        except Exception as e:
            return False, f"Funzione non definita in x = {x:.2f} - {str(e)}"

    # Controllo aggiuntivo per funzioni particolari
    if 'log' in f and x_min <= 0:
        return False, "Dominio del logaritmo non rispettato (x deve essere > 0)"

    if 'sqrt' in f and x_min < 0:
        return False, "Radice quadrata richiede x ≥ 0"

    return True, "Dominio valido"

def verifica_contrattivita(f, x0, h=1e-5):
    try:
        derivata = calcola_derivata(f, x0, h)
        return derivata, abs(derivata) < 1
    except:
        return float('nan'), False


def visualizza_grafico(iterazioni, funzione, x0, x_min, x_max, derivate):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Grafico principale
    ax1.set_title(f"Iterazione del Punto Fisso\n$g(x) = {funzione}$")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(x_min, x_max)
    ax1.grid(True)

    x_vals = np.linspace(x_min, x_max, 300)
    y_vals = [calcola_nuovo_valore_funzione(funzione, x) for x in x_vals]
    ax1.plot(x_vals, y_vals, 'b', label='g(x)')
    ax1.plot(x_vals, x_vals, 'k', label='y = x')

    # Elementi animati
    path_line, = ax1.plot([], [], 'r--', lw=1, alpha=0.7, label='Percorso iterazioni')
    start_dot = ax1.scatter([x0], [x0], color='green', s=100, zorder=5, label='Punto iniziale')
    end_dot = ax1.scatter([], [], color='purple', s=100, zorder=5, label='Punto corrente')

    # Legenda
    ax1.legend(loc='upper right')

    # Grafico derivata
    ax2.set_title("Andamento della derivata |g'(x)|")
    ax2.set_xlabel('Iterazioni')
    ax2.set_ylabel("|g'(x)|")
    ax2.axhline(1, color='red', linestyle='--', label='Soglia di contrattività')
    ax2.set_ylim(0, max(1.5, max(np.abs(derivate)) * 1.1))
    ax2.grid(True)

    der_line, = ax2.plot([], [], 'b-', label='Derivata')
    der_dots = ax2.scatter([], [], color='blue', s=30, zorder=5)

    # Testo animato
    text_artist = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top', ha='left',
                           bbox=dict(facecolor='white', alpha=0.8))

    def init():
        path_line.set_data([], [])
        end_dot.set_offsets(np.empty((0, 2)))
        der_line.set_data([], [])
        der_dots.set_offsets(np.empty((0, 2)))
        text_artist.set_text('')
        return path_line, end_dot, der_line, der_dots, text_artist

    def animate(frame):
        current_data = iterazioni[:frame + 1]
        current_der = derivate[:frame + 1]

        # Aggiorna percorso
        x_path, y_path = [], []
        for i in range(len(current_data)):
            if i == 0:
                x_path.append(current_data[i])
                y_path.append(current_data[i])
            else:
                x_path.extend([current_data[i - 1], current_data[i - 1]])
                y_path.extend([current_data[i - 1], current_data[i]])
                x_path.extend([current_data[i - 1], current_data[i]])
                y_path.extend([current_data[i], current_data[i]])
        path_line.set_data(x_path, y_path)

        # Aggiorna punto finale
        if current_data:
            end_dot.set_offsets([[current_data[-1], current_data[-1]]])

        # Aggiorna grafico derivata
        der_line.set_data(np.arange(frame + 1), np.abs(current_der))
        der_dots.set_offsets(np.c_[np.arange(frame + 1), np.abs(current_der)])

        # Testo
        text_str = f"Iterazione: {frame + 1}\n"
        if frame < len(iterazioni) - 1:
            text_str += f"xₙ = {iterazioni[frame]:.5f}\ng(xₙ) = {iterazioni[frame + 1]:.5f}"
        else:
            text_str += f"x finale = {iterazioni[-1]:.5f}"
        text_artist.set_text(text_str)

        return path_line, end_dot, der_line, der_dots, text_artist

    ani = FuncAnimation(fig, animate, frames=len(iterazioni), init_func=init, blit=True)

    try:
        ani.save('punto_fisso_avanzato.mp4', writer='ffmpeg', fps=2, extra_args=['-vcodec', 'libx264'])
        print("\nAnimazione salvata come 'punto_fisso_avanzato.mp4'")
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")

    plt.close()

iterazionePuntoFisso()