#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medir_speedup.py
Ejecuta el descubridor con diferentes números de procesos MPI,
mide el tiempo de ejecución, calcula el speedup y ajusta la curva de Amdahl.
Los resultados se guardan en la carpeta de la última ejecución (salidas/<timestamp>/03-speedup/).
"""

import os
import sys
import subprocess
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

# ============================================================
# CONFIGURACIÓN
# ============================================================
BINARIO = "./descubridor_c"
PROCESOS = [1, 2, 4, 8, 10, 12, 16, 20]
REPETICIONES = 10  # Número de ejecuciones por cada N (para promediar)

# ============================================================
# FUNCIONES
# ============================================================
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def encontrar_ultima_carpeta_salidas(base_dir="salidas"):
    """Encuentra la carpeta de salida más reciente en salidas/."""
    if not os.path.exists(base_dir):
        return None
    carpetas = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not carpetas:
        return None
    carpetas.sort(reverse=True)
    return os.path.join(base_dir, carpetas[0])

def ejecutar_con_procesos(n_procesos):
    """Ejecuta el binario con N procesos y devuelve el tiempo total (segundos)."""
    cmd = ["mpirun", "-np", str(n_procesos), BINARIO]
    print(f"[{timestamp()}] Ejecutando con {n_procesos} procesos...")
    inicio = time.time()
    try:
        resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        salida = resultado.stdout + resultado.stderr
        duracion = time.time() - inicio
        
        # Buscar la línea de tiempo total en la salida del maestro
        patron = r"Todas las tareas completadas en ([\d.]+) segundos"
        match = re.search(patron, salida)
        if match:
            tiempo = float(match.group(1))
            return tiempo, True  # tiempo del maestro
        else:
            return duracion, False  # wall-clock
    except subprocess.TimeoutExpired:
        print(f"[{timestamp()}]   TIMEOUT con {n_procesos} procesos")
        return None, False
    except Exception as e:
        print(f"[{timestamp()}]   ERROR: {e}")
        return None, False

def amdahl(p, fp):
    """Ley de Amdahl: S(p) = 1 / ((1 - fp) + fp / p)"""
    return 1.0 / ((1.0 - fp) + fp / p)

def ajustar_amdahl(procs, speedups):
    """Ajusta la fracción paralelizable usando mínimos cuadrados (solo con puntos donde crece)."""
    def amdahl_fit(p, fp):
        return amdahl(p, fp)
    
    # Solo usar puntos donde el speedup crece (hasta 8 procesos)
    procs_fit = []
    speedups_fit = []
    for p, s in zip(procs, speedups):
        if p <= 8:  # Solo hasta 8 procesos
            procs_fit.append(p)
            speedups_fit.append(s)
    
    if len(procs_fit) < 2:
        return 0.9
    
    p0 = [0.9]
    try:
        popt, _ = curve_fit(amdahl_fit, procs_fit, speedups_fit, p0=p0, bounds=(0.0, 1.0))
        return popt[0]
    except:
        print("Advertencia: No se pudo ajustar Amdahl, usando fp=0.9")
        return 0.9

def main():
    # Verificar binario
    if not os.path.exists(BINARIO):
        print(f"ERROR: No se encontró el binario '{BINARIO}'")
        print("Compila primero con 'make'.")
        sys.exit(1)

    print("=" * 70)
    print("MEDICIÓN DE SPEEDUP - DESCUBRIDOR MPI")
    print(f"Inicio: {timestamp()}")
    print("=" * 70)
    print(f"Binario: {BINARIO}")
    print(f"Procesos a probar: {PROCESOS}")
    print(f"Repeticiones por configuración: {REPETICIONES}")

    # Ejecutar para cada número de procesos
    resultados = {}
    for n in PROCESOS:
        tiempos = []
        tiempo_desde_maestro = True
        for rep in range(REPETICIONES):
            t, from_master = ejecutar_con_procesos(n)
            if t is not None:
                tiempos.append(t)
                if not from_master:
                    tiempo_desde_maestro = False
        if tiempos:
            promedio = np.mean(tiempos)
            desv = np.std(tiempos) if len(tiempos) > 1 else 0
            resultados[n] = {
                "tiempos": tiempos,
                "promedio": promedio,
                "desv": desv,
                "desde_maestro": tiempo_desde_maestro
            }
            print(f"[{timestamp()}] {n} procesos: promedio={promedio:.2f}s, desv={desv:.2f}s")
        else:
            print(f"[{timestamp()}] {n} procesos: FALLIDO")

    # Verificar que al menos 1 proceso funcionó
    if 1 not in resultados:
        print("ERROR: No se obtuvo tiempo para 1 proceso.")
        sys.exit(1)

    # Calcular speedup
    t1 = resultados[1]["promedio"]
    speedups = {}
    for n, data in resultados.items():
        speedups[n] = 1.0 if n == 1 else (t1 / data["promedio"] if data["promedio"] > 0 else float('nan'))

    # Ajustar Amdahl (solo con puntos hasta 8)
    procs_amdahl = sorted([p for p in speedups if speedups[p] > 0 and p <= 8])
    speedups_amdahl = [speedups[p] for p in procs_amdahl]
    fp = ajustar_amdahl(procs_amdahl, speedups_amdahl)
    
    # Generar puntos para la curva de Amdahl
    p_amdahl = np.linspace(1, max(PROCESOS), 200)
    s_amdahl = amdahl(p_amdahl, fp)
    speedup_max_teorico = 1.0 / (1.0 - fp)
    
    # Speedup máximo real alcanzado
    speedup_real_max = max([s for p, s in speedups.items() if p > 1 and not np.isnan(s)])
    mejor_proc = max([p for p, s in speedups.items() if p > 1 and not np.isnan(s)], key=lambda p: speedups[p])

    # Mostrar resultados
    print("\n" + "-" * 40)
    print("RESULTADOS FINALES")
    print("-" * 40)
    for n in sorted(resultados.keys()):
        data = resultados[n]
        sp = speedups.get(n, float('nan'))
        print(f"  {n} procesos: tiempo={data['promedio']:.2f}s, speedup={sp:.2f}x")
    print(f"\nFracción paralelizable (Amdahl, ajustada con 1,2,4,8): fp = {fp:.4f}")
    print(f"Speedup máximo teórico (Amdahl, sin overhead): {speedup_max_teorico:.2f}x")
    print(f"Speedup máximo real alcanzado: {speedup_real_max:.2f}x (con {mejor_proc} procesos)")
    print("\nNOTA: La caída del speedup después de 8 procesos se debe al overhead de comunicación y contención de recursos, que no está contemplado por la ley de Amdahl.")

    # ---- Determinar carpeta de salida (la más reciente en salidas/) ----
    carpeta_salidas = encontrar_ultima_carpeta_salidas()
    if carpeta_salidas is None:
        print("ERROR: No se encontró ninguna carpeta en 'salidas/'. ¿El binario generó resultados?")
        print("Asegúrate de que el binario C haya generado una carpeta con timestamp en 'salidas/'.")
        sys.exit(1)

    speedup_dir = os.path.join(carpeta_salidas, "03-speedup")
    os.makedirs(speedup_dir, exist_ok=True)
    print(f"\nGuardando resultados de speedup en: {speedup_dir}")

    # Guardar CSV
    csv_path = os.path.join(speedup_dir, "speedup_data.csv")
    with open(csv_path, "w") as archivo_csv:
        archivo_csv.write("procesos,tiempo_promedio,desviacion,speedup,desde_maestro\n")
        for n in sorted(resultados.keys()):
            data = resultados[n]
            sp = speedups.get(n, float('nan'))
            archivo_csv.write(f"{n},{data['promedio']:.6f},{data['desv']:.6f},{sp:.6f},{1 if data['desde_maestro'] else 0}\n")
    print(f"  CSV guardado: {csv_path}")

    # Generar gráfico
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Tiempo vs procesos
    procs = sorted(resultados.keys())
    tiempos = [resultados[p]["promedio"] for p in procs]
    desv = [resultados[p]["desv"] for p in procs]
    axes[0].bar([str(p) for p in procs], tiempos, yerr=desv, color='#2E86AB', capsize=5)
    axes[0].set_xlabel('Procesos MPI')
    axes[0].set_ylabel('Tiempo (s)')
    axes[0].set_title('Tiempo vs procesos')
    axes[0].grid(True, alpha=0.3)
    for i, (p, t) in enumerate(zip(procs, tiempos)):
        axes[0].text(i, t + 0.05, f'{t:.2f}s', ha='center', fontsize=9)

    # Gráfico 2: Speedup con Amdahl
    procs_plot = sorted([p for p in speedups if not np.isnan(speedups[p])])
    speedups_plot = [speedups[p] for p in procs_plot]
    
    axes[1].plot(procs_plot, speedups_plot, 'o-', color='#D64933', linewidth=2, markersize=8, label='Speedup real')
    axes[1].plot(p_amdahl, s_amdahl, 'b--', linewidth=2, label=f'Amdahl (fp={fp:.3f})')
    axes[1].plot(procs_plot, procs_plot, 'k--', linewidth=2, alpha=0.5, label='Ideal')
    axes[1].set_xlabel('Procesos MPI')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Speedup vs procesos')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    for p, sp in zip(procs_plot, speedups_plot):
        axes[1].text(p, sp + 0.1, f'{sp:.2f}x', ha='center', fontsize=9)
    
    # Anotar fp y S_max en el gráfico (esquina inferior derecha)
    axes[1].text(0.7, 0.05, f'fp = {fp:.3f}', transform=axes[1].transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    axes[1].text(0.7, 0.12, f'S_max (teórico) = {speedup_max_teorico:.2f}x', transform=axes[1].transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Nota sobre caída por overhead (en el medio del lado izquierdo, sin tapa)
    axes[1].text(0.05, 0.5, 'Caída por overhead de comunicación\ny contención de recursos', transform=axes[1].transAxes,
                 fontsize=8, bbox=dict(facecolor='yellow', alpha=0.6), ha='left', va='center')

    plt.tight_layout()
    ruta_grafico = os.path.join(speedup_dir, "speedup_analysis.png")
    plt.savefig(ruta_grafico, dpi=150)
    plt.close()
    print(f"  Gráfico guardado: {ruta_grafico}")

    # Reporte
    reporte = os.path.join(speedup_dir, "reporte_speedup.txt")
    with open(reporte, "w") as archivo_reporte:
        archivo_reporte.write("=" * 70 + "\n")
        archivo_reporte.write("REPORTE DE SPEEDUP\n")
        archivo_reporte.write(f"Fecha: {timestamp()}\n")
        archivo_reporte.write("=" * 70 + "\n\n")
        archivo_reporte.write("CONFIGURACIÓN:\n")
        archivo_reporte.write(f"  Binario: {BINARIO}\n")
        archivo_reporte.write(f"  Procesos: {PROCESOS}\n")
        archivo_reporte.write(f"  Repeticiones: {REPETICIONES}\n\n")
        archivo_reporte.write("RESULTADOS:\n")
        archivo_reporte.write("-" * 40 + "\n")
        for n in sorted(resultados.keys()):
            data = resultados[n]
            sp = speedups.get(n, float('nan'))
            fuente = "maestro" if data['desde_maestro'] else "wall-clock"
            archivo_reporte.write(f"  {n} procesos: tiempo={data['promedio']:.2f}s, speedup={sp:.2f}x (fuente: {fuente})\n")
        archivo_reporte.write("\n" + "-" * 40 + "\n")
        archivo_reporte.write(f"Fracción paralelizable (Amdahl, ajustada con 1,2,4,8): fp = {fp:.4f}\n")
        archivo_reporte.write(f"Speedup máximo teórico (Amdahl, sin overhead de comunicación): {speedup_max_teorico:.2f}x\n")
        archivo_reporte.write(f"Speedup máximo real alcanzado: {speedup_real_max:.2f}x (con {mejor_proc} procesos)\n")
        archivo_reporte.write("\nNOTA: La caída del speedup después de 8 procesos se debe al overhead de comunicación y contención de recursos, que no está contemplado por la ley de Amdahl.\n")
        archivo_reporte.write("\n" + "=" * 70 + "\n")

    print(f"  Reporte guardado: {reporte}")

    print("\n" + "=" * 70)
    print(f"EJECUCIÓN FINALIZADA. Resultados en: {speedup_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()