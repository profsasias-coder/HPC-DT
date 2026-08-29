#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoloGraficaSpeedUp.py
Genera gráficos de speedup a partir de un archivo CSV existente.
Uso: python3 SoloGraficaSpeedUp.py --csv speedup_data.csv [--output_dir ./]

python3 SoloGraficaSpeedUp.py --csv speedup_data.csv --output_dir ./resultados
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# FUNCIONES
# ============================================================
def ajustar_amdahl(procs, speedups, max_proc_opt=None):
    if hasattr(procs, 'tolist'):
        procs = procs.tolist()
    if hasattr(speedups, 'tolist'):
        speedups = speedups.tolist()
    
    if max_proc_opt is None:
        max_idx = np.argmax(speedups)
        max_proc_opt = procs[max_idx]
        procs_fit = []
        speedups_fit = []
        for p, s in zip(procs, speedups):
            if p <= max_proc_opt:
                procs_fit.append(p)
                speedups_fit.append(s)
            else:
                break
    else:
        procs_fit = [p for p, s in zip(procs, speedups) if p <= max_proc_opt]
        speedups_fit = [s for p, s in zip(procs, speedups) if p <= max_proc_opt]
    
    if len(procs_fit) < 2:
        print("Advertencia: no hay suficientes puntos para ajustar Amdahl, usando fp=0.9")
        return 0.9, np.linspace(1, max(procs), 200), None
    
    def amdahl(p, fp):
        return 1.0 / ((1.0 - fp) + fp / p)
    
    try:
        popt, _ = curve_fit(amdahl, procs_fit, speedups_fit, p0=[0.9], bounds=(0.0, 1.0))
        fp = popt[0]
    except Exception as e:
        print(f"Advertencia: falló el ajuste de Amdahl ({e}), usando fp=0.9")
        fp = 0.9
    
    p_amdahl = np.linspace(1, max(procs), 200)
    s_amdahl = amdahl(p_amdahl, fp)
    return fp, p_amdahl, s_amdahl

def generar_graficos(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    df = df.sort_values('procesos')
    procs = df['procesos'].values
    tiempos = df['tiempo_promedio'].values
    speedups = df['speedup'].values
    if 'desviacion' in df.columns:
        desv = df['desviacion'].values
    else:
        desv = None
    
    fp, p_amdahl, s_amdahl = ajustar_amdahl(procs, speedups)
    speedup_max_teorico = 1.0 / (1.0 - fp) if fp < 1.0 else float('inf')
    max_idx = np.argmax(speedups)
    max_proc = procs[max_idx]
    
    # ---- GRÁFICO 1: Barras de tiempos (individual) ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x_pos = range(len(procs))
    bars = ax1.bar(x_pos, tiempos, yerr=desv, color='#2E86AB', capsize=5)
    ax1.set_xlabel('Procesos MPI')
    ax1.set_ylabel('Tiempo (s)')
    ax1.set_title('Tiempo de ejecución vs procesos')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(p) for p in procs], rotation=0, ha='center')
    for i, (p, t) in enumerate(zip(procs, tiempos)):
        ax1.text(i, t + 0.05 * max(tiempos), f'{t:.1f}s', 
                 ha='center', va='bottom', fontsize=8, rotation=45)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'speedup_tiempos.png'), dpi=150)
    plt.close(fig1)
    print(f"  Gráfico de tiempos guardado: {os.path.join(output_dir, 'speedup_tiempos.png')}")
    
    # ---- GRÁFICO 2: Líneas de Speedup con Amdahl (individual) ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(procs, speedups, 'o-', color='#D64933', linewidth=2, markersize=8, label='Speedup real')
    ax2.plot(p_amdahl, s_amdahl, 'b--', linewidth=2, label=f'Amdahl (fp={fp:.3f})')
    ax2.plot(procs, procs, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax2.set_xlabel('Procesos MPI')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs procesos (con Amdahl)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(procs)
    ax2.set_xticklabels([str(p) for p in procs], rotation=0, ha='center')
    for p, s in zip(procs, speedups):
        ax2.text(p, s + 0.1, f'{s:.2f}x', ha='center', va='bottom', fontsize=8, rotation=45)
    ax2.text(0.7, 0.05, f'fp = {fp:.3f}', transform=ax2.transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    ax2.text(0.7, 0.12, f'S_max teórico = {speedup_max_teorico:.2f}x', transform=ax2.transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    if max_proc < max(procs):
        ax2.axvline(x=max_proc, color='red', linestyle=':', alpha=0.5, label=f'Punto óptimo ({max_proc} proc)')
        ax2.text(0.05, 0.85, 'Caída por overhead de comunicación\ny contención de recursos',
                 transform=ax2.transAxes, fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'speedup_amdahl.png'), dpi=150)
    plt.close(fig2)
    print(f"  Gráfico de speedup guardado: {os.path.join(output_dir, 'speedup_amdahl.png')}")
    
    # ---- GRÁFICO 3: COMBINADO (sin compartir ejes, cada uno con su propia escala) ----
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    
    # Subplot superior: Barras de tiempos
    x_pos2 = range(len(procs))
    bars2 = ax3.bar(x_pos2, tiempos, yerr=desv, color='#2E86AB', capsize=5)
    ax3.set_ylabel('Tiempo (s)')
    ax3.set_title('Tiempo de ejecución vs procesos')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(x_pos2)
    ax3.set_xticklabels([str(p) for p in procs], rotation=0, ha='center')
    for i, (p, t) in enumerate(zip(procs, tiempos)):
        ax3.text(i, t + 0.05 * max(tiempos), f'{t:.1f}s', 
                 ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Subplot inferior: Speedup con Amdahl
    ax4.plot(procs, speedups, 'o-', color='#D64933', linewidth=2, markersize=8, label='Speedup real')
    ax4.plot(p_amdahl, s_amdahl, 'b--', linewidth=2, label=f'Amdahl (fp={fp:.3f})')
    ax4.plot(procs, procs, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax4.set_xlabel('Procesos MPI')
    ax4.set_ylabel('Speedup')
    ax4.set_title('Speedup vs procesos (con Amdahl)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    # Usar valores reales de procesos para las marcas del eje X
    ax4.set_xticks(procs)
    ax4.set_xticklabels([str(p) for p in procs], rotation=0, ha='center')
    for p, s in zip(procs, speedups):
        ax4.text(p, s + 0.1, f'{s:.2f}x', ha='center', va='bottom', fontsize=8, rotation=45)
    ax4.text(0.7, 0.05, f'fp = {fp:.3f}', transform=ax4.transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    ax4.text(0.7, 0.12, f'S_max teórico = {speedup_max_teorico:.2f}x', transform=ax4.transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    if max_proc < max(procs):
        ax4.axvline(x=max_proc, color='red', linestyle=':', alpha=0.5, label=f'Punto óptimo ({max_proc} proc)')
        ax4.text(0.05, 0.85, 'Caída por overhead de comunicación\ny contención de recursos',
                 transform=ax4.transAxes, fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Ajustar espaciado entre subplots
    plt.subplots_adjust(hspace=0.4)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'speedup_combinado.png'), dpi=150)
    plt.close(fig3)
    print(f"  Gráfico combinado guardado: {os.path.join(output_dir, 'speedup_combinado.png')}")

def main():
    parser = argparse.ArgumentParser(description='Genera gráficos de speedup desde CSV.')
    parser.add_argument('--csv', type=str, required=True, help='Ruta al archivo speedup_data.csv')
    parser.add_argument('--output_dir', type=str, default='.', help='Directorio de salida para los gráficos')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"ERROR: No se encontró el archivo {args.csv}")
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    print(f"CSV cargado: {args.csv}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Filas: {len(df)}")
    
    required = ['procesos', 'tiempo_promedio', 'speedup']
    for col in required:
        if col not in df.columns:
            print(f"ERROR: El CSV debe contener la columna '{col}'")
            sys.exit(1)
    
    if 'desviacion' not in df.columns:
        df['desviacion'] = 0.0
    
    generar_graficos(df, args.output_dir)
    print(f"\n✅ Gráficos generados en: {args.output_dir}")

if __name__ == "__main__":
    main()