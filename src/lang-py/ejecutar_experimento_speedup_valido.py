#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejecutar_experimento_speedup_valido.py
Ejecuta experimento de speedup con 10 familias (fisicas + matematicas)
y configuracion reducida (maxiter=8, popsize=6) para que 1p termine en <60 min.
"""

import os
import sys
import subprocess
import time
import json
import shutil
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuracion
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DIR_DATOS = os.path.join(SCRIPT_DIR, "datos_experimento_valido")
DIR_RESULTADOS = os.path.join(SCRIPT_DIR, "resultados_experimento_valido")
os.makedirs(DIR_DATOS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# ============================================================================
# CONFIGURACION PARA SPEEDUP VALIDO (10 familias, carga reducida)
# ============================================================================

PROCESOS = [1, 4, 8, 10, 12]        # 1, 4 y 8 procesos
MAXITER = 8                 # Reducido de 15 a 8
POPSIZE = 6                 # Reducido de 8 a 6
TIMEOUT = 3600              # 60 min (suficiente para 1p con esta config)
FAMILIAS_MODO = "todas"     # 10 familias (fisicas + matematicas)
NUM_ARCHIVOS = 4            # Mismo numero de archivos que antes

# ============================================================================
# FUNCIONES CON TIMESTAMP
# ============================================================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_step(mensaje):
    print(f"[{timestamp()}] {mensaje}")

def print_header(mensaje):
    print("\n" + "=" * 70)
    print(f"[{timestamp()}] {mensaje}")
    print("=" * 70)

# ============================================================================

def generar_datos(num_archivos=4):
    """Genera datos usando generador_vibraciones_fallas.py"""
    print_step(f"Generando {num_archivos} archivos de datos...")
    cmd = [
        "python3", "generador_vibraciones_fallas.py",
        "--output_dir", DIR_DATOS
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_step(f"Datos generados en {DIR_DATOS}")
        return DIR_DATOS
    except Exception as e:
        print_step(f"Error generando datos: {e}")
        sys.exit(1)

def ejecutar_descubridor(procesos, input_dir, output_dir):
    """Ejecuta el descubridor mejorado con un numero de procesos."""
    print_step(f"Iniciando ejecucion con {procesos} procesos...")
    inicio = time.time()
    cmd = [
        "mpirun", "-np", str(procesos),
        "python3", "descubridor_mejorado.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--familias", FAMILIAS_MODO,
        "--maxiter", str(MAXITER),
        "--popsize", str(POPSIZE),
        "--timeout", str(TIMEOUT)
    ]
    try:
        subprocess.run(cmd, check=True, timeout=TIMEOUT)
        exito = True
        tiempo = time.time() - inicio
        print_step(f"Completado con {procesos} procesos en {tiempo:.2f}s")
    except subprocess.TimeoutExpired:
        tiempo = TIMEOUT
        exito = False
        print_step(f"TIMEOUT con {procesos} procesos despues de {TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        tiempo = time.time() - inicio
        exito = False
        print_step(f"ERROR con {procesos} procesos (codigo {e.returncode})")
    return tiempo, exito

def calcular_speedup(tiempos):
    """Calcula speedup a partir de tiempos. Solo si 1p existe y es valido."""
    if 1 not in tiempos or tiempos[1] is None or tiempos[1] <= 0:
        print("ERROR: No se dispone de tiempo secuencial valido para calcular speedup.")
        return None
    t1 = tiempos[1]
    speedups = {}
    for p, t in tiempos.items():
        if p == 1:
            speedups[p] = 1.0
        else:
            speedups[p] = t1 / t if t and t > 0 else float('inf')
    return speedups

def generar_grafico_speedup(tiempos, speedups, output_dir):
    """Genera grafico de speedup."""
    procs = sorted([p for p in tiempos.keys() if tiempos[p] is not None])
    tiempos_vals = [tiempos[p] for p in procs]
    speedup_vals = [speedups[p] for p in procs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([str(p) for p in procs], tiempos_vals, color='#2E86AB')
    axes[0].set_xlabel('Procesos')
    axes[0].set_ylabel('Tiempo (s)')
    axes[0].set_title('Tiempo de ejecucion')
    axes[0].grid(True, alpha=0.3)
    for x, y in zip(procs, tiempos_vals):
        axes[0].text(str(x), y + 5, f'{y:.1f}s', ha='center', fontsize=9)

    axes[1].plot(procs, speedup_vals, marker='o', color='#D64933', linewidth=2, label='Speedup real')
    axes[1].plot(procs, procs, 'k--', label='Ideal', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Procesos')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Speedup')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    for x, y in zip(procs, speedup_vals):
        axes[1].text(x, y + 0.2, f'{y:.2f}x', ha='center', fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(output_dir, "speedup_analysis.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta

def main():
    inicio_global = datetime.now()
    print_header("EXPERIMENTO DE SPEEDUP VALIDO - 10 FAMILIAS")
    print(f"Inicio global: {inicio_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Procesos a probar: {PROCESOS}")
    print(f"Familias: {FAMILIAS_MODO} (10 familias)")
    print(f"maxiter: {MAXITER}, popsize: {POPSIZE}, timeout: {TIMEOUT}s")
    print(f"Archivos: {NUM_ARCHIVOS}")

    # 1. Generar datos
    print_header("PASO 1: Generando datos")
    datos_dir = generar_datos(NUM_ARCHIVOS)

    # 2. Ejecutar para cada numero de procesos
    print_header("PASO 2: Ejecutando descubridor")
    tiempos = {}
    for idx, p in enumerate(PROCESOS, 1):
        print_step(f"--- Ejecucion {idx}/{len(PROCESOS)}: {p} procesos ---")
        out_dir = os.path.join(DIR_RESULTADOS, f"procesos_{p}")
        os.makedirs(out_dir, exist_ok=True)
        t, ok = ejecutar_descubridor(p, datos_dir, out_dir)
        tiempos[p] = t if ok else None
        if not ok:
            print_step(f"Ejecucion con {p} procesos fallida")

    # 3. Analizar resultados
    print_header("PASO 3: Analizando resultados")
    speedups = calcular_speedup(tiempos)

    # Mostrar resultados
    print("\n" + "-" * 40)
    print("RESULTADOS")
    print("-" * 40)
    for p in sorted(tiempos.keys()):
        if tiempos[p] is not None:
            print(f"  {p} procesos: {tiempos[p]:.2f}s")
        else:
            print(f"  {p} procesos: FALLIDO")
    if speedups:
        print("\nSPEEDUP:")
        for p in sorted(speedups.keys()):
            if p == 1:
                print(f"  {p} proceso: 1.00x")
            else:
                print(f"  {p} procesos: {speedups[p]:.2f}x")
    else:
        print("\nNo se pudo calcular speedup (falta tiempo secuencial)")

    # Generar grafico y reporte
    if speedups and len(tiempos) > 1:
        ruta_grafico = generar_grafico_speedup(tiempos, speedups, DIR_RESULTADOS)
        print_step(f"Grafico guardado en: {ruta_grafico}")

        # Guardar CSV
        df = pd.DataFrame({
            'procesos': list(tiempos.keys()),
            'tiempo': list(tiempos.values()),
            'speedup': [speedups.get(p, None) for p in tiempos.keys()]
        })
        df.to_csv(os.path.join(DIR_RESULTADOS, "speedup_data.csv"), index=False)

        # Reporte
        with open(os.path.join(DIR_RESULTADOS, "reporte.txt"), "w") as f:
            f.write("=" * 70 + "\n")
            f.write("REPORTE SPEEDUP VALIDO - 10 FAMILIAS\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("CONFIGURACION:\n")
            f.write(f"  Familias: {FAMILIAS_MODO}\n")
            f.write(f"  maxiter: {MAXITER}, popsize: {POPSIZE}\n")
            f.write(f"  Archivos: {NUM_ARCHIVOS}\n\n")
            f.write("TIEMPOS:\n")
            for p in sorted(tiempos.keys()):
                f.write(f"  {p} procesos: {tiempos[p]:.2f}s\n")
            if speedups:
                f.write("\nSPEEDUP:\n")
                for p in sorted(speedups.keys()):
                    if p == 1:
                        f.write(f"  {p} proceso: 1.00x\n")
                    else:
                        f.write(f"  {p} procesos: {speedups[p]:.2f}x\n")
            f.write("\n" + "=" * 70 + "\n")

        print_step(f"Reporte guardado en: {os.path.join(DIR_RESULTADOS, 'reporte.txt')}")

    fin_global = datetime.now()
    duracion = (fin_global - inicio_global).total_seconds()
    horas = int(duracion // 3600)
    minutos = int((duracion % 3600) // 60)
    segundos = int(duracion % 60)

    print_header("EXPERIMENTO FINALIZADO")
    print(f"Inicio: {inicio_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fin:    {fin_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duracion total: {horas}h {minutos}m {segundos}s")
    print("=" * 70)

if __name__ == "__main__":
    main()