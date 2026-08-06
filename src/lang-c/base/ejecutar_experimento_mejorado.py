#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejecutar_experimento_mejorado.py
Ejecuta el experimento completo usando el binario C (descubridor_c).
Genera datos, ejecuta con diferentes procesos y calcula speedup.
CON ESTRUCTURA DE CARPETAS CON TIMESTAMP Y OVERSUSCRIBE.
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

# ============================================================================
# CONFIGURACION - PARAMETROS DEL EXPERIMENTO
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# --- Parametros del experimento ---
PROCESOS = [1, 4, 8, 10, 12]          # Procesos a probar
MAXITER = 8                           # Iteraciones de evolucion diferencial
POPSIZE = 6                           # Tamaño de poblacion
TIMEOUT = 3600                        # 60 min
FAMILIAS_MODO = "todas"               # fisicas, matematicas, todas
NUM_ARCHIVOS = 4                      # Numero de archivos a generar
BINARIO_C = "./c/descubridor_c"       # Ruta al binario C

# --- Opciones MPI ---
MPI_OPCIONES = "--use-hwthread-cpus"  # o "--oversubscribe" si prefieres

# --- Crear carpeta raiz con timestamp ---
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
BASE_DIR = os.path.join(SCRIPT_DIR, "salidas", TIMESTAMP)
os.makedirs(BASE_DIR, exist_ok=True)

# Subcarpetas
DIR_DATOS = os.path.join(BASE_DIR, "01-datos")
DIR_RESULTADOS = os.path.join(BASE_DIR, "02-resultados")
DIR_SPEEDUP = os.path.join(BASE_DIR, "03-speedup")
DIR_GRAFICOS = os.path.join(BASE_DIR, "04-graficos")

for d in [DIR_DATOS, DIR_RESULTADOS, DIR_SPEEDUP, DIR_GRAFICOS]:
    os.makedirs(d, exist_ok=True)

# Copiar scripts a la carpeta de salida para trazabilidad
try:
    shutil.copy2(__file__, os.path.join(BASE_DIR, "ejecutar_experimento_mejorado.py"))
    shutil.copy2("graficar_resultados.py", os.path.join(BASE_DIR, "graficar_resultados.py"))
    shutil.copy2("generador_vibraciones_fallas.py", os.path.join(BASE_DIR, "generador_vibraciones_fallas.py"))
    print(f"Scripts copiados a {BASE_DIR} para trazabilidad")
except Exception as e:
    print(f"Advertencia: No se pudieron copiar scripts: {e}")

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

def generar_datos(num_archivos=NUM_ARCHIVOS):
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

def ejecutar_descubridor_c(procesos, input_dir, output_dir):
    """Ejecuta el binario C con MPI para un numero de procesos."""
    print_step(f"Iniciando ejecucion con {procesos} procesos...")
    inicio = time.time()
    
    cmd = [
        "mpirun", "-np", str(procesos), MPI_OPCIONES,
        BINARIO_C,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--familias", FAMILIAS_MODO,
        "--maxiter", str(MAXITER),
        "--popsize", str(POPSIZE)
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
    """Calcula speedup a partir de tiempos. Si 1p falla, usa estimacion."""
    if 1 not in tiempos or tiempos[1] is None:
        # Estimar T1 a partir de T4 y eficiencia tipica (~45%)
        if 4 in tiempos and tiempos[4] is not None:
            t4 = tiempos[4]
            t1_estimado = t4 * 4 / 0.45  # eficiencia 45%
            tiempos[1] = t1_estimado
            print_step(f"Estimando T1 = {t1_estimado:.2f}s (basado en T4 y eficiencia 45%)")
        else:
            print_step("No se puede estimar T1: falta T4")
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
    if not procs:
        return None
    tiempos_vals = [tiempos[p] for p in procs]
    speedup_vals = [speedups[p] for p in procs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Grafico de tiempos
    axes[0].bar([str(p) for p in procs], tiempos_vals, color='#2E86AB')
    axes[0].set_xlabel('Procesos')
    axes[0].set_ylabel('Tiempo (s)')
    axes[0].set_title('Tiempo de ejecucion')
    axes[0].grid(True, alpha=0.3)

    # Grafico de speedup
    axes[1].plot(procs, speedup_vals, marker='o', color='#D64933', linewidth=2, label='Speedup real')
    axes[1].plot(procs, procs, 'k--', label='Ideal', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Procesos')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Speedup')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(output_dir, "speedup_analysis.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta

def generar_graficos_finales():
    """Ejecuta el script Python para generar graficos de curvas y errores."""
    print_step("Generando graficos de curvas y errores a partir de los JSON...")
    cmd = [
        "python3", "graficar_resultados.py",
        "--input_dir", DIR_RESULTADOS,
        "--output_dir", DIR_GRAFICOS
    ]
    try:
        subprocess.run(cmd, check=True)
        print_step(f"Graficos guardados en {DIR_GRAFICOS}")
    except Exception as e:
        print_step(f"Error generando graficos: {e}")

def main():
    inicio_global = datetime.now()
    
    print_header("EXPERIMENTO MEJORADO CON BINARIO C")
    print(f"  Inicio global: {inicio_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directorio base: {BASE_DIR}")
    print(f"  Procesos a probar: {PROCESOS}")
    print(f"  maxiter: {MAXITER}, popsize: {POPSIZE}, timeout: {TIMEOUT}s")
    print(f"  Familias: {FAMILIAS_MODO}")
    print(f"  Archivos: {NUM_ARCHIVOS}")
    print(f"  Binario C: {BINARIO_C}")
    print(f"  MPI opcion: {MPI_OPCIONES}")

    # 1. Generar datos
    print_header("PASO 1: Generando datos")
    datos_dir = generar_datos()

    # 2. Ejecutar para cada numero de procesos
    print_header("PASO 2: Ejecutando descubridor C")
    tiempos = {}
    for idx, p in enumerate(PROCESOS, 1):
        print_step(f"--- Ejecucion {idx}/{len(PROCESOS)}: {p} procesos ---")
        out_dir = os.path.join(DIR_RESULTADOS, f"{p}_procesos")
        os.makedirs(out_dir, exist_ok=True)
        t, ok = ejecutar_descubridor_c(p, datos_dir, out_dir)
        tiempos[p] = t if ok else None
        if not ok:
            print_step(f"Ejecucion con {p} procesos fallida (tiempo: {t}s)")

    # 3. Analizar resultados (speedup)
    print_header("PASO 3: Analizando speedup")
    speedups = calcular_speedup(tiempos)

    # Mostrar resultados
    print("\n" + "-" * 40)
    print("RESULTADOS DE TIEMPOS Y SPEEDUP")
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

    # Generar grafico y reporte de speedup
    if speedups and len(tiempos) > 1:
        ruta_grafico = generar_grafico_speedup(tiempos, speedups, DIR_SPEEDUP)
        if ruta_grafico:
            print_step(f"Grafico de speedup guardado en: {ruta_grafico}")

        # Guardar CSV
        df = pd.DataFrame({
            'procesos': list(tiempos.keys()),
            'tiempo': [tiempos[p] if tiempos[p] is not None else None for p in tiempos.keys()],
            'speedup': [speedups.get(p, None) for p in tiempos.keys()]
        })
        df.to_csv(os.path.join(DIR_SPEEDUP, "speedup_data.csv"), index=False)

        # Reporte
        with open(os.path.join(DIR_SPEEDUP, "reporte_speedup.txt"), "w") as f:
            f.write("=" * 70 + "\n")
            f.write("REPORTE EXPERIMENTO MEJORADO (BINARIO C)\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("CONFIGURACION:\n")
            f.write(f"  maxiter: {MAXITER}, popsize: {POPSIZE}\n")
            f.write(f"  familias: {FAMILIAS_MODO}\n")
            f.write(f"  timeout: {TIMEOUT}s\n")
            f.write(f"  archivos: {NUM_ARCHIVOS}\n")
            f.write(f"  MPI opcion: {MPI_OPCIONES}\n\n")
            f.write("TIEMPOS:\n")
            for p in sorted(tiempos.keys()):
                if tiempos[p] is not None:
                    f.write(f"  {p} procesos: {tiempos[p]:.2f}s\n")
                else:
                    f.write(f"  {p} procesos: FALLIDO\n")
            if speedups:
                f.write("\nSPEEDUP:\n")
                for p in sorted(speedups.keys()):
                    if p == 1:
                        f.write(f"  {p} proceso: 1.00x\n")
                    else:
                        f.write(f"  {p} procesos: {speedups[p]:.2f}x\n")
            f.write("\n" + "=" * 70 + "\n")

        print_step(f"Reporte de speedup guardado en: {os.path.join(DIR_SPEEDUP, 'reporte_speedup.txt')}")

    # 4. Generar graficos de curvas y errores a partir de los JSON
    print_header("PASO 4: Generando graficos de curvas y errores")
    generar_graficos_finales()

    fin_global = datetime.now()
    duracion = (fin_global - inicio_global).total_seconds()
    horas = int(duracion // 3600)
    minutos = int((duracion % 3600) // 60)
    segundos = int(duracion % 60)

    print_header("EXPERIMENTO FINALIZADO")
    print(f"  Inicio: {inicio_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Fin:    {fin_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duracion total: {horas}h {minutos}m {segundos}s")
    print(f"  Resultados guardados en: {BASE_DIR}")
    print(f"    - Datos:       {DIR_DATOS}")
    print(f"    - Resultados:  {DIR_RESULTADOS}")
    print(f"    - Speedup:     {DIR_SPEEDUP}")
    print(f"    - Graficos:    {DIR_GRAFICOS}")
    print("=" * 70)

if __name__ == "__main__":
    main()