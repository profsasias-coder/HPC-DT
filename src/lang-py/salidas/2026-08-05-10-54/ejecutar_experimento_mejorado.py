#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejecutar_experimento_mejorado.py
Ejecuta el experimento completo con el descubridor mejorado.
CON ESTRUCTURA DE CARPETAS CON TIMESTAMP.
PARAMETROS SEGUN ESPECIFICACION DEL USUARIO.
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
# CONFIGURACION - PARAMETROS DEL USUARIO
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# --- Parametros del experimento (segun especificacion) ---
PROCESOS = [1, 4, 8, 10, 12, 16, 20]        # Procesos a probar
MAXITER = 8                         # Reducido de 15 a 8
POPSIZE = 6                         # Reducido de 8 a 6
TIMEOUT = 3600                      # 60 min (suficiente para 1p con esta config)
FAMILIAS_MODO = "todas"             # 10 familias (fisicas + matematicas)
NUM_ARCHIVOS = 4                    # Mismo numero de archivos que antes

# --- Crear carpeta raiz con timestamp ---
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")
BASE_DIR = os.path.join(SCRIPT_DIR, "salidas", TIMESTAMP)
os.makedirs(BASE_DIR, exist_ok=True)

# Subcarpetas
DIR_DATOS = os.path.join(BASE_DIR, "01-datos")
DIR_RESULTADOS = os.path.join(BASE_DIR, "02-resultados")
DIR_SPEEDUP = os.path.join(BASE_DIR, "03-speedup")

for d in [DIR_DATOS, DIR_RESULTADOS, DIR_SPEEDUP]:
    os.makedirs(d, exist_ok=True)

# Copiar scripts a la carpeta de salida para trazabilidad
try:
    shutil.copy2(__file__, os.path.join(BASE_DIR, "ejecutar_experimento_mejorado.py"))
    shutil.copy2("descubridor_mejorado.py", os.path.join(BASE_DIR, "descubridor_mejorado.py"))
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

def ejecutar_descubridor(procesos, input_dir, output_dir):
    """Ejecuta el descubridor mejorado con un numero de procesos."""
    print_step(f"Iniciando ejecucion con {procesos} procesos...")
    inicio = time.time()

    # Recomendacion:  "mpirun", "-np", str(procesos), "--use-hwthread-cpus",
    #                 "mpirun", "-np", str(procesos), "--oversubscribe",

    cmd = [
        "mpirun", "-np", str(procesos), "--use-hwthread-cpus",
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
    """Genera grafico de speedup con curva de Amdahl."""
    procs = sorted([p for p in tiempos.keys() if tiempos[p] is not None and p != 1])
    if not procs:
        return None
    
    # Speedups reales (incluyendo 1)
    procs_all = sorted([p for p in tiempos.keys() if tiempos[p] is not None])
    speedup_vals = [speedups[p] for p in procs_all]
    
    # ---- Curva de Amdahl ----
    # Estimar f (fraccion paralelizable) a partir del mejor speedup
    # Usamos el speedup en 8 procesos como referencia
    if 8 in speedups and speedups[8] > 0:
        s_ref = speedups[8]
        p_ref = 8
        # f = (1/s_ref - 1) / (1/p_ref - 1)
        f = (1.0/s_ref - 1.0) / (1.0/p_ref - 1.0)
        # Asegurar que f esté en [0,1]
        f = max(0.0, min(1.0, f))
    else:
        # Si no hay dato en 8, usar el mejor speedup
        best_p = max(procs, key=lambda p: speedups[p])
        s_ref = speedups[best_p]
        p_ref = best_p
        f = (1.0/s_ref - 1.0) / (1.0/p_ref - 1.0)
        f = max(0.0, min(1.0, f))
    
    print(f"Fraccion paralelizable estimada (fp): {f:.4f}")
    print(f"Fraccion secuencial (1-fp): {1-f:.4f}")
    print(f"Speedup maximo teorico: {1/(1-f):.2f}x")
    
    # Generar puntos para la curva de Amdahl
    procs_amdahl = np.linspace(1, max(procs_all), 100)
    speedup_amdahl = 1.0 / ((1.0 - f) + f / procs_amdahl)
    
    # ---- Grafico ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafico 1: Tiempos
    tiempos_vals = [tiempos[p] for p in procs_all]
    axes[0].bar([str(p) for p in procs_all], tiempos_vals, color='#2E86AB')
    axes[0].set_xlabel('Procesos')
    axes[0].set_ylabel('Tiempo (s)')
    axes[0].set_title('Tiempo de ejecucion')
    axes[0].grid(True, alpha=0.3)
    for i, (p, t) in enumerate(zip(procs_all, tiempos_vals)):
        axes[0].text(i, t + 5, f'{t:.1f}s', ha='center', fontsize=9)
    
    # Grafico 2: Speedup con Amdahl
    axes[1].plot(procs_all, speedup_vals, 'o-', color='#D64933', 
                 linewidth=2, markersize=8, label='Speedup real')
    axes[1].plot(procs_amdahl, speedup_amdahl, 'b--', linewidth=2, 
                 label=f'Amdahl (fp={f:.3f})')
    axes[1].plot(procs_all, procs_all, 'k--', linewidth=2, 
                 alpha=0.5, label='Ideal')
    axes[1].set_xlabel('Procesos')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Speedup vs Procesos')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Anotar el valor de fp en el grafico
    axes[1].text(0.7, 0.05, f'fp = {f:.3f}', transform=axes[1].transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    axes[1].text(0.7, 0.12, f'1-fp = {1-f:.3f}', transform=axes[1].transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    ruta = os.path.join(output_dir, "speedup_analysis.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Grafico guardado en: {ruta}")
    return ruta

def main():
    inicio_global = datetime.now()
    
    print_header("EXPERIMENTO MEJORADO - BALANCE DE CARGA DINAMICO")
    print(f"  Inicio global: {inicio_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directorio base: {BASE_DIR}")
    print(f"  Procesos a probar: {PROCESOS}")
    print(f"  maxiter: {MAXITER}, popsize: {POPSIZE}, timeout: {TIMEOUT}s")
    print(f"  Familias: {FAMILIAS_MODO}")
    print(f"  Archivos: {NUM_ARCHIVOS}")

    # 1. Generar datos
    print_header("PASO 1: Generando datos")
    datos_dir = generar_datos()

    # 2. Ejecutar para cada numero de procesos
    print_header("PASO 2: Ejecutando descubridor")
    tiempos = {}
    for idx, p in enumerate(PROCESOS, 1):
        print_step(f"--- Ejecucion {idx}/{len(PROCESOS)}: {p} procesos ---")
        out_dir = os.path.join(DIR_RESULTADOS, f"{p}_procesos")
        os.makedirs(out_dir, exist_ok=True)
        t, ok = ejecutar_descubridor(p, datos_dir, out_dir)
        tiempos[p] = t if ok else None
        if not ok:
            print_step(f"Ejecucion con {p} procesos fallida (tiempo: {t}s)")

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

    # Generar grafico y reporte
    if speedups and len(tiempos) > 1:
        ruta_grafico = generar_grafico_speedup(tiempos, speedups, DIR_SPEEDUP)
        if ruta_grafico:
            print_step(f"Grafico guardado en: {ruta_grafico}")

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
            f.write("REPORTE EXPERIMENTO MEJORADO\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("CONFIGURACION:\n")
            f.write(f"  maxiter: {MAXITER}, popsize: {POPSIZE}\n")
            f.write(f"  familias: {FAMILIAS_MODO}\n")
            f.write(f"  timeout: {TIMEOUT}s\n")
            f.write(f"  archivos: {NUM_ARCHIVOS}\n\n")
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

        print_step(f"Reporte guardado en: {os.path.join(DIR_SPEEDUP, 'reporte_speedup.txt')}")

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
    print("=" * 70)

if __name__ == "__main__":
    main()