#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graficar_resultados.py
Lee los archivos JSON generados por el descubridor en C y genera gráficos.
1. Comparación: curvas reales vs modelo descubierto (arriba) y error puntual (abajo) en una misma figura.
2. Barras de error por familia (figura separada).
"""

import os
import sys
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def encontrar_ultima_salida(base_dir="salidas"):
    """Encuentra la carpeta de salida más reciente (con timestamp)."""
    if not os.path.exists(base_dir):
        return None
    carpetas = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not carpetas:
        return None
    carpetas.sort(reverse=True)
    return os.path.join(base_dir, carpetas[0])

def graficar_resultados(salida_dir=None):
    """
    Genera gráficos a partir de los JSON en la carpeta de salida.
    Si no se especifica salida_dir, busca la carpeta más reciente.
    """
    if salida_dir is None:
        salida_dir = encontrar_ultima_salida()
        if salida_dir is None:
            print("ERROR: No se encontró ninguna carpeta de salida en 'salidas/'")
            print("Asegúrate de haber ejecutado primero el descubridor en C.")
            sys.exit(1)
    
    json_dir = os.path.join(salida_dir, "01-json")
    graficos_dir = os.path.join(salida_dir, "02-graficos")
    
    if not os.path.exists(json_dir):
        print(f"ERROR: No se encontró la carpeta de JSON en {json_dir}")
        sys.exit(1)
    
    os.makedirs(graficos_dir, exist_ok=True)
    
    archivos_json = glob.glob(os.path.join(json_dir, "*.json"))
    if not archivos_json:
        print(f"No se encontraron archivos JSON en {json_dir}")
        sys.exit(1)
    
    print(f"Procesando {len(archivos_json)} archivos JSON...")
    
    for json_file in archivos_json:
        print(f"  Leyendo: {os.path.basename(json_file)}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"    ERROR: No se pudo leer el archivo: {e}")
            continue
        
        if "resultado" not in data:
            print("    ERROR: El JSON no contiene la clave 'resultado'")
            continue
        
        resultado = data["resultado"]
        
        archivo = resultado.get("archivo", "desconocido")
        mejor_familia = resultado.get("familia", "desconocida")
        mejor_error = resultado.get("error", float('nan'))
        ecuacion = resultado.get("ecuacion", "Ecuación no disponible")
        
        t = np.array(resultado.get("t", []))
        x_real = np.array(resultado.get("x_real", []))
        x_desc = np.array(resultado.get("x_sim", []))
        
        if len(t) == 0 or len(x_real) == 0 or len(x_desc) == 0:
            print("    ERROR: Datos incompletos (t, x_real o x_sim vacíos)")
            continue
        
        # Calcular error puntual
        error = x_real - x_desc
        max_error = np.max(np.abs(error))
        
        errores_por_familia = resultado.get("errores_por_familia", {})
        
        nombre_base = os.path.splitext(archivo)[0]
        
        # ---- FIGURA PRINCIPAL: Curvas (arriba) + Error puntual (abajo) ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gráfico superior: Datos reales vs Modelo descubierto
        ax1.plot(t, x_real, 'b-', label='Datos reales', linewidth=2)
        ax1.plot(t, x_desc, 'r--', label=f'Modelo descubierto ({mejor_familia})', linewidth=2)
        ax1.set_ylabel('Vibración x(t)')
        ax1.set_title(f'Comparación: {archivo}\nMejor modelo descubierto: {mejor_familia} (MSE = {mejor_error:.2e})\n{ecuacion}', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico inferior: Error puntual e(t) = real - descubierto
        ax2.plot(t, error, 'g-', linewidth=1.5, label='Error: real - modelo descubierto')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Error e(t)')
        ax2.set_title(f'Error puntual (MSE = {mejor_error:.2e},  Máx|error| = {max_error:.2e})', fontsize=10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        ruta_comparacion = os.path.join(graficos_dir, f"comparacion_{nombre_base}.png")
        fig.savefig(ruta_comparacion, dpi=150)
        plt.close(fig)
        print(f"    Gráfico combinado (curvas + error) guardado: {ruta_comparacion}")
        
        # ---- GRÁFICO DE BARRAS: Error por familia ----
        if errores_por_familia:
            nombres = list(errores_por_familia.keys())
            errores = list(errores_por_familia.values())
            
            # Ordenar por error (ascendente)
            pares = sorted(zip(nombres, errores), key=lambda x: x[1])
            nombres_ord, errores_ord = zip(*pares)
            
            fig2, ax3 = plt.subplots(figsize=(10, 6))
            bars = ax3.bar(nombres_ord, errores_ord, color='skyblue')
            if bars:
                bars[0].set_color('green')  # Resaltar el mejor
            ax3.set_yscale('log')
            ax3.set_xlabel('Familia de modelos')
            ax3.set_ylabel('Error (MSE) - escala logarítmica')
            ax3.set_title(f'Error por familia de modelos para {archivo}')
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Anotar el valor del error encima de cada barra
            for bar, err in zip(bars, errores_ord):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                         f'{err:.2e}', ha='center', va='bottom', fontsize=7, rotation=45)
            
            plt.tight_layout()
            ruta_errores = os.path.join(graficos_dir, f"errores_{nombre_base}.png")
            fig2.savefig(ruta_errores, dpi=150)
            plt.close(fig2)
            print(f"    Gráfico de errores por familia guardado: {ruta_errores}")
        else:
            print("    No se generó gráfico de errores por familia (datos no disponibles)")
    
    print(f"\n✅ Gráficos generados exitosamente en: {graficos_dir}")

if __name__ == "__main__":
    graficar_resultados()