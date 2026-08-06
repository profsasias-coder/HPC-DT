#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generador_vibraciones_fallas.py
Genera datos de vibraciones de aerogeneradores con diferentes fallas.
Acepta argumentos para especificar directorio de salida.
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
import sys
import argparse
from datetime import datetime

def generar_vibracion_falla(
    tipo_falla="saludable",
    t_total=2.0,
    dt=0.001,
    severidad=0.3,
    rpm=15.0,
    seed=42,
    ruido_medicion=0.02
):
    np.random.seed(seed)
    t = np.arange(0, t_total, dt)
    n = len(t)
    
    f_rotor = rpm / 60.0
    omega_n = 2 * np.pi * f_rotor * 10
    zeta = 0.05
    
    F = np.zeros(n)
    
    if tipo_falla == "saludable":
        for arm in [1, 2, 3]:
            F += 0.5 * np.sin(2 * np.pi * f_rotor * arm * t)
            
    elif tipo_falla == "desgaste_rodamiento":
        f_rodamiento = 55.0
        F += severidad * np.sin(2 * np.pi * f_rodamiento * t)
        F += 0.3 * severidad * np.sin(2 * np.pi * 2 * f_rodamiento * t)
        F += 0.1 * np.sin(2 * np.pi * f_rotor * t) * np.sin(2 * np.pi * f_rodamiento * t)
        
    elif tipo_falla == "desbalanceo":
        F += severidad * np.sin(2 * np.pi * f_rotor * t)
        F += 0.3 * severidad * np.sin(2 * np.pi * 2 * f_rotor * t)
        
    elif tipo_falla == "falla_engranaje":
        for arm in [2, 3, 4, 5, 6, 7, 8]:
            amp = severidad / (arm - 1)
            F += amp * np.sin(2 * np.pi * f_rotor * arm * t)
            
    elif tipo_falla == "holgura":
        n_impactos = int(5 * severidad)
        for _ in range(n_impactos):
            t_impacto = np.random.uniform(0.5, t_total - 0.5)
            idx = int(t_impacto / dt)
            duracion = int(0.05 / dt)
            amp = np.random.uniform(0.5, 1.0) * severidad
            for j in range(duracion):
                if idx + j < n:
                    factor = np.exp(-(j / duracion * 5)**2)
                    F[idx + j] += amp * factor
    
    F += 0.01 * np.random.randn(n)
    
    def sistema(estado, t_idx):
        x, v = estado
        idx = int(t_idx / dt)
        idx = min(idx, len(F) - 1)
        F_curr = F[idx]
        dxdt = v
        dvdt = F_curr - 2*zeta*omega_n*v - omega_n**2 * x
        return [dxdt, dvdt]
    
    estado0 = [0.0, 0.0]
    sol = odeint(sistema, estado0, t, rtol=1e-6, atol=1e-8)
    x = sol[:, 0]
    x += ruido_medicion * np.std(x) * np.random.randn(n)
    
    return t, x, F, tipo_falla, severidad

def guardar_datos(t, x, F, tipo_falla, severidad, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"vibracion_{tipo_falla}_{timestamp}.csv"
    ruta = os.path.join(output_dir, nombre)
    
    df = pd.DataFrame({
        't': t,
        'x': x,
        'F': F,
        'falla': tipo_falla,
        'severidad': severidad
    })
    df.to_csv(ruta, index=False)
    return ruta

def main():
    parser = argparse.ArgumentParser(description='Generador de vibraciones con fallas')
    parser.add_argument('--output_dir', type=str, default='./datos',
                        help='Directorio de salida para los datos')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERADOR RAPIDO - VIBRACIONES CON FALLAS")
    print("=" * 70)
    print(f"Directorio de salida: {args.output_dir}")
    
    casos = [
        ("saludable", 0.0),
        ("desgaste_rodamiento", 0.5),
        ("desbalanceo", 0.5),
        ("falla_engranaje", 0.5),
    ]
    
    archivos = []
    for tipo, severidad in casos:
        t, x, F, _, _ = generar_vibracion_falla(tipo, t_total=2.0, severidad=severidad)
        ruta = guardar_datos(t, x, F, tipo, severidad, args.output_dir)
        archivos.append(ruta)
        print(f"Generado: {tipo} -> {ruta}")
    
    # Guardar lista de archivos
    with open(os.path.join(args.output_dir, "archivos_generados.txt"), "w") as f:
        for archivo in archivos:
            f.write(f"{archivo}\n")
    
    print(f"\nDatos generados correctamente en: {args.output_dir}")

if __name__ == "__main__":
    main()