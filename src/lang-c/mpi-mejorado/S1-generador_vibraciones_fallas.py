#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generador_vibraciones_fallas.py
Genera datos de vibraciones con mayor carga (más puntos).
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
import argparse
from datetime import datetime

def generar_vibracion_falla(
    tipo_falla="saludable",
    t_total=10.0,          # Aumentado de 2.0 a 10.0
    dt=0.0005,             # Reducido de 0.001 a 0.0005
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

def guardar_datos(t, x, F, tipo_falla, severidad, output_dir, idx):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"vibracion_{tipo_falla}_{severidad}_{idx}_{timestamp}.csv"
    ruta = os.path.join(output_dir, nombre)
    
    df = pd.DataFrame({
        't': t,
        'x': x,
        'F': F,
        'falla': tipo_falla,
        'severidad': severidad
    })
    df.to_csv(ruta, index=False, float_format='%.10f')
    return ruta

def main():
    parser = argparse.ArgumentParser(description='Generador de vibraciones con fallas (carga pesada)')
    parser.add_argument('--output_dir', type=str, default='./datos_pesados')
    parser.add_argument('--num_archivos', type=int, default=64)
    parser.add_argument('--t_total', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.0005)
    args = parser.parse_args()
    
    print("="*70)
    print("GENERADOR DE VIBRACIONES CON FALLAS (CARGA PESADA)")
    print(f"Directorio: {args.output_dir}")
    print(f"Archivos: {args.num_archivos}")
    print(f"t_total={args.t_total}s, dt={args.dt}s => {int(args.t_total/args.dt)} puntos/archivo")
    print("="*70)
    
    tipos = ["saludable", "desgaste_rodamiento", "desbalanceo", "falla_engranaje", "holgura"]
    severidades = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    archivos = []
    idx = 0
    i = 0
    while len(archivos) < args.num_archivos:
        tipo = tipos[i % len(tipos)]
        severidad = severidades[(i // len(tipos)) % len(severidades)]
        seed = 42 + i * 7
        t, x, F, _, _ = generar_vibracion_falla(
            tipo, t_total=args.t_total, dt=args.dt,
            severidad=severidad, seed=seed
        )
        ruta = guardar_datos(t, x, F, tipo, severidad, args.output_dir, idx)
        archivos.append(ruta)
        print(f"Generado: {tipo} (sev={severidad}) -> {os.path.basename(ruta)}")
        idx += 1
        i += 1
    
    with open(os.path.join(args.output_dir, "archivos_generados.txt"), "w") as f:
        for a in archivos:
            f.write(f"{a}\n")
    
    print(f"\n✅ Datos generados en: {args.output_dir}")

if __name__ == "__main__":
    main()