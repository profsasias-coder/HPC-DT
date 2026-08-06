#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
descubridor_mejorado.py
Descubridor de ecuaciones con mejoras de rendimiento.
CON MENSAJES DE PROGRESO EN MODO SECUENCIAL.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import pandas as pd
import os
import time
import glob
import json
import sys
import argparse
from datetime import datetime
from mpi4py import MPI
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FUNCIONES DE TIMESTAMP
# ============================================================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_step(mensaje, rank=0):
    if rank == 0:
        print(f"[{timestamp()}] {mensaje}")

# ============================================================================
# 1. FAMILIAS DE MODELOS (FISICAS Y MATEMATICAS)
# ============================================================================

class FamiliaBase:
    def __init__(self, nombre, parametros, ecuacion_template, tipo='fisica'):
        self.nombre = nombre
        self.parametros = parametros
        self.ecuacion_template = ecuacion_template
        self.tipo = tipo

    def generar_ecuacion(self, params):
        if params is None:
            return f"{self.nombre}: No convergio"
        if isinstance(params, np.ndarray):
            params = params.tolist()
        eq = self.ecuacion_template
        for i, p in enumerate(self.parametros):
            eq = eq.replace(f'{{{p}}}', f'{params[i]:.4f}')
        return eq

    def evaluar(self, params, t, x_real, F):
        raise NotImplementedError

    def simular(self, params, t, F):
        raise NotImplementedError


# -------------------- FAMILIAS FISICAS --------------------
class FamiliaOsciladorSimple(FamiliaBase):
    def __init__(self):
        super().__init__("Oscilador_Simple", ['omega_n', 'zeta'],
                         "d2x/dt2 + 2*{zeta}*{omega_n}*dx/dt + {omega_n}^2*x = F(t)")

    def _sistema(self, t, y, omega_n, zeta, F, t_eval):
        idx = np.searchsorted(t_eval, t, side='right') - 1
        idx = max(0, min(idx, len(F)-1))
        F_curr = F[idx]
        x, v = y
        return [v, F_curr - 2*zeta*omega_n*v - omega_n**2*x]

    def evaluar(self, params, t, x_real, F):
        omega_n, zeta = params
        try:
            sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                            t_eval=t, args=(omega_n, zeta, F, t),
                            rtol=1e-4, atol=1e-6, method='RK45')
            x_sim = sol.y[0]
            return np.mean((x_sim - x_real)**2)
        except:
            return 1e9

    def simular(self, params, t, F):
        omega_n, zeta = params
        sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                        t_eval=t, args=(omega_n, zeta, F, t),
                        rtol=1e-4, atol=1e-6, method='RK45')
        return sol.y[0]


class FamiliaOsciladorDuffing(FamiliaBase):
    def __init__(self):
        super().__init__("Oscilador_Duffing", ['omega_n', 'zeta', 'beta'],
                         "d2x/dt2 + 2*{zeta}*{omega_n}*dx/dt + {omega_n}^2*x + {beta}*x^3 = F(t)")

    def _sistema(self, t, y, omega_n, zeta, beta, F, t_eval):
        idx = np.searchsorted(t_eval, t, side='right') - 1
        idx = max(0, min(idx, len(F)-1))
        F_curr = F[idx]
        x, v = y
        return [v, F_curr - 2*zeta*omega_n*v - omega_n**2*x - beta*x**3]

    def evaluar(self, params, t, x_real, F):
        omega_n, zeta, beta = params
        try:
            sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                            t_eval=t, args=(omega_n, zeta, beta, F, t),
                            rtol=1e-4, atol=1e-6, method='RK45')
            x_sim = sol.y[0]
            return np.mean((x_sim - x_real)**2)
        except:
            return 1e9

    def simular(self, params, t, F):
        omega_n, zeta, beta = params
        sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                        t_eval=t, args=(omega_n, zeta, beta, F, t),
                        rtol=1e-4, atol=1e-6, method='RK45')
        return sol.y[0]


class FamiliaAmortiguamientoNoLineal(FamiliaBase):
    def __init__(self):
        super().__init__("Amortiguamiento_NoLineal", ['omega_n', 'zeta', 'gamma'],
                         "d2x/dt2 + 2*{zeta}*{omega_n}*dx/dt + {omega_n}^2*x + {gamma}*(dx/dt)^3 = F(t)")

    def _sistema(self, t, y, omega_n, zeta, gamma, F, t_eval):
        idx = np.searchsorted(t_eval, t, side='right') - 1
        idx = max(0, min(idx, len(F)-1))
        F_curr = F[idx]
        x, v = y
        return [v, F_curr - 2*zeta*omega_n*v - omega_n**2*x - gamma*v**3]

    def evaluar(self, params, t, x_real, F):
        omega_n, zeta, gamma = params
        try:
            sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                            t_eval=t, args=(omega_n, zeta, gamma, F, t),
                            rtol=1e-4, atol=1e-6, method='RK45')
            x_sim = sol.y[0]
            return np.mean((x_sim - x_real)**2)
        except:
            return 1e9

    def simular(self, params, t, F):
        omega_n, zeta, gamma = params
        sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                        t_eval=t, args=(omega_n, zeta, gamma, F, t),
                        rtol=1e-4, atol=1e-6, method='RK45')
        return sol.y[0]


class FamiliaForzamientoParametrico(FamiliaBase):
    def __init__(self):
        super().__init__("Forzamiento_Parametrico", ['omega_n', 'zeta', 'epsilon', 'omega_p'],
                         "d2x/dt2 + 2*{zeta}*{omega_n}*dx/dt + {omega_n}^2*(1 + {epsilon}*sin({omega_p}*t))*x = F(t)")

    def _sistema(self, t, y, omega_n, zeta, epsilon, omega_p, F, t_eval):
        idx = np.searchsorted(t_eval, t, side='right') - 1
        idx = max(0, min(idx, len(F)-1))
        F_curr = F[idx]
        x, v = y
        return [v, F_curr - 2*zeta*omega_n*v - omega_n**2*(1 + epsilon*np.sin(omega_p*t))*x]

    def evaluar(self, params, t, x_real, F):
        omega_n, zeta, epsilon, omega_p = params
        try:
            sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                            t_eval=t, args=(omega_n, zeta, epsilon, omega_p, F, t),
                            rtol=1e-4, atol=1e-6, method='RK45')
            x_sim = sol.y[0]
            return np.mean((x_sim - x_real)**2)
        except:
            return 1e9

    def simular(self, params, t, F):
        omega_n, zeta, epsilon, omega_p = params
        sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                        t_eval=t, args=(omega_n, zeta, epsilon, omega_p, F, t),
                        rtol=1e-4, atol=1e-6, method='RK45')
        return sol.y[0]


class FamiliaImpacto(FamiliaBase):
    def __init__(self):
        super().__init__("Impacto", ['omega_n', 'zeta', 'gap', 'k_impacto'],
                         "d2x/dt2 + 2*{zeta}*{omega_n}*dx/dt + {omega_n}^2*x + F_impacto(x) = F(t)")

    def _sistema(self, t, y, omega_n, zeta, gap, k_impacto, F, t_eval):
        idx = np.searchsorted(t_eval, t, side='right') - 1
        idx = max(0, min(idx, len(F)-1))
        F_curr = F[idx]
        x, v = y
        if x > gap:
            F_impacto = -k_impacto * (x - gap)
        elif x < -gap:
            F_impacto = -k_impacto * (x + gap)
        else:
            F_impacto = 0.0
        return [v, F_curr - 2*zeta*omega_n*v - omega_n**2*x + F_impacto]

    def evaluar(self, params, t, x_real, F):
        omega_n, zeta, gap, k_impacto = params
        try:
            sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                            t_eval=t, args=(omega_n, zeta, gap, k_impacto, F, t),
                            rtol=1e-4, atol=1e-6, method='RK45')
            x_sim = sol.y[0]
            return np.mean((x_sim - x_real)**2)
        except:
            return 1e9

    def simular(self, params, t, F):
        omega_n, zeta, gap, k_impacto = params
        sol = solve_ivp(self._sistema, [t[0], t[-1]], [0, 0],
                        t_eval=t, args=(omega_n, zeta, gap, k_impacto, F, t),
                        rtol=1e-4, atol=1e-6, method='RK45')
        return sol.y[0]


# -------------------- FAMILIAS MATEMATICAS --------------------
class FamiliaPolinomica(FamiliaBase):
    def __init__(self, grado=3):
        self.grado = grado
        params = [f'a{i}' for i in range(grado+1)] + [f'b{i}' for i in range(grado+1)] + ['c']
        super().__init__(f"Polinomica_grado{grado}", params,
                         "d2x/dt2 = sum(a_i*x^i) + sum(b_i*(dx/dt)^i) + c*F(t)", tipo='matematica')

    def evaluar(self, params, t, x_real, F):
        n = self.grado + 1
        a = params[:n]
        b = params[n:2*n]
        c = params[-1]
        dx = np.gradient(x_real, t)
        X = np.vstack([x_real**i for i in range(n)]).T
        dX = np.vstack([dx**i for i in range(n)]).T
        d2x = np.gradient(dx, t)
        pred = X @ a + dX @ b + c * F
        return np.mean((d2x - pred)**2)

    def simular(self, params, t, F):
        return np.zeros_like(t)


class FamiliaFourier(FamiliaBase):
    def __init__(self, n_armonicos=3):
        self.n = n_armonicos
        params = [f'a{k}' for k in range(1, n_armonicos+1)] + \
                 [f'b{k}' for k in range(1, n_armonicos+1)] + ['omega', 'c']
        super().__init__(f"Fourier_{n_armonicos}", params,
                         "d2x/dt2 = sum(a_k*cos(k*omega*t) + b_k*sin(k*omega*t)) + c*F(t)", tipo='matematica')

    def evaluar(self, params, t, x_real, F):
        n = self.n
        a = params[:n]
        b = params[n:2*n]
        omega = params[-2]
        c = params[-1]
        dx = np.gradient(x_real, t)
        d2x = np.gradient(dx, t)
        pred = np.zeros_like(t)
        for k in range(1, n+1):
            pred += a[k-1]*np.cos(k*omega*t) + b[k-1]*np.sin(k*omega*t)
        pred += c * F
        return np.mean((d2x - pred)**2)

    def simular(self, params, t, F):
        return np.zeros_like(t)


class FamiliaExponencial(FamiliaBase):
    def __init__(self):
        super().__init__("Exponencial", ['A', 'alpha', 'beta', 'gamma'],
                         "d2x/dt2 = A*exp(alpha*x) + beta*dx/dt + gamma*F(t)", tipo='matematica')

    def evaluar(self, params, t, x_real, F):
        A, alpha, beta, gamma = params
        dx = np.gradient(x_real, t)
        d2x = np.gradient(dx, t)
        pred = A * np.exp(alpha * x_real) + beta * dx + gamma * F
        return np.mean((d2x - pred)**2)

    def simular(self, params, t, F):
        return np.zeros_like(t)


# ============================================================================
# 2. OPTIMIZACION DE PARAMETROS
# ============================================================================

def obtener_bounds(familia):
    bounds = []
    for p in familia.parametros:
        if p.startswith('omega_n'):
            bounds.append((5.0, 50.0))
        elif p.startswith('zeta'):
            bounds.append((0.001, 0.5))
        elif p.startswith('beta'):
            bounds.append((-10.0, 10.0))
        elif p.startswith('gamma'):
            bounds.append((-10.0, 10.0))
        elif p.startswith('epsilon'):
            bounds.append((0.0, 1.0))
        elif p.startswith('omega_p'):
            bounds.append((0.1, 10.0))
        elif p.startswith('gap'):
            bounds.append((0.001, 1.0))
        elif p.startswith('k_impacto'):
            bounds.append((10.0, 1000.0))
        elif p.startswith('A') or p.startswith('a'):
            bounds.append((-100.0, 100.0))
        elif p.startswith('b'):
            bounds.append((-100.0, 100.0))
        elif p.startswith('c') or p.startswith('C'):
            bounds.append((-100.0, 100.0))
        elif p.startswith('omega'):
            bounds.append((0.1, 50.0))
        else:
            bounds.append((-100.0, 100.0))
    return bounds


def optimizar_familia(familia, t, x_real, F, maxiter=15, popsize=8):
    bounds = obtener_bounds(familia)

    def objective(params):
        return familia.evaluar(params, t, x_real, F)

    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter,
                                        popsize=popsize, disp=False, seed=42)
        mejores_params = result.x
        error = result.fun
        res = minimize(objective, mejores_params, method='Nelder-Mead',
                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success and res.fun < error:
            mejores_params = res.x
            error = res.fun
        return mejores_params, error
    except Exception:
        return None, 1e9


# ============================================================================
# 3. GRAFICOS DE COMPARACION
# ============================================================================

def generar_grafico_comparacion(t, x_real, x_sim, titulo, falla, severidad, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, x_real, 'b-', label='Datos reales', linewidth=1.5)
    axes[0].plot(t, x_sim, 'r--', label='Modelo descubierto', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Vibracion x(t)')
    axes[0].set_title(f'{titulo} - Comparacion Real vs Modelo')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    error = x_sim - x_real
    axes[1].plot(t, error, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Error (Modelo - Real)')
    axes[1].set_title(f'Error de ajuste (MSE = {np.mean(error**2):.2e})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"comparacion_{falla}_{severidad}_{timestamp}.png"
    ruta = os.path.join(output_dir, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta


# ============================================================================
# 4. PROGRAMA PRINCIPAL CON BALANCE DE CARGA DINAMICO Y PROGRESO VISUAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Descubridor mejorado')
    parser.add_argument('--input_dir', type=str, default='./datos',
                        help='Directorio con los datos de entrada')
    parser.add_argument('--output_dir', type=str, default='./resultados',
                        help='Directorio de salida')
    parser.add_argument('--familias', type=str, default='fisicas',
                        choices=['fisicas', 'matematicas', 'todas'],
                        help='Tipo de familias a usar')
    parser.add_argument('--maxiter', type=int, default=15,
                        help='Maximo de iteraciones para evolucion diferencial')
    parser.add_argument('--popsize', type=int, default=8,
                        help='Tamaño de la poblacion')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout por tarea en segundos')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print(f"[{timestamp()}] DESCUBRIDOR MEJORADO - BALANCE DE CARGA DINAMICO")
        print(f"Procesos MPI: {size}")
        print(f"Directorio entrada: {args.input_dir}")
        print(f"Directorio salida: {args.output_dir}")
        print(f"Familias: {args.familias}")
        print(f"maxiter: {args.maxiter}, popsize: {args.popsize}")
        print("=" * 70)

    # Seleccionar familias
    familias_fisicas = [
        FamiliaOsciladorSimple(),
        FamiliaOsciladorDuffing(),
        FamiliaAmortiguamientoNoLineal(),
        FamiliaForzamientoParametrico(),
        FamiliaImpacto()
    ]
    familias_matematicas = [
        FamiliaPolinomica(grado=2),
        FamiliaPolinomica(grado=3),
        FamiliaFourier(n_armonicos=2),
        FamiliaFourier(n_armonicos=3),
        FamiliaExponencial()
    ]

    if args.familias == 'fisicas':
        familias = familias_fisicas
    elif args.familias == 'matematicas':
        familias = familias_matematicas
    else:
        familias = familias_fisicas + familias_matematicas

    try:
        if rank == 0:
            archivos = glob.glob(os.path.join(args.input_dir, "vibracion_*.csv"))
            if not archivos:
                print(f"[{timestamp()}] ERROR: No hay archivos de vibracion en el directorio de entrada.")
                sys.exit(1)

            print(f"[{timestamp()}] Archivos: {len(archivos)}")
            print(f"[{timestamp()}] Familias: {len(familias)}")
            print(f"[{timestamp()}] Total tareas: {len(archivos) * len(familias)}")

            datos_archivos = []
            for archivo in archivos:
                df = pd.read_csv(archivo)
                datos_archivos.append({
                    'archivo': os.path.basename(archivo),
                    't': df['t'].values,
                    'x': df['x'].values,
                    'F': df['F'].values,
                    'falla': df['falla'].iloc[0] if 'falla' in df.columns else 'desconocida',
                    'severidad': df['severidad'].iloc[0] if 'severidad' in df.columns else 0.0
                })

            tareas = []
            for datos in datos_archivos:
                for familia in familias:
                    tareas.append((datos, familia))

            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "graficos"), exist_ok=True)

            resultados_totales = []
            graficos_generados = []

            # ===== MODO SECUENCIAL (1 proceso) =====
            if size == 1:
                print(f"[{timestamp()}] >>> EJECUTANDO EN MODO SECUENCIAL (1 PROCESO) <<<\n")
                tiempo_inicio_total = time.time()
                total_tareas = len(tareas)
                tarea_idx = 0

                for datos, familia in tareas:
                    tarea_idx += 1
                    archivo = datos['archivo']
                    nombre_familia = familia.nombre
                    print(f"[{timestamp()}] Tarea {tarea_idx}/{total_tareas}: {archivo} - {nombre_familia}")

                    t = datos['t']
                    x_real = datos['x']
                    F = datos['F']
                    params, error = optimizar_familia(familia, t, x_real, F,
                                                      maxiter=args.maxiter,
                                                      popsize=args.popsize)
                    resultados_totales.append({
                        'archivo': archivo,
                        'familia': familia.nombre,
                        'params': params.tolist() if params is not None else None,
                        'error': error,
                        'ecuacion': familia.generar_ecuacion(params)
                    })

                tiempo_total = time.time() - tiempo_inicio_total
                print(f"[{timestamp()}] Todas las tareas completadas en {tiempo_total:.2f}s")

            # ===== MODO PARALELO (varios procesos) =====
            else:
                print(f"[{timestamp()}] >>> EJECUTANDO EN MODO PARALELO ({size} PROCESOS) <<<\n")
                tareas_pendientes = list(tareas)
                workers_ocupados = {}
                tiempo_inicio_total = time.time()

                for worker_rank in range(1, min(size, len(tareas_pendientes) + 1)):
                    tarea = tareas_pendientes.pop(0)
                    comm.send(tarea, dest=worker_rank, tag=10)
                    workers_ocupados[worker_rank] = tarea
                    print(f"[{timestamp()}] Tarea enviada al worker {worker_rank}")

                while tareas_pendientes or workers_ocupados:
                    status = MPI.Status()
                    resultado = comm.recv(source=MPI.ANY_SOURCE, tag=11, status=status)
                    worker_rank = status.Get_source()
                    datos, familia, params, error = resultado
                    resultados_totales.append({
                        'archivo': datos['archivo'],
                        'familia': familia.nombre,
                        'params': params.tolist() if params is not None else None,
                        'error': error,
                        'ecuacion': familia.generar_ecuacion(params)
                    })

                    if tareas_pendientes:
                        nueva_tarea = tareas_pendientes.pop(0)
                        comm.send(nueva_tarea, dest=worker_rank, tag=10)
                        workers_ocupados[worker_rank] = nueva_tarea
                        print(f"[{timestamp()}] Worker {worker_rank} recibe nueva tarea. Pendientes: {len(tareas_pendientes)}")
                    else:
                        comm.send(None, dest=worker_rank, tag=99)
                        del workers_ocupados[worker_rank]
                        print(f"[{timestamp()}] Worker {worker_rank} finalizado")

                tiempo_total = time.time() - tiempo_inicio_total
                print(f"[{timestamp()}] Todas las tareas completadas en {tiempo_total:.2f}s")

            # ===== GENERAR GRAFICOS Y GUARDAR RESULTADOS =====
            print(f"[{timestamp()}] Generando graficos de comparacion...")
            for datos in datos_archivos:
                archivo_resultados = [r for r in resultados_totales if r['archivo'] == datos['archivo']]
                mejor = min(archivo_resultados, key=lambda x: x['error'])
                try:
                    mejor_familia = None
                    for f in familias:
                        if f.nombre == mejor['familia']:
                            mejor_familia = f
                            break
                    if mejor_familia and mejor['params'] is not None:
                        x_sim = mejor_familia.simular(mejor['params'], datos['t'], datos['F'])
                        ruta = generar_grafico_comparacion(
                            datos['t'], datos['x'], x_sim,
                            f"{datos['archivo']} - {mejor['familia']}",
                            datos['falla'], datos['severidad'],
                            os.path.join(args.output_dir, "graficos")
                        )
                        graficos_generados.append(ruta)
                except Exception as e:
                    print(f"[{timestamp()}] Error generando grafico para {datos['archivo']}: {e}")

            print(f"[{timestamp()}] Guardando resultados...")
            with open(os.path.join(args.output_dir, "resultados.txt"), "w") as f:
                f.write("=" * 70 + "\n")
                f.write("RESULTADOS POR ARCHIVO\n")
                f.write("=" * 70 + "\n\n")
                for archivo in archivos:
                    nombre_archivo = os.path.basename(archivo)
                    f.write(f"\n{nombre_archivo}:\n")
                    archivo_resultados = [r for r in resultados_totales if r['archivo'] == nombre_archivo]
                    mejor = min(archivo_resultados, key=lambda x: x['error'])
                    f.write(f"  MEJOR FAMILIA: {mejor['familia']}\n")
                    f.write(f"  ERROR: {mejor['error']:.4e}\n")
                    f.write(f"  ECUACION:\n")
                    f.write(f"    {mejor['ecuacion']}\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("GRAFICOS GENERADOS\n")
                f.write("=" * 70 + "\n")
                for g in graficos_generados:
                    f.write(f"  {g}\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("TIEMPOS DE EJECUCION\n")
                f.write("=" * 70 + "\n")
                f.write(f"Tiempo total: {tiempo_total:.2f} s\n")
                f.write(f"Procesos MPI: {size}\n")

            log_data = {
                'procesos': size,
                'fecha': datetime.now().isoformat(),
                'num_archivos': len(archivos),
                'num_familias': len(familias),
                'tiempo_total': tiempo_total,
                'resultados': resultados_totales
            }
            with open(os.path.join(args.output_dir, "log_speedup.json"), "w") as f:
                json.dump(log_data, f, indent=2)

            print(f"[{timestamp()}] Resultados guardados en: {args.output_dir}")

        else:
            # WORKER
            while True:
                tarea = comm.recv(source=0, tag=MPI.ANY_TAG)
                if tarea is None:
                    break
                datos, familia = tarea
                t = datos['t']
                x_real = datos['x']
                F = datos['F']
                params, error = optimizar_familia(familia, t, x_real, F,
                                                  maxiter=args.maxiter,
                                                  popsize=args.popsize)
                resultado = (datos, familia, params, error)
                comm.send(resultado, dest=0, tag=11)

    except Exception as e:
        if rank == 0:
            print(f"[{timestamp()}] ERROR: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

    comm.Barrier()
    if rank == 0:
        print(f"[{timestamp()}] Ejecucion finalizada.")


if __name__ == "__main__":
    main()