#include "utils.h"
#include "optimizador.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static double uniforme(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

double *evolucion_diferencial(EvolucionDiferencial *ed, double *mejor_error) {
    int N = ed->popsize;
    int D = ed->num_params;
    int maxiter = ed->maxiter;
    double *lb = ed->bounds_min;
    double *ub = ed->bounds_max;
    
    // Alocar poblacion (N individuos, cada uno con D parametros)
    double **poblacion = (double**)malloc(N * sizeof(double*));
    double **nuevos = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        poblacion[i] = (double*)malloc(D * sizeof(double));
        nuevos[i] = (double*)malloc(D * sizeof(double));
    }
    
    // Inicializar poblacion aleatoria
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            poblacion[i][j] = uniforme(lb[j], ub[j]);
        }
    }
    
    // Evaluar poblacion inicial
    double *fitness = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        fitness[i] = ed->funcion_objetivo(poblacion[i], ed->user_data);
    }
    
    // Encontrar el mejor
    int best_idx = 0;
    for (int i = 1; i < N; i++) {
        if (fitness[i] < fitness[best_idx]) best_idx = i;
    }

    if (fitness != NULL) print_step("    Poblacion inicial evaluada. Mejor error: %.6e", fitness[best_idx]);
    
    // Bucle principal
    for (int iter = 0; iter < maxiter; iter++) {
        for (int i = 0; i < N; i++) {
            // Seleccionar 3 individuos aleatorios distintos
            int r1, r2, r3;
            do { r1 = rand() % N; } while (r1 == i);
            do { r2 = rand() % N; } while (r2 == i || r2 == r1);
            do { r3 = rand() % N; } while (r3 == i || r3 == r1 || r3 == r2);
            
            double F = 0.8;  // factor de mutacion
            double CR = 0.9; // factor de cruce
            
            // Mutacion
            for (int j = 0; j < D; j++) {
                nuevos[i][j] = poblacion[r1][j] + F * (poblacion[r2][j] - poblacion[r3][j]);
                // Asegurar bounds
                if (nuevos[i][j] < lb[j]) nuevos[i][j] = lb[j];
                if (nuevos[i][j] > ub[j]) nuevos[i][j] = ub[j];
            }
            
            // Cruce
            int j_rand = rand() % D;
            for (int j = 0; j < D; j++) {
                if (j != j_rand && (double)rand()/RAND_MAX > CR) {
                    nuevos[i][j] = poblacion[i][j];
                }
            }
            
            // Seleccion
            double fitness_nuevo = ed->funcion_objetivo(nuevos[i], ed->user_data);
            if (fitness_nuevo <= fitness[i]) {
                memcpy(poblacion[i], nuevos[i], D * sizeof(double));
                fitness[i] = fitness_nuevo;
                if (fitness_nuevo < fitness[best_idx]) {
                    best_idx = i;
                }
            }
        }
	if (iter % 5 == 0 || iter == maxiter - 1) {
	    print_step("    Iteracion %d/%d - Mejor error: %.6e", iter+1, maxiter, fitness[best_idx]);
	}
    }
    
    // Mejor individuo
    double *mejor = (double*)malloc(D * sizeof(double));
    memcpy(mejor, poblacion[best_idx], D * sizeof(double));
    *mejor_error = fitness[best_idx];
    
    // Liberar memoria
    for (int i = 0; i < N; i++) {
        free(poblacion[i]);
        free(nuevos[i]);
    }
    free(poblacion); free(nuevos); free(fitness);
    
    return mejor;
}