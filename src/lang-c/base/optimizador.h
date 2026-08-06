#ifndef OPTIMIZADOR_H
#define OPTIMIZADOR_H

#include "familias.h"

typedef struct {
    int maxiter;
    int popsize;
    double *bounds_min;
    double *bounds_max;
    int num_params;
    double (*funcion_objetivo)(double *params, void *user_data);
    void *user_data;
} EvolucionDiferencial;

double *evolucion_diferencial(EvolucionDiferencial *ed, double *mejor_error);

#endif