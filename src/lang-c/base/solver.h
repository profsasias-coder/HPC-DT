#ifndef SOLVER_H
#define SOLVER_H

typedef struct {
    int n;              // numero de ecuaciones
    double t0, tf;      // tiempo inicial y final
    double dt;          // paso de tiempo
    double *y0;         // condiciones iniciales (n elementos)
    double *t_out;      // vector de tiempo (salida)
    double **y_out;     // matriz de salida (steps+1 x n)
    int steps;          // numero de pasos (salida)
} ODESystem;

typedef void (*ODEFunc)(double t, const double *y, double *dydt, void *params);

void rk4_solve(ODESystem *sys, ODEFunc f, void *params);
void rk4_solve_alloc(ODESystem *sys, ODEFunc f, void *params);
void rk4_solve_free(ODESystem *sys);

#endif