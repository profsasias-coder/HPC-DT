#include "solver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void rk4_step(double t, double *y, double dt, int n, ODEFunc f, void *params) {
    double *k1 = (double*)malloc(n * sizeof(double));
    double *k2 = (double*)malloc(n * sizeof(double));
    double *k3 = (double*)malloc(n * sizeof(double));
    double *k4 = (double*)malloc(n * sizeof(double));
    double *ytmp = (double*)malloc(n * sizeof(double));
    
    // k1
    f(t, y, k1, params);
    for (int i = 0; i < n; i++) ytmp[i] = y[i] + 0.5 * dt * k1[i];
    
    // k2
    f(t + 0.5*dt, ytmp, k2, params);
    for (int i = 0; i < n; i++) ytmp[i] = y[i] + 0.5 * dt * k2[i];
    
    // k3
    f(t + 0.5*dt, ytmp, k3, params);
    for (int i = 0; i < n; i++) ytmp[i] = y[i] + dt * k3[i];
    
    // k4
    f(t + dt, ytmp, k4, params);
    
    // Actualizar
    for (int i = 0; i < n; i++) {
        y[i] += (dt/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
    
    free(k1); free(k2); free(k3); free(k4); free(ytmp);
}

void rk4_solve_alloc(ODESystem *sys, ODEFunc f, void *params) {
    int n = sys->n;
    double dt = sys->dt;
    sys->steps = (int)((sys->tf - sys->t0) / dt);
    if (fabs(sys->tf - sys->t0 - sys->steps * dt) > 1e-12) {
        sys->steps++;
    }
    
    sys->t_out = (double*)malloc((sys->steps + 1) * sizeof(double));
    sys->y_out = (double**)malloc((sys->steps + 1) * sizeof(double*));
    for (int i = 0; i <= sys->steps; i++) {
        sys->y_out[i] = (double*)malloc(n * sizeof(double));
    }
    
    double *y = (double*)malloc(n * sizeof(double));
    memcpy(y, sys->y0, n * sizeof(double));
    
    sys->t_out[0] = sys->t0;
    memcpy(sys->y_out[0], y, n * sizeof(double));
    
    double t = sys->t0;
    for (int i = 1; i <= sys->steps; i++) {
        rk4_step(t, y, dt, n, f, params);
        t += dt;
        sys->t_out[i] = t;
        memcpy(sys->y_out[i], y, n * sizeof(double));
    }
    
    free(y);
}

void rk4_solve_free(ODESystem *sys) {
    if (sys->t_out) free(sys->t_out);
    if (sys->y_out) {
        for (int i = 0; i <= sys->steps; i++) {
            if (sys->y_out[i]) free(sys->y_out[i]);
        }
        free(sys->y_out);
    }
}