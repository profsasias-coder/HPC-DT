#include "familias.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Utilidades para derivadas numericas
static void derivada1(const double *x, const double *t, int n, double *dx) {
    if (n < 2) return;
    dx[0] = (x[1] - x[0]) / (t[1] - t[0]);
    for (int i = 1; i < n-1; i++)
        dx[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1]);
    dx[n-1] = (x[n-1] - x[n-2]) / (t[n-1] - t[n-2]);
}

static void derivada2(const double *x, const double *t, int n, double *d2x) {
    double *dx = (double*)malloc(n * sizeof(double));
    derivada1(x, t, n, dx);
    derivada1(dx, t, n, d2x);
    free(dx);
}

// ============================================================
// 1. POLINOMICA GENERICA (grado variable)
// ============================================================
double evaluar_polinomica(double *params, int grado, double *t, double *x, double *F, int n) {
    int nc = grado + 1;
    double *a = params;
    double *b = params + nc;
    double c = params[2 * nc];
    
    double *dx = (double*)malloc(n * sizeof(double));
    double *d2x_real = (double*)malloc(n * sizeof(double));
    derivada1(x, t, n, dx);
    derivada2(x, t, n, d2x_real);
    
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double sum_x = 0.0, sum_dx = 0.0;
        double x_pow = 1.0, dx_pow = 1.0;
        for (int k = 0; k <= grado; k++) {
            sum_x += a[k] * x_pow;
            sum_dx += b[k] * dx_pow;
            x_pow *= x[i];
            dx_pow *= dx[i];
        }
        double d2x_pred = sum_x + sum_dx + c * F[i];
        double diff = d2x_pred - d2x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    free(dx); free(d2x_real);
    return mse;
}

// ============================================================
// 2. FOURIER GENERICO (armonicos variable)
// ============================================================
double evaluar_fourier(double *params, int n_arm, double *t, double *x, double *F, int n) {
    double *a = params;
    double *b = params + n_arm;
    double omega = params[2 * n_arm];
    double c = params[2 * n_arm + 1];
    
    double *d2x_real = (double*)malloc(n * sizeof(double));
    derivada2(x, t, n, d2x_real);
    
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 1; k <= n_arm; k++) {
            double arg = k * omega * t[i];
            sum += a[k-1] * cos(arg) + b[k-1] * sin(arg);
        }
        double d2x_pred = sum + c * F[i];
        double diff = d2x_pred - d2x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    free(d2x_real);
    return mse;
}

// ============================================================
// 3. EXPONENCIAL
// ============================================================
double evaluar_exponencial(double *params, double *t, double *x, double *F, int n) {
    double A = params[0], alpha = params[1], beta = params[2], gamma = params[3];
    
    double *dx = (double*)malloc(n * sizeof(double));
    double *d2x_real = (double*)malloc(n * sizeof(double));
    derivada1(x, t, n, dx);
    derivada2(x, t, n, d2x_real);
    
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d2x_pred = A * exp(alpha * x[i]) + beta * dx[i] + gamma * F[i];
        double diff = d2x_pred - d2x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    free(dx); free(d2x_real);
    return mse;
}