#ifndef FAMILIAS_H
#define FAMILIAS_H

// Estructura para representar una familia de modelos
typedef struct {
    char *nombre;
    int num_params;
    double *bounds_min;
    double *bounds_max;
    double (*evaluar)(double *params, double *t, double *x_real, double *F, int n);
    void (*simular)(double *params, double *t, double *F, int n, double *x_sim);
    int tipo;  // 0=fisica, 1=matematica
} Familia;

// Prototipos de familias fisicas (5 originales + 3 nuevas)
double evaluar_oscilador_simple(double *params, double *t, double *x_real, double *F, int n);
void simular_oscilador_simple(double *params, double *t, double *F, int n, double *x_sim);

double evaluar_oscilador_duffing(double *params, double *t, double *x_real, double *F, int n);
void simular_oscilador_duffing(double *params, double *t, double *F, int n, double *x_sim);

double evaluar_amortiguamiento_nolineal(double *params, double *t, double *x_real, double *F, int n);
void simular_amortiguamiento_nolineal(double *params, double *t, double *F, int n, double *x_sim);

double evaluar_forzamiento_parametrico(double *params, double *t, double *x_real, double *F, int n);
void simular_forzamiento_parametrico(double *params, double *t, double *F, int n, double *x_sim);

double evaluar_impacto(double *params, double *t, double *x_real, double *F, int n);
void simular_impacto(double *params, double *t, double *F, int n, double *x_sim);

// NUEVAS familias fisicas
double evaluar_oscilador_amortiguamiento_cuadratico(double *params, double *t, double *x_real, double *F, int n);
void simular_oscilador_amortiguamiento_cuadratico(double *params, double *t, double *F, int n, double *x_sim);

double evaluar_resorte_asimetrico(double *params, double *t, double *x_real, double *F, int n);
void simular_resorte_asimetrico(double *params, double *t, double *F, int n, double *x_sim);

double evaluar_impacto_histeresis(double *params, double *t, double *x_real, double *F, int n);
void simular_impacto_histeresis(double *params, double *t, double *F, int n, double *x_sim);

// Prototipos de familias matematicas (2 originales + 2 nuevas)
double evaluar_polinomica(double *params, int grado, double *t, double *x, double *F, int n);
double evaluar_fourier(double *params, int n_armonicos, double *t, double *x, double *F, int n);
double evaluar_exponencial(double *params, double *t, double *x, double *F, int n);

// NUEVAS familias matematicas (wrappers)
double evaluar_polinomica_grado4(double *params, double *t, double *x_real, double *F, int n);
double evaluar_fourier_4(double *params, double *t, double *x_real, double *F, int n);

#endif