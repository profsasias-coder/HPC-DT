#include "familias.h"
#include "solver.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// 1. OSCILADOR SIMPLE (Original)
// ============================================================
typedef struct {
    double omega_n;
    double zeta;
    double *F;
    double *t;
    int n;
} OsciladorSimpleParams;

static void oscilador_simple_ode(double t, const double *y, double *dydt, void *params) {
    OsciladorSimpleParams *p = (OsciladorSimpleParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0];
    double v = y[1];
    dydt[0] = v;
    dydt[1] = F_curr - 2 * p->zeta * p->omega_n * v - p->omega_n * p->omega_n * x;
}

double evaluar_oscilador_simple(double *params, double *t, double *x_real, double *F, int n) {
    double omega_n = params[0];
    double zeta = params[1];
    
    ODESystem sys;
    sys.n = 2;
    sys.t0 = t[0];
    sys.tf = t[n-1];
    sys.dt = t[1] - t[0];
    double y0[] = {0.0, 0.0};
    sys.y0 = y0;
    
    OsciladorSimpleParams odeparams = {omega_n, zeta, F, t, n};
    
    rk4_solve_alloc(&sys, oscilador_simple_ode, &odeparams);
    
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    
    rk4_solve_free(&sys);
    return mse;
}

void simular_oscilador_simple(double *params, double *t, double *F, int n, double *x_sim) {
    double omega_n = params[0];
    double zeta = params[1];
    
    ODESystem sys;
    sys.n = 2;
    sys.t0 = t[0];
    sys.tf = t[n-1];
    sys.dt = t[1] - t[0];
    double y0[] = {0.0, 0.0};
    sys.y0 = y0;
    
    OsciladorSimpleParams odeparams = {omega_n, zeta, F, t, n};
    
    rk4_solve_alloc(&sys, oscilador_simple_ode, &odeparams);
    
    for (int i = 0; i < n && i <= sys.steps; i++) {
        x_sim[i] = sys.y_out[i][0];
    }
    
    rk4_solve_free(&sys);
}

// ============================================================
// 2. OSCILADOR DUFFING (Original)
// ============================================================
typedef struct {
    double omega_n, zeta, beta;
    double *F, *t;
    int n;
} DuffingParams;

static void duffing_ode(double t, const double *y, double *dydt, void *params) {
    DuffingParams *p = (DuffingParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*x - p->beta*x*x*x;
}

double evaluar_oscilador_duffing(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    DuffingParams p = {params[0], params[1], params[2], F, t, n};
    rk4_solve_alloc(&sys, duffing_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_oscilador_duffing(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    DuffingParams p = {params[0], params[1], params[2], F, t, n};
    rk4_solve_alloc(&sys, duffing_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}

// ============================================================
// 3. AMORTIGUAMIENTO NO LINEAL (Original)
// ============================================================
typedef struct {
    double omega_n, zeta, gamma;
    double *F, *t;
    int n;
} AmortiguamientoParams;

static void amortiguamiento_ode(double t, const double *y, double *dydt, void *params) {
    AmortiguamientoParams *p = (AmortiguamientoParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*x - p->gamma*v*v*v;
}

double evaluar_amortiguamiento_nolineal(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    AmortiguamientoParams p = {params[0], params[1], params[2], F, t, n};
    rk4_solve_alloc(&sys, amortiguamiento_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_amortiguamiento_nolineal(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    AmortiguamientoParams p = {params[0], params[1], params[2], F, t, n};
    rk4_solve_alloc(&sys, amortiguamiento_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}

// ============================================================
// 4. FORZAMIENTO PARAMETRICO (Original)
// ============================================================
typedef struct {
    double omega_n, zeta, epsilon, omega_p;
    double *F, *t;
    int n;
} ForzamientoParams;

static void forzamiento_ode(double t, const double *y, double *dydt, void *params) {
    ForzamientoParams *p = (ForzamientoParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*(1 + p->epsilon*sin(p->omega_p*t))*x;
}

double evaluar_forzamiento_parametrico(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ForzamientoParams p = {params[0], params[1], params[2], params[3], F, t, n};
    rk4_solve_alloc(&sys, forzamiento_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_forzamiento_parametrico(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ForzamientoParams p = {params[0], params[1], params[2], params[3], F, t, n};
    rk4_solve_alloc(&sys, forzamiento_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}

// ============================================================
// 5. IMPACTO (Original)
// ============================================================
typedef struct {
    double omega_n, zeta, gap, k_impacto;
    double *F, *t;
    int n;
} ImpactoParams;

static void impacto_ode(double t, const double *y, double *dydt, void *params) {
    ImpactoParams *p = (ImpactoParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    double F_impacto;
    if (x > p->gap) F_impacto = -p->k_impacto * (x - p->gap);
    else if (x < -p->gap) F_impacto = -p->k_impacto * (x + p->gap);
    else F_impacto = 0.0;
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*x + F_impacto;
}

double evaluar_impacto(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ImpactoParams p = {params[0], params[1], params[2], params[3], F, t, n};
    rk4_solve_alloc(&sys, impacto_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_impacto(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ImpactoParams p = {params[0], params[1], params[2], params[3], F, t, n};
    rk4_solve_alloc(&sys, impacto_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}

// ============================================================
// 6. OSCILADOR CON AMORTIGUAMIENTO CUADRATICO (NUEVA)
// ============================================================
typedef struct {
    double omega_n, zeta, alpha;
    double *F, *t;
    int n;
} OsciladorAmortCuadParams;

static void oscilador_amort_cuad_ode(double t, const double *y, double *dydt, void *params) {
    OsciladorAmortCuadParams *p = (OsciladorAmortCuadParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*x - p->alpha*v*v;
}

double evaluar_oscilador_amortiguamiento_cuadratico(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    OsciladorAmortCuadParams p = {params[0], params[1], params[2], F, t, n};
    rk4_solve_alloc(&sys, oscilador_amort_cuad_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_oscilador_amortiguamiento_cuadratico(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    OsciladorAmortCuadParams p = {params[0], params[1], params[2], F, t, n};
    rk4_solve_alloc(&sys, oscilador_amort_cuad_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}

// ============================================================
// 7. RESORTE ASIMETRICO (NUEVA)
// ============================================================
typedef struct {
    double omega_n, zeta, beta1, beta2;
    double *F, *t;
    int n;
} ResorteAsimetricoParams;

static void resorte_asimetrico_ode(double t, const double *y, double *dydt, void *params) {
    ResorteAsimetricoParams *p = (ResorteAsimetricoParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*x - p->beta1*x*x - p->beta2*x*x*x;
}

double evaluar_resorte_asimetrico(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ResorteAsimetricoParams p = {params[0], params[1], params[2], params[3], F, t, n};
    rk4_solve_alloc(&sys, resorte_asimetrico_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_resorte_asimetrico(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ResorteAsimetricoParams p = {params[0], params[1], params[2], params[3], F, t, n};
    rk4_solve_alloc(&sys, resorte_asimetrico_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}

// ============================================================
// 8. IMPACTO CON HISTERESIS (NUEVA)
// ============================================================
typedef struct {
    double omega_n, zeta, gap, k_impacto, delta;
    double *F, *t;
    int n;
} ImpactoHisteresisParams;

static void impacto_histeresis_ode(double t, const double *y, double *dydt, void *params) {
    ImpactoHisteresisParams *p = (ImpactoHisteresisParams*)params;
    int idx = (int)(t / (p->t[1] - p->t[0]));
    if (idx >= p->n) idx = p->n - 1;
    if (idx < 0) idx = 0;
    double F_curr = p->F[idx];
    double x = y[0], v = y[1];
    double F_impacto;
    if (x > p->gap) {
        F_impacto = -p->k_impacto * (x - p->gap - p->delta * v);
    } else if (x < -p->gap) {
        F_impacto = -p->k_impacto * (x + p->gap - p->delta * v);
    } else {
        F_impacto = 0.0;
    }
    dydt[0] = v;
    dydt[1] = F_curr - 2*p->zeta*p->omega_n*v - p->omega_n*p->omega_n*x + F_impacto;
}

double evaluar_impacto_histeresis(double *params, double *t, double *x_real, double *F, int n) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ImpactoHisteresisParams p = {params[0], params[1], params[2], params[3], params[4], F, t, n};
    rk4_solve_alloc(&sys, impacto_histeresis_ode, &p);
    double mse = 0.0;
    for (int i = 0; i < n && i <= sys.steps; i++) {
        double diff = sys.y_out[i][0] - x_real[i];
        mse += diff * diff;
    }
    mse /= n;
    rk4_solve_free(&sys);
    return mse;
}

void simular_impacto_histeresis(double *params, double *t, double *F, int n, double *x_sim) {
    ODESystem sys = {2, t[0], t[n-1], t[1]-t[0], (double[]){0.0, 0.0}, NULL, NULL, 0};
    ImpactoHisteresisParams p = {params[0], params[1], params[2], params[3], params[4], F, t, n};
    rk4_solve_alloc(&sys, impacto_histeresis_ode, &p);
    for (int i = 0; i < n && i <= sys.steps; i++) x_sim[i] = sys.y_out[i][0];
    rk4_solve_free(&sys);
}