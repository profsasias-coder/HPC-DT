#ifndef IO_H
#define IO_H

#include "familias.h"

typedef struct {
    double *t;
    double *x;
    double *F;
    char falla[50];
    double severidad;
    int n;
} DatosArchivo;

DatosArchivo* leer_csv(const char *filename);
void liberar_datos(DatosArchivo *datos);

// Funcion para guardar JSON completo
void guardar_resultados_json_completo(const char *output_dir,
                                      const char *nombre_archivo,
                                      const char *mejor_familia,
                                      double *mejor_params,
                                      int mejor_num_params,
                                      double mejor_error,
                                      const char *ecuacion,
                                      double *t,
                                      double *x_real,
                                      double *x_sim,
                                      int n,
                                      int maxiter,
                                      int popsize,
                                      char **nombres_familias,
                                      double *errores_familias,
                                      int num_familias_evaluadas);

#endif