#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <json-c/json.h>
#include <time.h>

DatosArchivo* leer_csv(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;
    
    char line[4096];
    int n = 0;
    while (fgets(line, sizeof(line), fp)) n++;
    rewind(fp);
    n--;  // cabecera
    
    DatosArchivo *datos = (DatosArchivo*)malloc(sizeof(DatosArchivo));
    datos->t = (double*)malloc(n * sizeof(double));
    datos->x = (double*)malloc(n * sizeof(double));
    datos->F = (double*)malloc(n * sizeof(double));
    datos->n = n;
    
    // Saltar cabecera
    if (fgets(line, sizeof(line), fp) == NULL) {
        free(datos->t); free(datos->x); free(datos->F); free(datos);
        fclose(fp);
        return NULL;
    }
    
    int i = 0;
    while (i < n && fscanf(fp, "%lf,%lf,%lf,%[^,],%lf",
                           &datos->t[i], &datos->x[i], &datos->F[i],
                           datos->falla, &datos->severidad) == 5) {
        i++;
    }
    datos->n = i;
    fclose(fp);
    return datos;
}

void liberar_datos(DatosArchivo *datos) {
    if (!datos) return;
    if (datos->t) free(datos->t);
    if (datos->x) free(datos->x);
    if (datos->F) free(datos->F);
    free(datos);
}

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
                                      int num_familias_evaluadas) {
    // Objeto raiz
    json_object *root = json_object_new_object();
    
    // Timestamp
    time_t rawtime;
    struct tm *timeinfo;
    char timestamp[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
    json_object_object_add(root, "timestamp", json_object_new_string(timestamp));
    
    // Configuracion
    json_object *config = json_object_new_object();
    json_object_object_add(config, "maxiter", json_object_new_int(maxiter));
    json_object_object_add(config, "popsize", json_object_new_int(popsize));
    json_object_object_add(root, "config", config);
    
    // Resultados para este archivo
    json_object *result = json_object_new_object();
    json_object_object_add(result, "archivo", json_object_new_string(nombre_archivo));
    json_object_object_add(result, "familia", json_object_new_string(mejor_familia));
    json_object_object_add(result, "error", json_object_new_double(mejor_error));
    json_object_object_add(result, "ecuacion", json_object_new_string(ecuacion));
    
    // Parametros del mejor modelo (array)
    json_object *params_arr = json_object_new_array();
    for (int i = 0; i < mejor_num_params; i++) {
        json_object_array_add(params_arr, json_object_new_double(mejor_params[i]));
    }
    json_object_object_add(result, "params", params_arr);
    
    // Datos simulados x_sim
    json_object *sim_arr = json_object_new_array();
    for (int i = 0; i < n; i++) {
        json_object_array_add(sim_arr, json_object_new_double(x_sim[i]));
    }
    json_object_object_add(result, "x_sim", sim_arr);
    
    // t y x_real (opcional, pero útil para graficar)
    json_object *t_arr = json_object_new_array();
    json_object *x_real_arr = json_object_new_array();
    for (int i = 0; i < n; i++) {
        json_object_array_add(t_arr, json_object_new_double(t[i]));
        json_object_array_add(x_real_arr, json_object_new_double(x_real[i]));
    }
    json_object_object_add(result, "t", t_arr);
    json_object_object_add(result, "x_real", x_real_arr);
    
    // Errores de todas las familias evaluadas
    json_object *errores_obj = json_object_new_object();
    for (int i = 0; i < num_familias_evaluadas; i++) {
        json_object_object_add(errores_obj, nombres_familias[i],
                               json_object_new_double(errores_familias[i]));
    }
    json_object_object_add(result, "errores_por_familia", errores_obj);
    
    // Agregar 'result' al root
    json_object_object_add(root, "resultado", result);
    
    // Crear nombre de archivo de salida (cambiamos .csv por .json)
    char out_filename[512];
    snprintf(out_filename, sizeof(out_filename), "%s/%s.json", output_dir, nombre_archivo);
    char *dot = strrchr(out_filename, '.');
    if (dot) strcpy(dot, ".json");
    
    // Guardar a archivo
    FILE *fp = fopen(out_filename, "w");
    if (fp) {
        fprintf(fp, "%s", json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY));
        fclose(fp);
    } else {
        fprintf(stderr, "Error: No se pudo guardar %s\n", out_filename);
    }
    
    json_object_put(root);  // Liberar memoria
}