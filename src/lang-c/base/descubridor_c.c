#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include "solver.h"
#include "familias.h"
#include "optimizador.h"
#include "io.h"
#include "utils.h"

// ============================================================
// Configuracion
// ============================================================
typedef struct {
    char *input_dir;
    char *output_base;   // directorio base de salida (se creará con timestamp dentro)
    char *familias_tipo;
    int maxiter;
    int popsize;
} Config;

// ============================================================
// Funcion objetivo para el optimizador (wrappers)
// ============================================================
typedef struct {
    Familia *familia;
    double *t;
    double *x_real;
    double *F;
    int n;
} UserData;

static double funcion_objetivo_wrapper(double *params, void *user_data) {
    UserData *data = (UserData*)user_data;
    return data->familia->evaluar(params, data->t, data->x_real, data->F, data->n);
}

// ============================================================
// Generar ecuacion en texto (similar a Python)
// ============================================================
static void generar_ecuacion_texto(char *buffer, size_t size, const char *nombre,
                                   double *params, int num_params) {
    char eq[1024] = "";
    
    if (strcmp(nombre, "Oscilador_Simple") == 0 && num_params >= 2) {
        snprintf(eq, sizeof(eq), "d2x/dt2 + 2*%.4f*%.4f*dx/dt + %.4f^2*x = F(t)",
                 params[1], params[0], params[0]);
    }
    else if (strcmp(nombre, "Oscilador_Duffing") == 0 && num_params >= 3) {
        snprintf(eq, sizeof(eq), "d2x/dt2 + 2*%.4f*%.4f*dx/dt + %.4f^2*x + %.4f*x^3 = F(t)",
                 params[1], params[0], params[0], params[2]);
    }
    else if (strcmp(nombre, "Amortiguamiento_NoLineal") == 0 && num_params >= 3) {
        snprintf(eq, sizeof(eq), "d2x/dt2 + 2*%.4f*%.4f*dx/dt + %.4f^2*x + %.4f*(dx/dt)^3 = F(t)",
                 params[1], params[0], params[0], params[2]);
    }
    else if (strcmp(nombre, "Forzamiento_Parametrico") == 0 && num_params >= 4) {
        snprintf(eq, sizeof(eq), "d2x/dt2 + 2*%.4f*%.4f*dx/dt + %.4f^2*(1 + %.4f*sin(%.4f*t))*x = F(t)",
                 params[1], params[0], params[0], params[2], params[3]);
    }
    else if (strcmp(nombre, "Impacto") == 0 && num_params >= 4) {
        snprintf(eq, sizeof(eq), "d2x/dt2 + 2*%.4f*%.4f*dx/dt + %.4f^2*x + F_impacto(x) = F(t)  (gap=%.4f, k=%.4f)",
                 params[1], params[0], params[0], params[2], params[3]);
    }
    else if (strcmp(nombre, "Polinomica_grado2") == 0 && num_params >= 7) {
        snprintf(eq, sizeof(eq), "d2x = a0 + a1*x + a2*x^2 + b0*dx + b1*dx^2 + c*F  [a=(%.2f,%.2f,%.2f) b=(%.2f,%.2f,%.2f) c=%.2f]",
                 params[0], params[1], params[2], params[3], params[4], params[5], params[6]);
    }
    else if (strcmp(nombre, "Polinomica_grado3") == 0 && num_params >= 9) {
        snprintf(eq, sizeof(eq), "d2x = a0 + a1*x + a2*x^2 + a3*x^3 + b0*dx + b1*dx^2 + b2*dx^3 + c*F  [a=(%.2f,%.2f,%.2f,%.2f) b=(%.2f,%.2f,%.2f) c=%.2f]",
                 params[0], params[1], params[2], params[3],
                 params[4], params[5], params[6], params[7], params[8]);
    }
    else if (strcmp(nombre, "Fourier_2") == 0 && num_params >= 6) {
        snprintf(eq, sizeof(eq), "d2x = a1*cos(wt)+a2*cos(2wt)+b1*sin(wt)+b2*sin(2wt)+c*F  [a=(%.2f,%.2f) b=(%.2f,%.2f) w=%.2f c=%.2f]",
                 params[0], params[1], params[2], params[3], params[4], params[5]);
    }
    else if (strcmp(nombre, "Fourier_3") == 0 && num_params >= 8) {
        snprintf(eq, sizeof(eq), "d2x = a1*cos(wt)+a2*cos(2wt)+a3*cos(3wt)+b1*sin(wt)+b2*sin(2wt)+b3*sin(3wt)+c*F  [a=(%.2f,%.2f,%.2f) b=(%.2f,%.2f,%.2f) w=%.2f c=%.2f]",
                 params[0], params[1], params[2],
                 params[3], params[4], params[5], params[6], params[7]);
    }
    else if (strcmp(nombre, "Exponencial") == 0 && num_params >= 4) {
        snprintf(eq, sizeof(eq), "d2x = A*exp(alpha*x) + beta*dx + gamma*F  [A=%.2f alpha=%.2f beta=%.2f gamma=%.2f]",
                 params[0], params[1], params[2], params[3]);
    }
    else {
        snprintf(eq, sizeof(eq), "Ecuacion no disponible para %s", nombre);
    }
    
    strncpy(buffer, eq, size - 1);
    buffer[size - 1] = '\0';
}

// ============================================================
// Crear lista de familias
// ============================================================
int crear_familias(const char *tipo, Familia **familias_out) {
    // Bounds fijos
    static double bounds_osc[4] = {5.0, 0.001, 50.0, 0.5};
    static double bounds_duffing[6] = {5.0, 0.001, -10.0, 50.0, 0.5, 10.0};
    static double bounds_amort[6] = {5.0, 0.001, -10.0, 50.0, 0.5, 10.0};
    static double bounds_forz[8] = {5.0, 0.001, 0.0, 0.1, 50.0, 0.5, 1.0, 10.0};
    static double bounds_impacto[8] = {5.0, 0.001, 0.001, 10.0, 50.0, 0.5, 1.0, 1000.0};

    static Familia familias[10];
    int count = 0;

    // ---- Familias fisicas ----
    if (strcmp(tipo, "fisicas") == 0 || strcmp(tipo, "todas") == 0) {
        familias[count].nombre = "Oscilador_Simple";
        familias[count].num_params = 2;
        familias[count].bounds_min = bounds_osc;
        familias[count].bounds_max = bounds_osc + 2;
        familias[count].evaluar = evaluar_oscilador_simple;
        familias[count].simular = simular_oscilador_simple;
        familias[count].tipo = 0;
        count++;

        familias[count].nombre = "Oscilador_Duffing";
        familias[count].num_params = 3;
        familias[count].bounds_min = bounds_duffing;
        familias[count].bounds_max = bounds_duffing + 3;
        familias[count].evaluar = evaluar_oscilador_duffing;
        familias[count].simular = simular_oscilador_duffing;
        familias[count].tipo = 0;
        count++;

        familias[count].nombre = "Amortiguamiento_NoLineal";
        familias[count].num_params = 3;
        familias[count].bounds_min = bounds_amort;
        familias[count].bounds_max = bounds_amort + 3;
        familias[count].evaluar = evaluar_amortiguamiento_nolineal;
        familias[count].simular = simular_amortiguamiento_nolineal;
        familias[count].tipo = 0;
        count++;

        familias[count].nombre = "Forzamiento_Parametrico";
        familias[count].num_params = 4;
        familias[count].bounds_min = bounds_forz;
        familias[count].bounds_max = bounds_forz + 4;
        familias[count].evaluar = evaluar_forzamiento_parametrico;
        familias[count].simular = simular_forzamiento_parametrico;
        familias[count].tipo = 0;
        count++;

        familias[count].nombre = "Impacto";
        familias[count].num_params = 4;
        familias[count].bounds_min = bounds_impacto;
        familias[count].bounds_max = bounds_impacto + 4;
        familias[count].evaluar = evaluar_impacto;
        familias[count].simular = simular_impacto;
        familias[count].tipo = 0;
        count++;
    }

    // ---- Familias matematicas ----
    if (strcmp(tipo, "matematicas") == 0 || strcmp(tipo, "todas") == 0) {
        // Polinomica grado 2 (7 params)
        static double bounds_poli2_min[7], bounds_poli2_max[7];
        for (int i = 0; i < 7; i++) { bounds_poli2_min[i] = -100.0; bounds_poli2_max[i] = 100.0; }
        familias[count].nombre = "Polinomica_grado2";
        familias[count].num_params = 7;
        familias[count].bounds_min = bounds_poli2_min;
        familias[count].bounds_max = bounds_poli2_max;
        count++;

        // Polinomica grado 3 (9 params)
        static double bounds_poli3_min[9], bounds_poli3_max[9];
        for (int i = 0; i < 9; i++) { bounds_poli3_min[i] = -100.0; bounds_poli3_max[i] = 100.0; }
        familias[count].nombre = "Polinomica_grado3";
        familias[count].num_params = 9;
        familias[count].bounds_min = bounds_poli3_min;
        familias[count].bounds_max = bounds_poli3_max;
        count++;

        // Fourier 2 (6 params)
        static double bounds_fourier2_min[6], bounds_fourier2_max[6];
        for (int i = 0; i < 6; i++) {
            if (i == 4) { // omega
                bounds_fourier2_min[i] = 0.1;
                bounds_fourier2_max[i] = 50.0;
            } else {
                bounds_fourier2_min[i] = -100.0;
                bounds_fourier2_max[i] = 100.0;
            }
        }
        familias[count].nombre = "Fourier_2";
        familias[count].num_params = 6;
        familias[count].bounds_min = bounds_fourier2_min;
        familias[count].bounds_max = bounds_fourier2_max;
        count++;

        // Fourier 3 (8 params)
        static double bounds_fourier3_min[8], bounds_fourier3_max[8];
        for (int i = 0; i < 8; i++) {
            if (i == 6) { // omega
                bounds_fourier3_min[i] = 0.1;
                bounds_fourier3_max[i] = 50.0;
            } else {
                bounds_fourier3_min[i] = -100.0;
                bounds_fourier3_max[i] = 100.0;
            }
        }
        familias[count].nombre = "Fourier_3";
        familias[count].num_params = 8;
        familias[count].bounds_min = bounds_fourier3_min;
        familias[count].bounds_max = bounds_fourier3_max;
        count++;

        // Exponencial (4 params)
        static double bounds_exp_min[4], bounds_exp_max[4];
        for (int i = 0; i < 4; i++) {
            bounds_exp_min[i] = -100.0;
            bounds_exp_max[i] = 100.0;
        }
        familias[count].nombre = "Exponencial";
        familias[count].num_params = 4;
        familias[count].bounds_min = bounds_exp_min;
        familias[count].bounds_max = bounds_exp_max;
        count++;
    }

    *familias_out = familias;
    return count;
}

// ============================================================
// Wrappers para familias matematicas
// ============================================================
static double evaluar_poli_wrapper_grado2(double *params, double *t, double *x_real, double *F, int n) {
    return evaluar_polinomica(params, 2, t, x_real, F, n);
}
static double evaluar_poli_wrapper_grado3(double *params, double *t, double *x_real, double *F, int n) {
    return evaluar_polinomica(params, 3, t, x_real, F, n);
}
static double evaluar_fourier_wrapper_2(double *params, double *t, double *x_real, double *F, int n) {
    return evaluar_fourier(params, 2, t, x_real, F, n);
}
static double evaluar_fourier_wrapper_3(double *params, double *t, double *x_real, double *F, int n) {
    return evaluar_fourier(params, 3, t, x_real, F, n);
}

// ============================================================
// Funcion para crear directorio recursivamente
// ============================================================
static void crear_directorio(const char *path) {
    char tmp[512];
    char *p = NULL;
    size_t len;
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/') tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0777);
            *p = '/';
        }
    }
    mkdir(tmp, 0777);
}

// ============================================================
// Funcion principal
// ============================================================
int main(int argc, char **argv) {
    // Valores por defecto (se pueden pasar como argumentos en el futuro)
    const char *input_dir = "./datos_ejemplo";
    const char *output_base = "./salidas";
    const char *familias_tipo = "todas";
    int maxiter = 8;
    int popsize = 6;

    // Crear carpeta con timestamp
    time_t rawtime;
    struct tm *timeinfo;
    char timestamp[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d-%H-%M", timeinfo);
    
    char output_dir[512];
    snprintf(output_dir, sizeof(output_dir), "%s/%s", output_base, timestamp);
    crear_directorio(output_dir);
    
    // Subcarpetas para JSON y gráficos
    char json_dir[512], graficos_dir[512];
    snprintf(json_dir, sizeof(json_dir), "%s/01-json", output_dir);
    snprintf(graficos_dir, sizeof(graficos_dir), "%s/02-graficos", output_dir);
    crear_directorio(json_dir);
    crear_directorio(graficos_dir);

    print_step("=== DESCUBRIDOR EN C (CON TIMESTAMP Y SALIDA JSON) ===");
    print_step("Directorio entrada: %s", input_dir);
    print_step("Directorio salida base: %s", output_base);
    print_step("Directorio salida esta ejecucion: %s", output_dir);
    print_step("Familias: %s, maxiter: %d, popsize: %d", familias_tipo, maxiter, popsize);

    // Leer archivos CSV
    DIR *dir = opendir(input_dir);
    if (!dir) {
        print_step("ERROR: No se puede abrir el directorio %s", input_dir);
        return 1;
    }

    struct dirent *entry;
    char archivos[100][256];
    int num_archivos = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".csv")) {
            strcpy(archivos[num_archivos], entry->d_name);
            num_archivos++;
        }
    }
    closedir(dir);

    if (num_archivos == 0) {
        print_step("ERROR: No se encontraron archivos .csv en %s", input_dir);
        return 1;
    }

    print_step("Archivos encontrados: %d", num_archivos);

    // Crear familias
    Familia *familias;
    int num_familias = crear_familias(familias_tipo, &familias);
    print_step("Familias cargadas: %d", num_familias);

    // Asignar funciones de evaluacion para las matematicas
    for (int i = 0; i < num_familias; i++) {
        if (strcmp(familias[i].nombre, "Polinomica_grado2") == 0) {
            familias[i].evaluar = evaluar_poli_wrapper_grado2;
        } else if (strcmp(familias[i].nombre, "Polinomica_grado3") == 0) {
            familias[i].evaluar = evaluar_poli_wrapper_grado3;
        } else if (strcmp(familias[i].nombre, "Fourier_2") == 0) {
            familias[i].evaluar = evaluar_fourier_wrapper_2;
        } else if (strcmp(familias[i].nombre, "Fourier_3") == 0) {
            familias[i].evaluar = evaluar_fourier_wrapper_3;
        }
    }

    // Procesar cada archivo
    double tiempo_total = 0.0;
    clock_t start_total = clock();

    for (int a = 0; a < num_archivos; a++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", input_dir, archivos[a]);
        print_step("Procesando archivo %d/%d: %s", a+1, num_archivos, archivos[a]);

        DatosArchivo *datos = leer_csv(path);
        if (!datos) {
            print_step("  ERROR: No se pudo leer %s", archivos[a]);
            continue;
        }

        // Arrays para guardar errores de todas las familias
        char **nombres_familias = (char**)malloc(num_familias * sizeof(char*));
        double *errores_familias = (double*)malloc(num_familias * sizeof(double));
        int num_familias_evaluadas = 0;

        double mejor_error = 1e9;
        double *mejor_params = NULL;
        char *mejor_familia = NULL;
        int mejor_num_params = 0;

        for (int f = 0; f < num_familias; f++) {
            Familia *fam = &familias[f];
            if (!fam->evaluar) continue;

            print_step("  Evaluando familia %d/%d: %s", f+1, num_familias, fam->nombre);

            // Configurar optimizador
            EvolucionDiferencial ed;
            ed.maxiter = maxiter;
            ed.popsize = popsize;
            ed.num_params = fam->num_params;
            ed.bounds_min = fam->bounds_min;
            ed.bounds_max = fam->bounds_max;

            UserData ud = {fam, datos->t, datos->x, datos->F, datos->n};
            ed.funcion_objetivo = funcion_objetivo_wrapper;
            ed.user_data = &ud;

            // Optimizar
            double error;
            double *params = evolucion_diferencial(&ed, &error);

            print_step("    Error final para %s: %.6e", fam->nombre, error);

            // Guardar error de esta familia
            nombres_familias[num_familias_evaluadas] = fam->nombre;
            errores_familias[num_familias_evaluadas] = error;
            num_familias_evaluadas++;

            if (error < mejor_error) {
                mejor_error = error;
                if (mejor_params) free(mejor_params);
                mejor_params = params;
                mejor_familia = fam->nombre;
                mejor_num_params = fam->num_params;
            } else {
                free(params);
            }
        }

        if (mejor_params) {
            print_step("  MEJOR para %s: %s (error = %.6e)", archivos[a], mejor_familia, mejor_error);

            // Generar ecuacion en texto
            char ecuacion[2048];
            generar_ecuacion_texto(ecuacion, sizeof(ecuacion), mejor_familia, mejor_params, mejor_num_params);
            print_step("  ECUACION: %s", ecuacion);

            // Simular el mejor modelo
            Familia *mejor_familia_ptr = NULL;
            for (int f = 0; f < num_familias; f++) {
                if (familias[f].evaluar && strcmp(familias[f].nombre, mejor_familia) == 0) {
                    mejor_familia_ptr = &familias[f];
                    break;
                }
            }

            if (mejor_familia_ptr && mejor_familia_ptr->simular) {
                double *x_sim = (double*)malloc(datos->n * sizeof(double));
                mejor_familia_ptr->simular(mejor_params, datos->t, datos->F, datos->n, x_sim);

                // Guardar JSON en la subcarpeta 01-json
                guardar_resultados_json_completo(json_dir,
                                                 archivos[a],
                                                 mejor_familia,
                                                 mejor_params,
                                                 mejor_num_params,
                                                 mejor_error,
                                                 ecuacion,
                                                 datos->t,
                                                 datos->x,
                                                 x_sim,
                                                 datos->n,
                                                 maxiter,
                                                 popsize,
                                                 nombres_familias,
                                                 errores_familias,
                                                 num_familias_evaluadas);
                print_step("    Resultados guardados en: %s/%s.json", json_dir, archivos[a]);

                free(x_sim);
            } else {
                print_step("    ADVERTENCIA: No se puede simular para %s", archivos[a]);
            }

            free(mejor_params);
        } else {
            print_step("  ADVERTENCIA: No se encontro un modelo valido para %s", archivos[a]);
        }

        free(nombres_familias);
        free(errores_familias);
        liberar_datos(datos);
    }

    clock_t end_total = clock();
    tiempo_total = (double)(end_total - start_total) / CLOCKS_PER_SEC;
    print_step("Tiempo total de ejecucion: %.2f segundos", tiempo_total);
    print_step("=== EJECUCION FINALIZADA ===");
    print_step("Resultados guardados en: %s", output_dir);

    return 0;
}