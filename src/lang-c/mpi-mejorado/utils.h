#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <time.h>
#include <stdarg.h>

// Imprime timestamp con formato [YYYY-MM-DD HH:MM:SS]
void print_timestamp(void);

// Imprime mensaje con timestamp: [YYYY-MM-DD HH:MM:SS] mensaje
void print_step(const char *format, ...);

#endif