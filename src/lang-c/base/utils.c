#include "utils.h"
#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <unistd.h>  // para fflush

void print_timestamp(void) {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    printf("[%s] ", buffer);
    fflush(stdout);
}

void print_step(const char *format, ...) {
    print_timestamp();
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}