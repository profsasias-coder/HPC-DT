CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -ljson-c
TARGET = descubridor_c
OBJS = solver.o familias_fisicas.o familias_matematicas.o optimizador.o io.o utils.o descubridor_c.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)