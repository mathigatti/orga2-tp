CC=gcc 
# Quitar -g (opcion de debugeo)
CFLAGS=-std=c99 -lm -O3
DEPS = tensorOps.h
OBJ = tensorOps.o

all: clean test

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

tensorOps_asm.o: tensorOps.asm $(DEPS)
	nasm -f elf64 $< -o tensorOps_asm.o

test: $(OBJ) tensorOps_asm.o
	gcc -o $@ test.c $^ $(CFLAGS)
	
clean:
	rm -f *.o *~
	rm -f test
