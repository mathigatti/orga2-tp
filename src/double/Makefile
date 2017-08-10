CC=gcc 
# Quitar -g (opcion de debugeo)
CFLAGS=-std=gnu11 -lm -O3
DEPS = nn.h helpers.h tensorOps.h
OBJ = nn.o helpers.o 

all: clean c_version asm_version

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

tensorOps_c.o: tensorOps.c $(DEPS)
	gcc -c -o $@ $< $(CFLAGS)

tensorOps_asm.o: tensorOps.asm $(DEPS)
	nasm -f elf64 $< -o tensorOps_asm.o

c_version: $(OBJ) tensorOps_c.o
	gcc -o $@ $^ $(CFLAGS)

asm_version: $(OBJ) tensorOps_asm.o
	gcc -o $@ $^ $(CFLAGS)
	
clean:
	rm -f *.o *~
	rm -f c_version
	rm -f asm_version