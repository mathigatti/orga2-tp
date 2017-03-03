CC=gcc 
# Quitar -g (opcion de debugeo)
CFLAGS=-std=c99 -lm -O3 -g 
DEPS = nn.h txtReader.c
OBJ = main.o 

all: neural-net

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

neural-net: $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

clean:
	rm -rf *.o *~

# Para compilar funciones en ASM
# nasm -f elf64 nn.asm -o nn_asm.o
# gcc -o neural-net main.c nn_asm.o -std=c99 -lm -g