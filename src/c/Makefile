CC=gcc
CFLAGS=-g -Wall -pedantic -m32 -I../include/

all: red-neuronal.o

red-neuronal.o: main.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf *.o