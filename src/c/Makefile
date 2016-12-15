CC=gcc 
CFLAGS=-std=c99 -lm 
DEPS = nn.h
OBJ = main.o

all: neural-net

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

neural-net: $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

clean:
	rm -rf *.o *~
