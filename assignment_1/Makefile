CC=gcc
DEPS=average.h
OBJ=average.o main.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CLFAGS)

average: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) 
