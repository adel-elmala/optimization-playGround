CC = gcc
CFLAGS = -Wall -pedantic 

SRC = 	src/testTimer.c 

HEADERS = include/adelTimer.h

INCLUDE = -I "./include" 
# INCLUDE += -I "../../common"

BIN = ./bin/testTimer.out

testTimer:$(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDE) $(SRC) -o $@ 
	mv $@ ./bin


.PHONY:setupDir,clean,valgrind
setupDir:
	mkdir bin src include 
	mv *.c ./src
	mv *.h ./include
clean:
	rm bin/*
valgrind:
	valgrind --leak-check=full --show-leak-kinds=all ./bin/main



