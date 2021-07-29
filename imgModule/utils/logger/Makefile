CC = gcc
CFLAGS = -Wall -pedantic 

SRC = 	src/testLogger.c \
		src/logger.c


HEADERS = include/logger.h \
		  include/loggerBase.h

INCLUDE = -I "./include" 
# INCLUDE += -I "../../common"

BIN = ./bin/testLogger.out

testLogger:$(SRC) $(HEADERS)
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



