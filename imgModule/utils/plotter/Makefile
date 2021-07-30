CC = gcc
CFLAGS = -Wall -pedantic 

SRC = 	src/plotter.c \
		./libs/gnuPlot/gnuplot_i.c

HEADERS = include/plotter.h

INCLUDE = -I "./include" 
INCLUDE += -I "./libs/gnuPlot/" 
# INCLUDE += -I "../../common"

BIN = ./bin/plotter.out

plotter:$(SRC) $(HEADERS)
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
	valgrind --leak-check=full --show-leak-kinds=all ./bin/plotter
	



