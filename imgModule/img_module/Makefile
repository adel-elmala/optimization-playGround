CC = gcc
CFlags = -msse2 -ggdb
LIBS = -lm 
INCLUDE = -I "./include/"
INCLUDE += -I "../utils/logger/include/"
INCLUDE += -I "../utils/timer/include/"
# INCLUDE += -I "../utils/plotter/include/"
# INCLUDE += -I "../utils/plotter/libs/gnuPlot/"

headers = 	./include/stb_image.h \
			./include/stb_image_write.h \
			./include/stb_image_resize.h

src = main.c \
	src/imgProcessingModule.c \
	../utils/logger/src/logger.c \
	# ../utils/plotter/src/plotter.c \
	# ../utils/plotter/libs/gnuPlot/gnuplot_i.c 

# VPATH = .:src:bin

imgModule:$(src) $(headers) 
	$(CC) $(CFlags) $(INCLUDE) $(src) $(LIBS)  -o $@ 

.PHONY:clean,valgrind

clean:
	rm -r ./bin/*

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all ./bin/imgModule


