CC = gcc
CFlags = -msse2 -ggdb
LIBS = -lm 
INCLUDE = -I "./include/"
INCLUDE += -I "../utils/logger/include/"
INCLUDE += -I "../utils/timer/include/"

headers = 	./include/stb_image.h \
			./include/stb_image_write.h \
			./include/stb_image_resize.h

src = main.c \
	src/imgProcessingModule.c \
	../utils/logger/src/logger.c	
# VPATH = .:src:bin

imgModule:$(src) $(headers) 
	$(CC) $(CFlags) $(src) $(LIBS) $(INCLUDE) -o $@ 
# bin = main.o imgProcessingModule.o logger.o
# imgModule:$(bin) $(headers) 
# 	$(CC) $(CFlags) $(bin) $(LIBS) $(INCLUDE) -o $@ 
# 	# mv $@ ./bin/$@

# main.o:main.c ./bin/imgProcessingModule.o ./bin/logger.o
# 	$(CC) -c $(CFlags) $^ $(LIBS) $(INCLUDE) -o $@ 
# 	mv $@ ./bin/$@


# imgProcessingModule.o:src/imgProcessingModule.c
# 	$(CC) -c $(CFlags) $< $(LIBS) $(INCLUDE) -o $@ 
# 	mv $@ ./bin/$@

# logger.o:../utils/logger/src/logger.c
# 	$(CC) -c $(CFlags) $< $(LIBS) $(INCLUDE) -o $@ 
# 	mv $@ ./bin/$@

.PHONY:clean,valgrind

clean:
	rm -r ./bin/*

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all ./bin/imgModule


