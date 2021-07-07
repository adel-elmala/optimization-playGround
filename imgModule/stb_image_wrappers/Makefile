CC = gcc
CFlags = -msse2
LIBS = -lm 
INCLUDE = -I "./include/"

headers = 	./include/stb_image.h \
			./include/stb_image_write.h \
			./include/stb_image_resize.h

src = main.c \
	src/imgProcessingModule.c

 
imgModule:$(src) $(headers) 
	$(CC) $(CFlags) $(src) $(LIBS) $(INCLUDE) -o $@ 
	mv $@ ./bin/$@

.PHONY:clean,valgrind

clean:
	rm -r ./bin/*

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all ./bin/imgModule


