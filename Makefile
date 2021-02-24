# SHELL := powershell.exe
# .SHELLFLAGS := -NoProfile -Command
XTENSOR_FLAGS = -I ${XTL_INCLUDE} -I ${XTENSOR_INCLUDE}
CFLAGS = -std=c++17 $(XTENSOR_FLAGS)

run: ./build/test
	./build/test

./build/test: tests/tensorTest.cpp ./build/tensor.o
	g++ $(CFLAGS) tests/tensorTest.cpp  ./build/tensor.o -o ./build/test


./build/tensor.o: src/tensor/tensor.cpp src/tensor/tensor.h
	g++ -c $(CFLAGS) src/tensor/tensor.cpp -o ./build/tensor.o

clean: 
	rm ./build/*.exe ./build/*.o
