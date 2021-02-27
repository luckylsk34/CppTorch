# SHELL := powershell.exe
# .SHELLFLAGS := -NoProfile -Command
XTENSOR_FLAGS = -I $(XTL_INCLUDE) -I $(XTENSOR_INCLUDE) -I $(XBLAS_INCLUDE)
CC = clang++
CFLAGS = -std=c++17 $(XTENSOR_FLAGS) -DXTENSOR_USE_FLENS_BLAS

.PHONY: run clean

run: ./build/trainTest.exe
	$<

./build/trainTest.exe: tests/trainTest.cpp ./build/layers.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/optimTest.exe: tests/optimTest.cpp ./build/layers.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/neuralnetTest.exe: tests/neuralnetTest.cpp ./build/layers.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/layertest.exe: tests/layerTest.cpp ./build/layers.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/dataTest.exe: tests/dataTest.cpp ./build/data.o
	$(CC) $(CFLAGS) $^ -o $@

./build/tensorTest.exe: tests/tensorTest.cpp ./build/tensor.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/losstest.exe: tests/lossTest.cpp ./build/loss.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/layers.o: src/layers/layers.cpp src/layers/layers.hpp src/layers/nn.hpp
	$(CC) $(CFLAGS) -c $< -o $@

./build/loss.o: src/loss/loss.cpp src/loss/loss.hpp
	$(CC) $(CFLAGS) -c $< -o $@

./build/data.o: src/data/data.cpp src/data/data.hpp
	$(CC) $(CFLAGS) -c $< -o $@

./build/tensor.o: src/tensor/tensor.cpp src/tensor/tensor.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm ./build/*
