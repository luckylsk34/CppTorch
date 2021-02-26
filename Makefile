# SHELL := powershell.exe
# .SHELLFLAGS := -NoProfile -Command
XTENSOR_FLAGS = -I $(XTL_INCLUDE) -I $(XTENSOR_INCLUDE)
CC = clang++
CFLAGS = -std=c++17 $(XTENSOR_FLAGS)

.PHONY: run clean

run: ./build/losstest.exe
	$<

./build/test.exe: tests/tensorTest.cpp ./build/tensor.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/losstest.exe: tests/lossTest.cpp ./build/loss.o 
	$(CC) $(CFLAGS) $^ -o $@

./build/loss.o: src/loss/loss.cpp src/loss/loss.hpp
	$(CC) $(CFLAGS) -c $< -o $@

./build/tensor.o: src/tensor/tensor.cpp src/tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@



clean:
	rm ./build/*
