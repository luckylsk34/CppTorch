# SHELL := powershell.exe
# .SHELLFLAGS := -NoProfile -Command
XTENSOR_FLAGS = -I $(XTL_INCLUDE) -I $(XTENSOR_INCLUDE)
CC = clang++
CFLAGS = -std=c++20 $(XTENSOR_FLAGS)

.PHONY: run clean

run: ./build/dataTest.exe
	$<

./build/dataTest.exe: tests/dataTest.cpp ./build/data.o
	$(CC) $(CFLAGS) $^ -o $@

./build/data.o: src/data/data.cpp src/data/data.hpp
	$(CC) $(CFLAGS) -c $< -o $@

./build/test.exe: tests/tensorTest.cpp ./build/tensor.o
	$(CC) $(CFLAGS) $^ -o $@

./build/tensor.o: src/tensor/tensor.cpp src/tensor/tensor.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm ./build/*
