# SHELL := powershell.exe
# .SHELLFLAGS := -NoProfile -Command
XTENSOR_FLAGS = -I $(XTL_INCLUDE) -I $(XTENSOR_INCLUDE)
CC = clang++
CFLAGS = -std=c++17 $(XTENSOR_FLAGS)

.PHONY: run clean

run: ./build/test.exe
	$<

./build/test.exe: tests/tensorTest.cpp ./build/tensor.o
	$(CC) $(CFLAGS) $^ -o $@

./build/tensor.o: src/tensor/tensor.cpp src/tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm ./build/*
