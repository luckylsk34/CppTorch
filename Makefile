# SHELL := powershell.exe
# .SHELLFLAGS := -NoProfile -Command
XTENSOR_FLAGS = -I $(XTL_INCLUDE) -I $(XTENSOR_INCLUDE)
CFLAGS = -std=c++17 $(XTENSOR_FLAGS)

.PHONY: run clean

run: ./build/test
	$<

./build/test: tests/tensorTest.cpp ./build/tensor.o
	g++ $(CFLAGS) $< $(word 2,$^) -o $@

./build/tensor.o: src/tensor/tensor.cpp src/tensor/tensor.h
	g++ -c $(CFLAGS) $< -o $@

clean:
	rm ./build/*
