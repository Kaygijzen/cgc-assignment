.PHONY: all clean

SRC=src/

INCLUDES=-Iexternal/argparse-2.9/include -Iexternal/libnpy/include
CFLAGS=-std=c++17 -O3 -march=native -Wall -Wextra -Wnarrowing -Wparentheses #-Werror -Wno-unused-parameter
CC=g++
BINS=bin/cgc_serial bin/cgc_mpi bin/cgc_cuda
MPICC=mpic++
NVCC=nvcc

all: $(BINS) Makefile

bin/cgc_serial: $(SRC)/serial.cpp $(SRC)/common.h
	$(CC) -o $@ $(SRC)/serial.cpp $(CFLAGS) $(INCLUDES)

bin/cgc_mpi: $(SRC)/mpi.cpp $(SRC)/common.h
	$(MPICC) -o $@ $(SRC)/mpi.cpp $(CFLAGS) $(INCLUDES)


bin/cgc_cuda: bin/cgc_kernel.o
	$(MPICC) bin/cgc_kernel.o $(SRC)/cuda.cpp -o $@ $(CFLAGS) $(INCLUDES) -lcudart -lcurand

bin/cgc_kernel.o: $(SRC)/cuda/module.cu $(SRC)/cuda/module.h 
	nvcc -c -g $(SRC)/cuda/module.cu -o $@ -I -dlink

clean:
	rm -rf $(BINS)

