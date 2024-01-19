.PHONY: all clean

SRC=src/

INCLUDES=-Iexternal/argparse-2.9/include -Iexternal/libnpy/include
CFLAGS=-std=c++17 -O3 -march=native -Wall -Wextra -Wnarrowing -Wparentheses #-Werror -Wno-unused-parameter
CC=g++
BINS=cgc_serial cgc_mpi cgc_cuda
MPICC=mpic++
NVCC=nvcc

all: $(BINS) Makefile

cgc_serial: $(SRC)/serial.cpp $(SRC)/common.h
	$(CC) -o $@ $(SRC)/serial.cpp $(CFLAGS) $(INCLUDES)

cgc_mpi: $(SRC)/mpi.cpp $(SRC)/common.h
	$(MPICC) -o $@ $(SRC)/mpi.cpp $(CFLAGS) $(INCLUDES)

cgc_openmp: $(SRC)/mpi_openmp.cpp $(SRC)/common.h
	$(MPICC) -fopenmp -o $@ $(SRC)/mpi_openmp.cpp $(CFLAGS) $(INCLUDES)


cgc_cuda: cgc_kernel.o
	$(MPICC) cgc_kernel.o $(SRC)/cuda.cpp -o $@ $(CFLAGS) $(INCLUDES) -lcudart -lcurand

cgc_kernel.o: $(SRC)/cuda/module.cu $(SRC)/cuda/module.h 
	nvcc -c -g $(SRC)/cuda/module.cu -o $@ -I -dlink

clean:
	rm -rf $(BINS)

