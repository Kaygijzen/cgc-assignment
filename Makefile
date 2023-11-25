.PHONY: all clean

SRC=src/

INCLUDES=-Iexternal/argparse-2.9/include -Iexternal/libnpy/include
CFLAGS=-std=c++17 -O3 -march=native -Wall -Wextra -Wnarrowing -Wparentheses #-Werror -Wno-unused-parameter
CC=g++
BINS=bin/cgc_serial bin/cgc_mpi bin/hello_world
MPICC=mpic++

all: $(BINS) Makefile

bin/cgc_serial: $(SRC)/serial.cpp $(SRC)/common.h
	$(CC) -o $@ $(SRC)/serial.cpp $(CFLAGS) $(INCLUDES)

bin/cgc_mpi: $(SRC)/mpi.cpp $(SRC)/common.h
	$(MPICC) -o $@ $(SRC)/mpi.cpp $(CFLAGS) $(INCLUDES)

bin/hello_world: $(SRC)/hello_world.cpp
	$(MPICC) -o $@ $(SRC)/hello_world.cpp $(CFLAGS)

# cgc_cuda:

clean:
	rm -rf $(BINS)

