#!/bin/bash

# REMOVE default cuda
module rm cuda90/toolkit/9.0.176
module rm openblas/dynamic/0.2.18
module rm nccl/cuda80/2.1.2
module rm gcc

# LOAD MODULES
module load cuda10.1
module load cuDNN/cuda10.1/7.6.4
module load cuda10.0/blas/10.0.130 
module load cmake/3.15.4
module load gcc/6.3.0

# set ENV
export CC=gcc
export CXX=g++