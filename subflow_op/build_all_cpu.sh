#!/bin/bash

./build_sub_conv2d_cpu.sh
./build_sub_matmul_cpu.sh

cp *.so ..
