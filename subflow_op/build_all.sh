#!/bin/bash

rm *.so
rm ../*.so
./build_sub_conv2d.sh
./build_sub_matmul.sh

cp *.so ..
