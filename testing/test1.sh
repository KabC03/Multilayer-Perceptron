#!/bin/bash


clang++ ./testing/test1.c++ -O3 -o ./bin/test1
./bin/test1

python3 ./testing/plot_single.py





