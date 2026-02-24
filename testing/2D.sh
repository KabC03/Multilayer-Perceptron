#!/bin/bash


clang++ ./testing/2D.c++ -O3 -o ./bin/2D
./bin/2D

python3 ./testing/plot_2D.py





