#!/bin/bash

#use ./compare.sh compare.csv PEMS_Conv2d_Comparativa/main.py


python3 "ConvTraff/main.py -d /home/luis.lopezc/data/2015.csv -v /home/luis.lopezc/data/2016.csv -t /home/luis.lopezc/data/2016.csv -ts 104988 -vs 200000 -tw 12 -fw 1 -e 25 -b 256"

