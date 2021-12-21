#!/bin/bash

#use ./compare.sh compare.csv PEMS_Conv2d_Comparativa/main.py
touch "$1";
FILE_COMPARE= readlink -f $1;

LEARNING_RATES=(1e-6,1e-5,1e-4, 1e-3);

for i in "${LEARNING_RATES[@]}"
do 

    python3 $2 "-d /home/luis/Documentos/pruebaRedes/datasetsMini/2015.csv 
    -v /home/luis/Documentos/pruebaRedes/datasetsMini/2016.csv 
    -t /home/luis/Documentos/pruebaRedes/datasetsMini/2016.csv 
    -ts 70 -vs 30 -l ${i} -cf ${FILE_COMPARE}"

done;