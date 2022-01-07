#!/bin/bash


python3 ConvTraff/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 70000 -vs 30000 -tw 12 -fw 1 -e 25 > ResultsFile/ConvTraff/res7000.txt

python3 ConvTraff_variable_input/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 70000 -vs 30000 -tw 12 -fw 1 -e 25 > ResultsFile/ConvTraff_variable_input/res7000.txt
