#!/bin/bash


python3 ConvTraff/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 10000 -vs 3000 -tw 12 -fw 1 -e 20 > ResultsFile/ConvTraff/res7000.txt

python3 ConvTraff_normal_layers/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 10000 -vs 3000 -tw 12 -fw 1 -e 20 > ResultsFile/ConvTraff_normal_layers/res7000.txt

python3 ConvTraff_depth_layers/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 10000 -vs 3000 -tw 12 -fw 1 -e 20 > ResultsFile/ConvTraff_depth_layers/res7000.txt

python3 ConvTraff_variable_input/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 10000 -vs 3000 -tw 12 -fw 1 -e 20 > ResultsFile/ConvTraff_variable_input/res7000.txt

python3 ConvTraff_transfer_learning_base/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-3/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-3/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-3/2016.csv -ts 10000 -vs 3000 -tw 12 -fw 1 -e 20 > ResultsFile/ConvTraff_transfer_learning_base/res7000.txt

python3 ConvTraff_transfer_learning/main.py -d /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2015.csv -v /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -t /home/uwicore/Documentos/pems-traffic-prediction-datasets/I5-S-4/2016.csv -ts 10000 -vs 3000 -tw 12 -fw 1 -e 20 > ResultsFile/ConvTraff_transfer_learning/res7000.txt