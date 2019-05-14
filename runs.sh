#!/bin/bash

# for DATA in wine_2.csv
# do
#   for EPOCHS in 100
#   do
#     for LAYERS in [10,10] [10,15,10] [5,5,5,5] [5,10,10] [15,10,5] [20] [20,20] []
#     do
#       python3 project4.py --validation multi \
#                           --data $DATA \
#                           --output final_all_runs_multi.csv \
#                           --k-val 5 \
#                           --layers $LAYERS \
#                           --epochs $EPOCHS \
#                           --noheader
#     done
#   done
# done


for DATA in breast-cancer-wisconsin-normalized.csv
do
  for EPOCHS in 100
  do
    for LAYERS in [10,10] [10,15,10] [] [5] [10] [20] [40] [20,20,20] [40,40]
    do
      python3 project4.py --validation binary \
                          --data $DATA \
                          --output breast_cancer_out.csv \
                          --k-val 5 \
                          --layers $LAYERS \
                          --epochs $EPOCHS \
                          --noheader
    done
  done
done
