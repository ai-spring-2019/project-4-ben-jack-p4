#!/bin/bash

for DATA in dna.csv car.csv wine_2.csv
do
  for EPOCHS in 100 500 1000
  do
    for LAYERS in [10,10] [10,15,10] [5,5,5,5] [5,10,10] [15,10,5]
    do
      python3 project4.py --validation multi \
                          --data $DATA \
                          --output final_all_runs_multi.csv \
                          --k-val 5 \
                          --layers $LAYERS \
                          --epochs $EPOCHS \
                          --noheader
    done
  done
done


for DATA in banana.csv breast-cancer-wisconsin-normalized.csv increment-3-bit.csv
do
  for EPOCHS in 100 500 1000
  do
    for LAYERS in [10,10] [10,15,10] [5,5,5,5] [5,10,10] [15,10,5]
    do
      python3 project4.py --validation multi \
                          --data $DATA \
                          --output final_all_runs_binary.csv \
                          --k-val 5 \
                          --layers $LAYERS \
                          --epochs $EPOCHS \
                          --noheader
    done
  done
done
