#!/bin/bash

cat $1 | grep "TD" | cut -d"V" -f 2|cut -f2 > ./Results/vavg.txt
cat $1 | grep "reward:" | cut -d "," -f 2 | cut -d " " -f 3 > ./Results/reward.txt
cat $1 | grep "TD" | cut -d"T" -f 2|cut -f2> ./Results/TD.txt
cat $1 | grep "nn.Sequential" | cut -d" " -f 3 > ./Results/conv1.txt
cat $1 | grep "nn.Sequential" | cut -d" " -f 5 > ./Results/conv2.txt
cat $1 | grep "nn.Sequential" | cut -d" " -f 7 > ./Results/conv3.txt
cat $1 | grep "nn.Sequential" | cut -d" " -f 9 > ./Results/lin1.txt
cat $1 | grep "nn.Sequential" | cut -d" " -f 11 |  cut -d"]" -f1 > ./Results/lin2.txt



th ./Results/Plot.lua


