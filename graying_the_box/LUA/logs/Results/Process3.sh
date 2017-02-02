#!/bin/bash

cat $1 | grep "TD" | cut -d"V" -f 2|cut -f2 > ./Results/vavg1.txt
cat $1 | grep "reward:" | cut -d "," -f 2 | cut -d " " -f 3 > ./Results/reward1.txt
cat $1 | grep "TD" | cut -d"T" -f 2|cut -f2> ./Results/TD1.txt

cat $2 | grep "TD" | cut -d"V" -f 2|cut -f2 > ./Results/vavg2.txt
cat $2 | grep "reward:" | cut -d "," -f 2 | cut -d " " -f 3 > ./Results/reward2.txt
cat $2 | grep "TD" | cut -d"T" -f 2|cut -f2> ./Results/TD2.txt

cat $3 | grep "TD" | cut -d"V" -f 2|cut -f2 > ./Results/vavg3.txt
cat $3 | grep "reward:" | cut -d "," -f 2 | cut -d " " -f 3 > ./Results/reward3.txt
cat $3 | grep "TD" | cut -d"T" -f 2|cut -f2> ./Results/TD3.txt

th ./Results/Plot3.lua


