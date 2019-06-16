#! /bin/bash
instance=$1
batch_size=$2
gpu_percent=$3
./og.sh 10 1 $instance -f /home/adhak001/openNetVM-dev/ml_models/ResNet50_ImageNet_CNTK.model -m 1 -b $batch_size -g $gpu_percent
