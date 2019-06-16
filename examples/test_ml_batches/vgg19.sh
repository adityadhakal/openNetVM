#! /bin/bash
instance=$1
batch_size=$2
./og.sh 10 1 $instance -f /home/adhak001/openNetVM-dev/ml_models/VGG19_ImageNet_Caffe.model -m 2 -b $batch_size &
sleep 5
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -lms 200 > "vgg19_gpu_utilization_batch_${2}.dat"
