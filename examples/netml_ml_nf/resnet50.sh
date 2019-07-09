#! /bin/bash
cpu_core=$1
service_id=$2
instance=$3

#model_num=$4
echo "This is NOT an error ./resnet50.sh <CPU CORE> <SERVICE ID> <INSTANCE ID> "
./og.sh $cpu_core $service_id $instance -m 6
