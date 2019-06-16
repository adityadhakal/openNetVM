#! /bin/bash

./perf_client -b 8 -c 8 -d -p 5000 -l 500 -m resnet50_netdef &
nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv -lms 180 > gpu_utilization.dat
