#! /bin/bash

for models in alexnet resnet50 vgg19
do
    output_file="${models}_summary.csv"
    rm $output_file

    printf "#Data for %s "$models >>$output_file
    printf "\n#rows 1-MPS only, 2-Netml, 3-2 streams, 4-batch of 2, 5-batch of 4, 6-batch of 8, 7 adaptive batching\n">>$output_file
    printf "#median throughput, std throughput, median gpu_latency, std gpu_latency, median cpu latency, std cpu latency\n">>$output_file
    for files in *$models.*
    do
	datamash -t, median 1 sstdev 1 median 2 sstdev 2 median 3 sstdev 3 <$files >>$output_file
    done
done
