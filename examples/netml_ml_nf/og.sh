#!/bin/bash


function usage {
        echo "$0 CPU-LIST SERVICE-ID [-p PRINT] [-n NF-ID] [-m model_num] [-b fixed_batch_size] [-a adaptive_batch_flag] [-s user_slo]"
        echo "$0 -F CONFIG_FILE [-p PRINT]"
        echo ""
        echo "$0 3 0 --> core 3, Service ID 0"
        echo "$0 -F example_config.json -p 1000 --> loads settings from example_config.json and print rate of 1000"
        echo "$0 3,7,9 1 --> cores 3,7, and 9 with Service ID 1"
        echo "$0 3,7,9 1 1000 --> cores 3,7, and 9 with Service ID 1 and Print Rate of 1000"
        exit 1
}

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

if [ -z $1 ]
then
  usage
fi

if [ $1 = "-F" ]
then
  config=$2
  shift 2
  exec sudo $SCRIPTPATH/build/app/bridge -F $config "$@"
fi

cpu=$1
service=$2
inst_id=$3
#print=$4
#file_name=$5
#model_name=$4

shift 3

if [ -z $service ]
then
    usage
fi

while getopts ":p:b:g:f:m:a:s:b:" opt; do
  case $opt in
    p) print="-p $OPTARG";;
    f) file_name="-f $OPTARG";;
    m) model_num="-m $OPTARG";;
    b) batch_size="-b $OPTARG";;
    a) adaptive_batching_flag="-a $OPTARG";;
    s) inference_slo="-s $OPTARG";;
    g) gpu_percent="-g $OPTARG";;
    \?) echo "Unknown option -$OPTARG" && usage
    ;;
  esac
done
echo sudo $SCRIPTPATH/build/app/bridge -l $cpu -n 3 --proc-type=secondary -- -r $service -n $inst_id $file_name $model_num -- $print $batch_size $gpu_percent $adaptive_batching_flag $inference_slo
exec sudo $SCRIPTPATH/build/app/bridge -l $cpu -n 3 --proc-type=secondary -- -r $service -n $inst_id $file_name $model_num -- $print $batch_size $gpu_percent $adaptive_batching_flag $inference_slo
