!# /bin/bash
gdb --args ./build/app/bridge -l 10 -n 3 --proc-type=secondary -- -r 1 -n 1 -f /home/adhak001/openNetVM-dev/ml_models/ResNet50_ImageNet_CNTK.model -m 1 -- -b 4 -g 100
