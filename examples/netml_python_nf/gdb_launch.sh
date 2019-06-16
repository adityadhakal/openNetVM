!# /bin/bash
gdb --args ./build/app/bridge -l 10 -n 3 --proc-type=secondary -- -r 1 -n 1 -m 6
