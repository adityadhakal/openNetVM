#ifndef _ONVM_STREAM_H
#define _ONVM_STREAM_H
#include <inttypes.h>
#include <cuda_runtime.h>
//#include "onvm_common.h"
//#ifdef ONVM_GPU

#define MAX_STREAMS 2
#define PARALLEL_EXECUTION 1
#define STREAMS_ENABLED 1
#define DEFAULT_STREAM 0

typedef struct stream_tracker {
	cudaStream_t stream;
	uint8_t status; //0 - being used in 2 executions, 1 being used in 1 execution, 1 slot available, 0 - available
	uint8_t id;	  //0 - id of the stream
} stream_tracker;

/* this function initializes the number of streams desired by the program */
int init_streams(void);

/* this function provides an empty stream */
stream_tracker *give_stream(void);

/* this function returns stream */
void return_stream(stream_tracker *stream);

#endif
//#endif
