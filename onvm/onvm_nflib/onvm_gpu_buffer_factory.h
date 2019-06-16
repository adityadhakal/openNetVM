#ifndef _ONVM_GPU_BUFFER_FACTORY_
#define _ONVM_GPU_BUFFER_FACTORY_

#ifdef ONVM_GPU
#include <inttypes.h>
#include "onvm_common.h"
#include "onvm_stream.h"
#include "onvm_netml.h"

#define DEV_BUFFER_PARTITIONS MAX_STREAMS //how many partitions dev buffer will have

#define SIZE_OF_A_MEM_BUFFER (MAX_IMAGES_BATCH_SIZE*SIZE_OF_AN_IMAGE_BYTES/DEV_BUFFER_PARTITIONS)

#define MAX_IMAGES_PER_PARTITION (MAX_IMAGES_BATCH_SIZE/DEV_BUFFER_PARTITIONS)


/* a function to resolve the CUDA IPC pointer for GPU image space 
 * Input includes the data pointer to store the image and then a flag saying it
 * is either a GPU pointer or CPU side pointer*/
void resolve_gpu_dev_buffer(void *ptr, void * output_ptr);

/* the device buffer pointer*/
void * input_dev_buffer;
void * output_dev_buffer;


/* the cpu side buffer */
void * input_cpu_buffer;
void * output_cpu_buffer;

void resolve_cpu_side_buffer(void *input_buffer, void * output_buffer);

struct cpu_buffer_state
{

  /* a number counting how many images are already occupied */
  uint8_t occupancy_indicator;

  /* the buffer's address itself */
  void *cpu_buffers[MAX_IMAGES_PER_PARTITION];
  /* the output buffer addresses */
  void *output_cpu_buffers[MAX_IMAGES_PER_PARTITION];
};

/* the dev buffer state local variable */
struct cpu_buffer_state cpu_buffer_state[DEV_BUFFER_PARTITIONS];

void give_cpu_addresses(uint8_t batch_id, void ** input_buffer, void ** output_buffer);

void return_cpu_buffer(uint8_t batch_id);

struct dev_buffer_state
{
  
  /* a number counting how many images are already occupied */
  uint8_t occupancy_indicator;
  
  /* the buffer's address itself */
  void *dev_buffers[MAX_IMAGES_PER_PARTITION];
  /* the output buffer addresses */
  void *output_dev_buffers[MAX_IMAGES_PER_PARTITION];
};

/* the dev buffer state local variable */
struct dev_buffer_state buffer_state[DEV_BUFFER_PARTITIONS];

/* provides the device address */
void give_device_addresses(uint8_t batch_id, void ** input_buffer, void ** output_buffer);

/* return the used device buffer back */
void return_device_buffer(uint8_t batch_id);

void resolve_gpu_dev_buffer_pointer(void * input, void * dev_output);

#endif
#endif
