#include <inttypes.h>
#include <stdio.h>
extern "C"{
#include "onvm_netml.h"
  #include "onvm_ml_libraries.h"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define NUM_THREADS_PER_BLOCK 32


/* Function to help with nice error message */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

/* the GPU kernel to perform data transfer */
__global__ void transfer_data(void *data_ptrs, void *destination){

  /* the ids for block and thread */
  uint16_t bid = blockIdx.x;
  uint16_t tid = threadIdx.x;
  int i;

  /* now we need to give each block the chunk info they will extract data from */
    //here we only have 1 thread...
  chunk_copy_info_t data_chunk = ((chunk_copy_info_t *) data_ptrs)[bid];

  //  chunk_copy_info_t *data_chunk = (chunk_copy_info_t *) (data_ptrs[bid]);
  // printf("Data pointer for chunk copy info %p \n",data_chunk);

  /* find out the size of each chunk .. all threads need to know this as some threads will be disabled eventually*/
  uint32_t size = data_chunk.image_chunk.size_in_bytes;
  uint32_t offset = data_chunk.image_chunk.start_offset;
  float * data = (float*)(data_chunk.src_cpy_ptr);
  float * empty_buffer = (float *)((char *) destination+offset);

  uint8_t num_of_data_points = size/SIZE_OF_EACH_ELEMENT;


  /* first find out how many data points each thread has to access and write */
  uint8_t num_data_points_per_thread = num_of_data_points/NUM_THREADS_PER_BLOCK;

  if(num_of_data_points%NUM_THREADS_PER_BLOCK)
    num_data_points_per_thread += 1;

  /* now move the data to the GPU buffer one by one */ 
  for(i = 0; i<num_data_points_per_thread; i++){
    uint16_t thread_position = tid+(NUM_THREADS_PER_BLOCK*i);
    if(thread_position<num_of_data_points){
      empty_buffer[thread_position] = data[thread_position];
    }
  }
}


// a kernel that does 1 threadblock per packet
__global__ void transfer_data_light(void * data_ptrs, void *destination){

  uint16_t bid  = blockIdx.x;

  //here we only have 1 thread...
  chunk_copy_info_t data_chunk = ((chunk_copy_info_t *) data_ptrs)[bid];
  //printf("ptr of data %p \t",data_chunk);

  //now find the bytes of the packet.
  //int size2 = data_chunk.image_chunk.size_in_bytes;//data_chunk->src_cpy_ptr;
  //printf("Size of data %d \n",size2);

  uint32_t size = data_chunk.image_chunk.size_in_bytes;
  
  uint32_t offset = data_chunk.image_chunk.start_offset;
  
  //where the data should be written
  char *empty_buffer = ((char *) destination)+offset;
  char * data = (char *) (data_chunk.src_cpy_ptr);
  //now loop through the number of bytes and write it in the arrary

  memcpy(empty_buffer,data,size);
  /*
  int i;
  for(i = 0; i<size; i++){
    empty_buffer[i] = data[i];
  }
  */
}

// a kernel that does 1 thread per packet
__global__ void transfer_data_light2(void * data_ptrs, void *destination, int num_of_payload_data){

  //uint16_t bid  = blockIdx.x;
  //uint16_t tid = threadIdx.x;
  uint16_t sid = blockIdx.x*blockDim.x + threadIdx.x; //bid*NUM_THREADS_PER_BLOCK;

  if(sid<num_of_payload_data){
  
  chunk_copy_info_t data_chunk = ((chunk_copy_info_t *) data_ptrs)[sid];
  //printf("ptr of data %p \t",data_chunk);
  
  //now find the bytes of the packet.
  //void *src_ptr = data_chunk.src_cpy_ptr;
  uint32_t size = data_chunk.image_chunk.size_in_bytes;
  uint32_t offset = data_chunk.image_chunk.start_offset;
  
  //printf("Size of data %p \n",src_ptr);
  //where the data should be written
  char *empty_buffer = ((char *) destination)+offset;
  char * data = (char *) (data_chunk.src_cpy_ptr);

  //TODO:CHECK WITH SAMEER. memcpy should copy the size of the data in packet not something else
  //memcpy(empty_buffer,data,sizeof(char)*num_of_payload_data);
  //memcpy(empty_buffer, data, sizeof(char)*size);

  memcpy(empty_buffer, data, size);

  
  //now loop through the number of bytes and write it in the arrary
  /*
  int i;
  for(i = 0; i<size; i++){
    empty_buffer[i] = data[i];
  }
  */
  }
}

__global__ void print_data(void *destination){
  float * data_array = (float *) destination;
  int i ;
  printf("---------- data in image buffer ------------\n");
  for(i = 0; i<100; i++){
    printf("%f ", data_array[i]);
  }
  printf("\n");
}

extern "C"
void transfer_to_gpu_copy(void * data_ptrs, int num_of_payload_data, void *cpu_destination, void * gpu_destination, cudaStream_t *stream){
  // this is simple copy operation by one CPU thread... just copy the data to cpu destination
  // and cudamemcpy them to GPU
  struct timespec begin_cpu_copy, end_cpu_copy, begin_cudamemcpy, end_cudamemcpy;
  int i = 0;
  size_t total_bytes =0;
  clock_gettime(CLOCK_MONOTONIC, &begin_cpu_copy);
  for(i = 0; i<num_of_payload_data; i++){
    chunk_copy_info_t data_chunk = ((chunk_copy_info_t *) data_ptrs)[i];
    //void * src_ptr = data_chunk.src_cpy_ptr;
    uint32_t size = data_chunk.image_chunk.size_in_bytes;
    uint32_t offset = data_chunk.image_chunk.start_offset;
    total_bytes += size;
    char *empty_buffer = ((char *) cpu_destination)+offset;
    char *data = (char *) (data_chunk.src_cpy_ptr);
    memcpy(empty_buffer,data,size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end_cpu_copy);
  
  //now as the packets data is copied into CPU buffer, we can cuda memcpy it into GPU
  //cudaMemcpyAsync(gpu_destination, cpu_destination, total_bytes, cudaMemcpyHostToDevice, *stream);
  //clock_gettime(CLOCK_MONOTONIC, &begin_cudamemcpy);
  cudaMemcpyAsync(gpu_destination, cpu_destination, total_bytes, cudaMemcpyHostToDevice, *stream);
  //clock_gettime(CLOCK_MONOTONIC, &end_cudamemcpy);
  //float cpu_copy_time = (end_cpu_copy.tv_sec-begin_cpu_copy.tv_sec)*1000.0+(end_cpu_copy.tv_nsec-begin_cpu_copy.tv_nsec)/1000000.0;
  //float cudamemcpy_time = (end_cudamemcpy.tv_sec-begin_cudamemcpy.tv_sec)*1000.0+(end_cudamemcpy.tv_nsec-begin_cudamemcpy.tv_nsec)/1000000.0;
  //printf("CPU copy time (ms),%f, CUDA memcpyasync(ms),%f,",cpu_copy_time, cudamemcpy_time);
  //printf("CPU copy time (ms), %f\n",cpu_copy_time);
}



// ONVM facing function to transfer to GPU
extern "C"
void transfer_to_gpu(void *data_ptrs,  int num_of_payload_data,void *destination,cudaStream_t *stream){
  /* Find the number of blocks and threads to use. 
   * It might be better to use lot of blocks with smaller number of threads... they might work better parellely with 
   * execution running in another stream.
  */

  //check if the pointers are reaching ehre or not..
  //int i = 0;
  //printf("num of payload %d \n",num_of_payload_data);
  /* 
  for(i = 0; i<num_of_payload_data; i++){
    printf("the pointer to chunk copy info %p \n",data_ptrs[i]);
  }
  */

  
  //let's get 1 block for every packet.
  //struct timespec begin_kernel, end_kernel;
  //clock_gettime(CLOCK_MONOTONIC, &begin_kernel);
  
  //transfer_data<<<num_of_payload_data, NUM_THREADS_PER_BLOCK, 0, *stream>>>(data_ptrs, destination);
  
  transfer_data_light<<<num_of_payload_data, 1, 0, *stream>>>(data_ptrs, destination);

  //transfer_data_light<<<2,1,0,*stream>>>(data_ptrs,destination);
  //transfer_data_light2<<<1,2,0,*stream>>>(data_ptrs,destination,2);
  
  //transfer_data_light2<<<(num_of_payload_data+NUM_THREADS_PER_BLOCK)/(NUM_THREADS_PER_BLOCK), NUM_THREADS_PER_BLOCK, 0, *stream>>>(data_ptrs, destination, num_of_payload_data);
  //cudaStreamSynchronize(*stream);
   
  //clock_gettime(CLOCK_MONOTONIC, &end_kernel);
  
  //float time_to_transfer = (end_kernel.tv_sec-begin_kernel.tv_sec)*1000.0+(end_kernel.tv_nsec-begin_kernel.tv_nsec)/1000000.0;
  //printf("Time to transfer data NetML way(ms),%f,",time_to_transfer);
  
  //cudaDeviceSynchronize();
  //cudaError_t error = cudaGetLastError();
  //printf("Last error %d \n",error);
  /* now lets print out the buffer to see if the data is copied */
  //print_data<<<1,1>>>(destination);
  //cudaDeviceSynchronize();
}

__global__ void check_data(void *ptr){
  image_batched_aggregation_info_t * image_list = (image_batched_aggregation_info_t *) ptr;
  printf("ready mask hahaha %d \n",image_list->ready_mask);

  int bytes_count = image_list->images[0].bytes_count;
  int status = image_list->images[0].usage_status;
  int packet_count = image_list->images[0].packets_count;

  printf("bytes count %d status %d packets_count %d\n", bytes_count, status, packet_count);

  void * source_addr = image_list->images[0].image_info.copy_info[0].src_cpy_ptr;
  int start_offset  = image_list->images[0].image_info.copy_info[0].image_chunk.start_offset;
  int size_bytes = image_list->images[0].image_info.copy_info[0].image_chunk.size_in_bytes;
  printf("source addr %p start offset %d size %d\n", source_addr, start_offset, size_bytes);
  
}
void check_kernel(void *ptr){
  check_data<<<1,1>>>(ptr);
  cudaDeviceSynchronize();
}


// https://www.cs.virginia.edu/~mwb7w/cuda_support/pinned_tradeoff.html

