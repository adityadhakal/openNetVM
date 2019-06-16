#ifndef _ONVM_IMAGES_H
#define _ONVM_IMAGES_H

#define NUM_IN_PKTS 96 //how many numbers are in payload
#define NUM_SIZE 4 //4 bytes number
#define IMAGE_NUM_ELE 3*224*224
#define IMAGE_SIZE 3*224*224*4
#define NUM_OF_PKTS IMAGE_SIZE/(NUM_IN_PKTS*NUM_SIZE) //how many packets an image makes
#define IMAGE_BATCH 36 //Max batch size
#define IMAGENET_OUTPUT_SIZE 1000 //output size of a single image inference.
#define NF_IMAGE_STATS_PERIOD_MS 100 //1 ms to check the stats
#define NF_INFERENCE_PERIOD 10 

//#include <rte_hash.h>
//#include <rte_timer.h>
#include <cuda_runtime.h>

typedef enum data_status{empty, occupied, ready} data_status; //empty... data can be filled, occupied: some pointers available, ready: ready to be processed i.e. data filled.
typedef struct __attribute__ ((__packed__)) data_struct{
  int file_id;
  int position;
  int number_of_elements;
  //float data_array[NUM_IN_PKTS];
}data_struct; //chage it to aligned

//void * gpu_image_buffers[MAX_IMAGE];

void transfer_to_gpu(void ** ptrs_from_pkts, int image_id, int num_of_pointers);

/* image data struct that uses CNTK like approach */

typedef struct image_data{
  int image_id; //this is more of a batch ID
  int batch_size; //number of image in batch...
  float image_data_arr[IMAGE_BATCH*IMAGE_NUM_ELE]; //the entire data array, so the mempool will give us place to keep entire data
  float output[IMAGE_BATCH*IMAGENET_OUTPUT_SIZE]; //this will store the imagenet output for whole batch
  int output_size; //the expected output size
  int data_per_image[IMAGE_BATCH]; //how much image data is already copied in a particular image.
  int image_ready[IMAGE_BATCH]; //which images in the batch are full
  
  float stats[3]; //0 - time for memcpy+execution, 1- time for execution only,
  struct timespec timestamps[6];// 0- when all_data is ready 1 -when placed in GPU execution queue, 2- function execution started, data transfer 3-when the GPU processing started, 4 -when the callback returns, 5- when NF encounters first piece of image packet
  int num_data_points_stored; //if this is same as IMAGE_NUM_ELE*BatchSize then we can process the image. 
  data_status status; // is the buffer empty (original state), occupied (some data in it), ready (ready to be processed)
} image_data;

/* image struct that uses NetML like approach */
typedef struct image_data_netml{
  //int image_id; //the image ID. Packets metadata has file_id, it should be mapped to image_id
  void* image_data_ptrs[NUM_OF_PKTS]; //the pointers to the payload of the images. It will hold the payload part of the packet
 
  // float *output_buffer; // a pointer to batch's output buffer. the output buffers are of fixed size (1000 floats) for resnet and vgg etc. NOTE: KEEP OUPUT WITH EXECUTION CONTEXT RATHER THAN WITH IMAGE
  int num_packets_arrived; //how many packets for this images have already been filled in the array above
  void *gpu_pointer; //the pointer to GPU buffer where this image will be transferred
}image_data_netml;

/* now a struct for managing batches for netml */
typedef struct image_batch_netml{
  int batch_id; //identifier of the batch
  image_data_netml *images[IMAGE_BATCH]; //pointers for number of images
  int num_images_in_batch; //how many images are in the above array now
  void *gpu_pointer_batch; //the pointer where the batch data starts in GPU
  cudaStream_t cuda_stream_batch; //the stream this batch will execute in
  int num_of_slots_empty; //number of images that can still be added to this batch
}image_batch_netml;

/*GPU data allocator for NetML */
typedef struct gpu_data_allocator{
  float *gpu_buffer; //the pointer to entire GPU buffer
  float *gpu_buffer_sections[NUM_GPU_SECTIONS]; //6 sections... 0 1-image, 1 2 images, 2 4-images, 3 8 images, 4 16 images, 5 1 image
  int free_image_in_each_section[NUM_GPU_SECTIONS];// the number of free images that remain in GPU sections
  int which_batch_owns_the_section[NUM_GPU_SECTIONS]; //which batch owns each section? only one batch can own a section how many images might be there. 0 indicates free section
}



// a flag to say if the NF has to finish all the works...
int gpu_finish_work_flag;


//void image_init(struct onvm_nf_info *nf, struct onvm_nf_info *original_nf);

/* count the current images pending */
//TODO: Remove this variable.
//void **image_buffers; //this maybe unused... 


#endif
