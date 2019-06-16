#ifndef ONVM_CNTK_API_H
#define ONVM_CNTK_API_H
#include "/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/onvm_ml_libraries.h"

#include <cuda_runtime.h>

/* This function loads the model file from disk
 * Arguments are:
 * model file path, reference to CPU side module pointer, reference to GPU side model pointer, flag 0 = load only on CPU, 1 = load only on GPU, 2 = load on both devices 
 
 *Inputs: 
 *File Path (Path to the file in the disk)
 *Load Flag (File to be loaded in CPU or GPU?


 *Input/Output:
 *cpu_function_pointer (The pointer to object for the CPU side model, used by NF to load the model to CPU)
 *gpu_function_pointer (The pointer to object for th GPU side model, used by Manager to load the model in GPU)
 *num_of_parameters (The number of parameters the model has )

 *Output
 *function returns 0 if successful on loading the model. However, since we use CNTK API the program will crash if the model is not loaded.
 */
int cntk_load_model(nflib_ml_fw_load_params_t *load_params, void *aio);
//int cntk_load_model(const char * file_path, int device_flag, void ** cpu_model_handle, void ** gpu_model_handle, int *num_of_parameters );


//function to count parameters
int cntk_count_parameters(void * gpu_function_pointer);


/* this function extracts the pointers for the GPU side models if there are GPU side models loaded 
   the cudaIpcMemHandles_t array is also submitted as parameter which the function will update with the cuda Mem handles

   *Input:
   gpu_function_pointer: the pointer to the object to the model in GPU
   
   *input/output:
   *mem_handles: empty buffer of cuda_memhandles for IPC handles. the program expects that the Manager knows how many mem_handles it needs and send the pointer to the buffer.
   manager can know how many handles it need by calling the count_parameters functions

*/
void cntk_get_cuda_pointers_handles(void * gpu_function_pointer, void * mem_handles);

/* get the number of inputs a model have 

 *This function has use.
*/
int cntk_count_inputs(void *function_pointer);

/* function to attach the GPU pointers with the models 
 * Input: CPU function pointer (NF loads the model in CPU and then gets the object here. Now we will link the model to the GPU model parameters that manager has loaded
 * Input: cuda_handles_for_gpu_data (CUDA IPC handles for the parameters)
 * Input: number of parameters (The number of parameters the model has)
 * Output: Integer 0 signifying success of linking.
*/
int cntk_link_pointers(nflib_ml_fw_link_params_t *link_params, void *aio);
//int link_gpu_pointers(void * cpu_function_pointer, void * cuda_handles_for_gpu_data, int number_of_parameters);

/* evaluate a data set 
   INPUT: input buffer (float input) CPU side buffer with image raw data
   *input_size : size of the input. Necessary for batching
   *function_pointer: the pointer to the GPU model object
   *cuda_event_flag: the barrier to make sure the CUDA even finished. Barely used and we should remove it.
   *batch size: batch size for image
   
Input/Output:
* output : output buffer to CPU for results after processing
* callback function: this will be called after evaluation is finished
*callback_data: the argument for callback function

   

*/
//int cntk_infer_batch(nflib_ml_fw_infer_params_t* infer_params, void *aio);
int cntk_infer_batch(nflib_ml_fw_infer_params_t* infer_params,  void *status);
//void evaluate_in_gpu_input_from_host(float *input, size_t input_size, float * output, void *function_pointer,size_t output_size, int batch_size,cudaHostFn_t callback_function, void *callback_data, void *evaluation_time );

/* evaluate an image in GPU 

 * input: Image: Image struct that contains detail about image batch. input and output
 * function pointer: the handle to the model in GPU
 * callback function and callback data: the GPU callback function and its arguments
 * gpu_barrier: if the GPU should have a barrier or not
 */

void evaluate_image_in_gpu(void * image, void *function_pointer,cudaHostFn_t callback_function, void * callback_data, int gpu_barrier_flag);

#define IMAGE_BATCH 32
#define IMAGE_NUM_ELE 3*224*224
#define IMAGENET_OUTPUT_SIZE 1000


typedef struct image_data{
  int image_id;
  void ** img_data_ptrs;
  int batch_size; //number of image in batch...
  float image_data_arr[IMAGE_BATCH*IMAGE_NUM_ELE]; //the entire data array, so the mempool will give us place to keep entire data
  float output[IMAGE_BATCH*IMAGENET_OUTPUT_SIZE]; //this will store the imagenet output for whole batch
  float stats[3]; //0 - time for memcpy+execution, 1- time for execution only,
  struct timespec timestamps[5];// 0- when all_data is ready 1 -when placed in GPU execution queue, 2- function execution started, data transfer 3-when the GPU processing started, 4 -when the callback returns
  int num_data_points_stored; //if this is same as IMAGE_NUM_ELE then we can process the image. can be used for batch too
  int status; // is the buffer empty (original state), occupied (some data in it), ready (ready to be processed)
} image_data;


#endif
