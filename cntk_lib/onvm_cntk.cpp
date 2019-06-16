#include <stdio.h>
#include <iostream>

#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "CNTKLibrary.h"
extern "C"{

  #include "onvm_cntk_api.h"
#include "onvm_ml_libraries.h"

}

using namespace CNTK;

/* function declarations */

/* information about how many GPU-side pointer, i.e. how many CUDA handles you have with this model 
 *
 * Input: pointer to object of model loaded in either GPU or CPU
 * Output: the number of parameter the model has
 */




//these modules will be shared so they can be global variables 
FunctionPtr rootFunc_cpu[NUMBER_OF_MODELS], rootFunc_gpu[NUMBER_OF_MODELS];

/* this function loads the model and gives back the pointer */
//extern "C"
//int cntk_load_model(const char * file_path_char, int load_flag, void ** cpu_function_pointer, void ** gpu_function_pointer, int * number_of_parameters)//, void **cpu_device_descriptor, void ** gpu_device_){
extern "C"
int cntk_load_model(nflib_ml_fw_load_params_t* load_params,  void *status)
{

  struct timespec load_begin, load_end;
  clock_gettime(CLOCK_MONOTONIC, &load_begin);
  static int count_call = 0;
  const char * file_path_char = load_params->file_path;
  printf("the file path is %s \n", file_path_char);
  int load_flag = load_params->load_options;
  
  setlocale(LC_ALL, "en_US.UTF-8");

    
  /* convert the filename to wchar_t */
  size_t filename_length = strlen(file_path_char);
  wchar_t file_path[filename_length];
  mbstowcs(file_path,file_path_char,filename_length+1);


  //CNTK::DeviceDescriptor cpu_device, gpu_device;

  //try loading the model
  try{
    //load the file to this variable
    switch(load_flag)
      {
      case 0:
	//rootFunc_cpu = Function::Load(file_path, cpu_device);
	rootFunc_cpu[count_call] = Function::Load(file_path, CNTK::DeviceDescriptor::CPUDevice());
	load_params->model_handle = (void *)(rootFunc_cpu[count_call].get());
	//model_handle = *cpu_function_pointer;
	break;
      case 1:
	//auto& gpu_device = CNTK::DeviceDescriptor::GPUDevice(0);
	rootFunc_gpu[count_call] = Function::Load(file_path, CNTK::DeviceDescriptor::GPUDevice(0));
	//*gpu_function_pointer = (void *) (rootFunc_gpu.get());
	load_params->model_handle = (void *) (rootFunc_gpu[count_call].get());
	//model_handle = *gpu_function_pointer;
	break;
	/*
      case 2:
	rootFunc_cpu = Function::Load(file_path, CNTK::DeviceDescriptor::CPUDevice());
	rootFunc_gpu = Function::Load(file_path, CNTK::DeviceDescriptor::GPUDevice(0));
	*cpu_function_pointer = (void *) (rootFunc_cpu.get());
	*gpu_function_pointer = (void *) (rootFunc_gpu.get());
	model_handle = *cpu_function_pointer;
	break;
	*/
      default:
    	  printf("The model can be loaded in either CPU : 0 or GPU : 1\n");
	break;

      }
    /*
    //if we have an entry for number of parameters, let's count the number of parameters
    if(number_of_parameters != NULL){
      *number_of_parameters = count_parameters(model_handle);
    }
    */
    clock_gettime(CLOCK_MONOTONIC, &load_end);
    float time_taken_to_load = (load_end.tv_sec-load_begin.tv_sec)*1000.0+(load_end.tv_nsec-load_begin.tv_nsec)/1000000.0;
    
    std::cout<<"Loaded the file .. time taken to load (milliseconds) "<<time_taken_to_load<<std::endl;
    count_call++;

    return 0;
  }
  catch(...){
    printf("Loading the file failed \n");
    return -1;
  }
  
}

/* gives the number of parameters in a ML function */
extern "C"
int cntk_count_parameters(void * gpu_function_pointer){
  Function *rootFunc = (Function *)gpu_function_pointer;
  int counter = 0;
  auto parameters = (rootFunc)->Parameters();
  for(auto& p : parameters){
    counter++;
  }
  std::cout<<"Number of parameters "<<counter<<std::endl;
  return counter;
}


/* this function extracts the GPU pointers and returns GPU cuda handles */
extern "C"
void cntk_get_cuda_pointers_handles(void *gpu_function_pointer, void *mem_handles){
  /* we do not need the number of mem handles as the user should know it or we can just simply iterate through it */
  //int number_of_handles_needed = count_parameters(gpu_function_pointer);
  //cudaIpcMemHandle_t *handles = (cudaIpcMemHandle_t*) malloc(sizeof(cudaIpcMemHandle_t)*number_of_handles_needed);

  /* also the cudaIpcMemHandle_t will be allocated by NFs*/
  cudaIpcMemHandle_t *handles = (cudaIpcMemHandle_t*)mem_handles;
  /* now need to extract the pointers from the GPU */
  Function *rootFunc = (Function*)gpu_function_pointer;
  int counter = 0;
  long total_size = 0;
  
  auto parameters = (rootFunc)->Parameters();

  for(auto& p : parameters){
    //auto gpu_ndarray = p.GetValue();
    cudaError_t cuda_error = cudaIpcGetMemHandle(&(handles[counter]), const_cast<float *>(p.GetValue()->DataBuffer<float>()));
    //check if the CUDA process went okay
    if(cuda_error != cudaSuccess)
      std::cout<<"CUDA ERROR : "<<cudaGetErrorString(cuda_error)<<std::endl;

    //first we need to find the shape...
    NDShape parameter_shape = p.Shape();

    //now loop through the dimensions and multiply them.
    std::vector<size_t> parameterShapeDim = parameter_shape.Dimensions();

    int size = 1;
    for(auto a : parameterShapeDim){
      size *= a;
    }
    total_size += size;
    
    //debug
    //std::cout<<"GPU pointer "<<p.GetValue()->DataBuffer<float>()<<std::endl;
    counter++;
  }

  printf(" The total size of all parameters is %f (megabytes)\n",(total_size *sizeof(float))/(1024*1024.0));
  

  //return (void *) handles;
}

/* the global variable that will not be freed... GPU side buffer */
ValuePtr inputVal, outputVal;
Variable inputvar, outputvar;

/* function to attach the GPU pointers with the models */
//int link_gpu_pointers(void * cpu_function_pointer, void * cuda_handles_for_gpu_data, int number_of_parameters)
extern "C"
int cntk_link_pointers(nflib_ml_fw_link_params_t* link_params,  void *status)
{
  struct timespec begin_linking, end_linking;
  clock_gettime(CLOCK_MONOTONIC, &begin_linking);
  void *cpu_function_pointer = link_params->model_handle;
  void *cuda_handles_for_gpu_data = link_params->cuda_handles_for_gpu_data;
  int number_of_parameters = link_params->number_of_parameters;
  
  //printf("Number of parameters %d \n",link_params->number_of_parameters);

  /* we assume the models are identical in CPU and GPU side */
  const auto& gpu_device = CNTK::DeviceDescriptor::GPUDevice(0);
  Function *rootFunc = (Function*) cpu_function_pointer;
  auto parameters = (rootFunc)->Parameters();
  int counter = 0;

  // a single variable should be enough
  //void ** gpu_pointer = (void **)malloc(sizeof(void *)*number_of_parameters);
  void * gpu_pointer;
  
  //since the number of paramters and the number of gpu pointers are the same... we can resolve the GPU pointers in the loop itself 
  for(auto& p: parameters){
    auto ndarray = p.GetValue();
    // resolve the cuda handles
    cudaError_t cuda_error = cudaIpcOpenMemHandle((void **)&(gpu_pointer), ((cudaIpcMemHandle_t*)cuda_handles_for_gpu_data)[counter],cudaIpcMemLazyEnablePeerAccess);
    if(cuda_error != cudaSuccess)
      std::cout<<"CUDA ERROR : "<<cudaGetErrorString(cuda_error)<<" gpu pointer "<<gpu_pointer<<"counter "<<counter<<std::endl;

    //else
    //  std::cout<<"CUDA pointer successfully converted "<<gpu_pointer[counter]<<std::endl;

    
    //now attach the gpu pointers
    (ndarray.get())->Attach_GPU_Address(gpu_device, (float *) gpu_pointer);
    counter++;
  }
  //free(gpu_pointer);

  clock_gettime(CLOCK_MONOTONIC, &end_linking);

  double linking_time = (end_linking.tv_sec-begin_linking.tv_sec)*1000.0+(end_linking.tv_nsec-begin_linking.tv_nsec)/1000000.0;
  printf("Total linking time in milliseconds %f \n", linking_time);

  // create a GPU side buffer
  std::vector<Variable> inputs = rootFunc->Arguments();

    //checking the input
    int i;
    //std::cout<<"The input in evaluate func \n";
    //for(i = 0; i<10 ; i++){
    //  std::cout<<input[i]<<" ";
    //}
    //std::cout<<"\n";
    // get the input and output
    std::vector<Variable> outputs = rootFunc->Outputs();

    //create a new vector
   inputvar = inputs[0];
   outputvar = outputs[0];
    long total_size = 1;

    for( auto inputvars : inputs){
      /* now create a new input */
      NDShape inputShape = inputvars.Shape();
      std::vector<size_t> inputShapeDim = inputShape.Dimensions();

      //std::cout<<"Input Dimensions "<<std::endl;
      //long total_size = 1;
      for(auto a : inputShapeDim){
        //std::cout<<"dimensions "<<a<<std::endl;
        total_size *= a;
      }
    }
    NDShape outputShape = outputvar.Shape();
    std::vector<size_t> outputsizedim = outputShape.Dimensions();
    int tot_output_size;
    for (auto a: outputsizedim){
      tot_output_size *= a;
    }

    
    std::vector<float> outputData(tot_output_size,0.0); //allocating output vector

    //let's try something... let's make our own GPU buffer
    void * gpu_input_buffer;
    cudaMalloc(&gpu_input_buffer, sizeof(float)*3*224*224);

    printf("GPU side pointer is %p \n", gpu_input_buffer);
      //allocate a new vector... to feed in the data..
      //for(int i =0; i<total_size;i++){
      //  inputData.at(i) = (float) (i%255);
      //}
      //const auto& gpu_device = CNTK::DeviceDescriptor::GPUDevice(0);
    struct timespec begin,end;

    printf("input size %d outputsize %d \n", total_size, tot_output_size);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    std::vector<float> inputData(total_size,0.0); //allocating a vector
    //inputVal = Value::CreateBatch(inputvar.Shape(), inputData, CNTK::DeviceDescriptor::CPUDevice() , false, gpu_input_buffer);
    inputVal = Value::CreateBatch(inputvar.Shape(), inputData, gpu_device , false, gpu_input_buffer);
    //inputVal = Value::CreateBatch_nocopy(inputvar.Shape(), inputData,gpu_device, false, gpu_input_buffer);
    //((inputVal->Data()).get())->Attach_GPU_Address(gpu_device, (float *) gpu_input_buffer);
     clock_gettime(CLOCK_MONOTONIC, &end);
     outputVal = Value::CreateBatch(outputvar.Shape(), outputData, gpu_device);
     double time_spent = (end.tv_sec-begin.tv_sec)*1000.0+(end.tv_nsec-begin.tv_nsec)/1000000.0;
     printf("Time spent creating a cpu side batch %f milliseconds\n",time_spent);
    printf("After conversion GPU side batch pointer is %p \n",(void *)inputVal->Data()->DataBuffer<float>());
     link_params->gpu_side_input_pointer = (void *)inputVal->Data()->DataBuffer<float>();
     link_params->gpu_side_output_pointer = (void *)outputVal->Data()->DataBuffer<float>();
    
  return 0;
}

//void CUDART_CB callback_function_example(cudaStream_t event, cudaError_t status, void *data);
//void CUDART_CB callback_function_example(cudaStream_t event, cudaError_t status, void *data){
//  std::cout<<"---- Callback called -----\n";
//}

/* evaluate a data set */
// let's hard code
//extern "C"
//void evaluate_in_gpu_input_from_host(float *input, size_t input_size, float *output, void *function_pointer, size_t output_size, int batch_size, cudaHostFn_t callback_function, void *callback_data, void* evaluation_time){
extern "C"
int cntk_infer_batch(nflib_ml_fw_infer_params_t* infer_params,  void *status){

	float * input = (float *)infer_params->input_data;
	size_t input_size = infer_params->input_size;
	float *output = infer_params->output;
	void *function_pointer = infer_params->model_handle;
	int batch_size = infer_params->batch_size;
	cudaHostFn_t callback_function = infer_params->callback_function;
	void * callback_data = infer_params->callback_data;

	const auto& gpu_device = CNTK::DeviceDescriptor::GPUDevice(0);

  //  cudaHostFn_t *callback_function = (cudaHostFn_t *) cb_function;
  //float cuda_time = 0;
  //struct timespec *timestamps = (struct timespec *)evaluation_time;
  //struct gpu_callback *callback_info = (struct gpu_callback*) callback_data;
  //clock_gettime(CLOCK_MONOTONIC, &(timestamps[0]));

  //convert to function
  Function *rootFunc = (Function *) function_pointer;

  /*
  std::vector<Variable> inputs = rootFunc->Arguments();

  //checking the input
  int i;
  //std::cout<<"The input in evaluate func \n";
  //for(i = 0; i<10 ; i++){
  //  std::cout<<input[i]<<" ";
  //}
  //std::cout<<"\n";

  //create a new vector
  Variable inputvar = inputs[0];
  long total_size = 1;

  for( auto inputvars : inputs){
    // now create a new input
    NDShape inputShape = inputvars.Shape();
    std::vector<size_t> inputShapeDim = inputShape.Dimensions();

    //std::cout<<"Input Dimensions "<<std::endl;
    //long total_size = 1;
    for(auto a : inputShapeDim){
      //std::cout<<"dimensions "<<a<<std::endl;
      total_size *= a;
    }
  }
  
  std::vector<float> inputData(input, input+input_size); //allocating a vector
  //allocate a new vector... to feed in the data..
  //for(int i =0; i<total_size;i++){
  //  inputData.at(i) = (float) (i%255);
  //}
  const auto& gpu_device = CNTK::DeviceDescriptor::GPUDevice(0);
    
  ValuePtr inputVal = Value::CreateBatch(inputvar.Shape(), inputData, gpu_device);
  */


  //create a map of input.
  std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputvar, inputVal } };

  // Alternatively, create a Value object and add it to the data map.
  std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputvar, nullptr } };
  //std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputvar, outputVal } };

  //creating cuda events
  //  cudaEventCreate(&(callback_info->start_event));
  //cudaEventCreate(&(callback_info->end_event));

  //struct timespec start1, finish1;

  /*the execution block */
//  clock_gettime(CLOCK_MONOTONIC, &timestamps[1]);
//  clock_gettime(CLOCK_MONOTONIC, &start1);
  //cudaEventRecord(callback_info->start_event);
  rootFunc->Evaluate(inputDataMap, outputDataMap, gpu_device);
  //cudaEventRecord(callback_info->end_event);
 // clock_gettime(CLOCK_MONOTONIC, &finish1);

  ValuePtr outputVal = outputDataMap[outputvar];
    //std::cout<<"output var shape "<<outputvar.Shape().TotalSize()<<std::endl;
  auto output_ndarray = outputVal->Data()->DataBuffer<float>();

  cudaMemcpyAsync(output, output_ndarray, sizeof(float)*1000*batch_size, cudaMemcpyDeviceToHost, 0);

  if(callback_function != NULL){
    cudaLaunchHostFunc(0, callback_function, callback_data);
  }

  //cudaMemcpy(output, output_ndarray, sizeof(float)*1000, cudaMemcpyDeviceToHost); //currently in zero stream
  //add callback if it exists



  //cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&cuda_time, callback_info->start_event, callback_info->end_event);
  //double time_taken = (finish1.tv_sec-start1.tv_sec)*1000.0+(finish1.tv_nsec-start1.tv_nsec)/1000000.0;

  //std::cout<<"CPU side time taken "<<time_taken<<" milli-seconds, CUDA timer time taken "<<cuda_time<<" milli-seconds\n";
  /*
  double exec_time = (execution_end.tv_sec-execution_begin.tv_sec)*1000000.0+(execution_end.tv_nsec-execution_begin.tv_nsec)/1000.0;
  //std::cout<<"time taken to run this module is "<<time_taken<<" micro-seconds ... and event time "<<cuda_time<<std::endl;
  std::cout<<""<<time_taken<<","<<cuda_time<<", exec_time "<<exec_time<<std::endl;  
  */
  //std::cout<<cuda_time<<std::endl;
  //for(int k = 0; k<1000; k++)
  //  std::cout<<result[k]<<" ";

  //std::cout<<std::endl;
  //free(result);
  return 0;
  
}


int count_inputs(void *function_pointer){
  //convert to function
  Function *rootFunc = (Function *) function_pointer;
  std::vector<Variable> inputs = rootFunc->Arguments();
  return inputs.size();
}  


void get_all_input_sizes(void *function_pointer,int *input_dim, int input_var_number){
  Function *rootFunc = (Function *) function_pointer;
  std::vector<Variable> inputs = rootFunc->Arguments();
  int counter = 0;
  auto input = inputs.at(input_var_number);
  
  NDShape inputShape = input.Shape();
  std::vector<size_t> inputShapeDim = inputShape.Dimensions();
  for(auto a: inputShapeDim){
    input_dim[counter++] = a;
  }
}

// new function to evaluate the images...
extern "C"
void evaluate_image_in_gpu(void *image, void * function_pointer,cudaHostFn_t callback_function, void *callback_data, int gpu_barrier_flag){
  //call the other function... just put a timestamp in here
  image_data * ready_image = (image_data *) image;

 // evaluate_in_gpu_input_from_host(ready_image->image_data_arr, ready_image->num_data_points_stored, ready_image->output, function_pointer, IMAGENET_OUTPUT_SIZE, ready_image->batch_size,callback_function, callback_data, &(ready_image->timestamps[2]));
}
