/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2017 George Washington University
 *            2015-2017 University of California Riverside
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * The name of the author may not be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * bridge.c - send all packets from one port out the other.
 ********************************************************************/

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_udp.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>

#include <rte_eal.h>
#include <rte_eal_memconfig.h>
#include <rte_lcore.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_cntk_api.h"
#include "onvm_images.h"
#include <cuda_runtime.h>

#define MSG_RING_NAME1 "msg_ring1"
#define MSG_RING_SIZE 128

#define NF_TAG "bridge"


/* Struct that contains information about this NF */
struct onvm_nf_info *nf_info;

/* ML related variables */
//static char *input_file_name = NULL;
static const char *ml_model = NULL;
static char *batchsize = NULL;
static char *gpu_percent = NULL;
int other_core_function(void *);
void * cpu_func_ptr = NULL;
void * gpu_func_ptr = NULL;

float *dummy_image;
float *dummy_output;

int function_to_process_gpu_message(struct onvm_nf_msg *message);
void store_the_gpu_pointers(struct onvm_nf_msg *message);
void ask_for_gpu_pointers(void);
void load_gpu_ptrs(struct onvm_nf_msg *message);
void voluntary_restart_the_nf(void);
void cuda_register_memseg_info(void);
void create_dummy_data(int img_size);
void check_number_of_sm(void);

cudaIpcMemHandle_t * cuda_handles;
models_attributes loaded_models;

/* number of package between each print */
static uint32_t print_delay = 1000000;

/*
 * Print a usage message
 */
static void
usage(const char *progname) {
        printf("Usage: %s [EAL args] -- [NF_LIB args] -- -p <print_delay> -f <file_name> -m <model_name>\n\n", progname);
}

/*
 * Parse the application arguments.
 */
static int
parse_app_args(int argc, char *argv[], const char *progname) {
        int c;

        while ((c = getopt (argc, argv, "p:b:g:")) != -1) {
                switch (c) {
                case 'p':
                        print_delay = strtoul(optarg, NULL, 10);
                        break;
			/*
		case 'f':
		        input_file_name = optarg;
		        break;
		case 'm':
		        ml_model = optarg;
		        break;
			*/
		case 'b':
		  batchsize = optarg;
		  break;
		case 'g':
		  gpu_percent = optarg;
		  break;
                case '?':
                        usage(progname);
                        if (optopt == 'p')
                                RTE_LOG(INFO, APP, "Option -%c requires an argument.\n", optopt);
                        else if (isprint(optopt))
                                RTE_LOG(INFO, APP, "Unknown option `-%c'.\n", optopt);
                        else
                                RTE_LOG(INFO, APP, "Unknown option character `\\x%x'.\n", optopt);
                        return -1;
                default:
                        usage(progname);
                        return -1;
                }
        }
        return optind;
}

/*
 * This function displays stats. It uses ANSI terminal codes to clear
 * screen when called. It is called from a single non-master
 * thread in the server process, when the process is run with more
 * than one lcore enabled.
 */
static void
do_stats_display(struct rte_mbuf* pkt) {
        const char clr[] = { 27, '[', '2', 'J', '\0' };
        const char topLeft[] = { 27, '[', '1', ';', '1', 'H', '\0' };
        static uint64_t pkt_process = 0;

        struct ipv4_hdr* ip;

        pkt_process += print_delay;

        /* Clear screen and move to top left */
        printf("%s%s", clr, topLeft);

        printf("PACKETS\n");
        printf("-----\n");
        printf("Port : %d\n", pkt->port);
        printf("Size : %d\n", pkt->pkt_len);
        printf("Type : %d\n", pkt->packet_type);
        printf("Number of packet processed : %"PRIu64"\n", pkt_process);

        ip = onvm_pkt_ipv4_hdr(pkt);
        if(ip != NULL) {
                onvm_pkt_print(pkt);
        }
        else {
                printf("Not IP4\n");
        }

        printf("\n\n");
}

static int
packet_handler(struct rte_mbuf *pkt, struct onvm_pkt_meta *meta, __attribute__((unused)) struct onvm_nf_info *nf_info) {
  //printf("--Packet arrived-- \n");
  static uint32_t counter = 0;
        if (counter++ == print_delay) {
                do_stats_display(pkt);
                counter = 0;
        }

	//copying the packet data to the right buffer ...
	//THis functionality has been moved to onvm_nflib
	if(onvm_pkt_ipv4_hdr(pkt) != NULL){
	  //void * packet_data = rte_pktmbuf_mtod_offset(pkt, void *, sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr));
	  //copy_data_to_image_batch(packet_data, nf_info,nf_info->user_batch_size);
	}
	
        if (pkt->port == 0) {
                meta->destination = 1;
        }
        else {
                meta->destination = 0;
        }
        meta->action = ONVM_NF_ACTION_OUT;
        return 0;
}

int function_to_process_gpu_message(struct onvm_nf_msg *message){

  // we have  a swtich case to handle the messages.
  switch (message->msg_type)
    {
         case MSG_GPU_MODEL_SEC:
	   //switch the GPU pointers and warm up the model
	   store_the_gpu_pointers(message);
	   break;
         case MSG_GPU_MODEL_PRI:
	   store_the_gpu_pointers(message);
	   load_gpu_ptrs(NULL);
	   break;
         case MSG_GET_GPU_READY:
	   load_gpu_ptrs(message);
	   break;
    case MSG_RESTART:
      prepare_to_restart(nf_info,message);
      break;
	   //case MSG_RESTART:
         default:
	   break;
    }
  return 0;
}   

void store_the_gpu_pointers(struct onvm_nf_msg *message){
    //first we should enable the GPU percentage
    char gpu_percentage_str[3];
    //int i = 0;
    sprintf(gpu_percentage_str, "%d", nf_info->gpu_percentage);

    if(gpu_percentage_str != NULL)
      setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_percentage_str, 1);//overwrite cuda_mps_active_thread_precentage
    
    printf("message ID %d \n",message->msg_type);


    //keeping the model information in the local variable
    memcpy(&loaded_models, message->msg_data, sizeof(models_attributes));

    //put back the message in the pool it came from
    cuda_handles = (cudaIpcMemHandle_t *) ((models_attributes *)(message->msg_data))->cuda_handles;

}
/* this function loads the GPU pointer */
void load_gpu_ptrs(struct onvm_nf_msg *message){

  //if the message is equal to null, then this is a primary NF so we can leave the GPU percentage as it is.
  if(message == NULL){
    printf("Primary NF\n");
    struct timespec begin,end;
    printf("GPU percentage is %s \n", gpu_percent);
    setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_percent, 1);//overwrite cuda_mps_active_thread_precentage
    clock_gettime(CLOCK_MONOTONIC, &begin);
    printf("the output of moving pointers = %d (0 means success) \n",link_gpu_pointers(cpu_func_ptr, cuda_handles, count_parameters(cpu_func_ptr)));
    //check number of SMs
    check_number_of_sm();
    //dummy test...
    //evaluate the image twice with Dummy data... measure the time taken.
    struct timespec timestamps[6];
    evaluate_in_gpu_input_from_host(dummy_image, (3*224*224), dummy_output, cpu_func_ptr,&timestamps, 0, NULL, NULL);
    evaluate_in_gpu_input_from_host(dummy_image, (3*224*224), dummy_output, cpu_func_ptr, &timestamps,0, NULL, NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double load_time = (end.tv_sec-begin.tv_sec)*1000000.0+(end.tv_nsec-begin.tv_nsec)/1000.0;
    printf("The time to switch pointers to GPU is %f microseconds \n",load_time);
    printf("since we are first NF, we will try to restart.. first wake up another NF\n");
    sleep(20);
    //trigger the NF restart...
    voluntary_restart_the_nf();
  }
  else
    {
      printf(" This is Secondary NF\n");
      //load the GPU percentage and then link the GPU pointers
      //we should get GPU percentage from the message
      char gpu_percent[3];
      sprintf(gpu_percent,"%d", *((int*) message->msg_data));
      
      setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_percent, 1);//overwrite cuda_mps_active_thread_precentage
      struct timespec begin,end;
      clock_gettime(CLOCK_MONOTONIC, &begin);
      printf("the output of moving pointers = %d (0 means success) \n",link_gpu_pointers(cpu_func_ptr, cuda_handles, count_parameters(cpu_func_ptr)));
      clock_gettime(CLOCK_MONOTONIC, &end);
      double load_time = (end.tv_sec-begin.tv_sec)*1000000.0+(end.tv_nsec-begin.tv_nsec)/1000.0;
      printf("The time to switch pointers to GPU is %f microseconds \n",load_time);

      //dummy test...
      //evaluate the image twice with Dummy data... measure the time taken.
      struct timespec timestamps[6];
      evaluate_in_gpu_input_from_host(dummy_image, (3*224*224), dummy_output, cpu_func_ptr,&timestamps, 0, NULL, NULL);
      evaluate_in_gpu_input_from_host(dummy_image, (3*224*224), dummy_output, cpu_func_ptr, &timestamps,0, NULL, NULL);
      //finish running the dummy test before saying GPU is ready.
      onvm_send_gpu_msg_to_mgr(nf_info, MSG_NF_GPU_READY);
      rte_free(dummy_image);
      rte_free(dummy_output);
    }
  nf_info->function_ptr = cpu_func_ptr;
  
}

/* the function to ask the manager for GPU pointers */
void ask_for_gpu_pointers(void ){
  //now send message to manager asking for the pointer...
  provide_gpu_model * give_gpu_model = (provide_gpu_model *) rte_malloc(NULL,sizeof(provide_gpu_model),0);
  give_gpu_model->model_index = atoi(ml_model); //alexnet
  //memcpy(name_of_model, name, strlen(name));
  //memcpy(_nfinfo,nf_info, sizeof(struct onvm_nf_info));
  give_gpu_model->nf = nf_info;
  //memcpy(give_gpu_model->model_name, name_of_model, sizeof(char)*20);
  onvm_send_gpu_msg_to_mgr(give_gpu_model,MSG_GIVE_GPU_MODEL);
  printf("Sent message to mgr asking for GPU Pointers\n");
}

void voluntary_restart_the_nf(void){

  printf("We are restarting the NF \n");
  onvm_send_gpu_msg_to_mgr(nf_info,MSG_NF_VOLUNTARY_RE);
  
}


void check_number_of_sm(void){
  int value;
  cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0);
  printf("The number of SMs are : %d \n", value);
}
//makes a dummy image input and output
void create_dummy_data(int img_size){
  dummy_image = (float *) rte_malloc(NULL, sizeof(float)*img_size, 0);
  dummy_output = (float *) rte_malloc(NULL, sizeof(float)*1000,0);
}

void cuda_register_memseg_info(void){

  //get the memory config
  struct rte_config * rte_config = rte_eal_get_configuration();

  //now get the memory locations
  struct rte_mem_config * memory_config = rte_config->mem_config;

  int i;
  struct timespec begin,end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  for(i = 0; i<RTE_MAX_MEMSEG_LISTS; i++){
    struct rte_memseg_list *memseg_ptr = &(memory_config->memsegs[i]);
    if(memseg_ptr->page_sz > 0 && memseg_ptr->socket_id == (int)rte_socket_id()){
      //printf("Pointer to huge page %p and size of the page %"PRIu64"\n",memseg_ptr->base_va, memseg_ptr->page_sz);
      cudaHostRegister(memseg_ptr->base_va, memseg_ptr->page_sz, cudaHostRegisterDefault);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double time_taken_to_register = (end.tv_sec-begin.tv_sec)*1000000.0 + (end.tv_nsec-begin.tv_nsec)/1000.0;
  printf("Total time taken to register the mempages to cuda is %f micro-seconds \n",time_taken_to_register);
    
}
  
/* function to load gpu file 
void load_gpu_file(void){
  // convert the filename to wchar_t 
  size_t filename_length = strlen(input_file_name);
  wchar_t file_name[filename_length];
  size_t wfilename_length = mbstowcs(file_name,input_file_name,filename_length+1);
  
  int flag = 0;
  struct timespec begin,end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  fprintf(stdout,"load file %d \n",load_model(file_name,&cpu_func_ptr, &gpu_func_ptr, flag, 0));
  clock_gettime(CLOCK_MONOTONIC, &end);
  double time_spent = (end.tv_sec-begin.tv_sec)*1000000.0 + (end.tv_nsec-begin.tv_nsec)/1000.0;
  printf("time taken to load the file %f microseconds\n", time_spent);
  printf(" length of the filename is %zu and the GPU pointer is %p CPU pointer is %p\n",wfilename_length, gpu_func_ptr, cpu_func_ptr);

  //now send message to manager asking for the pointer...
  provide_gpu_model * give_gpu_model = (provide_gpu_model *) rte_malloc(NULL,sizeof(provide_gpu_model),0);
  //struct onvm_nf_info * _nfinfo = (struct onvm_nf_info *) rte_malloc(NULL, sizeof(struct onvm_nf_info), 0);
  char * name_of_model = (char *) rte_malloc(NULL, sizeof(char)*20, 0);

  const char * name = ml_model;

  give_gpu_model->model_index = 0; //alexnet
  //memcpy(name_of_model, name, strlen(name));
  //memcpy(_nfinfo,nf_info, sizeof(struct onvm_nf_info));
  give_gpu_model->nf = nf_info;
  //memcpy(give_gpu_model->model_name, name_of_model, sizeof(char)*20);
  onvm_send_gpu_msg_to_mgr(give_gpu_model,MSG_GIVE_GPU_MODEL);
  printf("Sent message to mgr \n");
}
*/

/* the function executed in 2nd core 
int other_core_function(void *lcore){
  // message 
  fprintf(stdout,"Welcome to core %d \n",rte_lcore_id());

  // convert the filename to wchar_t 
  size_t filename_length = strlen(input_file_name);
  wchar_t file_name[filename_length];
  size_t wfilename_length = mbstowcs(file_name,input_file_name,filename_length+1);
  void * cpu_func_ptr = NULL;
  void * gpu_func_ptr = NULL;
  int flag = 0;
  struct timespec begin,end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  fprintf(stdout,"load file %d \n",load_model(file_name,&cpu_func_ptr, &gpu_func_ptr, flag, 0));
  clock_gettime(CLOCK_MONOTONIC, &end);
  double time_spent = (end.tv_sec-begin.tv_sec)*1000000.0 + (end.tv_nsec-begin.tv_nsec)/1000.0;
  printf("time taken to load the file %f microseconds\n", time_spent);
  printf("the core is %d and length of the filename is %zu and the GPU pointer is %p CPU pointer is %p\n",*((int *) lcore),wfilename_length, gpu_func_ptr, cpu_func_ptr);
    
  // create the message to send to another NF...
  struct rte_ring * message_ring;
  message_ring = rte_ring_lookup(MSG_RING_NAME1);
  if(message_ring == NULL)
    {
      printf("message ring couldn't be found \n");
    }

  // get the message out of the ring 
  struct onvm_nf_msg *message;
  while(rte_ring_dequeue(message_ring, (void **) &message)!= 0);
  void * cuda_handles= message->msg_data;

  // attach the GPU pointers to the existing CPU models.
  printf("The output of moving pointers %d \n",link_gpu_pointers(cpu_func_ptr, cuda_handles, count_parameters(cpu_func_ptr)));
  
  evaluate_in_gpu_input_from_host(NULL,0,NULL,cpu_func_ptr);
  evaluate_in_gpu_input_from_host(NULL,0,NULL,cpu_func_ptr);
  return *(int *)lcore;
}
*/
int main(int argc, char *argv[]) {
        int arg_offset;

        const char *progname = argv[0];

        if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, &nf_info)) < 0)
                return -1;
        argc -= arg_offset;
        argv += arg_offset;

        if (parse_app_args(argc, argv, progname) < 0) {
                onvm_nflib_stop(nf_info);
                rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");
        }

	/* launch a CPU thread and find the message ring of another NF */
	/*
	//finding current CPU lcore
	unsigned cur_lcore = rte_lcore_id();
	unsigned next_lcore = rte_get_next_lcore(cur_lcore,1,1); //gives you another lcore allocated to the NF

	void *lcore_args;
	lcore_args = &next_lcore;
	
	// launch a function in it to create a message ring. and post a message to the ring after 20 seconds 
	if(rte_eal_remote_launch(other_core_function, lcore_args ,next_lcore) == -EBUSY){
	  RTE_LOG(INFO,APP,"Core %d is being used \n",next_lcore);
	  return -1;
	}
	else
	  {
	    printf("executed something in another core \n");
	  }
	*/
	//register the funtion for processing the GPU related message from manager
	register_gpu_msg_handling_function(&function_to_process_gpu_message);
	//load_gpu_file();

	// loading the gpu model
	//load_ml_file(input_file_name, 0 /*cpu only*/, &cpu_func_ptr, &gpu_func_ptr, nf_info);

	//ask gpu pointers from the manager
	//ask_for_gpu_pointers();

	//check memsegs.. and pin them
	//cuda_register_memseg_info();

	//create a new memory for dummy image
	//create_dummy_data((3*224*224));

	//put in the batch size
	nf_info->user_batch_size = atoi(batchsize);
	
	//receive packets.
	onvm_nflib_run(nf_info, &packet_handler);
        printf("If we reach here, program is ending\n");
        return 0;
}
