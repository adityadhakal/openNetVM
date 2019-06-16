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
#include <rte_ip.h>
#include <rte_malloc.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_cntk_api.h"
//#include "onvm_netml.h"
#include <cuda_runtime.h>

#define MSG_RING_NAME1 "msg_ring1"
#define MSG_RING_SIZE 128

#define NF_TAG "bridge"


/* Struct that contains information about this NF */
struct onvm_nf_info *nf_info;

/* file name */
static char *input_file_name = NULL;
static const char *ml_model = NULL;
static char * number_of_loops  = NULL;
static char * gpu_cmd_percentage = NULL;
int other_core_function(void *);

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

        while ((c = getopt (argc, argv, "p:f:m:t:g:")) != -1) {
                switch (c) {
                case 'p':
                        print_delay = strtoul(optarg, NULL, 10);
                        break;
		case 'f':
		        input_file_name = optarg;
		        break;
		case 'm':
		        ml_model = optarg;
		        break;
		case 't':
		        number_of_loops = optarg;
		        break;
		case 'g':
            	        gpu_cmd_percentage = optarg;
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
  printf("--Packet arrived-- \n");
        static uint32_t counter = 0;
        if (counter++ == print_delay) {
                do_stats_display(pkt);
                counter = 0;
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

void * cpu_func_ptr = NULL;
void * gpu_func_ptr = NULL;
void * all_cuda_handels;
int function_to_process_gpu_message(struct onvm_nf_msg *message);
int move_the_model_to_gpu(struct onvm_nf_msg * message);
int function_to_process_gpu_message(struct onvm_nf_msg *message){

  // we have  a swtich case to handle the messages.
  switch (message->msg_type)
    {
    case MSG_GPU_MODEL:
      //switch the GPU pointers and warm up the model
      move_the_model_to_gpu(message);
      break;
    default:
      break;
    }
  return 0;
}   
int move_the_model_to_gpu(struct onvm_nf_msg *message){
    //first we should enable the GPU percentage
    char gpu_percentage_str[3];
    sprintf(gpu_percentage_str, "%d", nf_info->gpu_percentage);

    printf("GPU command percentage %s \n", gpu_cmd_percentage);
    if(gpu_cmd_percentage != NULL)
      setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_cmd_percentage, 1);//overwrite cuda_mps_active_thread_precentage
    
    printf("message ID %d \n",message->msg_type);
    cudaIpcMemHandle_t * cuda_handles = (cudaIpcMemHandle_t*)((models_attributes *) (message->msg_data))->cuda_handles;
    int i = 0;
    int num_parameters = count_parameters(cpu_func_ptr);

    printf("number of parameters in the models are %d \n", num_parameters);

    
    struct timespec begin,end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    printf("the output of moving pointers  %d \n",link_gpu_pointers(cpu_func_ptr, cuda_handles, count_parameters(cpu_func_ptr)));
    clock_gettime(CLOCK_MONOTONIC, &end);
    double load_time = (end.tv_sec-begin.tv_sec)*1000000.0+(end.tv_nsec-begin.tv_nsec)/1000.0;
    printf("The time to switch pointers to GPU is %f microseconds \n",load_time);

    //randomize the image data
    srand(2);
    float *images = (float *) rte_malloc(NULL, sizeof(float)*IMAGE_SIZE, 0);
    
  for(i = 0; i < IMAGE_SIZE; i++){
    images[i] = (float)(rand()%255);
  }

  int j ;
  int max_loop = atoi(number_of_loops);
  float evaluation_time[2];
  float * output = (float *) rte_malloc(NULL, sizeof(float)*1000, 0);
  //now time to run.. also check.. the time it took to execute
  struct timespec begin1, end1;
  for(j = 0; j<max_loop; j++){
    clock_gettime(CLOCK_MONOTONIC, &begin1);
    evaluate_in_gpu_input_from_host(images, IMAGE_SIZE, output, cpu_func_ptr, evaluation_time, 0);
    clock_gettime(CLOCK_MONOTONIC, &end1);
    double time_spent = (end1.tv_sec-begin1.tv_sec)*1000000.0+(end1.tv_nsec-begin1.tv_nsec)/1000.0;
    printf("the return time of function is %f \n", time_spent);
    //evaluate_in_gpu_input_from_host(images[0].image_data, IMAGE_SIZE, NULL, cpu_func_ptr);
    sleep(1);
  }
  //onvm_send_gpu_msg_to_mgr(give_gpu_model, MSG_NF_GPU_READY);
  return 0;
}

/* function to load gpu file */
void load_gpu_file(void);
void load_gpu_file(void){
  /* convert the filename to wchar_t */
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
  
  give_gpu_model->model_index = atoi(ml_model); //alexnet
  //memcpy(name_of_model, name, strlen(name));
  //memcpy(_nfinfo,nf_info, sizeof(struct onvm_nf_info));
  give_gpu_model->nf = nf_info;
  memcpy(give_gpu_model->model_name, ml_model, sizeof(char)*20);
  onvm_send_gpu_msg_to_mgr(give_gpu_model,MSG_GIVE_GPU_MODEL);
  printf("Sent message to mgr \n");
}

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
  }*/

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
	load_gpu_file();
	//init the images for GPU
	//image_init();

	//receive packets.
	onvm_nflib_run(nf_info, &packet_handler);
        printf("If we reach here, program is ending\n");
        return 0;
}
