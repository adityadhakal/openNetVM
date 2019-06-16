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
#include <sys/types.h>

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
#include "tensorrt_api.h"
#include "onvm_ml_libraries.h"
#include "onvm_cntk_api.h"

#include <cuda_runtime.h>
#include <cuda.h>

#define MSG_RING_NAME1 "msg_ring1"
#define MSG_RING_SIZE 128

#define NF_TAG "bridge"


/* Struct that contains information about this NF */
struct onvm_nf_info *nf_info;

/* ML related variables */
//static char *input_file_name = NULL;
static char *batchsize = NULL;
static char *gpu_percent = NULL;
void * cpu_func_ptr = NULL;
void * gpu_func_ptr = NULL;

void cuda_register_memseg_info(void);

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
  //printf("packet addr : %p \n", pkt);
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

void cuda_register_memseg_info(void){

  //get the memory config
  struct rte_config * rte_config = rte_eal_get_configuration();

  //now get the memory locations
  struct rte_mem_config * memory_config = rte_config->mem_config;

  int i;
  struct timespec begin,end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  cudaError_t cudaerror;
  printf("RTE_MAX_MEMSEG_LIST %d \n", RTE_MAX_MEMSEG_LISTS);
    printf("RTE_MAX_MEMSEG__PER_LIST %d \n", RTE_MAX_MEMSEG_PER_LIST);
  for(i = 0; i<RTE_MAX_MEMSEG_LISTS; i++){
    struct rte_memseg_list *memseg_ptr = &(memory_config->memsegs[i]);
    if(memseg_ptr->page_sz > 0 && memseg_ptr->socket_id == (int)rte_socket_id()){
      //printf("Pointer to huge page %p and size of the page %"PRIu64"\n",memseg_ptr->base_va, memseg_ptr->page_sz);
      cudaerror = cudaHostRegister(memseg_ptr->base_va, memseg_ptr->page_sz, cudaHostRegisterDefault);
      if(cudaerror != cudaSuccess){
	printf("Failed to pin error %d\n", cudaerror);
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double time_taken_to_register = (end.tv_sec-begin.tv_sec)*1000000.0 + (end.tv_nsec-begin.tv_nsec)/1000.0;
  printf("Total time taken to register the mempages to cuda is %f micro-seconds \n",time_taken_to_register);
    
}

ml_framework_operations_t ml_functions;

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

	
	//register the funtion for processing the GPU related message from manager
	//register_gpu_msg_handling_function(&function_to_process_gpu_message);

	//create a struct of functions from the library to register with NFlib.
	ml_fw_load_model load_mdl = cntk_load_model;
	ml_functions.load_model_fptr = load_mdl;
	ml_functions.link_model_fptr = cntk_link_pointers;
	ml_functions.infer_batch_fptr = cntk_infer_batch;
	nflib_register_ml_fw_operations(&ml_functions);

	//put in the batch size
	//nf_info->user_batch_size = atoi(batchsize);
	nf_info->user_batch_size = 1;

	pid_t pid = getpid();

	char huge_page_addr[128];
	char answer[128];
	//a better method came along.. to use environ and put it in the file
	sprintf(huge_page_addr, "grep \"kernelpagesize_kB=1048576\" /proc/%d/numa_maps | awk '{print $1;}'",pid);

	FILE* fp;
	fp = popen(huge_page_addr, "r");
	if(fp == NULL){
	  printf("Failed to run command\n");
	  exit(1);
	}

	/* now load the output in the path */
	while(fgets(answer, sizeof(answer)-1, fp) != NULL){
	  //printf("%s", path);
	}

	/* close the file */
	pclose(fp);

	unsigned long ul;
	sscanf(answer,"%lx", &ul);
	

	printf("huge_pages address %s pointer %p \n",answer, (void *)ul);
	//cudaError_t cudaerror;
	//cudaerror = cudaHostRegister((void *) ul, (1024*1024*1024), cudaHostRegisterDefault);
	//if(cudaerror == cudaSuccess)
	//printf("memory pinned-------\n");
	//receive packets.
 
	int answer2;
	cudaDeviceGetAttribute(&answer2,cudaDevAttrCanUseHostPointerForRegisteredMem, 0);
	printf("Can use host pointer for registered mem %d\n",answer2);
	onvm_nflib_run(nf_info, &packet_handler);
        printf("If we reach here, program is ending\n");
        return 0;
}
