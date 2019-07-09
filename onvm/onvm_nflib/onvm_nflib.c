/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2017 George Washington University
 *            2015-2017 University of California Riverside
 *            2016-2017 Hewlett Packard Enterprise Development LP
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
 ********************************************************************/

/******************************************************************************

 onvm_nflib.c


 File containing all functions of the NF API


 ******************************************************************************/

/***************************Standard C library********************************/

#include <getopt.h>
#include <signal.h>

/******************************DPDK libraries*********************************/
#include "rte_malloc.h"

/*****************************Internal headers********************************/

#include "onvm_nflib_internal.h"
#include "onvm_nflib.h"
#include <rte_eal_memconfig.h>
#include <rte_eal.h>
#ifdef ONVM_GPU
#include <cuda_runtime.h>
#include <strings.h> //for ffs
#endif
#include "onvm_includes.h"
#include "onvm_sc_common.h"

/**********************************Macros*************************************/

// Possible NF packet consuming modes
#define NF_MODE_UNKNOWN 0
#define NF_MODE_SINGLE 1
#define NF_MODE_RING 2

#define ONVM_NO_CALLBACK NULL

//#define ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
#define __DEBUG__NDSYNC_LOGS__

#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
#ifdef  __DEBUG_NDSYNC_LOGS__
onvm_interval_timer_t nd_ts;
int64_t max_nd, min_nd, avg_nd, delta_nd;
#endif
#endif
/*********************************************************************/
/*            NF LIB Feature flags specific functions                */
/*********************************************************************/

#ifdef TEST_MEMCPY_OVERHEAD
static inline void allocate_base_memory(void) {
	base_memory = calloc(1,2*MEMCPY_SIZE);
}
static inline void do_memcopy(void *from_pointer) {
	if(likely(base_memory && from_pointer)) {
		memcpy(base_memory,from_pointer,MEMCPY_SIZE);
	}
}
#endif //TEST_MEMCPY_OVERHEAD

/*********************************************************************/
nf_explicit_callback_function nf_ecb = NULL;
static uint8_t need_ecb = 0;
void register_explicit_callback_function(nf_explicit_callback_function ecb) {
	if (ecb) {
		nf_ecb = ecb;
	}
	return;
}
/******************************************************************************/
/*                        HISTOGRAM DETAILS                                   */
/******************************************************************************/

/*********************** ADITYA'S CODE ****************************************/
/**************** ONVM_GPU ******************/
#ifdef ONVM_GPU

gpu_message_processing_func nf_gpu_func;
void register_gpu_msg_handling_function(gpu_message_processing_func gmpf) {
	if(gmpf) {
		nf_gpu_func = gmpf;
	}
	return;
}

//old code
//timer threads...
#define NF_INFERENCE_PERIOD_MS 100

void initialize_ml_timers(struct onvm_nf_info *nf_info);
static void conduct_inference(__attribute__((unused)) struct rte_timer *ptr_timer,
		void *ptr_data);
struct rte_timer image_inference_timer;

/*
 //function to execute a dummy image
 void execute_dummy_image(void *function_ptr, int image_size);

 // copy the data to existing image batch 
 static inline void copy_data_to_image_batch(void *packet_data, struct onvm_nf_info *nf_info, int batch_size);
 */

//function to get the batch aggregation mempool and dev buffer mempool
static void get_batch_agg_and_dev_buffer_mempool(struct onvm_nf_info * info);

/* pinning the shared memory */
int cuda_register_shared_memory(void);

//GPU callback function to report after the evaluation is finished...
void gpu_image_callback_function(void *data);

//following are utilized while parsing the NF arguments so should stay
static char *ml_model_file;//filename of model
static uint16_t ml_model_number;//ml model number

/* A wrapper to replace NF callback */
typedef int (*process_batch_NF)(struct onvm_nf_info *nf_info,
		void **pkts, __attribute__ ((unused)) unsigned nb_pkts,
		__attribute__ ((unused)) pkt_handler_func handler);

//this variable will be updated by NF during init time so we know which function to send batch of packets to
process_batch_NF current_packet_processing_batch;

#endif

/************************************API**************************************/

#ifdef USE_CGROUPS_PER_NF_INSTANCE
#include <stdlib.h>
uint32_t get_nf_core_id(void);
void init_cgroup_info(struct onvm_nf_info *nf_info);
int set_cgroup_cpu_share(struct onvm_nf_info *nf_info, unsigned int share_val);

uint32_t get_nf_core_id(void) {
	return rte_lcore_id();
}

int set_cgroup_cpu_share(struct onvm_nf_info *nf_info, unsigned int share_val) {
	/*
	 unsigned long shared_bw_val = (share_val== 0) ?(1024):(1024*share_val/100); //when share_val is relative(%)
	 if (share_val >=100) {
	 shared_bw_val = shared_bw_val/100;
	 }*/

	unsigned long shared_bw_val = (share_val== 0) ?(1024):(share_val); //when share_val is absolute bandwidth
	const char* cg_set_cmd = get_cgroup_set_cpu_share_cmd(nf_info->instance_id, shared_bw_val);
	printf("\n CMD_TO_SET_CPU_SHARE: %s", cg_set_cmd);
	int ret = system(cg_set_cmd);
	if (0 == ret) {
		nf_info->cpu_share = shared_bw_val;
	}
	return ret;
}
void init_cgroup_info(struct onvm_nf_info *nf_info) {
	int ret = 0;
	const char* cg_name = get_cgroup_name(nf_info->instance_id);
	const char* cg_path = get_cgroup_path(nf_info->instance_id);
	printf("\n NF cgroup name and path: %s, %s", cg_name,cg_path);

	/* Check and create the CGROUP if necessary */
	const char* cg_crt_cmd = get_cgroup_create_cgroup_cmd(nf_info->instance_id);
	printf("\n CMD_TO_CREATE_CGROUP_for_NF: %d, %s", nf_info->instance_id, cg_crt_cmd);
	ret = system(cg_crt_cmd);

	/* Add the pid to the CGROUP */
	const char* cg_add_cmd = get_cgroup_add_task_cmd(nf_info->instance_id, nf_info->pid);
	printf("\n CMD_TO_ADD_NF_TO_CGROUP: %s", cg_add_cmd);
	ret = system(cg_add_cmd);

	/* Retrieve the mapped core-id */
	nf_info->core_id = get_nf_core_id();

	/* Initialize the cpu.shares to default value (100%) */
	ret = set_cgroup_cpu_share(nf_info, 0);

	printf("NF on core=%u added to cgroup: %s, ret=%d", nf_info->core_id, cg_name,ret);
	return;
}
#endif //USE_CGROUPS_PER_NF_INSTANCE

/******************************FTMB Helper Functions********************************/
#ifdef MIMIC_FTMB
//#define SV_ACCES_PER_PACKET (2)   //moved as configurable variable.
typedef struct pal_packet {
	uint64_t pal_counter;
	uint64_t pal_info;
}pal_packet_t;
static inline int send_packets_out(uint8_t port_id, uint16_t queue_id, struct rte_mbuf **tx_pkts, uint16_t nb_pkts) {

	return 0;
	uint16_t sent_packets = 0; //rte_eth_tx_burst(port_id,queue_id, tx_pkts, nb_pkts);
	if(unlikely(sent_packets < nb_pkts)) {
		uint16_t i = sent_packets;
		for(; i< nb_pkts;i++)
		onvm_nflib_drop_pkt(tx_pkts[i]);
	}
	/*
	 {
	 volatile struct tx_stats *tx_stats = &(ports->tx_stats);
	 tx_stats->tx[port_id] += sent_packets;
	 tx_stats->tx_drop[port_id] += (nb_pkts - sent_packets);
	 }
	 */
	return sent_packets;
	rte_eth_tx_burst(port_id,queue_id, tx_pkts, nb_pkts);
}
int generate_and_transmit_pals_for_batch(__attribute__((unused)) void **pktsTX, unsigned tx_batch_size, __attribute__((unused)) unsigned non_det_evt, __attribute__((unused)) uint64_t ts_info);
int generate_and_transmit_pals_for_batch(__attribute__((unused)) void **pktsTX, unsigned tx_batch_size, __attribute__((unused)) unsigned non_det_evt, __attribute__((unused)) uint64_t ts_info) {
	uint32_t i = 0;
	static uint64_t pal_counter = 0;
	struct rte_mbuf *out_pkt = NULL;
	struct onvm_pkt_meta *pmeta = NULL;
	struct ether_hdr *eth_hdr = NULL;
	pal_packet_t *p_hdr;
	size_t pkt_size = 0;
	size_t data_len = sizeof(struct ether_hdr) + sizeof(uint16_t) + sizeof(uint16_t);
	int ret = 0;
	//void *pkts[CLIENT_SHADOW_RING_SIZE];
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	static uint64_t pktcounterm=0; uint64_t start_cycle=0;onvm_interval_timer_t ts_p;
	pktcounterm+=(tx_batch_size*SV_ACCES_PER_PACKET);
	if(pktcounterm >= (1000*1000*20)) {
		onvm_util_get_start_time(&ts_p);
		start_cycle = onvm_util_get_current_cpu_cycles();
	}
#endif
	//for(i=0; i<(SV_ACCES_PER_PACKET); i++) {
	for(i=0; i<(tx_batch_size*SV_ACCES_PER_PACKET); i++) {

		//Allocate New Packet
		out_pkt = rte_pktmbuf_alloc(pktmbuf_pool);
		if (out_pkt == NULL) {
			rte_pktmbuf_free(out_pkt);
			//rte_exit(EXIT_FAILURE, "onvm_nflib:Failed to alloc packet for %x, %li!! \n", i, pal_counter);
			//RTE_LOG(ERROR, APP, "onvm_nflib:Failed to alloc packet for %x, %li!! \n", i, pal_counter);
			fprintf(stderr, "onvm_nflib:Failed to alloc packet for %x, %li!! \n", i, pal_counter);
			return -1;
		}

		//set packet properties
		pkt_size = sizeof(struct ether_hdr) + sizeof(pal_packet_t);
		out_pkt->data_len = MAX(pkt_size, data_len);//todo: crirical error if 0 or lesser than pkt_len: mooongen discards; check again and confirm
		out_pkt->pkt_len = pkt_size;

		//Set Ethernet Header info
		eth_hdr = onvm_pkt_ether_hdr(out_pkt);
		eth_hdr->ether_type = rte_cpu_to_be_16((ETHER_TYPE_RSYNC_DATA+1));
		//ether_addr_copy(&ports->mac[port], &eth_hdr->s_addr);
		//ether_addr_copy((const struct ether_addr *)&rsync_node_info.mac_addr_bytes, &eth_hdr->d_addr);

		//SET PAL DATA
		p_hdr = rte_pktmbuf_mtod_offset(out_pkt, pal_packet_t*, sizeof(struct ether_hdr));
		p_hdr->pal_counter = pal_counter++;
		p_hdr->pal_info = 0xBADABADBBADCBADD;

		//SEND PACKET OUT/SET METAINFO
		pmeta = onvm_get_pkt_meta(out_pkt);
		pmeta->destination =RSYNC_TX_OUT_PORT;
		pmeta->action = ONVM_NF_ACTION_DROP;

		send_packets_out(0, RSYNC_TX_PORT_QUEUE_ID_0, &out_pkt, 1);
		if(unlikely(-ENOBUFS == (ret = rte_ring_enqueue(tx_ring, out_pkt)))) {
			do {
#ifdef INTERRUPT_SEM
				onvm_nf_yeild(nf_info,YIELD_DUE_TO_FULL_TX_RING);
				ret = rte_ring_enqueue(tx_ring, out_pkt);
#endif
			}while((ret == -ENOBUFS) && keep_running);
		}
	}
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	if((pktcounterm)>=(1000*1000*20)) {
		fprintf(stdout, "PAL GENERATION TIME for (%x) SV : %li(ns) and %li (cycles) for packet:%d \n", SV_ACCES_PER_PACKET, onvm_util_get_elapsed_time(&ts_p), onvm_util_get_elapsed_cpu_cycles(start_cycle), (int)pkt_size);
		pktcounterm=0;
	}
#endif
	return ret;
}
#endif
/******************************Timer Helper functions*******************************/
#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
static void
stats_timer_cb(__attribute__((unused)) struct rte_timer *ptr_timer,
		__attribute__((unused)) void *ptr_data) {

#ifdef INTERRUPT_SEM
	counter = SAMPLING_RATE;
#endif //INTERRUPT_SEM

	//printf("\n On core [%d] Inside Timer Callback function: %"PRIu64" !!\n", rte_lcore_id(), rte_rdtsc_precise());
	//printf("Echo %d", system("echo > hello_timer.txt"));
	//printf("\n Inside Timer Callback function: %"PRIu64" !!\n", rte_rdtsc_precise());
}

static inline void
init_nflib_timers(void) {
	//unsigned cur_lcore = rte_lcore_id();
	//unsigned timer_core = rte_get_next_lcore(cur_lcore, 1, 1);
	//printf("cur_core [%u], timer_core [%u]", cur_lcore,timer_core);
	rte_timer_subsystem_init();
	printf("Timer subsystem init\n");
	rte_timer_init(&stats_timer);
	rte_timer_reset_sync(&stats_timer,
			(NF_STATS_PERIOD_IN_MS * rte_get_timer_hz()) / 1000,
			PERIODICAL,
			rte_lcore_id(),//timer_core
			&stats_timer_cb, NULL
	);
}
#endif
/******************************Timer Helper functions*******************************/
#ifdef INTERRUPT_SEM
void onvm_nf_yeild(__attribute__((unused))struct onvm_nf_info* info, __attribute__((unused)) uint8_t reason_rxtx) {

	/* For now discard the special NF instance and put all NFs to wait */
	// if ((!ONVM_SPECIAL_NF) || (info->instance_id != 1)) { }
#ifdef ENABLE_NF_YIELD_NOTIFICATION_COUNTER
	if(reason_rxtx) {
		this_nf->stats.tx_drop+=1;
	} else {
		this_nf->stats.yield_count +=1;
	}
#endif

#ifdef USE_POLL_MODE
	return;
#endif

	//do not block if running status is off.
	if(unlikely(!keep_running)) return;

	rte_atomic16_set(flag_p, 1);//rte_atomic16_cmpset(flag_p, 0, 1);
#ifdef USE_SEMAPHORE
	sem_wait(mutex);
#endif
	//rte_atomic16_set(flag_p, 0); //out of block rte_atomic16_cmpset(flag_p, 1, 0);

	//check and trigger explicit callabck before returning.
	if(need_ecb && nf_ecb) {
		need_ecb = 0;
		nf_ecb();
	}
}
#ifdef INTERRUPT_SEM
static inline void onvm_nf_wake_notify(__attribute__((unused))struct onvm_nf_info* info);
static inline void onvm_nf_wake_notify(__attribute__((unused))struct onvm_nf_info* info)
{
#ifdef USE_SEMAPHORE
	sem_post(mutex);
	//printf("Triggered to wakeup the NF thread internally");
#endif
	return;
}
static inline void onvm_nflib_implicit_wakeup(void);
static inline void onvm_nflib_implicit_wakeup(void) {
	// if ((rte_atomic16_read(flag_p) ==1)) {
	rte_atomic16_set(flag_p, 0);
	onvm_nf_wake_notify(nf_info);
	//}
}
#endif //#ifdef INTERRUPT_SEM

static inline void start_ppkt_processing_cost(uint64_t *start_tsc) {
	if (unlikely(counter % SAMPLING_RATE == 0)) {
		*start_tsc = onvm_util_get_current_cpu_cycles(); //compute_start_cycles(); //rte_rdtsc();
	}
}
static inline void end_ppkt_processing_cost(uint64_t start_tsc) {
	if (unlikely(counter % SAMPLING_RATE == 0)) {
		this_nf->stats.comp_cost = onvm_util_get_elapsed_cpu_cycles(start_tsc);
		if (likely(this_nf->stats.comp_cost > RTDSC_CYCLE_COST)) {
			this_nf->stats.comp_cost -= RTDSC_CYCLE_COST;
		}
#ifdef STORE_HISTOGRAM_OF_NF_COMPUTATION_COST
		hist_store_v2(&nf_info->ht2, this_nf->stats.comp_cost);
		//avoid updating 'nf_info->comp_cost' as it will be calculated in the weight assignment function
		//nf_info->comp_cost  = hist_extract_v2(&nf_info->ht2,VAL_TYPE_RUNNING_AVG);
#else   //just save the running average
		nf_info->comp_cost = (nf_info->comp_cost == 0)? (this_nf->stats.comp_cost): ((nf_info->comp_cost+this_nf->stats.comp_cost)/2);

#endif //STORE_HISTOGRAM_OF_NF_COMPUTATION_COST

#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
		counter = 1;
#endif  //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION

#ifdef ENABLE_ECN_CE
		hist_store_v2(&nf_info->ht2_q, rte_ring_count(rx_ring));
#endif
	}

#ifndef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
	counter++; //computing for first packet makes also account reasonable cycles for cache-warming.
#endif //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
}
#endif  //INTERRUPT_SEM
#ifdef ENABLE_NFV_RESL
static inline void
onvm_nflib_wait_till_notification(struct onvm_nf_info *nf_info) {
	//printf("\n Client [%d] is paused and waiting for SYNC Signal\n", nf_info->instance_id);
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	onvm_util_get_start_time(&ts);
#endif
	do {
		onvm_nf_yeild(nf_info,YEILD_DUE_TO_EXPLICIT_REQ);
		/* Next Check for any Messages/Notifications */
		onvm_nflib_dequeue_messages(nf_info);
	}while(((NF_PAUSED == (nf_info->status & NF_PAUSED))||(NF_WT_ND_SYNC_BIT == (nf_info->status & NF_WT_ND_SYNC_BIT))) && keep_running );

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	//printf("SIGNAL_TIME(PAUSE-->RESUME): %li ns\n", onvm_util_get_elapsed_time(&ts));
#endif
	//printf("\n Client [%d] completed wait on SYNC Signal \n", nf_info->instance_id);
}
#endif //ENABLE_NFV_RESL

static inline void onvm_nflib_check_and_wait_if_interrupted(
		__attribute__((unused))                                                                                                          struct onvm_nf_info *nf_info);
static inline void onvm_nflib_check_and_wait_if_interrupted(
		__attribute__((unused))                                                                                                          struct onvm_nf_info *nf_info) {
#if defined (INTERRUPT_SEM) && ((defined(NF_BACKPRESSURE_APPROACH_2) || defined(USE_ARBITER_NF_EXEC_PERIOD)) || defined(ENABLE_NFV_RESL))
	if(unlikely(NF_PAUSED == (nf_info->status & NF_PAUSED))) {
		//printf("\n Explicit Pause request from ONVM_MGR\n ");
		onvm_nflib_wait_till_notification(nf_info);
		//printf("\n Explicit Pause Completed by NF\n");
	}
	else if (unlikely(rte_atomic16_read(flag_p) ==1)) {
		//printf("\n Explicit Yield request from ONVM_MGR\n ");
		onvm_nf_yeild(nf_info,YEILD_DUE_TO_EXPLICIT_REQ);
		//printf("\n Explicit Yield Completed by NF\n");
	}
#endif
}

#if defined(ENABLE_SHADOW_RINGS)
static inline void onvm_nflib_handle_tx_shadow_ring(void);
static inline void onvm_nflib_handle_tx_shadow_ring(void) {

	/* Foremost Move left over processed packets from Tx shadow ring to the Tx Ring if any */
	if(unlikely( (rte_ring_count(tx_sring)))) {
		uint16_t nb_pkts = CLIENT_SHADOW_RING_SIZE;
		uint16_t tx_spkts;
		void *pkts[CLIENT_SHADOW_RING_SIZE];
		do
		{
			// Extract packets from Tx shadow ring
			tx_spkts = rte_ring_dequeue_burst(tx_sring, pkts, nb_pkts);

			//fprintf(stderr, "\n Move processed packets from Shadow Tx Ring to Tx Ring [%d] packets from shadow ring( Re-queue)!\n", tx_spkts);
			//Push the packets to the Tx ring
			//if(unlikely(rte_ring_enqueue_bulk(tx_ring, pkts, tx_spkts) == -ENOBUFS)) { //OLD API
			if(unlikely(rte_ring_enqueue_bulk(tx_ring, pkts, tx_spkts) == 0)) { //new API
#ifdef INTERRUPT_SEM
				//To preserve the packets, re-enqueue packets back to the the shadow ring
				rte_ring_enqueue_bulk(tx_sring, pkts, tx_spkts);

				//printf("\n Yielding till Tx Ring has space for tx_shadow buffer Packets \n");
				onvm_nf_yeild(nf_info,YIELD_DUE_TO_FULL_TX_RING);
				//printf("\n Resuming till Tx Ring has space for tx_shadow buffer Packets \n");
#endif
			}
		}while(rte_ring_count(tx_sring) && keep_running);
		this_nf->stats.tx += tx_spkts;
	}
}
#endif

#ifdef ENABLE_REPLICA_STATE_UPDATE
static inline void synchronize_replica_nf_state_memory(void) {

	//if(likely(nf_info->nf_state_mempool && pReplicaStateMempool))
	if(likely(dirty_state_map && dirty_state_map->dirty_index)) {
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
#if 0
		static int count = 0;uint64_t start_cycle=0;
		count++;
		if(count == 1000*1000*20) {
			onvm_util_get_start_time(&ts);
			start_cycle = onvm_util_get_current_cpu_cycles();
		}
#endif
#endif
		//Note: Must always ensure that dirty_map is carried over first; so that the remote replica can use this value to update only the changed states
		uint64_t dirty_index = dirty_state_map->dirty_index;
		uint64_t copy_index = 0;
		uint64_t copy_setbit = 0;
		uint16_t copy_offset = 0;
		for(;dirty_index;copy_index++) {
			copy_setbit = (1L<<(copy_index));
			if(dirty_index&copy_setbit) {
				copy_offset = copy_index*DIRTY_MAP_PER_CHUNK_SIZE;
				rte_memcpy(( ((uint8_t*)pReplicaStateMempool)+copy_offset),(((uint8_t*)nf_info->nf_state_mempool)+copy_offset),DIRTY_MAP_PER_CHUNK_SIZE);
				dirty_index^=copy_setbit;
			} // copy_index++;
		}
		dirty_state_map->dirty_index =0;
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
#if 0
		if(count == 1000*1000*20) {
			fprintf(stdout, "STATE REPLICATION TIME (Scan + Copy): %li(ns) and %li (cycles) \n", onvm_util_get_elapsed_time(&ts), onvm_util_get_elapsed_cpu_cycles(start_cycle));
			count=0;
		}
#endif
#endif
	}
	return;
}
#endif

#ifdef ENABLE_NFLIB_PER_FLOW_TS_STORE
//update the TS for the processed packet
static inline void update_processed_packet_ts(void **pkts, unsigned max_packets);
static inline void update_processed_packet_ts(void **pkts, unsigned max_packets) {
	if(!this_nf->per_flow_ts_info) return;
	uint16_t i, ft_index=0;
	uint64_t ts[NF_PKT_BATCH_SIZE];
	onvm_util_get_marked_packet_timestamp((struct rte_mbuf**)pkts, ts, max_packets);
	for(i=0; i< max_packets;i++) {
		struct onvm_pkt_meta *meta = onvm_get_pkt_meta((struct rte_mbuf*) pkts[i]);
#ifdef ENABLE_FT_INDEX_IN_META
		ft_index = meta->ft_index; //(uint16_t) MAP_SDN_FT_INDEX_TO_VLAN_STATE_TBL_INDEX(meta->ft_index);
#else
		{
			struct onvm_flow_entry *flow_entry = NULL;
			onvm_flow_dir_get_pkt((struct rte_mbuf*) pkts[i], &flow_entry);
			if(flow_entry) {
				ft_index = meta->ft_index; //(uint16_t) MAP_SDN_FT_INDEX_TO_VLAN_STATE_TBL_INDEX(flow_entry->entry_index);
			} else continue;
		}
#endif
		onvm_per_flow_ts_info_t *t_info = (onvm_per_flow_ts_info_t*)(((dirty_mon_state_map_tbl_t*)this_nf->per_flow_ts_info)+1);
		t_info[ft_index].ts = ts[i];
	}
}
#endif

static inline int onvm_nflib_fetch_packets(void **pkts, unsigned max_packets);
static inline int onvm_nflib_fetch_packets(void **pkts, unsigned max_packets) {
#if defined(ENABLE_SHADOW_RINGS)

	/* Address the buffers in the Tx Shadow Ring before starting to process the new packets */
	onvm_nflib_handle_tx_shadow_ring();

	/* First Dequeue the packets pulled from Rx Shadow Ring if not empty*/
	if (unlikely( (rte_ring_count(rx_sring)))) {
		max_packets = rte_ring_dequeue_burst(rx_sring, pkts, max_packets);
		fprintf(stderr, "Dequeued [%d] packets from shadow ring( Re-Run)!\n", max_packets);
	}
	/* ELSE: Get Packets from Main Rx Ring */
	else
#endif
	max_packets = (uint16_t) rte_ring_dequeue_burst(rx_ring, pkts, max_packets,
			NULL);

	if (likely(max_packets)) {
#if defined(ENABLE_SHADOW_RINGS)
		/* Also enqueue the packets pulled from Rx ring or Rx Shadow into Rx Shadow Ring */
		if (unlikely(rte_ring_enqueue_bulk(rx_sring, pkts, max_packets) == 0)) {
			fprintf(stderr, "Enqueue: %d packets to shadow ring Failed!\n", max_packets);
		}
#endif
	} else { //if(0 == max_packets){
#ifdef INTERRUPT_SEM
	//printf("\n Yielding till Rx Ring has Packets to process \n");
	onvm_nf_yeild(nf_info,YIELD_DUE_TO_EMPTY_RX_RING);
	//printf("\n Resuming from Rx Ring has Packets to process \n");
#endif
	}
	return max_packets;
}
static
inline int onvm_nflib_post_process_packets_batch(struct onvm_nf_info *nf_info,
		void **pktsTX, unsigned tx_batch_size,
		__attribute__((unused)) unsigned non_det_evt,
		__attribute__((unused))                                                                                                          uint64_t ts_info);
static
inline int onvm_nflib_post_process_packets_batch(struct onvm_nf_info *nf_info,
		void **pktsTX, unsigned tx_batch_size,
		__attribute__((unused)) unsigned non_det_evt,
		__attribute__((unused))                                                                                                          uint64_t ts_info) {
	int ret = 0;
	/* Perform Post batch processing actions */
	/** Atomic Operations:
	 * Synchronize the NF Memory State
	 * Update TS of last processed packet.
	 * Clear the Processed Batch of Rx packets.
	 */
#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
	if (unlikely(non_det_evt)) {
		if (likely(nf_info->bNDSycn)) { //if(unlikely(bNDSync)) {
			//explicitly move to pause state and wait till notified.
			//nf_info->status = NF_PAUSED|NF_WT_ND_SYNC_BIT;
			nf_info->status |= NF_WT_ND_SYNC_BIT;
#ifdef  __DEBUG_NDSYNC_LOGS__
			//printf("\n Client [%d] is halting due to second ND at: [%li] while first [%li] is not committed! waiting for SYNC Signal\n", nf_info->instance_id,ts_info, nf_info->bLastPktId);
			onvm_util_get_start_time(&nd_ts);
			//printf("RUN-->ND_SYNC): %i ns\n", 1);
#endif
			//wait_till_the NDSync is not signalled to be committed
			onvm_nflib_wait_till_notification(nf_info);
#ifdef  __DEBUG_NDSYNC_LOGS__
			// printf("\n\n\n\n\n\n\n\n$$$$$$$$$$$$WAIT_TIME(ND_SYNC): %li ns $$$$$$$$$$$$$\n\n\n\n\n\n\n", (delta_nd=onvm_util_get_elapsed_time(&nd_ts)));
			if(min_nd==0 || delta_nd < min_nd) min_nd= delta_nd;
			if(delta_nd > max_nd)max_nd=delta_nd;
			if(avg_nd) avg_nd = (avg_nd+delta_nd)/2;
			else avg_nd= delta_nd;
			//printf("\n\n In wait_till_Notitfication: WAIT_TIME_STATS(ND_SYNC):\n Cur=%li\n Min= %li\n Max: %li \n Avg: %li \n", delta_nd, min_nd, max_nd, avg_nd);
			nf_info->min_nd=min_nd; nf_info->max_nd=max_nd; nf_info->avg_nd=avg_nd;
#endif
		} else {
#ifdef  __DEBUG_NDSYNC_LOGS__
			//printf("\n Client [%d] got first ND SYNC event at: %li! \n", nf_info->instance_id, ts_info);
			onvm_util_get_start_time(&nd_ts);
#endif
		}
		nf_info->bNDSycn = 1; //bNDSync = 1;    //set the NDSync to True again
		nf_info->bLastPktId = ts_info;//can be used to check if Resume message carries TxTs of latest synced packet
	}
	//printf("\n %d", non_det_evt);
#endif
#ifdef REPLICA_STATE_UPDATE_MODE_PER_BATCH
	synchronize_replica_nf_state_memory();
#endif
#ifdef PER_FLOW_TS_UPDATE_PER_BATCH
	//update the TS for the processed batch of packets
	update_processed_packet_ts(pktsTX,tx_batch_size);
#endif

#if defined(SHADOW_RING_UPDATE_PER_BATCH)
	//rte_ring_enqueue_bulk(tx_sring, pktsTX, tx_batch_size);
	//void *pktsRX[NF_PKT_BATCH_SIZE];
	//rte_ring_sc_dequeue_bulk(rx_sring, pktsRX,NF_PKT_BATCH_SIZE ); //for now Bypass as we do at the end (down)
#endif

#if defined(TEST_MEMCPY_MODE_PER_BATCH)
	do_memcopy(nf_info->nf_state_mempool);
#endif //TEST_MEMCPY_OVERHEAD

#ifdef MIMIC_FTMB
	generate_and_transmit_pals_for_batch(pktsTX, tx_batch_size, non_det_evt,ts_info);
#endif
	//if(likely(tx_batch_size)) {
	//if(likely(-ENOBUFS != (ret = rte_ring_enqueue_bulk(tx_ring, pktsTX, tx_batch_size, NULL)))) {
	if (likely(
			0
					!= (ret = rte_ring_enqueue_bulk(tx_ring, pktsTX,
							tx_batch_size, NULL)))) {
		this_nf->stats.tx += tx_batch_size;
	} else {
#if defined(NF_LOCAL_BACKPRESSURE)
		do {
#ifdef INTERRUPT_SEM
			//printf("\n Yielding till Tx Ring has place to store Packets\n");
			onvm_nf_yeild(nf_info, YIELD_DUE_TO_FULL_TX_RING);
			//printf("\n Resuming from Tx Ring wait to store Packets\n");
#endif
			if (tx_batch_size > rte_ring_free_count(tx_ring)) {
				continue;
			}
			//if((ret = rte_ring_enqueue_bulk(tx_ring,pktsTX,tx_batch_size,NULL)) != -ENOBUFS){ ret = 0; break;}
			if((ret = rte_ring_enqueue_bulk(tx_ring,pktsTX,tx_batch_size,NULL)) != 0) {ret = 0; break;}
		}while (ret && ((this_nf->info->status==NF_RUNNING) && keep_running));
		this_nf->stats.tx += tx_batch_size;
#endif  //NF_LOCAL_BACKPRESSURE
	}
#if defined(ENABLE_SHADOW_RINGS)
	/* Finally clear all packets from the Tx Shadow Ring and also Rx shadow Ring ::only if packets from shadow ring have been flushed to Tx Ring: Reason, NF might get paused or stopped */
	if(likely(ret == 0)) {
		rte_ring_sc_dequeue_burst(tx_sring,pktsTX,rte_ring_count(tx_sring));
		if(unlikely(rte_ring_count(rx_sring))) {
			//These are the held packets in the NF in this round:
			rte_ring_sc_dequeue_burst(rx_sring,pktsTX,rte_ring_count(rx_sring));
			//fprintf(stderr, "BATCH END: %d packets still in Rx shadow ring!\n", rte_ring_sc_dequeue_burst(rx_sring,pkts,rte_ring_count(rx_sring)));
		}
	}
#endif
	//}
	return ret;
}
static inline int onvm_nflib_process_packets_batch(struct onvm_nf_info *nf_info,
		void **pkts, __attribute__ ((unused)) unsigned nb_pkts,
		__attribute__ ((unused)) pkt_handler_func handler);
static inline int onvm_nflib_process_packets_batch(struct onvm_nf_info *nf_info,
		void **pkts, __attribute__ ((unused)) unsigned nb_pkts,
		__attribute__ ((unused)) pkt_handler_func handler) {
	int ret = 0;
	uint16_t i = 0;
	uint16_t tx_batch_size = 0;

	void *pktsTX[NF_PKT_BATCH_SIZE];

	__attribute__ ((unused)) uint8_t bCurND = 0;
	__attribute__ ((unused)) uint64_t bCurNDPktId = 0;

#ifdef INTERRUPT_SEM
	// To account NFs computation cost (sampled over SAMPLING_RATE packets)
	uint64_t start_tsc = 0;
#endif
	//uint32_t imgs_aggregated=0;
	//uint32_t ret_sts =0;

	for (i = 0; i < nb_pkts; i++) {

#ifdef INTERRUPT_SEM
		start_ppkt_processing_cost(&start_tsc);
#endif

		/*
		 #ifdef ONVM_GPU

		 //we shall copy the packets here itself.. why should we give it to the handler

		 if(onvm_pkt_ipv4_hdr(pkts[i]) != NULL) {
		 //void *packet_data = rte_pktmbuf_mtod_offset((struct rte_mbuf *)pkts[i], void *, sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr));
		 //  copy_data_to_image_batch(packet_data, nf_info, nf_info->user_batch_size);
		 //send the packet to packet aggregation interface
		 //imgs_aggregated |= data_aggregation(pkts[i],nf_info->image_info,nf_info);

		 ret_sts = data_aggregation(pkts[i],nf_info->image_info, &imgs_aggregated);

		 //now check if there are any images to process
		 //if(nf_info->image_info->temp_mask) {
		 //there is an image available
		 //int num_of_images_ready = __builtin_popcount(batch_agg_info->ready_mask);
		 //aditya.. current work
		 //push data to GPU
		 //temporarily disable this
		 //load_data_to_gpu_and_execute(nf_info,nf_info->image_info, ml_operations, gpu_image_callback_function);

		 //}
		 }

		 #endif //onvm_gpu
		 #ifdef ONVM_GPU
		 ret = ret_sts;
		 #else
		 */
		ret = (*handler)((struct rte_mbuf*) pkts[i],
				onvm_get_pkt_meta((struct rte_mbuf*) pkts[i]), nf_info);

//#endif

#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT

		if (unlikely(
						onvm_get_pkt_meta((struct rte_mbuf*) pkts[i])->reserved_word
						& NF_NEED_ND_SYNC)) {
			bCurND = 1;
			bCurNDPktId = ((struct rte_mbuf*) pkts[i])->ol_flags;
		}

#endif

#if defined(TEST_MEMCPY_MODE_PER_PACKET)
		do_memcopy(nf_info->nf_state_mempool);
#endif

#ifdef INTERRUPT_SEM
		end_ppkt_processing_cost(start_tsc);
#endif  //INTERRUPT_SEM

		/* NF returns 0 to return packets or 1 to buffer */
		if (likely(ret == 0)) {
			pktsTX[tx_batch_size++] = pkts[i];

#ifdef REPLICA_STATE_UPDATE_MODE_PER_PACKET
			synchronize_replica_nf_state_memory();
#endif
#ifdef PER_FLOW_TS_UPDATE_PER_PKT
			//update the TS for the processed packet
			update_processed_packet_ts(&pkts[i],1);
#endif
#if defined(SHADOW_RING_UPDATE_PER_PKT)
			/* Move this processed packet (Head of Rx shadow Ring) to Tx Shadow Ring */
			void *pkt_rx;
			rte_ring_sc_dequeue(rx_sring, &pkt_rx);
			rte_ring_sp_enqueue(tx_sring, pkts[i]);
#endif
		} else {
#ifdef ENABLE_NF_TX_STAT_LOGS
			this_nf->stats.tx_buffer++;
#endif
#if defined(SHADOW_RING_UPDATE_PER_PKT)
			/* Remove this buffered packet from Rx shadow Ring, Should we buffer it separately, or assume NF has held on to it and NF state update reflects it. */
			void *pkt_rx;
			rte_ring_sc_dequeue(rx_sring, &pkt_rx);
			//rte_ring_sp_enqueue(rx_sring, pkts[i]); //TODO: Need separate buffer packets holder; cannot use the rx_sring
#endif
		}
	} //End Batch Process;

	/*
	 #ifdef ONVM_GPU
	 if(imgs_aggregated) {
	 load_data_to_gpu_and_execute(nf_info,nf_info->image_info, ml_operations, gpu_image_callback_function, imgs_aggregated);
	 }
	 #endif
	 */

#if defined(SHADOW_RING_UPDATE_PER_BATCH)
	void *pktsRX[NF_PKT_BATCH_SIZE];
	//clear rx_sring
	rte_ring_sc_dequeue_bulk(rx_sring, pktsRX,NF_PKT_BATCH_SIZE);
	//save processed packets in tx_sring
	rte_ring_enqueue_bulk(tx_sring, pktsTX, tx_batch_size);
#endif
	/*
	 #ifdef ONVM_GPU
	 if (likely(tx_batch_size)) {
	 rte_ring_enqueue_bulk(tx_ring, pktsTX, tx_batch_size, NULL);
	 }
	 #else
	 */
	/* Perform Post batch processing actions */
	if (likely(tx_batch_size)) {
		return onvm_nflib_post_process_packets_batch(nf_info, pktsTX,
				tx_batch_size, bCurND, bCurNDPktId);
	}
//#endif
	return ret;
}

#ifdef ONVM_GPU
static inline int onvm_nflib_process_packets_batch_gpu(struct onvm_nf_info *nf_info,
		void **pkts, __attribute__ ((unused)) unsigned nb_pkts,
		__attribute__ ((unused)) pkt_handler_func handler);

static inline int onvm_nflib_process_packets_batch_gpu(struct onvm_nf_info *nf_info,
		void **pkts, __attribute__ ((unused)) unsigned nb_pkts,
		__attribute__ ((unused)) pkt_handler_func handler) {
	int ret = 0;
	uint16_t i = 0;
	__attribute__ ((unused)) uint16_t tx_batch_size = 0;

	__attribute__ ((unused)) void *pktsTX[NF_PKT_BATCH_SIZE];

	uint32_t imgs_aggregated=0;
	uint32_t ttl_imgs_aggregated=0;

	for (i = 0; i < nb_pkts; i++) {

		//we shall copy the packets here itself.. why should we give it to the handler
		if(onvm_pkt_ipv4_hdr(pkts[i]) != NULL) {
			ret = data_aggregation(pkts[i],nf_info->image_info, &imgs_aggregated);
			ttl_imgs_aggregated |= imgs_aggregated;
			imgs_aggregated=0;
		}

		/* NF returns 0 to return packets or 1 to buffer */
		if (likely(ret == 0)) {
			pktsTX[tx_batch_size++] = pkts[i];
		} else {
#ifdef ENABLE_NF_TX_STAT_LOGS
			this_nf->stats.tx_buffer++;
#endif
		}
		onvm_get_pkt_meta((struct rte_mbuf*) pkts[i])->action = ONVM_NF_ACTION_NEXT;
	} //End Batch Process;

	if(nf_info->image_info->ready_mask) { //ttl_imgs_aggregated) { 
		ttl_imgs_aggregated=nf_info->image_info->ready_mask;
		load_data_to_gpu_and_execute(nf_info,nf_info->image_info, ml_operations, gpu_image_callback_function, ttl_imgs_aggregated);
	}

	if (likely(tx_batch_size)) {
		if( unlikely(0 == rte_ring_enqueue_bulk(tx_ring, pktsTX, tx_batch_size, NULL)))
		{
			nfs[nf_info->instance_id].stats.tx_drop += tx_batch_size;
			for (i = 0; i < tx_batch_size; i++) {
				rte_pktmbuf_free(pktsTX[i]);
			}
		}
	}

//	rte_ring_enqueue_bulk(tx_ring, pkts, nb_pkts, NULL);
//	for (i = 0; i < tx_batch_size; i++) {
//				rte_pktmbuf_free(pktsTX[i]);
//	}
	return ret;

}

#endif

int onvm_nflib_run_callback(struct onvm_nf_info* nf_info,
		pkt_handler_func handler, callback_handler_func callback) {
	void *pkts[NF_PKT_BATCH_SIZE]; //better to use (NF_PKT_BATCH_SIZE*2)
	uint16_t nb_pkts;

	pkt_hdl_func = handler;
	printf("\nClient process %d handling packets\n", nf_info->instance_id);
	printf("[Press Ctrl-C to quit ...]\n");

	/* Listen for ^C so we can exit gracefully */
	signal(SIGINT, onvm_nflib_handle_signal);

	onvm_nflib_notify_ready(nf_info);

	/* First Check for any Messages/Notifications */
	onvm_nflib_dequeue_messages(nf_info);

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	printf("WAIT_TIME(INIT-->START-->RUN-->RUNNING): %li ns\n", onvm_util_get_elapsed_time(&g_ts));
#endif

	for (; keep_running;) {
		/* check if signaled to block, then block:: TODO: Merge this to the Message above */
		onvm_nflib_check_and_wait_if_interrupted(nf_info);

		//printf("------ ***** ------- ##### We got inside keep_running after wait if interrupted ----- ***** ------\n");
#ifdef ONVM_GPU
		rte_timer_manage();
		//block the access to ring unless the NF have ring flag set
		if(nf_info->ring_flag) {
			/*
			 if(!nf_info->gpu_model){
			 //only to be used for testing overlap
			 nf_info->gpu_model = 6;
			 //our function to process the batch of packets
			 current_packet_processing_batch = onvm_nflib_process_packets_batch_gpu;
			 initialize_gpu(nf_info);
			 }
			 */
#endif
		nb_pkts = onvm_nflib_fetch_packets(pkts, NF_PKT_BATCH_SIZE);
		if (likely(nb_pkts)) {
			/* Give each packet to the user processing function */
			//nb_pkts = onvm_nflib_process_packets_batch(nf_info, pkts, nb_pkts,handler);
			nb_pkts = (*current_packet_processing_batch)(nf_info, pkts, nb_pkts,
					handler);
#ifdef ONVM_GPU
			//add the counters to number of packets that are outstanding...
			/* Since we know how many packets make images for us now.. so we can just determine how many images 
			 *are left here */
#endif //onvm_gpu
		}
#ifdef ONVM_GPU
	} //if(nf_info->ring_flag == 1)
#endif//onvm_gpu

#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
		rte_timer_manage();
#endif  //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION

		/* Finally Check for any Messages/Notifications */
		onvm_nflib_dequeue_messages(nf_info);
		//printf("------ ***** ------- ##### We got after nflib_dequeue if interrupted ----- ***** ------\n");

		if (callback) {
			keep_running = !(*callback)(nf_info) && keep_running;
		}
	}

	printf("\n NF is Exiting...!\n");
	onvm_nflib_cleanup(nf_info);
	return 0;
}

int onvm_nflib_run(struct onvm_nf_info* nf_info, pkt_handler_func handler) {
	return onvm_nflib_run_callback(nf_info, handler, ONVM_NO_CALLBACK);
}

int onvm_nflib_return_pkt(struct onvm_nf_info* nf_info, struct rte_mbuf* pkt) {
	return onvm_nflib_return_pkt_bulk(nf_info, &pkt, 1);
	/* FIXME: should we get a batch of buffered packets and then enqueue? Can we keep stats? */
	if (unlikely(rte_ring_enqueue(tx_ring, pkt) == -ENOBUFS)) {
		rte_pktmbuf_free(pkt);
		this_nf->stats.tx_drop++;
		return -ENOBUFS;
	} else {
#ifdef ENABLE_NF_TX_STAT_LOGS
		this_nf->stats.tx_returned++;
#endif
	}
	return 0;
}

int onvm_nflib_return_pkt_bulk(struct onvm_nf_info *nf_info,
		struct rte_mbuf** pkts, uint16_t count) {
	unsigned int i;
	if (pkts == NULL || count == 0)
		return -1;
	if (unlikely(
			rte_ring_enqueue_bulk(/*nfs[nf_info->instance_id].tx_q,*/tx_ring,
					(void **) pkts, count, NULL) == 0)) {
		nfs[nf_info->instance_id].stats.tx_drop += count;
		for (i = 0; i < count; i++) {
			rte_pktmbuf_free(pkts[i]);
		}
		return -ENOBUFS;
	} else
		nfs[nf_info->instance_id].stats.tx_returned += count;
	return 0;
}

void onvm_nflib_stop(__attribute__((unused))                                                                                                          struct onvm_nf_info* nf_info) {
	rte_exit(EXIT_SUCCESS, "Done.");
}

int onvm_nflib_init(int argc, char *argv[], const char *nf_tag,
		struct onvm_nf_info **nf_info_p) {
	int retval_parse, retval_final;
	struct onvm_nf_info *nf_info;
	int retval_eal = 0;
	//int use_config = 0;
	const struct rte_memzone *mz_nf;
	const struct rte_memzone *mz_port;
	const struct rte_memzone *mz_scp;
	const struct rte_memzone *mz_services;
	const struct rte_memzone *mz_nf_per_service;
	//struct rte_mempool *mp;
	struct onvm_service_chain **scp;

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	onvm_util_get_start_time(&g_ts);
#endif
	if ((retval_eal = rte_eal_init(argc, argv)) < 0)
		return -1;

	/* Modify argc and argv to conform to getopt rules for parse_nflib_args */
	argc -= retval_eal;
	argv += retval_eal;

	/* Reset getopt global variables opterr and optind to their default values */
	opterr = 0;
	optind = 1;

	if ((retval_parse = onvm_nflib_parse_args(argc, argv)) < 0)
		rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");

	/*
	 * Calculate the offset that the nf will use to modify argc and argv for its
	 * getopt call. This is the sum of the number of arguments parsed by
	 * rte_eal_init and parse_nflib_args. This will be decremented by 1 to assure
	 * getopt is looking at the correct index since optind is incremented by 1 each
	 * time "--" is parsed.
	 * This is the value that will be returned if initialization succeeds.
	 */
	retval_final = (retval_eal + retval_parse) - 1;

	/* Reset getopt global variables opterr and optind to their default values */
	opterr = 0;
	optind = 1;

	/* Lookup mempool for nf_info struct */
	nf_info_mp = rte_mempool_lookup(_NF_MEMPOOL_NAME);
	if (nf_info_mp == NULL)
		rte_exit(EXIT_FAILURE, "No Client Info mempool - bye\n");

	/* Lookup mempool for NF messages */
	nf_msg_pool = rte_mempool_lookup(_NF_MSG_POOL_NAME);
	if (nf_msg_pool == NULL)
		rte_exit(EXIT_FAILURE, "No NF Message mempool - bye\n");

	/* Initialize the info struct */
	nf_info = onvm_nflib_info_init(nf_tag);
	*nf_info_p = nf_info;

	pktmbuf_pool = rte_mempool_lookup(PKTMBUF_POOL_NAME);
	if (pktmbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get mempool for mbufs\n");

	/* Lookup mempool for NF structs */
	mz_nf = rte_memzone_lookup(MZ_NF_INFO);
	if (mz_nf == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get NF structure mempool\n");
	nfs = mz_nf->addr;

	mz_services = rte_memzone_lookup(MZ_SERVICES_INFO);
	if (mz_services == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot get service information\n");
	}
	services = mz_services->addr;

	mz_nf_per_service = rte_memzone_lookup(MZ_NF_PER_SERVICE_INFO);
	if (mz_nf_per_service == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot get NF per service information\n");
	}
	nf_per_service_count = mz_nf_per_service->addr;

	mz_port = rte_memzone_lookup(MZ_PORT_INFO);
	if (mz_port == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get port info structure\n");
	ports = mz_port->addr;

	mz_scp = rte_memzone_lookup(MZ_SCP_INFO);
	if (mz_scp == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get service chain info structre\n");
	scp = mz_scp->addr;
	default_chain = *scp;

	onvm_sc_print (default_chain);

	mgr_msg_ring = rte_ring_lookup(_MGR_MSG_QUEUE_NAME);
	if (mgr_msg_ring == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get MGR Message ring");
#ifdef ENABLE_SYNC_MGR_TO_NF_MSG
	mgr_rsp_ring = rte_ring_lookup(_MGR_RSP_QUEUE_NAME);
	if (mgr_rsp_ring == NULL)
	rte_exit(EXIT_FAILURE, "Cannot get MGR response (SYNC) ring");
#endif
	onvm_nflib_start_nf(nf_info);

#ifdef INTERRUPT_SEM
	init_shared_cpu_info(nf_info->instance_id);
#endif

#ifdef USE_CGROUPS_PER_NF_INSTANCE
	init_cgroup_info(nf_info);
#endif

#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
	init_nflib_timers();
#endif  //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION

#ifdef STORE_HISTOGRAM_OF_NF_COMPUTATION_COST
	hist_init_v2(&nf_info->ht2);    //hist_init( &ht, MAX_NF_COMP_COST_CYCLES);
#endif

#ifdef ENABLE_ECN_CE
	hist_init_v2(&nf_info->ht2_q);   //hist_init( &ht, MAX_NF_COMP_COST_CYCLES);
#endif

#ifdef TEST_MEMCPY_OVERHEAD
	allocate_base_memory();
#endif

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	int64_t ttl_elapsed = onvm_util_get_elapsed_time(&g_ts);
	printf("WAIT_TIME(INIT-->START-->Init_end): %li ns\n", ttl_elapsed);
#endif
	// Get the FlowTable Entries Exported to the NF.
	onvm_flow_dir_nf_init();

	RTE_LOG(INFO, APP, "Finished Process Init.\n");
	return retval_final;
}

int onvm_nflib_drop_pkt(struct rte_mbuf* pkt) {
	rte_pktmbuf_free(pkt);
	this_nf->stats.tx_drop++;
	return 0;
}

void notify_for_ecb(void) {
	need_ecb = 1;
#ifdef INTERRUPT_SEM
	if ((rte_atomic16_read(flag_p) ==1)) {
		onvm_nf_wake_notify(nf_info);
	}
#endif
	return;
}

int onvm_nflib_handle_msg(struct onvm_nf_msg *msg,
		__attribute__((unused))                                                                                                          struct onvm_nf_info *nf_info) {
	switch (msg->msg_type) {
	printf("\n Received MESSAGE [%d]\n!!", msg->msg_type);
case MSG_STOP:
	keep_running = 0;
	if (NF_PAUSED == (nf_info->status & NF_PAUSED)) {
		nf_info->status = NF_RUNNING;
#ifdef INTERRUPT_SEM
		onvm_nflib_implicit_wakeup(); //TODO: change this ecb call; split ecb call to two funcs. sounds stupid but necessary as cache update of flag_p takes time; otherwise results in sleep-wkup cycles
#endif
	}
	RTE_LOG(INFO, APP, "Shutting down...\n");
	break;
case MSG_NF_TRIGGER_ECB:
	notify_for_ecb();
	break;

	/* Aditya's edit ONVM_GPU*/
#ifdef ONVM_GPU
	case MSG_GPU_MODEL_PRI:
	if((*nf_gpu_func)(msg))
	printf("The NF didn't process GPU_MODEL message well \n");
	break;
	case MSG_GPU_MODEL_SEC:
	if((*nf_gpu_func)(msg))
	printf("The NF didn't process GPU_MODEL message well \n");
	break;
	case MSG_GET_GPU_READY:
	if((*nf_gpu_func)(msg))
	printf("The NF didn't process GET_GPU_READY message well \n");
	break;
	case MSG_RESTART:
	if((*nf_gpu_func)(msg))
	printf("The NF didn't process MSG_RESTART message well \n");
	break;
#endif
	/* Aditya's edit end */

case MSG_PAUSE:
	if (NF_PAUSED != (nf_info->status & NF_PAUSED)) {
		RTE_LOG(INFO, APP, "NF Status changed to Pause!...\n");
		nf_info->status |= NF_PAUSED;   //change == to |=
	}
	RTE_LOG(INFO, APP, "NF Pausing!...\n");
	break;
case MSG_RESUME: //MSG_RUN
#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
if (likely(nf_info->bNDSycn)) { //if(unlikely(bNDSync)) {
	nf_info->bNDSycn = 0;//bNDSync=0;
	//printf("\n Received NDSYNC_CLEAR AND RESUME MESSAGE\n!!");
#ifdef  __DEBUG_NDSYNC_LOGS__
	delta_nd=onvm_util_get_elapsed_time(&nd_ts);
	if(min_nd==0 || delta_nd < min_nd) min_nd= delta_nd;
	if(delta_nd > max_nd)max_nd=delta_nd;
	if(avg_nd) avg_nd = (avg_nd+delta_nd)/2;
	else avg_nd= delta_nd;
	//printf("\n\n IN RESUME NOTIFICATION:: WAIT_TIME_STATS(ND_SYNC):\n Cur=%li\n Min= %li\n Max: %li \n Avg: %li \n", delta_nd, min_nd, max_nd, avg_nd);
	nf_info->min_nd=min_nd; nf_info->max_nd=max_nd; nf_info->avg_nd=avg_nd;
#endif
}
#endif
	nf_info->status = NF_RUNNING; //set = RUNNING; works for both NF_PAUSED and NF_WT_ND_SYNC_BIT cases!
#ifdef INTERRUPT_SEM
			onvm_nflib_implicit_wakeup(); //TODO: change this ecb call; split ecb call to two funcs. sounds stupid but necessary as cache update of flag_p takes time; otherwise results in sleep-wkup cycles
#endif

	RTE_LOG(DEBUG, APP, "Resuming NF...\n");
	break;
case MSG_NOOP:
default:
	break;
	}
	return 0;
}

static inline void onvm_nflib_dequeue_messages(struct onvm_nf_info *nf_info) {
	// Check and see if this NF has any messages from the manager
	if (likely(rte_ring_count(nf_msg_ring) == 0)) {
		return;
	}
	struct onvm_nf_msg *msg = NULL;
	rte_ring_dequeue(nf_msg_ring, (void**) (&msg));
	onvm_nflib_handle_msg(msg, nf_info);
	rte_mempool_put(nf_msg_pool, (void*) msg);
}
/******************************Helper functions*******************************/
static struct onvm_nf_info *
onvm_nflib_info_init(const char *tag) {
	void *mempool_data;
	struct onvm_nf_info *info;

	if (rte_mempool_get(nf_info_mp, &mempool_data) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get client info memory");
	}

	if (mempool_data == NULL) {
		rte_exit(EXIT_FAILURE, "Client Info struct not allocated");
	}

	info = (struct onvm_nf_info*) mempool_data;
	info->instance_id = initial_instance_id;
	info->service_id = service_id;
	info->status = NF_WAITING_FOR_ID;
	info->tag = tag;

	info->pid = getpid();

#ifdef ONVM_GPU

	int original_instance_id = initial_instance_id;
	if(is_secondary_active_nf_id(info->instance_id)) {
		original_instance_id = get_associated_active_or_standby_nf_id(info->instance_id);
		printf("DEBUG... this is a secondary NF ---+++_---+++___ We get associated NF id %d \n",original_instance_id);
	}
	//info->image_info = &(all_images_information[original_instance_id]);
	printf("original instance id %d \n", original_instance_id);

	//initialize the histograms
	hist_init_v2(&(info->cpu_latency));
	hist_init_v2(&(info->gpu_latency));
	hist_init_v2(&(info->throughput_histogram));
	hist_init_v2(&(info->image_aggregation_latency));
	get_batch_agg_and_dev_buffer_mempool(info);//find the mempool for agg info and dev buffer
	initialize_ml_timers(info);

	//also initialize the gpu models and the model number before sending it to the manager
	info->gpu_model = ml_model_number;
	info->ring_flag = 0;//by default there will be a ring guard.

	//put the batch aggregation buffer in the NF
	struct rte_mempool * image_batch_aggregation_info;
	struct rte_mempool * image_batch_dev_buffer;
	uint16_t nf_id = info->instance_id;
	if(nf_id< MAX_ACTIVE_CLIENTS) {
		image_batch_aggregation_info = rte_mempool_lookup(get_image_batch_agg_name(nf_id));
		image_batch_dev_buffer = rte_mempool_lookup(get_image_dev_buffer_name(nf_id));
		printf("%s\n address %p",get_image_batch_agg_name(nf_id), image_batch_aggregation_info);
	}
	else
	{
		//just give the same
		uint16_t alternate_nf = get_associated_active_or_standby_nf_id(nf_id);
		image_batch_aggregation_info = rte_mempool_lookup(get_image_batch_agg_name(alternate_nf));
		image_batch_dev_buffer = rte_mempool_lookup(get_image_dev_buffer_name(alternate_nf));

	}
	int retval;
	retval = rte_mempool_get(image_batch_aggregation_info,(void **)&(info->image_info));

	/*
	 int i;
	 for(i = 0; i<MAX_IMAGES_BATCH_SIZE;i++){
	 printf("bytes count %ld i %d \n",info->image_info->images[i].bytes_count ,i);
	 info->image_info->images[i].packets_count = 0;
	 info->image_info->images[i].usage_status = 0;
	 }
	 */
	//memset(info->image_info,0,sizeof(image_batched_aggregation_info_t));
	/*
	 cudaError_t cudaerr;
	 cudaerr = cudaHostRegister(info->image_info,sizeof(image_batched_aggregation_info_t), cudaHostRegisterMapped);
	 if(cudaerr != cudaSuccess)
	 printf("Could not register image batch agg info %d\n", cudaerr);
	 */
	printf("mempool get retval %d \n", retval);
	//if(rte_mempool_get(image_batch_aggregation_info,(void **)(&info->image_info)) !=0){
	//		rte_exit(EXIT_FAILURE, "Failed to get batch aggregation info");
	//	}
	//now put it back
	rte_mempool_put(image_batch_aggregation_info, info->image_info);

	//now put the cpu buffer in the NF
	if(rte_mempool_get(image_batch_dev_buffer, (void **)&info->cpu_side_buffer)!= 0) {
		rte_exit(EXIT_FAILURE, "Failed to get CPU side buffer");
	}

	//now put it back
	rte_mempool_put(image_batch_dev_buffer, info->cpu_side_buffer);

	info->cpu_result_buffer = rte_malloc(NULL, SIZE_OF_AN_IMAGE_BYTES*MAX_IMAGES_BATCH_SIZE,0);
	//for the NF requesting CPU side buffer to copy packets into
	resolve_cpu_side_buffer(info->cpu_side_buffer, info->cpu_result_buffer);

#endif
	return info;
}

static void onvm_nflib_usage(const char *progname) {
	printf("Usage: %s [EAL args] -- "
#ifdef ENABLE_STATIC_ID
			"[-n <instance_id>]"
#endif
					"[-r <service_id>]\n\n", progname);
}

static int onvm_nflib_parse_args(int argc, char *argv[]) {
	const char *progname = argv[0];
	int c;

	opterr = 0;
#ifdef ENABLE_STATIC_ID
#ifdef ONVM_GPU
	while((c = getopt (argc, argv, "n:r:f:m:")) != -1)
#else
	while ((c = getopt (argc, argv, "n:r:")) != -1)
#endif //onvm_gpu
#else
	while ((c = getopt(argc, argv, "r:")) != -1)
#endif
		switch (c) {
#ifdef ENABLE_STATIC_ID
#ifdef ONVM_GPU
		case 'n':
		initial_instance_id = (uint16_t) strtoul(optarg, NULL, 10);
		break;
		case 'f':
		ml_model_file = optarg;
		printf("model file name is %s \n", ml_model_file);
		break;
		case 'm':
		ml_model_number = (uint16_t) strtoul(optarg, NULL, 10);
		break;
#else
		case 'n':
		initial_instance_id = (uint16_t) strtoul(optarg, NULL, 10);
		break;
#endif //onvm_gpu
#endif
		case 'r':
			service_id = (uint16_t) strtoul(optarg, NULL, 10);
			// Service id 0 is reserved
			if (service_id == 0)
				service_id = -1;
			break;
		case '?':
			onvm_nflib_usage(progname);
			if (optopt == 'n')
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint (optopt))
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
			return -1;
		default:
			return -1;
		}

	if (service_id == (uint16_t) - 1) {
		/* Service ID is required */
		fprintf(stderr, "You must provide a nonzero service ID with -r\n");
		return -1;
	}
	return optind;
}

static void onvm_nflib_handle_signal(int sig) {
	if (sig == SIGINT) {
		keep_running = 0;
#ifdef INTERRUPT_SEM
		onvm_nflib_implicit_wakeup();
#endif
	}
	/* TODO: Main thread for INTERRUPT_SEM case: Must additionally relinquish SEM, SHM */
}

static inline void onvm_nflib_cleanup(
		__attribute__((unused))                                                                                                          struct onvm_nf_info *nf_info) {
	struct onvm_nf_msg *shutdown_msg;
	nf_info->status = NF_STOPPED;

#ifndef ENABLE_MSG_CONSTRUCT_NF_INFO_NOTIFICATION
	/* Put this NF's info struct back into queue for manager to ack shutdown */
	if (mgr_msg_ring == NULL) {
		rte_mempool_put(nf_info_mp, nf_info); // give back memory
		rte_exit(EXIT_FAILURE, "Cannot get nf_info ring for shutdown");
	}

	if (rte_ring_enqueue(mgr_msg_ring, nf_info) < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for shutdown");
	}
	return;
#else
	/* Put this NF's info struct back into queue for manager to ack shutdown */
	if (mgr_msg_ring == NULL) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot get nf_info ring for shutdown");
	}
	if (rte_mempool_get(nf_msg_pool, (void**)(&shutdown_msg)) != 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot create shutdown msg");
	}

	shutdown_msg->msg_type = MSG_NF_STOPPING;
	shutdown_msg->msg_data = nf_info;

	if (rte_ring_enqueue(mgr_msg_ring, shutdown_msg) < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_mempool_put(nf_msg_pool, shutdown_msg);
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for shutdown");
	}
	return;
#endif
}

static inline int onvm_nflib_notify_ready(struct onvm_nf_info *nf_info) {
	int ret = 0;

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	onvm_util_get_start_time(&ts);
#endif
	nf_info->status = NF_WAITING_FOR_RUN;

#ifdef ENABLE_MSG_CONSTRUCT_NF_INFO_NOTIFICATION
	struct onvm_nf_msg *startup_msg;
	/* Put this NF's info struct onto queue for manager to process startup */
	ret = rte_mempool_get(nf_msg_pool, (void**)(&startup_msg));
	if (ret != 0) return ret;

	startup_msg->msg_type = MSG_NF_READY;
	startup_msg->msg_data = nf_info;
	ret = rte_ring_enqueue(mgr_msg_ring, startup_msg);
	if (ret < 0) {
		rte_mempool_put(nf_msg_pool, startup_msg);
		return ret;
	}
#else
	/* Put this NF's info struct onto queue for manager to process startup */
	ret = rte_ring_enqueue(mgr_msg_ring, nf_info);
	if (ret < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back memory
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager");
	}
#endif
	/* Wait for a client id to be assigned by the manager */
	RTE_LOG(INFO, APP, "Waiting for manager to put to RUN state...\n");
	struct timespec req = { 0, 1000 }, res = { 0, 0 };
	for (; nf_info->status == (uint16_t) NF_WAITING_FOR_RUN;) {
		nanosleep(&req, &res); //sleep(1); //better poll for some time and exit if failed within that time.?
	}

#ifdef ONVM_GPU

	printf("GPU model is %d\n",nf_info->gpu_model);
	if(!nf_info->gpu_model) {
		//in this case there will be no GPU loaded
		//we have to provide different data path
		current_packet_processing_batch = onvm_nflib_process_packets_batch;

	}
	else
	{
		printf("NF using GPU\n");
		//our function to process the batch of packets
		current_packet_processing_batch = onvm_nflib_process_packets_batch_gpu;

		int retval;
		void * status = NULL;
		/* create the argument list for loading the ml model */
		ml_load_params.file_path = nf_info->model_info->model_file_path;
		ml_load_params.load_options = 0; //For CPU side loading = 0, for gpu = 1

		/* in both NF running and pause case, we might need to load the ML model from disk to CPU */
		//ml_functions.load_model(nf_info->model_info.model_file_path, 0 /*load in CPU */, &(nf_info->ml_model_handle), &(nf_info->ml_model_handle), nf_info->model_info.model_handles.number_of_parameters);
		retval = (*(ml_operations->load_model_fptr))(&ml_load_params,status);
		nf_info->ml_model_handle = ml_load_params.model_handle;
		if(retval != 0)
		printf("Error while loading the model \n");

		/* this NF should have been registered to manager and have all info
		 * We can also then give its batch aggregation pointers and dev buffer pointer */
		nf_info->gpu_percentage = 70;
		printf("GPU Percentage now %d \n", nf_info->gpu_percentage);

		if(nf_info->gpu_percentage > 0) {

			//call a function to initialize GPU
			initialize_gpu(nf_info);
		}
	}
#endif //onvm_gpu
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	int64_t ttl_elapsed = onvm_util_get_elapsed_time(&ts);
	printf("WAIT_TIME(START-->RUN): %li ns\n", ttl_elapsed);
#endif
	return 0;
}

#ifdef ONVM_GPU

/* the function to initialize things on GPU */
void initialize_gpu(struct onvm_nf_info *nf_info) {

	nf_info->learned_max_batch_size=0;
	nf_info->adaptive_cur_batch_size=0;

	nf_info->aiad_aimd_decrease_factor=1;
	nf_info->aiad_aimd_increase_factor=1;

	nf_info->b_i_exceeding_slo_per_sec=0;
	nf_info->batches_inferred_per_sec=0;

	nf_info->under_provisioned_for_slo=0;
	nf_info->over_provisioned_for_slo=0;

	//0. fix the link parameters and infer parameters
	void * status = NULL;
	//nflib_ml_fw_link_params_t ml_link_params;
	//these things don't change once set so we can just set it here
	ml_link_params.file_path = nf_info->model_info->model_file_path;
	ml_link_params.model_handle = nf_info->ml_model_handle;
	ml_link_params.cuda_handles_for_gpu_data = nf_info->model_info->model_handles.cuda_handles;
	ml_link_params.number_of_parameters = nf_info->model_info->model_handles.number_of_parameters;
	ml_link_params.gpu_side_input_pointer = NULL;
	ml_link_params.gpu_side_output_pointer = NULL;
	printf("Linking the cuda memhandles from %p \n", ml_link_params.cuda_handles_for_gpu_data);
	printf("pointer to gpu agg buffer %p\n",nf_info->image_info);
	int retval;

	// 1. set the GPU percentage
	char gpu_percent[4];
	sprintf(gpu_percent,"%d", nf_info->gpu_percentage);
	setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_percent, 1);//overwrite cuda_mps_active_thread_precentage

	// 2. Create all streams
	retval = init_streams();
	if(retval)
	printf("Error while creating CUDA streams \n");

	// 3. Pin required memories

	retval = cuda_register_shared_memory();
	if(retval)
	printf("Could not pin the memory \n");

	// 4. Put the GPU model to GPU
	//retval = (*(ml_functions.get_gpu_ready))(nf_info->ml_model_handle, nf_info->model_info.model_handles->cuda_handles, nf_info->gpu_model_info.model_handles->number_of_parameters, nf_info->user_batch_size);
	retval = (*(ml_operations->link_model_fptr))(&ml_link_params,status);

	if(ml_link_params.gpu_side_input_pointer != NULL) {
		resolve_gpu_dev_buffer_pointer(ml_link_params.gpu_side_input_pointer, ml_link_params.gpu_side_output_pointer);
	}
	else {
		//convert all the gpu side buffers...
		resolve_gpu_dev_buffer(nf_info->gpu_input_buffer, nf_info->gpu_output_buffer);
	}

	if(retval != 0)
	printf("ERROR! while setting up the model in GPU \n");
}

#endif //onvm_gpu

#if 0
if(NF_PAUSED == nf_info->status) {
	onvm_nflib_wait_till_notification(nf_info);
}
if(NF_RUNNING != nf_info->status) {
	switch(nf_info->status) {
		case NF_PAUSED:
		onvm_nflib_wait_till_notification(nf_info);
		break;
		case NF_STOPPED:
		onvm_nflib_cleanup(nf_info);
		rte_exit(EXIT_FAILURE, "NF RUNfailed! moving to shutdown!");
		break;
		default:
		break;
	}
}
#endif

/* aditya's GPU related helper functions... */
#ifdef ONVM_GPU

/* we require these functions so we can keep track of how many images are present now */
//static image_data* get_image(int pkt_file_index, image_data **pending_img);
//static void delete_image(image_data * image, struct onvm_nf_info * nf_info);
//this function will be called to evaluate an image from mempool
//void evaluate_an_image_from_mempool(__attribute__((unused))struct rte_timer *timer_ptr, void *info, image_data * image);
//ml stats function
//void compute_ml_stats(struct rte_timer * timer_ptr,void *info);
/* the function to register ml operations... NF will call this function */
int nflib_register_ml_fw_operations(ml_framework_operations_t *ops) {
	ml_operations = ops;
	return 0;
}

/* function that pins all the shared memory... Necessary for NetML approach to work */
int cuda_register_shared_memory(void) {

	//get the memory config
	struct rte_config * rte_config = rte_eal_get_configuration();

	//now get the memory locations
	struct rte_mem_config * memory_config = rte_config->mem_config;

	int i;
	struct timespec begin,end;
	clock_gettime(CLOCK_MONOTONIC, &begin);
	cudaError_t cuda_err;
	for(i = 0; i<RTE_MAX_MEMSEG_LISTS; i++) {
		struct rte_memseg_list *memseg_ptr = &(memory_config->memsegs[i]);
		if(memseg_ptr->page_sz > 0 && memseg_ptr->socket_id == (int)rte_socket_id()) {
			//printf("Pointer to huge page %p and size of the page %"PRIu64"\n",memseg_ptr->base_va, memseg_ptr->page_sz);

			cuda_err = cudaHostRegister(memseg_ptr->base_va, memseg_ptr->page_sz, cudaHostRegisterDefault);
			if(cuda_err != cudaSuccess)
			printf("Could not register memory mem-addr %p size %ld cuda error %d \n", memseg_ptr->base_va, memseg_ptr->page_sz, cuda_err );
			else
			printf("registered cuda memory mem-addr %p size %ld cuda error %d \n", memseg_ptr->base_va, memseg_ptr->page_sz, cuda_err);
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double time_taken_to_register = (end.tv_sec-begin.tv_sec)*1000000.0 + (end.tv_nsec-begin.tv_nsec)/1000.0;
	printf("Total time taken to register the mempages to cuda is %f micro-seconds \n",time_taken_to_register);
	return 0;
}
struct rte_mempool *nf_batch_agg_pool;
struct rte_mempool *nf_batch_dev_buffer_pool;

//this function looksup image when required.
static void get_batch_agg_and_dev_buffer_mempool(struct onvm_nf_info * info)
{
	const char *batch_mempool_name;
	const char *dev_buffer_name;
	int instance_id = info->instance_id;
	//we have to attach the mempools here... and then put it back...
	printf("The instance ID of the nf is %d and associated id %d\n", instance_id,get_associated_active_or_standby_nf_id(instance_id));
	if(is_secondary_active_nf_id(info->instance_id)) {
		batch_mempool_name = get_image_batch_agg_name(get_associated_active_or_standby_nf_id(instance_id));
		dev_buffer_name = get_image_dev_buffer_name(get_associated_active_or_standby_nf_id(instance_id));
	}
	else
	{
		batch_mempool_name = get_image_batch_agg_name(instance_id);
		dev_buffer_name = get_image_dev_buffer_name(instance_id);
	}

	//printf("State mempool name is %s \n",state_mempool_name);
	nf_batch_agg_pool = rte_mempool_lookup(batch_mempool_name);
	nf_batch_dev_buffer_pool = rte_mempool_lookup(dev_buffer_name);

	if(nf_batch_agg_pool == NULL)
	printf("Cannot find batch agg pool mempool \n");
	if(nf_batch_dev_buffer_pool == NULL)
	printf("Cannot find dev buffer mempool \n");

	/*
	 //below code commented as we will be updating the address of mempool somewhere else
	 int retval;
	 void * address;
	 retval = rte_mempool_get(state_mempool, (void **) &address);
	 if(retval != 0){
	 printf("--- Couldn't get image mempool %d \n", retval);
	 }
	 info->image_info = address;
	 rte_mempool_put(state_mempool, info->image_info);
	 printf("Info image info mempool updated to %p \n",info->image_info);
	 */
}

/* this function sends message to onvm manager */
void onvm_send_gpu_msg_to_mgr(void *message_to_manager, int message_type) {
	printf("ONVM SEND GPU MSG TO MGR\n");
	struct onvm_nf_msg * msg_mgr;
	int ret = 0;
	ret = rte_mempool_get(nf_msg_pool,(void **) &msg_mgr);
	msg_mgr->msg_type = message_type;
	msg_mgr->msg_data = message_to_manager;
	struct rte_ring * mgr_ring;
	mgr_ring = rte_ring_lookup(_MGR_MSG_QUEUE_NAME);
	if(mgr_ring == NULL)
	printf("message ring couldn't be found \n");
	ret = rte_ring_enqueue(mgr_ring, msg_mgr);
	if (ret < 0) {
		rte_mempool_put(nf_msg_pool, msg_mgr);
		return;
	}
}

/* common function to load ml file 
 void load_ml_file(char * file_path, int cpu_gpu_flag, void ** cpu_func_ptr, void ** gpu_func_ptr, __attribute__((unused)) struct onvm_nf_info *nf_info){
 // convert the filename to wchar_t 
 size_t filename_length = strlen(file_path);
 wchar_t file_name[filename_length];
 size_t wfilename_length = mbstowcs(file_name,file_path,filename_length+1);
 
 if(wfilename_length == 0)
 fprintf(stdout, "Something went wrong with converting filename to wide letter \n");
 
 char device[5];
 if(cpu_gpu_flag == 0)
 strcpy(device, "CPU");
 else if(cpu_gpu_flag == 1)
 strcpy(device, "GPU");

 fprintf(stdout, "Loading ML file on %s device \n", device);

 int ret = load_model(file_name, cpu_func_ptr, gpu_func_ptr, cpu_gpu_flag, 0);

 // put it in NF info as well 
 //nf_info->function_ptr = gpu_func_ptr;

 if(ret)
 fprintf(stdout, "ML File loading failure \n");
 }
 */
//this function will evaluate your data...
/*
 void evaluate_the_image(void *function_ptr, void * input_buffer, float *stats, float *output){
 //printf("%p, %p, %p, %p \n",function_ptr, input_buffer, stats, output);
 evaluate_in_gpu_input_from_host((float *)input_buffer, IMAGE_SIZE*IMAGE_BATCH, output, function_ptr, stats, 0, NULL, NULL, IMAGE_BATCH);
 return;
 }
 */
//a more comprehensive evaluate function that can be called by the timer thread
/*
 void evaluate_an_image_from_mempool(__attribute__((unused))struct rte_timer *timer_ptr,void *info, image_data *ready_image){
 struct onvm_nf_info* nf_info = (struct onvm_nf_info *)info;
 //printf("DEBUG.. timer thread acting properly\n");
 if(nf_info->image_info == NULL){
 printf("image info address %p \n", nf_info->image_info);
 }
 //get the next image. only if there is any images to execute
 //printf("the image info address %p and nf instance ID %d the num of ready images--- nf_info address %p gpu model %d \n",nf_info->image_info,nf_info->instance_id, nf_info, nf_info->gpu_model);
 if(nf_info->image_info->num_of_ready_images <= 0){
 //  printf("Testing timer thread.. timer period %"PRIu64" \n", timer_ptr->period);
 return;
 }

 //printf("Number of ready images %d \n", nf_info->image_info->num_of_ready_images);

 
 //if the NF is destined to restart.. do not add to the GPU queue
 //otherwise grab one image and proceed
 //image_data *ready_image = nf_info->image_info->ready_images[nf_info->image_info->index_of_ready_image];
 
 //increase the index of ready image... as well as decrease the number of ready images
 nf_info->image_info->num_of_ready_images--;
 if(nf_info->image_info->num_of_ready_images < 0)
 nf_info->image_info->num_of_ready_images=0;
 
 nf_info->image_info->index_of_ready_image++;
 nf_info->image_info->index_of_ready_image %= MAX_IMAGE; 
 
 struct gpu_callback * callback_data = &(gpu_callbacks[gpu_queue_current_index]);
 callback_data->nf_info = nf_info;
 callback_data->ready_image = ready_image;
 
 //get the time when evaluate finishes
 struct timespec eval_end_time;

 //increase the index for queue and number of elements in the queue
 gpu_queue_current_index++;
 gpu_queue_current_index %= MAX_IMAGE; 
 num_elements_in_gpu_queue++;  //increase how many images are in GPU queue

 //time index... the cntk functions are synchronous in respect to CPU and tensorrt are not.. so there will be two different
 // times that we need to compute latency and throughput on...
 //we will declare a variable for the index. but we need to fix this with refactoring
 //seriously this is getting out of hand

 int time_index;

 
 //new evaluation function, older one is too cubersome
 if(nf_info->platform==cntk){
 evaluate_image_in_gpu(ready_image, nf_info->function_ptr, gpu_image_callback_function, (void *) callback_data, gpu_finish_work_flag);
 time_index = 1;
 }
 if(nf_info->platform == tensorrt){
 evaluate_image_in_gpu_tensorrt(ready_image, gpu_image_callback_function, (void *) callback_data, gpu_finish_work_flag);
 time_index = 3;
 }
 //post evaluation...put it in the GPU eval queue. if the above call is synchronous, then we can take a timestamp of finishing the image.
 gpu_queue_image_id[gpu_queue_current_index]=ready_image->image_id;
 clock_gettime(CLOCK_MONOTONIC, &eval_end_time); //read the current time
 ready_image->timestamps[1] = eval_end_time; //the index for timestamp is explained in onvm_image.h

 //store the throughputs
 if(num_throughput_stored < 1000){
 throughputs[num_throughput_stored] = nf_info->user_batch_size*(1000000.0/((float)time_difference_usec(&(ready_image->timestamps[5]),&(ready_image->timestamps[time_index]))));
 batch_fed[num_throughput_stored] = ready_image->timestamps[5];
 batch_processed[num_throughput_stored] = ready_image->timestamps[time_index];

 //ignoring first two data points
 if(num_throughput_stored > 10)  {
 //double current_total_latency = time_difference_usec(&batch_fed[num_throughput_stored-5], &batch_processed[num_throughput_stored]);
 //this latency measure is supposed to show the latency of last 10 batches
 double current_total_latency = (ready_image->timestamps[time_index].tv_sec-batch_fed[num_throughput_stored-10].tv_sec)*1000.0+(ready_image->timestamps[time_index].tv_nsec-batch_fed[num_throughput_stored-10].tv_nsec)/1000000.0;

 double current_throughput = (nf_info->user_batch_size*10*(1000.0/current_total_latency));
 printf("Image batch %d, single images processed till now %d, throughput of last 10 batches %f, latency including data movement of current batch %f (ms)\n", num_throughput_stored, num_throughput_stored*nf_info->user_batch_size, current_throughput, time_difference_usec(&(ready_image->timestamps[5]), &(ready_image->timestamps[time_index]))/1000.0);
 }
 if((num_throughput_stored%100) == 0){
 //final output
 printf("\n \n====== Summary output ======== \n");
 double total_latency = time_difference_usec(&batch_fed[num_throughput_stored-79], &batch_processed[num_throughput_stored]);
 double total_throughput = nf_info->user_batch_size*80*(1000000.0/total_latency);
 printf("based on data of 80 image batch or %d images, the throughput is %f images per second \n", 80*nf_info->user_batch_size, total_throughput);
 printf("\n \n ===== summary ended ======== \n");
 }
 }
 num_throughput_stored++;
 
 }
 */
#define MIN_GAP_THRESHOLD (50)	//Latency gap between SLO and observed latency below which the scaling cannot be performed.
#define MAX_OVERFLOW_THRESHOLD (5)
#define NEW_LEARNING_BATCH_APPROACH
#ifdef NEW_LEARNING_BATCH_APPROACH
#define SLO_OFFSET_THRESHOLD_PERCENTAGE (10)
//GPU callback function to report after the evaluation is finished...
inline void gpu_compute_batch_size_for_slo(void *data, uint32_t num_of_images_inferred, uint32_t cur_lat);
inline void gpu_compute_batch_size_for_slo(void *data, uint32_t num_of_images_inferred, uint32_t cur_lat) {
	struct gpu_callback *callback_data = (struct gpu_callback *) data;
	uint32_t slo_time_us = (callback_data->nf_info->inference_slo_ms*1000);
	//Exactly match ( rare)
	if(unlikely(slo_time_us == cur_lat)) {
		callback_data->nf_info->adaptive_cur_batch_size = num_of_images_inferred;
		callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
		return;
	}
	//exceeds the SLO
	else if (unlikely(cur_lat > slo_time_us)) {
		callback_data->nf_info->b_i_exceeding_slo_per_sec++;
		uint32_t overflow_us = (cur_lat - slo_time_us);
		if(unlikely(num_of_images_inferred == ((uint32_t) 1))) {
			//Must request to increase the Resource % for this NF
			callback_data->nf_info->under_provisioned_for_slo+=1;
			callback_data->nf_info->adaptive_cur_batch_size = 1;
			callback_data->nf_info->learned_max_batch_size = 1;
			printf("Need More GPU pct2: %d to meet the SLO!!! LearnedSize: %d CPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, slo_time_us);
			return;
		}
		else {
			//Compute the decrease factor to adapt batchsize to operate within SLO
			uint32_t decrease_factor = ((overflow_us)/(cur_lat/num_of_images_inferred));
			if(likely(decrease_factor == 0)) {
				decrease_factor = 1;
			} else {
				//decrease_factor = (((uint32_t)callback_data->nf_info->adaptive_cur_batch_size)>decrease_factor)?(decrease_factor):(callback_data->nf_info->adaptive_cur_batch_size-1);
				if(decrease_factor > ((uint32_t)callback_data->nf_info->adaptive_cur_batch_size)) decrease_factor = (callback_data->nf_info->adaptive_cur_batch_size-1);
			}
			//check if overflow is within permissible range
			if(likely(overflow_us <= (SLO_OFFSET_THRESHOLD_PERCENTAGE*slo_time_us/100))) {
				callback_data->nf_info->adaptive_cur_batch_size = num_of_images_inferred;
				callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
				return;
			}
			else {
				callback_data->nf_info->adaptive_cur_batch_size -= decrease_factor;
				callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
			}
		}
	}
	//Within SLO
	else {
		uint32_t headroom_us = (slo_time_us - cur_lat);
		//have we used smaller batch than already learned then ignore current timing for adjustment
		if(likely(num_of_images_inferred < ((uint32_t) callback_data->nf_info->adaptive_cur_batch_size))) {
			// no need to account for this result, as this is opportunistic obtained value; We already know better/higher value
			return;
		} else if (unlikely(num_of_images_inferred >= MAX_IMAGES_BATCH_SIZE)) {
			callback_data->nf_info->over_provisioned_for_slo+=headroom_us;
			printf("Have more than necessary GPU pct: %d to meet the SLO!!! LearnedSize: %d GPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, slo_time_us);
			return;
		}
		//check if enough headroom to increase batch size;
		if(likely(headroom_us >= (SLO_OFFSET_THRESHOLD_PERCENTAGE*slo_time_us/100))) {
			uint32_t increase_factor = (headroom_us)/(cur_lat/num_of_images_inferred);
			if(0 == increase_factor) {
				increase_factor+=1;
			}
			callback_data->nf_info->adaptive_cur_batch_size += increase_factor;
			callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
		}
	}

#if 0
	/* Within the SLO range: Can opportunistically increase the batch size : But how much to increase by? Note: Opportunistic way could have aggregated lesser as well */
	if((callback_data->nf_info->inference_slo_ms*1000) >= cur_lat) { //Compute the factor to increase based on linear interpolation?
		if(num_of_images_inferred < (uint32_t) callback_data->nf_info->adaptive_cur_batch_size) {
			// no need to account for this result, as this is opportunistic obtained value; We already know better/higher value
		}
		else { //if (num_of_images_inferred > callback_data->nf_info->adaptive_cur_batch_size) {
			callback_data->nf_info->adaptive_cur_batch_size = num_of_images_inferred;
			uint8_t increase_factor = ((callback_data->nf_info->inference_slo_ms*1000)-cur_lat)/(cur_lat/num_of_images_inferred);
			if(increase_factor == 0) {
				if(((callback_data->nf_info->inference_slo_ms*1000) -cur_lat) > ((MIN_GAP_THRESHOLD*cur_lat)/(num_of_images_inferred*100))) {
					callback_data->nf_info->adaptive_cur_batch_size += callback_data->nf_info->aiad_aimd_increase_factor;
					callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
				} else
				callback_data->nf_info->learned_max_batch_size = num_of_images_inferred;
			}
			//(increase_factor > callback_data->nf_info->aiad_aimd_increase_factor) &&
			else {
				if(((increase_factor + callback_data->nf_info->adaptive_cur_batch_size) < MAX_IMAGES_BATCH_SIZE)) {
					callback_data->nf_info->adaptive_cur_batch_size += increase_factor;
					callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
				}
				else {
					if((callback_data->nf_info->aiad_aimd_increase_factor + callback_data->nf_info->adaptive_cur_batch_size) < MAX_IMAGES_BATCH_SIZE) {
						callback_data->nf_info->adaptive_cur_batch_size += callback_data->nf_info->aiad_aimd_increase_factor;
						callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
					} else {
						//This is when we can ask for resource to be toned down for this NF
						callback_data->nf_info->over_provisioned_for_slo+=1;
						printf("Have more than necessary GPU pct: %d to meet the SLO!!! LearnedSize: %d GPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, (callback_data->nf_info->inference_slo_ms*1000));
					}
				}
			}
		}
	}

	/* Exceeding the SLO objective; Tone down the batch size */
	else {
		callback_data->nf_info->b_i_exceeding_slo_per_sec++;
		// Check if we had estimate earlier and current is lower than estimated value
		//if(num_of_images_inferred < callback_data->nf_info->adaptive_cur_batch_size) { //is this check necessary: we have exceeded, must reduce no matter what.
		//callback_data->nf_info->adaptive_cur_batch_size = num_of_images_inferred;
		//}
		/*
		 if(1 == num_of_images_inferred) {
		 callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
		 }
		 else*/{
			uint8_t decrease_factor = (cur_lat - (callback_data->nf_info->inference_slo_ms*1000))/(cur_lat/num_of_images_inferred);
			if(decrease_factor == 0) {
				if((callback_data->nf_info->adaptive_cur_batch_size >=2) && (cur_lat > ((callback_data->nf_info->inference_slo_ms*1000)*(100+MAX_OVERFLOW_THRESHOLD)/100))) {
					callback_data->nf_info->adaptive_cur_batch_size-=1;
				} else {
					callback_data->nf_info->under_provisioned_for_slo+=1;
					printf("Need More GPU pct1: %d to meet the SLO!!! LearnedSize: %d CPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, (callback_data->nf_info->inference_slo_ms*1000));
				}
				callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
			}
			/* Do we have sufficient head_room to decrease the batch_size? */ //Note: can be problematic say we have 8ms per img; 8 num_of_imgs_inferred, 64=current time and 50 SLO
			else { //&& (callback_data->nf_info->adaptive_cur_batch_size > callback_data->nf_info->aiad_aimd_decrease_factor)
				if( (decrease_factor < callback_data->nf_info->adaptive_cur_batch_size) ) {
					callback_data->nf_info->adaptive_cur_batch_size -= decrease_factor;
					callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
				}
				/* No head room to decrease the batch_size? */
				else {
					if(callback_data->nf_info->adaptive_cur_batch_size >= 2) { //callback_data->nf_info->aiad_aimd_decrease_factor
						callback_data->nf_info->adaptive_cur_batch_size--;
						callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
					}
					else {
						//Must request to increase the Resource % for this NF
						callback_data->nf_info->under_provisioned_for_slo+=1;
						printf("Need More GPU pct2: %d to meet the SLO!!! LearnedSize: %d CPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, (callback_data->nf_info->inference_slo_ms*1000));
					}
				}
			}
		}
	}
#endif
	return;
}
#endif //NEW_LEARNING_BATCH_APPROACH
void gpu_image_callback_function(void *data) {

	//just update the stats here for now...
	struct timespec call_back_time, image_start_aggr_timestamp, image_ready_timestamp;
	clock_gettime(CLOCK_MONOTONIC, &call_back_time);

	struct gpu_callback *callback_data = (struct gpu_callback *) data;

	callback_data->status = 0;

	/* we also have to clear the images status as inferred */
	int num_of_images_inferred = __builtin_popcount(callback_data->bitmask_images);

	int i;
	int bit_position;
	uint32_t temp_latency=0, nw_latency=0, cpu_latency=0;

	uint32_t gpu_latency= (call_back_time.tv_sec-callback_data->start_time.tv_sec)*1000000+(call_back_time.tv_nsec-callback_data->start_time.tv_nsec)/1000;
	hist_store_v2(&(callback_data->nf_info->gpu_latency),gpu_latency);

	for( i = 0; i<num_of_images_inferred; i++) {
		bit_position = ffs(callback_data->bitmask_images);
		CLEAR_BIT(callback_data->batch_aggregation->ready_mask,(bit_position));
		CLEAR_BIT(callback_data->bitmask_images,bit_position);
		bit_position--; //as it reports bit position 0 as 1.
		//printf("status[%d], The number of packets in the callback %d the bitmask is %x\n",callback_data->batch_aggregation->images[bit_position].usage_status, callback_data->batch_aggregation->images[bit_position].packets_count,callback_data->bitmask_images);

		//compute the latencies for this batch across cpu, gpu and n/w.
		image_start_aggr_timestamp = callback_data->batch_aggregation->images[bit_position].first_packet_time;
		image_ready_timestamp = callback_data->batch_aggregation->images[bit_position].last_packet_time;

		temp_latency = (call_back_time.tv_sec-image_ready_timestamp.tv_sec)*1000000+(call_back_time.tv_nsec-image_ready_timestamp.tv_nsec)/1000;
		if(temp_latency>cpu_latency) cpu_latency = temp_latency;

		temp_latency = (image_ready_timestamp.tv_sec - image_start_aggr_timestamp.tv_sec)*1000000+(image_ready_timestamp.tv_nsec - image_start_aggr_timestamp.tv_nsec)/1000;
		if(temp_latency>nw_latency) nw_latency = temp_latency;

		//printf("latency(ms):,%f \n batch_size,%d:", latency,num_of_images_inferred);

#ifdef HOLD_PACKETS_TILL_CALLBACK
		//REVERT FOR HOLDING THE PACKETS
		int j=0;
		/*
		 for(j = 0; j<callback_data->batch_aggregation->images[bit_position].packets_count; j++) {
		 //struct onvm_pkt_meta *meta = onvm_get_pkt_meta(callback_data->batch_aggregation->images[bit_position].image_packets[j]);
		 //printf("\n Action:%d, Destination:%d", meta->action, meta->destination); meta->action=ONVM_NF_ACTION_NEXT;
		 //((struct onvm_pkt_meta*)callback_data->batch_aggregation->images[bit_position].image_packets[j]->udata64)->action = ONVM_NF_ACTION_OUT; //ONVM_NF_ACTION_NEXT;
		 //onvm_nflib_drop_pkt(callback_data->batch_aggregation->images[bit_position].image_packets[j]);
		 }*/
		int ret = rte_ring_enqueue_bulk(tx_ring,(void**)(callback_data->batch_aggregation->images[bit_position].image_packets),callback_data->batch_aggregation->images[bit_position].packets_count,NULL);
		if(!ret) {
			for(j = 0; j<callback_data->batch_aggregation->images[bit_position].packets_count; j++) {
				onvm_nflib_drop_pkt(callback_data->batch_aggregation->images[bit_position].image_packets[j]);
			}
		}
#endif
		//onvm_nflib_return_pkt_bulk(callback_data->nf_info, callback_data->batch_aggregation->images[bit_position].image_packets, callback_data->batch_aggregation->images[bit_position].packets_count);
		//now clear the status of the image
		callback_data->batch_aggregation->images[bit_position].bytes_count = 0;
		callback_data->batch_aggregation->images[bit_position].packets_count = 0;
		callback_data->batch_aggregation->images[bit_position].usage_status = 0;
		//printf("Clearing image %d \n",bit_position);
		//now clear the bit so we can get another bit position
		//bit_position &= (0<<bit_position);
		//callback_data->batch_aggregation->ready_mask &= (0<<bit_position);

	}
	//printf("Callback ended, we inferred %d images \n", num_of_images_inferred);
	if(num_of_images_inferred) {
		hist_store_v2(&(callback_data->nf_info->cpu_latency),cpu_latency);

		//uint32_t latency = (call_back_time.tv_sec-callback_data->start_time.tv_sec)*1000000+(call_back_time.tv_nsec - callback_data->start_time.tv_nsec)/1000;

		number_of_images_since_last_computation += num_of_images_inferred;
		//printf("%d numbers of images processed %d \n",number_of_images_since_last_computation, num_of_images_inferred);

		/** Adapt batch size to meet the SLO latency objective **/
		if((callback_data->nf_info->inference_slo_ms) && (ADAPTIVE_BATCHING_SELF_LEARNING == callback_data->nf_info->enable_adaptive_batching)) {
			callback_data->nf_info->batches_inferred_per_sec++;
			if(0 == callback_data->nf_info->adaptive_cur_batch_size) {
				callback_data->nf_info->adaptive_cur_batch_size=num_of_images_inferred;
			}
			uint32_t cur_lat = gpu_latency;		//cpu_latency;
#ifdef NEW_LEARNING_BATCH_APPROACH
			gpu_compute_batch_size_for_slo(data, (uint32_t)num_of_images_inferred, cur_lat);
#else
			/* Within the SLO range: Can opportunistically increase the batch size : But how much to increase by? Note: Opportunistic way could have aggregated lesser as well */
			if((callback_data->nf_info->inference_slo_ms*1000) >= cur_lat) { //Compute the factor to increase based on linear interpolation?
				if(num_of_images_inferred < callback_data->nf_info->adaptive_cur_batch_size) {
					// no need to account for this result, as this is opportunistic obtained value; We already know better/higher value
				}
				else { //if (num_of_images_inferred > callback_data->nf_info->adaptive_cur_batch_size) {
					callback_data->nf_info->adaptive_cur_batch_size = num_of_images_inferred;
					uint8_t increase_factor = ((callback_data->nf_info->inference_slo_ms*1000)-cur_lat)/(cur_lat/num_of_images_inferred);
					if(increase_factor == 0) {
						if(((callback_data->nf_info->inference_slo_ms*1000) -cur_lat) > ((MIN_GAP_THRESHOLD*cur_lat)/(num_of_images_inferred*100))) {
							callback_data->nf_info->adaptive_cur_batch_size += callback_data->nf_info->aiad_aimd_increase_factor;
							callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
						} else
						callback_data->nf_info->learned_max_batch_size = num_of_images_inferred;
					}
					//(increase_factor > callback_data->nf_info->aiad_aimd_increase_factor) &&
					else {
						if(((increase_factor + callback_data->nf_info->adaptive_cur_batch_size) < MAX_IMAGES_BATCH_SIZE)) {
							callback_data->nf_info->adaptive_cur_batch_size += increase_factor;
							callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
						}
						else {
							if((callback_data->nf_info->aiad_aimd_increase_factor + callback_data->nf_info->adaptive_cur_batch_size) < MAX_IMAGES_BATCH_SIZE) {
								callback_data->nf_info->adaptive_cur_batch_size += callback_data->nf_info->aiad_aimd_increase_factor;
								callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
							} else {
								//This is when we can ask for resource to be toned down for this NF
								callback_data->nf_info->over_provisioned_for_slo+=1;
								printf("Have more than necessary GPU pct: %d to meet the SLO!!! LearnedSize: %d GPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, (callback_data->nf_info->inference_slo_ms*1000));
							}
						}
					}
				}
			}

			/* Exceeding the SLO objective; Tone down the batch size */
			else {
				callback_data->nf_info->b_i_exceeding_slo_per_sec++;
				// Check if we had estimate earlier and current is lower than estimated value
				//if(num_of_images_inferred < callback_data->nf_info->adaptive_cur_batch_size) { //is this check necessary: we have exceeded, must reduce no matter what.
				//callback_data->nf_info->adaptive_cur_batch_size = num_of_images_inferred;
				//}
				/*
				 if(1 == num_of_images_inferred) {
				 callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
				 }
				 else*/{
					uint8_t decrease_factor = (cur_lat - (callback_data->nf_info->inference_slo_ms*1000))/(cur_lat/num_of_images_inferred);
					if(decrease_factor == 0) {
						if((callback_data->nf_info->adaptive_cur_batch_size >=2) && (cur_lat > ((callback_data->nf_info->inference_slo_ms*1000)*(100+MAX_OVERFLOW_THRESHOLD)/100))) {
							callback_data->nf_info->adaptive_cur_batch_size-=1;
						} else {
							callback_data->nf_info->under_provisioned_for_slo+=1;
							printf("Need More GPU pct1: %d to meet the SLO!!! LearnedSize: %d CPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, (callback_data->nf_info->inference_slo_ms*1000));
						}
						callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
					}
					/* Do we have sufficient head_room to decrease the batch_size? */ //Note: can be problematic say we have 8ms per img; 8 num_of_imgs_inferred, 64=current time and 50 SLO
					else { //&& (callback_data->nf_info->adaptive_cur_batch_size > callback_data->nf_info->aiad_aimd_decrease_factor)
						if( (decrease_factor < callback_data->nf_info->adaptive_cur_batch_size) ) {
							callback_data->nf_info->adaptive_cur_batch_size -= decrease_factor;
							callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
						}
						/* No head room to decrease the batch_size? */
						else {
							if(callback_data->nf_info->adaptive_cur_batch_size >= 2) { //callback_data->nf_info->aiad_aimd_decrease_factor
								callback_data->nf_info->adaptive_cur_batch_size--;
								callback_data->nf_info->learned_max_batch_size = callback_data->nf_info->adaptive_cur_batch_size;
							}
							else {
								//Must request to increase the Resource % for this NF
								callback_data->nf_info->under_provisioned_for_slo+=1;
								printf("Need More GPU pct2: %d to meet the SLO!!! LearnedSize: %d CPULatency: %d SLO: %d\n", callback_data->nf_info->gpu_percentage, callback_data->nf_info->learned_max_batch_size, cur_lat, (callback_data->nf_info->inference_slo_ms*1000));
							}
						}
					}
				}
			}
#endif //NEW_LEARNING_BATCH_APPROACH
		}
	}
	//long timestamp = call_back_time.tv_sec*1000000+call_back_time.tv_nsec/1000;
	//printf("Timestamp: %ld TotalImages: %d BatchSize: %d LearnedSize: %d CPULatency: %d GPULatency: %d SLO: %d\n",timestamp, number_of_images_since_last_computation, num_of_images_inferred, callback_data->nf_info->learned_max_batch_size, cpu_latency, gpu_latency,(callback_data->nf_info->inference_slo_ms*1000));
	return_device_buffer(callback_data->stream_track->id);
	return_stream(callback_data->stream_track);
}

/* initializes the images... 
 *
 * THIS FUNCTIONALITY HAS BEEN MOVED TO MANAGER
 * onvm_init.c
 *
 void image_init(struct onvm_nf_info *nf, struct onvm_nf_info *original_nf){

 //first check if the alternate is active or not
 if(original_nf == NULL){
 //create buffer for list of pending image (first time)
 nf->image_info.image_pending = (void *) rte_malloc(NULL, sizeof(void *)*MAX_IMAGE, 0);
 nf->image_info.ready_images = (void *) rte_malloc(NULL, sizeof(void* )*MAX_IMAGE, 0);
 nf->image_info.num_of_ready_images = 0;
 nf->image_info.index_of_ready_image = 0;
 
 int i;
 //empty the array
 for(i = 0; i<MAX_IMAGE; i++)
 nf->image_info.image_pending[i] = NULL;
 }
 else
 {
 //attach to the alternate one's buffer
 nf->image_info = original_nf->image_info;
 } 
 }
 */

/* helper function to get the image index 
 static int get_image_index(image_data *image, image_data **image_list){
 int i;
 for(i = 0; i<MAX_IMAGE; i++){
 if(image_list[i] == image)
 return i;
 }
 return -1;
 }
 */

//get a new image from mempool
/*
 static image_data *get_image(int pkt_file_index, image_data **pending_img){
 int ret;
 struct image_data *img;
 //check if the image already exist...
 if(pending_img[pkt_file_index] != NULL){
 return pending_img[pkt_file_index];
 }
 //else make a new image
 ret = rte_mempool_get(nf_image_pool,(void **)(&img));
 img->num_data_points_stored = 0;
 img->image_id = pkt_file_index; //current logic is to just name the image ID as same as the packet ID. this is not transferrable across the NFs other than the alternate NF
 if(ret != 0){
 RTE_LOG(INFO, APP, "unable to allocate image from pool \n");
 return NULL;
 }

 //now we have to do the accounting... record it..
 pending_img[pkt_file_index] = img;
 return img;
 }

 //put the images back in the mempool
 static void delete_image(image_data *image, struct onvm_nf_info *nf_info){
 image_data ** image_list = (image_data **)nf_info->image_info->ready_images;
 int retval = get_image_index(image, image_list);//find where the mempool is located
 
 if(retval>=0)
 image_list[retval] = NULL;
 
 rte_mempool_put(nf_image_pool, (void *)image);
 }


 //consider single image.
 void copy_data_to_image(void *packet_data,struct onvm_nf_info *nf_info){
 image_data **pending_images = (image_data **)nf_info->image_info->image_pending;
 data_struct *pkt_data = (data_struct *)packet_data;
 image_data *image = get_image(pkt_data->file_id, pending_images);
 
 memcpy(&(image->image_data_arr[pkt_data->position]), pkt_data->data_array, sizeof(float)*pkt_data->number_of_elements);
 image->num_data_points_stored += pkt_data->number_of_elements;

 //printf("number of data points received %d \n",image->num_data_points_stored);
 //check if the image is ready for evaluation
 if(image->num_data_points_stored >= IMAGE_NUM_ELE){
 //add to the ready image list.
 nf_info->image_info->ready_images[(nf_info->image_info->num_of_ready_images+nf_info->image_info->index_of_ready_image)%MAX_IMAGE] = (void *)image;
 image->status = ready;
 image->output_size = IMAGENET_OUTPUT_SIZE; //outputsize for single Image
 image->batch_size = 1;
 nf_info->image_info->num_of_ready_images++;
 //printf("DEBUG.... all packets of image is received \n");
 //remove it from pending image list and put it on 
 pending_images[image->image_id] = NULL;
 }
 }

 //considers user provided batch sizes
 static inline void copy_data_to_image_batch(void *packet_data, struct onvm_nf_info *nf_info, int batch_size){
 image_data **pending_images = (image_data **)nf_info->image_info->image_pending;
 data_struct *pkt_data = (data_struct *)packet_data;
 int batch_id = pkt_data->file_id/batch_size; //which batch does the file belong to
 //printf("Debug, file_id %d the batch id is %d \n",pkt_data->file_id,batch_id);
 image_data *image = get_image(batch_id, pending_images);
 int batch_entry = pkt_data->file_id%batch_size; //where does the file belong in the batch

 
 //check if the batch is empty.. store a timer for first packet.
 if(image->num_data_points_stored == 0){
 clock_gettime(CLOCK_MONOTONIC, &(image->timestamps[5]));
 }
 
 //copy it correctly in a batch.
 memcpy(&(image->image_data_arr[batch_entry*IMAGE_NUM_ELE+pkt_data->position]), pkt_data->data_array, sizeof(float)*pkt_data->number_of_elements);
 image->num_data_points_stored += pkt_data->number_of_elements;
 
 //check if the image is ready for evaluation
 if(image->num_data_points_stored >= IMAGE_NUM_ELE*batch_size){
 //add to the ready image list.
 nf_info->image_info->ready_images[(nf_info->image_info->num_of_ready_images+nf_info->image_info->index_of_ready_image)%MAX_IMAGE] = (void *)image;
 image->status = ready;
 image->output_size = IMAGENET_OUTPUT_SIZE*batch_size;
 image->batch_size = batch_size;
 nf_info->image_info->num_of_ready_images++;
 //printf("DEBUG.... all packets of image is received \n");
 //evaluate if the image is ready to be evaluated.
 if(nf_info->candidate_for_restart != 1){
 evaluate_an_image_from_mempool(NULL, nf_info, image);
 }
 //remove it from pending image list and put it on 
 pending_images[image->image_id] = NULL;
 }
 }
 */

//timer function for inference Aditya check
static void conduct_inference(__attribute__((unused)) struct rte_timer *ptr_timer,void * info) {
	struct onvm_nf_info *nf_info = (struct onvm_nf_info *) info;

	uint32_t throughput = (number_of_images_since_last_computation*1000/NF_INFERENCE_PERIOD_MS);
	uint32_t cpu_latency = hist_extract_v2(&nf_info->cpu_latency,VAL_TYPE_99_PERCENTILE);
	uint32_t gpu_latency = hist_extract_v2(&nf_info->gpu_latency, VAL_TYPE_99_PERCENTILE);

	uint32_t batches_computed = nf_info->batches_inferred_per_sec*(1000/NF_INFERENCE_PERIOD_MS);
	uint32_t batches_above_slo = nf_info->b_i_exceeding_slo_per_sec*(1000/NF_INFERENCE_PERIOD_MS);

	nf_info->b_i_exceeding_slo_per_sec=0;
	nf_info->batches_inferred_per_sec=0;

	struct timespec timestamp;
	clock_gettime(CLOCK_MONOTONIC, &timestamp);
	uint64_t timestamp_64 = timestamp.tv_sec*1000000+timestamp.tv_nsec/1000;

	printf("%"PRIu32",%"PRIu32",%"PRIu32",%"PRIu32",%"PRIu32",%d,%"PRIu64"\n",throughput,gpu_latency,cpu_latency, batches_computed, batches_above_slo,nf_info->ring_flag,timestamp_64);
	number_of_images_since_last_computation = 0;
}

//initializes the timers....
void initialize_ml_timers(struct onvm_nf_info * nf_info) {
	//printf("Aditya's initialize the timer called ... image_state mempool address %p, nf_info address %p\n", nf_info->image_info, nf_info);
	//rte_timer_subsystem_init();//does this happen already.. check
	//rte_timer_init(&image_stats_timer);

	// this was just for inferring to check batch sizes
	rte_timer_init(&image_inference_timer);
	rte_timer_reset_sync(&image_inference_timer,
			(NF_INFERENCE_PERIOD_MS * rte_get_timer_hz())/1000,
			PERIODICAL,
			rte_lcore_id(),//timer_core
			&conduct_inference,
			(void *) nf_info
	);

}

/*
 void compute_ml_stats(__attribute__((unused))struct rte_timer *timer_ptr,void *info){
 struct onvm_nf_info *nf_info = (struct onvm_nf_info *) info;
 //compute the values in per second basis and feed into histogram.
 #ifdef ONVM_GPU_SAME_SIZE_PKTS
 int number_of_images_pending = nf_info->number_of_pkts_outstanding/NUM_OF_PKTS;
 hist_store_v2(&nf_info->image_queueing_rate, number_of_images_pending);
 nf_info->number_of_pkts_outstanding = 0;
 #endif
 hist_store_v2(&nf_info->image_processing_rate, nf_info->number_of_images_processed);
 nf_info->number_of_images_processed = 0;

 //print them if possible.
 //printf("+++++ Printing Histogram of Image Processing Rate +++++\n");
 //hist_print_v2(&nf_info->image_processing_rate);
 }

 //this function will put a marker in the onvm_nf_info so new images are not put in GPU queue
 //and we wait till number of images in GPU processing queue is zero
 //which will then trigger a message to manager saying NF is okay to restart
 void prepare_to_restart(struct onvm_nf_info *nf_info, __attribute__((unused))struct onvm_nf_msg *message){

 printf("Preparing this NF to restart... images in GPU queue %d \n", num_elements_in_gpu_queue);
 //first check how many images are in GPU queue
 if(num_elements_in_gpu_queue <= 0){
 //send message to manager saying can restart now
 onvm_send_gpu_msg_to_mgr(nf_info, MSG_NF_RESTART_OK);
 }
 //else put a flag in nf_info so we do not put the images in GPU queue anymore
 nf_info->candidate_for_restart = 1;
 }

 void execute_dummy_image(void *function_ptr, int img_size){
 float *dummy_image, *dummy_output;
 dummy_image = (float *) rte_malloc(NULL, sizeof(float)*img_size, 0);
 dummy_output = (float *) rte_malloc(NULL, sizeof(float)*1000,0);
 //dummy test...
 //evaluate the image twice with Dummy data... measure the time taken.
 struct timespec timestamps[6];
 evaluate_in_gpu_input_from_host(dummy_image, img_size, dummy_output, function_ptr, &timestamps, 0, NULL, NULL, 1);
 evaluate_in_gpu_input_from_host(dummy_image, img_size, dummy_output, function_ptr, &timestamps,0, NULL, NULL, 1);
 rte_free(dummy_image);
 rte_free(dummy_output);
 }

 */
#endif //ONVM_GPU

static inline void onvm_nflib_start_nf(struct onvm_nf_info *nf_info) {

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	onvm_util_get_start_time(&ts);
#endif
#ifdef ENABLE_MSG_CONSTRUCT_NF_INFO_NOTIFICATION
	struct onvm_nf_msg *startup_msg;
	/* Put this NF's info struct into queue for manager to process startup shutdown */
	if (rte_mempool_get(nf_msg_pool, (void**)(&startup_msg)) != 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot create shutdown msg");
	}
	startup_msg->msg_type = MSG_NF_STARTING;
	startup_msg->msg_data = nf_info;
	if (rte_ring_enqueue(mgr_msg_ring, startup_msg) < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_mempool_put(nf_msg_pool, startup_msg);
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for startup");
	}
#else
	/* Put this NF's info struct onto queue for manager to process startup */
	if (rte_ring_enqueue(mgr_msg_ring, nf_info) < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager");
	}
#endif

	/* Wait for a client id to be assigned by the manager */
	RTE_LOG(INFO, APP, "Waiting for manager to assign an ID...\n");
	struct timespec req = { 0, 1000 }, res = { 0, 0 };
	for (; nf_info->status == (uint16_t) NF_WAITING_FOR_ID;) {
		nanosleep(&req, &res); //sleep(1);
	}

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
	int64_t ttl_elapsed = onvm_util_get_elapsed_time(&ts);
	printf("WAIT_TIME(INIT-->START): %li ns\n", ttl_elapsed);
#endif

	/* This NF is trying to declare an ID already in use. */
	if (nf_info->status == NF_ID_CONFLICT) {
		rte_mempool_put(nf_info_mp, nf_info);
		rte_exit(NF_ID_CONFLICT, "Selected ID already in use. Exiting...\n");
	} else if (nf_info->status == NF_NO_IDS) {
		rte_mempool_put(nf_info_mp, nf_info);
		rte_exit(NF_NO_IDS, "There are no ids available for this NF\n");
	} else if (nf_info->status != NF_STARTING) {
		rte_mempool_put(nf_info_mp, nf_info);
		rte_exit(EXIT_FAILURE,
				"Error occurred during manager initialization\n");
	}
	RTE_LOG(INFO, APP, "Using Instance ID %d\n", nf_info->instance_id);
	RTE_LOG(INFO, APP, "Using Service ID %d\n", nf_info->service_id);

	/* Firt update this client structure pointer */
	this_nf = &nfs[nf_info->instance_id];

	/* Now, map rx and tx rings into client space */
	rx_ring = rte_ring_lookup(get_rx_queue_name(nf_info->instance_id));
	if (rx_ring == NULL)
		rte_exit(EXIT_FAILURE,
				"Cannot get RX ring - is server process running?\n");

	tx_ring = rte_ring_lookup(get_tx_queue_name(nf_info->instance_id));
	if (tx_ring == NULL)
		rte_exit(EXIT_FAILURE,
				"Cannot get TX ring - is server process running?\n");

#if defined(ENABLE_SHADOW_RINGS)
	rx_sring = rte_ring_lookup(get_rx_squeue_name(nf_info->instance_id));
	if (rx_sring == NULL)
	rte_exit(EXIT_FAILURE, "Cannot get RX Shadow ring - is server process running?\n");

	tx_sring = rte_ring_lookup(get_tx_squeue_name(nf_info->instance_id));
	if (tx_sring == NULL)
	rte_exit(EXIT_FAILURE, "Cannot get TX Shadow ring - is server process running?\n");
#endif

	nf_msg_ring = rte_ring_lookup(get_msg_queue_name(nf_info->instance_id));
	if (nf_msg_ring == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get nf msg ring");

#ifdef ENABLE_REPLICA_STATE_UPDATE
	pReplicaStateMempool = nfs[get_associated_active_or_standby_nf_id(nf_info->instance_id)].nf_state_mempool;
#endif
}

#ifdef INTERRUPT_SEM
static void set_cpu_sched_policy_and_mode(void) {
	return;

	struct sched_param param;
	pid_t my_pid = getpid();
	sched_getparam(my_pid, &param);
	param.__sched_priority = 20;
	sched_setscheduler(my_pid, SCHED_RR, &param);
}

static void
init_shared_cpu_info(uint16_t instance_id) {
	const char *sem_name;
	int shmid;
	key_t key;
	char *shm;

	sem_name = get_sem_name(instance_id);
	fprintf(stderr, "sem_name=%s for client %d\n", sem_name, instance_id);

#ifdef USE_SEMAPHORE
	mutex = sem_open(sem_name, 0, 0666, 0);
	if (mutex == SEM_FAILED) {
		perror("Unable to execute semaphore");
		fprintf(stderr, "unable to execute semphore for client %d\n", instance_id);
		sem_close(mutex);
		exit(1);
	}
#endif

	/* get flag which is shared by server */
	key = get_rx_shmkey(instance_id);
	if ((shmid = shmget(key, SHMSZ, 0666)) < 0) {
		perror("shmget");
		fprintf(stderr, "unable to Locate the segment for client %d\n", instance_id);
		exit(1);
	}

	if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
		fprintf(stderr, "can not attach the shared segment to the client space for client %d\n", instance_id);
		exit(1);
	}

	flag_p = (rte_atomic16_t *)shm;

	set_cpu_sched_policy_and_mode();
}
#endif //INTERRUPT_SEM

