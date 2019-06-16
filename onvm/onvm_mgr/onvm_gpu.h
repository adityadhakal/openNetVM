#ifndef ONVM_GPU_H
#define ONVM_GPU_H

#include "onvm_common.h"
#ifdef ONVM_GPU
#include <string.h>
#include <time.h>

#include "onvm_cntk_api.h"

#define GPU_MAX_RA_PER_NF	(100)			// indictes max % that can be allocated to an NF
#define MAX_GPU_OVERPRIVISION_VALUE	(200)	// indicates max % at which GPU can be over provisioned or oversubscribed.
#define DEFAULT_GPU_RA_VALUE	(30)		// indicates the default value to use \% for models that do not have profiled data. When set, also set nf->gpu_monitor_lat to TRUE.
typedef struct onvm_gpu_range_t {
	uint16_t min;
	uint16_t max;
}onvm_gpu_range_t;
typedef struct onvm_gpu_range_t onvm_gpu_throughput_range; // Number of Images per Second
typedef struct onvm_gpu_range_t onvm_gpu_latency_range_ms;// Latency in MilliSeconds.
typedef struct onvm_gpu_model_operational_range {
	uint16_t optimal_value;				//represents the knee % to be used
	uint16_t step_value;//represents the step increment or decrement that needs to be applied	(also can be used for applying over-provision unit for the NF)
	onvm_gpu_range_t operational_range;//represents the permissible GPU % range to operate this NF.
}onvm_gpu_model_operational_range_t;

typedef enum nf_gpu_ra_status {
	GPU_RA_NOT_SET=0,           //initial state of NF on bootup
	GPU_RA_NEEDS_ALLOCATION=1,//when NF requests for GPU but fails to get and triggers for reallocation
	GPU_RA_IS_SET=2,//When GPU RA is completed
	GPU_RA_NEEDS_READJUSTMENT=3,//When NF is marked for readjussment due to other NFs entry -- these NFs need to switch on peer/back.
	GPU_RA_NEED_TO_RELINQUISH=4,//When NF is marked to restart. (active-replica switch, active goes down while replica needs RA)
	GPU_RA_IS_WAITLISTED=5,//When NF requests and fails to get GPU allocation, (not sure, if we need this or use GPU_RA_NEEDS_ALLOCATION)
}nf_gpu_ra_status_e;
typedef struct gpu_ra_mgt_t {
	onvm_gpu_ra_info_t *gpu_ra_info;
	nf_gpu_ra_status_e ra_status[MAX_NFS];
	uint32_t nf_gpu_ra_list[MAX_NFS];
}gpu_ra_mgt_t;

/****************************************************************************************
 * 						MODEL LOADING FROM DISK AND RETRIEVAL
 ****************************************************************************************/
/* a function to initialize and load all the existing models */
//void load_model(struct gpu_file_listing *ml_file);
void init_ml_models(void);

/* load the historic per-computed data per file */
void load_old_data(model_profiler_data *runtime_data);

//variable for storing all models information //this struct is in onvm_common.h
struct gpu_file_listing * ml_files[NUMBER_OF_MODELS];

//cuda malloc pointers
void * gpu_side_input_buffer[MAX_NFS/2];
void * gpu_side_output_buffer[MAX_NFS/2];

#define NUM_OF_RUNTIME_DATAPOINTS 10

/* a function that provides NF with model's GPU pointers */
void * provide_nf_with_model(int gpu_model_id);

/* sends the input gpu buffer to the NF */
void * provide_nf_with_input_gpu_buffer(int service_id);

/* sends the output gpu buffer to the NF */
void * provide_nf_with_output_gpu_buffer(int service_id);

/****************************************************************************************
 * 						GPU Resource Allocation Management and Scheduling
 ****************************************************************************************/
/* Helper functions for finding the GPU percent wise compute time ... computed by undergrads */
//int suggest_gpu_percentage(float request_rate, int gpu_model);
int onvm_gpu_get_gpu_percentage_for_nf(struct onvm_nf_info *nf);

/* API to let dying NF release its GPU resource and reclaim it to available pool. */
int onvm_gpu_release_gpu_percentage_for_nf(struct onvm_nf_info *nf);

/* Function called by timer or periodic thread to check GPU RA MGT */
int onvm_gpu_check_gpu_ra_mgt(void);

/* find the throughput for certain model at certain percentage */
//float find_max_throughput(int gpu_model, int gpu_percentage);
/* the model that computes the GPU allocation and then recommends the new GPU percentage */
//void compute_GPU_allocation(struct onvm_nf_info *nf);
/****************************************************************************************
 * 						NF Status Mapping and Miscellaneous functions
 ****************************************************************************************/

/*we can restart the NF safely now */
void restart_nf(struct onvm_nf_info *nf);

/* inform the NF should get ready for restart */
void inform_NF_of_pending_restart(struct onvm_nf_info *nf);

/*get the shadow NF ready */
void get_shadow_NF_ready(struct onvm_nf_info *nf, int recommended_gpu_percentage);

/* we know the shadow NF is ready for GPU execution, can restart the original NF if it is restart ready */
void nf_is_gpu_ready(struct onvm_nf_info *nf);

//these two functions should check the nf_info struct so they only send the restart once.

/* NF says it is okay to be restarted... restart it only if the shadow NF is ready */
void nf_is_okay_to_restart(struct onvm_nf_info *nf);

#ifdef ONVM_GPU_TEST
/* the function to test above apparatus */
void voluntary_restart_the_nf(struct onvm_nf_info *nf);
#endif

/****************************************************************************************
 * 						GPU AND Node Orchestrator Functions
 ****************************************************************************************/
/* a struct that will have information about the model.. eg. file path, its runtime data file and model number */

/* file name business 
 char ml_file_name[NUMBER_OF_MODELS][100];
 const char * ml_model_name[NUMBER_OF_MODELS];
 */
/* actually fill in the file paths 
 const char *model_dir;

 */
/* loading the filename.... now we will have fixed number of files hardcoded 
 static inline void set_filename(void){
 model_dir = "/home/adhak001/openNetVM-dev/ml_models/";
 strcpy(ml_file_name[0], model_dir);
 strcat(ml_file_name[0],"AlexNet_ImageNet_CNTK.model");
 ml_model_name[0] = "alexnet_runtime.txt";
 strcpy(ml_file_name[1], model_dir);
 strcat(ml_file_name[1],"ResNet50_ImageNet_CNTK.model");
 ml_model_name[1] = "resnet50_runtime.txt";
 strcpy(ml_file_name[2], model_dir);
 strcat(ml_file_name[2],"VGG19_ImageNet_Caffe.model");
 ml_model_name[2] = "vgg19_rutime.txt";
 strcpy(ml_file_name[3], model_dir);
 strcat(ml_file_name[3],"ResNet152_ImageNet_CNTK.model");
 ml_model_name[3] = "resnet152_runtime.txt";
 strcpy(ml_file_name[4], model_dir);
 strcat(ml_file_name[4],"Fast-RCNN_Pascal.model");
 ml_model_name[4] = "Fast-RCNN_runtime.txt";
 strcpy(ml_file_name[5], model_dir);
 strcat(ml_file_name[5],"SRGAN.model");
 ml_model_name[5] = "SRGAN_runtime.txt";
 strcpy(ml_file_name[6], model_dir);
 strcat(ml_file_name[6], "resnet50_batch64.trt");
 ml_model_name[6] = "resnet50_tensorrt.txt";
 }

 */

/*
 static inline struct onvm_nf_info *shadow_nf(int instance_id){
 if(instance_id > 8)
 instance_id -= 8;
 else
 instance_id += 8;

 struct onvm_nf *cl;
 cl = &(nfs[instance_id]);
 
 return cl->info;
 }
 */
/* Data for ZMQ message passing */

/* Enum for orchestrator */
typedef enum nf_state {zstart, zrestart, zstop}nf_state;

typedef struct zinformation_format {
	int num_of_elements;
	int pid_array[5];
}zinfo_format;

/* message passing struct between ONVM manager and orchestrator */
typedef struct zmgr_msg_struct {
	nf_state state;
	size_t msg_size;
	struct timespec timestamp;
	zinfo_format information; //size of this information have to be known... I recommend an int (4 bytes) + int array of size 5 (20 bytes)
}zmgr_msg;
//check the information format below

//some variables
void * zmqContext;
void * zmqRequester;
const char * ipc_file_path;// = "ipc:///home/adhak001/dev/ipc_file";

/* the function to init the zmq */
void init_zmq(void);

/* the function to send the message to orchestrator */
int send_message_to_orchestrator(zmgr_msg *message);

zmgr_msg * create_zmsg(pid_t nf_pid[], int num_of_nfs, nf_state state);
#endif
#endif
