#include <stdio.h>
#include <stdlib.h>
#include <rte_malloc.h>

#include <zmq.h>
#include <assert.h>

#include "onvm_gpu.h"
#include "onvm_mgr.h"
#include "onvm_nf.h"
#include "onvm_netml.h"

#ifdef ONVM_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include "onvm_cntk_api.h"
#include "tensorrt_api.h"
#include "onvm_ml_libraries.h"

#define ONVM_MAX_GPU_ML_MODELS (NUMBER_OF_MODELS) 	//Note: Get rid of NUMBER_OF_MODELS
onvm_gpu_model_operational_range_t onvm_gpu_ml_model_profiler_data[ONVM_MAX_GPU_ML_MODELS];
onvm_gpu_ra_info_t gpu_ra_info = {.active_nfs=0, .gpu_ra_avail=MAX_GPU_OVERPRIVISION_VALUE, .gpu_ra_wtlst=0, .waitlisted_nfs=0};
gpu_ra_mgt_t gpu_ra_mgt = {.gpu_ra_info=&gpu_ra_info, .ra_status= {0,}};
//declaration of internal function
void load_gpu_model(struct gpu_file_listing *ml_file);
//void load_old_profiler_data(model_profiler_data *profiler_data);
void load_old_profiler_data(char * filename, int model_index);

cudaIpcMemHandle_t *input_memhandles;//[MAX_NFS/2];
cudaIpcMemHandle_t *output_memhandles;//[MAX_NFS/2];

static inline struct onvm_nf_info *shadow_nf(int);
static inline struct onvm_nf_info *shadow_nf(int instance_id) {
	//returns the info of shadow NF ID
	unsigned shadow_id = get_associated_active_or_standby_nf_id(instance_id);
	struct onvm_nf *cl;
	cl = &(nfs[shadow_id]);

	return cl->info;
}

/****************************************************************************************
 * 						MODEL LOADING FROM DISK AND RETRIEVAL
 ****************************************************************************************/

/* this function is called by the main so that the ML models can be loaded to manager */
void init_ml_models(void) {
	/* the directory where the model is put */
	const char *model_dir = "/home/adhak001/openNetVM-dev/ml_models/";

	/* the file name of ml models */
	const char *models[NUMBER_OF_MODELS];
	//models[0] = "AlexNet_ImageNet_CNTK.model";
	models[0] = "resnet50_batch64.trt";
	models[1] = "ResNet50_ImageNet_CNTK.model";
	models[2] = "VGG19_ImageNet_Caffe.model";
	models[3] = "ResNet152_ImageNet_CNTK.model";
	models[4] = "Fast-RCNN_Pascal.model";
	models[5] = "SRGAN.model";
	models[6] = "resnet50_batch64.trt";
	models[7] = "alexnet_batch128.trt";
	models[8] = "resnet152_batch128.trt";
	models[9] = "vgg19_batch128.trt";
	models[10] = "resnet34_batch128.trt";

	/*the file name of historical runtime data */
	const char *models_historical_dir = "/home/adhak001/openNetVM-dev/models_data/";
	const char *models_runtime[NUMBER_OF_MODELS];
	models_runtime[0] = "alexnet_cntk_runtime.txt";
	models_runtime[1] = "resnet50_cntk_runtime.txt";
	models_runtime[2] = "vgg19_cntk_runtime.txt";
	models_runtime[3] = "resnet152_cntk_runtime.txt";
	models_runtime[4] = "fastrcnn_cntk_runtime.txt";
	models_runtime[5] = "srgan_cntk_runtime.txt";
	models_runtime[6] = "resnet50_tensorrt.txt";
	models_runtime[7] = "alexnet_tensorrt.txt";
	models_runtime[8] = "resnet152_tensorrt.txt";
	models_runtime[9] = "vgg19_tensorrt.txt";
	models_runtime[10] = "resnet34_tensorrt.txt";
	/* platform type */
	ml_platform platforms[NUMBER_OF_MODELS];
	//platforms[0] = cntk;
	platforms[0] = tensorrt;
	platforms[1] = cntk;
	platforms[2] = cntk;
	platforms[3] = cntk;
	platforms[4] = cntk;
	platforms[5] = cntk;
	platforms[6] = tensorrt;
	platforms[7] = tensorrt;
	platforms[8] = tensorrt;
	platforms[9] = tensorrt;
	platforms[10] = tensorrt;
	/* now after setting all that up, let's allocate one information at a time and fill that up */
	struct rte_mempool *ml_model_mempool = rte_mempool_lookup(_GPU_MODELS_POOL_NAME);

	/* now let's loop over mempool and get one model at a time and fill up the details as well as load it up */
	//struct gpu_file_listing *ml_file2 = (struct gpu_file_listing *)rte_malloc(NULL,sizeof(struct gpu_file_listing)*7, 0);
	int i;
	struct gpu_file_listing *ml_file;
	for(i = 0; i < NUMBER_OF_MODELS; i++) {
		//ml_file = &ml_file2[i];
		//ml_file = &ml_files[i];
		rte_mempool_get(ml_model_mempool, (void **)&ml_file);

		ml_file->model_info.platform = platforms[i];
		ml_file->model_info.file_index = i;
		/* copy the directory and file name */
		strncpy(ml_file->model_info.model_file_path, model_dir, strlen(model_dir));
		strncat(ml_file->model_info.model_file_path, models[i], strlen(models[i]));
		/* copy the path of the runtime data file */
		strncpy(ml_file->attributes.profile_data.file_path, models_historical_dir, strlen(models_historical_dir));
		strncat(ml_file->attributes.profile_data.file_path, models_runtime[i], strlen(models_runtime[i]));
		/* now we can send this model to be populated with the load model function */

		cuInit(0); //initializing GPU

		//if(i>6)
		load_gpu_model(ml_file);

		ml_files[i] = ml_file;

	}
	printf("Loaded all the ML files ..\n");

	input_memhandles = (cudaIpcMemHandle_t *)rte_malloc(NULL,sizeof(cudaIpcMemHandle_t)*(MAX_NFS/2),0);
	output_memhandles = (cudaIpcMemHandle_t *)rte_malloc(NULL,sizeof(cudaIpcMemHandle_t)*(MAX_NFS/2),0);
	cudaError_t cuda_return;
	/* create cudaMalloc for multiple NFs */
	for(i = 0; i<MAX_NFS/2; i++) {
		cuda_return = cudaMalloc(&gpu_side_input_buffer[i],SIZE_OF_AN_IMAGE_BYTES*MAX_IMAGES_BATCH_SIZE);
		if(cuda_return != cudaSuccess) {
			printf("Cannot malloc dev buffer for NF\n");
		}
		cuda_return = cudaIpcGetMemHandle(&input_memhandles[i],gpu_side_input_buffer[i]);
		if(cuda_return != cudaSuccess) {
			printf("Cannot get ipc handle for NF\n");
		}

		//now output side buffer
		cuda_return = cudaMalloc(&gpu_side_output_buffer[i], sizeof(float)*1000*MAX_IMAGES_BATCH_SIZE);
		if(cuda_return != cudaSuccess) {
			printf("Cannot malloc dev buffer for NF\n");
		}

		cuda_return = cudaIpcGetMemHandle(&output_memhandles[i],gpu_side_output_buffer[i]);
		if(cuda_return != cudaSuccess) {
			printf("Cannot get ipc handle for NF\n");
		}
		else
		printf("got IPC for output buffer \n");

	}

}

/* loads the model for the manager */
void load_gpu_model(struct gpu_file_listing *ml_file) {

	int flag = 1; //all models are loaded to GPU, flag = 0 cpu only, flag = 1 gpu only
	int num_of_parameters = 0;//the number of parameters of a GPU side function

	/* create load model parameters */
	nflib_ml_fw_load_params_t load_model_params;
	load_model_params.file_path = ml_file->model_info.model_file_path;
	load_model_params.load_options = 1; //gpu model

	void *aio = NULL;

	/* load the ml model onto GPU */
	struct timespec begin,end;
	clock_gettime(CLOCK_MONOTONIC, &begin);
	if(ml_file->model_info.platform == cntk) {

		cntk_load_model(&load_model_params, aio); //loads the model
		printf("model handle %p \n", load_model_params.model_handle);
		ml_file->gpu_handle = load_model_params.model_handle;
		ml_file->model_info.model_handles.number_of_parameters = cntk_count_parameters(ml_file->gpu_handle);
		//we have to have loading API here.
		clock_gettime(CLOCK_MONOTONIC, &end);

		double time_taken_to_load = (end.tv_sec-begin.tv_sec)*1000000.0+(end.tv_nsec-begin.tv_nsec)/1000.0;
		printf("----- Time taken to load model %d is %f microseconds on GPU \n",ml_file->model_info.file_index,time_taken_to_load);

		//Now time to get file attributes..
		//general attributes..

		if(flag== 1)//gpu model
		{

			/* getting cuda handles */
			cudaIpcMemHandle_t * mem_handles= (cudaIpcMemHandle_t *) rte_malloc(NULL, ((sizeof(cudaIpcMemHandle_t))*(ml_file->model_info.model_handles.number_of_parameters)),0);

			printf("Number of cuda handles made %d\n",ml_file->model_info.model_handles.number_of_parameters);
			cntk_get_cuda_pointers_handles(ml_file->gpu_handle,mem_handles);
			ml_file->model_info.model_handles.cuda_handles = mem_handles;
			printf("Memhandle pointer %p \n", ml_file->model_info.model_handles.cuda_handles);
			//ml_file->model_info.model_handles.cuda_handles = mem_handles;
		}
		//update the attribute for CPU side..
		if(flag == 0)//cpu model
		{
			ml_file->model_info.model_handles.number_of_parameters = num_of_parameters;
		}
	}
	if(ml_file->model_info.platform == tensorrt) {
		printf("Loading tensorrt model \n");
	}
	// load the csv file for data
	load_old_profiler_data(ml_file->attributes.profile_data.file_path,ml_file->model_info.file_index);
}

void load_old_profiler_data(char * file_path,int model_index ) {//model_profiler_data *profiler_data) {
	FILE *fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	char buffer[1025];

	size_t bytes;

	int number_of_lines = 0;

	/* open the runtime file */
	fp = fopen(file_path,"r");
	if (fp == NULL) {
		printf("Couldn't open %s\n", file_path);
		return;
	}
	/* this file should be organized in following way */
	// optimal percentage, step, percentage_range...
	/* first count number of lines in the file */
	while((bytes=fread(buffer, 1, sizeof(buffer)-1, fp))) {
		//lastchar = buffer[bytes-1];
		for(char *c = buffer; (c = memchr(c, '\n', bytes-(c-buffer))); c++) {
			number_of_lines++;
		}
		//check the code here https://codereview.stackexchange.com/questions/156477/c-program-to-count-number-of-lines-in-a-file
	}
	//go back to beginning of the file
	rewind(fp);

	//now we know number of lines in the runtime data file, we need to allocate space for each data we collect...
	//currently we store the runtime latency and corresponding runtime percentages and number of SMSss.
	//the file looks like
	// sm, percentage, latency for now
	/*
	 profiler_data->number_of_values = number_of_lines;
	 profiler_data->num_of_sm = (int *)rte_malloc(NULL, sizeof(int)*profiler_data->number_of_values, 0);
	 profiler_data->runtime_percentages = (int *)rte_malloc(NULL, sizeof(int)*profiler_data->number_of_values, 0);
	 profiler_data->runtime_latency = (int *)rte_malloc(NULL, sizeof(int)*profiler_data->number_of_values, 0);

	 */

	//rather than reading the profiler data, we need to read it as the optimal value, step value and operational range
	//get the pointer to the operational range data
	onvm_gpu_model_operational_range_t *operational_range = &onvm_gpu_ml_model_profiler_data[model_index];
	memset(operational_range, 0, sizeof(*operational_range));
	const char * token;

	int i;
	for(i = 0; i<number_of_lines; i++) {
		if((read = getline(&line, &len, fp)) != -1) {
			token = strtok(line, ",");
			sscanf(token, "%"SCNd16, &(operational_range->optimal_value));
			//keep scanning the rest of the line
			//printf("Optimal value read %"PRIu16"\n", operational_range->optimal_value);

			token = strtok(NULL, ",");
			if(token != NULL)
			sscanf(token, "%"SCNd16, &(operational_range->step_value));

			token = strtok(NULL, ",");
			if(token != NULL)
			sscanf(token, "%"SCNd16, &(operational_range->operational_range.min));

			token = strtok(NULL, ",");
			if(token != NULL)
			sscanf(token, "%"SCNd16, &(operational_range->operational_range.max));
		}
	}
}

/* sends the model info for the NF */
void * provide_nf_with_model(int file_index) {

	//Now get all the attributes of the ML model we are checking
	return (void *) &(ml_files[file_index]->model_info);
}

/* sends the input gpu buffer to the NF */
void * provide_nf_with_input_gpu_buffer(int service_id) {
	return (void *) &(input_memhandles[service_id]);
}

/* sends the output gpu buffer to the NF */
void * provide_nf_with_output_gpu_buffer(int service_id) {
	return (void *) &(output_memhandles[service_id]);
}

//check if the alternate is not null

void restart_nf(struct onvm_nf_info *nf) {
	//get the PID and construct a zmsg and send it out to orchestartor
	printf("####____------#### restarting the NF %d \n", nf->pid);
	send_message_to_orchestrator(create_zmsg(&(nf->pid),1, zrestart));
}

//this function should be only called if the NF has been ordered to get ready, i.e. called by not active NF only
//in case of first NF, this will never get called...
void nf_is_gpu_ready(struct onvm_nf_info *nf) {
	//TODO just print for now
	printf("NF instance %d is now ready to process packets \n", nf->instance_id);
	//find the alternate NF... and then send it a message to shutdown...
	struct onvm_nf_info *alt_nf = shadow_nf(nf->instance_id);
	inform_NF_of_pending_restart(alt_nf);//sent message to tell the logic to restart it.

	//now let's change the access to the ring... the alt NF should be put in paused state and this NF in running state.
	//we have to send the message to NF to perform these.
	//the wakeup mgr is automatically called for that..
	onvm_nf_send_msg(nf->instance_id, MSG_RESUME,0,NULL);

	//send to active NF to stop
	onvm_nf_send_msg(alt_nf->instance_id, MSG_STOP, 0, NULL);
	printf("DEBUG, Sent message to both NFs to stop/wakeup etc.. \n");
}

/****************************************************************************************
 * 						GPU Resource Allocation Management and Scheduling
 ****************************************************************************************/
/* Helper functions for finding the GPU percent wise compute time ... computed by undergrads */
//int suggest_gpu_percentage(float request_rate, int gpu_model);
inline int onvm_gpu_adjust_nf_gpu_perecentage(struct onvm_nf_info *nf);
inline int onvm_gpu_set_gpu_percentage(struct onvm_nf_info *nf, uint16_t gpu_percent);
inline int onvm_gpu_check_any_readjustment(void);
inline int onvm_gpu_check_any_readjustment(void) {
	int i = 0;

	for(i=0; i<MAX_NFS; i++) {
		if(nfs[i].info->over_provisioned_for_slo || nfs[i].info->under_provisioned_for_slo) return 1;
	}
	return 0;
}

inline int onvm_gpu_set_gpu_percentage(struct onvm_nf_info *nf, uint16_t gpu_percent) {
	nf->gpu_percentage = gpu_percent;
	gpu_ra_mgt.nf_gpu_ra_list[nf->instance_id] = gpu_percent;
	gpu_ra_mgt.gpu_ra_info->gpu_ra_avail -= gpu_percent;
	gpu_ra_mgt.gpu_ra_info->active_nfs++;
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_IS_SET;
	return 0;
}
inline int onvm_gpu_set_wt_list_gpu_percentage(struct onvm_nf_info *nf, uint16_t gpu_percent) {
	nf->gpu_percentage = gpu_percent;
	gpu_ra_mgt.nf_gpu_ra_list[nf->instance_id] = gpu_percent;
	gpu_ra_mgt.gpu_ra_info->gpu_ra_wtlst+= gpu_percent;
	gpu_ra_mgt.gpu_ra_info->waitlisted_nfs++;
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_IS_WAITLISTED;
	return 0;
}
inline int compute_current_gpu_ra_stats(uint8_t *num_act_nfs, uint16_t *gpu_ra_avl_pct) {
	int i = 0;
	uint8_t act_nfs_count=0;
	uint16_t gpu_ra_used=0;

	for(;i<MAX_NFS;i++) {
		if ((onvm_nf_is_valid(&nfs[i])) && (nfs[i].info->gpu_percentage)) {
			act_nfs_count++;
			gpu_ra_used+=nfs[i].info->gpu_percentage;
		}
	}
	if(num_act_nfs) {
		*num_act_nfs=act_nfs_count;
	}
	if(gpu_ra_avl_pct) {
		*gpu_ra_avl_pct= ((MAX_GPU_OVERPRIVISION_VALUE < gpu_ra_used)?(0):(MAX_GPU_OVERPRIVISION_VALUE - gpu_ra_used));
	}
	return act_nfs_count;
}

//Note: Resource must be released/readjusted when NF terminates/quits or restarts.
inline int onvm_gpu_adjust_nf_gpu_perecentage(struct onvm_nf_info *nf) {

	/*
	 //For Low Priority: Just set to Minimum TODO: Should we enable low prio to use the full GPU when none are using-- yes. then this is not good!
	 if(0 == nf->gpu_priority) {
	 uint16_t underprovision_val = onvm_gpu_ml_model_profiler_data[nf->gpu_model].operational_range.min;
	 //can fit the NF resource within underprovision value
	 if(underprovision_val < gpu_ra_info.gpu_ra_val) {
	 //try to provide as much left over above the underprovision value;
	 nf->gpu_percentage = gpu_ra_info.gpu_ra_val;
	 gpu_ra_info.gpu_ra_val -= nf->gpu_percentage;//=0;
	 gpu_ra_info.active_nfs++;
	 } else {
	 nf->gpu_percentage = 0; //just waitlist this low priority NF.
	 }
	 return 0;
	 }
	 //High Priorty NFs

	 */
	if(!nf) return 0;
#if 0

#endif
	return 0;
}
/* find the throughput for certain model at certain percentage */
//float find_max_throughput(int gpu_model, int gpu_percentage);
/* the model that computes the GPU allocation and then recommends the new GPU percentage */
//void compute_GPU_allocation(struct onvm_nf_info *nf);
/* ***********
 * The engine to compute if we need to change the percentage
 ************
 * what do we need?
 * List of all the NFs that are using GPU.. and their Average Runtime of their program
 * The NFs should provide following things:
 * How many images they are getting Per_seconds
 * How many images they can process per_second
 * if the first one is greater than 2nd one.. we should consider resetting the percentage

 * We need to have this for multiple NFs and find optimal percentage
 * Already pre-compiled list of Runtime on different percentage for that NF
 */
/*
 void compute_GPU_allocation(struct onvm_nf_info *nf ){
 //get the reporting from an NF
 int nf_gpu_percentage = nf->gpu_percentage;
 //TODO: These should be got from histogram
 //float request_rate = nf->requests_per_second;
 //float throughput = nf->images_throughput;
 float request_rate = 0.0;
 float throughput = 0.0;


 float max_throughput = find_max_throughput(nf->gpu_model, nf_gpu_percentage);
 int recommended_gpu_percentage;

 printf(" nf_gpu_percentage %d request rate: %f and throughput %f \n",nf_gpu_percentage, request_rate, throughput);

 //recommend the NF to be restarted if the request per seconds are more than the throughput
 if(request_rate > 0.8*max_throughput)
 {
 // if the request rate is creeping up to 80% of max  we suggest that this NF be considered reallocating the resource
 nf->candidate_for_restart = 1;
 recommended_gpu_percentage = suggest_gpu_percentage(request_rate,nf->gpu_model);//find from the table

 //this means we are going to provide the shadow NF with the GPU percentage
 get_shadow_NF_ready(nf, recommended_gpu_percentage);

 //inform the NF that it is going to be restarted... only when the shadow NF replies, we going to inform previous of restart
 //inform_NF_of_pending_restart(nf);
 }

 }
 */

/* helper function to find the experimental throughput in for this model at this percentage
 float find_max_throughput(int model_index, int gpu_percentage){
 int i;
 int num_records = file_listing[model_index].attributes.num_of_runtimes;
 float *runtimes = file_listing[model_index].attributes.run_times;
 int *gpu_percentages = file_listing[model_index].attributes.gpu_percentages;
 for(i = 0; i<num_records; i++){
 if(gpu_percentages[i] >= gpu_percentage)
 return runtimes[i];
 }
 return runtimes[num_records-1];//send the best runtime
 }
 */

/* helper functions for model management
 int suggest_gpu_percentage(float request_rate, int model_index){
 //find the attributes...
 int num_records = file_listing[model_index].attributes.num_of_runtimes;
 float *runtimes = file_listing[model_index].attributes.run_times;
 int *gpu_percentages = file_listing[model_index].attributes.gpu_percentages;
 int i;
 for(i = 0; i < num_records; i++){
 if (runtimes[i]  >= 1.2*request_rate)
 return gpu_percentages[i];
 }
 return 100; //in case request rate is very large, give all the GPU to the program
 }
 */
/****************************************************************************************
 * 						GPU Resource Allocation and Management APIs
 ****************************************************************************************/
//Function to be called when NF termainates/killed/ or is transitioned to move out (special case move to pause state).
inline int onvm_gpu_release_gpu_percentage_for_nf(struct onvm_nf_info *nf) {
	if(!nf) return (0);
	if((GPU_RA_IS_SET != gpu_ra_mgt.ra_status[nf->instance_id]) && (GPU_RA_NEED_TO_RELINQUISH != gpu_ra_mgt.ra_status[nf->instance_id])) return 0;

	// release GPU from the NF and add back to the GPU RM Pool.
	gpu_ra_mgt.gpu_ra_info->gpu_ra_avail += nf->gpu_percentage;
	nf->gpu_percentage = 0;
	gpu_ra_mgt.gpu_ra_info->active_nfs--;
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_NOT_SET;
	gpu_ra_mgt.nf_gpu_ra_list[nf->instance_id] = 0;

	return 0;
}
/** API to get the NFs GPU % share: This will initially allocate 100% and oversubscribe to MAX (200%).
 Thereafter we need to reapportion GPU fairly amongst the contending NFs. (Policies: Uniform vs Rate vs cost vs Rate-cost proportional.)

 */
inline int onvm_gpu_get_gpu_percentage_for_nf(struct onvm_nf_info *nf) {
	if(!nf) return (0);

	//check for valid gpu model
	if( (0>= nf->gpu_model) || (ONVM_MAX_GPU_ML_MODELS <= nf->gpu_model)) return (0);

	//IF NF already has percentage set, then ignore the call :: Double check what should be done here.. not clear yet!
	if((nf->gpu_percentage) /*&& (GPU_RA_IS_SET == gpu_ra_mgt.ra_status[nf->instance_id])*/) return 0;

	//IF NF is marked for readjustment or is marked to relinquish its GPU resource then ignore.
	if(GPU_RA_NEEDS_READJUSTMENT == gpu_ra_mgt.ra_status[nf->instance_id] || GPU_RA_NEED_TO_RELINQUISH == gpu_ra_mgt.ra_status[nf->instance_id]) return 0;

	onvm_gpu_model_operational_range_t *gpu_ml_info = &(onvm_gpu_ml_model_profiler_data[nf->gpu_model]);
	//check if model has valid pre-profiled data
	nf->gpu_monitor_lat = (gpu_ml_info->optimal_value)?(0):(1);
	//set default as optimal value
	nf->gpu_percentage = (gpu_ml_info->optimal_value)?(gpu_ml_info->optimal_value):(DEFAULT_GPU_RA_VALUE);
	//set the GPU Resource Mgt state update
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_NOT_SET;

	/*
	 if(nf->gpu_percentage) {
	 //increment or decrement based on the run-time profiling results. TODO: handle this!
	 printf("NF Instance[%d] for model[%d] is already allocated:[%d], GPU (optimal=[%d], min=[%d], max=[%d]) RA(val=[%d], nfs=[%d]), \n",
	 nf->instance_id, nf->gpu_model, nf->gpu_percentage,
	 gpu_ml_info->optimal_value, gpu_ml_info->operational_range.min, gpu_ml_info->operational_range.max,
	 gpu_ra_info.gpu_ra_val, gpu_ra_info.active_nfs);
	 return 0;
	 
	 } else {
	 nf->gpu_percentage = gpu_ml_info->optimal_value;
	 }*/

	//Check current GPU status, whether it is underutilized (make sure to always allocate 100%+ of GPU resource)
	if(gpu_ra_info.gpu_ra_avail >= GPU_MAX_RA_PER_NF) {
		onvm_gpu_set_gpu_percentage(nf, GPU_MAX_RA_PER_NF);
	}
	// Check if request can be sufficiently met, with a step overprovision for NF?
	else if (/*(0 == nf->gpu_monitor_lat) && */gpu_ra_info.gpu_ra_avail > nf->gpu_percentage) {
		//Space to over-provision the NFs GPU percentage.
		uint16_t overprovision_val = gpu_ml_info->operational_range.max;//onvm_gpu_ml_model_profiler_data[nf->gpu_model].operational_range.max;
		if(overprovision_val > gpu_ra_info.gpu_ra_avail) {
			onvm_gpu_set_gpu_percentage(nf, gpu_ra_info.gpu_ra_avail); //nf->gpu_percentage = gpu_ra_info.gpu_ra_val;
		} else {
			onvm_gpu_set_gpu_percentage(nf, overprovision_val); //nf->gpu_percentage = overprovision_val;
		}
	}
	// Check if GPU can satisfy step under-provision below knee for this model?
	else {
		uint16_t underprovision_val = gpu_ml_info->operational_range.min; //onvm_gpu_ml_model_profiler_data[nf->gpu_model].operational_range.min;
		//can fit the NF resource within underprovision value
		if(underprovision_val < gpu_ra_info.gpu_ra_avail) {
			//try to provide as much left over above the underprovision value;
			onvm_gpu_set_gpu_percentage(nf, gpu_ra_info.gpu_ra_avail);//nf->gpu_percentage = gpu_ra_info.gpu_ra_val;
		} else {

			//need to readjust resources of all NFs to accomodate this otherwise deny admission to this NF
			onvm_gpu_set_wt_list_gpu_percentage(nf, nf->gpu_percentage);
			gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_NEEDS_ALLOCATION;

			//high priority: need to allocate and preempt low piority if any;
			if(nf->gpu_priority) {
				//high priority: need to allocate and preempt low piority if any;
			} else {
				nf->gpu_percentage = 0; //just waitlist this low priority NF.
				//gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_IS_WAITLISTED;
			}
		}
	}
	//try to adjust the NFs GPU % based on the Global contention ( call only in timer callback when any NF is waiting for gpu ra needs allocation)
	//onvm_gpu_adjust_nf_gpu_perecentage(nf);

	printf("NF Instance[%d: Prio:%d] for model[%d] is allocated:[%d], GPU ( RA_status=[%d], optimal=[%d], min=[%d], max=[%d]) RA(val=[%d], nfs=[%d]), \n",
			nf->instance_id, nf->gpu_priority, nf->gpu_model, nf->gpu_percentage, gpu_ra_mgt.ra_status[nf->instance_id],
			gpu_ml_info->optimal_value, gpu_ml_info->operational_range.min, gpu_ml_info->operational_range.max,
			gpu_ra_info.gpu_ra_avail, gpu_ra_info.active_nfs);

	return (0);
}

int onvm_gpu_check_gpu_ra_mgt(void) {
	uint8_t act_nfs;
	uint16_t gpu_ra_available;

	//check if any NFs are waiting for RA
	if((0 == gpu_ra_mgt.gpu_ra_info->gpu_ra_wtlst)||(0 == onvm_gpu_check_any_readjustment())) return 0;

	compute_current_gpu_ra_stats(&act_nfs, &gpu_ra_available);
	if((0==act_nfs)|| (gpu_ra_available == MAX_GPU_OVERPRIVISION_VALUE)) return 0;

	int i = 0;
	uint8_t needs_ra=0, readj_ra=0;
	uint8_t nfs_need_ra_list[MAX_NFS], nfs_readj_ra_list[MAX_NFS];
	//count the num of NFs that need GPU RA
	for(i=0; i<MAX_NFS; i++) {
		if((GPU_RA_NEEDS_ALLOCATION == gpu_ra_mgt.ra_status[i]) || (GPU_RA_IS_WAITLISTED == gpu_ra_mgt.ra_status[i])) {
			nfs_need_ra_list[needs_ra] = i;
			needs_ra+=1;
		}
		else if ((GPU_RA_NEEDS_READJUSTMENT == gpu_ra_mgt.ra_status[i])) {
			nfs_readj_ra_list[readj_ra] = i;
			readj_ra+=1;
		}
	}
	//TODO: Must have logic to track, find optimal Rate-cost proportional share for each NF and then trigger restart NFs for all that have changed GPU RA Profile.
	for(i=0; i<needs_ra; i++) {
		printf("Needs GPU Allocation: %d, %d, %d", i, nfs_need_ra_list[i], nfs_need_ra_list[i]);
	}
	for(i=0; i<readj_ra; i++) {
		printf("Needs GPU Allocation: %d, %d, %d", i, nfs_readj_ra_list[i], nfs_readj_ra_list[i]);
	}

	return 0;
}
/****************************************************************************************
 * 						NF Orchestrator specific functions
 ****************************************************************************************/
/*let the NF know that it is to be restarted.. similarly, let the other NF know that it should load the model in GPU and run dummy data */
void inform_NF_of_pending_restart(struct onvm_nf_info *nf) {
	onvm_nf_send_msg(nf->instance_id, MSG_RESTART, 0, NULL);
}

/* the function to send message to shadow NF */
void get_shadow_NF_ready(struct onvm_nf_info *shadow, int gpu_percentage) {
	struct get_alternate_NF_ready* alternate_message = (void *)rte_malloc(NULL, sizeof(int)+sizeof(void*), 0);
	alternate_message->gpu_percentage = gpu_percentage;

	struct onvm_nf_info * alternate_nf = shadow_nf(shadow->instance_id);

	if(onvm_nf_is_valid(&nfs[alternate_nf->instance_id])) {
		//alternate_message->image_info = alternate_nf->image_info;
		onvm_nf_send_msg((shadow_nf(shadow->instance_id))->instance_id, MSG_GET_GPU_READY, 0, alternate_message);
	}
	else
	{
		printf("Alternate NF not ready so No \"ready\" message passed \n");
	}

}

#ifdef ONVM_GPU_TEST
void voluntary_restart_the_nf(struct onvm_nf_info *nf) {
	//check if alternate is active... if not cancel the volutary restart
	struct onvm_nf_info *alternate_nf = shadow_nf(nf->instance_id);
	if(!onvm_nf_is_valid(&nfs[alternate_nf->instance_id])) {
		printf("The alternate NF is not running.. cannot initiate voluntary restart \n");
	}
	else
	{
		get_shadow_NF_ready(nf, 50);
	}

}
#endif

/* ------- <MESASAGING API > ******** */
void init_zmq(void) {
	ipc_file_path = "ipc:///home/adhak001/dev/ipc_file";
	zmqContext = zmq_init(1);
	zmqRequester = zmq_socket(zmqContext, ZMQ_PUSH);
	int rc = zmq_connect(zmqRequester, ipc_file_path);
	assert (rc == 0);
	printf("ZMQ apparatus ready \n");
}

/* Function to send message to orchestrator */
int send_message_to_orchestrator(zmgr_msg * message) {
	//char buffer[6];
	size_t msg_size = message->msg_size;
	zmq_send(zmqRequester, message, msg_size, 0);
	//now wait for the reply
	//zmq_recv(zmqRequester, buffer, 6 ,0); // we only expect "OK" .. there is no need to process the message now
	rte_free(message);
	return 0;
}

/* creates a zmesg to be sent */
zmgr_msg *create_zmsg(pid_t pid[], int num_nfs,nf_state state) {
	zmgr_msg *new_msg = (zmgr_msg*) rte_malloc(NULL, sizeof(zmgr_msg), 0);
	new_msg->state = state;
	new_msg->msg_size = sizeof(zmgr_msg);

	if(num_nfs >1) {
		new_msg->information.num_of_elements = num_nfs;
		memcpy(new_msg->information.pid_array, pid, sizeof(pid_t)*num_nfs);
	}
	else
	{
		new_msg->information.num_of_elements = pid[0];
	}
	clock_gettime(CLOCK_MONOTONIC, &(new_msg->timestamp));
	return new_msg;
}

#endif //ONVM_GPU
