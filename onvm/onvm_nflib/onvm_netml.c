#include <stdio.h>
#include "onvm_common.h"
#include "onvm_netml.h"
#include "onvm_gpu_buffer_factory.h"
#include <strings.h>//for ffs
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_byteorder.h>
#ifdef ONVM_GPU
//puts the data into 
#include "onvm_nflib.h"

uint32_t data_aggregation(struct rte_mbuf *pkt, image_batched_aggregation_info_t *image_agg, uint32_t *ready_images_index) {
	uint32_t ready_images=0;
	//static placeholder variable for a single image
	void *payload;
	image_chunk_header_t *chunk_header;
	//first find which image this packet belongs to
	payload = (void *)rte_pktmbuf_mtod_offset(pkt, void *, (sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr)));

	chunk_header =(image_chunk_header_t *)( (char * )payload + 2);// 2bytes offset to make it 4byte aligned address.

	//printf("  ID %d start offset %"PRIu32" bytes length %"PRIu32" \n",  chunk_header->image_id, chunk_header->image_chunk.start_offset, chunk_header->image_chunk.size_in_bytes);

	uint32_t image_id = (chunk_header->image_id);//*2;//TODO: hack to fix odd numbered image. DONE: Buffer overwriting incorrect index!

	image_aggregation_info_t *image = &(image_agg->images[image_id]);//(image_agg->images[chunk_header->image_id]);
	//image_aggregation_info_t *image2 = &(image_agg->images[image_id+1]);//(image_agg->images[chunk_header->image_id]);

	if(image->usage_status==0 || image->bytes_count==0) {
		image->packets_count=0;
		image->usage_status = 1;
		clock_gettime(CLOCK_MONOTONIC, &image->first_packet_time);
	}
	//printf("Image %"PRIu32" pointer image %d %p status %d bytes_count %d packets count %d bitmask %d size of struct %ld\n", chunk_header->image_id, image_id,image,(int)image->usage_status,(int)image->bytes_count,image->packets_count, image->usage_status, sizeof(image_batched_aggregation_info_t));
	//printf("Image %"PRIu32" pointer image %d %p status %d bytes_count %d packets count %d bitmask %d size of struct %ld\n", chunk_header->image_id, image_id+1,image2,(int)image2->usage_status,(int)image2->bytes_count,image2->packets_count, image2->usage_status, sizeof(image_batched_aggregation_info_t));

	if(1 == image->usage_status) {

		//first put the rte_mbuf address in the proper place and update the packet counter
		image->image_packets[image->packets_count] = pkt;

		//now put the chunk in right place
		image->image_info.image_id = chunk_header->image_id;
		image->image_info.copy_info[image->packets_count].src_cpy_ptr = (void *)((char *)chunk_header+sizeof(image_chunk_header_t));
		image->image_info.copy_info[image->packets_count].image_chunk = chunk_header->image_chunk;

		//now put the number of bytes
		image->bytes_count += chunk_header->image_chunk.size_in_bytes;

		//printf("packets count %d  ID %d start offset %"PRIu32" bytes length %"PRIu32" bytes_count %ld\n", image->packets_count, chunk_header->image_id, chunk_header->image_chunk.start_offset, chunk_header->image_chunk.size_in_bytes, image->bytes_count);
		//printf("Image chunk contents image id start offset %"PRIu32" and size is %"PRIu32"\n",image->copy_info.copy_info[image->packets_count].image_chunk.start_offset,image->copy_info.copy_info[image->packets_count].image_chunk.size_in_bytes);
		//printf("A value from packet %f \n",((float*)(image->copy_info.copy_info[image->packets_count].src_cpy_ptr))[0]);

		image->packets_count += 1;

		//if we have a right amount of bytes for an image, we should make it a ready image and then update the readymask
		if((image->packets_count == MAX_CHUNKS_PER_IMAGE)||(image->bytes_count >= SIZE_OF_AN_IMAGE_BYTES)) {
			image->usage_status = 2;
			//printf("Image %d is complete \n", image->image_info.image_id);
			SET_BIT(image_agg->ready_mask,(image_id+1));
			SET_BIT(ready_images, (image_id+1));
			clock_gettime(CLOCK_MONOTONIC, &image->last_packet_time);
			//++(*ready_images_count);
			//image_agg->ready_mask |= (1 << image_id);
			//image_agg->temp_mask |= (1<<image_id);
			//printf("Image mask : %"PRIu32"\n",image_agg->ready_mask);
			//image_id++;
			//image_id = (image_id%MAX_IMAGES_BATCH_SIZE);
		}
		//if(ready_images_index) (*ready_images_index) |= ready_images;
		if(ready_images) {*ready_images_index=ready_images;}
#ifdef HOLD_PACKETS_TILL_CALLBACK
		return 1;
#else
		return 0; //1;	//when 0 disable calbback release
#endif
	}
	//onvm_nflib_return_pkt(nf_info, pkt);
	//return ready_images;
	return 0;
}

/* the function to load and execute in GPU */
int load_data_to_gpu_and_execute(struct onvm_nf_info *nf_info,image_batched_aggregation_info_t * batch_agg_info, ml_framework_operations_t *ml_operations, cudaHostFn_t callback_function, uint32_t new_images) {
	int ret = 0;
	//prepare callback arguments
	//first find the callback
	//batch_agg_info->temp_mask = 0;
	//check if GPU is available
	/*
	 static int how_many_times_called = 0;

	 how_many_times_called++;

	 printf("This function is called %d many times --------\n",how_many_times_called);
	 */
	__attribute__((unused)) static uint32_t last_processed_index = 0; //Note: need to use this to avoid starvation and not able to touch higher indexed imamges, when always overshooting.
//	__attribute__((unused)) static onvm_interval_timer_t start_tsc = 0;
//	__attribute__((unused)) static onvm_interval_timer_t end_tsc = 0;
//	__attribute__((unused)) static uint64_t busy_interval_tsc = 0;

	stream_tracker *cuda_stream = give_stream_v2();//give_stream();
	if(cuda_stream != NULL) {

		uint32_t i;
		struct gpu_callback * callback_args = NULL;
		callback_args = &cuda_stream->callback_info;
#if 0  //This code is for explicit callback mode
		for(i =0; i<MAX_STREAMS*PARALLEL_EXECUTION; i++) {
			if(gpu_callbacks[i].status == 0) {
				gpu_callbacks[i].status = 1;
				callback_args = &(gpu_callbacks[i]);
				break;
			}
		}
		if(unlikely(NULL==callback_args)) {
			printf("Failed to get callback args\n");
			return_stream(cuda_stream);
			return 2;
		}
#endif

		//Check if images were remaining last time; then pick them.
		if(unlikely(0 == last_processed_index)) {last_processed_index = new_images;}
		//for Freshness set (to avoid stale images comment below line
		if(unlikely(nf_info->fixed_batch_size)) last_processed_index|=new_images;

		uint32_t num_of_images = __builtin_popcount(last_processed_index);//(last_processed_index)?(__builtin_popcount(last_processed_index)):(__builtin_popcount(new_images));
		//num_of_images = (nf_info->fixed_batch_size)? ((num_of_images>nf_info->fixed_batch_size)?(nf_info->fixed_batch_size):(num_of_images)):(num_of_images);
		if(unlikely(nf_info->fixed_batch_size)) {
			if(num_of_images > nf_info->fixed_batch_size) num_of_images = nf_info->fixed_batch_size;
			else {
				return_stream(cuda_stream);
				return 0;
			}
		}
		/* Should Adaptive batching be learning or not? */
		else if (ADAPTIVE_BATCHING_SELF_LEARNING == nf_info->enable_adaptive_batching) {
			// Check and cap to max batch size that is learnt and determined to not exceed SLO for the current operating settings
			if((nf_info->learned_max_batch_size) && (num_of_images > nf_info->learned_max_batch_size)) num_of_images = nf_info->learned_max_batch_size;
			//if((nf_info->learned_max_batch_size) && (num_of_images > nf_info->learned_max_batch_size)) num_of_images = nf_info->learned_max_batch_size;
		}

		uint32_t temp_bitmask = last_processed_index;//(last_processed_index)?(last_processed_index):(new_images);
		//uint32_t num_of_images = __builtin_popcount(new_images);

		void *in_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		void *out_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		void *in_cpu_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		void *out_cpu_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		uint32_t actual_images_in_batch = 0;
		uint32_t actual_images_in_batch_bitmask=0;
		void * input_dev_buffer = NULL;
		void * output_dev_buffer = NULL;
		void * cpu_side_buffer = NULL;
		void * cpu_side_output = NULL;

		for(i=0; i< num_of_images; i++) {
			//now get the GPU buffer for each image
			give_device_addresses(cuda_stream->id, &input_dev_buffer, &output_dev_buffer);
			if(NULL == input_dev_buffer || NULL == output_dev_buffer) break;
			//last_processed_index=0;
			int index = ffs(temp_bitmask);
			CLEAR_BIT(temp_bitmask, index);
			SET_BIT(actual_images_in_batch_bitmask, index);
			//SET_BIT(last_processed_index, index);
			actual_images_in_batch++;
			in_buffers[i] = input_dev_buffer;
			out_buffers[i] = output_dev_buffer;

			// for CPU buffer case
			give_cpu_addresses(cuda_stream->id,&cpu_side_buffer, &cpu_side_output);
			in_cpu_buffers[i] = cpu_side_buffer;
			out_cpu_buffers[i] = cpu_side_output;

		}
		//printf("number of images ready %d, can_be_processed=%d index of image %d \n", num_of_images, actual_images_in_batch, ffs(new_images)-1);

		//prepare execution arguments
		callback_args->bitmask_images= actual_images_in_batch_bitmask;//actual_images_in_batch;//new_images;
		callback_args->batch_aggregation = batch_agg_info;
		callback_args->stream_track = cuda_stream;
		callback_args->nf_info = nf_info;
		clock_gettime(CLOCK_MONOTONIC, &(callback_args->start_time));

		void * start_dev_buffer = in_buffers[0];
		void * start_output_buffer = out_buffers[0];

		for(i = 0; i< actual_images_in_batch; i++) { //for(i = 0; i<num_of_images; i++) {
			//find which image is ready
			int image_index = ffs(actual_images_in_batch_bitmask);// ffs(new_images);
			image_index -= 1;
			//printf("images ready %d index %d \n",num_of_images, image_index);
			if(batch_agg_info->images[image_index].usage_status == 2) {

				//now get the GPU buffer for each image
				//give_device_addresses(cuda_stream->id, &input_dev_buffer, &output_dev_buffer);

				// for CPU buffer case
				//give_cpu_addresses(cuda_stream->id,&cpu_side_buffer, &cpu_side_output);

				cpu_side_buffer = in_cpu_buffers[i];
				cpu_side_output = out_cpu_buffers[i];

				if(start_dev_buffer != NULL) {
					//we have GPU buffer available
					//printf("The image we are looking at is %d \n",image_index);

					//change the status
					batch_agg_info->images[image_index].usage_status = 3;

					#define ENABLE_GPU_NETML
#ifdef ENABLE_GPU_NETML
					//NetML transfer
					transfer_to_gpu((void *)(batch_agg_info->images[image_index].image_info.copy_info),batch_agg_info->images[image_index].packets_count,in_buffers[i],&(cuda_stream->stream));
#else
					//Copy Transfer
					transfer_to_gpu_copy((void *)(batch_agg_info->images[image_index].image_info.copy_info),batch_agg_info->images[image_index].packets_count,cpu_side_buffer,in_buffers[i],&(cuda_stream->stream));
#endif

					CLEAR_BIT(actual_images_in_batch_bitmask, (image_index+1));	//CLEAR_BIT(new_images, (image_index+1));
					CLEAR_BIT(batch_agg_info->ready_mask, (image_index+1));
					CLEAR_BIT(last_processed_index, (image_index+1));

					//printf("After posting image ready mask %"PRIu32",final_batch size %d \n", batch_agg_info->ready_mask, final_batch_size);
				}
				else
				{
					//printf("we could not get the GPU buffer\n");
					break;
				}
			}
		}

		//time to execute the image
		//prepare execution arguments;

		//we have GPU available

		nflib_ml_fw_infer_params_t infer_params;
		infer_params.batch_size = actual_images_in_batch;//__builtin_popcount(callback_args->bitmask_images);
		//printf("%d,",infer_params.batch_size);
		//printf("Batch size: %d Stream ID %"PRIu8" image mask %x\n",infer_params.batch_size, cuda_stream->id, callback_args->bitmask_images);
		infer_params.callback_data = callback_args;
		infer_params.callback_function = callback_function;
		infer_params.stream = &(cuda_stream->stream);
		infer_params.model_handle = nf_info->ml_model_handle;
		infer_params.input_data = start_dev_buffer;
		infer_params.input_size = SIZE_OF_AN_IMAGE_BYTES*infer_params.batch_size;
		infer_params.output = start_output_buffer;


		//struct timespec time_image_ready;
		//clock_gettime(CLOCK_MONOTONIC, &time_image_ready);
		//uint64_t image_ready_time = (time_image_ready.tv_sec)*1000000+(time_image_ready.tv_nsec)/1000;
		//printf("Image ready at time %"PRIu64"\n",image_ready_time);

		//conduct the inference.
		void * aio = NULL;
		//NO work in tensorrt now

		//struct timespec begin_infer, end_infer;
		//clock_gettime(CLOCK_MONOTONIC, &begin_infer);
		ml_operations->infer_batch_fptr(&infer_params,aio );
		cudaEventRecord(cuda_stream->event,cuda_stream->stream);
		//clock_gettime(CLOCK_MONOTONIC, &end_infer);

		//uint64_t infer_time = (end_infer.tv_sec-begin_infer.tv_sec)*1000000+(end_infer.tv_nsec-begin_infer.tv_nsec)/1000;
		//printf("Inference launch time (us) %"PRIu64" \n", infer_time);
	} else {
		//printf("GPU IS BUSY\n");
		return 1;// indicates busy
	}
	return ret;
}

/*
 void infer_the_image(uint32_t batch_size, void *callback_data, cudaHostFn_t callback_function,cudaStream_t *stream, void *model_handle, void* input_data, size_t input_size, float *output){

 nflib_ml_fw_infer_params_t infer_params;
 infer_params.batch_size = __builtin_popcount(callback_batch_info);
 infer_params.callback_data = callback_args;
 infer_params.callback_function = callback_function;
 infer_params.stream = &(cuda_stream->stream);
 infer_params.model_handle = nf_info->ml_model_handle;
 infer_params.input_data = start_dev_buffer;
 infer_params.input_size = SIZE_OF_AN_IMAGE_BYTES*infer_params.batch_size;
 infer_params.output = start_output_buffer;

 //conduct the inference.
 void * aio = NULL;
 //NO work in tensorrt now

 ml_operations->infer_batch_fptr(&infer_params,aio );

 return;
 }
 */

#endif
