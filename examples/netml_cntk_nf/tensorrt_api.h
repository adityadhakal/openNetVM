#ifndef _TENSORRT_API_
#define _TENSORRT_API_
#include "/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/onvm_ml_libraries.h"
#ifdef __cplusplus
extern "C"{

#endif

  /* function that will load the model in CPU.. however for Tensorrt, this function call won't do anything */
  int tensorrt_load_model(nflib_ml_fw_load_params_t *load_params, void *aio);
  
  /* the actual function to load the model into GPU */
  int tensorrt_link_model(nflib_ml_fw_link_params_t *load_params, void *aio);

  /* the function to infer the model */
  int tensorrt_infer_batch(nflib_ml_fw_infer_params_t* infer_params, void *aio);

  /* the function to get back the results if needed */
  int tensorrt_get_results(nflib_ml_fw_infer_params_t* infer_params, void *aio);

  /* the deinitialization module */
  int tensorrt_deinit(uint32_t options);


#ifdef __cplusplus
}
#endif
  
#endif
