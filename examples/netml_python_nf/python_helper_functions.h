#ifndef PYTHON_HELPER_FUNCTIONS_H
#define PYTHON_HELPER_FUNCTIONS_H


/* the functions to help with the loading of python model */
//initialize python.. sets up python environment
void python_initialize(void);


/* load model into python */
int python_load_model(nflib_ml_fw_load_params_t *load_params, void *aio);

/* link to python model */
int python_link_model(nflib_ml_fw_link_params_t *link_params, void *aio);

/* execute the image in python */
int python_execute_model(nflib_ml_fw_infer_params_t *infer_params, void *aio);



#endif
