#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <errno.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

PyObject *pModule, *empty_tensor, *myFileName, *pDict, *load_model_to_gpu, *create_gpu_tensor, *execute_image, *cuda_stream, *create_stream;

void python_initialize(){
  //initialize python
  printf("Initializing python \n");
  Py_Initialize();

  /*import few libraries to load the file */
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");

  /* load the file */
  myFileName = PyUnicode_FromString((char *) "python_executable");
  pModule = PyImport_Import(myFileName);


  PyErr_Print();
  /*load the namespace */
  pDict = PyModule_GetDict(pModule);

  /* load all functions */
  load_model_to_gpu = PyDict_GetItemString(pDict, (char *) "load_model_to_gpu");
  create_gpu_tensor = PyDict_GetItemString(pDict, (char *) "create_gpu_tensor");
  execute_image = PyDict_GetItemString(pDict, (char *)"execute_image");
  create_stream = PyDict_GetItemString(pDict, (char *)"create_stream");
}
  

int python_load_model(nflib_ml_fw_load_params_t *load_params, void *aio){
  python_initialize();
  printf("loaded python model \n");
  return 0;
}

/* create a GPU buffer for image as well as load the model to GPU */
int python_link_model(nflib_ml_fw_link_params_t *link_params, void *aio){
  PyObject_CallObject(load_model_to_gpu, NULL);
  //get empty tensor
  empty_tensor = PyObject_CallObject(create_gpu_tensor, NULL);
  long a = PyLong_AsLong(empty_tensor);
  void * empty_buffer = (void *) a;
  //get the stream
  cuda_stream = PyObject_CallObject(create_stream,NULL);
  a = PyLong_AsLong(cuda_stream);
  void *stream_ptr = (void *) a;
  
  //fill in the rest of the NF stuff here
}

int python_execute_model(nflib_ml_fw_infer_params_t *infer_params, void * aio){
  PyObject_CallObject(execute_image, NULL);
}
