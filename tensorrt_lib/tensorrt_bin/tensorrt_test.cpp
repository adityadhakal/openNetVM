#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include "buffers.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include "cudaWrapper.h"
#include "ioHelper.h"
//#include "ioHelper.cpp"


#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using namespace cudawrapper;


// important file wide variables
ICudaEngine* engine{nullptr};
Logger gLogger;
// Declaring execution context.
unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};

/*--------------------------- end logging ----------------------- */
// Number of times we run inference to calculate average time.
constexpr int ITERATIONS = 10;
/*
class Logger : public ILogger
{
  void log(Severity severity, const char* msg) override
  {
    // suppress info-level messages
    if (severity != Severity::kINFO)
      std::cout << msg << std::endl;
  }
} gLogger;
*/


// Returns empty string iff can't read the file
string readBuffer(string const& path)
{
  string buffer;
  ifstream stream(path.c_str(), ios::binary);

  if (stream)
    {
      stream >> noskipws;
      copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

  return buffer;
}
static int getBindingInputIndex(IExecutionContext* context)
{
  return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

void launchInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
  int inputId = getBindingInputIndex(context);

  cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  //cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
  context->enqueue(batchSize, bindings, stream, nullptr);
  cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
  //cudaMemcpy(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
}


void doInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
  CudaEvent start;
  CudaEvent end;
  double totalTime = 0.0;
  struct timespec begin,finish;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  int temp_batchsize = 1;
  for (int i = 0; i < ITERATIONS; ++i)
    {
      float elapsedTime;

      // Measure time it takes to copy input to GPU, run inference and move output back to CPU.
      cudaEventRecord(start, stream);
      //launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize);
      launchInference(context, stream, inputTensor, outputTensor, bindings, temp_batchsize);
      cudaEventRecord(end, stream);

      // Wait until the work is finished.
      cudaStreamSynchronize(stream);
      cudaEventElapsedTime(&elapsedTime, start, end);

      totalTime += elapsedTime;
    }
  clock_gettime(CLOCK_MONOTONIC, &finish);

  double time_elapsed = (finish.tv_sec-begin.tv_sec)*1000.0+(finish.tv_nsec-begin.tv_nsec)/1000000.0;

  cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << endl;
  std::cout<<" Inference of batch of "<< temp_batchsize << " number of iterations "<<ITERATIONS <<" total images "<< temp_batchsize*ITERATIONS<<" Time "<<time_elapsed<<" ms"<<" Throughput "<<(temp_batchsize*ITERATIONS)*1000/time_elapsed<<endl;
}


int init(const char * file_path)
{
  //  string buffer = readBuffer("my_engine.trt");
  string buffer = readBuffer(file_path);
  std::cout<<"Buffer size "<<buffer.size()<<std::endl;

  if (buffer.size())
    {
      // try to deserialize engine
      unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
      engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    }

  /* //not necessary as NF supplies them
  vector<float> inputTensor;
  vector<float> outputTensor;

  CudaStream stream;
  */
  void* bindings[2]{0};
  int batchSize = 8;

  // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
  assert(engine->getNbBindings() == 2);
  assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

  for (int i = 0; i < engine->getNbBindings(); ++i)
    {
      Dims dims{engine->getBindingDimensions(i)};
      size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
      // Create CUDA buffer for Tensor.
      cudaMalloc(&bindings[i], size * sizeof(float));

      // Resize CPU buffers to fit Tensor.
      if (engine->bindingIsInput(i))
	inputTensor.resize(size);
      else
	outputTensor.resize(size);
    }

  std::cout<<"The input tensor size is "<<inputTensor.size()<<" and the output tensor size is "<<outputTensor.size()<<std::endl;

  srand(0);
  
  for(int i = 0; i<inputTensor.size(); i++){
    inputTensor[i] = ((rand()%255)/255.0);
  }
  
  // Create Execution Context.
  context.reset(engine->createExecutionContext());
  doInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);
 
}
