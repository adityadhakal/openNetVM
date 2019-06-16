CPPFLAGS+=-I/home/adhak001/microsoft/cntk/Source/CNTKv2LibraryDll/API/
CPPFLAGS+=-I/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/
LDFLAGS= -L/home/adhak001/microsoft/cntk/build/release/lib/ -Wl,-rpath-link,/home/adhak001/microsoft/cntk/build/release/lib/ -Wl,-rpath,/home/adhak001/microsoft/cntk/build/release/lib/
LDLIBS= -lCntk.Core-2.6 -lCntk.Math-2.6 -lCntk.Eval-2.6 -lcudart -lcuda
CXX=g++
CXXFLAGS=-std=c++11 -O3 -g -shared -fPIC
#CXXFLAGS=-std=c++11 -O3 -g 
#libraries and flags for CUDA
CPPFLAGS+=-I/usr/local/cuda/include 
LDFLAGS+=-L/usr/local/cuda/lib64 

#all the source files

TARGET=libonvm_cntk.so

SOURCES=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)
OBJECTS=$(SOURCES: .cpp=.o)

all:$(TARGET)

$(TARGET):$(OBJECTS)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDLIBS) 

#$(TARGET):$(OBJECTS)
#	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) -c -o $(TARGET) $(OBJECTS) $(LDLIBS) 

clean:
	rm -rf *.so *.o

