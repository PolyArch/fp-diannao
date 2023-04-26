CPP=g++

OPT?=-O3

CFLAGS= --std=c++11 -g -ggdb -gdwarf-3 $(OPT) -fsanitize=address
MODULE          := conv1 conv2 class1 class2

.PHONY: all clean

all: $(MODULE)

HEADERS=dnn.hpp

# These tiling parameters are 100% arbitrary, and it may be advantageous to tune/remove/completely-change them for GPU
conv1: convolution.cpp $(HEADERS)
	$(CPP) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64        -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7

conv2: convolution.cpp $(HEADERS)
	$(CPP) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14   -DKx=3  -DKy=3  -DNi=512  -DNn=512      -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2

class1: classifier.cpp $(HEADERS)
	$(CPP) $^ $(CFLAGS) -o $@ -DNi=25088 -DNn=4096   -DTii=512 -DTi=64     -DTnn=32  -DTn=16

class2: classifier.cpp $(HEADERS)
	$(CPP) $^ $(CFLAGS) -o $@ -DNi=4096 -DNn=1024    -DTii=32 -DTi=32      -DTnn=32  -DTn=16

clean:
	@rm -f $(MODULE) 

