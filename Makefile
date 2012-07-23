# # ROOTLIBS := $(shell root-config --libs)
# # ROOTCFLAGS := $(shell root-config --cflags)
# # ROOTLIBS += lMinuit # Not included by default
# CFLAGS = $(shell root-config --cflags)
# LIBS   = $(shell root-config --libs)
# GLIBS  = $(shell root-config --glibs)
# GLIBS  += -lMinuit

all:	hough

hough:	houghtransform.cu
		nvcc -arch=sm_20 -o $@ $< 
# 		nvcc -arch=sm_20 --compiler-options "$(CFLAGS)" $(LIBS) --compiler-options "$(GLIBS)" -o $@ $< 
#test: lab8btest.cu
#		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $<
#
#lab8b: lab8b.cu
#		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $<
