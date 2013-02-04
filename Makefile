# # ROOTLIBS := $(shell root-config --libs)
# # ROOTCFLAGS := $(shell root-config --cflags)
# # ROOTLIBS += lMinuit # Not included by default
# CFLAGS = $(shell root-config --cflags)
# LIBS   = $(shell root-config --libs)
# GLIBS  = $(shell root-config --glibs)
# GLIBS  += -lMinuit

ROOTCFLAGS := -I$(shell root-config --incdir)
ROOTCFLAGS += -m64

ROOTLIBS := -L$(shell root-config --libdir)
ROOTLIBS += -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -lMinuit -lGui

# NVCCOPTIONS := -gencode arch=compute_10,code=sm_10 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
NVCCOPTIONS := -gencode arch=compute_35,code=sm_35 -g
# NVCCOPTIONS := -arch=sm_20

PATHTOCUSP = /private/herten/

OBJECTS = houghtransform.o AhTwoArraysToMatrix.o AhTranslatorFunction.o AhHoughTransformation.o


all:	hough

houghtransform.o:	houghtransform.cu
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -dc houghtransform.cu

AhTwoArraysToMatrix.o:	AhTwoArraysToMatrix.cu AhTwoArraysToMatrix.h AhTranslatorFunction.h
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -dc AhTwoArraysToMatrix.cu
AhHoughTransformation.o: AhHoughTransformation.cu AhHoughTransformation.h
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -dc AhHoughTransformation.cu

%.o:	%.cu %.h 
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -dc $<

hough:	$(OBJECTS)
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -o $@ $(OBJECTS) 

clean:
	rm -f *.o *~
