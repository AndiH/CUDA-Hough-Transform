# # ROOTLIBS := $(shell root-config --libs)
# # ROOTCFLAGS := $(shell root-config --cflags)
# # ROOTLIBS += lMinuit # Not included by default
# CFLAGS = $(shell root-config --cflags)
# LIBS   = $(shell root-config --libs)
# GLIBS  = $(shell root-config --glibs)
# GLIBS  += -lMinuit

PATHTOROOT = /home/herten/fairsoft/extpkg/tools/root
PATHTOCUSP = /private/herten/

ROOTLIBS := -m64 -I$(PATHTOROOT)/include -L$(PATHTOROOT)/lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -L$(PATHTOROOT)/lib -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -lMinuit

OBJECTS = houghtransform.o AhTwoArraysToMatrix.o AhTranslatorFunction.o 

all:	hough

houghtransform.o:	houghtransform.cu
	nvcc -arch=sm_20 $(ROOTLIBS) -I$(PATHTOCUSP) -c houghtransform.cu

AhTwoArraysToMatrix.o:	AhTwoArraysToMatrix.cu AhTwoArraysToMatrix.h AhTranslatorFunction.h
	nvcc -arch=sm_20 $(ROOTLIBS) -I$(PATHTOCUSP) -c AhTwoArraysToMatrix.cu

%.o:	%.cu %.h 
	nvcc -arch=sm_20 $(ROOTLIBS) -I$(PATHTOCUSP) -c $<

hough:	$(OBJECTS)
	nvcc -arch=sm_20 $(ROOTLIBS) -I$(PATHTOCUSP) -o $@ $(OBJECTS) 