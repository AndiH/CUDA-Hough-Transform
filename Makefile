NVCCOPTIONS := --gpu-architecture sm_20

#NVCCOPTIONS += -DUSE_DOUBLES=1 # uncomment if you'd like to run with double precision

ROOTCFLAGS := -I$(shell root-config --incdir)

ROOTLIBS := -L$(shell root-config --libdir)
ROOTLIBS += -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -lMinuit -lGui

PATHTOCUSP = /private/herten/cusplibrary

OBJECTS = houghtransform.o AhTwoArraysToMatrix.o AhTranslatorFunction.o AhHoughTransformation.o

all: hough

houghtransform.o: houghtransform.cu
	nvcc $(NVCCOPTIONS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -o $@ -c $<

%.o: %.cu %.h 
	nvcc $(NVCCOPTIONS) $(ROOTCFLAGS) -I$(PATHTOCUSP) -dlink -o $@ -c $<

hough: $(OBJECTS)
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) -o $@ $?

one: Makefile
	nvcc $(NVCCOPTIONS) $(ROOTLIBS) $(ROOTCFLAGS) -I$(PATHTOCUSP)  -o $@ houghtransform.cu AhHoughTransformation.cu AhTwoArraysToMatrix.cu AhTranslatorFunction.cu

clean: Makefile
	rm -f *.o *~
