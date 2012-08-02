# # ROOTLIBS := $(shell root-config --libs)
# # ROOTCFLAGS := $(shell root-config --cflags)
# # ROOTLIBS += lMinuit # Not included by default
# CFLAGS = $(shell root-config --cflags)
# LIBS   = $(shell root-config --libs)
# GLIBS  = $(shell root-config --glibs)
# GLIBS  += -lMinuit

PATHTOROOT = /home/herten/fairsoft/extpkg/tools/root

ROOTLIBS := -m64 -I$(PATHTOROOT)/include -L$(PATHTOROOT)/lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -L$(PATHTOROOT)/lib -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -lMinuit

all:	hough

hough:	houghtransform.cu
		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $< 