NVCCOPTIONS := -arch=sm_20


all: one good bad

demoGOOD.o: demo.cu Makefile
	nvcc $(NVCCOPTIONS) -dlink -o $@ -c $<

demoBAD.o: demo.cu Makefile
	nvcc $(NVCCOPTIONS) -o $@ -c $<

%.o: %.cu %.h Makefile
	nvcc $(NVCCOPTIONS) -o $@ -c $<

good: demoGOOD.o AhHoughTransformation.o
	nvcc $(NVCCOPTIONS) -o $@ $?

bad: demoBAD.o AhHoughTransformation.o
	nvcc $(NVCCOPTIONS) -o $@ $?

one: Makefile
	nvcc $(NVCCOPTIONS) -o $@ demo.cu AhHoughTransformation.cu

clean: Makefile
	rm -f *.o *~
