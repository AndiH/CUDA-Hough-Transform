# Hough Tracking in Thrust

My stuff for Online Tracking on GPUs.

Instead of invoking plain CUDA, this approach here uses **CUDA Thrust** and **cusp** and for displaying purposes CERN's **ROOT** classes.

Starting `houghtransform.cu`, the following steps are done with a number (6) of sample x y points:

* conformal map (on GPU)
* Hough transform (on GPU)
* histogram calculation (on GPU)

The first two bullet points are done in class `AhHoughTransformation`, the last bullet point (with some super features) is done by the class `AhTwoArraysToMatrix`. A Makefile is provided. See the descriptions below. Especially the last two lines.

I'm using **[Thrust](http://thrust.github.com/)**, **[cusp](https://code.google.com/p/cusp-library/)**, and **[ROOT](http://root.cern.ch/)**.

Let's look into the files and some highlights.

***Note**: The description below refers to a prior version of the code, where the Hough Transformation has not been extracted to a single class. Though this is not correct, the general notes are the same. Will change this description here later.*

## Conformal Map & Hough Transform on GPU — `houghtransform.cu`
Unfortunately not a class yet. But will be in some future time.

The file is quite deeply commented, so read into that for a more detailed understanding. I'm going to highlight some things here.

### Conformal mapping — `struct confMap`
Delivers a operator which is to be used in a `thrust::transform()` call. A `thrust::tuple` of x and y values is transformed by the formula

    x'=x/(x^2+y^2) and y'=-y/(x^2+y^2)

and a `thrust::tuple` of those two values is returned.  
This conformal mapping is essentially the first part of the procedure. From a physics point of view, it maps hits of circular particle trajectories to straight, origin-passing line-trajectories.

### Hough Transformation
General theory concerning Hough transform [can be found in the Wikipedia](http://en.wikipedia.org/wiki/Hough_transform).

The code starts by genearting a vector of angles.
#### Generating vectors with angles — `alphas`
The complete angular space (0° - 360°) is divided into chunks. By default, a chunck spans 30° – but this can easily be overwritten by a command line parameter.  
Thrusts `sequence()` method is used to generate a vector full of angles:

    thrust::sequence(alphas.begin(), alphas.end(), 0., everyXDegrees);
Starting at 0° a point is filled into `alphas`, its value incrementally increased by `everyXDegrees`.
#### Actual Hough Transformation — `struct transf`
Using Thrusts magical `zip_iterator` thingy a `std::vector` full of `thrust::host_vector`s is created. The size of the `std::vector` corresponds to the number of points (6), the size of the `thrust::host_vector` corresponds to the number of investigated `alphas` - being 360/everyXDegrees. Inside this `thrust::host_vector` for every possible angular value a Hough-transformed point corresponding to a x y pair is stored.

Looping over all x y points under investigation (6), the following is done:

* copy all angles onto GPU device (done by simply creating a `thrust::device_vector` of the `thrust::host_vector`)
* create a `thrust::constant_iterator` of the current x y point pair (being more precise: of the conformal map of the current x y point pair)
* use a `thrust::transform` to fill a temp vector full of values - for every to be investigated angle one value
    * the start iterator is a magical `zip_iterator` of a `tuple` of the begin iterator of the vector of the angles and the `constant_iterator` with the x y pairs
    * the end iterator is the same thing, but this time with the end iterator of the angle vector
    * every time the `transf` operator is called (see below) …
    * … and the value is written into a temp `device_vector` …
* which finally is pushed back into the `std::vector`

##### Actual actual Hough Transformation — `struct transf`
This is simple:  
**Input** is a `thrust::tuple` of

1. the angle
2. a `thrust::tuple` of
    1. the conformal mapped x value
    2. the conformal mapped y value
    
The last two being constant, so a `thrust::constant_iterator`.

**Return** is a double of the Hough-transformed points.

This transformation is done in the extracted `__device__` function `houghTransfFunction` and by the calculation of

    cos(angle * 1.7e-02) * x + sin(alpha * 1.7e-02) * y

## Histogram Filling on GPU — `AhTwoArraysToMatrix`
`AhTwoArraysToMatrix` is an extra class which does what it says: It gets two `thrust::host_vector`s as inputs (so, being precise, not actually two simple arrays) and gives back different versions of two dimensional histograms and matrices.

Everything is based on [an example](http://code.google.com/p/cusp-library/source/browse/examples/MatrixAssembly/unordered_triplets.cu) I found in the **cusp** package, like Thrust a template-oriented library – for sparse matrices.

Instead of going through the code in detail (which I might do in the future), here's the general idea of my code:

* you have two index vectors, **I** and **J**
* you have a third vector with weights, **V** (currently unsupported, **V**s elements are always 1)
* the elements of the original data vectors (**x'**, **y'**) are **translated** into coordinates of matrices, therefore becoming the content of the matrix index vectors **I** and **J**
* two `stable_sort_by_key`s are invoked to sort all elements having the same (**I**,**J**) indices behind one another
* it is calculated how many *different* point pairs there are (this determines the space for actual values in the sparse matrix)
* the elements of the weight vector **V** of n equal (**I**,**J**) pairs are summed together by `reduce_by_key` and set as the height (/multiplicity) of a bin at matrix position (**I'**, **J'**), the actual (**I**,**J**) element indices are set as those (**I'**,**J'**) matrix indices
* **Houston**, we have a histogram – in general at least:
    * the range of this matrix is still the *translated* range from 0 to NBinsMax.
    * To fix this, there's a `GetHistogram()` method which makes a **ROOT** `TH2D` from the matrix and sets its axis values in the correct, original way
    
That's it.


Some random special notes (maybe most interesting for you: timings, boundaries, vectors of threetuples of histogram-esk values):

#### Translators — `AhTranslatorFunction`
A separate class which provies two operators to translate and retranslate between floating point values of the two input arrays and the integer space of the matrix which holds the histogram.

* `int operator() (double value)` is used to fill the matrix.  
 **Input**: doubles.  
 **Output**: integers.
* `double operator() (int value)` is used to get an array of original-ranged values from the histogram. **Re**translation.  
 **Input**: integers. (Most of the time the (**I**,**J**) point pairs.)  
 **Output**: doubles.

The operator is overloaded, so be careful which one to choose – though by strictly using the proper data types you should automatically be fine.  
`AhTranslatorFunction` is used in a `thrust::transform()` call in the methods `TranslateValuesToMatrixCoordinates` and `RetranslateValuesFromMatrixCoordinates` – each time chosing the corresponding operator.

The operators need two parameters: `lowValue` which is the lower edge of the data range and `inverseStepWidth`. The last parameter is the **inverse** of

    fXStepWidth = (fXup - fXlow) / fNBinsX
which is essentially the width of one cell in x direction (y respectively). The inverse is chosen to circumvate double precision rounding/casting problems I encountered.

#### Timings — `doTiming`
Constructor flag, by default `doTiming = false`. If set to true, the code runs **ROOT** `TStopwatches` for different parts of the code.

There are five points of the code which deliver pointers to Stopwatches

* `GetSwTranslateValues()`
* `GetSwHistSort()`
* `GetSwHistSum()`
* `GetSwCreateTMatrixD()`
* `GetSwCreateTH2D()`

Be advised: Calling those Get methods without `doTiming = true` will lead to segmentation violations of the class. So: don't.


#### Boundaries – `DoBoundaryCheck()`
If the constructor is invoked with a (not default) `doBoundaryCheck = true` the code checks if all elements of both input vectors are within the histogram boundaries.

If they are, the class procedes just as usual, if they are not, the class exits with a (manual) cout error message.

**Attention:** Boundary check costs time, both GPU and CPU. For every vector a `thrust::min_element()` and a `thrust::max_element()` have to be found – four calls in whole! By default this boundary check is switched off.

#### cusp Matrix to ROOT Matrix — `GetTMatrixD()`
This is done on the CPU side. But maybe, some time, could be ported to the GPU? Although I highly doubt that – but then again, I don't exactly know all the possible ways to fill a TMatrixD.

**Attention**: At this point, I think, I switched cols and rows to match what I'd like to have in the end.

#### ROOT Matrix to ROOT Histogram — `GetHistogram()`
You wondered why I at all want a TMatrixD? Because **ROOT**s TH2D has a constructor for TMatrixD – which at least to some extend is faster then filling every bin of the histogram by `Fill(x,y)`.

This method sets the values on the axes to the original ones (the ones before translating into int matrix space). So this histogram can actually be used to do stuff with.

#### Calculated values without ROOT — `GetRetranslatedMatrixValues()` and `GetPlainMatrixValues()`
By using some clever Thrust calls (which I'm a bit proud of) those two methods provide each a `device_vector` containing three values (contained in a threetuple).

* `GetPlainMatrixValues()`: Tuple of `<int, int, double>`, representing (in that order) the x coordinate in the matrix, the y coordinate in the matrix and the multiplicity of the bin at that position
* `GetRetranslatedMatrixValues()`: Tuple of `<double, double, double>`, representing (in that order) the x coordinate in the double space of the retranslated histogram of original boundaries, the y point with the same properties and the multiplicity of the bin at that position

The last one, `GetRetranslatedMatrixValues()`, is probably the most interesting thingy of my program if you want to work with it afterwards. Wait, I print that again, in bold.

**The last one, `GetRetranslatedMatrixValues()`, is probably the most interesting thingy of my program if you want to work with it afterwards.**


There are still some bugs I'd like to get rid of, some tweaks to do, some features I'd like to add. It's still in development, so stay tuned.