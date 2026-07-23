#include<stdio.h>
#include<gsl/gsl_sf.h>


// see https://root-forum.cern.ch/t/problem-on-using-gsl-on-root/31225
// run on MacOS:
//   root [0] gSystem->Load("/opt/local/lib/libgsl.dylib");
//   root [1] .x testGsl.C++
// or
//   root [0] gSystem->AddLinkedLibs("-lgslcblas -lgsl");
//   root [1] .x testGsl.C++
// Python: find and load GSL library
//   libGslPath = str(ctypes.util.find_library("libgsl"))
//   ROOT.gSystem.Load(libGslPath)
void testGsl()
{
	double x[5] = {1, 2, 3, 4, 5};
	for (size_t i = 0; i < 5; ++i) {
		printf("J0(%.3f) = %.3f\n", x[i], gsl_sf_bessel_J0(x[i]));
	}
}
