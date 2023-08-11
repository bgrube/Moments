// on Linux:
// > gcc -fopenmp -o testOpenMp{,.c}
// > export OMP_NUM_THREADS=5
// > ./testOpenMp
//
// on MacOS:
// Apple does not ship OpenMP library -> install libomp e.g. via MacPorts or Homebrew
// see https://open-box.readthedocs.io/en/latest/installation/openmp_macos.html
// and https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave
// > sudo port install libomp
// > /usr/bin/clang -Xpreprocessor -fopenmp -I/opt/local/include/libomp -L/opt/local/lib/libomp -lomp -o testOpenMp{,.c}
// > export OMP_NUM_THREADS=5
// > ./testOpenMp


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


int
main(
	int   argc,
	char* argv[]
) {
	// Fork a team of threads giving them their own copies of variables
	int nmbThreads, threadId;
	#pragma omp parallel private(nmbThreads, threadId)
	{
		// Obtain thread number
		threadId = omp_get_thread_num();
		printf("Hello World from thread = %d\n", threadId);
		// Only master thread does this
		if (threadId == 0) {
			nmbThreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nmbThreads);
		}
	}
	// All threads join master thread and disband
}
