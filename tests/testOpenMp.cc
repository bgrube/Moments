// on Linux:
// > g++ $(root-config --cflags --libs) -lMathMore -fopenmp -o testOpenMp{,.cc}
// > export OMP_NUM_THREADS=5
// > ./testOpenMp
//
// on MacOS:
// Apple does not ship OpenMP library -> install libomp e.g. via MacPorts or Homebrew
// see https://open-box.readthedocs.io/en/latest/installation/openmp_macos.html
// and https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave
// > sudo port install libomp
// > /usr/bin/clang++ $(root-config --cflags --libs) -lMathMore -Xpreprocessor -fopenmp -I/opt/local/include/libomp -L/opt/local/lib/libomp -lomp -o testOpenMp{,.cc}
// > export OMP_NUM_THREADS=5
// > ./testOpenMp


#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>

#include "Math/SpecFuncMathMore.h"

#include "basisFunctions.C"


int
main(
	int   argc,
	char* argv[]
) {

	// // Fork a team of threads giving them their own copies of variables
	// int nmbThreads, threadId;
	// #pragma omp parallel private(nmbThreads, threadId)
	// {
	// 	// Obtain thread number
	// 	threadId = omp_get_thread_num();
	// 	#pragma omp critical
	// 	{
	// 		std::cout << "Hello World from thread = " << threadId << std::endl;
	// 	}
	// 	// Only master thread does this
	// 	if (threadId == 0) {
	// 		nmbThreads = omp_get_num_threads();
	// 		#pragma omp critical
	// 		{
	// 			std::cout << "Number of threads = " << nmbThreads << std::endl;
	// 		}
	// 	}
	// }
	// // All threads join master thread and disband

	const size_t        nmbEvents = 100000000;
	std::vector<double> theta    (nmbEvents);
	std::vector<double> phi      (nmbEvents);
	std::vector<double> Phi      (nmbEvents);
	std::vector<double> fcnValues(nmbEvents);
	const unsigned int  momentIndex  = 0;
	const unsigned int  L            = 1;
	const unsigned int  M            = 1;
	const double        polarization = 1.0;

	for (size_t nmbThreads = 1; nmbThreads < 256; nmbThreads *= 2)
	{
		omp_set_num_threads(nmbThreads);

		const auto start = std::chrono::high_resolution_clock::now();

		#pragma omp parallel for
		for (size_t i = 0; i < nmbEvents; ++i) {
			fcnValues[i] = f_basis(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
		}

		const auto end = std::chrono::high_resolution_clock::now();
		const std::chrono::duration<double> elapsed = end - start;

		std::cout << "Threads: " << nmbThreads
		          << ", total time: " << elapsed.count() << " seconds"
		          << ", total time * number of threads: " << elapsed.count() * nmbThreads << " seconds"
		          << std::endl;
	}

}
