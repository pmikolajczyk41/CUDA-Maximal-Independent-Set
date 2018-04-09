#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include "auxiliary.h"

/*	GPU info  */
void showoff() {
	printf("GPU:\n");
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}


/*	functions for interface	*/

void initializeHost(int N, int *& MIS, int *& Adj) {
	MIS = new int[N]();
	Adj = new int[N*N]();
}

void initializeDevice(int N, int*& MIS, int*& Adj, int*& MISDev, int*& AdjDev, int*&CurrentGraph) {
	//allocate device memory
	checkError(cudaMalloc((void**)&MISDev, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&AdjDev, N * N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&CurrentGraph, N * sizeof(int)), __LINE__);

	//copy matrix from host
	checkError(cudaMemcpy(AdjDev, Adj, N * N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);

	//set initial values for MISDev
	checkError(cudaMemset(MISDev, 0, N * sizeof(int)), __LINE__);
}

void freeMemory(int*& MISDev, int*& AdjDev, int*& CurrentGraphDev, int*& MIS, int*& Adj) {
	cudaFree(MISDev);
	cudaFree(AdjDev);
	cudaFree(CurrentGraphDev);
	delete[] MIS;
	delete[] Adj;
}

void output(int N, int*& MIS, std::chrono::duration<double, std::milli>& duration) {
	printf("result:\n");
	for (int i = 0; i < N; i++) 
		printf("%d ", (bool)MIS[i]);
	printf("\nfound in %.6f s\n\n", duration/1000);
}


/*	functions for algorithm	*/

void initializeDeviceSupport(int N, int ceilN, int*& WithHeavySubset, int*& HeavySet, int*& ScoreSet, int*& Degrees, unsigned int*& RandomChoice) {
	//allocate memory
	checkError(cudaMalloc((void**)&WithHeavySubset, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&HeavySet, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&ScoreSet, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&Degrees, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&RandomChoice, ceilN * sizeof(int)), __LINE__);
}

void freeDeviceSupport(int*& WithHeavySubset, int*& HeavySet, int*& ScoreSet, int*& Degrees, unsigned int*& RandomChoice) {
	cudaFree(WithHeavySubset);
	cudaFree(HeavySet);
	cudaFree(ScoreSet);
	cudaFree(Degrees);
	cudaFree(RandomChoice);
}