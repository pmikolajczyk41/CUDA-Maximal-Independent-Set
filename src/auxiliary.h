#pragma once

#include <curand.h>
#include <chrono>

//CUDA error handling
inline void checkError(cudaError_t error, int line) {
	if (error != cudaSuccess) {
		printf("CUDA error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, line);
		exit(EXIT_FAILURE);
	}
}

//CURAND error handling
inline void checkCurandError(curandStatus_t status, int line) {
	if (status != CURAND_STATUS_SUCCESS) {
		printf("CURAND error(code %d), line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

//info
void showoff();

//interface auxiliary methods
void initializeHost(int, int*&, int*&);
void initializeDevice(int, int*&, int*&, int*&, int*&, int*&);
void freeMemory(int*&, int*&, int*&, int*&, int*&);
void output(int, int*&, std::chrono::duration<double, std::milli>&);


//algorithm auxiliary methods
void initializeDeviceSupport(int, int, int*&, int*&, int*&, int*&, unsigned int*&);
void freeDeviceSupport(int*&, int*&, int*&, int*&, unsigned int*&);