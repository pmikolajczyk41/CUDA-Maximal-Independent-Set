#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"

//sets '1' value for every element
__global__ void setOnes(int* Array);

//prints given arrays
__global__ void checker(int N, int* MIS, int* Adj, int* CurrentGraph);

//gets subGraph and its adjacence matrix and modifies it into independent set - N^2 processors, O(1)
__global__ void indFind(int N, int* SubGraph, int* Adj, unsigned int* Random);

//updates previous MIS/current Graph by adding/erasing Independent Set - N processors, O(1)
__global__ void updateWithInd(int N, int* MIS, int* Graph, int* IndSet);

//updates current Graph by erasing neighbourhood of IndSet - N^2 processors, O(1)
__global__ void updateWithNeighs(int N, int* Graph, int* Adj, int* IndSet);

//reduces summing along rows by factor 1024  - N^2 or N processors, O(1)
__global__ void blockReduction(int* ArrayGlob, int* BlockSums);

//builds adjacence matrix in NewAdj with respect to subGraph - N^2 processors, O(1)
__global__ void correctEdges(int N, int* SubGraph, int* NewAdj, int* Adj);

//marks in HeavySet which vertices have degree no lower than lowerbound - N processors, O(1)
__global__ void markHeavy(int* Degrees, int* HeavySet, int lowerbound);

//substracts from WithHeavySubset HeavySet - N processors, O(1)
__global__ void removeVertices (int* WithHeavySubset, int* HeavySet);

//checks if any edge connects two vertices from MIS - N^2 processors, O(1)
__global__ void checkEdgesAndMarkNeighs(int N, int* MIS, int* Adj, int* Marked, int* flag);

//checks if every vertex is from either MIS or from N(MIS) - N processors, O(1)
__global__ void checkIfMarked(int* Marked, int* flag);

//marks in expectation 1/32 vertices from HeavySet - N processors, O(1)
__global__ void randChoice (int* Subset, unsigned int* Choice, int* HeavySet);

//prepares array for reduction summing - N^2 processors, O(1)
__global__ void prepareCountingScore(int N, int* ScoreArray, int* Adj, int* CurrentSet, int* WithHeavySubset);
