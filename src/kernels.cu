#include <cstdio>
#include "kernels.h"

__global__ void setOnes(int* Array) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	Array[thid] = 1;
}

__global__ void checker(int N, int* MIS, int* Adj, int* CurrentGraph) {
	for (int i = 0; i < N; i++)
		printf("%d ", MIS[i]);
	printf("\n\n");

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
			printf("%d ", Adj[j*N + i]);
		printf("\n");
	}

	printf("\n");
	for (int i = 0; i < N; i++)
		printf("%d ", (bool)CurrentGraph[i]);
	printf("\n\n");
}

__global__ void indFind(int N, int* SubGraph, int* Adj, unsigned int* Random) {
	//each thread handles one edge
	//if both ends belong to SubGraph, eliminate arbitrarily one of them

	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	
	//vertices corresponding to thread
	int u = thid / N;
	int v = thid % N;

	//eliminate half of the threads (for every pair of vertices there are two threads)
	if (v >= u) return;
	
	//if both belong to Subgraph and there is an edge between them
	bool bit = Random[u ^ v] & 1;
	if (Adj[thid] && SubGraph[u] && SubGraph[v]) {	//may not be valid in a moment
		SubGraph[u * bit + v * (1 - bit)] = 0;	//concurrent write, common value
	}
}

__global__ void updateWithInd(int N, int* MIS, int* Graph, int* IndSet) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	bool indSetThid = IndSet[thid];

	//MIS should be union of MIS and IndSet
	MIS[thid] |= indSetThid;

	//Graph should lose IndSet
	Graph[thid] &= !indSetThid;
}

__global__ void updateWithNeighs(int N, int* Graph, int* Adj, int* IndSet) {
	//each thread handles one edge
	//if one end belongs to IndSet, eliminate second from Graph

	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	//vertices corresponding to thread
	int u = thid / N;
	int v = thid % N;

	//eliminate half of the threads (for every pair of vertices there are two threads)
	if (v >= u) return;

	//if one belong to IndSet and there is an edge between them
	if (Adj[thid]) {
		if (IndSet[u]) Graph[v] = 0;		//CRCW common
		else if (IndSet[v]) Graph[u] = 0;	//CRCW common
	}
}

__global__ void blockReduction(int* ArrayGlob, int* BlockSums) {
	__shared__ int shData[1024];
	__shared__ int* Array;	//shifted EdgeCounter

	int thid = threadIdx.x;
	if (thid == 0) Array = ArrayGlob + blockDim.x * blockIdx.x;
	__syncthreads();

	shData[thid] = Array[thid];
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (thid < offset) shData[thid] += shData[thid + offset];
		__syncthreads();
	}

	if (thid == 0) BlockSums[blockIdx.x] = shData[0];
}

__global__ void correctEdges(int N, int* SubGraph, int* NewAdj, int* Adj) {
	//each thread handles one edge
	//if both ends belong to SubGraph, save edge

	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	//vertices corresponding to thread
	int u = thid / N;
	int v = thid % N;

	NewAdj[thid] = (Adj[thid] && SubGraph[u] && SubGraph[v]);
}

__global__ void markHeavy(int* Degrees, int* HeavySet, int lowerbound) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	//check if lowerbound is reached
	HeavySet[thid] = Degrees[thid] >= lowerbound;
}

__global__ void removeVertices(int* WithHeavySubset, int* HeavySet) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	//remove HeavySet
	WithHeavySubset[thid] &= !HeavySet[thid];
}

__global__ void checkEdgesAndMarkNeighs(int N, int* MIS, int* Adj, int* Marked, int* flag) {
	//each thread handles one edge

	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	//vertices corresponding to thread
	int u = thid / N;
	int v = thid % N;

	bool chosenU = MIS[u];
	bool chosenV = MIS[v];

	//if there is an edge between u and v...
	if (Adj[thid]) {
		//... and both ends belong to MIS, signal error
		if (chosenU && chosenV) *flag = 1;
		//... and one of them belongs to MIS, mark both
		else if (chosenU || chosenV) Marked[u] = Marked[v] = 1;
	}
	if (u == v && MIS[u])
		Marked[u] = 1;
}

__global__ void checkIfMarked(int* Marked, int* flag) {
	//each thread checks its own vertex
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	if (!Marked[thid]) *flag = 1;
}

__global__ void randChoice(int* Subset, unsigned int* Choice, int* HeavySet) {
	//each thread marks its own vertex
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	if (HeavySet[thid] && (Choice[thid] % 32 == 0))
		Subset[thid] = 1;
}

__global__ void prepareCountingScore(int N, int* ScoreArray, int* Adj, int* CurrentSet, int* WithHeavySubset) {
	//each thread handles one edge

	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	//vertices corresponding to thread
	int u = thid / N;
	int v = thid % N;
	
	//eliminate half of threads
	if (u > v) return;

	bool chosenU = CurrentSet[u];
	bool chosenV = CurrentSet[v];

	//mark those, who belong to CurrentSet
	if (u == v && chosenU) {
		ScoreArray[thid] = 1;
		return;
	}

	//mark edges inside CurrentSet
	if (chosenU && chosenV) {
		ScoreArray[thid] = -1;
	}

	//mark neighbours of CurrentSet
	if (chosenU || chosenV) {
		ScoreArray[thid] = 1;
	}
}
