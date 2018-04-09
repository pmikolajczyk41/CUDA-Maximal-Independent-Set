#include <cstdio>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <curand_kernel.h>
#include <ctime>
#include <chrono>
#include "kernels.h"
#include "graph_gen.h"
#include "auxiliary.h"
#include "serial.h"

//dimensions for CUDA
const int blockSize = 1024;

//size of graph, multiple of blockSize
const int N = 8192;
//number of blocks while N processors
const int ceilN = (N + blockSize - 1) / blockSize;
//number of blocks while N^2 processors
const int ceilNN = (N * N + blockSize - 1) / blockSize;

//logging option
#define LOGGING 0

//algorithm starting call and checking function
void solve(int*& MIS, int*& MISDev, int *&AdjDev, int *&CurrentGraphDev);
void checkMIS(int*& MIS, int*& Adj);


/*---------+------------+--------*/
/*         | INFTERFACE |        */
/*---------+------------+--------*/

int main() {
	//GPU info
	showoff();

	//OMP threads
	omp_set_num_threads(8);

	//time measurement
	auto start = std::chrono::high_resolution_clock::now();

	// host memory 
	int *MIS, *Adj;					//minimal independent set, matrix of adjacence
	initializeHost(N, MIS, Adj);	//allocate host memory and generate graph
	generateGraph(N, Adj);			//call graph creating method
	
	// device memory
	int* MISDev, *AdjDev, *CurrentGraph;
	initializeDevice(N, MIS, Adj, MISDev, AdjDev, CurrentGraph);

	//do the job, result should be in MIS, no memory was freed
	solve(MIS, MISDev, AdjDev, CurrentGraph);

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = finish - start;

	//print and check result
	output(N, MIS, duration);
	checkMIS(MISDev, AdjDev);

	//compare to serial algorithm
	serial(N, Adj);

	// free both host and device memory
	freeMemory(MISDev, AdjDev, CurrentGraph, MIS, Adj);
	return 0;
}


/*---------+----------------+---------*/
/*         | MAIN ALGORITHM |         */
/*---------+----------------+---------*/

int countLeft(int*& CurrentGraph);
void heavyFind(int currentCardinality, int*& CurrentGraph, int*& Adj, int*& Degrees, int*& WithHeavySubset, int*& HeavySet, int& heavySetCardinality);
void scoreFind(int*& Adj, int*& WithHeavySubset, int*& HeavySet, int heavySetCardinality, int*& ScoreFind, unsigned int*& RandomChoice, curandGenerator_t& rand_gen);

//core algorithm
void solve(int*& MIS, int *& MISDev, int *&Adj, int *&CurrentGraph) {
	printf("\n\n");
	printf("------------------------------------------------\n");
	printf("---------------PARALLEL ALGORITHM---------------\n\n");

	setOnes<<<ceilN, blockSize>>>(CurrentGraph);
	
	/*	Initially, MISDev (I set in article) = {0}^N (nothing has been chosen yet),
	CurrentGraphDev (H set in article) = {1}^N (we start from full graph) */

	//random generating
	unsigned int* RandomChoice;
	curandGenerator_t rand_gen;
	checkCurandError(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT), __LINE__);
	checkCurandError(curandSetPseudoRandomGeneratorSeed(rand_gen, 1234ULL), __LINE__);

							/* create main device arrays */
	int* WithHeavySubset;	//K set in the article	-	got from HeavyFind
	int* HeavySet;			//M set in the article	-	got from HeavyFind
	int* ScoreSet;			//T set in the article	-	got from ScoreFind
	int* IndSet;			//S set in the article	-	got from IndFind (modified T)
	int* Degrees;			//store current degrees of vertices in K

	initializeDeviceSupport(N, ceilN, WithHeavySubset, HeavySet, ScoreSet, Degrees, RandomChoice);

	//main loop should go until we are left with such a small graph, that brutal algorithm ought to be launched
	int currentCardinality = N;
	while (currentCardinality) {		//(O(lg^2 N)
		//set 0s
		checkError(cudaMemset(HeavySet, 0, N * sizeof(int)), __LINE__);
		checkError(cudaMemset(ScoreSet, 0, N * sizeof(int)), __LINE__);
		checkError(cudaMemset(Degrees, 0, N * sizeof(int)), __LINE__);

		//find heavy set
		int heavySetCardinality;
		heavyFind(currentCardinality, CurrentGraph, Adj, Degrees, WithHeavySubset, HeavySet, heavySetCardinality);		//O(log^2 N)

		//find best of heavy subsets
		scoreFind(Adj, WithHeavySubset, HeavySet, heavySetCardinality, ScoreSet, RandomChoice, rand_gen);		//O(log N)
		
		//find independent and update
		checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);
		IndSet = ScoreSet;
		indFind << < ceilNN, blockSize >> > (N, IndSet, Adj, RandomChoice);		//O(1)
		updateWithInd << <ceilN, blockSize >> > (N, MISDev, CurrentGraph, IndSet);	//O(1)
		updateWithNeighs << <ceilNN, blockSize >> > (N, CurrentGraph, Adj, IndSet);	//O(1)
		cudaDeviceSynchronize();

		//count remaining vertices
		int newCardinality = countLeft(CurrentGraph);
		if (newCardinality == currentCardinality) break;
		currentCardinality = newCardinality;
		if (LOGGING) printf("%d vertices left\n", currentCardinality);
	}

	//brutal finish
	if (currentCardinality && LOGGING) printf("\n\n...BRUTAL FINISH...\n");
	int* BrutalChosen = ScoreSet;	//steal memory
	while (currentCardinality) {
		checkError(cudaMemcpy(BrutalChosen, CurrentGraph, N * sizeof(int), cudaMemcpyDeviceToDevice), __LINE__);
		checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);
		indFind << < ceilNN, blockSize >> > (N, BrutalChosen, Adj, RandomChoice);	//O(1)
		updateWithInd << <ceilN, blockSize >> > (N, MISDev, CurrentGraph, BrutalChosen);	//O(1)
		updateWithNeighs << <ceilNN, blockSize >> > (N, CurrentGraph, Adj, BrutalChosen);	//O(1)
		cudaDeviceSynchronize();
		currentCardinality = countLeft(CurrentGraph);
		if (LOGGING) printf("%d vertices left\n", currentCardinality);
	}

	//copy computed Independent Set from device to host
	checkError(cudaMemcpy(MIS, MISDev, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

	//free allocated device memory
	freeDeviceSupport(WithHeavySubset, HeavySet, ScoreSet, Degrees, RandomChoice);

	if(LOGGING)	printf("\nFINISHED\n\n");
}

//the result is placed in last two arrays
void heavyFind(int currentCardinality, int*& CurrentGraph, int*& Adj, int*& Degrees, int*& WithHeavySubset, int*& HeavySet, int& heavySetCardinality) {
	//we want to return subset of no less cardinality than target
	int target = (int)(currentCardinality / log2(currentCardinality));
	
	if (LOGGING) {
		printf("\n...HEAVYFIND...\n");
		printf("\ttarget: %d vertices\n", target);
	}

	//work with a copy of H
	checkError(cudaMemcpy(WithHeavySubset, CurrentGraph, N * sizeof(int), cudaMemcpyDeviceToDevice), __LINE__);

	//create new adjacence matrix
	int* CurrentAdj;
	checkError(cudaMalloc((void**)&CurrentAdj, N * N * sizeof(int)), __LINE__);

	//for reducing
	int* BlockSumsHost = new int[ceilNN];
	int* BlockSumsDev;
	checkError(cudaMalloc((void**)&BlockSumsDev, ceilNN * sizeof(int)), __LINE__);

	int step = (int)ceil(log2(currentCardinality));
	while (step) {		//O(lg N)
		step--;
		//reset HeavySet
		checkError(cudaMemset(HeavySet, 0, N * sizeof(int)), __LINE__);

		int* hewi = new int[N]();
		cudaMemcpy(hewi, WithHeavySubset, N * sizeof(int), cudaMemcpyDeviceToHost);

		//actualize adjacence matrix
		correctEdges << <ceilNN, blockSize >> > (N, WithHeavySubset, CurrentAdj, Adj);	//O(1)

		//count degrees
		blockReduction << < ceilNN, blockSize >> > (CurrentAdj, BlockSumsDev);	//O(1)
		checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilNN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		//sum up results from blocks
		int* DegreesHost = new int[N]();

#pragma omp parallel for schedule(static)
		for (int i = 0; i < ceilNN; i++)	//O(lg N)
			DegreesHost[i / ceilN] += BlockSumsHost[i];

		checkError(cudaMemcpy(Degrees, DegreesHost, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
		delete[] DegreesHost;

		//mark and count heavy vertices
		markHeavy << <ceilN, blockSize >> > (Degrees, HeavySet, std::max(1,(1 << step) - 1));	//O(1)
		blockReduction << <ceilN, blockSize >> > (HeavySet, BlockSumsDev);			//O(1)
		checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		//check cardinality of HeavySet
		heavySetCardinality = 0;
#pragma omp parallel for shared(ceilN) reduction(+:heavySetCardinality)
		for (int i = 0; i < ceilN; i++)		//O(lg N)
			heavySetCardinality += BlockSumsHost[i];
		
		if (heavySetCardinality >= target) break;
		else {
			removeVertices << <ceilN, blockSize >> > (WithHeavySubset, HeavySet);	//O(1)
			cudaDeviceSynchronize();
		}
	}
	if (LOGGING) {
		int* WithHeavySubsetHost = new int[N];
		cudaMemcpy(WithHeavySubsetHost, WithHeavySubset, N * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\tWithHeavySet found:\n\t");
		for (int i = 0; i < N; i++)
			printf("%d ", (bool)WithHeavySubsetHost[i]);
		printf("\n\n");
		delete[]WithHeavySubsetHost;

		int* HeavySetHost = new int[N]();
		cudaMemcpy(HeavySetHost, HeavySet, N * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\tHeavySet found:\n\t");
		for (int i = 0; i < N; i++)
			printf("%d ", (bool)HeavySetHost[i]);
		printf("\n\n");
		delete[]HeavySetHost;
	}
	cudaFree(CurrentAdj);
	cudaFree(BlockSumsDev);
	delete[] BlockSumsHost;
}

//the result is placed in last parameter
void scoreFind(int*& Adj, int*& WithHeavySubset, int*& HeavySet, int heavySetCardinality, int*& ScoreSet, unsigned int*& RandomChoice, curandGenerator_t& rand_gen) {
	//target
	int cardinality = std::max(2, heavySetCardinality >> 5);		//point maximising function described in the article

	//keep best
	int bestScore = -INT_MAX;
	int* BestSet = new int[N]();	//allocated on host

	if (LOGGING) {
		printf("\n...SCOREFIND...\n");
		printf("\ttarget: %d vertices\n", cardinality);
	}

	//current computing
	int currentScore;
	int* CurrentSet;	//allocated on device, in case of improving score, it would be copied to host
	int* CurrentSetHost = new int[N];	//for reducing
	checkError(cudaMalloc((void**)&CurrentSet, N * sizeof(int)), __LINE__);

	//auxiliary array for counting score
	int* ScoreArray;
	checkError(cudaMalloc((void**)&ScoreArray, N * N * sizeof(int)), __LINE__);

	//for reducing
	int* BlockSumsHost = new int[ceilNN];
	int* BlockSumsDev;
	checkError(cudaMalloc((void**)&BlockSumsDev, ceilNN * sizeof(int)), __LINE__);

	for (int i = 0; i < 10; i++) {
		//reset
		checkError(cudaMemset(CurrentSet, 0, N * sizeof(int)), __LINE__);
		checkError(cudaMemset(ScoreArray, 0, N * N * sizeof(int)), __LINE__);
		checkError(cudaMemset(BlockSumsDev, 0, ceilNN * sizeof(int)), __LINE__);
		currentScore = 0;

		//generate random indicators
		checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);

		//generate current candidate
		randChoice << <ceilN, blockSize >> > (CurrentSet, RandomChoice, HeavySet);
		checkError(cudaMemcpy(CurrentSetHost, CurrentSet, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		//count its score - we use formula: |N_(K-T) (T)| - |E(T)| + |T|
		prepareCountingScore << <ceilNN, blockSize >> > (N, ScoreArray, Adj, CurrentSet, WithHeavySubset); //O(1)
		blockReduction << <ceilNN, blockSize >> > (ScoreArray, BlockSumsDev);	//O(1)
		checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilNN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

#pragma omp parallel for schedule(static)
		for (int i = 0; i < ceilNN; i++)
			if (!CurrentSetHost[i / blockSize] && BlockSumsHost[i]) BlockSumsHost[i] = 1;

#pragma omp parallel for schedule(static) reduction(+:currentScore)
		for (int i = 0; i < ceilNN; i++)
			currentScore += BlockSumsHost[i];

		if (currentScore > bestScore) {		//take best result
			bestScore = currentScore;
			checkError(cudaMemcpy(BestSet, CurrentSet, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);
		}
	}

	//show off the result
	if (LOGGING) {
		printf("\tbest found:\n\t");
		for (int i = 0; i < N; i++)
			printf("%d ", (bool)BestSet[i]);
		printf("\n\n");
	}

	checkError(cudaMemcpy(ScoreSet, BestSet, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);

	//free memory
	delete[] BestSet;
	delete[] BlockSumsHost;
	delete[] CurrentSetHost;
	cudaFree(CurrentSet);
	cudaFree(ScoreArray);
	cudaFree(BlockSumsDev);
}

//compare obtained MIS to Adj
void checkMIS(int*& MIS, int*& Adj) {
	printf("checking corectness of result... ");

	//communication with kernel
	int* flagHost = new int();
	int* flagDev;
	checkError(cudaMalloc((void**)&flagDev, sizeof(int)), __LINE__);
	checkError(cudaMemset(flagDev, 0, sizeof(int)), __LINE__);

	int* Marked;
	checkError(cudaMalloc((void**)&Marked, N * sizeof(int)), __LINE__);
	checkError(cudaMemset(Marked, 0, N * sizeof(int)), __LINE__);

	checkEdgesAndMarkNeighs << <ceilNN, blockSize >> > (N, MIS, Adj, Marked, flagDev);
	checkIfMarked << <ceilN, blockSize >> > (Marked, flagDev);
	checkError(cudaMemcpy(flagHost, flagDev, sizeof(int), cudaMemcpyDeviceToHost), __LINE__);
	
	if (*flagHost) printf("INCORRECT MIS!\n\n");
	else printf("CORRECT MIS!\n\n");

	cudaFree(Marked);
	cudaFree(flagDev);
	delete flagHost;
}

//counts vertices in Graph
int countLeft(int*& CurrentGraph) {
	int result = 0;
	int* BlockSumsDev;
	checkError(cudaMalloc((void**)&BlockSumsDev, ceilN * sizeof(int)), __LINE__);
	int* BlockSumsHost = new int[ceilN];

	blockReduction << <ceilN, blockSize >> > (CurrentGraph, BlockSumsDev);
	checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

#pragma omp parallel for schedule(static) reduction(+:result)
	for (int i = 0; i < ceilN; i++)
		result += BlockSumsHost[i];

	cudaFree(BlockSumsDev);
	delete[] BlockSumsHost;

	return result;
}