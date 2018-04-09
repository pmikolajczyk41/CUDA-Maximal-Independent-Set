#include "serial.h"
#include <cstdio>
#include <ctime>
#include <chrono>

void serial(int N, int*& Adj) {
	printf("\n\n\n");
	printf("------------------------------------------------\n");
	printf("----------------SERIAL ALGORITHM----------------\n");

	auto start = std::chrono::high_resolution_clock::now();

	int* MIS = new int[N]();	//result

	//do the job in O(N^2)
	for (int i = 0; i < N; i++) {
		bool adjacent = false;		//if it belongs to N(MIS)

		for (int j = 0; j < i; j++) 
			if (MIS[j] && Adj[j * N + i]) {
				adjacent = true;
				break;
			}
	
		if (adjacent) continue;
		MIS[i] = 1;		//next to be added
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = finish - start;

	//show results
	printf("result:\n");
	for (int i = 0; i < N; i++)
		printf("%d ", MIS[i]);
	printf("\nfound in %.6f s\n", duration/1000);

	delete[] MIS;
}