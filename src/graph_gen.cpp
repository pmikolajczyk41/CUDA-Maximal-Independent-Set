#include <ctime>
#include <cmath>
#include <cstdlib>

void star(int N, int*& Adj) {
	for (int i = 1; i < N; i++)
		Adj[i] = Adj[i*N] = 1;
}

void randomGraph(int N, int*& Adj) {
	srand(time(NULL));
	int density;

	//density = (int)(N * sqrt(N));
	density = (int)(2 * N);
	//density = (int)(N * log2(N));
	//density = (int)(0.2 * N * N);


	for (int i = 0; i < density; ++i) {
		int x = rand() % N;
		int y = rand() % N;
		if (x != y) {
			Adj[x + y * N] = 1;
			Adj[y + x * N] = 1;
		}
	}
}

void generateGraph(int N, int*& Adj) {
	//star(N, Adj);
	randomGraph(N, Adj);
}
