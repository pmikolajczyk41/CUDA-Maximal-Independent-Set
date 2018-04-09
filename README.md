# CUDA-Maximal-Independent-Set
CUDA implementation of a parallel algorithm for the Maximal Independent Set problem (by Karp &amp; Wigderson)

## Overall outline
The primary aim of this project was to implement an algorithm publicated by M. Karp and A. Wigderson (available at: http://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/KW85/KW85.pdf - but also included in repo). It was one of the pioneer proposition for parallel solution of Maximal Independent Set problem. Although there exists a quite simple sequential algorithm solving it in O(n) time (greedy approach), it is believed that it cannot be simply parallelized (this process would be related to the completeness in P class with respect to logspace reducibility).

This implementation follows strong recommendations from authors and uses randomness in one of the phases. Hence, the time complexity is O(log<sub>2</sub><sup>3</sup>n). Memory complexity (of this implementation) is O(n<sup>2</sup>).

## Implementation details
In spite of the fact that theoretically this algorithm should overtake sequential one, it is totally not practical. It harnesses powerful mathematical tools that requires much more work from processors than it is truly needed. Moreover, to simplify code and avoid sophisticated maneuvers, graph is represented as a matrix of adjacence, which implies square memory. In some places, decision had to be taken about which approach would be better: permuting rows and columns in matrix or maybe leaving matrix stable, but paying for it in threads running without much work to do. As it was rather educational project than a practical one, the second option was chosen.

Most of the computations are performed on GPU with CUDA kernels. CPU serves mainly as a host platform for calling kernels and organizating workflow, but still there are some snippets of code carrying out some computational work (often using OpenMP).

The sequential algorithm is also provided, just to compare effectiveness.

## Remarks
In the sources there is a simple random graph generator with 4 different densities proposed. It can be easily substituted for testing.

The whole project was realised in Visual Studio 2017 on machine with CPU: Intel Core i5-6300HQ with 8GB RAM and Nvidia GTX950M with CUDA Runtime API v9.0. Neither .sln file nor any analisis nor result files are included in this repo. But if you would like to, just write to me and I will be very glad to share with you any resource you will need.

## Requirements
1. CUDA compiler (nvcc)
2. C++ compiler supporting C++11 standard
3. OpenMP library
