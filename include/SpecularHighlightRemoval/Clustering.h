#ifndef GPUCLUSTERING_H
#define GPUCLUSTERING_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\count.h>

void GPULoadTempImages(int size);
void GPURunClustering(float *minimumChromaticityImage, float *maximumChromaticityImage, int *clusterImage, float *minCenters, float *maxCenters, int minClusterIndex, 
	int maxClusterIndex, int maxMinClusterIndex, size_t minPitch, size_t maxPitch, size_t clusterPitch, int rows, int cols);

#endif