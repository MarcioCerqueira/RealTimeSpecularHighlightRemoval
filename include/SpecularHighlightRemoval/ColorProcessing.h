#ifndef GPUCOLORPROCESSING_H
#define GPUCOLORPROCESSING_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\count.h>
#include <thrust\sort.h>
#include <thrust\remove.h>

void GPULoadRatio(int size);
void GPUComputeMinMaxRange(unsigned char *originalImage, unsigned char *minImage, unsigned char *maxImage, unsigned char *rangeImage, size_t originalPitch, 
	size_t minPitch, size_t maxPitch, size_t rangePitch, int rows, int cols);
void GPUComputePseudoChromaticity(unsigned char *originalImage, unsigned char *minImage, float *minChromaticityImage, float *maxChromaticityImage, unsigned char *maskImage,
	size_t originalPitch, size_t minPitch, size_t minChromaticityPitch, size_t maxChromaticityPitch, size_t maskPitch, float meanMin, int rows, int cols);
void GPUComputeIntensityRatio(int *clusterImage, unsigned char *rangeImage, unsigned char *maxImage, float *ratioImage, size_t clusterPitch, size_t rangePitch, 
	size_t maxPitch, size_t ratioPitch, float meanMin, float thPercent, int k, int useSort, int alpha, float beta, float gamma, int rows, int cols);
void GPUSeparateComponents(unsigned char *originalImage, unsigned char *specularImage, unsigned char *diffuseImage, unsigned char *maximumImage, unsigned char *rangeImage, 
	unsigned char *maskImage, float *ratioImage, size_t originalPitch, size_t specularPitch, size_t diffusePitch, size_t maximumPitch, size_t rangePitch, size_t maskPitch,
	size_t ratioPitch, int rows, int cols);
#endif