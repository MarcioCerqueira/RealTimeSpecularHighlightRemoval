#include "SpecularHighlightRemoval\ColorProcessing.h"

thrust::device_vector<float> thrustRatio;	
thrust::device_vector<float> thrustCompressedRatio;

__device__ int threadRoundValue(int x) {

	return ((x > 0) ? (int)((float)(x)+0.5) : (int)((float)(x)-0.5));

}

__global__ void computeMinMaxRange(unsigned char *originalImage, unsigned char *minImage, unsigned char *maxImage, unsigned char *rangeImage, size_t originalPitch, 
	size_t minPitch, size_t maxPitch, size_t rangePitch, int rows, int cols) {
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	unsigned char *originalImageRow = originalImage + y * originalPitch;
	unsigned char *minImageRow = minImage + y * minPitch;
	unsigned char *maxImageRow = maxImage + y * maxPitch;
	unsigned char *rangeImageRow = rangeImage + y * rangePitch;

	int red = originalImageRow[x * 3 + 2];
	int green = originalImageRow[x * 3 + 1];
	int blue = originalImageRow[x * 3 + 0];
	int minChannel = min(red, min(green, blue));
	int maxChannel = max(red, max(green, blue));
	minImageRow[x] = minChannel;
	maxImageRow[x] = maxChannel;
	rangeImageRow[x] = maxChannel - minChannel;

}

__global__ void computePseudoChromaticity(unsigned char *originalImage, unsigned char *minImage, float *minChromaticityImage, float *maxChromaticityImage, 
	unsigned char *maskImage, size_t originalPitch, size_t minPitch, size_t minChromaticityPitch, size_t maxChromaticityPitch, size_t maskPitch, float meanMin, int rows, 
	int cols) {
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	unsigned char *originalImageRow = originalImage + y * originalPitch;
	unsigned char *minImageRow = minImage + y * minPitch;
	float *minChromaticityImageRow = (float*)((char*)minChromaticityImage + y * minChromaticityPitch);
	float *maxChromaticityImageRow = (float*)((char*)maxChromaticityImage + y * maxChromaticityPitch);
	unsigned char *maskImageRow = maskImage + y * maskPitch;

	int minChannel = minImageRow[x];
	int mask = (minChannel > meanMin) ? 1 : 0;
	if(mask == 1) {
		float red = (int)originalImageRow[x * 3 + 2] - (float)minChannel + (float)meanMin;
		float green = (int)originalImageRow[x * 3 + 1] - (float)minChannel + (float)meanMin;
		float blue = (int)originalImageRow[x * 3 + 0] - (float)minChannel + (float)meanMin;
		float sum = red + green + blue;
		red /= sum;
		green /= sum;
		blue /= sum;
		minChromaticityImageRow[x] = min(red, min(green, blue));
		maxChromaticityImageRow[x] = max(red, max(green, blue));
	} else {
		minChromaticityImageRow[x] = 0;
		maxChromaticityImageRow[x] = 0;
	}
	maskImageRow[x] = mask;

}

__global__ void computeIntensityRatio(int *clusterImage, unsigned char *rangeImage, unsigned char *maxImage, float *ratio, size_t clusterPitch, size_t rangePitch, 
	size_t maxPitch, float meanMin, int cluster, int rows, int cols) {
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int *clusterImageRow = (int*)((char*)clusterImage + y * clusterPitch);
	unsigned char *rangeImageRow = rangeImage + y * rangePitch;
	unsigned char *maxImageRow = maxImage + y * maxPitch;
	
	int clusterIndex = clusterImageRow[x];
	int range = rangeImageRow[x];
	int maxChannel = maxImageRow[x];

	if(clusterIndex == cluster && range > meanMin) ratio[y * cols + x] = (float)maxChannel/(float)((float)range + 1e-10);
	else ratio[y * cols + x] = 0;

}

__global__ void assignIntensityRatio(int *clusterImage, float *ratioImage, float *compressedRatio, size_t clusterPitch, size_t ratioPitch, int index, int cluster, 
	int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int *clusterImageRow = (int*)((char*)clusterImage + y * clusterPitch);
	float *ratioImageRow = (float*)((char*)ratioImage + y * ratioPitch);
	
	if(clusterImageRow[x] == cluster) ratioImageRow[x] = compressedRatio[index];
	
}

__global__ void assignIntensityRatio(int *clusterImage, float *ratioImage, float intensityRatio, size_t clusterPitch, size_t ratioPitch, int cluster, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int *clusterImageRow = (int*)((char*)clusterImage + y * clusterPitch);
	float *ratioImageRow = (float*)((char*)ratioImage + y * ratioPitch);
	
	if(clusterImageRow[x] == cluster) ratioImageRow[x] = intensityRatio;
	
}

__global__ void separateComponents(unsigned char *originalImage, unsigned char *specularImage, unsigned char *diffuseImage, unsigned char *maximumImage, unsigned char *rangeImage, 
	unsigned char *maskImage, float *ratioImage, size_t originalPitch, size_t specularPitch, size_t diffusePitch, size_t maximumPitch, size_t rangePitch, size_t maskPitch,
	size_t ratioPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	unsigned char *originalImageRow = originalImage + y * originalPitch;
	unsigned char *specularImageRow = specularImage + y * specularPitch;
	unsigned char *diffuseImageRow = diffuseImage + y * diffusePitch;
	unsigned char *maskImageRow = maskImage + y * maskPitch;
	
	if(maskImageRow[x] == 1) {
		unsigned char *maxImageRow = maximumImage + y * maximumPitch;
		unsigned char *rangeImageRow = rangeImage + y * rangePitch;
		float *ratioImageRow = (float*)((char*)ratioImage + y * ratioPitch);
		int value = threadRoundValue((int)maxImageRow[x] - (float)ratioImageRow[x] * (int)rangeImageRow[x]);
		int specularity = max(value, 0);
		specularImageRow[x * 3 + 0] = specularity;
		specularImageRow[x * 3 + 1] = specularity;
		specularImageRow[x * 3 + 2] = specularity;
		diffuseImageRow[x * 3 + 0] = min(max(originalImageRow[x * 3 + 0] - specularity, 0), 255);
		diffuseImageRow[x * 3 + 1] = min(max(originalImageRow[x * 3 + 1] - specularity, 0), 255);
		diffuseImageRow[x * 3 + 2] = min(max(originalImageRow[x * 3 + 2] - specularity, 0), 255);
	} else {
		specularImageRow[x * 3 + 0] = 0;
		specularImageRow[x * 3 + 1] = 0;
		specularImageRow[x * 3 + 2] = 0;
		diffuseImageRow[x * 3 + 0] = originalImageRow[x * 3 + 0];
		diffuseImageRow[x * 3 + 1] = originalImageRow[x * 3 + 1];
		diffuseImageRow[x * 3 + 2] = originalImageRow[x * 3 + 2];
	}

}

struct is_not_zero2
{
    __host__ __device__
    bool operator()(const float x)
    {
      return (x != 0);
    }
};

struct greaterThan
{
	float _value;
	greaterThan(float value) {
		_value = value;
	}

    __host__ __device__
    bool operator()(const float x)
    {
      return (x > _value);
    }
};
int roundValue(int x) {

	return ((x > 0) ? (int)((float)(x)+0.5) : (int)((float)(x)-0.5));

}

int divUp(int a, int b) { 

    return (a + b - 1)/b;

}

void GPUCheckError(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

void GPULoadRatio(int size) {

	thrustRatio.resize(size);
	
}

void GPUComputeMinMaxRange(unsigned char *originalImage, unsigned char *minImage, unsigned char *maxImage, unsigned char *rangeImage, size_t originalPitch, 
	size_t minPitch, size_t maxPitch, size_t rangePitch, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	computeMinMaxRange<<<grid, threads>>>(originalImage, minImage, maxImage, rangeImage, originalPitch, minPitch, maxPitch, rangePitch, rows, cols);
	GPUCheckError("GPUComputeMinMaxRange");

}

void GPUComputePseudoChromaticity(unsigned char *originalImage, unsigned char *minImage, float *minChromaticityImage, float *maxChromaticityImage, unsigned char *maskImage,
	size_t originalPitch, size_t minPitch, size_t minChromaticityPitch, size_t maxChromaticityPitch, size_t maskPitch, float meanMin, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	computePseudoChromaticity<<<grid, threads>>>(originalImage, minImage, minChromaticityImage, maxChromaticityImage, maskImage, originalPitch, minPitch, 
		minChromaticityPitch, maxChromaticityPitch, maskPitch, meanMin, rows, cols);
	GPUCheckError("GPUComputePseudoChromaticity");

}

void GPUComputeIntensityRatio(int *clusterImage, unsigned char *rangeImage, unsigned char *maxImage, float *ratioImage, size_t clusterPitch, size_t rangePitch, 
	size_t maxPitch, size_t ratioPitch, float meanMin, float thPercent, int k, int useSort, int alpha, float beta, float gamma, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	for(int cluster = 1; cluster <= k; cluster++) {
		computeIntensityRatio<<<grid, threads>>>(clusterImage, rangeImage, maxImage, thrust::raw_pointer_cast(thrustRatio.data()), clusterPitch, rangePitch, maxPitch, 
			meanMin, cluster, rows, cols);
		int compressedSize = (rows * cols) - thrust::count(thrustRatio.begin(), thrustRatio.end(), 0);
		if(useSort) {
			thrustCompressedRatio.resize(compressedSize);
			thrust::copy_if(thrustRatio.begin(), thrustRatio.end(), thrustCompressedRatio.begin(), is_not_zero2());
			thrust::sort(thrustCompressedRatio.begin(), thrustCompressedRatio.end());
			assignIntensityRatio<<<grid, threads>>>(clusterImage, ratioImage, thrust::raw_pointer_cast(thrustCompressedRatio.data()), clusterPitch, ratioPitch, 
				roundValue(compressedSize * thPercent), cluster, rows, cols);
		} else {
			float sum = thrust::reduce(thrustRatio.begin(), thrustRatio.end());
			float estimatedRatio = (sum/(float)compressedSize);
			for(int iteration = 0; iteration < alpha; iteration++) {
				int greaterIndex = thrust::count_if(thrustRatio.begin(), thrustRatio.end(), greaterThan(estimatedRatio));
				int lesserIndex = compressedSize - greaterIndex;
				if((float)lesserIndex/(float)compressedSize > beta) estimatedRatio -= (estimatedRatio * gamma);
				else if((float)greaterIndex/(float)compressedSize > beta) estimatedRatio += (estimatedRatio * gamma);
				else break;
			}
			assignIntensityRatio<<<grid, threads>>>(clusterImage, ratioImage, estimatedRatio, clusterPitch, ratioPitch, cluster, rows, cols);
		}
	}
	GPUCheckError("GPUComputeIntensityRatio");

}

void GPUSeparateComponents(unsigned char *originalImage, unsigned char *specularImage, unsigned char *diffuseImage, unsigned char *maximumImage, unsigned char *rangeImage, 
	unsigned char *maskImage, float *ratioImage, size_t originalPitch, size_t specularPitch, size_t diffusePitch, size_t maximumPitch, size_t rangePitch, size_t maskPitch,
	size_t ratioPitch, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	separateComponents<<<grid, threads>>>(originalImage, specularImage, diffuseImage, maximumImage, rangeImage, maskImage, ratioImage, originalPitch, specularPitch, diffusePitch, 
		maximumPitch, rangePitch, maskPitch, ratioPitch, rows, cols);
	GPUCheckError("GPUSeparateComponents");

}