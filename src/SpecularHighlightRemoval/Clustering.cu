#include "SpecularHighlightRemoval\Clustering.h"

thrust::device_vector<float> thrustMinImage;
thrust::device_vector<float> thrustMaxImage;
__constant__ float constantMinCenters[3];
__constant__ float constantMaxCenters[3];

struct is_not_zero
{
    __host__ __device__
    bool operator()(const float x)
    {
      return (x != 0);
    }
};

__device__ float computeDistance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void loadClusters(float *minImage, float *maxImage, float *minCenters, float *maxCenters, int minClusterIndex, int maxClusterIndex, int maxMinClusterIndex, 
	size_t minPitch, size_t maxPitch, int rows, int cols) {

	int xMinCluster = minClusterIndex % cols;
	int yMinCluster = minClusterIndex / cols;
	int xMaxCluster = maxClusterIndex % cols;
	int yMaxCluster = maxClusterIndex / cols;
	int xMaxMinCluster = maxMinClusterIndex % cols;
	int yMaxMinCluster = maxMinClusterIndex / cols;
	
	float *minImageClusterRow1 = (float*)((char*)minImage + yMinCluster * minPitch);
	float *maxImageClusterRow1 = (float*)((char*)maxImage + yMinCluster * maxPitch);
	float *minImageClusterRow2 = (float*)((char*)minImage + yMaxCluster * minPitch);
	float *maxImageClusterRow2 = (float*)((char*)maxImage + yMaxCluster * maxPitch);
	float *minImageClusterRow3 = (float*)((char*)minImage + yMaxMinCluster * minPitch);
	float *maxImageClusterRow3 = (float*)((char*)maxImage + yMaxMinCluster * maxPitch);
		
	minCenters[0] = minImageClusterRow1[xMinCluster];
	minCenters[1] = minImageClusterRow2[xMaxCluster];
	minCenters[2] = minImageClusterRow3[xMaxMinCluster];
	
	maxCenters[0] = maxImageClusterRow1[xMinCluster];
	maxCenters[1] = maxImageClusterRow2[xMaxCluster];
	maxCenters[2] = maxImageClusterRow3[xMaxMinCluster];

}

__global__ void assignClusters(float *minImage, float *maxImage, int *clusterImage, int minClusterIndex, int maxClusterIndex, int maxMinClusterIndex, size_t minPitch, 
	size_t maxPitch, size_t clusterPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= cols || y >= rows) return;
	
	float *minImageRow = (float*)((char*)minImage + y * minPitch);
	float *maxImageRow = (float*)((char*)maxImage + y * maxPitch);
	int *clusterImageRow = (int*)((char*)clusterImage + y * clusterPitch);
	
	float minChannel = minImageRow[x];
	float maxChannel = maxImageRow[x];
	if(minChannel > 0 && maxChannel > 0) {
		float minDist = rows * cols;
		for(int c = 0; c < 3; c++) {
			float dist = computeDistance(minChannel, maxChannel, constantMinCenters[c], constantMaxCenters[c]);
			if(dist < minDist) {
				minDist = dist;
				clusterImageRow[x] = c + 1;
			}
		}
	} else clusterImageRow[x] = 0;

}

__global__ void copyToThrust(float *image, float *thrustImage, int *maskImage, int clusterIndex, size_t clusterPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= cols || y >= rows) return;
	int *maskImageRow = (int*)((char*)maskImage + y * clusterPitch);
	float *imageRow = (float*)((char*)image + y * clusterPitch);
	if(maskImageRow[x] == clusterIndex) thrustImage[y * cols + x] = imageRow[x];
	else thrustImage[y * cols + x] = 0;

}

void GPUCheckError2(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

int divUp2(int a, int b) { 
    return (a + b - 1)/b;
}

void GPULoadTempImages(int size) {
	thrustMinImage.resize(size);
	thrustMaxImage.resize(size);
}

void GPURunClustering(float *minimumChromaticityImage, float *maximumChromaticityImage, int *clusterImage, float *minCenters, float *maxCenters, int minClusterIndex, 
	int maxClusterIndex, int maxMinClusterIndex, size_t minPitch, size_t maxPitch, size_t clusterPitch, int rows, int cols) {

	float hostMinCenters[3], hostMaxCenters[3];
	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	loadClusters<<<1, 1>>>(minimumChromaticityImage, maximumChromaticityImage, minCenters, maxCenters, minClusterIndex, maxClusterIndex, maxMinClusterIndex, minPitch, maxPitch, rows, cols);
	cudaMemcpyToSymbol(constantMinCenters, minCenters, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(constantMaxCenters, maxCenters, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
	assignClusters<<<grid, threads>>>(minimumChromaticityImage, maximumChromaticityImage, clusterImage, minClusterIndex, maxClusterIndex, maxMinClusterIndex, minPitch, maxPitch, clusterPitch, rows, cols);
	for(int cluster = 1; cluster <= 3; cluster++) {
		copyToThrust<<<grid, threads>>>(minimumChromaticityImage, thrust::raw_pointer_cast(thrustMinImage.data()), clusterImage, cluster, clusterPitch, rows, cols);
		copyToThrust<<<grid, threads>>>(maximumChromaticityImage, thrust::raw_pointer_cast(thrustMaxImage.data()), clusterImage, cluster, clusterPitch, rows, cols);
		float size = thrust::count_if(thrustMinImage.begin(), thrustMinImage.end(), is_not_zero());
		float minSum = thrust::reduce(thrustMinImage.begin(), thrustMinImage.end());
		float maxSum = thrust::reduce(thrustMaxImage.begin(), thrustMaxImage.end());
		hostMinCenters[cluster - 1] = minSum/size;
		hostMaxCenters[cluster - 1] = maxSum/size;
	}
	cudaMemcpyToSymbol(constantMinCenters, hostMinCenters, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constantMaxCenters, hostMaxCenters, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	assignClusters<<<grid, threads>>>(minimumChromaticityImage, maximumChromaticityImage, clusterImage, minClusterIndex, maxClusterIndex, maxMinClusterIndex, minPitch, maxPitch, clusterPitch, rows, cols);
	GPUCheckError2("GPURunClustering");
	
}