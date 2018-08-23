#include "SpecularHighlightRemoval\SpecularHighlightRemoval.h"

SpecularHighlightRemoval::SpecularHighlightRemoval()
{

	thPercent = 0.5;
	alpha = 3;
	beta = 0.51;
	gamma = 0.025;
	useSort = true;

}

SpecularHighlightRemoval::~SpecularHighlightRemoval()
{

#ifdef REMOVE_SPECULAR_HIGHLIGHT_USING_CUDA
	cudaFree(&deviceMinCenters);
	cudaFree(&deviceMaxCenters);
#else
	delete [] ratio;
#endif

}

void SpecularHighlightRemoval::initialize(int imageRows, int imageCols)
{

#ifdef REMOVE_SPECULAR_HIGHLIGHT_USING_CUDA
	deviceDiffuseImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC3);
	deviceSpecularImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC3);
	deviceMinimumImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceMaximumImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceRangeImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceMaskImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceMinimumChromaticityImage = cv::gpu::GpuMat(imageRows, imageCols, CV_32FC1);
	deviceMaximumChromaticityImage = cv::gpu::GpuMat(imageRows, imageCols, CV_32FC1);
	deviceClusterImage = cv::gpu::GpuMat(imageRows, imageCols, CV_32S);
	deviceRatioImage = cv::gpu::GpuMat(imageRows, imageCols, CV_32FC1);
	GPULoadRatio(imageRows * imageCols);
	GPULoadTempImages(imageRows * imageCols);
	cudaMalloc(&deviceMinCenters, 3 * sizeof(float));
	cudaMalloc(&deviceMaxCenters, 3 * sizeof(float));
#else
	minimumImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	minimumChromaticityImage = cv::Mat::zeros(imageRows, imageCols, CV_32FC1);
	maximumChromaticityImage = cv::Mat::zeros(imageRows, imageCols, CV_32FC1);
	isLabelledImage = cv::Mat::zeros(imageRows, imageCols, CV_32FC1);
	clusterImage = cv::Mat(imageRows, imageCols, CV_32S);
	ratio = (float*)malloc(imageRows * imageCols * sizeof(float));
	maximumImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	rangeImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	maskImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	ratioImage = cv::Mat(imageRows, imageCols, CV_32FC1);
#endif
}

/*
Pipeline to remove specular highlights in real time
1. Compute minimum, maximum and range values for each pixel 
2. Compute the mean of minimum values
3. Estimate the pseudo-chromaticity values
4. Cluster regions in the minimum-maximum pseudo-chromaticity space
5. Estimate the single intensity ratio per cluster
6. Separate specular from diffuse components
*/

int compare(const void *a, const void *b) 
{ 
	return (*(float *)a > *(float *)b) ? 1 : ((*(float *)a == *(float *)b) ? 0 : -1);
}

cv::Mat SpecularHighlightRemoval::run(cv::Mat image)
{

	diffuseImage = image.clone();
	specularImage = image.clone();
	
#ifdef REMOVE_SPECULAR_HIGHLIGHT_USING_CUDA

	deviceOriginalImage = cv::gpu::GpuMat(image);
	GPUComputeMinMaxRange(deviceOriginalImage.ptr(), deviceMinimumImage.ptr(), deviceMaximumImage.ptr(), deviceRangeImage.ptr(), deviceOriginalImage.step, 
		deviceMinimumImage.step, deviceMaximumImage.step, deviceRangeImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
	cv::gpu::meanStdDev(deviceMinimumImage, minimumMean, stdDevMean);
	GPUComputePseudoChromaticity(deviceOriginalImage.ptr(), deviceMinimumImage.ptr(), deviceMinimumChromaticityImage.ptr<float>(), 
		deviceMaximumChromaticityImage.ptr<float>(), deviceMaskImage.ptr(), deviceOriginalImage.step, deviceMinimumImage.step, deviceMinimumChromaticityImage.step, 
		deviceMaximumChromaticityImage.step, deviceMaskImage.step, minimumMean(0), deviceOriginalImage.rows, deviceOriginalImage.cols);
	cv::gpu::minMaxLoc(deviceMinimumChromaticityImage, &minimumValue, &maximumMinimumValue, &minimumLocation, &maximumMinimumLocation, deviceMaskImage);
	cv::gpu::minMaxLoc(deviceMaximumChromaticityImage, 0, &maximumValue, 0, &maximumLocation, deviceMaskImage);
	GPURunClustering(deviceMinimumChromaticityImage.ptr<float>(), deviceMaximumChromaticityImage.ptr<float>(), deviceClusterImage.ptr<int>(), 
		deviceMinCenters, deviceMaxCenters, minimumLocation.y * deviceOriginalImage.cols + minimumLocation.x, maximumLocation.y * deviceOriginalImage.cols + 
		maximumLocation.x,  maximumMinimumLocation.y * deviceOriginalImage.cols + maximumMinimumLocation.x, deviceMinimumChromaticityImage.step, 
		deviceMaximumChromaticityImage.step, deviceClusterImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
	GPUComputeIntensityRatio(deviceClusterImage.ptr<int>(), deviceRangeImage.ptr(), deviceMaximumImage.ptr(), deviceRatioImage.ptr<float>(), deviceClusterImage.step,
		deviceRangeImage.step, deviceMaximumImage.step, deviceRatioImage.step, minimumMean(0), thPercent, 3, useSort, alpha, beta, gamma, deviceOriginalImage.rows, 
		deviceOriginalImage.cols);
	GPUSeparateComponents(deviceOriginalImage.ptr(), deviceSpecularImage.ptr(), deviceDiffuseImage.ptr(), deviceMaximumImage.ptr(), deviceRangeImage.ptr(), 
		deviceMaskImage.ptr(), deviceRatioImage.ptr<float>(), deviceOriginalImage.step, deviceSpecularImage.step, deviceDiffuseImage.step, deviceMaximumImage.step, 
		deviceRangeImage.step, deviceMaskImage.step, deviceRatioImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
	deviceSpecularImage.download(specularImage);
	deviceDiffuseImage.download(diffuseImage);
	return diffuseImage;
#else
	
	for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		int red = image.ptr<unsigned char>()[pixel * 3 + 2];
		int green = image.ptr<unsigned char>()[pixel * 3 + 1];
		int blue = image.ptr<unsigned char>()[pixel * 3 + 0];
		minimumImage.ptr<unsigned char>()[pixel] = std::min(red, std::min(green, blue));
		maximumImage.ptr<unsigned char>()[pixel] = std::max(red, std::max(green, blue));
		rangeImage.ptr<unsigned char>()[pixel] = maximumImage.ptr<unsigned char>()[pixel] - minimumImage.ptr<unsigned char>()[pixel];
	}
	
	minimumMean = cv::mean(minimumImage);
	
	for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		maskImage.ptr<unsigned char>()[pixel] = (minimumImage.ptr<unsigned char>()[pixel] > minimumMean(0)) ? 1 : 0;
		if(maskImage.ptr<unsigned char>()[pixel] == 1) {
			float redChromaticity = (int)image.ptr<unsigned char>()[pixel * 3 + 2] - (int)minimumImage.ptr<unsigned char>()[pixel] + (float)minimumMean(0);
			float greenChromaticity = (int)image.ptr<unsigned char>()[pixel * 3 + 1] - (int)minimumImage.ptr<unsigned char>()[pixel] + (float)minimumMean(0);
			float blueChromaticity = (int)image.ptr<unsigned char>()[pixel * 3 + 0] - (int)minimumImage.ptr<unsigned char>()[pixel] + (float)minimumMean(0);
			float sum = redChromaticity + greenChromaticity + blueChromaticity;
			redChromaticity /= sum;
			greenChromaticity /= sum;
			blueChromaticity /= sum;
			minimumChromaticityImage.ptr<float>()[pixel] = std::min(redChromaticity, std::min(greenChromaticity, blueChromaticity));
			maximumChromaticityImage.ptr<float>()[pixel] = std::max(redChromaticity, std::max(greenChromaticity, blueChromaticity));
		} 
	}

	cv::minMaxLoc(minimumChromaticityImage, &minimumValue, &maximumMinimumValue, &minimumLocation, &maximumMinimumLocation, maskImage);
	cv::minMaxLoc(maximumChromaticityImage, 0, &maximumValue, 0, &maximumLocation, maskImage);
	for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		if(maskImage.ptr<unsigned char>()[pixel] == 1) {
			float minimumChromaticity = minimumChromaticityImage.ptr<float>()[pixel];
			float maximumChromaticity = maximumChromaticityImage.ptr<float>()[pixel];
			float dist1 = estimateDistance(minimumChromaticity, maximumChromaticity, 
				(float)minimumChromaticityImage.ptr<float>()[minimumLocation.y * image.cols + minimumLocation.x], 
				(float)maximumChromaticityImage.ptr<float>()[minimumLocation.y * image.cols + minimumLocation.x]);
			float dist2 = estimateDistance(minimumChromaticity, maximumChromaticity, 
				(float)minimumChromaticityImage.ptr<float>()[maximumLocation.y * image.cols + maximumLocation.x], 
				(float)maximumChromaticityImage.ptr<float>()[maximumLocation.y * image.cols + maximumLocation.x]);
			float dist3 = estimateDistance(minimumChromaticity, maximumChromaticity, 
				(float)minimumChromaticityImage.ptr<float>()[maximumMinimumLocation.y * image.cols + maximumMinimumLocation.x], 
				(float)maximumChromaticityImage.ptr<float>()[maximumMinimumLocation.y * image.cols + maximumMinimumLocation.x]);
			if(dist1 <= dist2 && dist1 <= dist3) clusterImage.ptr<int>()[pixel] = 1;
			else if(dist2 < dist1 && dist2 < dist3) clusterImage.ptr<int>()[pixel] = 2;
			else clusterImage.ptr<int>()[pixel] = 3;			
		}	
	}
			
	for(int cluster = 1; cluster <= 3; cluster++) {
		int count = 0;
		minCenters[cluster - 1] = 0;
		maxCenters[cluster - 1] = 0;
		for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
			if(clusterImage.ptr<int>()[pixel] == cluster) {
				minCenters[cluster - 1] += minimumChromaticityImage.ptr<float>()[pixel];
				maxCenters[cluster - 1] += maximumChromaticityImage.ptr<float>()[pixel];
				count++;
			}
		}
		minCenters[cluster - 1] /= count;
		maxCenters[cluster - 1] /= count;
	}
			
	for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		if(maskImage.ptr<unsigned char>()[pixel] == 1) {
			float minimumChromaticity = minimumChromaticityImage.ptr<float>()[pixel];
			float maximumChromaticity = maximumChromaticityImage.ptr<float>()[pixel];
			float dist1 = estimateDistance(minimumChromaticity, maximumChromaticity, (float)minCenters[0], (float)maxCenters[0]);
			float dist2 = estimateDistance(minimumChromaticity, maximumChromaticity, (float)minCenters[1], (float)maxCenters[1]);
			float dist3 = estimateDistance(minimumChromaticity, maximumChromaticity, (float)minCenters[2], (float)maxCenters[2]);
			if(dist1 <= dist2 && dist1 <= dist3) clusterImage.ptr<int>()[pixel] = 1;
			else if(dist2 < dist1 && dist2 < dist3) clusterImage.ptr<int>()[pixel] = 2;
			else clusterImage.ptr<int>()[pixel] = 3;
		}
	}
	
	int k = 3;
	for(int cluster = 1; cluster <= k; cluster++) {

		float estimatedRatio = 0;
		int index = 0;
			
		if(useSort) {

			for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
				if(clusterImage.ptr<int>()[pixel] == cluster && rangeImage.ptr<unsigned char>()[pixel] > minimumMean(0)) {
					ratio[index] = (float)(int)maximumImage.ptr<unsigned char>()[pixel] / ((float)(int)rangeImage.ptr<unsigned char>()[pixel] + 1e-10);
					index++;
				}
			}
			qsort(ratio, index, sizeof(float), compare);
			estimatedRatio = ratio[round(index * thPercent)];
			
		} else {
			
			float sumValue = 0;
			for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
				if(clusterImage.ptr<int>()[pixel] == cluster && rangeImage.ptr<unsigned char>()[pixel] > minimumMean(0)) {
					ratio[index] = (float)(int)maximumImage.ptr<unsigned char>()[pixel] / ((float)(int)rangeImage.ptr<unsigned char>()[pixel] + 1e-10);
					sumValue += ratio[index];
					index++;
				}
			}
			
			estimatedRatio = sumValue/(float)index;
			for(int iteration = 0; iteration < alpha; iteration++) {
					
				int lessIndex = 0;
				int greaterIndex = 0;
				
				for(int idx = 0; idx < index; idx++) {
					if(ratio[idx] > estimatedRatio) greaterIndex++;
					else lessIndex++;
				}

				if((float)lessIndex/(float)index > beta) estimatedRatio -= (estimatedRatio * gamma);
				else if((float)greaterIndex/(float)index > beta) estimatedRatio += (estimatedRatio * gamma);
				else break;
					
			}
				
		}

		for(int pixel = 0; pixel < image.rows * image.cols; pixel++) 
			if(clusterImage.ptr<int>()[pixel] == cluster) 
				ratioImage.ptr<float>()[pixel] = estimatedRatio;

	}

	specularImage.setTo(0);
	for(int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		if(maskImage.ptr<unsigned char>()[pixel] == 1) {
			int value = round((int)maximumImage.ptr<unsigned char>()[pixel] - (float)ratioImage.ptr<float>()[pixel] * (int)rangeImage.ptr<unsigned char>()[pixel]);
			for(int ch = 0; ch < 3; ch++) {
				specularImage.ptr<unsigned char>()[pixel * 3 + ch] = std::max(value, 0);
				diffuseImage.ptr<unsigned char>()[pixel * 3 + ch] = std::min(std::max(image.ptr<unsigned char>()[pixel * 3 + ch] - specularImage.ptr<unsigned char>()[pixel * 3 + ch], 0), 255);
			}
		}
	}
	return diffuseImage;
#endif

}

float SpecularHighlightRemoval::estimateDistance(float x1, float y1, float x2, float y2) 
{
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

int SpecularHighlightRemoval::round(int x) {
	return (((x)>0)? (int)((float)(x)+0.5):(int)((float)(x)-0.5));
}
