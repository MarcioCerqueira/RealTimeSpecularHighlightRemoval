#ifndef SPECULAR_HIGHLIGHT_REMOVAL_H
#define SPECULAR_HIGHLIGHT_REMOVAL_H

#include <opencv2\opencv.hpp>
#include "useCUDA.h"
#ifdef REMOVE_SPECULAR_HIGHLIGHT_USING_CUDA
#include <opencv2\gpu\gpu.hpp>
#include <cuda_runtime_api.h>
#include "ColorProcessing.h"
#include "Clustering.h"
#endif

class SpecularHighlightRemoval
{

public:
	SpecularHighlightRemoval();
	~SpecularHighlightRemoval();
	
	void initialize(int imageRows, int imageCols);
	cv::Mat run(cv::Mat image);
	
	int getNumberOfIterations() { return alpha; }
	float getThreshold() { return beta; }
	float getStepValue() { return gamma; }
	bool isSortEnabled() { return useSort; }

	void setNumberOfIterations(int alpha) { this->alpha = alpha; }
	void setThreshold(float beta) { this->beta = beta; }
	void setStepValue(float gamma) { this->gamma = gamma; }
	void enableSort() { this->useSort = true; }
	void disableSort() { this->useSort = false; }
	
private:
	float estimateDistance(float x1, float y1, float x2, float y2);
	int round(int x);
#ifdef REMOVE_SPECULAR_HIGHLIGHT_USING_CUDA 
	cv::gpu::GpuMat deviceOriginalImage, deviceDiffuseImage, deviceSpecularImage, deviceMinimumImage, deviceMaximumImage, deviceRangeImage, deviceMaskImage;
	cv::gpu::GpuMat deviceMinimumChromaticityImage, deviceMaximumChromaticityImage, deviceClusterImage, deviceRatioImage;
	float *deviceMinCenters, *deviceMaxCenters;
	cv::Scalar stdDevMean;
#else
	cv::Mat minimumImage, maximumImage, rangeImage, maskImage;
	cv::Mat minimumChromaticityImage, maximumChromaticityImage, isLabelledImage, clusterImage, ratioImage;
	float *ratio, minCenters[3], maxCenters[3];	
#endif
	cv::Mat diffuseImage, specularImage;
	cv::Scalar minimumMean;
	double minimumValue, maximumValue, maximumMinimumValue;
	cv::Point minimumLocation, maximumLocation, maximumMinimumLocation;
	float thPercent;
	int alpha;
	float beta;
	float gamma;
	bool useSort;
};

#endif