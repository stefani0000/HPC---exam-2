#include "CpuGpuMat.h"
#include <stdlib.h>
#include <assert.h>
#include "KernelGpu.cuh"
#include <cuda_runtime_api.h>                        // cudaDeviceSynchronize()
#include <iostream>
#include <chrono>

using namespace std;
using namespace std:: chrono;


int main() {
    

    struct CpuGpuMat Mat1;
    struct CpuGpuMat Mat2;
    struct CpuGpuMat Mat3;
    int maskSize = 3;

    // matrix1
    Mat1.Rows = 10000; 
    Mat1.Cols = 10000;
    Mat1.Depth = 3;

    // matrix2 mask
    Mat2.Rows = maskSize;
    Mat2.Cols = maskSize;
    Mat2.Depth = 3;

    // matrix3 result
    Mat3.Rows = Mat1.Rows - maskSize + 1;
    Mat3.Cols = Mat1.Cols - maskSize + 1;
    Mat3.Depth = 1;

    Mat1.Size = Mat1.Rows * Mat1.Cols * Mat1.Depth;
    Mat2.Size = Mat2.Rows * Mat2.Cols * Mat2.Depth;
    Mat3.Size = Mat3.Rows * Mat3.Cols * Mat3.Depth;

    // cpu and gpu memory 
    Mat1.cpuP = (void*)malloc(Mat1.Size * sizeof(float));
    Mat2.cpuP = new float[Mat2.Size]{ 
        0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,    
        0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,    
        0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370     
    };
    Mat3.cpuP = (void*)malloc(Mat3.Size * sizeof(float));

    cudaError_t result1 = cudaMalloc(&Mat1.gpuP, Mat1.Size * sizeof(float));
    cudaError_t result2 = cudaMalloc(&Mat3.gpuP, Mat3.Size * sizeof(float));
	
   assert(result1 == cudaSuccess && result2 == cudaSuccess);

    // set values to cpu memory
    float* cpuFloatP = (float*)Mat1.cpuP;
    for (int i = 0; i < Mat1.Size; i++)
        cpuFloatP[i] = (float)i;

    // host -> device
    result1 = cudaMemcpy(Mat1.gpuP, Mat1.cpuP, Mat1.Size * sizeof(float), cudaMemcpyHostToDevice);
    result2 = cudaMemcpy(Mat3.gpuP, Mat3.cpuP, Mat3.Size * sizeof(float), cudaMemcpyHostToDevice);
    assert(result1 == cudaSuccess && result2 == cudaSuccess);

    // parallel conv
    high_resolution_clock::time_point start= high_resolution_clock::now();
    gpuMatrixConvulation3D(&Mat1, (float*)Mat2.cpuP , &Mat3);
    high_resolution_clock::time_point end= high_resolution_clock::now();
	chrono::duration<double>  duration = end - start;
	cout << duration.count()*1000 << endl;
    
    // device -> host
    cudaError_t result = cudaMemcpy(Mat3.cpuP, Mat3.gpuP, Mat3.Size * sizeof(float), cudaMemcpyDeviceToHost);
   // assert(result == cudaSuccess);


	if (result != cudaSuccess) {
    		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));    		
    		exit(EXIT_FAILURE); 
}


    cudaDeviceSynchronize();

    // cpu and gpu memory free
    result1 = cudaFree(Mat1.gpuP);
    result2 = cudaFree(Mat3.gpuP);
    assert(result1 == cudaSuccess && result2 == cudaSuccess);

    free(Mat1.cpuP);
    delete[] Mat2.cpuP;
    free(Mat3.cpuP);

    int x = 1;
    cout<<"\nPress any key and hit enter to end...";
    cin>>x;
    return 0;
}







