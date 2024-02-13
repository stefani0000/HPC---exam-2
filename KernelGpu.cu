#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelGpu.cuh"
#include <math.h>
#include <cuda_runtime_api.h>

// constant memory for the mask
__constant__ float constMask[27]; // 3x3x3 mask

__global__ void gpuMatrixConv3D(float* image, float* result, int imageRows, int imageCols, int imageDepth, int maskRows, int maskCols, int maskDepth, int resultRows, int resultCols) {
    // Calculate thread and block indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    // define the size of the shared memory tile
    // for a 3x3x3 mask.
    extern __shared__ float tile[];

    if (row < resultRows && col < resultCols && depth < imageDepth) {
        float sum = 0.0;

        // load a tile of the image into shared memory
        // assuming each thread loads one element.
        // might need to load multiple elements per thread to fully populate the tile.
        // 
        int linearIndex = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
        int startRow = blockIdx.y * blockDim.y - maskRows / 2;
        int startCol = blockIdx.x * blockDim.x - maskCols / 2;
        int startDepth = blockIdx.z * blockDim.z - maskDepth / 2;
        int indexRow = startRow + threadIdx.y;
        int indexCol = startCol + threadIdx.x;
        int indexDepth = startDepth + threadIdx.z;

        if (indexRow >= 0 && indexRow < imageRows && indexCol >= 0 && indexCol < imageCols && indexDepth >= 0 && indexDepth < imageDepth) {
            tile[linearIndex] = image[indexDepth * (imageRows * imageCols) + indexRow * imageCols + indexCol];
        } else {
            tile[linearIndex] = 0.0;
        }

        __syncthreads();

        // perform convolution using the tile in shared memory and the mask in constant memory
        // this is an example using a 3x3x3 mask.
        if (threadIdx.x < blockDim.x - maskCols + 1 && threadIdx.y < blockDim.y - maskRows + 1 && threadIdx.z < blockDim.z - maskDepth + 1) {
            for (int z = 0; z < maskDepth; z++) {
                for (int y = 0; y < maskRows; y++) {
                    for (int x = 0; x < maskCols; x++) {
                        sum += tile[(threadIdx.z + z) * (blockDim.x * blockDim.y) + (threadIdx.y + y) * blockDim.x + (threadIdx.x + x)] * constMask[z * (maskRows * maskCols) + y * maskCols + x];
                    }
                }
            }
            // the result
            if (row < resultRows && col < resultCols && depth < imageDepth) {
                result[depth * (resultRows * resultCols) + row * resultCols + col] = sum;
            }
        }
    }
}

void gpuMatrixConvulation3D(struct CpuGpuMat* image, const float* mask, struct CpuGpuMat* result) {
    // copy mask to constant memory
    cudaMemcpyToSymbol(constMask, mask, sizeof(float) * 27); // Assuming a 3x3x3 mask

    // grid and block dimensions
    int threadsPerBlock = 1024; // adjust based on the size of your data and shared memory options
    int gridCols = ceil(float(result->Cols) / float(threadsPerBlock));
    int gridRows = ceil(float(result->Rows) / float(threadsPerBlock));
    int gridDepth = ceil(float(image->Depth) / float(threadsPerBlock));

    dim3 gridDim(gridCols, gridRows, gridDepth);
    dim3 blockDim(threadsPerBlock, threadsPerBlock, threadsPerBlock);

    // calculate shared memory size based on tile and mask size
    size_t sharedMemSize = threadsPerBlock * threadsPerBlock * threadsPerBlock * sizeof(float);

    gpuMatrixConv3D<<<gridDim, blockDim, sharedMemSize>>>((float*)image->gpuP, (float*)result->gpuP, image->Rows, image->Cols, image->Depth, 3, 3, 3, result->Rows, result->Cols);
}
