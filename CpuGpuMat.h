#pragma once

struct CpuGpuMat {
    void* cpuP;     // RAM pointer
    void* gpuP;     // Graphics memory pointer
    int Rows;
    int Cols;
    int Depth;
    int Size;       // Total number of elements (Rows * Cols * Depth)
};