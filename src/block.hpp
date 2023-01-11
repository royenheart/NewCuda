/**
 * @file block.hpp
 * @author royenheart (royenheart@outlook.com)
 * @brief 图像显示 block
 * @version 0.1
 * @date 2023-01-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "image.hpp"
#include <cuda_runtime.h>

typedef struct DataBlock DataBlock;
typedef struct BoolBlock BoolBlock;

struct DataBlock {
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    IMAGE<float> *bitmap;
    
    cudaEvent_t start, stop;
    float totalTime;
    int frams;
};

struct BoolBlock {
    unsigned char *output_bitmap;
    bool *dev_outSrc;
    bool *dev_inSrc;
    IMAGE<bool> *bitmap;

    cudaEvent_t start, stop;
    float totalTime;
    int frams;
};