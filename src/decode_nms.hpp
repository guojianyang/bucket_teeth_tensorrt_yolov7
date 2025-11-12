// decode_nms.h
#pragma once
#include <cuda_runtime.h>

void decode_kernel_invoker(
    float* predict, 
    int num_boxes, 
    int num_classes, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray, 
    int max_objects,
    cudaStream_t stream
);

void nms_kernel_invoker(
    float* parray, 
    int max_objects, 
    float nms_threshold,
    cudaStream_t stream
);
