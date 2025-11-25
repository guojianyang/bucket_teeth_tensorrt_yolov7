// 在文件开头添加这些预处理器指令
// #define __CUDA_API_VERSION_INTERNAL

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <cuda_tools.hpp>


namespace CUDATools{
    #define checkCudaErrors(func)               \
    {                                   \
        cudaError_t e = (func);         \
        if(e != cudaSuccess)                                        \
            printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
    }

    float* read_output_data(int& num_elements) {
        // 读取保存的output_data文件
        FILE* in_file = fopen("workspace/output_data.bin", "rb");
        if (!in_file) {
            printf("Failed to open output_data.bin\n");
            num_elements = 0;
            return nullptr;
        }

        // 获取文件大小
        fseek(in_file, 0, SEEK_END);
        long file_size = ftell(in_file);
        fseek(in_file, 0, SEEK_SET);

        // 计算元素数量
        num_elements = file_size / sizeof(float);
        
        // 分配内存并读取数据
        float* data = new float[num_elements];
        size_t read_size = fread(data, sizeof(float), num_elements, in_file);
        fclose(in_file);

        printf("Successfully read %zu floats from output_data.bin\n", read_size);
        
        return data;
    }

    __host__ float* get_invert_affine_matrix(cv::Mat source_image, int dst_width, int dst_height){
        int width = source_image.cols;
        int height = source_image.rows;

        float scale_x = dst_width/(float)width;
        float scale_y = dst_height/(float)height;
        float scale= scale_x > scale_y ? scale_y : scale_x;
        float i2d[6];
        float* d2i = new float[6];// 动态分配内存
        i2d[0] = scale; i2d[1] = 0; i2d[2] = (-scale*width + dst_width + scale -1)*0.5;
        i2d[3] = 0; i2d[4] = scale; i2d[5] = (-scale*height + dst_height +scale -1)*0.5;
        cv::Mat m2x3_i2d = cv::Mat(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i = cv::Mat(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

        return d2i; //返回d2i float指针
    }

    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    /*
    定义grid和block(在decode算子中，grid和block都是一维)
    dim3 grid= （output_numboxes/256）// grid(100)
    dim3 block = (256)
    */
    __global__ void decode_kernel(float* output_data_device, int num_boxes, int num_class, float con_threshold,float* invert_affine_matrix, float* parray, int max_objects){
        int idx = blockDim.x*blockIdx.x + threadIdx.x;
        if(idx >= num_boxes) return; // 如果检测框的number大于设置的最大num，就跳过直接return
        // printf("idx : %d\n", idx); //打印本线程在总体线程中的thread索引值
        float* ptr =  output_data_device + idx*(5+num_class); // ptr代表一个boxes的其实地址，11个float为一个stride
        float objetcness = ptr[4];
        if(objetcness < con_threshold) {
            return;
        }
        int label = 0;
        float* class_ptr = ptr +5;
        float confidence = *class_ptr++;
        for(int i= 1; i < num_class; ++i, ++class_ptr){
            if(*class_ptr > confidence){
                confidence = *class_ptr;
                label = i;
            }

        }
        confidence = confidence * objetcness;
        if(confidence < con_threshold){
            return;
        }
        int index = atomicAdd(parray,1);
        if(index >= max_objects) return; // 添加边界检查
        float cx = *ptr++;
        float cy = *ptr++;
        float width = *ptr++;
        float height = *ptr++;
        float left = cx - width * 0.5f;
        float top  = cy - height * 0.5f;
        float right = cx + width * 0.5f;
        float bottom= cy + height * 0.5f;
        affine_project(invert_affine_matrix, left, top, &left, &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
        float* ptr_array = parray + 1 + index * 7; // 7代表输出的一个box中，有7个值： num_boxes, left, top, right, bottom, confidence, keep_flags
        *ptr_array++ = left;
        *ptr_array++ = top;
        *ptr_array++ = right;
        *ptr_array++ = bottom;
        *ptr_array++ = confidence;
        *ptr_array++ = float(label);
        *ptr_array++ = 1; // 1 = keep, 0 = ignore
    }

    __device__ float iou_box(float left1, float top1, float right1, float bottom1, float left2, float top2, float right2, float bottom2){
        // 完成iou_boxd手撕代码
        float cleft = max(left1, left2);
        float cright =min(right1, right2);
        float ctop = max(top1, top2);
        float cbottom = min(bottom1, bottom2);

        float i_area = max(cright-cleft, 0.0f) * max(cbottom-ctop, 0.0f);
        if(i_area == 0.0f ) return 0.0f;
        float a_area = max(right1-left1, 0.0f)*max(bottom1-top1, 0.0f);
        float b_area = max(right2-left2, 0.0f)*max(bottom2-top2, 0.0f);
        return i_area /(a_area + b_area - i_area);
    }

    __global__ void nms_kernel(float* parray_device, int max_objects, float nms_threshold){
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        int count = min(int(*parray_device), max_objects);
        if(position > count){
            return;
        }
        float* pcurrent = parray_device + 1 + position * 7;
        for(int i =0; i < count; i++){
            float* pitem = parray_device + 1 + i * 7;
            if(i == position || pcurrent[5] != pitem[5]){
                continue;
            }
            if(pitem[4] >= pcurrent[4]){
                if(pitem[4]==pcurrent[4] && i < position){
                    continue;
                }
                auto iou = iou_box(pcurrent[0], pcurrent[1],pcurrent[2], pcurrent[3],
                                pitem[0], pitem[1], pitem[2], pitem[3]);
                if(iou > nms_threshold){
                    pcurrent[6] = 0;
                    return;
                }

            }
        }

    }

    
    void decode_kernel_invoker(float* output_data_device, int num_boxes, int num_class, float con_threshold, float* invert_affine_matrix_device, float* parray_device, int max_object, cudaStream_t stream){
        auto grid = CUDATools::grid_dims(num_boxes);
        auto block = CUDATools::block_dims(num_boxes);
        decode_kernel<<<grid, block, 0, stream>>>(output_data_device, num_boxes, num_class, con_threshold, invert_affine_matrix_device, parray_device,max_object);
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_object, cudaStream_t stream){
        auto grid = CUDATools::grid_dims(max_object);
        auto block = CUDATools::block_dims(max_object);
        nms_kernel<<<grid, block, 0, stream>>>(parray, max_object, nms_threshold);
    }
};


// int main(){
//     int output_numbox = 25500;
//     int num_class = 6;
//     float confidence_threshold = 0.65;
//     float nms_threshold =0.45
//     int max_objects = 1024;
//     cudaStream_t stream1;//声明一个cuda流
//     cudaStreamCreate(&stream1);//创建并初始化cuda流

//     // 输入数据
//     float* output_data_host;
//     float* output_data_device;
//     // 仿射变换矩阵
//     float* invert_affine_matrix_host = nullptr;
//     float* invert_affine_matrix_device;

//     cv::Mat image(cv::imread("workspace/frame_000250.jpg"));
//     int dst_width = 640, dst_height = 640;

//     // if(invert_affine_matrix_host!=nullptr){
//     //     delete[] invert_affine_matrix_host;
//     // }
//     invert_affine_matrix_host =  get_invert_affine_matrix(image, dst_width,dst_height);

//     // 输出数据
//     float* parray_host = new float[7 * max_objects];
//     float* parray_device;
//     // checkCudaErrors(cudaMemsetAsync(parray_device, 0, sizeof(float), stream1)); // 初始化parray_device
//     int output_num_element = output_numbox*11;
//     output_data_host =  read_output_data(output_num_element);

//     int output_data_byte = sizeof(float)*output_numbox*11;
//     checkCudaErrors(cudaMalloc(&output_data_device, output_data_byte));//每个output_numbox中有11个元素（4边框+1obj+6类别）
//     checkCudaErrors(cudaMalloc(&invert_affine_matrix_device, sizeof(float) * 6));// 一个仿射变换矩阵中存在6个元素
//     // 初始化第一个元素为0（计数器）

//     checkCudaErrors(cudaMalloc(&parray_device, sizeof(float) * max_objects * 7));//parray_device中存放decode后boxes数据，每个boxes有6个数据
//     float zero = 0;
//     checkCudaErrors(cudaMemcpyAsync(parray_device, &zero, sizeof(float), cudaMemcpyHostToDevice, stream1));

//     // device中需要预先填入数据的有output_data_device, invert_affine_matrix,在核函数计算过程中填入的数据的变量为parray_device
//     checkCudaErrors(cudaMemcpyAsync(output_data_device, output_data_host, output_data_byte, cudaMemcpyHostToDevice, stream1));
//     checkCudaErrors(cudaMemcpyAsync(invert_affine_matrix_device, invert_affine_matrix_host, sizeof(float)*6, cudaMemcpyHostToDevice, stream1));

//     //设计grid和block
//     dim3 block(256);
//     dim3 grid((output_numbox + block.x - 1)/block.x); //向上取整，保留所有数据都被处理，grid（100）
//     // 完成核函数核心内容
//    decode_kernel<<<grid, block, 0, stream1>>>(output_data_device,output_numbox,num_class, confidence_threshold, invert_affine_matrix_device, parray_device, max_objects);
//    /*
//    直接利用decode_kernel的输出结果(parray_device)作为nms_kernel的输入
//    */
//    checkCudaErrors(cudaMemcpyAsync(parray_host, parray_device, sizeof(float) * max_objects*7, cudaMemcpyDeviceToHost, stream1));
//    // 等待流完成
//    checkCudaErrors(cudaStreamSynchronize(stream1));
//    printf("decode 解码后剩余的box数为: %d \n", int(parray_host[0]));
//    int num_input_nms = int(parray_host[0]);
//    auto nms_grid = CUDATools::grid_dims(max_objects);
//    dim3 nms_block = CUDATools::block_dims(max_objects);
//    nms_kernel<<<num_grid, nms_block, 0, stream1>>>(parray_device, max_objects, nms_threshold)





//    delete[] parray_host;
//    delete[] invert_affine_matrix_host;
//    delete[] output_data_host;
//    cudaFree(output_data_device);
//    cudaFree(invert_affine_matrix_device);
//    cudaFree(parray_device);
//    cudaStreamDestroy(stream1);



//     return 0;
// }


/*
运行命令行（加上so动态库）
export LD_LIBRARY_PATH=/home/guo/miniconda3/envs/trtpy/lib/python3.10/site-packages/trtpy/cpp-packages/opencv4.2/lib:$LD_LIBRARY_PATH
./decode_nms
*/



