/*
ptq_yolov7-w6_trained.onnx模型在三个框架中运行耗时对比：
onnxruntime框架：259ms；
openvino框架：127ms；
tensorRT框架(int8)：40ms（3090显卡）,20ms（jetson AGX）；
tensorRT框架(float32):302ms（jetson AGX）；
*/
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>
#include <nvToolsExt.h>

// cuda include
#include <cuda_runtime.h>
#include <cuda_tools.hpp>
#include <cuda.h>
#include <decode_nms.hpp>

//myself trt include
#include <trt_tensor.hpp>
#include <monopoly_allocator.hpp>
// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#define CURRENT_DEVICE_ID -1

using namespace std;
using namespace CUDATools;



namespace CUDATools {
    void decode_kernel_invoker(float* output_data_device, int num_boxes, int num_class, float con_threshold, float* invert_affine_matrix_device, float* parray_device, int max_object, cudaStream_t stream);
    void nms_kernel_invoker(float* parray, float nms_threshold, int max_object, cudaStream_t stream);
}


#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

static const char* theethlabels[] = {
    "2_t", "bucket", "5_t", "4_t", "3_t","1_t"
};

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
// template<typename _T>
// shared_ptr<_T> make_nvshared(_T* ptr){
//     return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
// }
// 修改为
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr);
}

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 在文件开头添加以下声明
__global__ void decode_kernel(
    float* predict, 
    int num_boxes, 
    int num_classes, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray, 
    int max_objects
);

__global__ void nms_kernel(
    float* parray, 
    int max_objects, 
    float nms_threshold
);




// 上一节的代码
bool build_model(){

    if(exists("ptq_yolov7-w6_trained.trtmodel")){
        printf("ptq_yolov7-w6_trained.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 1. 创建 builder 与 config
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config  = make_nvshared(builder->createBuilderConfig());

    // 2. 注意：TRT10 推荐使用 createNetwork() 而不是 createNetworkV2()
    uint32_t flag = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = make_nvshared(builder->createNetworkV2(flag));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile("ptq_yolov7-w6_trained.onnx",
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        printf("Failed to parse onnx.\n");
        return false;
    }
    //直接根据onnx结构图屏蔽并不需要的输出[1,2,3,4,5,6,7,8] ,[0]为需要输出
    // 4. 删除多余输出，只留 output[0]
    while (network->getNbOutputs() > 1) {
        network->unmarkOutput(*network->getOutput(1));
    }
    printf("Final output name = %s\n", network->getOutput(0)->getName());


    
    // 配置INT8
    config->setFlag(nvinfer1::BuilderFlag::kINT8);


    // TRT10 构建序列化的网络（替代 buildEngineWithConfig）
    auto serialized_model = make_nvshared(
        builder->buildSerializedNetwork(*network, *config)
    );
    if (!serialized_model) {
        printf("Build serialized network failed.\n");
        return false;
    }

    // 写入 trtmodel 文件
    FILE* f = fopen("ptq_yolov7-w6_trained.trtmodel", "wb");
    fwrite(serialized_model->data(), 1, serialized_model->size(), f);
    fclose(f);

    printf("TensorRT-10 Engine Build Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}
void inference_cuda_decode_nms_tensor(){

    TRTLogger logger;
    auto engine_data = load_file("ptq_yolov7-w6_trained.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }
    if(engine->getNbIOTensors() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbIOTensors() - 1);
        return;
    }
    cudaStream_t stream_0 = nullptr;
    checkCudaErrors(cudaStreamCreate(&stream_0));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    //2.创建的输入数据的host和device内存
    int input_batch = 1;
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;
    int input_numel = input_batch * input_channel * input_height * input_width;
    auto input = std::make_shared<TRT::Tensor>(input_batch, input_channel, input_height, input_width, TRT::DataType::Float, nullptr,CURRENT_DEVICE_ID);
    input->set_stream(stream_0,true);   

    //3.加载图片，并进行相关的仿射变换，并将图片从host to device
    auto image = cv::imread("frame_000250.jpg");
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image_frame_000250.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input->cpu<float>() + image_area * 0; // trt_tensor取代
    float* phost_g = input->cpu<float>() + image_area * 1; // trt_tensor取代
    float* phost_r = input->cpu<float>() + image_area * 2; // trt_tensor取代
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

    input->to_gpu(true); // trt_tensor取代

    // 创建输出数据的host和device内存
    auto output_dims = engine->getTensorShape("output");
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    std::vector<int> egnine_output_dims={input->batch(), output_dims.d[1], output_dims.d[2]};
    auto output = make_shared<TRT::Tensor>(egnine_output_dims, TRT::DataType::Float,nullptr,CURRENT_DEVICE_ID);
    output->set_stream(stream_0, true);    

    // 明确当前推理时，使用的数据输入大小，并使用enqueueV2进行推理。
    auto input_dims = engine->getTensorShape("input");
    input_dims.d[0] = input_batch;
    bool success =false;
    // execution_context->setInputShape("input", input_dims);
    // float* bindings[] = {input->gpu<float>(), output->gpu<float>()}; // trt_tensor取代
    execution_context->setTensorAddress("input", input->gpu<float>());
    execution_context->setTensorAddress("output", output->gpu<float>());
    execution_context->setInputShape("input", input_dims);

    for(int i = 0; i < 5; i++){
        success = execution_context->enqueueV3(stream_0); // 第一次 warm up 100次推理
        checkCudaErrors(cudaStreamSynchronize(stream_0));
        printf("warm up 第 %d 次推理完成!!!\n", i);
    }
    auto start_time = std::chrono::high_resolution_clock::now(); // 计时开始
    success = execution_context->enqueueV3(stream_0);
    checkCudaErrors(cudaStreamSynchronize(stream_0));

    auto end_time = std::chrono::high_resolution_clock::now(); //计时结束
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("纯推理时间: %d us\n", duration.count());
    printf("推理完成!!!\n");
    
    vector<vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.45;
    printf("output_numbox:%d \n", output_numbox); 
    //构建engine时，9个输出屏蔽了8个，只保留“output”，且在inference时，通过打印检测输出都是有效数据，接下来对输出做合理的后处理
    std::vector<int> affine_dims = {6};
    std::vector<int> parray_dims= {1+1024*7};
    auto invert_affine_matrix = make_shared<TRT::Tensor>(affine_dims, TRT::DataType::Float ,nullptr, CURRENT_DEVICE_ID);
    auto parray = make_shared<TRT::Tensor>(parray_dims, TRT::DataType::Float, nullptr, CURRENT_DEVICE_ID);
    invert_affine_matrix->set_stream(stream_0, true);
    parray->set_stream(stream_0, true);
    float* invert_affine_matrix_host = invert_affine_matrix->cpu<float>();
    for(int i = 0; i < 6; i++){
        if(d2i[i] != NULL){
            invert_affine_matrix_host[i] = d2i[i];
        }
    }
    int num_boxes = output_numbox;
    int num_class = num_classes;
    float con_threshold = 0.25;
    float max_objects = 1024;
    invert_affine_matrix->to_gpu(true);
    CUDATools::decode_kernel_invoker(output->gpu<float>(), num_boxes, num_class, con_threshold, invert_affine_matrix->gpu<float>(), parray->gpu<float>(), max_objects,stream_0);
    checkCudaErrors(cudaStreamSynchronize(stream_0));
    CUDATools::nms_kernel_invoker(parray->gpu<float>(), nms_threshold, max_objects, stream_0);
    parray->to_cpu(true);
    checkCudaErrors(cudaStreamSynchronize(stream_0));

    for(int i = 0;  i < num_boxes; i++){
        //将keep==0的值全部过滤掉
        float* item  = parray->cpu<float>() + 1 + i * 7;
        if(int(item[6]) != 1){
            continue;
        }
        float left = item[0];
        float top = item[1];
        float right = item[2];
        float bottom = item[3];
        float con_threshold = item[4];
        int label = int(item[5]);
        printf("label %d = %d \n", i, label);
        cv::Scalar color;
        tie(color[0], color[1], color[2]) = random_color(label);
        cv::rectangle(image, cv::Point(left,top), cv::Point(right, bottom), color, 2); // "1"代表框粗细
        auto name = theethlabels[label];
        auto caption = cv::format("%.2f",con_threshold);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left+text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("inference_cuda_decode_nms_tensor.jpg", image);
    printf("已生成 inference_cuda_decode_nms_tensor.jpg, 储存在/workspace下\n ");
}

void inference_cuda_decode_nms_tensor_cudaGraph(){

    TRTLogger logger;
    auto engine_data = load_file("ptq_yolov7-w6_trained.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }
    if(engine->getNbIOTensors() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbIOTensors() - 1);
        return;
    }
    cudaStream_t stream_0 = nullptr;
    checkCudaErrors(cudaStreamCreate(&stream_0));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    //2.创建的输入数据的host和device内存
    int input_batch = 1;
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;
    int input_numel = input_batch * input_channel * input_height * input_width;
    auto input = std::make_shared<TRT::Tensor>(input_batch, input_channel, input_height, input_width, TRT::DataType::Float, nullptr,CURRENT_DEVICE_ID);
    input->set_stream(stream_0,true);   

    //3.加载图片，并进行相关的仿射变换，并将图片从host to device
    auto image = cv::imread("frame_000250.jpg");
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image_frame_000250.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input->cpu<float>() + image_area * 0; // trt_tensor取代
    float* phost_g = input->cpu<float>() + image_area * 1; // trt_tensor取代
    float* phost_r = input->cpu<float>() + image_area * 2; // trt_tensor取代
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

    input->to_gpu(true); // trt_tensor取代

    // 创建输出数据的host和device内存
    auto output_dims = engine->getTensorShape("output");
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    std::vector<int> egnine_output_dims={input->batch(), output_dims.d[1], output_dims.d[2]};
    auto output = make_shared<TRT::Tensor>(egnine_output_dims, TRT::DataType::Float,nullptr,CURRENT_DEVICE_ID);
    output->set_stream(stream_0, true);    

    // 明确当前推理时，使用的数据输入大小，并使用enqueueV2进行推理。
    auto input_dims = engine->getTensorShape("input");
    input_dims.d[0] = input_batch;
    bool success =false;
    // execution_context->setInputShape("input", input_dims);
    // float* bindings[] = {input->gpu<float>(), output->gpu<float>()}; // trt_tensor取代
    execution_context->setTensorAddress("input", input->gpu<float>());
    execution_context->setTensorAddress("output", output->gpu<float>());
    execution_context->setInputShape("input", input_dims);

    // 设置cuda graph相关变量
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 开始记录graph
    checkCudaErrors(cudaStreamBeginCapture(stream_0, cudaStreamCaptureModeGlobal));

    // 执行一次推理以记录到graph
    success = execution_context->enqueueV3(stream_0);
    // checkCudaErrors(cudaStreamSynchronize(stream_0));

    // 结束记录并创建graph实例
    checkCudaErrors(cudaStreamEndCapture(stream_0, &graph));
    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    for(int i = 0; i < 5; i++){
        checkCudaErrors(cudaGraphLaunch(instance, stream_0)); // 使用graph执行
        checkCudaErrors(cudaStreamSynchronize(stream_0));
        printf("warm up 第 %d 次推理完成!!!\n", i);
    }

    auto start_time = std::chrono::high_resolution_clock::now(); // 计时开始
    checkCudaErrors(cudaGraphLaunch(instance, stream_0)); // 使用graph执行推理
    checkCudaErrors(cudaStreamSynchronize(stream_0));

    auto end_time = std::chrono::high_resolution_clock::now(); //计时结束
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("纯推理时间: %d us\n", duration.count());
    printf("推理完成!!!\n");
    
    vector<vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.45;
    printf("output_numbox:%d \n", output_numbox); 
    //构建engine时，9个输出屏蔽了8个，只保留“output”，且在inference时，通过打印检测输出都是有效数据，接下来对输出做合理的后处理
    std::vector<int> affine_dims = {6};
    std::vector<int> parray_dims= {1+1024*7};
    auto invert_affine_matrix = make_shared<TRT::Tensor>(affine_dims, TRT::DataType::Float ,nullptr, CURRENT_DEVICE_ID);
    auto parray = make_shared<TRT::Tensor>(parray_dims, TRT::DataType::Float, nullptr, CURRENT_DEVICE_ID);
    invert_affine_matrix->set_stream(stream_0, true);
    parray->set_stream(stream_0, true);
    float* invert_affine_matrix_host = invert_affine_matrix->cpu<float>();
    for(int i = 0; i < 6;  i++){
        if(d2i[i] != NULL){
            invert_affine_matrix_host[i] = d2i[i];
        }
    }
    int num_boxes = output_numbox;
    int num_class = num_classes;
    float con_threshold = 0.25;
    float max_objects = 1024;
    invert_affine_matrix->to_gpu(true);
    CUDATools::decode_kernel_invoker(output->gpu<float>(), num_boxes, num_class, con_threshold, invert_affine_matrix->gpu<float>(), parray->gpu<float>(), max_objects,stream_0);
    checkCudaErrors(cudaStreamSynchronize(stream_0));
    CUDATools::nms_kernel_invoker(parray->gpu<float>(), nms_threshold, max_objects, stream_0);
    parray->to_cpu(true);
    checkCudaErrors(cudaStreamSynchronize(stream_0));

    for(int i = 0;  i < num_boxes; i++){
        //将keep==0的值全部过滤掉
        float* item  = parray->cpu<float>() + 1 + i * 7;
        if(int(item[6]) != 1){
            continue;
        }
        float left = item[0];
        float top = item[1];
        float right = item[2];
        float bottom = item[3];
        float con_threshold = item[4];
        int label = int(item[5]);
        printf("label %d = %d \n", i, label);
        cv::Scalar color;
        tie(color[0], color[1], color[2]) = random_color(label);
        cv::rectangle(image, cv::Point(left,top), cv::Point(right, bottom), color, 2); // "1"代表框粗细
        auto name = theethlabels[label];
        auto caption = cv::format("%.2f",con_threshold);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left+text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("inference_cuda_decode_nms_tensor_cudaGraph.jpg", image);
    printf("已生成 inference_cuda_decode_nms_tensor_cudaGraph.jpg, 储存在/workspace下\n ");
}




void inference_cuda_decode_nms_tensor_cudaGraph_monopoly(std::shared_ptr<MonopolyAllocator<TRT::Tensor>>& tensor_allocator){

    TRTLogger logger;
    auto engine_data = load_file("ptq_yolov7-w6_trained.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }
    if(engine->getNbIOTensors() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbIOTensors() - 1);
        return;
    }
    cudaStream_t stream_0 = nullptr;
    checkCudaErrors(cudaStreamCreate(&stream_0));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    //2.创建的输入数据的host和device内存
    int input_batch = 1;
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;
    int input_numel = input_batch * input_channel * input_height * input_width;

    // 使用tensor池管理器来创建input_tensor
    tensor_allocator = std::make_shared<MonopolyAllocator<TRT::Tensor>>(4);
    auto input_tensor = tensor_allocator->query();
    if(input_tensor){
        input_tensor->data() = std::make_shared<TRT::Tensor>(input_batch, input_channel, input_height, input_width, TRT::DataType::Float, nullptr,CURRENT_DEVICE_ID);
        // input_tensor->release(); //目前无法理解，为什么要释放
    }else{
        printf("Failed to get input tensor from allocator! \n");
        return;
    }
    auto input = input_tensor->data();
    input->set_stream(stream_0,true);

    // auto input = std::make_shared<TRT::Tensor>(input_batch, input_channel, input_height, input_width, TRT::DataType::Float, nullptr,CURRENT_DEVICE_ID);
    // input->set_stream(stream_0,true);   

    //3.加载图片，并进行相关的仿射变换，并将图片从host to device
    auto image = cv::imread("frame_000250.jpg");
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image_frame_000250.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input->cpu<float>() + image_area * 0; // trt_tensor取代
    float* phost_g = input->cpu<float>() + image_area * 1; // trt_tensor取代
    float* phost_r = input->cpu<float>() + image_area * 2; // trt_tensor取代
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

    input->to_gpu(true); // trt_tensor取代

    // 创建输出数据的host和device内存
    auto output_dims = engine->getTensorShape("output");
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    std::vector<int> egnine_output_dims={input->batch(), output_dims.d[1], output_dims.d[2]};

    // 使用tensor池管理器来创建output_tensor
    auto output_tensor = tensor_allocator->query();
    if(output_tensor){
        output_tensor->data() = std::make_shared<TRT::Tensor>(egnine_output_dims, TRT::DataType::Float,nullptr,CURRENT_DEVICE_ID);
        // input_tensor->release(); //目前无法理解为什么要释放
    }else{
        printf("Failed to get input tensor from allocator! \n");
        return; 
    }
    auto output = output_tensor->data();
    output->set_stream(stream_0,true);
    // auto output = make_shared<TRT::Tensor>(egnine_output_dims, TRT::DataType::Float,nullptr,CURRENT_DEVICE_ID);
    // output->set_stream(stream_0, true);    

    // 明确当前推理时，使用的数据输入大小，并使用enqueueV2进行推理。
    auto input_dims = engine->getTensorShape("input");
    input_dims.d[0] = input_batch;
    bool success =false;
    // execution_context->setInputShape("input", input_dims);
    // float* bindings[] = {input->gpu<float>(), output->gpu<float>()}; // trt_tensor取代
    execution_context->setTensorAddress("input", input->gpu<float>());
    execution_context->setTensorAddress("output", output->gpu<float>());
    execution_context->setInputShape("input", input_dims);

    // 设置cuda graph相关变量
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 开始记录graph
    checkCudaErrors(cudaStreamBeginCapture(stream_0, cudaStreamCaptureModeGlobal));

    // 执行一次推理以记录到graph
    success = execution_context->enqueueV3(stream_0);
    // checkCudaErrors(cudaStreamSynchronize(stream_0));

    // 结束记录并创建graph实例
    checkCudaErrors(cudaStreamEndCapture(stream_0, &graph));
    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    for(int i = 0; i < 5; i++){
        checkCudaErrors(cudaGraphLaunch(instance, stream_0)); // 使用graph执行
        checkCudaErrors(cudaStreamSynchronize(stream_0));
        printf("warm up 第 %d 次使用独占分配器推理完成!!!\n", i);
    }

    auto start_time = std::chrono::high_resolution_clock::now(); // 计时开始
    checkCudaErrors(cudaGraphLaunch(instance, stream_0)); // 使用graph执行推理
    checkCudaErrors(cudaStreamSynchronize(stream_0));

    auto end_time = std::chrono::high_resolution_clock::now(); //计时结束
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("纯推理时间: %d us\n", duration.count());
    printf("使用独占分配器推理完成!!!\n");
    
    vector<vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.45;
    printf("output_numbox:%d \n", output_numbox); 
    //构建engine时，9个输出屏蔽了8个，只保留“output”，且在inference时，通过打印检测输出都是有效数据，接下来对输出做合理的后处理
    std::vector<int> affine_dims = {6};
    std::vector<int> parray_dims= {1+1024*7};
    auto invert_affine_matrix = make_shared<TRT::Tensor>(affine_dims, TRT::DataType::Float ,nullptr, CURRENT_DEVICE_ID);
    auto parray = make_shared<TRT::Tensor>(parray_dims, TRT::DataType::Float, nullptr, CURRENT_DEVICE_ID);
    invert_affine_matrix->set_stream(stream_0, true);
    parray->set_stream(stream_0, true);
    float* invert_affine_matrix_host = invert_affine_matrix->cpu<float>();
    for(int i = 0; i < 6;  i++){
        if(d2i[i] != NULL){
            invert_affine_matrix_host[i] = d2i[i];
        }
    }
    int num_boxes = output_numbox;
    int num_class = num_classes;
    float con_threshold = 0.25;
    float max_objects = 1024;
    invert_affine_matrix->to_gpu(true);
    CUDATools::decode_kernel_invoker(output->gpu<float>(), num_boxes, num_class, con_threshold, invert_affine_matrix->gpu<float>(), parray->gpu<float>(), max_objects,stream_0);
    checkCudaErrors(cudaStreamSynchronize(stream_0));
    CUDATools::nms_kernel_invoker(parray->gpu<float>(), nms_threshold, max_objects, stream_0);
    parray->to_cpu(true);
    checkCudaErrors(cudaStreamSynchronize(stream_0));

    for(int i = 0;  i < num_boxes; i++){
        //将keep==0的值全部过滤掉
        float* item  = parray->cpu<float>() + 1 + i * 7;
        if(int(item[6]) != 1){
            continue;
        }
        float left = item[0];
        float top = item[1];
        float right = item[2];
        float bottom = item[3];
        float con_threshold = item[4];
        int label = int(item[5]);
        printf("label %d = %d \n", i, label);
        cv::Scalar color;
        tie(color[0], color[1], color[2]) = random_color(label);
        cv::rectangle(image, cv::Point(left,top), cv::Point(right, bottom), color, 2); // "1"代表框粗细
        auto name = theethlabels[label];
        auto caption = cv::format("%.2f",con_threshold);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left+text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("inference_cuda_decode_nms_tensor_cudaGraph.jpg", image);
    printf("已生成 inference_cuda_decode_nms_tensor_cudaGraph.jpg, 储存在/workspace下\n ");
}


int main(){
    if(!build_model()){
        return -1;
    }
    // 申请一个独占数据管理器管理Tensor数据池
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator;
    // inference();
    // inference_cuda_decode_nms();
    inference_cuda_decode_nms_tensor_cudaGraph_monopoly(tensor_allocator);
    printf("inference 程序完成, main 结束返回！！！ \n");
    return 0;
}


/*
目前main.cpp中TensorRT的推理API接口是基于TensorRT-8.2.4的，但是现在所依赖的版本是TensorRT-10.0.1,请在main.cpp中将TensorRT构建引擎和推理相关的代码改成TensorRT-10版本的
*/