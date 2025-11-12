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
#include <onnx-tensorrt/NvOnnxParser.h>

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

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
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
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
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

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("ptq_yolov7-w6_trained.onnx", 1)){
        printf("Failed to parse ptq_yolov7-w6_trained.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    //遍历输入
//    for (int i = 0; i < network->getNbInputs(); ++i) {
//         string in_name = network->getInput(i)->getName();
//         printf("第 %d 个输入名字为：%s\n", i, in_name.c_str());
//         // if(
//         // in_name != "output")
//         //     network->unmarkOutput(*network->getOutput(i));
//     }
    //直接根据onnx结构图屏蔽并不需要的输出[1,2,3,4,5,6,7,8] ,[0]为需要输出
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));
    network->unmarkOutput(*network->getOutput(1));
    printf("目前的输出有 %d 个 \n", int(network->getNbOutputs()));

    string out_name = network->getOutput(0)->getName();
    printf("第 %d 个输出名字为：%s\n", 0, out_name.c_str());


   

    int maxBatchSize = 1;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    // auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    
    // 配置最小允许batch
    input_dims.d[0] = 1;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    // profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    // 配置最大允许batch
    // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    // input_dims.d[0] = maxBatchSize;
    // profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    // config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("ptq_yolov7-w6_trained.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done.\n");
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

void inference(){

    TRTLogger logger;
    auto engine_data = load_file("ptq_yolov7-w6_trained.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
    if(engine->getNbBindings() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbBindings() - 1);
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
    float scale_x = input->width() / (float)image.cols;
    float scale_y = input->height() / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input->width() + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input->height() + scale - 1) * 0.5;

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

    auto start_time = std::chrono::high_resolution_clock::now();
    input->to_gpu(true); // trt_tensor取代

    // 创建输出数据的host和device内存
    auto output_dims = engine->getBindingDimensions(1);
    
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    std::vector<int> egnine_output_dims={input->batch(),output_dims.d[1], output_dims.d[2]};
    auto output = make_shared<TRT::Tensor>(egnine_output_dims, TRT::DataType::Float,nullptr,CURRENT_DEVICE_ID);
    output->set_stream(stream_0, true);


    // 明确当前推理时，使用的数据输入大小，并使用enqueueV2进行推理。
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {static_cast<float*>(input->gpu()), static_cast<float*>(output->gpu())}; // trt_tensor取代
    bool success      = execution_context->enqueueV2((void**)bindings, stream_0, nullptr);
    output->to_cpu(true); //

    // decode box
    vector<vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;
    printf("output_numbox:%d \n", output_numbox);
    for(int i = 0; i < output_numbox; ++i){
        float* ptr = static_cast<float*>(output->cpu()) + i * output_dims.d[2];
        float objness = ptr[4];
        if(objness < confidence_threshold)
            continue;

        float* pclass = ptr + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;
        if(confidence < confidence_threshold)
            continue;

        float cx     = ptr[0];
        float cy     = ptr[1];
        float width  = ptr[2];
        float height = ptr[3];
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        float image_base_left   = d2i[0] * left   + d2i[2];
        float image_base_right  = d2i[0] * right  + d2i[2];
        float image_base_top    = d2i[0] * top    + d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
        // printf("confidence: %6f \n", confidence);
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms
    std::sort(bboxes.begin(), bboxes.end(), [](vector<float>& a, vector<float>& b){return a[5] > b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const vector<float>& a, const vector<float>& b){
        float cross_left   = std::max(a[0], b[0]);
        float cross_top    = std::max(a[1], b[1]);
        float cross_right  = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                         + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[4] == jbox[4]){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    printf("box_result.size = %d\n", box_result.size());
    //计算运行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("纯推理时间: %d us\n", duration.count());
    printf("推理完成!!!\n");

    for(int i = 0; i < box_result.size(); ++i){
        auto& ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;
        tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name      = theethlabels[class_label];
        auto caption   = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("image-draw_guojian_000252.jpg", image);
}
int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    printf("inference 程序完成, main 结束返回！！！ \n");
    return 0;
}