# Bucket_Teeth_TensorRT_Yolov7
=======
斗齿检测模型优化部署


# 备注：
1. 利用tensorrt构建engine引擎时，`ptq_yolov7-w6_trained.onnx`模型的输出存在9个输出的情况，但是我们只需其中第一个输出，需要屏蔽其他输出后再生成引擎[Release-v1.0](https://github.com/guojianyang/bucket_teeth_tensorrt_yolov7/tree/v1.0)；
2. 将main.cpp中的后处理部分（decode+nms）的传统串行计算变成cuda核函数的方式，main函数中增加推理函数`inference_cuda_decode_nms()`,(decode+nms)部分推理速度提升5-8倍[Release-v1.1](https://github.com/guojianyang/bucket_teeth_tensorrt_yolov7/tree/v1.1)。
3. 将main.cpp中的cuda的tensor算子启用和释放用class tensor统一管理，简化推理程序编写流程，main函数中增加推理函数`inference_cuda_decode_nms_tensor()`[Release-v1.2](https://github.com/guojianyang/bucket_teeth_tensorrt_yolov7/tree/v1.2)。
4. 尝试用jetson AGX 的DLA加速模型，因为DLA支持的算子不多，出现很多回退GPU的算子，导致用生成的模型做推理时出现延时更大的情况（30ms->100ms）,估计放弃此优化方案。但可记录DLA优化方式。[Release-v1.3](https://github.com/guojianyang/bucket_teeth_tensorrt_yolov7/releases/tag/v1.3)
# 运行步骤
## 1. 编译&&运行(makefile文件中自带运行命令)
首先检查workspace文件夹下是否存在(*.trtmodel)模型,若存在，则跳过build engine，若不存在，则需要花费较长时间（20-30min）构建engine文件，执行如下命令行：

```make run -j$(nproc)```

![现场实时检测](https://github.com/guojianyang/bucket_teeth_tensorrt_yolov7/blob/v1.2/workspace/inference_cuda_decode_nms_tensor.jpg)
