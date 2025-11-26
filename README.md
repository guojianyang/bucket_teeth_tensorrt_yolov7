# Bucket_Teeth_TensorRT_Yolov7
=======
斗齿检测模型优化部署


# 备注：
1. 利用tensorrt构建engine引擎时，`ptq_yolov7-w6_trained.onnx`模型的输出存在9个输出的情况，但是我们只需其中第一个输出，需要屏蔽其他输出后再生成引擎；
2. 将main.cpp中的后处理部分（decode+nms）的传统串行计算变成cuda核函数的方式，main函数中增加推理函数`inference_cuda_decode_nms()`,(decode+nms)部分推理速度提升5-8倍！！！（**新增**）

# 运行步骤
## 1. 编译&&运行(makefile文件中自带运行命令)
首先检查workspace文件夹下是否存在(*.trtmodel)模型,若存在，则跳过build engine，若不存在，则需要花费较长时间（20-30min）构建engine文件，执行如下命令行：

```make run -j$(nproc)```