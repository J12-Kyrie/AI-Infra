Step1 Pytorch自定义CUDA算子

总体架构图

Python(Pytorch) --> C++接口 --> C++调用CUDA --> CUDA Kernel

GPU kernel 传指针，线程独立做计算；

PyTorch自定义CUDA算子采用**三层架构**：最上层是Python接口，提供用户友好的API；中间层是C++包装器，处理张量操作和参数验证；底层是CUDA核函数，执行具体的GPU并行计算。这种设计既保证了易用性，又实现了高性能。

**第一步：算子定义与注册** 使用`TORCH_LIBRARY`宏定义算子签名

第二步：**CUDA实现**：编写核函数，利用GPU的大规模并行能力

第三步：Python接口封装




