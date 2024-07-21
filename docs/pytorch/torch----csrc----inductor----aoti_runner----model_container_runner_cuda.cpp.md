# `.\pytorch\torch\csrc\inductor\aoti_runner\model_container_runner_cuda.cpp`

```py
#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

namespace torch::inductor {

// 定义 AOTIModelContainerRunnerCuda 类的构造函数，初始化基类 AOTIModelContainerRunner
AOTIModelContainerRunnerCuda::AOTIModelContainerRunnerCuda(
    const std::string& model_so_path,    // 模型动态链接库路径
    size_t num_models,                   // 模型数量
    const std::string& device_str,       // 设备字符串描述
    const std::string& cubin_dir)        // CUDA 编译二进制文件目录
    : AOTIModelContainerRunner(
          model_so_path,                 // 调用基类构造函数初始化模型路径等信息
          num_models,
          device_str,
          cubin_dir) {}

// 定义 AOTIModelContainerRunnerCuda 类的析构函数
AOTIModelContainerRunnerCuda::~AOTIModelContainerRunnerCuda() {}

// 运行模型推断的函数，使用当前 CUDA 流
std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run(
    std::vector<at::Tensor>& inputs) {   // 输入张量列表
  at::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();  // 获取当前 CUDA 流
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream.stream()));  // 调用基类的 run 函数
}

// 使用指定 CUDA 流运行模型推断的函数
std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run_with_cuda_stream(
    std::vector<at::Tensor>& inputs,    // 输入张量列表
    at::cuda::CUDAStream cuda_stream) { // 指定的 CUDA 流
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream.stream()));  // 调用基类的 run 函数
}

} // namespace torch::inductor
#endif
```