# `.\pytorch\torch\csrc\inductor\aoti_runner\model_container_runner_cuda.h`

```py
#if !defined(C10_MOBILE) && !defined(ANDROID)
// 如果未定义 C10_MOBILE 和 ANDROID 宏，则执行以下代码

#pragma once
// 只包含一次当前头文件的指令

#include <c10/cuda/CUDAStream.h>
// 包含 CUDAStream 类的头文件

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
// 包含模型容器运行器的头文件

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
// 注意：由于活跃开发，以下 API 可能会发生变化
// 我们不保证这些 API 的向后兼容性

class TORCH_API AOTIModelContainerRunnerCuda : public AOTIModelContainerRunner {
// 定义 AOTIModelContainerRunnerCuda 类，继承自 AOTIModelContainerRunner

 public:
  // @param device_str: cuda device string, e.g. "cuda", "cuda:0"
  // 构造函数，接受模型共享对象路径、模型数量、CUDA 设备字符串和 cubin 目录作为参数
  AOTIModelContainerRunnerCuda(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "cuda",
      const std::string& cubin_dir = "");

  // 析构函数，释放对象资源
  ~AOTIModelContainerRunnerCuda();

  // 运行模型的函数，接受输入张量并返回输出张量的向量
  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs);

  // 在指定 CUDA 流上运行模型的函数，接受输入张量和 CUDA 流对象，并返回输出张量的向量
  std::vector<at::Tensor> run_with_cuda_stream(
      std::vector<at::Tensor>& inputs,
      at::cuda::CUDAStream cuda_stream);
};

} // namespace torch::inductor
#endif
// 结束条件编译指令块
```