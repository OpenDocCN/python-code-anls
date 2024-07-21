# `.\pytorch\torch\csrc\inductor\aoti_runner\model_container_runner_cpu.cpp`

```py
#if !defined(C10_MOBILE) && !defined(ANDROID)
// 如果未定义 C10_MOBILE 和 ANDROID 宏，则包含模型容器运行器头文件
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

// 定义 torch::inductor 命名空间
namespace torch::inductor {

// 注意：以下 API 可能会因为活跃的开发而发生变化
// 对于这些 API，我们不提供向后兼容性保证

// 构造函数：AOTIModelContainerRunnerCpu
AOTIModelContainerRunnerCpu::AOTIModelContainerRunnerCpu(
    const std::string& model_so_path,   // 模型共享对象文件路径
    size_t num_models)                  // 模型数量
    : AOTIModelContainerRunner(model_so_path, num_models, "cpu", "") {}  // 调用基类构造函数初始化

// 析构函数：AOTIModelContainerRunnerCpu
AOTIModelContainerRunnerCpu::~AOTIModelContainerRunnerCpu() {}

// 运行函数，接收输入张量向量，返回输出张量向量
std::vector<at::Tensor> AOTIModelContainerRunnerCpu::run(
    std::vector<at::Tensor>& inputs) {
  return AOTIModelContainerRunner::run(inputs);  // 调用基类的运行函数
}

} // namespace torch::inductor
#endif
```