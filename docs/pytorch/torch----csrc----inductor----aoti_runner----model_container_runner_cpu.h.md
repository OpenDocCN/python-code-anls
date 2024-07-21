# `.\pytorch\torch\csrc\inductor\aoti_runner\model_container_runner_cpu.h`

```py
#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once


// 如果 C10_MOBILE 和 ANDROID 宏都未定义，则执行以下代码
// 防止头文件被多次包含
#pragma once



#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>


// 包含模型容器运行器的头文件
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>



namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1);
  
  ~AOTIModelContainerRunnerCpu();
  
  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs);
};
} // namespace torch::inductor
#endif


// 声明 torch::inductor 命名空间
namespace torch::inductor {

// AOTIModelContainerRunnerCpu 类的声明，继承自 AOTIModelContainerRunner 类
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  // 构造函数，接受模型库路径和模型数量作为参数
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1);
  
  // 析构函数声明
  ~AOTIModelContainerRunnerCpu();
  
  // 运行函数声明，接受一个输入张量向量，并返回一个张量向量
  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs);
};

} // namespace torch::inductor
#endif



#endif


// 结束条件编译指令，确保头文件中的内容只被编译一次
```