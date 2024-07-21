# `.\pytorch\torch\csrc\inductor\aoti_runner\model_container_runner.h`

```py
#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once


// 如果未定义 C10_MOBILE 并且未定义 ANDROID，则执行以下代码
// 使用 #pragma once 确保头文件只被编译一次



#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>


// 引入 ATen 库中的 Tensor 头文件
// 引入 torch/csrc/inductor/aoti_runtime/interface.h 头文件



// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}


// 前向声明 DynamicLibrary 结构体，定义在 at 命名空间中



namespace torch::inductor {
using TensorConstantMap = std::unordered_map<std::string, at::Tensor*>;


// 定义命名空间 torch::inductor
// 定义 TensorConstantMap 作为键为字符串，值为 at::Tensor* 指针的无序映射类型
// 定义 AOTIModelContainerRunner 类，作为 Torch API 的模型容器运行器
class TORCH_API AOTIModelContainerRunner {
 public:
  // 禁用默认构造函数、拷贝构造函数、移动构造函数、拷贝赋值运算符和移动赋值运算符
  AOTIModelContainerRunner() = delete;
  AOTIModelContainerRunner(const AOTIModelContainerRunner& other) = delete;
  AOTIModelContainerRunner(AOTIModelContainerRunner&& other) = delete;
  AOTIModelContainerRunner& operator=(const AOTIModelContainerRunner& other) =
      delete;
  AOTIModelContainerRunner& operator=(AOTIModelContainerRunner&& other) =
      delete;
  // 析构函数声明
  ~AOTIModelContainerRunner();

  // 运行模型，返回输出张量的向量
  std::vector<at::Tensor> run(
      std::vector<at::Tensor>& inputs,
      AOTInductorStreamHandle cuda_stream_handle = nullptr);

  // 获取常量名到原始全限定名的映射
  std::unordered_map<std::string, std::string> getConstantNamesToOriginalFQNs()
      const;
  // 获取常量名到数据类型的映射
  std::unordered_map<std::string, int32_t> getConstantNamesToDtypes() const;
  // 更新非活动状态的常量缓冲区
  void update_inactive_constant_buffer(const TensorConstantMap& const_map);
  // 更新常量缓冲区，根据需要使用非活动状态和完全验证更新标志
  void update_constant_buffer(
      const TensorConstantMap& const_map,
      bool use_inactive,
      bool validate_full_updates);
  // 运行常量折叠，根据需要使用非活动状态和 CUDA 流句柄
  void run_const_fold(
      bool use_inactive,
      AOTInductorStreamHandle cuda_stream_handle = nullptr);
  // 交换常量缓冲区
  void swap_constant_buffer();

  // 获取调用规范的字符串向量
  std::vector<std::string> get_call_spec();

 protected:
  // 受保护的构造函数，从模型 so 文件路径、模型数量、设备字符串和 cubin 目录创建实例
  AOTIModelContainerRunner(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir);

  // 模型 so 文件的唯一指针
  std::unique_ptr<at::DynamicLibrary> model_so_;
  // 函数指针，用于创建模型容器、删除模型容器、获取输出数量、运行模型等
  decltype(&AOTInductorModelContainerCreateWithDevice) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumConstants) get_num_constants_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantName) get_constant_name_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantOriginalFQN)
      get_constant_original_fqn_func_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantDtype) get_constant_dtype_func_{
      nullptr};
  decltype(&AOTInductorModelContainerUpdateConstantBuffer)
      update_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateInactiveConstantBuffer)
      update_inactive_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerRunConstantFolding) run_const_fold_func_{
      nullptr};
  decltype(&AOTInductorModelContainerSwapConstantBuffer)
      swap_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerGetCallSpec) get_call_spec_func_{nullptr};

  // 模型容器句柄
  AOTInductorModelContainerHandle container_handle_ = nullptr;

  // TODO: 需要一个 OSS 代理执行器实现。目前 proxy_executor_handle_ 总是为 nullptr。
  AOTIProxyExecutorHandle proxy_executor_handle_ = nullptr;
};

// 命名空间 torch::inductor 结束
} // namespace torch::inductor

// 头文件保护结束
#endif
```