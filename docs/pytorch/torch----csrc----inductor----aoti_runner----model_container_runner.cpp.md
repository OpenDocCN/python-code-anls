# `.\pytorch\torch\csrc\inductor\aoti_runner\model_container_runner.cpp`

```
#if !defined(C10_MOBILE) && !defined(ANDROID)
// 如果未定义 C10_MOBILE 和 ANDROID，则进行以下操作

#include <ATen/DynamicLibrary.h>
// 包含动态库加载的头文件

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
// 包含 AOTI 模型容器运行器和张量转换器的头文件

namespace torch::inductor {

AOTIModelContainerRunner::AOTIModelContainerRunner(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir) {
  model_so_ = std::make_unique<at::DynamicLibrary>(model_so_path.c_str());
  // 使用给定的模型动态链接库路径创建 DynamicLibrary 对象
  TORCH_CHECK(model_so_, "Failed to load model: ", model_so_path);
  // 检查动态库加载是否成功

  create_func_ = reinterpret_cast<decltype(create_func_)>(
      model_so_->sym("AOTInductorModelContainerCreateWithDevice"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于创建模型容器

  delete_func_ = reinterpret_cast<decltype(delete_func_)>(
      model_so_->sym("AOTInductorModelContainerDelete"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于删除模型容器

  get_num_outputs_func_ = reinterpret_cast<decltype(get_num_outputs_func_)>(
      model_so_->sym("AOTInductorModelContainerGetNumOutputs"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于获取模型输出数量

  run_func_ = reinterpret_cast<decltype(run_func_)>(
      model_so_->sym("AOTInductorModelContainerRun"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于运行模型

  get_num_constants_func_ = reinterpret_cast<decltype(get_num_constants_func_)>(
      model_so_->sym("AOTInductorModelContainerGetNumConstants"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于获取常量数量

  get_constant_name_func_ = reinterpret_cast<decltype(get_constant_name_func_)>(
      model_so_->sym("AOTInductorModelContainerGetConstantName"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于获取常量名称

  get_constant_original_fqn_func_ =
      reinterpret_cast<decltype(get_constant_original_fqn_func_)>(
          model_so_->sym("AOTInductorModelContainerGetConstantOriginalFQN"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于获取常量原始全限定名

  get_constant_dtype_func_ =
      reinterpret_cast<decltype(get_constant_dtype_func_)>(
          model_so_->sym("AOTInductorModelContainerGetConstantDtype"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于获取常量数据类型

  update_constant_buffer_func_ =
      reinterpret_cast<decltype(update_constant_buffer_func_)>(
          model_so_->sym("AOTInductorModelContainerUpdateConstantBuffer"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于更新常量缓冲区

  update_inactive_constant_buffer_func_ =
      reinterpret_cast<decltype(update_inactive_constant_buffer_func_)>(
          model_so_->sym(
              "AOTInductorModelContainerUpdateInactiveConstantBuffer"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于更新非活跃状态的常量缓冲区

  run_const_fold_func_ = reinterpret_cast<decltype(run_const_fold_func_)>(
      model_so_->sym("AOTInductorModelContainerRunConstantFolding"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于运行常量折叠操作

  swap_constant_buffer_func_ =
      reinterpret_cast<decltype(swap_constant_buffer_func_)>(
          model_so_->sym("AOTInductorModelContainerSwapConstantBuffer"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于交换常量缓冲区

  get_call_spec_func_ = reinterpret_cast<decltype(get_call_spec_func_)>(
      model_so_->sym("AOTInductorModelContainerGetCallSpec"));
  // 将动态库中的符号指针转换为对应的函数指针类型，用于获取调用规范

  AOTI_RUNTIME_ERROR_CODE_CHECK(create_func_(
      &container_handle_,
      num_models,
      device_str.c_str(),
      cubin_dir.empty() ? nullptr : cubin_dir.c_str()));
  // 使用转换后的函数指针调用创建模型容器的函数，并进行错误检查
}

AOTIModelContainerRunner::~AOTIModelContainerRunner() {
  AOTIRuntimeError result = delete_func_(container_handle_);
  // 使用转换后的函数指针调用删除模型容器的函数
  TORCH_CHECK(
      result == AOTI_RUNTIME_SUCCESS, "AOTInductorModelContainerDelete failed");
  // 检查删除操作是否成功
}

std::vector<at::Tensor> AOTIModelContainerRunner::run(


注释：
    std::vector<at::Tensor>& inputs,
    AOTInductorStreamHandle cuda_stream_handle) {
  // 使用输入张量列表 inputs，通过 AOTInductorStreamHandle 来执行模型推断

  auto input_handles =
      torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(inputs);
  // 将输入张量 inputs 转换为输入句柄 input_handles，用于模型推断

  // 对于输出，仅分配一个向量来保存返回的张量句柄，而不在此处分配实际的输出张量存储空间
  size_t num_outputs = 0;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_outputs_func_(container_handle_, &num_outputs));
  // 获取模型容器中的输出数量，并将结果保存在 num_outputs 中

  std::vector<AtenTensorHandle> output_handles(num_outputs);
  // 根据输出数量 num_outputs 创建一个存储输出张量句柄的向量 output_handles

  AOTI_RUNTIME_ERROR_CODE_CHECK(run_func_(
      container_handle_,
      input_handles.data(),
      input_handles.size(),
      output_handles.data(),
      output_handles.size(),
      cuda_stream_handle,
      proxy_executor_handle_));
  // 使用模型运行函数 run_func_ 执行模型推断，将输入句柄和输出句柄传递给模型容器，
  // 并使用指定的 CUDA 流处理器进行异步推断

  return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
      output_handles.data(), output_handles.size());
  // 从输出句柄中生成张量，并将其返回
}
// 返回一个无序映射，将常量名称映射到原始完全限定名称（FQN）
std::unordered_map<std::string, std::string> AOTIModelContainerRunner::
    getConstantNamesToOriginalFQNs() const {
  // 创建一个空的结果映射
  std::unordered_map<std::string, std::string> result;
  // 获取常量的数量
  size_t num_constants{0};
  // 调用运行时错误检查宏，获取常量数量
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_constants_func_(container_handle_, &num_constants));
  // 遍历所有常量
  for (size_t i = 0; i < num_constants; ++i) {
    // 常量名称和原始完全限定名称的指针，初始化为空
    const char* name{nullptr};
    const char* original_fqn{nullptr};
    // 调用运行时错误检查宏，获取常量名称
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_name_func_(container_handle_, i, &name));
    // 调用运行时错误检查宏，获取常量原始完全限定名称
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_original_fqn_func_(container_handle_, i, &original_fqn));
    // 将常量名称和原始完全限定名称插入结果映射中
    result.emplace(name, original_fqn);
  }
  // 返回填充好的结果映射
  return result;
}

// 返回一个无序映射，将常量名称映射到数据类型（int32_t）
std::unordered_map<std::string, int32_t> AOTIModelContainerRunner::
    getConstantNamesToDtypes() const {
  // 创建一个空的结果映射
  std::unordered_map<std::string, int32_t> result;
  // 获取常量的数量
  size_t num_constants{0};
  // 调用运行时错误检查宏，获取常量数量
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_constants_func_(container_handle_, &num_constants));
  // 遍历所有常量
  for (size_t i = 0; i < num_constants; ++i) {
    // 常量名称和数据类型（int32_t）的指针，初始化为空和零
    const char* name{nullptr};
    int32_t dtype{0};
    // 调用运行时错误检查宏，获取常量名称
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_name_func_(container_handle_, i, &name));
    // 调用运行时错误检查宏，获取常量数据类型（int32_t）
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_dtype_func_(container_handle_, i, &dtype));
    // 将常量名称和数据类型插入结果映射中
    result.emplace(name, dtype);
  }
  // 返回填充好的结果映射
  return result;
}

// 更新常量缓冲区，将给定的常量映射写入到容器中
void AOTIModelContainerRunner::update_constant_buffer(
    const TensorConstantMap& const_map,
    bool use_inactive,
    bool check_full_update) {
  // 调用运行时错误检查宏，更新常量缓冲区
  AOTI_RUNTIME_ERROR_CODE_CHECK(update_constant_buffer_func_(
      container_handle_,
      (AOTInductorConstantMapHandle)&const_map,
      use_inactive,
      check_full_update));
}

// 更新非活动状态的常量缓冲区，将给定的常量映射写入到容器中
void AOTIModelContainerRunner::update_inactive_constant_buffer(
    const TensorConstantMap& const_map) {
  // 调用运行时错误检查宏，更新非活动状态的常量缓冲区
  AOTI_RUNTIME_ERROR_CODE_CHECK(update_inactive_constant_buffer_func_(
      container_handle_, (AOTInductorConstantMapHandle)&const_map));
}

// 执行常量折叠运算，可能使用非活动状态的常量，并在CUDA流上执行
void AOTIModelContainerRunner::run_const_fold(
    bool use_inactive,
    AOTInductorStreamHandle cuda_stream_handle) {
  // 调用运行时错误检查宏，执行常量折叠运算
  AOTI_RUNTIME_ERROR_CODE_CHECK(run_const_fold_func_(
      container_handle_,
      use_inactive,
      cuda_stream_handle,
      proxy_executor_handle_));
}

// 交换常量缓冲区，可能触发内部状态更新
void AOTIModelContainerRunner::swap_constant_buffer() {
  // 调用运行时错误检查宏，交换常量缓冲区
  AOTI_RUNTIME_ERROR_CODE_CHECK(swap_constant_buffer_func_(container_handle_));
}

// 获取调用规范，返回输入和输出规范字符串的向量
std::vector<std::string> AOTIModelContainerRunner::get_call_spec() {
  // 输入和输出规范字符串的指针，初始化为空
  const char* in_spec;
  const char* out_spec;
  // 调用运行时错误检查宏，获取调用规范
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_call_spec_func_(container_handle_, &in_spec, &out_spec));
  // 返回包含输入和输出规范字符串的向量
  return {in_spec, out_spec};
}

// 命名空间结束标记，这是torch::inductor命名空间的结束
} // namespace torch::inductor
#endif
```