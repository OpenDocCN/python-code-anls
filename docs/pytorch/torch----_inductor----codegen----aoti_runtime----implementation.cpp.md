# `.\pytorch\torch\_inductor\codegen\aoti_runtime\implementation.cpp`

```
// NOTE: Like interface.cpp, this file will be copied into AOTInductor
// generated output. This file is intended to keep implementation
// details separate from the implementation of the AOTI public
// interface. Note also that #includes should go into interface.cpp
// for simplicity of maintenance.

namespace torch {
namespace aot_inductor {

// 将输出的 ArrayRefTensor 转换为 AtenTensorHandle
template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor(); // 执行昂贵的复制操作，将输出复制到张量中
}

// 辅助函数：将多个输出转换为对应的多个 AtenTensorHandle
template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}

// 将多个输出转换为对应的多个 AtenTensorHandle
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

// 将 AtenTensorHandle 转换为 ArrayRefTensor
template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &data_ptr)); // 获取数据指针
  int64_t dim;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &dim)); // 获取张量维度
  int64_t numel;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(handle, &numel)); // 获取张量元素个数
  int64_t* sizes;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes)); // 获取张量尺寸数组指针
  int64_t* strides;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides)); // 获取张量步幅数组指针
  int32_t dtype;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &dtype)); // 获取张量数据类型
  int32_t device_type;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(handle, &device_type)); // 获取设备类型
  int32_t device_index;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &device_index)); // 获取设备索引

  // 使用获取的信息构造 ArrayRefTensor
  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel), // 数据指针和元素个数构成的迷你数组引用
      MiniArrayRef<const int64_t>(sizes, dim), // 尺寸数组和维度构成的迷你数组引用
      MiniArrayRef<const int64_t>(strides, dim), // 步幅数组和维度构成的迷你数组引用
      device_type, // 设备类型
      device_index); // 设备索引
}

// 辅助函数：将多个 AtenTensorHandle 转换为对应的多个 ArrayRefTensor
template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

// 将多个 AtenTensorHandle 转换为对应的多个 ArrayRefTensor
template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

// 断言 ArrayRefTensor 的元素个数是否与给定值相符
template <typename T>
void assert_numel(const ArrayRefTensor<T>& tensor, int64_t numel) {
  if (tensor.numel() != numel) {
    std::stringstream err;
    err << "incorrect numel for input tensor. expected " << numel << ", got " << tensor.numel();
    // 输出错误信息
    throw std::runtime_error(err.str());
  }
}
} // namespace aot_inductor
} // namespace torch
    // 抛出一个 std::runtime_error 异常，异常信息由 err.str() 提供
    throw std::runtime_error(err.str());
}
} // namespace aot_inductor
} // namespace torch
```