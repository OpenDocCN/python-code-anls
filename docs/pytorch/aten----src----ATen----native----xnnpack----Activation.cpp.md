# `.\pytorch\aten\src\ATen\native\xnnpack\Activation.cpp`

```
#ifdef USE_XNNPACK
// 包含 XNNPACK 相关的头文件
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/utils/Factory.h>

// 进入 ATen 的 native::xnnpack 命名空间
namespace at::native::xnnpack {

// 检查是否可以使用硬激活函数 hardswish
bool use_hardswish(
  const Tensor& input) {
  // 检查 XNNPACK 是否可用，输入张量维度至少为 1，设备为 CPU，数据类型为 float32，不需要梯度，并返回 true
  return xnnpack::available() &&
          (1 <= input.ndimension()) &&
          (input.device().is_cpu()) &&
          (kFloat == input.scalar_type()) &&
          !input.requires_grad() &&
           true;
}

// 实现硬激活函数 hardswish
static Tensor& hardswish_impl(Tensor& input, Tensor& output) {
  using namespace internal;

  // 创建硬激活函数的 XNNPACK 运算符
  xnn_operator_t hardswish_op{};
  const xnn_status create_status = xnn_create_hardswish_nc_f32(
    0, // flags
    &hardswish_op);

  // 检查硬激活函数创建是否成功
  TORCH_CHECK(
    xnn_status_success == create_status,
    "xnn_create_hardswish_nc_f32 failed!");

  // 在作用域内管理硬激活函数运算符的生命周期
  Operator hardswish_scoped_op(hardswish_op);

  // 重新调整硬激活函数的输入输出格式
  const xnn_status reshape_status = xnn_reshape_hardswish_nc_f32(
    hardswish_op,
    input.numel(),  // Batch
    1, // channels
    1, // input stride
    1, // output stride
    caffe2::pthreadpool_());  // threadpool

  // 检查重新调整操作是否成功
  TORCH_CHECK(
    xnn_status_success == reshape_status,
    "xnn_reshape_hardswish_nc_f32 failed!");

  // 设置硬激活函数的输入和输出数据指针
  const xnn_status setup_status = xnn_setup_hardswish_nc_f32(
    hardswish_op,
    input.data_ptr<float>(),
    output.data_ptr<float>());

  // 检查设置操作是否成功
  TORCH_CHECK(
    xnn_status_success == setup_status,
    "xnn_setup_hardswish_nc_f32 failed!");

  // 运行硬激活函数的 XNNPACK 运算符
  const xnn_status run_status = xnn_run_operator(
    hardswish_op,
    caffe2::pthreadpool_());  // threadpool

  // 内部断言确保运行操作成功
  TORCH_INTERNAL_ASSERT(
    xnn_status_success == run_status,
    "xnn_run_operator failed!");

  // 返回输出张量
  return output;
}

// 对外接口，应用硬激活函数到输入张量并返回结果
Tensor hardswish(const Tensor& input) {
  // 如果需要，为输入张量分配连续内存
  Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
    input, input.suggest_memory_format());

  // 根据填充后的输入张量尺寸和数据类型，创建输出张量
  Tensor output = mobile::empty_with_tail_padding(
    padded_input.sizes(),
    padded_input.options().dtype(),
    input.suggest_memory_format(),
    padded_input.opt_names());

  // 应用硬激活函数到填充后的输入张量，并返回连续的输出张量
  hardswish_impl(padded_input, output);
  return output.contiguous(input.suggest_memory_format());
}

// 对外接口，原地应用硬激活函数到输入张量并返回自身引用
Tensor& hardswish_(Tensor& input) {
  // 如果需要，为输入张量分配连续内存
  Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
    input, input.suggest_memory_format());

  // 如果输入张量已经是连续的并且已填充，不需要分配输出张量
  if (input.data_ptr() == padded_input.data_ptr()) {
    // 原地应用硬激活函数到输入张量并返回自身引用
    hardswish_impl(input, input);
    return input;
  } else {
    // 否则，根据填充后的输入张量尺寸和数据类型，创建输出张量
    Tensor output = mobile::empty_with_tail_padding(
      padded_input.sizes(),
      padded_input.options().dtype(),
      input.suggest_memory_format(),
      padded_input.opt_names());
    // 应用硬激活函数到填充后的输入张量，并将结果复制到原始输入张量中
    hardswish_impl(padded_input, output);
    return input.copy_(output);
  }
}

} // namespace at::native::xnnpack
#endif /* USE_XNNPACK */
```