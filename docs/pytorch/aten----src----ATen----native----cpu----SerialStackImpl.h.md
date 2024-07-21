# `.\pytorch\aten\src\ATen\native\cpu\SerialStackImpl.h`

```py
// Copyright 2004-present Facebook. All Rights Reserved.
// 版权声明，指出此代码版权归Facebook所有，保留所有权利

#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/Tensor.h>
// 引入ATen张量的核心头文件

#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
// 引入其他依赖的头文件

namespace at { namespace native { namespace detail {

struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
      : data_ptr(t.data_ptr()), inner_size(t.sizes()[dim] * inner) {}
};
// 输入元信息结构体，用于存储输入张量的数据指针和内部大小信息

// This kernel is used by two TensorList types:
// 1. stack_serial_kernel uses at::ArrayRef<Tensor>
// 2. Static runtime calls this kernel directly (csrc/jit/runtime/static/ops.cpp) with
//    ProcessedNodeInputWrapper.
// When making changes, make sure that they are compatible with both types!
template <typename scalar_t, typename TensorListType>
void stack_serial_kernel_impl(Tensor& result, TensorListType tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim <= result.dim(),
      "dim out of range in stack_serial_kernel_impl");
  // 断言检查，确保维度dim在有效范围内

  int64_t outer =
      result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  // 计算结果张量的外层大小

  scalar_t* result_data = result.data_ptr<scalar_t>();
  // 获取结果张量的数据指针

  int64_t ninputs = tensors.size();
  // 输入张量列表的大小

  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  // 创建存储输入元信息的向量，并预留空间

  for (const auto& tensor : tensors) {
    inputs.emplace_back(tensor, dim, tensor.strides()[dim]);
  }
  // 遍历输入张量列表，构造每个张量的元信息

  using Vec = vec::Vectorized<scalar_t>;
  // 使用向量化模板类Vec，用于向量化操作

  scalar_t* result_ptr = result_data;
  // 初始化结果指针

  for (const auto i : c10::irange(outer)) {
    for (const auto j : c10::irange(ninputs)) {
      int64_t local_inner = inputs[j].inner_size;
      // 获取当前输入张量的内部大小

      scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;
      // 计算当前输入张量在当前外层迭代下的起始指针位置

      if (local_inner < Vec::size()) {
        // 如果内部大小小于向量化宽度

        for (const auto k : c10::irange(local_inner)) {
          result_ptr[k] = input_ptr[k];
        }
        // 逐元素复制数据到结果张量中
      } else {
        vec::map(
            [](Vec x) { return x; }, result_ptr, input_ptr, local_inner);
        // 使用向量化操作复制数据到结果张量中
      }

      result_ptr += local_inner;
      // 更新结果指针位置
    }
  }
}

// Checks to see whether native stack can be invoked under these conditions:
// - result and input tensors are contiguous
// - only one thread is used
// - no type promotion has to occur
// - tensors dtype is Double or Float
template <typename TensorListType>
// 检查是否可以使用本地串行堆栈实现，返回布尔值
bool can_use_native_serial_stack_impl(Tensor& result, TensorListType tensors, int64_t dim) {
  // 检查张量列表是否为空，至少应包含一个张量
  TORCH_CHECK(tensors.size() > 0, "expected a non-empty list of Tensors");
  // 获取第一个张量作为参考
  const Tensor& first_tensor = tensors[0];
  
  // 检查堆叠维度是否在第一个张量的维度范围内 [0, firstTensor.dim())
  // dim == firstTensor.dim() 是有效输入，但会通过默认代码路径处理，使用unsqueeze
  if (dim >= first_tensor.dim()) return false;
  
  // 如果第一个张量是一维且元素数为零，则本地堆栈不适用
  if (first_tensor.numel() == 0 && first_tensor.dim() == 1) return false;
  
  // 检查结果张量的数据类型是否与第一个张量一致，不允许类型提升
  if (result.dtype() != first_tensor.dtype()) return false;

  // 获取第一个张量推荐的内存格式和数据类型
  auto first_tensor_mem_format = first_tensor.suggest_memory_format();
  ScalarType dtype = first_tensor.scalar_type();

  // 如果结果张量不是指定的内存格式，返回 false
  if (!result.is_contiguous(first_tensor_mem_format)) {
    return false;
  }

  // 快速路径仅适用于 Double 和 Float 数据类型
  if (dtype != ScalarType::Double && dtype != ScalarType::Float) {
    return false;
  }

  // 检查剩余输入张量
  auto const &first_tensor_shape = first_tensor.sizes();
  for (const auto i : c10::irange(1, tensors.size())) {
    auto const &tensor = tensors[i];
    // 检查每个张量的大小是否与第一个张量相同
    TORCH_CHECK(tensors[i].sizes() == first_tensor.sizes(),
      "stack expects each tensor to be equal size, but got ", first_tensor_shape,
      " at entry 0 and ", tensor.sizes(), " at entry ", i);

    // 每个张量必须是连续的，并且大小和步长必须相同，数据类型不能有提升
    if (!tensor.is_contiguous(first_tensor_mem_format) ||
      tensor.strides() != first_tensor.strides() ||
      tensor.dtype() != dtype) {
      return false;
    }
  }

  // 仅在不值得使用多线程或只有一个线程时，应使用快速本地堆栈
  // 注意，此处不检查 result.numel()，因为可能尚未调整大小，希望将该成本推迟到后面。
  int64_t numel_in_stack = first_tensor.numel() * tensors.size();
  return numel_in_stack < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
}

// 模板特化，用于指定是否应跳过重叠检查
template <typename TensorListType>
struct CanUseNativeSerialStack<TensorListType, false> {
  // 调用函数，检查输入张量是否可以使用本地串行堆栈，并进行重叠检查
  static bool call(Tensor& result, TensorListType tensors, int64_t dim) {
    // 输入张量不能与输出张量重叠
    for (const auto i : c10::irange(tensors.size())) {
      auto lap = at::get_overlap_status(result, tensors[i]);
      TORCH_CHECK(lap != at::MemOverlapStatus::Partial &&
          lap != at::MemOverlapStatus::Full, 0,
          "unsupported operation: the input tensors cannot refer to any of the "
          "output memory locations. Found overlap in input tensor ", i);
    }

    // 调用实现函数进行堆栈操作的检查
    return can_use_native_serial_stack_impl(result, tensors, dim);
  }
};
// 定义一个模板结构体 CanUseNativeSerialStack，用于在编译时根据类型参数 TensorListType 和常量 true 进行特化
struct CanUseNativeSerialStack<TensorListType, true> {
  // 定义静态成员函数 call，接受参数 result（引用类型的 Tensor）、tensors（类型为 TensorListType 的参数列表）、dim（整数类型）
  static bool call(Tensor& result, TensorListType tensors, int64_t dim) {
    // 调用实现函数 can_use_native_serial_stack_impl，传入 result、tensors 和 dim，并返回其结果
    return can_use_native_serial_stack_impl(result, tensors, dim);
  }
};

// 结束 at::native::detail 命名空间
}}}  // namespace at::native::detail
```