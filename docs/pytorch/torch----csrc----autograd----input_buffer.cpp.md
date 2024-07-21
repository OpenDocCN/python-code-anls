# `.\pytorch\torch\csrc\autograd\input_buffer.cpp`

```
// 包含 Torch 的自动求导库中定义的输入缓冲区相关头文件
#include <torch/csrc/autograd/input_buffer.h>

// 包含 ATen 库中定义的各种实用工具和特定张量类型的头文件
#include <ATen/CachedTensorUtils.h>
#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/SparseTensorUtils.h>

// 包含 C10 核心库中的设备守护、事件和流守护相关头文件
#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Optional.h>

// 包含标准库头文件
#include <cstddef>
#include <utility>
#include <vector>

// Torch 自动求导命名空间
namespace torch {
namespace autograd {

// 匿名命名空间，用于隐藏实现细节并提供局部作用域
namespace {

// 函数：记录流到任意变量的实现
// 根据变量的不同类型记录其数据指针到指定流中
void record_stream_any_impl(Variable& var, c10::Stream& stream) {
  // 使用变量的设备类型创建虚拟设备守护
  const auto guard = c10::impl::VirtualGuardImpl(device_of(var).value().type());

  // 如果变量是批处理张量
  if (C10_UNLIKELY(at::isBatchedTensor(var))) {
    auto* impl = at::maybeGetBatchedImpl(var);
    if (impl) {
      // 记录批处理实现的数据指针到流中
      guard.recordDataPtrOnStream(impl->value().storage().data_ptr(), stream);
    } else {
      // 断言失败，预期为批处理张量
      TORCH_INTERNAL_ASSERT(false, "Expected batched tensor");
    }
  } else {
    // 根据变量布局类型选择记录数据指针到流中
    switch (var.layout()) {
      // 稀疏张量的不同类型布局
      case c10::kSparseCsr:
      case c10::kSparseCsc:
      case c10::kSparseBsr:
      case c10::kSparseBsc: {
        auto* impl = at::sparse_csr::get_sparse_csr_impl(var);
        guard.recordDataPtrOnStream(
            impl->values().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->compressed_indices().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->plain_indices().storage().data_ptr(), stream);
        break;
      }
      // 普通稀疏张量
      case c10::kSparse: {
        auto* impl = at::sparse::get_sparse_impl(var);
        guard.recordDataPtrOnStream(
            impl->values().storage().data_ptr(), stream);
        guard.recordDataPtrOnStream(
            impl->indices().storage().data_ptr(), stream);
        break;
      }
      // 布局为 Strided 的普通张量
      case c10::kStrided:
        guard.recordDataPtrOnStream(var.storage().data_ptr(), stream);
        break;
      // 未知布局类型，抛出断言错误
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Unknown layout in record_stream_any_impl");
    }
  }
}

// 函数：判断变量是否可以就地累加
// 根据变量的特性判断是否可以在累加操作中就地处理
bool can_accumulate_inplace(const Variable& v) {
  return (
      // `v` 是常规张量
      !(at::isTensorSubclassLike(v) || v._is_zerotensor() || v.is_nested()) &&

      // 具有有利的内存布局
      v.is_non_overlapping_and_dense() &&

      // 我们拥有最后一个引用
      at::caching::adjusted_use_count(v) == 1 && v.has_storage() &&
      v.storage().use_count() == 1);
}

} // 匿名命名空间结束

// 静态函数：累加操作
// 在给定位置将变量向量中的变量累加
static void accumulate(
    std::vector<Variable>& buffer,
    const size_t pos,
  Variable&& var) {
// 断言：确保 `pos` 小于 `buffer` 的大小，以防止访问越界
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
// 获取 `buffer` 中索引为 `pos` 的旧变量的引用
  auto& old_var = buffer[pos];
// 如果我们是 `old_var` 的最后一个引用，并且其存储被使用，我们尝试重用它来存储输出。
// （或者，如果 `old_var` 是稀疏的，则 `var` 成为候选的输出 Tensor。）我们只在以下情况下执行这些操作：
//  1) GradMode 被禁用，因为 Autograd 对于原地操作有特殊处理，我们不希望触发这些处理。
//
//  2) 我们是最后一个引用。
//     （`.use_count` 和 `.storage().use_count()` 都为一）
//
//  3) 候选 Tensor 是连续的、非重叠的、稠密的，以及其他标准的 Tensor。
//
//  4) 候选 Tensor 是可变的。目前只有 ZeroTensor 是不可变的。
//
//  5) 另一个 Tensor 不是 Tensor 的子类（除了稀疏类型），因为预测任意子类的语义行为是困难的。
//
// NOLINTNEXTLINE(bugprone-branch-clone)
// 如果 GradMode 启用，则将 `old_var` 和 `var` 相加，并将结果存储在 `buffer[pos]` 中
  if (at::GradMode::is_enabled()) {
    buffer[pos] = old_var + var;
// 否则，如果 `old_var` 是稀疏的或者是稀疏 CSR 的，则根据条件进行选择性地累加 `var` 到 `old_var`
  } else if (
      // ATen 无法正确处理稀疏的加法...
      old_var.is_sparse() || old_var.is_sparse_csr()) {
    if (can_accumulate_inplace(var)) {
      buffer[pos] = var.add_(old_var);
    } else {
      buffer[pos] = var + old_var;
    }
// 否则，如果可以就地累加 `old_var`，并且 `var` 不是 Tensor 的子类，则执行就地加法操作
  } else if (
      can_accumulate_inplace(old_var) && !at::isTensorSubclassLike(var)) {
    buffer[pos] = old_var.add_(var);
// 否则，执行普通的加法操作，将 `old_var` 和 `var` 相加，并将结果存储在 `buffer[pos]` 中
  } else {
    buffer[pos] = old_var + var;
  }
}
// 结束 InputBuffer 类中的 add 方法的定义

void InputBuffer::add(
    size_t pos,
    Variable&& var,
    const std::optional<c10::Stream>& opt_producer_stream,
    const std::optional<c10::Stream>& opt_consumer_stream) {
  // 使用内部断言确保位置 pos 在缓冲区大小内
  TORCH_INTERNAL_ASSERT(pos < buffer.size());
  // 如果变量 var 未定义，则直接返回，不执行后续操作
  if (!var.defined()) {
    return;
  }

  // 切换到累积设备的逻辑说明：
  // 累积设备的选择是：
  //  (1) 如果 var 不是 CUDA 或者 privateuse1 变量，累积发生在 var 的设备上。
  //  (2) 如果 var 是 CUDA 或者 privateuse1 变量，并且它、消费者和生产者共享相同的设备：
  //       (2a) 使用消费者的流作为累积流
  //       (2b) 如果累积流与生产者流不同，同步累积流与生产者流
  //       (2c) 累积操作。
  //  (3) 如果 var 是 CUDA 或者 privateuse1 变量，并且它与消费者共享设备但不与生产者共享：
  //       (3a) 使用消费者的流作为累积流
  //       (3b) 将累积流与消费者设备的默认流同步
  //       (3c) 累积操作。
  //  (4) 如果 var 是 CUDA 或者 privateuse1 变量，并且它与生产者共享设备但不与消费者共享：
  //       (4a) 使用生产者设备的默认流作为累积流
  //       (4b) 将累积流与生产者流同步
  //       (4c) 累积操作。
  //  (5) 如果 var 是 CUDA 或者 privateuse1 变量，并且它既不与消费者也不与生产者共享设备：
  //       累积发生在 var 设备的默认流上。

  // 使用内部断言确认 var 的设备是否存在
  TORCH_INTERNAL_ASSERT(device_of(var));
  // 初始化一个空的累积流的可选值
  std::optional<c10::Stream> opt_accumulate_stream = c10::nullopt;
  // 获取 var 的设备类型
  const auto device_type = device_of(var).value().type();
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  // 如果 var 是 CUDA 或者 privateuse1 变量
  if (device_of(var)->is_cuda() || device_of(var)->is_privateuseone()) {
    // 检查是否在生产者设备上
    const auto on_producer =
        opt_producer_stream && device_of(var) == opt_producer_stream->device();
    // 检查是否在消费者设备上
    const auto on_consumer =
        opt_consumer_stream && device_of(var) == opt_consumer_stream->device();

    // 如果同时在生产者和消费者设备上
    if (on_producer && on_consumer) {
      // (2a) 使用消费者的流作为累积流
      opt_accumulate_stream = opt_consumer_stream;
      // 如果累积流与生产者流不同，则同步它们
      if (opt_accumulate_stream != opt_producer_stream) {
        // (2b) 创建一个事件对象，记录生产者流
        auto event = c10::Event{device_type};
        event.record(*opt_producer_stream);
        // 等待事件完成
        opt_accumulate_stream->wait(event);
        // 记录任何流的实现
        record_stream_any_impl(var, *opt_accumulate_stream);
      }
      // 继续进行累积操作
      // (2c) Accumulates.
      // (3), (3a), (3b), (3c), (4), (4a), (4b), (4c) 在此处省略，因为未在注释范围内
    }
  }
  } else {
    // 如果不在同一个设备上
    std::optional<c10::Stream> opt_sync_stream = c10::nullopt;
    // 创建一个虚拟的设备守卫对象
    const auto guard = c10::impl::VirtualGuardImpl{device_type};
    if (on_consumer && !on_producer) {
      // (3a) 情况：在consumer设备上，不在producer设备上
      opt_accumulate_stream = opt_consumer_stream;
      // 获取consumer流的默认同步流
      opt_sync_stream = guard.getDefaultStream(opt_consumer_stream->device());
    } else if (on_producer && !on_consumer) {
      // (4a) 情况：在producer设备上，不在consumer设备上
      opt_accumulate_stream =
          guard.getDefaultStream(opt_producer_stream->device());
      // 设置同步流为producer流
      opt_sync_stream = opt_producer_stream;
    } else {
      // (5) 情况：在相同设备上或者不清楚设备类型时
      // 使用变量的设备来获取默认流，并设置为累积流
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      opt_accumulate_stream = guard.getDefaultStream(*device_of(var));
    }
    if (opt_sync_stream && (opt_accumulate_stream != opt_sync_stream)) {
      // (3b), (4b) 情况：如果存在同步流且不是同一个流
      // 使用同步流的设备保护
      c10::OptionalDeviceGuard device_guard{opt_sync_stream->device()};
      // 创建事件对象并记录同步流
      auto event = c10::Event{device_type};
      event.record(*opt_sync_stream);
      // 等待累积流完成同步事件
      opt_accumulate_stream->wait(event);
      // 创建虚拟设备守卫对象
      const auto guard = c10::impl::VirtualGuardImpl(device_type);
      // 记录任何实现中的流
      record_stream_any_impl(var, *opt_accumulate_stream);
    }
  }

  auto& old_var = buffer[pos];
  if (!old_var.defined()) {
    // 如果旧变量未定义，将当前变量移动到缓冲区位置
    buffer[pos] = std::move(var);
  } else {
    if (opt_accumulate_stream) {
      // 如果存在累积流，使用流守卫对象
      c10::OptionalStreamGuard stream_guard{opt_accumulate_stream};
      // 在当前位置上累积变量
      accumulate(buffer, pos, std::move(var));
    } else {
      // (1) 非CUDA或私有使用变量
      // 在变量的设备上进行累积操作
      c10::OptionalDeviceGuard device_guard{device_of(var)};
      accumulate(buffer, pos, std::move(var));
    }
  }
}

// 定义 InputBuffer 类的成员函数 device()，返回首个非 CPU 设备的设备类型
auto InputBuffer::device() const -> at::Device {
  // 由于选择第一个非 CPU 的张量，所以不支持混合设备类型操作（例如同时 CUDA 和 XLA）。
  // 不过这种情况非常罕见，所以我们不用担心。
  for (auto& var : buffer) {
    if (var.defined()) {  // 检查张量是否已定义
      auto device = var.device();  // 获取张量的设备类型
      if (device.type() != at::kCPU) {  // 如果设备类型不是 CPU
        return device;  // 返回该设备类型
      }
    }
  }
  // 如果所有张量均为 CPU 类型，则返回 CPU 设备类型
  return at::kCPU;
}

// 定义 InputBuffer 类的成员函数 variables()，接受右值引用 g，并返回移动后的 buffer 向量
auto InputBuffer::variables(InputBuffer&& g) -> std::vector<Variable> {
  std::vector<Variable> result = std::move(g.buffer);  // 移动 g 的 buffer 到 result
  return result;  // 返回移动后的 result 向量
}

} // namespace autograd
} // namespace torch
```