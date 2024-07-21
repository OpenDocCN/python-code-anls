# `.\pytorch\torch\csrc\inductor\aoti_eager\kernel_meta_info.h`

```
#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 SymIntArrayRef 类型的头文件
#include <c10/core/SymIntArrayRef.h>

// 包含标准库中的字符串处理头文件
#include <string>

// torch::inductor 命名空间开始
namespace torch::inductor {

// 表示 ATen 操作的 AOTI 实现中，输入张量的元数据将缓存到磁盘以加速下一次运行。
// TensorMetadata 结构体用于表示每个输入张量的元数据，包括张量是否为符号化、数据类型、设备、大小和步长。
// 当输入张量的元数据与缓存的元数据相同时，将加载并执行缓存的内核库。否则，将再次调用 AOT Inductor 生成内核库。
// 除了 TensorMetadata，我们还为每个输入张量构建了 guard/TensorCheck，以支持符号化形状。
// 我们打算利用 TensorCheck 来确定正确的内核，而不是通过 TensorMetadata 比较。
// 假设一个操作有一个输入张量和两个内核：
//   kernel1: TensorMetadata(is_symbolic=false, dtype=Float, device=CPU, sizes=[s0, s1, s2], strides=[s1 * s2, s2, 1])
//   kernel2: TensorMetadata(is_symbolic=false, dtype=Float, device=CPU, sizes=[3, s1, s2], strides=[s1 * s2, s2, 1])
// 如果传递一个大小为 [3, 4, 5] 的张量给操作，kernel1 和 kernel2 都支持这种张量形状。
// 在这种情况下，我们需要使用 TensorCheck 加上一些启发式规则来确定正确的内核。
struct TensorMetadata {
  // 指示张量是否为符号化，未来可能通过 sizes_ 和 strides_ 推断。
  bool is_symbolic_;
  // 张量的数据类型（对于标量，我们将其包装为标量张量）
  c10::ScalarType dtype_;
  // 具体的标量值。为带标量参数的操作提供服务。
  c10::IValue scalar_value_;
  // 张量的设备。
  c10::Device device_;
  // 张量的大小。当前仅支持静态形状，并使用 int64_t 表示大小。将来，我们将创建符号大小并使用 SymInt 表示以支持符号化形状。
  std::vector<int64_t> sizes_;
  // 张量的步长。为支持符号化形状，它与 sizes_ 相同。
  std::vector<int64_t> strides_;

  // 构造函数，从给定的 ATen 张量创建 TensorMetadata 对象。
  TensorMetadata(const at::Tensor& src_tensor);
  // 构造函数，初始化所有成员变量。
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::Device device,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);
  // 构造函数，包括标量值参数。
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::IValue scalar_value,
      c10::Device device,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);

  // 重载运算符==，用于比较两个 TensorMetadata 对象是否相等。
  bool operator==(const TensorMetadata& other) const;
};

// 哈希结构体，用于计算 TensorMetadata 对象的哈希值。
struct TensorMetadataHash {
  size_t operator()(const TensorMetadata&) const;
};

// AOTI 内核元数据类型定义，是 TensorMetadata 的向量。
using AOTIKernelMetadata = std::vector<TensorMetadata>;

// 哈希结构体，用于计算 AOTIKernelMetadata 对象的哈希值。
struct AOTIKernelMetadataHash {
  size_t operator()(const AOTIKernelMetadata&) const;
};

} // namespace torch::inductor
#endif
```