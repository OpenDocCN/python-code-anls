# `.\pytorch\torch\csrc\inductor\aoti_eager\kernel_meta_info.cpp`

```
#if !defined(C10_MOBILE) && !defined(ANDROID)
// 如果未定义 C10_MOBILE 和 ANDROID，则编译以下代码块

#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
// 包含 kernel_meta_info.h 文件，用于引入相关的头文件

#include <iostream>
// 包含 iostream 头文件，用于标准输入输出流操作

namespace torch::inductor {
// 进入 torch::inductor 命名空间

TensorMetadata::TensorMetadata(const at::Tensor& src_tensor)
    : is_symbolic_(false),   // 初始化成员变量 is_symbolic_ 为 false
      device_(src_tensor.device()),   // 初始化成员变量 device_ 为 src_tensor 的设备信息
      sizes_(src_tensor.sizes().vec()),   // 初始化成员变量 sizes_ 为 src_tensor 的大小信息向量
      strides_(src_tensor.sizes().vec()) {}   // 初始化成员变量 strides_ 为 src_tensor 的步长信息向量

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides)
    : is_symbolic_(is_symbolic),   // 初始化成员变量 is_symbolic_
      dtype_(dtype),   // 初始化成员变量 dtype_
      scalar_value_((float)1.0),   // 初始化成员变量 scalar_value_，设置为浮点数 1.0
      device_(device),   // 初始化成员变量 device_
      sizes_(sizes),   // 初始化成员变量 sizes_
      strides_(strides) {   // 初始化成员变量 strides_
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
  // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 断言，如果 is_symbolic_ 为 true，则输出错误信息
}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::IValue scalar_value,
    c10::Device device,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides)
    : is_symbolic_(is_symbolic),   // 初始化成员变量 is_symbolic_
      dtype_(dtype),   // 初始化成员变量 dtype_
      scalar_value_(scalar_value),   // 初始化成员变量 scalar_value_
      device_(device),   // 初始化成员变量 device_
      sizes_(sizes),   // 初始化成员变量 sizes_
      strides_(strides) {   // 初始化成员变量 strides_
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
  // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 断言，如果 is_symbolic_ 为 true，则输出错误信息
}

bool TensorMetadata::operator==(const TensorMetadata& other) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
  // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 断言，如果 is_symbolic_ 为 true，则输出错误信息
  return this->is_symbolic_ == other.is_symbolic_ &&
      this->dtype_ == other.dtype_ &&
      this->scalar_value_ == other.scalar_value_ &&
      this->device_.type() == other.device_.type() &&
      this->sizes_ == other.sizes_ && this->strides_ == other.strides_;
  // 比较两个 TensorMetadata 对象的所有成员变量，返回比较结果
}

std::ostream& operator<<(
    std::ostream& stream,
    const TensorMetadata& tensor_metadata) {
  stream << "is_symbolic_: " << tensor_metadata.is_symbolic_ << std::endl;
  // 输出 tensor_metadata 的 is_symbolic_ 成员变量
  stream << "dtype_: " << tensor_metadata.dtype_ << std::endl;
  // 输出 tensor_metadata 的 dtype_ 成员变量
  stream << "scalar_value_: " << tensor_metadata.scalar_value_.type()->str()
         << "(" << tensor_metadata.scalar_value_ << ")" << std::endl;
  // 输出 tensor_metadata 的 scalar_value_ 成员变量的类型和值
  stream << "device_: " << tensor_metadata.device_ << std::endl;
  // 输出 tensor_metadata 的 device_ 成员变量
  stream << "sizes_: ";
  for (const auto& size : tensor_metadata.sizes_) {
    stream << size << " ";
  }
  stream << std::endl;
  // 输出 tensor_metadata 的 sizes_ 成员变量
  stream << "strides_: ";
  for (const auto& stride : tensor_metadata.strides_) {
    stream << stride << " ";
  }
  stream << std::endl;
  // 输出 tensor_metadata 的 strides_ 成员变量
  return stream;
  // 返回输出流对象
}

size_t TensorMetadataHash::operator()(
    const TensorMetadata& tensor_metadata) const {
  auto hash = std::hash<bool>()(tensor_metadata.is_symbolic_);
  // 使用 tensor_metadata 的 is_symbolic_ 成员变量生成哈希值
  hash = c10::hash_combine(
      hash, std::hash<c10::ScalarType>()(tensor_metadata.dtype_));
  // 使用 tensor_metadata 的 dtype_ 成员变量生成哈希值
  hash =
      c10::hash_combine(hash, c10::IValue::hash(tensor_metadata.scalar_value_));
  // 使用 tensor_metadata 的 scalar_value_ 成员变量生成哈希值
  hash = c10::hash_combine(
      hash, std::hash<c10::DeviceType>()(tensor_metadata.device_.type()));
  // 使用 tensor_metadata 的 device_ 成员变量的类型生成哈希值

  for (auto& e : tensor_metadata.sizes_) {
    // 遍历 tensor_metadata 中的每个元素 e 的哈希值，将其与当前 hash 值组合后更新 hash
    hash = c10::hash_combine(hash, std::hash<int64_t>()(e));
  }

  // 遍历 tensor_metadata 中 strides_ 容器的每个元素 e，计算其哈希值并更新 hash
  for (auto& e : tensor_metadata.strides_) {
    hash = c10::hash_combine(hash, std::hash<int64_t>()(e));
  }
  
  // 返回计算得到的最终 hash 值作为函数的结果
  return hash;
}

# 结束 AOTIKernelMetadataHash 结构体的 operator() 方法的定义
size_t AOTIKernelMetadataHash::operator()(const AOTIKernelMetadata& aoti_kernel_metadata) const {
    # 初始化哈希值为 0
    size_t hash = 0;
    # 遍历 AOTIKernelMetadata 结构体中的每一个元素
    for (auto& e : aoti_kernel_metadata) {
        # 使用 TensorMetadataHash 结构体的哈希函数计算当前元素 e 的哈希值，并与当前的 hash 值进行结合
        hash = c10::hash_combine(hash, TensorMetadataHash()(e));
    }
    # 返回计算得到的最终哈希值
    return hash;
}

# 结束 torch::inductor 命名空间的定义
} // namespace torch::inductor

# 结束预处理指令的条件编译部分
#endif
```