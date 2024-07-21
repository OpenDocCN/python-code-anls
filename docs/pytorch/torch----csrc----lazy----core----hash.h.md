# `.\pytorch\torch\csrc\lazy\core\hash.h`

```
/**
 * Hash utils in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/util.h
 */

#pragma once

#include <ATen/Tensor.h>         // 引入 ATen 库中的 Tensor 类
#include <c10/core/Scalar.h>     // 引入 c10 库中的 Scalar 类
#include <c10/util/int128.h>     // 引入 c10 库中的 int128 实用工具
#include <torch/csrc/Export.h>   // 引入 Torch 导出相关的头文件
#include <cstring>               // 引入 C 字符串操作相关的头文件
#include <set>                   // 引入 set 容器相关的头文件
#include <string>                // 引入 string 类
#include <vector>                // 引入 vector 容器相关的头文件

namespace torch {
namespace lazy {

using size_t = std::size_t;      // 使用 size_t 作为 std::size_t 的别名

/**
 * hash_t 类继承自 c10::uint128，提供了不同类型的构造函数和操作符重载
 * 用于表示 128 位的哈希值
 */
class TORCH_API hash_t : public c10::uint128 {
 public:
  hash_t(int8_t val) : uint128(static_cast<uint32_t>(val)) {}     // 使用 int8_t 类型构造 hash_t 对象
  hash_t(int16_t val) : uint128(static_cast<uint32_t>(val)) {}    // 使用 int16_t 类型构造 hash_t 对象
  hash_t(int32_t val) : uint128(static_cast<uint32_t>(val)) {}    // 使用 int32_t 类型构造 hash_t 对象
  hash_t(int64_t val) : uint128(static_cast<uint64_t>(val)) {}    // 使用 int64_t 类型构造 hash_t 对象
  hash_t(uint32_t val) : uint128(val) {}                          // 使用 uint32_t 类型构造 hash_t 对象
  hash_t(uint64_t val) : uint128(val) {}                          // 使用 uint64_t 类型构造 hash_t 对象
  hash_t(uint128 val) : uint128(val) {}                           // 使用 uint128 类型构造 hash_t 对象
  hash_t(uint64_t top, uint64_t bottom) : uint128(top, bottom) {} // 使用两个 uint64_t 类型构造 hash_t 对象
  hash_t() : uint128() {}                                         // 默认构造函数
};

// 使用 64 位哈希值计算数据的哈希值
size_t TORCH_API StdDataHash(const void* data, size_t size);

// 使用 64 位哈希值合并两个整数的哈希值
size_t TORCH_API StdHashCombine(uintmax_t a, uintmax_t b);

// 使用 128 位哈希值计算数据块的哈希值
hash_t TORCH_API HashBlock(const void* data, size_t n, const hash_t& seed);

// 使用 128 位哈希值计算数据的哈希值
hash_t TORCH_API DataHash(const void* data, size_t size);

// 使用 128 位哈希值合并两个哈希值
hash_t TORCH_API HashCombine(const hash_t& a, const hash_t& b);

// 使用 128 位哈希值减少哈希值
size_t TORCH_API HashReduce(const hash_t& a);

// 返回哈希值的字符串表示
std::string TORCH_API HashToString(const hash_t& a);

// 哈希值的散列函数对象，用于 STL 容器中的自定义哈希
struct HashReducer {
  size_t operator()(const hash_t& value) const {
    return HashReduce(value);
  }
};

// 计算 C 字符串的哈希值
static inline hash_t StringHash(const char* data) {
  return DataHash(data, std::strlen(data));
}

// 自动模板化实现 'arithmetic' 类型的哈希值计算
template <
    typename T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
hash_t Hash(const T& value) {
  return DataHash(&value, sizeof(value));
}

// macOS 构建时，vector<bool> 的特化实现
hash_t TORCH_API Hash(const std::vector<bool>& value);

// c10 库中特定类型的哈希值计算实现
static inline hash_t Hash(const c10::ScalarType& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::MemoryFormat& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::DeviceType& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::Device& value) {
  return HashCombine(Hash(value.type()), Hash(value.index()));
}

static inline hash_t Hash(const c10::Layout& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::Scalar& value) {
  switch (value.type()) {
    case c10::ScalarType::ComplexDouble:
      return Hash(value.toComplexDouble());
      // 继续添加其他 c10::ScalarType 的特定哈希计算
    # 如果值的类型是双精度浮点型（Double），则调用 Hash 函数计算哈希值并返回
    case c10::ScalarType::Double:
      return Hash(value.toDouble());
    # 如果值的类型是长整型（Long），则调用 Hash 函数计算哈希值并返回
    case c10::ScalarType::Long:
      return Hash(value.toLong());
    # 如果值的类型是布尔型（Bool），则调用 Hash 函数计算哈希值并返回
    case c10::ScalarType::Bool:
      return Hash(value.toBool());
    # 如果值的类型不是上述三种类型，则抛出内部错误断言，打印出未知的标量类型和具体的值类型
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown scalar type.", value.type());
  }
}

// 计算张量的哈希值
static inline hash_t TensorHash(const at::Tensor& tensor) {
  // 使张量变成连续的存储，以便于哈希计算
  at::Tensor ctensor = tensor.contiguous();
  // 计算张量的总字节数
  int64_t size = ctensor.numel() * ctensor.element_size();
  // 根据张量的数据类型选择相应的数据指针进行哈希计算
  switch (ctensor.scalar_type()) {
    case at::ScalarType::Bool:
      return DataHash(ctensor.const_data_ptr<bool>(), size);
    case at::ScalarType::Byte:
      return DataHash(ctensor.const_data_ptr<uint8_t>(), size);
    case at::ScalarType::Char:
      return DataHash(ctensor.const_data_ptr<int8_t>(), size);
    case at::ScalarType::Short:
      return DataHash(ctensor.const_data_ptr<int16_t>(), size);
    case at::ScalarType::Int:
      return DataHash(ctensor.const_data_ptr<int32_t>(), size);
    case at::ScalarType::Long:
      return DataHash(ctensor.const_data_ptr<int64_t>(), size);
    case at::ScalarType::Float:
      return DataHash(ctensor.const_data_ptr<float>(), size);
    case at::ScalarType::Double:
      return DataHash(ctensor.const_data_ptr<double>(), size);
    case at::ScalarType::BFloat16:
      return DataHash(ctensor.const_data_ptr<at::BFloat16>(), size);
    case at::ScalarType::Half:
      return DataHash(ctensor.const_data_ptr<at::Half>(), size);
    case at::ScalarType::ComplexFloat:
      return DataHash(ctensor.const_data_ptr<c10::complex<float>>(), size);
    case at::ScalarType::ComplexDouble:
      return DataHash(ctensor.const_data_ptr<c10::complex<double>>(), size);
    case at::ScalarType::UInt16:
      return DataHash(ctensor.const_data_ptr<uint16_t>(), size);
    case at::ScalarType::UInt32:
      return DataHash(ctensor.const_data_ptr<uint32_t>(), size);
    case at::ScalarType::UInt64:
      return DataHash(ctensor.const_data_ptr<uint64_t>(), size);
    default:
      // 如果出现不支持的标量类型，则断言失败
      TORCH_INTERNAL_ASSERT(false, "Unsupported scalar type:", ctensor.scalar_type());
  }
}

// 哈希字符串值
static inline hash_t Hash(const std::string& value) {
  return DataHash(value.data(), value.size());
}

// 哈希 c10::string_view 类型的值
static inline hash_t Hash(const c10::string_view& value) {
  return DataHash(value.data(), value.size());
}

// 哈希生成器对象
static inline hash_t Hash(const at::Generator& value) {
  return TensorHash(value.get_state());
}

// 对 std::optional 类型进行哈希计算
// 考虑到可能的空值情况，以区分不同的情况
template <typename T>
hash_t Hash(const std::optional<T>& value) {
  if (value.has_value()) {
    // 如果有值，则对值进行哈希
    return Hash(value.value());
  } else {
    // 如果为空，则返回预设的空值哈希
    return kNullOpt;
  }
}

// 从 glibc 的实现中借鉴的哈希可选项的方式
// 用于处理 std::optional 类型的哈希计算
// 确保在空值和非空值之间有区分
// 使用随机选择的64位整数来代替小常量，在运行时进行哈希计算
static const int64_t kNullOpt = 0x8655d738f3678dda;

// 容器的哈希计算
// 前向声明以允许对向量向量进行哈希处理。
template <typename T>
hash_t ContainerHash(const T& values);

// 对 std::vector<T> 进行哈希处理，调用 ContainerHash 进行实际处理
template <typename T>
hash_t Hash(const std::vector<T>& values) {
  return ContainerHash(values);
}

// 对 std::optional<std::vector<T>> 进行特殊处理，如果有值则哈希化其值，否则返回 kNullOpt
template <typename T>
hash_t Hash(const std::optional<std::vector<T>>& value) {
  if (value.has_value()) {
    return ContainerHash(value.value());
  } else {
    return kNullOpt;
  }
}

// 对 std::set<T> 进行哈希处理，调用 ContainerHash 进行实际处理
template <typename T>
hash_t Hash(const std::set<T>& values) {
  return ContainerHash(values);
}

// 对 std::pair<T, S> 进行哈希处理，组合哈希值
template <typename T, typename S>
hash_t Hash(const std::pair<T, S>& values) {
  return HashCombine(Hash(values.first), Hash(values.second));
}

// 对 hash_t 进行哈希处理，直接返回其值
static inline hash_t Hash(const hash_t& value) {
  return value;
}

// 对 c10::ArrayRef<T> 进行哈希处理，调用 ContainerHash 进行实际处理
template <typename T>
hash_t Hash(c10::ArrayRef<T> values) {
  return ContainerHash(values);
}

// 对容器类型 T 进行哈希处理，迭代容器中的每个元素并进行哈希组合
template <typename T>
hash_t ContainerHash(const T& values) {
  // 初始哈希值
  hash_t h(static_cast<uint64_t>(0x85ebca77c2b2ae63));
  // 遍历容器中的每个元素，依次进行哈希处理并组合结果
  for (const auto& value : values) {
    h = HashCombine(h, Hash(value));
  }
  return h;
}

// 可变参数哈希处理，递归调用 HashCombine 对参数进行哈希组合
template <typename T = void>
hash_t MHash() {
  return hash_t(static_cast<uint64_t>(0x165667b19e3779f9));
}

// 可变参数哈希处理，递归调用 HashCombine 对每个参数进行哈希处理并组合结果
template <typename T, typename... Targs>
hash_t MHash(T value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

} // namespace lazy
} // namespace torch
```