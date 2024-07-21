# `.\pytorch\aten\src\ATen\native\utils\ParamsHash.h`

```
#pragma once

#include <c10/util/irange.h>  // 引入范围库，用于迭代器的范围操作
#include <memory>             // 引入内存管理库，用于智能指针等内存操作
#include <mutex>              // 引入互斥锁库，用于多线程同步

namespace at::native {

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function

// 定义模板结构体 ParamsHash，用于参数 Params 的哈希计算
template <typename Params>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  // 断言 Params 必须是标准布局类型（POD，Plain Old Data），否则编译错误
  static_assert(std::is_standard_layout_v<Params>, "Params is not POD");

  // 哈希运算符重载，计算 Params 对象的哈希值
  size_t operator()(const Params& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    // 使用 FNV 哈希算法计算哈希值
    for (const auto i : c10::irange(sizeof(Params))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

// 定义模板结构体 ParamsEqual，用于比较两个 Params 对象是否相等
template <typename Params>
struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contents as char* when comparing
  // 断言 Params 必须是标准布局类型（POD，Plain Old Data），否则编译错误
  static_assert(std::is_standard_layout_v<Params>, "Params is not POD");

  // 比较运算符重载，判断两个 Params 对象是否相等
  bool operator()(const Params& a, const Params& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    // 使用 memcmp 函数比较两个内存块的内容是否相等
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

// Provide explicit byte-for-byte constructors to avoid unwittingly leaving
// padding bytes uninitialized (e.g., when passing Params by value)

// 定义模板结构体 ParamsWrapper，用于包装 POD 类型的参数，并提供明确的构造函数
template <typename T>
struct ParamsWrapper {
  T pod;
  static_assert(
      std::is_standard_layout_v<T>,
      "ParamsWrapper cannot wrap non-POD data");

  // 默认构造函数，将包装的 POD 类型参数初始化为零
  ParamsWrapper() {
    memset(&(this->pod), 0, sizeof(this->pod));
  }

  // 拷贝构造函数，按字节拷贝另一个 ParamsWrapper 对象的 pod 成员
  ParamsWrapper(const ParamsWrapper& other) {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
  }

  // 移动构造函数，按字节移动另一个 ParamsWrapper 对象的 pod 成员
  ParamsWrapper(ParamsWrapper&& other) noexcept {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
  }

  // 拷贝赋值运算符重载，按字节拷贝另一个 ParamsWrapper 对象的 pod 成员
  ParamsWrapper& operator=(const ParamsWrapper& other) {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
    return *this;
  }

  // 移动赋值运算符重载，按字节移动另一个 ParamsWrapper 对象的 pod 成员
  ParamsWrapper& operator=(ParamsWrapper&& other) noexcept {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
    return *this;
  }

  // 比较运算符重载，按字节比较两个 ParamsWrapper 对象的 pod 成员是否相等
  inline friend bool operator==(
      const ParamsWrapper& lhs,
      const ParamsWrapper& rhs) noexcept {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&(lhs.pod));
    auto ptr2 = reinterpret_cast<const uint8_t*>(&(rhs.pod));
    return memcmp(ptr1, ptr2, sizeof(lhs.pod)) == 0;
  }
};

// Wrapped version: this allows the outer struct to have custom copy and move
// constructors for additional safety

// 定义模板结构体 ParamsWrapperHash，用于包装 ParamsWrapper 并提供哈希计算
template <typename ParamsWrapper>
struct ParamsWrapperHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  // 断言 ParamsWrapper 的 pod 成员必须是标准布局类型（POD，Plain Old Data），否则编译错误
  static_assert(
      std::is_standard_layout_v<decltype(ParamsWrapper::pod)>,
      "ParamsWrapper cannot wrap non-POD data");

  // 哈希运算符重载，计算 ParamsWrapper 对象的哈希值
  size_t operator()(const ParamsWrapper& params_wrapper) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&(params_wrapper.pod));
    uint32_t value = 0x811C9DC5;
    // 使用 FNV 哈希算法计算哈希值
    for (const auto i : c10::irange(sizeof(params_wrapper.pod))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};
    // 对 params_wrapper.pod 中的数据进行逐字节处理
    for (const auto i : c10::irange(sizeof(params_wrapper.pod))) {
      // 按位异或操作，将 value 与 ptr[i] 进行异或运算
      value ^= ptr[i];
      // 将 value 乘以常数 0x01000193
      value *= 0x01000193;
    }
    // 将处理后的 value 转换为 size_t 类型并返回
    return (size_t)value;
  }
};

} // namespace at::native
```