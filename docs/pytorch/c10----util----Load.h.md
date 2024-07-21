# `.\pytorch\c10\util\Load.h`

```py
#pragma once
// 包含 C10 库的宏定义和标准库的字符串操作头文件
#include <c10/macros/Macros.h>
#include <cstring>

// 定义 c10 命名空间
namespace c10 {
// 定义 c10 内部细节的命名空间
namespace detail {

// 定义模板结构体 LoadImpl，用于加载不同类型的数据
template <typename T>
struct LoadImpl {
  // 静态函数，用于加载数据并返回指定类型 T
  C10_HOST_DEVICE static T apply(const void* src) {
    // 将 src 强制转换为 T 类型的指针，然后解引用获取数据
    return *reinterpret_cast<const T*>(src);
  }
};

// LoadImpl 的特化版本，用于加载布尔类型数据
template <>
struct LoadImpl<bool> {
  // 静态函数，用于加载布尔值并返回
  C10_HOST_DEVICE static bool apply(const void* src) {
    // 断言布尔类型的大小与字符类型的大小相同
    static_assert(sizeof(bool) == sizeof(char));
    // 注意：[加载布尔值]
    // 通过先加载为字节来保护免受无效布尔值的影响，然后再转换为 bool 类型（参见 gh-54789）
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

} // namespace detail

// load 函数模板，用于加载数据并返回指定类型 T
template <typename T>
C10_HOST_DEVICE T load(const void* src) {
  // 调用 LoadImpl 结构体的 apply 函数，加载并返回数据
  return c10::detail::LoadImpl<T>::apply(src);
}

// load 函数模板的特化版本，用于加载标量类型数据
template <typename scalar_t>
C10_HOST_DEVICE scalar_t load(const scalar_t* src) {
  // 调用 LoadImpl 结构体的 apply 函数，加载并返回数据
  return c10::detail::LoadImpl<scalar_t>::apply(src);
}

} // namespace c10
```