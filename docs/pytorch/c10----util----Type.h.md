# `.\pytorch\c10\util\Type.h`

```
#ifndef C10_UTIL_TYPE_H_
#define C10_UTIL_TYPE_H_

#include <cstddef>
#include <string>
#ifdef __GXX_RTTI
#include <typeinfo>
#endif // __GXX_RTTI

#include <c10/macros/Macros.h>

namespace c10 {

/// Utility to demangle a C++ symbol name.
/// C++符号名称解码的实用工具函数。
C10_API std::string demangle(const char* name);

/// Returns the printable name of the type.
/// 返回类型的可打印名称。
template <typename T>
inline const char* demangle_type() {
#ifdef __GXX_RTTI
  // 如果启用了RTTI，获取类型T的名称并解码成字符串
  static const auto& name = *(new std::string(demangle(typeid(T).name())));
  return name.c_str();
#else // __GXX_RTTI
  // 如果未启用RTTI，则返回一个指示RTTI已禁用的占位字符串
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

} // namespace c10

#endif // C10_UTIL_TYPE_H_


这段代码是一个 C++ 的头文件，提供了用于类型解码和名称显示的工具函数。
```