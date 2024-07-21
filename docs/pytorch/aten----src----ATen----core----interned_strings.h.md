# `.\pytorch\aten\src\ATen\core\interned_strings.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义文件

#include <ATen/core/aten_interned_strings.h>
// 包含 ATen 库中用于内部字符串处理的头文件
#include <ATen/core/symbol.h>
// 包含 ATen 库中的符号（symbol）相关的头文件

namespace c10 {

enum class _keys : unique_t {
    // 定义一个枚举类型 _keys，其值的类型为 unique_t
    #define DEFINE_KEY(ns, s) ns##_##s,
    // 使用宏定义，遍历所有的命名空间符号，将命名空间和符号连接成枚举常量
    FORALL_NS_SYMBOLS(DEFINE_KEY)
    // 调用宏展开，将所有命名空间的符号都列出来
    #undef DEFINE_KEY
    // 取消前面定义的宏

    num_symbols
    // 枚举的最后一个值，表示符号的总数目
};

#define DEFINE_SYMBOL(ns, s) \
  namespace ns { constexpr Symbol s(static_cast<unique_t>(_keys::ns##_##s)); }
// 使用宏定义，定义命名空间 ns 中的符号 s，通过 _keys 枚举值转换为 unique_t 类型，并且声明为 constexpr 类型的 Symbol 对象
FORALL_NS_SYMBOLS(DEFINE_SYMBOL)
// 调用宏展开，为所有命名空间下的符号生成符号定义
#undef DEFINE_SYMBOL
// 取消前面定义的宏

} // namespace c10
// 命名空间 c10 的结束位置
```