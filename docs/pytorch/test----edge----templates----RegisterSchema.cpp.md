# `.\pytorch\test\edge\templates\RegisterSchema.cpp`

```
// ${generated_comment}
// 定义宏，用于声明只包含方法的运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 Torch 库的头文件
#include <torch/library.h>

// 定义 at 命名空间
namespace at {

// 使用 TORCH_LIBRARY_FRAGMENT 宏定义 aten 库的 Torch 库片段
TORCH_LIBRARY_FRAGMENT(aten, m) {
    // 插入由代码生成器生成的 ATen 操作的模式注册
    ${aten_schema_registrations};
}

// 结束 at 命名空间
} // namespace at
```