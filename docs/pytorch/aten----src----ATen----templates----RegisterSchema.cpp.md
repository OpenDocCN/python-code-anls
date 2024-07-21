# `.\pytorch\aten\src\ATen\templates\RegisterSchema.cpp`

```
// 定义宏以仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入Torch库的头文件
#include <torch/library.h>

// 命名空间：at
namespace at {
    // Torch 库注册函数
    TORCH_LIBRARY(aten, m) {
        // 注册ATen操作的模式(schema)
        ${aten_schema_registrations};
        // 分布式操作
        // 实现位于 torch/csrc/jit/runtime/register_distributed_ops.cpp 中
        // 定义了一个名为 "get_gradients" 的方法，接受一个整数参数 context_id，返回一个 Tensor 到 Tensor 的字典
        m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
    }
    // 结束命名空间 at
    ${schema_registrations}
}  // namespace at
```