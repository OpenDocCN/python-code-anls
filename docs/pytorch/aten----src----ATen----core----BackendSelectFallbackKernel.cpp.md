# `.\pytorch\aten\src\ATen\core\BackendSelectFallbackKernel.cpp`

```py
# 包含 Torch 库的头文件
#include <torch/library.h>

# 定义 Torch 库的实现
TORCH_LIBRARY_IMPL(_, BackendSelect, m) {
    # 设置回退策略为 CppFunction 的 fallthrough()
    m.fallback(torch::CppFunction::makeFallthrough());
}
```