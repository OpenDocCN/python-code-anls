# `.\pytorch\aten\src\ATen\native\mps\operations\Indexing.h`

```py
// 设置编译选项，仅允许方法操作符使用 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入 ATen 库中的 MPS 模块相关头文件
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/TensorFactory.h>

// 引入 C10 库中的标量类型定义
#include <c10/core/ScalarType.h>

// 引入标准库中的无序映射容器
#include <unordered_map>

// 使用 ATen 命名空间下的 MPS 子命名空间
using namespace at::mps;
```