# `.\pytorch\torch\csrc\jit\runtime\serialized_shape_function_registry.h`

```py
#pragma once
// 使用#pragma once确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含torch库的导出头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含torch库的jit模块中的ir.h头文件

namespace torch::jit {
// 命名空间torch::jit，用于组织代码，避免命名冲突

TORCH_API const std::string& GetSerializedShapeFunctions();
// 声明一个函数GetSerializedShapeFunctions，返回一个const引用，类型为std::string，使用了TORCH_API宏

TORCH_API const OperatorMap<std::string>& GetShapeFunctionMappings();
// 声明一个函数GetShapeFunctionMappings，返回一个const引用，类型为OperatorMap<std::string>，使用了TORCH_API宏

TORCH_API const OperatorMap<std::pair<std::string, std::string>>&
// 声明一个函数GetBoundedShapeMappings，返回一个const引用，类型为OperatorMap<std::pair<std::string, std::string>>，使用了TORCH_API宏
GetBoundedShapeMappings();

} // namespace torch::jit
// 结束torch::jit命名空间
```