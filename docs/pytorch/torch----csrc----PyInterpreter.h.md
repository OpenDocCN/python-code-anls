# `.\pytorch\torch\csrc\PyInterpreter.h`

```
#pragma once

# 使用 `#pragma once` 指令，确保头文件只被编译一次，防止多重包含导致的重复定义错误


#include <c10/core/impl/PyInterpreter.h>
#include <torch/csrc/Export.h>

# 包含 `<c10/core/impl/PyInterpreter.h>` 和 `<torch/csrc/Export.h>` 头文件，用于声明下面代码中使用的类和函数


TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();

# 声明 `getPyInterpreter()` 函数，其返回类型为 `c10::impl::PyInterpreter*` 类型的指针，函数被声明为 `TORCH_PYTHON_API`，用于从 Torch Python API 中获取 Python 解释器对象。


TORCH_PYTHON_API bool isMainPyInterpreter();

# 声明 `isMainPyInterpreter()` 函数，其返回类型为 `bool`，函数被声明为 `TORCH_PYTHON_API`，用于检查当前 Python 解释器是否为主解释器。
```