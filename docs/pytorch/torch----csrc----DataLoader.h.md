# `.\pytorch\torch\csrc\DataLoader.h`

```
#pragma once
// 包含 Torch 库的 Python 头文件
#include <torch/csrc/python_headers.h>

// 声明一个外部的 PyMethodDef 数组 DataLoaderMethods
// NOLINTNEXTLINE 用于指定代码规则禁忌，此处禁止使用 C 风格数组和非常量全局变量，推荐使用现代化方式避免 C 风格数组
extern PyMethodDef DataLoaderMethods[];
```