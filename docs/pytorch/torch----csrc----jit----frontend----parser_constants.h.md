# `.\pytorch\torch\csrc\jit\frontend\parser_constants.h`

```py
#pragma once

// 在 C++ 中，此指令确保头文件只被编译一次，以防止重复包含


namespace torch {

// 声明命名空间 torch，用于组织代码，避免全局命名冲突


namespace jit {

// 声明命名空间 jit，用于在 torch 命名空间内进一步组织代码


static const char* valid_single_char_tokens = "+-*/%@()[]:,={}><.?!&^|~";

// 定义静态常量指针 valid_single_char_tokens，存储了一组有效的单字符运算符和特殊字符


} // namespace jit

// 结束 jit 命名空间的声明


} // namespace torch

// 结束 torch 命名空间的声明
```