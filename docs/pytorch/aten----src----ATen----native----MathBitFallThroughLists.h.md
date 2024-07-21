# `.\pytorch\aten\src\ATen\native\MathBitFallThroughLists.h`

```py
#pragma once

namespace at {
// 定义一个宏，用于注册张量视图相关的函数实现
#define TENSOR_UTILITIES_AND_CONSTRUCTORS(m) \
  // 注册 empty_like 函数的实现，使用默认的转发行为
  m.impl("empty_like", torch::CppFunction::makeFallthrough()); \
  // 注册 empty.memory_format 函数的实现，使用默认的转发行为
  m.impl("empty.memory_format", torch::CppFunction::makeFallthrough()); \
  // 注册 empty.out 函数的实现，使用默认的转发行为
  m.impl("empty.out", torch::CppFunction::makeFallthrough()); \
  // 注册 empty_strided 函数的实现，使用默认的转发行为
  m.impl("empty_strided", torch::CppFunction::makeFallthrough()); \
  // 注册 full_like 函数的实现，使用默认的转发行为
  m.impl("full_like", torch::CppFunction::makeFallthrough()); \
  // 注册 stride.int 函数的实现，使用默认的转发行为
  m.impl("stride.int", torch::CppFunction::makeFallthrough()); \
  // 注册 stride.Dimname 函数的实现，使用默认的转发行为
  m.impl("stride.Dimname", torch::CppFunction::makeFallthrough()); \
  // 注册 size.int 函数的实现，使用默认的转发行为
  m.impl("size.int", torch::CppFunction::makeFallthrough()); \
  // 注册 size.Dimname 函数的实现，使用默认的转发行为
  m.impl("size.Dimname", torch::CppFunction::makeFallthrough()); \
  // 注册 is_complex 函数的实现，使用默认的转发行为
  m.impl("is_complex", torch::CppFunction::makeFallthrough()); \
  // 注册 is_floating_point 函数的实现，使用默认的转发行为
  m.impl("is_floating_point", torch::CppFunction::makeFallthrough()); \
  // 注册 requires_grad_ 函数的实现，使用默认的转发行为
  m.impl("requires_grad_", torch::CppFunction::makeFallthrough());
}

// 定义一个宏，用于注册张量视图的原生函数实现
#define TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m) \
  // 注册 as_strided 函数的实现，使用默认的转发行为
  m.impl("as_strided", torch::CppFunction::makeFallthrough()); \
  // 注册 view 函数的实现，使用默认的转发行为
  m.impl("view", torch::CppFunction::makeFallthrough());
```