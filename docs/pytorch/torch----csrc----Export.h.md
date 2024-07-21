# `.\pytorch\torch\csrc\Export.h`

```
#pragma once
// 如果 THP_BUILD_MAIN_LIB 宏被定义，则将 TORCH_PYTHON_API 宏设置为 C10_EXPORT
// 否则，将 TORCH_PYTHON_API 宏设置为 C10_IMPORT
#include <c10/macros/Export.h>

#ifdef THP_BUILD_MAIN_LIB
#define TORCH_PYTHON_API C10_EXPORT
#else
#define TORCH_PYTHON_API C10_IMPORT
#endif
```