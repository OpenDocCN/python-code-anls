# `.\pytorch\aten\src\ATen\native\CanUse32BitIndexMath.h`

```py
#pragma once
#include <c10/macros/Export.h>  // 包含导出宏定义头文件
#include <limits>  // 包含数值极限定义头文件

namespace at {
class TensorBase;  // 前置声明 TensorBase 类
}

namespace at::native {

// 声明函数 canUse32BitIndexMath，用于检查是否可以使用32位索引数学运算
TORCH_API bool canUse32BitIndexMath(const at::TensorBase &t, int64_t max_elem=std::numeric_limits<int32_t>::max());

}  // namespace at::native
```