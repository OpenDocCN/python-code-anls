# `.\pytorch\aten\src\ATen\native\TransposeType.h`

```
#pragma once
// 包含 C10 库中的异常处理工具
#include <c10/util/Exception.h>

// 定义在 at::native 命名空间下的接口，用于不同的类 BLAS 库之间的通信
namespace at::native {

// 定义枚举类型 TransposeType，用于指示转置类型
enum class TransposeType {
  NoTranspose,      // 不进行转置
  Transpose,        // 转置
  ConjTranspose,    // 共轭转置
};

// 将 TransposeType 转换为 BLAS / LAPACK 格式的字符表示
static inline char to_blas(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return 'T';        // 若为 Transpose，则返回 'T'
    case TransposeType::NoTranspose: return 'N';      // 若为 NoTranspose，则返回 'N'
    case TransposeType::ConjTranspose: return 'C';    // 若为 ConjTranspose，则返回 'C'
  }
  // 如果传入了无效的转置类型，会触发断言错误
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

}  // namespace at::native
```