# `.\pytorch\aten\src\ATen\native\quantized\cpu\RuyUtils.h`

```py
#pragma once
#ifdef USE_RUY_QMATMUL
// 包含 Ruy 库的头文件
#include <ruy/ruy.h>

// 命名空间声明：at::native::ruy_utils
namespace at {
namespace native {
namespace ruy_utils {

// 声明获取 Ruy 上下文的函数
ruy::Context* get_ruy_context();

// 声明量化乘数的函数，计算固定点乘数和指数
void quantize_multiplier(double scale,
                         int* multiplier_fixedpoint,
                         int* multiplier_exponent);

} // namespace ruy_utils
} // namespace native
} // namespace

#endif // USE_RUY_QMATMUL
```