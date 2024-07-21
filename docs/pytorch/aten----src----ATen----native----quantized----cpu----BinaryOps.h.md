# `.\pytorch\aten\src\ATen\native\quantized\cpu\BinaryOps.h`

```py
#include <ATen/core/Tensor.h>

# 包含 ATen 核心库中的 Tensor 类的头文件


namespace at {
namespace native {

# 开始定义命名空间 at 和 native


TORCH_API Tensor
quantized_add(Tensor qa, Tensor qb, double scale, int64_t zero_point);

# 声明了一个函数 quantized_add，其返回类型为 Tensor 类型，使用了 TORCH_API 宏修饰，接受四个参数：两个 Tensor 对象 qa 和 qb，一个双精度浮点型参数 scale，一个 64 位整型参数 zero_point


} // namespace native
} // namespace at

# 结束定义命名空间 native 和 at
```