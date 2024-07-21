# `.\pytorch\aten\src\ATen\native\TensorCompare.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 头文件

namespace c10 {
class Scalar;
}
// 声明 c10 命名空间和 Scalar 类

namespace at {
class Tensor;
struct TensorIterator;
struct TensorIteratorBase;
}
// 声明 at 命名空间、Tensor 类和 TensorIterator 结构体

namespace at::native {

using reduce_minmax_fn =
    void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);
// 定义 reduce_minmax_fn 类型别名，表示接受特定参数的函数指针

using structured_reduce_minmax_fn =
    void (*)(const Tensor&, const Tensor&, const Tensor&, int64_t, bool);
// 定义 structured_reduce_minmax_fn 类型别名，表示接受特定参数的常量对象的函数指针

DECLARE_DISPATCH(structured_reduce_minmax_fn, max_stub);
DECLARE_DISPATCH(structured_reduce_minmax_fn, min_stub);
// 声明 max_stub 和 min_stub 函数的调度分发器

using where_fn = void (*)(TensorIterator &);
DECLARE_DISPATCH(where_fn, where_kernel);
// 声明 where_kernel 函数的调度分发器，接受 TensorIterator 参数

using is_infinity_op_fn = void (*)(TensorIteratorBase &);
DECLARE_DISPATCH(is_infinity_op_fn, isposinf_stub);
DECLARE_DISPATCH(is_infinity_op_fn, isneginf_stub);
// 声明 isposinf_stub 和 isneginf_stub 函数的调度分发器，接受 TensorIteratorBase 参数

using mode_fn = void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);
DECLARE_DISPATCH(mode_fn, mode_stub);
// 声明 mode_stub 函数的调度分发器

using clamp_tensor_fn = void (*)(TensorIteratorBase &);
DECLARE_DISPATCH(clamp_tensor_fn, clamp_stub);
// 声明 clamp_stub 函数的调度分发器，接受 TensorIteratorBase 参数

namespace detail {
    enum class ClampLimits {Min, Max, MinMax};
}
// 声明 detail 命名空间，定义 ClampLimits 枚举，表示极值类型

DECLARE_DISPATCH(void (*)(TensorIteratorBase &, const c10::Scalar&, const c10::Scalar&), clamp_scalar_stub);
DECLARE_DISPATCH(void (*)(TensorIteratorBase &, c10::Scalar), clamp_min_scalar_stub);
DECLARE_DISPATCH(void (*)(TensorIteratorBase &, c10::Scalar), clamp_max_scalar_stub);
// 声明 clamp_scalar_stub、clamp_min_scalar_stub 和 clamp_max_scalar_stub 函数的调度分发器

using isin_default_fn = void (*)(const Tensor&, const Tensor&, bool, const Tensor&);
DECLARE_DISPATCH(isin_default_fn, isin_default_stub);
// 声明 isin_default_stub 函数的调度分发器，接受特定参数

} // namespace at::native
// 结束 at::native 命名空间
```