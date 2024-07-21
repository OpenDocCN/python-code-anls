# `.\pytorch\aten\src\ATen\native\ReduceOps.h`

```py
#pragma once



// 通过指令指示，确保头文件只被编译一次
#include <ATen/native/DispatchStub.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
class Tensor;
}

namespace at::native {

// 定义函数指针类型 reduce_fn，用于表示一个接受 TensorIterator 引用参数的函数指针
using reduce_fn = void(*)(TensorIterator &);

// 声明各种 reduce 操作的调度函数指针
DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, nansum_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);
DECLARE_DISPATCH(reduce_fn, mean_stub);
DECLARE_DISPATCH(reduce_fn, and_stub);
DECLARE_DISPATCH(reduce_fn, or_stub);
DECLARE_DISPATCH(reduce_fn, min_values_stub);
DECLARE_DISPATCH(reduce_fn, max_values_stub);
DECLARE_DISPATCH(reduce_fn, argmax_stub);
DECLARE_DISPATCH(reduce_fn, argmin_stub);

// 定义函数指针类型 reduce_std_var_function，用于表示接受 TensorIterator 引用、double 和 bool 参数的函数指针
using reduce_std_var_function =
    void (*)(TensorIterator&, double correction, bool take_sqrt);

// 声明标准差和方差计算的调度函数指针
DECLARE_DISPATCH(reduce_std_var_function, std_var_stub);

// 定义函数指针类型 reduce_norm_fn，用于表示接受 Tensor&, const Tensor&, Scalar 和 optional<int64_t> 参数的函数指针
using reduce_norm_fn =
    void (*)(Tensor&, const Tensor&, const c10::Scalar&, std::optional<int64_t>);

// 声明归一化操作的调度函数指针
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel);

// 定义函数指针类型 reduce_fn_flag，用于表示接受 TensorIterator 引用和 Scalar 参数的函数指针
using reduce_fn_flag = void(*)(TensorIterator &, const c10::Scalar&);

// 声明 norm 操作的调度函数指针
DECLARE_DISPATCH(reduce_fn_flag, norm_stub);

// 定义函数指针类型 structured_cum_fn 和 cum_fn，用于表示不同累积函数的调度函数指针
using structured_cum_fn = void (*)(const Tensor&, const Tensor&, int64_t);
using cum_fn = void (*)(Tensor&, const Tensor&, int64_t);

// 声明累积函数的调度函数指针
DECLARE_DISPATCH(structured_cum_fn, cumsum_stub);
DECLARE_DISPATCH(structured_cum_fn, cumprod_stub);
DECLARE_DISPATCH(cum_fn, logcumsumexp_stub);

// 声明 aminmax_stub 和 aminmax_allreduce_stub 函数的调度函数指针
DECLARE_DISPATCH(void (*)(const Tensor&, int64_t, bool, Tensor&, Tensor&), aminmax_stub);
DECLARE_DISPATCH(void (*)(const Tensor&, Tensor&, Tensor&), aminmax_allreduce_stub);

// 用于 cuda/Normalization.cu 中的函数声明，返回两个输出张量的元组
TORCH_API std::tuple<Tensor&,Tensor&> var_mean_out(
    Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim,
    int64_t correction, bool keepdim);

} // namespace at::native
```