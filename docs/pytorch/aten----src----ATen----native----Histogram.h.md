# `.\pytorch\aten\src\ATen\native\Histogram.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 类定义头文件

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 类定义头文件

namespace at::native {

using histogramdd_fn = void(*)(const Tensor&, const std::optional<Tensor>&, bool, Tensor&, const TensorList&);
// 定义 histogramdd_fn 类型为函数指针，接受一系列参数，并返回 void

using histogramdd_linear_fn = void(*)(const Tensor&, const std::optional<Tensor>&, bool, Tensor&, const TensorList&, bool);
// 定义 histogramdd_linear_fn 类型为函数指针，接受一系列参数，并返回 void

using histogram_select_outer_bin_edges_fn = void(*)(const Tensor& input, const int64_t N, std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges);
// 定义 histogram_select_outer_bin_edges_fn 类型为函数指针，接受一系列参数，并返回 void

DECLARE_DISPATCH(histogramdd_fn, histogramdd_stub);
// 声明 histogramdd_stub 函数的调度器，接受 histogramdd_fn 类型的函数指针

DECLARE_DISPATCH(histogramdd_linear_fn, histogramdd_linear_stub);
// 声明 histogramdd_linear_stub 函数的调度器，接受 histogramdd_linear_fn 类型的函数指针

DECLARE_DISPATCH(histogram_select_outer_bin_edges_fn, histogram_select_outer_bin_edges_stub);
// 声明 histogram_select_outer_bin_edges_stub 函数的调度器，接受 histogram_select_outer_bin_edges_fn 类型的函数指针

} // namespace at::native
// 结束 at::native 命名空间
```