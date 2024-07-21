# `.\pytorch\aten\src\ATen\native\SegmentReduce.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 头文件，用于定义分发函数

#include <ATen/native/ReductionType.h>
// 包含 ATen 库中的 ReductionType 头文件，定义了各种数据减少操作的类型

#include <c10/core/Scalar.h>
// 包含 c10 库中的 Scalar 头文件，定义了标量类型

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional 头文件，定义了可选值的封装类型

namespace at {
// ATen 命名空间

class Tensor;
// 前向声明 Tensor 类，表示张量

namespace native {
// ATen::native 命名空间

using segment_reduce_lengths_fn = Tensor (*)(
    ReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
// 定义 segment_reduce_lengths_fn 类型别名，表示一个函数指针类型，用于执行长度分段减少操作

DECLARE_DISPATCH(segment_reduce_lengths_fn, _segment_reduce_lengths_stub);
// 声明 _segment_reduce_lengths_stub 分发函数，用于调度长度分段减少操作的具体实现

using segment_reduce_offsets_fn = Tensor (*)(
    ReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
// 定义 segment_reduce_offsets_fn 类型别名，表示一个函数指针类型，用于执行偏移量分段减少操作

DECLARE_DISPATCH(segment_reduce_offsets_fn, _segment_reduce_offsets_stub);
// 声明 _segment_reduce_offsets_stub 分发函数，用于调度偏移量分段减少操作的具体实现

using segment_reduce_lengths_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    ReductionType,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
// 定义 segment_reduce_lengths_backward_fn 类型别名，表示一个函数指针类型，用于执行长度分段减少操作的反向传播

DECLARE_DISPATCH(segment_reduce_lengths_backward_fn, _segment_reduce_lengths_backward_stub);
// 声明 _segment_reduce_lengths_backward_stub 分发函数，用于调度长度分段减少操作的反向传播的具体实现

using segment_reduce_offsets_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    ReductionType,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
// 定义 segment_reduce_offsets_backward_fn 类型别名，表示一个函数指针类型，用于执行偏移量分段减少操作的反向传播

DECLARE_DISPATCH(segment_reduce_offsets_backward_fn, _segment_reduce_offsets_backward_stub);
// 声明 _segment_reduce_offsets_backward_stub 分发函数，用于调度偏移量分段减少操作的反向传播的具体实现

} // namespace native
} // namespace at
```