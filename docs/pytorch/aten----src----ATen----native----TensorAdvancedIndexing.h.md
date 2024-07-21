# `.\pytorch\aten\src\ATen\native\TensorAdvancedIndexing.h`

```py
#pragma once
// 防止头文件被多次引用

// Indexing tensors by tensors
// 使用张量索引张量的相关声明和定义

#include <ATen/core/List.h>
// 引入 ATen 库中的 List 头文件
#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 Tensor 头文件
#include <ATen/native/DispatchStub.h>
// 引入 ATen 库中的 DispatchStub 头文件
#include <ATen/native/ReductionType.h>
// 引入 ATen 库中的 ReductionType 头文件

namespace at {
struct TensorIterator;
}
// 声明 at 命名空间下的 TensorIterator 结构体

namespace at::native {

using index_put_with_sort_fn = void(*)(Tensor &, const c10::List<std::optional<Tensor>> &, const Tensor &, bool accumulate, bool unsafe);
// 定义 index_put_with_sort_fn 类型别名，指向一个函数指针，用于执行带排序的索引赋值操作

using index_put_with_sort_quantized_fn = void(*)(Tensor& self, const c10::List<std::optional<Tensor>>& indices, const Tensor& value, double scale, int zero_point, bool unsafe);
// 定义 index_put_with_sort_quantized_fn 类型别名，指向一个函数指针，用于执行带排序的量化索引赋值操作

using gather_fn = void (*)(const Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
// 定义 gather_fn 类型别名，指向一个函数指针，用于执行 gather 操作

using scatter_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
// 定义 scatter_fn 类型别名，指向一个函数指针，用于执行 scatter 操作

using scatter_fill_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& src);
// 定义 scatter_fill_fn 类型别名，指向一个函数指针，用于执行 scatter_fill 操作

using scatter_add_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
// 定义 scatter_add_fn 类型别名，指向一个函数指针，用于执行 scatter_add 操作

using scatter_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                  const Tensor& src, const ReductionType& reduce);
// 定义 scatter_reduce_fn 类型别名，指向一个函数指针，用于执行 scatter_reduce 操作

using scatter_scalar_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                         const Scalar& value, const ReductionType& reduce);
// 定义 scatter_scalar_reduce_fn 类型别名，指向一个函数指针，用于执行 scatter_scalar_reduce 操作

using scatter_reduce_two_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                      const Tensor& src, const ReductionType& reduce);
// 定义 scatter_reduce_two_fn 类型别名，指向一个函数指针，用于执行 scatter_reduce_two 操作

DECLARE_DISPATCH(index_put_with_sort_fn, index_put_with_sort_stub);
// 声明 index_put_with_sort_stub 函数的调度分发

DECLARE_DISPATCH(index_put_with_sort_quantized_fn, index_put_with_sort_quantized_stub);
// 声明 index_put_with_sort_quantized_stub 函数的调度分发

DECLARE_DISPATCH(gather_fn, gather_stub);
// 声明 gather_stub 函数的调度分发

DECLARE_DISPATCH(scatter_fn, scatter_stub);
// 声明 scatter_stub 函数的调度分发

DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub);
// 声明 scatter_fill_stub 函数的调度分发

DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub);
// 声明 scatter_add_stub 函数的调度分发

DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub);
// 声明 scatter_reduce_stub 函数的调度分发

DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub);
// 声明 scatter_scalar_reduce_stub 函数的调度分发

DECLARE_DISPATCH(scatter_reduce_two_fn, scatter_reduce_two_stub);
// 声明 scatter_reduce_two_stub 函数的调度分发

TORCH_API Tensor& index_out(Tensor& result, const Tensor & self, const c10::List<std::optional<at::Tensor>>& indices);
// 声明 index_out 函数的 API 接口，用于处理张量索引输出

using scatter_add_expanded_index_fn = void(*)(const Tensor&, const Tensor&, const Tensor&);
// 定义 scatter_add_expanded_index_fn 类型别名，指向一个函数指针，用于执行扩展索引的 scatter_add 操作

using scatter_reduce_expanded_index_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const ReductionType& reduce, bool);
// 定义 scatter_reduce_expanded_index_fn 类型别名，指向一个函数指针，用于执行扩展索引的 scatter_reduce 操作

using gather_expanded_index_fn = void (*)(const Tensor&, const Tensor&, const Tensor&);
// 定义 gather_expanded_index_fn 类型别名，指向一个函数指针，用于执行扩展索引的 gather 操作

DECLARE_DISPATCH(scatter_add_expanded_index_fn, scatter_add_expanded_index_stub);
// 声明 scatter_add_expanded_index_stub 函数的调度分发

DECLARE_DISPATCH(scatter_reduce_expanded_index_fn, scatter_reduce_expanded_index_stub);
// 声明 scatter_reduce_expanded_index_stub 函数的调度分发

DECLARE_DISPATCH(gather_expanded_index_fn, gather_expanded_index_stub);
// 声明 gather_expanded_index_stub 函数的调度分发

} // namespace at::native
// 结束 at::native 命名空间的定义
```