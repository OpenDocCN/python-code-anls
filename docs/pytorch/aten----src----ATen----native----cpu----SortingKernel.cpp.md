# `.\pytorch\aten\src\ATen\native\cpu\SortingKernel.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS
// 定义宏，用于禁用 Torch 操作符的断言检查

#include <limits>
// 包含 C++ 标准库中的 <limits> 头文件

#include <ATen/native/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/TopKImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif
// 包含各种 ATen 和 C10 库的头文件，用于实现排序和张量操作

namespace at::native {

namespace {
// 匿名命名空间，用于封装局部实现的辅助函数和结构体

template <typename func_t>
void _dim_apply(
    const TensorBase &values,
    const TensorBase &indices,
    int64_t dim,
    const std::string& method_name,
    const func_t& f) {
  // 对指定维度上的张量应用函数，并更新索引
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .build();
  // 创建张量迭代器，配置迭代器以处理指定的操作

  auto values_dim_stride = values.stride(dim);
  auto indices_dim_stride = indices.stride(dim);
  auto dim_size = values.size(dim);
  // 获取指定维度的步幅和尺寸信息

  AT_DISPATCH_V2(
    iter.dtype(), "sorting_kernel_method_name", AT_WRAP([&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* values_data_bytes = data[0];
        auto* indices_data_bytes = data[1];

        if(values_data_bytes==nullptr || indices_data_bytes==nullptr){
          return;
        }
        // 循环处理数据的每个元素，应用给定函数

        for (const auto i C10_UNUSED : c10::irange(n)) {
          f(
            reinterpret_cast<scalar_t*>(values_data_bytes),
            values_dim_stride,
            reinterpret_cast<int64_t*>(indices_data_bytes),
            indices_dim_stride,
            dim_size
          );

          values_data_bytes += strides[0];
          indices_data_bytes += strides[1];
        }
      };

      int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, dim_size);
      iter.for_each(loop, /*grain_size=*/grain_size);
    }), kBool, kHalf, kBFloat16, AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
  );
  // 根据张量类型分发排序操作的内核函数，应用于指定的数据类型
}

template <typename scalar_t>
struct KeyValueCompAsc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (!_isnan<scalar_t>(get<0>(lhs)) && _isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) < get<0>(rhs));
  }
};
// 定义结构体，用于升序比较键值对

template <typename scalar_t>
struct KeyValueCompDesc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (_isnan<scalar_t>(get<0>(lhs)) && !_isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) > get<0>(rhs));
  }
};
// 定义结构体，用于降序比较键值对

#ifdef USE_FBGEMM
// 如果定义了 USE_FBGEMM 宏，则包含相关的 FBGEMM 头文件
// 检查是否可以使用基数排序的函数
static bool can_use_radix_sort(const TensorBase& values, const bool descending) {
  // 基数排序只能用于一维数据
  if (values.dim() != 1) return false;
  // 基数排序默认按升序排序
  if (descending) return false;
  // 基数排序仅适用于整数值
  if (!at::isIntegralType(values.scalar_type(), /*includeBool=*/false)) return false;
  // 当张量大小大于设定的 GRAIN_SIZE 或者基数排序加速使用 OpenMP 时，性能提升显著
  if (values.numel() < at::internal::GRAIN_SIZE || !fbgemm::is_radix_sort_accelerated_with_openmp()) return false;
  // TODO(DamianSzwichtenberg): 基数排序是一种稳定排序算法，这里是否应该检查 stable 是否设置为 true？

  return true;
}

// 并行排序的核函数，用于对一维张量进行排序
static void parallel_sort1d_kernel(
    const TensorBase& values,
    const TensorBase& indices) {
  AT_DISPATCH_INTEGRAL_TYPES(values.scalar_type(), "parallel_sort1d_kernel", [&] {
    const auto elements = values.numel();
    auto* const keys = values.data_ptr<scalar_t>();
    auto* const vals = indices.data_ptr<int64_t>();
    std::vector<scalar_t> tmp_keys(elements);
    std::vector<int64_t> tmp_vals(elements);
    const scalar_t* sorted_keys = nullptr;
    const int64_t* sorted_vals = nullptr;
    // 调用 fbgemm 库中的基数排序并行函数进行排序
    std::tie(sorted_keys, sorted_vals) = fbgemm::radix_sort_parallel(
        keys,
        vals,
        tmp_keys.data(),
        tmp_vals.data(),
        elements,
        std::numeric_limits<scalar_t>::max(),
        values.scalar_type() != ScalarType::Byte);

    // 检查是否排序在原位完成
    const bool sorted_in_place = keys == sorted_keys;
    if (!sorted_in_place) {
      const auto num_threads = at::get_num_threads();
      // 使用并行循环来映射排序结果到原始数据
      at::parallel_for(0, elements, elements / num_threads, [&](int64_t begin, int64_t end) {
        const auto job_size = end - begin;
        vec::map([](vec::Vectorized<scalar_t> x) -> vec::Vectorized<scalar_t> { return x; }, keys + begin, sorted_keys + begin, job_size);
        vec::map([](vec::Vectorized<int64_t> x) -> vec::Vectorized<int64_t> { return x; }, vals + begin, sorted_vals + begin, job_size);
      });
    }
  });
}

// 排序核函数，用于对张量中的数据进行排序
static void sort_kernel(
    const TensorBase& self,
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim,
    bool descending,
    bool stable) {
  // 根据值的维度确定要操作的维度
  dim = maybe_wrap_dim(dim, values.dim());
  // 填充索引张量以便排序
  _fill_indices(indices, dim);
  // 如果指定维度的步长为零，则不执行排序操作
  if (self.stride(dim) == 0) {
    // 检查步长是否为零，参考：https://github.com/pytorch/pytorch/issues/91420
    return;
  }
#ifdef USE_FBGEMM
  // 如果可以使用基数排序，则调用并行排序核函数
  if (can_use_radix_sort(values, descending)) {
    parallel_sort1d_kernel(values, indices);
    return;
  }
#endif
  // 否则，使用一般的排序操作
  _dim_apply(
    values, indices, dim,
    "sort_cpu", [&](
      auto* values, int64_t values_dim_stride,
      auto* indices, int64_t indices_dim_stride,
      int64_t dim_size
    ) {
      // 定义一个类型别名 scalar_t，表示 values 指针指向的类型
      using scalar_t = typename std::remove_pointer<decltype(values)>::type;
      // 创建 values_accessor，用于访问 values 数组的元素
      auto values_accessor = StridedRandomAccessor<scalar_t>(
        values, values_dim_stride);
      // 创建 indices_accessor，用于访问 indices 数组的元素
      auto indices_accessor = StridedRandomAccessor<int64_t>(
        indices, indices_dim_stride);
      // 创建 composite_accessor，将 values_accessor 和 indices_accessor 组合成一个访问器
      auto composite_accessor = CompositeRandomAccessorCPU<
        decltype(values_accessor), decltype(indices_accessor)
      >(values_accessor, indices_accessor);

      // 如果需要降序排序
      if (descending) {
        // 如果要求稳定排序
        if (stable) {
          // 使用稳定排序对 composite_accessor 进行降序排序
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
        else {
          // 使用普通排序对 composite_accessor 进行降序排序
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
      }
      else {
        // 如果要求稳定排序
        if (stable) {
          // 使用稳定排序对 composite_accessor 进行升序排序
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
        else {
          // 使用普通排序对 composite_accessor 进行升序排序
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
      }
    }
  );
} // 结束 topk_kernel 函数的定义

static void topk_kernel(
    const TensorBase &values, // values 参数：用于存储 topk 结果的张量
    const TensorBase &indices, // indices 参数：用于存储 topk 结果在原始张量中的索引的张量
    const TensorBase &self, // self 参数：输入的原始张量
    int64_t k, // k 参数：表示要获取的 topk 元素的数量
    int64_t dim, // dim 参数：沿着哪个维度计算 topk
    bool largest, // largest 参数：是否获取最大的 topk 元素
    bool sorted) { // sorted 参数：是否返回排序的结果
  auto sizes = self.sizes(); // 获取输入张量的大小

  // 根据输入张量的大小和指定的维度创建一个张量迭代器
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false) // 禁止检查所有张量是否具有相同的数据类型
    .resize_outputs(false) // 不调整输出张量的大小
    .declare_static_shape(sizes, /*squash_dims=*/dim) // 声明静态形状，可能压缩指定维度
    .add_output(values) // 将 values 参数添加为输出张量
    .add_output(indices) // 将 indices 参数添加为输出张量
    .add_const_input(self) // 将 self 参数添加为输入张量
    .build(); // 构建张量迭代器

  auto mode_values_stride = values.strides()[dim]; // 获取 values 参数在指定维度上的步长
  auto mode_indices_stride = indices.strides()[dim]; // 获取 indices 参数在指定维度上的步长
  auto tmp_values_stride = self.strides()[dim]; // 获取 self 参数在指定维度上的步长

  // 使用模板和自动分发，处理所有数据类型的情况，调用 topk 实现函数
  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "topk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      if (self.scalar_type() == ScalarType::BFloat16) {
        return topk_impl_loop<scalar_t, float>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n); // 调用 topk 实现循环函数，处理 BFloat16 类型的情况
      } else {
        return topk_impl_loop<scalar_t, scalar_t>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n); // 调用 topk 实现循环函数，处理其他数据类型的情况
      }
    };

    // 计算并发执行的粒度大小，基于内部 GRAIN_SIZE 常量和输入张量的大小
    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    // 使用迭代器并行执行循环函数
    iter.for_each(loop, /*grain_size=*/grain_size);
  });
}

} // 匿名命名空间的结束标记

// 注册 sort_stub 分发函数，将 sort_kernel 与其关联
REGISTER_DISPATCH(sort_stub, &sort_kernel);
// 注册 topk_stub 分发函数，将 topk_kernel 与其关联
REGISTER_DISPATCH(topk_stub, &topk_kernel);

} //at::native 命名空间的结束标记
```