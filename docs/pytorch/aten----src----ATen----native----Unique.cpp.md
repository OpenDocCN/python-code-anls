# `.\pytorch\aten\src\ATen\native\Unique.cpp`

```py
// 定义宏以限制为仅方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/irange.h>
#include <c10/util/Load.h>

// 如果未定义每个操作符的头文件，则包含下列头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/_unique_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

// 当数据类型为bool时，此unique实现通过使用归约算法，从UniqueCub.cu映射得来，
// 用于找到true值的数量。
std::tuple<Tensor, Tensor, Tensor> unique_cpu_bool_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts) {
  // 确保输入是连续的
  const Tensor& input = self.contiguous();
  // 获取输入数据的指针
  const bool* input_data = input.const_data_ptr<bool>();

  // 获取输入数据的元素数量
  int64_t numel = input.numel();
  // 创建空的输出张量
  Tensor output = at::empty({0}, self.options());
  // 创建空的反向索引张量
  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
  // 创建空的计数张量
  Tensor counts = at::empty({0}, self.options().dtype(kLong));

  // 如果输入张量为空，则根据需要调整反向索引张量的大小，并返回空的输出张量、反向索引张量和计数张量
  if (numel == 0) {
    if (return_inverse) {
      inverse_indices.resize_(input.sizes());
    }
    return std::make_tuple(output, inverse_indices, counts);
  }

  // 获取线程数
  int num_threads = at::get_num_threads();
  // 创建存储每个线程中true值数量的向量
  std::vector<int64_t> num_true_thread(num_threads, 0);

  // 定义并行操作的粒度大小
  const int64_t grain_size = at::internal::GRAIN_SIZE;
  // 并行处理每个元素，统计true值的数量
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    for (const auto i : c10::irange(begin, end)) {
      const bool value = c10::load(&input_data[i]);
      if (value) {
        num_true_thread[tid]++;
      }
    }
  });

  // 计算总的true值数量和false值数量
  int64_t num_true = std::accumulate(num_true_thread.begin(), num_true_thread.end(), 0);
  int64_t num_false = numel - num_true;
  // 计算输出张量的大小
  int num_out = ((num_true > 0) + (num_false > 0));

  // 定义false值在输出张量中的索引位置
  constexpr int false_idx = 0;
  // 定义true值在输出张量中的索引位置
  const int true_idx = num_false > 0;

  // 调整输出张量的大小
  output.resize_({num_out});
  // 如果需要计数，则调整计数张量的大小
  if (return_counts) {
    counts.resize_({num_out});
  }
  // 获取输出数据和计数数据的指针
  bool* output_data = output.data_ptr<bool>();
  int64_t* counts_data = return_counts ? counts.data_ptr<int64_t>() : nullptr;

  // 填充输出和计数数据
  if (num_false > 0) {
    output_data[false_idx] = false;
    if (return_counts) {
      counts_data[false_idx] = num_false;
    }
  }
  if (num_true > 0) {
    output_data[true_idx] = true;
    if (return_counts) {
      counts_data[true_idx] = num_true;
    }
  }

  // 如果需要返回反向索引，则调整其大小并获取其数据指针
  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
    int64_t* inverse_indices_data = inverse_indices.data_ptr<int64_t>();
    // 使用 ATen 库的 parallel_for 函数并行处理数据，范围从 0 到 numel，以 grain_size 为粒度
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      // 遍历指定范围的数据索引 i
      for (const auto i : c10::irange(begin, end)) {
        // 从 input_data 加载数据到布尔值 value
        const bool value = c10::load(&input_data[i]);
        // 根据 value 的真假将 inverse_indices_data[i] 设置为 true_idx 或 false_idx
        inverse_indices_data[i] = value ? true_idx : false_idx;
      }
    });
  }
  // 返回三元组，包括 output、inverse_indices 和 counts
  return std::make_tuple(output, inverse_indices, counts);
// }

// 检查索引 i 处的元素是否“唯一”，在排序序列中，第一个元素始终为 true。
//
// NaN 在排序序列中被传播到最后，考虑一个排序序列：
//   {1.0, 1.0, 2.0, 2.0, NaN, NaN, NaN}
//
// a. 如果 `equal_nan` == true，则结果为：
//   {T,   F,   T,   F,   T,   F,   F  }
//
// b. 如果 `equal_nan` == false，则结果为：
//   {T,   F,   T,   F,   T,   T,   T  }
//
template <typename scalar_t, bool equal_nan>
struct IsUnique {};

// 当 `equal_nan` == false 时的 IsUnique 结构体的特化版本
template <typename scalar_t>
struct IsUnique<scalar_t, false> {
  // 检查数据指针 data_ptr 中索引 i 处的元素是否与前一个元素不同
  inline bool operator() (scalar_t* data_ptr, int64_t i) {
    if (i == 0) { return true; }
    return c10::load(&data_ptr[i]) != c10::load(&data_ptr[i - 1]);
  }
};

// 当 `equal_nan` == true 时的 IsUnique 结构体的特化版本
template <typename scalar_t>
struct IsUnique<scalar_t, true> {
  // 检查数据指针 data_ptr 中索引 i 处的元素是否与前一个元素不同，并且不是 NaN
  inline bool operator() (scalar_t* data_ptr, int64_t i) {
    if (i == 0) { return true; }
    return (c10::load(&data_ptr[i]) != c10::load(&data_ptr[i - 1]))
        && !(_isnan(data_ptr[i]) && _isnan(data_ptr[i - 1]));
  }
};

// 注意：此处实现的 unique 使用了排序算法
//
// 整个算法源自 NumPy 的 numpy/lib/arraysetops.py，
// 首先对输入序列进行排序，然后将其转换为连续的唯一值。
//
// 本实现在 NumPy 版本的基础上进行了改进：并行计算 `inverse_indices` 和 `counts`，
// 在一个融合循环中几乎免费地实现了这一部分。
//
// 此内核还实现了一个 `equal_nan` 标志，其功能与 NumPy 的 unique 相同。当前此标志始终禁用。
//
// TODO: 添加 `bool` 特化版本，使用类似于 UniqueCub 的方法
//
template <typename scalar_t, typename CompareOp>
std::tuple<Tensor, Tensor, Tensor> unique_cpu_sorted_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    CompareOp is_unique) {
  // 对输入张量进行连续化处理
  const Tensor& input = self.contiguous();

  // 获取输入张量的元素个数
  int64_t numel = input.numel();

  // 创建空的输出张量
  Tensor output = at::empty({0}, self.options());

  // 创建空的反向索引张量和计数张量
  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor counts = at::empty({0}, self.options().dtype(kLong));

  // 如果输入张量为空
  if (numel == 0) {
    // 如果需要返回反向索引，则调整反向索引张量的大小
    if (return_inverse) {
      inverse_indices.resize_(input.sizes());
    }
    return std::make_tuple(output, inverse_indices, counts);
  }

  // index of first unique in each consecutive section
  // this is used to compute counts for parallelization purpose
  // 创建一个空的 Tensor 用于存储每个连续部分中第一个唯一元素的索引
  Tensor unique_index = at::empty({0}, self.options().dtype(kLong));

  // original behavior with unique on scalar tensor
  // is to return a output size of ([1]), `flatten` here will do the job
  // 在标量 Tensor 上调用 unique 的原始行为是返回 ([1]) 大小的输出，`flatten` 在此处实现相同效果
  auto input_flattened = input.flatten();

  // 对输入进行排序，同时返回排序后的 Tensor 和排序索引
  Tensor input_sorted, indices;
  std::tie(input_sorted, indices) = input_flattened.sort();

  // 获取排序后输入数据的指针和排序索引数据的指针
  scalar_t* input_sorted_data = input_sorted.data_ptr<scalar_t>();
  int64_t* indices_data = indices.data_ptr<int64_t>();

  // 获取当前线程数
  int num_threads = at::get_num_threads();
  // 创建一个 vector 用于存储每个线程中的唯一元素计数
  std::vector<int64_t> unique_count_thread(num_threads, 0);
  // 创建一个 vector 用于存储每个线程中的偏移量
  std::vector<int64_t> offset_thread(num_threads, 0);

  // 设置每个任务的粒度
  const int64_t grain_size = at::internal::GRAIN_SIZE;

  // 使用并行计算每个线程中的唯一元素计数
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    for (const auto i : c10::irange(begin, end)) {
      // 如果当前元素在排序后的输入中是唯一的，则增加当前线程的唯一元素计数
      if (is_unique(input_sorted_data, i)) {
        unique_count_thread[tid]++;
      }
    }
  });

  // 计算每个线程在输出中的偏移量，并计算总的唯一元素数量
  int64_t unique_count = 0;
  for (const auto t : c10::irange(num_threads)) {
    offset_thread[t] = unique_count;
    unique_count += unique_count_thread[t];
  }

  // 调整输出 Tensor 的大小以容纳所有唯一元素
  output.resize_({unique_count});
  scalar_t* output_data = output.data_ptr<scalar_t>();

  // 初始化反向索引数据指针为空
  int64_t* inverse_indices_data = nullptr;
  if (return_inverse) {
    // 如果需要返回反向索引，则调整反向索引 Tensor 的大小
    inverse_indices.resize_(input.sizes());
    inverse_indices_data = inverse_indices.data_ptr<int64_t>();
  }

  // 初始化计数数据指针为空
  int64_t* counts_data = nullptr;
  int64_t* unique_index_data = nullptr;
  if (return_counts) {
    // 如果需要返回计数，则调整计数 Tensor 和唯一索引 Tensor 的大小
    counts.resize_({unique_count});
    counts_data = counts.data_ptr<int64_t>();

    unique_index.resize_({unique_count + 1});
    unique_index_data = unique_index.data_ptr<int64_t>();
    unique_index_data[unique_count] = numel;
  }

  // 使用并行计算填充输出 Tensor 和相关的索引数据
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    int64_t offset = offset_thread[tid];

    for (const auto i : c10::irange(begin, end)) {
      // 如果当前元素在排序后的输入中是唯一的，则将其添加到输出中
      if (is_unique(input_sorted_data, i)) {
        output_data[offset] = c10::load(&input_sorted_data[i]);
        // 如果需要返回计数，则记录唯一索引
        if (return_counts) {
          unique_index_data[offset] = i;
        }
        offset++;
      }

      // 如果需要返回反向索引，则在反向索引中记录索引位置
      if (return_inverse) {
        int64_t inverse_index = offset - 1;
        int64_t perm = indices_data[i];
        inverse_indices_data[perm] = inverse_index;
      }
    }
  });

  if (return_counts) {
    // 计算相邻唯一元素之间的差异以获取计数
    at::parallel_for(0, unique_count, grain_size, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        counts_data[i] = unique_index_data[i + 1] - unique_index_data[i];
      }
    });
  }

  // 返回包含输出、反向索引和计数的元组
  return std::make_tuple(output, inverse_indices, counts);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_consecutive_cpu_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts) {
  const Tensor& input = self.contiguous();  // 获取输入张量的连续版本
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();  // 获取输入张量的常量数据指针
  int64_t numel = input.numel();  // 获取输入张量的元素数量
  Tensor output = at::empty({numel}, input.options());  // 创建一个空的张量作为输出
  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));  // 创建一个空的长整型张量用于存储逆索引
  Tensor counts = at::empty({0}, self.options().dtype(kLong));  // 创建一个空的长整型张量用于存储计数

  if (return_inverse) {  // 如果需要返回逆索引
    inverse_indices.resize_(input.sizes());  // 调整逆索引张量的大小与输入张量相同
  }

  if (numel > 0) {  // 如果输入张量不为空
    scalar_t *output_data = output.data_ptr<scalar_t>();  // 获取输出张量的数据指针
    int64_t *inverse_data = inverse_indices.data_ptr<int64_t>();  // 获取逆索引张量的数据指针
    int64_t *counts_data = nullptr;  // 初始化计数数据指针为nullptr
    scalar_t last_value = c10::load(input_data);  // 加载输入数据的第一个值作为初始值
    *output_data = last_value;  // 将初始值存储到输出张量中的第一个位置

    if (return_counts) {  // 如果需要返回计数
      counts.resize_({numel});  // 调整计数张量的大小为与输入张量相同
      counts_data = counts.data_ptr<int64_t>();  // 获取计数张量的数据指针
    }
    scalar_t *p = output_data;  // 设置指向输出数据的指针
    int64_t *q = counts_data;  // 设置指向计数数据的指针
    int64_t last = 0;  // 初始化上一个索引值为0
    if (return_inverse) {
      inverse_data[0] = 0;  // 如果需要返回逆索引，则将第一个位置的逆索引设置为0
    }
    for (const auto i : c10::irange(1, numel)) {  // 遍历输入张量的元素
      const auto value = c10::load(&input_data[i]);  // 加载当前位置的数据值
      if (value != last_value) {  // 如果当前值与上一个值不相同
        *(++p) = value;  // 在输出张量中存储当前值
        last_value = value;  // 更新上一个值为当前值
        if (return_counts) {
          *(q++) = i - last;  // 在计数张量中存储当前分段的长度
          last = i;  // 更新上一个索引为当前索引
        }
      }
      if (return_inverse) {
        inverse_data[i] = p - output_data;  // 在逆索引张量中存储当前位置的逆索引
      }
    }
    int64_t output_size = p - output_data + 1;  // 计算输出张量的有效大小
    if (return_counts) {
      *q = numel - last;  // 存储最后一个分段的长度到计数张量中
      counts.resize_({output_size});  // 调整计数张量的大小为有效大小
    }
    output.resize_({output_size});  // 调整输出张量的大小为有效大小
  }

  return std::make_tuple(output, inverse_indices, counts);  // 返回输出张量、逆索引张量和计数张量的元组
}

template<class ForwardIt>
ForwardIt _unique_dim_cpu_impl(ForwardIt first, ForwardIt last,
  std::vector<int64_t>& indices, Tensor inverse_indices_vec, Tensor counts) {
    if (first == last) {  // 如果迭代器范围为空
      return last;  // 直接返回last迭代器
    }

    TORCH_INTERNAL_ASSERT(inverse_indices_vec.is_contiguous(),
        "_unique_dim_cpu_impl only support contiguous inverse_indices_vec");  // 断言逆索引向量是连续的
    TORCH_INTERNAL_ASSERT(counts.is_contiguous(),
        "_unique_dim_cpu_impl only support contiguous counts");  // 断言计数张量是连续的

    int64_t *indices_data = indices.data();  // 获取索引向量的数据指针
    int64_t *inverse_data = inverse_indices_vec.data_ptr<int64_t>();  // 获取逆索引向量的数据指针
    int64_t *counts_data = counts.data_ptr<int64_t>();  // 获取计数张量的数据指针

    ForwardIt result = first;  // 设置结果迭代器为第一个迭代器
    ForwardIt previous = first;  // 设置前一个迭代器为第一个迭代器
    int64_t *current_counts = counts_data;  // 设置当前计数指针为计数数据的起始位置
    inverse_data[*(indices_data++)] = 0;  // 在逆索引向量中设置第一个索引的逆索引为0
    for (ForwardIt current = std::next(first); current != last; current++) {  // 遍历迭代器范围
      if (!at::equal(*current, *result)) {  // 如果当前值与结果值不相等
        *(++result) = std::move(*current);  // 在结果位置存储当前值
        *(current_counts++) = std::distance(previous, current);  // 存储当前分段的长度到计数数据中
        previous = current;  // 更新前一个迭代器为当前迭代器
      }
      inverse_data[*(indices_data++)] = std::distance(first, result);  // 在逆索引向量中存储当前位置的逆索引
    }
    *current_counts = std::distance(previous, last);  // 存储最后一个分段的长度到计数数据中
    return ++result;  // 返回最后一个结果迭代器的后一个位置
  }

template <typename scalar_t>
// 定义一个模板函数，用于在 CPU 上执行唯一值操作，返回一个包含三个 Tensor 的元组
std::tuple<Tensor, Tensor, Tensor> _unique_dim_cpu_template(
    const Tensor& self,                   // 输入的张量
    const int64_t dim,                    // 操作的维度
    const bool consecutive,               // 是否要求连续排序
    const bool return_inverse,            // 是否返回反向索引
    const bool return_counts) {           // 是否返回计数

    auto sizes = self.sizes().vec();      // 获取输入张量的尺寸向量
    auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);  // 统计尺寸向量中的零维数量

    // 如果指定维度的大小为0，则处理
    if (self.size(dim) == 0){
      TORCH_CHECK(
          num_zero_dims == 1,
          "Number of zero sized dimensions is more than one, so unique cannot be applied ")
      Tensor output = at::empty(sizes, self.options());  // 创建一个空的输出张量
      Tensor inverse_indices =
          at::empty({0}, self.options().dtype(kLong));   // 创建空的反向索引张量
      Tensor counts = at::empty({0}, self.options().dtype(kLong));  // 创建空的计数张量

      return std::make_tuple(output, inverse_indices, counts);  // 返回结果元组
    }

    TORCH_CHECK(num_zero_dims == 0,
    "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  // 将输入张量沿指定维度移动并展平为二维
  Tensor input_flat = self.moveaxis(dim, 0);  // 将指定维度移动到首维
  auto orig_sizes = input_flat.sizes().vec(); // 获取移动后的尺寸向量
  input_flat = input_flat.contiguous().view({input_flat.size(0), -1});  // 展平为二维数组

  std::vector<int64_t> indices(input_flat.size(0));  // 创建索引向量
  std::iota(indices.begin(), indices.end(), 0);      // 填充索引向量为 0 到 size-1
  int64_t numel = input_flat.size(1);                // 获取第二维的大小
  const scalar_t* input_flat_ptr = ((const scalar_t*)input_flat.const_data_ptr());  // 获取输入张量数据指针

  // 根据数据对索引进行排序
  if (!consecutive) {
    std::sort(indices.begin(), indices.end(),
      [&](int64_t a, int64_t b) -> bool {
        for (const auto i : c10::irange(numel)) {
          scalar_t lhs = c10::load(&input_flat_ptr[i + a * numel]);  // 加载左右两边的数据
          scalar_t rhs = c10::load(&input_flat_ptr[i + b * numel]);
          if (lhs < rhs) {   // 如果左边小于右边，则返回 true
            return true;
          } else if (lhs > rhs) {  // 如果左边大于右边，则返回 false
            return false;
          }
        }
        return false;  // 默认返回 false
      });
  }

  Tensor input_sorted;
  if (!consecutive) {
    input_sorted = at::empty(input_flat.sizes(), input_flat.options());  // 创建与输入相同的空张量
    for (const auto i : c10::irange(indices.size())) {
      input_sorted[i] = input_flat[indices[i]];  // 根据排序结果填充排序后的张量
    }
  } else {
    input_sorted = input_flat;  // 如果要求连续排序，则直接使用未排序的张量
  }

  Tensor inverse_indices = at::empty(indices.size(), self.options().dtype(kLong));  // 创建反向索引张量
  Tensor counts = at::zeros(indices.size(), self.options().dtype(kLong));           // 创建计数张量
  std::vector<Tensor> input_unbind = at::unbind(input_sorted, 0);  // 解绑排序后的张量
  auto last = _unique_dim_cpu_impl(
    input_unbind.begin(), input_unbind.end(), indices, inverse_indices, counts);  // 调用唯一值实现函数
  input_unbind.erase(last, input_unbind.end());  // 删除未使用的部分
  counts = at::narrow(counts, 0, 0, input_unbind.size());  // 缩小计数张量的大小

  // 将解绑后的张量重新堆叠起来
  auto output = at::stack(input_unbind, 0);
  auto new_sizes = std::vector<int64_t>(std::move(orig_sizes));
  new_sizes[0] = -1;  // 将首维大小设置为 -1
  output = output.view(new_sizes);  // 根据新尺寸重新视图化输出张量
  output = output.moveaxis(0, dim);  // 将首维移动回指定维度

  return std::make_tuple(output, inverse_indices, counts);  // 返回最终结果元组
}

} // namespace
// 定义名为 _unique_cpu 的函数，处理给定的张量 self，确定是否排序 sorted 和是否返回逆向映射 return_inverse
_unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse) {
  // 如果张量 self 的标量类型为布尔型 kBool
  if (self.scalar_type() == kBool) {
    // 使用布尔型模板 unique_cpu_bool_template 处理，不返回计数信息
    auto [output, inverse, _] = unique_cpu_bool_template(
        self, return_inverse, /* return_counts */false);
    // 返回输出和逆向映射的元组
    return std::make_tuple(output, inverse);
  }
  // 否则，根据张量的标量类型进行分发处理
  return AT_DISPATCH_V2(self.scalar_type(), "unique", [&] AT_WRAP({
    // 当前的 CPU 实现在进行 unique 操作时总是进行排序，因为这比使用哈希表更快
    auto [output, inverse, _] = unique_cpu_sorted_template<scalar_t>(
        self, return_inverse, /* return_counts */false, IsUnique<scalar_t, /* equal_nan */false>());
    // 返回输出和逆向映射的元组
    return std::make_tuple(output, inverse);
  }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// 定义名为 _unique2_cpu 的函数，处理给定的张量 self，确定是否排序 sorted、是否返回逆向映射 return_inverse 和是否返回计数 return_counts
std::tuple<Tensor, Tensor, Tensor>
_unique2_cpu(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  // 如果张量 self 的标量类型为布尔型 kBool
  if (self.scalar_type() == kBool) {
    // 使用布尔型模板 unique_cpu_bool_template 处理，返回逆向映射和计数信息
    return unique_cpu_bool_template(self, return_inverse, return_counts);
  }
  // 否则，根据张量的标量类型进行分发处理
  return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
    // 当前的 CPU 实现在进行 unique 操作时总是进行排序，因为这比使用哈希表更快
    return unique_cpu_sorted_template<scalar_t>(
        self, return_inverse, return_counts, IsUnique<scalar_t, /* equal_nan */ false>());
  }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// 定义名为 unique_dim_cpu 的函数，处理给定的张量 self 和维度 dim，确定是否排序 sorted、是否返回逆向映射 return_inverse 和是否返回计数 return_counts
std::tuple<Tensor, Tensor, Tensor>
unique_dim_cpu(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  // 根据张量的标量类型进行分发处理
  return AT_DISPATCH_V2(self.scalar_type(), "unique_dim", AT_WRAP([&] {
    // 当前的实现在使用维度 dim 进行操作时总是进行排序，因为不可哈希的张量
    return _unique_dim_cpu_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kBool, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// 定义名为 unique_dim_consecutive_cpu 的函数，处理给定的张量 self 和维度 dim，确定是否返回逆向映射 return_inverse 和是否返回计数 return_counts
std::tuple<Tensor, Tensor, Tensor>
unique_dim_consecutive_cpu(const Tensor& self, const int64_t dim, const bool return_inverse, const bool return_counts) {
  // 根据张量的标量类型进行分发处理
  return AT_DISPATCH_V2(self.scalar_type(), "unique_dim", AT_WRAP([&] {
    // 使用 _unique_dim_cpu_template 处理，维度 dim 上的操作是连续的，返回逆向映射和计数信息
    return _unique_dim_cpu_template<scalar_t>(self, dim, true, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kBool, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// 定义名为 unique_consecutive_cpu 的函数，处理给定的张量 self，确定是否返回逆向映射 return_inverse、是否返回计数 return_counts 和是否指定维度 dim
std::tuple<Tensor, Tensor, Tensor>
unique_consecutive_cpu(const Tensor& self, const bool return_inverse, const bool return_counts, std::optional<int64_t> dim) {
  // 如果未指定维度 dim 或者维度为 0 且张量 self 是一维的
  if (!dim.has_value() || (dim.value() == 0 && self.dim() == 1)) {
    // 根据张量的标量类型进行分发处理
    return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
      // 使用模板 unique_consecutive_cpu_template 进行处理，返回逆向映射和计数信息
      return unique_consecutive_cpu_template<scalar_t>(self, return_inverse, return_counts);
    }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kBool, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
  // 否则，对给定维度 dim 进行连续操作处理
  return unique_dim_consecutive_cpu(self, dim.value(), return_inverse, return_counts);
}

}  // namespace at::native
```