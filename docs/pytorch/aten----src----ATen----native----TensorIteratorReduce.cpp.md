# `.\pytorch\aten\src\ATen\native\TensorIteratorReduce.cpp`

```py
/// 定义预处理宏，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
/// 包含张量迭代器的头文件
#include <ATen/TensorIterator.h>
/// 包含并行处理的头文件
#include <ATen/Parallel.h>
/// 包含张量迭代器内部实现的头文件
#include <ATen/TensorIteratorInternal.h>

#ifndef AT_PER_OPERATOR_HEADERS
/// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含函数头文件
#include <ATen/Functions.h>
#else
/// 否则，包含 empty 操作的头文件
#include <ATen/ops/empty.h>
#endif

#include <c10/util/irange.h>

/// 包含了 TensorIterator 中并行规约实现的内容
namespace at {

/// 定义循环类型别名为 loop2d_t
using loop2d_t = TensorIteratorBase::loop2d_t;

/// 静态函数，判断是否使用两遍规约
static bool use_two_pass_reduction(TensorIteratorBase& iter);
/// 静态函数，执行两遍规约
static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop);
/// 静态函数，执行维度并行规约
static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop);

/// TensorIteratorBase 类的成员函数，实现并行规约
void TensorIteratorBase::parallel_reduce(loop2d_t loop) {
  /// 检查张量数目为两个，否则抛出异常
  TORCH_CHECK(ntensors() == 2, "parallel_reduce only supports one input and one output");
  /// 获取张量迭代器中的元素总数
  int64_t numel = this->numel();
  /// 如果元素数量小于内部定义的粒度或者只有一个线程可用或者已处于并行区域
  if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
      at::in_parallel_region()) {
    /// 使用串行方式执行循环
    serial_for_each(loop, {0, numel});
  } else if (use_two_pass_reduction(*this)) {
    /// 如果满足两遍规约条件，则执行两遍规约
    two_pass_reduction(*this, loop);
  } else {
    /// 否则执行维度并行规约
    parallel_dim_reduction(*this, loop);
  }
}

/// 静态函数，判断是否使用两遍规约
static bool use_two_pass_reduction(TensorIteratorBase& iter) {
  return iter.output(0).numel() == 1;
}

/// 静态函数，执行两遍规约
static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  /// 获取当前可用的最大线程数
  const int max_threads = at::get_num_threads();

  /// 获取输出张量的引用
  const auto& dst = iter.output(0);
  /// 在第 0 维度上添加一个新维度
  auto unsqueezed = dst.unsqueeze(0);
  /// 创建一个与 unsqueezed 相同形状的缓冲区
  auto buffer_shape = DimVector(unsqueezed.sizes());
  buffer_shape[0] = max_threads;
  /// 根据缓冲区形状和数据类型选项创建一个空张量
  auto buffer = at::empty(buffer_shape, dst.options());
  /// 将缓冲区填充为单位矩阵
  buffer.copy_(unsqueezed);

  /// 计算缓冲区的步长
  auto buffer_stride = buffer.strides()[0] * buffer.element_size();
  /// 获取缓冲区的第一个元素
  auto buffer_0 = buffer[0];
  /// 执行第一次规约操作，将结果存储在第一个缓冲区中
  auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.input(0));
  /// 内部断言，确保 first_reduce 的输出是 buffer_0 的别名
  TORCH_INTERNAL_ASSERT(first_reduce.output(0).is_alias_of(buffer_0));

  /// 使用并行方式执行循环
  at::parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    /// 获取当前线程的线程编号
    const auto thread_num = at::get_thread_num();
    /// 获取 first_reduce 的形状和步长
    auto shape = first_reduce.shape();
    auto strides = first_reduce.get_strides();

    /// 增加输出指针，使每个线程都有自己的输出切片
    auto base_ptrs = first_reduce.get_base_ptrs();
    base_ptrs[0] += buffer_stride * thread_num;

    /// 使用串行方式执行每个线程的循环操作
    at::internal::serial_for_each(shape, strides, base_ptrs.data(),
                                  base_ptrs.size(), loop, {begin, end});
  });

  /// 执行最终的规约操作，将结果存储在 unsqueezed 中
  auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
  /// 应用最终的规约操作到整个循环中
  final_reduce.for_each(loop);
}

/// 选择要并行化的维度。优先选择最外层大于可用线程数的维度。
static int find_split_dim(TensorIteratorBase& iter) {
  /// 获取当前可用线程数
  int num_threads = at::get_num_threads();
  /// 获取迭代器的形状
  auto shape = iter.shape();

  /// 从最外层维度开始查找
  int best_dim = iter.ndim() - 1;
  for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
    /// 如果该维度大于等于可用线程数，则选择该维度作为分割维度
    if (shape[dim] >= num_threads) {
      return dim;
    } else if (shape[dim] > shape[best_dim]) {
      best_dim = dim;
    }
  }


// 如果当前维度的形状大于最佳维度的形状，则更新最佳维度为当前维度
} else if (shape[dim] > shape[best_dim]) {
  best_dim = dim;
}



  AT_ASSERT(!iter.is_dim_reduced(best_dim));
  return best_dim;


// 断言：验证在迭代器中，最佳维度没有被降维（即未被减少）
AT_ASSERT(!iter.is_dim_reduced(best_dim));
// 返回找到的最佳维度
return best_dim;
}

// 定义静态函数，用于对迭代器进行列的四舍五入处理
static std::tuple<int64_t, int64_t>
round_columns(TensorIteratorBase& iter, int dim, int multiple, int64_t begin, int64_t end) {
  // 将 begin 调整为 multiple 的整数倍
  begin = begin - (begin % multiple);
  // 如果 end 不是最后一列，则将 end 调整为 multiple 的整数倍
  if (end != iter.shape()[dim]) {
    end = end - (end % multiple);
  }
  return std::make_tuple(begin, end);
}

// 并行降维函数，根据给定的循环处理函数将迭代器操作并行化
static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  // 断言迭代器的维度至少为 1
  AT_ASSERT(iter.ndim() >= 1);
  // 查找可以分割的维度
  int dim = find_split_dim(iter);
  // 获取迭代器操作的列数
  int64_t cols = iter.shape()[dim];
  // 获取元素大小
  int element_size = iter.element_size(/*arg=*/1);

  // 判断是否应当对列进行四舍五入处理
  bool should_round_columns = iter.strides(1)[dim] == element_size;
  // 使用并行处理的方式，遍历所有列
  at::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
    if (should_round_columns) {
      // 如果相邻列在内存中是连续的，将列数调整为 128 字节的整数倍
      int64_t cols_per_128_bytes = 128 / element_size;
      std::tie(begin, end) = round_columns(iter, dim, cols_per_128_bytes, begin, end);
    }
    // 如果起始和结束列相同，则直接返回
    if (begin == end) {
      return;
    }
    // 创建子迭代器并按指定维度进行缩小
    auto sub_iter = TensorIterator(iter);
    sub_iter.narrow(dim, begin, end - begin);
    // 对子迭代器应用给定的循环函数
    sub_iter.for_each(loop);
  });
}

// 对降维后的每个元素进行操作的函数
void TensorIteratorBase::foreach_reduced_elt(loop_subiter_t loop, bool parallelize) {
  // 断言输入张量数为 1
  AT_ASSERT(ninputs() == 1);
  // 断言输出张量数至少为 1
  AT_ASSERT(noutputs() >= 1);

  // 获取迭代器的形状
  auto shape = this->shape();
  // 如果输出张量元素个数为 0，则直接返回
  if (output(0).numel() == 0) {
    return;
  }
  // 如果输出张量元素个数为 1，则直接应用循环函数于当前迭代器
  if (output(0).numel() == 1) {
    loop(*this);
  }
  // 如果张量元素个数小于或等于 GRAIN_SIZE 或者单线程运行，或者当前已经处于并行区域或者禁用了并行化
  else if (numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
      at::in_parallel_region() || !parallelize) {
    // 计算降维后张量的非降维形状及其元素数目
    auto reduce_dims = num_reduce_dims();
    auto non_reduced_shape = shape.slice(reduce_dims, shape.size() - reduce_dims);
    int64_t non_reduced_numel = 1;
    for (const auto i : non_reduced_shape) {
      non_reduced_numel *= i;
    }
    // 使用 DimCounter 对非降维形状进行迭代
    DimCounter dims {non_reduced_shape, {0, non_reduced_numel}};
    while (!dims.is_done()) {
      // 复制当前迭代器并选择保留指定维度的全部数据
      TensorIterator reduced = *this;
      reduced.select_all_keeping_dim(reduce_dims, dims.values);
      // 对复制的迭代器应用给定的循环函数
      loop(reduced);
      dims.increment({1, 1});
    }
  }
  // 否则，使用并行化方式对张量的指定维度进行处理
  else {
    int dim = find_split_dim(*this);
    int64_t
```