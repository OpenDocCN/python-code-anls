# `.\pytorch\aten\src\ATen\CPUApplyUtils.h`

```py
#pragma once
// 包含必要的头文件

#include <ATen/CollapseDims.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>
#include <cstring>
#include <limits>

namespace at {

/*
 * apply 的基本策略如下：
 *
 * 1. 从最外层索引开始循环，直到达到一个非连续的维度，即该维度的步长不等于由外部维度定义的张量大小。
 *    我们称之为外部（连续）张量 A。注意，如果张量是连续的，则 A 等于整个张量。
 *    我们称内部张量为 B。
 *
 * 2. 我们遍历 B 中的索引，从其最外层维度开始。例如，如果 B 是一个 2x2 的矩阵，则我们会：
 *
 *    B[0][0]
 *    B[0][1]
 *    B[1][0]
 *    B[1][1]
 *
 *    我们将偏移设置为 (storageOffset + stride_B * index_B)，
 *    也就是说，我们基本上像对张量进行正常操作一样计算存储中的偏移量。
 *    但由于我们保证后续数据在内存中是连续的，所以我们可以简单地循环 sizeof(A) 次并执行操作，
 *    而无需遵循 A 的步长所描述的顺序。
 *
 * 3. 作为优化，我们合并内存中连续的 A 维度。例如，如果 A 是从一个 3x3x4x3 张量缩窄而来的 3x3x3x3 张量，
 *    那么前两个维度可以在 APPLY 的目的下合并，从而减少嵌套循环的数量。
 */

// 排序张量的步长，返回排列后的张量
inline Tensor sort_strides(Tensor& tensor_) {
  IntArrayRef strides = tensor_.strides();
  std::vector<int64_t> indices;
  indices.reserve(tensor_.ndimension());
  // 生成张量维度的索引数组
  for (const auto i : c10::irange(tensor_.ndimension())) {
    indices.push_back(i);
  }
  // 根据步长大小降序排序索引数组
  std::sort(indices.begin(), indices.end(), [&strides](int64_t i1, int64_t i2) {
    return strides[i1] > strides[i2];
  });
  // 根据排序后的索引数组重新排列张量
  Tensor tensor = tensor_.permute(indices);
  return tensor;
}

// 固定维度的张量迭代器
template <typename T, int N>
struct strided_tensor_iter_fixed {
 public:
  T* data_ = NULL;  // 数据指针
  int64_t dim_ = 0; // 张量维度

  int64_t counter_[N] = {0}; // 计数器数组
  int64_t sizes_[N] = {0};   // 张量尺寸数组
  int64_t strides_[N] = {0}; // 张量步长数组

  // 禁用拷贝构造函数和赋值操作符
  strided_tensor_iter_fixed(strided_tensor_iter_fixed const&) = delete;
  void operator=(strided_tensor_iter_fixed const& x) = delete;
  strided_tensor_iter_fixed(strided_tensor_iter_fixed&&) = default;

  // 构造函数，初始化迭代器
  strided_tensor_iter_fixed(
      Tensor& tensor,
      C10_UNUSED bool sort_strides = false)
      : data_(tensor.data_ptr<T>()) {
    std::memset(counter_, 0, sizeof(int64_t) * N);
    // 如果张量维度大于 0，则复制尺寸和步长数组
    if (tensor.dim() > 0) {
      std::memcpy(
          sizes_, tensor.sizes().data(), tensor.dim() * sizeof(int64_t));
      std::memcpy(
          strides_, tensor.strides().data(), tensor.dim() * sizeof(int64_t));
    }
    // 获取合并后的维度
    dim_ = std::get<1>(collapse_dims(sizes_, strides_, tensor.ndimension()));
  }
};

template <typename T>


这段代码中，主要涉及了张量操作的优化和迭代器的实现，注释解释了每个函数和结构的作用及其关键步骤。
// 定义一个结构体 strided_tensor_iter，用于迭代多维张量的元素
struct strided_tensor_iter {
 private:
 public:
  T* data_ = NULL;  // 指向张量数据的指针，默认为空指针
  int64_t dim_;     // 张量的维度

  std::vector<int64_t> counter_;  // 记录当前迭代位置的计数器数组
  std::vector<int64_t> sizes_;    // 张量的各维度大小数组
  std::vector<int64_t> strides_;  // 张量的各维度步长数组

  strided_tensor_iter(strided_tensor_iter const&) = delete;  // 禁用拷贝构造函数
  void operator=(strided_tensor_iter const& x) = delete;     // 禁用赋值运算符
  strided_tensor_iter(strided_tensor_iter&&) = default;      // 默认移动构造函数
  strided_tensor_iter(Tensor& tensor)
      : data_(tensor.data_ptr<T>()),                       // 初始化数据指针为张量的数据指针
        dim_(tensor.ndimension()),                         // 初始化维度为张量的维度数
        counter_(dim_, 0),                                 // 初始化计数器数组为全零
        sizes_(tensor.sizes().vec()),                      // 初始化大小数组为张量各维度的大小
        strides_(tensor.strides().vec()) {                 // 初始化步长数组为张量各维度的步长
    dim_ = std::get<1>(collapse_dims(sizes_.data(), strides_.data(), dim_));  // 调用 collapse_dims 函数，获取折叠后的维度
  }
};

// 检查数组中所有张量的元素个数是否相等
inline bool _all_equal_numel(at::ArrayRef<Tensor> tensors) {
  if (tensors.empty())
    return true;
  int64_t all_numel = tensors[0].numel();  // 获取第一个张量的元素个数
  for (const auto i : c10::irange(1, tensors.size())) {
    if (tensors[i].numel() != all_numel)   // 检查后续张量的元素个数是否与第一个相等
      return false;
  }
  return true;
}

// 构造包含所有张量尺寸信息的错误信息字符串
inline std::string _all_equal_numel_error(at::ArrayRef<Tensor> tensors) {
  std::ostringstream oss;
  oss << "inconsistent tensor size, expected ";
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].sizes() << ", ";  // 拼接每个张量的尺寸信息
  }
  oss << "and " << tensors[tensors.size() - 1].sizes()
      << " to have the same number of elements, but got ";
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].numel() << ", ";  // 拼接每个张量的元素个数
  }
  oss << "and " << tensors[tensors.size() - 1].numel()
      << " elements respectively";
  return oss.str();  // 返回完整的错误信息字符串
}

// 执行应用前的准备工作，检查设备类型、布局和所有张量的元素个数
inline bool _apply_preamble(ArrayRef<Tensor> tensors) {
  checkDeviceType("CPU_tensor_apply", tensors, kCPU);    // 检查张量是否在 CPU 上
  checkLayout("CPU_tensor_apply", tensors, kStrided);    // 检查张量是否是连续布局
  if (!_all_equal_numel(tensors))                       // 检查所有张量的元素个数是否相等
    AT_ERROR(_all_equal_numel_error(tensors));          // 若不相等则抛出错误
  // 空张量没有元素
  for (auto& t : tensors)
    if (t.numel() == 0)
      return false;                                    // 如果存在空张量则返回 false
  return true;                                          // 所有条件满足，返回 true
}

// 获取数组中张量的最大维度
inline int64_t _max_dim_tensors(ArrayRef<Tensor> tensors) {
  int64_t dim = 0;
  for (auto& t : tensors)
    dim = std::max(dim, t.ndimension());  // 比较获取最大的张量维度
  return dim;                            // 返回最大的张量维度
}

// 空实现，用于模板参数展开
inline void iterate(int64_t /*size*/){};

// 递归调用，更新迭代器位置和数据指针
template <typename Arg, typename... Args>
inline void iterate(int64_t size, Arg& iter, Args&... iter_tail) {
  iter.counter_[iter.dim_ - 1] += size;                                     // 更新计数器
  iter.data_ = iter.data_ + size * iter.strides_[iter.dim_ - 1];             // 更新数据指针
  iterate(size, iter_tail...);                                              // 递归调用处理剩余参数
}

// 检查是否继续迭代的条件，所有迭代器都未到达末尾则继续
inline bool iterate_continue() {
  return true;
};

// 递归调用，检查所有迭代器是否都未到达末尾
template <typename Arg, typename... Args>
inline bool iterate_continue(Arg& iter, Args&... iter_tail) {
  return iter.counter_[iter.dim_ - 1] < iter.sizes_[iter.dim_ - 1] &&
      iterate_continue(iter_tail...);  // 逐个检查迭代器是否未到达末尾
}

// 获取单次迭代的最大尺寸，避免超出张量的实际大小
inline int64_t max_iterate_size() {
  return std::numeric_limits<int64_t>::max();  // 返回 int64_t 的最大值
};

// 递归调用，获取所有迭代器中最小的可迭代尺寸
template <typename Arg, typename... Args>
inline int64_t max_iterate_size(Arg& iter, Args&... iter_tail) {
  return std::min(
      (iter.sizes_[iter.dim_ - 1] - iter.counter_[iter.dim_ - 1]),
      max_iterate_size(iter_tail...));  // 比较获取最小的可迭代尺寸
}

// 空实现，用于模板参数展开
inline void iterate_overflow(){};

// 递归调用，用于展开模板参数
template <typename Arg, typename... Args>
/*
  Iterate overflow handling for tensor iterators

  This function recursively adjusts iterator counters and data pointers when the iterator
  exceeds its boundary, ensuring correct iteration over multi-dimensional tensors.
*/
inline void iterate_overflow(Arg& iter, Args&... iter_tail) {
  // Check if the current iterator has overflowed
  if (iter.counter_[iter.dim_ - 1] == iter.sizes_[iter.dim_ - 1]) {
    // Iterate backwards through dimensions to handle overflow
    for (int64_t i = iter.dim_ - 1; i > 0; i--) {
      if (iter.counter_[i] == iter.sizes_[i]) {
        // Reset current dimension counter and adjust data pointer
        iter.counter_[i] = 0;
        iter.counter_[i - 1]++;
        iter.data_ = iter.data_ - (iter.sizes_[i] * iter.strides_[i]) + iter.strides_[i - 1];
      }
    }
  }
  // Recursively call for remaining iterators
  iterate_overflow(iter_tail...);
}

/*
  Forward iterator adjustment function

  Adjusts data pointers of iterators based on the given offset, facilitating movement
  through tensor dimensions.
*/
inline void forward(int64_t /*offset*/){};

/*
  Forward iterator adjustment function (variadic)

  Adjusts data pointers of multiple iterators based on the given offset, iterating through
  each dimension accordingly.
*/
template <typename Arg, typename... Args>
inline void forward(int64_t offset, Arg& iter, Args&... iter_tail) {
  int64_t multi = offset;
  // Iterate backwards through dimensions to compute offsets
  for (int64_t i = iter.dim_ - 1; i >= 0; i--) {
    int64_t inc = multi % iter.sizes_[i];
    multi = multi / iter.sizes_[i];
    iter.data_ = iter.data_ + inc * iter.strides_[i];
    iter.counter_[i] += inc;
  }
  // Recursively call forward for remaining iterators
  forward(offset, iter_tail...);
}

/*
  Maximum dimension calculator

  Returns the maximum dimensionality among a set of tensor iterators.
*/
inline int64_t max_dim() {
  return 0;
}

/*
  Maximum dimension calculator (variadic)

  Returns the maximum dimensionality among a set of tensor iterators.
*/
template <typename Arg, typename... Args>
inline int64_t max_dim(Arg& iter, Args&... iter_tail) {
  // Recursively find the maximum dimension among all iterators
  return std::max(iter.dim_, max_dim(iter_tail...));
}

/*
  Apply a pointwise operator to sequence of tensors

  Applies a given operation (op) to a sequence of tensors, iterating through their elements
  based on specified iterators and handling overflow across dimensions.
*/
inline void apply_op(){};

/*
  Apply a pointwise operator to sequence of tensors (variadic)

  Applies a given operation (op) to a sequence of tensors, iterating through their elements
  based on specified iterators and handling overflow across dimensions.
*/
template <typename Op, typename... Args>
inline void apply_op(
    int64_t numel,
    int64_t offset,
    const Op& op,
    Args... iters) {
  // For 0-dim tensors, directly apply the operator
  if (numel == 1 && max_dim(iters...) == 0) {
    op(*iters.data_...);
    return;
  }
  // Adjust iterators by the given offset
  if (offset > 0)
    forward(offset, iters...);
  // Iteratively apply the operator to tensor elements
  for (int64_t i = 0; i < numel;) {
    for (; iterate_continue(iters...) && i < numel;) {
      op(*iters.data_...);
      iterate(1, iters...);
      i++;
    }
    iterate_overflow(iters...);
  }
}

/*
  Apply a pointwise operator to two tensors

  Applies a given operation (op) pointwise to two tensors, utilizing optimized
  fixed-size iterators when the maximum dimensionality is low (<= 8).
*/
template <typename scalar1, typename scalar2, typename Op>
inline void CPU_tensor_apply2(Tensor tensor1, Tensor tensor2, const Op op) {
  // Check and handle any pre-application conditions
  if (!_apply_preamble({tensor1, tensor2}))
    return;
  // Select appropriate iterator type based on tensor dimensionality
  if (_max_dim_tensors({tensor1, tensor2}) <= 8) {
    apply_op(
        tensor1.numel(),
        0,
        op,
        strided_tensor_iter_fixed<scalar1, 8>(tensor1),
        strided_tensor_iter_fixed<scalar2, 8>(tensor2));
  } else {
    apply_op(
        tensor1.numel(),
        0,
        op,
        strided_tensor_iter<scalar1>(tensor1),
        strided_tensor_iter<scalar2>(tensor2));
  }
}

/*
  Apply a pointwise operator to three tensors

  Applies a given operation (op) pointwise to three tensors, utilizing optimized
  fixed-size iterators when the maximum dimensionality is low (<= 8).
*/
template <typename scalar1, typename scalar2, typename scalar3, typename Op>
inline void CPU_tensor_apply3(
    Tensor tensor1,
    Tensor tensor2,
    Tensor tensor3,
    const Op op) {
  // Check and handle any pre-application conditions
  if (!_apply_preamble({tensor1, tensor2, tensor3}))
    return;
  // Select appropriate iterator type based on tensor dimensionality
  if (_max_dim_tensors({tensor1, tensor2, tensor3}) <= 8) {
    // Apply the operation using fixed-size iterators

        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter_fixed<scalar1, 8>(tensor1),
            strided_tensor_iter_fixed<scalar2, 8>(tensor2),
            strided_tensor_iter_fixed<scalar3, 8>(tensor3));
    } else {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter<scalar1>(tensor1),
            strided_tensor_iter<scalar2>(tensor2),
            strided_tensor_iter<scalar3>(tensor3));
    }
}
    # 如果张量的元素数目为8的倍数，则使用固定步长的迭代器处理
    if (tensor1.numel() % 8 == 0) {
        # 调用函数 apply_op 处理张量，传入操作符 op 和固定步长迭代器的结果
        apply_op(
            tensor1.numel(),  # 张量1的元素数目
            0,  # 起始位置设为0
            op,  # 操作符
            strided_tensor_iter_fixed<scalar1, 8>(tensor1),  # 张量1的固定步长迭代器
            strided_tensor_iter_fixed<scalar2, 8>(tensor2),  # 张量2的固定步长迭代器
            strided_tensor_iter_fixed<scalar3, 8>(tensor3));  # 张量3的固定步长迭代器
    } else {
        # 如果张量的元素数目不是8的倍数，则使用普通步长的迭代器处理
        apply_op(
            tensor1.numel(),  # 张量1的元素数目
            0,  # 起始位置设为0
            op,  # 操作符
            strided_tensor_iter<scalar1>(tensor1),  # 张量1的普通步长迭代器
            strided_tensor_iter<scalar2>(tensor2),  # 张量2的普通步长迭代器
            strided_tensor_iter<scalar3>(tensor3));  # 张量3的普通步长迭代器
    }
// 结束 at 命名空间的定义

template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename Op>
// 定义一个模板函数，接受四个张量和一个操作符 Op
inline void CPU_tensor_apply4(
    Tensor tensor1,  // 第一个张量
    Tensor tensor2,  // 第二个张量
    Tensor tensor3,  // 第三个张量
    Tensor tensor4,  // 第四个张量
    const Op op) {   // 传入的操作符 Op
  // 如果应用前处理返回 false，则直接返回
  if (!_apply_preamble({tensor1, tensor2, tensor3, tensor4}))
    return;
  // 如果四个张量中的最大维度都小于等于 8
  if (_max_dim_tensors({tensor1, tensor2, tensor3, tensor4}) <= 8) {
    // 调用 apply_op 函数，使用固定步长为 8 的张量迭代器
    apply_op(
        tensor1.numel(),  // 张量1 的元素数量
        0,                // 起始索引为 0
        op,               // 操作符 Op
        strided_tensor_iter_fixed<scalar1, 8>(tensor1),  // 张量1 的固定步长迭代器
        strided_tensor_iter_fixed<scalar2, 8>(tensor2),  // 张量2 的固定步长迭代器
        strided_tensor_iter_fixed<scalar3, 8>(tensor3),  // 张量3 的固定步长迭代器
        strided_tensor_iter_fixed<scalar4, 8>(tensor4)); // 张量4 的固定步长迭代器
  } else {
    // 否则调用 apply_op 函数，使用普通步长的张量迭代器
    apply_op(
        tensor1.numel(),  // 张量1 的元素数量
        0,                // 起始索引为 0
        op,               // 操作符 Op
        strided_tensor_iter<scalar1>(tensor1),  // 张量1 的普通步长迭代器
        strided_tensor_iter<scalar2>(tensor2),  // 张量2 的普通步长迭代器
        strided_tensor_iter<scalar3>(tensor3),  // 张量3 的普通步长迭代器
        strided_tensor_iter<scalar4>(tensor4)); // 张量4 的普通步长迭代器
  }
}

// 结束 CPU_tensor_apply4 函数模板的定义
} // namespace at
// 结束 at 命名空间的定义
```