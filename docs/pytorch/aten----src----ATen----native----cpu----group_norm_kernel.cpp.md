# `.\pytorch\aten\src\ATen\native\cpu\group_norm_kernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义预处理指令，用于指定仅支持特定方法的操作符

#include <ATen/native/group_norm.h>
// 包含 GroupNorm 的原生实现头文件

#include <algorithm>
#include <array>
#include <numeric>
// 包含标准库中的算法和数组相关头文件

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/moments_utils.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/OpMathType.h>
#include <c10/util/irange.h>
// 包含 ATen 库中的各种头文件，用于张量操作和向量化功能

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif
// 根据条件包含不同的 ATen 库头文件

namespace at::native {

namespace {
// 进入 at::native 命名空间并定义匿名命名空间，用于限定实现细节的作用域

template <typename T, typename PT>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  // 定义 GroupNorm 内核的实现，接受输入参数并操作张量

  TORCH_CHECK(X.numel() == N * C * HxW);
  // 检查输入张量 X 的元素数量是否符合预期

  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  // 检查 gamma 张量是否已定义且元素数量与通道数 C 相符

  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  // 检查 beta 张量是否已定义且元素数量与通道数 C 相符

  const int64_t G = group;
  // 定义 G 为 group 参数的值，表示分组规模

  const int64_t D = C / G;
  // 计算 D，表示每个分组的通道数

  const T* X_data = X.const_data_ptr<T>();
  // 获取输入张量 X 的常量数据指针

  const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;
  // 获取 gamma 张量的常量数据指针，如果未定义则设为 nullptr

  const PT* beta_data = beta.defined() ? beta.const_data_ptr<PT>() : nullptr;
  // 获取 beta 张量的常量数据指针，如果未定义则设为 nullptr

  T* Y_data = Y.data_ptr<T>();
  // 获取输出张量 Y 的数据指针

  PT* mean_data = mean.data_ptr<PT>();
  // 获取均值张量 mean 的数据指针

  PT* rstd_data = rstd.data_ptr<PT>();
  // 获取标准差的倒数张量 rstd 的数据指针

  const bool gamma_null = (gamma_data == nullptr);
  // 检查 gamma_data 是否为 nullptr，用于后续条件判断

  const bool beta_null = beta_data == nullptr;
  // 检查 beta_data 是否为 nullptr，用于后续条件判断

  const int64_t inner_size = D * HxW;
  // 计算内部大小，即每个分组内部的元素数量

  using opmath_t = at::opmath_type<T>;
  // 定义操作类型 opmath_t，根据 T 类型确定

  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    // 并行循环，处理每个分组内的数据

    for (const auto i : c10::irange(start, end)) {
      // 对于当前处理的分组索引 i

      const T* X_ptr = X_data + i * inner_size;
      // 获取当前分组在输入张量中的起始指针位置

      auto [mean_val, rstd_val] = RowwiseMoments(X_ptr, inner_size);
      // 调用 RowwiseMoments 函数计算当前分组的均值和标准差

      rstd_val = opmath_t(1) / std::sqrt(std::max(rstd_val, opmath_t(0)) + eps);
      // 计算修正后的标准差的倒数，避免除以零情况

      if (gamma_null && beta_null) {
        // 如果 gamma 和 beta 均未定义

        T* Y_ptr = Y_data + i * inner_size;
        // 获取输出张量中当前分组的起始位置

        for (const auto j : c10::irange(inner_size)) {
          // 对于当前分组内的每个元素 j

          Y_ptr[j] = (X_ptr[j] - mean_val) * rstd_val;
          // 计算归一化后的输出值，并存入 Y_ptr 中
        }
      } else {
        // 如果 gamma 或 beta 至少有一个已定义

        const int64_t g = i % G;
        // 计算当前处理的分组索引 g

        for (const auto j : c10::irange(D)) {
          // 对于当前分组内的每个通道 j

          const int64_t c = g * D + j;
          // 计算当前通道在整个张量中的索引 c

          const opmath_t scale = rstd_val * (gamma_null ? opmath_t(1) : opmath_t(gamma_data[c]));
          // 计算当前通道的缩放比例

          const opmath_t bias = -scale * mean_val + (beta_null ? opmath_t(0) : opmath_t(beta_data[c]));
          // 计算当前通道的偏置

          X_ptr = X_data + (i * D + j) * HxW;
          // 更新输入数据指针到当前通道的起始位置

          T* Y_ptr = Y_data + (i * D + j) * HxW;
          // 更新输出数据指针到当前通道的起始位置

          for (const auto k : c10::irange(HxW)) {
            // 对于当前通道内的每个元素 k

            Y_ptr[k] = scale * X_ptr[k] + bias;
            // 计算经缩放和偏置后的输出值，并存入 Y_ptr 中
          }
        }
      }
      mean_data[i] = mean_val;
      // 将计算得到的均值存入均值张量中的相应位置

      rstd_data[i] = rstd_val;
      // 将计算得到的标准差的倒数存入标准差张量中的相应位置
    }
  });
}

template <typename T>
typename std::enable_if<std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<T, T>>::type
ColumnwiseMoments(
    const T* X_data,
    int64_t HxW,
    int64_t C,



    // 定义 ColumnwiseMoments 函数模板，计算输入张量的列方向均值和标准差

    // 参数说明：
    // - X_data: 输入张量的数据指针
    // - HxW: 每个通道的大小（高度乘以宽度）
    // - C: 输入张量的通道数

    const int64_t D = C;
    // 设置 D 为通道数 C，因为这里计算的是每列的统计信息

    T mean = at::opmath_type<T>(0);
    // 初始化均值为零

    T mean_of_square = at::opmath_type<T>(0);
    // 初始化平方均值为零

    const T* X_ptr = X_data;
    // 设置 X_ptr 指向输入数据的起始位置

    for (int64_t i = 0; i < C; ++i) {
        // 对于每个通道 i

        const T* X_col_ptr = X_ptr + i * HxW;
        // 获取当前通道的起始位置

        for (int64_t j = 0; j < HxW; ++j) {
            // 对于当前通道内的每个元素 j

            mean += X_col_ptr[j];
            // 累加当前元素到均值中

            mean_of_square += X_col_ptr[j] * X_col_ptr[j];
            // 累加当前元素的平方到平方均值中
        }
    }

    mean /= static_cast<T>(C * HxW);
    // 计算均值，除以总元素数量

    mean_of_square /= static_cast<T>(C * HxW);
    // 计算平方均值，除以总元素数量

    T variance = mean_of_square - mean * mean;
    // 计算方差

    variance = std::max(variance, static_cast<T>(0));
    // 确保方差为非负数

    T std_dev = std::sqrt(variance);
    // 计算标准差

    return std::make_tuple(mean, std_dev);
    // 返回均值和标准差的元组
}



} // 匿名命名空间的结束

void GroupNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  // 实现 GroupNorm 的核心函数，调用内部实现函数处理输入张量

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, X.scalar_type(), "GroupNorm", [&] {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，根据输入张量的数据类型分发不同的处理逻辑

    GroupNormKernelImplInternal<scalar_t, accscalar_t>(
        X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
    // 调用内部模板函数 GroupNormKernelImplInternal 处理具体逻辑
  });
}

} // at::native 命名空间的结束



#endif



#include <ATen/ops/empty.h>
// 根据条件包含 ATen 库的 empty.h 头文件



} // at::native 命名空间的结束



} // 匿名命名空间的结束



#endif



#include <ATen/Functions.h>
// 根据条件包含 ATen 库的 Functions.h 头文件



#endif



} // at::native 命名空间的结束



#endif



} // at::native 命名空间的结束



} // at::native 命名空间的结束



} // at::native 命名空间的结束



} // at::native 命名空间的结束



} // at::native 命名空间的结束
    // 使用 Vectorized<T> 类型的别名 Vec 进行向量化计算
    using Vec = vec::Vectorized<T>;
    // 定义向量大小 K，使用 constexpr 使其成为编译时常量
    constexpr int64_t K = Vec::size();
    // 计算内部循环的大小，使其为 K 的整数倍
    const int64_t inner_size = D / K * K;
    // 初始化累加向量 acc0_vec 和 acc1_vec，初始值为零向量
    Vec acc0_vec{0}, acc1_vec{0};
    // 外部循环遍历图像的每个像素点
    for (const auto m : c10::irange(HxW)) {
        // 获取指向图像数据 X_ptr 的指针，偏移量为 m * C
        const T* X_ptr = X_data + m * C;
        // 初始化 d 为零
        int64_t d = 0;
        // 内部循环处理每个像素点的数据
        for (; d < inner_size; d += K) {
            // 加载 X_ptr + d 处的数据到向量 x_vec
            Vec x_vec = Vec::loadu(X_ptr + d);
            // 累加 x_vec 到 acc0_vec
            acc0_vec += x_vec;
            // 累加 x_vec 的平方到 acc1_vec
            acc1_vec += x_vec * x_vec;
        }
        // 处理剩余部分，若 D - d 大于零
        if (D - d > 0) {
            // 加载 X_ptr + d 处开始，剩余 D - d 大小的数据到向量 x_vec
            Vec x_vec = Vec::loadu(X_ptr + d, D - d);
            // 累加 x_vec 到 acc0_vec
            acc0_vec += x_vec;
            // 累加 x_vec 的平方到 acc1_vec
            acc1_vec += x_vec * x_vec;
        }
    }
    // 计算 acc0_vec 的所有元素的总和，得到均值 mean_val
    T mean_val = vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, acc0_vec);
    // 计算 acc1_vec 的所有元素的总和，得到标准差的平方 rstd_val
    T rstd_val = vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, acc1_vec);
    // 返回均值和标准差的平方的元组
    return std::tuple<T, T>(mean_val, rstd_val);
// 结束一个 C++ 函数模板定义
}

// 如果 T 不是 at::BFloat16 或 at::Half，返回一个包含两个 opmath_type<T> 类型元素的元组
template <typename T>
typename std::enable_if<!std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<at::opmath_type<T>, at::opmath_type<T>>>::type
ColumnwiseMoments(
    const T* X_data,       // 输入数据的指针，类型为 T
    int64_t HxW,           // 数据的高度乘以宽度
    int64_t C,             // 数据的通道数
    int64_t D) {           // 数据的维度
  using opmath_t = at::opmath_type<T>;    // 定义 T 对应的 opmath_type 类型
  using Vec = vec::Vectorized<T>;         // 使用 Vec 表示 T 对应的向量化类型
  using fVec = vec::Vectorized<opmath_t>; // 使用 fVec 表示 opmath_t 对应的向量化类型
  constexpr int64_t K = Vec::size();      // K 是 Vec 的长度
  const int64_t inner_size = D / K * K;    // 内部循环的大小，保证是 K 的整数倍
  fVec acc0_fvec{0}, acc1_fvec{0}, zero{0};  // 初始化向量化累加器和零向量
  for (const auto m : c10::irange(HxW)) {  // 外部循环遍历高度乘以宽度的范围
    const T* X_ptr = X_data + m * C;      // 指向当前数据块的指针
    int64_t d = 0;                        // 初始化内部循环计数器
    for (; d < inner_size; d += K) {      // 内部循环，每次处理 K 个元素
      Vec x_bvec = Vec::loadu(X_ptr + d); // 加载未对齐的向量化数据
      auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);  // 将数据转换为浮点类型
      acc0_fvec += x_fvec0 + x_fvec1;     // 更新累加器 acc0_fvec
      acc1_fvec += x_fvec0 * x_fvec0 + x_fvec1 * x_fvec1;  // 更新累加器 acc1_fvec
    }
    if (D - d > 0) {                      // 处理剩余不足 K 的数据
      Vec x_bvec = Vec::loadu(X_ptr + d, D - d);  // 加载剩余数据
      auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);  // 将数据转换为浮点类型
      if (D - d > fVec::size()) {         // 如果剩余数据大于一个 fVec 的长度
        x_fvec1 = fVec::set(zero, x_fvec1, D - d - fVec::size());  // 设置第二部分数据
        acc0_fvec += x_fvec0 + x_fvec1;   // 更新累加器 acc0_fvec
        acc1_fvec += x_fvec0 * x_fvec0 + x_fvec1 * x_fvec1;  // 更新累加器 acc1_fvec
      } else {                            // 剩余数据不足一个 fVec 的长度
        x_fvec0 = fVec::set(zero, x_fvec0, D - d);  // 设置第一部分数据
        acc0_fvec += x_fvec0;             // 更新累加器 acc0_fvec
        acc1_fvec += x_fvec0 * x_fvec0;   // 更新累加器 acc1_fvec
      }
    }
  }
  opmath_t mean_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, acc0_fvec);  // 计算均值
  opmath_t rstd_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, acc1_fvec);  // 计算标准差的平方
  return std::tuple<opmath_t, opmath_t>(mean_val, rstd_val);  // 返回均值和标准差的平方
}

// 如果 T 和 opmath_t 相同，计算输入数据 X_ptr 的均值和标准差的平方
template <typename T, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
CalcMeanVar(
  const T* X_ptr,    // 输入数据的指针，类型为 T
  opmath_t* mean_ptr,  // 均值的指针，类型为 opmath_t
  opmath_t* rstd_ptr,  // 标准差的平方的指针，类型为 opmath_t
  int64_t C) {        // 数据的通道数
  using Vec = vec::Vectorized<T>;   // 使用 Vec 表示 T 对应的向量化类型
  vec::map2<T>(         // 对数据进行逐元素的映射操作
          [](Vec x, Vec y) { return x + y; },   // 映射函数，计算 x 和 y 的和
          mean_ptr,     // 输出均值的指针
          X_ptr,        // 输入数据的指针
          mean_ptr,     // 输出均值的指针
          C);           // 数据的通道数
  vec::map2<T>(         // 对数据进行逐元素的映射操作
      [](Vec x, Vec y) { return x * x + y; },   // 映射函数，计算 x 的平方加上 y
      rstd_ptr,         // 输出标准差的平方的指针
      X_ptr,            // 输入数据的指针
      rstd_ptr,         // 输出标准差的平方的指针
      C);               // 数据的通道数
}

// 如果 T 不是 at::BFloat16 或 at::Half，计算输入数据 X_ptr 的均值和标准差的平方
template <typename T, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
CalcMeanVar(
  const T* X_ptr,        // 输入数据的指针，类型为 T
  opmath_t* mean_ptr,    // 均值的指针，类型为 opmath_t
  opmath_t* rstd_ptr,    // 标准差的平方的指针，类型为 opmath_t
  int64_t C) {           // 数据的通道数
  using fVec = vec::Vectorized<opmath_t>;  // 使用 fVec 表示 opmath_t 对应的向量化类型
  using Vec = vec::Vectorized<T>;          // 使用 Vec 表示 T 对应的向量化类型
  int64_t d = 0;                           // 初始化内部循环计数器
  for (; d < C - (C % Vec::size()); d += Vec::size()) {  // 内部循环，每次处理一个向量化数据块
    Vec data_bvec = Vec::loadu(X_ptr + d);  // 加载未对齐的向量化数据
    fVec mean_fvec0 = fVec::loadu(mean_ptr + d);  // 加载均值的向量化数据
    fVec mean_fvec1 = fVec::loadu(mean_ptr + d + fVec::size());  // 加载均值的向量化数据
    fVec rstd_fvec0 = fVec::loadu(rstd_ptr + d);  // 加载标准差的平方的向量化数据
    fVec rstd_fvec1 = fVec::loadu(rstd_ptr + d + fVec::size());  // 加载标准差的平方的向量化数据
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);  // 将数据转换为浮点类型
    mean_fvec0 = data_fvec0 + mean_fvec0;  // 更新均值的向量化数据
    mean_fvec1 = data_fvec1 + mean_fvec1;  // 更新均值的向量化数据
    rstd_fvec0 = data_fvec
    // 更新 rstd_fvec0，加上 data_fvec0 的平方
    rstd_fvec0 = data_fvec0 * data_fvec0 + rstd_fvec0;
    // 更新 rstd_fvec1，加上 data_fvec1 的平方
    rstd_fvec1 = data_fvec1 * data_fvec1 + rstd_fvec1;
    // 将 mean_fvec0 存储到 mean_ptr + d 处
    mean_fvec0.store(mean_ptr + d);
    // 将 mean_fvec1 存储到 mean_ptr + d + fVec::size() 处
    mean_fvec1.store(mean_ptr + d + fVec::size());
    // 将 rstd_fvec0 存储到 rstd_ptr + d 处
    rstd_fvec0.store(rstd_ptr + d);
    // 将 rstd_fvec1 存储到 rstd_ptr + d + fVec::size() 处
    rstd_fvec1.store(rstd_ptr + d + fVec::size());
  }
  // 如果还有剩余的 C - d 大于 0
  if (C - d > 0) {
    // 加载 X_ptr + d 处开始的 C - d 大小的数据到 Vec 对象 data_bvec
    Vec data_bvec = Vec::loadu(X_ptr + d, C - d);
    // 加载 mean_ptr + d 处开始的 (C - d) 或 fVec::size() 大小的数据到 mean_fvec0
    fVec mean_fvec0 = fVec::loadu(mean_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    // 加载 mean_ptr + d + fVec::size() 处开始的剩余数据到 mean_fvec1
    fVec mean_fvec1 = fVec::loadu(mean_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    // 加载 rstd_ptr + d 处开始的 (C - d) 或 fVec::size() 大小的数据到 rstd_fvec0
    fVec rstd_fvec0 = fVec::loadu(rstd_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    // 加载 rstd_ptr + d + fVec::size() 处开始的剩余数据到 rstd_fvec1
    fVec rstd_fvec1 = fVec::loadu(rstd_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    // 将 data_bvec 转换为浮点数并分别赋值给 data_fvec0 和 data_fvec1
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);
    // 更新 mean_fvec0，加上 data_fvec0
    mean_fvec0 = data_fvec0 + mean_fvec0;
    // 更新 mean_fvec1，加上 data_fvec1
    mean_fvec1 = data_fvec1 + mean_fvec1;
    // 更新 rstd_fvec0，加上 data_fvec0 的平方
    rstd_fvec0 = data_fvec0 * data_fvec0 + rstd_fvec0;
    // 更新 rstd_fvec1，加上 data_fvec1 的平方
    rstd_fvec1 = data_fvec1 * data_fvec1 + rstd_fvec1;
    // 将 mean_fvec0 存储到 mean_ptr + d 处，存储大小为 (C - d) 或 fVec::size()
    mean_fvec0.store(mean_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    // 将 mean_fvec1 存储到 mean_ptr + d + fVec::size() 处，存储大小为 (C - d) 或剩余部分大小
    mean_fvec1.store(mean_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    // 将 rstd_fvec0 存储到 rstd_ptr + d 处，存储大小为 (C - d) 或 fVec::size()
    rstd_fvec0.store(rstd_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    // 将 rstd_fvec1 存储到 rstd_ptr + d + fVec::size() 处，存储大小为 (C - d) 或剩余部分大小
    rstd_fvec1.store(rstd_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
  }
}

template <typename T, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ApplyScaleBias(
  T* Y_ptr,                      // 输出数据指针，用于存储处理后的数据
  const T* X_ptr,                // 输入数据指针，用于读取待处理的数据
  const opmath_t* scale_ptr,     // 缩放因子指针，用于每个数据点的缩放操作
  const opmath_t* bias_ptr,      // 偏置指针，用于每个数据点的偏置操作
  int64_t C                      // 数据通道数
) {
  using Vec = vec::Vectorized<T>; // 使用 Vectorized 类型 Vec 处理 T 类型的数据向量化操作
  vec::map3<T>(
    [](Vec x, Vec scale, Vec bias) { return x * scale + bias; },  // 对每个数据向量进行缩放和偏置操作
    Y_ptr,                        // 输出数据指针
    X_ptr,                        // 输入数据指针
    scale_ptr,                    // 缩放因子指针
    bias_ptr,                     // 偏置指针
    C                             // 数据通道数
  );
}

// std::is_same<T, at::BFloat16> || std::is_same<T, at::Half>
template <typename T, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ApplyScaleBias(
  T* Y_ptr,                      // 输出数据指针，用于存储处理后的数据
  const T* X_ptr,                // 输入数据指针，用于读取待处理的数据
  const opmath_t* scale_ptr,     // 缩放因子指针，用于每个数据点的缩放操作
  const opmath_t* bias_ptr,      // 偏置指针，用于每个数据点的偏置操作
  int64_t C                      // 数据通道数
) {
  using fVec = vec::Vectorized<opmath_t>;  // 使用 Vectorized 类型 fVec 处理 opmath_t 类型的数据向量化操作
  using Vec = vec::Vectorized<T>;          // 使用 Vectorized 类型 Vec 处理 T 类型的数据向量化操作
  int64_t d = 0;                          // 初始化迭代器 d 为 0
  for (; d < C - (C % Vec::size()); d += Vec::size()) {  // 使用向量化处理主循环，直到剩余数据小于一个向量的长度
    Vec data_bvec = Vec::loadu(X_ptr + d);                  // 加载未对齐的输入数据向量
    fVec scale_fvec0 = fVec::loadu(scale_ptr + d);          // 加载未对齐的缩放因子向量
    fVec scale_fvec1 = fVec::loadu(scale_ptr + d + fVec::size());  // 加载未对齐的缩放因子向量的第二部分
    fVec bias_fvec0 = fVec::loadu(bias_ptr + d);            // 加载未对齐的偏置向量
    fVec bias_fvec1 = fVec::loadu(bias_ptr + d + fVec::size());    // 加载未对齐的偏置向量的第二部分
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);  // 转换输入数据向量为浮点向量
    fVec out0 = data_fvec0 * scale_fvec0 + bias_fvec0;         // 计算第一个输出向量
    fVec out1 = data_fvec1 * scale_fvec1 + bias_fvec1;         // 计算第二个输出向量
    convert_from_float<T>(out0, out1).store(Y_ptr + d);         // 将处理后的数据存储到输出指针中
  }
  if (C - d > 0) {  // 处理剩余的数据，若存在
    Vec data_bvec = Vec::loadu(X_ptr + d, C - d);              // 加载未对齐的输入数据向量
    fVec scale_fvec0 = fVec::loadu(scale_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));  // 加载未对齐的缩放因子向量
    fVec scale_fvec1 = fVec::loadu(scale_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);  // 加载未对齐的缩放因子向量的第二部分
    fVec bias_fvec0 = fVec::loadu(bias_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));    // 加载未对齐的偏置向量
    fVec bias_fvec1 = fVec::loadu(bias_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);    // 加载未对齐的偏置向量的第二部分
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);  // 转换输入数据向量为浮点向量
    fVec out0 = data_fvec0 * scale_fvec0 + bias_fvec0;         // 计算第一个输出向量
    fVec out1 = data_fvec1 * scale_fvec1 + bias_fvec1;         // 计算第二个输出向量
    convert_from_float<T>(out0, out1).store(Y_ptr + d, C - d);  // 将处理后的数据存储到输出指针中
  }
}

template <typename T, typename PT>
void GroupNormKernelImplChannelsLastInternal(
    const Tensor& X,        // 输入张量 X
    const Tensor& gamma,    // 缩放因子张量 gamma
    const Tensor& beta,     // 偏置张量 beta
    int64_t N,              // batch 大小
    int64_t C,              // 通道数
    int64_t HxW,            // 高度乘宽度
    int64_t group,          // 组大小
    double eps,             // 用于数值稳定性的小值 epsilon
    Tensor& Y,              // 输出张量 Y
    Tensor& mean,           // 平均值张量 mean
    // 检查输入张量 X 的元素数量是否等于 N * C * HxW
    TORCH_CHECK(X.numel() == N * C * HxW);
    // 检查 gamma 张量是否未定义或其元素数量是否等于 C
    TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
    // 检查 beta 张量是否未定义或其元素数量是否等于 C
    TORCH_CHECK(!beta.defined() || beta.numel() == C);
    
    // 定义变量 G 为分组数 group，D 为每组的通道数 C / G
    const int64_t G = group;
    const int64_t D = C / G;
    
    // 获取输入张量 X 的常量数据指针
    const T* X_data = X.const_data_ptr<T>();
    // 获取 gamma 张量的常量数据指针，如果未定义则设为 nullptr
    const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;
    // 获取 beta 张量的常量数据指针，如果未定义则设为 nullptr
    const PT* beta_data = beta.defined() ? beta.const_data_ptr<PT>() : nullptr;
    
    // 获取输出张量 Y 的数据指针
    T* Y_data = Y.data_ptr<T>();
    // 获取均值张量 mean 的数据指针
    PT* mean_data = mean.data_ptr<PT>();
    // 获取标准差的倒数张量 rstd 的数据指针
    PT* rstd_data = rstd.data_ptr<PT>();
    
    // 定义 opmath_t 为 ATen 库中 T 类型的数学运算类型
    using opmath_t = at::opmath_type<T>;
    
    // 计算标准化的缩放因子 s，为 1 / (D * HxW)，其中 D 为每组通道数，HxW 为空间维度
    const opmath_t s = opmath_t(1) / static_cast<opmath_t>(D * HxW);
    
    // 判断 gamma_data 是否为 nullptr
    const bool gamma_null = (gamma_data == nullptr);
    // 判断 beta_data 是否为 nullptr
    const bool beta_null = beta_data == nullptr;
    
    // 注意：关于选择的算法：
    //
    // 在通道维度为最后维度时，GroupNorm 的输入形状为 {N, H, W, GD}，
    // 均值和标准差按每个 n 和 g 收集，涉及非相邻维度上的归约。
    // 我们可以在以下两种实现中并行处理：
    //
    // 实现一：在 N * G 上并行。只需要一个 OpenMP 会话，但每个线程的内存访问不连续。
    //
    // 实现二：在 N * HxW 上并行。每个线程的内存访问是连续的，但需要额外的临时缓冲区，大小为 {T, N, 2C}。
    //
    // 一般情况下，当 HxW 足够大时，实现二的性能更好，因为每个线程的数据量 {NHWC / T} 显著大于临时缓冲区 {2NC}。
    //
    constexpr int64_t feature_map_threshold = 1024;
    if (HxW < feature_map_threshold) {
        // 实现一：在 N * G 上并行。
        //
        // 对于每个 HxW 平面，缩放和偏置只计算一次
        Tensor buffer = at::empty({N * G, 2 * D}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
        // 获取缓冲区数据的指针
        opmath_t* buffer_data = buffer.data_ptr<opmath_t>();
    at::parallel_for(0, N * G, 1, [&](int64_t begin, int64_t end) {
      int64_t n{0}, g{0};
      data_index_init(begin, n, N, g, G);
      for (const auto i : c10::irange(begin, end)) {
        // step-1: for each n and g, collect sum of x and x2
        //
        // Note that using vec::map_reduce_all here is simpler to write
        // but it is slower since horizontal reduce from vec to scalar is slow.
        // So it is better to reduce with a vec across all HxW plain,
        // and do a horizontal add just once for each {n, g}.
        //
        // 计算每个 n 和 g，收集 x 和 x^2 的和
        // 使用 vec::map_reduce_all 可以更简单地编写，但由于从向量到标量的水平归约较慢，
        // 因此最好在所有 HxW 平面上使用向量进行归约，并仅在每个 {n, g} 上进行一次水平加法。

        auto [mean_val, rstd_val] = ColumnwiseMoments(
                X_data + n * HxW * C + g * D,
                HxW,
                C,
                D);

        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, opmath_t(0));
        rstd_val = opmath_t(1) / std::sqrt(rstd_val + eps);
        mean_data[i] = mean_val;
        rstd_data[i] = rstd_val;

        // step-2: calculate scale and bias
        opmath_t* scale_ptr = buffer_data + i * 2 * D;
        opmath_t* bias_ptr = scale_ptr + D;
        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[d] = rstd_val * (gamma_null ? opmath_t(1) : opmath_t(gamma_data[c]));
          bias_ptr[d] = -scale_ptr[d] * mean_val + (beta_null ? opmath_t(0) : opmath_t(beta_data[c]));
        }

        // step-3: apply scale and bias
        for (const auto m : c10::irange(HxW)) {
          const T* X_ptr = X_data + n * HxW * C + m * C + g * D;
          T* Y_ptr = Y_data + n * HxW * C + m * C + g * D;
          ApplyScaleBias<T, opmath_t>(Y_ptr, X_ptr, scale_ptr, bias_ptr, D);
        }

        data_index_step(n, N, g, G);
      }
    });



  } else {
    // impl-2: parallel on N * HxW.
    //
    // temp buffer holding x and x2
    int num_threads = at::get_num_threads();
    Tensor buffer = at::empty({num_threads, N, 2 * C},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value)).zero_();
    opmath_t* buffer_data = buffer.data_ptr<opmath_t>();
    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    opmath_t* tmp_buffer_data = tmp_buffer.data_ptr<opmath_t>();
    // step-1: accumulate on dimension of C
    //
    // In order to improve multi-core performance when N=1,
    // we parallel on the all the outer dimensions of N and HxW,
    // leaving the most inner dimension C for vectorization.
    //
    // Note that parallel on {N, HxW, G} is not feasible for some common configs,
    // e.g. say input shape is {1, 32, h, w} and G = 8,
    //   this will give D = 4 which is unable to take full SIMD length.
    //
    // To avoid thread conflict, we make use of a temp buffer of {T, N, 2C},
    //   firstly, reduce from {N, HxW, C} to {T, N, 2C}
    //

    // 实现-2：在 N * HxW 上并行。
    //
    // 临时缓冲区，保存 x 和 x^2
    int num_threads = at::get_num_threads();
    Tensor buffer = at::empty({num_threads, N, 2 * C},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value)).zero_();
    opmath_t* buffer_data = buffer.data_ptr<opmath_t>();
    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    opmath_t* tmp_buffer_data = tmp_buffer.data_ptr<opmath_t>();
    // step-1: accumulate on dimension of C
    //
    // 为了在 N=1 时提高多核性能，
    // 我们在 N 和 HxW 的所有外部维度上并行，
    // 将最内部的维度 C 留给矢量化。
    //
    // 注意，对 {N, HxW, G} 的并行对于一些常见的配置是不可行的，
    // 例如，假设输入形状为 {1, 32, h, w} 并且 G = 8，
    // 这将导致 D = 4，无法利用完整的 SIMD 长度。
    //
    // 为了避免线程冲突，我们使用了一个 {T, N, 2C} 的临时缓冲区，
    //   首先，从 {N, HxW, C} 减少到 {T, N, 2C}
    //
    // 使用ATen的并行函数对0到N*HxW的范围进行并行操作
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      // 获取当前线程的线程ID
      int tid = at::get_thread_num();
      // 计算当前线程对应的buffer数据起始指针
      opmath_t* buffer_ptr = buffer_data + tid * N * 2 * C;

      // 初始化数据索引n和m，以及相关计算
      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      // 对于指定范围内的每个元素进行计算
      for (const auto i : c10::irange(begin, end)) {
        // 计算当前元素对应的均值和标准差的指针位置
        opmath_t* mean_ptr = buffer_ptr + n * 2 * C;
        opmath_t* rstd_ptr = mean_ptr + C;
        // 获取输入数据X中当前元素对应的指针位置
        const T* X_ptr = X_data + i * C;
        // 调用CalcMeanVar函数计算当前元素的均值和标准差
        CalcMeanVar<T, opmath_t>(X_ptr, mean_ptr, rstd_ptr, C);
        // 更新数据索引n和m
        data_index_step(n, N, m, HxW);
      }
    });

    // step-2: 计算均值和标准差
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        // 初始化均值和标准差的值
        opmath_t mean_val{0}, rstd_val{0};
        // 遍历深度维度D
        for (const auto d : c10::irange(D)) {
          // 遍历线程数目标
          for (const auto t : c10::irange(num_threads)) {
            // 计算当前线程的buffer数据起始指针
            opmath_t* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            // 累加均值和标准差的计算结果
            mean_val += buffer_ptr[g * D + d];
            rstd_val += buffer_ptr[g * D + d + C];
           }
        }
        // 乘以缩放系数s
        mean_val *= s;
        // 计算标准差的平方并进行平方根计算
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, opmath_t(0));
        rstd_val = opmath_t(1) / std::sqrt(rstd_val + eps);
        // 将均值和标准差的计算结果存储到临时缓冲区中
        tmp_buffer_data[n * 2 * G + 2 * g] = mean_val;
        tmp_buffer_data[n * 2 * G + 2 * g + 1] = rstd_val;
      }
    }

    // step-3: 计算缩放和偏置
    //
    // 均值和标准差的形状为{N, G}，gamma和beta的形状为{G, D}。
    // 缩放和偏置的形状为{N, C}，因此我们可以在最后一步直接在维度C上进行向量化。
    //
    // 我们可以将步骤3和4合并为一个会话，但这种方式更好：
    //   a. D可能太小无法进行向量化；
    //   b. 避免重复计算缩放和偏置，每个HxW平面共享相同的缩放和偏置。
    //
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        // 计算当前缩放和偏置的起始指针
        opmath_t* scale_ptr = buffer_data + n * 2 * C;
        opmath_t* bias_ptr = scale_ptr + C;
        // 获取均值和标准差的值
        opmath_t mean_val = tmp_buffer_data[n * 2 * G + 2 * g];
        opmath_t rstd_val = tmp_buffer_data[n * 2 * G + 2 * g + 1];
        // 将均值和标准差的值存储到输出数组中
        mean_data[n * G + g] = mean_val;
        rstd_data[n * G + g] = rstd_val;

        // 遍历深度维度D
        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          // 计算当前缩放和偏置的值
          scale_ptr[c] = rstd_val * (gamma_null ? opmath_t(1) : opmath_t(gamma_data[c]));
          bias_ptr[c] = -scale_ptr[c] * mean_val + (beta_null ? opmath_t(0) : opmath_t(beta_data[c]));
        }
      }
    }

    // step-4: 应用缩放和偏置
    //
    // 在N和HxW的所有外部维度上并行执行，并在C上进行向量化。
    //
    // 使用 ATen 库的 parallel_for 函数并行处理数组索引范围 [0, N * HxW)
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      // 初始化循环变量 n 和 m
      int64_t n{0}, m{0};
      // 调用函数 data_index_init 对 begin、n、N、m 和 HxW 进行初始化
      data_index_init(begin, n, N, m, HxW);
      // 对于从 begin 到 end 的每一个 i（包括 begin，但不包括 end）
      for (const auto i : c10::irange(begin, end)) {
        // 计算 X_ptr 的指针位置，基于 X_data 和通道数 C
        const T* X_ptr = X_data + i * C;
        // 计算 Y_ptr 的指针位置，基于 Y_data 和通道数 C
        T* Y_ptr = Y_data + i * C;
        // 计算 scale_ptr 的指针位置，基于 buffer_data、n 和 2 * C
        opmath_t* scale_ptr = buffer_data + n * 2 * C;
        // 计算 bias_ptr 的指针位置，基于 scale_ptr 和 C
        opmath_t* bias_ptr = scale_ptr + C;
        // 调用 ApplyScaleBias 函数，对 Y_ptr 应用 X_ptr 的数据，并使用 scale_ptr 和 bias_ptr 进行缩放和偏置处理，通道数为 C
        ApplyScaleBias<T, opmath_t>(Y_ptr, X_ptr, scale_ptr, bias_ptr, C);
        // 调用函数 data_index_step 更新 n 和 m 的值，基于 N 和 HxW
        data_index_step(n, N, m, HxW);
      }
    });
  }
}

void GroupNormKernelImpl(
    const Tensor& X,                    // 输入张量 X
    const Tensor& gamma,                // 缩放参数 gamma
    const Tensor& beta,                 // 偏移参数 beta
    int64_t N,                          // 批次大小
    int64_t C,                          // 通道数
    int64_t HxW,                        // 高度乘以宽度
    int64_t group,                      // 分组数
    double eps,                         // epsilon 值
    Tensor& Y,                          // 输出张量 Y
    Tensor& mean,                       // 均值张量 mean
    Tensor& rstd) {                     // 标准差倒数张量 rstd
  const bool mixed_type = is_mixed_type(X, gamma, beta);  // 检查是否为混合类型
  switch (X.suggest_memory_format()) {                    // 根据建议的内存格式进行分支
    case at::MemoryFormat::Contiguous: {                  // 若为连续内存格式
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;        // 使用类型 scalar_t 进行数学操作
        if (mixed_type) {                                 // 如果是混合类型
          GroupNormKernelImplInternal<scalar_t, param_t>(  // 调用混合类型的内部实现
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          GroupNormKernelImplInternal<scalar_t, scalar_t>( // 调用相同类型的内部实现
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast:                   // 若为 ChannelsLast 内存格式
    case at::MemoryFormat::ChannelsLast3d: {               // 或 ChannelsLast3d 内存格式
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;         // 使用类型 scalar_t 进行数学操作
        if (mixed_type) {                                  // 如果是混合类型
          GroupNormKernelImplChannelsLastInternal<scalar_t, param_t>( // 调用 ChannelsLast 混合类型的内部实现
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          GroupNormKernelImplChannelsLastInternal<scalar_t, scalar_t>( // 调用 ChannelsLast 相同类型的内部实现
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");  // 不支持的内存格式错误提示
  }
}


template <typename T, typename opmath_t>
typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* dY,
    const T* X,
    opmath_t* ds,
    opmath_t* db) {
  using Vec = at::vec::Vectorized<opmath_t>;    // 使用 opmath_t 类型的向量化 Vec
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* dY_ptr = dY + i * HxW;           // 指向 dY 的指针偏移
      const T* X_ptr = X + i * HxW;             // 指向 X 的指针偏移
      ds[i] = at::vec::map2_reduce_all<T>(      // 使用向量化操作计算 ds[i]
          [](Vec x, Vec y) { return x * y; },   // 映射和归约操作：x * y
          [](Vec x, Vec y) { return x + y; },   // 映射和归约操作：x + y
          dY_ptr,                               // 输入数据指针 dY_ptr
          X_ptr,                                // 输入数据指针 X_ptr
          HxW);                                 // 数据大小 HxW
      db[i] = at::vec::reduce_all<T>(           // 使用向量化操作计算 db[i]
          [](Vec& x, Vec& y) { return x + y; }, // 归约操作：x + y
          dY_ptr,                               // 输入数据指针 dY_ptr
          HxW);                                 // 数据大小 HxW
    }
  });
}

template <typename T, typename opmath_t>
typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* dY,
    const T* X,
    opmath_t* ds,
    opmath_t* db) {
  using Vec = vec::Vectorized<T>;               // 使用 T 类型的向量化 Vec
  using fVec = vec::Vectorized<opmath_t>;       // 使用 opmath_t 类型的向量化 fVec
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = Vec::size();         // 向量化大小 K
    const int64_t inner_size = HxW / K * K;     // 内部大小
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    // 创建一个固定大小为 K/2 的数组 ds_arr，用于存储 ds 的中间结果
    std::array<opmath_t, K / 2> ds_arr;
    // 创建一个固定大小为 K/2 的数组 db_arr，用于存储 db 的中间结果
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<opmath_t, K / 2> db_arr;
    // 遍历范围从 start 到 end 的索引 i
    for (const auto i : c10::irange(start, end)) {
      // 获取当前 dY 和 X 的指针，这些指针分别指向当前数据块的位置
      const T* dY_ptr = dY + i * HxW;
      const T* X_ptr = X + i * HxW;
      // 初始化 ds_vec 和 db_vec，用于累加 ds 和 db 的向量结果
      fVec ds_vec(0);
      fVec db_vec(0);
      // 对当前数据块中每个 K 大小的元素进行处理
      for (int64_t j = 0; j < inner_size; j += K) {
        // 从 dY_ptr 和 X_ptr 加载 K 大小的数据块到 SIMD 向量 dy_bvec 和 x_bvec
        const Vec dy_bvec = Vec::loadu(dY_ptr + j);
        const Vec x_bvec = Vec::loadu(X_ptr + j);
        // 将 SIMD 向量转换为浮点数向量，并分别存储在 x_fvec0, x_fvec1, dy_fvec0, dy_fvec1 中
        auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
        auto [dy_fvec0, dy_fvec1] = convert_to_float<T>(dy_bvec);
        // 计算 ds_vec 和 db_vec 的累加结果
        ds_vec = ds_vec + dy_fvec0 * x_fvec0;
        ds_vec = ds_vec + dy_fvec1 * x_fvec1;
        db_vec = db_vec + dy_fvec0 + dy_fvec1;
      }
      // 将 ds_vec 和 db_vec 的结果存储回 ds_arr 和 db_arr 中
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      // 对 ds_arr 和 db_arr 中的数据求和，得到 ds_val 和 db_val 的中间结果
      opmath_t ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), opmath_t(0));
      opmath_t db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), opmath_t(0));
      // 处理剩余部分，从 inner_size 到 HxW 的每个元素，累加到 ds_val 和 db_val 中
      for (const auto j : c10::irange(inner_size, HxW)) {
        ds_val += opmath_t(dY_ptr[j]) * opmath_t(X_ptr[j]);
        db_val += opmath_t(dY_ptr[j]);
      }
      // 将最终结果 ds_val 和 db_val 分别存储回 ds 和 db 的数组中
      ds[i] = ds_val;
      db[i] = db_val;
    }
  });


这段代码是一个并行计算的例子，使用了 SIMD（Single Instruction, Multiple Data）指令集优化了向量操作，对输入数据进行了并行处理并计算出结果存储在数组 ds 和 db 中。
}

template <typename PT, typename opmath_t>
inline typename std::enable_if<std::is_same<PT, opmath_t>::value, void>::type
CalcDsDb(
    const opmath_t* ds_ptr,
    const opmath_t* db_ptr,
    const PT* gamma_ptr,
    const int64_t d,
    const int64_t K,
    void* ds_arr,
    void* db_arr) {
    // 创建存储 ds 和 db 向量化结果的对象，初始值为0
    vec::Vectorized<opmath_t> ds_vec(0);
    vec::Vectorized<opmath_t> db_vec(0);
    // 循环处理每个向量
    for (int64_t j = 0; j < d; j += K) {
        // 如果 gamma_ptr 为 nullptr，则创建一个全1的向量
        const vec::Vectorized<PT> gamma_vec = (gamma_ptr == nullptr)
            ? vec::Vectorized<PT>(1)
            : vec::Vectorized<PT>::loadu(gamma_ptr + j);
        // 加载 ds_ptr 和 db_ptr 对应位置的向量，并乘以 gamma_vec
        ds_vec = ds_vec + vec::Vectorized<PT>::loadu(ds_ptr + j) * gamma_vec;
        db_vec = db_vec + vec::Vectorized<PT>::loadu(db_ptr + j) * gamma_vec;
    }
    // 将向量化计算结果存储到 ds_arr 和 db_arr 中
    ds_vec.store(ds_arr);
    db_vec.store(db_arr);
}

template <typename PT, typename opmath_t>
inline typename std::enable_if<!std::is_same<PT, opmath_t>::value, void>::type
CalcDsDb(
    const opmath_t* ds_ptr,
    const opmath_t* db_ptr,
    const PT* gamma_ptr,
    const int64_t d,
    const int64_t K,
    void* ds_arr,
    void* db_arr) {
  using fVec = at::vec::Vectorized<opmath_t>;
  using Vec = at::vec::Vectorized<PT>;
  // 创建存储 ds 和 db 向量化结果的对象，初始值为0
  fVec ds_acc(0);
  fVec db_acc(0);
  // 循环处理每个向量
  for (int64_t j = 0; j < d; j += K) {
    // 如果 gamma_ptr 为 nullptr，则创建一个全1的向量；否则加载对应位置的向量
    const Vec gamma_vec = (gamma_ptr == nullptr) ? Vec(1) : Vec::loadu(gamma_ptr + j);
    auto [gamma_vec0, gamma_vec1] = convert_to_float<PT>(gamma_vec);
    // 加载 ds_ptr 和 db_ptr 对应位置的向量，并乘以 gamma_vec 中的浮点数部分
    ds_acc += fVec::loadu(ds_ptr + j) * gamma_vec0;
    ds_acc += fVec::loadu(ds_ptr + j + fVec::size()) * gamma_vec1;
    db_acc += fVec::loadu(db_ptr + j) * gamma_vec0;
    db_acc += fVec::loadu(db_ptr + j + fVec::size()) * gamma_vec1;
  }
  // 将向量化计算结果存储到 ds_arr 和 db_arr 中
  ds_acc.store(ds_arr);
  db_acc.store(db_arr);
}

template <typename T, typename PT, typename opmath_t>
void GroupNormInputBackward(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* dY,
    const T* X,
    const PT* mean,
    const PT* rstd,
    const PT* gamma,
    const opmath_t* ds,
    const opmath_t* db,
    T* dX) {
  const int64_t G = group;
  const int64_t D = C / G;
  const opmath_t s = opmath_t(1) / static_cast<opmath_t>(D * HxW);
  const bool gamma_null = (gamma == nullptr);
  // 并行处理每个组的数据
  at::parallel_for(0, N * G, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = vec::Vectorized<PT>::size();
    const int64_t d = D / K * K;
    // 创建存储 ds 和 db 向量化结果的数组
    std::array<opmath_t, at::vec::Vectorized<opmath_t>::size()> ds_arr;
    std::array<opmath_t, at::vec::Vectorized<opmath_t>::size()> db_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    // 循环处理每个数据组
    for (int64_t i = start; i < end; ++i) {
        // 处理每个组的 ds 和 db
    // 对于范围 [start, end) 中的每个索引 i 进行循环迭代
    for (const auto i : c10::irange(start, end)) {
      // 计算 i 对 G 取模的结果
      const int64_t g = i % G;
      // 获取 ds 数组中第 i 行的起始指针
      const opmath_t* ds_ptr = ds + i * D;
      // 获取 db 数组中第 i 行的起始指针
      const opmath_t* db_ptr = db + i * D;
      // 如果 gamma_null 为真，则 gamma_ptr 为 nullptr；否则为 gamma 数组中第 g 行的起始指针
      const PT* gamma_ptr = gamma_null ? nullptr : (gamma + g * D);
      // 调用 CalcDsDb 函数计算 ds_arr 和 db_arr 数组的值
      CalcDsDb(ds_ptr, db_ptr, gamma_ptr, d, K, ds_arr.data(), db_arr.data());
      // 计算 ds_arr 数组的累加和
      opmath_t ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), opmath_t(0));
      // 计算 db_arr 数组的累加和
      opmath_t db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), opmath_t(0));
      // 对于范围 [d, D) 中的每个索引 j 进行循环迭代
      for (const auto j : c10::irange(d, D)) {
        // 如果 gamma_null 为真，则 gamma_v 为 1；否则为 gamma 数组中的值
        const opmath_t gamma_v = gamma_null ? opmath_t(1) : opmath_t(gamma[g * D + j]);
        // 计算 ds_val 和 db_val 的更新值
        ds_val += ds_ptr[j] * gamma_v;
        db_val += db_ptr[j] * gamma_v;
      }
      // 计算 c2 的值，这里是一个复杂的表达式
      const opmath_t c2 =
          (db_val * opmath_t(mean[i]) - ds_val) * opmath_t(rstd[i]) * opmath_t(rstd[i]) * opmath_t(rstd[i]) * s;
      // 计算 c3 的值，这里也是一个复杂的表达式
      const opmath_t c3 = -c2 * opmath_t(mean[i]) - db_val * opmath_t(rstd[i]) * s;

      // 对于范围 [0, D) 中的每个索引 j 进行循环迭代
      for (const auto j : c10::irange(D)) {
        // 计算 gamma 数组中的索引值
        const int64_t c = g * D + j;
        // 获取 dY 和 X 中第 (i, j) 元素的指针
        const T* dY_ptr = dY + (i * D + j) * HxW;
        const T* X_ptr = X + (i * D + j) * HxW;
        // 获取 dX 中第 (i, j) 元素的指针
        T* dX_ptr = dX + (i * D + j) * HxW;
        // 计算 c1 的值，这里是一个复杂的表达式
        const opmath_t c1 = opmath_t(rstd[i]) * (gamma_null ? opmath_t(1) : opmath_t(gamma[c]));
        // 对于范围 [0, HxW) 中的每个索引 k 进行循环迭代
        for (const auto k : c10::irange(HxW)) {
          // 计算 dX_ptr[k] 的更新值
          dX_ptr[k] = c1 * opmath_t(dY_ptr[k]) + c2 * opmath_t(X_ptr[k]) + c3;
        }
      }
    }
}

template <typename PT, typename opmath_t>
typename std::enable_if<std::is_same<PT, opmath_t>::value, void>::type
GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const PT* mean,
    const PT* rstd,
    const opmath_t* ds,
    const opmath_t* db,
    PT* dgamma) {
  // 定义组数 G，并计算每组的维度 D
  const int64_t G = group;
  const int64_t D = C / G;
  // 计算模板类型 PT 的向量化大小 K
  constexpr int64_t K = at::vec::Vectorized<PT>::size();
  // 使用 Vec 类型表示 PT 的向量化操作
  using Vec = at::vec::Vectorized<PT>;
  // 计算每个向量的内部大小
  const int64_t inner_size = D / K * K;
  // 遍历每个组 g
  for (const auto g : c10::irange(G)) {
    int64_t i = 0;
    // 对于每个组内的内部大小，以向量化大小 K 为步长遍历
    for (; i < inner_size; i += K) {
      Vec acc_vec{0};
      // 遍历每个样本 n
      for (const auto n : c10::irange(N)) {
        // 计算 ds 和 db 的指针位置
        const PT* ds_ptr = ds + n * C + g * D + i;
        const PT* db_ptr = db + n * C + g * D + i;
        // 加载 ds 和 db 的向量数据
        auto ds_vec = Vec::loadu(ds_ptr);
        auto db_vec = Vec::loadu(db_ptr);
        // 加载 mean 和 rstd 的向量数据
        auto mean_vec = Vec(mean[n * G + g]);
        auto rstd_vec = Vec(rstd[n * G + g]);
        // 计算并累加梯度
        acc_vec += (ds_vec - db_vec * mean_vec) * rstd_vec;
      }
      // 存储计算得到的梯度结果
      acc_vec.store(dgamma + g * D + i);
    }
    // 处理剩余部分，不足一个向量化大小 K 的情况
    if (D - i > 0) {
      Vec acc_vec{0};
      // 遍历每个样本 n
      for (const auto n : c10::irange(N)) {
        // 计算 ds 和 db 的指针位置
        const PT* ds_ptr = ds + n * C + g * D + i;
        const PT* db_ptr = db + n * C + g * D + i;
        // 加载 ds 和 db 的部分向量数据
        auto ds_vec = Vec::loadu(ds_ptr, D - i);
        auto db_vec = Vec::loadu(db_ptr, D - i);
        // 加载 mean 和 rstd 的向量数据
        auto mean_vec = Vec(mean[n * G + g]);
        auto rstd_vec = Vec(rstd[n * G + g]);
        // 计算并累加梯度
        acc_vec += (ds_vec - db_vec * mean_vec) * rstd_vec;
      }
      // 存储计算得到的梯度结果
      acc_vec.store(dgamma + g * D + i, D - i);
    }
  }
}

template <typename PT, typename opmath_t>
typename std::enable_if<!std::is_same<PT, opmath_t>::value, void>::type
GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const PT* mean,
    const PT* rstd,
    const opmath_t* ds,
    const opmath_t* db,
    PT* dgamma) {
  // 定义组数 G，并计算每组的维度 D
  const int64_t G = group;
  const int64_t D = C / G;
  // 使用 Vec 类型表示 PT 的向量化操作
  using Vec = at::vec::Vectorized<PT>;
  // 使用 fVec 类型表示 opmath_t 的向量化操作
  using fVec = at::vec::Vectorized<opmath_t>;
  // 计算模板类型 PT 的向量化大小 K
  constexpr int64_t K = Vec::size();
  // 计算每个向量的内部大小
  const int64_t inner_size = D / K * K;
  // 遍历每个组 g
  for (const auto g : c10::irange(G)) {
    int64_t i = 0;
    // 对于每个组内的内部大小，以向量化大小 K 为步长遍历
    for (; i < inner_size; i += K) {
      fVec acc0_vec{0}, acc1_vec{0};
      // 遍历每个样本 n
      for (const auto n : c10::irange(N)) {
        // 计算 ds 和 db 的指针位置
        const opmath_t* ds_ptr = ds + n * C + g * D + i;
        const opmath_t* db_ptr = db + n * C + g * D + i;
        // 加载 ds 和 db 的向量数据
        fVec ds_vec0, ds_vec1, db_vec0, db_vec1;
        ds_vec0 = fVec::loadu(ds_ptr);
        ds_vec1 = fVec::loadu(ds_ptr + fVec::size());
        db_vec0 = fVec::loadu(db_ptr);
        db_vec1 = fVec::loadu(db_ptr + fVec::size());
        // 加载 mean 和 rstd 的向量数据
        fVec mean_vec = fVec(opmath_t(mean[n * G + g]));
        fVec rstd_vec = fVec(opmath_t(rstd[n * G + g]));
        // 计算并累加梯度
        acc0_vec += (ds_vec0 - db_vec0 * mean_vec) * rstd_vec;
        acc1_vec += (ds_vec1 - db_vec1 * mean_vec) * rstd_vec;
      }
      // 将浮点数转换为 PT 类型，并存储计算得到的梯度结果
      convert_from_float<PT>(acc0_vec, acc1_vec).store(dgamma + g * D + i);
    }
    # 检查 D - i 是否大于 0，进入条件语句
    if (D - i > 0) {
      # 初始化两个向量，用于累加计算
      fVec acc0_vec{0}, acc1_vec{0};
      # 对于范围内的每个 n 进行循环迭代
      for (const auto n : c10::irange(N)) {
        # 计算 ds_ptr 和 db_ptr 指针的位置
        const opmath_t* ds_ptr = ds + n * C + g * D + i;
        const opmath_t* db_ptr = db + n * C + g * D + i;
        # 初始化四个向量，用于加载数据
        fVec ds_vec0, ds_vec1, db_vec0, db_vec1;
        # 加载 ds_ptr 的数据到 ds_vec0 和 ds_vec1
        ds_vec0 = fVec::loadu(
            ds_ptr, (D - i) > fVec::size() ? fVec::size() : (D - i));
        ds_vec1 = fVec::loadu(
            ds_ptr + fVec::size(),
            (D - i) > fVec::size() ? (D - i - fVec::size()) : 0);
        # 加载 db_ptr 的数据到 db_vec0 和 db_vec1
        db_vec0 = fVec::loadu(
            db_ptr, (D - i) > fVec::size() ? fVec::size() : (D - i));
        db_vec1 = fVec::loadu(
            db_ptr + fVec::size(),
            (D - i) > fVec::size() ? (D - i - fVec::size()) : 0);
        # 加载均值和标准差倒数到向量
        fVec mean_vec = fVec(opmath_t(mean[n * G + g]));
        fVec rstd_vec = fVec(opmath_t(rstd[n * G + g]));
        # 计算累加向量 acc0_vec 和 acc1_vec
        acc0_vec += (ds_vec0 - db_vec0 * mean_vec) * rstd_vec;
        acc1_vec += (ds_vec1 - db_vec1 * mean_vec) * rstd_vec;
      }
      # 将累加结果转换为指定精度 PT 并存储到 dgamma 中
      convert_from_float<PT>(acc0_vec, acc1_vec).store(dgamma + g * D + i, D - i);
    }
}

template <typename PT, typename opmath_t>
typename std::enable_if<std::is_same<PT, opmath_t>::value, void>::type
BetaBackward(int64_t N, int64_t C, const opmath_t* db, PT* dbeta) {
  // 使用模板参数 PT 和 opmath_t，若二者相同，则执行该函数
  using Vec = at::vec::Vectorized<PT>;
  // 定义模板类型 Vec，用于载入和存储 PT 类型的向量化数据
  constexpr int64_t K = Vec::size();
  // 定义常量 K，表示向量化操作的长度
  Vec acc_vec{0}, zero{0};
  // 初始化 Vec 类型的累加向量 acc_vec 和零向量 zero
  const int64_t inner_size = C / K * K;
  // 计算最内层循环的长度 inner_size，确保对齐到 K 的倍数
  int64_t i = 0;
  // 初始化循环变量 i
  for (; i < inner_size; i += K) {
    // 外层循环，每次增加 K，以处理向量化累加操作
    for (const auto n : c10::irange(N)) {
      // 内层循环，遍历数据批次 N
      acc_vec += Vec::loadu(db + n * C + i);
      // 将 db 中偏移量为 n * C + i 的数据加载到 acc_vec 中进行累加
    }
    acc_vec.store(dbeta + i);
    // 将累加的结果存储到 dbeta 中的偏移量 i 处
    acc_vec = Vec::set(acc_vec, zero);
    // 将 acc_vec 重置为全零向量
  }
  if (C - i > 0) {
    // 处理剩余部分，若 C - i 大于零
    for (const auto n : c10::irange(N)) {
      // 内层循环，遍历数据批次 N
      acc_vec += Vec::loadu(db + n * C + i, C - i);
      // 加载剩余部分数据到 acc_vec 进行累加
    }
    acc_vec.store(dbeta + i, C - i);
    // 将累加的结果存储到 dbeta 中的偏移量 i 处，长度为 C - i
    acc_vec = Vec::set(acc_vec, zero, C - i);
    // 将 acc_vec 重置为全零向量，长度为 C - i
  }
}

template <typename PT, typename opmath_t>
typename std::enable_if<!std::is_same<PT, opmath_t>::value, void>::type
BetaBackward(int64_t N, int64_t C, const opmath_t* db, PT* dbeta) {
  // 使用模板参数 PT 和 opmath_t，若二者不同，则执行该函数
  using Vec = at::vec::Vectorized<PT>;
  // 定义模板类型 Vec，用于载入和存储 PT 类型的向量化数据
  using fVec = at::vec::Vectorized<opmath_t>;
  // 定义模板类型 fVec，用于载入和存储 opmath_t 类型的向量化数据
  constexpr int64_t K = Vec::size();
  // 定义常量 K，表示向量化操作的长度
  fVec acc0_vec{0}, acc1_vec{0}, zero{0};
  // 初始化 fVec 类型的累加向量 acc0_vec 和 acc1_vec，以及零向量 zero
  const int64_t inner_size = C / K * K;
  // 计算最内层循环的长度 inner_size，确保对齐到 K 的倍数
  int64_t i = 0;
  // 初始化循环变量 i
  for (; i < inner_size; i += K) {
    // 外层循环，每次增加 K，以处理向量化累加操作
    for (const auto n : c10::irange(N)) {
      // 内层循环，遍历数据批次 N
      fVec db_vec0, db_vec1;
      // 声明 fVec 类型的变量 db_vec0 和 db_vec1
      db_vec0 = fVec::loadu(db + n * C + i);
      // 加载 db 中偏移量为 n * C + i 的数据到 db_vec0
      db_vec1 = fVec::loadu(db + n * C + i + fVec::size());
      // 加载 db 中偏移量为 n * C + i + fVec::size() 的数据到 db_vec1
      acc0_vec += db_vec0;
      acc1_vec += db_vec1;
      // 分别累加 db_vec0 和 db_vec1 到 acc0_vec 和 acc1_vec
    }
    convert_from_float<PT>(acc0_vec, acc1_vec).store(dbeta + i);
    // 将累加的结果转换为 PT 类型，并存储到 dbeta 中的偏移量 i 处
    acc0_vec = fVec::set(acc0_vec, zero);
    // 将 acc0_vec 重置为全零向量
    acc1_vec = fVec::set(acc1_vec, zero);
    // 将 acc1_vec 重置为全零向量
  }
  if (C - i > 0) {
    // 处理剩余部分，若 C - i 大于零
    for (const auto n : c10::irange(N)) {
      // 内层循环，遍历数据批次 N
      fVec db_vec0, db_vec1;
      // 声明 fVec 类型的变量 db_vec0 和 db_vec1
      db_vec0 = fVec::loadu(
          db + n * C + i, (C - i) > fVec::size() ? fVec::size() : (C - i));
      // 加载 db 中偏移量为 n * C + i 的数据，最多加载 fVec::size() 长度
      db_vec1 = fVec::loadu(
          db + n * C + i + fVec::size(),
          (C - i) > fVec::size() ? (C - i - fVec::size()) : 0);
      // 加载 db 中偏移量为 n * C + i + fVec::size() 的数据，最多加载剩余长度
      acc0_vec += db_vec0;
      acc1_vec += db_vec1;
      // 分别累加 db_vec0 和 db_vec1 到 acc0_vec 和 acc1_vec
    }
    convert_from_float<PT>(acc0_vec, acc1_vec).store(dbeta + i, C - i);
    // 将累加的结果转换为 PT 类型，并存储到 dbeta 中的偏移量 i 处，长度为 C - i
    acc0_vec = fVec::set(acc0_vec, zero, C - i);
    // 将 acc0_vec 重置为全零向量，长度为 C - i
    acc1_vec = fVec::set(acc1_vec, zero, C - i);
    // 将 acc1_vec 重置为全零向量，长度为 C - i
  }
}

template <typename T, typename PT>
void GroupNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    // 检查输入张量的元素数量是否正确
    TORCH_CHECK(dY.numel() == N * C * HxW);
    // 检查输入张量的元素数量是否正确
    TORCH_CHECK(X.numel() == N * C * HxW);
    // 检查均值张量的元素数量是否正确
    TORCH_CHECK(mean.numel() == N * group);
    // 检查归一化标准差张量的元素数量是否正确
    TORCH_CHECK(rstd.numel() == N * group);
    // 检查 gamma 张量是否已定义，如果定义了，检查其元素数量是否正确
    TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
    
    // 获取 dY 张量的常量数据指针
    const T* dY_data = dY.const_data_ptr<T>();
    // 获取 X 张量的常量数据指针
    const T* X_data = X.const_data_ptr<T>();
    // 获取 mean 张量的常量数据指针
    const PT* mean_data = mean.const_data_ptr<PT>();
    // 获取 rstd 张量的常量数据指针
    const PT* rstd_data = rstd.const_data_ptr<PT>();
    // 获取 gamma 张量的常量数据指针，如果未定义则置为 nullptr
    const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;
    
    // 获取 dX 张量的数据指针，如果未定义则置为 nullptr
    T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
    // 获取 dgamma 张量的数据指针，如果未定义则置为 nullptr
    PT* dgamma_data = dgamma.defined() ? dgamma.data_ptr<PT>() : nullptr;
    // 获取 dbeta 张量的数据指针，如果未定义则置为 nullptr
    PT* dbeta_data = dbeta.defined() ? dbeta.data_ptr<PT>() : nullptr;
    
    // 定义 opmath_t 类型为 at::opmath_type<T>
    using opmath_t = at::opmath_type<T>;
    // 创建一个空张量 ds，形状为 {N, C}，使用与 X 相同的选项和数据类型
    Tensor ds = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    // 创建一个空张量 db，形状为 {N, C}，使用与 X 相同的选项和数据类型
    Tensor db = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    
    // 获取 ds 张量的 opmath_t 类型数据指针
    opmath_t* ds_data = ds.data_ptr<opmath_t>();
    // 获取 db 张量的 opmath_t 类型数据指针
    opmath_t* db_data = db.data_ptr<opmath_t>();
    
    // 调用 ComputeInternalGradients 函数计算内部梯度
    ComputeInternalGradients<T, opmath_t>(N, C, HxW, dY_data, X_data, ds_data, db_data);
    
    // 如果 dX_data 不为 nullptr，则调用 GroupNormInputBackward 函数计算输入梯度
    if (dX_data != nullptr) {
        GroupNormInputBackward<T, PT, opmath_t>(
            N,
            C,
            HxW,
            group,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            gamma_data,
            ds_data,
            db_data,
            dX_data);
    }
    
    // 如果 dgamma_data 不为 nullptr，则调用 GammaBackward 函数计算 gamma 的梯度
    if (dgamma_data != nullptr) {
        GammaBackward(
            N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
    }
    
    // 如果 dbeta_data 不为 nullptr，则调用 BetaBackward 函数计算 beta 的梯度
    if (dbeta_data != nullptr) {
        BetaBackward(N, C, db_data, dbeta_data);
    }
// 当前函数用于计算在Channels Last格式下，每行数据的均值和方差的增量更新
template <typename T, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
DsDbRowwiseMomentsChannelsLast(
  const T* dY_ptr,           // 输入参数：指向梯度数据的指针
  const T* X_ptr,            // 输入参数：指向输入数据的指针
  opmath_t* ds_ptr,          // 输出参数：指向均值增量更新的指针
  opmath_t* db_ptr,          // 输出参数：指向方差增量更新的指针
  int64_t C) {               // 输入参数：通道数

  using Vec = vec::Vectorized<T>;  // 使用Vectorized<T>定义Vec类型
  constexpr int64_t K = vec::Vectorized<T>::size();  // 向量化操作的向量大小K
  const int64_t inner_size = C / K * K;  // 内部循环处理的数据量，确保是K的整数倍
  int64_t d = 0;  // 循环索引

  // 主循环，处理能够以Vec大小K进行向量化的部分
  for (; d < inner_size; d += K) {
    Vec ds_dev = Vec::loadu(ds_ptr + d);  // 加载ds_ptr中偏移d处的数据到ds_dev向量
    Vec db_vec = Vec::loadu(db_ptr + d);  // 加载db_ptr中偏移d处的数据到db_vec向量
    Vec x_vec = Vec::loadu(X_ptr + d);    // 加载X_ptr中偏移d处的数据到x_vec向量
    Vec dy_vec = Vec::loadu(dY_ptr + d);  // 加载dY_ptr中偏移d处的数据到dy_vec向量

    // 计算均值的增量更新
    ds_dev += x_vec * dy_vec;
    // 计算方差的增量更新
    db_vec += dy_vec;

    // 将更新后的均值数据存储回ds_ptr中
    ds_dev.store(ds_ptr + d);
    // 将更新后的方差数据存储回db_ptr中
    db_vec.store(db_ptr + d);
  }

  // 处理剩余的不足以向量化的部分
  if (C - d > 0) {
    Vec ds_dev = Vec::loadu(ds_ptr + d, C - d);  // 加载ds_ptr中剩余部分的数据到ds_dev向量
    Vec db_vec = Vec::loadu(db_ptr + d, C - d);  // 加载db_ptr中剩余部分的数据到db_vec向量
    Vec x_vec = Vec::loadu(X_ptr + d, C - d);    // 加载X_ptr中剩余部分的数据到x_vec向量
    Vec dy_vec = Vec::loadu(dY_ptr + d, C - d);  // 加载dY_ptr中剩余部分的数据到dy_vec向量

    // 计算均值的增量更新
    ds_dev += x_vec * dy_vec;
    // 计算方差的增量更新
    db_vec += dy_vec;

    // 将更新后的均值数据存储回ds_ptr中
    ds_dev.store(ds_ptr + d, C - d);
    // 将更新后的方差数据存储回db_ptr中
    db_vec.store(db_ptr + d, C - d);
  }
}

// 当前函数用于处理不同数据类型T和opmath_t的情况下的Channels Last格式的均值和方差增量更新
template <typename T, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
DsDbRowwiseMomentsChannelsLast(
  const T* dY_ptr,           // 输入参数：指向梯度数据的指针
  const T* X_ptr,            // 输入参数：指向输入数据的指针
  opmath_t* ds_ptr,          // 输出参数：指向均值增量更新的指针
  opmath_t* db_ptr,          // 输出参数：指向方差增量更新的指针
  int64_t C) {               // 输入参数：通道数

  using fVec = vec::Vectorized<opmath_t>;  // 使用Vectorized<opmath_t>定义fVec类型
  using Vec = vec::Vectorized<T>;          // 使用Vectorized<T>定义Vec类型
  int64_t d = 0;  // 循环索引

  // 主循环，处理能够以Vec大小进行向量化的部分
  for (; d < C - (C % Vec::size()); d += Vec::size()) {
    fVec ds_dev0 = fVec::loadu(ds_ptr + d);                  // 加载ds_ptr中偏移d处的数据到ds_dev0向量
    fVec ds_dev1 = fVec::loadu(ds_ptr + d + fVec::size());   // 加载ds_ptr中偏移d+fVec::size()处的数据到ds_dev1向量
    fVec db_vec0 = fVec::loadu(db_ptr + d);                  // 加载db_ptr中偏移d处的数据到db_vec0向量
    fVec db_vec1 = fVec::loadu(db_ptr + d + fVec::size());   // 加载db_ptr中偏移d+fVec::size()处的数据到db_vec1向量
    Vec x_vec = Vec::loadu(X_ptr + d);                       // 加载X_ptr中偏移d处的数据到x_vec向量
    Vec dy_vec = Vec::loadu(dY_ptr + d);                     // 加载dY_ptr中偏移d处的数据到dy_vec向量

    // 转换T类型向量到opmath_t类型向量，用于后续计算
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);

    // 计算均值的增量更新
    ds_dev0 += x_vec0 * dy_vec0;
    ds_dev1 += x_vec1 * dy_vec1;
    // 计算方差的增量更新
    db_vec0 += dy_vec0;
    db_vec1 += dy_vec1;

    // 将更新后的均值数据存储回ds_ptr中
    ds_dev0.store(ds_ptr + d);
    ds_dev1.store(ds_ptr + d + fVec::size());
    // 将更新后的方差数据存储回db_ptr中
    db_vec0.store(db_ptr + d);
    db_vec1.store(db_ptr + d + fVec::size());
  }

  // 处理剩余的不足以向量化的部分
  if (C - d > 0) {
    fVec ds_dev0 = fVec::loadu(ds_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));  // 加载ds_ptr中剩余部分的数据到ds_dev0向量
    fVec ds_dev1 = fVec::loadu(ds_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);  // 加载ds_ptr中剩余部分的数据到ds_dev1向量
    fVec db_vec0 = fVec::loadu(db_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));  // 加载db_ptr中剩余部分的数据到db_vec0向量
    fVec db_vec1 = fVec::loadu(db_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);  // 加载db_ptr中剩余部分的数据到db_vec1向量
    Vec x_vec = Vec::loadu(X_ptr + d, C - d);    // 加载X_ptr中剩余部分的数据到x_vec向量
    Vec dy_vec = Vec::loadu(dY_ptr + d, C - d);  // 加载dY_ptr中剩余部分的数据到dy_vec向量

    // 转换T类型向量到opmath_t类型向量，用于后续计算
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
    auto
    // 在 ds_dev1 中存储数据，从 ds_ptr + d + fVec::size() 处开始存储，
    // 存储长度为 (C - d) 和 fVec::size() 中较小的值，确保不超过可存储的最大长度
    ds_dev1.store(ds_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    // 在 db_vec0 中存储数据，从 db_ptr + d 处开始存储，
    // 存储长度为 (C - d) 和 fVec::size() 中较小的值，确保不超过可存储的最大长度
    db_vec0.store(db_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    // 在 db_vec1 中存储数据，从 db_ptr + d + fVec::size() 处开始存储，
    // 存储长度为 (C - d) 和 fVec::size() 中较小的值，确保不超过可存储的最大长度
    db_vec1.store(db_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
}
template <typename T>
// 如果 T 和 at::opmath_type<T> 相同，则返回一个包含两个 Vectorized<T> 对象的元组
inline typename std::enable_if<std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<
    vec::Vectorized<T>,
    vec::Vectorized<T>>>::type
load_util(const T* data_ptr, int64_t n) {
  using Vec = vec::Vectorized<T>;
  // 加载 data_ptr 指向的数据到 vec0，最多加载 Vec::size() 个元素
  auto vec0 = Vec::loadu(data_ptr, n > Vec::size() ? Vec::size() : n);
  // 加载 data_ptr + Vec::size() 指向的数据到 vec1，最多加载剩余的元素
  auto vec1 = Vec::loadu(
      data_ptr + Vec::size(), n > Vec::size() ? (n - Vec::size()) : 0);
  // 返回两个 Vectorized<T> 对象的元组
  return std::tuple<Vec, Vec>(vec0, vec1);
}

template <typename T>
// 如果 T 和 at::opmath_type<T> 不相同，则返回一个包含两个 Vectorized<at::opmath_type<T>> 对象的元组
inline typename std::enable_if<!std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<
    vec::Vectorized<at::opmath_type<T>>,
    vec::Vectorized<at::opmath_type<T>>>
    >::type
load_util(const T* data_ptr, int64_t n) {
  using Vec = vec::Vectorized<T>;
  // 加载 data_ptr 指向的数据到 vec，加载 n 个元素
  auto vec = Vec::loadu(data_ptr, n);
  // 转换 vec 到 float 类型并返回
  return convert_to_float<T>(vec);
}

template <typename T, typename PT, typename opmath_t>
// 如果 T 和 opmath_t 相同，则应用输入梯度到 channels last col major order
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastColMov(
  const T* dY_data,
  const T* X_data,
  T* dX_data,
  const PT* rstd,
  const PT* gamma,
  opmath_t c2,
  opmath_t c3,
  int64_t HxW,
  int64_t C,
  int64_t D) {
  const bool gamma_null = (gamma == nullptr);
  int64_t d = 0;
  auto K = vec::Vectorized<T>::size();
  // 对每个 K 元素循环，处理 D/K 个组
  for (; d < D / K * K; d += K) {
    auto c1 = vec::Vectorized<T>(*rstd) *
        // 如果 gamma 不为空，则加载 gamma + d 处的值作为 Vectorized<T> 对象
        (gamma_null ? vec::Vectorized<T>(1)
                    : vec::Vectorized<T>::loadu(gamma + d));
    // 对每个 HxW 的区域循环
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      T* dX_ptr = dX_data + m * C;
      // 加载 dY_ptr + d 处的数据到 dy_vec
      auto dy_vec = vec::Vectorized<T>::loadu(dY_ptr + d);
      // 加载 X_ptr + d 处的数据到 x_vec
      auto x_vec = vec::Vectorized<T>::loadu(X_ptr + d);
      // 计算 dx_vec 并存储到 dX_ptr + d 处
      auto dx_vec = c1 * dy_vec +
        vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
      dx_vec.store(dX_ptr + d);
    }
  }
  // 处理剩余的 D-d 个元素
  if (D - d > 0) {
    auto c1 = vec::Vectorized<T>(*rstd) *
        // 如果 gamma 不为空，则加载 gamma + d 处的值作为 Vectorized<T> 对象，加载 D-d 个元素
        (gamma_null ? vec::Vectorized<T>(1)
                    : vec::Vectorized<T>::loadu(gamma + d, D - d));
    // 对每个 HxW 的区域循环
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      T* dX_ptr = dX_data + m * C;
      // 加载 dY_ptr + d, D-d 处的数据到 dy_vec
      auto dy_vec = vec::Vectorized<T>::loadu(dY_ptr + d, D - d);
      // 加载 X_ptr + d, D-d 处的数据到 x_vec
      auto x_vec = vec::Vectorized<T>::loadu(X_ptr + d, D - d);
      // 计算 dx_vec 并存储到 dX_ptr + d, D-d 处
      auto dx_vec = c1 * dy_vec +
        vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
      dx_vec.store(dX_ptr + d, D - d);
    }
  }
}

template <typename T, typename PT, typename opmath_t>
// 如果 T 和 opmath_t 不相同，则应用输入梯度到 channels last col major order
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastColMov(
    const T* dY_data,
    const T* X_data,
    T* dX_data,
    const PT* rstd,
    const PT* gamma,
    opmath_t c2,
    opmath_t c3,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<opmath_t>;
  const bool gamma_null = (gamma == nullptr);
  auto K = Vec::size();
  int64_t d = 0;
  // 对每个 K 元素循环，处理 D/K 个组
  for (; d < D / K * K; d += K) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))
                                      : load_util(gamma + d, K);
    // 如果 gamma_null 为 true，则 c1_0 和 c1_1 初始化为 fVec(1)，否则调用 load_util 函数加载 gamma + d 到 K 个元素
    c1_0 = c1_0 * fVec(opmath_t(*rstd));
    // 将 c1_0 乘以标量值 *rstd 转换为 fVec 类型
    c1_1 = c1_1 * fVec(opmath_t(*rstd));
    // 将 c1_1 乘以标量值 *rstd 转换为 fVec 类型
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      // 获取输入数据 X 中第 m 组的指针，偏移量为 m * C
      const T* dY_ptr = dY_data + m * C;
      // 获取梯度数据 dY 中第 m 组的指针，偏移量为 m * C
      T* dX_ptr = dX_data + m * C;
      // 获取输出梯度数据 dX 中第 m 组的指针，偏移量为 m * C

      Vec dy_vec = Vec::loadu(dY_ptr + d);
      // 从 dY_ptr + d 处加载向量 dy_vec，长度为 Vec 的长度
      Vec x_vec = Vec::loadu(X_ptr + d);
      // 从 X_ptr + d 处加载向量 x_vec，长度为 Vec 的长度
      auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
      // 将 x_vec 转换为浮点类型 x_vec0 和 x_vec1
      auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
      // 将 dy_vec 转换为浮点类型 dy_vec0 和 dy_vec1
      fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      // 计算第一个输出梯度向量 dx_vec0
      fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      // 计算第二个输出梯度向量 dx_vec1
      convert_from_float<T>(dx_vec0, dx_vec1).store(dX_ptr + d);
      // 将浮点类型的 dx_vec0 和 dx_vec1 转换为 T 类型，并存储到 dX_ptr + d 处
    }
  }
  if (D - d > 0) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))
                                      : load_util(gamma + d, D - d);
    // 如果 gamma_null 为 true，则 c1_0 和 c1_1 初始化为 fVec(1)，否则调用 load_util 函数加载 gamma + d 到 D - d 个元素
    c1_0 = c1_0 * fVec(opmath_t(*rstd));
    // 将 c1_0 乘以标量值 *rstd 转换为 fVec 类型
    c1_1 = c1_1 * fVec(opmath_t(*rstd));
    // 将 c1_1 乘以标量值 *rstd 转换为 fVec 类型
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      // 获取输入数据 X 中第 m 组的指针，偏移量为 m * C
      const T* dY_ptr = dY_data + m * C;
      // 获取梯度数据 dY 中第 m 组的指针，偏移量为 m * C
      T* dX_ptr = dX_data + m * C;
      // 获取输出梯度数据 dX 中第 m 组的指针，偏移量为 m * C
      Vec dy_vec = Vec::loadu(dY_ptr + d, D - d);
      // 从 dY_ptr + d 处加载向量 dy_vec，长度为 D - d
      Vec x_vec = Vec::loadu(X_ptr + d, D - d);
      // 从 X_ptr + d 处加载向量 x_vec，长度为 D - d
      auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
      // 将 x_vec 转换为浮点类型 x_vec0 和 x_vec1
      auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
      // 将 dy_vec 转换为浮点类型 dy_vec0 和 dy_vec1
      fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      // 计算第一个输出梯度向量 dx_vec0
      fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      // 计算第二个输出梯度向量 dx_vec1
      convert_from_float<T>(dx_vec0, dx_vec1).store(dX_ptr + d, D - d);
      // 将浮点类型的 dx_vec0 和 dx_vec1 转换为 T 类型，并存储到 dX_ptr + d 处，长度为 D - d
    }
  }
// 模板函数：应用输入梯度（通道为最后维度，按行移动）
template <typename T, typename PT, typename opmath_t>
// 当 T 与 opmath_t 类型相同时，启用此函数，返回值类型为 void
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastRowMov(
  const T* dY_data,          // 输入梯度 dY 的数据指针
  const T* X_data,           // 输入数据 X 的数据指针
  T* dX_data,                // 输出梯度 dX 的数据指针
  const PT* rstd,            // 标准差的逆值数据指针
  const PT* gamma,           // gamma 参数数据指针，如果为 nullptr 则 gamma_null 为 true
  opmath_t c2,               // 常数 c2，用于计算 dX
  opmath_t c3,               // 常数 c3，用于计算 dX
  int64_t HxW,               // 输入数据的 H * W 维度大小（未使用）
  int64_t C,                 // 输入数据的通道数（未使用）
  int64_t D) {               // 输入数据的总大小

  const bool gamma_null = (gamma == nullptr); // 判断 gamma 是否为 nullptr

  int64_t d = 0;  // 循环计数器
  auto K = vec::Vectorized<T>::size(); // 向量化操作的向量大小

  // 对每个 K 大小的向量进行循环处理
  for (; d < D / K * K; d += K) {
    auto c1 = vec::Vectorized<T>(*rstd) *  // 计算 c1，使用 rstd 初始化
      (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d)); // 根据 gamma_null 加载 gamma 数据
    auto dy_vec = vec::Vectorized<T>::loadu(dY_data + d);  // 加载 dY 数据向量
    auto x_vec = vec::Vectorized<T>::loadu(X_data + d);    // 加载 X 数据向量
    auto dx_vec = c1 * dy_vec +                        // 计算 dx 向量
      vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);  
    dx_vec.store(dX_data + d);  // 将 dx 向量存储到 dX 中
  }

  // 处理剩余不足一个完整向量大小的部分
  if (D - d > 0) {
    auto c1 = vec::Vectorized<T>(*rstd) *  // 计算 c1，使用 rstd 初始化
      (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d, D - d)); // 根据 gamma_null 加载 gamma 数据
    auto dy_vec = vec::Vectorized<T>::loadu(dY_data + d, D - d);  // 加载剩余部分的 dY 数据向量
    auto x_vec = vec::Vectorized<T>::loadu(X_data + d, D - d);    // 加载剩余部分的 X 数据向量
    auto dx_vec = c1 * dy_vec +                        // 计算 dx 向量
      vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
    dx_vec.store(dX_data + d, D - d);  // 将 dx 向量存储到 dX 中
  }
}

// 当 T 与 opmath_t 类型不同时，启用此函数，返回值类型为 void
template <typename T, typename PT, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastRowMov(
    const T* dY_data,          // 输入梯度 dY 的数据指针
    const T* X_data,           // 输入数据 X 的数据指针
    T* dX_data,                // 输出梯度 dX 的数据指针
    const PT* rstd,            // 标准差的逆值数据指针
    const PT* gamma,           // gamma 参数数据指针，如果为 nullptr 则 gamma_null 为 true
    opmath_t c2,               // 常数 c2，用于计算 dX
    opmath_t c3,               // 常数 c3，用于计算 dX
    int64_t HxW,               // 输入数据的 H * W 维度大小（未使用）
    int64_t C,                 // 输入数据的通道数（未使用）
    int64_t D) {               // 输入数据的总大小

  using Vec = vec::Vectorized<T>;  // 使用 Vec 作为 T 的向量化类型
  using fVec = vec::Vectorized<opmath_t>; // 使用 fVec 作为 opmath_t 的向量化类型
  const bool gamma_null = (gamma == nullptr); // 判断 gamma 是否为 nullptr
  auto K = Vec::size(); // 向量化操作的向量大小
  int64_t d = 0;  // 循环计数器

  // 对每个 K 大小的向量进行循环处理
  for (; d < D / K * K; d += K) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1)) // 根据 gamma_null 初始化 c1_0, c1_1
                                      : load_util(gamma + d, K);  // 加载 gamma 数据到 c1_0, c1_1
    c1_0 = c1_0 * fVec(opmath_t(*rstd));  // 计算 c1_0
    c1_1 = c1_1 * fVec(opmath_t(*rstd));  // 计算 c1_1
    Vec dy_vec = Vec::loadu(dY_data + d);  // 加载 dY 数据向量
    Vec x_vec = Vec::loadu(X_data + d);    // 加载 X 数据向量
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);  // 将 X 数据向量转换为浮点数向量
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);  // 将 dY 数据向量转换为浮点数向量
    fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);  // 计算 dx_vec0
    fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);  // 计算 dx_vec1
    convert_from_float<T>(dx_vec0, dx_vec1).store(dX_data + d);  // 将浮点数向量转换并存储到 dX 中
  }

  // 处理剩余不足一个完整向量大小的部分
  if (D - d > 0) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))  // 根据 gamma_null 初始化 c1_0, c1_1
                                      : load_util(gamma + d, D - d);  // 加载 gamma 数据到 c1_0, c1_1
    c1_0 = c1_0 * fVec(opmath_t(*rstd));  // 计算 c1_0
    c1_1 = c1_1 * fVec(opmath_t(*rstd));  // 计算 c1_1
    Vec dy_vec = Vec::loadu(dY_data + d, D - d);  // 加载剩余部分的 dY 数据向量
    Vec x_vec = Vec::loadu(X_data + d, D - d);    // 加载剩余部分的 X 数据向量
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);  // 将 X 数据向量转换为浮点数向量
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);  // 将 dY 数据向量转换为浮点数向量
    fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);  // 计算 dx_vec0
    fVec dx_vec1 = c
    // 计算 dx_vec1，其中 c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3)
    fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
    // 将浮点向量 dx_vec1 转换为指定类型 T，然后将结果存储到 dX_data + d 的位置，长度为 D - d
    convert_from_float<T>(dx_vec0, dx_vec1).store(dX_data + d, D - d);
  }
}

template <typename T, typename PT, typename opmath_t>
inline typename std::
    enable_if<std::is_same<T, opmath_t>::value, std::tuple<opmath_t, opmath_t>>::type
    CalcInternalGradientsChannelsLast(
    const T* X_data,
    const T* dY_data,
    const PT* gamma_ptr,
    opmath_t* ds_ptr,
    opmath_t* db_ptr,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;  // 使用 Vec 类型来处理向量化操作，T 是原始数据类型
  const bool gamma_null = (gamma_ptr == nullptr);  // 检查 gamma_ptr 是否为空
  constexpr int64_t K = Vec::size();  // K 是向量的大小
  const int64_t inner_size = D / K * K;  // 计算内部循环的大小，确保对齐
  int64_t d = 0;  // 初始化 d，用于迭代处理数据
  opmath_t ds_gamma{0}, db_gamma{0};  // 初始化梯度值 ds_gamma 和 db_gamma

  // 外层循环，处理所有的向量化数据块
  for (; d < inner_size; d += K) {
    Vec acc0_vec{0}, acc1_vec{0};  // 初始化累加向量 acc0_vec 和 acc1_vec

    // 内层循环，遍历每个 HxW 像素点
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;  // 获取输入数据 X 的指针
      const T* dY_ptr = dY_data + m * C;  // 获取梯度数据 dY 的指针
      Vec x_vec = Vec::loadu(X_ptr + d);  // 加载 X 数据的向量化表达
      Vec dy_vec = Vec::loadu(dY_ptr + d);  // 加载 dY 数据的向量化表达
      acc0_vec += x_vec * dy_vec;  // 计算 ds 梯度的累加向量
      acc1_vec += dy_vec;  // 计算 db 梯度的累加向量
    }

    acc0_vec.store(ds_ptr + d);  // 将 ds 梯度累加结果存储到 ds_ptr 中
    acc1_vec.store(db_ptr + d);  // 将 db 梯度累加结果存储到 db_ptr 中

    // 计算 gamma 的梯度 ds_gamma 和 db_gamma
    ds_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc0_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d)));
    db_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc1_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d)));
  }

  // 处理剩余的不足一个向量长度的数据
  if (D - d > 0) {
    Vec acc0_vec{0}, acc1_vec{0};  // 初始化累加向量 acc0_vec 和 acc1_vec

    // 内层循环，遍历每个 HxW 像素点
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;  // 获取输入数据 X 的指针
      const T* dY_ptr = dY_data + m * C;  // 获取梯度数据 dY 的指针
      Vec x_vec = Vec::loadu(X_ptr + d, D - d);  // 加载部分 X 数据的向量化表达
      Vec dy_vec = Vec::loadu(dY_ptr + d, D - d);  // 加载部分 dY 数据的向量化表达
      acc0_vec += x_vec * dy_vec;  // 计算 ds 梯度的累加向量
      acc1_vec += dy_vec;  // 计算 db 梯度的累加向量
    }

    acc0_vec.store(ds_ptr + d, D - d);  // 将 ds 梯度累加结果存储到 ds_ptr 中
    acc1_vec.store(db_ptr + d, D - d);  // 将 db 梯度累加结果存储到 db_ptr 中

    // 计算 gamma 的梯度 ds_gamma 和 db_gamma
    ds_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc0_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d, D - d)));
    db_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc1_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d, D - d)));
  }

  return std::tuple<opmath_t, opmath_t>(ds_gamma, db_gamma);  // 返回 ds_gamma 和 db_gamma 的元组
}

template <typename T, typename PT, typename opmath_t>
inline typename std::
    enable_if<!std::is_same<T, opmath_t>::value, std::tuple<opmath_t, opmath_t>>::type
    CalcInternalGradientsChannelsLast(
        const T* X_data,
        const T* dY_data,
        const PT* gamma_ptr,
        opmath_t* ds_ptr,
        opmath_t* db_ptr,
        int64_t HxW,
        int64_t C,
        int64_t D) {
  using Vec = vec::Vectorized<T>;  // 使用 Vec 类型来处理向量化操作，T 是原始数据类型
  using fVec = vec::Vectorized<opmath_t>;  // 使用 fVec 类型来处理 opmath_t 类型的向量化操作
  const bool gamma_null = (gamma_ptr == nullptr);  // 检查 gamma_ptr 是否为空
  constexpr int64_t K = Vec::size();  // K 是向量的大小
  const int64_t inner_size = D / K * K;  // 计算内部循环的大小，确保对齐
  float ds_gamma{0}, db_gamma{0};  // 初始化梯度值 ds_gamma 和 db_gamma
  int64_t d = 0;  // 初始化 d，用于迭代处理数据

  // 外层循环，处理所有的向量化数据块
  for (; d < inner_size; d += K) {
    fVec acc0_vec0{0}, acc0_vec1{0}, acc1_vec0{0}, acc1_vec1{0};  // 初始化累加向量
    // 对于每个索引 m 在范围 [0, HxW) 中循环迭代
    for (const auto m : c10::irange(HxW)) {
      // 计算 X 数据的指针 X_ptr，位移为 m * C，指向当前处理位置的数据块
      const T* X_ptr = X_data + m * C;
      // 计算 dY 数据的指针 dY_ptr，位移为 m * C，指向当前处理位置的梯度数据块
      const T* dY_ptr = dY_data + m * C;
      // 从 X_ptr 和 dY_ptr 中加载 Vec 类型的数据到 x_vec 和 dy_vec
      Vec x_vec = Vec::loadu(X_ptr + d);
      Vec dy_vec = Vec::loadu(dY_ptr + d);
      // 将 x_vec 和 dy_vec 转换为浮点数，存储在 x_vec0, x_vec1 和 dy_vec0, dy_vec1 中
      auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
      auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
      // 计算累加向量 acc0_vec0 和 acc0_vec1，分别为 x_vec0 * dy_vec0 和 x_vec1 * dy_vec1
      acc0_vec0 += x_vec0 * dy_vec0;
      acc0_vec1 += x_vec1 * dy_vec1;
      // 计算累加向量 acc1_vec0 和 acc1_vec1，分别为 dy_vec0 和 dy_vec1
      acc1_vec0 += dy_vec0;
      acc1_vec1 += dy_vec1;
    }
    // 将 acc0_vec0 和 acc0_vec1 存储到 ds_ptr 中，位置为 d 和 d + fVec::size()
    acc0_vec0.store(ds_ptr + d);
    acc0_vec1.store(ds_ptr + d + fVec::size());
    // 将 acc1_vec0 和 acc1_vec1 存储到 db_ptr 中，位置为 d 和 d + fVec::size()
    acc1_vec0.store(db_ptr + d);
    acc1_vec1.store(db_ptr + d + fVec::size());
    // 加载 gamma_vec0 和 gamma_vec1，如果 gamma_null 为 true，则设置为 fVec(1)，否则从 gamma_ptr + d 处加载
    auto [gamma_vec0, gamma_vec1] = gamma_null ?
      std::tuple<fVec, fVec>(fVec(1), fVec(1)) : load_util(gamma_ptr + d, K);
    // 计算 ds_gamma 的累加，对 acc0_vec0 * gamma_vec0 和 acc0_vec1 * gamma_vec1 进行向量加法
    ds_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc0_vec0 * gamma_vec0);
    // 计算 ds_gamma 的累加，对 acc0_vec1 * gamma_vec1 进行向量加法
    ds_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc0_vec1 * gamma_vec1);
    // 计算 db_gamma 的累加，对 acc1_vec0 * gamma_vec0 和 acc1_vec1 * gamma_vec1 进行向量加法
    db_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc1_vec0 * gamma_vec0);
    // 计算 db_gamma 的累加，对 acc1_vec1 * gamma_vec1 进行向量加法
    db_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc1_vec1 * gamma_vec1);
  }
  // 继续处理剩余的 d 值，范围为 [0, D)，每次增加 d 的值
  for (; d < D; d++) {
    // 初始化累加器 acc0 和 acc1 为零
    opmath_t acc0{0}, acc1{0};
    // 对于每个索引 m 在范围 [0, HxW) 中循环迭代
    for (const auto m : c10::irange(HxW)) {
      // 计算 X 数据的指针 X_ptr，位移为 m * C，指向当前处理位置的数据块
      const T* X_ptr = X_data + m * C;
      // 计算 dY 数据的指针 dY_ptr，位移为 m * C，指向当前处理位置的梯度数据块
      const T* dY_ptr = dY_data + m * C;
      // 累加 X_ptr[d] * dY_ptr[d] 到 acc0，累加 dY_ptr[d] 到 acc1
      acc0 += opmath_t(X_ptr[d]) * opmath_t(dY_ptr[d]);
      acc1 += opmath_t(dY_ptr[d]);
    }
    // 将 acc0 存储到 ds_ptr[d] 中，将 acc1 存储到 db_ptr[d] 中
    ds_ptr[d] = acc0;
    db_ptr[d] = acc1;
    // 加载 gamma_val，如果 gamma_null 为 true，则设置为 1，否则从 gamma_ptr[d] 处加载
    opmath_t gamma_val = gamma_null ? opmath_t(1) : opmath_t(gamma_ptr[d]);
    // 累加 ds_gamma，对 acc0 * gamma_val 进行加法
    ds_gamma += acc0 * gamma_val;
    // 累加 db_gamma，对 acc1 * gamma_val 进行加法
    db_gamma += acc1 * gamma_val;
  }

  // 返回 ds_gamma 和 db_gamma 的元组作为结果
  return std::tuple<opmath_t, opmath_t>(ds_gamma, db_gamma);
  // 根据输入的各个参数进行检查，确保张量的元素数量正确
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * group);
  TORCH_CHECK(rstd.numel() == N * group);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  // 计算每个分组的通道数和分组数
  int64_t D = C / group;
  int64_t G = group;

  // 获取输入张量的数据指针
  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const PT* mean_data = mean.const_data_ptr<PT>();
  const PT* rstd_data = rstd.const_data_ptr<PT>();
  const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;

  // 获取输出张量的数据指针，如果未定义则设为nullptr
  T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
  PT* dgamma_data = dgamma.defined() ? dgamma.data_ptr<PT>() : nullptr;
  PT* dbeta_data = dbeta.defined() ? dbeta.data_ptr<PT>() : nullptr;

  // 检查gamma是否为null
  const bool gamma_null = (gamma_data == nullptr);

  // 定义数学操作类型
  using opmath_t = at::opmath_type<T>;

  // 创建空的张量ds和db
  Tensor ds = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
  Tensor db = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));

  // 获取ds和db的数据指针
  opmath_t* ds_data = ds.data_ptr<opmath_t>();
  opmath_t* db_data = db.data_ptr<opmath_t>();

  // 计算s的值，这是一个常数
  const opmath_t s = opmath_t(1) / static_cast<opmath_t>(D * HxW);

  // 与前向传播的通道最后实现类似，通道最后的反向传播也有两种实现方式。
  // impl-1: 在 N * G 上并行。只需一个 omp 会话用于输入梯度，但每个线程的内存访问是非连续的。
  //
  // impl-2: 在 N * HxW 上并行。每个线程的内存访问是连续的，但需要额外的临时缓冲区大小为 {T, N, 2C}。

  // 通常情况下，当 HxW 足够大时，impl-2 的性能更好，因为每个线程的数据量 {NHWC / T} 明显大于临时缓冲区 {2NC}。

  // 定义一个特征图的阈值，以便在选择两种实现方式之间进行切换
  constexpr int64_t feature_map_threshold = 2048;

  // 根据特征图的大小选择不同的实现方式
  if (HxW < feature_map_threshold) {
    // impl-1: 在 N * G 上并行。
    // 使用 ATen 的 parallel_for 函数并行处理从 0 到 N * G 的索引范围
    at::parallel_for(0, N * G, 1, [=](int64_t begin, int64_t end) {
      // 初始化变量 n 和 g，分别表示 batch 和 group 的索引
      int64_t n{0}, g{0};
      // 调用 data_index_init 函数，初始化 n 和 g 的值
      data_index_init(begin, n, N, g, G);
      // 遍历 begin 到 end 的范围
      for (const auto i : c10::irange(begin, end)) {
        // Step 1. 计算内部梯度。

        // 获取当前线程的 ds_ptr 和 db_ptr
        opmath_t* ds_ptr = ds_data + i * D;
        opmath_t* db_ptr = db_data + i * D;

        // 获取指向 X 数据的指针，X 是一个四维张量，根据 n 和 g 计算偏移量
        const T* X_ptr = X_data + n * HxW * C + g * D;
        // 获取指向 dY 数据的指针，同样根据 n 和 g 计算偏移量
        const T* dY_ptr = dY_data + n * HxW * C + g * D;

        // gamma_ptr 指向 gamma 数据的指针，如果 gamma_null 为真则为空指针，否则根据 g 计算偏移量
        const PT* gamma_ptr = gamma_null ? gamma_data : (gamma_data + g * D);

        // 调用 CalcInternalGradientsChannelsLast 函数计算内部梯度 ds_gamma 和 db_gamma
        auto [ds_gamma, db_gamma] = CalcInternalGradientsChannelsLast<T, PT, opmath_t>(
          X_ptr,
          dY_ptr,
          gamma_ptr,
          ds_ptr,
          db_ptr,
          HxW,
          C,
          D);

        // Step 2. 计算 dX。

        // 获取指向 dX 数据的指针，根据 n 和 g 计算偏移量
        T* dX_ptr = dX_data + n * HxW * C + g * D;

        // rstd_ptr 指向 rstd 数据的指针，根据 i 计算偏移量
        const PT* rstd_ptr = rstd_data + i;

        // 计算常量 c2 和 c3，用于输入梯度计算
        const opmath_t c2 = (db_gamma * opmath_t(mean_data[i]) - ds_gamma) *
            opmath_t(rstd_data[i]) * opmath_t(rstd_data[i]) * opmath_t(rstd_data[i]) * s;
        const opmath_t c3 = -c2 * opmath_t(mean_data[i]) - db_gamma * opmath_t(rstd_data[i]) * s;

        // 调用 ApplyInputGradientsChannelsLastColMov 函数应用输入梯度
        ApplyInputGradientsChannelsLastColMov<T, PT, opmath_t>(dY_ptr, X_ptr, dX_ptr, rstd_ptr, gamma_ptr, c2, c3, HxW, C, D);

        // 更新索引 n 和 g 的值
        data_index_step(n, N, g, G);
      }
    });

  } else {
    // impl-2: 在 N * HxW 上并行处理。

    // 获取线程数
    int num_threads = at::get_num_threads();

    // 创建并初始化 buffer 张量，用于存储计算结果
    Tensor buffer = at::empty({num_threads, N, 2 * C},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value)).zero_();
    opmath_t* buffer_data = buffer.data_ptr<opmath_t>();

    // 创建并初始化 tmp_buffer 张量，用于存储临时计算结果
    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    opmath_t* tmp_buffer_data = tmp_buffer.data_ptr<opmath_t>();

    // Step 1. 每个线程计算它们自己的内部梯度到 buffer 中。
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      // 获取当前线程的编号
      int tid = at::get_thread_num();
      // 获取当前线程在 buffer 中的起始位置
      opmath_t* buffer_ptr = buffer_data + tid * N * 2 * C;
      // 初始化变量 n 和 m，分别表示 batch 和 spatial 的索引
      int64_t n{0}, m{0};
      // 调用 data_index_init 函数，初始化 n 和 m 的值
      data_index_init(begin, n, N, m, HxW);
      // 遍历 begin 到 end 的范围
      for (const auto i : c10::irange(begin, end)) {
        // 获取当前线程的 ds_ptr 和 db_ptr
        opmath_t* ds_ptr = buffer_ptr + n * 2 * C;
        opmath_t* db_ptr = ds_ptr + C;

        // 获取指向 X 数据的指针，X 是一个二维张量，根据 i 计算偏移量
        const T* X_ptr = X_data + i * C;
        // 获取指向 dY 数据的指针，同样根据 i 计算偏移量
        const T* dY_ptr = dY_data + i * C;

        // 调用 DsDbRowwiseMomentsChannelsLast 函数计算 ds 和 db
        DsDbRowwiseMomentsChannelsLast<T, opmath_t>(dY_ptr, X_ptr, ds_ptr, db_ptr, C);

        // 更新索引 n 和 m 的值
        data_index_step(n, N, m, HxW);
      }
    });

    // Step 2. 收集每个线程的内部梯度，并计算最终结果到 ds、db 和 tmp_buffer 中。
    // 遍历 N 次，每次迭代访问一个索引 n
    for (const auto n : c10::irange(N)) {
      // 遍历 G 次，每次迭代访问一个索引 g
      for (const auto g : c10::irange(G)) {
        // 初始化 ds_gamma 和 db_gamma 为零
        opmath_t ds_gamma{0}, db_gamma{0};
        // 遍历 D 次，每次迭代访问一个索引 d
        for (const auto d : c10::irange(D)) {
          // 初始化 ds_val 和 db_val 为零
          opmath_t ds_val{0}, db_val{0};
          // 遍历 num_threads 次，每次迭代访问一个线程索引 t
          for (const auto t : c10::irange(num_threads)) {
            // 计算当前线程的 buffer 数据指针
            opmath_t* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            // 计算 gamma_val，如果 gamma_null 为 true，则为 1，否则为 gamma_data[g * D + d] 的值
            opmath_t gamma_val = gamma_null ? opmath_t(1) : opmath_t(gamma_data[g * D + d]);
            // 更新 ds_gamma 和 db_gamma
            ds_gamma += buffer_ptr[g * D + d] * gamma_val;
            db_gamma += buffer_ptr[g * D + d + C] * gamma_val;
            // 更新 ds_val 和 db_val
            ds_val += buffer_ptr[g * D + d];
            db_val += buffer_ptr[g * D + d + C];
          }
          // 将 ds_val 和 db_val 分别存入 ds_data 和 db_data
          ds_data[n * C + g * D + d] = ds_val;
          db_data[n * C + g * D + d] = db_val;
        }
        // 将 ds_gamma 和 db_gamma 分别存入 tmp_buffer_data
        tmp_buffer_data[n * 2 * G + 2 * g] = ds_gamma;
        tmp_buffer_data[n * 2 * G + 2 * g + 1] = db_gamma;
      }
    }

    // 步骤 3：计算 dx
    if (dX_data != nullptr) {
      // 并行计算，遍历每个像素点 i
      at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
        int64_t n{0}, m{0};
        // 初始化数据索引 n 和 m
        data_index_init(begin, n, N, m, HxW);
        // 遍历像素点 i 的范围
        for (const auto i : c10::irange(begin, end)) {
          // 遍历 G 次，每次迭代访问一个索引 g
          for (const auto g : c10::irange(G)) {
            // 获取当前像素点 i 的 X_ptr、dY_ptr 和 dX_ptr
            const T* X_ptr = X_data + i * C + g * D;
            const T* dY_ptr = dY_data + i * C + g * D;
            T* dX_ptr = dX_data + i * C + g * D;
            // 获取当前均值 mean_ptr、标准差倒数 rstd_ptr 和 gamma_ptr
            const PT* mean_ptr = mean_data + n * G + g;
            const PT* rstd_ptr = rstd_data + n * G + g;
            const PT* gamma_ptr = gamma_null ? gamma_data : (gamma_data + g * D);
            // 获取 tmp_buffer_data 中的 ds_val 和 db_val
            opmath_t ds_val = tmp_buffer_data[n * 2 * G + 2 * g];
            opmath_t db_val = tmp_buffer_data[n * 2 * G + 2 * g + 1];

            // 计算常数 c2 和 c3
            const opmath_t c2 = (db_val * opmath_t(*mean_ptr) - ds_val) *
                opmath_t(*rstd_ptr) * opmath_t(*rstd_ptr)* opmath_t(*rstd_ptr) * s;
            const opmath_t c3 = -c2 * opmath_t(*mean_ptr) - db_val * opmath_t(*rstd_ptr) * s;

            // 应用输入梯度到 dX_ptr
            ApplyInputGradientsChannelsLastRowMov<T, PT, opmath_t>(dY_ptr, X_ptr, dX_ptr, rstd_ptr, gamma_ptr, c2, c3, HxW, C, D);
          }
          // 更新数据索引 n 和 m
          data_index_step(n, N, m, HxW);
        }
      });
    }

  }

  // 最后计算 dgamma 和 dbeta
  if (dgamma_data != nullptr) {
    // 计算 dgamma
    GammaBackward(
        N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
  }
  if (dbeta_data != nullptr) {
    // 计算 dbeta
    BetaBackward(N, C, db_data, dbeta_data);
  }
} // 结束 GroupNormBackwardKernelImpl 函数定义

void GroupNormBackwardKernelImpl(
    const Tensor& dY,  // 反向传播的梯度张量
    const Tensor& X,   // 输入张量 X
    const Tensor& mean,  // 均值张量
    const Tensor& rstd,  // 标准差的倒数张量
    const Tensor& gamma,  // 缩放参数张量
    int64_t N,  // 维度 N
    int64_t C,  // 维度 C
    int64_t HxW,  // 维度 HxW
    int64_t group,  // 组数
    Tensor& dX,  // X 的梯度张量
    Tensor& dgamma,  // gamma 的梯度张量
    Tensor& dbeta) {  // beta 的梯度张量
  // 在训练中，建议使用 Amp 来启用低精度数据类型，
  // 例如 BFloat16 或 Half。
  // 这将保持模块参数在 opmath 数据类型（例如 float），
  // 而输入/输出将使用较低精度的数据类型。
  // 使用 BFloat16 或 Half 参数可能会导致高精度损失。
  const bool mixed_type = is_mixed_type(dY, mean);

  switch (X.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;
        if(mixed_type) {
          // 调用内部函数，处理混合数据类型情况
          GroupNormBackwardKernelImplInternal<scalar_t, param_t>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        } else {
          // 调用内部函数，处理同一数据类型情况
          GroupNormBackwardKernelImplInternal<scalar_t, scalar_t>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;
        if(mixed_type) {
          // 调用内部函数，处理混合数据类型情况（通道为最后维度的情况）
          GroupNormBackwardKernelImplChannelsLastInternal<scalar_t, param_t>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        } else {
          // 调用内部函数，处理同一数据类型情况（通道为最后维度的情况）
          GroupNormBackwardKernelImplChannelsLastInternal<scalar_t, scalar_t>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        }
      });
      break;
    }
    default:
      // 不支持的内存格式错误提示
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");
  }

}

} // 结束 at::native 命名空间

// 注册 GroupNormKernel 的调度器
REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
// 注册 GroupNormBackwardKernel 的调度器
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);
```