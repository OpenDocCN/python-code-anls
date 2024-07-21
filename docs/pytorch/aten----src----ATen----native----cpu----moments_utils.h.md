# `.\pytorch\aten\src\ATen\native\cpu\moments_utils.h`

```
#pragma once
// 使用#pragma once确保头文件只被编译一次

#include <array>
// 包含array头文件，用于使用std::array

#include <cstring>
// 包含cstring头文件，用于C风格字符串操作

#include <numeric>
// 包含numeric头文件，用于进行数值操作

#include <utility>
// 包含utility头文件，用于通用编程工具

#include <vector>
// 包含vector头文件，用于使用std::vector

#include <ATen/Parallel.h>
// 包含ATen并行处理的头文件

#include <ATen/OpMathType.h>
// 包含ATen中操作数学类型的头文件

#include <ATen/cpu/vec/vec.h>
// 包含ATen中向量化处理的头文件

#include <ATen/native/cpu/utils.h>
// 包含ATen中CPU工具函数的头文件

#include <c10/util/SmallVector.h>
// 包含c10中SmallVector的头文件

#include <c10/util/irange.h>
// 包含c10中整数范围迭代器的头文件

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {

template<typename T> using opmath_t = at::opmath_type<T>;
// 定义模板类型别名opmath_t，用于表示ATen中的操作数学类型

constexpr int64_t kChunkSize = 16;
// 定义constexpr常量kChunkSize为16，用于表示块大小

template <typename T>
void AddMoments(
    int64_t m0_add,
    const T& m1_add,
    const T& m2_add,
    int64_t& m0,
    T& m1,
    T& m2) {
  // 添加矩时的函数模板，用于更新均值和方差的累加值
  const int64_t n = m0 + m0_add;
  // 计算新的样本总数
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  // 计算加权系数c，避免除以零情况
  const T delta = m1_add - m1;
  // 计算均值的变化量
  m1 += c * delta;
  // 更新均值
  m2 += m2_add + delta * delta * c * static_cast<T>(m0);
  // 更新方差
  m0 = n;
  // 更新样本总数
}

template <typename T>
C10_ALWAYS_INLINE void AddMomentsVec(
    int64_t m0_add,
    const vec::Vectorized<T>& m1_add,
    const vec::Vectorized<T>& m2_add,
    int64_t& m0,
    vec::Vectorized<T>& m1,
    vec::Vectorized<T>& m2) {
  // 向量化版本的添加矩时函数模板，用于并行更新均值和方差的累加值
  using Vec = vec::Vectorized<T>;
  const int64_t n = m0 + m0_add;
  // 计算新的样本总数
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  // 计算加权系数c，避免除以零情况
  const Vec c_vec(c);
  // 将c转换为向量形式
  const Vec delta = m1_add - m1;
  // 计算均值的向量化变化量
  m1 += c_vec * delta;
  // 更新均值向量
  m2 += m2_add + delta * delta * c_vec * Vec(static_cast<T>(m0));
  // 更新方差向量
  m0 = n;
  // 更新样本总数
}

template <typename T>
inline typename std::enable_if<std::is_same<T, opmath_t<T>>::value, void>::type
UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<vec::Vectorized<opmath_t<T>>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    vec::Vectorized<opmath_t<T>>& m1_stk0,
    vec::Vectorized<opmath_t<T>>& m2_stk0) {
  // 更新均值和方差的向量化函数，针对ATen中操作数学类型
  using Vec = vec::Vectorized<opmath_t<T>>;
  Vec m1_vec(0);
  // 初始化均值向量
  Vec m2_vec(0);
  // 初始化方差向量
  for (const auto j : c10::irange(m0)) {
    // 迭代处理样本数据
    const Vec x_vec = Vec::loadu(X_ptr + j * Vec::size());
    // 加载当前块的数据到向量x_vec
    const Vec delta_vec = x_vec - m1_vec;
    // 计算当前块数据与均值的向量化差值
    m1_vec += delta_vec * c_vecs[j];
    // 更新均值向量
    m2_vec += delta_vec * (x_vec - m1_vec);
    // 更新方差向量
  }
  AddMomentsVec(m0, m1_vec, m2_vec, m0_stk0, m1_stk0, m2_stk0);
  // 调用向量化矩时函数更新累加的均值和方差向量
}

// 每个bfloat16/half向量将被转换为两个float向量，
// 并依次累加到m1_stk0/m2_stk0上。
template <typename T>
inline typename std::enable_if<!std::is_same<T, at::opmath_type<T>>::value, void>::type
UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<vec::Vectorized<at::opmath_type<T>>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    vec::Vectorized<at::opmath_type<T>>& m1_stk0,
    vec::Vectorized<at::opmath_type<T>>& m2_stk0) {
  // 更新均值和方差的向量化函数，针对非ATen中操作数学类型
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<at::opmath_type<T>>;
  // 定义向量类型Vec和浮点向量类型fVec
  fVec m1_fvec0(0), m1_fvec1(0);
  // 初始化浮点均值向量
  fVec m2_fvec0(0), m2_fvec1(0);
  // 初始化浮点方差向量
  for (const auto j : c10::irange(m0)) {
    // 迭代处理样本数据
    const Vec x_bvec = Vec::loadu(X_ptr + j * Vec::size());
    // 加载当前块的数据到向量x_bvec
    auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
    // 将bfloat16/half向量转换为两个float向量
    const fVec delta_fvec0 = x_fvec0 - m1_fvec0;
    // 计算第一个浮点均值向量的变化量
    const fVec delta_fvec1 = x_fvec1 - m1_fvec1;
    // 计算第二个浮点均值向量的变化量
    # 计算 m1_fvec0，使用 delta_fvec0 乘以 c_vecs[j]，并将结果累加到 m1_fvec0 上
    m1_fvec0 += delta_fvec0 * c_vecs[j];
    
    # 计算 m1_fvec1，使用 delta_fvec1 乘以 c_vecs[j]，并将结果累加到 m1_fvec1 上
    m1_fvec1 += delta_fvec1 * c_vecs[j];
    
    # 计算 m2_fvec0，使用 delta_fvec0 乘以 (x_fvec0 - m1_fvec0)，并将结果累加到 m2_fvec0 上
    m2_fvec0 += delta_fvec0 * (x_fvec0 - m1_fvec0);
    
    # 计算 m2_fvec1，使用 delta_fvec1 乘以 (x_fvec1 - m1_fvec1)，并将结果累加到 m2_fvec1 上
    m2_fvec1 += delta_fvec1 * (x_fvec1 - m1_fvec1);
    
  }
  
  # 调用函数 AddMomentsVec，传递 m0, m1_fvec0, m2_fvec0, m0_stk0, m1_stk0, m2_stk0 作为参数
  AddMomentsVec(m0, m1_fvec0, m2_fvec0, m0_stk0, m1_stk0, m2_stk0);
  
  # 再次调用函数 AddMomentsVec，传递 m0, m1_fvec1, m2_fvec1, m0_stk0, m1_stk0, m2_stk0 作为参数
  AddMomentsVec(m0, m1_fvec1, m2_fvec1, m0_stk0, m1_stk0, m2_stk0);
}

// Compute rowwise moments by Welford algorithm and cascade sum to improve
// numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
// https://en.wikipedia.org/wiki/Pairwise_summation
// 实现 Welford 算法计算行向量的矩，通过级联求和提升数值稳定性
// 相关链接：
//   - 计算方差的算法：https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
//   - 两两求和：https://en.wikipedia.org/wiki/Pairwise_summation
template <typename T, int64_t kMaxDepth>
std::pair<opmath_t<T>, opmath_t<T>> RowwiseMomentsImpl(const T* X, int64_t N, int64_t ddof = 0) {
  using math_t = opmath_t<T>;

  constexpr int64_t kVecSize = vec::Vectorized<T>::size();  // 向量大小
  constexpr int64_t kAccVecSize = vec::Vectorized<math_t>::size();  // 累加向量大小
  const int64_t n = N / kVecSize;  // 向量个数
  const int64_t m = divup(n, kChunkSize);  // 划分块数
  const int64_t depth = utils::CeilLog2(m);  // 深度

  using Vec = vec::Vectorized<math_t>;
  const Vec kZeroVec(math_t(0));  // 零向量
  c10::SmallVector<int64_t, kMaxDepth> m0_stk(depth, 0);  // 均值个数栈
  c10::SmallVector<Vec, kMaxDepth> m1_stk(depth, kZeroVec);  // 均值栈
  c10::SmallVector<Vec, kMaxDepth> m2_stk(depth, kZeroVec);  // 方差栈

  // 遍历块
  for (const auto i : c10::irange(m)) {
    const T* X_ptr = X + i * kChunkSize * kVecSize;  // 当前块的起始指针
    const int64_t m0 = std::min(kChunkSize, n - i * kChunkSize);  // 当前块中的元素个数
    static std::array<Vec, kChunkSize> c_vecs = ([]() {
      std::array<Vec, kChunkSize> result;
      for (const auto i : c10::irange(kChunkSize)) {
        result[i] = Vec(math_t(1) / static_cast<math_t>(i + 1));  // 逆元素向量
      }
      return result;
    })();
    UpdateMomentsVec(m0, X_ptr, c_vecs, m0_stk[0], m1_stk[0], m2_stk[0]);  // 更新矩向量

    int64_t mask = i + 1;  // 掩码
    for (int64_t j = 1; j < depth && (mask & 1) == 0; ++j) {
      AddMomentsVec(
          m0_stk[j - 1],
          m1_stk[j - 1],
          m2_stk[j - 1],
          m0_stk[j],
          m1_stk[j],
          m2_stk[j]);  // 添加矩向量
      m0_stk[j - 1] = 0;  // 清零
      m1_stk[j - 1] = kZeroVec;  // 清零
      m2_stk[j - 1] = kZeroVec;  // 清零
      mask >>= 1;  // 右移
    }
  }
  for (const auto i : c10::irange(1, depth)) {
    AddMomentsVec(
        m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);  // 添加矩向量
  }

  std::array<math_t, kAccVecSize> m1_arr{};  // 均值数组
  std::array<math_t, kAccVecSize> m2_arr{};  // 方差数组
  m1_stk[0].store(m1_arr.data());  // 存储均值到数组
  m2_stk[0].store(m2_arr.data());  // 存储方差到数组

  int64_t m0 = 0;  // 均值个数
  math_t m1 = 0;  // 均值
  math_t m2 = 0;  // 方差
  for (int64_t i = n * kVecSize; i < N; ++i) {
    math_t x = static_cast<math_t>(X[i]);  // 转换为 math_t 类型
    const math_t delta = x - m1;  // 与均值的差
    ++m0;  // 均值个数加一
    m1 += delta / static_cast<math_t>(m0);  // 更新均值
    m2 += delta * (x - m1);  // 更新方差
  }
  // 对于 BFloat16，每个向量在 m1_arr/m2_arr 中保存 2*n 的累加结果
  int64_t m0_add = n * kVecSize / kAccVecSize;  // 累加均值个数
  for (const auto i : c10::irange(kAccVecSize)) {
    AddMoments(m0_add, m1_arr[i], m2_arr[i], m0, m1, m2);  // 添加矩
  }

  return std::make_pair(m1, m2 / static_cast<math_t>(N - ddof));  // 返回均值和无偏方差
}

template <typename T>
std::pair<opmath_t<T>, opmath_t<T>> RowwiseMoments(const T* X, int64_t N, int64_t ddof = 0) {
  using Vec = vec::Vectorized<T>;  // 向量类型
  constexpr int64_t kVecSize = Vec::size();  // 向量大小
  const int64_t n = N / kVecSize;  // 向量个数
  const int64_t m = divup(n, kChunkSize);  // 划分块数
  const int64_t depth = utils::CeilLog2(m);  // 深度
  if (depth <= 4) {  // 如果深度小于等于 4
    // 如果深度小于或等于4，调用 RowwiseMomentsImpl 模板函数处理输入 X，数据数量 N，自由度修正 ddof
    return RowwiseMomentsImpl<T, 4>(X, N, ddof);
    // 如果深度在5到8之间（包括8），调用 RowwiseMomentsImpl 模板函数处理输入 X，数据数量 N，自由度修正 ddof
    } else if (depth <= 8) {
        return RowwiseMomentsImpl<T, 8>(X, N, ddof);
    // 如果深度在9到16之间（包括16），调用 RowwiseMomentsImpl 模板函数处理输入 X，数据数量 N，自由度修正 ddof
    } else if (depth <= 16) {
        return RowwiseMomentsImpl<T, 16>(X, N, ddof);
    // 如果深度在17到32之间（包括32），调用 RowwiseMomentsImpl 模板函数处理输入 X，数据数量 N，自由度修正 ddof
    } else if (depth <= 32) {
        return RowwiseMomentsImpl<T, 32>(X, N, ddof);
    // 对于更大深度（大于32），调用 RowwiseMomentsImpl 模板函数处理输入 X，数据数量 N，自由度修正 ddof，使用模板参数 64
    } else {
        return RowwiseMomentsImpl<T, 64>(X, N, ddof);
    }
}
} // namespace CPU_CAPABILITY
} // namespace native
} // namespace at
```