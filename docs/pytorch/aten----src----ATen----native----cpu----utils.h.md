# `.\pytorch\aten\src\ATen\native\cpu\utils.h`

```py
#pragma once
// 预处理命令，确保头文件只包含一次

#include <ATen/Parallel.h>
// 引入 ATen 库的并行处理头文件

#include <ATen/cpu/vec/vec.h>
// 引入 ATen 库的向量化头文件

#include <c10/util/llvmMathExtras.h>
// 引入 C10 库的 LLVM 数学工具额外函数头文件

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif
// 如果定义了 USE_FBGEMM 宏，则引入 FBGEMM 库的头文件

namespace at {
namespace native {

template <typename T>
inline void _store(T* dst, at::vec::Vectorized<T> src) {
  // 将向量化的数据 src 存储到目标指针 dst 处
  src.store(dst);
}

inline void _store(at::BFloat16* dst, at::vec::Vectorized<float> src) {
  // 将向量化的 float 数据 src 转换成 BFloat16 类型后存储到目标指针 dst 处
  auto res = at::vec::convert_float_bfloat16(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

inline void _store(at::Half* dst, at::vec::Vectorized<float> src) {
  // 将向量化的 float 数据 src 转换成 Half 类型后存储到目标指针 dst 处
  auto res = at::vec::convert_float_half(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

inline namespace CPU_CAPABILITY {

template <typename T>
inline T data_index_init(T offset) {
  // 初始化数据索引偏移量
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  // 递归初始化数据索引偏移量，并设置每个维度的索引值 x
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  // 数据索引步进函数，返回始终为真
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  // 递归调用数据索引步进函数，并更新每个维度的索引值 x
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

// Helper struct for bfloat16 vectorization
// BFloat16 向量化的辅助结构体
using namespace vec;
struct Vec2 {
  Vectorized<float> val0, val1;
  Vec2(Vectorized<float> v0, Vectorized<float> v1) : val0(v0), val1(v1) {}
  Vec2(float v) : val0(v), val1(v) {}
  static Vec2 loadu(const BFloat16* ptr) {
    auto [v0, v1] = convert_bfloat16_float(Vectorized<BFloat16>::loadu(ptr));
    return {v0, v1};
  }
  static Vec2 loadu(const float* ptr) {
    return {Vectorized<float>::loadu(ptr), Vectorized<float>::loadu(ptr + Vectorized<float>::size())};
  }
  void store(BFloat16* ptr) const {
    Vectorized<BFloat16> val = convert_float_bfloat16(val0, val1);
    val.store(ptr);
  }
  void store(float* ptr) const {
    val0.store(ptr);
    val1.store(ptr + Vectorized<float>::size());
  }
};
inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.val0 + b.val0, a.val1 + b.val1}; }
inline Vec2 operator*(const Vec2& a, const Vec2& b) { return {a.val0 * b.val0, a.val1 * b.val1}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.val0 - b.val0, a.val1 - b.val1}; }
inline Vec2 operator/(const Vec2& a, const Vec2& b) { return {a.val0 / b.val0, a.val1 / b.val1}; }
inline Vec2 maximum(const Vec2& a, const Vec2& b) { return {vec::maximum(a.val0, b.val0), vec::maximum(a.val1, b.val1)}; }
inline Vec2 minimum(const Vec2& a, const Vec2& b) { return {vec::minimum(a.val0, b.val0), vec::minimum(a.val1, b.val1)}; }

template <typename scalar_t> struct VectorizedType { using type = Vectorized<scalar_t>; };
template <> struct VectorizedType<BFloat16> { using type = Vec2; };
template <typename scalar_t> using VecType = typename VectorizedType<scalar_t>::type;
// Helper function to load two float vectors from memory containing BFloat16 data
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const BFloat16* ptr) {
  return convert_bfloat16_float(Vectorized<BFloat16>::loadu(ptr));
}

// Helper function to load two float vectors from memory containing Half precision data
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const Half* ptr) {
  return convert_half_float(Vectorized<Half>::loadu(ptr));
}

// Helper function to load two float vectors from memory containing float data
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const float* ptr) {
  using Vec = Vectorized<float>;
  // Load two vectors of float data using Vectorized operations
  return std::make_tuple(Vec::loadu(ptr), Vec::loadu(ptr + Vec::size()));
}

// Helper function to load two float vectors from memory containing BFloat16 data with a specified count
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const BFloat16* ptr, int64_t count) {
  return convert_bfloat16_float(Vectorized<BFloat16>::loadu(ptr, count));
}

// Helper function to load two float vectors from memory containing Half precision data with a specified count
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const Half* ptr, int64_t count) {
  return convert_half_float(Vectorized<Half>::loadu(ptr, count));
}

// Helper function to load two float vectors from memory containing float data with a specified count
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const float* ptr, int64_t count) {
  using Vec = Vectorized<float>;
  if (count > Vec::size()) {
    // Load two vectors of float data, considering the specified count
    return std::make_tuple(Vec::loadu(ptr), Vec::loadu(ptr + Vec::size(), count - Vec::size()));
  } else {
    // Load one vector of float data and pad the second vector with zeros
    return std::make_tuple(Vec::loadu(ptr, count), Vec(0));
  }
}

} // namespace

namespace utils {

// Function to compute the ceiling of the logarithm base 2 of a value
template <typename T>
T CeilLog2(const T& x) {
  if (x <= 2) {
    return 1;
  }
  // Calculate the ceiling of log2(x) using llvm::findLastSet function
  // Adjust for exact powers of 2 by subtracting 1 from x before computation
  return static_cast<T>(llvm::findLastSet(static_cast<uint64_t>(x) - 1)) + 1;
}

// Function to transpose a matrix
// src has shape M by N, with leading dimension ld_src
// dst has shape N by M, with leading dimension ld_dst
template <typename T>
inline void transpose(int64_t M, int64_t N, const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  for (int64_t j = 0; j < N; j++) {
    for (int64_t i = 0; i < M; i++) {
      // Transpose elements from src to dst based on specified dimensions
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

#ifdef USE_FBGEMM
// Specialization of transpose function for float type using FBGEMM library
template <>
inline void transpose<float>(int64_t M, int64_t N, const float* src, int64_t ld_src, float* dst, int64_t ld_dst) {
  // Check if FBGEMM is supported on the current CPU
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  // Perform matrix transpose using FBGEMM transpose_simd function
  fbgemm::transpose_simd<float>(M, N, src, ld_src, dst, ld_dst);
}
#endif

// Function to parallelize sparse CSR matrix operations
template <typename index_t, typename F>
inline void parallel_sparse_csr(
    const TensorAccessor<index_t, 1>& crow_acc,
    const int64_t M,
    const int64_t nnz,
    const F& f) {
  // Verify the size of crow_acc matches M + 1
  TORCH_CHECK(crow_acc.size(0) == M + 1);

  // Determine thread splits based on number of threads and nnz to balance workload
  int num_threads = at::get_num_threads();
  std::vector<int64_t> thread_splits(num_threads + 1, M);

  int64_t thread_averge_payload = std::max((int64_t)1, divup(nnz, num_threads));

  thread_splits[0] = 0;
  int64_t sum = 0;
  int64_t t = 1;
  for (const auto m : c10::irange(M)) {
    int64_t row_start = crow_acc[m];
    int64_t row_end = crow_acc[m + 1];
    // Compute thread splits dynamically based on nnz distribution
    sum += row_end - row_start;
    if (sum > thread_averge_payload && t < num_threads) {
      thread_splits[t++] = m + 1;
      sum = 0;
    }
  }
  thread_splits[num_threads] = M;
}
    // 计算当前线程处理的行数总和
    sum += row_end - row_start;
    // 如果总和超过预设的平均线程负载阈值，则进行线程分割
    if (sum > t * thread_averge_payload) {
      // 将当前线程分割点记录在数组中，并增加线程计数器 t
      thread_splits[t] = m;
      t++;
    }
  }
  // 由于计算 `thread_averge_payload` 时可能存在舍入误差，需要恢复最后一个索引
  // 将其作为最后一个线程分割点，确保所有数据均被处理
  thread_splits[num_threads] = M;

  // 使用 ATen 的并行处理函数，以多线程方式执行 f 函数
  at::parallel_for(0, num_threads, 1, [&](int64_t cbegin, int64_t cend) {
    // 获取当前线程的线程 ID
    int tid = at::get_thread_num();
    // 获取当前线程的起始和结束索引
    int64_t begin = thread_splits[tid];
    int64_t end = thread_splits[tid + 1];
    // 调用函数 f 处理从 begin 到 end 的数据段
    f(begin, end);
  });
}

} // namespace utils

} // namespace native
} // namespace at
```