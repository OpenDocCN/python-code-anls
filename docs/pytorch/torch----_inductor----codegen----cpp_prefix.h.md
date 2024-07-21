# `.\pytorch\torch\_inductor\codegen\cpp_prefix.h`

```py
#pragma once

#include <algorithm>  // 包含算法标准库，提供各种算法函数
#include <atomic>     // 包含原子操作标准库，提供原子操作类型和函数
#include <cmath>      // 包含数学函数标准库，提供数学计算函数
#include <cstdlib>    // 包含标准库，提供通用实用程序函数
#include <limits>     // 包含数值极限标准库，提供各种数据类型的极限值
#include <memory>     // 包含智能指针和内存管理标准库，提供内存分配和管理功能
#include <omp.h>      // 包含 OpenMP 并行编程标准库，提供多线程并行处理支持

// WARNING: be extra careful when including more ATen/c10 header files here!
// Because AOTInductor generated code will copy-paste this cpp_prefix.h for
// the CPU backend, we have to make sure the used headers are implemented
// in a header-only way, i.e. all the function and class definitions are
// in .h files instead of .cpp files, to avoid ABI backward-compatiblity breakage.

#include <ATen/NumericUtils.h>    // 包含 ATen 数值工具标准库，提供数值相关工具函数
#include <ATen/core/PhiloxRNGEngine.h>  // 包含 ATen Philox 随机数引擎标准库，提供随机数引擎实现

#include <c10/util/Float8_e4m3fn.h>  // 包含 c10 浮点数处理标准库，提供特定浮点数类型支持
#include <c10/util/Float8_e5m2.h>    // 包含 c10 浮点数处理标准库，提供特定浮点数类型支持
#include <c10/util/BFloat16.h>       // 包含 c10 BFloat16 标准库，提供 BFloat16 数据类型支持
#include <c10/util/BFloat16-math.h>  // 包含 c10 BFloat16 数学标准库，提供 BFloat16 数学函数支持
#include <c10/util/generic_math.h>   // 包含 c10 通用数学标准库，提供通用数学函数支持
#include <c10/util/Half.h>           // 包含 c10 半精度浮点数标准库，提供半精度浮点数支持
#include <c10/util/TypeCast.h>       // 包含 c10 类型转换标准库，提供类型转换功能支持

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON)
#define INDUCTOR_USE_VECTOR_TYPES() 1  // 如果支持向量类型，则定义为 1
#else
#define INDUCTOR_USE_VECTOR_TYPES() 0  // 否则定义为 0
#endif

#if INDUCTOR_USE_VECTOR_TYPES()
#include <ATen/cpu/vec/functional.h>  // 包含 ATen 向量函数标准库，提供向量化操作支持
#include <ATen/cpu/vec/vec.h>         // 包含 ATen 向量类型标准库，提供向量数据类型支持
#else
// For calc_erfinv
#include <ATen/native/Math.h>         // 包含 ATen 数学库，提供数学计算支持
#endif

typedef at::Half half;                 // 定义 half 类型为 ATen 半精度浮点数
typedef at::BFloat16 bfloat16;         // 定义 bfloat16 类型为 ATen BFloat16 浮点数

typedef at::Float8_e4m3fn float8_e4m3fn;  // 定义 float8_e4m3fn 类型为 ATen Float8_e4m3fn 类型
typedef at::Float8_e5m2 float8_e5m2;      // 定义 float8_e5m2 类型为 ATen Float8_e5m2 类型

template <typename T>
struct Welford {
  T mean = T(0);        // 平均值初始化为 0
  T m2 = T(0);           // m2 初始化为 0
  int64_t index = 0;     // 索引初始化为 0
};

template <typename T>
struct IsVecType: std::false_type {};  // 默认情况下，IsVecType 结构体为 false_type

#if INDUCTOR_USE_VECTOR_TYPES()
template <typename T>
struct IsVecType<at::vec::Vectorized<T>>: std::true_type {};  // 如果支持向量类型，则为 true_type
#endif

template <typename T>
struct WeightRecp {
  using scalar_t = typename T::value_type;  // 使用 T 的值类型作为 scalar_t 类型
  int64_t N;                               // 记录数据点数量 N
  std::vector<scalar_t> weight_recps;      // 存储权重倒数的向量
  WeightRecp(int64_t N) : N(N) {           // 构造函数，初始化 N
    weight_recps.reserve(N);               // 预留空间以存储 N 个数据点的权重
    for (const auto i : c10::irange(N)) {  // 遍历从 0 到 N-1
      weight_recps.push_back(              // 计算权重倒数并存储到 weight_recps 中
          scalar_t(static_cast<double>(1) / static_cast<double>(i + 1)));
    }
  }
};

template <typename T>
Welford<T> welford_combine(const Welford<T> &a, const Welford<T> &b) {
  if (a.index == 0) {                       // 如果 a 的索引为 0
    return b;                               // 返回 b 的 Welford 结构体
  }
  if (b.index == 0) {                       // 如果 b 的索引为 0
    return a;                               // 返回 a 的 Welford 结构体
  }
  auto delta = b.mean - a.mean;             // 计算 b.mean 和 a.mean 的差值
  auto new_index = a.index + b.index;       // 计算新的索引
  auto wb_over_w = T(b.index) / T(new_index);  // 计算权重比例
  auto result = Welford<T>{                 // 构造新的 Welford 结构体
    a.mean + delta * wb_over_w,             // 更新均值
    a.m2 + b.m2 + delta * delta * T(a.index) * wb_over_w,  // 更新 m2
    new_index,                              // 更新索引
  };
  return result;                            // 返回更新后的 Welford 结构体
}

template <typename T>
Welford<T> welford_combine(const Welford<T> &acc, T data, const WeightRecp<T>* w=nullptr) {
  // Add a single data point
  int64_t index = acc.index + 1;            // 计算新的索引
  auto delta = data - acc.mean;             // 计算数据点和当前均值的差值
  T new_mean;
  if constexpr (!IsVecType<T>::value) {     // 如果 T 不是向量类型
    new_mean = acc.mean + delta / T(index); // 更新均值
  } else {                                  // 否则
    new_mean = acc.mean +
      ((w == nullptr || acc.index >= w->weight_recps.size())
            ? delta / T(index)
            : delta * T(w->weight_recps[acc.index]));  // 根据权重更新均值
  }
  auto new_delta = data - new_mean;         // 计算新的 delta
  auto result = Welford<T>{                 // 构造新的 Welford 结构体
    new_mean,                               // 更新均值
    acc.m2 + new_delta * delta * T(index),  // 更新 m2
    index,                                  // 更新索引
  };
  return result;                            // 返回更新后的 Welford 结构体
}
    new_mean,  // 更新后的均值
    acc.m2 + delta * new_delta,  // 更新后的第二中心矩
    index  // 索引值
  };
  return result;  // 返回结果数组
// 参考 https://github.com/pytorch/pytorch/blob/b5b36cf0c4e1958f1ff25120f5d4beeef3288187/
// aten/src/ATen/native/SharedReduceOps.h#L419-L445

// 定义模板函数，比较两个数值大小，并处理 NaN 情况
template <typename scalar_t>
inline bool greater_or_nan(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) {
  // 如果 a 是 NaN
  if (at::_isnan(a)) {
    // 如果 b 也是 NaN，选择索引更小的那个
    if (at::_isnan(b)) {
      return idx_a < idx_b;
    }
    // 如果 b 不是 NaN，返回 true
    return true;
  }
  // 如果 a 不是 NaN，比较 a 和 b 的大小，或者选择索引更小的那个
  return (a == b) ? idx_a < idx_b : (a > b);
}

// 定义模板函数，比较两个数值大小，并处理 NaN 情况
template <typename scalar_t>
inline bool less_or_nan(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) {
  // 如果 a 是 NaN
  if (at::_isnan(a)) {
    // 如果 b 也是 NaN，选择索引更小的那个
    if (at::_isnan(b)) {
      return idx_a < idx_b;
    }
    // 如果 b 不是 NaN，返回 true
    return true;
  }
  // 如果 a 不是 NaN，比较 a 和 b 的大小，或者选择索引更小的那个
  return (a == b) ? idx_a < idx_b : (a < b);
}

// 如果启用了向量类型，则定义向量化的向下移位函数
#if INDUCTOR_USE_VECTOR_TYPES()
template <typename scalar_t>
inline at::vec::Vectorized<scalar_t> vec_shuffle_down(at::vec::Vectorized<scalar_t> x, size_t n) {
  using Vec = at::vec::Vectorized<scalar_t>;
  // 定义一个数组，用于存储向量化数据
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  // 将向量化数据存储到数组中
  x.store(array);
  // 执行向下移位操作
  for (size_t i = 0; i + n < Vec::size(); i += 2 * n) {
    array[i] = array[i + n];
  }
  // 加载处理后的向量数据并返回
  return Vec::loadu(array);
}

// 如果 CPU 支持 AVX2，则定义 AVX2 版本的向下移位函数
#ifdef CPU_CAPABILITY_AVX2
inline at::vec::Vectorized<float> vec_shuffle_down(at::vec::Vectorized<float> x, size_t n) {
  using vec_t = at::vec::Vectorized<float>;
#define SHUFFLE_MASK(z, y, x, w) ((z << 6) | (y << 4) | (x << 2) | w)
  switch (n) {
  case 1:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(1, 1, 3, 3)));
  case 2:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(2, 2, 2, 2)));
  case 4:
    return vec_t(_mm256_permute2f128_ps(x, x, SHUFFLE_MASK(1, 1, 1, 1)));
  }
  // 抛出错误，处理未支持的向下移位值
  TORCH_CHECK(false, "Unhandled vec_shuffle_down value ", n);
}
#endif

// 如果 CPU 支持 AVX512，则定义 AVX512 版本的向下移位函数
#ifdef CPU_CAPABILITY_AVX512
inline at::vec::Vectorized<float> vec_shuffle_down(at::vec::Vectorized<float> x, size_t n) {
  using vec_t = at::vec::Vectorized<float>;
#define SHUFFLE_MASK(z, y, x, w) ((z << 6) | (y << 4) | (x << 2) | w)
  switch (n) {
    case 1:
      return vec_t(_mm512_permute_ps(x, SHUFFLE_MASK(1, 1, 3, 3)));
    case 2:
      return vec_t(_mm512_permute_ps(x, SHUFFLE_MASK(2, 2, 2, 2)));
    case 4:
      return vec_t(_mm512_permutexvar_ps(
          _mm512_set_epi32(
              12, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4),
          x));
    case 8:
      return vec_t(_mm512_permutexvar_ps(
          _mm512_set_epi32(8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8), x));
  }
  // 抛出错误，处理未支持的向下移位值
  TORCH_CHECK(false, "Unhandled vec_shuffle_down value ", n);
}
#endif

// 定义模板函数，将向量化的 Welford 结构体数据归约到单一的 Welford 结构体
template <typename scalar_t>
Welford<scalar_t> welford_vec_reduce_all(Welford<at::vec::Vectorized<scalar_t>> acc) {
  using Vec = at::vec::Vectorized<scalar_t>;
  // 循环归约，直到向量大小
  for (size_t n = 1; n < Vec::size(); n *= 2) {
    auto index = acc.index;
    // 使用向下移位函数处理平均值和平方平均值
    auto shuffled = Welford<Vec>{
      vec_shuffle_down(acc.mean, n),
      vec_shuffle_down(acc.m2, n),
      index,
    };
    // 将当前累加器的统计信息与打乱后的数据统计信息合并，更新累加器
    acc = welford_combine(acc, shuffled);
  }

  // 创建一个 Welford 对象来保存计算结果
  Welford<scalar_t> result;
  // 使用 Vec 对齐的方式分配内存，用于存储统计数据的数组
  alignas(alignof(Vec)) scalar_t array[Vec::size()];

  // 将累加器中的均值存储到数组中
  acc.mean.store(array);
  // 将数组中的第一个元素作为计算结果对象的均值
  result.mean = array[0];

  // 将累加器中的 m2 值（方差）存储到数组中
  acc.m2.store(array);
  // 将数组中的第一个元素作为计算结果对象的 m2 值（方差）
  result.m2 = array[0];

  // 将累加器中的索引值赋给计算结果对象的索引
  result.index = acc.index;

  // 返回最终的 Welford 对象，其中包含了均值、方差和索引信息
  return result;
#endif

// 模板函数：求取两个数的模
template <typename T, typename U> inline typename std::common_type<T, U>::type mod(T a, U b) { return a % b; }

// 特化模板函数：求取两个浮点数的模
template <> inline float mod(float a, float b) { return std::fmod(a, b); }

// 特化模板函数：求取两个双精度浮点数的模
template <> inline double mod(double a, double b) { return std::fmod(a, b); }

// 函数模板：比较两个数并处理 NaN 的最大值传播
template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {  // 检查 a 是否为 NaN
    return a;  // 若 a 是 NaN，则直接返回 a
  }
  return a > b ? a : b;  // 否则返回较大的数
}

// 函数模板：比较两个数并处理 NaN 的最小值传播
template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {  // 检查 a 是否为 NaN
    return a;  // 若 a 是 NaN，则直接返回 a
  }
  return a < b ? a : b;  // 否则返回较小的数
}

// constexpr 函数：将 uint32_t 转换为 [0, 1) 之间的均匀分布浮点数
constexpr float uint32_to_uniform_float(uint32_t value) {
  // scale 是一个常量，用于将 uint32_t 转换为浮点数
  constexpr float scale = 4.6566127342e-10;
  return static_cast<float>(value & 0x7FFFFFFF) * scale;  // 将 uint32_t 的值转换为浮点数并返回
}

// 函数：生成 [0, 1) 之间均匀分布的随机数
float normalized_rand_cpu(uint32_t seed, uint32_t offset) {
  return uint32_to_uniform_float(at::Philox4_32(seed, 0, offset)());  // 使用 Philox4_32 算法生成随机数并转换为浮点数返回
}

// 函数：生成标准正态分布的随机数
float randn_cpu(uint32_t seed, uint32_t offset) {
  at::Philox4_32 engine(seed, 0, offset);  // 创建 Philox4_32 随机数生成引擎
  return engine.randn(10);  // 生成标准正态分布的随机数并返回
}

// 函数：生成指定范围内的随机整数
int64_t randint64_cpu(uint32_t seed, uint32_t offset, int64_t low, int64_t high) {
  auto gen = at::Philox4_32(seed, 0, offset);  // 创建 Philox4_32 随机数生成引擎
  uint64_t r0 = gen();  // 生成第一个随机数
  uint64_t r1 = gen();  // 生成第二个随机数
  uint64_t result = r0 | (r1 << 32);  // 将两个随机数合并为一个 64 位整数
  return static_cast<int64_t>(result % (high - low)) + low;  // 将结果映射到指定范围并返回
}

// 结构模板：将浮点类型映射为整数类型
template <typename T> struct AsIntegerType { typedef T type; };
template <> struct AsIntegerType<float> { typedef uint32_t type; };
template <> struct AsIntegerType<double> { typedef uint64_t type; };
template <> struct AsIntegerType<bfloat16> { typedef uint16_t type; };

// 函数模板：如果 T 不是浮点类型，则获取地址处的值
template <typename T>
typename std::enable_if_t<!std::is_floating_point_v<T>, T>
inline fetch_value(volatile T *addr) {
  return *addr;  // 返回地址处的值
}

// 函数模板：如果 T 是浮点类型，则获取地址处的值并返回浮点数对象
template <typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, T>
inline fetch_value(volatile T *addr) {
  return T(addr->x, T::from_bits());  // 返回浮点数对象
}

// 函数模板：如果 T 不是整数类型，则对地址处的值执行原子加法操作
template <typename T>
typename std::enable_if_t<!std::is_integral_v<T>>
atomic_add(volatile T *addr, T offset) {
  typedef typename AsIntegerType<T>::type alt_type;  // 将 T 映射为整数类型

  static_assert(sizeof(std::atomic<alt_type>) == sizeof(T),
                "std::atomic issue");  // 检查 std::atomic 是否符合预期

  alt_type expected;  // 期望值
  alt_type desired;  // 目标值

  std::atomic<alt_type> *atomic_addr = (std::atomic<alt_type> *)addr;  // 将地址转换为 std::atomic 类型指针
  do {
    T val = fetch_value(addr);  // 获取地址处的值
    reinterpret_cast<T *>(&expected)[0] = val;  // 设置期望值
    reinterpret_cast<T *>(&desired)[0] = val + offset;  // 设置目标值
  } while (!atomic_addr->compare_exchange_weak(expected, desired,
                                               std::memory_order_relaxed));  // 使用弱比较交换操作进行原子更新
}

// 注释：在 C++20 中，float 类型被 fetch_add 支持，但性能可能不如 compare_exchange_weak，可通过 microbenchmark inductor_cpu_atomic.py 进行检查
template <typename T>
typename std::enable_if_t<std::is_integral_v<T>>
// 实现原子加法操作，用于对地址处的值进行原子加操作
// 使用 static_assert 确保 std::atomic<T> 的大小与 T 相同，否则抛出错误信息
atomic_add(volatile T *addr, T offset) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T),
                "std::atomic issue");
  // 将地址转换为 std::atomic<T>* 类型，以便进行原子操作
  std::atomic<T> *atomic_addr = (std::atomic<T> *)addr;
  // 执行原子加操作，使用 std::memory_order_relaxed 内存序
  atomic_addr->fetch_add(offset, std::memory_order_relaxed);
}

// 根据线程数和矩阵维度计算线程的分块数量
void mm_get_thread_blocking(
    int num_threads,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t M0,
    int64_t N0,
    int64_t K0,
    int64_t& Mt,
    int64_t& Nt,
    int64_t& Kt) {
  // Lambda 函数：计算一个数的因子，并返回因子数组和因子个数
  auto get_factors = [](int64_t number) {
    int count = 0;
    // 循环查找小于等于平方根的因子
    for (int64_t i = std::sqrt(number); i > 0; --i) {
      if (number % i == 0) {
        count += 2;
      }
    }
    auto factors = std::make_unique<int64_t[]>(count);
    int index = 0;
    // 再次循环获取所有因子，并存入 factors 数组
    for (int64_t i = std::sqrt(number); i > 0; --i) {
      if (number % i == 0) {
        factors[index++] = number / i;
        factors[index++] = i;
      }
    }
    return std::make_tuple(std::move(factors), count);
  };

  // Lambda 函数：根据因子计算线程的分块数量
  auto get_blocking = [](int64_t num_threads,
                         int64_t factor,
                         int64_t m_blocks,
                         int64_t n_blocks,
                         int64_t k_blocks) {
    // 计算 n 方向的线程块数量
    int64_t thread_block_n = (n_blocks + factor - 1) / factor;
    int64_t cofactor = num_threads / factor;
    // 计算 m 方向的线程块数量
    int64_t thread_block_m = (m_blocks + cofactor - 1) / cofactor;
    // 返回计算得到的线程块数量和 k 方向的块数
    return std::make_tuple(thread_block_m, thread_block_n, k_blocks);
  };

  // 计算 M、N、K 方向上的块数
  int64_t m_blocks = (M + M0 - 1) / M0;
  int64_t n_blocks = (N + N0 - 1) / N0;
  int64_t k_blocks = (K + K0 - 1) / K0;

  // 获取线程数的因子数组及其数量
  auto [factors, count] = get_factors(num_threads);
  // 断言确保因子数量大于 0
  assert(count > 0);

  // 遍历所有因子，寻找合适的因子来划分线程块
  for (int i = 0; i < count; ++i) {
    int64_t factor = factors[i];
    // 如果 n_blocks 能被当前因子整除，并且 m_blocks 能被 (num_threads / factor) 整除
    if (n_blocks % factor == 0 &&
        m_blocks % (num_threads / factor) == 0) {
      // 根据找到的因子计算线程块的分配
      std::tie(Mt, Nt, Kt) = get_blocking(
          num_threads, factor, m_blocks, n_blocks, k_blocks);
      return;
    }
  }

  // 如果没有找到合适的因子，则再次遍历因子数组
  for (int i = 0; i < count; ++i) {
    int64_t factor = factors[i];
    // 如果 n_blocks 能被当前因子整除
    if (n_blocks % factor == 0) {
      // 根据找到的因子计算线程块的分配
      std::tie(Mt, Nt, Kt) = get_blocking(
          num_threads, factor, m_blocks, n_blocks, k_blocks);
      return;
    }
    int64_t cofactor = num_threads / factor;
    // 如果 m_blocks 能被 (num_threads / factor) 整除
    if (m_blocks % cofactor == 0) {
      // 根据找到的因子计算线程块的分配
      std::tie(Mt, Nt, Kt) = get_blocking(
          num_threads, factor, m_blocks, n_blocks, k_blocks);
      return;
    }
  }

  // 如果所有条件都不满足，触发断言错误，程序不应该运行到这里
  assert(false && "Should not reach here.");
  // 返回一个虚假值以避免编译器警告
  return;
}

// 内联函数：根据线程 ID 和线程块的大小获取线程块的起始和结束位置
inline void mm_get_thread_blocks(
    int thread_id,
    int64_t M_blocks,
    int64_t N_blocks,
    int64_t K_blocks,
    int64_t Mt_blocks,
    int64_t Nt_blocks,
    int64_t Kt_blocks,
    int64_t& m_block_start,
    int64_t& m_block_end,
    int64_t& n_block_start,
    int64_t& n_block_end,
    int64_t& k_block_start,
    # 计算每个线程处理的 K 维度块的起始和结束索引
    int64_t num_Kt = (K_blocks + Kt_blocks - 1) / Kt_blocks;
    k_block_start = (thread_id % num_Kt) * Kt_blocks;
    k_block_end = std::min(k_block_start + Kt_blocks, K_blocks);
    
    # 根据线程ID调整为处理 N 维度块的起始和结束索引
    thread_id /= num_Kt;
    int64_t num_Nt = (N_blocks + Nt_blocks - 1) / Nt_blocks;
    n_block_start = (thread_id % num_Nt) * Nt_blocks;
    n_block_end = std::min(n_block_start + Nt_blocks, N_blocks);
    
    # 根据线程ID计算处理 M 维度块的起始和结束索引
    thread_id /= num_Nt;
    m_block_start = std::min(thread_id * Mt_blocks, M_blocks);
    m_block_end = std::min(m_block_start + Mt_blocks, M_blocks);
}

// 结构体定义，表示 AMX 瓦片配置
struct amx_tilecfg {
  uint8_t palette_id;           // 调色板 ID
  uint8_t start_row;            // 起始行
  uint8_t reserved_0[14];       // 保留字段
  uint16_t colsb[16];           // 列偏移数组，最多包含 16 个元素
  uint8_t rows[16];             // 行数组，最多包含 16 个元素
};

// AMXState 类定义
class AMXState {
 private:
  amx_tilecfg tilecfg_;         // AMX 瓦片配置结构体
  uint8_t rows_;                // 行数
  uint16_t colsb_;              // 列偏移
  uint8_t num_tile_rows_;       // 瓦片行数
  uint8_t num_tile_columns_;    // 瓦片列数

 public:
  // 构造函数，初始化各成员变量
  AMXState() : rows_(0), colsb_(0), num_tile_rows_(0), num_tile_columns_(0) {
    memset(&tilecfg_, 0, sizeof(tilecfg_));   // 将 tilecfg_ 清零
  }

  // 配置函数，根据传入参数进行配置
  inline void configure(
      uint8_t rows,
      uint16_t colsb,
      uint8_t num_tile_rows,
      uint8_t num_tile_columns,
      void (*loadconfig)(const amx_tilecfg&)) {
    // 检查当前配置是否与传入参数相同，若相同则直接返回
    if (tilecfg_.palette_id == 1 && rows_ == rows && colsb_ == colsb &&
        num_tile_rows_ == num_tile_rows &&
        num_tile_columns_ == num_tile_columns) {
      return;
    }
    // 更新配置参数
    tilecfg_.palette_id = 1;
    rows_ = rows;
    colsb_ = colsb;
    num_tile_rows_ = num_tile_rows;
    num_tile_columns_ = num_tile_columns;
    const auto num_c_tiles = num_tile_rows * num_tile_columns;
    
    // 配置 C 类型瓦片
    for (int i = 0; i < num_c_tiles; i++) {
      tilecfg_.rows[i] = rows;
      tilecfg_.colsb[i] = 64;
    }
    
    // 配置 A 类型瓦片
    for (int i = 0; i < num_tile_rows; i++) {
      tilecfg_.rows[i + num_c_tiles] = rows;
      tilecfg_.colsb[i + num_c_tiles] = colsb;
    }
    
    // 配置 B 类型瓦片
    for (int i = 0; i < num_tile_columns; i++) {
      tilecfg_.rows[i + num_c_tiles + num_tile_rows] = colsb / 4;
      tilecfg_.colsb[i + num_c_tiles + num_tile_rows] = 64;
    }
    
    // 调用外部加载配置函数，传入更新后的 tilecfg_
    loadconfig(tilecfg_);
  }

  // 释放函数，重置配置并释放资源
  inline void release(void (*tile_release)()) {
    tilecfg_.palette_id = 0;    // 重置调色板 ID
    tile_release();             // 调用释放函数
  }
};
```