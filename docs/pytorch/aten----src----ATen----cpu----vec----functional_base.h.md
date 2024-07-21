# `.\pytorch\aten\src\ATen\cpu\vec\functional_base.h`

```
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec.h>         // 引入向量化操作相关的头文件
#include <c10/util/irange.h>          // 引入用于迭代范围的头文件

namespace at::vec {

// slow path
// 对向量化操作的结果进行归约
template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,                  // 向量化操作的函数对象
    vec::Vectorized<scalar_t> acc_vec,  // 初始的向量化结果
    int64_t size) {                     // 数据的大小
  using Vec = vec::Vectorized<scalar_t>;  // 使用向量化类型 Vec
  scalar_t acc_arr[Vec::size()];        // 创建数组存储向量化结果
  acc_vec.store(acc_arr);               // 将向量化结果存储到数组中
  for (const auto i : c10::irange(1, size)) {  // 对数据进行迭代
    std::array<scalar_t, Vec::size()> acc_arr_next = {0};  // 创建下一个向量化结果数组
    acc_arr_next[0] = acc_arr[i];       // 将当前位置的结果存入下一个数组的第一个位置
    Vec acc_vec_next = Vec::loadu(acc_arr_next.data());  // 加载下一个向量化结果
    acc_vec = vec_fun(acc_vec, acc_vec_next);  // 应用向量化操作函数
  }
  acc_vec.store(acc_arr);               // 将最终向量化结果存储到数组中
  return acc_arr[0];                    // 返回归约后的结果
}

// 结构体模板 VecReduceAllSIMD 的特化，用于 SIMD 加速归约操作
template <typename scalar_t, typename Op>
struct VecReduceAllSIMD {
  static inline scalar_t apply(const Op& vec_fun, const Vectorized<scalar_t>& acc_vec) {
    return vec_reduce_all(vec_fun, acc_vec, Vectorized<scalar_t>::size());  // 调用通用的归约函数
  }
};

// 根据条件编译的不同 SIMD 指令集，执行不同的向量化归约操作
#if defined(__GNUC__) && (__GNUC__ > 5) && !defined(_MSC_VER) && !defined(C10_MOBILE)

#if defined(CPU_CAPABILITY_AVX2)
// AVX2 指令集的特化版本，用于 float 类型的向量化归约
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(const Op& vec_fun, const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;
    // 128-bit shuffle
    Vec v1 = _mm256_permute2f128_ps(v, v, 0x1);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0xB1);
    v = vec_fun(v, v1);
    return _mm256_cvtss_f32(v);  // 将向量化结果转换为 float 类型并返回
  }
};
#endif // defined(CPU_CAPABILITY_AVX2)

#if defined(CPU_CAPABILITY_AVX512)
// AVX512 指令集的特化版本，用于 float 类型的向量化归约
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(const Op& vec_fun, const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;
    // 256-bit shuffle
    Vec v1 = _mm512_shuffle_f32x4(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f32x4(v, v, 0xB1);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0xB1);
    v = vec_fun(v, v1);
    return _mm512_cvtss_f32(v);  // 将向量化结果转换为 float 类型并返回
  }
};
#endif // defined(CPU_CAPABILITY_AVX512)

#endif // defined(__GNUC__) && (__GNUC__ > 5) && !defined(_MSC_VER) && !defined(C10_MOBILE)

// ARM 架构的特化版本，用于 float 类型的向量化归约
#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(const Op& vec_fun, const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;

    // 128-bit shuffle: [a1, a2, a3, a4, a5, a6, a7, a8] -> [a5, a6, a7, a8, a1, a2, a3, a4]
    Vec v1 = {v.get_high(), v.get_low()};
    // [a1+a5, a2+a6, a3+a7, a4+a8, -, -, -, -] ('+' stands for the reduction function. Note that the last 4 elements are not required)
    v = vec_fun(v, v1);
    // 64位元素重排：[a1+a5, a2+a6, a3+a7, a4+a8, -, -, -, -] -> [a3+a7, a4+a8, a1+a5, a2+a6, -, -, -, -]
    float32x4_t v1_1 = vextq_f32(v.get_low(), v.get_low(), 2);
    v1 = {v1_1, v1_1};
    // 使用 v1 对 v 进行某种向量操作
    v = vec_fun(v, v1);

    // 32位元素重排：[a1+a3+a5+a7, a2+a4+a6+a8, a1+a3+a5+a7, a2+a4+a6+a8, -, -, -, -] -> [a2+a4+a6+a8, a1+a3+a5+a7, a2+a4+a6+a8, a1+a3+a5+a7, -, -, -, -]
    v1_1 = vrev64q_f32(v.get_low());
    v1 = {v1_1, v1_1};
    // 使用 v1 对 v 进行某种向量操作
    v = vec_fun(v, v1);

    // 返回向量 v 低位的第一个元素
    return v.get_low()[0];
}
// 结束if预处理指令块，适用于__aarch64__已定义
};
#endif // defined(__aarch64__)

// 模板函数：对向量化的操作应用标准化的向量减少操作
template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(const Op& vec_fun, const Vectorized<scalar_t>& acc_vec) {
  return VecReduceAllSIMD<scalar_t, Op>::apply(vec_fun, acc_vec);
}

// 模板函数：将非减少浮点类型标准化的向量操作减少为单一输出
template <typename scalar_t, typename Op,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t reduce_all(const Op& vec_fun, const scalar_t* data, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  // 如果数据量小于向量大小，则直接进行向量减少操作
  if (size < Vec::size())
    return vec_reduce_all(vec_fun, Vec::loadu(data, size), size);
  int64_t d = Vec::size();
  Vec acc_vec = Vec::loadu(data);
  // 以向量大小为步长循环处理数据
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec = vec_fun(acc_vec, data_vec);
  }
  // 处理剩余的部分数据
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(vec_fun, acc_vec);
}

// 模板函数：类似reduce_all，但将结果减少为两个输出
template <typename scalar_t, typename Op1, typename Op2,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline std::pair<scalar_t, scalar_t> reduce2_all(const Op1& vec_fun1, const Op2& vec_fun2,
    const scalar_t* data, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  // 如果数据量小于向量大小，则分别对两个操作进行向量减少操作
  if (size < Vec::size()) {
    auto loaded_data = Vec::loadu(data, size);
    return std::pair<scalar_t, scalar_t>(
      vec_reduce_all(vec_fun1, loaded_data, size),
      vec_reduce_all(vec_fun2, loaded_data, size));
  }
  int64_t d = Vec::size();
  Vec acc_vec1 = Vec::loadu(data);
  Vec acc_vec2 = Vec::loadu(data);
  // 以向量大小为步长循环处理数据，分别应用两个操作函数
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec1 = vec_fun1(acc_vec1, data_vec);
    acc_vec2 = vec_fun2(acc_vec2, data_vec);
  }
  // 处理剩余的部分数据
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec1 = Vec::set(acc_vec1, vec_fun1(acc_vec1, data_vec), size - d);
    acc_vec2 = Vec::set(acc_vec2, vec_fun2(acc_vec2, data_vec), size - d);
  }
  return std::pair<scalar_t, scalar_t>(
    vec_reduce_all(vec_fun1, acc_vec1),
    vec_reduce_all(vec_fun2, acc_vec2));
}

// 模板函数：将映射和减少操作结合，将非减少浮点类型标准化的向量操作减少为单一输出
template <typename scalar_t, typename MapOp, typename ReduceOp,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  // 如果数据量小于向量大小，则直接进行映射和减少操作
  if (size < Vec::size())
    return vec_reduce_all(red_fun, map_fun(Vec::loadu(data, size)), size);
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data));
  // 以向量大小为步长循环处理数据，先映射再减少
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    data_vec = map_fun(data_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  // 处理剩余的部分数据
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    # 对 data_vec 应用 map_fun 函数，处理数据向量
    data_vec = map_fun(data_vec);

    # 使用 red_fun 函数对 acc_vec 进行更新，根据 data_vec 计算结果，并设置到新的向量 acc_vec 中
    # 使用 Vec::set 函数将计算后的结果设置到 acc_vec 中，保留原向量的大小减去 d 的部分
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  
  # 使用 vec_reduce_all 函数对 acc_vec 执行归约操作，使用 red_fun 函数对所有元素进行归约
  # 返回最终的归约结果
  return vec_reduce_all(red_fun, acc_vec);
# 模板函数：map2_reduce_all，用于对两个数据数组进行映射和归约操作
template <typename scalar_t, typename MapOp, typename ReduceOp,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t map2_reduce_all(
    const MapOp& map_fun,  // 映射操作函数对象
    const ReduceOp& red_fun,  // 归约操作函数对象
    const scalar_t* data,  // 第一个输入数据数组指针
    const scalar_t* data2,  // 第二个输入数据数组指针
    int64_t size) {  // 数据数组的大小

  using Vec = vec::Vectorized<scalar_t>;  // 使用模板化的矢量类型Vec

  // 如果数据数组大小小于矢量化操作的长度，则使用非矢量化方式处理
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);  // 加载第一个数据数组的部分数据到矢量
    Vec data2_vec = Vec::loadu(data2, size);  // 加载第二个数据数组的部分数据到矢量
    data_vec = map_fun(data_vec, data2_vec);  // 对加载的数据进行映射操作
    return vec_reduce_all(red_fun, data_vec, size);  // 对映射结果进行归约并返回结果
  }

  int64_t d = Vec::size();  // 初始化矢量化操作的步长
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2));  // 初始化累加矢量，对第一组数据进行映射操作

  // 对剩余的数据进行矢量化映射和归约操作
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);  // 加载第一个数据数组的一部分数据到矢量
    Vec data2_vec = Vec::loadu(data2 + d);  // 加载第二个数据数组的一部分数据到矢量
    data_vec = map_fun(data_vec, data2_vec);  // 对加载的数据进行映射操作
    acc_vec = red_fun(acc_vec, data_vec);  // 对映射结果进行归约并累加到acc_vec中
  }

  // 处理剩余不足一个完整矢量长度的数据
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);  // 加载第一个数据数组的剩余部分数据到矢量
    Vec data2_vec = Vec::loadu(data2 + d, size - d);  // 加载第二个数据数组的剩余部分数据到矢量
    data_vec = map_fun(data_vec, data2_vec);  // 对加载的数据进行映射操作
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);  // 对映射结果进行归约并设置到acc_vec中
  }

  return vec_reduce_all(red_fun, acc_vec);  // 对最终累加结果进行归约并返回
}

# 模板函数：map3_reduce_all，用于对三个数据数组进行映射和归约操作
template <typename scalar_t, typename MapOp, typename ReduceOp,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t map3_reduce_all(
    const MapOp& map_fun,  // 映射操作函数对象
    const ReduceOp& red_fun,  // 归约操作函数对象
    const scalar_t* data,  // 第一个输入数据数组指针
    const scalar_t* data2,  // 第二个输入数据数组指针
    const scalar_t* data3,  // 第三个输入数据数组指针
    int64_t size) {  // 数据数组的大小

  using Vec = vec::Vectorized<scalar_t>;  // 使用模板化的矢量类型Vec

  // 如果数据数组大小小于矢量化操作的长度，则使用非矢量化方式处理
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);  // 加载第一个数据数组的部分数据到矢量
    Vec data2_vec = Vec::loadu(data2, size);  // 加载第二个数据数组的部分数据到矢量
    Vec data3_vec = Vec::loadu(data3, size);  // 加载第三个数据数组的部分数据到矢量
    data_vec = map_fun(data_vec, data2_vec, data3_vec);  // 对加载的数据进行映射操作
    return vec_reduce_all(red_fun, data_vec, size);  // 对映射结果进行归约并返回结果
  }

  int64_t d = Vec::size();  // 初始化矢量化操作的步长
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2), Vec::loadu(data3));  // 初始化累加矢量，对三组数据进行映射操作

  // 对剩余的数据进行矢量化映射和归约操作
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);  // 加载第一个数据数组的一部分数据到矢量
    Vec data2_vec = Vec::loadu(data2 + d);  // 加载第二个数据数组的一部分数据到矢量
    Vec data3_vec = Vec::loadu(data3 + d);  // 加载第三个数据数组的一部分数据到矢量
    data_vec = map_fun(data_vec, data2_vec, data3_vec);  // 对加载的数据进行映射操作
    acc_vec = red_fun(acc_vec, data_vec);  // 对映射结果进行归约并累加到acc_vec中
  }

  // 处理剩余不足一个完整矢量长度的数据
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);  // 加载第一个数据数组的剩余部分数据到矢量
    Vec data2_vec = Vec::loadu(data2 + d, size - d);  // 加载第二个数据数组的剩余部分数据到矢量
    Vec data3_vec = Vec::loadu(data3 + d, size - d);  // 加载第三个数据数组的剩余部分数据到矢量
    data_vec = map_fun(data_vec, data2_vec, data3_vec);  // 对加载的数据进行映射操作
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);  // 对映射结果进行归约并设置到acc_vec中
  }

  return vec_reduce_all(red_fun, acc_vec);  // 对最终累加结果进行归约并返回
}

# 模板函数：map，用于对单个数据数组进行映射操作
template <typename scalar_t, typename Op,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map(
    const Op& vec_fun,  // 映射操作函数对象
    scalar_t* output_data,  // 输出数据数组指针
    const scalar_t* input_data,  // 输入数据数组指针
    int64_t size) {  // 数据数组的大小

  using Vec = vec::Vectorized<scalar_t>;  // 使用模板化的矢量类型Vec
  int64_t d = 0;  // 初始化迭代器

  // 对数据进行矢量化映射操作
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    # 使用 vec_fun 函数处理从 input_data + d 处加载的数据，并将结果存储到 output_data + d 处
    Vec output_vec = vec_fun(Vec::loadu(input_data + d));
    # 将处理后的向量数据存储到指定的内存位置 output_data + d
    output_vec.store(output_data + d);
  }
  # 如果剩余的数据大小大于 0，则继续处理
  if (size - d > 0) {
    # 使用 vec_fun 函数处理从 input_data + d 处加载的数据（同时指定加载的大小为 size - d），并将结果存储到 output_data + d 处
    Vec output_vec = vec_fun(Vec::loadu(input_data + d, size - d));
    # 将处理后的向量数据存储到指定的内存位置 output_data + d，并指定存储的大小为 size - d
    output_vec.store(output_data + d, size - d);
  }
# 结束当前函数模板的定义
}

# 定义一个模板函数 `map2`，用于将操作 `vec_fun` 应用于两个输入数组，生成一个输出数组
template <typename scalar_t, typename Op,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map2(
    const Op& vec_fun,               # 函数对象，用于操作两个输入向量
    scalar_t* output_data,           # 输出数据数组的指针
    const scalar_t* input_data,      # 第一个输入数据数组的指针
    const scalar_t* input_data2,     # 第二个输入数据数组的指针
    int64_t size) {                  # 数据数组的大小
  using Vec = vec::Vectorized<scalar_t>;   # 使用向量化类型 Vec，根据 scalar_t 类型进行向量化
  int64_t d = 0;                    # 初始化循环变量 d
  for (; d < size - (size % Vec::size()); d += Vec::size()) {   # 循环处理向量化操作
    Vec data_vec = Vec::loadu(input_data + d);   # 加载第一个输入数据向量
    Vec data_vec2 = Vec::loadu(input_data2 + d); # 加载第二个输入数据向量
    Vec output_vec = vec_fun(data_vec, data_vec2);   # 使用 vec_fun 对输入向量执行操作
    output_vec.store(output_data + d);   # 存储操作结果到输出数组中
  }
  if (size - d > 0) {               # 处理剩余的不足一个向量长度的数据
    Vec data_vec = Vec::loadu(input_data + d, size - d);   # 加载剩余部分的第一个输入数据向量
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d); # 加载剩余部分的第二个输入数据向量
    Vec output_vec = vec_fun(data_vec, data_vec2);   # 使用 vec_fun 对剩余部分的输入向量执行操作
    output_vec.store(output_data + d, size - d);    # 存储操作结果到输出数组中
  }
}

# 定义一个模板函数 `map3`，用于将操作 `vec_fun` 应用于三个输入数组，生成一个输出数组
template <typename scalar_t, typename Op,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map3(
    const Op& vec_fun,               # 函数对象，用于操作三个输入向量
    scalar_t* output_data,           # 输出数据数组的指针
    const scalar_t* input_data1,     # 第一个输入数据数组的指针
    const scalar_t* input_data2,     # 第二个输入数据数组的指针
    const scalar_t* input_data3,     # 第三个输入数据数组的指针
    int64_t size) {                  # 数据数组的大小
  using Vec = vec::Vectorized<scalar_t>;   # 使用向量化类型 Vec，根据 scalar_t 类型进行向量化
  int64_t d = 0;                    # 初始化循环变量 d
  for (; d < size - (size % Vec::size()); d += Vec::size()) {   # 循环处理向量化操作
    Vec data_vec1 = Vec::loadu(input_data1 + d);  # 加载第一个输入数据向量
    Vec data_vec2 = Vec::loadu(input_data2 + d);  # 加载第二个输入数据向量
    Vec data_vec3 = Vec::loadu(input_data3 + d);  # 加载第三个输入数据向量
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);   # 使用 vec_fun 对输入向量执行操作
    output_vec.store(output_data + d);   # 存储操作结果到输出数组中
  }
  if (size - d > 0) {               # 处理剩余的不足一个向量长度的数据
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);   # 加载剩余部分的第一个输入数据向量
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);   # 加载剩余部分的第二个输入数据向量
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);   # 加载剩余部分的第三个输入数据向量
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);   # 使用 vec_fun 对剩余部分的输入向量执行操作
    output_vec.store(output_data + d, size - d);    # 存储操作结果到输出数组中
  }
}

# 定义一个模板函数 `map4`，用于将操作 `vec_fun` 应用于四个输入数组，生成一个输出数组
template <typename scalar_t, typename Op,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map4(
    const Op& vec_fun,               # 函数对象，用于操作四个输入向量
    scalar_t* output_data,           # 输出数据数组的指针
    const scalar_t* input_data1,     # 第一个输入数据数组的指针
    const scalar_t* input_data2,     # 第二个输入数据数组的指针
    const scalar_t* input_data3,     # 第三个输入数据数组的指针
    const scalar_t* input_data4,     # 第四个输入数据数组的指针
    int64_t size) {                  # 数据数组的大小
  using Vec = vec::Vectorized<scalar_t>;   # 使用向量化类型 Vec，根据 scalar_t 类型进行向量化
  int64_t d = 0;                    # 初始化循环变量 d
  for (; d < size - (size % Vec::size()); d += Vec::size()) {   # 循环处理向量化操作
    Vec data_vec1 = Vec::loadu(input_data1 + d);  # 加载第一个输入数据向量
    Vec data_vec2 = Vec::loadu(input_data2 + d);  # 加载第二个输入数据向量
    Vec data_vec3 = Vec::loadu(input_data3 + d);  # 加载第三个输入数据向量
    Vec data_vec4 = Vec::loadu(input_data4 + d);  # 加载第四个输入数据向量
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);   # 使用 vec_fun 对输入向量执行操作
    output_vec.store(output_data + d);   # 存储操作结果到输出数组中
  }
  if (size - d > 0) {               # 处理剩余的不足一个向量长度的数据
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);   # 加载剩余部分的第一个输入数据向量
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);   # 加载剩余部分的第二个输入数据向量
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);   # 加载剩余部分的第三个输入数据向量
    Vec data_vec4 = Vec::loadu(input_data4 + d, size - d);   # 加载剩余部分的第四个输入数据向量
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);   # 使用 vec_fun 对剩余部分的输入向量执行操作
    output_vec.store(output_data + d, size - d);    # 存储操作结果到输出数组中
    // 将 output_data + d 处的数据存储到 output_vec 中，存储 size - d 大小的数据
    output_vec.store(output_data + d, size - d);
}

} // namespace at::vec
```