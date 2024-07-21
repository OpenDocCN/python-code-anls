# `.\pytorch\aten\src\ATen\cpu\vec\functional_bfloat16.h`

```py
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec.h>

namespace at::vec {

// BFloat16 specification
// 定义模板，用于确定不同标量类型的别名
template <typename scalar_t> struct VecScalarType { using type = scalar_t; };
// 对于 BFloat16 类型，使用 float 作为其标量类型的别名
template <> struct VecScalarType<BFloat16> { using type = float; };
// 对于 Half 类型，同样使用 float 作为其标量类型的别名
template <> struct VecScalarType<Half> { using type = float; };

// This is different from at::acc_type since we only need to specialize BFloat16
// 定义类型别名，用于将标量类型转换为向量类型
template <typename scalar_t>
using vec_scalar_t = typename VecScalarType<scalar_t>::type;

// Vector conversion between float and bfloat16/half
// 实现将 BFloat16 向量转换为 float 向量的函数模板特化
template <>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float<BFloat16> (const Vectorized<BFloat16>& a) {
  return convert_bfloat16_float(a);
}

// 实现将 Half 向量转换为 float 向量的函数模板特化
template <>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float<Half> (const Vectorized<Half>& a) {
  return convert_half_float(a);
}

// 实现将 float 向量转换为 BFloat16 向量的函数模板特化
template <>
inline Vectorized<BFloat16> convert_from_float<BFloat16>(const Vectorized<float>& a, const Vectorized<float>& b) {
  return convert_float_bfloat16(a, b);
}

// 实现将 float 向量转换为 Half 向量的函数模板特化
template <>
inline Vectorized<Half> convert_from_float<Half>(const Vectorized<float>& a, const Vectorized<float>& b) {
  return convert_float_half(a, b);
}

// 实现将标量数组加载为 float 向量的函数模板特化
template <>
inline void load_to_float<BFloat16> (const BFloat16 *data, Vectorized<float> &out1, Vectorized<float> &out2) {
  load_fp32_from_bf16(data, out1, out2);
}

// 实现将标量数组加载为 float 向量的函数模板特化
template <>
inline void load_to_float<Half> (const Half *data, Vectorized<float> &out1, Vectorized<float> &out2) {
  load_fp32_from_fp16(data, out1, out2);
}

// 实现将标量加载为 float 向量的函数模板特化
template <>
inline void load_to_float<BFloat16> (const BFloat16 *data, Vectorized<float> &out) {
  load_fp32_from_bf16(data, out);
}

// 实现将标量加载为 float 向量的函数模板特化
template <>
inline void load_to_float<Half> (const Half *data, Vectorized<float> &out) {
  load_fp32_from_fp16(data, out);
}

// Note that we already have specialized member of Vectorized<scalar_t> for BFloat16
// so the following functions would run smoothly:
//   using Vec = Vectorized<BFloat16>;
//   Vec one = Vec(BFloat16(1));
//   vec::map([](Vec x) { return one / (one + x.exp()); }, y_ptr, x_ptr, N);
//

} // namespace at::vec
// 为了仍然需要专门化"functional"吗？
//   如果我们在Vectorized<>级别进行专门化，上述示例将需要3对
//   bf16->fp32/fp32->bf16的转换，分别对应".exp()"、"+"和"/"。
//   如果我们在vec::map<>()级别进行专门化，只需要一对bf16->fp32/fp32->bf16的转换，
//   用于输入和输出的BFloat16向量。
//
// 以下BFloat16功能仅为输入和输出向量进行数据类型转换
// （reduce功能仅将最终标量转换回bf16）。
// 相比于Vectorized<>专门化，
//   1. 性能更好，因为数据类型转换更少；
//   2. 由于立即结果保留在fp32中，舍入误差更少；
//   3. 累积在fp32数据类型上完成。
//
// 如果您计划扩展此文件，请确保在
//   aten/src/ATen/test/vec_test_all_types.cpp 中添加单元测试。
//
template <typename scalar_t, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline float reduce_all(const Op& vec_fun, const scalar_t* data, int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  // 如果数据大小小于bVec::size()，则处理边界情况
  if (size < bVec::size()) {
    // 加载数据到bVec
    bVec data_bvec = bVec::loadu(data, size);
    // 将bVec类型转换为float类型的向量对
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 如果数据大小大于fVec::size()，则对数据进行处理
    if (size > fVec::size()) {
      // 对data_fvec0应用vec_fun操作，并将结果设置回data_fvec0
      data_fvec0 = fVec::set(data_fvec0, vec_fun(data_fvec0, data_fvec1), size - fVec::size());
      // 对fVec类型的数据应用vec_fun，并返回结果
      return vec_reduce_all<float>(vec_fun, data_fvec0, fVec::size());
    } else {
      // 对fVec类型的数据应用vec_fun，并返回结果
      return vec_reduce_all<float>(vec_fun, data_fvec0, size);
    }
  }
  // 如果数据大小不小于bVec::size()，则进行正常处理
  int64_t d = bVec::size();
  // 初始化累加bVec向量
  bVec acc_bvec = bVec::loadu(data);
  // 将累加向量转换为float类型的向量对
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  // 循环处理数据，每次处理bVec::size()大小的数据块
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载数据到bVec
    bVec data_bvec = bVec::loadu(data + d);
    // 将bVec类型转换为float类型的向量对
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 应用vec_fun操作到累加向量的每个分量上
    acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = vec_fun(acc_fvec1, data_fvec1);
  }
  // 处理剩余的数据块，大小为(size - d)
  if (size - d > 0) {
    // 加载数据到bVec
    bVec data_bvec = bVec::loadu(data + d, size - d);
    // 将bVec类型转换为float类型的向量对
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 如果剩余数据块大小大于fVec::size()，则对数据进行处理
    if (size - d > fVec::size()) {
      // 应用vec_fun操作到累加向量的每个分量上
      acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
      // 对acc_fvec1应用vec_fun操作，并将结果设置回acc_fvec1的后部分
      acc_fvec1 = fVec::set(acc_fvec1, vec_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      // 对acc_fvec0应用vec_fun操作，并将结果设置回acc_fvec0的后部分
      acc_fvec0 = fVec::set(acc_fvec0, vec_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  // 应用vec_fun操作到acc_fvec0的每个分量上，并返回结果
  acc_fvec0 = vec_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(vec_fun, acc_fvec0);
}

// 另一个reduce函数的模板，与上述函数类似，但返回一个包含两个float的pair
template <typename scalar_t, typename Op1, typename Op2,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline std::pair<float, float> reduce2_all(const Op1& vec_fun1, const Op2& vec_fun2,
    const scalar_t* data, int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  // 如果数据大小小于bVec::size()，则处理边界情况
  if (size < bVec::size()) {
    // 加载以非对齐方式加载数据并创建 bVec 对象
    bVec data_bvec = bVec::loadu(data, size);
    // 将 bVec 对象转换为浮点数向量 data_fvec0 和 data_fvec1
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 如果 size 大于 fVec::size()
    if (size > fVec::size()) {
      // 创建包含部分数据和计算的 fVec 对象 acc1_fvec
      fVec acc1_fvec = fVec::set(data_fvec0, vec_fun1(data_fvec0, data_fvec1), size - fVec::size());
      // 创建包含部分数据和计算的 fVec 对象 acc2_fvec
      fVec acc2_fvec = fVec::set(data_fvec0, vec_fun2(data_fvec0, data_fvec1), size - fVec::size());
      // 返回两个浮点数标量构成的 pair
      return std::pair<scalar_t, scalar_t>(
          // 对 acc1_fvec 使用 vec_fun1 进行归约操作
          vec_reduce_all<float>(vec_fun1, acc1_fvec, fVec::size()),
          // 对 acc2_fvec 使用 vec_fun2 进行归约操作
          vec_reduce_all<float>(vec_fun2, acc2_fvec, fVec::size()));
    } else {
      // 返回两个浮点数标量构成的 pair，使用 data_fvec0 进行归约操作
      return std::pair<scalar_t, scalar_t>(
          vec_reduce_all<float>(vec_fun1, data_fvec0, size),
          vec_reduce_all<float>(vec_fun2, data_fvec0, size));
    }
  }
  // 初始化 d 为 bVec 的大小
  int64_t d = bVec::size();
  // 加载数据并创建 acc_bvec 对象
  bVec acc_bvec = bVec::loadu(data);
  // 将 acc_bvec 转换为浮点数向量 acc1_fvec0 和 acc1_fvec1
  auto [acc1_fvec0, acc1_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  // 将 acc_bvec 转换为另一组浮点数向量 acc2_fvec0 和 acc2_fvec1
  auto [acc2_fvec0, acc2_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  // 循环处理数据，每次处理一个 bVec 的大小，直到剩余数据小于一个 bVec 的大小
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载数据并创建 data_bvec 对象
    bVec data_bvec = bVec::loadu(data + d);
    // 将 data_bvec 转换为浮点数向量 data_fvec0 和 data_fvec1
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 使用 vec_fun1 更新 acc1_fvec0 和 acc1_fvec1
    acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
    acc1_fvec1 = vec_fun1(acc1_fvec1, data_fvec1);
    // 使用 vec_fun2 更新 acc2_fvec0 和 acc2_fvec1
    acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
    acc2_fvec1 = vec_fun2(acc2_fvec1, data_fvec1);
  }
  // 如果剩余数据大于 0
  if (size - d > 0) {
    // 加载部分数据并创建 data_bvec 对象
    bVec data_bvec = bVec::loadu(data + d, size - d);
    // 将 data_bvec 转换为浮点数向量 data_fvec0 和 data_fvec1
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 如果剩余数据大于 fVec::size()
    if (size - d > fVec::size()) {
      // 使用 vec_fun1 更新 acc1_fvec0
      acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
      // 对 acc1_fvec1 使用 vec_fun1 并设置部分数据
      acc1_fvec1 = fVec::set(acc1_fvec1, vec_fun1(acc1_fvec1, data_fvec1), size - d - fVec::size());
      // 使用 vec_fun2 更新 acc2_fvec0
      acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
      // 对 acc2_fvec1 使用 vec_fun2 并设置部分数据
      acc2_fvec1 = fVec::set(acc2_fvec1, vec_fun2(acc2_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      // 对 acc1_fvec0 使用 vec_fun1 并设置部分数据
      acc1_fvec0 = fVec::set(acc1_fvec0, vec_fun1(acc1_fvec0, data_fvec0), size - d);
      // 对 acc2_fvec0 使用 vec_fun2 并设置部分数据
      acc2_fvec0 = fVec::set(acc2_fvec0, vec_fun2(acc2_fvec0, data_fvec0), size - d);
    }
  }
  // 使用 vec_fun1 和 vec_fun2 对 acc1_fvec0 和 acc2_fvec0 进行最终的归约操作
  acc1_fvec0 = vec_fun1(acc1_fvec0, acc1_fvec1);
  acc2_fvec0 = vec_fun2(acc2_fvec0, acc2_fvec1);
  // 返回两个浮点数标量构成的 pair
  return std::pair<scalar_t, scalar_t>(
      vec_reduce_all<float>(vec_fun1, acc1_fvec0),
      vec_reduce_all<float>(vec_fun2, acc2_fvec0));
// 结束前一个函数模板的定义，开始定义另一个模板函数，用于将浮点数减少映射到一个标量上
template <typename scalar_t, typename MapOp, typename ReduceOp,
          // 启用条件：如果标量类型是浮点数且可以减少
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
// 内联函数：将映射和减少操作应用于数据数组，并返回结果作为 float 类型
inline float map_reduce_all(
    const MapOp& map_fun,              // 映射操作器对象
    const ReduceOp& red_fun,           // 减少操作器对象
    const scalar_t* data,              // 数据数组的指针
    int64_t size) {                    // 数据数组的大小

  using bVec = vec::Vectorized<scalar_t>; // 使用标量类型的向量化类型
  using fVec = vec::Vectorized<float>;    // 使用 float 类型的向量化类型

  // 如果数据量小于向量化类型的大小
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);    // 将部分数据加载到向量中
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec); // 将数据转换为 float 类型
    // 如果数据量大于 float 类型的向量化大小
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0); // 对第一个向量应用映射操作
      data_fvec1 = map_fun(data_fvec1); // 对第二个向量应用映射操作
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size()); // 将减少操作应用到第一个向量并设置结果
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size()); // 对结果向量应用减少操作并返回最终标量结果
    } else {
      data_fvec0 = map_fun(data_fvec0); // 对第一个向量应用映射操作
      return vec_reduce_all<float>(red_fun, data_fvec0, size); // 对结果向量应用减少操作并返回最终标量结果
    }
  }

  int64_t d = bVec::size(); // 计算向量化类型的大小
  bVec acc_bvec = bVec::loadu(data); // 加载第一个向量数据
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec); // 将第一个向量数据转换为 float 类型
  acc_fvec0 = map_fun(acc_fvec0); // 对第一个向量应用映射操作
  acc_fvec1 = map_fun(acc_fvec1); // 对第二个向量应用映射操作

  // 对数据进行向量化处理
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d); // 加载另一个向量的数据
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec); // 将另一个向量的数据转换为 float 类型
    data_fvec0 = map_fun(data_fvec0); // 对第一个向量应用映射操作
    data_fvec1 = map_fun(data_fvec1); // 对第二个向量应用映射操作
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0); // 将减少操作应用到第一个向量的结果
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1); // 将减少操作应用到第二个向量的结果
  }

  // 处理剩余不足一个向量化大小的数据
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d); // 加载剩余数据到向量
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec); // 将剩余数据转换为 float 类型
    // 如果剩余数据大于 float 类型的向量化大小
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0); // 对第一个向量应用映射操作
      data_fvec1 = map_fun(data_fvec1); // 对第二个向量应用映射操作
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0); // 将减少操作应用到第一个向量的结果
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size()); // 将减少操作应用到第二个向量的结果并设置最终值
    } else {
      data_fvec0 = map_fun(data_fvec0); // 对第一个向量应用映射操作
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d); // 将减少操作应用到第一个向量的结果并设置最终值
    }
  }

  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1); // 对最终的向量化结果应用减少操作
  return vec_reduce_all<float>(red_fun, acc_fvec0); // 对最终的结果向量应用减少操作并返回最终标量结果
}
    // 如果计算尺寸大于向量大小
    if (size > fVec::size()) {
      // 对 data_fvec0 和 data2_fvec0 应用 map_fun 函数
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      // 对 data_fvec1 和 data2_fvec1 应用 map_fun 函数
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      // 使用 data_fvec0 计算并设置向量数据，使用 red_fun 函数对 data_fvec0 和 data_fvec1 进行归约
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      // 对结果向量进行全局归约操作，使用 red_fun 函数进行归约
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      // 对 data_fvec0 和 data2_fvec0 应用 map_fun 函数
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      // 对结果向量进行全局归约操作，使用 red_fun 函数进行归约
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  // 初始化向量大小为 bVec 的大小
  int64_t d = bVec::size();
  // 从 data 中加载未对齐的 bVec 数据到 acc_bvec
  bVec acc_bvec = bVec::loadu(data);
  // 将 acc_bvec 转换为浮点数向量 acc_fvec0 和 acc_fvec1
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  // 从 data2 中加载未对齐的 bVec 数据到 acc2_bvec
  bVec acc2_bvec = bVec::loadu(data2);
  // 将 acc2_bvec 转换为浮点数向量 acc2_fvec0 和 acc2_fvec1
  auto [acc2_fvec0, acc2_fvec1] = convert_to_float<scalar_t>(acc2_bvec);
  // 对 acc_fvec0 和 acc2_fvec0 应用 map_fun 函数
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0);
  // 对 acc_fvec1 和 acc2_fvec1 应用 map_fun 函数
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1);
  // 循环处理每个 bVec 大小的数据块，从 d 开始，直到 size - (size % bVec::size())
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 从 data + d 处加载未对齐的 bVec 数据到 data_bvec
    bVec data_bvec = bVec::loadu(data + d);
    // 将 data_bvec 转换为浮点数向量 data_fvec0 和 data_fvec1
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 从 data2 + d 处加载未对齐的 bVec 数据到 data2_bvec
    bVec data2_bvec = bVec::loadu(data2 + d);
    // 将 data2_bvec 转换为浮点数向量 data2_fvec0 和 data2_fvec1
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    // 对 data_fvec0 和 data2_fvec0 应用 map_fun 函数
    data_fvec0 = map_fun(data_fvec0, data2_fvec0);
    // 对 data_fvec1 和 data2_fvec1 应用 map_fun 函数
    data_fvec1 = map_fun(data_fvec1, data2_fvec1);
    // 使用 red_fun 函数对 acc_fvec0 和 data_fvec0 进行归约
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    // 使用 red_fun 函数对 acc_fvec1 和 data_fvec1 进行归约
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  // 处理剩余的数据，从 d 处到 size 大小
  if (size - d > 0) {
    // 从 data + d 处加载未对齐的 bVec 数据块，大小为 size - d
    bVec data_bvec = bVec::loadu(data + d, size - d);
    // 将 data_bvec 转换为浮点数向量 data_fvec0 和 data_fvec1
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 从 data2 + d 处加载未对齐的 bVec 数据块，大小为 size - d
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    // 将 data2_bvec 转换为浮点数向量 data2_fvec0 和 data2_fvec1
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    // 如果剩余大小大于 fVec 的大小
    if (size - d > fVec::size()) {
      // 对 data_fvec0 和 data2_fvec0 应用 map_fun 函数
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      // 对 data_fvec1 和 data2_fvec1 应用 map_fun 函数
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      // 使用 red_fun 函数对 acc_fvec0 和 data_fvec0 进行归约
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      // 使用 red_fun 函数对 acc_fvec1 和 data_fvec1 进行归约，并将结果设置到 acc_fvec1 中
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      // 对 data_fvec0 和 data2_fvec0 应用 map_fun 函数
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      // 使用 red_fun 函数对 acc_fvec0 和 data_fvec0 进行归约，并将结果设置到 acc_fvec0 中
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  // 对 acc_fvec0 和 acc_fvec1 进行最后一次归约操作
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  // 返回结果向量的全局归约结果，使用 red_fun 函数进行归约
  return vec_reduce_all<float>(red_fun, acc_fvec0);
// 模板函数定义，用于映射和归约操作，处理浮点数类型数据
template <typename scalar_t, typename MapOp, typename ReduceOp,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline float map3_reduce_all(
    const MapOp& map_fun,             // 映射操作的函数对象
    const ReduceOp& red_fun,          // 归约操作的函数对象
    const scalar_t* data,             // 第一个数据数组的指针
    const scalar_t* data2,            // 第二个数据数组的指针
    const scalar_t* data3,            // 第三个数据数组的指针
    int64_t size) {                   // 数据数组的大小

  using bVec = vec::Vectorized<scalar_t>;  // 使用模板元编程的向量化类型定义
  using fVec = vec::Vectorized<float>;     // 使用模板元编程的浮点向量化类型定义

  if (size < bVec::size()) {  // 如果数据量小于向量化长度
    bVec data_bvec = bVec::loadu(data, size);          // 加载第一个数据数组
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);  // 转换为浮点型向量
    bVec data2_bvec = bVec::loadu(data2, size);        // 加载第二个数据数组
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);  // 转换为浮点型向量
    bVec data3_bvec = bVec::loadu(data3, size);        // 加载第三个数据数组
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);  // 转换为浮点型向量

    if (size > fVec::size()) {  // 如果数据量大于浮点向量化长度
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);  // 对数据进行映射操作
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);  // 对数据进行映射操作
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());  // 执行归约操作
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());  // 对浮点向量进行全局归约操作
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);  // 对数据进行映射操作
      return vec_reduce_all<float>(red_fun, data_fvec0, size);  // 对浮点向量进行全局归约操作
    }
  }

  // 大数据量情况下的处理
  int64_t d = bVec::size();  // 向量化长度
  bVec acc_bvec = bVec::loadu(data);  // 加载第一个数据数组的部分
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);  // 转换为浮点型向量
  bVec acc2_bvec = bVec::loadu(data2);  // 加载第二个数据数组的部分
  auto [acc2_fvec0, acc2_fvec1] = convert_to_float<scalar_t>(acc2_bvec);  // 转换为浮点型向量
  bVec acc3_bvec = bVec::loadu(data3);  // 加载第三个数据数组的部分
  auto [acc3_fvec0, acc3_fvec1] = convert_to_float<scalar_t>(acc3_bvec);  // 转换为浮点型向量

  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0, acc3_fvec0);  // 对数据进行映射操作
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1, acc3_fvec1);  // 对数据进行映射操作

  // 处理剩余未处理的数据
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);  // 加载未处理数据
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);  // 转换为浮点型向量
    bVec data2_bvec = bVec::loadu(data2 + d);  // 加载未处理数据
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);  // 转换为浮点型向量
    bVec data3_bvec = bVec::loadu(data3 + d);  // 加载未处理数据
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);  // 转换为浮点型向量

    data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);  // 对数据进行映射操作
    data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);  // 对数据进行映射操作

    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);  // 执行归约操作
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);  // 执行归约操作
  }

  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);  // 加载剩余未处理数据
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);  // 转换为浮点型向量
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);  // 加载剩余未处理数据
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);  // 转换为浮点型向量
    bVec data3_bvec = bVec::loadu(data3 + d, size - d);  // 加载剩余未处理数据
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);  // 转换为浮点型向量

    // 对剩余数据进行映射操作和归约操作
    acc_fvec0 = red_fun(acc_fvec0, map_fun(data_fvec0, data2_fvec0, data3_fvec0));
    acc_fvec1 = red_fun(acc_fvec1, map_fun(data_fvec1, data2_fvec1, data3_fvec1));
  }
    # 如果剩余的大小超过了fVec::size()，则执行以下操作
    if (size - d > fVec::size()) {
      # 对data_fvec0, data2_fvec0, data3_fvec0进行映射操作，并将结果赋给data_fvec0
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      # 对data_fvec1, data2_fvec1, data3_fvec1进行映射操作，并将结果赋给data_fvec1
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      # 对acc_fvec0进行归约操作，将结果赋给acc_fvec0
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      # 对acc_fvec1进行归约操作，将结果赋给acc_fvec1，并且在fVec::size()之后的部分
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      # 如果剩余大小不超过fVec::size()，则执行以下操作
      # 对data_fvec0, data2_fvec0, data3_fvec0进行映射操作，并将结果赋给data_fvec0
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      # 对acc_fvec0进行归约操作，将结果赋给acc_fvec0，并且在剩余大小size - d的位置
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  # 对acc_fvec0和acc_fvec1进行最终的归约操作，并将结果赋给acc_fvec0
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  # 对所有float类型数据执行归约操作，并返回结果
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <typename scalar_t, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size) {
  // 使用 Vectorized 类型根据 scalar_t 加载数据向量化操作
  using bVec = vec::Vectorized<scalar_t>;
  // 使用 Vectorized 类型指定 float 类型向量化操作
  using fVec = vec::Vectorized<float>;
  // 初始化循环变量 d
  int64_t d = 0;
  // 处理主循环，处理整数倍 bVec::size() 的数据
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载 bVec 大小的数据到 data_bvec
    bVec data_bvec = bVec::loadu(input_data + d);
    // 将 loaded 数据转换为 float，并存储在 data_fvec0 和 data_fvec1 中
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 对每个数据应用 vec_fun 函数
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    // 将 float 数据转换回 scalar_t，并存储在 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    // 将结果存储到 output_data 中
    output_bvec.store(output_data + d);
  }
  // 处理剩余部分，即不足一个 bVec::size() 的数据
  if (size - d > 0) {
    // 加载剩余数据到 data_bvec
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    // 将 loaded 数据转换为 float，并存储在 data_fvec0 和 data_fvec1 中
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    // 对每个数据应用 vec_fun 函数
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    // 将 float 数据转换回 scalar_t，并存储在 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    // 将结果存储到 output_data 中
    output_bvec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const float* input_data,
    int64_t size) {
  // 使用 Vectorized 类型根据 scalar_t 加载数据向量化操作
  using bVec = vec::Vectorized<scalar_t>;
  // 使用 Vectorized 类型指定 float 类型向量化操作
  using fVec = vec::Vectorized<float>;
  // 初始化循环变量 d
  int64_t d = 0;
  // 处理主循环，处理整数倍 bVec::size() 的数据
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载两个 fVec::size() 大小的数据到 data_fvec0 和 data_fvec1
    fVec data_fvec0 = fVec::loadu(input_data + d);
    fVec data_fvec1 = fVec::loadu(input_data + d + fVec::size());
    // 对每个数据应用 vec_fun 函数
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    // 将 float 数据转换回 scalar_t，并存储在 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    // 将结果存储到 output_data 中
    output_bvec.store(output_data + d);
  }
  // 处理剩余部分，即不足一个 bVec::size() 的数据
  if (size - d > 0) {
    fVec data_fvec0, data_fvec1;
    if (size - d > fVec::size()) {
      // 加载两个 fVec::size() 大小的数据到 data_fvec0 和 data_fvec1
      data_fvec0 = fVec::loadu(input_data + d);
      data_fvec1 = fVec::loadu(input_data + d + fVec::size(), size - d - fVec::size());
    } else {
      // 选择与 bVec::loadu(ptr, size) 行为一致，未初始化 data_fvec1
      data_fvec0 = fVec::loadu(input_data + d, size - d);
    }
    // 对每个数据应用 vec_fun 函数
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    // 将 float 数据转换回 scalar_t，并存储在 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    // 将结果存储到 output_data 中
    output_bvec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  // 使用 Vectorized 类型根据 scalar_t 加载数据向量化操作
  using bVec = vec::Vectorized<scalar_t>;
  // 使用 Vectorized 类型指定 float 类型向量化操作
  using fVec = vec::Vectorized<float>;
  // 初始化循环变量 d
  int64_t d = 0;
  // 处理主循环，处理整数倍 bVec::size() 的数据
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    # 从内存中加载输入数据的一部分为一个位向量对象，并存储到 data_bvec 中
    bVec data_bvec = bVec::loadu(input_data + d);
    # 将加载的位向量数据转换为浮点数向量，存储在 data_fvec0 和 data_fvec1 中
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    # 从内存中加载第二个输入数据的一部分为一个位向量对象，并存储到 data2_bvec 中
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    # 将加载的第二个位向量数据转换为浮点数向量，存储在 data2_fvec0 和 data2_fvec1 中
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    # 对两个浮点数向量应用向量函数 vec_fun，得到 output_fvec0 和 output_fvec1
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    # 将得到的浮点数向量转换回位向量对象，存储到 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    # 将输出位向量对象的数据存储到内存中的输出数据位置，偏移量为 d
    output_bvec.store(output_data + d);
  }
  # 处理剩余不足一整个位向量的部分
  if (size - d > 0) {
    # 从内存中加载输入数据的剩余部分为一个位向量对象，存储到 data_bvec 中
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    # 将加载的位向量数据转换为浮点数向量，存储在 data_fvec0 和 data_fvec1 中
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    # 从内存中加载第二个输入数据的剩余部分为一个位向量对象，存储到 data2_bvec 中
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    # 将加载的第二个位向量数据转换为浮点数向量，存储在 data2_fvec0 和 data2_fvec1 中
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    # 对两个浮点数向量应用向量函数 vec_fun，得到 output_fvec0 和 output_fvec1
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    # 将得到的浮点数向量转换回位向量对象，存储到 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    # 将输出位向量对象的数据存储到内存中的输出数据位置，偏移量为 d，长度为 size - d
    output_bvec.store(output_data + d, size - d);
  }
// 带有类型模板参数 `scalar_t` 和操作器类型模板参数 `Op` 的函数模板 `map3`
// 当 `scalar_t` 是减少浮点类型时，使用 SFINAE 技术进行条件编译，使其生效
template <typename scalar_t, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map3(
    const Op& vec_fun,                      // 操作函数对象的引用参数
    scalar_t* output_data,                  // 输出数据的指针
    const scalar_t* input_data1,            // 输入数据1的指针
    const scalar_t* input_data2,            // 输入数据2的指针
    const scalar_t* input_data3,            // 输入数据3的指针
    int64_t size) {                         // 数据大小

  using bVec = vec::Vectorized<scalar_t>;   // 使用 `scalar_t` 实例化的向量化类型
  using fVec = vec::Vectorized<float>;      // 使用 `float` 实例化的向量化类型
  int64_t d = 0;                            // 迭代器 `d` 初始化为 0
  // 以向量化长度 `bVec::size()` 为步长迭代处理数据
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载 `input_data1` 中偏移为 `d` 的向量数据到 `data1_bvec`
    bVec data1_bvec = bVec::loadu(input_data1 + d);
    // 将 `data1_bvec` 转换为 `float` 类型并分别赋值给 `data1_fvec0` 和 `data1_fvec1`
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    // 同样方式处理 `input_data2` 和 `input_data3`
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    // 调用 `vec_fun` 对向量化的数据进行操作，得到输出的 `float` 向量
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1);
    // 将输出的 `float` 向量转换为 `scalar_t` 类型的向量，并存储到 `output_data` 中偏移为 `d` 的位置
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  // 处理剩余不足一个向量长度的数据
  if (size - d > 0) {
    // 加载剩余数据，并处理方式同上
    bVec data1_bvec = bVec::loadu(input_data1 + d, size - d);
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d, size - d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    // 调用 `vec_fun` 处理剩余部分的数据
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1);
    // 将处理结果存储到 `output_data` 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}
    // 调用 vec_fun 函数计算并返回 output_fvec1 向量，使用 data1_fvec1、data2_fvec1、data3_fvec1、data4_fvec1 作为参数
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1, data4_fvec1);
    
    // 将浮点类型向量 output_fvec0 和 output_fvec1 转换为布尔向量，存储到 output_bvec 中
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    
    // 将 output_bvec 中的数据存储到 output_data + d 处，向量长度为 size - d
    output_bvec.store(output_data + d);
  }
  
  // 如果 size - d 大于 0，则执行以下代码块
  if (size - d > 0) {
    // 从 input_data1 + d 处加载 size - d 长度的布尔向量到 data1_bvec
    bVec data1_bvec = bVec::loadu(input_data1 + d, size - d);
    
    // 将 data1_bvec 布尔向量转换为浮点向量，存储到 data1_fvec0 和 data1_fvec1 中
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    
    // 同上，加载 input_data2 + d 处的布尔向量到 data2_bvec，并转换为浮点向量存储到 data2_fvec0 和 data2_fvec1 中
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    
    // 同上，加载 input_data3 + d 处的布尔向量到 data3_bvec，并转换为浮点向量存储到 data3_fvec0 和 data3_fvec1 中
    bVec data3_bvec = bVec::loadu(input_data3 + d, size - d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    
    // 同上，加载 input_data4 + d 处的布尔向量到 data4_bvec，并转换为浮点向量存储到 data4_fvec0 和 data4_fvec1 中
    bVec data4_bvec = bVec::loadu(input_data4 + d, size - d);
    auto [data4_fvec0, data4_fvec1] = convert_to_float<scalar_t>(data4_bvec);
    
    // 调用 vec_fun 函数计算 output_fvec0 和 output_fvec1 浮点向量
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0, data4_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1, data4_fvec1);
    
    // 将 output_fvec0 和 output_fvec1 转换为布尔向量，存储到 output_data + d 处，向量长度为 size - d
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

} // namespace at::vec
```