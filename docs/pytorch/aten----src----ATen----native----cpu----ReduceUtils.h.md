# `.\pytorch\aten\src\ATen\native\cpu\ReduceUtils.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/Parallel.h>
// 包含 ATen 库中的并行计算相关头文件
#include <ATen/NumericUtils.h>
// 包含 ATen 库中的数值工具相关头文件
#include <ATen/cpu/vec/vec.h>
// 包含 ATen 库中的矢量化支持相关头文件
#include <ATen/cpu/vec/functional.h>
// 包含 ATen 库中的矢量化函数相关头文件
#include <ATen/native/ReductionType.h>
// 包含 ATen 库中的归约类型相关头文件
#include <c10/util/irange.h>
// 包含 c10 库中的整数范围遍历相关头文件
#include <ATen/OpMathType.h>
// 包含 ATen 库中的操作数数学类型相关头文件
#include <ATen/native/cpu/utils.h>
// 包含 ATen 库中的 CPU 工具函数相关头文件
#include <ATen/OpMathType.h>
// 包含 ATen 库中的操作数数学类型相关头文件

namespace at::native {
inline namespace CPU_CAPABILITY {
// 命名空间 at::native 下的内联命名空间 CPU_CAPABILITY

using namespace vec;
// 使用 vec 命名空间中的内容

#define AT_DISPATCH_REDUCTION_TYPES(op, ...)                                   \
  [&] {                                                                        \
    switch (op) {                                                              \
      case ReductionType::SUM: {                                               \
        static constexpr auto reduce = ReductionType::SUM;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MEAN: {                                              \
        static constexpr auto reduce = ReductionType::MEAN;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MIN: {                                               \
        static constexpr auto reduce = ReductionType::MIN;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MAX: {                                               \
        static constexpr auto reduce = ReductionType::MAX;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::PROD: {                                              \
        static constexpr auto reduce = ReductionType::PROD;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    }                                                                          \
  }()
// 宏定义 AT_DISPATCH_REDUCTION_TYPES(op, ...)，根据归约类型分发操作

template <typename scalar_t, ReductionType reduce>
inline vec_scalar_t<scalar_t> init_value() {
  // 模板函数 init_value，初始化归约操作的起始值

  using acc_t = vec_scalar_t<scalar_t>;
  // 使用 acc_t 作为 vec_scalar_t<scalar_t> 的别名

  acc_t val;
  // 声明 acc_t 类型的变量 val

  if (reduce == ReductionType::SUM ||
      reduce == ReductionType::MEAN) {
    // 如果归约类型为 SUM 或 MEAN
    val = static_cast<acc_t>(0);
    // 初值为 0
  } else if (reduce == ReductionType::PROD) {
    // 如果归约类型为 PROD
    val = static_cast<acc_t>(1);
    // 初值为 1
  } else if (reduce == ReductionType::MAX) {
    // 如果归约类型为 MAX
    val = -std::numeric_limits<acc_t>::infinity();
    // 初值为负无穷大
  } else {
    // 如果归约类型为 MIN
    TORCH_INTERNAL_ASSERT(reduce == ReductionType::MIN);
    // 内部断言确保 reduce 确实为 MIN
    val = std::numeric_limits<acc_t>::infinity();
    // 初值为正无穷大
  }
  return val;
  // 返回计算好的起始值
}

template <typename scalar_t, ReductionType reduce>
// 模板函数定义，接受数据类型和归约类型作为参数
// 返回初始化值，根据可选的初始标量值（initial）或模板参数（reduce）来决定
inline vec_scalar_t<scalar_t> init_value(const std::optional<Scalar>& initial) {
  // 定义累加器类型
  using acc_t = vec_scalar_t<scalar_t>;
  // 如果给定了初始值，将其转换为累加器类型并返回
  if (initial.has_value()) {
    return initial.value().to<acc_t>();
  } else {
    // 否则使用模板参数 reduce 调用另一个 init_value 函数重载
    return init_value<scalar_t, reduce>();
  }
}

// 初始化函数，用指定值填充大小为 size 的数组 out
template <typename scalar_t>
inline void init(scalar_t* out, int64_t size, const vec_scalar_t<scalar_t>& val) {
  // 定义向量化类型 Vec
  using Vec = Vectorized<vec_scalar_t<scalar_t>>;
  // 调用 map 函数，使用 val 初始化数组 out 的每个元素
  map<scalar_t>(
      [val](Vec x) { return Vec(val); },
      out,
      out,
      size);
}

// 使用初始值初始化数组，根据模板参数 reduce 决定初始值的获取方式
template <typename scalar_t, ReductionType reduce>
inline void init(scalar_t* out, int64_t size, const std::optional<Scalar>& initial) {
  // 定义累加器类型
  using acc_t = vec_scalar_t<scalar_t>;
  // 获取初始值
  acc_t val = init_value<scalar_t, reduce>(initial);
  // 使用初始值 val 初始化数组 out
  init(out, size, val);
}

// 重载的 init 函数，用于 scatter_reduce 中的 include_self 参数
template <typename scalar_t, ReductionType reduce>
inline void init(scalar_t* out, int64_t size, bool include_self = false) {
  // 定义累加器类型
  using acc_t = vec_scalar_t<scalar_t>;
  // 如果不包含自身元素，则获取初始值并初始化数组 out
  if (!include_self) {
    acc_t val = init_value<scalar_t, reduce>();
    init(out, size, val);
  }
}

// _init 函数的模板定义，根据 include_self 参数选择初始化方式
template <typename scalar_t, ReductionType reduce>
inline void _init(scalar_t* self_ptr, at::opmath_type<scalar_t>* buffer_ptr, int64_t size, bool include_self) {
  // 如果不包含自身元素，则调用 init 函数初始化 buffer_ptr 指向的数组
  if (!include_self) {
    init<at::opmath_type<scalar_t>, reduce>(buffer_ptr, size, include_self);
  } else {
    // 否则，将 self_ptr 拷贝到 buffer_ptr 指向的数组中
    vec::convert(self_ptr, buffer_ptr, size);
  }
}

// 非 Vec2 类型的标量类型的 _max 函数重载，返回 x 和 y 中的最大值
template <typename scalar_t>
inline typename std::enable_if<!std::is_same<scalar_t, Vec2>::value, scalar_t>::type
_max(const scalar_t& x, const scalar_t& y) {
  // 如果 y 是 NaN，则返回 y，否则返回 x 和 y 中的最大值
  return at::_isnan(y) ? y : std::max(x, y);
}

// 向量化类型 Vectorized<scalar_t> 的 _max 函数重载，使用 vec::maximum 返回 x 和 y 中的最大值
template <typename scalar_t>
inline Vectorized<scalar_t> _max(const Vectorized<scalar_t>& x, const Vectorized<scalar_t>& y) {
  // vec::maximum 会传播 NaN
  return vec::maximum(x, y);
}

// Vec2 类型的 _max 函数重载，返回 x 和 y 中的最大值
template <typename vec_t>
inline typename std::enable_if<std::is_same<vec_t, Vec2>::value, Vec2>::type
_max(const vec_t& x, const vec_t& y) {
  // vec::maximum 会传播 NaN
  return maximum(x, y);
}

// 非 Vec2 类型的标量类型的 _min 函数重载，返回 x 和 y 中的最小值
template <typename scalar_t>
inline typename std::enable_if<!std::is_same<scalar_t, Vec2>::value, scalar_t>::type
_min(const scalar_t& x, const scalar_t& y) {
  // 如果 y 是 NaN，则返回 y，否则返回 x 和 y 中的最小值
  return at::_isnan(y) ? y : std::min(x, y);
}

// 向量化类型 Vectorized<scalar_t> 的 _min 函数重载，使用 vec::minimum 返回 x 和 y 中的最小值
template <typename scalar_t>
inline Vectorized<scalar_t> _min(const Vectorized<scalar_t>& x, const Vectorized<scalar_t>& y) {
  // vec::minimum 会传播 NaN
  return vec::minimum(x, y);
}

// Vec2 类型的 _min 函数重载，返回 x 和 y 中的最小值
template <typename vec_t>
inline typename std::enable_if<std::is_same<vec_t, Vec2>::value, Vec2>::type
_min(const vec_t& x, const vec_t& y) {
  // vec::minimum 会传播 NaN
  return minimum(x, y);
}

// 对于浮点数类型的 map_acc 函数模板定义，使用 Op 函数对象处理向量数据
template <typename scalar_t, typename accumut, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map_acc(
    const Op& vec_fun,
    accumut* output_data,
    const accumut* input_data,
    const scalar_t* input_data2,
    // 循环处理向量化操作，处理给定大小的数据
    int64_t size) {
      // 使用 Vectorized 类型来进行标量类型的向量化计算
      using Vec = vec::Vectorized<scalar_t>;
      // 使用 Vectorized 类型来进行累积类型的向量化计算
      using aVec = vec::Vectorized<accumut>;
      // 初始化变量 d，用于迭代处理数据
      int64_t d = 0;
      // 定义每个向量的大小
      constexpr int64_t kVecSize = Vec::size();
      // 定义累积向量的大小
      constexpr int64_t kaVecSize = aVec::size();
      // 以向量大小 kVecSize 为步长循环处理数据，直到剩余数据不足一个向量大小
      for (d = 0; d < size - (size % kVecSize); d += kVecSize) {
        // 从 input_data2 中加载 kVecSize 大小的数据到 data2_vec 向量
        Vec data2_vec = Vec::loadu(input_data2 + d);
        // 将 data2_vec 向量中的数据转换为浮点数并存储在 data2_avec0 和 data2_avec1 中
        auto [data2_avec0, data2_avec1] = convert_to_float<scalar_t>(data2_vec);
        // 从 input_data 中加载累积向量数据到 input_vec0 和 input_vec1 中
        aVec input_vec0 = aVec::loadu(input_data + d);
        aVec input_vec1 = aVec::loadu(input_data + d + kaVecSize);
        // 使用 vec_fun 函数处理 input_vec0 和 data2_avec0 并将结果存储到 output_data 中
        vec_fun(input_vec0, data2_avec0).store(output_data + d);
        // 使用 vec_fun 函数处理 input_vec1 和 data2_avec1 并将结果存储到 output_data 中
        vec_fun(input_vec1, data2_avec1).store(output_data + d + kaVecSize);
      }
      // 处理剩余不足一个向量大小的数据
      if (size - d > 0) {
        // 计算尾部数据的大小
        int64_t tail_size = size - d;
        // 从 input_data2 中加载 tail_size 大小的数据到 data2_vec 向量
        Vec data2_vec = Vec::loadu(input_data2 + d, tail_size);
        // 将 data2_vec 向量中的数据转换为浮点数并存储在 data2_avec0 和 data2_avec1 中
        auto [data2_avec0, data2_avec1] = convert_to_float<scalar_t>(data2_vec);
        // 如果剩余数据大于累积向量的大小
        if (tail_size > kaVecSize) {
          // 从 input_data 中加载 input_vec0 和 input_vec1 数据
          aVec input_vec0 = aVec::loadu(input_data + d);
          aVec input_vec1 = aVec::loadu(input_data + d + kaVecSize, tail_size - kaVecSize);
          // 使用 vec_fun 处理 input_vec0 和 data2_avec0 并将结果存储到 output_data 中
          vec_fun(input_vec0, data2_avec0).store(output_data + d);
          // 使用 vec_fun 处理 input_vec1 和 data2_avec1 并将结果存储到 output_data 中
          vec_fun(input_vec1, data2_avec1).store(output_data + d + kaVecSize, tail_size - kaVecSize);
        } else {
          // 从 input_data 中加载 tail_size 大小的数据到 input_vec0
          aVec input_vec0 = aVec::loadu(input_data + d, tail_size);
          // 使用 vec_fun 处理 input_vec0 和 data2_avec0 并将结果存储到 output_data 中
          vec_fun(input_vec0, data2_avec0).store(output_data + d, tail_size);
        }
      }
    }
// namespace CPU_CAPABILITY 和 namespace at::native 之间的实现代码段

// for Max and Min, propagate NaN:
// 根据 reduce 参数的不同，执行不同的更新操作，用于计算最大值、最小值、求和、均值或乘积
template <typename T, ReductionType reduce>
inline T update(const T& x, const T& y) {
  if (reduce == ReductionType::SUM ||
      reduce == ReductionType::MEAN) {
    return x + y;  // 求和或均值操作
  } else if (reduce == ReductionType::PROD) {
    return x * y;  // 乘积操作
  } else if (reduce == ReductionType::MAX) {
    return _max(x, y);  // 最大值操作
  } else {
    TORCH_INTERNAL_ASSERT(reduce == ReductionType::MIN);
    return _min(x, y);  // 最小值操作
  }
}

// 根据 reduce 参数的不同，更新输出数组 out，使用输入数据 data 的前 K 个元素
template <typename scalar_t, ReductionType reduce>
inline void update(scalar_t* out, const scalar_t* data, int64_t K) {
  using Vec = vec::Vectorized<vec_scalar_t<scalar_t>>;
  // 使用 map2 函数将 update 函数应用于 Vec 类型的输入向量
  map2<scalar_t>(
      [](Vec x, Vec y) { return update<Vec, reduce>(x, y); },
      out,
      out,
      data,
      K);
}

// 对浮点数类型 scalar_t 进行特化版本的 update 函数，根据 reduce 参数执行不同的更新操作
template <typename scalar_t, ReductionType reduce,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void update(at::opmath_type<scalar_t>* out, const scalar_t* data, int64_t K) {
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<opmath_t>;
  // 使用 map_acc 函数将 update 函数应用于 Vec 类型的输入向量
  map_acc<scalar_t, opmath_t>(
      [](Vec x, Vec y) { return update<Vec, reduce>(x, y); },
      out,
      out,
      data,
      K);
}

// 根据 reduce 参数的不同，更新输出数组 out 的内容，执行均值操作时对输出进行归一化
template <typename scalar_t, ReductionType reduce>
inline void write(scalar_t* out, int64_t count, int64_t K) {
  using Vec = vec::Vectorized<vec_scalar_t<scalar_t>>;
  if (reduce == ReductionType::MEAN) {
    if (count > 0) {
      // 执行均值操作，对输出向量中的每个元素除以 count
      vec::map<scalar_t>(
          [count](Vec x) { return x / Vec(count); },
          out,
          out,
          K);
    }
  }
}
```