# `.\pytorch\aten\src\ATen\native\cpu\Loops.h`

```
#pragma once
// 这个文件提供了两个函数来帮助编写逐元素内核：
//
//   cpu_kernel(TensorIterator iter, <lambda>)
//   cpu_kernel_vec(TensorIterator iter, <lambda>, <vec_lambda>)
//
// 这两个函数可以生成矢量化的代码。cpu_kernel 的实现依赖于编译器的自动矢量化。
// cpu_kernel_vec 的实现在可用时使用 x86 SIMD 指令集。这些函数仅意在 ATen/native/cpu 子目录中使用，
// 因为其他目录中的文件未启用 AVX/AVX2 编译。更多详情请见 README.md。
//
// 例如，要为 float 编写一个乘法内核：
//
//   cpu_kernel(iter, [](float a, float b) { return a * b; });
//
// 或者可以这样写：
//
//   cpu_kernel_vec(iter,
//     [](float a, float b) { return a * b; },
//     [](Vectorized<float> a, Vectorized<float> b) { return a * b; });
//
// 详细实现请参见 BinaryOpsKernel.cpp
//

#include <stdint.h>
#include <c10/util/C++17.h>
#include <c10/util/Load.h>
#include <c10/util/irange.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cpu/IsContiguous.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cpu/vec/vec.h>

#include <utility>

namespace at { namespace native { inline namespace CPU_CAPABILITY {

using namespace vec;

// 模板函数：根据 traits 和索引序列 INDEX 解引用数据数组，返回参数元组
template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_impl(char* C10_RESTRICT data[], const int64_t* strides, int64_t i,
                 std::index_sequence<INDEX...>) {
  return std::make_tuple(
      c10::load<typename traits::template arg<INDEX>::type>(
          data[INDEX] + i * strides[INDEX])...);
}

// 函数：根据 traits 解引用数据数组，返回参数元组
template <typename traits>
typename traits::ArgsTuple
dereference(char* C10_RESTRICT data[], const int64_t* strides, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

// 模板函数：根据 traits 和索引序列 INDEX 解引用数据数组，返回矢量化后的参数元组
template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_vec_impl(char* C10_RESTRICT data[],
                     const typename traits::result_type& opt_scalar,
                     size_t S,
                     int64_t i,
                     std::index_sequence<INDEX...>) {
  using Vec = typename traits::result_type;
  using scalar_t = typename Vec::value_type;
  return std::make_tuple(
      S == INDEX + 1 ?
      opt_scalar :
      Vec::loadu(data[INDEX] + i * sizeof(scalar_t))...);
}

// 函数：根据 traits 解引用数据数组，返回矢量化后的参数元组
template <typename traits>
typename traits::ArgsTuple
dereference_vec(char* C10_RESTRICT data[], const typename traits::result_type& opt_scalar, size_t S, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_vec_impl<traits>(data, opt_scalar, S, i, Indices{});
}

// 函数模板：用于非 void 返回类型的函数 traits，未命名类型指针置空
template <typename func_t,
    typename std::enable_if<!std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
inline void
// 执行操作函数，用于处理具有一组输入和一个输出的操作
// 使用 char* C10_RESTRICT data[] 表示数据数组，const int64_t* strides 表示步长数组
// i 表示当前迭代的起始位置，n 表示迭代的结束位置，op 表示操作的函数对象
template <typename func_t>
void execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  // 获取操作函数 op 的特征 traits
  using traits = function_traits<func_t>;
  // 定义结果类型为操作函数的返回类型
  using result_type = typename traits::result_type;

  // 循环迭代从 i 到 n 的区间
  for (; i < n; i++) {
    // 计算输出指针位置，指向输出数据的位置
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    // 调用 c10::guts::apply 函数，将操作函数 op 应用于输入数据集合
    *out_ptr = c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[1],  // 输入数据数组的起始地址
        &strides[1],  // 输入数据步长数组的起始地址
        i));  // 当前迭代的索引
  }
}

// 特化模板，处理操作函数返回值类型为 void 的情况
// 使用 char* C10_RESTRICT data[] 表示数据数组，const int64_t* strides 表示步长数组
// i 表示当前迭代的起始位置，n 表示迭代的结束位置，op 表示操作的函数对象
template <typename func_t,
    typename std::enable_if<std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
inline void execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  // 获取操作函数 op 的特征 traits
  using traits = function_traits<func_t>;

  // 循环迭代从 i 到 n 的区间
  for (; i < n; i++) {
    // 调用 c10::guts::apply 函数，将操作函数 op 应用于输入数据集合
    c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[0],  // 输入数据数组的起始地址
        &strides[0],  // 输入数据步长数组的起始地址
        i));  // 当前迭代的索引
  }
}

// 基础循环操作函数，支持不同类型的输入和输出数据
// 使用 char* C10_RESTRICT data[] 表示数据数组，const int64_t* strides_ 表示步长数组
// i 表示当前迭代的起始位置，n 表示迭代的结束位置，op 表示操作的函数对象
template <typename func_t>
inline void basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  // 获取操作函数 op 的特征 traits
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  // 创建临时步长数组，用于支持旧版 GCC 中的自动向量化
  int64_t strides[ntensors];
  // 复制步长数据到临时数组中
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  // 调用执行操作函数，处理数据操作
  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

// 用于处理返回元组的递归可变模板
// T 表示元组类型，N 表示元组成员数量
template<class T, size_t N>
struct TupleOutput {
  // 处理元组成员的方法，递归地调用处理
  static void handle(char *C10_RESTRICT data[], const int64_t *strides, int64_t i,
                     const T &tuple) {
    TupleOutput<T, N - 1>::handle(data, strides, i, tuple);

    // 获取当前元组成员
    auto output = std::get<N - 1>(tuple);
    using output_type = decltype(output);
    // 计算输出指针位置，指向输出数据的位置
    output_type * out_ptr = (output_type *)(data[N - 1] + i * strides[N - 1]);
    // 将元组成员的值写入到输出位置
    *out_ptr = output;
  }
};

// 上述递归模板的基础情况，处理元组中只有一个成员的情况
template<class T>
struct TupleOutput<T, 1> {
  // 处理元组中唯一的成员
  static void handle(char *C10_RESTRICT data[], const int64_t *strides, int64_t i,
                     const T &tuple) {
    // 获取元组中的成员
    auto output = std::get<0>(tuple);
    using output_type = decltype(output);
    // 计算输出指针位置，指向输出数据的位置
    output_type* out_ptr = (output_type *)(data[0] + i * strides[0]);
    // 将元组成员的值写入到输出位置
    *out_ptr = output;
  }
};

// 处理元组输出的函数模板，调用 TupleOutput 处理返回元组的所有成员
// 使用 char* C10_RESTRICT data[] 表示数据数组，const int64_t* strides 表示步长数组
// i 表示当前迭代的起始位置，tuple 表示返回的元组对象
template<class... Args>
void handle_tuple_outputs(char* C10_RESTRICT data[],
                          const int64_t* strides,
                          int64_t i,
                          const std::tuple<Args...> &tuple) {
  // 调用 TupleOutput 处理返回的元组成员
  TupleOutput<decltype(tuple), sizeof...(Args)>::handle(data, strides, i, tuple);
}
// 通过`handle_tuple_outputs`函数的元组成员输出张量
template <typename func_t>
inline void
multiple_outputs_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;

  using result_type = typename traits::result_type;
  constexpr int num_outputs = std::tuple_size<result_type>::value;
  constexpr int ntensors = traits::arity + num_outputs;

  // 复制步长到临时数组以帮助在较旧的GCC版本中进行自动向量化
  int64_t strides[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  // 开始主循环，处理每个索引i直到n
  for (; i < n; i++) {
    // 调用op函数，应用于data和strides的解引用结果
    auto output = c10::guts::apply(op, dereference<traits>(
      &data[num_outputs],
      &strides[num_outputs],
      i));
    // 处理输出元组，将结果存入data和strides中
    handle_tuple_outputs(data, strides, i, output);
  }
}

// 显式向量化循环实现。所有输入和输出必须具有相同类型，并且除了一个标量输入（步长为0）外，必须是连续的。
// 标量的位置由参数`S`指示。如果`S`为0，则没有标量输入。
template <typename func_t, typename vec_func_t>
inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  using traits = function_traits<vec_func_t>;
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vectorized<scalar_t>;
  constexpr int ntensors = traits::arity + 1;

  // 复制data_到data数组中，以便在限制条件下使用
  char* C10_RESTRICT data[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    data[arg] = data_[arg];
  }

  // 初始化标量优化，如果S > 0，则将第S个位置的数据视为标量，否则初始化为0
  Vec opt_scalar = Vec(S > 0 ? *(scalar_t*)data[S] : scalar_t(0));
  int64_t i = 0;
  // 主循环，以向量化大小的步长处理数据
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    // 使用dereference_vec解引用数据，并结合标量优化opt_scalar和索引i，i + Vec::size()
    auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
    auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
    // 应用向量化函数vop到args1和args2，然后将结果存储到data[0]中的相应位置
    auto out1 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args1));
    auto out2 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args2));
    out1.store(data[0] + i * sizeof(scalar_t));
    out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
  }
  // 处理剩余的非向量化数据，使用基本循环处理
  if (i < n) {
    int64_t strides[ntensors];
    for (const auto arg : c10::irange(ntensors)) {
      // 如果S > 0且arg == S，则步长为0，否则为标量大小
      strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
    }
    basic_loop(data, strides, i, n, std::forward<func_t>(op));
  }
}

// 对连续标量进行展开检查
template <typename traits, typename cb_t>
inline void unroll_contiguous_scalar_checks(
    const int64_t* /*strides*/,
    std::index_sequence<>,
    cb_t&& cb) {
  // 调用回调函数cb，参数为0
  cb(0);
}

// 递归展开连续标量检查，使用索引序列INDEX0, INDEX...
template <typename traits, typename cb_t, size_t INDEX0, size_t ...INDEX>
inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<INDEX0, INDEX...>,
    cb_t&& cb) {
  // 如果下一个索引INDEX0 + 1处的数据是连续的标量，则调用cb(INDEX0 + 1)
  if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
    cb(INDEX0 + 1);
  } else {
    // 否则继续递归检查下一个INDEX...
    unroll_contiguous_scalar_checks<traits, cb_t, INDEX...>(strides, std::index_sequence<INDEX...>(), std::forward<cb_t>(cb));
  }
}
    // 调用模板函数 unroll_contiguous_scalar_checks，传入参数：
    // - strides: 表示步长的数组
    // - std::index_sequence<INDEX...>{}: 编译时整数序列，用于展开模板参数
    // - std::forward<cb_t>(cb): 转发参数 cb，可能是回调函数或函数对象
    unroll_contiguous_scalar_checks<traits>(strides, std::index_sequence<INDEX...>{}, std::forward<cb_t>(cb));
  }
}

// 结构模板：VectorizedLoop2d，用于处理二维向量化循环
template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
  op_t op;            // 操作函数对象
  vop_t vop;          // 向量化操作函数对象

  // traits 类型定义为 op_t 函数特征
  using traits = function_traits<op_t>;
  // ntensors 常量定义为操作数数量加一
  static constexpr int ntensors = traits::arity + 1;
  // data_t 类型定义为包含 ntensors 个 char* 的数组
  using data_t = std::array<char*, ntensors>;

  // 构造函数，接受 op 和 vop 作为参数
  VectorizedLoop2d(const op_t &op, vop_t vop):
    op(op), vop(std::move(vop)) {}

  // 静态函数：advance，用于将 data 指针数组中的指针按 outer_strides 数组移动
  static void advance(data_t &data, const int64_t *outer_strides) {
    for (const auto arg : c10::irange(data.size())) {
      data[arg] += outer_strides[arg];
    }
  }

  // 操作符重载函数调用运算符，实现二维向量化循环
  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    data_t data;
    std::copy_n(base, ntensors, data.data());
    const int64_t *outer_strides = &strides[ntensors];

    // 如果 strides 是连续的
    if (is_contiguous<traits>(strides)) {
      // 对 size1 次循环，执行向量化循环操作
      for (const auto i C10_UNUSED : c10::irange(size1)) {
        vectorized_loop(data.data(), size0, 0, op, vop);  // 调用向量化循环函数
        advance(data, outer_strides);  // 移动 data 指针数组的指针
      }
    } else {
      // 使用 Indices 定义 traits::arity 次循环索引序列
      using Indices = std::make_index_sequence<traits::arity>;
      // 对 strides 进行非展开的标量检查
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          // 对 size1 次循环，执行向量化循环操作
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            vectorized_loop(data.data(), size0, idx, op, vop);  // 调用向量化循环函数
            advance(data, outer_strides);  // 移动 data 指针数组的指针
          }
        } else {
          // 对 size1 次循环，执行基本循环操作
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            basic_loop(data.data(), strides, 0, size0, op);  // 调用基本循环函数
            advance(data, outer_strides);  // 移动 data 指针数组的指针
          }
        }
      });
    }
  }
};

// 函数模板：make_vectorized_loop2d，创建 VectorizedLoop2d 对象的辅助函数
template <typename op_t, typename vop_t>
VectorizedLoop2d<op_t, vop_t> make_vectorized_loop2d(
    const op_t &op, const vop_t &vop) {
  return VectorizedLoop2d<op_t, vop_t>(op, vop);
}

// 函数模板：cpu_kernel，用于处理 CPU 上的张量迭代器
template <typename func_t>
void cpu_kernel(TensorIteratorBase& iter, func_t&& op, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // 断言：迭代器的输入张量数量等于 op 函数的参数数量
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  // 断言：迭代器的输出张量数量为 1
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // 断言：不需要在 CPU 上进行动态类型转换
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  // 迭代器调用 for_each 函数，执行 op 函数的基本循环
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    // basic_loop 可以处理任意步长的 1D 切片，iter.for_each 仅发送 1D 切片到循环 lambda 函数
    basic_loop(data, strides, 0, n, std::forward<func_t>(op));
  }, grain_size);
  // 转换迭代器的输出张量
  iter.cast_outputs();
}

// 函数模板：gpu_kernel_multiple_outputs，用于编写需要多个输出的逐元素内核
// 它遵循与 cpu_kernel 相似的结构，但使用 multiple_outputs_loop 函数代替 basic_loop 函数
// 目前尚未添加 needs_dynamic_casting 检查，因为传递给 multiple_outputs_loop 的 lambda (func_t)
// 返回 std::tuple 而不是 scalar_t
// 未在 CPU 上实现此检查
// 未来可以扩展`needs_dynamic_casting`来支持`std::tuple`和`thrust::tuple`两种类型。
template <typename func_t>
void cpu_kernel_multiple_outputs(TensorIteratorBase& iter, func_t&& op, int64_t grain_size = at::internal::GRAIN_SIZE) {
  // 使用函数特性traits获取func_t的信息，包括参数数量等
  using traits = function_traits<func_t>;
  // 断言输入张量迭代器的输入数量等于func_t的参数数量
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);

  // 对迭代器中的每个元素执行操作op，并使用multiple_outputs_loop处理多个输出
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    multiple_outputs_loop(data, strides, 0, n, std::forward<func_t>(op));
  }, grain_size);
  // 将输出数据类型进行强制转换（cast）
  iter.cast_outputs();
}

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  // 使用函数特性traits获取func_t的信息，包括参数数量等
  using traits = function_traits<func_t>;
  // 断言输入张量迭代器的输入数量等于func_t的参数数量
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  // 断言输出数量为1
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // CPU上目前不支持动态类型转换，但某些内核（如Fill）可能会显式使用动态类型转换，因此检查是否需要跳过此检查
  if constexpr (check_dynamic_cast) {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  }

  // 对迭代器中的每个元素执行向量化操作op和vop
  iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
  // 将输出数据类型进行强制转换（cast）
  iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIteratorBase& iter, func_t&& op, const Range& range) {
  // 使用函数特性traits获取func_t的信息，包括参数数量等
  using traits = function_traits<func_t>;
  // 判断func_t返回类型是否为void
  constexpr bool result_void = std::is_void<typename traits::result_type>::value;
  // 断言输入张量迭代器的输入数量等于func_t的参数数量，并根据返回类型判断输出数量是否合理
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity &&
                        ((result_void && iter.noutputs() == 0) || (!result_void && iter.noutputs() == 1)));
  // CPU上目前不支持动态类型转换，因此断言不需要动态类型转换
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  // 对迭代器中的每个元素执行序列化操作op
  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    basic_loop(data, strides, 0, n, std::forward<func_t>(op));
  }, range);
  // 将输出数据类型进行强制转换（cast）
  iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIteratorBase& iter, func_t&& op) {
  // 简化版本的序列化操作，覆盖整个张量范围
  cpu_serial_kernel(iter, op, {0, iter.numel()});
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, const Range& range) {
  // 使用函数特性traits获取func_t的信息，包括参数数量等
  using traits = function_traits<func_t>;
  // 断言输入张量迭代器的输入数量等于func_t的参数数量
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  // 断言输出数量为1
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // CPU上目前不支持动态类型转换，因此断言不需要动态类型转换
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  // 对迭代器中的每个元素执行向量化操作op和vop，并在给定范围内进行序列化操作
  iter.serial_for_each(make_vectorized_loop2d(op, vop), range);
  // 将输出数据类型进行强制转换（cast）
  iter.cast_outputs();
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop) {
  // 简化版本的向量化序列化操作，覆盖整个张量范围
  cpu_serial_kernel_vec(iter, op, vop, {0, iter.numel()});
}
```