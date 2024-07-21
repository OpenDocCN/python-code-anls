# `.\pytorch\aten\src\ATen\native\cpu\Reduce.h`

```py
#pragma once

#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>
#include <c10/util/TypeList.h>
#include <c10/core/Scalar.h>
#include <c10/util/irange.h>

#include <sstream>
#include <type_traits>

// 声明命名空间 at::native::CPU_CAPABILITY
namespace at { namespace native { inline namespace CPU_CAPABILITY {

// 使用命名空间 vec
using namespace vec;

// 定义宏 VEC_LOOP_HEADER，设置函数类型和数据参数
#define VEC_LOOP_HEADER(func_t, data) \
  // 定义标量类型为 func_t 函数的返回类型
  using scalar_t = typename function_traits<func_t>::result_type; \
  // 使用 Vectorized 类模板，其中标量类型为 scalar_t
  using Vec = Vectorized<scalar_t>; \
  // 指向输出数据的指针，从 data[0] 中获取
  char* out_ptr = data[0]; \
  // 使用 (void) out_ptr; 防止编译器警告，这里实际没有其他操作

// 判断是否在维度 0 上连续进行的规约操作
template <typename traits>
inline bool is_contiguous_reduction(const int64_t* strides) {
  // 如果 strides[0] 等于 0，并且 strides[1] 等于 traits::arg2_t 的大小，返回 true
  return strides[0] == 0 &&
         strides[1] == sizeof(typename traits::arg2_t);
}

// 判断是否在维度 1 上外部进行的规约操作
template <typename traits>
inline bool is_outer_reduction(const int64_t* strides) {
  // 如果 strides[0] 等于 0，并且 strides[2] 等于 traits::result_type 的大小，
  // 并且 strides[3] 等于 traits::arg2_t 的大小，返回 true
  return strides[0] == 0 &&
         strides[2] == sizeof(typename traits::result_type) &&
         strides[3] == sizeof(typename traits::arg2_t);
}

// 向量化的规约函数，处理输入数据，进行向量化操作
template <typename func_t, typename vec_func_t>
inline void vectorized_reduction(char** data, int64_t n, int64_t stride,
                                        func_t op, vec_func_t vop, bool reduce) {
  // 调用宏 VEC_LOOP_HEADER，设置标量类型和输出指针
  VEC_LOOP_HEADER(func_t, data)
  // 从 data[1] 中获取输入数据指针 in1_ptr
  const char* in1_ptr = data[1];
  // 声明 Vec 类型的数组 acc，大小为 4
  Vec acc[4];
  // 遍历范围为 0 到 3，加载数据到 acc 数组
  for (const auto j : c10::irange(4)) {
    acc[j] = Vec::loadu(in1_ptr + j * Vec::size() * sizeof(scalar_t));
  }
  // 遍历范围为 1 到 n-1，处理每个输入数据
  for (const auto i : c10::irange(1, n)) {
    // 计算当前输入数据指针 ptr
    const char* ptr = in1_ptr + stride * i;
    // 使用 vop 函数对 acc 数组进行向量化操作
    acc[0] = vop(acc[0], Vec::loadu(ptr + (0 * Vec::size() * sizeof(scalar_t))));
    acc[1] = vop(acc[1], Vec::loadu(ptr + (1 * Vec::size() * sizeof(scalar_t))));
    acc[2] = vop(acc[2], Vec::loadu(ptr + (2 * Vec::size() * sizeof(scalar_t))));
    acc[3] = vop(acc[3], Vec::loadu(ptr + (3 * Vec::size() * sizeof(scalar_t))));
  }
  // 如果 reduce 为 true，则进行最终规约操作
  if (reduce) {
    // 声明标量类型的 buffer 数组，大小为 Vec::size()
    scalar_t buffer[Vec::size()];
    // 将 acc 数组的结果进行向量化操作
    acc[0] = vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]));
    // 将结果存储到 buffer 中
    acc[0].store(buffer);
    // 迭代处理 buffer 数组，使用 op 函数进行最终规约
    for (const auto j : c10::irange(1, Vec::size())) {
      buffer[0] = op(buffer[0], buffer[j]);
    }
    // 将最终结果存储到 out_ptr 指向的位置
    auto dst = (scalar_t*)out_ptr;
    *dst = op(*dst, buffer[0]);
  } else {
    // 否则，将向量化处理后的结果存储到 out_ptr 指向的位置
    for (const auto j : c10::irange(4)) {
      auto dst = out_ptr + j * Vec::size() * sizeof(scalar_t);
      acc[j] = vop(acc[j], Vec::loadu(dst));
      acc[j].store(dst);
    }
  }
}

// 外部循环的一元操作函数，处理数据和步长，执行 n 次操作
template <typename F>
inline void UNARY_OUTER_LOOP(char* data[2], const int64_t strides[2], int64_t n, F f) {
  // 遍历范围为 0 到 n-1，执行循环体中的函数 f
  for (const auto j C10_UNUSED : c10::irange(n)) {
    f();
    // 更新 data[0] 和 data[1] 的指针位置，分别增加 strides[0] 和 strides[1]
    data[0] += strides[0];
    data[1] += strides[1];
  }
}

// 计算规约操作 out = op(out, in) 的内部向量化函数
template <typename func_t, typename vec_func_t>
inline void vectorized_inner_reduction(char** data, int64_t n, func_t op, vec_func_t vop) {
  // 调用宏 VEC_LOOP_HEADER，设置标量类型和输出指针
  VEC_LOOP_HEADER(func_t, data)
  // 计算向量的步长，以及处理的向量数量
  int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
  int64_t count = n / (4 * Vec::size());
  // 如果 count 大于 0，则进行向量化内部规约操作
  if (count > 0) {
    # 调用向量化的归约函数，对数据进行归约操作
    vectorized_reduction(data, count, vector_stride, op, vop, /*reduce=*/true);
  }
  # 创建一个指向字符指针的数组，包含三个指针，分别指向 data[0], data[0], data[1]
  char* ptrs[3] = { data[0], data[0], data[1] };
  # 创建一个整型数组 strides，包含三个元素，分别为 0, 0, sizeof(scalar_t)
  int64_t strides[] = { 0, 0, sizeof(scalar_t) };
  # 调用基本循环函数，对 ptrs 数组中的指针进行基本循环操作
  basic_loop(ptrs, strides, count * 4 * Vec::size(), n, op);
}

// 结束一个函数或者结构体定义的语句块

// computes the reduction out = op(out, in)
template <typename func_t, typename vec_func_t>
inline void vectorized_outer_reduction(char** data, int64_t inner_stride, int64_t size0, int64_t size1, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(func_t, data)

  // reduce down each column of 4 * Vec::size() elements (128 or 256 bytes)
#if defined(CPU_CAPABILITY_AVX512)
  int64_t outer_stride[2] = { 256, 256 };
#else
  int64_t outer_stride[2] = { 128, 128 };
#endif
  UNARY_OUTER_LOOP(data, outer_stride, size1 / (4 * Vec::size()), [&] {
    // 对每列包含 4 * Vec::size() 元素的数据进行向下归约
    vectorized_reduction(data, size0, inner_stride, op, vop, /*reduce=*/false);
  });

  // reduce down the remaining columns
  int64_t step[] = { sizeof(scalar_t), sizeof(scalar_t) };
  int64_t remaining = size1 % (4 * Vec::size());
  UNARY_OUTER_LOOP(data, step, remaining, [&] {
    // 对剩余的列进行向下归约
    char* ptrs[3] = { data[0], data[0], data[1] };
    int64_t strides[] = { 0, 0, inner_stride };
    basic_loop(ptrs, strides, 0, size0, op);
  });
}

template<typename traits, typename res_t>
static void set_result(const int index, const res_t result, const TensorIteratorBase &iter, const int num_outputs) {
  // static_assert(std::is_same<res_t, typename traits::arg2_t>::value, "data types must match");
  if (index < num_outputs) {
    // 设置计算结果到输出张量的指定位置
    char *out = (char *) iter.data_ptr(index);
    *(res_t *) out = result;
  }
}

template<typename traits, typename res_t>
static void set_results(const res_t result, const TensorIteratorBase &iter, const int num_outputs) {
  AT_ASSERT(num_outputs == 1);
  set_result<traits>(0, result, iter, num_outputs);
}

template<typename traits, std::size_t i = 0, typename... tuple_t>
inline typename std::enable_if<i == sizeof...(tuple_t), std::size_t>::type
for_each_in_tuple(const std::tuple<tuple_t...>& /*t*/, const TensorIteratorBase& /*iter*/, const int /*num_outputs*/) {
  // 对元组中的每个元素进行循环处理，直到处理完所有元素
  return i;
}

template<typename traits, std::size_t i = 0, typename... tuple_t>
inline typename std::enable_if<i < sizeof...(tuple_t), std::size_t>::type
for_each_in_tuple(const std::tuple<tuple_t...>& t, const TensorIteratorBase &iter, const int num_outputs) {
  if (i < (size_t)num_outputs) {
    // 对元组中的每个元素设置结果到输出张量的指定位置
    set_result<traits>(i, std::get<i>(t), iter, num_outputs);
    return for_each_in_tuple<traits, i + 1, tuple_t...>(t, iter, num_outputs);
  }
  return i;
}

template<typename traits, typename... res_t>
static void set_results(const std::tuple<res_t...>& result, const TensorIteratorBase &iter, const int num_outputs) {
  AT_ASSERT(num_outputs >= 1);
  // 将元组中的结果设置到输出张量的指定位置
  std::size_t result_size = for_each_in_tuple<traits>(result, iter, num_outputs);
  AT_ASSERT((size_t)num_outputs == result_size);
}

template <typename T, typename... Args>
struct all_same : std::conjunction<
  std::is_same<T, Args>...
> {};

// data_t is the input/output data type.
// acc_t is a type that contains all the necessary data
// to continue reducing.
// index_t is a one-dimensional index
//
// ops_t is such that &ops_t::reduce, &ops_t::combine, and &ops_t::project exist and satisfy
// `binary_kernel_reduce` function template definition, which performs reduction operations on a tensor iterator.
template <typename ops_t, typename init_t>
void binary_kernel_reduce(TensorIteratorBase& iter, ops_t ops, init_t init) {
  // Type aliases for the function pointers `ops_t::reduce`, `ops_t::combine`, and `ops_t::project`
  using rf_t = decltype(&ops_t::reduce);
  using cf_t = decltype(&ops_t::combine);
  using pf_t = decltype(&ops_t::project);

  // Type traits for the function types involved in reduction
  using r_traits = binary_function_traits<rf_t>;
  using c_traits = binary_function_traits<cf_t>;
  using p_traits = unary_function_traits<pf_t>;

  // Type definitions for accumulator type (`acc_t`) and data type (`data_t`) based on function traits
  using acc_t = typename p_traits::arg1_t;
  using data_t = typename r_traits::arg2_t;

  // Static assertions to ensure consistency in types across operations
  static_assert(
    all_same<
      acc_t,
      init_t,
      typename r_traits::arg1_t,
      typename r_traits::result_type,
      typename c_traits::arg1_t,
      typename c_traits::arg2_t,
      typename c_traits::result_type>::value,
    "all accumulate types must match");
  static_assert(
    std::is_default_constructible<acc_t>::value,
    "the accumulate type must be default-constructible"
  );

  // Determine the number of output elements from the iterator
  const int num_outputs = iter.noutputs();

  // Iterate over each reduced element in the iterator
  iter.foreach_reduced_elt([&ops, &init, num_outputs](TensorIteratorBase &sub_iter) {
    // Lambda function that defines the reduction process over a subrange
    auto reduction_body = [&ops, &sub_iter, num_outputs](acc_t acc, int64_t begin, int64_t end) -> acc_t {
      // Number of tensors in the sub iterator
      int ntensors = sub_iter.ntensors();

      // Serial loop over the subrange of the iterator
      sub_iter.serial_for_each([&acc, &ops, num_outputs, ntensors, begin](char** data, const int64_t* strides, int64_t size) {
        // Assertion to check tensor count consistency
        AT_ASSERT(ntensors - num_outputs == 1);

        // Pointer to input data
        char *in = data[ntensors - 1];

        // Stride of the input tensor
        int64_t stride = strides[ntensors - 1];

        // Loop over the size of the subrange
        for (const auto i : c10::irange(size)) {
          // Perform reduction operation on the accumulator with loaded data element
          acc = ops.reduce(acc, c10::load<data_t>(in), begin + i);
          // Move to the next element in the input tensor
          in += stride;
        }
      }, {begin, end});

      // Translate the index of the accumulator using view offsets and return
      return ops.translate_idx(acc, sub_iter.view_offsets()[0]);
    };

    // Initialize the accumulator with the initial value (`init`)
    acc_t total_acc = init;

    // Get the number of elements in the subrange
    auto numel = sub_iter.numel();

    // Condition to determine whether to perform parallel reduction
    if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
        at::in_parallel_region()) {
      // Perform reduction sequentially if conditions are met
      total_acc = reduction_body(total_acc, 0, numel);
    }
    } else {
      // 获取当前可用的最大线程数
      int max_threads = at::get_num_threads();
      // 断言最大线程数大于0，确保线程数合理
      AT_ASSERT(max_threads > 0);
      // 静态断言，确保模板参数acc_t不是bool类型，因为并发修改std::vector<bool>的不同引用是未定义行为
      static_assert(
        !std::is_same<acc_t, bool>::value,
        "Concurrently modifying different references into std::vector<bool> is UB."
      );
      // 创建一个大小为max_threads的std::vector，每个元素初始化为init
      std::vector<acc_t> buffer((unsigned)max_threads, init);
      // 使用并行for循环，将reduction_body函数应用于数据区间，并将结果存储在buffer中
      at::parallel_for(0, numel, internal::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          // 获取当前线程的buffer引用，并更新其中的值
          auto& acc = buffer[at::get_thread_num()];
          acc = reduction_body(acc, begin, end);
        }
      );
      // 将所有buffer中的值使用ops.combine合并到total_acc中
      for (const auto i : c10::irange(max_threads)) {
        total_acc = ops.combine(total_acc, buffer[i]);
      }
    }
    // 将结果投影到类型r_traits，并设置到sub_iter和num_outputs中
    set_results<r_traits>(ops.project(total_acc), sub_iter, num_outputs);
  });


这段代码主要涉及并行计算和结果处理：

1. 获取当前可用的最大线程数并进行断言验证。
2. 创建一个存储线程结果的缓冲区，并使用并行for循环在每个线程上执行reduction_body函数。
3. 将各个线程的计算结果合并到总计数器(total_acc)中。
4. 最后，使用ops对总计数器中的值进行投影并设置到指定位置(sub_iter和num_outputs)。
// 结束函数 binary_kernel_reduce_lastdim，其功能是在指定迭代器上执行最内层维度的二元约简操作
template <typename reduce_func_t>
void binary_kernel_reduce_lastdim(TensorIteratorBase& iter, reduce_func_t reduce_op) {
  auto shape = iter.shape(); // 获取迭代器的形状信息
  int64_t dim_size = shape[0]; // 获取最内层维度的大小
  int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / dim_size); // 计算并行执行的粒度

  TensorIterator sub_iter(iter); // 创建一个子迭代器来并行处理所有非约简维度
  sub_iter.narrow(0, 0, 1); // 缩小迭代器的维度范围为最内层维度

  // 定义并行处理函数，对输入数据进行约简操作
  auto loop = [&](char** data, const int64_t* strides, int64_t size) {
    char* out = data[0]; // 输出数据的指针
    char* in = data[1]; // 输入数据的指针
    for (int64_t i = 0; i < size; ++i) { // 迭代处理每个元素
      reduce_op(out, in, dim_size); // 调用约简操作函数
      out += strides[0]; // 更新输出数据指针
      in += strides[1]; // 更新输入数据指针
    }
  };

  // 使用指定的粒度执行子迭代器上的循环处理
  sub_iter.for_each(loop, grain_size);
}

}}}  // namespace at::native::<anonymous>
```