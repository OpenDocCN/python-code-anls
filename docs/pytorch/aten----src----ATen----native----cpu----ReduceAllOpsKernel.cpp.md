# `.\pytorch\aten\src\ATen\native\cpu\ReduceAllOpsKernel.cpp`

```
// 定义宏以仅支持方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入头文件，包括张量操作、减少操作、工具函数等
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>

// 引入调度和并行处理相关的头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/OpMathType.h>

// 引入CPU相关的循环和数学运算头文件
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

// 定义命名空间 at::native
namespace at::native {
namespace {

// 使用向量化命名空间 vec
using namespace vec;

// 模板函数：使用向量化操作进行全部减少操作
template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl_vec(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op,
    vec_func_t vop) {
  // 获取输入张量的元素个数
  const int64_t input_numel = input.numel();
  // 获取输入张量数据的指针
  auto input_data = input.const_data_ptr<scalar_t>();

  // 使用并行减少函数 parallel_reduce 进行操作
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t /*ident*/) -> scalar_t {
      // 在指定范围内执行向量化的全部减少操作
      scalar_t partial_out = vec::reduce_all<scalar_t>(
        [=](Vectorized<opmath_type<scalar_t>> x, Vectorized<opmath_type<scalar_t>> y) {
          return vop(x, y);
        },
        input_data + start,
        end - start);
      return partial_out;
    }, op);
  
  // 将结果填充到输出张量中
  output.fill_(result);
}

// 模板函数：普通的全部减少操作，不使用向量化
template <typename scalar_t, typename func_t>
inline void reduce_all_impl(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op) {
  // 获取输入张量的元素个数
  const int64_t input_numel = input.numel();
  // 获取输入张量数据的指针
  auto input_data = input.const_data_ptr<scalar_t>();

  // 使用并行减少函数 parallel_reduce 进行操作
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t ident) -> scalar_t {
      // 初始化局部结果为标识值
      scalar_t partial_out = ident;
      // 在指定范围内执行普通的全部减少操作
      for (const auto i : c10::irange(start, end)) {
         partial_out = op(partial_out, input_data[i]);
      }
      return partial_out;
    }, op);
  
  // 将结果填充到输出张量中
  output.fill_(result);
}

// 静态函数：最小值全部减少的具体实现
static void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    // 如果输入张量为布尔类型，使用串行 CPU 内核进行最小值全部减少
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    bool result_data = true;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data && a;
    });
    result.fill_(result_data);
  } else if(input.scalar_type() == ScalarType::Long) {
    // 对于 int64_t 类型，由于向量化实现性能问题，使用标量路径执行全部减少操作
    reduce_all_impl<int64_t>(result, input, upper_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return min_impl(a, b); });
  } else {


这段代码主要是针对张量的减少操作进行了一些实现，包括向量化的减少和普通的减少操作。
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "min_all", [&] {
      // 根据输入张量的数据类型分发处理，包括kHalf和kBFloat16类型
      using Vec = Vectorized<opmath_type<scalar_t>>;
      // 定义Vectorized类型Vec，用于处理当前标量类型的向量化操作
      reduce_all_impl_vec<scalar_t>(result, input, upper_bound<scalar_t>(),
        // 调用reduce_all_impl_vec函数，对输入张量input进行归约操作，结果存储在result中
        [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
        // 定义lambda函数，实现标量类型的最小值比较操作
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
      // 定义lambda函数，实现Vec类型的最小值比较操作
    });
  }
}

// 定义静态函数 max_all_kernel_impl，计算输入张量中的全局最大值
static void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  // 检查输入张量的数据类型是否为布尔类型
  if (input.scalar_type() == ScalarType::Bool) {
    // 创建张量迭代器，遍历输入张量
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    // 初始化结果数据为 false
    bool result_data  = false;
    // 使用串行 CPU 核函数，将每个元素与当前结果数据进行逻辑或操作
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data || a;
    });
    // 将结果数据填充到输出张量中
    result.fill_(result_data);
  } else if (input.scalar_type() == ScalarType::Long) {
    // 如果数据类型为 int64_t，由于向量化实现性能问题，使用标量路径
    reduce_all_impl<int64_t>(result, input, lower_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return max_impl(a, b); });
  } else {
    // 对于除布尔和 int64_t 外的所有数据类型，使用 AT_DISPATCH_ALL_TYPES_AND2 宏展开
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "max_all", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      // 使用向量化方式实现全局最大值计算，调用 reduce_all_impl_vec 函数
      reduce_all_impl_vec<scalar_t>(result, input, lower_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return max_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); });
    });
  }
}

// 用于不支持 AVX/AVX2 的操作的模板函数
template <typename scalar_t, typename func_t1, typename func_t2>
inline void reduce_all_impl_two_outputs(
    Tensor& output1,
    Tensor& output2,
    const Tensor& input,
    const std::pair<scalar_t, scalar_t>& ident_v,
    func_t1 reduce_chunk_func,
    func_t2 reduce_acc_func) {
  using scalar_t_pair = std::pair<scalar_t, scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  // 使用 parallel_reduce 函数并行执行归约操作，处理输入数据
  scalar_t_pair result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t_pair& ident) -> scalar_t_pair {
      scalar_t_pair partial_out(ident);
      // 对输入数据的指定范围内进行归约操作
      for (const auto i : c10::irange(start, end)) {
         partial_out = reduce_chunk_func(partial_out, input_data[i]);
      }
      return partial_out;
    },
    reduce_acc_func
  );
  // 将归约结果填充到输出张量1和输出张量2中
  output1.fill_(result.first);
  output2.fill_(result.second);
}

// 用于支持向量化的两个输出的模板函数
template <typename scalar_t, typename func_t, typename vec_func_t1, typename vec_func_t2>
inline void reduce_all_impl_vec_two_outputs(
    Tensor& output1,
    Tensor& output2,
    const Tensor& input,
    const std::pair<scalar_t, scalar_t>& ident_v,
    func_t reduce_acc_func,
    vec_func_t1 reduce_chunk_func1,
    vec_func_t2 reduce_chunk_func2) {
  using Vec = Vectorized<opmath_type<scalar_t>>;
  using scalar_t_pair = std::pair<scalar_t, scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  // 注意: parallel_reduce 不支持布尔类型
  // 使用 parallel_reduce 函数并行执行归约操作，处理输入数据
  std::pair<scalar_t, scalar_t> result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t_pair& /* ident */) -> scalar_t_pair {
      scalar_t_pair partial_out(ident_v);
      // 对输入数据的指定范围内进行向量化归约操作
      for (const auto i : c10::irange(start, end)) {
         partial_out.first = reduce_chunk_func1(partial_out.first, input_data[i]);
         partial_out.second = reduce_chunk_func2(partial_out.second, input_data[i]);
      }
      return partial_out;
    },
    reduce_acc_func
  );
  // 将归约结果填充到输出张量1和输出张量2中
  output1.fill_(result.first);
  output2.fill_(result.second);
}
    scalar_t_pair partial_out = vec::reduce2_all<scalar_t>(
        [=](Vec x, Vec y) { return reduce_chunk_func1(x, y); },
        [=](Vec x, Vec y) { return reduce_chunk_func2(x, y); },
        input_data + start,
        end - start);

定义一个变量 `partial_out`，调用 `vec::reduce2_all<scalar_t>` 方法，对输入数据进行归约操作，返回一个包含两个元素的结构体 `scalar_t_pair`。使用了两个 lambda 函数作为参数，分别执行 `reduce_chunk_func1` 和 `reduce_chunk_func2` 函数来处理输入数据的向量块。


      return partial_out;
    },
    reduce_acc_func
  );

将 `partial_out` 作为参数传递给另一个函数，并返回其结果。这里的代码片段似乎与前一个语句存在语法错误或不完整的结构。


  output1.fill_(result.first);
  output2.fill_(result.second);

填充 `output1` 和 `output2`，分别使用 `result` 结构体的第一个和第二个成员。通常情况下，这些操作将 `partial_out` 的结果分配给输出张量的两个部分，以便后续处理或进一步使用。
} // namespace

static void aminmax_allreduce_kernel(
    const Tensor& input,
    Tensor& min_result,
    Tensor& max_result) {
  // 检查输入张量的数据类型是否为布尔型
  if (input.scalar_type() == ScalarType::Bool) {
    // 配置张量迭代器，将输入张量添加为输入
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    // 初始化布尔型的最小值和最大值结果
    bool min_result_data = true;
    bool max_result_data = false;
    // 应用序列化 CPU 内核，遍历输入张量的每个元素
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      // 计算布尔型的最小值和最大值
      min_result_data = min_result_data && a;
      max_result_data = max_result_data || a;
    });
    // 将计算得到的最小值和最大值填充到结果张量中
    min_result.fill_(min_result_data);
    max_result.fill_(max_result_data);
  } else if (input.scalar_type() == ScalarType::Long) {
    // 对于 int64_t 类型，矢量化实现存在性能问题，使用标量路径
    using int64_t_pair = std::pair<int64_t, int64_t>;
    // 调用具有两个输出的全局最小和最大值实现
    reduce_all_impl_two_outputs<int64_t>(min_result, max_result, input,
      int64_t_pair(upper_bound<int64_t>(), lower_bound<int64_t>()),
      // 在块上进行归约
      [=](int64_t_pair a, int64_t b) -> int64_t_pair {
        return int64_t_pair(min_impl(a.first, b), max_impl(a.second, b));
      },
      // 合并两个输入
      [=](int64_t_pair a, int64_t_pair b) -> int64_t_pair {
        return int64_t_pair(min_impl(a.first, b.first), max_impl(a.second, b.second));
      }
    );
  } else {
    // 对于除布尔型和 int64_t 外的所有数据类型，包括浮点数和整数
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "aminmax_cpu", [&] {
      // 使用向量化操作类型
      using Vec = Vectorized<opmath_type<scalar_t>>;
      // 使用标量对类型进行两个输出的全局最小和最大值归约实现
      using scalar_t_pair = std::pair<scalar_t, scalar_t>;
      reduce_all_impl_vec_two_outputs<scalar_t>(
        min_result,
        max_result,
        input,
        scalar_t_pair(upper_bound<scalar_t>(), lower_bound<scalar_t>()),
        // 归约两个输入
        [=] (scalar_t_pair a , scalar_t_pair b) -> scalar_t_pair {
          return scalar_t_pair(
            min_impl(a.first, b.first), max_impl(a.second, b.second));
        },
        // 合并两个输入向量
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); }
      );
    });
  }
}

} // namespace at::native

REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl);
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl);
REGISTER_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel);
```