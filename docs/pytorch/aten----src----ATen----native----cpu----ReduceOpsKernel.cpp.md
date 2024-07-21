# `.\pytorch\aten\src\ATen\native\cpu\ReduceOpsKernel.cpp`

```
// 定义宏以仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含标准库头文件<algorithm>
#include <algorithm>

// 包含 ATen 张量相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/LogAddExp.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 的整体功能头文件，否则包含 imag 操作相关的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

// 包含 c10 工具中的 Optional 和 irange 相关头文件
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

// 包含 ATen 的积累类型头文件
#include <ATen/AccumulateType.h>

// 进入 ATen 的 native 命名空间
namespace at::native {

// 匿名命名空间，用于定义局部函数
namespace {

// 使用 vec 命名空间中的功能
using namespace vec;

// 定义静态模板函数 cpu_cum_base_kernel，用于执行累积基本核心计算
template <typename scalar_t, typename func_t>
static inline void cpu_cum_base_kernel(const Tensor& result,
                                       const Tensor& self,
                                       int64_t dim,
                                       const func_t& f,
                                       scalar_t init_val) {
  // 如果结果张量的尺寸与自身张量不同，则调整结果张量的尺寸为自身张量的尺寸
  if (result.sizes() != self.sizes()) {
    at::native::resize_output(result, self.sizes());
  }

  // 如果自身张量的元素个数为 0，则直接返回
  if (self.numel() == 0) {
    return;
  }

  // 获取自身张量的维度数量
  const auto input_ndim = self.dim();

  // 如果自身张量的维度数量为 0，则用自身张量填充结果张量，并返回
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  // TODO 这里可能应该使用 at::native::make_reduction
  // 创建一个张量迭代器配置，用于执行迭代计算
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)  // 不检查所有张量是否具有相同的数据类型
    .resize_outputs(false)        // 不调整输出张量的尺寸
    // NOLINTNEXTLINE(bugprone-argument-comment)
    .declare_static_shape(self.sizes(), /*squash_dim=*/dim)  // 声明静态形状
    .add_output(result)           // 添加输出张量
    .add_const_input(self)        // 添加常量输入张量
    .build();                     // 构建迭代器配置

  // 确保结果张量和自身张量在指定维度 dim 上的步长不为空
  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  // 定义一个循环函数，用于执行核心计算
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    // 使用 c10::irange(n) 遍历 n 次
    for (const auto i C10_UNUSED : c10::irange(n)) {
      // 调用传入的函数对象 f，执行核心计算
      f(
        (scalar_t*)result_data_bytes, result_dim_stride,
        (scalar_t*)self_data_bytes, self_dim_stride, init_val
      );

      // 更新结果数据和自身数据的字节偏移
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  // 定义一个内部的粒度大小，用于控制每次迭代的工作量
  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, self.size(dim));

  // 使用迭代器执行循环操作，指定循环函数和粒度大小
  iter.for_each(loop, grain_size);
}

// 定义静态函数 cumsum_cpu_kernel，用于执行累积求和操作的 CPU 核心计算
static void cumsum_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  // 获取经包装的维度，以及在该维度上确保非空的自身张量尺寸
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  // 根据输入张量的标量类型，分派具体的类型处理
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumsum_out_cpu", [&] {
    // 使用模板函数 cpu_cum_base_kernel 处理输入数据，对每个元素进行累加操作
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      // 定义结果数据指针及步长，输入数据指针及步长，并初始化累加值
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        // 将初始值转换为指定类型 scalar_t
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        // 遍历输入数据的维度范围
        for (const auto i : c10::irange(self_dim_size)) {
          // 累加当前元素到 cum_number
          cum_number += self_data[i * self_dim_stride];
          // 将累加结果存入结果数据中
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 0
    );
  });
}

static void cumprod_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  // 计算累积乘积的 CPU 内核函数
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  // 确定有效的维度索引
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);
  // 确保自身张量在给定维度上非空

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumprod_out_cpu", [&] {
    // 分发到所有数据类型，包括复数类型
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        // 初始化累积数字
        for (const auto i : c10::irange(self_dim_size)) {
          // 迭代处理每个元素
          cum_number *= self_data[i * self_dim_stride];
          // 计算累积乘积
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
          // 将结果存入输出张量
        }
      }, /*init_val=*/ 1
    );
  });
}

static void logcumsumexp_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  // 计算对数累积和的 CPU 内核函数
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  // 确定有效的维度索引
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);
  // 确保自身张量在给定维度上非空

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "logcumsumexp_out_cpu", [&] {
    // 分发到浮点和复数类型
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto cum_number = (accscalar_t)init_val;
        // 初始化累积数字
        for (const auto i : c10::irange(self_dim_size)) {
          // 迭代处理每个元素
          accscalar_t x = self_data[i * self_dim_stride];

          cum_number = _log_add_exp_helper(x, cum_number);
          // 调用辅助函数计算对数累积和
          result_data[i * result_dim_stride] = static_cast<scalar_t>(cum_number);
          // 将结果存入输出张量
        }
      }, /*init_val=*/ -std::numeric_limits<scalar_t>::infinity()
    );
  });
}

static void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    // 分发到浮点类型，包括半精度和 BF16
    binary_kernel_reduce(
        iter,
        WelfordOps<
            scalar_t,
            double,
            int64_t,
            std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
        WelfordData<double, int64_t>());
    // 使用 Welford 算法进行二元内核约简操作
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  // Workaround for the error: '*' in boolean context, suggest '&&' instead
  // [-Werror=int-in-bool-context]
  // 解决错误：在布尔上下文中的 '*'，建议使用 '&&' 替代
  if (iter.dtype() == ScalarType::Bool) {
    // 如果张量类型是布尔型
    using scalar_t = bool;
    binary_kernel_reduce_vec(
        iter,
        [=](scalar_t a, scalar_t b)
            __ubsan_ignore_undefined__ -> scalar_t { return a && b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
            __ubsan_ignore_undefined__ { return a && b; },
        // NOLINTNEXTLINE(bugprone-argument-comment)
        /*identity=*/1);
    // 使用向量化的二元约简操作
  } else {
    // 对于其他数据类型
    # 使用宏 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 处理所有数据类型，包括 kBFloat16 和 kHalf
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "prod_out_cpu", [&] {
      # 调用 binary_kernel_reduce_vec 函数，对迭代器 iter 进行二元运算的向量化归约
      binary_kernel_reduce_vec(
          iter,
          # 第一个参数为标量的二元操作，忽略未定义行为，返回 a * b 的结果
          [=](scalar_t a, scalar_t b)
              __ubsan_ignore_undefined__ -> scalar_t { return a * b; },
          # 第二个参数为向量化的二元操作，忽略未定义行为，返回 a * b 的结果
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
              __ubsan_ignore_undefined__ { return a * b; },
          // NOLINTNEXTLINE(bugprone-argument-comment)
          /*identity=*/1);
    });
  // 如果当前数据类型是 float 或者 double 或者 BFloat16，并且可以在最后一个维度上进行向量化处理
  if (val == 2.0 && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    // 如果满足条件，使用向量化路径处理
    // 使用具体的归约操作符 NormTwoOps 处理归约
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else {
    // 对于其他情况，使用一般的归约操作符 NormOps 处理归约，并传入具体的值作为参数
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t, out_t>{acc_t(val)}, acc_t(0));
  }
}
    // 根据浮点类型和 BFloat16 分派处理函数 "norm_cpu"
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
        // 使用 float 作为累加类型（accumulator type）用于 BFloat16
        using acc_t = at::opmath_type<scalar_t>;
        // 调用二进制核函数进行最后一个维度的规范化计算
        binary_kernel_reduce_lastdim(iter, [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
          // 将结果数据和自身数据的字节转换为标量类型 scalar_t*
          scalar_t* result_data = (scalar_t*)result_data_bytes;
          scalar_t* self_data = (scalar_t*)self_data_bytes;

          // 使用 Vectorized 类处理标量类型 scalar_t
          using Vec = Vectorized<scalar_t>;
          // 使用 Vectorized 类处理累加类型 acc_t
          using fVec = Vectorized<acc_t>;
          // 初始化累加向量，初始值为 0
          fVec acc_vec{acc_t(0)};
          // 缓冲区大小为 fVec::size()
          acc_t buffer[fVec::size()];
          // 初始化维度索引 d
          int64_t d = 0;
          // 遍历数据，每次处理 Vec::size() 大小的数据
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            // 加载 self_data 中的数据到 data_vec
            Vec data_vec = Vec::loadu(self_data + d);
            // 调用 norm_two_reduce_step 函数处理 acc_vec 和 data_vec
            norm_two_reduce_step(acc_vec, data_vec);
          }
          // 将 acc_vec 中的值存储到 buffer 中
          acc_vec.store(buffer);
          // 将 buffer 中除了第一个元素外的值累加到第一个元素
          for (int j = 1; j < fVec::size(); j++) {
            buffer[0] = buffer[0] + buffer[j];
          }
          // 处理剩余不足一个 Vec::size() 的数据
          for (; d < size; d++) {
            // 将 self_data[d] 转换为累加类型 acc_t，计算平方后加入 buffer[0]
            acc_t data_val = acc_t(self_data[d]);
            buffer[0] += data_val * data_val;
          }
          // 将 buffer[0] 的平方根赋值给 result_data[0]
          result_data[0] = scalar_t(std::sqrt(buffer[0]));
        });
      });
  } else {
    // 处理其他情况，包括半精度、BFloat16 和复数类型
    if (iter.dtype(0) == kHalf) {
      // 调用 norm_kernel_cpu_impl 处理 Half 和 float 类型的情况
      return norm_kernel_cpu_impl<at::Half, float>(iter, val);
    } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
      // 半精度到 float 的类型提升和规范化处理
      return norm_kernel_cpu_impl<at::Half, float, float>(iter, val);
    } else if(iter.dtype(0) == kBFloat16) {
      // 处理 BFloat16 类型
      return norm_kernel_cpu_impl<at::BFloat16, float>(iter, val);
    } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
      // BFloat16 到 float 的类型提升和规范化处理
      return norm_kernel_cpu_impl<at::BFloat16, float, float>(iter, val);
    }

    // 处理其他浮点和复杂类型的情况，调用 norm_kernel_cpu_impl 处理
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_cpu", [&] {
      norm_kernel_cpu_impl<scalar_t>(iter, val);
    });

    // 对于复数输出，上述内核函数不处理虚数部分，因此需要将其清零
    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

// 实现按位与操作的内核函数
static void and_kernel_impl(TensorIterator& iter) {
  // 如果数据类型为字节型（uint8）
  if (iter.dtype() == ScalarType::Byte) {
    // 参考 [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        // Lambda 表达式：对两个 uint8_t 类型进行按位与操作，返回结果为 uint8_t
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a && b) ? 1 : 0; },
        // Lambda 表达式：对两个 Vectorized<uint8_t> 对象进行按位与操作，并取反
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          // 注意：!= 操作返回 0xFF 而非 0x01，因此需取反以得到期望的结果
          return (a != Vectorized<uint8_t>(0)).neg() & (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/true);
  } else {
    binary_kernel_reduce_vec(
        iter,
        // Lambda 表达式：对两个 bool 类型进行逻辑与操作，返回结果为 bool
        [=](bool a, bool b) -> bool { return a && b; },
        // Lambda 表达式：对两个 Vectorized<bool> 对象进行逻辑与操作
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          // 添加此处实现，而非在 vec256_base 中，以避免返回值的不一致性
          // vec256_base 中的其他比较运算符返回 -1/0（所有位为 1 / 所有位为 0）作为 true/false
          // 遵循 AVX2 约定，这在与其他向量化操作结合时很方便
          // 例如，可以使用逻辑操作结果作为位操作的掩码来获取/重置向量中的多个元素
          //
          // 在此方法中，用户期望 all() 等方法返回 1/0 作为 true/false。
          Vectorized<bool> c = Vectorized<bool>();

          // 遍历向量中的元素，计算逻辑与结果
          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] && b[i];
          }
          return c;
        },
        /*ident=*/true);
  }
}

// 实现按位或操作的内核函数
static void or_kernel_impl(TensorIterator& iter) {
  // 如果数据类型为字节型（uint8）
  if (iter.dtype() == ScalarType::Byte) {
    // 参考 [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        // Lambda 表达式：对两个 uint8_t 类型进行按位或操作，返回结果为 uint8_t
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a || b) ? 1 : 0; },
        // Lambda 表达式：对两个 Vectorized<uint8_t> 对象进行按位或操作，并取反
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          return (a != Vectorized<uint8_t>(0)).neg() | (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/false);
  } else {
    binary_kernel_reduce_vec(
        iter,
        // Lambda 表达式：对两个 bool 类型进行逻辑或操作，返回结果为 bool
        [=](bool a, bool b) -> bool { return a || b; },
        // Lambda 表达式：对两个 Vectorized<bool> 对象进行逻辑或操作
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          Vectorized<bool> c = Vectorized<bool>();

          // 遍历向量中的元素，计算逻辑或结果
          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] || b[i];
          }
          return c;
        },
        /*ident=*/false);
  }
}

// 定义模板结构体，用于处理最小值操作
template<typename scalar_t>
struct MinValuesOps: public at::native::MinOps<scalar_t> {
  using arg_t = typename MinOps<scalar_t>::arg_t;
  static scalar_t project(arg_t arg) {
    return arg.first;
  }
};

// 实现最小值操作的内核函数
static void min_values_kernel_impl(TensorIterator& iter) {
  // 如果数据类型为长整型（int64_t）
  if (iter.dtype() == kLong) {
    // 此情况特殊，因为 Vectorized<int64_t> 无法处理 upper_bound<int64_t>()
    // 参见：https://github.com/pytorch/pytorch/issues/43254
    using scalar_t = int64_t;
    // 使用二元内核函数对迭代器执行操作，使用MinValuesOps<scalar_t>进行操作
    // 并且传入std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1)作为参数
    binary_kernel_reduce(
      iter,
      MinValuesOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
    // 函数执行完毕后直接返回
    return;
  }
  // 针对所有类型（包括kBFloat16, kHalf, kBool），使用CPU上下文进行分派
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
    // 使用二元内核函数进行向量化操作，传入两个lambda函数
    // 第一个lambda函数用于标量操作，返回两个标量的最小值
    // 第二个lambda函数用于向量操作，返回两个向量的最小值
    // upper_bound<scalar_t>()的结果被转换为double类型并作为参数传入
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return minimum(a, b); },
      static_cast<double>(upper_bound<scalar_t>()));
  });
}  // anonymous namespace



REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl);



REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);



// mean implementation for CPU is in aten/src/ATen/native/ReduceOps.cpp
// but mean_stub must be defined for CPU as well
REGISTER_DISPATCH(mean_stub, nullptr);



REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl);



REGISTER_DISPATCH(and_stub, &and_kernel_impl);



REGISTER_DISPATCH(or_stub, &or_kernel_impl);



REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl);



REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl);



REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl);



REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl);



REGISTER_DISPATCH(cumprod_stub, &cumprod_cpu_kernel);



REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel);



REGISTER_DISPATCH(logcumsumexp_stub, &logcumsumexp_cpu_kernel);



}  // namespace at::native
```