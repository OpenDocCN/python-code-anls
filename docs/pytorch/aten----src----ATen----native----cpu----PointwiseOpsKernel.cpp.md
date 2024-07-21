# `.\pytorch\aten\src\ATen\native\cpu\PointwiseOpsKernel.cpp`

```
// 定义宏，禁用包含运算符的断言
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库的分发机制
#include <ATen/Dispatch.h>
// 包含 ATen 库中的点对点运算相关头文件
#include <ATen/native/PointwiseOps.h>
// 包含 ATen 库中的张量迭代器相关头文件
#include <ATen/native/TensorIterator.h>
// 包含 ATen 库中 CPU 环境下的循环优化相关头文件
#include <ATen/native/cpu/Loops.h>
// 包含 C10 核心库中的标量定义
#include <c10/core/Scalar.h>
// 包含 ATen 库中 CPU 环境下的向量化功能头文件
#include <ATen/cpu/vec/functional.h>

namespace at::native {

// 匿名命名空间，用于限定函数的作用域
namespace {

// 定义 addcmul_cpu_kernel 函数，用于执行 addcmul 操作
static void addcmul_cpu_kernel(TensorIteratorBase& iter, const Scalar& value) {
  // 获取迭代器中的数据类型
  ScalarType dtype = iter.common_dtype();
  // 检查数据类型是否为降维浮点类型
  if (at::isReducedFloatingType(dtype)) {
    // 如果是降维浮点类型，则调用模板函数处理
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "addcmul_cpu_out", [&]() {
      // 将标量值转换为 float 类型
      float float_val = value.to<float>();
      // 创建 float 类型的向量化对象
      auto float_vec = Vectorized<float>(float_val);
      // 调用 CPU 向量化函数，对迭代器进行操作
      cpu_kernel_vec(
          iter,
          // 核心运算，对每个标量值进行计算
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return float(self_val) + float_val * float(t1_val) * float(t2_val);
          },
          // 向量化操作，对向量化数据进行计算
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) -> Vectorized<scalar_t> {
            auto [self_vec0, self_vec1] = convert_to_float<scalar_t>(self_vec);
            auto [t1_vec0, t1_vec1] = convert_to_float<scalar_t>(t1_vec);
            auto [t2_vec0, t2_vec1] = convert_to_float<scalar_t>(t2_vec);
            self_vec0 = self_vec0 + float_vec * t1_vec0 * t2_vec0;
            self_vec1 = self_vec1 + float_vec * t1_vec1 * t2_vec1;
            return convert_from_float<scalar_t>(self_vec0, self_vec1);
          });
    });
  } else {
    // 如果不是降维浮点类型，则调用通用类型模板函数处理
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::ComplexHalf,
                                           dtype, "addcmul_cpu_out", [&] {
      // 将标量值转换为对应类型的标量
      scalar_t scalar_val = value.to<scalar_t>();
      // 创建对应类型的向量化对象
      auto scalar_vec = Vectorized<scalar_t>(scalar_val);
      // 调用 CPU 向量化函数，对迭代器进行操作
      cpu_kernel_vec(
          iter,
          // 核心运算，对每个标量值进行计算
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return self_val + scalar_val * t1_val * t2_val;
          },
          // 向量化操作，对向量化数据进行计算
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) {
            return self_vec + scalar_vec * t1_vec * t2_vec;
          });
    });
  }
}

// 定义 addcdiv_cpu_kernel 函数，用于执行 addcdiv 操作
static void addcdiv_cpu_kernel(TensorIteratorBase& iter, const Scalar& value) {
  // 获取迭代器中的数据类型
  ScalarType dtype = iter.common_dtype();
  // 检查数据类型是否为降维浮点类型
  if (at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "addcdiv_cpu_out", [&]() {
      // 将输入的值转换为单精度浮点数
      float float_val = value.to<float>();
      // 创建单精度浮点数的向量
      auto float_vec = Vectorized<float>(float_val);
      // 调用 CPU 内核函数处理向量化计算
      cpu_kernel_vec(
          iter,
          // 标量版本的计算函数，对每个元素执行加法、乘法和除法操作
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return float(self_val) + float_val * float(t1_val) / float(t2_val);
          },
          // 向量版本的计算函数，对整个向量执行加法、乘法和除法操作
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) -> Vectorized<scalar_t> {
              // 将输入向量转换为单精度浮点数向量
              auto [self_vec0, self_vec1] = convert_to_float<scalar_t>(self_vec);
              auto [t1_vec0, t1_vec1] = convert_to_float<scalar_t>(t1_vec);
              auto [t2_vec0, t2_vec1] = convert_to_float<scalar_t>(t2_vec);
              // 执行向量化的加法、乘法和除法操作
              self_vec0 = self_vec0 + float_vec * t1_vec0 / t2_vec0;
              self_vec1 = self_vec1 + float_vec * t1_vec1 / t2_vec1;
              // 将结果向量转换回原始类型的向量
              return convert_from_float<scalar_t>(self_vec0, self_vec1);
          });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcdiv_cpu_out", [&] {
      // 将输入的值转换为指定类型的标量
      scalar_t scalar_val = value.to<scalar_t>();
      // 创建指定类型的标量向量
      auto scalar_vec = Vectorized<scalar_t>(scalar_val);
      // 调用 CPU 内核函数处理向量化计算
      cpu_kernel_vec(
          iter,
          // 标量版本的计算函数，对每个元素执行加法、乘法和除法操作
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return self_val + scalar_val * t1_val / t2_val;
          },
          // 向量版本的计算函数，对整个向量执行加法、乘法和除法操作
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) {
            return self_vec + scalar_vec * t1_vec / t2_vec;
          });
    });
  }
}

static void smooth_l1_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double beta) {
  // 获取迭代器中第一个操作数的数据类型
  ScalarType dtype = iter.dtype(0);
  // 如果数据类型为 kBFloat16
  if (dtype == kBFloat16) {
    // 将 norm 转换为 float 类型
    auto norm_val = norm.to<float>();
    // 将 beta 转换为 float 类型
    float beta_val(beta);
    // 创建 norm_val 的向量化对象
    auto norm_val_vec = Vectorized<float>(norm_val);
    // 创建 beta_val 的向量化对象
    auto beta_val_vec = Vectorized<float>(beta_val);
    // 创建常量向量，分别表示 -1、0、1
    const auto neg_1_vec = Vectorized<float>(-1);
    const auto zero_vec = Vectorized<float>(0);
    const auto pos_1_vec = Vectorized<float>(1);
    // 使用向量化的 CPU 内核函数进行计算
    cpu_kernel_vec(iter,
      // 核心计算逻辑，处理 BFloat16 类型的输入、目标和梯度输出
      [=](BFloat16 input, BFloat16 target, BFloat16 grad_output) -> BFloat16 {
        // 计算 x = input - target
        const auto x = float(input) - float(target);
        // 根据 x 的值进行不同情况的计算返回
        if (x <= -beta){
          return -norm_val * float(grad_output);
        }else if (x >= beta){
          return norm_val * float(grad_output);
        }else{
          return norm_val * x * float(grad_output) / beta;
        }
      },
      // 向量化版本的计算，处理向量化的 BFloat16 输入、目标和梯度输出
      [norm_val_vec, beta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vectorized<BFloat16> input, Vectorized<BFloat16> target, Vectorized<BFloat16> grad_output) -> Vectorized<BFloat16> {
        // 使用两次 blendv 调用模拟三种情况
        // 1        如果 x >= beta
        // -1       如果 x <= -beta
        // x / beta 如果 |x| < beta
        auto [input0, input1] = convert_bfloat16_float(input);
        auto [target0, target1] = convert_bfloat16_float(target);
        auto [grad_output0, grad_output1] = convert_bfloat16_float(grad_output);
        auto x = input0 - target0;
        auto pos_or_neg_1_vec = Vectorized<float>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        auto x_abs = x.abs();
        auto output = Vectorized<float>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        input0 = norm_val_vec * output * grad_output0;

        x = input1 - target1;
        pos_or_neg_1_vec = Vectorized<float>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        x_abs = x.abs();
        output = Vectorized<float>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        input1 = norm_val_vec * output * grad_output1;
        return convert_float_bfloat16(input0, input1);
      }
    );
  } else {
    // 处理除了 kBFloat16 外的所有其他数据类型
    AT_DISPATCH_ALL_TYPES(dtype, "smooth_l1_backward_cpu_out", [&] {
    // 将 norm 转换为当前数据类型的标量类型
    auto norm_val = norm.to<scalar_t>();
    // 将 beta 转换为当前数据类型的标量类型
    scalar_t beta_val(beta);
    // 创建 norm_val 的向量化对象
    auto norm_val_vec = Vectorized<scalar_t>(norm_val);
    // 创建 beta_val 的向量化对象
    auto beta_val_vec = Vectorized<scalar_t>(beta_val);
    // 创建常量向量，分别表示 -1、0、1
    const auto neg_1_vec = Vectorized<scalar_t>(-1);
    const auto zero_vec = Vectorized<scalar_t>(0);
    const auto pos_1_vec = Vectorized<scalar_t>(1);
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        // 计算差值 x = input - target
        const auto x = input - target;
        // 根据 x 的大小返回不同的梯度修正值
        if (x <= -beta)
          return -norm_val * grad_output;
        else if (x >= beta)
          return norm_val * grad_output;
        else
          return norm_val * x * grad_output / beta;
      },
      [norm_val_vec, beta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vectorized<scalar_t> input, Vectorized<scalar_t> target, Vectorized<scalar_t> grad_output) -> Vectorized<scalar_t> {
        // 使用两次 blendv 调用模拟三种情况
        // 1        如果 x >= beta
        // -1       如果 x <= -beta
        // x / beta 如果 |x| < beta
        const auto x = input - target;
        // 根据 x 的正负情况选择 -1 或 1
        const auto pos_or_neg_1_vec = Vectorized<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        // 计算 x 的绝对值
        const auto x_abs = x.abs();
        // 根据 x 的绝对值是否大于等于 beta 选择 x / beta 或 pos_or_neg_1_vec
        const auto output = Vectorized<scalar_t>::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        // 返回修正后的输出向量
        return norm_val_vec * output * grad_output;
      }
    );
  });
  }


这段代码使用了 C++ 的 lambda 表达式和向量化操作，其中 `cpu_kernel_vec` 函数接受两个 lambda 表达式作为参数。第一个 lambda 表达式计算单个标量的梯度修正值，根据输入和目标的差值 x 的大小情况返回不同的值。第二个 lambda 表达式接受向量化的输入、目标和梯度值，并使用 `blendv` 方法模拟三种不同的条件：如果 x 大于等于 beta，则返回 1；如果 x 小于等于 -beta，则返回 -1；否则返回 x 除以 beta。
} // 匿名命名空间的结束

static void huber_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double delta) {
  // 获取迭代器的数据类型
  ScalarType dtype = iter.dtype(0);
  // 根据数据类型调度不同的函数，处理浮点数类型和特定的类型
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "huber_backward_cpu_out", [&] {
    // 将 norm 转换为当前数据类型的标量
    auto norm_val = norm.to<scalar_t>();
    // 创建 delta 对应的标量值
    scalar_t delta_val(delta);
    // 创建 norm_val 和 delta_val 的向量化形式
    auto norm_val_vec = Vectorized<scalar_t>(norm_val);
    auto delta_val_vec = Vectorized<scalar_t>(delta_val);
    // 创建常量向量：-1, 0, 1
    const auto neg_1_vec = Vectorized<scalar_t>(-1);
    const auto zero_vec = Vectorized<scalar_t>(0);
    const auto pos_1_vec = Vectorized<scalar_t>(1);
    // 调用 CPU 核函数进行向量化计算
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        // 计算输入与目标之间的差值 x
        const auto x = input - target;
        // 根据 x 的值返回不同的梯度计算结果
        if (x <= -delta) {
          return -norm_val * grad_output * delta;
        } else if (x >= delta) {
          return norm_val * grad_output * delta;
        } else {
          return norm_val * x * grad_output;
        }
      },
      [norm_val_vec, delta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vectorized<scalar_t> input, Vectorized<scalar_t> target, Vectorized<scalar_t> grad_output) -> Vectorized<scalar_t> {
        // 使用两个 blendv 调用来模拟三种情况
        // delta     如果 x >= delta
        // -delta    如果 x <= -delta
        // x         如果 |x| < delta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vectorized<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vectorized<scalar_t>::blendv(
            x, pos_or_neg_1_vec * delta_val_vec, x_abs >= delta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
}

static void mse_backward_cpu_kernel(TensorIterator& iter, const Scalar& value) {
  // 获取迭代器的数据类型
  ScalarType dtype = iter.dtype(0);
  // 根据数据类型调度不同的函数，处理所有的类型
  AT_DISPATCH_ALL_TYPES(dtype, "mse_backward_cpu_out", [&] {
    // 将 value 转换为当前数据类型的标量
    scalar_t scalar_val = value.to<scalar_t>();
    // 创建 scalar_val 的向量化形式
    auto scalar_vec = Vectorized<scalar_t>(scalar_val);
    // 调用 CPU 核函数进行向量化计算
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
          // 返回 MSE 损失的梯度计算结果
          return scalar_val * (self_val - t1_val) * t2_val;
        },
        [=](Vectorized<scalar_t> self_vec,
            Vectorized<scalar_t> t1_vec,
            Vectorized<scalar_t> t2_vec) {
          // 返回向量化的 MSE 损失的梯度计算结果
          return scalar_vec * (self_vec - t1_vec) *  t2_vec;
    });
  });
}

} // 匿名命名空间的结束

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel); // 注册 addcmul 的 CPU 核函数
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cpu_kernel); // 注册 addcdiv 的 CPU 核函数
REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_cpu_kernel); // 注册 smooth_l1_backward 的 CPU 核函数
REGISTER_DISPATCH(huber_backward_stub, &huber_backward_cpu_kernel); // 注册 huber_backward 的 CPU 核函数
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_cpu_kernel); // 注册 mse_backward 的 CPU 核函数

} // namespace at::native  // 结束 at::native 命名空间
```