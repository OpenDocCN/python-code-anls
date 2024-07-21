# `.\pytorch\aten\src\ATen\native\cpu\Activation.cpp`

```
// 定义编译选项，禁用 Torch 的操作符断言
#define TORCH_ASSERT_NO_OPERATORS
// 如果未定义 _USE_MATH_DEFINES 宏，则定义该宏
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// 包含 Torch 的激活函数头文件
#include <ATen/native/Activation.h>

// 包含 C++ 标准库的头文件
#include <cmath>
#include <functional>

// 包含 Torch 的核心分发和数学操作类型头文件
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>

// 包含 C10 标量类型的头文件
#include <c10/core/Scalar.h>

namespace at::native {

// 命名空间开始

// 声明静态函数 log_sigmoid_cpu_kernel，处理输入输出 TensorBase 对象
namespace {
static void log_sigmoid_cpu_kernel(TensorBase &output, TensorBase &buffer, const TensorBase &input) {
  // 检查输入张量是否是降维的浮点类型
  if (at::isReducedFloatingType(input.scalar_type())) {
    // 根据输入的浮点类型分发执行 log_sigmoid_cpu 函数
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&]() {
      // 使用 Vectorized 类型定义标量类型
      using Vec = Vectorized<scalar_t>;
      // 获取输出、缓冲和输入数据指针
      scalar_t* output_data = output.data_ptr<scalar_t>();
      scalar_t* buffer_data = buffer.data_ptr<scalar_t>();
      const scalar_t* input_data = input.const_data_ptr<scalar_t>();
      // 并行循环处理输入张量中的每个元素
      parallel_for(0, input.numel(), 1, [&] (int64_t begin, int64_t end) {
        // 计算当前处理的数据块大小
        int64_t size = end - begin;
        int64_t d = 0;
        // 以 Vectorized 类型的大小作为步长循环处理数据
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          // 加载输入数据到 Vectorized 对象
          Vec data_vec = Vec::loadu(input_data + begin + d);
          // 将 Vectorized 对象的数据转换为 float 类型
          auto [data_vec0, data_vec1] = convert_to_float<scalar_t>(data_vec);
          // 计算每个元素的 log_sigmoid 操作
          Vectorized<float> min_vec = minimum(data_vec0, Vectorized<float>(float(0)));
          Vectorized<float> buffer_vec0 = data_vec0.abs().neg().exp();
          Vectorized<float> output_vec0 = min_vec - buffer_vec0.log1p();
          min_vec = minimum(data_vec1, Vectorized<float>(float(0)));
          Vectorized<float> buffer_vec1 = data_vec1.abs().neg().exp();
          Vectorized<float> output_vec1 = min_vec - buffer_vec1.log1p();
          // 将 float 类型的结果转换回标量类型并存储到缓冲和输出数据中
          convert_from_float<scalar_t>(buffer_vec0, buffer_vec1).store(buffer_data + begin + d);
          convert_from_float<scalar_t>(output_vec0, output_vec1).store(output_data + begin + d);
        }
        // 处理剩余的数据块（少于 Vectorized 大小的数据）
        if (size - d > 0) {
          // 加载剩余数据到 Vectorized 对象
          Vec data_vec = Vec::loadu(input_data + begin + d, size - d);
          auto [data_vec0, data_vec1] = convert_to_float<scalar_t>(data_vec);
          // 计算每个元素的 log_sigmoid 操作
          Vectorized<float> min_vec = minimum(data_vec0, Vectorized<float>(float(0)));
          Vectorized<float> buffer_vec0 = data_vec0.abs().neg().exp();
          Vectorized<float> output_vec0 = min_vec - buffer_vec0.log1p();
          min_vec = minimum(data_vec1, Vectorized<float>(float(0)));
          Vectorized<float> buffer_vec1 = data_vec1.abs().neg().exp();
          Vectorized<float> output_vec1 = min_vec - buffer_vec1.log1p();
          // 将 float 类型的结果转换回标量类型并存储到缓冲和输出数据中
          convert_from_float<scalar_t>(buffer_vec0, buffer_vec1).store(buffer_data + begin + d, size - d);
          convert_from_float<scalar_t>(output_vec0, output_vec1).store(output_data + begin + d, size - d);
        }
      });
    });
  } else {
    // 使用 AT_DISPATCH_FLOATING_TYPES 宏，生成根据输入类型不同的模板化代码块 "log_sigmoid_cpu"
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&] {
      // 定义别名 Vec 作为 Vectorized 类的实例，用于处理标量类型 scalar_t
      using Vec = Vectorized<scalar_t>;
      // 获取输出数据的指针，并转换为 scalar_t 类型
      scalar_t* output_data = output.data_ptr<scalar_t>();
      // 获取缓冲区数据的指针，并转换为 scalar_t 类型
      scalar_t* buffer_data = buffer.data_ptr<scalar_t>();
      // 获取输入数据的常量指针，并转换为 scalar_t 类型
      const scalar_t* input_data = input.const_data_ptr<scalar_t>();
      // 并行循环，处理从 0 到输入数据元素个数的索引范围
      parallel_for(0, input.numel(), 1, [&] (int64_t begin, int64_t end) {
        // 计算当前处理范围的大小
        int64_t size = end - begin;
        // 初始化 d 为 0，用于迭代处理向量化大小的数据块
        int64_t d = 0;
        // 迭代处理所有完整的 Vec::size() 大小的向量化数据块
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          // 加载以 input_data + begin + d 为起点的 Vec::size() 大小的数据向量
          Vec data_vec = Vec::loadu(input_data + begin + d);
          // 计算 data_vec 中各元素与标量 0 的较小值向量
          Vec min_vec = vec::minimum(data_vec, Vec(scalar_t(0)));
          // 计算 data_vec 中各元素的绝对值的负指数向量
          Vec buffer_vec = data_vec.abs().neg().exp();
          // 计算 log1p(buffer_vec) 后与 min_vec 的向量差
          Vec output_vec = min_vec - buffer_vec.log1p();
          // 将 buffer_vec 的数据存储到 buffer_data + begin + d 处
          buffer_vec.store(buffer_data + begin + d);
          // 将 output_vec 的数据存储到 output_data + begin + d 处
          output_vec.store(output_data + begin + d);
        }
        // 处理剩余不足 Vec::size() 大小的数据块
        if (size - d > 0) {
          // 加载以 input_data + begin + d 为起点的剩余数据向量
          Vec data_vec = Vec::loadu(input_data + begin + d, size - d);
          // 计算 data_vec 中各元素与标量 0 的较小值向量
          Vec min_vec = vec::minimum(data_vec, Vec(scalar_t(0)));
          // 计算 data_vec 中各元素的绝对值的负指数向量
          Vec buffer_vec = data_vec.abs().neg().exp();
          // 计算 log1p(buffer_vec) 后与 min_vec 的向量差
          Vec output_vec = min_vec - buffer_vec.log1p();
          // 将 buffer_vec 的部分数据存储到 buffer_data + begin + d 处
          buffer_vec.store(buffer_data + begin + d, size - d);
          // 将 output_vec 的部分数据存储到 output_data + begin + d 处
          output_vec.store(output_data + begin + d, size - d);
        }
      });
    });
}

// 定义静态函数，实现 log sigmoid 激活函数的反向传播的 CPU 内核
static void log_sigmoid_backward_cpu_kernel(TensorIterator& iter) {
  // 检查迭代器的数据类型是否为浮点数类型的降维类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用宏展开不同浮点数类型的处理逻辑
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "log_sigmoid_backward_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;
      auto zero_val = float(0);
      auto zero_vec = Vectorized<float>(zero_val);
      auto one_val = float(1);
      auto one_vec = Vectorized<float>(one_val);
      // 使用向量化的方式执行 CPU 内核操作
      cpu_kernel_vec(iter,
        // 核心计算函数，计算 log sigmoid 的反向传播
        [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
          auto in_negative = float(a) < float(0);
          auto max_deriv = in_negative ? float(1) : float(0);
          auto sign = in_negative ? float(1) : -float(1);
          return (max_deriv - sign * (float(b) / (float(1) + b))) * float(c);
        },
        // 向量化版本的核心计算函数
        [=](Vec a, Vec b, Vec c) -> Vec {
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          auto [b0, b1] = convert_to_float<scalar_t>(b);
          auto [c0, c1] = convert_to_float<scalar_t>(c);
          auto mask = a0 < zero_vec;
          auto max_deriv_vec = Vectorized<float>::blendv(zero_vec, one_vec, mask);
          auto sign_vec = Vectorized<float>::blendv(one_vec.neg(), one_vec, mask);
          a0 = (max_deriv_vec - sign_vec * (b0 / (one_vec + b0))) * c0;
          mask = a1 < zero_vec;
          max_deriv_vec = Vectorized<float>::blendv(zero_vec, one_vec, mask);
          sign_vec = Vectorized<float>::blendv(one_vec.neg(), one_vec, mask);
          a1 = (max_deriv_vec - sign_vec * (b1 / (one_vec + b1))) * c1;
          return convert_from_float<scalar_t>(a0, a1);
        });
    });
  } else {
    // 处理非降维浮点数类型的情况
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_sigmoid_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto zero_val = scalar_t(0);
    auto zero_vec = Vec(zero_val);
    auto one_val = scalar_t(1);
    auto one_vec = Vec(one_val);
    // 使用向量化的方式执行 CPU 内核操作
    cpu_kernel_vec(iter,
      // 核心计算函数，计算 log sigmoid 的反向传播
      [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        auto in_negative = a < scalar_t(0);
        auto max_deriv = in_negative ? scalar_t(1) : scalar_t(0);
        auto sign = in_negative ? scalar_t(1) : -scalar_t(1);
        return (max_deriv - sign * (b / (scalar_t(1) + b))) * c;
      },
      // 向量化版本的核心计算函数
      [=](Vec a, Vec b, Vec c) -> Vec {
        auto mask = a < zero_vec;
        auto max_deriv_vec = Vec::blendv(zero_vec, one_vec, mask);
        auto sign_vec = Vec::blendv(one_vec.neg(), one_vec, mask);
        return (max_deriv_vec - sign_vec * (b / (one_vec + b))) * c;
      });
  });
  }
}

// 定义阈值化操作的 CPU 内核函数
static void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold_scalar,
    const Scalar& value_scalar) {
  // 检查迭代器的数据类型是否为浮点数类型的降维类型
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "threshold_cpu", [&]() {
      // 使用 AT_DISPATCH_REDUCED_FLOATING_TYPES 宏根据迭代器的数据类型进行条件分发
      using Vec = Vectorized<float>;
      // 将 threshold_scalar 转换为 float 类型
      float threshold = threshold_scalar.to<float>();
      // 创建一个 Vec 对象，用 threshold 初始化
      Vec threshold_v = Vec(threshold);
      // 将 value_scalar 转换为与迭代器元素相同类型的标量值
      scalar_t value = value_scalar.to<scalar_t>();
      // 创建一个 Vec 对象，用 value 初始化
      Vec value_v = Vec(float(value));
      // 调用 cpu_kernel_vec 处理迭代器 iter
      cpu_kernel_vec(
          iter,
          // 标量版本的 lambda 函数，判断 x 是否小于等于 threshold，返回相应值
          [&](scalar_t x, scalar_t other) -> scalar_t {
            return float(x) <= threshold ? value : other;
          },
          // 向量化版本的 lambda 函数，将 x 和 other 转换为 float 后，根据条件混合 value 和 other
          [&](Vectorized<scalar_t> x, Vectorized<scalar_t> other) -> Vectorized<scalar_t> {
            auto [x0, x1] = convert_to_float<scalar_t>(x);
            auto [other0, other1] = convert_to_float<scalar_t>(other);
            return convert_from_float<scalar_t>(Vec::blendv(other0, value_v, x0 <= threshold_v),
                                                Vec::blendv(other1, value_v, x1 <= threshold_v));
          });
    });
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "threshold_cpu", [&] {
      // 使用 AT_DISPATCH_ALL_TYPES 宏根据迭代器的数据类型进行条件分发
      using Vec = Vectorized<scalar_t>;
      // 将 threshold_scalar 转换为与迭代器元素相同类型的标量值
      scalar_t threshold = threshold_scalar.to<scalar_t>();
      // 创建一个 Vec 对象，用 threshold 初始化
      Vec threshold_v = Vec(threshold);
      // 将 value_scalar 转换为与迭代器元素相同类型的标量值
      scalar_t value = value_scalar.to<scalar_t>();
      // 创建一个 Vec 对象，用 value 初始化
      Vec value_v = Vec(value);
      // 调用 cpu_kernel_vec 处理迭代器 iter
      cpu_kernel_vec(
          iter,
          // 标量版本的 lambda 函数，判断 x 是否小于等于 threshold，返回相应值
          [&](scalar_t x, scalar_t other) -> scalar_t {
            return x <= threshold ? value : other;
          },
          // 向量化版本的 lambda 函数，根据条件混合 value 和 other
          [&](Vec x, Vec other) -> Vec {
            return Vec::blendv(other, value_v, x <= threshold_v);
          });
    });
  }
// 定义一个名为 `elu_kernel` 的函数，接受一个 `TensorIteratorBase` 类的参数 `it` 和三个标量参数 `alpha`、`scale`、`input_scale`
void elu_kernel(TensorIteratorBase& it, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  // 检查迭代器中数据类型是否为降低的浮点类型
  if (at::isReducedFloatingType(it.common_dtype())) {
    // 在降低的浮点类型上分发操作，命名为 `elu_cpu`
    AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "elu_cpu", [&]() {
      // 计算负系数、正系数和输入缩放系数，转换为 float 类型
      auto negcoef = alpha.to<float>() * scale.to<float>();
      auto poscoef = scale.to<float>();
      auto negiptcoef = input_scale.to<float>();
      // 创建 Vectorized 对象来处理向量化操作
      const Vectorized<float> negcoef_vec(negcoef);
      const Vectorized<float> negiptcoef_vec(negiptcoef);
      const Vectorized<float> poscoef_vec(poscoef);
      const Vectorized<float> one_vec(static_cast<float>(1));
      const Vectorized<float> zero_vec(static_cast<float>(0));
      // 调用 CPU 核函数 `cpu_kernel_vec`
      cpu_kernel_vec(
        it,
        // 对每个元素执行的操作，根据元素值是否小于等于 0 来决定返回值
        [negcoef, negiptcoef, poscoef](scalar_t a) -> scalar_t {
          return float(a) <= float(0) ? (std::exp(float(a) * negiptcoef) - float(1)) * negcoef : float(a) * poscoef;
        },
        // 对向量执行的操作，根据向量元素的比较结果来决定返回值
        [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &one_vec, &zero_vec](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          auto cmp0 = (a0 > zero_vec);
          auto cmp1 = (a1 > zero_vec);
          // 定义一个函数用于根据条件选择返回值
          auto get_res_masked = [&](Vectorized<float>& cmp, Vectorized<float>& a) {
            return !cmp.zero_mask() ? a * poscoef_vec :
              Vectorized<float>::blendv(((a * negiptcoef_vec).exp() - one_vec) * negcoef_vec, a * poscoef_vec, cmp);
          };
          auto res0 = get_res_masked(cmp0, a0);
          auto res1 = get_res_masked(cmp1, a1);
          return convert_from_float<scalar_t>(res0, res1);
        });
    });
  } else {
    // 在浮点类型上分发操作，命名为 `elu_cpu`
    AT_DISPATCH_FLOATING_TYPES(it.common_dtype(), "elu_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;
      // 计算负系数、正系数和输入缩放系数，转换为当前标量类型
      auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
      auto poscoef = scale.to<scalar_t>();
      auto negiptcoef = input_scale.to<scalar_t>();
      // 创建 Vectorized 对象来处理向量化操作
      const Vec negcoef_vec(negcoef);
      const Vec negiptcoef_vec(negiptcoef);
      const Vec poscoef_vec(poscoef);
      const Vec one_vec(static_cast<scalar_t>(1));
      const Vec zero_vec(static_cast<scalar_t>(0));
      // 调用 CPU 核函数 `cpu_kernel_vec`
      cpu_kernel_vec(
          it,
          // 对每个元素执行的操作，根据元素值是否小于等于 0 来决定返回值
          [negcoef, negiptcoef, poscoef](scalar_t a) -> scalar_t {
            return a <= scalar_t(0) ? (std::exp(a * negiptcoef) - scalar_t(1)) * negcoef : a * poscoef;
          },
          // 对向量执行的操作，根据向量元素的比较结果来决定返回值
          [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &one_vec, &zero_vec](Vec a) -> Vec {
            auto cmp = (a > zero_vec);
            if (!cmp.zero_mask()) {  // 只需计算 a * poscoef（非常快速）
              return a * poscoef_vec;
            } else {
              return Vec::blendv(((a * negiptcoef_vec).exp() - one_vec) * negcoef_vec, a * poscoef_vec, cmp);
            }
          });
    });
  }
}
    AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "elu_backward_cpu", [&]() {
    // 使用宏展开，根据输入张量的通用数据类型派发不同浮点类型的ELU反向传播计算

    auto negcoef = alpha.to<float>() * scale.to<float>();
    // 计算负部系数，alpha乘以缩放比例得到

    auto poscoef = scale.to<float>();
    // 计算正部系数，直接使用缩放比例

    auto negiptcoef = input_scale.to<float>();
    // 计算输入缩放系数，将输入的缩放比例转换为浮点数

    const Vectorized<float> negcoef_vec(negcoef);
    // 创建负部系数的向量化对象

    const Vectorized<float> negiptcoef_vec(negiptcoef);
    // 创建输入缩放系数的向量化对象

    const Vectorized<float> poscoef_vec(poscoef);
    // 创建正部系数的向量化对象

    const Vectorized<float> zero_vec(static_cast<float>(0));
    // 创建零向量化对象，用于比较操作中

    cpu_kernel_vec(
        it,
        // 在CPU上执行向量化计算，处理每个元素对的ELU反向传播
        [negcoef, negiptcoef, poscoef, is_result](scalar_t a, scalar_t b) -> scalar_t {
          if (is_result) {
            // 如果是结果张量，根据b是否小于等于零返回计算结果
            return float(b) <= float(0) ? float(a) * negiptcoef * (float(b) + negcoef) : float(a) * poscoef;
          } else {
            // 如果不是结果张量，根据b是否小于等于零返回计算结果
            return float(b) <= float(0) ? float(a) * negiptcoef * negcoef * std::exp(float(b) * negiptcoef): float(a) * poscoef;
          }
        },
        // 使用向量化的计算方式处理每对向量化的输入张量a和b
        [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &zero_vec, is_result](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          auto [b0, b1] = convert_to_float<scalar_t>(b);
          auto cmp0 = (b0 > zero_vec);
          auto cmp1 = (b1 > zero_vec);
          auto get_res_masked = [&](Vectorized<float>& cmp, Vectorized<float>& a, Vectorized<float>& b) {
            if (is_result) {
              return !cmp.zero_mask() ? a * poscoef_vec :
                Vectorized<float>::blendv(a * negiptcoef_vec * (b + negcoef_vec), a * poscoef_vec, cmp);
            } else {
              return Vectorized<float>::blendv(a * negiptcoef_vec * negcoef_vec * (b * negiptcoef_vec).exp(), a * poscoef_vec, cmp);
            }
          };
          auto res0 = get_res_masked(cmp0, a0, b0);
          auto res1 = get_res_masked(cmp1, a1, b1);
          return convert_from_float<scalar_t>(res0, res1);
        });
    });
  } else {
    // 使用 AT_DISPATCH_FLOATING_TYPES 宏来处理浮点类型，生成名为 "elu_backward_cpu" 的分发器函数
    AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_backward_cpu", [&]() {
      // 定义 Vectorized<scalar_t> 为 Vec
      using Vec = Vectorized<scalar_t>;
      // 计算负数和正数系数
      auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
      auto poscoef = scale.to<scalar_t>();
      auto negiptcoef = input_scale.to<scalar_t>();
      // 创建 Vectorized 类型的常量向量
      const Vec negcoef_vec(negcoef);
      const Vec negiptcoef_vec(negiptcoef);
      const Vec poscoef_vec(poscoef);
      const Vec zero_vec(static_cast<scalar_t>(0));
      // 调用 cpu_kernel_vec 函数进行向量化 CPU 计算
      cpu_kernel_vec(
          it,
          // lambda 函数定义，根据 is_result 参数不同返回不同计算结果
          [negcoef, negiptcoef, poscoef, is_result](scalar_t a, scalar_t b) -> scalar_t {
            if (is_result) {
              // 如果是结果计算，则根据 b 的值返回不同的计算结果
              return b <= scalar_t(0) ? a * negiptcoef * (b + negcoef) : a * poscoef;
            } else {
              // 如果不是结果计算，则同样根据 b 的值返回不同的计算结果
              return b <= scalar_t(0) ? a * negiptcoef * negcoef * std::exp(b * negiptcoef) : a * poscoef;
            }
          },
          // 向量化版本的 lambda 函数，根据 is_result 参数不同进行向量化计算
          [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &zero_vec, is_result](Vec a, Vec b) -> Vec {
            // 比较操作，生成掩码 cmp
            auto cmp = (b > zero_vec);
            if (is_result) {
              if (!cmp.zero_mask()) {  // 只有计算 a * poscoef 的部分需要执行，这是非常快速的
                return a * poscoef_vec;
              } else {
                // 根据掩码选择不同计算路径
                return Vec::blendv(a * negiptcoef_vec * (b + negcoef_vec), a * poscoef_vec, cmp);
              }
            } else {
              // 根据掩码选择不同计算路径
              return Vec::blendv(a * negiptcoef_vec * negcoef_vec * (b * negiptcoef_vec).exp(), a * poscoef_vec, cmp);
            }
          }
      );
    });
}

// TODO(yangxm): Add another fast kernel using formula
// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
// and the fast tanh impl from Eigen.

// 实现 GELU 内核函数，通过 TensorIteratorBase 处理，使用指定的近似方法
void GeluKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  auto grain_size = at::internal::GRAIN_SIZE;  // 获取当前的计算粒度

  // Numbers based on benchmarking.
  // Benchmark: benchmarks/operator_benchmarks/pt/gelu_test.py

#ifdef C10_MOBILE
  // Benchmarked on S8 US phone.
  // Internal benchmarking that converts operator benchmark into
  // a torchscript module and run that on mobile.
  // Same benchmark as server side.
  // 在 S8 美国手机上进行基准测试
  // 将操作基准测试转换为 TorchScript 模块，在移动设备上运行
  // 与服务器端相同的基准测试
  constexpr int64_t GELU_MIN_ELEMENTS_FOR_MULTI_THREADING{6144};
#else
  // Benchmarked on i9 8 core 16 thread machine.
  // 1 thread: cd benchmark/operator_benchmarks;
  //           python -m pt.gelu_test --tag_filter long --omp_num_threads 1
  // 2 threads: cd benchmark/operator_benchmarks;
  //           python -m pt.gelu_test --tag_filter long --omp_num_threads 1
  // 在 i9 8 核 16 线程机器上进行基准测试
  // 1 线程：cd benchmark/operator_benchmarks;
  //         python -m pt.gelu_test --tag_filter long --omp_num_threads 1
  // 2 线程：cd benchmark/operator_benchmarks;
  //         python -m pt.gelu_test --tag_filter long --omp_num_threads 1
  constexpr int64_t GELU_MIN_ELEMENTS_FOR_MULTI_THREADING{16384};
#endif

  // 根据元素数量决定是否启用多线程
  if (it.numel() > GELU_MIN_ELEMENTS_FOR_MULTI_THREADING) {
    grain_size = it.numel() / at::get_num_threads();
  }

  // 根据选择的近似类型执行不同的 GELU 计算方式
  if (approximate == GeluType::Tanh) {
    // 如果数据类型为减少浮点数类型
    if (at::isReducedFloatingType(it.common_dtype())) {
      // 使用 SIMD 向量化操作处理计算
      AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "GeluKernelImpl", [&]() {
        // 预定义常量向量化
        auto kBetaVec = Vectorized<float>((float)(M_SQRT2 * M_2_SQRTPI * 0.5));
        auto kKappaVec = Vectorized<float>((float)(0.044715));
        auto kOneVec = Vectorized<float>((float)(1));
        auto kPointFiveVec = Vectorized<float>((float)(0.5));

        // 调用 CPU 核函数处理向量化计算
        cpu_kernel_vec(
            it,
            [](scalar_t x) -> scalar_t {
              // 定义常量并执行 GELU 近似计算
              const float kBeta = float(M_SQRT2 * M_2_SQRTPI * 0.5);
              const float kKappa = float(0.044715);
              float x_cube = float(x) * float(x) * float(x);
              float inner = kBeta * (float(x) + kKappa * x_cube);
              return float(0.5) * float(x) * (float(1) + std::tanh(inner));
            },
            [&](Vectorized<scalar_t> x) -> Vectorized<scalar_t> {
              // 执行 SIMD 向量化计算
              auto [x0, x1] = convert_to_float<scalar_t>(x);
              auto x0_cube = x0 * x0 * x0;
              auto x1_cube = x1 * x1 * x1;
              auto inner_vec0 = kBetaVec * (x0 + kKappaVec * x0_cube);
              auto inner_vec1 = kBetaVec * (x1 + kKappaVec * x1_cube);
              auto res0 = kPointFiveVec * x0 * (kOneVec + inner_vec0.tanh());
              auto res1 = kPointFiveVec * x1 * (kOneVec + inner_vec1.tanh());
              return convert_from_float<scalar_t>(res0, res1);
            },
            grain_size);
      });
    }
  }
}
    } else {
      // 如果不满足特定的浮点类型条件，执行以下代码块
      AT_DISPATCH_FLOATING_TYPES(
          it.dtype(), "GeluKernelImpl", [&]() {
        // 使用指定的数据类型scalar_t和指定的内核函数"GeluKernelImpl"进行调度
        using Vec = vec::Vectorized<scalar_t>;
        // 定义常量向量化对象
        const Vec kBetaVec(scalar_t(M_SQRT2 * M_2_SQRTPI * 0.5));
        const Vec kKappaVec(scalar_t(0.044715));
        const Vec kOneVec(scalar_t(1));
        const Vec kPointFiveVec(scalar_t(0.5));
        // 调用CPU向量化内核函数
        cpu_kernel_vec(
            it,
            [](scalar_t x) {
              // 定义局部变量和表达式
              const scalar_t kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
              const scalar_t kKappa = 0.044715;
              auto x_cube = x * x * x;
              auto inner = kBeta * (x + kKappa * x_cube);
              return scalar_t(0.5) * x * (scalar_t(1) + std::tanh(inner));
            },
            [&](Vec x_vec) {
              // 向量化计算
              auto x_cube = x_vec * x_vec * x_vec;
              auto inner_vec = kBetaVec * (x_vec + kKappaVec * x_cube);
              return kPointFiveVec * x_vec * (kOneVec + inner_vec.tanh());
            },
            grain_size);
      });
    }
    } else {
      // 如果不是降低的浮点类型，执行以下代码块
      if (at::isReducedFloatingType(it.common_dtype())) {
        // 判断是否为降低浮点类型，使用指定的数据类型scalar_t和指定的内核函数"GeluKernelImpl"进行调度
        AT_DISPATCH_REDUCED_FLOATING_TYPES(it.dtype(), "GeluKernelImpl", [&]() {
          // 定义向量化的常量对象
          auto kAlphaVec = Vectorized<float>((float)(M_SQRT1_2));
          auto kOneVec = Vectorized<float>((float)(1));
          auto kPointFiveVec = Vectorized<float>((float)(0.5));
          // 调用CPU向量化内核函数
          cpu_kernel_vec(
              it,
              [](scalar_t x) -> scalar_t {
                // 定义局部变量和表达式
                const float kAlpha = float(M_SQRT1_2);
                return float(x) * float(0.5) * (float(1) + std::erf(float(x) * kAlpha));
              },
              [&](Vectorized<scalar_t> x) -> Vectorized<scalar_t> {
                // 向量化计算
                auto [x0, x1] = convert_to_float<scalar_t>(x);
                auto res0 = x0 * kPointFiveVec * (kOneVec + (x0 * kAlphaVec).erf());
                auto res1 = x1 * kPointFiveVec * (kOneVec + (x1 * kAlphaVec).erf());
                return convert_from_float<scalar_t>(res0, res1);
              },
              grain_size);
        });
      } else {
        // 否则，执行以下代码块
        AT_DISPATCH_FLOATING_TYPES(
            it.dtype(), "GeluKernelImpl", [&]() {
          // 使用指定的数据类型scalar_t和指定的内核函数"GeluKernelImpl"进行调度
          using Vec = vec::Vectorized<scalar_t>;
          const Vec kAlphaVec(scalar_t(M_SQRT1_2));
          const Vec kOneVec(scalar_t(1));
          const Vec kPointFiveVec(scalar_t(0.5));
          // 调用CPU向量化内核函数
          cpu_kernel_vec(
              it,
              [](scalar_t x) {
                // 定义局部变量和表达式
                const scalar_t kAlpha = scalar_t(M_SQRT1_2);
                return x * scalar_t(0.5) * (scalar_t(1) + std::erf(x * kAlpha));
              },
              [&](Vec x_vec) {
                // 向量化计算
                return x_vec * kPointFiveVec *
                    (kOneVec + (x_vec * kAlphaVec).erf());
              },
              grain_size);
        });
      }
    }
}

// GeluBackwardKernelImpl 函数实现，用于反向传播 GELU 激活函数的梯度
void GeluBackwardKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  // 如果使用的是 Tanh 近似方法
  if (approximate == GeluType::Tanh) {
    // 空的条件分支，因为这个分支下没有具体的代码
    // 可能是为了保留将来扩展的可能性而保留的空分支
  } else {
    // 根据输入张量的数据类型分发不同的实现
    AT_DISPATCH_FLOATING_TYPES(
        it.dtype(), "GeluBackwardKernelImpl", [&]() {
      // 使用 Vectorized 宏定义的向量类型进行计算
      using Vec = vec::Vectorized<scalar_t>;
      // 定义常数向量
      const Vec kBetaVec(scalar_t(M_SQRT2 * M_2_SQRTPI * 0.5));
      const Vec kKappaVec(scalar_t(0.044715));
      const Vec kOneVec(scalar_t(1));
      const Vec kThreeVec(scalar_t(3));
      const Vec kPointFiveVec(scalar_t(0.5));
      
      // 调用 CPU 内核向量化函数进行计算
      cpu_kernel_vec(
          it,
          // 匿名函数，计算每个元素的梯度
          [](scalar_t dy, scalar_t x) {
            // 定义 GELU 函数的常数
            const scalar_t kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
            const scalar_t kKappa = 0.044715;
            auto x_sq = x * x;
            auto x_cube = x_sq * x;
            auto inner = kBeta * (x + kKappa * x_cube);
            auto tanh_inner = std::tanh(inner);

            auto left = scalar_t(0.5) * x;
            auto right = scalar_t(1) + tanh_inner;

            auto left_derivative = scalar_t(0.5) * right;

            auto tanh_derivative = scalar_t(1) - tanh_inner * tanh_inner;
            auto inner_derivative =
              kBeta * (scalar_t(1) + scalar_t(3) * kKappa * x_sq);
            auto right_derivative = left * tanh_derivative * inner_derivative;

            return dy * (left_derivative + right_derivative);
          },
          // 向量化计算的部分
          [&](Vec dy_vec, Vec x_vec) {
            auto x_sq = x_vec * x_vec;
            auto x_cube = x_vec * x_vec * x_vec;
            auto inner_vec =
                kBetaVec * (x_vec + kKappaVec * x_cube);
            auto tanh_inner_vec = inner_vec.tanh();

            auto left_vec = kPointFiveVec * x_vec;
            auto right_vec = kOneVec + tanh_inner_vec;

            auto left_derivative_vec = kPointFiveVec * right_vec;

            auto tanh_derivative_vec =
                kOneVec - tanh_inner_vec * tanh_inner_vec;
            auto inner_derivative_vec =
                kBetaVec * (kOneVec + kThreeVec * kKappaVec * x_sq);
            auto right_derivative_vec =
                left_vec * tanh_derivative_vec * inner_derivative_vec;

            return dy_vec * (left_derivative_vec + right_derivative_vec);
          });
    });
  }
}  // end of GeluBackwardKernelImpl
    // 检查是否为降维浮点类型，根据结果选择不同的处理方式
    if (at::isReducedFloatingType(it.common_dtype())) {
      // 在降维浮点类型的上下文中执行以下操作
      AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "GeluBackwardKernelImpl", [&]() {
        // 定义常量向量 kAlphaVec，其值为 sqrt(1/2)
        auto kAlphaVec = Vectorized<float>((float)(M_SQRT1_2));
        // 定义常量向量 kBetaVec，其值为 2/sqrt(pi) * sqrt(1/2) * 0.5
        auto kBetaVec = Vectorized<float>((float)(M_2_SQRTPI * M_SQRT1_2 * 0.5));
        // 定义常量向量 kOneVec，其值为 1
        auto kOneVec = Vectorized<float>((float)(1));
        // 定义常量向量 kPointFiveVec，其值为 0.5
        auto kPointFiveVec = Vectorized<float>((float)(0.5));
        // 定义常量向量 kMinusPointFiveVec，其值为 -0.5
        auto kMinusPointFiveVec = Vectorized<float>((float)(-0.5));
        // 调用 CPU 内核向量化函数，对每个元素进行计算
        cpu_kernel_vec(
            it,
            // 内部 lambda 函数计算梯度下降的处理方式
            [](scalar_t dy, scalar_t x) -> scalar_t {
                // 定义常量 kAlpha，其值为 sqrt(1/2)
                const float kAlpha = float(M_SQRT1_2);
                // 定义常量 kBeta，其值为 2/sqrt(pi) * sqrt(1/2) * 0.5
                const float kBeta = float(M_2_SQRTPI) * float(M_SQRT1_2) * float(0.5);
                // 计算累积分布函数 CDF
                const float cdf =
                    float(0.5) * (float(1) + std::erf(float(x) * kAlpha));
                // 计算概率密度函数 PDF
                const float pdf = kBeta * std::exp(float(x) * float(x) * float(-0.5));
                // 返回梯度下降的计算结果
                return float(dy) * (cdf + float(x) * pdf);
            },
            // 内部 lambda 函数处理向量化的梯度下降
            [&](Vectorized<scalar_t> dy, Vectorized<scalar_t> x) -> Vectorized<scalar_t> {
                // 将向量 x 转换为浮点数向量
                auto [x0, x1] = convert_to_float<scalar_t>(x);
                // 将向量 dy 转换为浮点数向量
                auto [dy0, dy1] = convert_to_float<scalar_t>(dy);
                // 计算向量化的 CDF 值
                auto cdf_vec0 = kPointFiveVec * (kOneVec + (x0 * kAlphaVec).erf());
                auto cdf_vec1 = kPointFiveVec * (kOneVec + (x1 * kAlphaVec).erf());
                // 计算向量化的 PDF 值
                auto pdf_vec0 = kBetaVec * (x0 * x0 * kMinusPointFiveVec).exp();
                auto pdf_vec1 = kBetaVec * (x1 * x1 * kMinusPointFiveVec).exp();
                // 计算向量化的结果
                auto res0 = dy0 * (cdf_vec0 + x0 * pdf_vec0);
                auto res1 = dy1 * (cdf_vec1 + x1 * pdf_vec1);
                // 将浮点数向量结果转换回原始类型的向量
                return convert_from_float<scalar_t>(res0, res1);
            });
      });
    } else {
      // 在普通浮点类型的上下文中执行以下操作
      AT_DISPATCH_FLOATING_TYPES(
          it.dtype(), "GeluBackwardKernelImpl", [&]() {
        // 使用 Vec 类型的向量定义 kAlphaVec 常量，其值为 sqrt(1/2)
        using Vec = vec::Vectorized<scalar_t>;
        const Vec kAlphaVec(scalar_t(M_SQRT1_2));
        // 使用 Vec 类型的向量定义 kBetaVec 常量，其值为 2/sqrt(pi) * sqrt(1/2) * 0.5
        const Vec kBetaVec(scalar_t(M_2_SQRTPI * M_SQRT1_2 * 0.5));
        // 使用 Vec 类型的向量定义 kOneVec 常量，其值为 1
        const Vec kOneVec(scalar_t(1));
        // 使用 Vec 类型的向量定义 kPointFiveVec 常量，其值为 0.5
        const Vec kPointFiveVec(scalar_t(0.5));
        // 使用 Vec 类型的向量定义 kMinusPointFiveVec 常量，其值为 -0.5
        const Vec kMinusPointFiveVec(scalar_t(-0.5));
        // 调用 CPU 内核向量化函数，对每个元素进行计算
        cpu_kernel_vec(
            it,
            // 内部 lambda 函数计算梯度下降的处理方式
            [](scalar_t dy, scalar_t x) {
              // 定义常量 kAlpha，其值为 sqrt(1/2)
              const scalar_t kAlpha = scalar_t(M_SQRT1_2);
              // 定义常量 kBeta，其值为 2/sqrt(pi) * sqrt(1/2) * 0.5
              const scalar_t kBeta = M_2_SQRTPI * M_SQRT1_2 * scalar_t(0.5);
              // 计算累积分布函数 CDF
              const scalar_t cdf =
                  scalar_t(0.5) * (scalar_t(1) + std::erf(x * kAlpha));
              // 计算概率密度函数 PDF
              const scalar_t pdf = kBeta * std::exp(x * x * scalar_t(-0.5));
              // 返回梯度下降的计算结果
              return dy * (cdf + x * pdf);
            },
            // 内部 lambda 函数处理向量化的梯度下降
            [&](Vec dy_vec, Vec x_vec) {
              // 计算向量化的 CDF 值
              const Vec cdf_vec =
                  kPointFiveVec * (kOneVec + (x_vec * kAlphaVec).erf());
              // 计算向量化的 PDF 值
              const Vec pdf_vec =
                  kBetaVec * (x_vec * x_vec * kMinusPointFiveVec).exp();
              // 返回向量化的结果
              return dy_vec * (cdf_vec + x_vec * pdf_vec);
            });
      });
    }
}

// 定义 hardsigmoid_kernel 函数，处理给定的张量迭代器
void hardsigmoid_kernel(TensorIteratorBase& iter) {
  // 检查张量迭代器的数据类型是否为缩减浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用 AT_DISPATCH_REDUCED_FLOATING_TYPES 宏根据迭代器的数据类型分发函数
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardsigmoid_cpu", [&]() {
      // 定义常量和向量化类型
      const float zero(0.0f);
      const float three(3.0f);
      const float six(6.0f);
      using Vec = vec::Vectorized<float>;
      const Vec kZeroVec(zero);
      const Vec kThreeVec(three);
      const Vec kSixVec(six);
      // 调用 cpu_kernel_vec 处理迭代器
      cpu_kernel_vec(
          iter,
          // 标量版本的处理函数
          [&](scalar_t self_val) -> scalar_t {
            return std::min(std::max(float(self_val) + three, zero), six) / six;
          },
          // 向量化版本的处理函数
          [&](vec::Vectorized<scalar_t> self_val) -> vec::Vectorized<scalar_t> {
            auto [self_val0, self_val1] = convert_to_float<scalar_t>(self_val);
            self_val0 = minimum(
              maximum(self_val0 + kThreeVec, kZeroVec),
              kSixVec
            ) / kSixVec;
            self_val1 = minimum(
              maximum(self_val1 + kThreeVec, kZeroVec),
              kSixVec
            ) / kSixVec;
            return convert_from_float<scalar_t>(self_val0, self_val1);
          });
    });
  } else {
    // 处理非缩减浮点类型的情况
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardsigmoid_cpu", [&] {
      // 定义常量和向量化类型
      const scalar_t zero(0.0f);
      const scalar_t three(3.0f);
      const scalar_t six(6.0f);
      using Vec = vec::Vectorized<scalar_t>;
      const Vec kZeroVec(zero);
      const Vec kThreeVec(three);
      const Vec kSixVec(six);
      // 调用 cpu_kernel_vec 处理迭代器
      cpu_kernel_vec(
          iter,
          // 标量版本的处理函数
          [&](scalar_t self_val) {
            return std::min(std::max(self_val + three, zero), six) / six;
          },
          // 向量化版本的处理函数
          [&](Vec self_val) {
            return vec::minimum(
              vec::maximum(self_val + kThreeVec, kZeroVec),
              kSixVec
            ) / kSixVec;
          });
    });
  }
}

// 定义 hardsigmoid_backward_kernel 函数，处理给定的张量迭代器
void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  // 检查迭代器的公共数据类型是否为缩减浮点类型
  if (at::isReducedFloatingType(iter.common_dtype())) {
    // 使用 AT_DISPATCH_REDUCED_FLOATING_TYPES 宏根据公共数据类型分发函数
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.common_dtype(), "hardsigmoid_backward", [&]() {
      // 定义常量和向量化类型
      const float zero(0.0f);
      const float three(3.0f);
      const float neg_three(-3.0f);
      const float one_sixth(1.0f / 6.0f);
      using Vec = Vectorized<float>;
      Vec kZeroVec(0.0f);
      Vec kOneSixthVec(1.0f / 6.0f);
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) -> scalar_t {
          // 如果 self_val 的值在 -3 和 3 之间，则返回 grad_val 乘以 1/6，否则返回 0
          return (float(self_val) > neg_three && float(self_val) < three)
            ? float(grad_val) * one_sixth
            : zero;
        },
        [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
          // 将 Vectorized 类型的 self_val 和 grad_val 转换为 float 类型
          auto [self_val0, self_val1] = convert_to_float<scalar_t>(self_val);
          auto [grad_val0, grad_val1] = convert_to_float<scalar_t>(grad_val);
          // 创建一个向量化的掩码，标记 self_val0 中在 -3 和 3 之间的元素
          Vec gradNonZeroMask = (self_val0 > neg_three) & (self_val0 < three);
          // 根据掩码，选择性地应用 grad_val0 * kOneSixthVec 或者 kZeroVec 到 self_val0
          self_val0 = Vec::blendv(kZeroVec, grad_val0 * kOneSixthVec, gradNonZeroMask);
          // 创建一个向量化的掩码，标记 self_val1 中在 -3 和 3 之间的元素
          gradNonZeroMask = (self_val1 > neg_three) & (self_val1 < three);
          // 根据掩码，选择性地应用 grad_val1 * kOneSixthVec 或者 kZeroVec 到 self_val1
          self_val1 = Vec::blendv(kZeroVec, grad_val1 * kOneSixthVec, gradNonZeroMask);
          // 将处理后的 self_val0 和 self_val1 转换回 Vectorized<scalar_t> 类型并返回
          return convert_from_float<scalar_t>(self_val0, self_val1);
        });
    });
  } else {
    // 如果 iter 的数据类型是浮点类型，则执行以下代码
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardsigmoid_backward", [&] {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t neg_three(-3.0f);
    const scalar_t one_sixth(1.0f / 6.0f);
    using Vec = Vectorized<scalar_t>;
    Vec kZeroVec(0.0f);
    Vec kOneSixthVec(1.0f / 6.0f);
    // 调用 cpu_kernel_vec 函数
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) {
          // 如果 self_val 的值在 -3 和 3 之间，则返回 grad_val 乘以 1/6，否则返回 0
          return (self_val > neg_three && self_val < three)
            ? grad_val * one_sixth
            : zero;
        },
        [=](Vec grad_val, Vec self_val) {
          // 创建一个向量化的掩码，标记 self_val 中在 -3 和 3 之间的元素
          Vec gradNonZeroMask = (self_val > neg_three) & (self_val < three);
          // 根据掩码，选择性地应用 grad_val * kOneSixthVec 或者 kZeroVec 到 grad_val
          return Vec::blendv(kZeroVec, grad_val * kOneSixthVec, gradNonZeroMask);
        });
  });
  }
void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏展开，以处理所有浮点数类型和 kBFloat16、kHalf 类型
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "hardshrink_cpu", [&] {
    // 将 lambd 转换为当前类型的标量值
    auto lambd_val = lambd.to<scalar_t>();
    // 定义 Vectorized 类型别名为 Vec，以便于使用向量化指令
    using Vec = Vectorized<scalar_t>;
    // 调用 cpu_kernel_vec 函数处理迭代器 iter
    cpu_kernel_vec(
        iter,
        // 标量操作：对每个 self_val 执行硬阈值函数
        [=](scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : self_val;
        },
        // 向量操作：对每个向量化 self_val 执行硬阈值函数
        [=](Vec self_val) {
          return Vec::blendv(self_val, Vec(0), (self_val >= -lambd_val) & (self_val <= lambd_val));
        });
  });
}

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  // 检查迭代器的数据类型是否是降低浮点类型（reduced floating type）
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用 AT_DISPATCH_REDUCED_FLOATING_TYPES 宏展开，处理所有降低的浮点数类型
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.common_dtype(), "softshrink_cpu", [&]() {
    // 将 lambd 转换为 float 类型的标量值
    auto lambd_val = lambd.to<float>();
    // 创建 Vectorized<float> 类型的 lambdVec
    auto lambdVec = Vectorized<float>(lambd_val);
    // 调用 cpu_kernel_vec 函数处理迭代器 iter
    cpu_kernel_vec(
      iter,
      // 标量操作：对每个标量 a 执行软阈值函数
      [=](scalar_t a) -> scalar_t {
        return float(a) > lambd_val ? a - lambd_val : (float(a) < -lambd_val ? a + lambd_val : float(0));
      },
      // 向量操作：对每个向量化 self_val 执行软阈值函数
      [=](Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
          auto [self_val0, self_val1] = convert_to_float<scalar_t>(self_val);
          auto self_val_t0 = convert_from_float<scalar_t>((self_val0 > lambdVec) & (self_val0 - lambdVec), (self_val1 > lambdVec) & (self_val1 - lambdVec));
          auto self_val_t1 = convert_from_float<scalar_t>((self_val0 < -lambd_val) & (self_val0 + lambdVec), (self_val1 < -lambd_val) & (self_val1 + lambdVec));
          return (self_val_t0 | self_val_t1);
      });
    });
  } else {
    // 使用 AT_DISPATCH_FLOATING_TYPES 宏展开，处理所有浮点数类型
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softshrink_cpu", [&]() {
    // 将 lambd 转换为当前类型的标量值
    auto lambd_val = lambd.to<scalar_t>();
    // 创建 Vectorized<scalar_t> 类型的 lambdVec
    auto lambdVec = Vectorized<scalar_t>(lambd_val);
    // 调用 cpu_kernel_vec 函数处理迭代器 iter
    cpu_kernel_vec(
      iter,
      // 标量操作：对每个标量 a 执行软阈值函数
      [=](scalar_t a) -> scalar_t {
        return a > lambd_val ? a - lambd_val : (a < -lambd_val ? a + lambd_val : scalar_t(0));
      },
      // 向量操作：对每个向量化 self_val 执行软阈值函数
      [=](Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
          Vectorized<scalar_t> self_val_t0, self_val_t1;
          self_val_t0 = (self_val > lambdVec) & (self_val - lambdVec);
          self_val_t1 = (self_val < -lambd_val) & (self_val + lambdVec);
          return (self_val_t0 | self_val_t1);
      });
  });
  }
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏展开，处理所有浮点数类型和 kBFloat16、kHalf 类型
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "shrink_backward_cpu", [&] {
    // 将 lambd 转换为当前类型的标量值
    auto lambd_val = lambd.to<scalar_t>();
    // 调用 cpu_kernel_vec 函数处理迭代器 iter
    cpu_kernel_vec(
        iter,
        // 标量操作：对每个 grad_val 和 self_val 执行收缩反向函数
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : grad_val;
        },
        // 向量操作：对每个向量化 grad_val 和 self_val 执行收缩反向函数
        [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) {
          return ((self_val < -lambd_val) | (self_val > lambd_val)) & grad_val;
        });
  });
}
void hardtanh_backward_kernel(TensorIterator& iter, const Scalar& min, const Scalar& max) {
  // 检查迭代器的数据类型是否是减少后的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 根据实际浮点类型分发具体的 CPU 计算内核
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardshrink_backward_cpu", [&]() {
      // 将最小值和最大值转换为 float 类型
      auto min_val = min.to<float>();
      auto max_val = max.to<float>();
      // 调用 CPU 向量化内核函数，对每个元素执行操作
      cpu_kernel_vec(
          iter,
          // 根据条件返回梯度值或零
          [=](scalar_t grad_val, scalar_t self_val) -> scalar_t {
            return (float(self_val) <= min_val || float(self_val) >= max_val) ? scalar_t(0) : grad_val;
          },
          // 向量化版本的处理函数，处理多个元素
          [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
            auto [grad_val0, grad_val1] = convert_to_float<scalar_t>(grad_val);
            auto [self_val0, self_val1] = convert_to_float<scalar_t>(self_val);
            // 返回处理后的向量化结果
            return convert_from_float<scalar_t>(
              ((self_val0 > min_val) & (self_val0 < max_val)) & grad_val0,
              ((self_val1 > min_val) & (self_val1 < max_val)) & grad_val1
            );
          });
    });
  } else {
    // 如果数据类型是浮点类型而非减少后的浮点类型，则执行以下操作
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_backward_cpu", [&] {
    auto min_val = min.to<scalar_t>();
    auto max_val = max.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        // 根据条件返回梯度值或零
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val <= min_val || self_val >= max_val) ? scalar_t(0) : grad_val;
        },
        // 向量化版本的处理函数，处理多个元素
        [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) {
          return ((self_val > min_val) & (self_val < max_val)) & grad_val;
        });
  });
  }
}

void hardswish_kernel(TensorIterator& iter) {
  // 检查迭代器的数据类型是否是减少后的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 根据实际浮点类型分发具体的 CPU 计算内核
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardswish_cpu", [&]() {
    const float zero(0.0f);
    const float three(3.0f);
    const float six(6.0f);
    using Vec = vec::Vectorized<float>;
    // 定义常量向量化操作需要的值
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kSixVec(six);
    // 调用 CPU 向量化内核函数，对每个元素执行操作
    cpu_kernel_vec(
      iter,
      // 根据硬 swish 函数的定义，计算每个元素的结果
      [&](scalar_t x) -> scalar_t {
        return float(x) * std::min(std::max(float(x) + three, zero), six) / six;
      },
      // 向量化版本的处理函数，处理多个元素
      [&](vec::Vectorized<scalar_t> x_vec) {
        auto [x_vec0, x_vec1] = convert_to_float<scalar_t>(x_vec);
        // 执行硬 swish 函数的向量化计算
        x_vec0 = x_vec0 * minimum(
          maximum(x_vec0 + kThreeVec, kZeroVec),
          kSixVec
        ) / kSixVec;
        x_vec1 = x_vec1 * minimum(
          maximum(x_vec1 + kThreeVec, kZeroVec),
          kSixVec
        ) / kSixVec;
        return convert_from_float<scalar_t>(x_vec0, x_vec1);
      });
    });
  } else {
    // 如果数据类型是浮点类型而非减少后的浮点类型，则执行以下操作
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardswish_cpu", [&]() {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t six(6.0f);
    using Vec = vec::Vectorized<scalar_t>;
    // 定义常量向量化操作需要的值
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kSixVec(six);
    cpu_kernel_vec(
      iter,
      [&](scalar_t x) {  // 定义lambda函数，参数为scalar_t类型x，执行以下操作
        // 返回x乘以在[min(x + three, zero)，six]范围内的值，再除以six
        return x * std::min(std::max(x + three, zero), six) / six;
      },
      [&](Vec x_vec) {  // 定义lambda函数，参数为Vec类型x_vec，执行以下操作
        // 返回x_vec乘以在[min(x_vec + kThreeVec, kZeroVec)，kSixVec]范围内的值，再除以kSixVec
        return x_vec * vec::minimum(
          vec::maximum(x_vec + kThreeVec, kZeroVec),
          kSixVec
        ) / kSixVec;
      }
    );
  });
  }
// 定义硬切函数的反向传播内核函数，接受一个张量迭代器作为参数
void hardswish_backward_kernel(TensorIterator& iter) {
  // 如果张量的数据类型是减少的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 根据张量的数据类型分发到相应的处理函数上下文中
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardswish_backward_cpu", [&]() {
      // 定义一些常量值
      const float zero(0.0f);
      const float three(3.0f);
      const float neg_three(-3.0f);
      const float one_half(0.5f);
      // 使用 Vectorized 类来处理向量化操作
      using Vec = vec::Vectorized<float>;
      // 定义一些常用的向量常量
      const Vec kZeroVec(zero);
      const Vec kThreeVec(three);
      const Vec kNegThreeVec(neg_three);
      const Vec kOneHalfVec(one_half);
      // 调用 CPU 内核函数处理张量迭代器
      cpu_kernel_vec(
        iter,
        // 逐元素处理张量，计算梯度的反向传播
        [&](scalar_t grad_val, scalar_t self_val) -> scalar_t {
          if (float(self_val) < neg_three) {
            return zero;
          } else if (float(self_val) <= three) {
            return float(grad_val) * ((float(self_val) / three) + one_half);
          } else {
            return grad_val;
          }
        },
        // 处理向量化的张量，加速计算
        [&](vec::Vectorized<scalar_t> grad_val, vec::Vectorized<scalar_t> self_val) {
          auto [self_val0, self_val1] = convert_to_float<scalar_t>(self_val);
          auto [grad_val0, grad_val1] = convert_to_float<scalar_t>(grad_val);
          // 使用向量化处理，根据条件混合计算结果
          self_val0 = Vec::blendv(
            Vec::blendv(
              grad_val0 * ((self_val0 / kThreeVec) + kOneHalfVec),
              grad_val0,
              self_val0 >= kThreeVec
            ),
            kZeroVec,
            self_val0 < kNegThreeVec
          );
          self_val1 = Vec::blendv(
            Vec::blendv(
              grad_val1 * ((self_val1 / kThreeVec) + kOneHalfVec),
              grad_val1,
              self_val1 >= kThreeVec
            ),
            kZeroVec,
            self_val1 < kNegThreeVec
          );
          return convert_from_float<scalar_t>(self_val0, self_val1);
        }
      );
    });
  } else {
    // 如果张量的数据类型是浮点类型，但不是减少的浮点类型，进入这个分支
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardswish_backward_cpu", [&]() {
      // 定义一些常量值
      const scalar_t zero(0.0f);
      const scalar_t three(3.0f);
      const scalar_t neg_three(-3.0f);
      const scalar_t one_half(0.5f);
      // 使用 Vectorized 类来处理向量化操作
      using Vec = vec::Vectorized<scalar_t>;
      // 定义一些常用的向量常量
      const Vec kZeroVec(zero);
      const Vec kThreeVec(three);
      const Vec kNegThreeVec(neg_three);
      const Vec kOneHalfVec(one_half);
      // 调用 CPU 内核函数处理张量迭代器
      cpu_kernel_vec(
        iter,
        // 逐元素处理张量，计算梯度的反向传播
        [&](scalar_t grad_val, scalar_t self_val) {
          if (self_val < neg_three) {
            return zero;
          } else if (self_val <= three) {
            return grad_val * ((self_val / three) + one_half);
          } else {
            return grad_val;
          }
        },
        // 处理向量化的张量，加速计算
        [&](Vec grad_val, Vec self_val) {
          return Vec::blendv(
            Vec::blendv(
              grad_val * ((self_val / kThreeVec) + kOneHalfVec),
              grad_val,
              self_val >= kThreeVec
            ),
            kZeroVec,
            self_val < kNegThreeVec
          );
        }
      );
    });
  }
}

// 定义静态函数，处理 leaky_relu 的内核函数，接受一个张量迭代器和负值参数作为参数
static void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  // 如果张量的数据类型是减少的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 根据张量的数据类型分发到相应的处理函数上下文中
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "leaky_relu_cpu", [&]() {
      // 定义一个向量，元素值为 0
      auto zero_vec = Vectorized<float>((float)(0));
    // 创建一个包含单个浮点数 1 的向量化对象
    auto one_vec = Vectorized<float>((float)(1));
    // 将 negval_ 转换为 float 类型
    float negval = negval_.to<float>();
    // 创建一个包含单个浮点数 negval 的向量化对象
    Vectorized<float> negval_v = Vectorized<float>(negval);
    // 调用 CPU 内核函数，处理迭代器 iter 中的每个标量值 a
    cpu_kernel_vec(
        iter,
        // 对每个标量 a 执行以下操作并返回结果标量
        [&](scalar_t a) -> scalar_t {
          // 如果 a 大于 0，则返回 a；否则返回 a 乘以 negval
          return float(a) > float(0) ? float(a) : float(a) * negval;
        },
        // 对每个向量 a 执行以下操作并返回结果向量
        [&](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
          // 将向量 a 中的元素转换为 float 类型
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          // 根据 a0 和 a1 的条件选择，对向量 a 进行操作并返回结果向量
          auto res0 = a0 * (Vectorized<float>::blendv(negval_v, one_vec, a0 > zero_vec));
          auto res1 = a1 * (Vectorized<float>::blendv(negval_v, one_vec, a1 > zero_vec));
          return convert_from_float<scalar_t>(res0, res1);
        });
    // 结束 CPU 内核函数调用
    });
  } else {
    // 处理浮点数类型的迭代器 iter，使用 "leaky_relu_cpu" 分发函数
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_cpu", [&] {
      // 使用 Vectorized<scalar_t> 类型的向量 Vec
      using Vec = Vectorized<scalar_t>;
      // 创建一个包含标量 0 的向量化对象
      auto zero_vec = Vec((scalar_t)(0));
      // 创建一个包含标量 1 的向量化对象
      auto one_vec = Vec((scalar_t)(1));
      // 将 negval_ 转换为 scalar_t 类型
      scalar_t negval = negval_.to<scalar_t>();
      // 创建一个包含 negval 的向量化对象
      Vec negval_v = Vec(negval);
      // 调用 CPU 内核函数，处理迭代器 iter 中的每个向量 a
      cpu_kernel_vec(
          iter,
          // 对每个标量 a 执行以下操作并返回结果标量
          [&](scalar_t a) -> scalar_t {
            // 如果 a 大于 0，则返回 a；否则返回 a 乘以 negval
            return a > scalar_t(0) ? a : a * negval;
          },
          // 对每个向量 a 执行以下操作并返回结果向量
          [&](Vec a) -> Vec {
            // 根据 a 中元素的条件选择，创建混合向量 r
            auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
            // 返回向量 a 与混合向量 r 的乘积结果向量
            return a * r;
          });
    });
  }
}

// 定义静态函数，用于计算 Leaky ReLU 激活函数的反向传播
static void leaky_relu_backward_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  // 检查是否是降低浮点数类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用宏进行浮点数类型的调度，这里使用了 Leaky ReLU 的反向传播实现
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "leaky_relu_backward_cpu", [&]() {
      auto zero_vec = Vectorized<float>((float)(0));
      auto one_vec = Vectorized<float>((float)(1));
      float negval = negval_.to<float>();
      Vectorized<float> negval_v = Vectorized<float>(negval);
      // 在 CPU 上执行向量化操作
      cpu_kernel_vec(
        iter,
        // 点对点操作，根据条件返回不同的值
        [&](scalar_t a, scalar_t b) -> scalar_t {
          return float(a) > float(0) ? float(b) : float(b) * negval;
        },
        // 向量化版本的点对点操作
        [&](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          auto [b0, b1] = convert_to_float<scalar_t>(b);
          auto res0 = b0 * (Vectorized<float>::blendv(negval_v, one_vec, a0 > zero_vec));
          auto res1 = b1 * (Vectorized<float>::blendv(negval_v, one_vec, a1 > zero_vec));
          return convert_from_float<scalar_t>(res0, res1);
        });
    });
  } else {
    // 浮点数类型的调度，用于非降低浮点数类型
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_backward_cpu", [&] {
      using Vec = Vectorized<scalar_t>;
      auto zero_vec = Vec((scalar_t)(0));
      auto one_vec = Vec((scalar_t)(1));
      scalar_t negval = negval_.to<scalar_t>();
      Vec negval_v = Vec(negval);
      // 在 CPU 上执行向量化操作
      cpu_kernel_vec(
          iter,
          // 点对点操作，根据条件返回不同的值
          [&](scalar_t a, scalar_t b) -> scalar_t {
            return a > scalar_t(0) ? b : b * negval;
          },
          // 向量化版本的点对点操作
          [&](Vec a, Vec b) -> Vec {
            auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
            return b * r;
          });
    });
  }
}

// 定义函数，用于执行 Softplus 激活函数的计算
void softplus_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
    // 检查是否是降低浮点数类型
    if (at::isReducedFloatingType(iter.dtype())) {
    // 使用宏进行浮点数类型的调度，这里使用了 Softplus 函数的实现
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "softplus_cpu", [&]() {
      using Vec = Vectorized<float>;
      auto beta = beta_.to<float>();
      auto threshold = threshold_.to<float>();
      const Vec beta_vec(beta);
      const Vec threshold_vec(threshold);
      // 在 CPU 上执行向量化操作
      cpu_kernel_vec(
          iter,
          // 点对点操作，根据条件返回不同的值
          [beta, threshold](scalar_t a) -> scalar_t {
            return (float(a) * beta) > threshold ? a
              : static_cast<scalar_t>((std::log1p(std::exp(float(a) * beta))) / beta);
          },
          // 向量化版本的点对点操作
          [beta_vec, threshold_vec](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
            auto [a0, a1] = convert_to_float<scalar_t>(a);
            a0 = Vec::blendv((a0 * beta_vec).exp().log1p() / beta_vec, a0, (a0 * beta_vec) > threshold_vec);
            a1 = Vec::blendv((a1 * beta_vec).exp().log1p() / beta_vec, a1, (a1 * beta_vec) > threshold_vec);
            return convert_from_float<scalar_t>(a0, a1);
          }
      );
    });
  } else {
    // 浮点数类型的调度，用于非降低浮点数类型
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softplus_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    const Vec beta_vec(beta);
    const Vec threshold_vec(threshold);
    // 创建一个包含阈值的向量 threshold_vec

    cpu_kernel_vec(
        iter,
        // 对每个元素执行以下 lambda 函数
        [beta, threshold](scalar_t a) -> scalar_t {
          return (a * beta) > threshold ? a
            : static_cast<scalar_t>(std::log1p(std::exp(a * beta))) / beta;
        },
        // 对每个向量执行以下 lambda 函数
        [beta_vec, threshold_vec](Vec a) -> Vec {
          return Vec::blendv((a * beta_vec).exp().log1p() / beta_vec, a, (a * beta_vec) > threshold_vec);
        }
    );
    // 使用 cpu_kernel_vec 函数并传入两个 lambda 函数作为参数，分别处理标量和向量操作
  });
  }
  // 结束函数定义
}

// 定义 softplus_backward_kernel 函数，处理反向传播的 Softplus 操作
void softplus_backward_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  // 检查迭代器中的数据类型是否为降维后的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用宏展开，处理降维后的浮点类型
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "softplus_backward_cpu", [&]() {
    using Vec = Vectorized<float>;
    auto beta = beta_.to<float>();
    auto threshold = threshold_.to<float>();
    const Vec beta_vec(beta);
    const Vec threshold_vec(threshold);
    const Vec one_vec(static_cast<float>(1.0));
    // 调用 CPU 内核函数，处理向量化操作
    cpu_kernel_vec(
        iter,
        // 定义标量处理函数，计算 Softplus 反向传播
        [beta, threshold](scalar_t a, scalar_t b) -> scalar_t {
          float z = std::exp(float(b) * beta);
          return (float(b) * beta) > threshold ? a : static_cast<scalar_t>(float(a) * z / (z + float(1.)));
        },
        // 定义向量处理函数，使用向量化指令加速计算
        [beta_vec, one_vec, threshold_vec](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          auto [b0, b1] = convert_to_float<scalar_t>(b);
          Vec z = (b0 * beta_vec).exp();
          a0 = Vec::blendv(a0 * z / (z + one_vec), a0, (b0 * beta_vec) > threshold_vec);
          z = (b1 * beta_vec).exp();
          a1 = Vec::blendv(a1 * z / (z + one_vec), a1, (b1 * beta_vec) > threshold_vec);
          return convert_from_float<scalar_t>(a0, a1);
        });
    });
  } else {
    // 处理非降维浮点类型的情况
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softplus_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    const Vec beta_vec(beta);
    const Vec threshold_vec(threshold);
    const Vec one_vec(static_cast<scalar_t>(1.0));
    // 调用 CPU 内核函数，处理向量化操作
    cpu_kernel_vec(
        iter,
        // 定义标量处理函数，计算 Softplus 反向传播
        [beta, threshold](scalar_t a, scalar_t b) -> scalar_t {
          scalar_t z = std::exp(b * beta);
          return (b * beta) > threshold ? a : a * z / (z + scalar_t(1.));
        },
        // 定义向量处理函数，使用向量化指令加速计算
        [beta_vec, one_vec, threshold_vec](Vec a, Vec b) -> Vec {
          const Vec z = (b * beta_vec).exp();
          return Vec::blendv(a * z / (z + one_vec), a, (b * beta_vec) > threshold_vec);
        }
    );
  });
  }
}

// 定义 glu_kernel 函数，处理门控线性单元 (GLU) 操作
void glu_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中的数据类型是否为降维后的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用宏展开，处理降维后的浮点类型
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "glu_cpu", [&]() {
    const float float_one_val(1);
    const Vectorized<float> float_one_vec(float_one_val);
    // 调用 CPU 内核函数，处理向量化操作
    cpu_kernel_vec(
      iter,
      // 定义标量处理函数，计算门控线性单元 (GLU) 操作
      [float_one_val](scalar_t a, scalar_t b) -> scalar_t {
        return float(a) * (float_one_val / (float_one_val + std::exp(- float(b))));
      },
      // 定义向量处理函数，使用向量化指令加速计算
      [float_one_vec](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
        auto [a0, a1] = convert_to_float<scalar_t>(a);
        auto [b0, b1] = convert_to_float<scalar_t>(b);
        return convert_from_float<scalar_t>(a0 * (float_one_vec / (float_one_vec + b0.neg().exp())),
                                            a1 * (float_one_vec / (float_one_vec + b1.neg().exp())));
      });
    });
  } else {
    // 使用AT_DISPATCH_FLOATING_TYPES宏，根据iter的数据类型动态选择对应的函数模板"glu_cpu"
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_cpu", [&] {
        // 定义类型别名Vec，表示scalar_t类型的向量化操作
        using Vec = Vectorized<scalar_t>;
        // 创建常量one_val，值为1，用于向量化计算
        const scalar_t one_val(1);
        // 创建向量化的常量向量one_vec，其中每个元素的值都是one_val
        const Vec one_vec(one_val);
        // 调用cpu_kernel_vec函数，对iter进行处理，传入两个lambda表达式
        cpu_kernel_vec(
          iter,
          // 第一个lambda表达式：接受标量a和b，返回a乘以一个sigmoid变换后的值
          [one_val](scalar_t a, scalar_t b) -> scalar_t {
            return a * (one_val / (one_val + std::exp(-b)));
          },
          // 第二个lambda表达式：接受向量a和向量b，返回a与sigmoid变换后的b的向量运算结果
          [one_vec](Vec a, Vec b) -> Vec {
            return a * (one_vec / (one_vec + b.neg().exp()));
          }
        );
    });
    // 函数调用结束
    }
}

// 定义一个名为 glu_jvp_kernel 的函数，接受一个 TensorIteratorBase 的引用参数
void glu_jvp_kernel(TensorIteratorBase& iter) {
  // 根据迭代器的数据类型分发到对应的浮点类型处理函数 "glu_jvp_cpu"
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_jvp_cpu", [&] {
    // 使用 Vectorized 类别的当前数据类型作为 scalar_t 的别名
    using Vec = Vectorized<scalar_t>;
    // 声明并初始化一个常量标量 one 为 1
    const scalar_t one(1);
    // 使用标量 one 初始化一个 Vectorized 对象 ones
    const Vec ones(one);
    // 调用 cpu_kernel_vec 函数处理迭代器 iter
    cpu_kernel_vec(
      iter,
      // Lambda 函数，计算 JVP 变换的 CPU 实现
      [one](scalar_t res, scalar_t b, scalar_t da, scalar_t db) -> scalar_t {
        // 计算 sig_b，对应于 sigmoid 函数的结果
        const auto sig_b = one / (one + std::exp(-b));
        // 返回 JVP 的结果
        return da * sig_b + res * (db - sig_b * db);
      },
      // Lambda 函数，向量化版本的 JVP 计算
      [ones](Vec res, Vec b, Vec da, Vec db) -> Vec {
        // 计算 sig_b 的向量化形式
        const auto sig_b = ones / (ones + b.neg().exp());
        // 返回向量化的 JVP 计算结果
        return da * sig_b + res * (db - sig_b * db);
      }
    );
  });
}

// 定义一个名为 glu_backward_kernel 的函数，接受一个 TensorIterator 的引用参数
void glu_backward_kernel(TensorIterator& iter) {
  // 如果数据类型是减少浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 根据具体的减少浮点类型分发到对应的函数 "glu_backward_cpu"
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "glu_backward_cpu", [&]() {
      // 声明并初始化一个常量标量 float_one_val 为 1
      const float float_one_val(1);
      // 使用标量 float_one_val 初始化一个 Vectorized<float> 对象 float_one_vec
      const Vectorized<float> float_one_vec(float_one_val);
      // 调用 cpu_kernel_vec 函数处理迭代器 iter
      cpu_kernel_vec(
        iter,
        // Lambda 函数，计算 glu 的反向传播的 CPU 实现
        [float_one_val](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
          return  (float_one_val - float(a)) * float(a) * float(b) * float(c);
        },
        // Lambda 函数，向量化版本的 glu 反向传播计算
        [float_one_vec](Vectorized<scalar_t> a, Vectorized<scalar_t> b, Vectorized<scalar_t> c) -> Vectorized<scalar_t> {
          auto [a0, a1] = convert_to_float<scalar_t>(a);
          auto [b0, b1] = convert_to_float<scalar_t>(b);
          auto [c0, c1] = convert_to_float<scalar_t>(c);
          // 计算向量化的 glu 反向传播结果
          a0 = (float_one_vec - a0) * a0 * b0 * c0;
          a1 = (float_one_vec - a1) * a1 * b1 * c1;
          return convert_from_float<scalar_t>(a0, a1);
        });
    });
  } else {
    // 如果数据类型是浮点类型
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_backward_cpu", [&] {
      // 使用 Vectorized 类别的当前数据类型作为 scalar_t 的别名
      using Vec = Vectorized<scalar_t>;
      // 声明并初始化一个常量标量 one_val 为 1
      const scalar_t one_val(1);
      // 使用标量 one_val 初始化一个 Vectorized 对象 one_vec
      const Vec one_vec(one_val);
      // 调用 cpu_kernel_vec 函数处理迭代器 iter
      cpu_kernel_vec(
        iter,
        // Lambda 函数，计算 glu 的反向传播的 CPU 实现
        [one_val](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
          return (one_val - a) * a * b * c;
        },
        // Lambda 函数，向量化版本的 glu 反向传播计算
        [one_vec](Vec a, Vec b, Vec c) -> Vec {
          return (one_vec - a) * a * b * c;
        }
      );
    });
  }
}

// 定义一个名为 silu_kernel 的函数，接受一个 TensorIteratorBase 的引用参数
void silu_kernel(TensorIteratorBase& iter) {
  // 如果数据类型是减少浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 根据具体的减少浮点类型分发到对应的函数 "silu_cpu"
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "silu_cpu", [&]() {
      // 声明并初始化一个 Vectorized<float> 类型的常量 kOneVec，值为 1.0f
      const Vectorized<float> kOneVec(1.0f);
      // 调用 cpu_kernel_vec 函数处理迭代器 iter
      cpu_kernel_vec(
          iter,
          // Lambda 函数，计算 SiLU 激活函数的 CPU 实现
          [](scalar_t x) -> scalar_t {
            return float(x) / (1.0f + std::exp(-float(x)));
          },
          // Lambda 函数，向量化版本的 SiLU 激活函数计算
          [kOneVec](Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
            auto [x_vec0, x_vec1] = convert_to_float<scalar_t>(x_vec);
            return convert_from_float<scalar_t>(
              x_vec0 / (kOneVec + x_vec0.neg().exp()),
              x_vec1 / (kOneVec + x_vec1.neg().exp()));
          });
    });
  } else {
    // 如果数据类型是浮点类型
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      iter.dtype(), "silu_cpu", [&]() {
        // 使用宏根据迭代器的数据类型调度对应的函数，函数名为 "silu_cpu"
        const Vectorized<scalar_t> kOneVec(scalar_t(1));
        // 创建一个包含值为 1 的向量化对象 kOneVec

        cpu_kernel_vec(
            iter,
            // 对每个元素应用 SILU（Sigmoid Linear Unit）函数：x / (1 + exp(-x))
            [](scalar_t x) {
              return x / (scalar_t(1) + std::exp(-x));
            },
            // 对向量化对象应用 SILU 函数：x_vec / (kOneVec + (-x_vec).exp())
            [kOneVec](Vectorized<scalar_t> x_vec) {
              return x_vec / (kOneVec + x_vec.neg().exp());
            });
      });
    }
}

// 定义一个函数 silu_backward_kernel，接受一个 Tensor 迭代器
void silu_backward_kernel(TensorIteratorBase& iter) {
  // 检查迭代器的数据类型是否为浮点类型的简化类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 如果是简化的浮点类型，使用宏展开处理器，命名为 "silu_backward_cpu"
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "silu_backward_cpu", [&]() {
      // 定义一个常量向量化对象 kOneVec，初始化为 1.0f
      const Vectorized<float> kOneVec(1.0f);
      // 调用 CPU 内核向量化函数 cpu_kernel_vec
      cpu_kernel_vec(
          iter,
          // Lambda 函数，计算 Sigmoid 线性单元的反向传播
          [](scalar_t dy, scalar_t x) -> scalar_t {
            const float sigmoid =
                1.0f / (1.0f + std::exp(-float(x)));
            return dy * sigmoid * (1.0f + x * (1.0f - sigmoid));
          },
          // Lambda 函数，处理向量化输入
          [kOneVec](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
            auto [x_vec0, x_vec1] = convert_to_float<scalar_t>(x_vec);
            auto [dy_vec0, dy_vec1] = convert_to_float<scalar_t>(dy_vec);
            const Vectorized<float> sigmoid0 =
                kOneVec / (kOneVec + x_vec0.neg().exp());
            const Vectorized<float> sigmoid1 =
                kOneVec / (kOneVec + x_vec1.neg().exp());
            return convert_from_float<scalar_t>(
              dy_vec0 * sigmoid0 * (kOneVec + x_vec0 * (kOneVec - sigmoid0)),
              dy_vec1 * sigmoid1 * (kOneVec + x_vec1 * (kOneVec - sigmoid1)));
          });
    });
  } else {
    // 如果不是简化的浮点类型，使用宏展开处理器，命名为 "silu_backward_cpu"
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      iter.dtype(), "silu_backward_cpu", [&]() {
        // 定义一个常量向量化对象 kOneVec，使用当前标量类型初始化为 1
        const Vectorized<scalar_t> kOneVec(scalar_t(1));
        // 调用 CPU 内核向量化函数 cpu_kernel_vec
        cpu_kernel_vec(
            iter,
            // Lambda 函数，计算 Sigmoid 线性单元的反向传播
            [](scalar_t dy, scalar_t x) {
              const scalar_t sigmoid =
                  scalar_t(1) / (scalar_t(1) + std::exp(-x));
              return dy * sigmoid * (scalar_t(1) + x * (scalar_t(1) - sigmoid));
            },
            // Lambda 函数，处理向量化输入
            [kOneVec](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) {
              const Vectorized<scalar_t> sigmoid =
                  kOneVec / (kOneVec + x_vec.neg().exp());
              return dy_vec * sigmoid * (kOneVec + x_vec * (kOneVec - sigmoid));
            });
      });
  }
}

// 定义一个函数 mish_kernel，接受一个 Tensor 迭代器
void mish_kernel(TensorIteratorBase& iter) {
  // 检查迭代器的数据类型是否为浮点类型的简化类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 如果是简化的浮点类型，使用宏展开处理器，命名为 "mish_cpu"
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "mish_cpu", [&]() {
      // 调用 CPU 内核向量化函数 cpu_kernel_vec
      cpu_kernel_vec(
          iter,
          // Lambda 函数，计算 Mish 激活函数的反向传播
          [](scalar_t x) -> scalar_t{
            return static_cast<scalar_t>(float(x) * std::tanh(std::log1p(std::exp(float(x)))));
          },
          // Lambda 函数，处理向量化输入
          [](Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
            auto [x_vec0, x_vec1] = convert_to_float<scalar_t>(x_vec);
            return convert_from_float<scalar_t>(
              x_vec0 * x_vec0.exp().log1p().tanh(),
              x_vec1 * x_vec1.exp().log1p().tanh()
            );
          });
    });
  } else {
    // 如果不是简化的浮点类型，使用宏展开处理器，命名为 "mish_cpu"
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mish_cpu", [&]() {
        using Vec = Vectorized<scalar_t>;
        // 调用 CPU 内核向量化函数 cpu_kernel_vec
        cpu_kernel_vec(
            iter,
            // Lambda 函数，计算 Mish 激活函数的反向传播
            [](scalar_t x) -> scalar_t{
              return static_cast<scalar_t>(x * std::tanh(std::log1p(std::exp(x))));
            },
            // Lambda 函数，处理向量化输入
            [](Vec x_vec) -> Vec {
              return x_vec * x_vec.exp().log1p().tanh();
            });
      });
  }
}
// 定义反向传播 Mish 激活函数的 CPU 实现的核心函数
void mish_backward_kernel(TensorIterator& iter) {
  // 检查是否为降维后的浮点数类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用宏展开特定的降维后浮点数类型，并命名函数为 "mish_backward_cpu"
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "mish_backward_cpu", [&]() {
      // 使用 Vectorized<float> 类型别名 Vec
      using Vec = Vectorized<float>;
      // 创建常量向量，初始化为 1.0f
      const Vec kOneVec(1.0f);
      // 调用 CPU 向量化核函数
      cpu_kernel_vec(
          iter,
          // 匿名函数，计算 Mish 激活函数的反向传播
          [](scalar_t dy, scalar_t x) -> scalar_t {
            // 计算 sigmoid 函数的值
            const float sigmoid =
                1.0f / (1.0f + std::exp(-float(x)));
            // 计算 tanh_softplus 函数的值
            const float tanh_softplus = std::tanh(std::log1p(std::exp(float(x))));
            // 返回 Mish 激活函数的导数值
            return dy * (tanh_softplus + x * sigmoid * (1.0f - tanh_softplus * tanh_softplus));
          },
          // 匿名函数，处理向量化计算
          [kOneVec](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
            auto [x_vec0, x_vec1] = convert_to_float<scalar_t>(x_vec);
            auto [dy_vec0, dy_vec1] = convert_to_float<scalar_t>(dy_vec);
            // 计算两个向量的 sigmoid 函数值
            const Vec sigmoid0 = kOneVec / (kOneVec + x_vec0.neg().exp());
            const Vec sigmoid1 = kOneVec / (kOneVec + x_vec1.neg().exp());
            // 计算两个向量的 tanh_softplus 函数值
            const Vec tanh_softplus0 = x_vec0.exp().log1p().tanh();
            const Vec tanh_softplus1 = x_vec1.exp().log1p().tanh();
            // 返回向量化后的 Mish 激活函数的导数值
            return convert_from_float<scalar_t>(
              dy_vec0 * (tanh_softplus0 + x_vec0 * sigmoid0 * (kOneVec - tanh_softplus0 * tanh_softplus0)),
              dy_vec1 * (tanh_softplus1 + x_vec1 * sigmoid1 * (kOneVec - tanh_softplus1 * tanh_softplus1))
            );
          });
    });
  } else {
    // 如果不是降维后的浮点数类型，使用标准浮点数类型进行处理
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mish_backward_cpu", [&]() {
        using Vec = Vectorized<scalar_t>;
        // 创建常量向量，初始化为 1
        const Vec kOneVec(scalar_t(1));
        // 调用 CPU 向量化核函数
        cpu_kernel_vec(
            iter,
            // 匿名函数，计算 Mish 激活函数的反向传播
            [](scalar_t dy, scalar_t x) -> scalar_t {
              // 计算 sigmoid 函数的值
              const scalar_t sigmoid =
                  scalar_t(1) / (scalar_t(1) + std::exp(-x));
              // 计算 tanh_softplus 函数的值
              const scalar_t tanh_softplus = std::tanh(std::log1p(std::exp(x)));
              // 返回 Mish 激活函数的导数值
              return dy * (tanh_softplus + x * sigmoid * (scalar_t(1) - tanh_softplus * tanh_softplus));
            },
            // 匿名函数，处理向量化计算
            [kOneVec](Vec dy_vec, Vec x_vec) -> Vec {
              // 计算向量的 sigmoid 函数值
              const Vec sigmoid = kOneVec / (kOneVec + x_vec.neg().exp());
              // 计算向量的 tanh_softplus 函数值
              const Vec tanh_softplus = x_vec.exp().log1p().tanh();
              // 返回向量化后的 Mish 激活函数的导数值
              return dy_vec * (tanh_softplus + x_vec * sigmoid * (kOneVec - tanh_softplus * tanh_softplus));
            });
      });
  }
}

// 定义 PReLU 激活函数的 CPU 实现的核心函数
void prelu_kernel(TensorIterator& iter) {
  // 使用宏展开浮点数类型，并命名函数为 "prelu_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_cpu", [&]() {
    // 使用 Vectorized<scalar_t> 类型别名 Vec
    using Vec = Vectorized<scalar_t>;
    // 调用 CPU 向量化核函数
    cpu_kernel_vec(
      iter,
      // 匿名函数，计算 PReLU 激活函数
      [](scalar_t input, scalar_t weight) {
        // 根据 input 的值返回 input 或者 weight * input
        return (input > scalar_t(0)) ? input : weight * input;
      },
      // 匿名函数，处理向量化计算
      [](Vec input, Vec weight) {
        // 使用向量融合操作，根据 input 的值返回 weight * input 或者 input
        return Vec::blendv(weight * input, input, input > Vec(0));
      });
  });
}

// 定义 PReLU 激活函数的反向传播的 CPU 实现的核心函数
void prelu_backward_kernel(TensorIterator& iter) {
  // 使用宏展开浮点数类型，并命名函数为 "prelu_backward_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_backward_cpu", [&]() {
    cpu_kernel_multiple_outputs(iter,
      [](scalar_t input, scalar_t weight, scalar_t grad) -> std::tuple<scalar_t, scalar_t> {
        // 定义一个掩码，用于判断输入是否大于零
        auto mask = input > scalar_t{0};
        // 根据掩码条件选择计算结果，如果输入大于零，则 grad_input = grad，否则 grad_input = weight * grad
        auto grad_input = mask ? grad : weight * grad;
        // 根据掩码条件选择计算结果，如果输入大于零，则 grad_weight = 0，否则 grad_weight = input * grad
        auto grad_weight = mask ? scalar_t{0} : input * grad;
        // 返回计算结果作为元组
        return {grad_input, grad_weight};
      });
  });
} // 结束 at::native 命名空间

} // 结束代码文件的命名空间

REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel);
// 注册 hardsigmoid_stub，关联到 hardsigmoid_kernel 函数

REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel);
// 注册 hardsigmoid_backward_stub，关联到 hardsigmoid_backward_kernel 函数

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
// 注册 threshold_stub，关联到 threshold_kernel 函数

REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel);
// 注册 leaky_relu_stub，关联到 leaky_relu_kernel 函数

REGISTER_DISPATCH(leaky_relu_backward_stub, &leaky_relu_backward_kernel);
// 注册 leaky_relu_backward_stub，关联到 leaky_relu_backward_kernel 函数

REGISTER_DISPATCH(prelu_stub, &prelu_kernel);
// 注册 prelu_stub，关联到 prelu_kernel 函数

REGISTER_DISPATCH(prelu_backward_stub, &prelu_backward_kernel);
// 注册 prelu_backward_stub，关联到 prelu_backward_kernel 函数

REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel);
// 注册 hardtanh_backward_stub，关联到 hardtanh_backward_kernel 函数

REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel);
// 注册 hardshrink_stub，关联到 hardshrink_kernel 函数

REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel);
// 注册 softshrink_stub，关联到 softshrink_kernel 函数

REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel);
// 注册 shrink_backward_stub，关联到 shrink_backward_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(log_sigmoid_cpu_stub, &log_sigmoid_cpu_kernel);
// 还注册 AVX512 版本的 log_sigmoid_cpu_stub，关联到 log_sigmoid_cpu_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(log_sigmoid_backward_stub, &log_sigmoid_backward_cpu_kernel);
// 还注册 AVX512 版本的 log_sigmoid_backward_stub，关联到 log_sigmoid_backward_cpu_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(glu_stub, &glu_kernel);
// 还注册 AVX512 版本的 glu_stub，关联到 glu_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(glu_backward_stub, &glu_backward_kernel);
// 还注册 AVX512 版本的 glu_backward_stub，关联到 glu_backward_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(glu_jvp_stub, &glu_jvp_kernel);
// 还注册 AVX512 版本的 glu_jvp_stub，关联到 glu_jvp_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(elu_stub, &elu_kernel);
// 还注册 AVX512 版本的 elu_stub，关联到 elu_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(elu_backward_stub, &elu_backward_kernel);
// 还注册 AVX512 版本的 elu_backward_stub，关联到 elu_backward_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(GeluKernel, &GeluKernelImpl);
// 还注册 AVX512 版本的 GeluKernel，关联到 GeluKernelImpl 函数

ALSO_REGISTER_AVX512_DISPATCH(GeluBackwardKernel, &GeluBackwardKernelImpl);
// 还注册 AVX512 版本的 GeluBackwardKernel，关联到 GeluBackwardKernelImpl 函数

ALSO_REGISTER_AVX512_DISPATCH(hardswish_stub, &hardswish_kernel);
// 还注册 AVX512 版本的 hardswish_stub，关联到 hardswish_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(hardswish_backward_stub, &hardswish_backward_kernel);
// 还注册 AVX512 版本的 hardswish_backward_stub，关联到 hardswish_backward_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(softplus_stub, &softplus_kernel);
// 还注册 AVX512 版本的 softplus_stub，关联到 softplus_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(softplus_backward_stub, &softplus_backward_kernel);
// 还注册 AVX512 版本的 softplus_backward_stub，关联到 softplus_backward_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(silu_stub, &silu_kernel);
// 还注册 AVX512 版本的 silu_stub，关联到 silu_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(silu_backward_stub, &silu_backward_kernel);
// 还注册 AVX512 版本的 silu_backward_stub，关联到 silu_backward_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(mish_stub, &mish_kernel);
// 还注册 AVX512 版本的 mish_stub，关联到 mish_kernel 函数

ALSO_REGISTER_AVX512_DISPATCH(mish_backward_stub, &mish_backward_kernel);
// 还注册 AVX512 版本的 mish_backward_stub，关联到 mish_backward_kernel 函数

} // 结束 at::native 命名空间
```