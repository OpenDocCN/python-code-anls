# `.\pytorch\aten\src\ATen\native\RangeFactories.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/RangeFactories.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>
#include <cmath>
#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/range_native.h>
#endif

// 进入 ATen 的 native 命名空间
namespace at::native {

// 函数：使用 linspace 创建等间距数列到给定张量 result
Tensor& linspace_out(const Tensor& start, const Tensor& end, int64_t steps, Tensor& result) {
  // 检查输入张量 start 和 end 的维度是否为零
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  // 调用 ATen 的 linspace_out 函数生成等间距数列并存储到 result 中
  return at::linspace_out(result, start.item(), end.item(), steps);
}

// 函数：使用 linspace 创建等间距数列从张量 start 到标量 end 到给定张量 result
Tensor& linspace_out(const Tensor& start, const Scalar& end, int64_t steps, Tensor& result) {
  // 检查输入张量 start 的维度是否为零
  TORCH_CHECK(start.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  // 调用 ATen 的 linspace_out 函数生成等间距数列并存储到 result 中
  return at::linspace_out(result, start.item(), end, steps);
}

// 函数：使用 linspace 创建等间距数列从标量 start 到张量 end 到给定张量 result
Tensor& linspace_out(const Scalar& start, const Tensor& end, int64_t steps, Tensor& result) {
  // 检查输入张量 end 的维度是否为零
  TORCH_CHECK(end.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  // 调用 ATen 的 linspace_out 函数生成等间距数列并存储到 result 中
  return at::linspace_out(result, start, end.item(), steps);
}

// 函数：使用 linspace 创建等间距数列从标量 start 到标量 end 到给定张量 result
Tensor& linspace_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  // 检查步长是否为非负数
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  // 如果结果张量的元素数量不等于 steps，则重新调整其大小
  if (result.numel() != steps) {
    result.resize_({steps});
  }

  // 如果结果张量的设备是元设备，则直接返回结果
  if (result.device() == kMeta) {
    return result;
  }

  // 如果步长为 0，则跳过
  if (steps == 0) {
    // skip
  } else if (steps == 1) {  // 如果步长为 1，则用 start 填充结果张量
    result.fill_(start);
  } else {  // 否则，生成连续的张量并填充结果
    // 如果结果张量不是连续的，则先进行连续化
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    // 使用 TensorIterator 执行 linspace 操作
    auto iter = TensorIterator::borrowing_nullary_op(r);
    linspace_stub(iter.device_type(), iter, start, end, steps);
    // 如果结果张量不是连续的，则复制连续化的结果回原始结果张量
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  }

  // 返回最终的结果张量
  return result;
}

// 函数：使用 logspace 创建对数间距数列到给定张量 result
Tensor& logspace_out(const Tensor& start, const Tensor& end, int64_t steps, double base, Tensor& result) {
  // 检查输入张量 start 和 end 的维度是否为零
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  // 调用 ATen 的 logspace_out 函数生成对数间距数列并存储到 result 中
  return at::logspace_out(result, start.item(), end.item(), steps, base);
}

// 函数：使用 logspace 创建对数间距数列从张量 start 到标量 end 到给定张量 result
Tensor& logspace_out(const Tensor& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  // 检查输入张量 start 的维度是否为零
  TORCH_CHECK(start.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  // 调用 ATen 的 logspace_out 函数生成对数间距数列并存储到 result 中
  return at::logspace_out(result, start.item(), end, steps, base);
}

// 其余函数未提供，省略
// ...

} // namespace at::native
// 计算给定起始点、结束点（作为标量）、步数、基数和结果张量来生成对数空间值，将结果存储在给定的结果张量中
Tensor& logspace_out(const Scalar& start, const Tensor& end, int64_t steps, double base, Tensor& result) {
  // 检查结束张量是否为零维，因为 logspace 只支持零维的起始和结束张量
  TORCH_CHECK(end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  // 调用 ATen 提供的 logspace_out 函数来执行实际的计算和填充操作
  return at::logspace_out(result, start, end.item(), steps, base);
}

// 计算给定起始点、结束点（作为标量）、步数、基数和结果张量来生成对数空间值，将结果存储在给定的结果张量中
Tensor& logspace_out(const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  // 检查步数是否为非负数
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  // 如果结果张量的元素数量不等于步数，则调整结果张量的形状为(steps,)
  if (result.numel() != steps) {
    result.resize_({steps});
  }

  // 如果结果张量的设备是 kMeta，直接返回结果张量
  if (result.device() == kMeta) {
    return result;
  }

  // 将结果张量转为连续的张量 r
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  // 根据步数执行不同的操作
  if (steps == 0) {
    // 如果步数为 0，则跳过
  } else if (steps == 1) {
    // 如果步数为 1，根据结果张量的数据类型，填充 r
    if (isComplexType(r.scalar_type())){
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isComplexType(r.scalar_type())) {
    // 如果结果张量的数据类型是复数类型，则执行复数类型的并行计算
    AT_DISPATCH_COMPLEX_TYPES(r.scalar_type(), "logspace_cpu", [&]() {
      // 获取基数、起始点、结束点的实数值
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // 获取结果张量的数据指针
      scalar_t *data_ptr = r.data_ptr<scalar_t>();
      // 计算步长
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      // 计算中点位置
      const int64_t halfway = steps / 2;
      // 并行计算每个步骤的值
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        scalar_t is = static_cast<scalar_t>(p_begin);
        for (int64_t i = p_begin; i < p_end; ++i, is+=1) { //std::complex does not support ++operator
          if (i < halfway) {
            data_ptr[i] = std::pow(scalar_base, scalar_start + step*is);
          } else {
            data_ptr[i] = std::pow(scalar_base, scalar_end - (step * static_cast<scalar_t>(steps - i - 1)));
          }
        }
      });
    });
  } else {
    // 如果结果张量的数据类型不是复数类型，则执行标量类型的并行计算
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, r.scalar_type(), "logspace_cpu", [&]() {
      // 获取基数、起始点、结束点的实数值
      double scalar_base = static_cast<double>(base); // will be autopromoted anyway
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // 获取结果张量的数据指针
      scalar_t *data_ptr = r.data_ptr<scalar_t>();
      // 计算步长
      double step = static_cast<double>(scalar_end - scalar_start) / (steps - 1);
      // 计算中点位置
      const int64_t halfway = steps / 2;
      // 并行计算每个步骤的值
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        for (const auto i : c10::irange(p_begin, p_end)) {
          if (i < halfway) {
            data_ptr[i] = std::pow(scalar_base, scalar_start + step*i);
          } else {
            data_ptr[i] = std::pow(scalar_base, scalar_end - step * (steps - i - 1));
          }
        }
      });
    });
  }

  // 如果结果张量不是连续的，则将 r 的值复制到结果张量中
  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  // 返回结果张量
  return result;
}
Tensor& range_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, result.scalar_type(), "range_cpu", [&]() {
    // 使用 accscalar_t 作为 acc_type 的类型，不允许累加
    using accscalar_t = at::acc_type<scalar_t, false>;
    // 将 start、end、step 转换为 accscalar_t 类型
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    // 检查步长是否为非零
    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    // 检查 start 和 end 是否为有限数值
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    // 检查上界和下界与步长符号是否一致
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and lower bound inconsistent with step sign");

    // 计算结果张量的大小
    int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
    // 如果结果张量的元素数量不等于 size，则调整结果张量的大小
    if (result.numel() != size) {
      result.resize_({size});
    }

    // 如果结果张量在 kMeta 设备上，则直接返回
    if (result.device() == kMeta) {
      return;
    }

    // 如果结果张量不是连续的，则创建其连续版本
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    // 获取结果张量数据指针
    scalar_t *data_ptr = r.data_ptr<scalar_t>();

    // 并行计算结果张量的值
    at::parallel_for(0, size, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      accscalar_t is = p_begin;
      for (int64_t i = p_begin; i < p_end; ++i, ++is) {
        data_ptr[i] = xstart + is * xstep;
      }
    });

    // 如果结果张量不是连续的，则将连续版本的数据拷贝回原始结果张量
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  // 返回结果张量
  return result;
}

Tensor& range_out_no_step(const Scalar& start, const Scalar& end, Tensor& result) {
  // 调用 range_out 函数，并指定步长为 1
  return range_out(start, end, /*step = */ 1, result);
}

Tensor& arange_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, result.scalar_type(), "arange_cpu", [&]() {
    // 使用 accscalar_t 作为 acc_type 的类型，不允许累加
    using accscalar_t = at::acc_type<scalar_t, false>;
    // 将 start、end、step 转换为 accscalar_t 类型
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    // 检查步长是否为非零
    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    // 检查 start 和 end 是否为有限数值
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    // 检查上界和下界与步长符号是否一致
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and larger bound inconsistent with step sign");

    // 使用 double 精度计算 (start - end) / step，以保证跨设备的一致性
    // accscalar_t 可能在 GPU 上为 float32（对于 float32 的 scalar_t），但在 CPU 上为 double
    // 但我们希望在 CPU 和 GPU 上都能保证输出大小的一致性，因此使用 double
    double size_d;
    // 如果 scalar_t 和 int64_t 相同，执行以下代码块
    if constexpr (std::is_same_v<scalar_t, int64_t>) {
      // 计算步长的符号
      int64_t sgn = (xstep > 0) - (xstep < 0);
      // 计算输出张量的大小，使用向上取整确保足够容纳范围内所有元素
      size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
    } else {
      // 对于其他类型，计算输出张量的大小，将范围转换为 double 进行计算
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                         / step.to<double>());
    }

    // 检查输出张量的大小是否在有效范围内，避免溢出
    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
             "invalid size, possible overflow?");

    // 将计算得到的大小转换为 int64_t 类型
    int64_t size = static_cast<int64_t>(size_d);
    // 获取输出张量的元素个数
    int64_t numel = result.numel();

    // 如果输出张量的元素个数与计算得到的大小不一致，则进行调整
    if (numel != size) {
      if(numel > 0){
        // 发出警告，说明输出张量的形状和元素数量不匹配
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      // 调整输出张量的大小为计算得到的大小
      result.resize_({size});
    }

    // 如果输出张量的设备是 kMeta，直接返回
    if (result.device() == kMeta) {
      return;
    }

    // 如果输出张量不是连续的，将其变为连续的
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    // 创建一个 Tensor 迭代器，用于生成一个从 start 到 (start + size * step) 的序列
    auto iter = TensorIterator::borrowing_nullary_op(r);
    // 使用 arange_stub 在迭代器上执行操作，生成指定范围内的数值序列
    arange_stub(iter.device_type(), iter, start, size, step);
    // 如果输出张量不是连续的，则将生成的连续张量复制回结果张量
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  // 返回处理后的结果张量
  return result;
}

DEFINE_DISPATCH(arange_stub);
DEFINE_DISPATCH(linspace_stub);

} // namespace at::native


注释：


// 结束当前命名空间 at::native 的定义
}

// 定义名为 arange_stub 的分发函数
DEFINE_DISPATCH(arange_stub);

// 定义名为 linspace_stub 的分发函数
DEFINE_DISPATCH(linspace_stub);

// 声明结束命名空间 at::native
} // namespace at::native


这段代码主要涉及 C++ 的命名空间和函数定义：

1. `}` 行表示结束当前的命名空间 `at::native` 的定义。
2. `DEFINE_DISPATCH(arange_stub);` 和 `DEFINE_DISPATCH(linspace_stub);` 是宏或函数定义，用于声明和定义名为 `arange_stub` 和 `linspace_stub` 的分发函数。
3. `}` 行再次结束了命名空间 `at::native` 的定义，这次是通过注释来指出该命名空间的结束位置。
4. `// namespace at::native` 是一个注释，指出前面的 `}` 是结束命名空间 `at::native` 的位置。

这些注释确保了读者能够理解每行代码的具体作用和位置，尤其是在复杂的 C++ 命名空间和宏定义的情况下。
```