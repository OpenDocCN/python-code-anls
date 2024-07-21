# `.\pytorch\aten\src\ATen\native\quantized\cpu\TensorShape.cpp`

```py
/* 定义宏，用于仅声明方法操作符 */
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
/* 包含张量核心头文件 */
#include <ATen/core/Tensor.h>
/* 包含列表核心头文件 */
#include <ATen/core/List.h>
/* 包含调度头文件 */
#include <ATen/Dispatch.h>
/* 包含维度包装实用工具头文件 */
#include <ATen/WrapDimUtils.h>
/* 包含列表引用头文件 */
#include <ATen/core/IListRef.h>
/* 包含CPU循环头文件 */
#include <ATen/native/cpu/Loops.h>
/* 包含量化操作CPU头文件 */
#include <ATen/native/quantized/cpu/QuantizedOps.h>
/* 包含张量迭代器头文件 */
#include <ATen/native/TensorIterator.h>
/* 包含张量形状头文件 */
#include <ATen/native/TensorShape.h>
/* 包含整数范围头文件 */
#include <c10/util/irange.h>
/* 包含Torch库头文件 */
#include <torch/library.h>

/* 如果未定义每个操作符的头文件，则包含功能和本地功能头文件 */
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
/* 否则，包含cat、cat_native、copy_native、quantize_per_tensor、zeros_like_ops头文件 */
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

/* 包含算法和向量头文件 */
#include <algorithm>
#include <vector>

/* 命名空间at内的命名空间native中 */
namespace at {
namespace native {

/* 定义分派qcat_nhwc_stub */
DEFINE_DISPATCH(qcat_nhwc_stub);
/* 定义分派qcat_relu_nhwc_stub */
DEFINE_DISPATCH(qcat_relu_nhwc_stub);

/* 匿名命名空间开始 */
namespace {

/* 判断是否使用NHWC快速路径的函数 */
bool is_cat_nhwc_fast_path(const MaterializedITensorListRef& qxs, int64_t dim) {
  /* 使用TORCH_CHECK断言列表不为空 */
  TORCH_CHECK(!qxs.empty());
  /* 初始化是否为快速路径为真，并检查每个张量是否满足NHWC格式要求 */
  bool is_fast_path = dim == 1;
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const at::Tensor& qx : qxs) {
    is_fast_path &= qx.dim() == 4;
    is_fast_path &= qx.is_contiguous(c10::MemoryFormat::ChannelsLast);
  }
  /* 返回是否为快速路径 */
  return is_fast_path;
}

/* 检查量化方案是否有效的函数 */
bool is_valid_quantization_scheme(const Tensor& t) {
  /* 获取张量的量化方案类型 */
  const auto qtype = t.qscheme();
  /* 返回是否为PerTensorAffine或PerTensorSymmetric的布尔值 */
  return (qtype == kPerTensorAffine) || (qtype == kPerTensorSymmetric);
}

/* 检查所有输入张量是否共享量化参数的函数 */
bool all_inputs_sharing_qparams(const MaterializedITensorListRef& qxs) {
  /* 初始化是否有效为真 */
  bool is_valid = true;
  /* 遍历输入张量列表，检查是否共享量化参数 */
  for (const auto i : c10::irange(1, qxs.size())) {
    is_valid |= qxs[0].get().is_quantized();
    is_valid |= qxs[i].get().is_quantized() == qxs[0].get().is_quantized();
    is_valid |= qxs[i].get().qscheme() == qxs[0].get().qscheme();
    is_valid |= qxs[i].get().dtype() == qxs[0].get().dtype();
    /* 如果量化方案为PerTensorAffine，还需比较量化比例和零点 */
    if (qxs[0].get().qscheme() == kPerTensorAffine) {
        is_valid |= qxs[i].get().q_scale() == qxs[0].get().q_scale();
      is_valid |= qxs[i].get().q_zero_point() == qxs[0].get().q_zero_point();
    } else if (qxs[0].get().qscheme() == kPerChannelAffine) {
        /* 如果量化方案为PerChannelAffine，则还需比较通道量化比例和零点 */
        is_valid |= qxs[i].get().q_per_channel_scales().equal(qxs[0].get().q_per_channel_scales());
      is_valid |= qxs[i].get().q_per_channel_zero_points().equal(qxs[0].get().q_per_channel_zero_points());
    } else {
        /* 否则，抛出错误，说明量化方案未识别 */
        TORCH_CHECK(false, "Unrecognized qscheme:", toString(qxs[0].get().qscheme()));
    }
  }
  /* 返回是否有效 */
  return is_valid;
}

/* 量化连接实现的模板函数 */
/* 注意：此函数使用去量化 */
template <bool ReLUFused>
Tensor quantized_cat_impl(
    const MaterializedITensorListRef& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  /* 如果符合NHWC快速路径条件 */
  if (is_cat_nhwc_fast_path(qxs, dim)) {
    /* 如果融入ReLU操作 */
    if (ReLUFused) {
      /* 调用qcat_relu_nhwc_stub分派方法进行量化连接 */
      return qcat_relu_nhwc_stub(at::kCPU, qxs, dim, scale, zero_point);
    } else {
      /* 否则，调用qcat_nhwc_stub分派方法进行量化连接 */
      return qcat_nhwc_stub(at::kCPU, qxs, dim, scale, zero_point);
    }
  }

  const auto x_dtype = qxs[0].get().scalar_type();
  const auto x_qscheme = qxs[0].get().qscheme();
  std::vector<Tensor> xs;
  xs.reserve(qxs.size());
  // 循环遍历输入的量化张量列表，确保它们的数据类型一致
  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(x_dtype == qx.scalar_type(), "All dtypes must be the same.");
    // 检查量化方案是否一致
    TORCH_CHECK(
        x_qscheme == qx.qscheme(), "Quantization schemes must be the same.");
    // 将去量化后的张量添加到 xs 向量中
    xs.push_back(qx.dequantize());
  }
  // 将 xs 中的张量沿指定维度进行拼接
  const Tensor y = at::cat(xs, dim);
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(x_dtype, "qcat", [&]() {
    // 对拼接后的张量进行量化
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    qy = at::quantize_per_tensor(y, scale, zero_point, SCALAR_TYPE);
    // 如果开启了 ReLU 后处理，则对 qy 应用 ReLU 函数
    if (ReLUFused) {
      // 创建迭代器用于对 qy 中的每个元素执行操作
      auto iter = TensorIterator::unary_op(qy, qy);
      // 在 CPU 上执行核函数，将小于 zero_point 的值设为 zero_point
      cpu_kernel(iter, [&](scalar_t value) -> scalar_t {
        return scalar_t(std::max<underlying_t>(value.val_, zero_point));
      });
    }
  });
  // 返回量化后的张量 qy
  return qy;
} // namespace

// 定义一个模板函数 quantized_cat_impl，用于将量化后的张量列表沿指定维度拼接，返回拼接后的张量
template <bool ReLUFused>
Tensor quantized_cat_impl(
    ITensorListRef qxs,     // 输入的量化张量列表
    int64_t dim,            // 拼接的维度
    double scale,           // 拼接后张量的缩放因子
    int64_t zero_point) {   // 拼接后张量的零点

  // 调用 materialize() 方法，将懒加载的张量列表 qxs 转换为实际张量列表
  return quantized_cat_impl<ReLUFused>(qxs.materialize(), dim, scale, zero_point);
}

// 定义一个模板函数 qcat，用于将普通张量列表中的量化张量沿指定维度拼接，返回拼接后的张量
template <bool ReLUFused = false>
Tensor qcat(
    const c10::List<Tensor>& qxs,           // 输入的普通张量列表
    int64_t dim,                            // 拼接的维度
    std::optional<double> scale,            // 拼接后张量的缩放因子（可选）
    std::optional<int64_t> zero_point) {    // 拼接后张量的零点（可选）

  // 检查第一个张量是否采用有效的量化方案
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]), "Only per-tensor quantization is supported in 'cat'!")

  // 确定缩放因子和零点的值
  double _scale = scale.has_value() ? scale.value() : qxs.get(0).q_scale();
  int64_t _zero_point = zero_point.has_value() ? zero_point.value() : qxs.get(0).q_zero_point();

  // 调用 quantized_cat_impl 函数，实现拼接操作
  return quantized_cat_impl<ReLUFused>(qxs, dim, _scale, _zero_point);
}

// 定义一个模板函数 qcat_out，用于将量化张量列表中的量化张量沿指定维度拼接，结果存储在输出张量 out 中
template <bool ReLUFused = false>
Tensor qcat_out(const c10::List<Tensor>& qxs, int64_t dim, Tensor out) {
  // 检查第一个张量是否采用有效的量化方案
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]), "Only per-tensor quantization is supported in 'cat'!")
  
  // 检查输出张量是否采用有效的量化方案
  TORCH_CHECK(is_valid_quantization_scheme(out), "Only per-tensor quantization is supported in 'cat'!")

  // 调用 quantized_cat_impl 函数，实现拼接操作，并将结果复制到输出张量 out 中
  auto out_ = quantized_cat_impl<ReLUFused>(qxs, dim, out.q_scale(), out.q_zero_point());
  at::native::copy_(out, out_, /*non_blocking=*/false);

  // 返回输出张量 out
  return out;
}

// 声明 Torch 库的实现，为 quantized::cat 和 quantized::cat_relu 注册对应的函数 qcat<false> 和 qcat<true>
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat"), TORCH_FN(qcat<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu"), TORCH_FN(qcat<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_out"), TORCH_FN(qcat_out<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu_out"), TORCH_FN(qcat_out<true>));
}

// 定义函数 cat_quantized_cpu，用于将量化张量列表沿指定维度拼接，返回拼接后的张量
Tensor cat_quantized_cpu(const ITensorListRef& qxs, int64_t dim) {
  // 将懒加载的张量列表 qxs 转换为实际张量列表
  auto materialized = qxs.materialize();

  // 检查第一个张量是否采用有效的量化方案
  TORCH_CHECK(is_valid_quantization_scheme(materialized[0]), "Only per-tensor quantization is supported in 'cat'!");

  // 检查所有输入张量是否共享相同的量化参数
  TORCH_CHECK(all_inputs_sharing_qparams(materialized), "All inputs should share the same quantization parameters.");

  // 检查拼接维度是否为零维度
  check_cat_no_zero_dim(materialized);

  // 将维度转换为兼容的维度
  dim = legacy_cat_wrap_dim(dim, materialized);

  // 确定缩放因子和零点的值
  double _scale = materialized[0].get().q_scale();
  int64_t _zero_point = materialized[0].get().q_zero_point();

  // 调用 quantized_cat_impl 函数，实现拼接操作
  return quantized_cat_impl<false>(materialized, dim, _scale, _zero_point);
}

// 定义函数 cat_out_quantized_cpu，用于将量化张量列表沿指定维度拼接，结果存储在输出张量 out 中
Tensor& cat_out_quantized_cpu(const ITensorListRef& qxs, int64_t dim, Tensor& out) {
  // 将懒加载的张量列表 qxs 转换为实际张量列表
  auto materialized = qxs.materialize();

  // 检查第一个张量是否采用有效的量化方案
  TORCH_CHECK(is_valid_quantization_scheme(materialized[0]), "Only per-tensor quantization is supported in 'cat'!")

  // 检查输出张量是否采用有效的量化方案
  TORCH_CHECK(is_valid_quantization_scheme(out), "Only per-tensor quantization is supported in 'cat'!")

  // 检查拼接维度是否为零维度
  check_cat_no_zero_dim(materialized);

  // 将维度转换为兼容的维度
  dim = legacy_cat_wrap_dim(dim, materialized);

  // 调用 quantized_cat_impl 函数，实现拼接操作，并将结果复制到输出张量 out 中
  auto out_ = quantized_cat_impl<false>(qxs, dim, out.q_scale(), out.q_zero_point());
  at::native::copy_(out, out_, /*non_blocking=*/false);

  // 返回输出张量 out 的引用
  return out;
}

} // namespace native
} // namespace at
```