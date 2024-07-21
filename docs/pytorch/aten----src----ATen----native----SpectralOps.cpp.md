# `.\pytorch\aten\src\ATen\native\SpectralOps.cpp`

```
// 定义宏以仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含通用操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含具体操作的头文件
#else
#include <ATen/ops/_cufft_clear_plan_cache_native.h>
#include <ATen/ops/_cufft_get_plan_cache_max_size_native.h>
#include <ATen/ops/_cufft_get_plan_cache_size_native.h>
#include <ATen/ops/_cufft_set_plan_cache_max_size_native.h>
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_fft_c2r.h>
#include <ATen/ops/_fft_r2c.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft2_native.h>
#include <ATen/ops/fft_fft_native.h>
#include <ATen/ops/fft_fftfreq_native.h>
#include <ATen/ops/fft_fftn_native.h>
#include <ATen/ops/fft_fftshift_native.h>
#include <ATen/ops/fft_hfft2_native.h>
#include <ATen/ops/fft_hfft_native.h>
#include <ATen/ops/fft_hfftn_native.h>
#include <ATen/ops/fft_ifft2_native.h>
#include <ATen/ops/fft_ifft_native.h>
#include <ATen/ops/fft_ifftn_native.h>
#include <ATen/ops/fft_ifftshift_native.h>
#include <ATen/ops/fft_ihfft2_native.h>
#include <ATen/ops/fft_ihfft_native.h>
#include <ATen/ops/fft_ihfftn_native.h>
#include <ATen/ops/fft_irfft2_native.h>
#include <ATen/ops/fft_irfft_native.h>
#include <ATen/ops/fft_irfftn_native.h>
#include <ATen/ops/fft_rfft2_native.h>
#include <ATen/ops/fft_rfft_native.h>
#include <ATen/ops/fft_rfftfreq_native.h>
#include <ATen/ops/fft_rfftn_native.h>
#include <ATen/ops/istft_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/stft.h>
#include <ATen/ops/stft_native.h>
#include <ATen/ops/unfold_backward.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

#include <algorithm>

// 命名空间 at::native 下的匿名命名空间
namespace at::native {

// 匿名命名空间，用于实现局部函数或者变量

// 促进输入到 FFT 函数的类型提升
// * 整数提升为默认的浮点类型
// * 如果 require_complex=True，则所有类型提升为复数类型
// * 对于半精度数据类型抛出错误，以便将来支持
ScalarType promote_type_fft(ScalarType type, bool require_complex, Device device) {
  // 如果已经是复数类型，则直接返回
  if (at::isComplexType(type)) {
    return type;
  }
  // 如果不是浮点类型，则提升为默认浮点类型
  if (!at::isFloatingType(type)) {
    type = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  // 检查是否可能支持半精度
  const bool maybe_support_half = (
    // 只有 CUDA 支持半精度，但是由于元张量没有设备，我们倾向于接受它

    // 半精度类型可能受支持，但当前不支持
    // 如果要求半精度，则抛出错误
    if (type == ScalarType::Half && device != DeviceType::CUDA) {
        AT_ERROR("Half precision is not supported for FFT operations");
    }

    // 返回提升后的数据类型
    return type;
}
    device.is_cuda() || device.is_meta()
  );



  // 检查设备是否支持半精度（half），如果支持则要求数据类型为半精度、单精度或双精度；否则要求数据类型为单精度或双精度
  if (maybe_support_half) {
    TORCH_CHECK(type == kHalf || type == kFloat || type == kDouble, "Unsupported dtype ", type);
  } else {
    TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);
  }



  // 如果不需要复数类型，直接返回当前数据类型
  if (!require_complex) {
    return type;
  }



  // 如果需要复数类型，则根据当前数据类型进行类型提升到复数类型
  // 根据当前数据类型进行选择
  switch (type) {
  case kHalf: return kComplexHalf;
  case kFloat: return kComplexFloat;
  case kDouble: return kComplexDouble;
  default: TORCH_INTERNAL_ASSERT(false, "Unhandled dtype");
  }
}

// 根据 promote_type_fft 函数推广张量的数据类型
Tensor promote_tensor_fft(const Tensor& t, bool require_complex=false) {
  // 获取当前张量的数据类型
  auto cur_type = t.scalar_type();
  // 根据 promote_type_fft 函数推广数据类型
  auto new_type = promote_type_fft(cur_type, require_complex, t.device());
  // 如果数据类型没有改变，则直接返回原张量，否则将张量转换为新类型
  return (cur_type == new_type) ? t : t.to(new_type);
}

// 将 NumPy 兼容的标准化模式字符串转换为枚举值
// 注意：NumPy 的标准化模式根据方向有不同的含义。例如，“forward” 对于前向变换转换为 `by_n`，对于后向变换则转换为 `none`。
fft_norm_mode norm_from_string(std::optional<c10::string_view> norm, bool forward) {
  if (!norm || *norm == "backward") {
    // 如果模式未指定或为 "backward"，根据前向还是后向返回相应的标准化模式
    return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  }

  if (*norm == "forward") {
    // 如果模式为 "forward"，根据前向还是后向返回相应的标准化模式
    return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
  }

  if (*norm == "ortho") {
    // 如果模式为 "ortho"，返回按照根号 N 标准化的模式
    return fft_norm_mode::by_root_n;
  }

  // 如果模式无效，则抛出错误信息
  TORCH_CHECK(false, "Invalid normalization mode: \"", *norm, "\"")
}

// 调整 x 张量的形状，使得 x.size(dims[i]) == sizes[i]，可以通过零填充或从 0 开始切片 x
Tensor resize_fft_input(Tensor x, IntArrayRef dims, SymIntArrayRef sizes) {
  // 确保 dims 和 sizes 的大小相等
  TORCH_INTERNAL_ASSERT(dims.size() == sizes.size());
  bool must_copy = false;
  auto x_sizes = x.sym_sizes();
  // 初始化零填充的数量
  SymDimVector pad_amount(x_sizes.size() * 2);
  for (const auto i : c10::irange(dims.size())) {
    if (sizes[i] == -1) {
      continue;
    }

    if (x_sizes[dims[i]] < sizes[i]) {
      // 如果当前维度大小小于目标大小，标记为需要复制并计算零填充的数量
      must_copy = true;
      auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
      pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];
    }

    if (x_sizes[dims[i]] > sizes[i]) {
      // 如果当前维度大小大于目标大小，则对该维度进行切片操作
      x = x.slice_symint(dims[i], 0, sizes[i]);
    }
  }

  // 只在必要时调用零填充函数，因为零填充会复制整个张量
  return must_copy ? at::constant_pad_nd_symint(x, pad_amount) : x;
}

// 检查输出张量 out 是否定义，并根据其调用对应的 FFT 实现函数
Tensor fft_r2c_maybe_out(
    c10::string_view fname, const Tensor& out, const Tensor& input,
    IntArrayRef dim, int64_t norm, bool onesided) {
  if (out.defined()) {
    // 如果 out 已定义，则检查其类型是否为复数类型，然后调用相应的 FFT 实现函数
    TORCH_CHECK(out.is_complex(), fname,
                " expects a complex output tensor, but got ", out.scalar_type());
    auto out_mut = out;
    return at::_fft_r2c_outf(input, dim, norm, onesided, out_mut);
  }
  // 如果 out 未定义，则直接调用功能型 FFT 函数
  return at::_fft_r2c(input, dim, norm, onesided);
}

// 检查输出张量 out 是否定义，并根据其调用对应的 IFFT 实现函数
Tensor fft_c2r_maybe_out(
    c10::string_view fname, const Tensor& out, const Tensor& input,
    IntArrayRef dim, int64_t norm, SymInt last_dim_size) {
  // 如果 out 已定义，则检查其类型是否为浮点类型，然后调用相应的 IFFT 实现函数
  if (out.defined()) {
    TORCH_CHECK(out.is_floating_point(), fname,
                " expects a floating point output tensor, but got ", out.scalar_type());
    auto out_mut = out;
    return at::_fft_c2r_symint_outf(input, dim, norm, last_dim_size, out_mut);
  }
  // 如果 out 未定义，则直接调用功能型 IFFT 函数
  return at::_fft_c2r_symint(input, dim, norm, last_dim_size);
}

Tensor fft_c2c_maybe_out(
    c10::string_view fname, const Tensor& out, const Tensor& input,
    # 检查是否输出张量已定义，如果未定义则返回 false
    if (out.defined()) {
        # 检查输出张量是否为复数类型，否则抛出错误信息
        TORCH_CHECK(out.is_complex(), fname,
                    " expects a complex output tensor, but got ", out.scalar_type());
        # 将可变引用指向输出张量
        auto out_mut = out;
        # 调用带有输出张量参数的复数到复数的 FFT 函数
        return at::_fft_c2c_outf(input, dim, norm, forward, out_mut);
    }
    # 如果输出张量未定义，调用默认的复数到复数的 FFT 函数
    return at::_fft_c2c(input, dim, norm, forward);
// 复数到实数的快速傅立叶变换（FFT）

Tensor fft_c2r(c10::string_view function_name,
               Tensor out, Tensor input, std::optional<SymInt> n_opt,
               int64_t unwrapped_dim, std::optional<c10::string_view> norm_str,
               bool forward) {
  // 检查输出张量是否未定义或者是浮点型
  TORCH_CHECK(!out.defined() || out.is_floating_point(), function_name,
              " expects a floating point output tensor, but got ", out.scalar_type());
  // 将输入张量提升为复数类型的张量
  input = promote_tensor_fft(input, /*require_complex=*/true);
  // 获取输入张量的维度
  const auto input_dim = input.dim();
  // 包装维度以确保有效性
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  // 如果提供了数据点数目 n_opt，则使用它；否则计算默认值
  const auto n = n_opt.value_or(2*(input.sym_sizes()[dim] - 1));
  // 检查数据点数目是否有效
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  // 如果提供了 n_opt，则调整输入张量的大小以适应 FFT 输入要求
  if (n_opt) {
    input = resize_fft_input(input, dim, n/2 + 1);
  }
  // 根据字符串获取归一化方式
  const auto norm = norm_from_string(norm_str, forward);
  // 如果是正向变换，需要对输入张量取共轭
  if (forward) {
    // FIXME: _fft does not support complex_output=false with inverse=false
    input = input.conj();
  }
  // 调用 fft_c2r_maybe_out 执行 FFT 变换
  return fft_c2r_maybe_out(
      function_name, out, input, dim, static_cast<int64_t>(norm), n);
}

// 实数到复数的快速傅立叶变换（FFT）

Tensor fft_r2c(c10::string_view function_name,
               Tensor out, Tensor input, std::optional<SymInt> n_opt,
               int64_t unwrapped_dim, std::optional<c10::string_view> norm_str,
               bool forward, bool onesided) {
  // 检查输入张量是否为实数类型
  TORCH_CHECK(!input.is_complex(), function_name,
              " expects a real input tensor, but got ", input.scalar_type());
  // 检查输出张量是否未定义或者是复数类型
  TORCH_CHECK(!out.defined() || out.is_complex(), function_name,
              " expects a complex output tensor, but got ", out.scalar_type());
  // 提升输入张量为适合 FFT 的类型
  input = promote_tensor_fft(input);
  // 获取输入张量的维度
  const auto input_dim = input.dim();
  // 包装维度以确保有效性
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  // 如果提供了数据点数目 n_opt，则使用它；否则使用输入张量的默认大小
  const auto n = n_opt.value_or(input.sym_sizes()[dim]);
  // 检查数据点数目是否有效
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  // 如果提供了 n_opt，则调整输入张量的大小以适应 FFT 输入要求
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  // 根据字符串获取归一化方式
  const auto norm = norm_from_string(norm_str, forward);

  Tensor ret;
  // 如果输出张量已定义且为正向变换，则使用指定的 out 执行 FFT
  if (out.defined() && forward) {
    ret = at::_fft_r2c_out(out, input, dim, static_cast<int64_t>(norm), onesided);
  } else {
    // 否则调用普通的实数到复数 FFT
    ret = at::_fft_r2c(input, dim, static_cast<int64_t>(norm), onesided);
  }

  // 如果是逆向变换，则对输出张量进行共轭处理
  if (!forward) {
    // FIXME: _fft_r2c doesn't support native r2c IFFT
    return out.defined() ? at::conj_physical_out(out, ret) : ret.conj();
  } else {
    return ret;
  }
}
// 对输入进行傅里叶变换，返回变换结果张量
Tensor fft_c2c(c10::string_view function_name,
               Tensor out, Tensor input, std::optional<SymInt> n_opt,
               int64_t unwrapped_dim, std::optional<c10::string_view> norm_str,
               bool forward) {
  // 检查输入张量是否为复数类型，若不是则抛出错误
  TORCH_CHECK(input.is_complex(), function_name,
              " expects a complex input tensor, but got ", input.scalar_type());
  // 获取输入张量的维度信息
  const auto input_dim = input.dim();
  // 根据指定的维度对输入张量进行包装，确保维度索引的有效性
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  // 获取变换点数 n，若未指定则使用输入张量在指定维度上的大小
  const auto n = n_opt.value_or(input.sym_sizes()[dim]);
  // 检查变换点数 n 是否有效
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  // 若指定了 n_opt，则调整输入张量的尺寸以符合变换要求
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  // 根据给定的规范字符串获取变换的归一化方式，并执行变换
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, forward));
  // 调用实际执行傅里叶变换的函数，并返回结果张量
  return fft_c2c_maybe_out(function_name, out, input, dim, norm, forward);
}

// 表示要进行变换的维度及该维度上的信号形状
struct ShapeAndDims {
  SymDimVector shape; // 信号形状向量
  DimVector dim;      // 变换维度向量
};

// 预处理 n 维 FFT 的 `s` 和 `dim` 参数
// 包装维度并应用默认行为
// 同时检查变换维度是否唯一且变换形状非空
ShapeAndDims canonicalize_fft_shape_and_dim_args(
    Tensor input, at::OptionalSymIntArrayRef shape, at::OptionalIntArrayRef dim) {
  // 获取输入张量的维度数
  const int64_t input_dim = input.dim();
  // 获取输入张量的符号化尺寸
  const SymIntArrayRef input_sizes = input.sym_sizes();
  ShapeAndDims ret; // 返回的结构体

  // 若指定了 dim 参数
  if (dim) {
    // 将 dim 转为 DimVector 类型并确保维度索引的有效性
    ret.dim.resize(dim->size());
    std::copy(dim->begin(), dim->end(), ret.dim.begin());
    maybe_wrap_dims(ret.dim, input_dim, /*wrap_scalars=*/false);

    // 检查变换维度是否唯一
    DimVector copy = ret.dim;
    std::sort(copy.begin(), copy.end());
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    TORCH_CHECK(duplicate == copy.end(), "FFT dims must be unique");
  }

  // 若指定了 shape 参数
  if (shape) {
    // 若同时指定了 dim 和 shape 参数，确保它们的长度相同
    TORCH_CHECK(!dim ||
                dim->size() == shape->size(),
                "When given, dim and shape arguments must have the same length");
    // 检查 shape 的长度不超过输入张量的维度数
    TORCH_CHECK(static_cast<int64_t>(shape->size()) <= input_dim,
                "Got shape with ", shape->size(), " values but input tensor "
                "only has ", input_dim, " dimensions.");
    const int64_t transform_ndim = shape->size();
    // 若未指定 dim 参数，则默认使用最后 shape.size() 维度作为变换维度
    if (!dim) {
      ret.dim.resize(transform_ndim);
      std::iota(ret.dim.begin(), ret.dim.end(), input_dim - transform_ndim);
    }

    // 将 shape 中的 -1 转换为默认长度
    ret.shape.resize(transform_ndim);
    for (const auto i : c10::irange(transform_ndim)) {
      const auto n = (*shape)[i];
      ret.shape[i] = n == -1 ? input_sizes[ret.dim[i]] : n;
    }
  } else if (!dim) {
    // 若未指定 shape 和 dim 参数，则使用输入张量的所有维度
    ret.dim.resize(input_dim);
    std::iota(ret.dim.begin(), ret.dim.end(), int64_t{0});
    ret.shape.resize(input_dim);
    std::copy(input_sizes.begin(), input_sizes.end(), ret.shape.begin());
  } else {
    // 调整返回值的形状大小，根据维度的数量来进行调整
    ret.shape.resize(ret.dim.size());
    // 遍历维度的数量，将输入尺寸中对应维度的大小赋值给返回值的形状
    for (const auto i : c10::irange(ret.dim.size())) {
      ret.shape[i] = input_sizes[ret.dim[i]];
    }
  }

  // 验证返回值的每个形状维度是否大于零，否则抛出异常
  for (const auto & shape : ret.shape) {
    TORCH_CHECK(shape > 0,
                "Invalid number of data points (", shape, ") specified");
  }

  // 返回调整后的返回值对象
  return ret;
// Complex to complex n-dimensional fft
// 执行复数到复数的n维傅立叶变换
Tensor fftn_c2c(
    c10::string_view function_name,  // 函数名称的字符串视图
    Tensor out,                      // 输出张量
    const Tensor& input,             // 输入张量的常量引用
    SymIntArrayRef shape,            // 形状的符号整数数组引用
    IntArrayRef dim,                 // 维度的整数数组引用
    std::optional<c10::string_view> norm_str,  // 可选的归一化字符串视图
    bool forward                      // 布尔值，指示是否进行正向变换
) {
  TORCH_CHECK(input.is_complex(), function_name, " expects a complex input tensor, but got", input.scalar_type());
  // 检查输入张量是否为复数类型，否则抛出错误
  Tensor x = resize_fft_input(input, dim, shape);
  // 调整输入张量以适应傅立叶变换的输入要求
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, forward));
  // 将归一化字符串视图转换为整数，并赋值给norm
  constexpr c10::string_view fname = "fftn";  // 函数名常量字符串视图
  return fft_c2c_maybe_out(fname, out, x, dim, norm, forward);
  // 调用复数到复数傅立叶变换函数，并返回结果
}

}  // namespace (anonymous)

// torch.fft.fft, analogous to NumPy's numpy.fft.fft
// 类似于NumPy的numpy.fft.fft的torch.fft.fft函数
Tensor fft_fft_symint(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm      // 可选的归一化字符串视图
) {
  return self.is_complex() ?
    fft_c2c("fft", {}, self, n, dim, norm, /*forward=*/true) :
    fft_r2c("fft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
  // 如果输入张量为复数，则调用复数到复数的傅立叶变换，否则调用实数到复数的傅立叶变换
}

Tensor& fft_fft_symint_out(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm,     // 可选的归一化字符串视图
    Tensor& out                               // 输出张量的引用
) {
  if (self.is_complex()) {
    fft_c2c("fft", out, self, n, dim, norm, /*forward=*/true);
    // 如果输入张量为复数，则调用复数到复数的傅立叶变换，将结果存入输出张量
  } else {
    fft_r2c("fft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
    // 如果输入张量为实数，则调用实数到复数的傅立叶变换，将结果存入输出张量
  }
  return out;
  // 返回输出张量的引用
}

Tensor fft_ifft_symint(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm      // 可选的归一化字符串视图
) {
  return self.is_complex() ?
    fft_c2c("ifft", {}, self, n, dim, norm, /*forward=*/false) :
    fft_r2c("ifft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
  // 如果输入张量为复数，则调用复数到复数的逆傅立叶变换，否则调用实数到复数的逆傅立叶变换
}

Tensor& fft_ifft_symint_out(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm,     // 可选的归一化字符串视图
    Tensor& out                               // 输出张量的引用
) {
  if (self.is_complex()) {
    fft_c2c("ifft", out, self, n, dim, norm, /*forward=*/false);
    // 如果输入张量为复数，则调用复数到复数的逆傅立叶变换，将结果存入输出张量
  } else {
    fft_r2c("ifft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
    // 如果输入张量为实数，则调用实数到复数的逆傅立叶变换，将结果存入输出张量
  }
  return out;
  // 返回输出张量的引用
}

Tensor fft_rfft_symint(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm      // 可选的归一化字符串视图
) {
  return fft_r2c("rfft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
  // 调用实数到复数的快速傅立叶变换，返回结果
}

Tensor& fft_rfft_symint_out(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm,     // 可选的归一化字符串视图
    Tensor& out                               // 输出张量的引用
) {
  fft_r2c("rfft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
  // 调用实数到复数的快速傅立叶变换，将结果存入输出张量
  return out;
  // 返回输出张量的引用
}

Tensor fft_irfft_symint(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm      // 可选的归一化字符串视图
) {
  return fft_c2r("irfft", {}, self, n, dim, norm, /*forward=*/false);
  // 调用复数到实数的逆快速傅立叶变换，返回结果
}

Tensor& fft_irfft_symint_out(
    const Tensor& self,                       // 输入张量的常量引用
    std::optional<SymInt> n,                  // 可选的符号整数
    int64_t dim,                              // 维度
    std::optional<c10::string_view> norm,     // 可选的归一化字符串视图
    Tensor& out                               // 输出张量的引用
) {
  fft_c2r("irfft", out, self, n, dim, norm, /*forward=*/false);
  // 调用复数到实数的逆快速傅立叶变换，将结果存入输出张量
  return out;
  // 返回输出张量的引用
}
// 计算高速傅立叶变换的半整数点版本，返回结果张量
Tensor fft_hfft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                std::optional<c10::string_view> norm) {
  // 调用 fft_c2r 函数进行高速傅立叶变换，使用 "hfft" 算法
  return fft_c2r("hfft", {}, self, n, dim, norm, /*forward=*/true);
}

// 计算高速傅立叶变换的半整数点版本，将结果存入预先分配的输出张量 out，并返回 out 引用
Tensor& fft_hfft_symint_out(const Tensor& self, std::optional<SymInt> n,
                     int64_t dim, std::optional<c10::string_view> norm, Tensor& out) {
  // 调用 fft_c2r 函数进行高速傅立叶变换，使用 "hfft" 算法，结果存入 out
  fft_c2r("hfft", out, self, n, dim, norm, /*forward=*/true);
  return out;
}

// 计算反高速傅立叶变换的半整数点版本，返回结果张量
Tensor fft_ihfft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                 std::optional<c10::string_view> norm) {
  // 调用 fft_r2c 函数进行反高速傅立叶变换，使用 "ihfft" 算法
  return fft_r2c("ihfft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
}

// 计算反高速傅立叶变换的半整数点版本，将结果存入预先分配的输出张量 out，并返回 out 引用
Tensor& fft_ihfft_symint_out(const Tensor& self, std::optional<SymInt> n,
                     int64_t dim, std::optional<c10::string_view> norm, Tensor& out) {
  // 调用 fft_r2c 函数进行反高速傅立叶变换，使用 "ihfft" 算法，结果存入 out
  fft_r2c("ihfft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
  return out;
}

// 计算多维高速傅立叶变换的半整数点版本，返回结果张量
Tensor fft_fftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                at::OptionalIntArrayRef dim,
                std::optional<c10::string_view> norm) {
  // 标准化输入形状和维度参数
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // 将输入张量提升为复数类型的张量
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  // 调用 fftn_c2c 函数进行多维高速傅立叶变换，使用 "fftn" 算法，正向变换
  return fftn_c2c("fftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/true);
}

// 计算多维高速傅立叶变换的半整数点版本，将结果存入预先分配的输出张量 out，并返回 out 引用
Tensor& fft_fftn_symint_out(const Tensor& self,
                     at::OptionalSymIntArrayRef s,
                     at::OptionalIntArrayRef dim,
                     std::optional<c10::string_view> norm, Tensor& out) {
  // 标准化输入形状和维度参数
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // 将输入张量提升为复数类型的张量
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  // 调用 fftn_c2c 函数进行多维高速傅立叶变换，使用 "fftn" 算法，正向变换，结果存入 out
  fftn_c2c("fftn", out, input, desc.shape, desc.dim, norm, /*forward=*/true);
  return out;
}

// 计算多维反高速傅立叶变换的半整数点版本，返回结果张量
Tensor fft_ifftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                at::OptionalIntArrayRef dim,
                std::optional<c10::string_view> norm) {
  // 标准化输入形状和维度参数
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // 将输入张量提升为复数类型的张量
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  // 调用 fftn_c2c 函数进行多维反高速傅立叶变换，使用 "ifftn" 算法，反向变换
  return fftn_c2c("ifftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/false);
}

// 计算多维反高速傅立叶变换的半整数点版本，将结果存入预先分配的输出张量 out，并返回 out 引用
Tensor& fft_ifftn_symint_out(const Tensor& self,
                      at::OptionalSymIntArrayRef s,
                      at::OptionalIntArrayRef dim,
                      std::optional<c10::string_view> norm, Tensor& out) {
  // 标准化输入形状和维度参数
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // 将输入张量提升为复数类型的张量
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  // 调用 fftn_c2c 函数进行多维反高速傅立叶变换，使用 "ifftn" 算法，反向变换，结果存入 out
  fftn_c2c("ifftn", out, input, desc.shape, desc.dim, norm, /*forward=*/false);
  return out;
}
static Tensor fft_rfftn_impl(Tensor out, const Tensor& self,
                             at::OptionalSymIntArrayRef s,
                             at::OptionalIntArrayRef dim,
                             const std::optional<c10::string_view>& norm_str) {
    // 检查输入张量是否为实数类型，如果不是则报错
    TORCH_CHECK(!self.is_complex(), "rfftn expects a real-valued input tensor, but got ", self.scalar_type());
    // 规范化 FFT 的形状和维度参数
    auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
    // 检查是否至少有一个轴进行 FFT 变换
    TORCH_CHECK(!desc.shape.empty(), "rfftn must transform at least one axis");
    // 将输入张量升级为适合 FFT 的张量，确保其为实数类型
    Tensor input = promote_tensor_fft(self, /*require_complex=*/false);
    // 调整 FFT 输入的大小，以匹配指定的维度和形状
    Tensor x = resize_fft_input(input, desc.dim, desc.shape);
    // 根据规范化后的参数，解析并获取归一化常数
    const auto norm = static_cast<int64_t>(norm_from_string(norm_str, /*forward=*/true));
    // 定义 FFT 函数名称为 "rfftn"
    constexpr c10::string_view fname = "rfftn";
    // 执行实数输入的 FFT，返回结果
    return fft_r2c_maybe_out(fname, out, x, desc.dim, norm, /*onesided=*/true);
}

Tensor fft_rfftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                        at::OptionalIntArrayRef dim,
                        std::optional<c10::string_view> norm_str) {
    // 调用 FFT 实现函数，并返回结果张量
    return fft_rfftn_impl({}, self, s, dim, norm_str);
}

Tensor& fft_rfftn_symint_out(const Tensor& self,
                             at::OptionalSymIntArrayRef s,
                             at::OptionalIntArrayRef dim,
                             std::optional<c10::string_view> norm_str, Tensor& out) {
    // 调用 FFT 实现函数，将结果写入预分配的输出张量中，并返回输出张量的引用
    fft_rfftn_impl(out, self, s, dim, norm_str);
    return out;
}

static ShapeAndDims canonicalize_fft_c2r_shape_and_dim_args(
    c10::string_view fname, const Tensor& self,
    const at::OptionalSymIntArrayRef& s,
    const at::OptionalIntArrayRef& dims,
    SymInt& last_dim_size) {
    // 规范化 FFT 的形状和维度参数
    auto desc = canonicalize_fft_shape_and_dim_args(self, s, dims);
    // 检查是否至少有一个轴进行 FFT 变换
    TORCH_CHECK(!desc.shape.empty(), fname, " must transform at least one axis");

    // 预期的埃尔米特对称维度的输出大小
    last_dim_size = [&] {
        // 如果最后一个维度的形状未指定或为 -1，则进行默认处理
        if (!s.has_value() || (s->back() == -1)) {
            const auto last_dim = desc.dim.back();
            return 2 * (self.sym_sizes()[last_dim] - 1);
        }
        return desc.shape.back();
    }();
    // 检查埃尔米特对称维度的输出大小是否有效
    TORCH_CHECK(last_dim_size >= 1, "Invalid number of data points (", last_dim_size, ") specified");

    // 预期复埃尔米特数据的输入大小
    desc.shape.back() = last_dim_size / 2 + 1;
    // 返回规范化后的形状和维度信息
    return desc;
}
static Tensor fft_irfftn_impl(Tensor out, const Tensor& self,
                              at::OptionalSymIntArrayRef s,
                              at::OptionalIntArrayRef dim,
                              const std::optional<c10::string_view>& norm_str) {
  // 初始化最后一个维度大小为零
  SymInt last_dim_size = 0;
  // 规范化 FFT 的实部转换为复数的形状和维度参数
  auto desc = canonicalize_fft_c2r_shape_and_dim_args(
      "irfftn", self, s, dim, last_dim_size);
  // 提升输入张量以满足 FFT 的要求，需要确保为复数类型
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  // 调整 FFT 输入的形状和维度
  Tensor x = resize_fft_input(input, desc.dim, desc.shape);
  // 从字符串中解析规范化参数，转换为整数
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, /*forward=*/false));
  // 函数名称常量
  constexpr c10::string_view fname = "irfftn";
  // 调用复数到实部的 FFT，可能将结果存入 out 张量
  return fft_c2r_maybe_out(fname, out, x, desc.dim, norm, last_dim_size);
}

Tensor fft_irfftn_symint(const Tensor& self,
                  at::OptionalSymIntArrayRef s,
                  at::OptionalIntArrayRef dim,
                  std::optional<c10::string_view> norm_str) {
  // 调用实部反 FFT 实现函数，返回结果张量
  return fft_irfftn_impl({}, self, s, dim, norm_str);
}

Tensor& fft_irfftn_symint_out(const Tensor& self,
                       at::OptionalSymIntArrayRef s,
                       at::OptionalIntArrayRef dim,
                       std::optional<c10::string_view> norm_str, Tensor& out) {
  // 调用实部反 FFT 实现函数，将结果存入给定的 out 张量
  fft_irfftn_impl(out, self, s, dim, norm_str);
  return out;
}

static Tensor fft_hfftn_impl(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim,
    std::optional<c10::string_view> norm_str,
    const Tensor& out) {
  // 函数名称常量
  constexpr c10::string_view fname = "hfftn";
  // 初始化最后一个维度大小为零
  SymInt last_dim_size = 0;
  // 规范化 FFT 的实部转换为复数的形状和维度参数
  auto desc = canonicalize_fft_c2r_shape_and_dim_args(
      fname, self, s, dim, last_dim_size);
  // 提升输入张量以满足 FFT 的要求，需要确保为复数类型
  auto input = promote_tensor_fft(self, /*require_complex=*/true);
  // 调整 FFT 输入的形状和维度
  auto x = resize_fft_input(input, desc.dim, desc.shape);
  // 从字符串中解析规范化参数，转换为整数
  const auto norm = static_cast<int64_t>(
      norm_from_string(norm_str, /*forward=*/true));

  Tensor tmp;
  // 如果有多于一个维度，则执行复数到复数的 FFT
  if (desc.dim.size() > 1) {
    auto c2c_dims = IntArrayRef(desc.dim).slice(0, desc.dim.size() - 1);
    tmp = at::_fft_c2c(x, c2c_dims, norm, /*forward=*/true);
  } else {
    tmp = x;
  }

  // 获取最后一个维度
  const auto last_dim = desc.dim.back();
  // 对 tmp 张量执行共轭操作
  tmp = tmp.conj();
  // 执行复数到实部的 FFT，可能将结果存入 out 张量
  return fft_c2r_maybe_out(fname, out, tmp, last_dim, norm, last_dim_size);
}

Tensor fft_hfftn_symint(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim,
    std::optional<c10::string_view> norm) {
  // 调用实部的高维 FFT 实现函数，返回结果张量
  return fft_hfftn_impl(self, s, dim, norm, {});
}

const Tensor& fft_hfftn_symint_out(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim, std::optional<c10::string_view> norm,
    const Tensor& out) {
  // 调用实部的高维 FFT 实现函数，将结果存入给定的 out 张量
  fft_hfftn_impl(self, s, dim, norm, out);
  return out;
}

static Tensor fft_ihfftn_impl(
    const Tensor& self,
    const at::OptionalSymIntArrayRef& s,
    const at::OptionalIntArrayRef& dim,
    const std::optional<c10::string_view>& norm_str,
    // 定义函数 ihfftn，接受两个输入参数 self 和 s，以及一个输出参数 out，返回一个 Tensor
    const Tensor& ihfftn(
        // 声明常量 fname 为字符串 "ihfftn"
        constexpr c10::string_view fname = "ihfftn";
        // 标准化 FFT 的形状和维度参数，并返回描述符
        auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
        // 检查是否至少有一个轴需要进行 ihfftn 变换
        TORCH_CHECK(!desc.shape.empty(), "ihfftn must transform at least one axis");
        // 为输入张量 self 提升 FFT 运算所需，要求不是复数形式
        auto input = promote_tensor_fft(self, /*require_complex=*/false);
        // 调整 FFT 输入的形状以匹配 desc.dim 和 desc.shape
        auto x = resize_fft_input(input, desc.dim, desc.shape);
        // 根据 norm_str 解析得到 norm 值，要求是反向变换
        const auto norm = static_cast<int64_t>(
            norm_from_string(norm_str, /*forward=*/false));
    
        // 获取 desc.dim 中的最后一个维度
        const auto last_dim = desc.dim.back();
        // 执行实数到复数的 FFT 变换，返回临时结果 tmp
        auto tmp = at::_fft_r2c(x, last_dim, norm, /*onesided=*/true);
        // 如果 desc.dim 的大小为 1，则根据 out 是否已定义执行不同操作
        if (desc.dim.size() == 1) {
            return out.defined() ? at::conj_physical_out(tmp, out) : tmp.conj();
        }
    
        // 对 tmp 执行共轭操作
        tmp = at::conj_physical(tmp);
        // 获取 c2c_dims，即 desc.dim 去除最后一个维度后的部分
        auto c2c_dims = IntArrayRef(desc.dim).slice(0, desc.dim.size() - 1);
        // 执行可能带有输出的 C2C FFT 变换，返回结果
        return fft_c2c_maybe_out(fname, out, tmp, c2c_dims, norm, /*forward=*/false);
    }
}

// 对称整数参数的逆快速傅里叶变换（IFFT）函数，返回处理后的张量
Tensor fft_ihfftn_symint(
    const Tensor& self,                             // 输入张量
    at::OptionalSymIntArrayRef s,                   // 对称整数数组引用（可选）
    at::OptionalIntArrayRef dim,                    // 维度数组引用（可选）
    std::optional<c10::string_view> norm) {         // 标准化字符串视图（可选）
  return fft_ihfftn_impl(self, s, dim, norm, {});   // 调用内部的逆快速傅里叶变换实现函数，返回结果张量
}

// 对称整数参数的逆快速傅里叶变换（IFFT）函数，将结果写入预分配的输出张量，返回处理后的输出张量
const Tensor& fft_ihfftn_symint_out(
    const Tensor& self,                             // 输入张量
    at::OptionalSymIntArrayRef s,                   // 对称整数数组引用（可选）
    at::OptionalIntArrayRef dim,                    // 维度数组引用（可选）
    std::optional<c10::string_view> norm,           // 标准化字符串视图（可选）
    const Tensor& out) {                            // 输出张量
  fft_ihfftn_impl(self, s, dim, norm, out);         // 调用内部的逆快速傅里叶变换实现函数，将结果写入输出张量
  return out;                                       // 返回输出张量的引用
}

// 对称整数参数的二维快速傅里叶变换（FFT）函数，返回处理后的张量
Tensor fft_fft2_symint(const Tensor& self,          // 输入张量
                at::OptionalSymIntArrayRef s,       // 对称整数数组引用（可选）
                IntArrayRef dim,                    // 维度数组引用
                std::optional<c10::string_view> norm) {  // 标准化字符串视图（可选）
  return native::fft_fftn_symint(self, s, dim, std::move(norm));  // 调用本地的 n 维快速傅里叶变换（FFT）函数，返回结果张量
}

// 对称整数参数的二维快速傅里叶变换（FFT）函数，将结果写入预分配的输出张量，返回处理后的输出张量
Tensor& fft_fft2_symint_out(const Tensor& self,    // 输入张量
                     at::OptionalSymIntArrayRef s, // 对称整数数组引用（可选）
                     IntArrayRef dim,              // 维度数组引用
                     std::optional<c10::string_view> norm,  // 标准化字符串视图（可选）
                     Tensor& out) {                // 输出张量
  return native::fft_fftn_symint_out(self, s, dim, std::move(norm), out);  // 调用本地的 n 维快速傅里叶变换（FFT）函数，将结果写入输出张量，返回输出张量的引用
}

// 对称整数参数的二维逆快速傅里叶变换（IFFT）函数，返回处理后的张量
Tensor fft_ifft2_symint(const Tensor& self,         // 输入张量
                at::OptionalSymIntArrayRef s,       // 对称整数数组引用（可选）
                IntArrayRef dim,                    // 维度数组引用
                std::optional<c10::string_view> norm) {  // 标准化字符串视图（可选）
  return native::fft_ifftn_symint(self, s, dim, std::move(norm));  // 调用本地的 n 维逆快速傅里叶变换（IFFT）函数，返回结果张量
}

// 对称整数参数的二维逆快速傅里叶变换（IFFT）函数，将结果写入预分配的输出张量，返回处理后的输出张量
Tensor& fft_ifft2_symint_out(const Tensor& self,    // 输入张量
                      at::OptionalSymIntArrayRef s, // 对称整数数组引用（可选）
                      IntArrayRef dim,              // 维度数组引用
                      std::optional<c10::string_view> norm,  // 标准化字符串视图（可选）
                      Tensor& out) {                // 输出张量
  return native::fft_ifftn_symint_out(self, s, dim, std::move(norm), out);  // 调用本地的 n 维逆快速傅里叶变换（IFFT）函数，将结果写入输出张量，返回输出张量的引用
}

// 对称整数参数的二维实部快速傅里叶变换（RFFT）函数，返回处理后的张量
Tensor fft_rfft2_symint(const Tensor& self,         // 输入张量
                at::OptionalSymIntArrayRef s,       // 对称整数数组引用（可选）
                IntArrayRef dim,                    // 维度数组引用
                std::optional<c10::string_view> norm) {  // 标准化字符串视图（可选）
  return native::fft_rfftn_symint(self, s, dim, std::move(norm));  // 调用本地的 n 维实部快速傅里叶变换（RFFT）函数，返回结果张量
}

// 对称整数参数的二维实部快速傅里叶变换（RFFT）函数，将结果写入预分配的输出张量，返回处理后的输出张量
Tensor& fft_rfft2_symint_out(const Tensor& self,    // 输入张量
                      at::OptionalSymIntArrayRef s, // 对称整数数组引用（可选）
                      IntArrayRef dim,              // 维度数组引用
                      std::optional<c10::string_view> norm,  // 标准化字符串视图（可选）
                      Tensor& out) {                // 输出张量
  return native::fft_rfftn_symint_out(self, s, dim, std::move(norm), out);  // 调用本地的 n 维实部快速傅里叶变换（RFFT）函数，将结果写入输出张量，返回输出张量的引用
}

// 对称整数参数的二维逆实部快速傅里叶变换（IRFFT）函数，返回处理后的张量
Tensor fft_irfft2_symint(const Tensor& self,        // 输入张量
                  at::OptionalSymIntArrayRef s,     // 对称整数数组引用（可选）
                  IntArrayRef dim,                  // 维度数组引用
                  std::optional<c10::string_view> norm) {  // 标准化字符串视图（可选）
  return native::fft_irfftn_symint(self, s, dim, std::move(norm));  // 调用本地的 n 维逆实部快速傅里叶变换（IRFFT）函数，返回结果张量
}

// 对称整数参数的二维逆实部快速傅里叶变换（IRFFT）函数，将结果写入预分配的输出张量，返回处理后的输出张量
Tensor& fft_irfft2_symint_out(const Tensor& self,   // 输入张量
                        at::OptionalSymIntArrayRef s,  // 对称整数数组引用（可选）
                        IntArrayRef dim,               // 维度数组引用
                        std::optional<c10::string_view> norm,  // 标准化字符串视图（可选）
                        Tensor& out) {                 // 输出张量
  return native::fft_irfftn_symint_out(self, s, dim, std::move(norm), out);  // 调用本地的 n 维逆实部快速傅里叶变换（IRFFT）函数，将结果写入输出张量，返回输出张量的引用
}

// 对称整数参数的二维 Hermite 插值的快速傅里叶变换（HFFT）函数，将结果写入预分配的输出张量，返回处理后的输出张量的常量引用
const Tensor& fft_hfft2_symint_out(
    const Tensor& self,                             // 输入张量
    at::OptionalSymIntArrayRef s,                   // 对称整数数组
    // 调用 native 命名空间中的 fft_ihfftn_symint_out 函数，传入以下参数：
    // - self: 作为输入的张量对象
    // - s: 表示是否进行共轭操作的标志位
    // - dim: 执行傅里叶逆变换的维度
    // - std::move(norm): 可选的标准化向量（可能为空）
    // - out: 输出张量，用于存储结果
    return native::fft_ihfftn_symint_out(self, s, dim, std::move(norm), out);
}

// 对于给定的输入张量和可选的符号整数数组引用，执行反向离散傅立叶变换（IHFFT）。
Tensor fft_ihfft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                  IntArrayRef dim, std::optional<c10::string_view> norm) {
  // 调用底层实现的对称整数参数版本的多维反向离散傅立叶变换
  return native::fft_ihfftn_symint(self, s, dim, std::move(norm));
}

// 计算离散傅立叶变换频率的快速计算方法，并将结果写入输出张量。
Tensor& fft_fftfreq_out(int64_t n, double d, Tensor& out) {
  // 获取输出张量的数据类型
  ScalarType dtype = out.scalar_type();
  // 检查数据类型必须是浮点数或复数类型
  TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype),
              "fftfreq requires a floating point or complex dtype");
  // 使用arange生成等间隔的数列，填充输出张量
  at::arange_out(out, n);
  // 取右半部分切片，填充负半部分的数据
  auto right_slice = out.slice(0, (n + 1) / 2, 0);
  at::arange_out(right_slice, -(n/2), 0, 1);
  // 返回归一化后的结果张量
  return out.mul_(1.0 / (n * d));  // 比使用div_稍快
}

// 计算离散傅立叶变换频率的快速计算方法，并返回结果张量。
Tensor fft_fftfreq(int64_t n, double d,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 构建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 创建一个空张量以存储结果
  auto out = at::empty({n}, options);
  // 调用out作为输出张量的fft_fftfreq_out方法来计算结果
  return native::fft_fftfreq_out(n, d, out);
}

// 计算实数离散傅立叶变换频率的快速计算方法，并将结果写入输出张量。
Tensor& fft_rfftfreq_out(int64_t n, double d, Tensor& out) {
  // 获取输出张量的数据类型
  ScalarType dtype = out.scalar_type();
  // 检查数据类型必须是浮点数或复数类型
  TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype),
              "rfftfreq requires a floating point or complex dtype");
  // 使用native::arange_out生成等间隔的数列，填充输出张量
  native::arange_out(n/2 + 1, out);
  // 返回归一化后的结果张量
  return out.mul_(1.0 / (n * d));  // 比使用div_稍快
}

// 计算实数离散傅立叶变换频率的快速计算方法，并返回结果张量。
Tensor fft_rfftfreq(int64_t n, double d,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 构建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 创建一个空张量以存储结果
  auto out = at::empty({n/2 + 1}, options);
  // 调用out作为输出张量的fft_rfftfreq_out方法来计算结果
  return native::fft_rfftfreq_out(n, d, out);
}

// 如果指定了维度数组，则根据self.dim()进行包装。否则返回所有维度的向量。
static DimVector default_alldims(const Tensor& self, at::OptionalIntArrayRef dim_opt) {
  DimVector dim;
  // 如果dim_opt有值，则将其展开为IntArrayRef
  if (dim_opt) {
    IntArrayRef dim_unwrapped = *dim_opt;
    // 将dim的大小调整为与dim_unwrapped相同
    dim.resize(dim_unwrapped.size());
    // 遍历dim_unwrapped数组，根据self.dim()包装维度
    for (const auto i : c10::irange(dim.size())) {
      dim[i] = maybe_wrap_dim(dim_unwrapped[i], self.dim(), /*wrap_scalars=*/false);
    }
  } else {
    // 如果dim_opt为空，则将dim的大小调整为self.dim()，并填充为连续的整数序列
    dim.resize(self.dim());
    std::iota(dim.begin(), dim.end(), 0);
  }
  // 返回处理后的dim向量
  return dim;
}

// 将输入张量在指定维度上进行移位，以实现频域中心化。
Tensor fft_fftshift(const Tensor& x, at::OptionalIntArrayRef dim_opt) {
  // 获取默认的维度向量
  auto dim = default_alldims(x, dim_opt);

  // 获取输入张量的符号整数数组引用
  SymIntArrayRef x_sizes = x.sym_sizes();
  // 创建移位向量
  SymDimVector shift(dim.size());
  // 遍历维度向量，计算移位量
  for (const auto i : c10::irange(dim.size())) {
    shift[i] = x_sizes[dim[i]] / 2;
  }

  // 在指定维度上对输入张量进行滚动移位，返回结果张量
  return at::roll_symint(x, shift, dim);
}
// 定义一个函数 fft_ifftshift，用于对输入张量 x 进行 FFT 前的逆移位操作
Tensor fft_ifftshift(const Tensor& x, at::OptionalIntArrayRef dim_opt) {
  // 获得默认维度列表
  auto dim = default_alldims(x, dim_opt);

  // 获取张量 x 的符号化尺寸
  SymIntArrayRef x_sizes = x.sym_sizes();
  // 创建一个与维度列表长度相同的符号化维度向量 shift
  SymDimVector shift(dim.size());
  // 遍历维度列表，计算每个维度的 shift
  for (const auto i : c10::irange(dim.size())) {
    shift[i] = (x_sizes[dim[i]] + 1) / 2;
  }

  // 调用 ATen 函数 at::roll_symint 进行符号化整数的滚动操作，返回处理后的张量
  return at::roll_symint(x, shift, dim);
}

// 获取 CuFFT 计划缓存的最大大小
int64_t _cufft_get_plan_cache_max_size(DeviceIndex device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheMaxSize(device_index);
}

// 设置 CuFFT 计划缓存的最大大小
void _cufft_set_plan_cache_max_size(DeviceIndex device_index, int64_t max_size) {
  detail::getCUDAHooks().cuFFTSetPlanCacheMaxSize(device_index, max_size);
}

// 获取当前 CuFFT 计划缓存的大小
int64_t _cufft_get_plan_cache_size(DeviceIndex device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheSize(device_index);
}

// 清除 CuFFT 计划缓存
void _cufft_clear_plan_cache(DeviceIndex device_index) {
  detail::getCUDAHooks().cuFFTClearPlanCache(device_index);
}

// 写入函数，将 optional 类型的值写入流 SS 中
template <typename Stream, typename T>
static Stream& write_opt(Stream& SS, const optional<T>& value) {
  if (value) {
    SS << *value;
  } else {
    SS << "None";
  }
  return SS;
}

/* 短时傅里叶变换（STFT），用于信号分析。
 *
 * 该函数模仿 librosa 的设计，支持复数时域信号和复数窗口。
 */
Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
            const optional<int64_t> win_lengthOpt, const std::optional<Tensor>& window_opt,
            const bool center, c10::string_view mode, const bool normalized,
            const optional<bool> onesidedOpt, const optional<bool> return_complexOpt) {
  // 从可选的窗口 tensor 中借用一个 MaybeOwned<Tensor> 对象
  c10::MaybeOwned<Tensor> window_maybe_owned = at::borrow_from_optional_tensor(window_opt);
  // 获得窗口 tensor 的引用
  const Tensor& window = *window_maybe_owned;

  // 如果窗口未定义，则发出警告
  if (!window.defined()) {
    TORCH_WARN_ONCE(
        "A window was not provided. A rectangular window will be applied,"
        "which is known to cause spectral leakage. "
        "Other windows such as torch.hann_window or torch.hamming_window "
        "can are recommended to reduce spectral leakage."
        "To suppress this warning and use a rectangular window, explicitly set "
        "`window=torch.ones(n_fft, device=<device>)`.");
  }

  // 定义宏 REPR，用于生成表示 STFT 函数调用的字符串描述
  #define REPR(SS) \
    SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", normalized=" << normalized << ", onesided="; \
    write_opt(SS, onesidedOpt) << ", return_complex=";
    // 将 SS 和 return_complexOpt 作为参数调用 write_opt 函数，并添加右括号和空格
    write_opt(SS, return_complexOpt) << ") ";

  // 检查窗口是否已定义，若定义则检查其设备是否与当前对象的设备一致
  TORCH_CHECK(!window.defined() || window.device() == self.device(),
              "stft input and window must be on the same device but got self on ",
              self.device(), " and window on ", window.device())

  // 默认初始化 hop_length 和 win_length
  auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  auto win_length = win_lengthOpt.value_or(n_fft);

  // 确定是否返回复数类型的输出
  const bool return_complex = return_complexOpt.value_or(
      self.is_complex() || (window.defined() && window.is_complex()));

  // 如果不返回复数类型，则进行额外的检查和警告
  if (!return_complex) {
    TORCH_CHECK(return_complexOpt.has_value(),
        "stft requires the return_complex parameter be given for real inputs, "
        "and will further require that return_complex=True in a future PyTorch release.");

    TORCH_WARN_ONCE(
        "stft with return_complex=False is deprecated. In a future pytorch "
        "release, stft will return complex tensors for all inputs, and "
        "return_complex=False will raise an error.\n"
        "Note: you can still call torch.view_as_real on the complex output to "
        "recover the old return format.");
  }

  // 检查输入张量的类型，必须是浮点数或复数类型
  if (!at::isFloatingType(self.scalar_type()) && !at::isComplexType(self.scalar_type())) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor of floating point or complex values";
    AT_ERROR(ss.str());
  }

  // 检查输入张量的维度，必须是1D或2D
  if (self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor";
    AT_ERROR(ss.str());
  }

  // 如果输入是1D张量，则扩展为2D，第一维度为batch
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
  }

  // 如果 center 参数为真，则对输入进行填充，以确保输出的时间维度与输入一致
  if (center) {
    const auto input_shape = input.sizes();
    const auto input_dim = input_shape.size();
    const auto extra_dims = std::max(size_t{3}, input_dim) - input_dim;
    const auto pad_amount = n_fft / 2;

    DimVector extended_shape(extra_dims, 1);
    extended_shape.append(input_shape.begin(), input_shape.end());
    input = at::pad(input.view(extended_shape), {pad_amount, pad_amount}, mode);
    input = input.view(IntArrayRef(input.sizes()).slice(extra_dims));
  }

  // 获取输入张量的批次大小和长度
  int64_t batch = input.size(0);
  int64_t len = input.size(1);

  // 检查 n_fft 参数的有效性
  if (n_fft <= 0 || n_fft > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < n_fft < " << len
             << ", but got n_fft=" << win_length;
    AT_ERROR(ss.str());
  }

  // 检查 hop_length 参数的有效性
  if (hop_length <= 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected hop_length > 0, but got hop_length=" << hop_length;
    AT_ERROR(ss.str());
  }

  // 检查 win_length 参数的有效性
  if (win_length <= 0 || win_length > n_fft) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft, but got win_length="
             << win_length;
    AT_ERROR(ss.str());
  }

  // 如果窗口已定义，则检查其维度和大小是否符合预期
  if (window.defined() && (window.dim() != 1 || window.size(0) != win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to win_length="
             << win_length << ", but got window with size " << window.sizes();
    // 抛出错误信息并终止程序执行
    AT_ERROR(ss.str());
  }
  // 取消宏定义 REPR
  #undef REPR
  // 复制 window 到 window_，以备后续可能的修改
  auto window_ = window;
  // 如果窗口长度小于 FFT 窗口长度
  if (win_length < n_fft) {
    // 在信号中心填充数据
    auto left = (n_fft - win_length) / 2;
    // 如果存在窗口函数
    if (window.defined()) {
      // 创建一个全零的 n_fft 长度的张量作为 window_
      window_ = at::zeros({n_fft}, window.options());
      // 将原始窗口函数复制到新的 window_ 中心部分
      window_.narrow(0, left, win_length).copy_(window);
    } else {
      // 创建一个全零的 n_fft 长度的张量，并用值 1 填充中心部分
      window_ = at::zeros({n_fft}, self.options());
      window_.narrow(0, left, win_length).fill_(1);
    }
  }
  // 计算输出帧的数量
  int64_t n_frames = 1 + (len - n_fft) / hop_length;
  // 将输入信号 input 重塑为 (batch, n_frames, n_fft) 的张量
  input = input.as_strided(
    {batch, n_frames, n_fft},
    {input.stride(0), hop_length * input.stride(1), input.stride(1)}
  );
  // 如果定义了窗口函数，将 input 乘以 window_
  if (window_.defined()) {
    input = input.mul(window_);
  }

  // 进行 FFT 转换并转置以得到 (batch x fft_size x num_frames) 的输出
  const bool complex_fft = input.is_complex();
  const auto onesided = onesidedOpt.value_or(!complex_fft);

  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
  Tensor out;
  if (complex_fft) {
    // 如果输入是复数，执行复数到复数的 FFT 变换
    TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
    out = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/true);
  } else {
    // 如果输入是实数，执行实数到复数的 FFT 变换
    out = at::_fft_r2c(input, input.dim() - 1, static_cast<int64_t>(norm), onesided);
  }
  // 转置输出张量的第 1 和第 2 维度
  out.transpose_(1, 2);

  // 如果 self 的维度为 1，压缩第 0 维
  if (self.dim() == 1) {
    out.squeeze_(0);
  }

  // 如果需要返回复数结果，则直接返回 out
  if (return_complex) {
    return out;
  } else {
    // 否则将 out 视为实数张量返回
    return at::view_as_real(out);
  }
}

// 定义 stft 函数，进行短时傅里叶变换
Tensor stft(
    const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt, const std::optional<Tensor>& window_opt,
    const bool normalized,
    const optional<bool> onesidedOpt, const optional<bool> return_complexOpt) {
  // 调用 ATen 提供的 stft 函数进行短时傅里叶变换计算
  return at::stft(
      self, n_fft, hop_lengthOpt, win_lengthOpt, window_opt,
      /*center=*/false, /*mode=*/"constant", normalized, onesidedOpt,
      return_complexOpt);
}

// 从旧式实数张量创建复数张量，支持 istft 转换到需要复数输入的过渡阶段
// 注意：这可能返回输入张量的视图，或者必要时进行克隆
static Tensor as_complex(const Tensor& self) {
  // 检查是否可以将输入张量视为复数张量的视图
  const bool can_view_as_complex = [&]{
    auto strides = self.strides();
    for (const auto i : c10::irange(static_cast<int64_t>(strides.size()) - 1)) {
      if (strides[i] % 2 != 0) {
        return false;
      }
    }
    return strides.back() == 1 && self.storage_offset() % 2 == 0;
  }();
  // 根据是否可以视为复数张量，返回对应的复数张量
  return at::view_as_complex(can_view_as_complex ? self : self.clone(MemoryFormat::Contiguous));
}

/* 逆短时傅里叶变换
 *
 * 这是模仿 librosa 的实现，但支持复数时域信号和复数窗口。
 */
Tensor istft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
             const optional<int64_t> win_lengthOpt, const std::optional<Tensor>& window_opt,
             const bool center, const bool normalized, const std::optional<bool> onesidedOpt,
             const optional<int64_t> lengthOpt, const bool return_complex) {
  // 从可选的张量中获取窗口信息
  c10::MaybeOwned<Tensor> window_maybe_owned = at::borrow_from_optional_tensor(window_opt);
  const Tensor& window = *window_maybe_owned;

  // 如果未提供窗口，发出警告
  if (!window.defined()) {
    TORCH_WARN_ONCE(
        "A window was not provided. A rectangular window will be applied."
        "Please provide the same window used by stft to make the inversion "
        "lossless."
        "To suppress this warning and use a rectangular window, explicitly set "
        "`window=torch.ones(n_fft, device=<device>)`.");
  }

  // 定义一个宏用于生成 istft 的描述字符串
  #define REPR(SS) \
    SS << "istft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", center=" << center << ", normalized=" << normalized << ", onesided="; \
    write_opt(SS, onesidedOpt) << ", length="; \
    // 构建字符串流，用于生成输出消息
    write_opt(SS, lengthOpt) << ", return_complex=" << return_complex << ") "
  
  // 检查窗口与输入张量的设备匹配性
  TORCH_CHECK(!window.defined() || window.device() == self.device(),
              "istft input and window must be on the same device but got self on ",
              self.device(), " and window on ", window.device())
  
  // 设置默认的 hop_length 和 win_length
  const auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  const auto win_length = win_lengthOpt.value_or(n_fft);
  
  // 检查输入张量是否为复数类型
  TORCH_CHECK(self.is_complex(),
              "istft requires a complex-valued input tensor matching the "
              "output from stft with return_complex=True.");
  // 将输入张量视为实部解析后的结果
  Tensor input = at::view_as_real(self.resolve_conj());
  // 获取输入张量的维度信息
  const auto input_dim = input.dim();
  // 获取输入张量中的帧数
  const auto n_frames = input.size(-2);
  // 获取输入张量中的 FFT 大小
  const auto fft_size = input.size(-3);
  
  // 计算预期的输出信号长度
  const auto expected_output_signal_len = n_fft + hop_length * (n_frames - 1);
  
  // 创建选项，以便在指定设备和数据类型下操作
  const auto options = at::device(input.device()).dtype(input.dtype());
  // 检查输入张量是否为空
  if (input.numel() == 0) {
    std::ostringstream ss;
    REPR(ss) << ": input tensor cannot be empty.";
    AT_ERROR(ss.str());
  }
  // 检查输入张量的维度是否为3或4
  if (input_dim != 3 && input_dim != 4) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor with 3 or 4 dimensions, but got " << input_dim;
    AT_ERROR(ss.str());
  }
  // 检查输入张量的最后一个维度是否为2，对应实部和虚部
  if (input.size(-1) != 2) {
    std::ostringstream ss;
    REPR(ss) << ": expected the last dimension to be 2 (corresponding to real and imaginary parts), but got " << self.size(-1);
    AT_ERROR(ss.str());
  }
  
  // 检查是否为单侧频谱
  const bool onesided = onesidedOpt.value_or(fft_size != n_fft);
  if (onesided) {
    // 当为单侧频谱时，检查输入张量的频率维度是否匹配预期
    if (n_fft / 2 + 1 != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft / 2 + 1 when onesided=True, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  } else {
    // 当为双侧频谱时，检查输入张量的频率维度是否匹配预期
    if (n_fft != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft when onesided=False, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  }
  
  // 检查 hop_length 和 win_length 的取值范围
  if (!(0 < hop_length && hop_length <= win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < hop_length <= win_length";
    AT_ERROR(ss.str());
  }
  
  // 检查 win_length 和 n_fft 的取值范围
  if (!(0 < win_length && win_length <= n_fft)) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft";
    AT_ERROR(ss.str());
  }
  // 检查窗口张量的形状是否有效，应为1维且长度为 win_length
  if (window.defined()) {
    if (window.dim() != 1 || window.size(0) != win_length) {
      std::ostringstream ss;
      REPR(ss) << ": Invalid window shape. window has to be 1D and length of `win_length`";
      AT_ERROR(ss.str());
    }
  }
  
  // 如果未定义窗口张量，则创建长度为 win_length 的全1张量
  Tensor window_tmp = window.defined() ? window : at::ones({win_length,}, options);
  // 当 win_length 不等于 n_fft 时，通过填充0使窗口居中
  if (win_length != n_fft) {
    int64_t left = (n_fft - win_length) / 2;
    window_tmp = at::constant_pad_nd(window_tmp, {left, n_fft - win_length - left}, 0);
  TORCH_INTERNAL_ASSERT(window_tmp.size(0) == n_fft);
  // 确保窗口临时张量的第一个维度大小等于 n_fft

  if (input_dim == 3) {
    input = input.unsqueeze(0);
    // 如果输入的维度为 3，则在第0维度上添加一个维度
  }

  input = as_complex(input.transpose(1, 2));  // size: (channel, n_frames, fft_size)
  // 将输入张量转换为复数表示，并将第1和第2维度进行转置

  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n;
  // 根据 normalized 变量选择 FFT 的归一化模式

  if (return_complex) {
    TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
    // 如果需要返回复数结果，则检查不能使用单边输出（onesided）

    input = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/false);  // size: (channel, n_frames, n_fft)
    // 对输入进行复数到复数的 FFT 变换，输入数据沿倒数第2维进行变换
  } else {
    TORCH_CHECK(!window.defined() || !window.is_complex(),
                "Complex windows are incompatible with return_complex=False");
    // 如果不需要返回复数结果，则检查窗口不能是复数

    if (!onesided) {
      input = input.slice(-1, 0, n_fft / 2 + 1);
      // 如果不是单边输出，则从倒数第2维度上切片，保留前 n_fft/2 + 1 个元素
    }

    input = at::_fft_c2r(input, input.dim() - 1, static_cast<int64_t>(norm), n_fft);  // size: (channel, n_frames, n_fft)
    // 对输入进行复数到实数的 FFT 变换，输入数据沿倒数第2维进行变换
  }

  TORCH_INTERNAL_ASSERT(input.size(2) == n_fft);
  // 确保变换后的输入张量的第2维度大小等于 n_fft

  Tensor y_tmp = input * window_tmp.view({1, 1, n_fft});  // size: (channel, n_frames, n_fft)
  // 将变换后的输入张量与窗口临时张量进行逐元素相乘，得到临时输出张量 y_tmp

  Tensor y = at::unfold_backward(
    y_tmp,
    /*input_sizes=*/{y_tmp.size(0), expected_output_signal_len},
    /*dim=*/1,
    /*size=*/n_fft,
    /*step=*/hop_length);
  // 使用反折叠操作将 y_tmp 张量转换为预期输出信号长度的张量 y

  window_tmp = window_tmp.pow(2).expand({1, n_frames, n_fft});  // size: (1, n_frames, n_fft)
  // 将窗口临时张量每个元素平方，并扩展为指定形状

  Tensor window_envelop = at::unfold_backward(
    window_tmp,
    /*input_sizes=*/{1, expected_output_signal_len},
    /*dim=*/1,
    /*size=*/n_fft,
    /*step=*/hop_length); // size: (1, expected_output_signal_len)
  // 使用反折叠操作将窗口包络张量转换为指定形状的窗口包络

  TORCH_INTERNAL_ASSERT(expected_output_signal_len == y.size(1));
  // 确保预期输出信号长度与 y 张量的第1维度大小相等
  TORCH_INTERNAL_ASSERT(expected_output_signal_len == window_envelop.size(1));
  // 确保预期输出信号长度与窗口包络张量的第1维度大小相等

  // We need to trim the front padding away if centered
  const auto start = center ? n_fft / 2 : 0;
  // 如果居中则从前面修剪填充

  const auto end = [&] () -> int64_t {
    if (lengthOpt.has_value()) {
      return start + *lengthOpt;
      // 如果指定了长度参数，则返回修剪的结束位置
    }
    if (center) {
      return -(n_fft / 2);
      // 如果居中但没有指定长度，则返回修剪的结束位置
    }
    return expected_output_signal_len;
    // 否则返回预期输出信号长度
  }();

  y = y.slice(1, start, end, 1);
  // 根据计算的开始和结束位置在第1维度上对 y 张量进行切片

  window_envelop = window_envelop.slice(1, start, end, 1);
  // 根据计算的开始和结束位置在第1维度上对窗口包络张量进行切片

  const auto window_envelop_lowest = window_envelop.abs().min().lt(1e-11);
  // 计算窗口包络张量的绝对值最小值是否小于 1e-11

  if (at::is_scalar_tensor_true(window_envelop_lowest)) {
    std::ostringstream ss;
    REPR(ss) << "window overlap add min: " << window_envelop_lowest;
    AT_ERROR(ss.str());
    // 如果最小值小于阈值，则抛出错误信息
  }

  y = (y / window_envelop);  // size: (channel, expected_output_signal_len)
  // 将 y 张量按元素除以窗口包络张量，得到最终的输出张量 y

  if (input_dim == 3) {
    y = y.squeeze(0);
    // 如果输入的维度为 3，则压缩第0维度
  }

  // zero padding if the given lengthOpt is longer than expected
  if(end > expected_output_signal_len) {
    TORCH_WARN_ONCE(
      "The length of signal is shorter than the length parameter. Result is being padded with zeros in the tail. "
      "Please check your center and hop_length settings."
    );
    y = at::constant_pad_nd(y, {0, end - expected_output_signal_len}, 0);
    // 如果指定的结束位置大于预期的输出信号长度，则在尾部用零进行填充
  }

  return y;
  // 返回最终的输出张量 y
#undef REPR
}

// 填充具有共轭对称性的 FFT 输入数据
void _fft_fill_with_conjugate_symmetry_(const Tensor& input, IntArrayRef dim_) {
  // 获取输入张量的大小和步长信息
  const auto input_sizes = input.sizes();
  const auto input_strides = input.strides();
  // 检查维度数组不为空
  TORCH_CHECK(!dim_.empty());

  // 将维度从IntArrayRef转换为DimVector，并确保维度合法
  DimVector dim(dim_.begin(), dim_.end());
  at::maybe_wrap_dims(dim, input_strides.size(), /*wrap_scalars=*/false);

  // 如果张量为空或指定维度的尺寸小于等于2，则无需写入元素
  if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
    return;  // 不需要写入任何元素
  }

  // 小尺寸的维度可以视为批处理维度，因为它们不会被镜像
  dim.erase(
      std::remove_if(dim.begin(), dim.end(), [&](int64_t dim) {
        return (input_sizes[dim] <= 2);
      }),
      dim.end());

  // 使用TensorIterator来合并批处理维度
  // 注意：无法使用TensorIterator的循环，因为我们需要负步长
  auto iter = TensorIteratorConfig()
      .add_output(input)
      .add_input(input)
      .resize_outputs(false)
      .declare_static_shape(input_sizes, dim)
      .build();

  // 获取TensorIterator的步长和形状信息
  const auto iter_strides = iter.strides(0);
  const auto iter_sizes = iter.shape();
  const auto ndim = static_cast<int64_t>(iter_strides.size() + dim.size());
  DimVector in_strides(ndim), signal_half_sizes(ndim);

  // 从TensorIterator获取合并的批处理维度
  std::copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
  std::copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

  // 直接从输入获取转换后的维度信息
  const auto element_size = iter.element_size(0);
  for (const auto i : c10::irange(dim.size())) {
    // 转换为字节步长以匹配TensorIterator
    in_strides[iter_strides.size() + i] = input_strides[dim[i]] * element_size;
    signal_half_sizes[iter_strides.size() + i] = input_sizes[dim[i]];
  }

  // 对于最后一个维度，使用负步长以执行镜像操作
  signal_half_sizes.back() = (input_sizes[dim.back()] - 1) / 2;
  auto out_strides = in_strides;
  out_strides.back() *= -1;

  // 获取输入数据的指针并设置输入和输出数据指针
  auto* data_ptr = static_cast<char*>(input.data_ptr());
  const auto* in_data = data_ptr + input_strides[dim.back()] * element_size;
  auto* out_data = data_ptr + (
      input_strides[dim.back()] * (input_sizes[dim.back()] - 1) * element_size);

  // 通过步长重新排序维度以最大化数据局部性
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::sort(dim_permute.begin(), dim_permute.end(),
      [&](auto dim1, auto dim2) {
        return in_strides[dim1] < in_strides[dim2];
      });

  DimVector temp(ndim);
  auto apply_permutation = [&] (DimVector & vec) {
    // 将索引按照重新排序的顺序复制到临时数组，然后再复制回去
    for (const auto i : c10::irange(ndim)) {
      temp[i] = vec[dim_permute[i]];
    }
  // 将 temp 赋值给 vec
  vec = temp;
};

// 对输入、输出和信号半大小应用排列
apply_permutation(in_strides);
apply_permutation(out_strides);
apply_permutation(signal_half_sizes);

// 在新的排列顺序中查找 dims.slice(dims.size() - 1)。
// 这些是需要显式埃尔米特镜像的维度
DimVector mirror_dims;
mirror_dims.reserve(dim.size() - 1);
for (const auto i : c10::irange(ndim)) {
  // 如果 dim_permute[i] >= static_cast<int64_t>(iter_strides.size())，表示不是批处理维度
  // 并且 dim_permute[i] != ndim - 1，表示不是最后一个维度，最后一个维度将单独使用负步幅进行镜像
  if (dim_permute[i] >= static_cast<int64_t>(iter_strides.size()) &&
      dim_permute[i] != ndim - 1) {
    mirror_dims.push_back(i);
  }
}
// 断言镜像维度的数量等于 dim.size() - 1
TORCH_INTERNAL_ASSERT(mirror_dims.size() == dim.size() - 1);

// 调度到 CPU 或 CUDA 内核执行实际的共轭镜像操作
fft_fill_with_conjugate_symmetry_stub(
    input.device().type(), input.scalar_type(),
    mirror_dims, signal_half_sizes, in_strides, in_data, out_strides, out_data);
}

// 定义一个名为 `fft_fill_with_conjugate_symmetry_stub` 的调度函数
DEFINE_DISPATCH(fft_fill_with_conjugate_symmetry_stub);

// 结束 `at::native` 命名空间的定义
} // namespace at::native
```