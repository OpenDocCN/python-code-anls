# `.\pytorch\aten\src\ATen\native\mkl\SpectralOps.cpp`

```py
// 定义编译时宏，用于仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入头文件，包括 Tensor 类和配置文件等
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

// 条件编译，根据 AT_PER_OPERATOR_HEADERS 宏选择不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#include <ATen/ops/empty.h>
#endif

// 根据 MKL 或 PocketFFT 是否启用，引入并行处理和迭代器等相关头文件
#if AT_MKL_ENABLED() || AT_POCKETFFT_ENABLED()
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>

// 命名空间开始，native 命名空间内部
namespace at { namespace native {

// 实现一个静态模板函数，用于填充具有共轭对称性的数据切片
template <typename scalar_t>
static __ubsan_ignore_undefined__  // UBSAN 在使用指针的负索引时会给出误报
void _fft_fill_with_conjugate_symmetry_slice(
    Range range, at::ArrayRef<bool> is_mirrored_dim, IntArrayRef signal_half_sizes,
    IntArrayRef in_strides, const scalar_t * in_ptr,
    IntArrayRef out_strides, scalar_t * out_ptr) {
  const auto ndim = signal_half_sizes.size();
  DimVector iter_index(ndim, 0);

  // 显式循环处理一行数据，然后使用 lambda 函数迭代处理 n 维数据
  // 这会使 iter_index 前进一行，同时更新 in_ptr 和 out_ptr 指向新的数据行
  auto advance_index = [&] () __ubsan_ignore_undefined__ {
    for (const auto i : c10::irange(1, iter_index.size())) {
      if (iter_index[i] + 1 < signal_half_sizes[i]) {
        ++iter_index[i];
        in_ptr += in_strides[i];
        if (is_mirrored_dim[i]) {
          if (iter_index[i] == 1) {
            out_ptr += (signal_half_sizes[i] - 1) * out_strides[i];
          } else {
            out_ptr -= out_strides[i];
          }
        } else {
          out_ptr += out_strides[i];
        }
        return;
      }

      in_ptr -= in_strides[i] * iter_index[i];
      if (is_mirrored_dim[i]) {
        out_ptr -= out_strides[i];
      } else {
        out_ptr -= out_strides[i] * iter_index[i];
      }
      iter_index[i] = 0;
    }
  };

  // 处理数据切片，如果 range.begin 大于 0，则更新 iter_index 和指针以引用切片的起始位置
  if (range.begin > 0) {
    iter_index[0] = range.begin % signal_half_sizes[0];
    auto linear_idx = range.begin / signal_half_sizes[0];

    // 其余部分未完整显示，需继续处理完整代码以完善此段注释


这段代码中，我们开始处理 ATen 库中与 FFT 相关的操作和数据填充函数的定义。
    // 遍历除第一个维度外的所有维度，直到线性索引为零或超过数组维度
    for (size_t i = 1; i < ndim && linear_idx > 0; ++i) {
      // 计算当前维度的索引值
      iter_index[i] = linear_idx % signal_half_sizes[i];
      // 更新线性索引以便处理下一个维度
      linear_idx = linear_idx / signal_half_sizes[i];

      // 如果当前维度的索引大于零，执行以下操作
      if (iter_index[i] > 0) {
        // 更新输入指针以指向当前维度的正确位置
        in_ptr += in_strides[i] * iter_index[i];
        // 根据是否镜像维度，更新输出指针以指向正确位置
        if (is_mirrored_dim[i]) {
          out_ptr += out_strides[i] * (signal_half_sizes[i] - iter_index[i]);
        } else {
          out_ptr += out_strides[i] * iter_index[i];
        }
      }
    }
  }

  // 计算剩余处理的元素个数
  auto numel_remaining = range.end - range.begin;

  // 如果第一个维度是镜像的
  if (is_mirrored_dim[0]) {
    // 显式处理 Hermitian 镜像维度的循环
    if (iter_index[0] > 0) {
      // 确定循环的结束位置，以保证不超出数组范围
      auto end = std::min(signal_half_sizes[0], iter_index[0] + numel_remaining);
      // 复制并取共轭将结果写入输出数组
      for (const auto i : c10::irange(iter_index[0], end)) {
        out_ptr[(signal_half_sizes[0] - i) * out_strides[0]] = std::conj(in_ptr[i * in_strides[0]]);
      }
      // 更新剩余处理的元素个数
      numel_remaining -= (end - iter_index[0]);
      // 重置第一个维度的迭代索引并前进到下一个索引位置
      iter_index[0] = 0;
      advance_index();
    }

    // 处理剩余的元素
    while (numel_remaining > 0) {
      // 确定循环的结束位置，以保证不超出数组范围
      auto end = std::min(signal_half_sizes[0], numel_remaining);
      // 复制并取共轭将结果写入输出数组
      out_ptr[0] = std::conj(in_ptr[0]);
      for (const auto i : c10::irange(1, end)) {
        out_ptr[(signal_half_sizes[0] - i) * out_strides[0]] = std::conj(in_ptr[i * in_strides[0]]);
      }
      // 更新剩余处理的元素个数
      numel_remaining -= end;
      // 前进到下一个索引位置
      advance_index();
    }
  } else {
    // 处理非镜像维度的简单共轭复制操作
    while (numel_remaining > 0) {
      // 确定循环的结束位置，以保证不超出数组范围
      auto end = std::min(signal_half_sizes[0], iter_index[0] + numel_remaining);
      // 复制并取共轭将结果写入输出数组
      for (int64_t i = iter_index[0]; i != end; ++i) {
        out_ptr[i * out_strides[0]] = std::conj(in_ptr[i * in_strides[0]]);
      }
      // 更新剩余处理的元素个数
      numel_remaining -= (end - iter_index[0]);
      // 重置第一个维度的迭代索引并前进到下一个索引位置
      iter_index[0] = 0;
      advance_index();
    }
  }
}

static void _fft_fill_with_conjugate_symmetry_cpu_(
    ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef signal_half_sizes,
    IntArrayRef in_strides_bytes, const void * in_data,
    IntArrayRef out_strides_bytes, void * out_data) {

  // Convert strides from bytes to elements
  // 将字节单位的步长转换为元素单位的步长
  const auto element_size = scalarTypeToTypeMeta(dtype).itemsize();
  const auto ndim = signal_half_sizes.size();
  DimVector in_strides(ndim), out_strides(ndim);
  for (const auto i : c10::irange(ndim)) {
    // 确保输入步长是元素大小的整数倍
    TORCH_INTERNAL_ASSERT(in_strides_bytes[i] % element_size == 0);
    in_strides[i] = in_strides_bytes[i] / element_size;
    // 确保输出步长是元素大小的整数倍
    TORCH_INTERNAL_ASSERT(out_strides_bytes[i] % element_size == 0);
    out_strides[i] = out_strides_bytes[i] / element_size;
  }

  // Construct boolean mask for mirrored dims
  // 构建用于镜像维度的布尔掩码
  c10::SmallVector<bool, at::kDimVectorStaticSize> is_mirrored_dim(ndim, false);
  for (const auto& dim : mirror_dims) {
    is_mirrored_dim[dim] = true;
  }

  // Calculate the total number of elements to process
  // 计算需要处理的总元素数
  const auto numel = c10::multiply_integers(signal_half_sizes);
  // Dispatch function for complex types using ATen's parallelization
  // 使用 ATen 的并行机制分发复杂类型的处理函数
  AT_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry", [&] {
    at::parallel_for(0, numel, at::internal::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          // Slice and process a portion of the data
          // 对数据的一部分进行切片和处理
          _fft_fill_with_conjugate_symmetry_slice(
              {begin, end}, is_mirrored_dim, signal_half_sizes,
              in_strides, static_cast<const scalar_t*>(in_data),
              out_strides, static_cast<scalar_t*>(out_data));
        });
  });
}

// Register this one implementation for all cpu types instead of compiling multiple times
// 注册此实现以处理所有 CPU 类型，避免多次编译
REGISTER_ARCH_DISPATCH(fft_fill_with_conjugate_symmetry_stub, DEFAULT, &_fft_fill_with_conjugate_symmetry_cpu_)
REGISTER_AVX2_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cpu_)
REGISTER_AVX512_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cpu_)
REGISTER_ZVECTOR_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cpu_)
REGISTER_VSX_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cpu_)

// _out variants can be shared between PocketFFT and MKL
// _out 变种可以在 PocketFFT 和 MKL 之间共享
Tensor& _fft_r2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool onesided, Tensor& out) {
  // Perform MKL-based real-to-complex FFT and resize the output accordingly
  // 执行基于 MKL 的实到复 FFT 并相应调整输出大小
  auto result = _fft_r2c_mkl(self, dim, normalization, /*onesided=*/true);
  if (onesided) {
    // Resize the output tensor to match the result and copy data
    // 将输出张量大小调整为与结果相同并复制数据
    resize_output(out, result.sizes());
    return out.copy_(result);
  }

  // Resize the output tensor to match the original input sizes
  // 将输出张量大小调整为与原始输入相同
  resize_output(out, self.sizes());

  // Slice the output tensor and copy the relevant part of the result
  // 对输出张量进行切片，并复制结果的相关部分
  auto last_dim = dim.back();
  auto last_dim_halfsize = result.sizes()[last_dim];
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  out_slice.copy_(result);

  // Apply conjugate symmetry filling for FFT
  // 对 FFT 应用共轭对称性填充
  at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  return out;
}
Tensor& _fft_c2r_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         int64_t last_dim_size, Tensor& out) {
  // 调用 _fft_c2r_mkl 函数计算离散傅立叶逆变换，结果存储在 result 中
  auto result = _fft_c2r_mkl(self, dim, normalization, last_dim_size);
  // 调整输出张量 out 的尺寸以匹配 result 的尺寸
  resize_output(out, result.sizes());
  // 将 result 的内容复制到输出张量 out 中，并返回 out
  return out.copy_(result);
}

Tensor& _fft_c2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool forward, Tensor& out) {
  // 调用 _fft_c2c_mkl 函数计算复数到复数的傅立叶变换或逆变换，结果存储在 result 中
  auto result = _fft_c2c_mkl(self, dim, normalization, forward);
  // 调整输出张量 out 的尺寸以匹配 result 的尺寸
  resize_output(out, result.sizes());
  // 将 result 的内容复制到输出张量 out 中，并返回 out
  return out.copy_(result);
}

}} // namespace at::native
#endif /* AT_MKL_ENABLED() || AT_POCKETFFT_ENABLED() */

#if AT_POCKETFFT_ENABLED()
#include <pocketfft_hdronly.h>

namespace at { namespace native {

namespace {
using namespace pocketfft;

// 从张量 t 中获取步长并返回
stride_t stride_from_tensor(const Tensor& t) {
  stride_t stride(t.strides().begin(), t.strides().end());
  // 将每个步长乘以元素大小，以便在傅立叶变换中正确处理内存布局
  for(auto& s: stride) {
   s *= t.element_size();
  }
  return stride;
}

// 从张量 t 中获取形状并返回
inline shape_t shape_from_tensor(const Tensor& t) {
  return shape_t(t.sizes().begin(), t.sizes().end());
}

// 返回张量 t 的复数数据的指针（可变版本）
template<typename T>
inline std::complex<T> *tensor_cdata(Tensor& t) {
  return reinterpret_cast<std::complex<T>*>(t.data_ptr<c10::complex<T>>());
}

// 返回张量 t 的复数数据的指针（常量版本）
template<typename T>
inline const std::complex<T> *tensor_cdata(const Tensor& t) {
  return reinterpret_cast<const std::complex<T>*>(t.const_data_ptr<c10::complex<T>>());
}

// 计算给定类型 T 的归一化系数，根据 size 和 normalization 参数
template<typename T>
T compute_fct(int64_t size, int64_t normalization) {
  constexpr auto one = static_cast<T>(1);
  switch (static_cast<fft_norm_mode>(normalization)) {
    case fft_norm_mode::none: return one;
    case fft_norm_mode::by_n: return one / static_cast<T>(size);
    case fft_norm_mode::by_root_n: return one / std::sqrt(static_cast<T>(size));
  }
  // 如果 normalization 参数不支持，则抛出错误
  AT_ERROR("Unsupported normalization type", normalization);
}

// 在张量 t 的指定维度 dim 上计算归一化系数，根据 normalization 参数
template<typename T>
T compute_fct(const Tensor& t, IntArrayRef dim, int64_t normalization) {
  // 如果 normalization 参数为 none，则返回 1
  if (static_cast<fft_norm_mode>(normalization) == fft_norm_mode::none) {
    return static_cast<T>(1);
  }
  const auto& sizes = t.sizes();
  int64_t n = 1;
  // 计算维度 dim 中所有尺寸的乘积
  for(auto idx: dim) {
    n *= sizes[idx];
  }
  // 根据计算出的尺寸 n 和 normalization 参数计算归一化系数
  return compute_fct<T>(n, normalization);
}

} // anonymous namespace

// 执行离散傅立叶逆变换，并返回结果张量
Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  auto in_sizes = self.sizes();
  // 复制输入张量的尺寸
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  // 将指定维度 dim 中的最后一个维度改为 last_dim_size
  out_sizes[dim.back()] = last_dim_size;
  // 根据修改后的尺寸创建空的输出张量 out，类型匹配输入张量的实部类型
  auto out = at::empty(out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  // 将 dim 转换为 pocketfft 库的形状
  pocketfft::shape_t axes(dim.begin(), dim.end());
  // 如果输入张量的类型是 kComplexFloat，则调用 c2r 函数执行复数到实数的傅立叶逆变换
  if (self.scalar_type() == kComplexFloat) {
    pocketfft::c2r(shape_from_tensor(out), stride_from_tensor(self), stride_from_tensor(out), axes, false,
                   tensor_cdata<float>(self),
                   out.data_ptr<float>(), compute_fct<float>(out, dim, normalization));
  } else {
    // 使用 `pocketfft` 库中的 c2r 函数执行反向的复数到实数的快速傅立叶变换
    // shape_from_tensor(out): 从张量 `out` 获取形状信息
    // stride_from_tensor(self): 从张量 `self` 获取步幅信息
    // stride_from_tensor(out): 从张量 `out` 获取步幅信息
    // axes: 变换的轴列表
    // false: 不进行逆变换
    // tensor_cdata<double>(self): 获取张量 `self` 的数据指针，数据类型为双精度浮点数
    // out.data_ptr<double>(): 获取张量 `out` 的数据指针，数据类型为双精度浮点数
    // compute_fct<double>(out, dim, normalization): 计算函数，用于处理输出的维度和归一化参数
    }
  return out;
// 定义一个名为 `_fft_r2c_mkl` 的函数，接受一些参数并返回一个 Tensor 对象
Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  // 检查输入张量是否为浮点类型
  TORCH_CHECK(self.is_floating_point());
  // 获取输入张量的大小信息
  auto input_sizes = self.sizes();
  // 复制输入大小到输出大小的向量
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  // 获取指定维度的最后一个维度
  auto last_dim = dim.back();
  // 计算最后一个维度的一半大小（实数到复数的变换）
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  // 如果指定了单边频谱（onesided），则修改输出大小
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  // 创建一个空的输出张量，数据类型为输入张量的复数类型
  auto out = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  // 将维度数组转换为 pocketfft 库使用的形状描述
  pocketfft::shape_t axes(dim.begin(), dim.end());
  // 根据输入张量的数据类型选择不同的函数进行实数到复数的 Fourier 变换
  if (self.scalar_type() == kFloat) {
    pocketfft::r2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, true,
                   self.const_data_ptr<float>(),
                   tensor_cdata<float>(out), compute_fct<float>(self, dim, normalization));
  } else {
    pocketfft::r2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, true,
                   self.const_data_ptr<double>(),
                   tensor_cdata<double>(out), compute_fct<double>(self, dim, normalization));
  }

  // 如果没有指定单边频谱，则根据对称性填充输出张量
  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  }
  // 返回计算后的输出张量
  return out;
}

// 定义一个名为 `_fft_c2c_mkl` 的函数，接受一些参数并返回一个 Tensor 对象
Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  // 检查输入张量是否为复数类型
  TORCH_CHECK(self.is_complex());
  // 如果未指定变换维度，则直接复制输入张量并返回
  if (dim.empty()) {
    return self.clone();
  }

  // 创建一个空的输出张量，与输入张量相同的大小和数据类型
  auto out = at::empty(self.sizes(), self.options());
  // 将维度数组转换为 pocketfft 库使用的形状描述
  pocketfft::shape_t axes(dim.begin(), dim.end());
  // 根据输入张量的数据类型选择不同的函数进行复数到复数的 Fourier 变换
  if (self.scalar_type() == kComplexFloat) {
    pocketfft::c2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, forward,
                   tensor_cdata<float>(self),
                   tensor_cdata<float>(out), compute_fct<float>(self, dim, normalization));
  } else {
    pocketfft::c2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, forward,
                   tensor_cdata<double>(self),
                   tensor_cdata<double>(out), compute_fct<double>(self, dim, normalization));
  }

  // 返回计算后的输出张量
  return out;
}
  switch (c10::toRealValueType(dtype)) {
    // 根据 Tensor 的数据类型选择对应的 MKL 的数据类型
    case ScalarType::Float: return DFTI_SINGLE;
    case ScalarType::Double: return DFTI_DOUBLE;
    // 如果类型不支持，则抛出错误信息
    default: TORCH_CHECK(false, "MKL FFT doesn't support tensors of type: ", dtype);
  }
}();
// 信号类型（实数或复数）的配置
const DFTI_CONFIG_VALUE signal_type = [&]{
  if (forward) {
    // 如果是正向变换，根据输入是否为复数确定信号类型
    return complex_input ? DFTI_COMPLEX : DFTI_REAL;
  } else {
    // 如果是反向变换，根据输出是否为复数确定信号类型
    return complex_output ? DFTI_COMPLEX : DFTI_REAL;
  }
}();
// 使用信号尺寸创建描述符
using MklDimVector = c10::SmallVector<MKL_LONG, at::kDimVectorStaticSize>;
MklDimVector mkl_signal_sizes(sizes.begin() + 1, sizes.end());
DftiDescriptor descriptor;
descriptor.init(prec, signal_type, signal_ndim, mkl_signal_sizes.data());
// 非原地 FFT 变换
MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE));
// 批处理模式
MKL_LONG mkl_batch_size = sizes[0];
MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, mkl_batch_size));

// 批处理维度步长，即每个数据之间的距离
TORCH_CHECK(in_strides[0] <= MKL_LONG_MAX && out_strides[0] <= MKL_LONG_MAX);
MKL_LONG idist = in_strides[0];
MKL_LONG odist = out_strides[0];
MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_INPUT_DISTANCE, idist));
MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_OUTPUT_DISTANCE, odist));

// 信号步长
// 第一个值为偏移量，设置为零（被忽略）
MklDimVector mkl_istrides(1 + signal_ndim, 0), mkl_ostrides(1 + signal_ndim, 0);
for (int64_t i = 1; i <= signal_ndim; i++) {
  TORCH_CHECK(in_strides[i] <= MKL_LONG_MAX && out_strides[i] <= MKL_LONG_MAX);
  mkl_istrides[i] = in_strides[i];
  mkl_ostrides[i] = out_strides[i];
}
MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_INPUT_STRIDES, mkl_istrides.data()));
MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_OUTPUT_STRIDES, mkl_ostrides.data()));
// 如果涉及实数的共轭域，设置标准的 CCE 存储类型
// 这将在未来成为 MKL 的默认设置
if (!complex_input || !complex_output) {
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
}
// 如果请求了重新缩放
const auto norm = static_cast<fft_norm_mode>(normalization);
int64_t signal_numel = c10::multiply_integers(IntArrayRef(sizes.data() + 1, signal_ndim));
if (norm != fft_norm_mode::none) {
  // 根据归一化模式选择缩放系数
  const double scale = (
    (norm == fft_norm_mode::by_root_n) ?
    1.0 / std::sqrt(static_cast<double>(signal_numel)) :
    1.0 / static_cast<double>(signal_numel));
  const auto scale_direction = forward ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE;
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), scale_direction, scale));
}

if (sizeof(MKL_LONG) < sizeof(int64_t)) {
    # 检查输入信号的元素数量是否在允许的范围内，即不超过 MKL_LONG_MAX
    TORCH_CHECK(signal_numel <= MKL_LONG_MAX,
                "MKL FFT: input signal numel exceeds allowed range [1, ", MKL_LONG_MAX, "]");
  }

  // 完成描述符的设置和配置
  MKL_DFTI_CHECK(DftiCommitDescriptor(descriptor.get()));

  // 返回已配置好的描述符对象
  return descriptor;
}

// 执行通用的 FFT 操作（可以是复数到复数、实数到单边复数或单边复数到实数）
static Tensor& _exec_fft(Tensor& out, const Tensor& self, IntArrayRef out_sizes,
                         IntArrayRef dim, int64_t normalization, bool forward) {
  // 获取输入张量的维度数
  const auto ndim = self.dim();
  // 信号维度的数量
  const int64_t signal_ndim = dim.size();
  // 批次维度的数量
  const auto batch_dims = ndim - signal_ndim;

  // 排列维度使得批次维度首先出现，并且按照步幅顺序
  // 这样做可以在折叠到单个批次维度时最大化数据局部性
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  // 标记需要变换的维度
  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  // 将批次维度和非批次维度分开，以便进行排序
  auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
                                  [&](int64_t d) { return !is_transformed_dim[d]; });
  // 根据输入张量的步幅排序维度
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end,
            [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  // 将未变换的维度添加到排列后的维度中
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  // 按照新的排列顺序重新排列输入张量
  auto input = self.permute(dim_permute);

  // 将批次维度折叠成单个维度
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1; // 第一个维度是未知的批次维度
  std::copy(input.sizes().cbegin() + batch_dims, input.sizes().cend(), batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  // 获取批次大小
  const auto batch_size = input.sizes()[0];
  // 信号大小的维度向量，包括批次大小
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (const auto i : c10::irange(signal_ndim)) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    // 确保输入和输出的大小符合预期的FFT要求
    TORCH_INTERNAL_ASSERT(in_size == signal_size[i + 1] ||
                          in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(out_size == signal_size[i + 1] ||
                          out_size == (signal_size[i + 1] / 2) + 1);
    // 更新信号大小向量
    signal_size[i + 1] = std::max(in_size, out_size);
  }

  // 重置批次大小
  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (const auto i : c10::irange(dim.size())) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }

  // 调整输出张量的大小和存储顺序
  const auto value_type = c10::toRealValueType(input.scalar_type());
  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // 获取MKL FFT计划描述符
  auto descriptor = _plan_mkl_fft(
      input.strides(), out.strides(), signal_size, input.is_complex(),
      out.is_complex(), normalization, forward, value_type);

  // 执行FFT运算
  if (forward) {
    MKL_DFTI_CHECK(DftiComputeForward(descriptor.get(), const_cast<void*>(input.const_data_ptr()), out.data_ptr()));
  } else {
    MKL_DFTI_CHECK(DftiComputeBackward(descriptor.get(), const_cast<void*>(input.const_data_ptr()), out.data_ptr()));
  }

  // 就地重塑为原始批次形状并反转维度排列
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    // 根据维度置换数组 `dim_permute` 更新输出数组的步长 `out_strides`
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];

    // 更新 `batch_numel` 以反映当前维度的大小
    batch_numel *= out_sizes[dim_permute[i]];
  }

  // 对于剩余的维度，更新输出数组的步长 `out_strides`
  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }

  // 使用给定的大小和步长重新调整输出数组 `out`
  out.as_strided_(out_sizes, out_strides, out.storage_offset());

  // 返回调整后的输出数组 `out`
  return out;
// 结束 at::native 命名空间的定义

// Sort transform dimensions by input layout, for best performance
// exclude_last is for onesided transforms where the last dimension cannot be reordered
// 根据输入布局对转换维度进行排序，以获得最佳性能
// exclude_last 参数用于单边转换，其中最后一个维度无法重新排序
static DimVector _sort_dims(const Tensor& self, IntArrayRef dim, bool exclude_last=false) {
  // 将传入的维度 dim 复制到 sorted_dims 中
  DimVector sorted_dims(dim.begin(), dim.end());
  // 获取输入张量的步长信息
  auto self_strides = self.strides();
  // 对 sorted_dims 进行排序，排序规则是根据 self_strides 中的值进行比较
  std::sort(sorted_dims.begin(), sorted_dims.end() - exclude_last,
            [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  // 返回排序后的维度列表
  return sorted_dims;
}

// n-dimensional complex to real IFFT
// 将 n 维复数转换为实数的逆FFT
Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  // 检查输入张量是否为复数类型
  TORCH_CHECK(self.is_complex());
  // 复杂情况下，将输入赋值给 input 变量
  auto input = self;
  // 如果 dim 的大小大于 1
  if (dim.size() > 1) {
    // 获取除最后一个维度外的所有维度
    auto c2c_dims = dim.slice(0, dim.size() - 1);
    // 使用 _fft_c2c_mkl 函数进行多维度 C2C 转换
    input = _fft_c2c_mkl(self, c2c_dims, normalization, /*forward=*/false);
    // 将 dim 更新为最后一个维度
    dim = dim.slice(dim.size() - 1);
  }

  // 获取输入张量的大小信息
  auto in_sizes = input.sizes();
  // 复制输入张量的大小信息到 out_sizes
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  // 将 out_sizes 中 dim 的最后一个维度更新为 last_dim_size
  out_sizes[dim.back()] = last_dim_size;
  // 创建一个与 out_sizes 大小相同且数据类型为实数的空张量 out
  auto out = at::empty(out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  // 执行 FFT 操作，将结果存储到 out 中
  return _exec_fft(out, input, out_sizes, dim, normalization, /*forward=*/false);
}

// n-dimensional real to complex FFT
// 将 n 维实数转换为复数的FFT
Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  // 检查输入张量是否为浮点类型
  TORCH_CHECK(self.is_floating_point());
  // 获取输入张量的大小信息
  auto input_sizes = self.sizes();
  // 复制输入张量的大小信息到 out_sizes
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  // 获取 dim 的最后一个维度
  auto last_dim = dim.back();
  // 计算最后一个维度的一半大小
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  // 如果 onesided 为 true，则更新 out_sizes 中最后一个维度的大小为 last_dim_halfsize
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  // 使用 _sort_dims 函数对维度 dim 进行排序，exclude_last 参数为 true
  auto sorted_dims = _sort_dims(self, dim, /*exclude_last=*/true);
  // 创建一个与 out_sizes 大小相同且数据类型为复数的空张量 out
  auto out = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  // 执行 FFT 操作，将结果存储到 out 中
  _exec_fft(out, self, out_sizes, sorted_dims, normalization, /*forward=*/true);

  // 如果 onesided 为 false，则使用 _fft_fill_with_conjugate_symmetry_ 函数进行对称填充
  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  }
  // 返回结果张量 out
  return out;
}

// n-dimensional complex to complex FFT/IFFT
// 多维度复数到复数的FFT/IFFT
Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  // 检查输入张量是否为复数类型
  TORCH_CHECK(self.is_complex());
  // 如果 dim 为空，则直接返回输入张量的克隆
  if (dim.empty()) {
    return self.clone();
  }

  // 使用 _sort_dims 函数对维度 dim 进行排序
  const auto sorted_dims = _sort_dims(self, dim);
  // 创建一个与输入张量大小相同且数据类型相同的空张量 out
  auto out = at::empty(self.sizes(), self.options());
  // 执行 FFT 操作，将结果存储到 out 中
  return _exec_fft(out, self, self.sizes(), sorted_dims, normalization, forward);
}

}} // namespace at::native
#endif

// 结束 at::native 命名空间的条件编译
# 抛出错误，指示 ATen 没有编译支持 FFT
Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

# 抛出错误，指示 ATen 没有编译支持 FFT
Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

# 抛出错误，指示 ATen 没有编译支持 FFT
Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

# 抛出错误，指示 ATen 没有编译支持 FFT
Tensor& _fft_r2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool onesided, Tensor& out) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

# 抛出错误，指示 ATen 没有编译支持 FFT
Tensor& _fft_c2r_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         int64_t last_dim_size, Tensor& out) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

# 抛出错误，指示 ATen 没有编译支持 FFT
Tensor& _fft_c2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool forward, Tensor& out) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}
```