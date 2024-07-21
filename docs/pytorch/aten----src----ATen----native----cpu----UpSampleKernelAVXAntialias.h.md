# `.\pytorch\aten\src\ATen\native\cpu\UpSampleKernelAVXAntialias.h`

```
/*
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is

    Copyright © 2010-2022 by Alex Clark and contributors

Like PIL, Pillow is licensed under the open source HPND License
*/

// This code is heavily inspired from PILLOW-SIMD's implementation:
// https://github.com/uploadcare/pillow-simd/blob/simd/master/src/libImaging/Resample.c

#pragma once
#ifdef CPU_CAPABILITY_AVX2
// TODO: This file only supports AVX2. We could split the AVX kernels into
// smaller logical blocks in order to port them into the Vec.h logic. This would
// allow to support other vectorization architectures and perhaps also support
// the non-vectorized fallback (we'd need to make sure it's not slower than the
// current fallback).

#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace {

// Convert a 32-bit integer into a 128-bit SIMD register.
static inline __m128i mm_cvtsi32_si128(const uint8_t* C10_RESTRICT ptr, bool i32_aligned) {
  int32_t v;
  if (i32_aligned) {
    v = *(const int32_t*)ptr;
  } else {
    std::memcpy(&v, ptr, 4);
  }
  return _mm_cvtsi32_si128(v);
}

// Convert unsigned 8-bit integers from memory to 32-bit signed integers in a 128-bit SIMD register.
static inline __m128i mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr, bool i32_aligned) {
  return _mm_cvtepu8_epi32(mm_cvtsi32_si128(ptr, i32_aligned));
}

// Write the lower 32 bits of 'data' into 'output', assuming 'output' is a pointer to uint8_t.
static inline void _write_endline_rgb_as_uint32(
    uint8_t* C10_RESTRICT output,
    uint32_t data
) {
  // data is (R G B X), output is (X1 X2 X3 | R1 B1 G1 R2 ...)
  // Here we explicitly set X as R1
  uint8_t* data_ptr = reinterpret_cast<uint8_t*>(&data);
  data_ptr[3] = output[3];
  std::memcpy(output, data_ptr, 4);
}

// Unpack a packed RGB tensor into an RGBA tensor where A is hard-coded to 0.
at::Tensor unpack_rgb(const at::Tensor& packed_tensor) {
  const uint8_t* packed = (const uint8_t*)packed_tensor.const_data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);

  constexpr int rgba_size = 4;
  auto unpacked_tensor = at::empty({rgba_size, packed_tensor.size(1), packed_tensor.size(2)}, at::CPU(at::kByte));
  uint8_t* unpacked = (uint8_t*) unpacked_tensor.data_ptr<uint8_t>();

  auto stride_i = packed_tensor.stride(2);
  auto stride_j = packed_tensor.stride(0);

  // Iterate through pixels and channels to unpack RGB to RGBA format.
  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(rgba_size)) {
      unpacked[rgba_size * i + j] = (j < num_channels) ? packed[stride_i * i + stride_j * j] : 0;
    }
  }
  return unpacked_tensor;
}

// Pack an RGBA tensor into a packed RGB tensor.
void pack_rgb(
    const at::Tensor& unpacked_tensor, // IN
    const at::Tensor& packed_tensor // OUT
*/
// 将解压的通道最后的 3 通道或 4 通道张量转换为原始数据布局。

uint8_t* unpacked = (uint8_t*)unpacked_tensor.data_ptr<uint8_t>();
// 获取解压张量的数据指针，并将其转换为 uint8_t 类型的指针

uint8_t* packed = (uint8_t*)packed_tensor.data_ptr<uint8_t>();
// 获取打包张量的数据指针，并将其转换为 uint8_t 类型的指针

auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
// 计算打包张量中的像素数量，即 size(1) * size(2)

auto num_channels = packed_tensor.size(0);
// 获取打包张量中的通道数量，即 size(0)

auto unpacked_increment = unpacked_tensor.size(0);
// 获取解压张量的增量（步长），即 size(0)

auto packed_increment = packed_tensor.stride(2);
// 获取打包张量在第二维度上的步长

auto packed_stride = packed_tensor.stride(0);
// 获取打包张量在第零维度上的步长

TORCH_INTERNAL_ASSERT(unpacked_increment == 3 || unpacked_increment == 4);
// 使用 Torch 的内部断言确保解压增量为 3 或 4

for (const auto i C10_UNUSED : c10::irange(num_pixels)) {
    // 对于每一个像素的迭代，使用 c10 库中的范围迭代器
    for (const auto j : c10::irange(num_channels)) {
        // 对于每一个通道的迭代，使用 c10 库中的范围迭代器
        packed[j * packed_stride] = unpacked[j];
        // 将解压数据拷贝到打包数据中，根据通道索引和打包步长进行写入
    }
    unpacked += unpacked_increment;
    // 更新解压指针以便下一个像素
    packed += packed_increment;
    // 更新打包指针以便下一个像素
}
    unsigned int horiz_weights_precision) {
// 定义一个函数，接受多个参数，其中包括水平权重的精度

  // Interpolation horizontal pass: we compute x-axis (image width) interpolation outputs.
  // 水平插值处理：计算 x 轴（图像宽度）的插值输出。

  // Input data is stored as
  //   input = [r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], ...]
  // 输入数据存储格式如下：
  //   input = [r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], ...]

  // Weights are float values computed for each output pixel and rescaled to uint16:
  //   weights[i] = [w[i, 0], w[i, 1], ..., w[i, K-1]]
  // 权重是为每个输出像素计算的浮点值，并重新缩放为 uint16 类型：
  //   weights[i] = [w[i, 0], w[i, 1], ..., w[i, K-1]]

  // We want to compute the output as following:
  //   output = [oR[0], oG[0], oB[0], oA[0], oR[1], oG[1], oB[1], oA[1], ...]
  // 我们希望计算如下的输出：
  //   output = [oR[0], oG[0], oB[0], oA[0], oR[1], oG[1], oB[1], oA[1], ...]

  // where
  //   oR[yoffset + i] = r[yoffset + xmin[i]] * w[i, 0] + ... + r[yoffset + xmin[i] + K-1] * w[i, K-1]
  //   oG[yoffset + i] = g[yoffset + xmin[i]] * w[i, 0] + ... + g[yoffset + xmin[i] + K-1] * w[i, K-1]
  //   oB[yoffset + i] = b[yoffset + xmin[i]] * w[i, 0] + ... + b[yoffset + xmin[i] + K-1] * w[i, K-1]
  // 其中
  //   oR[yoffset + i] = r[yoffset + xmin[i]] * w[i, 0] + ... + r[yoffset + xmin[i] + K-1] * w[i, K-1]
  //   oG[yoffset + i] = g[yoffset + xmin[i]] * w[i, 0] + ... + g[yoffset + xmin[i] + K-1] * w[i, K-1]
  //   oB[yoffset + i] = b[yoffset + xmin[i]] * w[i, 0] + ... + b[yoffset + xmin[i] + K-1] * w[i, K-1]

  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_horizontal<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem
  // 待办事项：我们可能希望将此部分合并到回退代码中（当前称为 basic_loop_aa_horizontal<uint8_t>）
  // 尽管如果我们将所有代码移植到使用 Vec.h，可能就不再需要这样做，因为这可能会为我们提供另一种回退实现

  const int16_t* kk = (int16_t*)(horiz_indices_weights[3].const_data_ptr<double>());

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  auto xin = unpacked_input.size(2);
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_input.size(0));

  const int64_t* idx_ptr_xmin = horiz_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size = horiz_indices_weights[1].const_data_ptr<int64_t>();

  uint8_t* unpacked_output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* unpacked_input_p = unpacked_input.const_data_ptr<uint8_t>();

  int64_t yy = 0;
  auto xout_stride = xout * num_channels;
  auto xin_stride = xin * num_channels;
  for (; yy < yout - 3; yy += 4) {
    // 调用水平插值函数，处理四行像素数据
    ImagingResampleHorizontalConvolution8u4x(
        unpacked_output_p + yy * xout_stride,
        unpacked_output_p + (yy + 1) * xout_stride,
        unpacked_output_p + (yy + 2) * xout_stride,
        unpacked_output_p + (yy + 3) * xout_stride,
        xout,
        unpacked_input_p + yy * xin_stride,
        unpacked_input_p + (yy + 1) * xin_stride,
        unpacked_input_p + (yy + 2) * xin_stride,
        unpacked_input_p + (yy + 3) * xin_stride,
        xin,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels,
        yy + 3 == yout - 1);
  }
  for (; yy < yout; yy++) {
    // 处理剩余不足四行的像素数据，调用对应的水平插值函数
    ImagingResampleHorizontalConvolution8u(
        unpacked_output_p + yy * xout_stride,
        xout,
        unpacked_input_p + yy * xin_stride,
        xin,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels,
        yy == yout - 1);
  }
// 垂直重采样函数，用于图像处理，处理解压输出、输入张量，应用垂直方向的插值操作
void ImagingResampleVertical(
    const at::Tensor & unpacked_output,  // 解压后的输出张量，存储插值结果
    const at::Tensor & unpacked_input,   // 解压后的输入张量，包含原始像素数据
    int ksize,                           // 插值核大小
    const std::vector<at::Tensor>& vert_indices_weights,  // 垂直方向索引和权重的张量数组
    unsigned int vert_weights_precision) {

  // 垂直插值过程：计算沿着Y轴的插值输出
  // 输入数据按以下方式存储：
  //   input = [r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], ...]
  // 权重是为每个输出像素计算的浮点值，并重新缩放为uint16：
  //   weights[i] = [w[i, 0], w[i, 1], ..., w[i, K-1]]
  // 我们希望计算的输出如下所示：
  //   output = [oR[0], oG[0], oB[0], oA[0], oR[1], oG[1], oB[1], oA[1], ...]
  // 其中
  //   oR[xoffset + i] = r[xoffset + ymin[i]] * w[i, 0] + ... + r[xoffset + ymin[i] + (K-1) * xsize] * w[i, K-1]
  //   oG[xoffset + i] = g[xoffset + ymin[i]] * w[i, 0] + ... + g[xoffset + ymin[i] + (K-1) * xsize] * w[i, K-1]
  //   oB[xoffset + i] = b[xoffset + ymin[i]] * w[i, 0] + ... + b[xoffset + ymin[i] + (K-1) * xsize] * w[i, K-1]

  // TODO: 可能需要将此部分合并到回退代码中（当前称为basic_loop_aa_vertical<uint8_t>）
  // 如果/当我们将所有这些代码移植到使用Vec.h时，这可能是不需要的，因为这可能会为我们提供另一个回退实现

  // 从vert_indices_weights中获取kk数组，这是指向int16_t类型的指针
  const int16_t* kk = (int16_t*)(vert_indices_weights[3].const_data_ptr<double>());

  // 从vert_indices_weights中获取idx_ptr_xmin和idx_ptr_size数组，这是指向int64_t类型的指针
  const int64_t* idx_ptr_xmin = vert_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size = vert_indices_weights[1].const_data_ptr<int64_t>();

  // 获取解压后输出张量的指针
  uint8_t* unpacked_output_p = unpacked_output.data_ptr<uint8_t>();
  // 获取解压后输入张量的指针
  const uint8_t* unpacked_input_p = unpacked_input.const_data_ptr<uint8_t>();

  // 获取输出张量的宽度和高度
  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  // 获取输入张量的通道数
  const auto num_channels = unpacked_input.size(0);
  // 内部断言，确保输入输出张量的通道数相同
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_output.size(0));

  // 计算输出张量的横向步长
  auto xout_stride = xout * num_channels;

  // 对于输出张量的每一行，进行垂直插值卷积
  for (const auto yy : c10::irange(yout)) {
    // 获取当前行的插值权重数组
    const auto* k = &kk[yy * ksize];
    // 获取当前行的xmin和size值
    auto ids_min = idx_ptr_xmin[yy];
    auto ids_size = idx_ptr_size[yy];
    // 调用垂直插值卷积函数，将结果写入解压后的输出张量中
    ImagingResampleVerticalConvolution8u(
        unpacked_output_p + yy * xout_stride,
        unpacked_input_p,
        xout,
        ids_min,
        ids_size,
        k,
        vert_weights_precision,
        num_channels);
  }
}

// 这是此文件中唯一的公共入口点。它支持uint8类型的双线性或双三次模式，当通道数小于等于4时，可以选择是否进行抗锯齿处理。
// 该实现基于PIL-SIMD。
// 当AVX不受支持或通道数大于4时，它的等效实现（回退）是separable_upsample_generic_Nd_kernel_impl()。
// 可以进行一些未来的改进：在此文件中查找TODO。
// 有关如何计算权重以及如何在整数上执行乘法的详细信息，请参阅[权重计算和乘法技巧的说明]。
// For details on how the AVX kernels are implemented, see
// https://gist.github.com/NicolasHug/47c97d731f05eaad5694c173849b86f5
// See also [ Support for antialias=False as a subcase of antialias=True ] to
// learn more about how the antialias=False case is computed. The same holds
// here: all these kernels are general enough to handle an arbitrary number of
// weights, but when aa=False they could be optimized further.

// 定义一个模板函数，用于 AVX 加速的双线性和双三次插值处理，输入为 uint8 类型的张量
template <typename scale_type, class F>
void upsample_avx_bilinear_bicubic_uint8(
    const at::Tensor& input_,  // 输入张量，原始图像
    const at::Tensor& output,  // 输出张量，处理后的图像
    bool align_corners,        // 是否对齐图像角点
    const scale_type& scales,  // 缩放因子
    bool antialias) {          // 是否开启抗锯齿

  auto batch_size = input_.size(0);     // 输入批次大小
  auto num_channels = input_.size(1);   // 输入通道数
  auto xin = input_.size(3);            // 输入图像宽度
  auto yin = input_.size(2);            // 输入图像高度
  auto xout = output.size(3);           // 输出图像宽度
  auto yout = output.size(2);           // 输出图像高度

  // 如果输入和输出大小相同，则直接复制输入到输出并返回
  if (xin == xout && yin == yout) {
    output.copy_(input_);
    return;
  }

  at::Tensor input = input_;
  // 如果输入张量不是连续的或者内存格式不是 channels first 或 channels last，则将其转换为 channels last 格式
  if (!(input.is_contiguous() || input.is_contiguous(at::MemoryFormat::ChannelsLast))) {
    // 如果输入既不是 channels first 也不是 channels last，则显式转换为 channels last 格式
    input = input.contiguous(at::MemoryFormat::ChannelsLast);
  }

  auto need_horizontal = xout != xin;  // 是否需要水平插值
  auto need_vertical = yout != yin;    // 是否需要垂直插值

  int ksize_horiz, ksize_vert;  // 水平和垂直方向的插值核尺寸
  std::vector<at::Tensor> horiz_indices_weights, vert_indices_weights;  // 水平和垂直方向的索引权重
  unsigned int horiz_weights_precision, vert_weights_precision;  // 水平和垂直方向权重的精度

  // 是否跳过解包步骤，这在输入通道数为 3 或 4 且输入张量以 channels last 格式连续时成立
  bool skip_unpacking = (num_channels == 3 || num_channels == 4) && input.is_contiguous(at::MemoryFormat::ChannelsLast);
  // 是否跳过打包步骤，这在输出通道数为 3 或 4 且输出张量以 channels last 格式连续时成立
  bool skip_packing = (num_channels == 3 || num_channels == 4) && output.is_contiguous(at::MemoryFormat::ChannelsLast);

  // 如果需要进行水平插值
  if (need_horizontal) {
    int interp_dim = 3;  // 插值的维度为 3（宽度）
    auto stride = (skip_unpacking) ? num_channels : 4;  // 计算步长，根据是否跳过解包确定
    // 调用模板参数 F 中的静态函数 compute_index_ranges_int16_weights 来计算水平插值的索引范围和权重
    std::tie(horiz_indices_weights, ksize_horiz, horiz_weights_precision) =
        F::compute_index_ranges_int16_weights(
            /*input_size=*/xin,
            /*output_size=*/xout,
            /*stride=*/stride,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }

  // 如果需要进行垂直插值
  if (need_vertical) {
    int interp_dim = 2;  // 插值的维度为 2（高度）
    auto stride = (skip_unpacking) ? num_channels * xout : 4 * xout;  // 计算步长，根据是否跳过解包确定
    // 调用模板参数 F 中的静态函数 compute_index_ranges_int16_weights 来计算垂直插值的索引范围和权重
    std::tie(vert_indices_weights, ksize_vert, vert_weights_precision) =
        F::compute_index_ranges_int16_weights(
            /*input_size=*/yin,
            /*output_size=*/yout,
            /*stride=*/stride,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }
}
    // 使用 std::tie 解构函数返回值，将计算出的顶点索引、权重及精度分别赋值给变量
    std::tie(vert_indices_weights, ksize_vert, vert_weights_precision) =
        F::compute_index_ranges_int16_weights(
            /*input_size=*/yin,                   // 输入尺寸 yin
            /*output_size=*/yout,                 // 输出尺寸 yout
            /*stride=*/stride,                    // 步长 stride
            /*ndims=*/4,                          // 维度数 4
            /*reshape_dim=*/interp_dim,           // 重塑维度 interp_dim
            /*align_corners=*/align_corners,      // 是否对齐角落像素 align_corners
            /*opt_scale=*/scales[interp_dim - 2], // 优化比例尺度 scales[interp_dim - 2]
            /*antialias=*/antialias,              // 是否使用抗锯齿 antialias
            /*align_i32=*/true);                  // 是否对齐到32位 align_i32

  }

  // 创建水平和垂直插值时使用的缓冲区
  at::Tensor buffer_horiz, buffer_vert;
  // 小优化：如果只进行水平或垂直插值，且不需要重新打包，可以避免分配额外的缓冲区
  if (need_horizontal && (need_vertical || !skip_packing)) {
    auto c = (skip_unpacking) ? num_channels : 4;
    buffer_horiz = at::empty({c, yin, xout}, input.options()); // 创建指定形状的空张量 buffer_horiz
  }
  if (need_vertical && !skip_packing) {
    auto c = (skip_unpacking) ? num_channels : 4;
    buffer_vert = at::empty({c, yout, xout}, input.options()); // 创建指定形状的空张量 buffer_vert
  }

  // 遍历每个批次的数据
  for (const auto i : c10::irange(batch_size)) {

    // 如果跳过解包，则直接使用输入张量；否则对输入张量进行解包
    at::Tensor unpacked_input = (skip_unpacking) ? input[i] : unpack_rgb(input[i]);
    at::Tensor unpacked_output;

    // 如果需要水平插值
    if (need_horizontal) {
      // 如果同时需要垂直插值或者不跳过打包，则使用水平缓冲区；否则使用输出张量
      at::Tensor unpacked_output_temp = (need_vertical || !skip_packing) ? buffer_horiz : output[i];

      // 如果跳过解包且通道数为3，则使用三通道水平插值函数；否则使用四通道水平插值函数
      if (skip_unpacking && num_channels == 3) {
        ImagingResampleHorizontal<3>(
          unpacked_output_temp,
          unpacked_input,
          ksize_horiz,
          horiz_indices_weights,
          horiz_weights_precision);
      } else {
        ImagingResampleHorizontal<4>(
            unpacked_output_temp,
            unpacked_input,
            ksize_horiz,
            horiz_indices_weights,
            horiz_weights_precision);
      }
      unpacked_output = unpacked_input = unpacked_output_temp; // 更新解包输出为当前使用的缓冲区
    }
    // 如果需要垂直插值
    if (need_vertical) {
      // 如果跳过打包，则直接使用输出张量；否则使用垂直缓冲区
      unpacked_output = (skip_packing) ? output[i] : buffer_vert;

      // 执行垂直插值操作
      ImagingResampleVertical(
          unpacked_output,
          unpacked_input,
          ksize_vert,
          vert_indices_weights,
          vert_weights_precision
      );
    }

    // 断言解包输出已定义
    TORCH_INTERNAL_ASSERT(unpacked_output.defined());

    // 如果不跳过打包，则对解包输出进行重新打包
    if (!skip_packing) {
      pack_rgb(unpacked_output, output[i]);
    }
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,  // 输出行0的指针
    uint8_t* C10_RESTRICT lineOut1,  // 输出行1的指针
    uint8_t* C10_RESTRICT lineOut2,  // 输出行2的指针
    uint8_t* C10_RESTRICT lineOut3,  // 输出行3的指针
    int64_t out_xsize,               // 输出图像的水平尺寸
    const uint8_t* C10_RESTRICT lineIn0,  // 输入行0的指针
    const uint8_t* C10_RESTRICT lineIn1,  // 输入行1的指针
    const uint8_t* C10_RESTRICT lineIn2,  // 输入行2的指针
    const uint8_t* C10_RESTRICT lineIn3,  // 输入行3的指针
    int64_t in_xsize,                // 输入图像的水平尺寸
    const int64_t* idx_ptr_xmin,     // 输入行的最小索引指针
    const int64_t* idx_ptr_size,     // 输入行的大小指针
    const int16_t* kk,               // 滤波器系数数组
    int kmax,                        // 最大滤波器系数索引
    unsigned int coefs_precision,    // 系数精度
    int64_t num_channels,            // 图像通道数
    const auto ids_min = idx_ptr_xmin[out_x];  // 输出图像的最小索引
    const auto ids_size = idx_ptr_size[out_x];  // 输出图像的大小
    const auto * k = &kk[out_x * kmax];  // 当前输出像素位置的滤波器系数指针
    int64_t i = 0;                   // 循环计数器

    auto sss0 = initial;             // 初始值变量sss0
    auto sss1 = initial;             // 初始值变量sss1

    const auto * lineIn0_min = lineIn0 + ids_min;  // 输入行0的起始位置
    const auto * lineIn1_min = lineIn1 + ids_min;  // 输入行1的起始位置
    const auto * lineIn2_min = lineIn2 + ids_min;  // 输入行2的起始位置
    const auto * lineIn3_min = lineIn3 + ids_min;  // 输入行3的起始位置

    // block 4
    for (; i < ids_size - b4_delta; i += 4) {
      // 循环处理每4个元素，直到剩余少于4个元素
      // 加载权重向量中的4个值
      // mmk0 = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      // mmk1 = [wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ...]
      const auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[i]);
      const auto mmk1 = _mm256_set1_epi32(*(int32_t*)&k[i + 2]);

      // RGBA: 从输入行0和行1加载8个像素（每行4个像素）：
      // source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //   R0 G0 B0 A0  R1 G1 B1 A1  R2 G2 B2 A2  R3 G3 B3 A3
      // ]
      // RGB: 从输入行0和行1加载10个像素（每行5个像素）：
      // source = [
      //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
      //   R0 G0 B0 R1  G1 B1 R2 G2  B2 R3 G3 B3  R4 G4 B4 R5
      // ]
      auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn0_min + stride * i))),
          _mm_loadu_si128((__m128i *) (lineIn1_min + stride * i)), 1);

      // 应用低位掩码：
      // RGBA:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      // RGB:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  0 0 0 0]
      auto pix1 = _mm256_shuffle_epi8(source, mask_low);
      // 计算输出值，对每个通道使用32位精度：C += w0 * C0 + w1 * C1
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk0));

      // 应用高位掩码：
      // RGBA:
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  A2 0 A3 0]
      // RGB:
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  0 0 0 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  0 0 0 0]
      auto pix2 = _mm256_shuffle_epi8(source, mask_high);
      // 计算输出值，对每个通道使用32位精度：C += w2 * C2 + w3 * C3
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix2, mmk1));

      // 处理下两行（行2和行3）的相同操作：
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn2_min + stride * i))),
          _mm_loadu_si128((__m128i *) (lineIn3_min + stride * i)), 1);
      auto pix3 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix3, mmk0));
      auto pix4 = _mm256_shuffle_epi8(source2, mask_high);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix4, mmk1));
    }

    // block 2
    // 遍历处理像素数据，每次处理两个像素
    for (; i < ids_size - b2_delta; i += 2) {
      // 从权重向量中加载两个值
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      const auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // 从输入行0和行1中加载4个像素（每行两个像素）：
      // RGBA 模式下的源数据：source1 = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  R1 G1 B1 A1  0 0 0 0  0 0 0 0
      // ]
      // RGB 模式下的源数据：source1 = [
      //   r0 g0 b0 r1  g1 b1 r2  0 0 0 0  0 0 0 0
      //   R0 G0 B0 R1  G1 B1 R2  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i *) (lineIn0_min + stride * i))),
          _mm_loadl_epi64((__m128i *) (lineIn1_min + stride * i)), 1);
      // 应用低位掩码：
      // RGBA 模式下：[r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      // RGB 模式下：[r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  0 0 0 0]
      auto pix1 = _mm256_shuffle_epi8(source1, mask_low);
      // 计算输出值，对每个通道使用32位精度：C += w0 * C0 + w1 * C1
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // 处理行2和行3的像素，与上述过程相同：
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i *) (lineIn2_min + stride * i))),
          _mm_loadl_epi64((__m128i *) (lineIn3_min + stride * i)), 1);
      auto pix2 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // block 1
    const auto i32_aligned = num_channels == 4;
    // 处理单独像素的剩余部分
    for (; i < ids_size - 1; i++) {
      // 从权重向量中加载一个值
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      const auto mmk = _mm256_set1_epi32(k[i]);

      // 从输入行0和行1中加载2个像素（每行一个像素）：
      // RGBA 模式下的像素：pix1 = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  a0 0 0 0
      //   R0 0 0 0  G0 0 0 0  B0 0 0 0  A0 0 0 0
      // ]
      // RGB 模式下的像素：pix1 = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  r1 0 0 0
      //   R0 0 0 0  G0 0 0 0  B0 0 0 0  R1 0 0 0
      // ]
      auto pix1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0_min + stride * i, i32_aligned)),
          mm_cvtepu8_epi32(lineIn1_min + stride * i, i32_aligned), 1);
      // 计算输出值，对每个通道使用32位精度：C += w0 * C0
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // 处理行2和行3的像素，与上述过程相同
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn2_min + stride * i, i32_aligned)),
          mm_cvtepu8_epi32(lineIn3_min + stride * i, i32_aligned), 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }
    if (i == ids_size - 1) {
      // 如果当前处理的元素是最后一个
      auto mmk = _mm256_set1_epi32(k[i]);
      // 对于 num_channels == 3（每像素3字节），允许读取4字节，确保不超出内存边界
      // 第0、1、2行不会超出分配的内存边界
      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0_min + stride * i, i32_aligned)),
          mm_cvtepu8_epi32(lineIn1_min + stride * i, i32_aligned), 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      auto p0 = mm_cvtepu8_epi32(lineIn2_min + stride * i, i32_aligned);
      __m128i p1;
      // 如果 num_channels == 3 并且（是最后一行且超出最大范围），处理边界情况
      if (num_channels == 3 && C10_UNLIKELY(is_last_line && ids_min + stride * i + 4 >= max_in_x_strided)) {
        uint8_t input[4];
        std::memcpy(input, lineIn3_min + stride * i, 3);
        p1 = mm_cvtepu8_epi32(input, true);
      } else {
        p1 = mm_cvtepu8_epi32(lineIn3_min + stride * i, i32_aligned);
      }
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // 将定点值转换回整数（截断）
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    // 使用有符号饱和转换将打包的有符号32位整数转换为打包的16位整数
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d 0 0 0 0 0 0 0 0)
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    // 使用无符号饱和转换将打包的有符号16位整数转换为打包的8位整数
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);

    // 将输出写入单个 uint32
    // (a b c d) -> x_uint32
    auto o0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(sss0));
    auto o1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    auto o2 = _mm_cvtsi128_si32(_mm256_castsi256_si128(sss1));
    auto o3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));

    const auto out_x_strided = stride * out_x;
    # 如果通道数为3且输出的位置接近末尾，执行以下操作：
    if (num_channels == 3 && C10_UNLIKELY(out_x_strided + 4 >= max_out_x_strided)) {
      // 使用 4 字节的 memcpy 操作比使用 3 字节更快，这是一个边界情况，我们要将 4 个字节（R G B | X）
      // 写入输出缓冲区，其中 X 是一个垃圾值，输出缓冲区中的第 4 个字节（R1）包含先前由另一行计算得到的正确值。
      // 换句话说，我们不能简单地通过将寄存器中的 4 个字节写入输出来覆盖它。我们将执行以下操作：
      //               v----------|
      // Output = [... X1 X2 X3 | R1 G1 B1 R2 ...]
      // 首先，将 R1 的值写入到（R G B | X）的第 4 个字节 -> （R G B | R1）
      // 然后，从寄存器向输出写入 4 个字节：（X1 X2 X3 | R1）->（R G B | R1）
      // Output = [... R G B | R1 G1 B1 R2 ...]

      _write_endline_rgb_as_uint32(lineOut0 + out_x_strided, o0);
      _write_endline_rgb_as_uint32(lineOut1 + out_x_strided, o1);
      _write_endline_rgb_as_uint32(lineOut2 + out_x_strided, o2);

      // 如果是最后一行，由于超出内存边界，无法访问下一个 4 字节
      if (C10_UNLIKELY(is_last_line)) {
        // 执行 memcpy 操作将 o3 的 num_channels 个字节复制到 lineOut3 的 out_x_strided 处
        std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, num_channels);
      } else {
        // 否则，将 o3 作为 uint32 写入 lineOut3 的 out_x_strided 处
        _write_endline_rgb_as_uint32(lineOut3 + out_x_strided, o3);
      }
    } else if (num_channels == 3) {
      // 如果通道数为3，执行以下操作：
      // 使用 memcpy 操作将 o0、o1、o2 和 o3 分别作为 4 个字节写入到 lineOut0、lineOut1、lineOut2、lineOut3 的 out_x_strided 处
      std::memcpy(lineOut0 + out_x_strided, (uint8_t *) &o0, 4);
      std::memcpy(lineOut1 + out_x_strided, (uint8_t *) &o1, 4);
      std::memcpy(lineOut2 + out_x_strided, (uint8_t *) &o2, 4);
      std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, 4);
    } else {
      // 否则，当通道数为4时，将 o0、o1、o2 和 o3 分别作为 uint32 写入到 lineOut0、lineOut1、lineOut2、lineOut3 的 out_x_strided 处
      *(uint32_t *)(lineOut0 + out_x_strided) = o0;
      *(uint32_t *)(lineOut1 + out_x_strided) = o1;
      *(uint32_t *)(lineOut2 + out_x_strided) = o2;
      *(uint32_t *)(lineOut3 + out_x_strided) = o3;
    }
}

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,        // 输出线条的指针，用于存储水平重采样后的结果
    int64_t out_xsize,                    // 输出线条的宽度
    const uint8_t* C10_RESTRICT lineIn,   // 输入线条的指针，用于提供源数据
    int64_t in_xsize,                     // 输入线条的宽度
    const int64_t* idx_ptr_xmin,          // 输入线条的最小索引指针
    const int64_t* idx_ptr_size,          // 输入线条的大小索引指针
    const int16_t* kk,                    // 权重系数数组指针
    int kmax,                             // 最大权重系数数目
    unsigned int coefs_precision,         // 权重系数精度
    int64_t num_channels,                 // 输入线条的通道数目
    __m128i sss;                          // 128位整数寄存器，用于存储累加结果
    const auto ids_min = idx_ptr_xmin[out_x];  // 计算输出最小索引值
    const auto ids_size = idx_ptr_size[out_x]; // 计算输出大小索引值
    const auto * k = &kk[out_x * kmax];    // 计算权重系数数组的偏移量
    int64_t i = 0;                         // 迭代器初始化

    const auto * lineIn_min = lineIn + ids_min;  // 计算输入线条的最小位置

    if (ids_size < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));  // 如果输出大小小于8，设置sss为32位整数寄存器，进行初始化
    }

    // block 2
    for (; i < ids_size - b2_delta; i += 2) {
      // 从权重向量加载2个值
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);  // 将权重数组中的值加载到128位整数寄存器mmk中
      // 从输入线条加载像素值
      // RGBA: source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  g1 b1 r2 g2  0 0 0 0  0 0 0 0
      // ]
      auto source = _mm_loadl_epi64((__m128i *) (lineIn_min + stride * i));  // 加载输入线条中的像素数据到128位整数寄存器source中
      // 将source强制转换为epi16类型，并重新排列RGBARGBA -> RRGGBBAA
      auto pix = _mm_shuffle_epi8(source, mask_low128);  // 使用掩码对source进行重新排列，得到像素值pix
      // 对每个通道使用32位精度计算输出值 C += w0 * C0 + w1 * C1
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));  // 使用像素值pix和权重值mmk进行累加计算，存储结果到sss寄存器
    }

    // block 1
    const auto i32_aligned = num_channels == 4;  // 检查是否通道数为4
    for (; i < ids_size - 1; i++) {
      // 从权重向量加载1个值
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      auto mmk = _mm_set1_epi32(k[i]);  // 将权重数组中的值加载到128位整数寄存器mmk中
      // 从输入线条加载一个像素值
      // RGBA: pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  a0 0 0 0
      // ]
      // RGB: pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  r1 0 0 0
      // ]
      auto pix = mm_cvtepu8_epi32(lineIn_min + stride * i, i32_aligned);  // 加载输入线条中的像素数据到128位整数寄存器pix中
      // 对每个通道使用32位精度计算输出值 C += w0 * C0
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));  // 使用像素值pix和权重值mmk进行累加计算，存储结果到sss寄存器
    }

    if (i == ids_size - 1) {
      // 最后一个元素
      auto mmk = _mm_set1_epi32(k[i]);  // 将权重数组中的最后一个值加载到128位整数寄存器mmk中
      __m128i pix;
      auto p = lineIn_min + stride * i;  // 计算最后一个像素在输入线条中的位置
      if (num_channels == 3 && C10_UNLIKELY(is_last_line && ids_min + stride * i + 4 >= max_in_x_strided)) {
        uint8_t input[4];
        std::memcpy(input, p, 3);  // 如果通道数为3且满足特定条件，则将输入的3个字节复制到input数组中
        pix = mm_cvtepu8_epi32(input, true);  // 将input数组的值加载到128位整数寄存器pix中
      } else {
        pix = mm_cvtepu8_epi32(p, i32_aligned);  // 否则，加载输入线条中的像素数据到128位整数寄存器pix中
      }
      // 对每个通道使用32位精度计算输出值 C += w0 * C0
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));  // 使用像素值pix和权重值mmk进行累加计算，存储结果到sss寄存器
    }

    // 将定点值转换回整数（截断）
    sss = _mm_srai_epi32(sss, coefs_precision);  // 对sss寄存器中的值进行算术右移，实现定点数值的转换
    // 使用有符号饱和将打包的32位整数转换为16位整数
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d 0 0 0 0 0 0 0 0)
    sss = _mm_packs_epi32(sss, zero);  // 使用饱和运算将sss寄存器中的32位整数转换为16位整数，并将结果存储回sss寄存器
    // 将打包的有符号16位整数转换为使用无符号饱和度的打包的8位整数
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss = _mm_packus_epi16(sss, zero);
    
    // 将输出写入单个uint32中
    // (a b c d) -> x_uint32
    auto o = _mm_cvtsi128_si32(sss);
    
    // 计算输出行的偏移量
    const auto out_x_strided = stride * out_x;
    
    // 如果通道数为3且超出了最大输出行的限制
    if (num_channels == 3 && C10_UNLIKELY(out_x_strided + 4 >= max_out_x_strided)) {
      // 如果是最后一行，不能访问下一个4字节，因为超出了内存边界。
      // 将3字节从寄存器复制到输出缓冲区
      std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 3);
    } else if (num_channels == 3) {
      // 对于通道数为3的情况，直接复制4字节，因为复制4字节比3字节更快
      // 在这里我们简单地写入4字节 (... R G B X 0 0 0 0 0 ...)，其中X是垃圾值，
      // 我们将在下一次迭代中覆盖它
      std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 4);
    } else {
      // 对于通道数为4的情况，lineOut + out_x_strided 应该对齐到uint32
      // 将o写入lineOut + out_x_strided位置
      *(uint32_t *)(lineOut + out_x_strided) = o;
    }
// 关闭上一个函数的大括号，这里是一个函数定义的结束
}

// 定义一个函数 ImagingResampleVerticalConvolution8u，接受多个参数并执行垂直插值操作
void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,         // 输出行指针，用于存储插值结果
    const uint8_t* C10_RESTRICT lineIn,    // 输入行指针，包含需要插值的数据
    int64_t xsize,                         // 输入/输出行的宽度
    int64_t ids_min,                       // 输入行的起始索引
    int64_t ids_size,                      // 插值大小
    const int16_t* k,                      // 插值权重数组
    unsigned int coefs_precision,           // 权重的精度
    int64_t num_channels) {                 // 通道数

  // 插值的垂直通道处理，处理单行数据
  // - 我们使用块大小为 8、2 和 1 处理 x 轴数据
  // - 将权重向量的大小划分为给定输出索引的总和: K = n * 2 + m.

  // xsize = 输出宽度，也等于输入宽度
  // ids_size = 插值大小
  // ids_min = 输入 y 起始索引
  const auto stride = num_channels * sizeof(uint8_t);

  // 内部断言，确保步长为 3 或 4
  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  const int64_t data_size = xsize * stride;  // 数据大小
  const int64_t data_stride = stride;        // 数据步长
  constexpr auto vec_size = 256 / 8;         // 向量大小为 32 字节，即 256 位

  const auto initial = _mm_set1_epi32(1 << (coefs_precision - 1));        // 初始化 128 位整数向量
  const auto initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1)); // 初始化 256 位整数向量
  const auto zero = _mm_setzero_si128();                                  // 初始化 128 位整数零向量
  const auto zero_256 = _mm256_setzero_si256();                           // 初始化 256 位整数零向量

  int64_t j = 0;  // 初始化数据索引 j

  // 块大小为 8 的循环处理
  const auto b8_usable_vec_stride = (vec_size / data_stride) * data_stride;
  for (; j < data_size - vec_size; j += b8_usable_vec_stride) {
    auto sss0 = initial_256;   // 初始化 256 位整数向量 sss0
    auto sss1 = initial_256;   // 初始化 256 位整数向量 sss1
    auto sss2 = initial_256;   // 初始化 256 位整数向量 sss2
    auto sss3 = initial_256;   // 初始化 256 位整数向量 sss3
    int64_t i = 0;             // 初始化内部索引 i
    const auto * lineIn_min = lineIn + j + ids_min;  // 计算输入行的起始指针
    // 循环处理像素数据，每次处理两个像素
    for (; i < ids_size - 1; i += 2) {
      // 从权重向量中加载两个值，使用 AVX 指令集加载 256 位整数数据
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // RGBA 模式下，每行加载 8 个像素数据
      // source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
      // ]
      // RGB 模式下，每行加载 10 个像素数据，实际处理 8 个像素
      // source1 = [
      //    r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
      //    r4 g4 b4 r5  g5 b5 r6 g6  b6 r7 g7 b7  r8 g8 b8 r9
      // ]
      auto source1 =
          _mm256_loadu_si256((__m256i*)(lineIn_min + data_size * i));
      auto source2 =
          _mm256_loadu_si256((__m256i*)(lineIn_min + data_size * (i + 1)));

      // 将 source1 和 source2 交错排列，并转换为 epi16 类型
      // RGBA 模式下，pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      // RGB 模式下，pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  0 0 0 0
      // ]
      auto source_lo = _mm256_unpacklo_epi8(source1, source2);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      // 使用权重值 mmk 对 pix1 进行加权累加，结果保存在 sss0 中
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // RGBA 模式下，pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  a2 0 A2 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  a3 0 A3 0
      // ]
      // RGB 模式下，pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  0 0 0 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  0 0 0 0
      // ]
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      // 使用权重值 mmk 对 pix2 进行加权累加，结果保存在 sss1 中
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      // 对每个 128 位通道的高半部分进行相同处理
      auto source_hi = _mm256_unpackhi_epi8(source1, source2);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, zero_256);
      // 使用权重值 mmk 对 pix3 进行加权累加，结果保存在 sss2 中
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, zero_256);
      // 使用权重值 mmk 对 pix4 进行加权累加，结果保存在 sss3 中
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // 使用单个权重值进行相同的处理
    // 使用 AVX2 指令集进行向量化计算，处理输入数据中的每个元素
    for (; i < ids_size; i += 1) {
      // 创建包含当前系数的 AVX2 向量
      auto mmk = _mm256_set1_epi32(k[i]);

      // 加载未对齐的 AVX2 向量，表示输入数据中的当前行
      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn_min + i * data_size));

      // 拆分 AVX2 向量的低位部分并扩展为 16 位
      auto source_lo = _mm256_unpacklo_epi8(source1, zero_256);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      // 使用当前系数进行乘法累加操作
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      // 拆分 AVX2 向量的高位部分并扩展为 16 位
      auto source_hi = _mm256_unpackhi_epi8(source1, zero_256);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }

    // 将固定点数值转换回整数值（向下取整）
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);

    // 将打包的有符号 32 位整数转换为打包的 16 位整数，使用有符号饱和转换
    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);

    // 将打包的有符号 16 位整数转换为打包的 8 位整数，使用无符号饱和转换
    sss0 = _mm256_packus_epi16(sss0, sss2);

    // 存储结果到输出行，每次存储 32 字节
    _mm256_storeu_si256((__m256i*)(lineOut + j), sss0);
  }

  // TODO: 是否需要处理块 4 ???
  // 块 2 的处理
  // 计算可用的向量步长，确保对齐要求
  const auto b2_usable_vec_stride = (8 / data_stride) * data_stride;
  for (; j < data_size - vec_size / 4; j += b2_usable_vec_stride) {
    // 初始化累加器
    auto sss0 = initial;
    auto sss1 = initial;
    // 设置循环变量 i 的初始值
    int64_t i = 0;
    // 计算当前处理行的最小值偏移
    const auto * lineIn_min = lineIn + j + ids_min;
    for (; i < ids_size - 1; i += 2) {
      // 从权重向量中加载两个值
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // 加载每行的两个像素
      // RGBA: source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source1 = [
      //    r0 g0 b0 r1  g1 b1 r2 g2  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_loadl_epi64((__m128i *) (lineIn_min + i * data_size));
      auto source2 = _mm_loadl_epi64((__m128i *) (lineIn_min + (i + 1) * data_size));
      // 将source1和source2交错，并将结果转换为epi16类型
      // RGBA: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      // RGB: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // 使用32位精度计算每个通道的输出值作为 C += w0 * c0 + w1 * C0
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      // RGBA: pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      // RGB: pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  0 0 0 0
      // ]
      pix = _mm_unpackhi_epi8(source, zero);
      // 使用32位精度计算每个通道的输出值作为 C += w0 * c1 + w1 * C1
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    // 与上面相同的处理，但是只使用单个权重值
    for (; i < ids_size; i += 1) {
      auto mmk = _mm_set1_epi32(k[i]);

      auto source1 = _mm_loadl_epi64((__m128i*) (lineIn_min + i * data_size));

      auto source = _mm_unpacklo_epi8(source1, zero);
      auto pix1 = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix1, mmk));
      auto pix2 = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix2, mmk));
    }
    // 将定点值转换回整数（截断）
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);
    // 使用有符号饱和转换将打包的有符号32位整数转换为有符号16位整数
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm_packs_epi32(sss0, sss1);
    // 使用无符号饱和转换将打包的有符号16位整数转换为无符号8位整数
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm_packus_epi16(sss0, sss0);
    // 将两个像素存储到输出中
    _mm_storel_epi64((__m128i*)(lineOut + j), sss0);
  }

  // block 1
  const auto b1_usable_vec_stride = (4 / data_stride) * data_stride;
  const auto i32_aligned = num_channels == 4;
  for (; j < data_size - 4; j += b1_usable_vec_stride) {
    auto sss = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;
    // 以步长为2遍历ids数组，加载权重向量中的两个值
    // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
    auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

    // 每行加载一个像素值
    // RGBA情况下，source1 = [r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0]
    // RGB情况下，source1 = [r0 g0 b0 r1  0 0 0 0  0 0 0 0  0 0 0 0]
    auto source1 = mm_cvtsi32_si128(lineIn_min + i * data_size, i32_aligned);
    auto source2 = mm_cvtsi32_si128(lineIn_min + (i + 1) * data_size, i32_aligned);

    // 将source1和source2交错排列，并转换为epi16类型
    // RGBA情况下，pix = [r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0]
    // RGB情况下，pix = [r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0]
    auto source = _mm_unpacklo_epi8(source1, source2);
    auto pix = _mm_unpacklo_epi8(source, zero);

    // 使用权重mmk对每个通道的像素进行加权累加，结果以32位精度存储
    sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
}

// 处理剩余的单个像素值
for (; i < ids_size; i++) {
    auto mmk = _mm_set1_epi32(k[i]);

    // 加载单行像素值
    auto pix = mm_cvtepu8_epi32(lineIn_min + i * data_size, i32_aligned);

    // 使用权重mmk对每个通道的像素进行加权累加，结果以32位精度存储
    sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
}

// 对累加结果进行右移操作，以降低精度
sss = _mm_srai_epi32(sss, coefs_precision);

// 将累加结果从32位打包成16位，超过16位范围的值截断
sss = _mm_packs_epi32(sss, zero);

// 将打包后的16位结果再次打包成8位，超过8位范围的值截断
sss = _mm_packus_epi16(sss, zero);

// 将结果向量sss的最低位转换为普通整数类型o
auto o = _mm_cvtsi128_si32(sss);

// 将o的内容复制到输出数组lineOut的当前位置j处，写入4个字节的数据
// 即使通道数小于4（例如num_channels=3），也会写入第四个字节（例如X），在下一步会被新数据覆盖
// 写入操作不会超出lineOut内存分配的边界
std::memcpy(lineOut + j, (uint8_t *) &o, 4);
    // 使用 SSE 指令集进行像素计算，循环处理每个像素
    for (; i < ids_size; i++) {
      // 使用 _mm_set1_epi32 创建一个包含 k[i] 所有元素的向量
      auto mmk = _mm_set1_epi32(k[i]);

      // 计算指向输入数据起始位置的指针 p
      const uint8_t * p = lineIn_min + i * data_size;
      __m128i pix;
      
      // 根据通道数 num_channels 判断处理方式
      if (num_channels == 3) {
        // 对于 RGB 彩色图像，从 p 复制三个字节到 input 数组
        uint8_t input[4];
        std::memcpy(input, p, 3);
        // 将 input 数组转换成 SSE 寄存器类型 pix
        pix = mm_cvtepu8_epi32(input, true);
      } else {
        // 对于其他情况，直接将 p 指向的数据转换成 SSE 寄存器类型 pix
        pix = mm_cvtepu8_epi32(p, true);
      }
      
      // 使用 SSE 指令计算乘积并累加到 sss 变量中
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // 将固定点数值转换回整数（截断处理）
    sss = _mm_srai_epi32(sss, coefs_precision);
    // 将打包的有符号 32 位整数转换成有符号 16 位整数（使用有符号饱和）
    sss = _mm_packs_epi32(sss, zero);
    // 将打包的有符号 16 位整数转换成无符号 8 位整数（使用无符号饱和）
    sss = _mm_packus_epi16(sss, zero);
    // 将一个像素存储到输出数组 lineOut
    auto o = _mm_cvtsi128_si32(sss);

    // 如果是 RGB 彩色图像并且 j + 4 >= data_size，则只复制三个字节到 lineOut
    if (num_channels == 3 && C10_UNLIKELY(j + 4 >= data_size)) {
      std::memcpy(lineOut + j, (uint8_t *) &o, 3);
    } else {
      // 否则复制四个字节到 lineOut
      std::memcpy(lineOut + j, (uint8_t *) &o, 4);
    }
}

} // anonymous namespace
#endif // CPU_CAPABILITY_AVX2
```