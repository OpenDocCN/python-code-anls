# `.\pytorch\aten\src\ATen\native\cpu\DepthwiseConvKernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/DepthwiseConvKernel.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at::native {
namespace {

struct Arguments final {
  // 输入层维度
  int64_t batch;        // 批处理大小
  int64_t in_rows;      // 输入行数
  int64_t in_cols;      // 输入列数
  int64_t stride;       // 步长
  int64_t pad_rows;     // 填充行数
  int64_t pad_cols;     // 填充列数

  // 输出层维度
  int64_t out_rows;     // 输出行数
  int64_t out_cols;     // 输出列数
  int64_t out_channels; // 输出通道数
};

// 计算卷积输出尺寸的函数
inline std::vector<int64_t> calculate_conv_output_size(
    const IntArrayRef input_size,
    const IntArrayRef weight_size,
    const IntArrayRef stride,
    const IntArrayRef padding) {
  // 内部函数，计算单个维度的输出尺寸
  const auto calc_output_dimension = [](
    const int64_t input, const int64_t kernel, const int64_t stride, const int64_t padding) {
    return 1 + (input - kernel + 2 * padding) / stride;
  };

  // 返回计算得到的输出尺寸向量
  return std::vector<int64_t> {
    input_size[0],
    weight_size[0],
    calc_output_dimension(input_size[2], weight_size[2], stride[0], padding[0]),
    calc_output_dimension(input_size[3], weight_size[3], stride[1], padding[1]),
  };
}

#ifdef __ARM_NEON__

// 使用 NEON 指令集进行的 Winograd F(2x2, 3x3) 输入变换
inline void winograd_f2k3_input_transform_inplace__neon(
    float32x4_t* const d0,
    float32x4_t* const d1,
    float32x4_t* const d2,
    float32x4_t* const d3) {
  const float32x4_t wd0 = *d0 - *d2;
  const float32x4_t wd1 = *d1 + *d2;
  const float32x4_t wd2 = -*d1 + *d2;
  const float32x4_t wd3 = *d1 - *d3;
  *d0 = wd0;
  *d1 = wd1;
  *d2 = wd2;
  *d3 = wd3;
}

// 使用 NEON 指令集进行的 Winograd F(2x2, 3x3) 输出变换
inline void winograd_f2k3_output_transform_inplace__neon(
    float32x4_t* const m0,
    float32x4_t* const m1,
    const float32x4_t* const m2,
    const float32x4_t* const m3) {
  *m0 = *m0 + *m1 + *m2;
  *m1 = *m1 - *m2 - *m3;
}

// 使用 NEON 指令集进行的浮点数向量乘加操作
inline float32x4_t
vmuladdq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b) {
#if defined(__aarch64__)
  return vfmaq_f32(c, a, b);
#else
  return vmlaq_f32(c, a, b);
#endif
}

// 使用 NEON 指令集进行的浮点数向量乘减操作
inline float32x4_t
vmulsubq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b) {
#if defined(__aarch64__)
  return vfmsq_f32(c, a, b);
#else
  return vmlsq_f32(c, a, b);
#endif
}

// 使用 NEON 指令集进行的 Winograd F(2x2, 3x3) 卷积核变换
inline void winograd_f2k3_kernel_transform__neon(
    const float32x4_t g0,
    const float32x4_t g1,
    const float32x4_t g2,
    float32x4_t* const transform0,
    float32x4_t* const transform1,
    float32x4_t* const transform2,
    float32x4_t* const transform3) {
  const float32x4_t const_half = vdupq_n_f32(0.5f);
  float32x4_t half_g0_plus_g2 = const_half * (g0 + g2);
  *transform0 = g0;
  *transform1 = vmuladdq_f32(half_g0_plus_g2, const_half, g1);
  *transform2 = vmulsubq_f32(half_g0_plus_g2, const_half, g1);
  *transform3 = g2;
}

// 使用 NEON 指令集进行的浮点数向量四维矩阵转置
inline float32x4x4_t v4f_transpose4x4__neon(const float32x4x4_t m) {
  float32x4x4_t ret;
  vst4q_f32((float*)(&ret), m);
  return ret;
}

#endif // __ARM_NEON__

} // namespace
} // namespace at::native
  // 定义 vbias，将 bias 的值插入到向量中，其余部分填充为 0
  const float32x4_t vbias = vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
  // 定义 kernel_tile 结构体，用于存储转换后的卷积核
  float32x4x4_t kernel_tile;

  {
    // 加载卷积核的第一行
    const float32x4_t g0 = vld1q_f32(kernel);
    // 加载卷积核的第二行
    const float32x4_t g1 = vld1q_f32(kernel + 3);
    // 加载卷积核的第三行，并忽略其第四个元素
    const float32x4_t g2 =
        vextq_f32(vld1q_f32(kernel + 5), vld1q_f32(kernel + 5), 1);
    float32x4x4_t w;
    // 对卷积核进行 Winograd F(2x2, 3x3) 变换
    winograd_f2k3_kernel_transform__neon(
        g0, g1, g2, &w.val[0], &w.val[1], &w.val[2], &w.val[3]);
    // 将结果转置，得到转换后的卷积核
    w = v4f_transpose4x4__neon(w);

    // 将转换后的卷积核进一步变换，并存入 kernel_tile 中
    winograd_f2k3_kernel_transform__neon(
        w.val[0],
        w.val[1],
        w.val[2],
        &kernel_tile.val[0],
        &kernel_tile.val[1],
        &kernel_tile.val[2],
        &kernel_tile.val[3]);
  }

#define TILE                                                  \
  // 对输入进行 Winograd F(2x2, 3x3) 变换，直接在原地修改 input_tile
  winograd_f2k3_input_transform_inplace__neon(                \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
  // 将 input_tile 转置
  input_tile = v4f_transpose4x4__neon(input_tile);            \
  // 再次对转置后的 input_tile 进行 Winograd F(2x2, 3x3) 变换
  winograd_f2k3_input_transform_inplace__neon(                \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
                                                              \
  // 对每一行进行乘法操作，将 input_tile 与 kernel_tile 相乘
  for (const auto row : c10::irange(4)) {                         \
    input_tile.val[row] =                                     \
        vmulq_f32(input_tile.val[row], kernel_tile.val[row]); \
  }                                                           \
                                                              \
  // 添加偏置项 vbias 到 input_tile 的第二行
  input_tile.val[1] = input_tile.val[1] + vbias;              \
  // 对转置后的 input_tile 进行输出变换
  winograd_f2k3_output_transform_inplace__neon(               \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
  // 再次转置 input_tile，并对其进行输出变换
  input_tile = v4f_transpose4x4__neon(input_tile);            \
  winograd_f2k3_output_transform_inplace__neon(               \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3])

  // 非填充模式。

  // 迭代非填充输出瓦片。
  // TODO: 避免通过区分非填充与填充情况来减少 W 的溢出。
  for (int64_t oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
    for (int64_t otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
        // 对于每个输出列 otw，循环执行以下操作

        // 计算输入块的起始行号和列号
        int64_t ih = oth * 2 - args.pad_rows;
        int64_t iw = otw * 2 - args.pad_cols;

        // 快速路径，所有访问都在边界内
        if (C10_LIKELY(
                ih >= 0 && iw >= 0 && ih + 3 < args.in_rows &&
                    iw + 3 < args.in_cols && 2 * oth + 1 < args.out_rows &&
                    2 * otw + 1 < args.out_cols
                )) {
            // 创建包含四行的输入矩阵块
            float32x4x4_t input_tile;
            for (const auto row : c10::irange(4)) {
                // 使用 NEON 加载四行数据到 input_tile
                input_tile.val[row] =
                    vld1q_f32(input + (ih + row) * args.in_cols + iw);
            }

            // 处理输入矩阵块的操作（未提供具体细节，标记为 TILE）

            // 将处理后的结果写入输出矩阵
            for (const auto row : c10::irange(2)) {
                // 使用 NEON 将 input_tile 中的数据存储到输出数组中
                vst1_f32(
                    output + (oth * 2 + row) * args.out_cols + otw * 2,
                    vget_low_f32(input_tile.val[row]));
            }
        } else {
            // 创建临时的 4x4 块，用于处理边界外的情况
            float block[4][4];
            for (const auto row : c10::irange(4)) {
                for (const auto col : c10::irange(4)) {
                    if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
                        iw + col < args.in_cols) {
                        // 如果在边界内，从输入数组中复制数据到 block 中
                        block[row][col] = input[(ih + row) * args.in_cols + iw + col];
                    } else {
                        // 如果在边界外，将 block 元素设置为 0.0
                        block[row][col] = 0.0;
                    }
                }
            }

            // 使用 NEON 加载 block 数据到 input_tile
            float32x4x4_t input_tile;
            for (const auto row : c10::irange(4)) {
                input_tile.val[row] = vld1q_f32(&block[row][0]);
            }

            // 处理输入矩阵块的操作（未提供具体细节，标记为 TILE）

            // 创建临时的 2x2 块，用于存储处理后的输出数据
            float oblock[2][2];
            for (const auto row : c10::irange(2)) {
                // 使用 NEON 将 input_tile 中的数据存储到 oblock 中
                vst1_f32(&oblock[row][0], vget_low_f32(input_tile.val[row]));
            }

            // 将 oblock 中的处理后的数据写入输出数组
            for (const auto row : c10::irange(2)) {
                for (const auto col : c10::irange(2)) {
                    if (2 * oth + row < args.out_rows &&
                        2 * otw + col < args.out_cols) {
                        // 将 oblock 数据写入到输出数组的正确位置
                        output[(2 * oth + row) * args.out_cols + 2 * otw + col] =
                            oblock[row][col];
                    }
                }
            }
        }
    }
#else

void convolution_depthwise3x3_winograd_impl(
    const Arguments&,                      // 定义名为 Arguments 的结构体作为参数
    const float* const,                    // 常量指针，指向浮点数数组
    const float* const,                    // 常量指针，指向浮点数数组
    const float* const,                    // 常量指针，指向浮点数数组
    float* const) {                        // 指向浮点数数组的常量指针
}

#endif /* __ARM_NEON__ */

Tensor _convolution_depthwise3x3_winograd(
    const Tensor & input,                  // 输入张量引用
    const Tensor & kernel,                 // 卷积核张量引用
    const Tensor & bias_potentially_undefined, // 可能未定义的偏置张量引用
    const IntArrayRef stride,              // 步长的整数数组引用
    const IntArrayRef padding,             // 填充的整数数组引用
    const int64_t groups)                  // 组数的整数
{
  const IntArrayRef input_sizes = input.sizes();       // 输入张量的尺寸数组引用
  const IntArrayRef kernel_sizes = kernel.sizes();     // 卷积核张量的尺寸数组引用

  Tensor output = at::empty(                            // 创建一个空张量作为输出
    calculate_conv_output_size(input_sizes, kernel_sizes, stride, padding),  // 计算卷积输出的尺寸
    input.options());                                   // 使用输入张量的选项（数据类型等）

  const IntArrayRef output_sizes = output.sizes();       // 输出张量的尺寸数组引用

  const Arguments args {                                // 定义名为 args 的 Arguments 结构体
      input_sizes[0],     // Input N                      // 输入批次大小 N
      input_sizes[2],     // Input H                      // 输入高度 H
      input_sizes[3],     // Input W                      // 输入宽度 W
      stride[0],          // Stride                       // 步长
      padding[0],         // Padding Rows                 // 填充行数
      padding[1],         // Padding Columns              // 填充列数
      output_sizes[2],    // Output H                     // 输出高度
      output_sizes[3],    // Output W                     // 输出宽度
      output_sizes[1],    // Output C                     // 输出通道数
  };

  const int64_t input_hxw = args.in_rows * args.in_cols;  // 输入图像的高度乘宽度
  const int64_t output_hxw = args.out_rows * args.out_cols;  // 输出图像的高度乘宽度

  const Tensor bias = bias_potentially_undefined.defined() ?   // 如果偏置张量已定义
                      bias_potentially_undefined :
                      at::zeros({kernel_sizes[0]}, input.options());  // 否则创建一个与卷积核数量相同的零张量

  auto input_data = input.const_data_ptr<float>();         // 输入张量的浮点数常量数据指针
  auto kernel_data = kernel.const_data_ptr<float>();       // 卷积核张量的浮点数常量数据指针
  auto bias_data = bias.const_data_ptr<float>();           // 偏置张量的浮点数常量数据指针
  auto output_data = output.data_ptr<float>();             // 输出张量的浮点数数据指针

  at::parallel_for(0, args.batch * args.out_channels, 0, [&](int64_t start, int64_t end) {  // 并行循环
    for (const auto k : c10::irange(start, end)) {         // 遍历计算范围内的 k
      const int64_t g = k % args.out_channels;             // 计算当前输出通道组
      const int64_t i = k / (args.out_channels / groups);  // 计算当前输入批次索引
      convolution_depthwise3x3_winograd_impl(              // 调用深度可分离3x3 Winograd卷积实现函数
          args,
          input_data + i * input_hxw,                     // 输入数据起始位置
          kernel_data + g * 3 * 3,                        // 卷积核数据起始位置
          bias_data + g,                                  // 偏置数据位置
          output_data + k * output_hxw);                   // 输出数据起始位置
    }
  });

  return output;                                          // 返回输出张量
}

}  // namespace

ALSO_REGISTER_AVX512_DISPATCH(convolution_depthwise3x3_winograd_stub, &_convolution_depthwise3x3_winograd);  // 在 AVX512 下注册深度可分离3x3 Winograd卷积存根

}  // namespace at::native
```