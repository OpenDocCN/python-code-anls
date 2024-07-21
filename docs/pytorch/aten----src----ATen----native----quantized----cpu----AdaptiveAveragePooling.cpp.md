# `.\pytorch\aten\src\ATen\native\quantized\cpu\AdaptiveAveragePooling.cpp`

```
// 定义编译时仅用于 Torch 断言的宏
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

// 根据不同的宏定义条件选择性包含头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/_adaptive_avg_pool3d_native.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <ATen/native/quantized/cpu/QnnpackUtils.h>

// 定义命名空间 at 下的 native 命名空间
namespace at {
namespace native {

// 定义函数指针，用于分发量化自适应平均池化操作
DEFINE_DISPATCH(qadaptive_avg_pool2d_nhwc_stub);
DEFINE_DISPATCH(qadaptive_avg_pool3d_ndhwc_stub);

// 匿名命名空间，包含一些内部使用的辅助函数
namespace {

// 内联函数，计算输入矩阵中每次平均计算的起始索引
inline int start_index(int out_idx, int out_len, int in_len) {
  /*
   * out_idx: 输出矩阵的当前索引
   * out_len: 输出矩阵的维度大小
   * in_len: 输入矩阵的维度大小
   * 基本上，in_len / out_len 给出每个平均计算中的元素数量。
   * 此函数计算输入矩阵上的起始索引。
   */
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return (int)std::floor((float)(out_idx * in_len) / out_len);
}

// 内联函数，计算输入矩阵中每次平均计算的结束索引
inline int end_index(int out_idx, int out_len, int in_len) {
  /*
   * 参数定义与 start_index 相同。
   * 此函数计算输入矩阵上的结束索引。
   */
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return (int)std::ceil((float)((out_idx + 1) * in_len) / out_len);
}

// 自适应平均池化的模板函数，适用于 2D 和 3D 输入
template <typename scalar_t>
static void adaptive_avg_pool_single_out_frame(
    scalar_t* input_p,
    scalar_t* output_p,
    int64_t sizeC,
    int64_t isizeD, // 对于 2D 设置为 1
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeD, // 对于 2D 设置为 1
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideC,
    int64_t istrideD,  // 对于 2D 设置为 1
    int64_t istrideH,
    int64_t istrideW) {
  // 使用 ATen 提供的并行执行工具进行并行计算
  at::parallel_for(0, sizeC, 0, [&](int64_t start, int64_t end) {
    for (const auto c : c10::irange(start, end)) {
      /* loop over output */
      // 循环遍历输出

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t od, oh, ow;
      // 定义输出的深度、高度、宽度

      for (od = 0; od < osizeD; od++) {
        // 循环遍历输出的深度方向
        int istartD = start_index(od, osizeD, isizeD);
        int iendD = end_index(od, osizeD, isizeD);
        int kD = iendD - istartD;
        // 计算当前深度方向上的起始索引、结束索引及长度

        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        float kDr = 1.0 / kD;
        // 计算深度方向上的归一化因子

        for (oh = 0; oh < osizeH; oh++) {
          // 循环遍历输出的高度方向
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;
          // 计算当前高度方向上的起始索引、结束索引及长度

          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          float kDHr = kDr / kH;
          // 计算高度方向上的归一化因子

          for (ow = 0; ow < osizeW; ow++) {
            // 循环遍历输出的宽度方向
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;
            // 计算当前宽度方向上的起始索引、结束索引及长度

            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
            float kDHWr = kDHr / kW;
            // 计算宽度方向上的归一化因子

            /* local pointers */
            // 定义本地指针

            scalar_t* ip = input_p +
                           c * istrideC +
                           istartD * istrideD +
                           istartH * istrideH +
                           istartW * istrideW;
            // 计算输入指针的位置

            scalar_t* op = output_p +
                           c * osizeD * osizeH * osizeW +
                           od * osizeH * osizeW +
                           oh * osizeW +
                           ow;
            // 计算输出指针的位置

            /* compute local average: */
            // 计算本地平均值：

            int64_t sum = 0;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int id, ih, iw;
            // 定义循环变量

            for (id = 0; id < kD; id++) {
              // 循环遍历深度方向上的长度
              for (ih = 0; ih < kH; ih++) {
                // 循环遍历高度方向上的长度
                for (iw = 0; iw < kW; iw++) {
                  // 循环遍历宽度方向上的长度
                  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
                  int64_t val = (ip +
                                 id * istrideD +
                                 ih * istrideH +
                                 iw * istrideW)->val_;
                  // 获取当前位置的值
                  sum += val;
                  // 累加求和
                }
              }
            }

            /* set output to local average */
            // 将输出设置为本地平均值
            // TODO: add the max/min clip
            // 待完成：添加最大/最小值限制

            op->val_ = static_cast<typename scalar_t::underlying>(
                std::nearbyint(sum * kDHWr));
            // 计算并赋值最终的输出值
          } // ow
        } // oh
      } // od
    }
  });
template <int64_t DIM>
std::vector<int64_t> get_output_shape(
    const Tensor& input,
    IntArrayRef output_size) {
  // 遍历输入张量的维度，确保除了批量维度外的所有空间维度都不为空
  for (const auto i : c10::irange(1, input.dim())) {
    // 允许空批量的情况
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pooling", DIM, "d(): ",
        "expected input to have non-empty spatial "
        "dimensions, but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  // 检查输入张量的维度是否符合预期，应为 DIM+1 或 DIM+2
  TORCH_CHECK(
      (input.dim() == DIM + 1 || input.dim() == DIM + 2),
      "non-empty ",
      DIM + 1,
      "D or ",
      DIM + 2,
      "D (batch mode) tensor expected for input");

  /* Channels */
  // 获取通道数
  const int64_t sizeC = input.size(-(DIM+1));

  // 初始化输出形状向量
  std::vector<int64_t> output_shape;
  output_shape.reserve(input.dim());

  // 如果输入张量的维度为 DIM+2，说明包含批量维度
  if (input.dim() == DIM + 2) {
    // 将批量维度大小加入输出形状
    output_shape.push_back(input.size(0));
  }

  // 将通道维度大小加入输出形状
  output_shape.push_back(sizeC);

  // 将用户指定的输出大小加入输出形状
  for (const auto size : output_size) {
    output_shape.push_back(size);
  }

  return output_shape;
}

template <int32_t kSpatialDim, typename scalar_t>
Tensor _adaptive_avg_pool(const Tensor& input,
                          IntArrayRef output_size,
                          Tensor& output) {
  // 调用 get_output_shape 函数获取输出形状
  const auto output_shape = get_output_shape<kSpatialDim>(input, output_size);

  /* sizes */
  // 获取输入张量的通道数、空间维度的大小
  int64_t sizeC = input.size(-(kSpatialDim + 1));
  int64_t isizeD = kSpatialDim == 2 ? 1 : input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);

  // 获取输出张量的空间维度大小
  auto osizeD = kSpatialDim == 2 ? 1 : output_shape[output_shape.size() - 3];
  auto osizeH = output_shape[output_shape.size() - 2];
  auto osizeW = output_shape[output_shape.size() - 1];

  // 获取批量大小，如果输出形状的大小为 kSpatialDim + 1，则批量大小为 1
  int64_t sizeB = output_shape.size() ==(kSpatialDim + 1) ? 1 : output_shape[0];

  // 如果输入张量是按通道最后内存格式或按通道的三维内存格式连续
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
      input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    // 为 NDHWC 格式快速路径创建输出张量
    auto in_stride = input.strides();
    output = at::_empty_affine_quantized(
        output_shape,
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);

    // 调用 qadaptive_avg_pool3d_ndhwc_stub 函数进行池化操作
    qadaptive_avg_pool3d_ndhwc_stub(
        input.device().type(),
        input,
        output,
        sizeB,
        sizeC,
        isizeD,
        isizeH,
        isizeW,
        osizeD,
        osizeH,
        osizeW,
        in_stride[0],
        in_stride[in_stride.size() - (kSpatialDim + 1)],
        in_stride[in_stride.size() - kSpatialDim],
        in_stride[in_stride.size() - 2],
        in_stride[in_stride.size() - 1]);
    return output;
  } else {
    // 否则，创建不按通道内存格式连续的输出张量
    output = at::_empty_affine_quantized(
        output_shape, input.options(), input.q_scale(), input.q_zero_point());

    // 对输入张量进行连续化处理，并获取其数据指针
    auto input_contig = input.contiguous();
    auto input_data = input_contig.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto in_stride = input_contig.strides();
    adaptive_avg_pool_single_out_frame<scalar_t>(
        input_data,
        output_data,
        // 将批次和通道合并成一个维度
        sizeB * sizeC,                       // 合并后的大小
        isizeD,                              // 输入数据的深度维度大小
        isizeH,                              // 输入数据的高度维度大小
        isizeW,                              // 输入数据的宽度维度大小
        osizeD,                              // 输出数据的深度维度大小
        osizeH,                              // 输出数据的高度维度大小
        osizeW,                              // 输出数据的宽度维度大小
        in_stride[in_stride.size() - (kSpatialDim + 1)],  // 输入数据的步长数组的倒数第 kSpatialDim+1 个元素
        in_stride[in_stride.size() - kSpatialDim],          // 输入数据的步长数组的倒数第 kSpatialDim 个元素
        in_stride[in_stride.size() - 2],                     // 输入数据的步长数组的倒数第 2 个元素
        in_stride[in_stride.size() - 1]                      // 输入数据的步长数组的最后一个元素
    );
    return output;
}
Tensor& adaptive_avg_pool3d_out_quantized_cpu(
    const at::Tensor& input,      // 输入张量
    IntArrayRef output_size,      // 输出尺寸数组
    at::Tensor& output) {         // 输出张量的引用

#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    // 如果使用 QNNPACK 引擎

    if (input.scalar_type() == kQUInt8 &&
        enable_qnnpack_for_ada_avgpool(input, output_size)) {
      // 如果输入张量的数据类型为 kQUInt8，并且满足启用 QNNPACK 自适应平均池化的条件
      return qnnpack_adaptive_avg_pool2d(input, output_size);
      // 调用 QNNPACK 自适应平均池化函数并返回结果
    }

  }
#endif

  // 默认情况下，使用普通的自适应平均池化函数
  Tensor output;
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "adaptive_avg_pool2d_quantized_cpu", [&]() {
        output = q_adaptive_avg_pool3d<scalar_t>(output, input, output_size);
      });
  return output;
  // 返回处理后的输出张量的引用
}
    # 输出警告信息，提示“Quantized Adaptive Average Pool 3D”在QNNPACK中尚未实现
    TORCH_WARN("Quantized Adaptive Average Pool 3D is not implemented for ",
               "QNNPACK. Falling back to default implementation.");
    # 结束函数或代码块
    }
#endif
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "adaptive_avg_pool3d_quantized_cpu", [&]() {
        output = q_adaptive_avg_pool3d<scalar_t>(output, input, output_size);
      });
  return output;
}

// 定义了一个函数 adaptive_avg_pool3d_quantized_cpu，用于对输入张量进行三维自适应平均池化的量化CPU实现
Tensor adaptive_avg_pool3d_quantized_cpu(
    const at::Tensor& input,
    IntArrayRef output_size) {
  Tensor output;  // 定义了一个输出张量
  // 调用 ATen 库中的 adaptive_avg_pool3d_out_quantized_cpu 函数，对输入张量进行三维自适应平均池化，结果存储在 output 中
  return at::native::adaptive_avg_pool3d_out_quantized_cpu(input, output_size, output);
}

// 结束 native 命名空间
} // namespace native

// 结束 at 命名空间
} // namespace at
```