# `.\pytorch\aten\src\ATen\native\AveragePool3d.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/Parallel.h>
#include <ATen/native/Pool.h>
#include <c10/util/irange.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#endif

namespace at::meta {
using namespace ::at::native;

// 定义 avg_pool3d 的元信息函数
TORCH_META_FUNC(avg_pool3d) (
  const Tensor& input,              // 输入张量
  IntArrayRef kernel_size,          // 池化核大小
  IntArrayRef stride,               // 步长
  IntArrayRef padding,              // 填充
  bool ceil_mode,                   // 是否使用 ceil 模式
  bool count_include_pad,           // 是否包含填充
  std::optional<int64_t> divisor_override // 用于覆盖的除数
) {
  // #20866, #22032: 确保官方 C++ API 中满足该条件
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.size(0);     // 批次大小
  const int64_t nslices = input.size(-4);   // 通道数
  const int64_t itime = input.size(-3);     // 输入时间维度大小
  const int64_t iheight = input.size(-2);   // 输入高度维度大小
  const int64_t iwidth = input.size(-1);    // 输入宽度维度大小

  // 计算输出尺寸
  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  // 检查池化形状是否合法
  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    padT, padH, padW,
    1, 1, 1,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    "avg_pool3d()"
    /*check_input_size=*/ true);



  /* resize output */
  // 如果输入张量的维度为4，设置输出张量的形状为 {nslices, otime, oheight, owidth}
  if (input.ndimension() == 4) {
    set_output_raw_strided(0, {nslices, otime, oheight, owidth}, {}, input.options());
  }
  // 如果输入张量的维度不为4，设置输出张量的形状为 {nbatch, nslices, otime, oheight, owidth}
  else {
    set_output_raw_strided(0, {nbatch, nslices, otime, oheight, owidth}, {}, input.options());
  }
}

TORCH_META_FUNC(avg_pool3d_backward) (
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  // 检查 kernel_size 的维度是否为1或3，确保合法性
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  // 检查 stride 的维度是否为空、1或3，确保合法性
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  // 检查 padding 的维度是否为1或3，确保合法性
  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  // 检查输入张量的维度是否为4D或5D（批处理模式）
  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  // 检查 divisor_override 的值是否为非零（如果有提供）
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  // 获取输入张量在不同维度上的大小
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* XXX shape check behavior from TH */
  // 计算池化操作后的输出形状，用于形状检查
  const int64_t otime_for_shape_check = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight_for_shape_check = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  // 执行池化反向传播的形状检查
  avg_pool3d_backward_shape_check(
    input,
    gradOutput_,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    padT, padH, padW,
    itime, iheight, iwidth,
    otime_for_shape_check, oheight_for_shape_check, owidth_for_shape_check,
    "avg_pool3d_backward()");

  /* resize output */
  // 设置输出张量的大小和选项
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

} // namespace at::meta

namespace at::native {

namespace {

template <typename scalar_t>
// 定义一个静态函数，用于执行3D平均池化操作的输出帧计算
static void avg_pool3d_out_frame(
          // 输入数据的指针，指向一个常量标量类型的数组
          const scalar_t *input_p,
          // 输出数据的指针，指向一个标量类型的数组
          scalar_t *output_p,
          // 输入数据的切片数目
          int64_t nslices,
          // 输入数据的时间维度大小
          int64_t itime,
          // 输入数据的宽度维度大小
          int64_t iwidth,
          // 输入数据的高度维度大小
          int64_t iheight,
          // 输出数据的时间维度大小
          int64_t otime,
          // 输出数据的宽度维度大小
          int64_t owidth,
          // 输出数据的高度维度大小
          int64_t oheight,
          // 池化核的时间维度大小
          int kT,
          // 池化核的宽度维度大小
          int kW,
          // 池化核的高度维度大小
          int kH,
          // 时间维度上的步长
          int dT,
          // 宽度维度上的步长
          int dW,
          // 高度维度上的步长
          int dH,
          // 时间维度上的填充大小
          int padT,
          // 宽度维度上的填充大小
          int padW,
          // 高度维度上的填充大小
          int padH,
          // 是否包括填充值在内进行计算
          bool count_include_pad,
          // 可选参数，用于覆盖默认的除数
          std::optional<int64_t> divisor_override)
{
  // 使用ATen库的并行for循环，对输入数据进行并行处理
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // 遍历范围 [start, end) 内的每个索引 k
    for (const auto k : c10::irange(start, end)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 声明整型变量 i, j, ti
      int64_t i, j, ti;

      /* local pointers. */
      // 设置输入数据的指针 ip，指向输入张量中第 k 个切片的起始位置
      const scalar_t *ip = input_p + k * itime * iwidth * iheight;
      // 设置输出数据的指针 op，指向输出张量中第 k 个切片的起始位置
      scalar_t *op = output_p + k * otime * owidth * oheight;
      // 将输出张量中第 k 个切片的所有元素初始化为 0
      for (i = 0; i < otime * oheight * owidth; ++i)
        *(op + i) = 0;

      /* loop over output */
      // 遍历输出张量的时间维度
      for (ti = 0; ti < otime; ti++)
      {
        // 遍历输出张量的高度维度
        for (i = 0; i < oheight; i++)
        {
          // 遍历输出张量的宽度维度
          for (j = 0; j < owidth; j++)
          {
            /* compute pool range. */
            // 计算池化操作的范围
            int64_t tstart = ti * dT - padT;
            int64_t hstart = i  * dH - padH;
            int64_t wstart = j  * dW - padW;
            int64_t tend = std::min(tstart + kT, itime + padT);
            int64_t hend = std::min(hstart + kH, iheight + padH);
            int64_t wend = std::min(wstart + kW, iwidth + padW);
            int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
            tstart = std::max(tstart, (int64_t) 0);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            tend = std::min(tend, itime);
            hend = std::min(hend, iheight);
            wend = std::min(wend, iwidth);

            // 如果池化范围不合法，直接跳过当前位置的输出
            if (tstart >= tend || hstart >= hend || wstart >= wend) {
              ++op;
              continue;
            }

            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            // 声明整型变量 divide_factor
            int divide_factor;
            // 根据是否存在 divisor_override 的值决定 divide_factor 的值
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              // 根据 count_include_pad 的值确定 divide_factor 的计算方式
              if(count_include_pad) {
                divide_factor = pool_size;
              } else {
                // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
                divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
              }
            }

            /* compute local sum: */
            // 计算局部区域内的数据和 sum
            scalar_t sum = 0.0;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            // 声明整型变量 x, y, z
            int64_t x, y, z;

            // 遍历局部区域内的每个元素，计算它们的和并累加到 sum 中
            for (z = tstart; z < tend; z++)
            {
              for (y = hstart; y < hend; y++)
              {
                for (x = wstart; x < wend; x++)
                {
                  sum +=  *(ip + z * iwidth * iheight + y * iwidth + x);
                }
              }
            }

            /* set output to local max */
            // 将输出设置为局部和的平均值
            *op++ += sum / divide_factor;
          }
        }
      }
    }
  });
} // 结束匿名命名空间

} // 结束匿名命名空间

TORCH_IMPL_FUNC(avg_pool3d_out_cpu) (
  const Tensor& input_,                     // 输入张量，包含要进行池化操作的数据
  IntArrayRef kernel_size,                   // 池化核的大小
  IntArrayRef stride,                        // 池化操作的步长
  IntArrayRef padding,                       // 填充大小
  bool ceil_mode,                            // 是否使用 ceil 模式
  bool count_include_pad,                    // 是否包含填充在内
  std::optional<int64_t> divisor_override,   // 可选的除数覆盖值
  const Tensor& output                       // 输出张量，用于存放池化结果
) {
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);   // 获取池化核的时间维度大小
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);  // 获取池化核的高度维度大小
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);  // 获取池化核的宽度维度大小

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);  // 获取池化操作的时间步长
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);  // 获取池化操作的高度步长
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);  // 获取池化操作的宽度步长

  const int padT = safe_downcast<int, int64_t>(padding[0]);  // 获取时间维度的填充大小
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);  // 获取高度维度的填充大小
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);  // 获取宽度维度的填充大小

  const int64_t nslices = input_.size(-4);  // 输入张量的通道数
  const int64_t itime = input_.size(-3);    // 输入张量的时间维度大小
  const int64_t iheight = input_.size(-2);  // 输入张量的高度维度大小
  const int64_t iwidth = input_.size(-1);   // 输入张量的宽度维度大小

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);     // 计算池化输出的时间维度大小
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode); // 计算池化输出的高度维度大小
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);   // 计算池化输出的宽度维度大小

  /* get contiguous input */
  Tensor input = input_.contiguous();  // 获取连续的输入张量数据

  if (input.ndimension() == 4) /* non-batch mode */  // 非批处理模式下
  {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "avg_pool3d_out_frame",
      [&] {
        const scalar_t *input_data = input.const_data_ptr<scalar_t>();  // 获取输入数据的指针
        scalar_t *output_data = output.data_ptr<scalar_t>();  // 获取输出数据的指针

        avg_pool3d_out_frame(
          input_data, output_data, nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH,
          count_include_pad,
          divisor_override);  // 执行 3D 平均池化操作
    });
  }
  else  /* batch mode */  // 批处理模式下
  {
    const int64_t nbatch = input.size(0);  // 获取批处理大小
    const int64_t istride = nslices * itime * iwidth * iheight;  // 输入张量的步长
    const int64_t ostride = nslices * otime * owidth * oheight;  // 输出张量的步长
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      // 使用宏展开，处理输入张量的浮点类型及长整型
      "avg_pool3d_out_frame",
      // 调用 avg_pool3d_out_frame 函数进行平均池化操作
      [&] {
        // 获取输入张量的指针，指向对应类型的数据
        const scalar_t *input_data = input.const_data_ptr<scalar_t>();
        // 获取输出张量的指针，指向对应类型的数据
        scalar_t *output_data = output.data_ptr<scalar_t>();

        // 并行遍历处理张量的第一维，按指定的起始和结束索引
        at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
          // 循环处理每个批次中的数据片段
          for (const auto p : c10::irange(start, end)) {
            // 调用 avg_pool3d_out_frame 函数对输入数据进行平均池化处理
            avg_pool3d_out_frame(
              input_data + p * istride,  // 输入数据的起始位置
              output_data + p * ostride, // 输出数据的起始位置
              nslices,    // 输入张量的通道数
              itime, iwidth, iheight,   // 输入张量的时间、宽度、高度尺寸
              otime, owidth, oheight,   // 输出张量的时间、宽度、高度尺寸
              kT, kW, kH,   // 池化核的时间、宽度、高度尺寸
              dT, dW, dH,   // 池化操作的时间、宽度、高度步长
              padT, padW, padH,   // 输入张量的时间、宽度、高度填充
              count_include_pad,   // 是否包含填充值在内进行计算
              divisor_override    // 用于覆盖默认的池化因子
            );
          }
        });
    });
  }
// 匿名命名空间，用于定义局部静态变量和函数，限制其作用域不超出当前文件
namespace {

// 定义静态函数 avg_pool3d_backward_out_frame，用于计算 3D 平均池化的反向传播
template <typename scalar_t>
static void avg_pool3d_backward_out_frame(
          scalar_t *gradInput_p,                   // 梯度输入指针
          const scalar_t *gradOutput_p,            // 梯度输出指针（前一层传递过来的梯度）
          int64_t nslices,                        // 输入数据的通道数
          int64_t itime,                          // 输入时间维度大小
          int64_t iwidth,                         // 输入宽度维度大小
          int64_t iheight,                        // 输入高度维度大小
          int64_t otime,                          // 输出时间维度大小
          int64_t owidth,                         // 输出宽度维度大小
          int64_t oheight,                        // 输出高度维度大小
          int kT,                                 // 时间维度的池化核大小
          int kW,                                 // 宽度维度的池化核大小
          int kH,                                 // 高度维度的池化核大小
          int dT,                                 // 时间维度的步长
          int dW,                                 // 宽度维度的步长
          int dH,                                 // 高度维度的步长
          int padT,                               // 时间维度的填充大小
          int padW,                               // 宽度维度的填充大小
          int padH,                               // 高度维度的填充大小
          bool count_include_pad,                 // 是否计算填充区域的池化核大小
          std::optional<int64_t> divisor_override // 可选的覆盖因子，用于池化大小的调整
)
{
  // 使用并行计算来分配工作负载
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // 对每一个通道的数据进行迭代计算
    for (const auto k : c10::irange(start, end)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t i, j, ti;

      /* local pointers */
      // 指向梯度输入和梯度输出的指针
      scalar_t *ip = gradInput_p + k * itime * iwidth * iheight;
      const scalar_t *op = gradOutput_p + k * otime * owidth * oheight;
      
      // 将梯度输入的所有元素初始化为0
      for (i = 0; i < itime * iwidth * iheight; i++)
        *(ip + i) = 0;

      /* loop over output */
      // 遍历输出的每一个元素
      for (ti = 0; ti < otime; ti++)
      {
        for (i = 0; i < oheight; i++)
        {
          for (j = 0; j < owidth; j++)
          {
            // 计算池化区域的起始和结束位置
            int64_t tstart = ti * dT - padT;
            int64_t hstart = i  * dH - padH;
            int64_t wstart = j  * dW - padW;
            int64_t tend = std::min(tstart + kT, itime + padT);
            int64_t hend = std::min(hstart + kH, iheight + padH);
            int64_t wend = std::min(wstart + kW, iwidth + padW);
            
            // 计算池化区域的大小
            int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
            
            // 确保池化区域不超出输入尺寸的边界
            tstart = std::max(tstart, (int64_t) 0);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            tend = std::min(tend, itime);
            hend = std::min(hend, iheight);
            wend = std::min(wend, iwidth);

            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int divide_factor;
            // 根据是否有覆盖因子来确定池化的除数
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if(count_include_pad) {
                divide_factor = pool_size;
              } else {
                // 如果不包含填充区域，重新计算池化区域的大小作为除数
                divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
              }
            }

            /* scatter gradients out to footprint: */
            // 将梯度值按照池化区域分布到输入的各个位置
            scalar_t val = *op++;

            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int64_t x,y,z;
            for (z = tstart; z < tend; z++)
            {
              for (y = hstart; y < hend; y++)
              {
                for (x = wstart; x < wend; x++)
                {
                  *(ip + z * iheight * iwidth + y * iwidth + x) += val / divide_factor;
                }
              }
            }
          }
        }
      }
    }
  });
}

} // namespace
    }
  });



    // 结束了一个函数的定义，可能是一个事件处理函数或者回调函数
    }
    // 结束了一个代码块（可能是一个条件语句、循环语句或函数体），在此之前可能有一些逻辑处理
  });
    // 用于结束一个事件监听器或回调函数的声明，这里的 `});` 可能表示一个闭包或匿名函数的结束
} // 结束匿名命名空间

} // 结束匿名命名空间
// 实现 TORCH_IMPL_FUNC 函数，用于计算 avg_pool3d 操作的反向传播，对应 CPU 实现
TORCH_IMPL_FUNC(avg_pool3d_backward_out_cpu) (
  // 梯度输出张量
  const Tensor& gradOutput_,
  // 输入张量
  const Tensor& input,
  // 池化核大小
  IntArrayRef kernel_size,
  // 步长
  IntArrayRef stride,
  // 填充
  IntArrayRef padding,
  // 是否向上取整模式
  bool ceil_mode,
  // 是否包括填充
  bool count_include_pad,
  // 除数覆盖选项
  std::optional<int64_t> divisor_override,
  // 梯度输入张量
  const Tensor& gradInput
) {
  // 计算池化核的时间维度
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  // 计算池化核的高度维度，如果仅有一维，则与 kT 相同
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  // 计算池化核的宽度维度，如果仅有一维，则与 kT 相同
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  // 计算步长的时间维度，如果未指定则默认与 kT 相同
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  // 计算步长的高度维度，如果未指定则默认与 dT 相同；如果仅有一维则与 dT 相同
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  // 计算步长的宽度维度，如果未指定则默认与 dT 相同；如果仅有一维则与 dT 相同
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  // 计算填充的时间维度
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  // 计算填充的高度维度，如果仅有一维则与 padT 相同
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  // 计算填充的宽度维度，如果仅有一维则与 padT 相同
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  // 输入张量的时间切片数
  const int64_t nslices = input.size(-4);
  // 输入张量的时间维度大小
  const int64_t itime = input.size(-3);
  // 输入张量的高度维度大小
  const int64_t iheight = input.size(-2);
  // 输入张量的宽度维度大小
  const int64_t iwidth = input.size(-1);

  /* 获取连续的梯度输出 */
  // 使梯度输出张量连续
  Tensor gradOutput = gradOutput_.contiguous();

  // 梯度输出张量的时间维度大小
  const int64_t otime = gradOutput.size(-3);
  // 梯度输出张量的高度维度大小
  const int64_t oheight = gradOutput.size(-2);
  // 梯度输出张量的宽度维度大小
  const int64_t owidth = gradOutput.size(-1);

  // 将梯度输入张量置零
  gradInput.zero_();

  /* 反向传播 */
  // 如果输入张量为四维（非批处理模式）
  if (input.ndimension() == 4) /* non-batch mode*/
  {
    // 根据输入张量的数据类型进行派发，并命名为 "avg_pool3d_backward_out_frame"
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "avg_pool3d_backward_out_frame",
      [&] {
       // 获取梯度输入数据的指针
       scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
       // 获取梯度输出数据的常量指针
       const scalar_t *gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

       // 调用 avg_pool3d_backward_out_frame 函数进行帧的梯度反向传播
       avg_pool3d_backward_out_frame(
         gradInput_data, gradOutput_data,
         nslices,
         itime, iwidth, iheight,
         otime, owidth, oheight,
         kT, kW, kH,
         dT, dW, dH,
         padT, padW, padH,
         count_include_pad,
         divisor_override);
    });
  }
  else /* batch mode */
  {
    // 获取批处理数
    const int64_t nbatch = input.size(0);
    // 计算输入张量每个样本的步长
    const int64_t istride = nslices * itime * iwidth * iheight;
    // 计算梯度输出张量每个样本的步长
    const int64_t ostride = nslices * otime * owidth * oheight;
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND 宏来处理输入的浮点类型和长整型，生成对应的代码块
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      // 定义并命名 lambda 函数 "avg_pool3d_backward_out_frame"，用于反向传播平均池化的输出帧
      "avg_pool3d_backward_out_frame",
      // lambda 函数定义开始
      [&] {
        // 获取梯度输入数据的指针
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        // 获取梯度输出数据的常量指针
        const scalar_t *gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

        // 使用并行化方式遍历批次中的每个元素
        at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
          // 对于范围 [start, end) 内的每个索引 p
          for (const auto p : c10::irange(start, end)) {
            // 调用 avg_pool3d_backward_out_frame 函数，对梯度输入数据和梯度输出数据进行操作
            avg_pool3d_backward_out_frame(
              gradInput_data  + p * istride, gradOutput_data + p * ostride, nslices,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              kT, kW, kH,
              dT, dW, dH,
              padT, padW, padH,
              count_include_pad,
              divisor_override
            );
          }
        });
    });
  }
}

// 在命名空间 at::native 中结束命名空间的定义
} // namespace at::native
```