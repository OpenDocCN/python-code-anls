# `.\pytorch\aten\src\ATen\native\AdaptiveMaxPooling3d.cpp`

```py
// 定义宏以启用仅方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量类的头文件
#include <ATen/core/Tensor.h>
// 包含调度相关的头文件
#include <ATen/Dispatch.h>
// 包含并行处理的头文件
#include <ATen/Parallel.h>
// 包含范围迭代工具的头文件
#include <c10/util/irange.h>
// 包含元组的头文件
#include <tuple>

// 包含自适应池化相关的头文件
#include <ATen/native/AdaptivePooling.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择包含的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool3d_backward_native.h>
#include <ATen/ops/adaptive_max_pool3d_native.h>
#endif

// 进入 AT 命名空间中的 meta 子命名空间
namespace at::meta {

// 定义自适应三维最大池化函数的元信息函数
TORCH_META_FUNC(adaptive_max_pool3d) (const Tensor& input, IntArrayRef output_size) {
  // 获取输入张量的维度数
  auto ndim = input.ndimension();
  // 检查输入张量的维度是否为 4 或 5
  TORCH_CHECK(
    ndim == 4 || ndim == 5,
    "adaptive_max_pool3d(): Expected 4D or 5D tensor, but got: ", input.sizes());
  // 检查非批处理维度的大小是否大于 0
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  // 检查输出大小是否为 3
  TORCH_CHECK(
      output_size.size() == 3,
      "adaptive_max_pool3d(): internal error: output_size.size() must be 3");

  // 初始化一些变量
  int dimD = 0;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  // 如果输入张量的维度为 5，则设置 sizeB 并增加 dimD
  if (ndim == 5) {
    sizeB = input.size(0);
    dimD++;
  }

  // 获取维度 dimD 的大小
  sizeD = input.size(dimD);

  // 获取输出大小的各维度值
  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  // 根据输入张量的维度不同，设置输出张量的大小和数据类型
  if (ndim == 4) {
    set_output_raw_strided(0, {sizeD, osizeT, osizeH, osizeW}, {}, input.options());
    // 设置包含每个输出点最大输入位置的索引张量
    set_output_raw_strided(1, {sizeD, osizeT, osizeH, osizeW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {sizeB, sizeD, osizeT, osizeH, osizeW}, {}, input.options());
    // 设置包含每个输出点最大输入位置的索引张量
    set_output_raw_strided(1, {sizeB, sizeD, osizeT, osizeH, osizeW}, {}, input.options().dtype(kLong));
  }
}

// 定义自适应三维最大池化函数的反向传播的元信息函数
TORCH_META_FUNC(adaptive_max_pool3d_backward)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices) {
    // 检查梯度输出张量是否为空
    at::native::adaptive_pool_empty_output_check(gradOutput, "adaptive_max_pool3d_backward");
    // 设置反向传播输出张量的大小和数据类型与输入张量相同
    set_output_raw_strided(0, input.sizes(), {}, input.options());
}
} // namespace meta

// 进入 AT 命名空间中的 native 子命名空间
namespace at::native {

// 匿名命名空间，定义了自适应三维最大池化的单个输出帧的模板函数
namespace {

// 模板函数，用于计算单个输出帧的自适应三维最大池化
template <typename scalar_t>
static void adaptive_max_pool3d_single_out_frame(
          const scalar_t *input_p,
          scalar_t *output_p,
          int64_t *ind_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW,
          int64_t istrideD,
          int64_t istrideT,
          int64_t istrideH,
          int64_t istrideW)
{
  // 使用并行化策略对 sizeD 维度进行遍历
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {


这些注释详细解释了每行代码的功能和目的，确保了代码的每一部分都得到了充分的解释和理解。
    for (const auto d : c10::irange(start, end)) {
      /* 遍历输出 */
      int64_t ot, oh, ow;
      for(ot = 0; ot < osizeT; ot++)
      {
        int64_t istartT = start_index(ot, osizeT, isizeT);
        int64_t iendT   = end_index(ot, osizeT, isizeT);
        int64_t kT = iendT - istartT;

        for(oh = 0; oh < osizeH; oh++)
        {
          int64_t istartH = start_index(oh, osizeH, isizeH);
          int64_t iendH   = end_index(oh, osizeH, isizeH);
          int64_t kH = iendH - istartH;

          for(ow = 0; ow < osizeW; ow++)
          {

            int64_t istartW = start_index(ow, osizeW, isizeW);
            int64_t iendW   = end_index(ow, osizeW, isizeW);
            int64_t kW = iendW - istartW;

            /* 定义局部指针 */
            const scalar_t *ip = input_p   + d*istrideD + istartT *istrideT + istartH*istrideH + istartW*istrideW;
            scalar_t *op = output_p  + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;
            int64_t *indp = ind_p   + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;

            /* 计算局部最大值： */
            int64_t it = 0, ih = 0, iw = 0;
            int64_t maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
            for(it = 0; it < kT; it++)
            {
              for(ih = 0; ih < kH; ih++)
              {
                for(iw = 0; iw < kW; iw++)
                {
                  scalar_t val = *(ip + it*istrideT + ih*istrideH + iw*istrideW);
                  if ((val > maxval) || std::isnan(val))
                  {
                    maxval = val;
                    maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
                  }
                }
              }
            }

            /* 将输出设置为局部最大值 */
            *op = maxval;

            /* 存储最大值的位置 */
            *indp = maxindex;
          }
        }
      }
    }
{
  // 并行处理每个 batch 中的样本
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      // 调用 adaptive_max_pool3d_single_out_frame 函数处理单个样本的自适应最大池化操作
      adaptive_max_pool3d_single_out_frame<scalar_t>(input_data+b*istrideB, output_data+b*sizeD*osizeT*osizeH*osizeW,
                                                     indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                     sizeD,
                                                     isizeT, isizeH, isizeW,
                                                     osizeT, osizeH, osizeW,
                                                     istrideD, istrideT,
                                                     istrideH, istrideW);
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool3d_backward_single_out_frame(
          scalar_t *gradInput_p,
          const scalar_t *gradOutput_p,
          const int64_t *ind_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  // 并行处理每个深度维度上的梯度反向传播
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      // 计算当前深度维度的梯度输入和梯度输出的偏移量
      scalar_t *gradInput_p_d = gradInput_p + d*isizeT*isizeH*isizeW;
      const scalar_t *gradOutput_p_d = gradOutput_p + d*osizeT*osizeH*osizeW;
      const int64_t *ind_p_d = ind_p + d*osizeT*osizeH*osizeW;

      /* 计算最大值点 */
      int64_t ot, oh, ow;
      for(ot = 0; ot < osizeT; ot++)
      {
        for(oh = 0; oh < osizeH; oh++)
        {
          for(ow = 0; ow < osizeW; ow++)
          {
            /* 获取最大值的位置 */
            int64_t maxp = ind_p_d[ot*osizeH*osizeW + oh*osizeW + ow];

            /* 更新梯度 */
            gradInput_p_d[maxp] += gradOutput_p_d[ot*osizeH*osizeW + oh*osizeW + ow];
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool3d_backward_out_frame(
          scalar_t *gradInput_data,
          const scalar_t *gradOutput_data,
          const int64_t *indices_data,
          int64_t sizeB,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  // 并行处理每个 batch 中的样本
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    # 使用范围遍历，遍历从 `start` 到 `end` 之间的整数，每次迭代得到的整数为 `b`
    for (const auto b : c10::irange(start, end)) {
        # 调用自定义函数 `adaptive_max_pool3d_backward_single_out_frame`，传入以下参数：
        # - gradInput_data+b*sizeD*isizeT*isizeH*isizeW：梯度输入数据的起始位置
        # - gradOutput_data+b*sizeD*osizeT*osizeH*osizeW：梯度输出数据的起始位置
        # - indices_data+b*sizeD*osizeT*osizeH*osizeW：索引数据的起始位置
        # - sizeD：输入数据的通道数
        # - isizeT, isizeH, isizeW：输入数据的时间维度、高度和宽度
        # - osizeT, osizeH, osizeW：输出数据的时间维度、高度和宽度
        adaptive_max_pool3d_backward_single_out_frame<scalar_t>(
            gradInput_data+b*sizeD*isizeT*isizeH*isizeW,
            gradOutput_data+b*sizeD*osizeT*osizeH*osizeW,
            indices_data+b*sizeD*osizeT*osizeH*osizeW,
            sizeD,
            isizeT, isizeH, isizeW,
            osizeT, osizeH, osizeW);
    }
} // namespace
    // 增加每个维度的大小标记
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  // 计算输入张量和梯度张量在各个维度上的尺寸
  sizeD = input.size(dimD);
  isizeT = input.size(dimT);
  isizeH = input.size(dimH);
  isizeW = input.size(dimW);
  osizeT = gradOutput_.size(dimT);
  osizeH = gradOutput_.size(dimH);
  osizeW = gradOutput_.size(dimW);

  /* backprop */
  // 如果输入张量是四维的
  if (input.ndimension() == 4) {
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16,
        input.scalar_type(), "adaptive_max_pool3d_backward", [&] {
          /* get raw pointers */
          // 获取梯度输入数据、梯度输出数据和索引数据的原始指针
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices.const_data_ptr<int64_t>();

          // 调用三维自适应最大池化反向传播函数
          adaptive_max_pool3d_backward_single_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16,
        input.scalar_type(), "adaptive_max_pool3d_backward", [&] {
          /* get raw pointers */
          // 获取梯度输入数据、梯度输出数据和索引数据的原始指针
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices.const_data_ptr<int64_t>();

          // 调用多帧三维自适应最大池化反向传播函数
          adaptive_max_pool3d_backward_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              sizeB,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  }
}
// 结束 at::native 命名空间
} // namespace at::native
```