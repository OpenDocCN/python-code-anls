# `.\pytorch\aten\src\ATen\native\FractionalMaxPool2d.cpp`

```
// 定义宏，指定仅支持方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含张量操作的核心头文件
#include <ATen/core/Tensor.h>
// 包含分发机制
#include <ATen/Dispatch.h>
// 包含并行处理支持
#include <ATen/Parallel.h>
// 包含张量元数据
#include <ATen/TensorMeta.h>
// 包含分数最大池化相关函数的头文件
#include <ATen/native/FractionalMaxPooling.h>
// 包含循环范围计算的工具
#include <c10/util/irange.h>

// 根据条件选择是否包含通用函数和本地函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/fractional_max_pool2d_backward_native.h>
#include <ATen/ops/fractional_max_pool2d_native.h>
#endif

// 包含元组和向量的标准库头文件
#include <tuple>
#include <vector>

// 定义ATen命名空间
namespace at {

// 定义meta命名空间，用于元编程函数定义
namespace meta {

// 定义fractional_max_pool2d元函数，接收输入张量、池化尺寸、输出尺寸和随机采样张量作为参数
TORCH_META_FUNC(fractional_max_pool2d) (
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples
) {
  // 检查池化尺寸参数是否为二维
  TORCH_CHECK(
      pool_size.size() == 2,
      "fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints")
  // 检查输出尺寸参数是否为二维
  TORCH_CHECK(
      output_size.size() == 2,
      "fractional_max_pool2d: output_size must either be a single Int or tuple of Ints")

  // 初始化变量，设定默认批次数、平面维度、高度维度、宽度维度、输出高度和输出宽度
  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t heightDim = 1;
  int64_t widthDim = 2;
  int64_t outputH = output_size[0];
  int64_t outputW = output_size[1];
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  // 获取输入张量的维度数
  int64_t ndims = input.ndimension();
  // 检查输入张量是否为3D或4D张量
  TORCH_CHECK(ndims == 3 || ndims == 4,
              "fractional_max_pool2d(): Expected 3D or 4D tensor, but got: ", input.sizes());
  // 对于每个非批次维度，检查其大小是否大于0
  for (const auto i : c10::irange(1, ndims)) {
    TORCH_CHECK(input.size(i) > 0,
                "fractional_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but got",
                input.sizes(), " with dimension ", i, " being empty.");
  }

  // 如果输入张量为4D，更新批次数和维度索引
  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  // 获取平面数、输入高度和宽度
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(heightDim);
  int inputW = input.size(widthDim);

  // 检查池化和输出尺寸是否符合输入尺寸
  TORCH_CHECK(outputH + poolSizeH - 1 <= inputH,
    "fractional_max_pool2d(): pool height ", poolSizeH,
    " too large relative to input height ", inputH);
  TORCH_CHECK(outputW + poolSizeW - 1 <= inputW,
    "fractional_max_pool2d(): pool width ", poolSizeW,
    " too large relative to input width ", inputW);

  // 根据输入张量维度数选择设置输出张量结构
  if (ndims == 3) {
    set_output_raw_strided(0, {numPlanes, outputH, outputW}, {}, input.options());
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {numPlanes, outputH, outputW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {numBatch, numPlanes, outputH, outputW}, {}, input.options());
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {numBatch, numPlanes, outputH, outputW}, {}, input.options().dtype(kLong));
  }
}

} // namespace meta
} // namespace at
// 定义 TORCH_META_FUNC 宏函数，用于定义 fractional_max_pool2d_backward 函数
TORCH_META_FUNC(fractional_max_pool2d_backward)(
  const at::Tensor& gradOutput_,                       // 输入参数，梯度输出张量
  const at::Tensor& input,                             // 输入参数，输入张量
  IntArrayRef pool_size /* unused */,                   // 输入参数，池大小（未使用）
  IntArrayRef output_size,                              // 输入参数，输出大小
  const at::Tensor& indices) {                          // 输入参数，索引张量

  int numBatch = 1;                                     // 初始化批次数为1
  int planeDim = 0;                                     // 平面维度索引为0
  int heightDim = 1;                                    // 高度维度索引为1
  int widthDim = 2;                                     // 宽度维度索引为2

  int outputH = output_size[0];                          // 输出高度
  int outputW = output_size[1];                          // 输出宽度

  int ndims = input.ndimension();                        // 输入张量的维度数
  if (ndims == 4) {                                      // 如果输入张量是4维的
    numBatch = input.size(0);                            // 更新批次数为输入张量的第0维大小
    planeDim = 1;                                        // 更新平面维度索引为1
    heightDim++;                                         // 更新高度维度索引
    widthDim++;                                          // 更新宽度维度索引
  }

  /* sizes */
  int numPlanes = input.size(planeDim);                  // 获取平面数
  int inputH = input.size(heightDim);                    // 获取输入高度
  int inputW = input.size(widthDim);                     // 获取输入宽度

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();            // 获取连续的梯度输出张量

  // 检查梯度输出的宽度和高度是否与预期相符
  TORCH_CHECK(outputW == gradOutput.size(widthDim),
    "fractional_max_pool2d_backward(): gradOutput width unexpected");
  TORCH_CHECK(outputH == gradOutput.size(heightDim),
    "fractional_max_pool2d_backward(): gradOutput height unexpected");

  /* resize */
  if (ndims == 3) {
    // 如果输入张量是3维的，则设置输出为三维形状
    set_output_raw_strided(0, {numPlanes, inputH, inputW}, {}, input.options());
  } else {
    // 如果输入张量是4维的，则设置输出为四维形状
    set_output_raw_strided(0, {numBatch, numPlanes, inputH, inputW}, {}, input.options());
  }
}
} // namespace meta

namespace native {
namespace {

template <typename scalar_t>
// 定义模板函数，用于处理单个批次帧的分数最大池化
static void fractional_max_pool2d_out_single_batch_frame(
  const scalar_t* input,                                // 输入参数，输入数据
  scalar_t* output,                                     // 输出参数，输出数据
  int64_t* indices,                                     // 输出参数，池化索引
  const scalar_t* randomSamples,                        // 输入参数，随机采样
  int numPlanes,                                        // 输入参数，平面数
  int inputW, int inputH,                               // 输入参数，输入宽度和高度
  int outputW, int outputH,                             // 输入参数，输出宽度和高度
  int poolSizeW, int poolSizeH) {                       // 输入参数，池大小宽度和高度

  // 并行处理每个平面
  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    // 遍历 planes 区间 [start, end)，其中每个 plane 包含两个随机样本，一个用于 W，一个用于 H
    for (const auto plane : c10::irange(start, end)) {

      /* 每个 plane 包含两个随机样本，一个用于 W，一个用于 H */
      const scalar_t* randomSamplesForPlane = randomSamples + plane * 2;

      /* 生成 W 方向的间隔序列 */
      auto sequenceW = generate_intervals<scalar_t>(
          randomSamplesForPlane[0], inputW, outputW, poolSizeW);
      
      /* 生成 H 方向的间隔序列 */
      auto sequenceH = generate_intervals<scalar_t>(
          randomSamplesForPlane[1], inputH, outputH, poolSizeH);

      /* 遍历输出的每个位置 */
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int h, w;

      // 获取当前 plane 的输入和输出的起始位置
      const scalar_t* inputForPlane = input + plane * inputW * inputH;
      scalar_t* outputForPlane = output + plane * outputW * outputH;
      int64_t* indicesForPlane = indices + plane * outputW * outputH;

      // 遍历输出的每个 h 方向
      for (h = 0; h < outputH; ++h) {
        // 获取 H 方向的起始位置
        int inputHStart = sequenceH[h];

        // 遍历输出的每个 w 方向
        for (w = 0; w < outputW; ++w) {
          // 获取 W 方向的起始位置
          int inputWStart = sequenceW[w];

          // 初始化局部最大值及其索引
          int h2 = inputHStart, w2 = inputWStart;
          scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
          int64_t maxIndex = h2 * inputW + w2;

          // 在池化窗口内寻找最大值及其索引
          for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
            for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
              // 断言确保索引在合法范围内
              AT_ASSERT(h2 >= 0 && h2 < inputH);
              AT_ASSERT(w2 >= 0 && w2 < inputW);

              // 计算在输入平面中的索引
              int planeIndex = h2 * inputW + w2;
              scalar_t val = inputForPlane[planeIndex];

              // 更新最大值及其索引
              if (val > maxVal || std::isnan(val)) {
                maxVal = val;
                maxIndex = planeIndex;
              }
            }
          }

          // 将池化后的最大值存入输出数组
          outputForPlane[h * outputW + w] = maxVal;
          // 将最大值的索引存入索引数组
          indicesForPlane[h * outputW + w] = maxIndex;
        }
      }
    }
  });
}

template <typename scalar_t>
// 定义静态函数，用于执行分数最大池化的反向传播，处理单个批次的帧
static void fractional_max_pool2d_out_frame(
  const scalar_t* input,  // 输入数据的指针
  scalar_t* output,       // 输出数据的指针
  int64_t* indices,       // 池化索引的指针
  const scalar_t* randomSamples,  // 随机采样值的指针
  int numBatch, int numPlanes,    // 批次数和平面数
  int inputW, int inputH,         // 输入宽度和高度
  int outputW, int outputH,       // 输出宽度和高度
  int poolSizeW, int poolSizeH) { // 池化窗口的宽度和高度
    if(numBatch == 1) {  // 如果批次数为1
      // 对单个批次的帧执行分数最大池化
      fractional_max_pool2d_out_single_batch_frame<scalar_t>(
        input,
        output,
        indices,
        randomSamples,
        numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH
      );
      return;  // 返回
    }
    // 使用并行处理，对每个批次的帧进行分数最大池化
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      for (const auto batch : c10::irange(start, end)) {
        // 对单个批次的帧执行分数最大池化
        fractional_max_pool2d_out_single_batch_frame<scalar_t>(
          input + batch * numPlanes * inputH * inputW,
          output + batch * numPlanes * outputH * outputW,
          indices + batch * numPlanes * outputH * outputW,
          randomSamples + batch * numPlanes * 2,
          numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
      }
    });
  }

template <typename scalar_t>
// 定义静态函数，用于执行分数最大池化的反向传播，处理单个批次的帧
static void fractional_max_pool2d_backward_out_single_batch_frame(
  scalar_t* gradInput,            // 梯度输入数据的指针
  const scalar_t* gradOutput,     // 梯度输出数据的指针
  const int64_t* indices,         // 池化索引的指针
  int numPlanes,                  // 平面数
  int inputW, int inputH,         // 输入宽度和高度
  int outputW, int outputH) {     // 输出宽度和高度
  // 使用并行处理，对每个平面进行分数最大池化的反向传播
  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (const auto plane : c10::irange(start, end)) {
      scalar_t* gradInputForPlane = gradInput + plane * inputW * inputH;
      const scalar_t* gradOutputForPlane = gradOutput + plane * outputW * outputH;
      const int64_t* indicesForPlane = indices + plane * outputW * outputH;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int h, w;
      // 遍历输出图像的每一个像素
      for (h = 0; h < outputH; ++h) {
        for (w = 0; w < outputW; ++w) {
          int outputIndex = h * outputW + w;
          int64_t index = indicesForPlane[outputIndex];
          // 断言索引在合法范围内
          AT_ASSERT(index >= 0 && index < inputW * inputH);

          // 累加梯度
          gradInputForPlane[index] += gradOutputForPlane[outputIndex];
        }
      }
    }
  });
}

template <typename scalar_t>
// 定义静态函数，用于执行分数最大池化的反向传播，处理整个批次的帧
static void fractional_max_pool2d_backward_out_frame(
  scalar_t* gradInput,            // 梯度输入数据的指针
  const scalar_t* gradOutput,     // 梯度输出数据的指针
  const int64_t* indices,         // 池化索引的指针
  int numBatch, int numPlanes,    // 批次数和平面数
  int inputW, int inputH,         // 输入宽度和高度
  int outputW, int outputH) {     // 输出宽度和高度
    if(numBatch == 1) {  // 如果批次数为1
      // 对单个批次的帧执行分数最大池化的反向传播
      fractional_max_pool2d_backward_out_single_batch_frame<scalar_t>(
        gradInput, gradOutput, indices,
        numPlanes,
        inputW, inputH, outputW, outputH
      );
      return;  // 返回
    }
    // 使用ATen库的parallel_for函数实现并行计算，将任务拆分为多个线程执行
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      // 遍历每个批次中的元素范围
      for (const auto batch : c10::irange(start, end)) {
        // 调用fractional_max_pool2d_backward_out_single_batch_frame模板函数，
        // 对单个批次的梯度计算进行反向传播
        fractional_max_pool2d_backward_out_single_batch_frame<scalar_t>(
          // 计算输入梯度的起始地址，根据批次、通道数、输入高度、输入宽度定位
          gradInput + batch * numPlanes * inputH * inputW,
          // 计算输出梯度的起始地址，根据批次、通道数、输出高度、输出宽度定位
          gradOutput + batch * numPlanes * outputH * outputW,
          // 计算最大池化操作中的索引起始地址，根据批次、通道数、输出高度、输出宽度定位
          indices + batch * numPlanes * outputH * outputW,
          // 传递通道数
          numPlanes,
          // 传递输入图像宽度
          inputW,
          // 传递输入图像高度
          inputH,
          // 传递输出图像宽度
          outputW,
          // 传递输出图像高度
          outputH);
      }
    });
} // 结束函数 fractional_max_pool2d_out_cpu

} // 匿名命名空间结束

TORCH_IMPL_FUNC(fractional_max_pool2d_out_cpu) (
  const at::Tensor& input_,                     // 输入张量（可能是4维的）
  IntArrayRef pool_size,                        // 池化窗口尺寸
  IntArrayRef output_size,                      // 输出尺寸
  const at::Tensor& randomSamples_,             // 随机采样张量
  const at::Tensor& output,                     // 输出张量
  const at::Tensor& indices) {                  // 索引张量

  fractional_max_pool_check_shape</*ndim*/ 2>(input_, randomSamples_); // 检查输入形状是否符合要求

  if (output.numel() == 0) {                    // 如果输出张量为空
    return;                                     // 直接返回
  }

  int64_t numBatch = 1;                         // 批次数初始化为1
  int64_t planeDim = 0;                         // 平面维度索引初始化为0
  int64_t heightDim = 1;                        // 高度维度索引初始化为1
  int64_t widthDim = 2;                         // 宽度维度索引初始化为2
  int64_t outputH = output_size[0];              // 输出高度从输出尺寸中获取
  int64_t outputW = output_size[1];              // 输出宽度从输出尺寸中获取
  int64_t poolSizeH = pool_size[0];              // 池化高度从池化尺寸中获取
  int64_t poolSizeW = pool_size[1];              // 池化宽度从池化尺寸中获取

  /* 获取连续的输入和样本 */
  auto input = input_.contiguous();             // 获取输入张量的连续版本
  auto randomSamples = randomSamples_.contiguous(); // 获取随机采样张量的连续版本

  int64_t ndims = input.ndimension();           // 获取输入张量的维度数

  if (ndims == 4) {                             // 如果输入张量是4维的
    numBatch = input.size(0);                   // 更新批次数
    planeDim++;                                 // 更新平面维度索引
    heightDim++;                                // 更新高度维度索引
    widthDim++;                                 // 更新宽度维度索引
  }

  /* 尺寸信息 */
  int64_t numPlanes = input.size(planeDim);     // 计算平面数
  int64_t inputH = input.size(heightDim);       // 计算输入高度
  int64_t inputW = input.size(widthDim);        // 计算输入宽度

  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16,
    kHalf,
    input.scalar_type(),
    "fractional_max_pool2d_out_frame", [&] {
      auto input_data = input.const_data_ptr<scalar_t>();     // 获取输入数据指针
      auto output_data = output.data_ptr<scalar_t>();         // 获取输出数据指针
      auto indices_data = indices.data_ptr<int64_t>();        // 获取索引数据指针
      auto randomSamples_data = randomSamples.const_data_ptr<scalar_t>(); // 获取随机采样数据指针
      fractional_max_pool2d_out_frame<scalar_t>(
        input_data,
        output_data,
        indices_data,
        randomSamples_data,
        numBatch, numPlanes,
        inputW, inputH,
        outputW, outputH,
        poolSizeW, poolSizeH);                             // 执行分数最大池化操作
    }
  );
}

TORCH_IMPL_FUNC(fractional_max_pool2d_backward_cpu) (
  const at::Tensor& gradOutput_,                // 梯度输出张量
  const at::Tensor& input,                      // 输入张量
  IntArrayRef pool_size,                        // 池化窗口尺寸
  IntArrayRef output_size,                      // 输出尺寸
  const at::Tensor& indices,                    // 索引张量
  const at::Tensor& gradInput) {                // 梯度输入张量

  gradInput.zero_();                            // 将梯度输入张量初始化为零

  int numBatch = 1;                             // 批次数初始化为1
  int planeDim = 0;                             // 平面维度索引初始化为0
  int heightDim = 1;                            // 高度维度索引初始化为1
  int widthDim = 2;                             // 宽度维度索引初始化为2

  int outputH = output_size[0];                  // 输出高度从输出尺寸中获取
  int outputW = output_size[1];                  // 输出宽度从输出尺寸中获取

  int ndims = input.ndimension();               // 获取输入张量的维度数
  if (ndims == 4) {                             // 如果输入张量是4维的
    numBatch = input.size(0);                   // 更新批次数
    planeDim = 1;                               // 更新平面维度索引
    heightDim++;                                // 更新高度维度索引
    widthDim++;                                 // 更新宽度维度索引
  }

  /* 尺寸信息 */
  int numPlanes = input.size(planeDim);         // 计算平面数
  int inputH = input.size(heightDim);           // 计算输入高度
  int inputW = input.size(widthDim);            // 计算输入宽度

  /* 获取连续的梯度输出 */
  auto gradOutput = gradOutput_.contiguous();   // 获取梯度输出张量的连续版本

  /* 反向传播 */
  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16,
    kHalf,
    input.scalar_type(),
    "fractional_max_pool2d_backward_frame", [&] {
      fractional_max_pool2d_backward_frame<scalar_t>(
        gradOutput.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        indices.data_ptr<int64_t>(),
        gradInput.data_ptr<scalar_t>(),
        numBatch, numPlanes,
        inputW, inputH,
        outputW, outputH,
        pool_size[1], pool_size[0]);           // 执行分数最大池化的反向传播操作
    }
  );
}
    input.scalar_type(), "fractional_max_pool2d_backward_out_frame", [&] {
      auto gradInput_data = gradInput.data_ptr<scalar_t>();  
      // 获取梯度输入数据的指针，类型为 scalar_t
      auto gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
      // 获取梯度输出数据的指针，类型为 scalar_t，是常量指针
      auto indices_data = indices.const_data_ptr<int64_t>();
      // 获取索引数据的指针，类型为 int64_t，是常量指针
      fractional_max_pool2d_backward_out_frame<scalar_t>(
        gradInput_data,  // 梯度输入数据的指针
        gradOutput_data,  // 梯度输出数据的指针
        indices_data,  // 索引数据的指针
        numBatch, numPlanes,  // 批次数和平面数
        inputW, inputH,  // 输入宽度和高度
        outputW, outputH  // 输出宽度和高度
      );
    }
}

} // 结束 at::native 命名空间

} // 结束 at 命名空间
```