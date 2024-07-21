# `.\pytorch\aten\src\ATen\native\FractionalMaxPool3d.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/FractionalMaxPooling.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/fractional_max_pool3d_backward_native.h>
#include <ATen/ops/fractional_max_pool3d_native.h>
#endif

#include <vector>

// 命名空间定义
namespace at::meta {

// 定义预计算元函数 'fractional_max_pool3d'
TORCH_PRECOMPUTE_META_FUNC(fractional_max_pool3d)(
  // 输入参数：输入张量 input_
  const at::Tensor& input_,
  // 池化尺寸，三个维度的大小
  IntArrayRef pool_size,
  // 输出尺寸，三个维度的大小
  IntArrayRef output_size,
  // 随机采样张量
  const at::Tensor& randomSamples
) {
  // 检查池化尺寸必须为三维
  TORCH_CHECK(
      pool_size.size() == 3,
      "fractional_max_pool3d: kernel_size must either be a single Int or tuple of three Ints")
  // 检查输出尺寸必须为三维
  TORCH_CHECK(
      output_size.size() == 3,
      "fractional_max_pool3d: output_size must either be a single Int or tuple of three Ints")
  
  // 提取输出尺寸的各个维度值
  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];
  // 提取池化尺寸的各个维度值
  int64_t poolSizeT = pool_size[0];
  int64_t poolSizeH = pool_size[1];
  int64_t poolSizeW = pool_size[2];

  // 初始化批次数为1
  int64_t numBatch = 1;
  // 平面维度索引
  int64_t planeDim = 0;
  // 时间维度索引
  int64_t timeDim = 1;
  // 高度维度索引
  int64_t heightDim = 2;
  // 宽度维度索引
  int64_t widthDim = 3;

  // 获取输入张量的维度数
  int64_t ndims = input_.ndimension();
  // 检查输入张量必须是4维或者5维
  TORCH_CHECK(ndims == 4 || ndims == 5,
              "fractional_max_pool3d_out(): Expected 4D or 5D tensor, but got: ",
              input_.sizes());
  
  // 对于每个非批次维度，检查其大小不能为0
  for (const auto i : c10::irange(1, ndims)) {
    TORCH_CHECK(input_.size(i) > 0,
                "fractional_max_pool3d_out(): Expected input to have non-zero size for non-batch dimensions, but got",
                input_.sizes(), " with dimension ", i, " being empty.");
  }

  // 如果张量是5维的，更新相关维度索引和批次数
  if (ndims == 5) {
    numBatch = input_.size(0);
    planeDim++;
    timeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  // 提取平面数
  int64_t numPlanes = input_.size(planeDim);
  // 提取输入的时间维度大小
  int64_t inputT = input_.size(timeDim);
  // 提取输入的高度维度大小
  int64_t inputH = input_.size(heightDim);
  // 提取输入的宽度维度大小
  int64_t inputW = input_.size(widthDim);

  // 检查池化后的时间维度不能超过输入时间维度
  TORCH_CHECK(outputT + poolSizeT - 1 < inputT,
           "fractional_max_pool3d_out(): pool time ", poolSizeT,
           " too large relative to input time ", inputT);
  // 检查池化后的宽度维度不能超过输入宽度维度
  TORCH_CHECK(outputW + poolSizeW - 1 < inputW,
           "fractional_max_pool3d_out(): pool width ", poolSizeW,
           " too large relative to input width ", inputW);
  // 检查池化后的高度维度不能超过输入高度维度
  TORCH_CHECK(outputH + poolSizeH - 1 < inputH,
           "fractional_max_pool3d_out(): pool height ", poolSizeH,
           " too large relative to input height ", inputH);

  // 如果张量是4维的
  if (ndims == 4) {
    /* resize output */
    // 设置输出张量的原始步进化表示
    set_output_raw_strided(0, {numPlanes, outputT, outputH, outputW}, {}, input_.options());
    /* indices will contain the locations for each output point */
    // 设置输出张量的原始步进化表示，数据类型为长整型
    set_output_raw_strided(1, {numPlanes, outputT, outputH, outputW}, {}, input_.options().dtype(kLong));
  } else {
    # 设置第一个输出张量的形状和选项
    set_output_raw_strided(0, {numBatch, numPlanes, outputT, outputH, outputW}, {}, input_.options());
    # 设置第二个输出张量的形状为相同，但数据类型为长整型
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {numBatch, numPlanes, outputT, outputH, outputW}, {}, input_.options().dtype(kLong));
  }

  # 返回预计算的 TORCH_PRECOMPUTE_STRUCT(fractional_max_pool3d) 实例，并设置其参数
  return TORCH_PRECOMPUTE_STRUCT(fractional_max_pool3d)().set_numBatch(numBatch).set_numPlanes(numPlanes).set_inputT(inputT).set_inputH(inputH).set_inputW(inputW)
                                                         .set_poolSizeT(poolSizeT).set_poolSizeH(poolSizeH).set_poolSizeW(poolSizeW)
                                                         .set_outputT(outputT).set_outputH(outputH).set_outputW(outputW);
} // namespace at::meta
// 定义了一个静态函数，用于在三维分数最大池化的CPU实现中处理单个批次的帧
static void fractional_max_pool3d_out_frame(
  // 输入数据指针，指向输入张量的数据
  const scalar_t* input,
  // 输出数据指针，指向输出张量的数据
  scalar_t* output,
  // 索引数据指针，指向输出最大值索引的数据
  int64_t* indices,
  // 随机采样数据指针，指向随机采样数据的数据
  const scalar_t* randomSamples,
  // 批次数目
  int64_t numBatch,
  // 平面数目（通道数）
  int64_t numPlanes,
  // 输入的时间维度大小
  int64_t inputT,
  // 输入的高度维度大小
  int64_t inputH,
  // 输入的宽度维度大小
  int64_t inputW,
  // 输出的时间维度大小
  int64_t outputT,
  // 输出的高度维度大小
  int64_t outputH,
  // 输出的宽度维度大小
  int64_t outputW,
  // 池化核的时间维度大小
  int64_t poolSizeT,
  // 池化核的高度维度大小
  int64_t poolSizeH,
  // 池化核的宽度维度大小
  int64_t poolSizeW) {
    // 如果批次数目为1，则调用单批次帧的分数最大池化函数
    if(numBatch == 1) {
      fractional_max_pool3d_out_single_batch_frame<scalar_t>(
        input, output, indices, randomSamples,
        numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW,
        poolSizeT, poolSizeH, poolSizeW
      );
      return;
    }

    // 使用ATen库的并行函数，对每个批次进行并行处理
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      // 遍历每个批次
      for (const auto batch : c10::irange(start, end)) {
        // 调用单批次帧的分数最大池化函数，处理当前批次的数据
        fractional_max_pool3d_out_single_batch_frame<scalar_t>(
          // 计算当前批次在输入张量中的偏移量，以处理该批次的数据
          input + batch * numPlanes * inputW * inputH * inputT,
          // 计算当前批次在输出张量中的偏移量，以保存处理结果
          output + batch * numPlanes * outputW * outputH * outputT,
          // 计算当前批次在索引张量中的偏移量，以保存最大值的索引
          indices + batch * numPlanes * outputW * outputH * outputT,
          // 计算当前批次在随机采样数据中的偏移量，以使用对应的采样数据
          randomSamples + batch * numPlanes * 3,
          // 平面数目（通道数）
          numPlanes,
          // 输入的时间维度大小
          inputT, inputH, inputW,
          // 输出的时间维度大小
          outputT, outputH, outputW,
          // 池化核的时间维度大小
          poolSizeT, poolSizeH, poolSizeW
        );
      }
    });
  }

} // 匿名命名空间结束

// Torch的CPU实现函数，用于执行三维分数最大池化操作
TORCH_IMPL_FUNC(fractional_max_pool3d_out_cpu)(
  // 输入张量的常量引用
  const at::Tensor& input_,
  // 池化核的时间维度大小
  int64_t poolSizeT,
  // 池化核的高度维度大小
  int64_t poolSizeH,
  // 池化核的宽度维度大小
  int64_t poolSizeW,
  // 输出的时间维度大小
  int64_t outputT,
  // 输出的高度维度大小
  int64_t outputH,
  // 输出的宽度维度大小
  int64_t outputW,
  // 随机采样张量的常量引用
  const at::Tensor& randomSamples_,
  // 批次数目
  int64_t numBatch,
  // 平面数目（通道数）
  int64_t numPlanes,
  // 输入的时间维度大小
  int64_t inputT,
  // 输入的高度维度大小
  int64_t inputH,
  // 输入的宽度维度大小
  int64_t inputW,
  // 输出张量的引用
  const at::Tensor& output,
  // 索引张量的引用
  const at::Tensor& indices) {

  // 检查输入张量和随机采样数据的形状是否匹配
  fractional_max_pool_check_shape</*ndim*/ 3>(input_, randomSamples_);

  // 如果输出张量的元素个数为0，则直接返回
  if (output.numel() == 0) {
    return;
  }

  /* get contiguous input and samples */
  // 获取输入张量的连续版本
  auto input = input_.contiguous();
  // 获取随机采样张量的连续版本
  auto randomSamples = randomSamples_.contiguous();

  // 使用ATen的宏，针对浮点数类型和其他特定类型执行操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16,
    kHalf,
    input.scalar_type(),
    // 操作名称，用于错误消息中的标识
    "fractional_max_pool3d_out_frame",
    [&] {
      // 调用具体的三维分数最大池化帧处理函数
      fractional_max_pool3d_out_frame<scalar_t>(
        // 输入张量的常量数据指针，指向其数据
        input.const_data_ptr<scalar_t>(),
        // 输出张量的数据指针，指向其数据
        output.data_ptr<scalar_t>(),
        // 索引张量的数据指针，指向其数据
        indices.data_ptr<int64_t>(),
        // 随机采样张量的常量数据指针，指向其数据
        randomSamples.const_data_ptr<scalar_t>(),
        // 批次数目
        numBatch,
        // 平面数目（通道数）
        numPlanes,
        // 输入的时间维度大小
        inputT, inputH, inputW,
        // 输出的时间维度大小
        outputT, outputH, outputW,
        // 池化核的时间维度大小
        poolSizeT, poolSizeH, poolSizeW
      );
    }
  );
}
    // 对于每一个平面（plane），遍历从 start 到 end 的范围
    for (const auto plane : c10::irange(start, end)) {
      // 计算当前平面在梯度输入（gradInput）中的起始位置
      scalar_t* gradInputForPlane = gradInput + plane * inputT * inputH * inputW;
      // 获取当前平面在梯度输出（gradOutput）中的起始位置
      const scalar_t* gradOutputForPlane = gradOutput +
                  plane * outputT * outputH * outputW;
      // 获取当前平面在索引数组（indices）中的起始位置
      const int64_t* indicesForPlane = indices + plane * outputT * outputH * outputW;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 定义循环中使用的临时变量 h, w, t
      int64_t h, w, t;
      // 遍历输出张量的时间维度（outputT）
      for (t = 0; t < outputT; ++t) {
        // 遍历输出张量的高度维度（outputH）
        for (h = 0; h < outputH; ++h) {
          // 遍历输出张量的宽度维度（outputW）
          for (w = 0; w < outputW; ++w) {
            // 计算在输出张量中的索引
            int64_t outputIndex = t * outputH * outputW + h * outputW + w;
            // 获取对应的输入索引
            int64_t index = indicesForPlane[outputIndex];
            // 断言输入索引的合法性，即必须在合理范围内
            AT_ASSERT(index >= 0 && index < inputT * inputH * inputW);
            // 将梯度输出值累加到对应的梯度输入位置
            gradInputForPlane[index] += gradOutputForPlane[outputIndex];
          }
        }
      }
    }
  });
}// anonymous namespace

template<typename scalar_t>
static void fractional_max_pool3d_backward_out_frame(
  scalar_t* gradInput,
  const scalar_t* gradOutput,
  const int64_t* indices,
  int64_t numBatch, int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW) {
    // 如果 batch 大小为 1，则调用单批次的反向传播函数
    if(numBatch == 1) {
      fractional_max_pool3d_backward_out_single_batch_frame<scalar_t>(
        gradInput, gradOutput, indices,
        numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW
      );
      return;
    }

    // 并行处理每个 batch 中的数据
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      for (const auto batch : c10::irange(start, end)) {
        // 调用单批次的反向传播函数，处理当前批次的数据
        fractional_max_pool3d_backward_out_single_batch_frame<scalar_t>(
          gradInput + batch * numPlanes * inputW * inputH * inputT,
          gradOutput + batch * numPlanes * outputW * outputH * outputT,
          indices + batch * numPlanes * outputW * outputH * outputT,
          numPlanes,
          inputT, inputH, inputW,
          outputT, outputH, outputW
        );
      }
    });
  }


void fractional_max_pool3d_backward_out_cpu_template(
  const Tensor& input,
  const Tensor& gradOutput_,
  Tensor& gradInput,
  IntArrayRef output_size,
  IntArrayRef pool_size /* unused */,
  const Tensor& indices) {

  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];

  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t timeDim = 1;
  int64_t heightDim = 2;
  int64_t widthDim = 3;

  int64_t ndims = input.ndimension();
  // 如果输入数据是 5 维，则更新批次相关的维度信息
  if (ndims == 5) {
    numBatch = input.size(0);
    planeDim = 1;
    heightDim++;
    widthDim++;
    timeDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputT = input.size(timeDim);
  int64_t inputH = input.size(heightDim);
  int64_t inputW = input.size(widthDim);

  // 检查梯度输出的尺寸是否符合预期
  TORCH_CHECK(outputT == gradOutput_.size(timeDim),
           "fractional_max_pool3d_backward_out(): gradOutput time unexpected");
  TORCH_CHECK(outputH == gradOutput_.size(heightDim),
           "fractional_max_pool3d_backward_out(): ",
           "gradOutput height unexpected");
  TORCH_CHECK(outputW == gradOutput_.size(widthDim),
           "fractional_max_pool3d_backward_out(): gradOutput width unexpected");

  /* get contiguous gradOutput */
  // 获取连续的梯度输出张量
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  // 调整梯度输入张量的尺寸和梯度清零
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* backprop */
  // 分发不同数据类型的反向传播处理
  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16,
    kHalf,
    input.scalar_type(),
    "fractional_max_pool3d_backward_out_frame",
    [&]{
      fractional_max_pool3d_backward_out_frame<scalar_t>(
        gradInput.data_ptr<scalar_t>(),
        gradOutput.const_data_ptr<scalar_t>(),
        indices.const_data_ptr<int64_t>(),
        numBatch, numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW
      );
    }
  );
}
# 定义一个函数，用于计算三维分数最大池化操作的反向传播，针对 CPU 计算
Tensor& fractional_max_pool3d_backward_out_cpu(const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices,
  at::Tensor& gradInput) {
  # 调用模板函数实现具体的三维分数最大池化反向传播计算，将结果写入 gradInput 中
  fractional_max_pool3d_backward_out_cpu_template(
    input,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  # 返回计算得到的梯度输入 gradInput
  return gradInput;
}

# 定义一个函数，用于计算三维分数最大池化操作的反向传播，针对 CPU 计算
Tensor fractional_max_pool3d_backward_cpu(
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices) {
  # 创建一个与 input 具有相同选项的空张量 gradInput
  Tensor gradInput = at::empty({0}, input.options());
  # 调用模板函数实现具体的三维分数最大池化反向传播计算，将结果写入 gradInput 中
  fractional_max_pool3d_backward_out_cpu_template(
    input,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  # 返回计算得到的梯度输入 gradInput
  return gradInput;
}

} // namespace at::native
```