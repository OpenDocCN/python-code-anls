# `.\pytorch\aten\src\ATen\native\MaxUnpooling.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/MaxUnpoolKernel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_unpool2d_native.h>
#include <ATen/ops/max_unpool3d_native.h>
#endif

namespace at::native {

// 定义在 CPU 上执行的二维最大反池化操作的前向函数，将结果写入输出张量
Tensor& max_unpooling2d_forward_out_cpu(
    const Tensor& self_,            // 输入张量
    const Tensor& indices_,         // 最大值池化过程中记录的索引张量
    IntArrayRef output_size,        // 输出张量的大小（高度和宽度）
    Tensor& output) {               // 输出张量

  // See Note [Writing Nondeterministic Operations]
  // 标记为非确定性操作，因为可能存在重复的索引
  at::globalContext().alertNotDeterministic("max_unpooling2d_forward_out");

  auto oheight = output_size[0];    // 输出张量的高度
  auto owidth = output_size[1];     // 输出张量的宽度

  // 检查索引张量的数据类型是否为 int64
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ", indices_.scalar_type());

  // 检查输出大小参数是否为两个元素（高度和宽度）
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size, but got ", output_size.size(), " elements.");

  // 检查输入张量的维度是否为3或4（即是否为3D或4D张量）
  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with ", self_.ndimension(), " dimensions.");

  // 检查索引张量的形状是否与输入张量相同
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Expected shape of indices to be same as that of the input tensor (", self_.sizes(),
      ") but got indices tensor with shape: ", indices_.sizes());

  // 检查除了批次维度外的所有维度是否具有非零大小
  for (const auto i : c10::irange(1, self_.ndimension())) {
    TORCH_CHECK(self_.size(i) > 0, "max_unpooling2d_forward_out_cpu(): ",
                "Expected input to have non-zero size for non-batch dimensions, but got ",
                self_.sizes(), " with dimension ", i , " being empty.");
  }

  // 根据输入张量的内存格式获取连续的 self 和 indices 张量
  auto memory_format = self_.suggest_memory_format();
  auto self = self_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  // 根据输入张量的维度选择合适的输出张量形状
  if (self.ndimension() == 3) {
    int64_t numChannels = self.size(0);
    output.resize_({numChannels, oheight, owidth});  // 为三维张量设置输出形状
  } else {
    int64_t numBatch = self.size(0);
    int64_t numChannels = self.size(1);
    output.resize_({numBatch, numChannels, oheight, owidth}, memory_format);  // 为四维张量设置输出形状
  }
  output.zero_();  // 将输出张量初始化为零

  // 如果输出张量的元素数量不为零，则执行最大反池化的 CPU 内核操作
  if (output.numel() != 0) {
    max_unpool2d_kernel(kCPU, output, self, indices);
  }

  return output;  // 返回输出张量
};

// 在 CPU 上执行二维最大反池化操作的前向函数，返回新的输出张量
Tensor max_unpooling2d_forward_cpu(
    const Tensor& self,         // 输入张量
    const Tensor& indices,      // 最大值池化过程中记录的索引张量
    IntArrayRef output_size) {  // 输出张量的大小（高度和宽度）

  auto output = at::empty({0}, self.options());  // 创建一个空的输出张量
  at::native::max_unpooling2d_forward_out_cpu(self, indices, output_size, output);  // 调用前向函数进行计算
  return output;  // 返回新的输出张量
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const char *fn_name) {


  // 检查 indices 张量的数据类型是否为 int64
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  // 检查 input 张量的维度是否为 4 或 5
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with ", input.ndimension(), " dimensions.");
  // 检查 output_size 应该包含三个元素（深度、高度、宽度）
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size, but got ", output_size.size(), " elements.");
  // 检查 stride 应该包含三个元素（深度、高度、宽度）
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stride, but got: ", stride.size(), " elements.");
  // 检查 padding 应该包含三个元素（深度、高度、宽度）
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding, but got: ", padding.size(), " elements.");
  // 检查 indices 张量的形状是否与 input 张量相同
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Expected shape of indices to be same as that of the input tensor (", input.sizes(),
      ") but got indices tensor with shape: ", indices.sizes());

  // 对于每个非批处理维度，检查输入张量的大小是否大于零
  for (const auto i : c10::irange(1, input.ndimension())) {
    TORCH_CHECK(input.size(i) > 0, fn_name,
                ": Expected input to have non-zero size for non-batch dimensions, but got ",
                input.sizes(), " with dimension ", i , " being empty.");
  }

  // 检查 stride 的三个元素是否大于零
  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

  // 获取输出尺寸的深度、高度和宽度
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  // 定义用于索引的维度号，根据输入张量的维度数决定
  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  // 如果输入张量维度为 5，则更新索引维度号
  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  // 获取输入张量中通道数（或切片数）
  int nslices = input.size(dimn);

  // 如果定义了梯度输出 gradOutput
  if (gradOutput.defined()) {
    // 检查 gradOutput 的深度、高度和宽度是否与预期输出尺寸一致
    if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
        oW != gradOutput.size(dimw)) {
      AT_ERROR(
          "Inconsistent gradOutput size. oT= ",
          oT,
          ", oH= ",
          oH,
          ", oW= ",
          oW,
          ". gradOutput: ",
          gradOutput.size(dimt),
          "x",
          gradOutput.size(dimh),
          "x",
          gradOutput.size(dimw));
    }
    // 检查 gradOutput 和 input 张量的维度数以及通道数是否相同
    TORCH_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
            gradOutput.size(dimn) == nslices,
        "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
  }
}

Tensor& max_unpooling3d_forward_out_cpu(const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // 标记不确定性操作，因为存在重复的索引
  // 在存在重复索引时是不确定性的操作
  at::globalContext().alertNotDeterministic("max_unpooling3d_forward_out");

  // 检查输出张量是否是连续的
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

  // 保证输入张量和索引张量是连续的
  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  // 执行形状检查，确保参数和张量维度匹配
  max_unpooling3d_shape_check(
    self_, Tensor(), indices_, output_size, stride, padding, "max_unpooling3d_forward_out_cpu()");

  // 提取输出尺寸的三个维度
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  // 根据输入张量的维度设置输出张量的形状
  if (self_.ndimension() == 5) {
    output.resize_({self.size(0), self.size(1), oT, oH, oW});
  } else {
    output.resize_({self.size(0), oT, oH, oW});
  }
  
  // 将输出张量的所有元素置零
  output.zero_();
  
  // 如果输出张量不为空，则执行最大解池化的核心计算
  if (output.numel() != 0) {
    max_unpool3d_kernel(kCPU, output, self, indices);
  }

  // 返回输出张量
  return output;
}

Tensor max_unpooling3d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  // 创建一个空的输出张量，与输入张量使用相同的选项
  auto output = at::empty({0}, self.options());
  
  // 调用带输出参数的函数进行计算
  at::native::max_unpooling3d_forward_out_cpu(
      self, indices, output_size, stride, padding, output);
  
  // 返回计算结果的输出张量
  return output;
}

// 定义核心计算函数的分发器
DEFINE_DISPATCH(max_unpool2d_kernel);
DEFINE_DISPATCH(max_unpool3d_kernel);

} // namespace at::native
```