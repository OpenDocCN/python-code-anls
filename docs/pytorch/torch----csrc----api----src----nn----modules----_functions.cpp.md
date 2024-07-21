# `.\pytorch\torch\csrc\api\src\nn\modules\_functions.cpp`

```py
// 引入必要的头文件，包括C++标准库和PyTorch相关库的头文件
#include <c10/util/irange.h>
#include <torch/nn/modules/_functions.h>

// 使用torch::autograd命名空间
using namespace torch::autograd;

// torch::nn::functions命名空间
namespace torch {
namespace nn {
namespace functions {

// CrossMapLRN2d类的前向传播函数
Variable CrossMapLRN2d::forward(
    AutogradContext* ctx,                 // 自动求导上下文指针
    const Variable& input,                // 输入变量
    const CrossMapLRN2dOptions& options) { // CrossMapLRN2d的选项参数

  // 保存选项参数到上下文的saved_data中
  ctx->saved_data["size"] = options.size();
  ctx->saved_data["alpha"] = options.alpha();
  ctx->saved_data["beta"] = options.beta();
  ctx->saved_data["k"] = options.k();
  ctx->saved_data["scale"] = torch::Tensor(); // 初始化scale为空Tensor

  // 检查输入的维度是否为4
  TORCH_CHECK(input.dim() == 4);

  // 如果scale未定义，则根据input的options创建一个空的Tensor
  ctx->saved_data["scale"] = ctx->saved_data["scale"].toTensor().defined()
      ? ctx->saved_data["scale"]
      : torch::empty({0}, input.options());

  // 创建一个空的output Tensor
  torch::Tensor output = torch::empty({0}, input.options());

  // 获取输入的通道数
  int64_t channels = input.size(1);

  // 将output Tensor resize为和input相同的大小
  output.resize_as_(input);
  // 将scale Tensor resize为和input相同的大小
  ctx->saved_data["scale"].toTensor().resize_as_(input);

  // 使用output存储input的平方值
  auto input_square = output;
  torch::pow_out(input_square, input, 2);

  // 计算前填充量
  int64_t pre_pad =
      static_cast<int64_t>((ctx->saved_data["size"].toInt() - 1) / 2 + 1);
  int64_t pre_pad_crop = pre_pad > channels ? channels : pre_pad;

  // 获取scale中第一个通道的Tensor，并清零
  auto scale_first = ctx->saved_data["scale"].toTensor().select(1, 0);
  scale_first.zero_();

  // 计算第一个特征图的归一化
  for (const auto c : c10::irange(pre_pad_crop)) {
    scale_first.add_(input_square.select(1, c));
  }

  // 重复计算以供后续特征图的归一化
  torch::Tensor scale_previous, scale_current, square_next, square_previous;

  for (const auto c : c10::irange(1, channels)) {
    scale_previous = ctx->saved_data["scale"].toTensor().select(1, c - 1);
    scale_current = ctx->saved_data["scale"].toTensor().select(1, c);
    scale_current.copy_(scale_previous);

    if (c < channels - pre_pad + 1) {
      square_next = input_square.select(1, c + pre_pad - 1);
      scale_current.add_(square_next, 1);
    }

    if (c > pre_pad) {
      square_previous = input_square.select(1, c - pre_pad);
      scale_current.add_(square_previous, -1);
    }
  }

  // 对scale进行归一化调整和加上常数k
  ctx->saved_data["scale"]
      .toTensor()
      .mul_(
          ctx->saved_data["alpha"].toDouble() / ctx->saved_data["size"].toInt())
      .add_(ctx->saved_data["k"].toInt());

  // 对output进行幂运算调整
  torch::pow_out(
      output,
      ctx->saved_data["scale"].toTensor(),
      -ctx->saved_data["beta"].toDouble());

  // 将output乘以input，得到最终的输出
  output.mul_(input);

  // 保存用于反向传播的变量
  ctx->save_for_backward({input, output});

  // 返回最终的output作为前向传播的结果
  return output;
}

// CrossMapLRN2d类的反向传播函数
variable_list CrossMapLRN2d::backward(
    AutogradContext* ctx,
    const variable_list& grad_outputs) {
  
  // 略，后续部分未提供，此处省略反向传播函数的注释
  // 需要继续完成反向传播函数的注释
}
    // 定义梯度输出变量，grad_outputs 是一个变量列表，包含所有梯度输出
    variable_list grad_outputs) {
  // 获取第一个梯度输出
  auto grad_output = grad_outputs[0];
  // 获取输入变量，这里假设 ctx 是一个上下文对象，通过它可以获取保存的变量列表
  auto input = ctx->get_saved_variables()[0];
  // 获取输出变量
  auto output = ctx->get_saved_variables()[1];
  // 创建一个与 grad_output 形状相同的空张量，作为梯度输入
  auto grad_input = torch::empty({0}, grad_output.options());

  // 获取输入的批量大小、通道数、高度和宽度
  int64_t batch_size = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  // 创建一个空张量 padded_ratio，其形状为 channels + size - 1, input_height, input_width
  auto padded_ratio = torch::empty(
      {channels + ctx->saved_data["size"].toInt() - 1,
       input_height,
       input_width},
      input.options());
  // 创建一个与 input_height 和 input_width 相同形状的空张量 accum_ratio
  auto accum_ratio = torch::empty({input_height, input_width}, input.options());
  // 计算缓存比例值
  double cache_ratio_value = 2 * ctx->saved_data["alpha"].toDouble() *
      ctx->saved_data["beta"].toDouble() / ctx->saved_data["size"].toInt();
  // 计算 inversePrePad 值
  int64_t inversePrePad = static_cast<int64_t>(
      ctx->saved_data["size"].toInt() -
      (ctx->saved_data["size"].toInt() - 1) / 2);

  // 将 grad_input 调整为与 input 相同的形状
  grad_input.resize_as_(input);
  // 计算 grad_input 的值
  torch::pow_out(
      grad_input,
      ctx->saved_data["scale"].toTensor(),
      -ctx->saved_data["beta"].toDouble())
      .mul_(grad_output);

  // 将 padded_ratio 张量清零
  padded_ratio.zero_();
  // 获取 padded_ratio 的中心部分，从第 inversePrePad 个位置开始的 channels 个元素
  auto padded_ratio_center = padded_ratio.narrow(0, inversePrePad, channels);

  // 遍历每个批次中的元素
  for (const auto n : c10::irange(batch_size)) {
    // 计算 padded_ratio_center 的值
    torch::mul_out(padded_ratio_center, grad_output[n], output[n]);
    // 将 padded_ratio_center 除以 ctx->saved_data["scale"] 中的张量值
    padded_ratio_center.div_(ctx->saved_data["scale"].toTensor()[n]);
    // 计算累积和 accum_ratio
    torch::sum_out(
        accum_ratio,
        padded_ratio.narrow(0, 0, ctx->saved_data["size"].toInt() - 1),
        0,
        /*keepdim=*/false);
    // 遍历每个通道中的元素
    for (const auto c : c10::irange(channels)) {
      // 累积 accum_ratio 的值
      accum_ratio.add_(padded_ratio[c + ctx->saved_data["size"].toInt() - 1]);
      // 更新 grad_input[n][c] 的值
      grad_input[n][c].addcmul_(input[n][c], accum_ratio, -cache_ratio_value);
      // 减去 padded_ratio[c] 的值
      accum_ratio.add_(padded_ratio[c], -1);
    }
  }

  // 返回变量列表，包括 grad_input 和几个空 Variable()
  return variable_list{
      grad_input, Variable(), Variable(), Variable(), Variable()};
}
}

// 结束 functions 命名空间
} // namespace functions

// 结束 nn 命名空间
} // namespace nn

// 结束 torch 命名空间
} // namespace torch
```