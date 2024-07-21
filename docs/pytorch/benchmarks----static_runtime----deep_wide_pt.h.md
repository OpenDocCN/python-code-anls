# `.\pytorch\benchmarks\static_runtime\deep_wide_pt.h`

```
#pragma once

#include <ATen/CPUFunctions.h>  // 引入CPU功能函数库
#include <ATen/NativeFunctions.h>  // 引入ATen本地函数库
#include <torch/torch.h>  // 引入PyTorch主库

// 定义一个深度和宽度神经网络模型，继承自torch::nn::Module
struct DeepAndWide : torch::nn::Module {
  DeepAndWide(int num_features = 50) {
    // 初始化模型参数
    mu_ = register_parameter("mu_", torch::randn({1, num_features}));  // 均值参数
    sigma_ = register_parameter("sigma_", torch::randn({1, num_features}));  // 标准差参数
    fc_w_ = register_parameter("fc_w_", torch::randn({1, num_features + 1}));  // 全连接层权重参数
    fc_b_ = register_parameter("fc_b_", torch::randn({1}));  // 全连接层偏置参数
  }

  // 前向传播函数定义
  torch::Tensor forward(
      torch::Tensor ad_emb_packed,
      torch::Tensor user_emb,
      torch::Tensor wide) {
    // 对宽度特征进行预处理
    auto wide_offset = wide + mu_;  // 加上均值
    auto wide_normalized = wide_offset * sigma_;  // 标准化
    auto wide_noNaN = wide_normalized;  // 无NaN值处理
    // Placeholder for ReplaceNaN
    auto wide_preproc = torch::clamp(wide_noNaN, -10.0, 10.0);  // 使用clamp函数限制范围

    auto user_emb_t = torch::transpose(user_emb, 1, 2);  // 转置用户嵌入张量
    auto dp_unflatten = torch::bmm(ad_emb_packed, user_emb_t);  // 执行批量矩阵乘法
    auto dp = torch::flatten(dp_unflatten, 1);  // 展平结果
    auto input = torch::cat({dp, wide_preproc}, 1);  // 拼接处理后的宽度特征和深度特征

    auto fc1 = torch::nn::functional::linear(input, fc_w_, fc_b_);  // 执行线性变换
    auto pred = torch::sigmoid(fc1);  // 使用sigmoid函数得到预测结果
    return pred;  // 返回预测结果张量
  }

  torch::Tensor mu_, sigma_, fc_w_, fc_b_;  // 模型参数：均值、标准差、全连接层权重、全连接层偏置
};

// 使用本地函数和预分配张量实现的深度和宽度神经网络模型
// 可以用作静态运行时的“光速”版本
struct DeepAndWideFast : torch::nn::Module {
  DeepAndWideFast(int num_features = 50) {
    mu_ = register_parameter("mu_", torch::randn({1, num_features}));  // 初始化均值参数
    sigma_ = register_parameter("sigma_", torch::randn({1, num_features}));  // 初始化标准差参数
    fc_w_ = register_parameter("fc_w_", torch::randn({1, num_features + 1}));  // 初始化全连接层权重参数
    fc_b_ = register_parameter("fc_b_", torch::randn({1}));  // 初始化全连接层偏置参数
    allocated = false;  // 分配标志初始化为假
    prealloc_tensors = {};  // 预分配张量列表初始化为空
  }

  torch::Tensor forward(
      torch::Tensor ad_emb_packed,
      torch::Tensor user_emb,
      torch::Tensor wide) {
    torch::NoGradGuard no_grad;  // 禁用梯度计算

    if (!allocated) {
      auto wide_offset = at::add(wide, mu_);  // 加上均值
      auto wide_normalized = at::mul(wide_offset, sigma_);  // 标准化
      // Placeholder for ReplaceNaN
      auto wide_preproc = at::cpu::clamp(wide_normalized, -10.0, 10.0);  // 使用clamp函数限制范围

      auto user_emb_t = at::native::transpose(user_emb, 1, 2);  // 转置用户嵌入张量
      auto dp_unflatten = at::cpu::bmm(ad_emb_packed, user_emb_t);  // 执行CPU上的批量矩阵乘法
      // auto dp = at::native::flatten(dp_unflatten, 1);  // 展平结果
      auto dp = dp_unflatten.view({dp_unflatten.size(0), 1});  // 重塑张量形状为指定大小
      auto input = at::cpu::cat({dp, wide_preproc}, 1);  // 拼接处理后的宽度特征和深度特征

      // fc1 = torch::nn::functional::linear(input, fc_w_, fc_b_);
      fc_w_t_ = torch::t(fc_w_);  // 计算权重张量的转置
      auto fc1 = torch::addmm(fc_b_, input, fc_w_t_);  // 执行矩阵乘法并添加偏置

      auto pred = at::cpu::sigmoid(fc1);  // 使用sigmoid函数得到预测结果

      // 将使用过的张量存储在预分配张量列表中
      prealloc_tensors = {
          wide_offset,
          wide_normalized,
          wide_preproc,
          user_emb_t,
          dp_unflatten,
          dp,
          input,
          fc1,
          pred};
      allocated = true;  // 设置分配标志为真

      return pred;  // 返回预测结果张量
    }
    } else {
      // 可能的优化：可以将 add 和 mul 操作融合在一起（例如使用 Eigen 库）。
      at::add_out(prealloc_tensors[0], wide, mu_);
      at::mul_out(prealloc_tensors[1], prealloc_tensors[0], sigma_);

      // 对 prealloc_tensors[1] 中的数据进行裁剪，确保数值在 [-10.0, 10.0] 范围内。
      at::native::clip_out(
          prealloc_tensors[1], -10.0, 10.0, prealloc_tensors[2]);

      // 可能的优化：可以预先对原始张量进行转置。
      // prealloc_tensors[3] = at::native::transpose(user_emb, 1, 2);
      if (prealloc_tensors[3].data_ptr() != user_emb.data_ptr()) {
        // 如果 prealloc_tensors[3] 和 user_emb 不是同一个张量，则进行重设操作。
        auto sizes = user_emb.sizes();
        auto strides = user_emb.strides();
        prealloc_tensors[3].set_(
            user_emb.storage(),
            0,
            {sizes[0], sizes[2], sizes[1]},
            {strides[0], strides[2], strides[1]});
      }

      // 可能的优化：可以直接调用 MKLDNN 库。
      // 使用 BMM 进行批矩阵乘操作。
      at::cpu::bmm_out(ad_emb_packed, prealloc_tensors[3], prealloc_tensors[4]);

      // 如果 prealloc_tensors[5] 和 prealloc_tensors[4] 不是同一个张量，则重新初始化视图。
      if (prealloc_tensors[5].data_ptr() != prealloc_tensors[4].data_ptr()) {
        prealloc_tensors[5] =
            prealloc_tensors[4].view({prealloc_tensors[4].size(0), 1});
      }

      // 可能的优化：可以使用精心构造的张量视图，替换 cat 操作，传递给上面的 _out 操作。
      // 在第二维度上连接 prealloc_tensors[5] 和 prealloc_tensors[2]。
      at::cpu::cat_outf(
          {prealloc_tensors[5], prealloc_tensors[2]}, 1, prealloc_tensors[6]);

      // 使用 addmm 进行矩阵相加乘操作。
      at::cpu::addmm_out(
          prealloc_tensors[7], fc_b_, prealloc_tensors[6], fc_w_t_, 1, 1);

      // 对结果进行 sigmoid 激活操作。
      at::cpu::sigmoid_out(prealloc_tensors[7], prealloc_tensors[8]);

      // 返回 sigmoid 操作后的结果。
      return prealloc_tensors[8];
    }
  }
  // 声明需要使用的变量
  torch::Tensor mu_, sigma_, fc_w_, fc_b_, fc_w_t_;
  std::vector<torch::Tensor> prealloc_tensors;
  bool allocated = false;
// 定义一个以分号结尾的空代码块，通常作为语法占位符或是某些特定代码结构的标志
};

// 声明一个返回类型为 torch::jit::Module 的函数 getDeepAndWideSciptModel，接受一个整数参数 num_features，默认值为 50
torch::jit::Module getDeepAndWideSciptModel(int num_features = 50);

// 声明一个返回类型为 torch::jit::Module 的函数 getTrivialScriptModel，无参数
torch::jit::Module getTrivialScriptModel();

// 声明一个返回类型为 torch::jit::Module 的函数 getLeakyReLUScriptModel，无参数
torch::jit::Module getLeakyReLUScriptModel();

// 声明一个返回类型为 torch::jit::Module 的函数 getLeakyReLUConstScriptModel，无参数
torch::jit::Module getLeakyReLUConstScriptModel();

// 声明一个返回类型为 torch::jit::Module 的函数 getLongScriptModel，无参数
torch::jit::Module getLongScriptModel();

// 声明一个返回类型为 torch::jit::Module 的函数 getSignedLog1pModel，无参数
torch::jit::Module getSignedLog1pModel();
```