# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qnnpack_utils.h`

```py
#pragma once
// 指令，确保头文件只被编译一次

#include <ATen/Tensor.h>
// 包含 PyTorch ATen 张量的头文件

#include <c10/core/QScheme.h>
// 包含 PyTorch 的量化方案枚举定义的头文件

#ifdef USE_PYTORCH_QNNPACK
// 如果定义了 USE_PYTORCH_QNNPACK，则编译以下内容

// TODO: 重构 QnnpackUtils.h 文件，以便将量化操作所需的代码与通用的 qnnpack 特定量化工具分离

#include <ATen/native/quantized/cpu/QnnpackUtils.h>
// 包含 QNNPACK 的实用工具函数的头文件

#include <pack_block_sparse.h>
// 包含块稀疏打包相关的头文件

#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
// 包含稀疏量化参数打包的头文件

namespace ao {
namespace sparse {

struct TORCH_API PackedLinearWeightQnnp
    : public LinearPackedParamsBase {
  // 定义稀疏线性权重的结构体，并继承自线性打包参数的基类

  PackedLinearWeightQnnp(const at::Tensor& weight, const std::optional<at::Tensor>& bias, const int64_t out_features_block_size /* block sparsity size across output_features */, const int64_t in_features_block_size /* block sparsity size across input_features */);
  // 构造函数，接受权重张量、可选的偏置张量以及输出特征块大小和输入特征块大小作为参数

  explicit PackedLinearWeightQnnp(const BCSRSerializationType& serialized);
  // 显式构造函数，接受 BCSR 序列化类型作为参数

  std::optional<at::Tensor> orig_bias_;
  // 原始偏置张量的可选包装

  // 为了符合 qnnpack 操作符的预期，存在一个单独的偏置副本，当可选偏置不存在时，我们可以填充零。
  // 如果偏置存在，则 bias_ 是 orig_bias_ 的引用。
  at::Tensor bias_;

  c10::QScheme q_scheme_;
  // 量化方案枚举值

  double input_scale_;
  // 输入缩放比例

  std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix_;
  // BCSR 矩阵的唯一指针

  at::Tensor w_scales_;
  // 权重缩放张量

  std::vector<uint8_t> w_zero_points_;
  // 权重零点的向量

  std::vector<float> requantization_scales_;
  // 重量化比例的向量

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      sparse_linear_op_{nullptr};
  // 稀疏线性操作符的唯一指针，使用 QNNPACK 操作符删除器进行管理

  int64_t output_channels_;
  // 输出通道数

  int64_t input_channels_;
  // 输入通道数

  // 反序列化的张量被存储以维护基础 BCSR 数据的生命周期。
  // 如果 PackedLinearWeightQnnp 是通过预打包而不是反序列化创建的，则这些张量将为空。
  at::Tensor deserialized_bcsr_row_block_indices_;
  at::Tensor deserialized_bcsr_col_block_indices_;
  at::Tensor deserialized_bcsr_weight_values_;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    TORCH_CHECK(
        false, "Static quantized sparse linear unimplemented on QNNPACK");
    // 应用函数，用于将输入张量应用于稀疏线性操作
    // 抛出错误，因为 QNNPACK 上的静态量化稀疏线性操作尚未实现
  }

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    TORCH_CHECK(
        false, "Static quantized sparse linear unimplemented on QNNPACK");
    // 应用 ReLU 函数，用于将输入张量应用于稀疏线性操作，并加上 ReLU 激活
    // 抛出错误，因为 QNNPACK 上的静态量化稀疏线性操作尚未实现
  }

  at::Tensor apply_dynamic(const at::Tensor& input) override;
  // 应用动态量化稀疏线性操作的函数声明

  at::Tensor apply_dynamic_relu(const at::Tensor& input) override;
  // 应用动态量化稀疏线性操作并加上 ReLU 激活的函数声明

  LinearPackedSerializationType unpack() override;
  // 解包线性打包序列化类型的函数声明

  BCSRSerializationType serialize() override;
  // 序列化 BCSR 类型的函数声明

  static c10::intrusive_ptr<LinearPackedParamsBase> deserialize(
      const BCSRSerializationType& serialized);
  // 反序列化函数声明，返回线性打包参数基类的内部指针

  std::optional<at::Tensor> bias() override {
    // 返回可选偏置张量的函数声明
  // 返回存储的原始偏置值
  return orig_bias_;
}

// 预打包函数，将权重和偏置（可选）打包成线性层的参数
static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const int64_t out_features_block_size,
    const int64_t in_features_block_size);

private:
// 模板函数，用于实现应用线性变换（可融合ReLU操作）
template <bool ReluFused>
at::Tensor apply_impl(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point);

// 模板函数，用于实现应用动态线性变换（可融合ReLU操作）
template <bool ReluFused>
at::Tensor apply_dynamic_impl(const at::Tensor& input);
};

}}  // namespace ao::sparse

#endif // USE_PYTORCH_QNNPACK


注释：

// 结束 ao::sparse 命名空间的定义

#endif // 如果定义了 USE_PYTORCH_QNNPACK，则结束当前文件的条件编译
```