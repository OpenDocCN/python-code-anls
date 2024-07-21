# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\fbgemm_utils.h`

```
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/QScheme.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmSparse.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>

namespace ao {
namespace sparse {

// 定义稀疏量化线性层的权重打包结构体，继承自LinearPackedParamsBase
struct TORCH_API PackedLinearWeight
    : public LinearPackedParamsBase {
  // 构造函数，初始化稀疏矩阵、偏置、列偏移、量化参数等
  PackedLinearWeight(std::unique_ptr<fbgemm::BCSRMatrix<int8_t>> w,
                     std::optional<at::Tensor> bias,
                     std::vector<int32_t> col_offsets,
                     std::vector<float> w_scale,
                     std::vector<int32_t> w_zp,
                     c10::QScheme q_scheme,
                     const int64_t out_features_block_size /* block sparsity size across output_features */,
                     const int64_t in_features_block_size /* block sparsity size across input_features */)
      : LinearPackedParamsBase(
            out_features_block_size,
            in_features_block_size),
        w(std::move(w)),
        bias_(std::move(bias)),
        col_offsets(std::move(col_offsets)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(q_scheme) {}

  // 稀疏矩阵的唯一指针，存储稀疏权重
  std::unique_ptr<fbgemm::BCSRMatrix<int8_t>> w;
  // 可选的偏置张量
  std::optional<at::Tensor> bias_;
  // 列偏移，用于稀疏矩阵计算
  std::vector<int32_t> col_offsets;
  // 权重缩放因子
  std::vector<float> w_scale;
  // 权重零点
  std::vector<int32_t> w_zp;
  // 量化方案
  c10::QScheme q_scheme;

  // 应用线性变换，支持输出缩放和零点偏移
  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用带ReLU的线性变换，支持输出缩放和零点偏移
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用动态量化线性变换，当前不支持QNNPACK后端
  at::Tensor apply_dynamic(const at::Tensor& input) override {
    TORCH_INTERNAL_ASSERT(
        false,
        "Sparse quantized dynamic linear with fused relu is not yet "
        "supported on qnnpack backend.");
    return at::Tensor();
  }

  // 应用带ReLU的动态量化线性变换，当前不支持QNNPACK后端
  at::Tensor apply_dynamic_relu(const at::Tensor& input) override {
    TORCH_INTERNAL_ASSERT(
        false,
        "Sparse quantized dynamic linear with fused relu is not yet "
        "supported on qnnpack backend.");
    return at::Tensor();
  }

  // 解压缩稀疏矩阵的序列化类型
  LinearPackedSerializationType unpack() override;

  // 序列化稀疏矩阵的BCSR格式
  BCSRSerializationType serialize() override;

  // 反序列化稀疏线性打包参数
  static c10::intrusive_ptr<LinearPackedParamsBase> deserialize(
      const BCSRSerializationType& serialized);

  // 获取偏置张量的可选接口
  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  // 预打包稀疏线性参数，包括权重、偏置以及块稀疏大小
  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      const at::Tensor& weight,
      const std::optional<at::Tensor>& bias,
      const int64_t out_features_block_size,
      const int64_t in_features_block_size);

 private:
  // 实现带ReLU的线性变换
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
};

}}  // namespace ao::sparse

#endif // USE_FBGEMM

// 注册稀疏线性参数
namespace ao {
namespace sparse {
int register_linear_params();
}}  // namespace ao::sparse
```