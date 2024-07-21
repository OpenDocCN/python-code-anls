# `.\pytorch\aten\src\ATen\native\ao_sparse\library.cpp`

```
// 定义预处理宏，用于声明仅支持方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 Torch 库的头文件
#include <torch/library.h>

// 包含自定义类和相关的头文件
#include <torch/custom_class.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>

// 注册稀疏运算符
TORCH_LIBRARY(sparse, m) {
  // 注册稀疏线性参数
  ao::sparse::register_linear_params();

  // 定义稀疏量化线性运算的原型
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  // 定义带 ReLU 的稀疏量化线性运算的原型
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_relu(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));

  // 定义动态量化的稀疏线性运算的原型
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));
  // 定义带 ReLU 的动态量化稀疏线性运算的原型
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_relu_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));

  // 定义稀疏线性参数预打包函数的原型
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_prepack(Tensor W, Tensor? B, int out_features_block_size, int in_features_block_size) -> __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack"));

  // 定义稀疏线性参数解包函数的原型
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_unpack(__torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin, int[] block_pattern)"));
}
```