# `.\pytorch\torch\csrc\distributed\c10d\quantization\quantization.h`

```py
// 声明命名空间 torch::distributed::c10d::quantization，用于量化相关功能
namespace torch::distributed::c10d::quantization {

// 声明函数 _float_to_bfloat16_cpu，将输入张量从 float 转换为 bfloat16，运行在 CPU 上
at::Tensor _float_to_bfloat16_cpu(const at::Tensor& input);

// 声明函数 _bfloat16_to_float_cpu，将输入张量从 bfloat16 转换为 float，运行在 CPU 上
at::Tensor _bfloat16_to_float_cpu(const at::Tensor& input);

} // namespace torch::distributed::c10d::quantization
```