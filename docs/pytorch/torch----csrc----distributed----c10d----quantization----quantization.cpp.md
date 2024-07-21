# `.\pytorch\torch\csrc\distributed\c10d\quantization\quantization.cpp`

```py
// 包含头文件：Torch的分布式量化模块的相关头文件
#include <torch/csrc/distributed/c10d/quantization/quantization.h>
#include <torch/csrc/distributed/c10d/quantization/quantization_utils.h>
#include <torch/library.h>

// 命名空间定义：torch::distributed::c10d::quantization
namespace torch::distributed::c10d::quantization {

// 静态函数定义：将float类型数据转换为BFloat16类型数据的参考实现
static void FloatToBFloat16Quantized_ref(
    const float* const input, // 输入的float数据指针
    const size_t nrows,       // 输入矩阵的行数
    const size_t ncols,       // 输入矩阵的列数
    uint16_t* const output) { // 输出的BFloat16数据指针
  // 遍历每一行
  for (const auto row : c10::irange(nrows)) {
    const float* input_row = input + row * ncols;     // 当前行的输入数据指针
    uint16_t* output_row = output + row * ncols;      // 当前行的输出数据指针

    // 遍历当前行的每一列
    for (const auto col : c10::irange(ncols)) {
      // 将float数据强制转换为uint32_t，然后将其整体右移16位，得到BFloat16数据
      output_row[col] =
          (*reinterpret_cast<const uint32_t*>(input_row + col) + (1 << 15)) >>
          16;
    }
  }
}

// 静态函数定义：将BFloat16类型数据转换为float类型数据的参考实现
static void BFloat16QuantizedToFloat_ref(
    const at::BFloat16* const input, // 输入的BFloat16数据指针
    const size_t nrows,              // 输入矩阵的行数
    const size_t ncols,              // 输入矩阵的列数
    float* const output) {           // 输出的float数据指针
  // 遍历每一行
  for (const auto row : c10::irange(nrows)) {
    const at::BFloat16* input_row = input + row * ncols; // 当前行的输入数据指针
    float* output_row = output + row * ncols;            // 当前行的输出数据指针

    // 遍历当前行的每一列
    for (const auto col : c10::irange(ncols)) {
      // 将BFloat16数据强制转换为uint16_t，然后左移16位，得到float数据
      uint32_t val_fp32 = static_cast<uint32_t>(
                              reinterpret_cast<const uint16_t*>(input_row)[col])
          << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32; // 存储为float数据
    }
  }
}

// 函数定义：将输入的Tensor对象转换为BFloat16类型的Tensor对象
at::Tensor _float_to_bfloat16_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input); // 断言输入Tensor在CPU上
  // 断言输入Tensor是二维的
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes(); // 获取输入Tensor的尺寸
  const auto nrows = input_sizes[0];      // 获取行数
  const auto ncols = input_sizes[1];      // 获取列数
  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kHalf)); // 创建相同尺寸的输出Tensor，数据类型为BFloat16

  // 调用FloatToBFloat16Quantized_ref函数，进行数据转换
  FloatToBFloat16Quantized_ref(
      input.const_data_ptr<float>(), // 输入Tensor的float数据指针
      nrows,
      ncols,
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr<at::Half>())); // 输出Tensor的BFloat16数据指针

  return output; // 返回转换后的Tensor对象
}

// 函数定义：将输入的BFloat16类型的Tensor对象转换为float类型的Tensor对象
at::Tensor _bfloat16_to_float_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input); // 断言输入Tensor在CPU上
  // 断言输入Tensor是二维的
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes(); // 获取输入Tensor的尺寸
  const auto nrows = input_sizes[0];      // 获取行数
  const auto ncols = input_sizes[1];      // 获取列数

  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kFloat)); // 创建相同尺寸的输出Tensor，数据类型为float

  // 调用BFloat16QuantizedToFloat_ref函数，进行数据转换
  BFloat16QuantizedToFloat_ref(
      reinterpret_cast<const at::BFloat16*>(input.const_data_ptr<at::Half>()), // 输入Tensor的BFloat16数据指针
      nrows,
      ncols,
      output.mutable_data_ptr<float>()); // 输出Tensor的float数据指针

  return output; // 返回转换后的Tensor对象
}

// Torch库注册：注册BFloat16到float转换的函数
TORCH_LIBRARY(quantization, m) {
  m.def("_Bfloat16QuantizedToFloat(Tensor input) -> Tensor"); // 定义BFloat16到float转换的接口
  m.def("_FloatToBfloat16Quantized(Tensor input) -> Tensor"); // 定义float到BFloat16转换的接口
}

// Torch库实现注册：注册CPU环境下具体的转换函数实现
TORCH_LIBRARY_IMPL(quantization, CPU, m) {
  m.impl("_Bfloat16QuantizedToFloat", _bfloat16_to_float_cpu); // 实现BFloat16到float转换的具体函数
  m.impl("_FloatToBfloat16Quantized", _float_to_bfloat16_cpu); // 实现float到BFloat16转换的具体函数
}

} // namespace torch::distributed::c10d::quantization
```