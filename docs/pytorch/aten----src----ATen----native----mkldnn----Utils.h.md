# `.\pytorch\aten\src\ATen\native\mkldnn\Utils.h`

```py
#pragma once

#include <ATen/Config.h>  // 包含 ATen 库的配置文件
#include <ATen/core/List.h>  // 包含 ATen 核心列表的头文件
#include <ATen/core/Tensor.h>  // 包含 ATen 核心张量的头文件
#include <c10/util/ArrayRef.h>  // 包含 c10 的 ArrayRef 实用工具头文件
#include <c10/util/strides.h>  // 包含 c10 的 strides 实用工具头文件

#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>  // 如果不是 s390x 和 powerpc 架构，则包含 cpuinfo.h 头文件
#endif

#include <vector>  // 包含 C++ 标准库的向量容器

#if AT_MKLDNN_ENABLED()
#include <ideep/tensor.hpp>  // 如果 MKLDNN 被启用，则包含 ideep 的 tensor 头文件
#endif // AT_MKLDNN_ENABLED()

namespace at { namespace native {

// 定义 mkldnn_layer_norm_last_index_weight_bias_f32 函数，用于执行 MKLDNN 加速的 Layer Normalization 操作
std::tuple<Tensor, Tensor, Tensor> mkldnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,  // 输入张量
    IntArrayRef normalized_shape,  // 规范化形状
    const Tensor& weight,  // 权重张量
    const Tensor& bias,  // 偏置张量
    double eps,  // epsilon 参数
    bool inplace = false);  // 是否原地操作，默认为 false

// 计算池化操作的输出尺寸
std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,  // 输入尺寸
    IntArrayRef kernel_size,  // 核大小
    IntArrayRef stride,  // 步幅
    IntArrayRef padding_l,  // 左填充
    IntArrayRef padding_r,  // 右填充
    IntArrayRef dilation,  // 膨胀系数
    bool ceil_mode);  // 是否使用向上取整模式

// 检查 MKLDNN 二元融合操作的输入张量
void check_mkldnn_binary_fusion_inputs(
    const Tensor& input,  // 输入张量
    const Tensor& other,  // 其他张量
    const Tensor& weight,  // 权重张量
    const Tensor& bias);  // 偏置张量

// 定义填充函数的右填充部分
inline std::vector<int64_t> padding_r(
    IntArrayRef padding,  // 填充
    IntArrayRef output_padding)  // 输出填充
{
  // ConvTranpose 填充调整
  //
  // PyTorch 使用 padding/output_padding:
  //   osize = (isize - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
  //
  // MKLDNN 使用 padding_l/padding_r:
  //   osize = (isize - 1) * stride - padding_l - padding_r + dilation * (kernel_size - 1) + 1
  //
  // 因此：padding_l = padding，padding_r = padding - output_padding
  //
  auto dim = padding.size();  // 获取填充的维度
  std::vector<int64_t> pad_r(dim);  // 创建存储右填充结果的向量
  for (const auto d : c10::irange(dim)) {  // 遍历填充的维度
    pad_r[d] = padding[d] - output_padding[d];  // 计算右填充
  }
  return pad_r;  // 返回右填充结果向量
}

// 如果输入张量是连续的，将其转换为默认连续步长以提高性能
inline Tensor may_convert_to_default_contiguous_strides(const Tensor& input) {
  auto input_size = input.sizes().vec();  // 获取输入张量的尺寸向量
  auto input_stride = input.strides().vec();  // 获取输入张量的步长向量
  auto input_default_contiguous_strides = c10::contiguous_strides(input_size);  // 计算默认连续步长
  if (input.is_contiguous() && input_stride != c10::IntArrayRef(input_default_contiguous_strides)) {
     return input.as_strided(input_size, input_default_contiguous_strides);  // 如果张量已经是连续的，则返回原张量；否则返回步长已被转换的张量
  }
  return input;  // 返回输入张量
}

#if AT_MKLDNN_ENABLED()

using AttrFunction = std::function<ideep::attr_t(
    torch::List<std::optional<at::Scalar>>,  // 包含可选标量的列表
    std::optional<c10::string_view>)>;  // 可选的字符串视图

// 获取 MKLDNN 一元融合操作的属性函数映射
const std::map<c10::string_view, AttrFunction>& fusion_unary_attr_map();

// 获取 MKLDNN 一元融合操作的算法映射
const std::map<c10::string_view, ideep::algorithm>& fusion_unary_alg_map();

// 获取 MKLDNN 二元融合操作的算法映射
const std::map<c10::string_view, ideep::algorithm>& fusion_binary_alg_map();

#endif // AT_MKLDNN_ENABLED()

};  // 结束 native 命名空间

#if defined(__aarch64__)
// 检查是否支持 ARM 架构的 MKLDNN BF16 加速
inline bool mkldnn_bf16_device_check_arm() {
  return cpuinfo_initialize() && cpuinfo_has_arm_bf16();  // 初始化 CPU 信息并检查 ARM BF16 支持情况
}
#else
// 对于非 ARM 架构，始终返回 false
constexpr bool mkldnn_bf16_device_check_arm() {
  return false;  // 返回 false
}
#endif
#if AT_MKLDNN_ENABLED()
// 如果 MKLDNN 可用

inline bool mkldnn_bf16_device_check() {
// 内联函数，检查是否支持 MKLDNN 的 BF16 数据类型

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
  // 如果是 x86_64 架构或者 _M_X64 并且不是 ARM64EC 架构
  // 使用 ideep 来检查 BF16 是否支持，因为 cpuinfo 没有 avx_ne_convert 检查
  return ideep::has_bf16_type_support();
#else
  // 否则调用 ARM 平台的 BF16 设备检查函数
  return mkldnn_bf16_device_check_arm();
#endif
}

inline bool mkldnn_fp16_device_check() {
// 内联函数，检查是否支持 MKLDNN 的 FP16 数据类型

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
  // 如果是 x86_64 架构或者 _M_X64 并且不是 ARM64EC 架构
  return ideep::has_fp16_type_support();
#else
  // 否则返回 false
  return false;
#endif
}

#else
// 如果 MKLDNN 不可用，则以下函数返回 false

inline bool mkldnn_bf16_device_check() {
  return false;
}

inline bool mkldnn_fp16_device_check() {
  return false;
}
#endif

inline void mkldnn_check_low_precision(ScalarType input_t, std::string name) {
// 内联函数，检查低精度设置

  if (input_t == ScalarType::BFloat16) {
    // 如果输入类型是 BFloat16
    TORCH_CHECK(
        mkldnn_bf16_device_check(),
        name,
        ": bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq");
    // 使用 TORCH_CHECK 检查是否支持 BF16，否则抛出异常
  } else if (input_t == ScalarType::Half) {
    // 如果输入类型是 Half
    TORCH_CHECK(
        mkldnn_fp16_device_check(),
        name,
        ": fp16 path needs the cpu support avx_ne_convert or avx512_fp16");
    // 使用 TORCH_CHECK 检查是否支持 FP16，否则抛出异常
  }
}
```