# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qlinear_unpack.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <torch/custom_class.h>

#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#endif

namespace ao {
namespace sparse {

// 注册线性参数函数声明
int register_linear_params();

#ifdef USE_FBGEMM

// 解包线性权重的函数
LinearPackedSerializationType PackedLinearWeight::unpack() {
  auto packW = w.get();  // 获取指向封装的权重的指针

  const int64_t N = static_cast<int64_t>(packW->R);  // 输出通道数
  const int64_t K = static_cast<int64_t>(packW->C);  // 输入通道数

  at::Tensor weight_origin;  // 存储原始权重的张量
  if (q_scheme == c10::kPerTensorAffine) {
    // 创建一个仿射量化的空张量
    weight_origin = at::_empty_affine_quantized(
        {N, K}, at::device(c10::kCPU).dtype(c10::kQInt8), w_scale[0], w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    // 创建一个通道仿射量化的空张量，并设置每个通道的标度和零点
    at::Tensor scales = at::empty(
        {static_cast<long>(w_scale.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(w_scale.begin(), w_scale.end(), scales.mutable_data_ptr<float>());

    at::Tensor zero_points = at::empty(
        {static_cast<long>(w_zp.size())},
        at::device(c10::kCPU).dtype(c10::kInt));
    std::copy(w_zp.begin(), w_zp.end(), zero_points.mutable_data_ptr<int>());

    // 创建通道仿射量化的空张量
    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales,
        zero_points,
        0, // 输出通道轴为0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  // 获取原始权重数据的 int8 指针并解包到 packW 中
  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());
  packW->unpack(weight_ptr_int8);

  const std::vector<int64_t> block_pattern(
      {out_features_block_size_, in_features_block_size_});

  // 返回包含权重张量、偏置和块模式的元组
  return std::make_tuple(std::move(weight_origin), bias_, block_pattern);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

// QNNPACK 解包线性权重的函数
LinearPackedSerializationType PackedLinearWeightQnnp::unpack() {
  const int64_t N = static_cast<int64_t>(output_channels_);  // 输出通道数
  const int64_t K = static_cast<int64_t>(input_channels_);   // 输入通道数

  float* w_scales_ptr = w_scales_.data_ptr<float>();  // 权重标度的指针

  at::Tensor weight_origin;  // 存储原始权重的张量
  if (q_scheme_ == c10::kPerTensorAffine) {
    // 创建一个仿射量化的空张量
    weight_origin = at::_empty_affine_quantized(
        {N, K},
        at::device(c10::kCPU).dtype(c10::kQInt8),
        w_scales_ptr[0],
        w_zero_points_[0] - 128);
  } else if (q_scheme_ == c10::kPerChannelAffine) {
    // 创建一个通道仿射量化的空张量，并设置每个通道的标度和零点
    at::Tensor scales = at::empty(
        {static_cast<long>(output_channels_)},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        w_scales_ptr,
        w_scales_ptr + output_channels_,
        scales.mutable_data_ptr<float>());

    at::Tensor zero_points = at::empty(
        {static_cast<long>(output_channels_)},
        at::device(c10::kCPU).dtype(c10::kInt));



**继续完整的注释，以保证代码的完整性。**
    // 使用 std::transform 函数对 w_zero_points_ 中的每个元素进行转换
    std::transform(
        w_zero_points_.begin(),
        w_zero_points_.begin() + output_channels_,
        zero_points.mutable_data_ptr<int>(),
        [](uint8_t v) { return static_cast<int>(v) - 128; });

    // 创建量化权重张量 weight_origin，采用每通道的仿射量化
    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales,
        zero_points,
        0, // 输出通道轴为 0
        device(c10::kCPU).dtype(c10::kQInt8));

  }

  // 将 weight_origin 的数据指针转换为 int8_t 类型的指针
  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  // 使用 bcsr_matrix_ 对象的 unpack 方法，解包权重数据
  bcsr_matrix_->unpack(
      weight_ptr_int8,
      output_channels_,
      input_channels_,
      w_zero_points_.data());

  // 创建并初始化 block_pattern 向量，包含 out_features_block_size_ 和 in_features_block_size_ 两个元素
  std::vector<int64_t> block_pattern(
      {out_features_block_size_, in_features_block_size_});

  // 返回一个 tuple，包含 weight_origin（作为右值引用）、bias_ 和 block_pattern（作为右值引用）
  return std::make_tuple(
      std::move(weight_origin), bias_, std::move(block_pattern));
}

#endif // USE_FBGEMM


// 结束条件编译指令，用于检查是否定义了 USE_FBGEMM 宏
#endif // USE_FBGEMM

namespace {


// 创建匿名命名空间，限制其中定义的类、函数、变量只在当前文件可见
class QLinearUnpackWeightInt8 final {
 public:
  // 静态方法，接收一个 LinearPackedParamsBase 类型的智能指针参数，返回 LinearPackedSerializationType 类型
  static LinearPackedSerializationType run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    // 调用 packed_weight 智能指针指向对象的 unpack() 方法，返回解压缩后的数据
    return packed_weight->unpack();
  }
};


// 实现 Torch 库中 sparse 命名空间的扩展
TORCH_LIBRARY_IMPL(sparse, CatchAll, m) {
  // 注册线性参数的实现函数
  register_linear_params();
  // 注册 sparse::qlinear_unpack 的实现函数为 QLinearUnpackWeightInt8::run
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_unpack"),
      TORCH_FN(QLinearUnpackWeightInt8::run));
}


}  // namespace


}}  // namespace ao::sparse



// 结束匿名命名空间，并结束 ao::sparse 命名空间
```