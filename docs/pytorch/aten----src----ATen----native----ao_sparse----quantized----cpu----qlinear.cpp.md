# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qlinear.cpp`

```py
// 定义宏，用于在头文件中只包含操作符方法
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 PyTorch 的 Tensor 类及其相关头文件
#include <ATen/core/Tensor.h>
// 包含 PyTorch 的并行处理头文件
#include <ATen/Parallel.h>
// 包含 PyTorch 的自定义类头文件
#include <torch/custom_class.h>
// 包含 PyTorch 的库头文件
#include <torch/library.h>

// 包含 AO Sparse 模块中量化 CPU 实现的 FBGEMM 相关头文件
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
// 包含 AO Sparse 模块中量化 CPU 实现的打包参数相关头文件
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
// 包含 C++ 标准库中的范围库头文件
#include <c10/util/irange.h>

// 根据宏定义选择性地包含不同的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#endif

// AO Sparse 命名空间
namespace ao {
namespace sparse {

// 注册线性参数函数声明
int register_linear_params();

#ifdef USE_FBGEMM

// 定义 PackedLinearWeight 类模板的成员函数 apply_impl，根据是否融合ReLU进行选择性编译
template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_impl(
    const at::Tensor& input,   // 输入张量
    double output_scale,       // 输出缩放因子
    int64_t output_zero_point) {  // 输出零点

  // 强调模型在不同机器上具有相同数值精度，如果不支持 FBGEMM 则终止运行
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  // TODO: 调用 contiguous 方法以便进行 JIT 优化
  auto input_contig = input.contiguous();
  // 获取输入数据的指针，假设输入类型为 quint8
  const auto* input_ptr =
      reinterpret_cast<uint8_t*>(input_contig.data_ptr<c10::quint8>());

  // 检查输入张量维度至少为2
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // 计算批次大小
  int64_t batch_size = size_to_dim_(input.dim() - 1, input.sizes());

  // 获取打包后的权重对象
  auto packW = w.get();

  // 获取输出通道数及输入特征数
  int64_t out_channels = static_cast<int64_t>(packW->R);
  int64_t K = input.size(input.dim() - 1);
  // 检查输入特征数与权重打包后的列数是否一致
  TORCH_CHECK(
      K == static_cast<int64_t>(packW->C),
      "The number of columns in the packW should be equal to K: " +
          std::to_string(K));

  // 获取输入的量化参数
  float input_scale_float = input.q_scale();
  int32_t input_zero_point_int32 = input.q_zero_point();

  // 初始化输出缩放因子和激活乘以权重缩放因子
  std::vector<float> output_multiplier_float(1, 0.0);
  std::vector<float> act_times_w_scale(1, 0.0);
  // 检查权重缩放因子和零点向量大小是否一致
  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");
  
  // 根据量化方案进行不同的量化计算
  if (q_scheme == c10::kPerTensorAffine) {
    // 处理每张量的量化方案
    act_times_w_scale[0] = (input_scale_float * w_scale[0]);
    output_multiplier_float[0] =
        act_times_w_scale[0] / static_cast<float>(output_scale);
  } else if (q_scheme == c10::kPerChannelAffine) {
    // 处理每通道的量化方案
    output_multiplier_float.resize(out_channels, 0.0);
    act_times_w_scale.resize(out_channels, 1.0f);
    for (const auto i : c10::irange(out_channels)) {
      act_times_w_scale[i] = (input_scale_float * w_scale[i]);
      output_multiplier_float[i] =
          act_times_w_scale[i] / static_cast<float>(output_scale);
  }
  // 关闭 bias_ 的空值检查，输出零点的整型转换
  }
  int32_t output_zero_point_int32 = static_cast<int32_t>(output_zero_point);

  const float* bias_ptr = nullptr;
  at::Tensor bias;
  // 如果存在偏置值，将其赋给 bias，并保证其连续性
  if (this->bias_.has_value()) {
    bias = this->bias_.value();
    bias = bias.contiguous();
    // 检查 bias 是否为一维向量（1D Tensor）
    TORCH_CHECK(bias.dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查 bias 是否包含 out_channels 个元素
    TORCH_CHECK(
        bias.size(0) == out_channels,
        "bias should have out_channels elements: " +
            std::to_string(out_channels));
    // 将 bias 数据的指针重新解释为 float 类型的指针
    bias_ptr = reinterpret_cast<float*>(bias.data_ptr<float>());
  }

  // 结果矩阵是二维的，按照输入的原始左手边维度来查看它。以下是两个示例：
  // 1. 如果输入张量为 {batch_size, K}，输出张量为 {batch_size, out_channels}。
  // 2. 如果输入张量为 {x, batch_size, K}，输出张量为 {x, batch_size, out_channels}。
  std::vector<int64_t> out_sizes = input.sizes().vec();
  // 将输出张量的最后一个维度设置为 out_channels
  out_sizes.back() = out_channels; // NOLINT
  // 分配输出张量和用于 fbgemmPacked 使用的缓冲区
  auto output_tr = at::_empty_affine_quantized(
      out_sizes,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  auto output = at::_empty_affine_quantized(
      out_sizes,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);

  auto buffer = at::empty(out_sizes, output.options().dtype(at::kInt));

  // fbgemm 内核计算如下：
  // C(output) = A(weight) x B(input)，其中 C, A, B 是 out_channels x batch_size,
  // out_channels x K, K x batch_size 的矩阵。因此需要对输入进行转置。
  auto input_tr = at::_empty_affine_quantized(
      input.sizes(),
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      input_scale_float,
      input_zero_point_int32);

  auto* input_tr_ptr =
      reinterpret_cast<uint8_t*>(input_tr.data_ptr<c10::quint8>());
  // TODO: 如果始终保持激活张量为转置状态，则可以删除内核前后的激活转置。
  // 使用 fbgemm::transpose_simd 对输入进行转置
  fbgemm::transpose_simd<uint8_t>(
      batch_size, K, input_ptr, K, input_tr_ptr, batch_size);

  // 获取线程数并并行处理任务
  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    // 遍历任务范围内的每个任务 ID
    for (const auto task_id : c10::irange(begin, end)) {
      // 设置量化后矩阵乘法的重量化参数
      fbgemm::trRequantizationParams_t reqParams = {
          input_zero_point_int32,    // 输入量化零点（整数）
          w_zp.data(),               // 权重量化零点数组的指针
          output_zero_point_int32,   // 输出量化零点（整数）
          static_cast<float>(output_scale),  // 输出的量化比例因子
          col_offsets.data(),        // 列偏移量数组的指针
          /*activation offsets*/ nullptr,   // 激活函数的偏移量（未使用，设为 nullptr）
          bias_ptr,                  // 偏置指针
          act_times_w_scale.data()}; // 激活函数乘以权重的比例因子数组的指针

      if (q_scheme == c10::kPerTensorAffine) {
        // 处理每张量的量化方案
        //
        // 在执行 uint8 * int8 矩阵乘法后，此操作执行：
        //  1) 将行和列偏移量加到相应的行和列中。
        //  2) 加入偏置项。

        // 执行矩阵乘法
        fbgemm::fbgemmSparseDenseInt8MM<
            ReluFused,
            fbgemm::QuantizationGranularity::TENSOR>(
            batch_size,
            w,
            input_tr_ptr,
            /*ldb=*/batch_size,
            /*C_i32=*/buffer.data_ptr<int32_t>(),
            /*C_u8=*/reinterpret_cast<uint8_t*>(output_tr.data_ptr<c10::quint8>()),
            /*ldc=*/batch_size,
            /*rParams=*/reqParams,
            /*accum=*/false,
            /*thread_id=*/task_id,
            /*num_threads=*/num_tasks);
      } else if (q_scheme == c10::kPerChannelAffine) {
        // 处理每通道的量化方案
        //
        // 在执行 uint8 * int8 矩阵乘法后，此操作执行：
        //  1) 将行和列偏移量加到相应的行和列中。
        //  2) 加入偏置项。

        // 执行矩阵乘法
        fbgemm::fbgemmSparseDenseInt8MM<
            ReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL>(
            batch_size,
            w,
            input_tr_ptr,
            /*ldb=*/batch_size,
            /*C_i32=*/buffer.data_ptr<int32_t>(),
            /*C_u8=*/reinterpret_cast<uint8_t*>(output_tr.data_ptr<c10::quint8>()),
            /*ldc=*/batch_size,
            /*rParams=*/reqParams,
            /*accum=*/false,
            /*thread_id=*/task_id,
            /*num_threads=*/num_tasks);
      }
    }
  });

  // 将 output_tr 转置回 batch_size x out_channels 的形状
  fbgemm::transpose_simd<uint8_t>(
      out_channels,
      batch_size,
      reinterpret_cast<uint8_t*>(output_tr.data_ptr<c10::quint8>()),
      batch_size,
      reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
      out_channels);

  // 返回输出结果
  return output;
}

// 定义 QLinearInt8 类模板，模板参数为 ReluFused
namespace {

template <bool ReluFused>
class QLinearInt8 final {
 public:
  // 静态方法 run，接受输入张量 input、线性参数 packed_weight、输出比例 output_scale 和输出零点 output_zero_point
  static at::Tensor run(
      const at::Tensor& input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    // 如果 ReluFused 为 true，则调用 packed_weight 的 apply_relu 方法
    if (ReluFused) {
      return packed_weight->apply_relu(input, output_scale, output_zero_point);
    } else {
      // 否则调用 packed_weight 的 apply 方法
      return packed_weight->apply(input, output_scale, output_zero_point);
    }
  }
};

// 实现 sparse 库的 QuantizedCPU 的 TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
  // 注册线性参数函数
  register_linear_params();
  // 实现 sparse::qlinear 的方法
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear"),
      TORCH_FN(QLinearInt8<false>::run));
  // 实现 sparse::qlinear_relu 的方法
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_relu"),
      TORCH_FN(QLinearInt8<true>::run));
}

} // namespace
}} // namespace ao::sparse
```