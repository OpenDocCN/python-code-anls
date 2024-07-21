# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qlinear_prepack.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 Tensor 头文件，包括 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>
// 引入 ATen 库的上下文定义
#include <ATen/Context.h>
// 引入范围遍历工具
#include <c10/util/irange.h>
// 引入 PyTorch 自定义类支持的头文件
#include <torch/custom_class.h>

// 引入量化的初始化函数和工具函数
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

// 根据不同条件引入不同的 ATen 函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/zeros.h>
#endif

// 引入标准库中的算法实现
#include <algorithm>

// 定义 ao 命名空间和 sparse 子命名空间
namespace ao {
namespace sparse {

// 注册线性参数的函数声明
int register_linear_params();

#ifdef USE_FBGEMM
// 匿名命名空间，用于局部函数或变量的定义
namespace {
// 计算转置时的列偏移量
// 注意这包括列的总和以及标量项 B_zero_point * K，
// 而激活的行偏移仅包括 A 行的总和。
void calc_col_offsets_transpose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t* B_zero_point,
    int32_t* col_offsets,
    c10::QScheme qtype) {
  // 遍历列索引
  for (const auto i : c10::irange(N)) {
    int32_t sum = 0;
    // 遍历行索引
    for (const auto j : c10::irange(K)) {
      sum += Bint8[i * K + j];
    }
    // 根据量化类型计算列偏移量
    if (qtype == c10::kPerTensorAffine) {
      col_offsets[i] = sum - B_zero_point[0] * K;
    } else {
      col_offsets[i] = sum - B_zero_point[i] * K;
    }
  }
}
} // namespace

// 实现线性权重预打包函数
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeight::
    prepack(
        const at::Tensor& weight,
        const std::optional<at::Tensor>& bias,
        const int64_t out_features_block_size,
        const int64_t in_features_block_size) {
  // 检查权重张量的维度是否为 2
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for ao::sparse::qlinear_prepack (fbgemm) should"
      " be 2-dimensional.");

  // 检查输出和输入特征块大小是否符合预期
  TORCH_CHECK(
      out_features_block_size == 1 && in_features_block_size == 4,
      "The out and in features block sizes for ao::sparse::qlinear_prepack",
      " (fbgemm) should be 1 and 4 respectively (got ", out_features_block_size,
      " and ", in_features_block_size, ")");

  // 获取权重张量的维度信息
  auto N = weight.size(0);
  auto K = weight.size(1);

  // 创建连续的权重张量副本
  auto weight_contig = weight.contiguous();
  // 获取量化类型
  const auto qtype = weight.qscheme();
  // 初始化权重零点的整型向量
  std::vector<int32_t> weight_zero_points_int32(1, 0);
  // 根据量化类型设置权重零点值
  if (qtype == c10::kPerTensorAffine) {
    weight_zero_points_int32[0] = weight.q_zero_point();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_zero_points_int32.resize(N, 0);
    for (const auto i : c10::irange(N)) {
      weight_zero_points_int32[i] =
          weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  }
  // 检查权重零点值是否为 0
  TORCH_CHECK(
      std::all_of(
          weight_zero_points_int32.cbegin(),
          weight_zero_points_int32.cend(),
          [](int32_t i) { return i == 0; }),
      "zero point(s) should be 0 for the weight tensor of ao::sparse::qlinear op");
  
  // 初始化权重缩放因子的浮点向量
  std::vector<float> weight_scales_float(1, 0.0);
  // 根据量化类型设置权重缩放因子
  if (qtype == c10::kPerTensorAffine) {
    // 如果量化类型为整体量化
    weight_scales_float[0] = weight.q_scale();
  } else if (qtype == c10::kPerChannelAffine) {
    // 如果量化类型为通道间仿射量化，初始化长度为N的权重比例向量
    weight_scales_float.resize(N, 0.0);
    // 遍历每个通道，将其对应的量化比例转换为浮点数并存储在weight_scales_float中
    for (const auto i : c10::irange(N)) {
      weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }

  // 将权重张量转换为int8类型的指针
  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

  // 计算转置后的列偏移量
  std::vector<int32_t> col_offsets(N);
  calc_col_offsets_transpose(
      /*K=*/K,
      /*N=*/N,
      /*Bint8=*/weight_ptr_int8,
      /*B_zero_point=*/weight_zero_points_int32.data(),
      /*col_offsets=*/col_offsets.data(),
      /*qtype=*/qtype);

  // 如果存在偏置项，则对其进行处理
  std::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    const at::Tensor& bias_vec = bias.value();
    // 检查偏置项是一个向量（1维张量）
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查偏置项的长度与N相等
    TORCH_CHECK(
        bias_vec.size(0) == N,
        "bias should have N elements: " + std::to_string(N));
    // 将偏置项转换为连续的张量
    bias_contig = bias->contiguous();
  }

  // 将权重张量转换为BCSR格式
  auto bcsr = fbgemm::fbgemmDenseToBCSR<int8_t>(N, K, weight_ptr_int8);
  // 创建PackedLinearWeight对象，并返回其指针
  auto ret_ptr = c10::make_intrusive<PackedLinearWeight>(
      std::move(bcsr),
      bias_contig,
      col_offsets,
      weight_scales_float,
      weight_zero_points_int32,
      qtype,
      out_features_block_size,
      in_features_block_size);
  return ret_ptr;
#ifdef USE_PYTORCH_QNNPACK
// 如果使用了 PyTorch QNNPACK，定义一个函数 prepack，用于预打包线性层参数
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightQnnp::
    prepack(
        const at::Tensor& weight,
        const std::optional<at::Tensor>& bias,
        const int64_t out_features_block_size,
        const int64_t in_features_block_size) {
  // 初始化 QNNPACK，确保 QNNPACK 被正确初始化
  at::native::initQNNPACK();
  // 创建并返回一个 PackedLinearWeightQnnp 对象
  return c10::make_intrusive<PackedLinearWeightQnnp>(
      weight, bias, out_features_block_size, in_features_block_size);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 定义 PackedLinearWeightQnnp 构造函数，初始化线性层打包参数
PackedLinearWeightQnnp::PackedLinearWeightQnnp(
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const int64_t out_features_block_size,
    const int64_t in_features_block_size)
    : LinearPackedParamsBase(out_features_block_size, in_features_block_size),
      orig_bias_(bias),
      q_scheme_(weight.qscheme()),
      output_channels_(weight.size(0)),
      input_channels_(weight.size(1)) {
  // 检查权重张量的维度必须为 2
  TORCH_CHECK(
      weight.dim() == 2,
      "ao::sparse::qlinear (qnnpack): Weight tensor rank should be == 2");
  // 检查行块大小必须大于 0
  TORCH_CHECK(out_features_block_size > 0, "Row block size must be > 0.");
  TORCH_CHECK(in_features_block_size > 0, "Row block size must be > 0.");

  // 如果提供了偏置，则使用给定的偏置，否则创建一个与输出通道数相同的零张量
  if (bias.has_value()) {
    bias_ = bias.value();
  } else {
    bias_ = at::zeros(output_channels_, weight.options().dtype(at::kFloat));
  }
  // 检查偏置张量维度必须为 1，且大小与输出通道数相同
  TORCH_CHECK(
      (bias_.ndimension() == 1 && bias_.size(0) == output_channels_),
      "ao::sparse::qlinear_prepack (qnnpack): Given weight of size ",
      weight.sizes(),
      ", expected bias to be 1-dimensional with ",
      output_channels_,
      " elements",
      ", but got bias of size ",
      bias_.sizes(),
      " instead");

  // 确保权重张量是连续的
  at::Tensor weight_contig = weight.contiguous();

  // 生成权重的零点和缩放因子张量
  std::tie(w_zero_points_, w_scales_) =
      make_zero_points_and_scales_tensor(weight_contig);
  const float* weight_scales_data = w_scales_.const_data_ptr<float>();

  // 创建 QNNPACK 权重张量，进行量化
  at::Tensor qnnp_weight = at::_empty_affine_quantized(
      weight_contig.sizes(),
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      weight_scales_data[0],
      w_zero_points_[0]);
  auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
  auto wt_numel = weight_contig.numel();
  int8_t* w_data =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

  // 将权重数据从 qint8 转换为 quint8，同时偏置为 128
  for (const auto i : c10::irange(wt_numel)) {
    qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
  }

  // 生成块压缩稀疏行（BCSR）矩阵
  bcsr_matrix_ = qnnpack::generateBlockCSRMatrix<uint32_t>(
      reinterpret_cast<uint8_t*>(qnnp_w_data),
      output_channels_,
      input_channels_,
      out_features_block_size,
      in_features_block_size,
      w_zero_points_.data());
}
#endif // USE_PYTORCH_QNNPACK
# 定义了一个名为 QLinearPackWeightInt8 的 C++ 类，用于量化线性层的权重打包
class QLinearPackWeightInt8 final {
 public:
  # 静态方法，用于运行量化线性层权重的打包操作
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      const at::Tensor& weight,
      const std::optional<at::Tensor>& bias,
      const int64_t out_features_block_size,
      const int64_t in_features_block_size) {
    # 获取当前的全局上下文
    auto& ctx = at::globalContext();

    # 如果使用 FBGEMM 引擎
#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      # 调用 PackedLinearWeight 的预打包方法，返回打包后的线性参数对象
      return PackedLinearWeight::prepack(
          weight, bias, out_features_block_size, in_features_block_size);
    }
#endif

    # 如果使用 QNNPACK 引擎
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      # 调用 PackedLinearWeightQnnp 的预打包方法，返回打包后的线性参数对象
      return PackedLinearWeightQnnp::prepack(
          weight, bias, out_features_block_size, in_features_block_size);
    }
#endif

    # 若未找到适用的引擎，抛出错误信息
    TORCH_CHECK(
        false,
        "Didn't find engine for operation ao::sparse::qlinear_prepack ",
        toString(ctx.qEngine()));
  }
};

# 在 sparse 命名空间内注册了 QuantizedCPU 实现的 sparse::qlinear_prepack 方法
TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
  # 注册线性层参数
  register_linear_params();
  # 将 sparse::qlinear_prepack 方法实现为 QLinearPackWeightInt8 类的 run 静态方法
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_prepack"),
      TORCH_FN(QLinearPackWeightInt8::run));
}
}  // namespace ao::sparse
```