# `.\pytorch\aten\src\ATen\native\quantized\cpu\qlinear_prepack.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_saturate_weight_to_fp16.h>
#include <ATen/ops/_saturate_weight_to_fp16_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <utility>
#include <vector>

// 声明一个函数 register_linear_params，返回类型为 int
int register_linear_params();

#ifdef USE_FBGEMM
namespace {
// 计算列偏移量的函数
// 注意这里包括了列的总和以及标量项 B_zero_point * K，
// 而 PackAWithQuantRowOffset 创建的行偏移量只包括 A 行的总和。
void calc_col_offsets_transpose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t* B_zero_point,
    int32_t* col_offsets,
    c10::QScheme qtype) {
  // 遍历列数 N
  for (const auto i : c10::irange(N)) {
    int32_t sum = 0;
    // 遍历行数 K
    for (const auto j : c10::irange(K)) {
      sum += Bint8[i * K + j];  // 计算第 i 列的总和
    }
    // 根据量化类型 qtype 计算列偏移量
    if (qtype == c10::kPerTensorAffine) {
      col_offsets[i] = sum - B_zero_point[0] * K;
    } else {
      col_offsets[i] = sum - B_zero_point[i] * K;
    }
  }
}
} // namespace

// 实现 PackedLinearWeight 类的 prepack 函数
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeight::prepack(
    at::Tensor weight,
    std::optional<at::Tensor> bias) {
  // 检查权重张量维度是否为 2
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack (fbgemm) should"
      " be 2-dimensional.");

  auto N = weight.size(0);  // 获取权重张量的行数
  auto K = weight.size(1);  // 获取权重张量的列数

  // TODO: 为了进一步的 JIT 优化，需要保证权重张量是连续的。
  auto weight_contig = weight.contiguous();
  const auto qtype = weight.qscheme();  // 获取量化方案类型
  std::vector<int32_t> weight_zero_points_int32(1, 0);  // 初始化权重零点的整型向量
  if (qtype == c10::kPerTensorAffine) {
    weight_zero_points_int32[0] = weight.q_zero_point();  // 设置权重的零点值
  } else if (qtype == c10::kPerChannelAffine) {
    // 对于每个通道的量化，设置权重的每个通道零点值
    weight_zero_points_int32.resize(N, 0);
    for (const auto i : c10::irange(N)) {
      weight_zero_points_int32[i] =
          weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  }
  std::vector<float> weight_scales_float(1, 0.0);  // 初始化权重的浮点比例因子向量
  if (qtype == c10::kPerTensorAffine) {
    weight_scales_float[0] = weight.q_scale();  // 设置权重的量化比例因子
  } else if (qtype == c10::kPerChannelAffine) {
    // 对于每个通道的量化，设置权重的每个通道比例因子
    weight_scales_float.resize(N, 0.0);
    for (const auto i : c10::irange(N)) {
      weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }
  }
}

int8_t* weight_ptr_int8 =
    reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

std::vector<int32_t> col_offsets(N);
calc_col_offsets_transpose(
    /*K=*/K,
    /*N=*/N,
    /*Bint8=*/weight_ptr_int8,
    /*B_zero_point=*/weight_zero_points_int32.data(),
    /*col_offsets=*/col_offsets.data(),
    /*qtype=*/qtype);

std::optional<at::Tensor> bias_contig;
if (bias.has_value()) {
  at::Tensor bias_vec = bias.value();
  TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
  TORCH_CHECK(
      bias_vec.size(0) == N,
      "bias should have N elements: " + std::to_string(N));
  bias_contig = bias->contiguous();
}

auto ret_ptr = c10::make_intrusive<PackedLinearWeight>(
    std::make_unique<fbgemm::PackBMatrix<int8_t>>(
        /*trans=*/fbgemm::matrix_op_t::Transpose,
        /*nRow=*/K,
        /*nCol=*/N,
        /*smat=*/weight_ptr_int8,
        /*ld=*/K,
        /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
        /*groups=*/1),
    bias_contig,
    col_offsets,
    weight_scales_float,
    weight_zero_points_int32,
    qtype);
return ret_ptr;


注释：


// 结束函数体，关闭所有的大括号，表示代码块的结束
  }
}

// 将 weight_contig 的数据指针解释为 int8_t 类型的指针
int8_t* weight_ptr_int8 =
    reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

// 创建一个大小为 N 的整数向量 col_offsets，用于存储计算后的列偏移量
std::vector<int32_t> col_offsets(N);
// 调用 calc_col_offsets_transpose 函数，计算权重矩阵的转置，填充 col_offsets
calc_col_offsets_transpose(
    /*K=*/K,  // 矩阵的行数 K
    /*N=*/N,  // 矩阵的列数 N
    /*Bint8=*/weight_ptr_int8,  // 指向权重矩阵的 int8_t 类型指针
    /*B_zero_point=*/weight_zero_points_int32.data(),  // 权重矩阵的零点数组
    /*col_offsets=*/col_offsets.data(),  // 存储列偏移量的数组
    /*qtype=*/qtype);  // 权重的量化类型

// 创建一个可选类型的 Tensor 对象 bias_contig
std::optional<at::Tensor> bias_contig;
// 如果 bias 有值
if (bias.has_value()) {
  // 将 bias 的值赋给 bias_vec
  at::Tensor bias_vec = bias.value();
  // 检查 bias_vec 的维度是否为 1
  TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
  // 检查 bias_vec 的大小是否为 N
  TORCH_CHECK(
      bias_vec.size(0) == N,
      "bias should have N elements: " + std::to_string(N));
  // 如果通过检查，则使 bias_contig 指向 bias 的连续版本
  bias_contig = bias->contiguous();
}

// 创建一个 PackedLinearWeight 对象的智能指针 ret_ptr
auto ret_ptr = c10::make_intrusive<PackedLinearWeight>(
    // 创建一个 int8_t 类型的 PackBMatrix 对象，用于包装权重矩阵的转置
    std::make_unique<fbgemm::PackBMatrix<int8_t>>(
        /*trans=*/fbgemm::matrix_op_t::Transpose,  // 矩阵的转置操作
        /*nRow=*/K,  // 矩阵的行数 K
        /*nCol=*/N,  // 矩阵的列数 N
        /*smat=*/weight_ptr_int8,  // 指向权重矩阵的 int8_t 类型指针
        /*ld=*/K,  // 矩阵的 leading dimension
        /*pmat=*/nullptr, // PackBMatrix 管理 pmat 的所有权，因此传入 nullptr
        /*groups=*/1),  // 矩阵的组数，默认为 1 组
    bias_contig,  // 权重矩阵的偏置项
    col_offsets,  // 存储列偏移量的数组
    weight_scales_float,  // 权重矩阵的浮点类型的缩放因子
    weight_zero_points_int32,  // 权重矩阵的零点数组
    qtype);  // 权重的量化类型
// 返回指向 PackedLinearWeight 对象的智能指针 ret_ptr
return ret_ptr;
#ifdef USE_FBGEMM
// 如果使用 FBGEMM，定义一个返回 LinearPackedParamsBase 指针的方法 prepack
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightFp16::prepack(
    at::Tensor weight,  // 输入权重张量
    std::optional<at::Tensor> bias) {  // 可选的偏置张量

  // 将输入的权重张量转换为半精度浮点数
  weight = at::_saturate_weight_to_fp16(weight);

  // 获取权重张量的维度信息
  const int64_t K = weight.size(1);  // 列数
  const int64_t N = weight.size(0);  // 行数

  // 将权重张量进行内存连续化
  at::Tensor weight_contig = weight.contiguous();
  // 获取权重张量连续化后的数据指针
  float* weight_contig_ptr = weight_contig.data_ptr<float>();

  // TODO(mingzhe09088):
  // 考虑在 PackedGemmMatrixFP16 中使用一个函数对象
  // (XQ) 的评论：我不确定这里使用 make_unique 是安全的。
  // make_unique 是通过常规的 "new" 创建的，并且在此函数中通过 TypeMetaData::deleteFn 释放。
  // 如果张量在跨 DLL 边界时创建和释放，这可能会非常问题。
  // 创建包含权重数据的 PackedLinearWeightFp16 实例并返回其指针
  auto ptr = c10::make_intrusive<PackedLinearWeightFp16>(
      std::make_unique<fbgemm::PackedGemmMatrixFP16>(
          fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr),
      bias);
  return ptr;
}
#endif // USE_FBGEMM

#if AT_MKLDNN_ENABLED()
// 如果启用了 MKL-DNN，定义一个返回 LinearPackedParamsBase 指针的方法 prepack
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightsOnednn::prepack(
    at::Tensor weight,  // 输入权重张量
    std::optional<at::Tensor> bias_in) {  // 可选的输入偏置张量

  // 检查权重张量的维度是否为 2
  TORCH_CHECK(
      weight.dim() == 2,
      "quantized::linear_prepack (onednn): Weight tensor rank should be == 2");

  // 获取权重张量的行数
  int64_t rows_w = weight.size(0);

  // 如果提供了偏置张量，将其赋值给 bias_fp32，否则创建一个与权重行数相同的全零张量
  at::Tensor bias_fp32;
  if (bias_in.has_value()) {
    bias_fp32 = bias_in.value();
  } else {
    bias_fp32 = at::zeros(rows_w, weight.options().dtype(at::kFloat));
  }

  // 检查偏置张量的维度和尺寸是否符合预期
  TORCH_CHECK(
      !bias_fp32.defined() ||
          (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == rows_w),
      "quantized::linear_prepack (onednn): Given weight of size ",
      weight.sizes(),
      ", expected bias to be 1-dimensional with ",
      rows_w,
      " elements",
      ", but got bias of size ",
      bias_fp32.sizes(),
      " instead");

  // 将权重张量进行内存连续化
  at::Tensor weight_contig = weight.contiguous();

  // 初始化 MKL-DNN
  at::native::initMKLDNN();

  // TODO 实际上在这里调用 pre-pack，但在去除偏置项的预打包步骤中需要更新
  // 将预打包的线性权重设置为 nullptr，因为我们在第一次调用运算符运行时调用 pre-pack。
  // 有关详细信息，请参阅 Linear.cpp。TODO: 更新以实际调用 pre-pack
  auto wt_ptr = c10::make_intrusive<PackedLinearWeightsOnednn>(
      nullptr,
      weight_contig, /* int8_t weight */
      bias_fp32.contiguous()); /* fp32 偏置 */
  return wt_ptr;
}
#endif // AT_MKLDNN_ENABLED
    std::optional<at::Tensor> bias) {
  // 检查权重张量维度是否为2，因为 quantized::linear_prepack (onednn) 需要二维权重
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack (onednn) should"
      " be 2-dimensional.");

  // Weight
  // 获取权重张量的尺寸，并转换为向量形式
  std::vector<int64_t> dims = weight.sizes().vec();
  auto N = weight.size(0); // 获取权重张量的第一个维度大小
  std::vector<int32_t> wgt_zero_points; // 权重的零点
  ideep::scale_t wgt_scales; // 权重的缩放因子
  const auto qtype = weight.qscheme(); // 获取量化方案类型

  // 根据量化类型进行处理
  if (qtype == c10::kPerTensorAffine) {
    // 对称量化时，权重的零点必须为0
    TORCH_CHECK(
        weight.q_zero_point() == 0,
        "quantized::linear_prepack: ONEDNN only supports symmetric quantization of weight,"
        " whose zero point must be 0, but got ", weight.q_zero_point());
    wgt_zero_points = std::vector<int32_t>(1, weight.q_zero_point()); // 设置权重零点
    wgt_scales = ideep::scale_t(1, 1.0/weight.q_scale()); // ONEDNN 和 PyTorch 的缩放因子是互倒的
  } else if (qtype == c10::kPerChannelAffine) {
    // 通道量化时，需要处理每个通道的零点和缩放因子
    wgt_zero_points.resize(N); // 调整零点的大小以匹配通道数
    wgt_scales.resize(N); // 调整缩放因子的大小以匹配通道数
    for (int i = 0; i < N; ++i) {
      wgt_zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>(); // 获取每个通道的零点
      TORCH_CHECK(
          wgt_zero_points[i] == 0,
          "quantized::linear_prepack: ONEDNN only supports symmetric quantization of weight,"
          " whose zero point must be 0, but got ",  wgt_zero_points[i], ", at index ", i);
      wgt_scales[i] = 1.0f / weight.q_per_channel_scales()[i].item<float>(); // ONEDNN 和 PyTorch 的缩放因子是互倒的
    }
  } else {
    // 不支持的量化方案
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // Prepack weight
  auto weight_copy = weight.clone(); // 克隆权重张量
  ideep::tensor wgt = ideep::tensor({dims, dnnl::memory::data_type::s8}, weight_copy.data_ptr()); // 使用ONEDNN创建张量对象
  wgt.transpose_(0, 1); // ONEDNN要求转置权重
  auto src_dims = ideep::dims(); // 在预打包时未知
  ideep::attr_t op_attr;
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, 0); // 设置零点掩码
  auto w_desc = ideep::matmul_forward::expected_weights_desc(wgt.get_dims(), src_dims, dnnl::memory::data_type::s8,
                                                             dnnl::memory::data_type::u8, op_attr); // 期望的权重描述
  ideep::tensor exp_wgt(w_desc); // 创建预打包权重张量
  exp_wgt.feed_from(wgt); // 将数据填充到预打包权重张量中
  ideep::tensor * packed_weight_p = new ideep::tensor(std::move(exp_wgt)); // 创建指向预打包权重张量的指针
  packed_weight_p->set_scale(wgt_scales); // 设置权重的缩放因子
  packed_weight_p->set_zero_point(wgt_zero_points); // 设置权重的零点
  std::unique_ptr<ideep::tensor> weight_ptr(packed_weight_p); // 创建独特指针，管理预打包权重张量的所有权

  // Bias
  std::optional<ideep::tensor> onednn_bias{c10::nullopt}; // ONEDNN中的偏置

  // 如果存在偏置值
  if (bias.has_value()) {
    auto& b = bias.value(); // 获取偏置值的引用
    auto bias_size = b.sizes().vec(); // 获取偏置值的尺寸
    bias_size.insert(bias_size.begin(), 1); // 在偏置尺寸前插入一个元素
    TORCH_CHECK(
        bias_size[1] == weight_ptr->get_dim(1),
        "bias should have N elements: ",
        std::to_string(weight_ptr->get_dim(1)),
        ", but got ", bias_size[1]); // 检查偏置的尺寸是否匹配

    auto bias_desc = ideep::tensor::desc(bias_size, dnnl::memory::data_type::f32); // 创建偏置的描述
    ideep::tensor packed_bias; // 打包后的偏置
    packed_bias.init(bias_desc, b.data_ptr()); // 使用偏置数据初始化打包后的偏置张量


这段代码主要是对量化线性预打包函数 `quantized::linear_prepack` 中权重和偏置进行处理和打包的过程。
    # 使用 std::optional 类型来包装 packed_bias，创建一个可选的 onednn_bias 对象
    onednn_bias = std::optional<ideep::tensor>(packed_bias);
    # 使用 c10::make_intrusive 创建一个指向 PackedLinearWeightsOnednn 对象的智能指针 ret_ptr
    auto ret_ptr = c10::make_intrusive<PackedLinearWeightsOnednn>(
        PackedLinearWeightsOnednn{
            # 将 weight_ptr 移动到 PackedLinearWeightsOnednn 对象中
            std::move(weight_ptr),
            # 使用之前创建的 onednn_bias 作为参数传入
            onednn_bias,
            # 将 weight 作为参数传入 PackedLinearWeightsOnednn 对象中
            weight,
            # 将 bias 作为参数传入 PackedLinearWeightsOnednn 对象中
            bias});
    # 返回指向 PackedLinearWeightsOnednn 对象的智能指针 ret_ptr
    return ret_ptr;
#ifdef USE_FBGEMM
    // 如果使用了 FBGEMM 引擎或者 X86 引擎
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      // 调用 FBGEMM 提供的预打包函数，将权重和偏置打包为线性参数
      return PackedLinearWeight::prepack(std::move(weight), std::move(bias));
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    // 如果使用了 QNNPACK 引擎
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      // 调用 QNNPACK 提供的预打包函数，将权重和偏置打包为线性参数
      return PackedLinearWeightsQnnp::prepack(
          std::move(weight), std::move(bias));
    }
#endif

#if AT_MKLDNN_ENABLED()
    // 如果启用了 ONEDNN 引擎
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      // 调用 ONEDNN 提供的预打包函数，将权重和偏置打包为线性参数
      return PackedLinearWeightsOnednn::prepack(std::move(weight), std::move(bias));
    }
#endif // #if AT_MKLDNN_ENABLED()

    // 如果以上条件均不满足，则抛出错误信息并中止程序执行
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack ",
        toString(ctx.qEngine()));
}
    # 检查当前运行的量化引擎是否为 QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
        # 如果是 QNNPACK，抛出错误，因为 quantized::linear_prepack_fp16 不支持 QNNPACK
        TORCH_CHECK(
            false,
            "quantized::linear_prepack_fp16 is currently "
            "not supported by QNNPACK");
    }
#ifdef USE_PYTORCH_QNNPACK
#if AT_MKLDNN_ENABLED()
    // 检查当前运行环境是否为 ONEDNN，如果是，则抛出异常，因为不支持 FP16 格式的量化线性预打包
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      TORCH_CHECK(
          false,
          "quantized::linear_prepack_fp16 is currently "
          "not supported by ONEDNN");
    }
#endif // #if AT_MKLDNN_ENABLED()
    // 如果未找到适合 quantized::linear_prepack_fp16 操作的引擎，则抛出异常
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack_fp16 ",
        toString(ctx.qEngine()));
  }
};

class QLinearPackWeightInt8Legacy final {
 public:
  // 该类用于运行使用旧版本 quantized.linear_prepack 的模型权重打包操作，抛出异常提示用户需要使用新版本定义
  static Tensor run(at::Tensor weight, std::optional<Tensor> bias) {
    TORCH_CHECK(false,
        "This model uses an outdated version of quantized.linear_prepack. "
        "Please re-export your model using the newer definitions in torch.jit.quantized");
  }
};

class QLinearPackWeightFp16Legacy final {
 public:
  // 该类用于运行使用旧版本 quantized.linear_prepack_fp16 的模型权重打包操作，抛出异常提示用户需要使用新版本定义
  static Tensor run(at::Tensor weight, std::optional<Tensor> bias) {
    TORCH_CHECK(false,
        "This model uses an outdated version of quantized.linear_prepack_fp16. "
        "Please re-export your model using the newer definitions in torch.jit.quantized");
  }
};

class QLinearPackWeightInt8Onednn final {
 public:
  // 如果 ONEDNN 可用，则将权重打包为 ONEDNN 张量，否则抛出异常
  static at::Tensor run(
    at::Tensor weight, // Not QTensor
    std::optional<torch::List<int64_t>> input_shape) {
#if AT_MKLDNN_ENABLED()
    return pack_weight_to_onednn_tensor(weight, input_shape);
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.");
#endif
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 注册量化线性参数
  register_linear_params();
  // 将 quantized::linear_prepack 的实现绑定到 QLinearPackWeightInt8::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
  // 将 quantized::linear_prepack_legacy 的实现绑定到 QLinearPackWeightInt8Legacy::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_legacy"), TORCH_FN(QLinearPackWeightInt8Legacy::run));
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // 注册量化线性参数
  register_linear_params();
  // 将 quantized::linear_prepack_fp16 的实现绑定到 QLinearPackWeightFp16::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
  // 将 quantized::linear_prepack_fp16_legacy 的实现绑定到 QLinearPackWeightFp16Legacy::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  // 注册量化线性参数
  register_linear_params();
  // 将 _quantized::linear_prepack 的实现绑定到 QLinearPackWeightInt8::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
}

TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  // 注册量化线性参数
  register_linear_params();
  // 将 _quantized::linear_prepack_fp16 的实现绑定到 QLinearPackWeightFp16::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
  // 将 _quantized::linear_prepack_fp16_legacy 的实现绑定到 QLinearPackWeightFp16Legacy::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  // 将 onednn::qlinear_prepack 的实现绑定到 QLinearPackWeightInt8Onednn::run 函数
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_prepack"), TORCH_FN(QLinearPackWeightInt8Onednn::run));
}

} // namespace
} // namespace native
} // namespace at
```