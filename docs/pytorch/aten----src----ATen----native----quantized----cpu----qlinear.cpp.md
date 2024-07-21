# `.\pytorch\aten\src\ATen\native\quantized\cpu\qlinear.cpp`

```
// 定义预处理指令，以仅包含 Torch 的操作方法和运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

// 根据情况包含不同的头文件，定义了一些 ATen 操作的具体实现
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>         // for _empty_affine_q...
#include <ATen/ops/_empty_affine_quantized_native.h>  // for empty_affine_qu...
#include <ATen/ops/empty.h>                           // for empty
#include <ATen/ops/quantize_per_channel_native.h>     // for quantize_per_ch...
#include <ATen/ops/quantize_per_tensor_native.h>      // for quantize_per_te...
#include <ATen/ops/zeros.h>
#endif

// 包含 C++ 标准库的头文件
#include <c10/util/irange.h>
#include <algorithm>
#include <string>

// 声明函数 register_linear_params，返回整型
int register_linear_params();

// 根据预处理宏 USE_FBGEMM，定义了一个模板函数 apply_impl，接受输入张量 input，应用于线性加权
#ifdef USE_FBGEMM
template <bool ReluFused>
at::Tensor& PackedLinearWeight::apply_impl(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    at::Tensor& output) {
  // uint8 * int8 -> uint8 (no quantization/dequantization)

  // We make a strong guarantee that models using these operators will have
  // the same numerics across different machines. Therefore, we do not provide
  // a fallback path and rather fail loudly if we cannot run FBGEMM.
  // 检查当前 CPU 是否支持 FBGEMM，若不支持则抛出错误信息
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  // 检查输入张量的数据类型是否为 quint8
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
                "Expected input data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(input.scalar_type()));

  // TODO: contiguous is called for further jit optimizations.
  // 获取输入张量的连续版本，以进行进一步的 JIT 优化
  auto input_contig = input.expect_contiguous();
  // 获取输入张量数据的指针，并将其解释为 uint8 类型的指针
  const auto* input_ptr =
      reinterpret_cast<uint8_t*>(input_contig->data_ptr<c10::quint8>());

  // 检查输入张量的维度是否大于等于 2
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // 设置矩阵 C(output) = A(input) x B(weight)，其中 C、A、B 分别为 M x N、M x K、K x N 的矩阵
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 计算输入张量的最后一个维度的大小 M
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  // 获取权重张量 w 的打包数据
  auto packB = w.get();

  // 获取权重张量的维度信息
  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = input.sizes()[input.dim() - 1];
  // 检查权重张量的行数是否等于 K
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输入张量的量化比例因子和零点
  float input_scale_float = input.q_scale();
  int32_t input_zero_point_int32 = input.q_zero_point();

  // 初始化输出的倍乘因子和激活乘以权重的比例
  std::vector<float> output_multiplier_float(1, 0.0);
  std::vector<float> act_times_w_scale(1, 0.0);
  // 检查权重的量化比例因子和零点向量的大小是否相同
  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");
  if (q_scheme == c10::kPerTensorAffine) {
    // 处理按张量的量化方案
    act_times_w_scale[0] = (input_scale_float * w_scale[0]);
    output_multiplier_float[0] =
        act_times_w_scale[0] / static_cast<float>(output_scale);
  } else if (q_scheme == c10::kPerChannelAffine) {
    // 处理按通道的量化方案
    // 调整输出的倍乘因子和激活乘以权重的比例向量的大小
    output_multiplier_float.resize(N, 0.0);
    act_times_w_scale.resize(N, 1.0f);
    // 遍历每个通道，计算激活乘以权重的比例
    for (const auto i : c10::irange(N)) {
      act_times_w_scale[i] = (input_scale_float * w_scale[i]);
      output_multiplier_float[i] =
          act_times_w_scale[i] / static_cast<float>(output_scale);
    }
  }
  // 获取输出的零点值
  int32_t output_zero_point_int32 = static_cast<int32_t>(output_zero_point);

  // 初始化偏置指针为空
  const float* bias_ptr = nullptr;
  // 创建偏置的连续版本
  c10::MaybeOwned<at::Tensor> bias_contig;
  // 如果存在偏置，则进行以下操作
  if (this->bias_.has_value()) {
    auto& bias = this->bias_.value();
    // 获取偏置张量的连续版本
    bias_contig = bias.expect_contiguous();
    // 检查偏置张量是否为 1 维张量
    TORCH_CHECK(bias_contig->dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查偏置张量的第一个维度大小是否等于 N，确保偏置张量包含 N 个元素
    TORCH_CHECK(
        bias_contig->sizes()[0] == N, "bias should have N elements: " + std::to_string(N));
    // 将偏置张量的数据指针转换为 float* 类型
    bias_ptr = reinterpret_cast<float*>(bias_contig->data_ptr<float>());
  }

  // 结果矩阵是二维的，在保持原输入左侧维度的基础上查看它。这里给出两个示例：
  // 1. 如果输入张量为 {M, K}，输出张量为 {M, N}。
  // 2. 如果输入张量为 {b, M, K}，输出张量为 {b, M, N}。
  at::DimVector out_sizes(input.sizes());
  // 将输出张量的最后一个维度大小设为 N
  out_sizes.back() = N;
  // 调整输出张量的尺寸
  output.resize_(out_sizes);

  // 为 fbgemmPacked 分配一个使用的缓冲区
  auto buffer = at::empty(out_sizes, output.options().dtype(at::kInt));

  // 将输出张量的数据指针解释为 uint8_t* 类型
  auto output_data = reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>());

  // 获取线程数
  int num_tasks = at::get_num_threads();
  // 并行执行任务
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    }
  });

  // 返回输出张量
  return output;


这些注释解释了每行代码的具体作用，包括数据验证、内存管理、张量大小调整和并行任务执行。
}

at::Tensor PackedLinearWeight::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  // 分配输出张量
  auto output = at::_empty_affine_quantized(
      {0},
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  // 调用实现函数，不使用ReLU激活函数
  apply_impl<false>(input, output_scale, output_zero_point, output);
  return output;
}

at::Tensor PackedLinearWeight::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  // 分配输出张量
  auto output = at::_empty_affine_quantized(
      {0},
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  // 调用实现函数，使用ReLU激活函数
  apply_impl<true>(input, output_scale, output_zero_point, output);
  return output;
}

at::Tensor& PackedLinearWeight::apply_out(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    at::Tensor& output) {
  // 检查输出张量的属性
  TORCH_CHECK(
      (output.device() == c10::kCPU) && (output.dtype() == c10::kQUInt8) &&
      (output.q_scale() == output_scale) &&
      (output.q_zero_point() == output_zero_point));
  // 调用实现函数，不使用ReLU激活函数
  return apply_impl<false>(input, output_scale, output_zero_point, output);
}

at::Tensor& PackedLinearWeight::apply_relu_out(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    at::Tensor& output) {
  // 检查输出张量的属性
  TORCH_CHECK(
      (output.device() == c10::kCPU) && (output.dtype() == c10::kQUInt8) &&
      (output.q_scale() == output_scale) &&
      (output.q_zero_point() == output_zero_point));
  // 调用实现函数，使用ReLU激活函数
  return apply_impl<true>(input, output_scale, output_zero_point, output);
}

at::Tensor PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32(
  at::Tensor input,
  double input_scale,
  int64_t input_zero_point) {
  // 检查输入张量是否未量化
  TORCH_CHECK(!input.is_quantized(), "Input tensor for apply_with_input_q_dq_qweight_dq_output_fp32 is quantized; "
  "Expected input tensor in PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32 to be full precision.");

  // 调用实现函数，不使用ReLU激活函数
  return apply_with_input_q_dq_qweight_dq_output_fp32_impl<false>(input, input_scale, input_zero_point);
}

at::Tensor PackedLinearWeight::apply_with_input_q_dq_qweight_dq_relu_output_fp32(
  at::Tensor input,
  double input_scale,
  int64_t input_zero_point) {
  // 检查输入张量是否未量化
  TORCH_CHECK(!input.is_quantized(), "Input tensor for apply_with_input_q_dq_qweight_dq_output_fp32 is quantized; "
  "Expected input tensor in PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32 to be full precision.");

  // 调用实现函数，使用ReLU激活函数
  return apply_with_input_q_dq_qweight_dq_output_fp32_impl<true>(input, input_scale, input_zero_point);
}


template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32_impl(
    const at::Tensor& input,
    double input_scale,
    int64_t input_zero_point) {
  // 检查当前 CPU 是否支持 FBGEMM，否则抛出错误
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  // 确保输入张量是连续的，并获取其指针
  auto input_contig = input.expect_contiguous();
  const auto* input_ptr = input_contig->const_data_ptr<float>();

  // 检查输入张量的维度是否大于等于2
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // 计算输入张量中最后一维的大小
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  // 获取权重张量的包装对象
  auto packB = w.get();

  // 获取权重张量的列数 N
  int64_t N = static_cast<int64_t>(packB->numCols());
  // 获取输入张量的最后一维大小 K
  int64_t K = input.sizes()[input.dim() - 1];
  // 检查权重张量的行数是否等于 K
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  // 将输入量化的比例因子从浮点数转换为单精度浮点数
  float input_scale_float = input_scale;
  // 输入量化的零点值转换为 32 位整数
  int32_t input_zero_point_int32 = input_zero_point;

  // 检查权重张量的比例因子和零点向量是否具有相同的大小
  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");

  // 如果存在偏置项，则确保其为一维向量且长度为 N
  const float* bias_ptr = nullptr;
  c10::MaybeOwned<at::Tensor> bias_contig;
  if (this->bias_.has_value()) {
    auto& bias = this->bias_.value();
    bias_contig = bias.expect_contiguous();
    TORCH_CHECK(bias_contig->dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_contig->sizes()[0] == N, "bias should have N elements: " + std::to_string(N));
    bias_ptr = bias_contig->data_ptr<float>();
  }

  // 设置输出张量的大小，并将最后一维的大小设置为 N
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  // 分配输出张量和用于 fbgemmPacked 使用的缓冲区
  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
  auto buffer = at::empty_like(
      output,
      output.options().dtype(at::kInt),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // 获取输出张量的数据指针
  auto output_data = output.data_ptr<float>();

  // 获取线程数，用于并行化任务
  int num_tasks = at::get_num_threads();
  // 并行执行任务，每个任务执行以下内容
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    // 使用输入张量创建量化行偏移的 PackA 对象
    fbgemm::PackAWithQuantRowOffset<uint8_t> packA(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr,
        /*scale=*/input_scale_float,
        /*zero_pt=*/input_zero_point_int32);

    // 创建一个 DoNothing 对象，用于处理没有操作的情况
    fbgemm::DoNothing<float, float> doNothingObj{};
    // 使用范围循环遍历任务 ID 范围内的每个任务
    for (const auto task_id : c10::irange(begin, end)) {
      // 检查量化方案是否为每张量仿射量化
      if (q_scheme == c10::kPerTensorAffine) {
        // 处理每张量量化的操作
        //
        // 在执行 uint8 * int8 矩阵乘法后，此操作执行以下步骤：
        //  1) 对行和列分别添加行偏移量和列偏移量。
        //  2) 添加偏置项。
        fbgemm::ReQuantizeForFloat<ReluFused>
            outputProcObj(
                doNothingObj,  // 无操作对象
                input_scale_float,  // 输入缩放因子（浮点数）
                w_scale.data(),  // 权重缩放因子（数据）
                input_zero_point_int32,  // 输入零点（32位整数）
                w_zp.data(),  // 权重零点（数据）
                packA.getRowOffsetBuffer(),  // 行偏移量缓冲区
                col_offsets.data(),  // 列偏移量（数据）
                bias_ptr,  // 偏置指针
                N /* nCol */);  // 列数 N

        // 执行 GEMM（广义矩阵乘法）
        fbgemm::fbgemmPacked(
            /*packA=*/packA,  // 矩阵 A 的打包对象
            /*packB=*/packB,  // 矩阵 B 的打包对象
            /*C=*/output_data,  // 输出矩阵 C
            /*C_buffer=*/buffer.data_ptr<int32_t>(),  // 输出缓冲区
            /*ldc=*/N,  // 矩阵 C 的列数
            /*outProcess=*/outputProcObj,  // 输出处理对象
            /*thread_id=*/task_id,  // 线程 ID
            /*num_threads=*/num_tasks);  // 总线程数
      } else if (q_scheme == c10::kPerChannelAffine) {
        // 处理每通道仿射量化的操作
        //
        // 在执行 uint8 * int8 矩阵乘法后，此操作执行以下步骤：
        //  1) 对行和列分别添加行偏移量和列偏移量。
        //  2) 添加偏置项。
        fbgemm::ReQuantizeForFloat<
            ReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL>
            outputProcObj(
                doNothingObj,  // 无操作对象
                input_scale_float,  // 输入缩放因子（浮点数）
                w_scale.data(),  // 权重缩放因子（数据）
                input_zero_point_int32,  // 输入零点（32位整数）
                w_zp.data(),  // 权重零点（数据）
                packA.getRowOffsetBuffer(),  // 行偏移量缓冲区
                col_offsets.data(),  // 列偏移量（数据）
                bias_ptr,  // 偏置指针
                N /* nCol */);  // 列数 N

        // 执行 GEMM（广义矩阵乘法）
        fbgemm::fbgemmPacked(
            /*packA=*/packA,  // 矩阵 A 的打包对象
            /*packB=*/packB,  // 矩阵 B 的打包对象
            /*C=*/output_data,  // 输出矩阵 C
            /*C_buffer=*/buffer.data_ptr<int32_t>(),  // 输出缓冲区
            /*ldc=*/N,  // 矩阵 C 的列数
            /*outProcess=*/outputProcObj,  // 输出处理对象
            /*thread_id=*/task_id,  // 线程 ID
            /*num_threads=*/num_tasks);  // 总线程数
      }
    }
// 当使用 QNNPACK 和 XNNPACK 时执行以下代码块
#ifdef USE_PYTORCH_QNNPACK

#ifdef USE_XNNPACK
// TODO: 在将来添加 per_channel 支持时更新此处
template <typename scalar_t, bool kReluFused>
// 应用于 xnnpack 的 packed linear weights 的实现，返回一个张量
at::Tensor PackedLinearWeightsQnnp::apply_impl_xnnp(
    const at::Tensor& input, // 输入张量
    double output_scale, // 输出的缩放因子
    int64_t output_zero_point) { // 输出的零点
  using underlying_t = typename scalar_t::underlying; // scalar_t 的底层类型

  std::lock_guard<std::mutex> lock(qnnp_mutex_); // 使用互斥锁保护并发访问

  const std::string func_name = kReluFused ? "quantized::linear_relu (xnnpack)"
                                           : "quantized::linear (xnnpack)";
  // 检查输入张量的维度是否大于等于 2
  TORCH_CHECK(
      input.dim() >= 2, func_name, ": Input tensor rank should be >= 2.");
  // 检查是否不支持 per_channel 操作
  TORCH_CHECK(
      !per_channel(),
      func_name,
      ": xnnpack does not currently have per_channel support.");

  const auto input_contig = input.contiguous(); // 获取连续存储的输入张量
  const auto input_scale = input_contig.q_scale(); // 获取输入张量的量化缩放因子

  const size_t rows_w = bias_.size(0); // 获取偏置的行数
  const size_t cols_w = input_contig.size(input_contig.dim() - 1); // 获取输入张量的最后一维大小

  auto status = xnn_status_invalid_state; // 初始化状态为无效状态

  // 如果尚未创建运算符或者输入缩放因子发生变化，则创建一个新的运算符
  if (!xnnp_linear_op ||
      (!this->input_scale.has_value() ||
       this->input_scale.value() != input_scale)) {
    // 更新输入缩放因子以便缓存运算符
    this->input_scale = input_scale;

    xnn_operator_t xnnp_op = nullptr; // 初始化 xnnpack 运算符为空指针

    const float* weight_scales_data = w_scales.const_data_ptr<float>(); // 获取权重缩放因子数据的指针

    // 准备权重
    underlying_t w_zp = static_cast<underlying_t>(
        orig_weight.q_zero_point() +
        (std::is_same<underlying_t, uint8_t>::value ? 128 : 0));

    // 创建一个与原始权重相同大小的仿射量化空张量
    at::Tensor xnnp_weight = at::_empty_affine_quantized(
        orig_weight.sizes(),
        c10::CppTypeToScalarType<scalar_t>::value,
        weight_scales_data[0],
        w_zp);

    // 复制原始权重并根据需要处理数据类型更改
    at::native::xnnp_utils::q8_copy_int8_weight_and_add_offset<scalar_t>(
        orig_weight, xnnp_weight);

    // 原始偏置是浮点数，因此在这里重新量化
    at::Tensor qbias = quant_utils::QuantizeBias(false, bias_, orig_weight, input_scale);

    // 输出的下限和上限
    auto output_min = kReluFused
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).first
        : std::numeric_limits<underlying_t>::min();
    auto output_max = kReluFused
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).second
        : std::numeric_limits<underlying_t>::max();

    // 创建一个运算符
    // 调用 XNNPACK 库创建全连接层操作符，设置各种参数
    status = at::native::xnnp_utils::xnnp_create_fully_connected_nc(
        cols_w, /* input_channels */               // 输入通道数
        rows_w, /* output_channels */              // 输出通道数
        cols_w, /* input_stride */                 // 输入步长
        rows_w, /* output_stride */                // 输出步长
        input_contig.q_zero_point(),              // 输入张量的量化零点
        input_contig.q_scale(),                   // 输入张量的量化比例
        w_zp,                                     // 权重的量化零点
        weight_scales_data[0],                    // 权重的量化比例
        reinterpret_cast<const underlying_t*>(
            xnnp_weight.template data_ptr<scalar_t>()),  // 权重数据的指针转换为 scalar_t 类型
        reinterpret_cast<int32_t*>(qbias.data_ptr<c10::qint32>()),  // 偏置的数据指针转换为 qint32 类型
        output_zero_point,                        // 输出张量的量化零点
        output_scale,                             // 输出张量的量化比例
        output_min,                               // 输出张量的最小值
        output_max,                               // 输出张量的最大值
        0, /* flags */                            // 标志位，这里为 0
        &xnnp_op);                                // 指向创建的操作符的指针
    xnnp_linear_op = xnnpack_operator(xnnp_op);   // 将 XNNPACK 操作符转换为 XNNPACK 运算符

    // 检查 XNNPACK 操作符创建是否成功
    TORCH_CHECK(
        status == xnn_status_success,
        func_name,
        ": xnn create operator failed(",
        status,
        ")");

  }

  /*
   * 分配输出张量和 XNNPACK 使用的缓冲区
   * 这里生成的矩阵是二维的，根据输入的左手维度进行视图处理。
   * 这里有两个示例：
   * 1. 如果输入张量是 {M, K}，输出张量是 {M, N}。
   * 2. 如果输入张量是 {b, M, K}，输出张量是 {b, M, N}。
   */
  std::vector<int64_t> out_sizes = input.sizes().vec();  // 获取输入张量的尺寸
  out_sizes.back() = static_cast<int64_t>(rows_w);       // 设置输出张量的最后一个维度为 rows_w
  at::Tensor output = at::native::empty_affine_quantized(  // 创建一个量化空张量
      out_sizes,
      c10::CppTypeToScalarType<scalar_t>::value,
      c10::nullopt /* layout */,
      c10::kCPU,
      c10::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      input.suggest_memory_format());  // 根据输入张量建议的内存布局格式初始化输出张量

  // 计算 batch_size
  size_t rows_input = 1;
  for (const auto i : c10::irange(input_contig.dim() - 1)) {
    rows_input *= input_contig.size(i);  // 计算输入张量的所有维度乘积，除去最后一个维度
  }

  // 重塑操作符
  status = at::native::xnnp_utils::xnnp_reshape_fully_connected_nc(
      xnnp_linear_op.get(),
      rows_input, /* batch_size */          // 批处理大小
      caffe2::pthreadpool_());             // 使用 pthreadpool 进行线程管理

  // 设置操作符
  status = at::native::xnnp_utils::xnnp_setup_fully_connected_nc(
      xnnp_linear_op.get(),
      reinterpret_cast<const underlying_t*>(
          input_contig.template data_ptr<scalar_t>()),  // 输入张量的数据指针转换为 scalar_t 类型
      reinterpret_cast<underlying_t*>(output.template data_ptr<scalar_t>())  // 输出张量的数据指针转换为 scalar_t 类型
    );

  // 检查操作符设置是否成功
  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn setup operator failed(",
      status,
      ")");

  // 运行操作符
  status = xnn_run_operator(
      xnnp_linear_op.get(),        // 线性操作符
      caffe2::pthreadpool_()       // 线程池
  );
  // 检查操作符运行是否成功
  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn run operator failed(",
      status,
      ")");

  return output;  // 返回输出张量
  // 结束ifdef USE_XNNPACK条件编译

#endif // USE_XNNPACK

// 实现应用于输入的带条件ReLU融合线性权重函数
template <bool ReluFused>
at::Tensor PackedLinearWeightsQnnp::apply_impl(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  // 检查输入张量的维度是否至少为2
  TORCH_CHECK(
      input.dim() >= 2,
      "quantized::linear(): Input tensor rank should be >= 2");
  // 检查输入张量是否为无符号8位整数类型
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
                "quantized::linear (qnnpack): Expected input data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(input.scalar_type()));

  // 获得连续内存的输入张量副本
  auto input_contig = input.contiguous();

  // 权重打包不是线程安全的，使用互斥锁确保线程安全
  std::lock_guard<std::mutex> lock(qnnp_mutex_);
  // 获取权重对象的指针
  auto packB = w.get();
  // 计算权重的行数和列数
  size_t rows_w = bias_.size(0);
  size_t cols_w = input_contig.size(input_contig.dim() - 1);
  // 获取输入张量的量化比例
  auto input_scale = input_contig.q_scale();

  // 如果输入比例未定义或者与当前输入比例不同，执行以下操作
  if (!this->input_scale.has_value() ||
      this->input_scale.value() != input_scale) {
    // 获取原始权重并调整为从int8到uint8
    auto weight_contig = orig_weight;
    auto bias_fp32 = bias_;
    int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();

    // 获取权重缩放比例数据
    float* weight_scales_data = w_scales.data_ptr<float>();
    // 在此处计算重新量化比例，由模块拥有，然后传递给qnnpack后端
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales, input_scale, output_scale, requantization_scales);

    // 创建空的仿射量化的权重张量
    at::Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8),
        weight_scales_data[0],
        w_zero_points[0]);
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    auto wt_numel = weight_contig.numel();
    // 将权重数据转换为uint8，并调整偏移量
    for (const auto i : c10::irange(wt_numel)) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }
    // 原始偏置为浮点数，因此在此处进行重新量化
    const bool is_per_channel = orig_weight.qscheme() == at::kPerChannelAffine;
    at::Tensor qbias = quant_utils::QuantizeBias(is_per_channel, bias_fp32, weight_contig, input_scale);

    // 更新输入比例，以避免重新打包
    this->input_scale = input_scale;
    // 重置权重指针
    w.reset();
    // 创建qnnpack的PackBMatrix对象并更新packB指针
    w = std::make_unique<qnnpack::PackBMatrix>(
        cols_w /* input_channels */,
        rows_w /* output_channels */,
        w_zero_points.data(),
        requantization_scales.data(),
        reinterpret_cast<uint8_t*>(qnnp_w_data),
        reinterpret_cast<int32_t*>(qbias.data_ptr<c10::qint32>()));
    packB = w.get();
    // 如果在预打包时释放权重，在移动设备上重置原始权重
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
      // 在移动设备上，通过重置intrusive_ptr来释放原始权重
      // 在此之后调用unpack将引发断言
      orig_weight.reset();
  }
  // 遍历输入张量的所有维度，计算输入张量除去最后一维外的元素个数的乘积，得到输入张量的行数
  size_t rows_input = 1;
  size_t cols_input = input_contig.size(input_contig.dim() - 1);
  for (const auto i : c10::irange(input_contig.dim() - 1)) {
    rows_input *= input_contig.size(i);
  }

  // 检查输入张量的最后一维（列数）是否与权重张量的第二维大小相匹配，若不匹配则抛出错误
  TORCH_CHECK(
      cols_input == cols_w,
      "quantized::linear(): input size does not match weight dimension 1 size: \
         got ",
      cols_input,
      " but expected ",
      cols_w);

  // 分配输出张量和 QNNPACK 使用的缓冲区
  // 这里的输出矩阵是二维的，可以视作具有输入张量原始左手侧维度的视图。以下是两个例子：
  // 1. 如果输入张量是 {M, K}，输出张量是 {M, N}。
  // 2. 如果输入张量是 {b, M, K}，输出张量是 {b, M, N}。
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = static_cast<long>(rows_w);
  at::Tensor output = at::_empty_affine_quantized(
      out_sizes,
      input.options(),
      output_scale,
      output_zero_point);

  // 根据是否启用了 ReLU 融合，确定输出张量的最小值和最大值
  auto output_min = ReluFused
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU)
            .first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = ReluFused
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU)
            .second
      : std::numeric_limits<uint8_t>::max();
  
  // 检查打包的权重是否为空，若为空则抛出错误
  TORCH_INTERNAL_ASSERT(packB != nullptr, "Packed Weights are NULL");
  
  // 调用 QNNPACK 的线性操作函数，执行量化的矩阵乘法
  const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinear(
      rows_input /* batch_size */,
      cols_input /* input_channels */,
      rows_w /* output_channels */,
      input_contig.q_zero_point(),
      w_zero_points.data(),
      requantization_scales.data(),
      output_zero_point,
      output_min,
      output_max,
      (uint8_t*)input_contig.data_ptr<c10::quint8>(),
      cols_input /* input_stride */,
      packB->getPackedWeights(),
      (uint8_t*)output.data_ptr<c10::quint8>(),
      rows_w /* output_stride */,
      // TODO (Ashkan): Disabling temporarily.
      // Throws a floating point exception with OSS pthreadpool.
      caffe2::pthreadpool_() /* threadpool */);

  // 检查 QNNPACK 线性操作的执行状态，确保成功执行
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Linear operator");

  // 返回输出张量
  return output;
#if AT_MKLDNN_ENABLED()
// 模板函数：应用线性权重到ONEDNN引擎的实现，支持不同的后操作
template <PostOps post_op>
at::Tensor PackedLinearWeightsOnednn::apply_impl(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point,
    torch::List<at::Scalar> post_op_args) {
  // 获取输入张量的维度数
  const int64_t dim = input.dim();
  // 检查输入张量的维度是否为至少1
  TORCH_CHECK(
      dim != 0,
      "qlinear (ONEDNN): input dim should be at least 1, but got 0");
  // 检查输入张量的数据类型是否为QUInt8
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::QUInt8,
      "qlinear (ONEDNN): data type of input should be QUint8.");

  // 确保输入张量是连续的
  auto input_contig = input.expect_contiguous();
  // 获取权重张量的引用
  auto& w = *(weight_.get());
  // 获取输入张量的尺寸
  auto K = input.size(dim - 1), M = input.numel() / K, N = w.get_dim(1);
  // 定义输入张量的维度和数据类型（无符号8位整数）
  auto input_dims = {M, K};
  auto input_data_type = dnnl::memory::data_type::u8;
  // 创建ONEDNN张量描述符
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  // 定义ONEDNN操作的属性
  ideep::attr_t op_attr = ideep::attr_t();
  
  // 根据后操作类型设置ONEDNN操作属性
  if (post_op == Relu) {
    op_attr = ideep::attr_t::fuse_relu();
  } else if (post_op == LeakyRelu) {
    auto alpha = post_op_args.get(0).to<double>();
    op_attr = ideep::attr_t::fuse_relu(/*scale=*/1.0f, /*alpha=*/alpha);
  } else if (post_op == Tanh) {
      // Perform hyperbolic tangent
     ```cpp
    // 如果后操作是Tanh，则应用双曲正切
    op_attr = ideep::attr_t::fuse_tanh();
  } else if (post_op == Sigmoid) {
    // 如果后操作是Sigmoid，则应用Sigmoid函数
    op_attr = ideep::attr_t::fuse_sigmoid();
  }

  // 在ONEDNN引擎上应用线性权重，根据后操作属性选择相应的实现
  return input_contig.with_version_counter()
      .typeMeta()
      .toTensor();
}

#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
// 模板函数：应用ReLU激活函数后的线性权重到ONEDNN引擎的实现
template <PostOps post_op>
at::Tensor PackedLinearWeightsOnednn::apply_impl_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point,
    torch::List<at::Scalar> post_op_args) {
  // 获取输入张量的维度数
  const int64_t dim = input.dim();
  // 检查输入张量的维度是否为至少1
  TORCH_CHECK(
      dim != 0,
      "qlinear (ONEDNN): input dim should be at least 1, but got 0");
  // 检查输入张量的数据类型是否为QUint8
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::QUInt8,
      "qlinear (ONEDNN): data type of input should be QUint8.");

  // 确保输入张量是连续的
  auto input_contig = input.expect_contiguous();
  // 获取权重张量的引用
  auto& w = *(weight_.get());
  // 获取输入张量的尺寸
  auto K = input.size(dim - 1), M = input.numel() / K, N = w.get_dim(1);
  // 定义输入张量的维度和数据类型（无符号8位整数）
  auto input_dims = {M, K};
  auto input_data_type = dnnl::memory::data_type::u8;
  // 创建ONEDNN张量描述符
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  // 定义ONEDNN操作的属性
  ideep::attr_t op_attr = ideep::attr_t();
  
  // 根据后操作类型设置ONEDNN操作属性，并在ReLU之后应用
  if (post_op == Relu) {
    op_attr = ideep::attr_t::fuse_relu();
  } else if (post_op == LeakyRelu) {
    auto alpha = post_op_args.get(0).to<double>();
    op_attr = ideep::attr_t::fuse_relu(/*scale=*/1.0f, /*alpha=*/alpha);
  } else if (post_op == Tanh) {
    op_attr = ideep::attr_t::fuse_tanh();
  } else if (post_op == Sigmoid) {
    op_attr = ideep::attr_t::fuse_sigmoid();
  }

  // 在ONEDNN引擎上应用ReLU后的线性权重，根据后操作属性选择相应的实现
  return input_contig.with_version_counter()
      .typeMeta()
      .toTensor();
}
    // 定义操作属性，这里是 fuse_tanh()，返回一个 ideep::attr_t 对象
    op_attr = ideep::attr_t::fuse_tanh();
  }
  // 创建 ideep::tensor 对象 x，使用输入描述符和输入数据指针
  ideep::tensor x(input_desc, input_contig->data_ptr<c10::quint8>());
  // 定义目标维度 dst_dims，这里是一个包含 M 和 N 的初始化列表
  auto dst_dims = {M, N};
  // 获取输入的量化参数
  double input_scale = input.q_scale();
  int64_t input_zero_point = input.q_zero_point();
  // 创建源张量的量化比例，这里使用 ideep::scale_t 初始化
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/input_scale);
  // 获取权重张量的量化比例
  const ideep::scale_t& weights_scales = w.get_scale();
  // 根据输出量化参数创建目标张量的量化比例
  const ideep::scale_t& dst_scales = ideep::scale_t(1, 1.0/output_scale);
  // 创建源张量的零点，这里使用 ideep::zero_point_t 初始化
  const ideep::zero_point_t& src_zero_point = ideep::zero_point_t(1, input_zero_point);
  // 根据输出的零点创建目标张量的零点
  const ideep::zero_point_t& dst_zero_point = ideep::zero_point_t(1, output_zero_point);
  // 计算：使用 ideep::matmul_forward 执行矩阵乘法前向传播，支持非对称量化
  // 分配输出 Tensor
  at::Tensor output = at::_empty_affine_quantized(
      dst_dims,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  // 如果输出 Tensor 的元素数为 0，则直接返回空的输出 Tensor
  if (output.numel() == 0) {
    return output;
  }
  // 创建 ideep::tensor 对象 y，使用目标维度和 u8 数据类型，以及输出 Tensor 的数据指针
  ideep::tensor y({dst_dims, ideep::tensor::data_type::u8,
                   {output.strides().cbegin(), output.strides().cend()}},
                  output.data_ptr());
  // 检查是否存在偏置项
  bool with_bias = bias_.has_value();
  // 如果存在偏置项
  if (with_bias) {
    // 偏置项可能在外部被修改（例如通过量化偏置校正）。
    // 如果是这样，更新预打包的偏置项。
    if (bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
      bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
    }
  }
  // 获取偏置项 b，如果不存在偏置项，则创建空的 ideep::tensor 对象
  const auto& b = with_bias ? bias_.value() : ideep::tensor();
  // 初始化原始缓存键
  PrimitiveCacheKey cache_key = std::make_tuple(
      input_scale, input_zero_point, input_dims, output_scale, output_zero_point, num_threads, /*accum scale*/1.0, /*accum zero point*/0);
  // 使用 call_once 初始化原语缓存，lambda 函数包含线性参数的初始化和重排序操作
  c10::call_once(*cache_initialized_flag, [&](){
      LinearParams params;
      ideep::matmul_forward::prepare</*is_dynamic=*/false>(
          params, x, w, b, y,
          src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, 1.0f, 1.0f, op_attr);
      // 缓存中保存线性原语的参数
      get_cache() = LinearPrimitiveCache(cache_key, params);
      // 如果权重描述符不同，则重新排序权重 w
      w = w.reorder_if_differ_in(params.pd.weights_desc());
  });
  // 如果命中了缓存，则直接使用缓存中的参数计算
  if (get_cache().hit(cache_key)) {
    LinearParams& params = get_cache().get_param();
    ideep::matmul_forward::compute<false, false>(params, x, w, b, y);
  } else {
    // 否则，调用 ideep::matmul_forward::compute 执行矩阵乘法计算
    ideep::matmul_forward::compute(x, w, b, y, src_scales, weights_scales,
                                   dst_scales, src_zero_point, dst_zero_point,
                                   1.0f, 1.0f, op_attr);
  }
  // 获取输入的尺寸，并更新为输出尺寸
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  // 如果输出 Tensor 的尺寸与更新后的尺寸相同，则直接返回输出 Tensor
  if (output.sizes().vec() == out_sizes)
    return output;
  // 否则，调整输出 Tensor 的形状为更新后的尺寸，并返回
  return output.reshape(out_sizes);
}
// 结束函数定义

at::Tensor PackedLinearWeightsOnednn::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<NoPostOp>(
      std::move(input), output_scale, output_zero_point);
}
// 调用 apply_impl 函数，将输入的 Tensor 应用于无后处理操作，返回处理后的 Tensor

at::Tensor PackedLinearWeightsOnednn::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<Relu>(
      std::move(input), output_scale, output_zero_point);
}
// 调用 apply_impl 函数，将输入的 Tensor 应用于 ReLU 后处理操作，返回处理后的 Tensor

at::Tensor PackedLinearWeightsOnednn:: apply_leaky_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point,
    double negative_slope) {
  torch::List<at::Scalar> post_op_args =
      {at::Scalar(negative_slope)};
  return apply_impl<LeakyRelu>(
      std::move(input), output_scale, output_zero_point, post_op_args);
}
// 调用 apply_impl 函数，将输入的 Tensor 应用于带泄漏的 ReLU 后处理操作，返回处理后的 Tensor

at::Tensor PackedLinearWeightsOnednn:: apply_tanh(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<Tanh>(
      std::move(input), output_scale, output_zero_point);
}
// 调用 apply_impl 函数，将输入的 Tensor 应用于 Tanh 后处理操作，返回处理后的 Tensor

static at::Tensor linear_int8_with_onednn_weight(
    at::Tensor input, // int8 CPU Tensor, not QTensor
    double input_scale,
    int64_t input_zero_point,
    at::Tensor onednn_weight, // int8 tensor from MkldnnCPU
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias, // plain tensor
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<at::Tensor> other, // extra input for binary post-op
    double other_scale,
    int64_t other_zero_point,
    const c10::string_view& binary_post_op, // e.g. "none", "sum", "add"
    double binary_alpha,
    const c10::string_view& unary_post_op, // e.g. "none", "relu"
    torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    c10::string_view& unary_post_op_algorithm) {
  using ideep::tensor;
  const int64_t dim = input.dim();
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Byte,
      "qlinear with mkldnn tensor: data type of input should be uint8 (unsigned char).");
  TORCH_CHECK(onednn_weight.scalar_type() == c10::ScalarType::Char,
      "qlinear with mkldnn tensor: data type of weight should be int8 (char).");
  TORCH_CHECK(
      weight_scales.scalar_type() == c10::ScalarType::Float, "weight scales should be dtype c10::ScalarType::Float.");
  TORCH_CHECK(
      binary_alpha == 1.0f, "onednn qlinear: alpha != 1 for binary post op is not yet supported.");
  bool fp32_output = output_dtype.has_value() && (output_dtype.value() == c10::kFloat);
  bool bf16_output = output_dtype.has_value() && (output_dtype.value() == c10::kBFloat16);
  if (fp32_output || bf16_output) {
    TORCH_CHECK(
        output_scale == 1.0f && output_zero_point == 0, "onednn qlinear: expect scale=1 and zero point=0 for fp32 output");
  }
  if (binary_post_op != "none") {
    // 检查是否支持二元后处理操作
    // 目前仅支持 "none" 类型
    TORCH_CHECK(false, "onednn qlinear: only 'none' is supported for binary post-op.");
  }
}
// 执行线性整数运算与 OneDNN 权重的计算，包含一系列参数的检查与设定
    /* 支持二进制后操作的情况如下:
      +-------------------+--------------+---------------+
      | 额外输入的数据类型 | 输出数据类型 | 后操作         |
      +-------------------+--------------+---------------+
      | Fp32/bf16         | fp32/bf16    | sum           |
      +-------------------+--------------+---------------+
      | Fp32/bf16         | int8         | add           |
      +-------------------+--------------+---------------+
      | int8              | fp32/bf16    | 不支持        |
      +-------------------+--------------+---------------+
      | int8              | int8         | sum           |
      +-------------------+--------------+---------------+
    */
    TORCH_CHECK(other.has_value(), "onednn qlinear: the extra input is missing for post op ", binary_post_op);
    // 检查是否存在额外的输入值，如果不存在则报错，指出丢失的二进制后操作的额外输入
    
    if (fp32_output || bf16_output) {
      // 如果输出数据类型为 fp32 或 bf16
      TORCH_CHECK(
          other_scale == 1.0f && other_zero_point == 0,
          "onednn qlinear: expect extra input scale = 1.0 and zero point = 0 when output dtype is ", output_dtype.value(),
          ", but got ", other_scale, " and ", other_zero_point, ", respectively"
      );
      // 检查额外输入的标度和零点是否符合预期值
    }
    
    if (binary_post_op == "sum") {
      // 如果二进制后操作为 "sum"
      auto expected_dtype = output_dtype.has_value() ? output_dtype.value() : c10::kByte;
      // 获取预期的数据类型，如果没有指定，则默认为 c10::kByte
      TORCH_CHECK(
          other.value().scalar_type() == expected_dtype,
          "onednn qlinear: the dtype of extra input for binary post op should be ", expected_dtype,
          " (same as output dtype), but got ", other.value().scalar_type()
      );
      // 检查额外输入的数据类型是否符合预期，即与输出数据类型一致
    }
    
    }
    
    // 如果输入具有超过两个维度，我们将把它重塑为二维形式进行计算，并随后重塑输出。
    auto input_contig =
        dim == 2 ? input.contiguous() : input.reshape({-1, input.size(dim - 1)}).contiguous();
    // 如果输入张量的维度超过两个，则将其重塑为二维形式，以便进行计算，并确保是连续的
    
    auto src = at::native::itensor_from_tensor(input_contig);
    // 从输入张量创建一个内部张量表示
    
    auto packed_weight = at::native::itensor_from_mkldnn(onednn_weight);
    // 从 MKL-DNN 格式的权重数据创建一个打包的权重张量表示
    
    int64_t K = input.size(dim - 1), M = input.numel() / K, N = packed_weight.get_dim(1);
    // 获取输入张量的相关维度信息，用于后续的计算
    
    auto output_size = input.sizes().vec();
    output_size[dim - 1] = N;
    // 设置输出张量的大小，保持除了指定维度外的其他维度不变
    
    std::optional<ideep::tensor> onednn_bias{c10::nullopt};
    bool with_bias = bias.has_value();
    at::Tensor bias_val_float;
    if (with_bias) {
      bias_val_float = bias.value().to(at::kFloat);
      if (bias_val_float.dim() == 1) {
        auto b_reshape = bias_val_float.reshape({1, bias_val_float.size(0)});
        onednn_bias = at::native::itensor_view_from_dense(b_reshape);
      } else {
        onednn_bias = at::native::itensor_view_from_dense(bias_val_float);
      }
    }
    // 处理偏置项，将其转换为浮点型张量并创建相应的内部张量表示
    
    std::vector<int64_t> src_dims = {M, K};
    std::vector<int64_t> dst_dims = {M, N};
    // 定义输入和输出张量的维度信息
    
    at::Tensor output = binary_post_op == "sum" ?
        other.value() :
        at::empty(
          dst_dims,
          device(c10::kCPU)
              .dtype(fp32_output ? c10::kFloat : (bf16_output ? c10::kBFloat16 : c10::kByte))
        );
    // 根据二进制后操作的类型创建输出张量，如果是 "sum" 则使用预先计算的额外输入，否则根据输出数据类型创建空的张量
    
    if (output.numel() == 0) {
  return output;
  // 返回函数的输出 tensor

tensor dst = at::native::itensor_view_from_dense(output);
// 根据 dense tensor 创建一个视图 tensor dst

static tensor empty_tensor;
static tensor::desc empty_tensor_desc;
// 定义静态的空 tensor 和其描述符

tensor src1 = binary_post_op == "add" ?
    at::native::itensor_view_from_dense(other.value().reshape({-1, other.value().size(dim - 1)})) :
    empty_tensor;
// 根据 binary_post_op 的类型选择性地创建 src1 tensor 视图，如果是 "add" 操作则使用 other 的视图，
// 否则使用空 tensor

// 创建 OneDNN 原语
auto src_desc = tensor::desc(src_dims, ideep::data_type::u8, ideep::format_tag::any);
// 根据 src_dims 创建输入张量的描述符，数据类型为 u8

auto weights_desc = packed_weight.get_desc();
// 获取 packed_weight 的描述符

auto dst_dtype = dst.get_data_type();
auto dst_desc = tensor::desc(dst_dims, dst_dtype, ideep::format_tag::any);
// 根据 dst_dims 和 dst 的数据类型创建输出张量的描述符

auto bias_desc = with_bias ?
    tensor::desc(onednn_bias.value().get_dims(), ideep::data_type::f32, ideep::format_tag::any) :
    empty_tensor_desc;
// 如果有 bias，则创建偏置 tensor 的描述符，数据类型为 f32，否则使用空 tensor 的描述符

// 获取原语的操作属性
// 注意：output_scale 和 output_zero_point 用于最终输出的重新量化。
// other_scale 和 other_zero_point 用于 other 的反量化。
auto other_desc = binary_post_op == "add" ? src1.get_desc() : empty_tensor_desc;
auto op_attr = onednn_utils::create_attr_by_post_op(
  binary_post_op,
  binary_alpha,
  other_scale,
  other_zero_point,
  other_desc,
  unary_post_op,
  unary_post_op_args,
  unary_post_op_algorithm
);
// 根据不同的后操作类型创建操作属性

if (input_scale != 1.0f) {
  op_attr.set_scales_mask(DNNL_ARG_SRC, 0);
}
// 如果输入的缩放因子不为 1.0，则设置输入的缩放掩码

if (input_zero_point != 0) {
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
}
// 如果输入的零点不为 0，则设置输入的零点掩码

op_attr.set_scales_mask(DNNL_ARG_WEIGHTS, ideep::utils::op_scale_mask(weight_scales.numel()));
// 设置权重的缩放掩码

if (output_scale != 1.0f) {
  op_attr.set_scales_mask(DNNL_ARG_DST, 0);
}
// 如果输出的缩放因子不为 1.0，则设置输出的缩放掩码

if (output_zero_point != 0) {
  op_attr.set_zero_points_mask(DNNL_ARG_DST, 0);
}
// 如果输出的零点不为 0，则设置输出的零点掩码

op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
// 设置原语的临时内存模式为用户定义的模式

auto engine = ideep::engine::cpu_engine();
// 获取 CPU 引擎

auto primitive_desc = with_bias ?
    dnnl::matmul::primitive_desc(engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr) :
    dnnl::matmul::primitive_desc(engine, src_desc, weights_desc, dst_desc, op_attr);
// 根据是否有偏置创建矩阵乘法原语的描述

auto primitive = dnnl::matmul(primitive_desc);
// 创建矩阵乘法的原语

// 如果需要，重新排序权重
auto expected_weight = packed_weight.reorder_if_differ_in(primitive_desc.weights_desc());

// 准备参数并执行原语
tensor scratchpad(primitive_desc.scratchpad_desc());
ideep::exec_args args;
args.insert({DNNL_ARG_SRC, src});
args.insert({DNNL_ARG_WEIGHTS, expected_weight});
args.insert({DNNL_ARG_DST, dst});
args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
if (with_bias) {
  args.insert({DNNL_ARG_BIAS, onednn_bias.value()});
}
// 插入执行参数

tensor src_scales_t = tensor(ideep::scale_t(1, input_scale));
tensor wei_scales_t = at::native::itensor_from_tensor(weight_scales);
tensor dst_scales_t = tensor(ideep::scale_t(1, output_scale));
tensor src_zp_t = tensor(ideep::zero_point_t(1, input_zero_point));
tensor dst_zp_t = tensor(ideep::zero_point_t(1, output_zero_point));
// 创建缩放因子和零点的 tensor

if (input_scale != 1.0f) {
    # 将 src_scales_t 插入到 args 中，对应于 DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  # 如果 output_scale 不等于 1.0f，则将 dst_scales_t 插入到 args 中，对应于 DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST
  if (output_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_t});
  }
  # 将 wei_scales_t 插入到 args 中，对应于 DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});
  # 如果 input_zero_point 不等于 0，则将 src_zp_t 插入到 args 中，对应于 DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC
  if (input_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_t});
  }
  # 如果 output_zero_point 不等于 0，则将 dst_zp_t 插入到 args 中，对应于 DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST
  if (output_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_t});
  }
  # 如果 binary_post_op 等于 "add"，则将 src1 插入到 args 中，对应于 DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1
  if (binary_post_op == "add") {
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, src1});
  }
  # 执行计算图的原语，使用默认的流对象 ideep::stream::default_stream() 和 args 参数
  primitive.execute(ideep::stream::default_stream(), args);
  # 如果 dim 等于 2，则返回 output；否则返回 output.reshape(output_size)
  return dim == 2 ? output : output.reshape(output_size);
// 定义 QLinearInt8 模板类，用于执行量化整型线性操作（可选带 ReLU）
template <bool ReluFused>
class QLinearInt8 final {
 public:
  // 静态成员函数 run，执行量化整型线性操作
  static at::Tensor run(
      at::Tensor input,  // 输入张量
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,  // 线性参数的包装指针
      double output_scale,  // 输出的量化比例因子
      int64_t output_zero_point) {  // 输出的量化零点
    if (ReluFused) {
      // 如果启用了 ReLU 融合，则应用带 ReLU 的线性操作
      return packed_weight->apply_relu(
          std::move(input), output_scale, output_zero_point);
    } else {
      // 否则，应用普通的线性操作
      return packed_weight->apply(
          std::move(input), output_scale, output_zero_point);
    }
  }
};

// 定义 QLinearLeakyReluInt8 类，用于执行带 Leaky ReLU 的量化整型线性操作
class QLinearLeakyReluInt8 final {
 public:
  // 静态成员函数 run，执行带 Leaky ReLU 的量化整型线性操作
  static at::Tensor run(
      at::Tensor input,  // 输入张量
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,  // 线性参数的包装指针
      double output_scale,  // 输出的量化比例因子
      int64_t output_zero_point,  // 输出的量化零点
      double negative_slope) {  // Leaky ReLU 的负斜率参数
    auto& ctx = at::globalContext();
    // 如果 MKLDNN 加速可用
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      // 如果量化引擎为 ONEDNN，则应用带 Leaky ReLU 的线性操作
      return dynamic_cast<PackedLinearWeightsOnednn*>(packed_weight.get())->apply_leaky_relu(
          std::move(input), output_scale, output_zero_point, negative_slope);
    }
#endif
    // 如果不满足条件，抛出错误信息
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_leaky_relu ",
        toString(ctx.qEngine()));
  }
};

// 定义 QLinearTanhInt8 类，用于执行带 Tanh 的量化整型线性操作
class QLinearTanhInt8 final {
 public:
  // 静态成员函数 run，执行带 Tanh 的量化整型线性操作
  static at::Tensor run(
      at::Tensor input,  // 输入张量
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,  // 线性参数的包装指针
      double output_scale,  // 输出的量化比例因子
      int64_t output_zero_point) {  // 输出的量化零点
    auto& ctx = at::globalContext();
    // 如果 MKLDNN 加速可用
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      // 如果量化引擎为 ONEDNN，则应用带 Tanh 的线性操作
      return dynamic_cast<PackedLinearWeightsOnednn*>(packed_weight.get())->apply_tanh(
          std::move(input), output_scale, output_zero_point);
    }
#endif
    // 如果不满足条件，抛出错误信息
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_tanh ",
        toString(ctx.qEngine()));
  }
};

// 定义 QLinearInt8FusedQDQ 模板类，用于执行带量化和 Dequantize 的量化整型线性操作（可选带 ReLU）
template <bool ReluFused>
class QLinearInt8FusedQDQ final {
 public:
  // 静态成员函数 run，执行带量化和 Dequantize 的量化整型线性操作
  static at::Tensor run(
      at::Tensor input,  // 输入张量
      double input_scale,  // 输入的量化比例因子
      int64_t input_zero_point,  // 输入的量化零点
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {  // 线性参数的包装指针
    if (ReluFused) {
      // 如果启用了 ReLU 融合，则应用带 ReLU 的量化和 Dequantize 线性操作
      return packed_weight->apply_with_input_q_dq_qweight_dq_relu_output_fp32(
          std::move(input), input_scale, input_zero_point);
    } else {
      // 否则，应用普通的量化和 Dequantize 线性操作
      return packed_weight->apply_with_input_q_dq_qweight_dq_output_fp32(
          std::move(input), input_scale, input_zero_point);
    }
  }
};
// 定义 QLinearOnednn 类，用于执行基于 OneDNN 的量化线性运算
class QLinearOnednn final {
 public:
  // 静态方法，执行基于点操作的量化线性运算
  static Tensor run_pointwise(
      Tensor act, // 输入激活张量，int8 CPU 张量，非量化张量
      double act_scale, // 激活张量的量化比例因子
      int64_t act_zero_point, // 激活张量的零点
      Tensor onednn_weight, // 来自 MkldnnCPU 的 int8 张量
      Tensor weight_scales, // 权重的量化比例因子
      Tensor weight_zero_points, // 权重的零点
      std::optional<Tensor> bias, // 可选的偏置张量
      double output_scale, // 输出的量化比例因子
      int64_t output_zero_point, // 输出的零点
      std::optional<c10::ScalarType> output_dtype, // 可选的输出数据类型
      c10::string_view post_op_name, // 后操作名称
      torch::List<std::optional<at::Scalar>> post_op_args, // 后操作参数列表
      c10::string_view post_op_algorithm) { // 后操作算法名称
#if AT_MKLDNN_ENABLED()
    // 静态变量，表示另一个参数的可选张量
    static std::optional<at::Tensor> other = c10::nullopt;
    // 常量字符串，二元后操作名称为 "none"
    static const c10::string_view binary_post_op = "none";
    // 调用 OneDNN 加权线性函数 linear_int8_with_onednn_weight
    return linear_int8_with_onednn_weight(
        act, act_scale, act_zero_point,
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, /*other scale*/1.0, /*other zp*/0,
        binary_post_op, /*binary alpha*/1.0,
        post_op_name, post_op_args, post_op_algorithm
    );
#endif
    // 如果未启用 Mkldnn，抛出错误信息
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }

  // 静态方法，执行基于张量的点操作的量化线性运算
  static Tensor run_pointwise_tensor(
      Tensor act, // 输入激活张量，int8 CPU 张量，非量化张量
      Tensor act_scale, // 激活张量的量化比例因子张量
      Tensor act_zero_point, // 激活张量的零点张量
      Tensor onednn_weight, // 来自 MkldnnCPU 的 int8 张量
      Tensor weight_scales, // 权重的量化比例因子张量
      Tensor weight_zero_points, // 权重的零点张量
      std::optional<Tensor> bias, // 可选的偏置张量
      double output_scale, // 输出的量化比例因子
      int64_t output_zero_point, // 输出的零点
      std::optional<c10::ScalarType> output_dtype, // 可选的输出数据类型
      c10::string_view post_op_name, // 后操作名称
      torch::List<std::optional<at::Scalar>> post_op_args, // 后操作参数列表
      c10::string_view post_op_algorithm) { // 后操作算法名称
#if AT_MKLDNN_ENABLED()
    // 检查激活张量的量化比例因子和零点张量的大小是否为1
    TORCH_CHECK(act_scale.numel() == 1 && act_zero_point.numel() == 1,
        "onednn int8 linear: act scale/zp size should be 1");
    // 静态变量，表示另一个参数的可选张量
    static std::optional<at::Tensor> other = c10::nullopt;
    // 常量字符串，二元后操作名称为 "none"
    static const c10::string_view binary_post_op = "none";
    // 调用 OneDNN 加权线性函数 linear_int8_with_onednn_weight
    return linear_int8_with_onednn_weight(
        act, act_scale.item().toDouble(), act_zero_point.item().toLong(),
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, /*other scale*/1.0, /*other zp*/0,
        binary_post_op, /*binary alpha*/1.0,
        post_op_name, post_op_args, post_op_algorithm
    );
#endif
    // 使用 TORCH_CHECK 宏来验证条件为 false，如果是，则抛出错误信息并终止程序
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }

  // 定义静态函数 run_pointwise_binary，用于执行逐点二进制操作
  static Tensor run_pointwise_binary(
      // act: 表示输入的 int8 CPU 张量，不是量化张量
      Tensor act,
      // act_scale: 输入张量的缩放因子
      double act_scale,
      // act_zero_point: 输入张量的零点偏移量
      int64_t act_zero_point,
      // onednn_weight: 来自 MkldnnCPU 的 int8 张量
      Tensor onednn_weight,
      // weight_scales: 权重的缩放因子
      Tensor weight_scales,
      // weight_zero_points: 权重的零点偏移量
      Tensor weight_zero_points,
      // bias: 可选的偏置张量
      std::optional<Tensor> bias,
      // output_scale: 输出张量的缩放因子
      double output_scale,
      // output_zero_point: 输出张量的零点偏移量
      int64_t output_zero_point,
      // output_dtype: 可选的输出数据类型
      std::optional<c10::ScalarType> output_dtype,
      // other: 用于二进制后操作的额外输入张量
      std::optional<at::Tensor> other,
      // other_scale: 额外输入张量的缩放因子
      double other_scale,
      // other_zero_point: 额外输入张量的零点偏移量
      int64_t other_zero_point,
      // binary_post_op: 二进制后操作类型，例如 "none", "sum", "add"
      c10::string_view binary_post_op,
      // binary_alpha: 二进制操作的 alpha 参数
      double binary_alpha,
      // unary_post_op: 一元后操作类型，例如 "none", "relu"
      c10::string_view unary_post_op,
      // unary_post_op_args: 一元后操作的参数列表
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      // unary_post_op_algorithm: 一元后操作的算法描述
      c10::string_view unary_post_op_algorithm) {
#if AT_MKLDNN_ENABLED()
    // 如果使用了 MKLDNN 加速，则调用基于 MKLDNN 的 int8 线性计算函数
    return linear_int8_with_onednn_weight(
        act, act_scale, act_zero_point,
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, other_scale, other_zero_point,
        binary_post_op, binary_alpha,
        unary_post_op, unary_post_op_args, unary_post_op_algorithm
    );
#endif
    // 如果未使用 MKLDNN 加速，报错并提示未实现相关功能
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }

  static Tensor run_pointwise_binary_tensor(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::optional<at::Tensor> other, // extra input for binary post-op
      double other_scale,
      int64_t other_zero_point,
      c10::string_view binary_post_op, // e.g. "none", "sum", "add"
      double binary_alpha,
      c10::string_view unary_post_op, // e.g. "none", "relu"
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      c10::string_view unary_post_op_algorithm) {
#if AT_MKLDNN_ENABLED()
    // 检查输入的 scale 和 zero_point 是否是单元素（即长度为 1）
    TORCH_CHECK(act_scale.numel() == 1 && act_zero_point.numel() == 1,
        "onednn int8 linear: act scale/zp size should be 1");
    // 调用基于 MKLDNN 的 int8 线性计算函数
    return linear_int8_with_onednn_weight(
        act, act_scale.item().toDouble(), act_zero_point.item().toLong(),
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, other_scale, other_zero_point,
        binary_post_op, binary_alpha,
        unary_post_op, unary_post_op_args, unary_post_op_algorithm
    );
#endif
    // 如果未使用 MKLDNN 加速，报错并提示未实现相关功能
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }
};
// 实现了 onednn 库在 MkldnnCPU 上的 Torch 库函数注册
TORCH_LIBRARY_IMPL(onednn, MkldnnCPU, m) {
  // 注册 onednn::qlinear_pointwise 函数的实现为 QLinearOnednn::run_pointwise
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise"),
      TORCH_FN(QLinearOnednn::run_pointwise));
  // 注册 onednn::qlinear_pointwise.tensor 函数的实现为 QLinearOnednn::run_pointwise_tensor
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.tensor"),
      TORCH_FN(QLinearOnednn::run_pointwise_tensor));
  // 注册 onednn::qlinear_pointwise.binary 函数的实现为 QLinearOnednn::run_pointwise_binary
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary"),
      TORCH_FN(QLinearOnednn::run_pointwise_binary));
  // 注册 onednn::qlinear_pointwise.binary_tensor 函数的实现为 QLinearOnednn::run_pointwise_binary_tensor
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary_tensor"),
      TORCH_FN(QLinearOnednn::run_pointwise_binary_tensor));
}

} // namespace
} // namespace native
} // namespace at
```