# `.\pytorch\aten\src\ATen\native\quantized\cpu\qlinear_dynamic.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_native.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_native.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <string>

// 声明一个函数，返回类型为 int，用于注册线性参数
int register_linear_params();

#ifdef USE_FBGEMM
// 如果使用了 FBGEMM 库，则定义一个模板函数 apply_dynamic_impl
template <bool ReluFused>
// 该函数接受一个 Tensor 类型的输入参数和返回一个 Tensor
at::Tensor PackedLinearWeight::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {
  using at::Tensor;
  // fp32 * int8 -> fp32 (with quantization on activation, and dequantization
  // on the result).

  // We make a strong guarantee that models using these operators will have
  // the same numerics across different machines. Therefore, we do not provide
  // a fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  // TODO: contiguous is called for further jit optimizations.
  // 对输入进行连续性处理，以便进行进一步的即时编译优化。
  auto input_contig = input.contiguous();
  // 获取输入张量的指向常量数据的指针
  const auto* input_ptr = input_contig.const_data_ptr<float>();

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
  // matrices, respectively.
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 计算输入张量的最后一个维度的大小，即M
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  // 获取权重矩阵 packB
  auto packB = w.get();

  // 获取权重矩阵 packB 的列数 N
  int64_t N = static_cast<int64_t>(packB->numCols());
  // 获取输入张量的最后一个维度的大小 K
  int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  // Calculate statistics for quantization of the input Tensor
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 计算输入张量的量化统计信息：最小值和最大值
  float x_min, x_max;
  fbgemm::FindMinMax(
      /*m=*/input_ptr,
      /*min=*/&x_min,
      /*max=*/&x_max,
      /*len=*/input.numel());

  // Input tensor is quantized as 8-bit unsigned values
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;

  // Calculate scale and zero point for quantization of input tensor
  // 计算输入张量的量化比例因子和零点
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/
      is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  q_params.precision = precision;

  // ReQuantizeForFloat requires pointers to the zero point values,
  // since in the case of rowwise quantization these will be arrays rather
  // than scalars. But in this case, we're doing whole-tensor quantization so
  // we just pass a pointer to the scale values (and internally
  // ReQuantizeForFloat won't index past 0.
  
  // 准备偏置项指针
  const float* bias_ptr = nullptr;
  at::Tensor bias_vec;
  if (bias_.has_value()) {
    // 如果存在偏置项，则获取其张量
    bias_vec = bias_.value();
    // 检查偏置项是一个一维向量（1D Tensor）
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查偏置项的大小是否与 N 相等
    TORCH_CHECK(
        bias_vec.size(0) == N,
        "bias should have N elements: " + std::to_string(N));
    // TODO: contiguous is called for further jit optimizations.
    // 对偏置项进行连续性处理，以便进行进一步的即时编译优化。
    auto bias_contig = bias_vec.contiguous();
  }
    // 获取偏置连续数据的指针，数据类型为 float
    bias_ptr = bias_contig.data_ptr<float>();
  }
  // 这里得到的矩阵是二维的，我们将其与输入的原始左手维度视图进行关联。
  // 以下是两个示例：
  // 1. 如果输入张量为 {M, K}，输出张量为 {M, N}。
  // 2. 如果输入张量为 {b, M, K}，输出张量为 {b, M, N}。
  std::vector<int64_t> out_sizes = input.sizes().vec();
  // 将输出张量的最后一个维度大小设为 N
  out_sizes.back() = N;
  // 分配输出张量和用于 fbgemmPacked 使用的缓冲区
  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
  auto buffer = at::empty_like(
      output,
      output.options().dtype(at::kInt),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // 获取线程数量，用于并行处理
  int num_tasks = at::get_num_threads();
  // 并行处理任务，使用 lambda 表达式定义任务范围
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    // 该操作包括以下步骤：
    // 1) 根据上面计算的统计数据对输入矩阵进行量化
    // 2) 创建一个“行缓冲”向量，其中包含必须添加到整数矩阵乘法操作中的偏移值，
    //    以确保正确性。这个“行缓冲”也称为行偏移，在权重使用仿射量化时是必需的。
    // 3) 将结果量化后的矩阵打包成向量寄存器和缓存友好的瓦片。
    //
    // 注意这些操作不是立即执行的，而是在下面的 fbgemmPacked 调用内部执行。

    fbgemm::PackAWithQuantRowOffset<uint8_t> packA(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr, // 目前，packA 管理 `pmat` 的所有权。
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        /*scale=*/q_params.scale,
        /*zero_pt=*/q_params.zero_point);
    // TODO: 考虑一种预分配和重用 pmat 缓冲区的方法。

    // 这是管道的末端，通过结果矩阵传递。
    fbgemm::DoNothing<float, float> doNothingObj{};
    // 使用范围迭代器遍历任务范围内的每个任务ID
    for (const auto task_id : c10::irange(begin, end)) {
      // 检查量化方案是否为每张量仿射量化
      if (q_scheme == c10::kPerTensorAffine) {
        // 处理每张量量化。
        //
        // 在执行 uint8 * int8 矩阵乘法后，此操作执行以下步骤：
        //  1) 向行和列添加偏移量。
        //  2) 将结果反量化为浮点数。
        //  3) 添加偏置项。
        fbgemm::ReQuantizeForFloat<ReluFused> outputProcObj(
            /*nextop=*/doNothingObj,  // 下一个操作对象
            /*Aq_scale=*/q_params.scale,  // A 的量化比例
            /*Bq_scale=*/w_scale.data(),  // B 的量化比例数组
            /*Aq_zero_point=*/q_params.zero_point,  // A 的零点
            /*Bq_zero_point=*/w_zp.data(),  // B 的零点数组
            /*row_offsets=*/packA.getRowOffsetBuffer(),  // 行偏移数组
            /*col_offsets=*/col_offsets.data(),  // 列偏移数组
            /*bias=*/bias_ptr,  // 偏置指针
            /*nCol=*/N);  // 列数

        // 执行 GEMM（通用矩阵乘法）
        fbgemm::fbgemmPacked(
            /*packA=*/packA,  // A 的打包对象
            /*packB=*/*packB,  // B 的打包对象
            /*C=*/output.data_ptr<float>(),  // 输出指针（浮点数）
            /*C_buffer=*/buffer.data_ptr<int32_t>(),  // 缓冲区指针（int32_t）
            /*ldc=*/N,  // 输出列数
            /*outProcess=*/outputProcObj,  // 输出处理对象
            /*thread_id=*/task_id,  // 线程 ID
            /*num_threads=*/num_tasks);  // 线程总数

      } else if (q_scheme == c10::kPerChannelAffine) {
        // 处理每通道仿射量化。
        //
        // 在执行 uint8 * int8 矩阵乘法后，此操作执行以下步骤：
        //  1) 向行和列添加偏移量。
        //  2) 将结果反量化为浮点数。
        //  3) 添加偏置项。
        fbgemm::ReQuantizeForFloat<
            ReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL>
            outputProcObj(
                /*nextop=*/doNothingObj,  // 下一个操作对象
                /*Aq_scale=*/q_params.scale,  // A 的量化比例
                /*Bq_scale=*/w_scale.data(),  // B 的量化比例数组
                /*Aq_zero_point=*/q_params.zero_point,  // A 的零点
                /*Bq_zero_point=*/w_zp.data(),  // B 的零点数组
                /*row_offsets=*/packA.getRowOffsetBuffer(),  // 行偏移数组
                /*col_offsets=*/col_offsets.data(),  // 列偏移数组
                /*bias=*/bias_ptr,  // 偏置指针
                /*nCol=*/N);  // 列数

        // 执行 GEMM（通用矩阵乘法）
        fbgemm::fbgemmPacked(
            /*packA=*/packA,  // A 的打包对象
            /*packB=*/*packB,  // B 的打包对象
            /*C=*/output.data_ptr<float>(),  // 输出指针（浮点数）
            /*C_buffer=*/buffer.data_ptr<int32_t>(),  // 缓冲区指针（int32_t）
            /*ldc=*/N,  // 输出列数
            /*outProcess=*/outputProcObj,  // 输出处理对象
            /*thread_id=*/task_id,  // 线程 ID
            /*num_threads=*/num_tasks);  // 线程总数
      }
    }
  });

  return output;  // 返回输出结果
#ifdef USE_FBGEMM


// 如果使用 FBGEMM 库，则编译以下代码块

#endif // USE_FBGEMM


template <bool ReluFused>
at::Tensor PackedLinearWeightsQnnp::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {


// 实现应用动态量化的函数模板 PackedLinearWeightsQnnp::apply_dynamic_impl

  if (reduce_range) {
    // 如果 reduce_range 参数为 true，则发出警告，因为当前 qnnpack 在这种情况下会被错误地忽略；这可能在未来的发布版本中改变。
    TORCH_WARN_ONCE("Currently, qnnpack incorrectly ignores reduce_range when it is set to true; this may change in a future release.");
  }

  // 检查输入张量的维度是否大于或等于 2
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");

  // 对输入张量进行连续性处理
  auto input_contig = input.contiguous();

  // 加锁，以确保权重打包操作的线程安全性
  std::lock_guard<std::mutex> lock(qnnp_mutex_);

  // 获取权重 w
  auto packB = w.get();

  // 计算权重的行数和列数
  size_t rows_w = bias_.size(0);
  size_t cols_w = input_contig.size(input_contig.dim() - 1);

  // 获取偏置向量
  at::Tensor bias_vec = bias_;

  // 检查偏置向量的维度是否为 1
  TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

  // 对偏置向量进行连续性处理
  auto bias_contig = bias_vec.contiguous();

  // 获取偏置向量的常量数据指针
  const float* bias_ptr = bias_contig.const_data_ptr<float>();

  // 计算输入张量的量化统计信息（最小值和最大值）
  float x_min;
  float x_max;
  if (input.numel() > 0) {
    x_min = input_contig.min().item<float>();
    x_max = input_contig.max().item<float>();
  } else {
    // 如果输入张量为空，则输出数据为空，因此使用任意的量化参数
    x_min = 0;
    x_max = 0;
  }

  // 选择输入张量的量化参数
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255);

  // 获取权重比例因子数据指针
  float* weight_scales_data = w_scales.data_ptr<float>();

  // 如果输入的量化尺度值不可用或者与当前的量化尺度值不同，则生成重新量化的比例
  if (!input_scale.has_value() || input_scale.value() != q_params.scale) {
    generate_requantization_scales(
        w_scales,
        q_params.scale,
        1.f,
        requantization_scales);
  }

  // 如果输入的量化尺度值不可用
  if (!input_scale.has_value()) {
    // 获取原始权重并将其调整为 uint8 类型（从 int8）
    auto weight_contig = orig_weight;

    // 分配仿射量化的张量，不论是否每通道量化，这个分配仅用于权重的打包，并将被释放。尽管如此，我们应该保持一致性。需要修复这个问题。
    // 为权重分配仿射量化张量
    Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8),
        weight_scales_data[0],
        w_zero_points[0]);
    // 获取 QNNPACK 权重数据的指针，使用 quint8 类型存储
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    // 获取 PyTorch 权重数据的指针，使用 qint8 类型存储
    int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
    // 获取权重张量的元素数量
    auto wt_numel = weight_contig.numel();
    // 将 qint8 类型的权重数据转换为 quint8 类型，并加上 128 进行偏移
    for (const auto i : c10::irange(wt_numel)) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }

    // 重置智能指针 w，为 qnnpack::PackBMatrix 创建新的实例
    w.reset();
    // 创建 PackBMatrix 对象，用于存储压缩后的权重数据
    w = std::make_unique<qnnpack::PackBMatrix>(
        cols_w /* input_channels */,
        rows_w /* output_channels */,
        w_zero_points.data(),
        requantization_scales.data(),
        (uint8_t*)qnnp_w_data,
        nullptr);
    // 获取指向 PackBMatrix 对象的指针
    packB = w.get();
    // 如果在预打包时释放权重，重置原始权重智能指针
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
      orig_weight.reset();
    }
  }

  // 更新输入的量化比例，以避免再次打包权重
  input_scale = q_params.scale;

  // 对输入进行量化
  Tensor q_input = at::quantize_per_tensor(
      input_contig, q_params.scale, q_params.zero_point, c10::kQUInt8);

  // 将输出尺寸调整为与输入的左侧维度相对应
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = rows_w;
  // 创建一个与输出尺寸相匹配的空张量
  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));

  // 计算输入张量的行数和列数
  size_t rows_input = 1;
  size_t cols_input = input_contig.size(input_contig.dim() - 1);
  for (const auto i : c10::irange(input_contig.dim() - 1)) {
    rows_input *= input_contig.size(i);
  }
  // 调用 QNNPACK 的线性运算函数 qnnpackLinearDynamic
  pytorch_qnnp_status runStatus = qnnpack::qnnpackLinearDynamic(
      rows_input /* batch_size */,
      cols_input /* input_channels */,
      rows_w /* output_channels */,
      q_input.q_zero_point(),
      w_zero_points.data(),
      /* for dynamic should really be called dequant scale */
      requantization_scales.data(),
      (uint8_t*)q_input.data_ptr<c10::quint8>(),
      cols_input /* input_stride */,
      packB->getPackedWeights(),
      bias_ptr,
      output.data_ptr<float>(),
      rows_w /* output_stride */,
      caffe2::pthreadpool_() /* threadpool */);

  // 断言 QNNPACK 线性操作是否成功运行
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Linear operator");

  // 如果启用了 ReluFused，对输出进行 ReLU 激活
  if (ReluFused) {
    output.relu_();
  }
  // 返回输出张量
  return output;
#ifdef USE_PYTORCH_QNNPACK

// 如果定义了 USE_PYTORCH_QNNPACK，编译以下代码块


template <bool ReluFused>
at::Tensor& PackedLinearWeightFp16::apply_dynamic_impl(
    const at::Tensor& input,
    at::Tensor& output) {

// 定义了一个模板函数 apply_dynamic_impl，接受输入张量和输出张量引用作为参数


const at::Tensor input_contig = input.contiguous();

// 创建一个连续存储的输入张量，确保数据在内存中连续


const float* input_ptr = input_contig.const_data_ptr<float>();

// 获取连续存储的输入张量的指针，指向其中的 float 数据


auto& packed_weight_fp16 = *w;

// 获取 fp16 格式的权重数据，存储在 packed_weight_fp16 中


TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
TORCH_CHECK(input.dim() >= 2);

// 检查输入张量的最后一个维度的大小与权重行数是否匹配，以及输入张量的维度是否大于等于 2


// NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
const int64_t N = packed_weight_fp16.numCols();

// 计算矩阵乘法的维度 M 和 N，其中 M 是输入张量最后一个维度的大小，N 是权重的列数


std::vector<int64_t> output_sizes = input.sizes().vec();
TORCH_CHECK(!output_sizes.empty())
output_sizes.back() = N;

// 创建输出张量的大小向量，确保其非空，并将最后一个维度大小设置为 N


// Resize output Tensor
output.resize_(output_sizes);

// 调整输出张量的大小为 output_sizes 中指定的大小


auto output_data = output.data_ptr<float>();

// 获取输出张量的数据指针，指向其中的 float 数据


int num_tasks = at::get_num_threads();
at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
  for (const auto task_id : c10::irange(begin, end)) {
    // Call the fp16 gemm interface
    fbgemm::cblas_gemm_compute(
        /*transa=*/fbgemm::matrix_op_t::NoTranspose,
        /*m=*/static_cast<int>(M),
        /*A=*/input_ptr,
        /*Bp=*/packed_weight_fp16,
        /*beta=*/0.0f,
        /*C=*/output_data,
        /*thread_id=*/static_cast<int>(task_id),
        /*num_threads=*/num_tasks);
  }
});

// 使用多线程并行执行 fp16 gemm 计算，将结果存储在 output_data 中


// Add bias term
if (bias_.has_value()) {
  TORCH_CHECK(bias_->dim() == 1);
  output.add_(*bias_);
}

// 如果存在偏置项，则将其加到输出张量上


return output;
}

// 返回处理后的输出张量引用


at::Tensor PackedLinearWeightFp16::apply_dynamic(
    at::Tensor input,
    bool /* reduce_range */) {
  at::Tensor output = at::empty({0}, input.options().dtype(at::kFloat));
  return apply_dynamic_impl</*ReluFused=*/false>(input, output);
}

// 应用动态计算，返回不带 ReLU 的输出张量


at::Tensor PackedLinearWeightFp16::apply_dynamic_relu(
    at::Tensor input,
    bool /* reduce_range */) {
  at::Tensor output = at::empty({0}, input.options().dtype(at::kFloat));
  return apply_dynamic_impl</*ReluFused=*/true>(input, output);
}

// 应用动态计算，返回带 ReLU 的输出张量


at::Tensor& PackedLinearWeightFp16::apply_dynamic_out(
    const at::Tensor& input,
    at::Tensor& output,
    bool /* reduce_range */) {
  TORCH_CHECK((output.device() == c10::kCPU) && (output.dtype() == at::kFloat));
  return apply_dynamic_impl<false>(input, output);
}

// 将动态计算结果存储到预先分配的输出张量中，不带 ReLU


at::Tensor& PackedLinearWeightFp16::apply_dynamic_relu_out(
    const at::Tensor& input,
    at::Tensor& output,
    bool /* reduce_range */) {
  TORCH_CHECK((output.device() == c10::kCPU) && (output.dtype() == at::kFloat));
  return apply_dynamic_impl<true>(input, output);
}

// 将动态计算结果存储到预先分配的输出张量中，带 ReLU


#endif // USE_FBGEMM

// 如果定义了 USE_FBGEMM，编译以下代码块
#if AT_MKLDNN_ENABLED()
template <bool ReluFused>
at::Tensor PackedLinearWeightsOnednn::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {
  // Dynamic: fp32 * int8 -> fp32
  using at::Tensor;

  // 检查输入张量的维度是否至少为2
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // 检查输入张量的数据类型是否为float
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float,
      "qlinear_dynamic (ONEDNN): data type of input should be float.");

  // 将输入张量转为连续存储
  auto input_contig = input.contiguous();
  const int64_t dim = input.dim();
  // 如果输入张量的维度是2，则不改变形状，否则将其形状改为[-1, 最后一个维度大小]
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});
  auto input_dims = input_reshaped.sizes().vec();
  // 定义输入数据类型为f32
  auto input_data_type = dnnl::memory::data_type::f32;
  // 创建用于描述输入张量的ideep::tensor::desc对象
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  // 根据是否融合ReLU选择操作属性
  ideep::attr_t op_attr = ReluFused ? ideep::attr_t::fuse_relu() : ideep::attr_t();
  // 初始化ideep::tensor对象x，使用input_contig的数据指针
  ideep::tensor x;
  x.init(input_desc, input_contig.data_ptr());

  // 查找量化参数
  float x_max = 0, x_min = 0;
#ifdef USE_FBGEMM
  // 如果定义了USE_FBGEMM，使用FBGEMM的FindMinMax函数查找最大值和最小值
  fbgemm::FindMinMax(
      /*m=*/input_contig.data_ptr<float>(),
      /*min=*/&x_min,
      /*max=*/&x_max,
      /*len=*/input.numel());
#else
  // 否则，使用PyTorch的aminmax函数查找最大值和最小值
  if (input_contig.numel() > 0) {
    auto [t_min, t_max] = at::aminmax(input_contig);
    x_max = t_max.item<float>();
    x_min = t_min.item<float>();
  }
#endif

  // 定义量化精度为8位
  const int precision = 8;
  // 选择量化参数
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/(1 << precision) - 1,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  // 创建源零点的向量，仅包含一个元素，即量化参数的零点
  const std::vector<int32_t>& src_zero_point = std::vector<int32_t>(1, q_params.zero_point);

  // 获取权重
  auto w = *(weight_.get());
  // 定义目标张量的维度
  auto dst_dims = {x.get_dim(0), w.get_dim(1)};
  // 定义源缩放因子
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/q_params.scale);
  // 获取权重的缩放因子
  const ideep::scale_t& weights_scales = w.get_scale();

  // 分配输出张量
  at::Tensor output = at::empty(dst_dims, input.options().dtype(at::kFloat));
  // 如果输出张量的元素个数为0，则直接返回空张量
  if (output.numel() == 0) return output;

  // 初始化ideep::tensor对象y，使用output的数据指针和维度信息
  ideep::tensor y({dst_dims, ideep::tensor::data_type::f32,
                   {output.strides().cbegin(), output.strides().cend()}},
                  output.data_ptr());

  // 判断是否存在偏置项
  bool with_bias = bias_.has_value();
  if (with_bias) {
    // 如果偏置项可能在外部被修改，更新预装的偏置项
    if (bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
      bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
  }
  }
  // 如果启用偏置，则使用偏置值；否则使用空的ideep::tensor()
  const auto& b = with_bias ? bias_.value() : ideep::tensor();
  // 当首次调用时初始化原语缓存，之后不再更新
  int num_threads = at::get_num_threads();
  // 创建原语缓存的键，包括量化参数、输入维度、线程数等信息
  PrimitiveCacheKey cache_key = std::make_tuple(
      q_params.scale, q_params.zero_point, input_dims, 1.0, 0, num_threads, /*accum scale*/1.0, /*accum zero point*/0);
  c10::call_once(*cache_initialized_flag, [&](){
      // 在原语缓存首次初始化时，准备线性参数
      LinearParams params;
      ideep::matmul_forward::prepare</*is_dynamic=*/true>(
          params, x, w, b, y,
          src_scales, weights_scales, ideep::scale_t(),
          src_zero_point, ideep::zero_point_t(), 1.0f, 1.0f, op_attr);
      // 将参数与缓存键结合，存入原语缓存
      get_cache() = LinearPrimitiveCache(cache_key, params);
      // 重新调整权重张量的顺序以匹配参数描述
      w = w.reorder_if_differ_in(params.pd.weights_desc());
  });
  // 如果命中原语缓存，则直接使用缓存中的参数进行计算
  if (get_cache().hit_dynamic(cache_key)) {
    LinearParams& params = get_cache().get_param();
    ideep::matmul_forward::compute(params, x, w, b, y, src_scales, src_zero_point);
  } else {
    // 否则，直接进行矩阵乘法前向计算
    ideep::matmul_forward::compute(x, w, b, y,
                                   src_scales, weights_scales, ideep::scale_t(),
                                   src_zero_point, ideep::zero_point_t(),
                                   1.0f, 1.0f, op_attr);
  }
  // 获取输入张量的维度，并调整输出张量的最后一个维度以匹配权重张量的第二个维度
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = w.get_dim(1);
  // 如果输出张量的尺寸与调整后的尺寸相匹配，则直接返回输出张量
  if (output.sizes().vec() == out_sizes)
    return output;
  // 否则，按照调整后的尺寸重新塑形输出张量并返回
  return output.reshape(out_sizes);
    // 如果使用 FBGEMM 库，则执行下面的代码块
#ifdef USE_FBGEMM
    // 检查当前 CPU 是否支持 FBGEMM，如果不支持则抛出错误信息
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

    // 检查权重张量的维度是否为2，即二维张量
    TORCH_CHECK(
        weight.dim() == 2,
        "The dimension of weight tensor should be equal to 2");

    // 使用预包装函数将权重和偏置进行打包
    auto packed_weight = PackedLinearWeightFp16::prepack(weight, bias);

    // 对输入张量应用动态量化线性变换操作，并将结果保存到输出张量
    auto output = packed_weight->apply_dynamic(std::move(input));

    // 如果 ReluFused 为 true，则在输出张量上执行 relu 操作
    if (ReluFused) {
      output.relu_();
    }
    // 返回处理后的输出张量
    return output;

// 如果未使用 FBGEMM 库，则执行下面的代码块
#else // USE_FBGEMM
    // 我们确保使用这些运算符的模型在不同机器上具有相同的数值计算结果。
    // 因此，如果没有使用 FBGEMM 操作符，我们不提供回退路径，并且在无法运行 FBGEMM 时会直接报错。
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}
    // 返回计算结果的张量
    return output;
  }

  // 返回输出张量的元信息，根据输入张量和给定的权重、偏置计算
  static at::Tensor meta(
      at::Tensor input,                             // 输入张量
      const at::Tensor& weight,                     // 权重张量（常量引用）
      const at::Tensor& bias) {                     // 偏置张量（常量引用）
    // 我们强烈保证使用这些运算符的模型在不同机器上具有相同的数值。因此，
    // 如果不能运行 FBGEMM，我们不提供备用路径，而是直接失败报错。
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

    // 检查权重张量的维度是否为2
    TORCH_CHECK(
        weight.dim() == 2,
        "The dimension of weight tensor should be equal to 2");

    // 获取权重张量的输出通道数
    auto out_channel = weight.sym_sizes().vec()[0];
    // 获取输入张量的符号化尺寸，并修改最后一个维度为输出通道数
    auto out_sizes = input.sym_sizes().vec();
    out_sizes[out_sizes.size() - 1] = out_channel;

    // 返回一个符号化整数张量，其尺寸与修改后的输入张量尺寸相同，使用与输入张量相同的选项
    return at::empty_symint(out_sizes, input.options());
  }
#else // USE_FBGEMM
// 如果未定义 USE_FBGEMM，则以下是对应的静态方法，用于在 FBGEMM 不可用时报错
static at::Tensor run(
    at::Tensor /* input */,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  // 我们承诺使用这些运算符的模型在不同机器上具有相同的数值。因此，如果无法运行 FBGEMM，则不提供备用路径，而是会有明确的失败提示。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

// 对应的 meta 方法，在 FBGEMM 不可用时报错
static at::Tensor meta(
    at::Tensor /* input */,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}
#endif // USE_FBGEMM

// 如果定义了 USE_FBGEMM，则以下是对应的函数封装，用于调用 fbgemm_pack_gemm_matrix_fp16 函数
at::Tensor wrapped_fbgemm_pack_gemm_matrix_fp16(const at::Tensor& weight) {
#ifdef USE_FBGEMM
  // 检查 weight 的维度是否为 2，因为 fbgemm weight packing 只能处理矩阵而非向量。
  TORCH_CHECK(
      weight.dim() == 2,
      "fbgemm weight packing only packs matrices not vectors.");
  // 调用 fbgemm_pack_gemm_matrix_fp16 函数进行矩阵打包
  return at::native::fbgemm_pack_gemm_matrix_fp16(weight);
#else // USE_FBGEMM
  // 如果未定义 USE_FBGEMM，则报错
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

// 如果定义了 USE_FBGEMM，则以下是对应的函数封装，用于返回一个空的字节张量作为元数据
at::Tensor wrapped_fbgemm_pack_gemm_matrix_fp16_meta(const at::Tensor& weight) {
#ifdef USE_FBGEMM
  // 严格来说，这不正确。但我们无法知道打包矩阵的确切大小，因为它由对象本身维护，所以我们返回这里的视图。
  return at::empty({8}, weight.options().dtype(at::kByte));
#else // USE_FBGEMM
  // 如果未定义 USE_FBGEMM，则报错
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

// 如果定义了 USE_FBGEMM，则以下是对应的函数封装，用于调用 fbgemm_linear_fp16_weight 函数
at::Tensor wrapped_fbgemm_linear_fp16_weight(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, int64_t out_channel) {
#ifdef USE_FBGEMM
  // 调用 fbgemm_linear_fp16_weight 函数
  return at::native::fbgemm_linear_fp16_weight(input, weight, bias);
#else // USE_FBGEMM
  // 如果未定义 USE_FBGEMM，则报错
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

// 如果定义了 USE_FBGEMM，则以下是对应的函数封装，用于返回一个空的符号整数张量作为元数据
at::Tensor wrapped_fbgemm_linear_fp16_weight_meta(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, int64_t out_channel) {
#ifdef USE_FBGEMM
  // 对于 meta 函数，需要用户显式提供维度，因为我们无法访问 weight。
  auto out_sizes = input.sym_sizes().vec();
  if (out_channel == -1) {
    out_sizes.pop_back();
  } else {
    out_sizes.back() = out_channel;
  }
  // 返回一个根据输入形状和选项创建的空符号整数张量
  return at::empty_symint(out_sizes, input.options());
#else // USE_FBGEMM
  // 如果未定义 USE_FBGEMM，则报错
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}
// 定义 Torch 库的 quantized 分支的实现，针对 CPU
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // 注册线性模型的参数
  register_linear_params();
  // 实现 quantized::linear_dynamic 方法，使用 QLinearDynamicInt8<false>::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic"),
      TORCH_FN(QLinearDynamicInt8<false>::run));
  // 实现 quantized::linear_relu_dynamic 方法，使用 QLinearDynamicInt8<true>::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic"),
      TORCH_FN(QLinearDynamicInt8<true>::run));
  // 实现 quantized::linear_dynamic_fp16 方法，使用 QLinearDynamicFp16<false>::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16"),
      TORCH_FN(QLinearDynamicFp16<false>::run));
  // 实现 quantized::linear_dynamic_fp16_unpacked_weight 方法，使用 QLinearUnpackedDynamicFp16::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16_unpacked_weight"),
      TORCH_FN(QLinearUnpackedDynamicFp16::run));
  // 实现 quantized::linear_relu_dynamic_fp16 方法，使用 QLinearDynamicFp16<true>::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic_fp16"),
      TORCH_FN(QLinearDynamicFp16<true>::run));
}

// 定义 Torch 库的 quantized 分支的 Meta 实现
TORCH_LIBRARY_IMPL(quantized, Meta, m) {
  // 实现 quantized::linear_dynamic_fp16_unpacked_weight 方法的元函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16_unpacked_weight"),
      TORCH_FN(QLinearUnpackedDynamicFp16::meta));
}

// 定义 Torch 库的 _quantized 分支的实现，针对 CPU
TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  // 注册线性模型的参数
  register_linear_params();
  // 实现 _quantized::linear_dynamic 方法，使用 QLinearDynamicInt8<false>::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::linear_dynamic"),
      TORCH_FN(QLinearDynamicInt8<false>::run));
  // 实现 _quantized::wrapped_fbgemm_pack_gemm_matrix_fp16 方法，使用 wrapped_fbgemm_pack_gemm_matrix_fp16 函数
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16"),
      wrapped_fbgemm_pack_gemm_matrix_fp16);
  // 实现 _quantized::wrapped_fbgemm_linear_fp16_weight 方法，使用 wrapped_fbgemm_linear_fp16_weight 函数
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_linear_fp16_weight"),
      wrapped_fbgemm_linear_fp16_weight);
}

// 定义 Torch 库的 _quantized 分支的 Meta 实现
TORCH_LIBRARY_IMPL(_quantized, Meta, m) {
  // 实现 _quantized::wrapped_fbgemm_pack_gemm_matrix_fp16 方法的元函数
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16"),
      wrapped_fbgemm_pack_gemm_matrix_fp16_meta);
  // 实现 _quantized::wrapped_fbgemm_linear_fp16_weight 方法的元函数
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_linear_fp16_weight"),
      wrapped_fbgemm_linear_fp16_weight_meta);
}

// 结束 namespace 声明
} // namespace
} // namespace native
} // namespace at
```