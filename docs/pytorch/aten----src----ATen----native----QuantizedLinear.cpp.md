# `.\pytorch\aten\src\ATen\native\QuantizedLinear.cpp`

```py
// 定义编译选项，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含标准库头文件
#include <vector>

// 包含 ATen 库中的必要头文件
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>

// 根据不同的编译选项选择不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_native.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_native.h>
#include <ATen/ops/fbgemm_linear_int8_weight_fp32_activation_native.h>
#include <ATen/ops/fbgemm_linear_int8_weight_native.h>
#include <ATen/ops/fbgemm_linear_quantize_weight_native.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16_native.h>
#include <ATen/ops/fbgemm_pack_quantized_matrix_native.h>
#endif

// 包含 C10 实用工具中的 irange.h 头文件
#include <c10/util/irange.h>

// 如果定义了 USE_FBGEMM，则包含相关的 FBGEMM 头文件
#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtils.h>
#endif // USE_FBGEMM

// 声明 caffe2 命名空间，并注册一些特定类型
namespace caffe2 {
CAFFE_KNOWN_TYPE(c10::intrusive_ptr<LinearPackedParamsBase>);
} // namespace caffe2

// 在使用 FBGEMM 时，注册进一步的类型
#ifdef USE_FBGEMM
namespace caffe2 {
CAFFE_KNOWN_TYPE(fbgemm::PackBMatrix<int8_t>);
CAFFE_KNOWN_TYPE(c10::intrusive_ptr<PackedLinearWeightFp16>);
} // namespace caffe2
#endif // USE_FBGEMM

// 定义 ATen 库的 native 命名空间
namespace at::native {

// 在使用 FBGEMM 时定义函数 fbgemm_linear_int8_weight_fp32_activation
Tensor fbgemm_linear_int8_weight_fp32_activation(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& packed,
    const Tensor& col_offsets,
    const Scalar& weight_scale,
    const Scalar& weight_zero_point,
    // 这个操作的作用：
    // 1) 根据预先计算的统计信息对输入矩阵进行量化
    // 2) 创建一个“行缓冲”向量，其中包含必须添加到整数矩阵乘法操作中的偏移值，以确保正确性
    // 3) 将结果的量化矩阵打包为向量寄存器和缓存友好的块
    //
    // 注意：这些步骤并非立即执行，而是在下面的 fbgemmPacked 调用中执行
    fbgemm::PackAWithQuantRowOffset<uint8_t> pack_a(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr, // pack_a 管理 `pmat` 的所有权
        /*scale=*/q_params.scale,
        /*zero_pt=*/q_params.zero_point);

// 这是流水线的最后阶段，通过这个对象传递结果矩阵
fbgemm::DoNothing<float, float> kDoNothingObj{};
    for (const auto task_id : c10::irange(begin, end)) {
      // 对于给定范围内的每个任务ID，执行以下操作：
      //  1) 矩阵乘法完成后，将uint8 * int8结果矩阵乘积执行以下操作：
      //     a) 向行和列添加偏移量
      //     b) 对结果进行反量化为浮点数
      //     c) 添加偏置项
      fbgemm::ReQuantizeForFloat</* FUSE_RELU */ false> output_proc_obj(
          /*nextop=*/kDoNothingObj,
          /*Aq_scale=*/q_params.scale,
          /*Bq_scale=*/&weight_scale_float,
          /*Aq_zero_point=*/q_params.zero_point,
          /*Bq_zero_point=*/&weight_zero_point_int32,
          /*row_offsets=*/pack_a.getRowOffsetBuffer(),
          /*col_offsets=*/col_offsets_data,
          /*bias=*/bias_contig_data,
          /*nCol=*/N);
      // 执行矩阵乘法
      fbgemm::fbgemmPacked(
          /*packA=*/pack_a,
          /*packB=*/pack_b,
          /*C=*/output.data_ptr<float>(),
          /*C_buffer=*/buffer.data_ptr<int32_t>(),
          /*ldc=*/N,
          /*outProcess=*/output_proc_obj,
          /*thread_id=*/task_id,
          /*num_threads=*/num_tasks);
    }
  });

  // 返回处理后的输出数据
  return output;
}

Tensor fbgemm_linear_int8_weight(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& packed,
    const Tensor& col_offsets,
    const Scalar& weight_scale,
    const Scalar& weight_zero_point,
    const Tensor& bias) {
  // 调用 fbGEMM 库中的整数8位加权线性函数，返回加权结果
  return at::native::fbgemm_linear_int8_weight_fp32_activation(
      input,
      weight,
      packed,
      col_offsets,
      weight_scale,
      weight_zero_point,
      bias);
}

namespace {

// 计算列偏移量
// 注意这包括列的总和以及标量项 B_zero_point * K，
// 而由 PackAWithQuantRowOffset 创建的 row_offsets 仅包括 A 行的总和。
void CalcColOffsetsTranspose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t B_zero_point,
    int32_t* col_offsets) {
  // 遍历每一列
  for (const auto i : c10::irange(N)) {
    int32_t sum = 0;
    // 对当前列中的每个元素求和
    for (const auto j : c10::irange(K)) {
      sum += Bint8[i * K + j];
    }
    // 计算并存储列偏移量，减去 B_zero_point 乘以 K
    col_offsets[i] = sum - B_zero_point * K;
  }
}

} // namespace

std::tuple<Tensor, Tensor, double, int64_t> fbgemm_linear_quantize_weight(
    // 函数签名，声明一个线性量化权重的函数，接受一个权重张量作为参数
    const Tensor& weight) {
  // 发出警告，提醒函数已经废弃，并将在未来的PyTorch版本中移除
  TORCH_WARN_ONCE("fbgemm_linear_quantize_weight is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们强烈保证使用这些运算符的模型在不同机器上具有相同的数值计算结果。
  // 因此，如果我们无法运行FBGEMM，我们不提供回退路径，而是直接报错。
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
  // 将权重张量转换为连续的张量
  const Tensor weight_contig = weight.contiguous();

  // 计算权重的统计信息
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float w_min;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float w_max;
  // 调用FBGEMM库的函数，计算权重张量的最小值和最大值
  fbgemm::FindMinMax(
      /*m=*/weight_contig.data_ptr<float>(),
      /*min=*/&w_min,
      /*max=*/&w_max,
      /*len=*/weight_contig.numel());

  // 选择将权重量化为8位有符号整数的参数
  constexpr bool kIsSigned = true;
  constexpr int kPrecision = 8;
  constexpr int kBound = (1 << (kPrecision - 1));
  // 调用FBGEMM库的函数，选择量化参数
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/w_min,
      /*max=*/w_max,
      /*qmin=*/kIsSigned ? -kBound : 0,
      /*qmax=*/kIsSigned ? (kBound - 1) : (1 << kPrecision) - 1,
      /*preserve_sparsity=*/false);
  // 设置量化的精度
  q_params.precision = kPrecision;

  // 创建一个与weight_contig具有相同大小和类型的空张量
  Tensor quantized = at::native::empty_like(
      weight_contig,
      at::kChar,
      weight_contig.options().layout_opt(),
      weight_contig.options().device_opt(),
      weight_contig.options().pinned_memory_opt(),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  
  // 使用FBGEMM库的函数，将浮点数权重量化为8位整数，并存储在quantized张量中
  fbgemm::Quantize<int8_t, false /*LEGACY*/>(
      /*src=*/weight_contig.data_ptr<float>(),
      /*dst=*/quantized.data_ptr<int8_t>(),
      /*len=*/weight_contig.numel(),
      /*qparams=*/q_params);

  // 计算权重的列偏移，并将其存储在一个张量中，以备后用
  // 类似于量化过程，这可以执行一次并进行缓存
  Tensor col_offsets = at::empty(
      {weight_contig.size(0)},
      at::kInt,
      weight_contig.options().layout_opt(),
      weight_contig.options().device_opt(),
      weight_contig.options().pinned_memory_opt(),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用一个函数，计算权重的列偏移，并将结果存储在col_offsets张量中
  CalcColOffsetsTranspose(
      /*K=*/quantized.size(1),
      /*N=*/quantized.size(0),
      /*Bint8=*/quantized.data_ptr<int8_t>(),
      /*B_zero_point=*/q_params.zero_point,
      /*col_offsets=*/col_offsets.data_ptr<int32_t>());

  // 返回一个元组，包含量化后的权重、列偏移、量化参数的比例和零点
  return std::make_tuple(
      quantized, col_offsets, q_params.scale, q_params.zero_point);
}

// 函数：将输入的权重张量打包成量化矩阵的FBGEMM表示
Tensor fbgemm_pack_quantized_matrix(const Tensor& weight) {
  // 发出警告：fbgemm_pack_quantized_matrix已经被弃用，并将在未来的PyTorch版本中移除
  TORCH_WARN_ONCE("fbgemm_pack_quantized_matrix is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们保证使用这些操作符的模型在不同机器上具有相同的数值结果。因此，如果我们无法运行FBGEMM，就不提供回退路径，而是直接报错。
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
  
  // 获取权重张量的维度大小
  const int64_t K = weight.size(1);
  const int64_t N = weight.size(0);

  // 将权重张量转换为连续的Tensor
  const Tensor weight_contig = weight.contiguous();

  // 获取指向权重数据的指针
  const int8_t* weight_ptr = weight_contig.const_data_ptr<int8_t>();

  // 创建fbgemm::PackBMatrix对象，将权重数据打包为FBGEMM所需的格式
  auto ptr = std::make_unique<fbgemm::PackBMatrix<int8_t>>(
      /*trans=*/fbgemm::matrix_op_t::Transpose,
      /*nRow=*/K,
      /*nCol=*/N,
      /*smat=*/weight_ptr,
      /*ld=*/K,
      /*pmat=*/nullptr, // PackBMatrix管理pmat的所有权
      /*groups=*/1);

  // 使用cpp_custom_type_hack::create创建一个自定义类型的Tensor并返回
  return cpp_custom_type_hack::create(std::move(ptr), weight.options());
}

// 函数：将输入的权重张量打包成量化矩阵的FBGEMM表示（重载版本）
Tensor fbgemm_pack_quantized_matrix(
    const Tensor& weight,
    int64_t K,
    int64_t N) {
  // 当https://github.com/pytorch/pytorch/issues/24354问题解决后，替换此处的警告
  // TORCH_WARN(
  //     "fbgemm_pack_quantized_matrix(weight, K, N) will be deprecated soon."
  //     "Please use fbgemm_pack_quantized_matrix(weight) instead.");
  
  // 调用不带K和N参数的fbgemm_pack_quantized_matrix函数
  return at::native::fbgemm_pack_quantized_matrix(weight);
}

// 匿名命名空间：将16位无符号整数表示的原始浮点数转换为半精度浮点数
float RawUint16ToFp16(unsigned short value) {
  // 将原始的16位半精度浮点数转换为32位单精度浮点数
  const unsigned short sign_bits = value >> 15;
  const unsigned short exponent_bits = value >> 10 & 0x1f;
  const unsigned short significand_bits = value & 0x3ff;

  const float sign = sign_bits ? -1 : 1;
  const float significand =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const float exponent = exponent_bits - 0xf;

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return sign * std::ldexp(significand, exponent);
}

// 模板函数：检查元素是否超出指定最大值并饱和处理
template <typename T>
bool CheckAndSaturate(T max_val, T* element) {
  if (*element > max_val) {
    *element = max_val;
    return true;
  }
  if (*element < -max_val) {
    *element = -max_val;
    return true;
  }
  return false;
}

// 函数：处理权重值的饱和范围，确保在FP16量化的有效范围内
void HandleWeightsSaturation(int64_t N, float* weight) {
  const float kFp16Max = RawUint16ToFp16(0x7BFF);
  bool found_out_of_range = false;
  for (const auto i : c10::irange(N)) {
    # 如果使用 CheckAndSaturate 函数检测到 weight + i 大于 kFp16Max，则执行以下代码块
    if (CheckAndSaturate<float>(kFp16Max, weight + i)) {
      # 如果检测到 weight + i 超出范围，将 found_out_of_range 设为 true
      found_out_of_range = true;
    }
  }
  # 如果 found_out_of_range 为 true，表示发现超出范围的 weight
  if (found_out_of_range) {
    # 输出警告信息，指示找到超出范围的 weight
    TORCH_WARN("FOUND weight out of range ");
  }
}

} // namespace

// 将输入权重张量打包成半精度浮点数的 GEMM 矩阵
Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  // 发出警告，提示函数已过时，并将在未来的 PyTorch 版本中移除
  TORCH_WARN_ONCE("fbgemm_pack_gemm_matrix_fp16 is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们承诺使用这些运算符的模型在不同的机器上具有相同的数值计算结果。
  // 因此，如果不能运行 FBGEMM，我们不提供备用路径，而是会有明确的错误提示。
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  // 获取权重张量的维度信息
  const int64_t K = weight.size(1);
  const int64_t N = weight.size(0);
  // 创建权重张量的连续副本
  Tensor weight_contig = weight.contiguous();
  // 获取连续副本的数据指针，指向其数据的起始位置
  float* weight_contig_ptr = weight_contig.data_ptr<float>();
  // 对权重数据进行饱和处理
  HandleWeightsSaturation(K * N, weight_contig_ptr);

  // TODO(mingzhe09088):
  // 在此处考虑使用一个函数对象，用于 PackedGemmMatrixFP16
  // (XQ) 的注释：并不确定这里的 make_unique 是否安全。make_unique 是用常规的 "new" 创建的，
  // 并且在此函数中通过 TypeMetaData::deleteFn 释放。如果张量在这个翻译单元内创建并释放，
  // 这是完全安全的。如果该张量跨越动态链接库边界，可能会有很大问题。
  auto ptr = std::make_unique<fbgemm::PackedGemmMatrixFP16>(
      fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr);
  // 创建包装了 PackedGemmMatrixFP16 的智能指针
  c10::intrusive_ptr<LinearPackedParamsBase> packed_weight =
      c10::make_intrusive<PackedLinearWeightFp16>(std::move(ptr), c10::nullopt);
  // 创建包装了 packed_weight 的 unique_ptr 包装器
  auto unique_ptr_wrapper =
      std::make_unique<decltype(packed_weight)>(std::move(packed_weight));
  // 创建一个自定义类型的张量，使用 cpp_custom_type_hack::create 方法
  return cpp_custom_type_hack::create(
      std::move(unique_ptr_wrapper), weight.options());
}

// 使用半精度浮点数的权重和激活函数执行线性运算
Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    // 发出一次性警告，指示该函数即将被弃用并将在将来的PyTorch版本中移除
    TORCH_WARN_ONCE("fbgemm_linear_fp16_weight_fp32_activation is deprecated "
                    "and will be removed in a future PyTorch release.")
    
    // 我们强烈保证使用这些操作符的模型在不同机器上具有相同的数值计算结果。因此，如果我们无法运行FBGEMM，则不提供备用路径，而是直接失败。
    TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
    
    // 通过调用contiguous()方法，确保输入张量input是连续存储的
    const Tensor input_contig = input.contiguous();
    // 获取input张量的float类型数据指针
    const float* input_ptr = input_contig.const_data_ptr<float>();
    
    // 从拥有的张量中提取出PackedGemmMatrixFP16实例，该张量包含了压缩的FP16权重矩阵
    const fbgemm::PackedGemmMatrixFP16& packed_weight_fp16 =
        *c10::dynamic_intrusive_pointer_cast<PackedLinearWeightFp16>(
             cpp_custom_type_hack::cast<
                 c10::intrusive_ptr<LinearPackedParamsBase>>(packed_weight))
             ->w;
    
    // 检查输入张量的最后一个维度的大小与压缩权重矩阵的行数相匹配
    TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
    // 检查输入张量的维度至少为2
    TORCH_CHECK(input.dim() >= 2);
    // 检查偏置项张量的维度为1
    TORCH_CHECK(bias.dim() == 1);
    
    // 计算M和N的值，其中M是输入张量最后一个维度之前的所有维度的乘积，N是压缩权重矩阵的列数
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
    const int64_t N = packed_weight_fp16.numCols();
    // 创建一个输出张量，形状与输入张量相同，但最后一个维度改为N，并且数据类型为float
    std::vector<int64_t> output_size = input.sizes().vec();
    output_size.back() = N;
    Tensor output = at::empty(output_size, input.options().dtype(at::kFloat));
    
    // 调用fp16 gemm接口进行矩阵乘法计算，计算结果存储在输出张量中
    fbgemm::cblas_gemm_compute(
        fbgemm::matrix_op_t::NoTranspose,
        M,
        input_ptr,
        packed_weight_fp16,
        0.0f,
        output.data_ptr<float>());
    
    // 添加偏置项到输出张量中
    output.add_(bias);
    
    // 返回计算结果的输出张量
    return output;
}

// 如果使用了 FBGEMM，调用对应的 fp16 权重计算函数
Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  return at::native::fbgemm_linear_fp16_weight_fp32_activation(
      input, packed_weight, bias);
}

#else // 如果没有使用 FBGEMM

// 这个函数是被废弃的，用于计算带有 fp32 激活的 int8 权重
Tensor fbgemm_linear_int8_weight_fp32_activation(
    const Tensor& /*input*/,
    const Tensor& /*weight*/,
    const Tensor& /*packed*/,
    const Tensor& /*col_offsets*/,
    const Scalar& /*weight_scale*/,
    const Scalar& /*weight_zero_point*/,
    const Tensor& /*bias*/) {
  TORCH_WARN_ONCE("fbgemm_linear_int8_weight_fp32_activation is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们强烈保证使用这些运算符的模型在不同机器上具有相同的数值。因此，如果无法运行 FBGEMM，则会失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

// 这个函数是被废弃的，用于计算 int8 权重
Tensor fbgemm_linear_int8_weight(
    const Tensor& /*input*/,
    const Tensor& /*weight*/,
    const Tensor& /*packed*/,
    const Tensor& /*col_offsets*/,
    const Scalar& /*weight_scale*/,
    const Scalar& /*weight_zero_point*/,
    const Tensor& /*bias*/) {
  TORCH_WARN_ONCE("fbgemm_linear_int8_weight is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们强烈保证使用这些运算符的模型在不同机器上具有相同的数值。因此，如果无法运行 FBGEMM，则会失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

// 这个函数是被废弃的，用于量化权重
std::tuple<Tensor, Tensor, double, int64_t> fbgemm_linear_quantize_weight(
    const Tensor& /*weight*/) {
  TORCH_WARN_ONCE("fbgemm_linear_quantize_weight is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们强烈保证使用这些运算符的模型在不同机器上具有相同的数值。因此，如果无法运行 FBGEMM，则会失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

// 这个函数是被废弃的，用于打包量化矩阵
Tensor fbgemm_pack_quantized_matrix(const Tensor& /*input*/) {
  TORCH_WARN_ONCE("fbgemm_pack_quantized_matrix is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们强烈保证使用这些运算符的模型在不同机器上具有相同的数值。因此，如果无法运行 FBGEMM，则会失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

// 这个函数是被废弃的，用于打包量化矩阵
Tensor fbgemm_pack_quantized_matrix(
    const Tensor& /*input*/,
    int64_t /*K*/,
    int64_t /*N*/) {


    // 函数签名，这里的 `int64_t` 可能是函数的返回类型或参数类型，但需要完整的函数定义才能确认具体含义。
    // 此处代码可能有缺失，因为函数名和函数体并未完全展示。


  TORCH_WARN_ONCE("fbgemm_pack_quantized_matrix is deprecated "
                  "and will be removed in a future PyTorch release.")


  // 发出一次性警告，指出`fbgemm_pack_quantized_matrix`函数已废弃
  // 并将在未来的PyTorch版本中移除。


  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");


  // 我们强烈保证使用这些操作符的模型在不同机器上具有相同的数值特性。
  // 因此，如果我们无法运行FBGEMM，我们不会提供备用路径，而是会大声失败。
  // 使用TORCH_CHECK宏来检查条件是否为假，如果条件为假，输出给定的错误消息。
}

// 结束 namespace at::native

Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  // 发出一次性警告，指出 fbgemm_pack_gemm_matrix_fp16 函数已经废弃
  TORCH_WARN_ONCE("fbgemm_pack_gemm_matrix_fp16 is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们承诺，使用这些操作符的模型在不同机器上将具有相同的数值计算结果。
  // 因此，如果没有 FBGEMM 操作符可用，我们不提供备选方案，而是会明确失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  // 发出一次性警告，指出 fbgemm_linear_fp16_weight_fp32_activation 函数已经废弃
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight_fp32_activation is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们承诺，使用这些操作符的模型在不同机器上将具有相同的数值计算结果。
  // 因此，如果没有 FBGEMM 操作符可用，我们不提供备选方案，而是会明确失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  // 发出一次性警告，指出 fbgemm_linear_fp16_weight 函数已经废弃
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight is deprecated "
                  "and will be removed in a future PyTorch release.")

  // 我们承诺，使用这些操作符的模型在不同机器上将具有相同的数值计算结果。
  // 因此，如果没有 FBGEMM 操作符可用，我们不提供备选方案，而是会明确失败。
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

#endif // USE_FBGEMM

} // 结束 namespace at::native


这些注释提供了对每个函数的解释，包括函数废弃的警告以及失败时的错误信息说明。
```