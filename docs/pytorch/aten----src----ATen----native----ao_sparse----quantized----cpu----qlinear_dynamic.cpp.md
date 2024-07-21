# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qlinear_dynamic.cpp`

```py
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/empty.h>
#endif



// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含 <ATen/Functions.h>
// 否则，包含 <ATen/ops/quantize_per_tensor.h> 和 <ATen/ops/empty.h>



namespace ao {
namespace sparse {



// 进入 ao::sparse 命名空间



int register_linear_params();



// 声明一个名为 register_linear_params 的整型函数，返回类型为 int



#ifdef USE_PYTORCH_QNNPACK



// 如果定义了 USE_PYTORCH_QNNPACK，则编译以下内容



template <>
at::Tensor PackedLinearWeightQnnp::apply_dynamic_impl<true>(
    const at::Tensor& input) {



// 特化模板 PackedLinearWeightQnnp::apply_dynamic_impl<true>，处理动态输入
// 使用 qnnpack 后端时，不支持稀疏量化动态线性加 ReLU 操作



  TORCH_INTERNAL_ASSERT(
      false,
      "Sparse quantized dynamic linear with fused relu is not yet "
      "supported on qnnpack backend.");
  return at::Tensor();



// 强制内部断言，因为稀疏量化动态线性加 ReLU 操作在 qnnpack 后端尚未支持
// 返回一个空的 Tensor



template <>
at::Tensor PackedLinearWeightQnnp::apply_dynamic_impl<false>(
    const at::Tensor& input) {



// 特化模板 PackedLinearWeightQnnp::apply_dynamic_impl<false>，处理静态输入
// 检查输入张量的维度应该 >= 2



  TORCH_CHECK(
      input.dim() >= 2,
      "quantized_sparse_linear(): Input tensor rank should be >= 2");



// 使用 TORCH_CHECK 进行检查，确保输入张量的秩至少为 2



  const auto rows_input = c10::multiply_integers(input.sizes().begin(), input.sizes().end() - 1);
  const auto cols_input = static_cast<int64_t>(input.size(input.dim() - 1));
  TORCH_CHECK(
      cols_input == input_channels_,
      "quantized_sparse_linear: Input tensor's last and weight tensor's"
      " second dimension must match.");



// 计算输入的行数和列数，确保输入张量的最后一维与权重张量的第二维匹配



  // On empty input, no output data will be generated,
  // so use arbitrary qparams.
  float x_min = 0;
  float x_max = 0;



// 对于空输入，不会生成任何输出数据，因此使用任意的量化参数



  // Otherwise...
  if (input.numel() > 0) {
    x_min = input.min().item<float>();
    x_max = input.max().item<float>();
  }



// 否则，计算输入张量的最小和最大值作为量化的范围



  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255);



// 使用 quant_utils::ChooseQuantizationParams 函数选择量化参数



  // Quantize input
  at::Tensor q_input = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);



// 对输入张量进行量化



  auto q_input_contig = q_input.contiguous();



// 将量化后的张量转换为连续的张量



  if (sparse_linear_op_ == nullptr) {



// 如果 sparse_linear_op_ 为空指针，则执行以下操作



    // We calculate requant scale here as the vector holding the requant scale
    // is owned by this module. The pointer is then passed to qnnpack backend.
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales_, q_input_contig.q_scale(), 1.f, requantization_scales_);
    input_scale_ = q_input_contig.q_scale();
    pytorch_qnnp_operator_t sparse_linear_op{nullptr};



// 在这里计算重新量化的尺度，因为保存重新量化尺度的向量由此模块拥有
// 然后将指针传递给 qnnpack 后端






#endif



// 结束条件编译指令 USE_PYTORCH_QNNPACK 区域






}
}



// 结束 ao::sparse 命名空间
    pytorch_qnnp_status status =
        pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
            input_channels_,  // 输入通道数
            output_channels_,  // 输出通道数
            q_input_contig.q_zero_point(),  // 输入张量的量化零点
            w_zero_points_.data(),  // 权重张量的零点数组
            bcsr_matrix_->col_indices_data_ptr(),  // BCSR 格式的稀疏矩阵列索引数据指针
            bcsr_matrix_->row_values_data_ptr(),  // BCSR 格式的稀疏矩阵行值数据指针
            bcsr_matrix_->values.data(),  // BCSR 格式的稀疏矩阵值数据
            bcsr_matrix_->row_block_size, /* out_features_block_size */  // 稀疏矩阵的行块大小，用于输出特征块大小
            bcsr_matrix_->col_block_size, /* in_features_block_size */  // 稀疏矩阵的列块大小，用于输入特征块大小
            bcsr_matrix_->indices_dtype,  // 稀疏矩阵的索引数据类型
            0, /* output zero point: not used */  // 输出的量化零点，这里未使用
            std::numeric_limits<uint8_t>::min(),  // 输出值的最小值限制
            std::numeric_limits<uint8_t>::max(),  // 输出值的最大值限制
            0, /* flags */  // 标志位，这里为零
            requantization_scales_.data(),  // 重新量化的比例因子数组
            true, /* use prepacking kernel */  // 是否使用预打包内核
            &sparse_linear_op);  // 输出的稀疏线性运算符指针
    TORCH_CHECK(
        status == pytorch_qnnp_status_success,  // 检查创建稀疏线性运算符是否成功
        "Failed to create sparse linear operator on"
        " qnnpack backend.");

    sparse_linear_op_ =
        std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(
            sparse_linear_op);  // 使用智能指针管理稀疏线性运算符的生命周期

  }

  // Input on next iteration can be different, thus resulting in
  // different input scale. This will require us to recalculate requantization
  // scales.
  if (input_scale_ != q_input_contig.q_scale()) {
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales_, q_input_contig.q_scale(), 1.f, requantization_scales_);
    // 如果输入张量的量化尺度发生变化，需要重新生成重新量化的比例因子
  }
  // Update input related quantization params in the operator.
  sparse_linear_op_->dynamic_conv_quantization_params.input_zero_point =
      q_input_contig.q_zero_point();  // 更新稀疏线性运算符中的输入量化参数的零点
  sparse_linear_op_->dynamic_conv_quantization_params.multipliers =
      requantization_scales_.data();  // 更新稀疏线性运算符中的输入量化参数的比例因子数组

  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = output_channels_;  // 更新输出张量的最后一个维度大小为输出通道数

  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));  // 创建输出张量

  pytorch_qnnp_status status =
      pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
          sparse_linear_op_.get(),  // 获取稀疏线性运算符指针
          rows_input, /* batch size */  // 批量大小
          reinterpret_cast<uint8_t*>(q_input_contig.data_ptr<c10::quint8>()),  // 输入张量的数据指针
          cols_input, /* num input channels */  // 输入通道数
          bias_.data_ptr<float>(),  // 偏置的数据指针
          output.data_ptr<float>(),  // 输出张量的数据指针
          output_channels_);  // 输出通道数
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,  // 检查设置稀疏线性运算符是否成功
      "Failed to setup sparse linear operator on"
      " qnnpack backend.");

  status = pytorch_qnnp_run_operator(
      sparse_linear_op_.get(), caffe2::pthreadpool_());  // 运行稀疏线性运算符
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,  // 检查运行稀疏线性运算符是否成功
      "Failed to run sparse linear operator on"
      " qnnpack backend.");

  return output;  // 返回计算的输出张量
}

// 定义了一个名为 QLinearDynamicInt8 的模板类，用于动态量化线性运算
namespace {

template <bool ReluFused>
class QLinearDynamicInt8 final {
public:
  // 静态成员函数，用于执行动态量化线性运算
  static at::Tensor run(
      const at::Tensor& input,  // 输入张量
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    auto& ctx = at::globalContext();  // 获取全局上下文

#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 QNNPACK 引擎
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      // 如果启用了 ReLU 融合
      if (ReluFused) {
        // 调用 packed_weight 对象的 apply_dynamic_relu 方法处理输入
        return packed_weight->apply_dynamic_relu(input);
      } else {
        // 否则调用 packed_weight 对象的 apply_dynamic 方法处理输入
        return packed_weight->apply_dynamic(input);
      }
    }
#endif

    // 如果未找到匹配的引擎，则抛出异常
    TORCH_CHECK(
        false,
        "Didn't find engine for operation ao::sparse::qlinear_dynamic",
        toString(ctx.qEngine()));
  }
};

// 使用 TORCH_LIBRARY_IMPL 宏在 sparse 命名空间中注册 CPU 实现
TORCH_LIBRARY_IMPL(sparse, CPU, m) {
  // 注册 sparse::qlinear_dynamic 操作的实现函数为 QLinearDynamicInt8<false>::run
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_dynamic"),
      TORCH_FN(QLinearDynamicInt8<false>::run));
  // 注册 sparse::qlinear_relu_dynamic 操作的实现函数为 QLinearDynamicInt8<true>::run
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_relu_dynamic"),
      TORCH_FN(QLinearDynamicInt8<true>::run));
}

} // namespace
}} // namespace ao::sparse
```