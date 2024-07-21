# `.\pytorch\aten\src\ATen\native\nested\cuda\NestedTensorTransformerFunctions.cpp`

```
// 包含标准库头文件
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <c10/util/Exception.h>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/NonSymbolicBC.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_from_padded_native.h>
#include <ATen/ops/narrow_native.h>
#endif

#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorTransformerUtils.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {
namespace {
// 计算填充张量的元素总数
int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.const_data_ptr<int64_t>();
  int64_t numel = 0;
  for (const auto row_num : c10::irange(sizes_num_rows)) {
    const auto* row_ptr = sizes_data + row_num * sizes_row_length;
    int64_t prod = 1;
    for (const auto idx : c10::irange(sizes_row_length)) {
      prod *= row_ptr[idx];
    }
    numel += prod;
  }
  return numel;
}
} // namespace

// 使用 CUDA 实现的 nested_from_padded 函数，根据输入参数生成嵌套张量
Tensor nested_from_padded_cuda(
    const Tensor& padded,
    const Tensor& sizes,
    bool do_transform_0213) {
  // 检查填充张量的维度是否在指定范围内
  if (padded.dim() > 1 && padded.dim() < 5) {
    // 如果不符合特定维度条件，则调用通用版本函数
    if(!(padded.dim() == 4 && do_transform_0213) && !(padded.dim() == 3 && !do_transform_0213)){
      return at::native::nested_from_padded_generic(padded, sizes, do_transform_0213);
    }
    // 检查填充张量的数据类型是否为 fp32 或 fp16
    if (padded.dtype() != kFloat && padded.dtype() != kHalf) {
      // 提示警告信息，只支持 fp32 或 fp16 类型，然后调用通用版本函数
      TORCH_WARN_ONCE(
          "nested_from_padded CUDA kernels only support fp32/fp16; falling "
          "back to slower generic kernel");
      return at::native::nested_from_padded_generic(padded, sizes, do_transform_0213);
    }

    // 计算目标偏移量
    Tensor target_offsets =
        NestedTensor_batch_offsets_from_size_tensor(sizes, 0);

    // 创建填充张量的大小信息张量，并生成空输出张量
    Tensor padded_sizes_tensor = at::tensor(padded.sizes());
    Tensor output = at::empty({padded_tensor_numel(sizes)}, padded.options());

    // 将目标大小信息重塑为一维张量
    Tensor target_size_sizes = sizes.reshape(-1);

    // 创建元数据张量，将各种大小和偏移量连接在一起，并将其转移到 CUDA 设备上
    Tensor metadata =
        at::cat({target_size_sizes, padded_sizes_tensor, target_offsets});
    metadata = metadata.to(at::Device(kCUDA), kInt, true, true);

    // 获取元数据中的指针
    auto output_size_ptr = metadata.data_ptr<int>();
    auto input_size_ptr = output_size_ptr + target_size_sizes.numel();
    auto offsets_ptr = input_size_ptr + padded_sizes_tensor.numel();

    // 创建填充张量的连续版本
    Tensor padded_contiguous = padded.contiguous();
    # 检查输入张量的数据类型是否为 float
    if (padded.dtype() == kFloat) {
      # 如果需要进行 transform_0213 变换
      if (do_transform_0213) {
        # 调用 CUDA 函数移除填充并进行变换
        remove_padding_transform0213_kernelLauncher(
            padded_contiguous.data_ptr<float>(),  # 输入张量的 float 数据指针
            output.data_ptr<float>(),              # 输出张量的 float 数据指针
            offsets_ptr,                           # 偏移指针
            input_size_ptr,                        # 输入大小指针
            output_size_ptr,                       # 输出大小指针
            padded_contiguous.dim() - 2,           # 输入张量维度减去2
            padded_contiguous.sizes()[0]);         # 输入张量第一维大小
      } else {
        # 调用 CUDA 函数移除填充
        remove_padding_kernelLauncher(
            padded_contiguous.data_ptr<float>(),  # 输入张量的 float 数据指针
            output.data_ptr<float>(),              # 输出张量的 float 数据指针
            offsets_ptr,                           # 偏移指针
            input_size_ptr,                        # 输入大小指针
            output_size_ptr,                       # 输出大小指针
            padded_contiguous.dim() - 1,           # 输入张量维度减去1
            padded_contiguous.sizes()[0]);         # 输入张量第一维大小
      }
    # 如果输入张量的数据类型为 half
    } else if (padded.dtype() == kHalf) {
      # 如果需要进行 transform_0213 变换
      if (do_transform_0213) {
        # 调用 CUDA 函数移除填充并进行变换
        remove_padding_transform0213_kernelLauncher(
            padded_contiguous.data_ptr<c10::Half>(),  # 输入张量的 half 数据指针
            output.data_ptr<c10::Half>(),              # 输出张量的 half 数据指针
            offsets_ptr,                              # 偏移指针
            input_size_ptr,                           # 输入大小指针
            output_size_ptr,                          # 输出大小指针
            padded_contiguous.dim() - 2,              # 输入张量维度减去2
            padded_contiguous.sizes()[0]);            # 输入张量第一维大小
      } else {
        # 调用 CUDA 函数移除填充
        remove_padding_kernelLauncher(
            padded_contiguous.data_ptr<c10::Half>(),  # 输入张量的 half 数据指针
            output.data_ptr<c10::Half>(),              # 输出张量的 half 数据指针
            offsets_ptr,                              # 偏移指针
            input_size_ptr,                           # 输入大小指针
            output_size_ptr,                          # 输出大小指针
            padded_contiguous.dim() - 1,              # 输入张量维度减去1
            padded_contiguous.sizes()[0]);            # 输入张量第一维大小
      }
    # 如果输入张量数据类型既不是 float 也不是 half
    } else {
      # 抛出错误，只支持 fp32/fp16 类型的填充输入
      AT_ERROR("Only support fp32/fp16 for padded input");
    }
    # 返回一个 NestedTensorImpl 类型的张量，将输出张量包装并传递给 make_tensor 函数
    return at::detail::make_tensor<NestedTensorImpl>(std::move(output), sizes);
  } else {
    # 如果不需要移除填充，则调用通用的嵌套张量创建函数
    return at::native::nested_from_padded_generic(padded, sizes);
  }
}

Tensor batch_offsets_from_efficient_size(const Tensor& ef_sizes) {
  // 获取指向 ef_sizes 数据的指针
  int64_t* nt_sizes_ptr = ef_sizes.data_ptr<int64_t>();
  // 获取 ef_sizes 的第一个维度大小
  int64_t ef_sizes_size_0 = ef_sizes.sizes()[0];
  // 创建一个空的 Tensor，用于存储偏移量，大小为 {1 + ef_sizes_size_0}
  Tensor offsets = at::empty({1 + ef_sizes_size_0}, at::kLong);
  // 获取 offsets 的可变数据指针
  int64_t* offsets_ptr = offsets.mutable_data_ptr<int64_t>();
  // 设置 offsets 的第一个元素为 0
  offsets_ptr[0] = 0;
  // 获取 ef_sizes 的第二个维度大小
  int64_t ef_sizes_size_1 = ef_sizes.sizes()[1];
  // 对 ef_sizes 的第一个维度进行迭代
  for (const auto i : c10::irange(ef_sizes_size_0)) {
    // 初始化 prod 变量为 1
    int64_t prod = 1;
    // 对 ef_sizes 的第二个维度进行迭代
    for (const auto j : c10::irange(ef_sizes_size_1)) {
      // 计算累积乘积
      prod = prod * nt_sizes_ptr[i * ef_sizes_size_1 + j];
    }
    // 设置 offsets 的第 i+1 个元素为前一个元素加上当前累积乘积
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  // 返回计算得到的偏移量 Tensor
  return offsets;
}

Tensor NestedTensor_to_padded_tensor_cuda(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
  // 检查输入张量 t 的元素数量是否大于 0
  TORCH_CHECK(t.numel() > 0, "to_padded_tensor: at least one constituent tensor should have non-zero numel")
  // 获取张量 t 的维度
  int64_t t_dim = t.dim();
  // 检查张量 t 是否在 2 到 4 维之间，且数据类型为 kFloat、kDouble 或 kHalf
  if (t_dim >= 2 && t_dim <= 4 &&
      (t.dtype() == at::kFloat || t.dtype() == at::kDouble ||
       t.dtype() == at::kHalf)) {
    // 获取 NestedTensor 的实现指针
    auto* nt_input = get_nested_tensor_impl(t);
    // 检查 NestedTensor 是否是连续的
    TORCH_CHECK(
        nested_tensor_impl_is_contiguous(nt_input),
        "for now to_padded_tensor only supports contiguous nested tensor");
    // 获取 NestedTensor 内部缓冲区的引用
    const auto& nt_buffer = nt_input->get_buffer();

    // 如果张量 t 是三维且第二维大小已知且 output_size 未提供
    if (t_dim == 3 && nt_input->opt_size(2) && (*nt_input->opt_size(2) > 0) &&
        !(output_size.has_value())) {
      // 获取 NestedTensor 内部尺寸
      Tensor nt_sizes = nt_input->get_nested_sizes();
      // 提取尺寸的第一维度
      Tensor sizes_dim1 = at::native::narrow_symint(nt_sizes, 1, 0, 1);
      // 提取尺寸的第二维度
      Tensor sizes_dim2 = at::native::narrow_symint(nt_sizes, 1, 1, 1);
      // 根据尺寸创建新的 NestedTensorImpl
      Tensor result = at::detail::make_tensor<NestedTensorImpl>(
          nt_input->get_buffer(), sizes_dim1 * sizes_dim2[0]);
      // 断言 result 的维度为 2
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.dim() == 2);
      // 递归调用 NestedTensor_to_padded_tensor_cuda 处理 result
      result =
          NestedTensor_to_padded_tensor_cuda(result, padding, output_size);
      // 重塑 result 的形状
      return result.reshape({result.sizes()[0], -1, *nt_input->opt_size(2)});
    }

    // 获取 NestedTensor 内部尺寸
    Tensor nt_sizes = nt_input->get_nested_sizes();
    // 计算批量偏移量
    Tensor offsets = batch_offsets_from_efficient_size(nt_sizes);
    // 获取 NestedTensor 的最大尺寸
    auto new_size = NestedTensor_get_max_size(*nt_input);
    // 将第一个维度大小插入到 new_size 的开头
    new_size.insert(new_size.begin(), nt_sizes.sizes()[0]);

    // 如果提供了 output_size，则将输出张量填充到指定大小
    if (output_size.has_value()) {
      auto output_size_ = output_size.value();
      // 检查 output_size 的长度是否与 NestedTensor 的维度匹配
      TORCH_CHECK(
          output_size_.size() == new_size.size(),
          "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
      // 对 new_size 进行逐个维度的比较和更新
      for (uint64_t i = 0; i < new_size.size(); i++) {
        // 检查 output_size 中的值是否大于等于 NestedTensor 的填充尺寸
        TORCH_CHECK(
            output_size_[i] >= new_size[i],
            "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
        // 更新 new_size 中的每个维度
        new_size[i] = output_size_[i];
      }
    }

    // 创建一个空的输出张量，使用 nt_buffer 的选项
    Tensor output = at::empty(IntArrayRef(new_size), nt_buffer.options());

    // 获取 NestedTensor 的输入维度和批量大小
    int64_t input_dim = nt_sizes.sizes()[1];
    int64_t batch_size = nt_sizes.sizes()[0];
    int64_t output_batch_size = new_size[0];
    // TODO: Remove need for cat here
    // 使用 at::cat 将 offsets 和 nt_sizes.reshape(-1) 拼接成一个新的 Tensor，metadata
    at::Tensor metadata = at::cat({offsets, nt_sizes.reshape(-1)});
    // 将 metadata Tensor 移动到 CUDA 设备，并转换其数据类型为整数
    metadata = metadata.to(at::Device(kCUDA), at::kInt);
    
    // 将 metadata Tensor 按照给定大小分割成两个 Tensor，split[0] 和 split[1]
    std::vector<Tensor> split =
        at::split_with_sizes(metadata, {offsets.numel(), nt_sizes.numel()}, 0);
    
    // 更新 offsets 和 nt_sizes 为分割后的 Tensor
    offsets = split[0];
    nt_sizes = split[1];
    
    // 根据 nt_buffer 中数据的类型，调度并执行 CUDA 内核函数 add_padding_kernelLauncher
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        nt_buffer.scalar_type(), "NestedTensor_to_padded_tensor_cuda", [&]() {
          add_padding_kernelLauncher(
              nt_buffer.data_ptr<scalar_t>(),            // 输入数据指针
              output.data_ptr<scalar_t>(),               // 输出数据指针
              (scalar_t)(padding),                       // 填充值
              offsets.data_ptr<int>(),                   // 偏移量数据指针
              nt_sizes.data_ptr<int>(),                  // 每个样本大小数据指针
              input_dim,                                 // 输入维度
              new_size,                                  // 新的大小
              batch_size,                                // 批量大小
              output_batch_size);                        // 输出批量大小
        });
    
    // 返回填充后的输出 Tensor
    return output;
}

std::tuple<
    Tensor,                                     // 输出：注意力权重
    Tensor,                                     // 输出：对数加权和指数
    Tensor,                                     // 输出：乘法种子
    Tensor,                                     // 输出：乘法偏移
    c10::SymInt,                                // 输出：调试用的注意力掩码
    c10::SymInt,                                // 输出：累积查询序列长度
    Tensor,                                     // 输出：累积键值序列长度
    Tensor>                                     // 输出：输出形状
_scaled_dot_product_flash_attention_nestedtensor_cuda(
    const Tensor& query,                        // 输入：查询张量
    const Tensor& key,                          // 输入：键张量
    const Tensor& value,                        // 输入：值张量
    double dropout_p,                           // 输入：Dropout 概率
    bool is_causal,                             // 输入：是否因果
    bool return_debug_mask,                     // 输入：是否返回调试用的掩码
    std::optional<double> scale) {              // 输入：缩放因子（可选）
  Tensor query_buffer_reshaped,                 // 查询张量重塑后的缓冲区
      key_buffer_reshaped,                     // 键张量重塑后的缓冲区
      value_buffer_reshaped,                   // 值张量重塑后的缓冲区
      cumulative_sequence_length_q,            // 累积查询序列长度
      cumulative_sequence_length_kv,           // 累积键值序列长度
      output_shape;                            // 输出形状
  int64_t max_seqlen_batch_q{0},               // 最大序列长度（查询部分）
      max_seqlen_batch_kv{0};                  // 最大序列长度（键值部分）
  
  // 执行预处理，获取重塑后的张量和其他返回值
  std::tie(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      output_shape) = preprocessing::sdpa_nested_preprocessing(query, key, value);

  // 调用 Flash Attention 的前向计算函数
  auto
      [attention,
       logsumexp,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
      at::_flash_attention_forward(
          query_buffer_reshaped,
          key_buffer_reshaped,
          value_buffer_reshaped,
          cumulative_sequence_length_q,
          cumulative_sequence_length_kv,
          max_seqlen_batch_q,
          max_seqlen_batch_kv,
          dropout_p,
          is_causal,
          return_debug_mask,
          scale,
          c10::nullopt,
          c10::nullopt);
  
  // 将注意力张量重塑并转置以转换非零元素到批量大小和序列长度
  attention = wrap_buffer(attention.view(-1), output_shape).transpose(1, 2);
  
  // 返回计算结果的元组
  return std::make_tuple(
      attention,
      logsumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention_nestedtensor_cuda(
    const Tensor& query,                        // 输入：查询张量
    const Tensor& key,                          // 输入：键张量
    const Tensor& value,                        // 输入：值张量
    const std::optional<at::Tensor>&  attn_bias, // 输入：注意力偏置（可选）
    bool compute_log_sumexp,                    // 输入：是否计算对数加权和
    double dropout_p,                           // 输入：Dropout 概率
    bool is_causal,                             // 输入：是否因果
    bool return_debug_mask,                     // 输入：是否返回调试用的掩码
  std::optional<double> scale) {
  Tensor query_buffer_reshaped, key_buffer_reshaped, value_buffer_reshaped,
      cumulative_sequence_length_q, cumulative_sequence_length_kv, output_shape;
  int64_t max_seqlen_batch_q{0};
  int64_t max_seqlen_batch_k{0};
  // 调用预处理函数，获取处理后的张量和相关信息
  std::tie(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      output_shape) = preprocessing::sdpa_nested_preprocessing(query, key, value);

  // 根据是否需要因果掩码，选择不同的掩码类型
  sdp::CustomMaskType custom_mask_type = is_causal
      ? sdp::CustomMaskType::CausalFromTopLeft
      : sdp::CustomMaskType::NoCustomMask;

  // 根据注意力机制的实现，执行前向传播
  auto [attention, log_sumexp, seed, offset, max_seqlen_q, max_seqlen_batch_kv] = at::_efficient_attention_forward(
      query_buffer_reshaped.unsqueeze(0),          // 将查询张量扩展维度后传入
      key_buffer_reshaped.unsqueeze(0),            // 将键张量扩展维度后传入
      value_buffer_reshaped.unsqueeze(0),          // 将值张量扩展维度后传入
      c10::nullopt,                                // 不传入任何附加参数
      cumulative_sequence_length_q,                // 累积查询序列长度
      cumulative_sequence_length_kv,               // 累积键值序列长度
      max_seqlen_batch_q,                          // 查询序列的最大长度
      max_seqlen_batch_k,                          // 键值序列的最大长度
      dropout_p,                                   // dropout 概率
      static_cast<int64_t>(custom_mask_type),       // 控制掩码类型
      compute_log_sumexp,                          // 是否计算 log_sumexp
      scale);                                      // 注意力矩阵的缩放因子

  // 将注意力张量重新整形以转换 nnz 到 batch_size 和 seq_len
  attention = wrap_buffer(attention.view(-1), output_shape).transpose(1, 2);
  // 返回注意力机制的计算结果及其相关信息
  return std::make_tuple(std::move(attention), std::move(log_sumexp), std::move(seed), std::move(offset));
// 定义一个函数，用于计算 scaled dot product attention 的反向传播过程，返回三个梯度张量
std::tuple<at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_flash_attention_backward_nested(
    const at::Tensor& grad_out_,                            // 输入参数：输出梯度张量
    const at::Tensor& query,                                 // 输入参数：查询张量
    const at::Tensor& key,                                   // 输入参数：键张量
    const at::Tensor& value,                                 // 输入参数：值张量
    const at::Tensor& out,                                   // 输入参数：输出张量
    const at::Tensor& logsumexp,                             // 输入参数：logsumexp 张量
    const Tensor& cumulative_sequence_length_q,              // 输入参数：查询序列累计长度张量
    const Tensor& cumulative_sequence_length_k,              // 输入参数：键序列累计长度张量
    const int64_t max_seqlen_batch_q,                        // 输入参数：查询序列的最大长度
    const int64_t max_seqlen_batch_k,                        // 输入参数：键序列的最大长度
    double dropout_p,                                        // 输入参数：dropout 概率
    bool is_causal,                                          // 输入参数：是否是因果注意力
    const at::Tensor& philox_seed,                           // 输入参数：随机数种子张量
    const at::Tensor& philox_offset,                         // 输入参数：随机数偏移张量
    std::optional<double> scale                              // 输入参数：可选的缩放因子
  ) {
  // 如果 grad_out_ 未定义，返回空张量元组
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  // 定义用于存储预处理结果的张量变量
  Tensor grad_out_buffer_reshaped, query_buffer_reshaped, key_buffer_reshaped,
      value_buffer_reshaped, output_buffer_reshaped;
  
  // 调用预处理函数进行反向传播预处理，获取预处理后的张量结果
  std::tie(
      grad_out_buffer_reshaped,
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      output_buffer_reshaped) =
      preprocessing::sdpa_nested_preprocessing_backward(
          grad_out_,
          query,
          key,
          value,
          out,
          cumulative_sequence_length_q,
          cumulative_sequence_length_k,
          max_seqlen_batch_q,
          max_seqlen_batch_k);

  // 定义用于存储查询、键、值梯度的张量变量
  Tensor grad_q, grad_k, grad_v;
  
  // 调用 _flash_attention_backward 函数计算 scaled dot product attention 的反向传播梯度
  std::tie(grad_q, grad_k, grad_v) = at::_flash_attention_backward(
    grad_out_buffer_reshaped,
    query_buffer_reshaped,
    key_buffer_reshaped,
    value_buffer_reshaped,
    output_buffer_reshaped,
    logsumexp,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_seqlen_batch_q,
    max_seqlen_batch_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale);

  // 将梯度张量重新包装成原始形状
  grad_q = wrap_buffer(grad_q.view(-1), query.transpose(1,2)._nested_tensor_size()).transpose(1,2);
  grad_k = wrap_buffer(grad_k.view(-1), key.transpose(1,2)._nested_tensor_size()).transpose(1,2);
  grad_v = wrap_buffer(grad_v.view(-1), value.transpose(1,2)._nested_tensor_size()).transpose(1,2);

  // 返回查询、键、值梯度的张量元组
  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace native
} // namespace at
```