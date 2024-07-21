# `.\pytorch\aten\src\ATen\native\transformers\sdp_utils_cpp.h`

```py
#pragma once
#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymFloat.h>
#include <c10/util/string_view.h>
#include <c10/util/Array.h>
#include <cmath>
#include <cstdint>
#include <functional>

// 命名空间 sdp 开始
namespace sdp {

// 定义常量 num_backends，表示后端数量为 5
constexpr int32_t num_backends = 5;

// 枚举类型 SDPBackend，用于表示不同的后端选项
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2,
  cudnn_attention = 3,
  overrideable = 4
};

// 自定义枚举类型 CustomMaskType，表示不同的自定义掩码类型
enum class CustomMaskType {
  NoCustomMask = 0,
  CausalFromTopLeft = 1,
  CausalFromBottomRight = 2,
  NumCustomMaskTypes,
};

// 结构体 sdp_params，包含注意力机制的参数
struct sdp_params {
  at::Tensor query;                     // 查询张量
  at::Tensor key;                       // 键张量
  at::Tensor value;                     // 值张量
  std::optional<at::Tensor> attn_mask;  // 可选的注意力掩码张量
  double dropout;                       // dropout 概率
  bool is_causal;                       // 是否是因果的
};

// 函数声明：根据 sdp_params 选择 SDP 后端
SDPBackend select_sdp_backend_cpp(sdp_params const& kernel_params);

// 内联函数：计算缩放因子
inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale) {
  // 计算 softmax 缩放因子
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

// 使用 c10 命名空间中的 array_of 函数
using c10::array_of;

// 内联函数：检查输入张量是否需要梯度
inline bool input_requires_grad(sdp_params const& params) {
  // 检查是否有输入张量需要梯度，且梯度模式是否启用
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

// 内联函数：检查是否有嵌套输入张量
inline bool has_for_nested_inputs(sdp_params const& params) {
  // 检查是否存在嵌套输入张量，并且布局为 kStrided
  return
      (params.query.is_nested() && params.query.layout() == c10::kStrided) ||
      (params.key.is_nested() && params.key.layout() == c10::kStrided) ||
      (params.value.is_nested() && params.value.layout() == c10::kStrided);
}

// 内联函数：检查是否有密集输入张量
inline bool has_for_dense_inputs(sdp_params const& params) {
  // 检查是否所有输入张量均为非嵌套
  return !params.query.is_nested() || !params.key.is_nested() || !params.value.is_nested();
}

// 内联函数：检查是否仅有密集输入张量
inline bool has_only_dense_inputs(sdp_params const& params) {
  // 检查是否所有输入张量均为非嵌套
  return !params.query.is_nested() && !params.key.is_nested() && !params.value.is_nested();
}

// 模板函数：检查张量的数据类型
template <typename dtype_vector>
inline bool check_tensor_dtype(
    sdp_params const& params,
    dtype_vector allowed_dtypes,
    bool debug) {
  auto query_dtype = params.query.dtype();
  // 检查所有输入张量的数据类型是否在允许的数据类型列表中，并且与查询张量的数据类型相同
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), query_dtype) !=
         allowed_dtypes.end()))) {
    // 如果 debug 标志为真，输出警告信息，指出预期的 dtype，并展示实际传入的参数的 dtype
    if (debug) {
      TORCH_WARN(
          "Expected query, key and value to all be of dtype: {",
          c10::Join(", ", allowed_dtypes),
          "}. Got ",
          "Query dtype: ",
          params.query.dtype(),
          ", Key dtype: ",
          params.key.dtype(),
          ", and Value dtype: ",
          params.value.dtype(),
          " instead.");
    }
    // 返回假表示出现了类型不匹配，警告信息已输出
    return false;
  }
  // 返回真表示类型匹配，没有警告信息输出
  return true;
}

// 尝试广播参数大小，并检查是否符合条件
inline bool try_broadcast_param_size(
    const c10::SymInt q_size,         // 查询大小
    const c10::SymInt k_size,         // 键大小
    const c10::SymInt v_size,         // 值大小
    c10::string_view param_name,      // 参数名称
    bool debug) {                     // 调试标志

  // 计算三者中的最大大小
  auto max_size = std::max({q_size, k_size, v_size});

  // 检查是否有任何一个参数大小不等于最大大小且不为1
  if ((q_size != max_size && q_size != 1) ||
      (k_size != max_size && k_size != 1) ||
      (v_size != max_size && v_size != 1)) {
    if (debug) {
      // 如果调试开启，发出警告，说明具体的参数大小情况
      TORCH_WARN(
          "Both fused kernels require query, key and value to have broadcastable ",
          param_name,
          "got Query ",
          param_name,
          q_size,
          ", Key ",
          param_name,
          k_size,
          ", Value ",
          param_name,
          v_size,
          " instead.");
    }
    return false;  // 返回 false 表示不符合条件
  }
  return true;  // 返回 true 表示符合条件
}

// 检查是否存在序列长度为0以及头维度一致的嵌套张量的辅助函数
inline bool check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
    at::Tensor const& param,          // 参数张量
    c10::string_view param_name,      // 参数名称
    bool debug) {                     // 调试标志

  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  const at::Tensor& sizes = nt_tensor_impl->get_nested_sizes();
  auto num_head_dims = nt_tensor_impl->opt_size(1);

  // 如果头维度不存在
  if (!num_head_dims.has_value()) {
    if (debug) {
      // 如果调试开启，发出警告，说明头维度不支持的情况
      TORCH_WARN(
          "Fused kernels do not support ragged num_head_dims, ",
          param_name,
          "has a ragged num_heads.");
    }
    return false;  // 返回 false 表示不符合条件
  }

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = param.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // 遍历张量，检查每个张量的第二维是否为0
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] == 0) {
      if (debug) {
        // 如果调试开启，发出警告，说明序列长度为0的情况
        TORCH_WARN(
            "Fused kernels do not support seq_len == 0, ",
            param_name,
            "has a seq len of 0.");
      }
      return false;  // 返回 false 表示不符合条件
    }
  }
  return true;  // 返回 true 表示符合条件
}

// 检查是否存在序列长度为0的嵌套张量
inline bool check_for_seq_len_0_nested_tensor(sdp_params const& params, bool debug) {
  // 当调用此函数时，确保张量维度为4
  bool q_is_safe = params.query.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.query, "query ", debug)
      : true;

  // 如果 query 不安全，直接返回 false
  if (!q_is_safe) {
    return false;
  }

  bool k_is_safe = params.key.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.key, "key ", debug)
      : true;

  // 如果 key 不安全，直接返回 false
  if (!k_is_safe) {
    return false;
  }

  bool v_is_safe = params.value.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.value, "value ", debug)
      : true;

  // 如果 value 不安全，直接返回 false
  if (!v_is_safe) {
    // 返回 false，表示未通过检查
    return false;
  }

  // 现在我们知道所有的输入都没有不一致的 num_heads，因此可以安全地访问 .size(1)
  auto q_num_heads = params.query.size(1);
  auto k_num_heads = params.key.size(1);
  auto v_num_heads = params.value.size(1);
  // 检查 query、key、value 的 num_heads 是否相同
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  // 如果 num_heads 不相同
  if (!same_num_heads) {
    // 如果需要对参数进行梯度计算
    if (input_requires_grad(params)){
      // 如果开启了调试模式，发出警告信息
      if (debug) {
        TORCH_WARN(
              "Both fused kernels do not support training with broadcasted NT inputs.");
      }
      // 返回 false，表示未通过检查
      return false;
    }
    // 尝试调整参数的尺寸以匹配 num_heads
    return try_broadcast_param_size(
        q_num_heads, k_num_heads, v_num_heads, "num heads ", debug);
  }

  // 如果所有的 num_heads 都相同，返回 true，表示通过检查
  return true;
// 检查是否存在嵌套张量，如果存在则返回 false
inline bool check_nested_tensor(sdp_params const& params, bool debug) {
    // 如果参数中存在非稠密输入，返回 false
    if (!has_only_dense_inputs(params)) {
        // 如果 debug 标志为 true，输出警告信息
        if (debug) {
            TORCH_WARN(
                "Both fused kernels of cpp version currently do not support Nested Tensor inputs.");
        }
        return false;  // 返回 false 表示存在嵌套张量
    }
    return true;  // 没有嵌套张量，返回 true
}

// 检查是否存在 dropout，如果存在则返回 false
inline bool check_for_dropout(sdp_params const& params, bool debug) {
    // 如果 dropout 概率大于 0，返回 false
    if (params.dropout > 0.0) {
        // 如果 debug 标志为 true，输出警告信息
        if (debug) {
            TORCH_WARN("Both fused kernels do not support non-zero dropout.");
        }
        return false;  // 返回 false 表示存在 dropout
    }
    return true;  // 没有 dropout，返回 true
}

// 检查是否需要计算梯度且存在嵌套输入，如果是则返回 false
inline bool check_requires_grad_and_nested(sdp_params const& params, bool debug) {
    // 如果输入需要计算梯度，返回 false
    if (input_requires_grad(params)) {
        // 如果 debug 标志为 true，输出警告信息
        if (debug) {
            TORCH_WARN(
                "Memory efficient attention currently doesn't support training with NT inputs.");
        }
        return false;  // 返回 false 表示需要计算梯度且存在嵌套输入
    }
    return true;  // 不需要计算梯度或者不存在嵌套输入，返回 true
}

// 检查是否存在注意力机制的掩码，如果存在则返回 false
inline bool check_for_attn_mask(sdp_params const& params, bool debug) {
    // 如果存在注意力机制的掩码，返回 false
    if (params.attn_mask.has_value()) {
        // 如果 debug 标志为 true，输出警告信息
        if (debug) {
            TORCH_WARN("Flash Attention does not support non-null attn_mask.");
        }
        return false;  // 返回 false 表示存在注意力机制的掩码
    }
    return true;  // 不存在注意力机制的掩码，返回 true
}

// 检查注意力机制掩码的形状是否符合要求，如果不符合则返回 false
inline bool check_attn_mask_shape(sdp_params const& params, bool debug) {
    auto attn_mask = params.attn_mask;
    // 如果没有提供注意力机制的掩码，直接返回 true
    if (!attn_mask.has_value()) {
        return true;
    }
    // 如果注意力机制的掩码需要计算梯度，返回 false
    if (attn_mask.value().requires_grad()) {
        return false;
    }
    // 获取参数中的各维度大小
    auto batchSize = params.query.sym_size(0);
    auto qSize = params.query.sym_size(2);
    auto kvSize = params.key.sym_size(2);
    auto num_head = params.query.sym_size(1);
    // 检查掩码的形状是否符合要求
    if (attn_mask.value().sym_size(-2) != qSize && attn_mask.value().sym_size(-2) != 1) {
        return false;
    }
    if (attn_mask.value().sym_size(-1) != kvSize && attn_mask.value().sym_size(-1) != 1) {
        return false;
    }
    // 如果掩码是二维的，直接返回 true
    if (attn_mask.value().dim() == 2) {
        return true;
    } else if (attn_mask.value().dim() == 4) {
        // 如果掩码是四维的，检查各维度大小是否符合要求
        if ((attn_mask.value().sym_size(0) == 1 || attn_mask.value().sym_size(0) == batchSize)
            && (attn_mask.value().sym_size(1) == 1 || attn_mask.value().sym_size(1) == num_head)) {
            return true;
        }
    }
    // 如果 debug 标志为 true，输出警告信息
    if (debug) {
        TORCH_WARN("Please use the following attn mask shapes: ",
            "2d - ({Q_seq_len, 1}  x {KV_seq_len, 1}); ",
            "4d - ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})");
    }
    return false;  // 返回 false 表示掩码形状不符合要求
}

// 检查张量的形状是否符合要求，如果不符合则返回 false
inline bool check_tensor_shapes(sdp_params const& params, bool debug) {
    // 获取 query 张量的维度
    auto query_dim = params.query.dim();
    // 如果 query、key 和 value 张量的维度不全为 4，返回 false
    if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
          (query_dim == 4))) {
        // 如果 debug 标志为 true，输出警告信息
        if (debug) {
            TORCH_WARN(
                "Both fused kernels requires query, key and value to be 4 dimensional, but got Query dim: ",
                query_dim,
                ", Key dim: ",
                params.key.dim(),
                ", Value dim: ",
                params.value.dim(),
                " instead.");
        }
        return false;  // 返回 false 表示张量形状不符合要求
    }
    return true;  // 张量形状符合要求，返回 true
}
inline bool check_safe_kv_broadcast(at::Tensor const& param, bool debug) {
  // 获取参数 param 的嵌套张量实现
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  // 获取序列长度
  auto seq_len = nt_tensor_impl->opt_size(2);
  // 如果序列长度不存在
  if (!seq_len.has_value()) {
    // 如果启用了调试模式，发出警告信息
    if (debug) {
      TORCH_WARN(
          "对于融合内核，如果一个键/值批处理大小需要广播而另一个不需要，则另一个必须具有一致的 seq_len 维度。");
    }
    // 返回 false，表示检查未通过
    return false;
  }
  // 返回 true，表示检查通过
  return true;
}

inline bool check_batch_size_and_num_heads_dense(sdp_params const& params, bool debug) {
  // 期望在检查张量形状后调用，确保 size() 调用不会因为输入是四维而出错

  // 获取查询、键、值的批处理大小
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  // 检查批处理大小是否一致
  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  // 获取查询、键、值的头数
  auto q_num_heads = params.query.sym_size(1);
  auto k_num_heads = params.key.sym_size(1);
  auto v_num_heads = params.value.sym_size(1);

  // 检查头数是否一致
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  // 如果批处理大小或头数不一致
  if (!(same_batch_size && same_num_heads)) {
    // 如果启用了调试模式，发出警告信息，显示当前张量的尺寸
    if (debug) {
      TORCH_WARN(
          "对于稠密输入，融合内核要求查询、键和值具有相同的批处理大小和头数。 Query.sizes(): ",
          params.query.sizes(),
          ", Key sizes(): ",
          params.key.sizes(),
          ", Value sizes(): ",
          params.value.sizes(),
          "。若要广播稠密输入，请尝试在传递到内核之前使用 unsqueeze 和 expand_to。");
    }
    // 返回 false，表示检查未通过
    return false;
  }
  // 返回 true，表示检查通过
  return true;
}

inline bool check_batch_size_nested(sdp_params const& params, bool debug) {
  // 期望在检查张量形状后调用，确保 size() 调用不会因为输入是四维而出错

  // 获取查询、键、值的批处理大小
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  // 检查批处理大小是否一致
  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  // 如果批处理大小不一致
  if (!same_batch_size) {
    // 如果输入需要梯度计算
    if (input_requires_grad(params)){
      // 如果启用了调试模式，发出警告信息
      if (debug) {
        TORCH_WARN(
            "两个融合内核不支持使用广播的嵌套张量进行训练。");
      }
      // 返回 false，表示检查未通过
      return false;
    }
    // 尝试广播批处理大小参数
    broadcastable_batch_size = try_broadcast_param_size(
        q_batch_size, k_batch_size, v_batch_size, "batch size ", debug);

    // 如果只有 k 或 v 需要广播批处理大小，则另一个必须具有一致的 seq_len 维度
    // 如果广播批处理大小为真，则执行以下逻辑
    if (broadcastable_batch_size) {
      // 如果键的批处理大小为1且值的批处理大小不为1，并且检查键值广播安全性失败，则返回false
      if (k_batch_size == 1 && v_batch_size != 1 &&
          !check_safe_kv_broadcast(params.value, debug)) {
        return false;
      }
      // 如果值的批处理大小为1且键的批处理大小不为1，并且检查键值广播安全性失败，则返回false
      if (v_batch_size == 1 && k_batch_size != 1 &&
          !check_safe_kv_broadcast(params.key, debug)) {
        return false;
      }
    }
  }
  // 返回广播批处理大小的值
  return broadcastable_batch_size;
namespace sdp
{

// 这里定义了一个命名空间 `sdp`，用于封装以下函数和变量。


inline bool check_nonzero_sequence_lengths_dense(sdp_params const& params, bool debug) {

// 内联函数，用于检查密集型操作中序列长度是否为非零。参数包括操作参数和调试标志。


  // In some cases people will pass in 0 sized tensors, this will
  // cause the fused path to error with unaligned mask

// 在某些情况下，可能会传递大小为0的张量，这会导致融合路径出现未对齐掩码的错误。


  bool zero_seq_len_q = params.query.sym_size(-2) == 0;
  bool zero_seq_len_k = params.key.sym_size(-2) == 0;

// 检查查询和键的序列长度是否为零。


  if (zero_seq_len_q || zero_seq_len_k) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels do not support zero seq_len_q or seq_len_kv.");
    }
    return false;
  }

// 如果查询或键的序列长度为零，则发出警告（如果调试标志为真），并返回假。


  return true;
}

// 如果序列长度非零，则返回真。


template<bool ignore_singleton_dim>
inline bool check_last_dim_stride_equals_1_dense(sdp_params const& params, bool debug) {

// 内联模板函数，用于检查密集型注意力操作中最后一个维度的步幅是否为1。参数包括操作参数和调试标志。


  // The stride checking for NestedTensors is done within the kernel
  // And .contiguous will be called if needed

// 嵌套张量的步幅检查由内核完成，如有必要将调用 `.contiguous` 方法。


  // This function checks that the last dimension of the inputs to
  // fused_attention have stride 1

// 此函数检查输入到融合注意力的张量的最后一个维度是否具有步幅为1。


  bool qkv_strides_equal_1 = params.query.sym_stride(-1) == 1 &&
      params.key.sym_stride(-1) == 1 && params.value.sym_stride(-1) == 1;

// 检查查询、键和值的最后一个维度是否都具有步幅为1。


  if (ignore_singleton_dim){
    qkv_strides_equal_1 = qkv_strides_equal_1 || params.query.sym_size(-1) == 1;
  }

// 如果忽略单维度，检查查询的最后一个维度是否为1。


  bool mask_stride_equal_1 = params.attn_mask.has_value()
      ? params.attn_mask.value().sym_stride(-1) == 1
      : true;

// 检查注意力掩码的最后一个维度是否为1（如果存在的话）。


  if (!(qkv_strides_equal_1 && mask_stride_equal_1)) {
    if (debug) {
      std::ostringstream epilogue_message;
      if (params.attn_mask.has_value()) {
        epilogue_message << ", Attn_mask.stride(-1): "
                         << params.attn_mask.value().sym_stride(-1);
      }
      epilogue_message << " instead.";
      TORCH_WARN(
          "Both fused kernels require the last dimension of the input to have stride 1. ",
          "Got Query.stride(-1): ",
          params.query.sym_stride(-1),
          ", Key.stride(-1): ",
          params.key.sym_stride(-1),
          ", Value.stride(-1): ",
          params.value.sym_stride(-1),
          epilogue_message.str());
    }

// 如果查询、键或值的最后一个维度步幅不为1或者注意力掩码的最后一个维度步幅不为1，则发出警告（如果调试标志为真）。


    return false;
  }

// 如果条件不满足，则返回假。


  return true;
}

// 如果所有条件都满足，则返回真。


inline bool check_runtime_disabled_flash(sdp_params const& params, bool debug) {

// 内联函数，用于检查是否在运行时禁用了闪存SDP内核。参数包括操作参数和调试标志。


  // We check the global context to see if user has explicitly turned of flash
  // sdp kernels

// 我们检查全局上下文，看用户是否明确禁用了闪存SDP内核。


  if (!at::globalContext().userEnabledFlashSDP()) {
    if (debug) {
      TORCH_WARN("Flash attention has been runtime disabled.");
    }
    return false;
  }

// 如果未启用闪存SDP内核，则发出警告（如果调试标志为真），并返回假。


  return true;
}

// 如果启用了闪存SDP内核，则返回真。


inline bool check_runtime_disabled_mem_efficient(sdp_params const& params, bool debug) {

// 内联函数，用于检查是否在运行时禁用了内存高效SDP内核。参数包括操作参数和调试标志。


  // We check the global context to see if user has explicitly turned of
  // mem_efficient sdp kernels

// 我们检查全局上下文，看用户是否明确禁用了内存高效SDP内核。


  if (!at::globalContext().userEnabledMemEfficientSDP()) {
    if (debug) {
      TORCH_WARN("Memory Efficient attention has been runtime disabled.");
    }
    return false;
  }

// 如果未启用内存高效SDP内核，则发出警告（如果调试标志为真），并返回假。


  return true;
}

// 如果启用了内存高效SDP内核，则返回真。


} // namespace sdp

// 结束命名空间 `sdp` 的定义。
```