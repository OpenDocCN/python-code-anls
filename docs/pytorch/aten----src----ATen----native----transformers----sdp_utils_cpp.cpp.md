# `.\pytorch\aten\src\ATen\native\transformers\sdp_utils_cpp.cpp`

```
// 包含头文件 ATen/native/transformers/sdp_utils_cpp.h
#include <ATen/native/transformers/sdp_utils_cpp.h>

// 定义命名空间 sdp
namespace sdp {

// 定义匿名命名空间，内部函数和变量仅在当前文件中可见
namespace {

// 定义优先级顺序数组的函数，根据参数返回默认顺序
std::array<SDPBackend, num_backends> priority_order_cpp(sdp_params const& params) {
  // 默认顺序为 Flash Attention 和 Math
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::math};

  return default_order;
}

// 检查头部维度大小的函数，根据参数和调试标志检查最后一个维度是否相同
bool check_head_dim_size_cpp(sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  // 如果 q, k, v 的最后一个维度不相等
  if (!(query_size_last == key_size_last &&
        query_size_last == value_size_last)) {
    // 如果调试模式开启，打印警告信息
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

// 使用 Flash Attention 的函数，根据参数和调试标志检查是否支持 Flash Attention
bool use_flash_attention_cpp(sdp_params const& params, bool debug) {
  // 支持的 Flash Attention 数据类型数组
  constexpr auto cpp_supported_flash_dtypes =
      array_of<at::ScalarType>(at::kFloat, at::kDouble, at::kBFloat16, at::kHalf);

  // 定义约束函数数组，检查是否可以运行 Flash Kernel
  constexpr auto constraints = array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_flash,
      check_nested_tensor,
      check_for_dropout,
      check_tensor_shapes,
      check_batch_size_and_num_heads_dense,
      check_attn_mask_shape,
      check_head_dim_size_cpp,
      check_nonzero_sequence_lengths_dense,
      check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>);
  
  // 遍历约束函数数组，如果有任何一个约束不满足，则返回 false
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  // 检查张量的数据类型是否符合 Flash Attention 的要求
  return check_tensor_dtype(params, cpp_supported_flash_dtypes, debug);
}

} // namespace

// 选择 SDP 后端的函数，根据内核参数返回最佳后端
SDPBackend select_sdp_backend_cpp(sdp_params const& kernel_params) {
  // 获取全局上下文
  auto& ctx = at::globalContext();
  
  // 如果既未启用数学 SDP 也未启用 Flash SDP，则返回错误后端
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP()) {
    return SDPBackend::error;
  }
  
  // 获取理想的内核顺序
  const auto ordering = priority_order_cpp(kernel_params);

  // 当调试模式关闭时，不打印调试信息
  bool print_debug = false;
  
  // 遍历优先级顺序数组
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::flash_attention:
        // 如果支持 Flash Attention，返回 Flash Attention 后端
        if (use_flash_attention_cpp(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::math:
        // 如果启用了数学 SDP，返回数学后端
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      default:
        // 如果后端无效，抛出错误
        TORCH_CHECK(false, "Invalid backend");
    }
  }

  // 默认返回错误后端
  return SDPBackend::error;
}

} // namespace sdp
  }
  }
  // 如果程序执行到这里，则发生了两件事情：
  // 1. use_flash_attention 没有满足运行的约束条件
  // 2. 用户明确禁用了数学内核
  // 因此，我们重新运行内核检查，并启用调试模式以打印出内核未被选中的原因

  // 设置打印调试信息为真
  print_debug = true;
  // 输出警告信息，说明未使用闪存注意力内核的原因
  TORCH_WARN("Flash attention kernel not used because:");
  // 使用闪存注意力内核的 C++ 版本进行调试打印
  use_flash_attention_cpp(kernel_params, print_debug);
  // 检查打印调试信息，如果为真，则输出错误信息并终止执行
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  // 返回 SDPBackend 类的错误状态
  return SDPBackend::error;
}
} // namespace sdp
```