# `.\pytorch\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp`

```py
/**
 * Note [SDPA Runtime Dispatch]
 * SDPA relies on a runtime dispatch mechanism to select the appropriate
 * kernel. This file contains exposes this through the `select_sdp_backend`
 * The basic structure of this function is to call `priority_order` to get a
 * list of backends to try, and then iterate through them until one succeeds.
 * Each backend defines a use_<backend> function that returns true if the
 * backend can be run with the given SDP parameters. The use_<backend> function
 * will iterate over a list of "filters" that check for specific properties of
 * the SDP parameters. If all filters pass, the backend can be used and use_<backend>
 * returns true. If any filter fails, then use_<backend> returns false.
 *
 * In order to aid in debugging, each filter takes sdp_params and a debug flag.
 * If the debug flag is set, the filter will print a warning message if it fails.
 * The behavior of select_sdp_backend is to return the first backend that
 * succeeds. If no backend is viable then it will run each use_<backend> function
 * with debug=true and return SDPBackend::error.
 */

namespace sdp {
namespace {
// flash_attention V2 is universally faster than efficient_attention and Math
/**
 * Define the default order of SDP backends to try based on priority for the given `params`.
 * The order is: cudnn_attention, flash_attention, efficient_attention, math.
 */
std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::cudnn_attention,
      SDPBackend::flash_attention,
      SDPBackend::efficient_attention,
      SDPBackend::math};
  return default_order;
}

/**
 * Check if tensor cores can be utilized based on the device properties `dprops` and whether
 * the operation is using half-precision arithmetic `is_half`.
 * Returns true if tensor cores can be used; false otherwise.
 */
bool use_tensor_cores(sdp_params const& params, cudaDeviceProp* dprops, bool is_half) {
  if (dprops->major >= 8) {
    return true;
  }
  if (dprops->major >= 7) {
    return is_half;
  }
  return false;
}

/**
 * Determine the minimum alignment requirement for GEMM operations based on the SDP parameters `params`.
 * Uses current CUDA device properties to adjust alignment requirements if tensor cores are used.
 * Returns the minimum alignment in terms of matrix dimensions `m` and `n`.
 */
int64_t minimum_gemm_alignment(sdp_params const& params) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_half = (params.query.dtype() == at::kHalf) ||
      (params.query.dtype() == at::kBFloat16);
  bool use_tc = use_tensor_cores(params, dprops, is_half);
  int64_t matmul_alignment_mn = 1;
  if (dprops->major >= 8) {
    matmul_alignment_mn = 4;
  }
  int64_t bits_per_scalar = is_half ? 16 : 32;
  if (use_tc) {
    matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar);
  }
  return matmul_alignment_mn;
}
} // namespace
} // namespace sdp
// 检查 Flash attention 中注意力头维度的大小是否符合要求
bool check_head_dim_size_flash(sdp_params const& params, bool debug) {
  // 定义最大允许的维度大小
  const auto max_size = c10::SymInt(256);
  // 获取查询张量的最后一个维度大小
  const auto query_size_last = params.query.sym_size(-1);
  // 获取键张量的最后一个维度大小
  const auto key_size_last = params.key.sym_size(-1);
  // 获取值张量的最后一个维度大小
  const auto value_size_last = params.value.sym_size(-1);
  // 检查是否所有的头维度大小相等且小于等于256
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last <= max_size))) {
    // 如果调试模式开启，则输出警告信息
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension and to be less than or equal to 256.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    // 返回 false 表示未通过检查
    return false;
  }
  // 返回 true 表示通过检查
  return true;
}

// 检查嵌套张量输入情况下的 Flash attention 注意力头维度大小是否符合要求
bool check_head_dim_size_flash_nested(sdp_params const& params, bool debug) {
  // 定义最大允许的维度大小
  const auto max_size = c10::SymInt(256);
  // 获取查询张量的最后一个维度大小
  const auto query_size_last = params.query.sym_size(-1);
  // 获取键张量的最后一个维度大小
  const auto key_size_last = params.key.sym_size(-1);
  // 获取值张量的最后一个维度大小
  const auto value_size_last = params.value.sym_size(-1);
  // 检查是否所有的头维度大小相等、是8的倍数且小于等于256
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last % 8 == 0) &&
        (query_size_last <= max_size))) {
    // 如果调试模式开启，则输出警告信息
    if (debug) {
      TORCH_WARN(
          "For NestedTensor inputs, Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 256.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    // 返回 false 表示未通过检查
    return false;
  }
  // 返回 true 表示通过检查
  return true;
}

// 检查内存高效注意力机制中注意力头维度大小是否符合要求
bool check_head_dim_size_mem_efficient(sdp_params const& params, bool debug) {
  // 获取查询张量的最后一个维度大小
  const auto query_size_last = params.query.sym_size(-1);
  // 获取值张量的最后一个维度大小
  const auto value_size_last = params.value.sym_size(-1);
  // 获取最小的 GEMM 对齐大小
  const int64_t alignment = minimum_gemm_alignment(params);
  // 检查是否查询和键的最后一个维度相等，且能够被对齐大小整除
  // 同时检查值的最后一个维度是否能够被对齐大小整除，并且都大于0
  if (!(query_size_last == params.key.sym_size(-1) &&
        query_size_last % alignment == 0 && query_size_last > 0 &&
        value_size_last % alignment == 0 && value_size_last > 0)) {
    // 如果调试模式开启，则输出警告信息
    if (debug) {
      TORCH_WARN(
          "Mem efficient attention requires last dimension of inputs to be divisible by ",
          alignment,
          ". ",
          "Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    // 返回 false 表示未通过检查
    return false;
  }
  // 返回 true 表示通过检查
  return true;
}

// 定义用于表示 GPU 架构版本的结构模板
template <int Major, int Minor>
struct SMVersion {
  // 定义主版本号和次版本号
  static constexpr int major = Major;
  static constexpr int minor = Minor;
  // 默认构造函数
  constexpr SMVersion() = default;
};
/**
 * Checks if the current CUDA device architecture is inclusively within the specified range.
 *
 * @param lower_bound The lower bound of the CUDA device architecture range.
 * @param upper_bound The upper bound of the CUDA device architecture range.
 * @param dprops The pointer to cudaDeviceProp struct containing device properties.
 * @return True if the current CUDA device architecture is within the specified range, false otherwise.
 */
template <typename lower_bound, typename upper_bound>
bool check_sm_version(cudaDeviceProp * dprops) {
  // Check if the current device architecture is greater than or equal to the lower bound
  bool is_gte_lower_bound = dprops->major > lower_bound::major ||
      (dprops->major == lower_bound::major &&
       dprops->minor >= lower_bound::minor);
  // Check if the current device architecture is less than or equal to the upper bound
  bool is_lte_upper_bound = dprops->major < upper_bound::major ||
      (dprops->major == upper_bound::major &&
       dprops->minor <= upper_bound::minor);
  // Return true if the device architecture satisfies both bounds
  return is_gte_lower_bound && is_lte_upper_bound;
}

/**
 * Checks if the current CUDA device supports the Flash Attention operation.
 *
 * @param params The parameters for the current operation.
 * @param debug Flag indicating whether to output debug information.
 * @return True if the device supports Flash Attention, false otherwise.
 */
bool check_flash_attention_hardware_support(sdp_params const& params, bool debug) {
  // Check GPU compatibility for Flash Attention based on selected compilation (CUDA or ROCm)
  using sm80 = SMVersion<8, 0>;  // Define minimum supported architecture (sm_80)
  using sm90 = SMVersion<9, 0>;  // Define maximum supported architecture (sm_90)
#if USE_ROCM
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Check if the GPU supports Flash Attention via ROCm backend
  if (hipSuccess != aotriton::v2::flash::check_gpu(stream)) {
      auto dprops = at::cuda::getCurrentDeviceProperties();
      if (debug) {
          // Output warning if Flash Attention is not supported on the current AMD GPU architecture
          TORCH_WARN(
                  "Flash attention was not compiled for current AMD GPU architecture. Attempting to run on architecture ", dprops->gcnArchName);
      }
      return false;
  }
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();
  // Check if the CUDA device architecture supports Flash Attention
  if (!check_sm_version<sm80, sm90>(dprops)) {
    if (debug) {
      // Output warning if Flash Attention is not supported on the current CUDA GPU architecture
      TORCH_WARN(
          "Flash attention only supports gpu architectures in the range [sm80, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
#endif
  return true;
}

/**
 * Checks if the current CUDA device supports Mem Efficient Attention.
 *
 * @param params The parameters for the current operation.
 * @param debug Flag indicating whether to output debug information.
 * @return True if the device supports Mem Efficient Attention, false otherwise.
 */
bool check_mem_efficient_hardware_support(sdp_params const& params, bool debug) {
  // Mem Efficient Attention supports hardware in the range [sm_50, sm_90]
  using sm50 = SMVersion<5, 0>;   // Define minimum supported architecture (sm_50)
  using sm90 = SMVersion<9, 0>;   // Define maximum supported architecture (sm_90)
#if USE_ROCM
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Check if the GPU supports Mem Efficient Attention via ROCm backend
  if (hipSuccess != aotriton::v2::flash::check_gpu(stream)) {
      auto dprops = at::cuda::getCurrentDeviceProperties();
      if (debug) {
          // Output warning if Mem Efficient Attention is not supported on the current AMD GPU architecture
          TORCH_WARN(
                  "Mem Efficient attention was not compiled for current AMD GPU architecture. Attempting to run on architecture ", dprops->gcnArchName);
      }
      return false;
  }
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();
  // Check if the CUDA device architecture supports Mem Efficient Attention
  if (!check_sm_version<sm50, sm90>(dprops)) {
    if (debug) {
      // Output warning if Mem Efficient Attention is not supported on the current CUDA GPU architecture
      TORCH_WARN(
          "Mem Efficient Attention only supports gpu architectures in the range [sm50, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
#endif
  return true;
}
// 检查在sm86和sm89设备上的要求：
// 如果head_dim大于192且设备在[sm86, sm89]范围内，则Flash Attention在反向传播时会引发错误。
bool check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89(
    sdp_params const& params,   // 输入参数：sdp_params结构体引用，包含Flash Attention的参数
    bool debug) {               // 输入参数：调试模式开关

  // 定义sm86和sm89的CUDA架构版本
  using sm86 = SMVersion<8, 6>;
  using sm89 = SMVersion<8, 9>;

  // 获取当前CUDA设备的属性
  auto dprops = at::cuda::getCurrentDeviceProperties();

  // 检查当前设备是否为sm86或sm89
  bool is_sm86_or_sm89 = check_sm_version<sm86, sm89>(dprops);

  // 检查head_dim是否大于192
  bool is_head_dim_gt192 = params.query.sym_size(-1) > 192;

  // 检查head_dim是否小于等于224
  bool is_head_dim_lte224 = params.query.sym_size(-1) <= 224;

  // 检查是否定义了dropout
  bool is_dropout = params.dropout > 0.0;

  // 检查head_dim是否在(192, 224]范围内，并且当前设备为sm86或sm89
  bool cond1 = is_head_dim_gt192 && is_head_dim_lte224;

  // 检查head_dim是否大于224并且定义了dropout，并且当前设备为sm86或sm89
  bool cond2 = params.query.sym_size(-1) > 224 && is_dropout;

  // 如果需要计算梯度、当前设备为sm86或sm89，并且满足cond1或cond2条件，则报错并返回false
  if (input_requires_grad(params) && is_sm86_or_sm89 && (cond1 || cond2)) {
    if (debug) {
      // 在调试模式下输出警告信息，说明Flash Attention不支持特定条件下的训练
      TORCH_WARN(
          "Flash attention currently doesn't support training with head_dim ∈ (192, 224] or "
          "(head_dim ∈ (224, 256] and dropout > 0.0) on gpu architectures in the range[sm86, sm89].",
          "Attempting to run with dropout set to: ", params.dropout,
          " and head_dim: ",
          params.query.sym_size(-1), " on a sm ", dprops->major, ".",
          dprops->minor, " gpu.");
    }
    return false;
  }
  // 满足所有条件，返回true
  return true;
}

// 检查FlashAttention是否支持非方形序列长度
bool check_flash_causal_non_square_seqlens(sdp_params const& params, bool debug) {
  // 如果启用了causal并且查询和键不是嵌套结构，并且查询的序列长度不等于键的序列长度，则不支持非方形掩码
  if (params.is_causal &&
      !params.query.is_nested() && !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      // 在调试模式下输出警告信息，说明Flash Attention不支持特定条件下的causal设置
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ", params.query.sym_size(-2), " seqlen_k: ",
          params.key.sym_size(-2), ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  // 满足所有条件，返回true
  return true;
}

// 检查所有张量是否在GPU设备上
bool check_all_tensors_on_device(sdp_params const& params, bool debug) {
  // 检查所有张量是否都在CUDA设备上
  // 这应该由存根调度处理，但是当从Python直接调用can_use_*_attention时，我们需要确保张量在CUDA上
  if (params.query.device().type() != at::DeviceType::CUDA) {
    if (debug) {
      // 在调试模式下输出警告信息，说明所有张量需要在CUDA设备上
      TORCH_WARN(
          "All tensors need to be on cuda device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  // 满足所有条件，返回true
  return true;
}
// 检查 CUDNN 张量的形状是否符合要求
bool check_cudnn_tensor_shapes(sdp_params const& params, bool debug) {
  // 获取查询张量的第二个符号维度大小
  const auto s_q = params.query.sym_size(2);
  // 获取键张量的第二个符号维度大小
  const auto s_k = params.key.sym_size(2);
  // 获取头部维度大小
  const auto head_dim = params.query.sym_size(3);
  // 获取当前 CUDNN 版本号
  long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
  
  // 如果 CUDNN 版本号大于等于 9.0.0
  if (cudnn_version >= 90000) {
    // 检查头部维度是否为 8 的倍数且不超过 256
    if (head_dim % 8 != 0 || head_dim > 256) {
      // 如果 debug 模式开启，输出警告信息
      if (debug) {
        TORCH_WARN("head_dim should be a multiple of 8 and no more than 256");
      }
      // 返回 false 表示检查未通过
      return false;
    }
  } else {
    // 如果 CUDNN 版本号小于 9.0.0
    // 检查头部维度是否为 8 的倍数且不超过 128
    if (head_dim % 8 != 0 || head_dim > 128) {
      // 如果 debug 模式开启，输出警告信息
      if (debug) {
        TORCH_WARN("head_dim should be a multiple of 8 and no more than 128");
      }
      // 返回 false 表示检查未通过
      return false;
    }
  }
  
  // 如果 CUDNN 版本号小于 8.9.3
  if (cudnn_version < 8903) {
    // 如果 debug 模式开启，输出警告信息
    if (debug) {
      TORCH_WARN("SDPA fprop requires cudnn 8.9.3 or higher");
    }
    // 返回 false 表示检查未通过
    return false;
  }
  
  // 如果 dropout 不为 0 且 CUDNN 版本号小于 8.9.6
  if (params.dropout != 0.0 && cudnn_version < 8906) {
    // 如果 debug 模式开启，输出警告信息
    if (debug) {
      TORCH_WARN("Dropout reference is only supported on 8.9.6 onwards.");
    }
    // 返回 false 表示检查未通过
    return false;
  }
  
  // 如果 CUDNN 版本号小于 9.0.0
  if (cudnn_version < 90000) {
    // 如果 s_q 小于 64
    if (s_q < 64) {
      // 如果 debug 模式开启，输出警告信息
      if (debug) {
        TORCH_WARN("s_q less than 64 is not supported before cudnn 9.0.0");
      }
      // 返回 false 表示检查未通过
      return false;
    }
    // 如果 s_q 或 s_k 不是 64 的倍数且 dropout 不为 0
    if ((s_q % 64 != 0 || s_k % 64 != 0) && params.dropout != 0.0) {
      // 如果 debug 模式开启，输出警告信息
      if (debug) {
        TORCH_WARN(
            "s_q not a multiple of 64 with padding/dropout is not supported with cudnn version 9.0.0");
      }
      // 返回 false 表示检查未通过
      return false;
    }
  }
  
  // 如果 s_k 不是 64 的倍数且 CUDNN 版本号小于 8.9.6
  if (s_k % 64 != 0 && cudnn_version < 8906) {
    // 如果 debug 模式开启，输出警告信息
    if (debug) {
      TORCH_WARN("not-multiple-of-64 seq_kv is not supported below 8.9.6");
    }
    // 返回 false 表示检查未通过
    return false;
  }
  
  // 所有检查通过，返回 true 表示检查通过
  return true;
}
# 检查 cuDNN 布局是否正确
bool check_cudnn_layout(sdp_params const& params, bool debug) {
    # 获取查询张量的各维度大小
    const int64_t h = params.query.size(1);
    const int64_t s_q = params.query.size(2);
    const int64_t d = params.query.size(3);
    const int64_t s_k = params.key.size(2);
    const int64_t s_v = params.value.size(2);
    
    # 检查是否符合 cuDNN 的 "packed QKV" 布局
    const bool packed_query_layout_ok = (params.query.stride(0) == s_q * 3 * h * d) &&
                                 (params.query.stride(1) == d) &&
                                 (params.query.stride(2) == 3 * h * d) &&
                                 (params.query.stride(3) == 1);
    const bool packed_key_layout_ok = (params.key.stride(0) == s_k * 3 * h * d) &&
                               (params.key.stride(1) == d) &&
                               (params.key.stride(2) == 3 * h * d) &&
                               (params.key.stride(3) == 1);
    const bool packed_value_layout_ok = (params.value.stride(0) == s_v * 3 * h * d) &&
                                 (params.value.stride(1) == d) &&
                                 (params.value.stride(2) == 3 * h * d) &&
                                 (params.value.stride(3) == 1);

    # 检查是否满足 packed 布局要求
    const bool packed_layout_ok = packed_query_layout_ok && packed_key_layout_ok && packed_value_layout_ok;

    # 检查是否满足非 packed 布局要求
    const bool query_layout_ok = (params.query.stride(0) == s_q * h * d) &&
                               (params.query.stride(1) == d) &&
                               (params.query.stride(2) == h * d) &&
                               (params.query.stride(3) == 1);
    const bool key_layout_ok = (params.key.stride(0) == s_k * h * d) &&
                              (params.key.stride(1) == d) &&
                              (params.key.stride(2) == h * d) &&
                              (params.key.stride(3) == 1);
    const bool value_layout_ok = (params.value.stride(0) == s_v * h * d) &&
                               (params.value.stride(1) == d) &&
                               (params.value.stride(2) == h * d) &&
                               (params.value.stride(3) == 1);

    # 检查是否满足任何布局要求
    const bool layout_ok = query_layout_ok && key_layout_ok && value_layout_ok;

    # 如果不满足 packed 布局和非 packed 布局的要求
    if (!packed_value_layout_ok && !layout_ok) {
    # 如果调试模式开启
    if (debug) {
      # 如果紧凑布局不正确
      if (!packed_layout_ok) {
        # 如果紧凑布局的查询部分不正确
        if (!packed_query_layout_ok) {
          # 发出警告，说明查询张量不符合 cuDNN 支持的紧凑 QKV 布局
          TORCH_WARN("Query tensor was not in cuDNN-supported packed QKV layout", params.query.strides());
        }
        # 如果紧凑布局的键部分不正确
        if (!packed_key_layout_ok) {
          # 发出警告，说明键张量不符合 cuDNN 支持的紧凑 QKV 布局
          TORCH_WARN("Key tensor was not in cuDNN-supported packed QKV layout", params.key.strides());
        }
        # 如果紧凑布局的值部分不正确
        if (!packed_value_layout_ok) {
          # 发出警告，说明值张量不符合 cuDNN 支持的紧凑 QKV 布局
          TORCH_WARN("Value tensor was not in cuDNN-supported packed QKV layout", params.value.strides());
        }
      }
      # 如果非紧凑布局不正确
      if (!layout_ok) {
        # 如果非紧凑布局的查询部分不正确
        if (!query_layout_ok) {
          # 发出警告，说明查询张量不符合 cuDNN 支持的非紧凑 QKV 布局
          TORCH_WARN("Query tensor was not in cuDNN-supported unpacked QKV layout", params.query.strides());
        }
        # 如果非紧凑布局的键部分不正确
        if (!key_layout_ok) {
          # 发出警告，说明键张量不符合 cuDNN 支持的非紧凑 QKV 布局
          TORCH_WARN("Key tensor was not in cuDNN-supported unpacked QKV layout", params.key.strides());
        }
        # 如果非紧凑布局的值部分不正确
        if (!value_layout_ok) {
          # 发出警告，说明值张量不符合 cuDNN 支持的非紧凑 QKV 布局
          TORCH_WARN("Value tensor was not in cuDNN-supported unpacked QKV layout", params.value.strides());
        }
      }
    }
    # 返回 false，表示布局检查未通过
    return false;
  }
  # 返回 true，表示布局检查通过
  return true;
bool check_cudnn_hardware_support(sdp_params const& params, bool debug) {
  // 使用 SMVersion 模板定义 sm80 和 sm90 类型，表示 GPU 架构版本
  using sm80 = SMVersion<8, 0>;
  using sm90 = SMVersion<9, 0>;
  // 获取当前 CUDA 设备的属性
  auto dprops = at::cuda::getCurrentDeviceProperties();
  // 检查当前设备是否支持指定的 GPU 架构版本
  if (!check_sm_version<sm80, sm90>(dprops)) {
    // 如果不支持，且 debug 模式打开，输出警告信息
    if (debug) {
      TORCH_WARN(
          "cuDNN MHA only supports gpu architectures in the range [sm80, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    // 返回 false 表示硬件不支持
    return false;
  }
  // 硬件支持的话返回 true
  return true;
}

bool check_is_causal(sdp_params const& params, bool debug) {
  // 检查输入是否因果
  if (!params.is_causal) {
    // 如果不因果，且 debug 模式打开，输出警告信息
    if (debug) {
      TORCH_WARN("CuDNN requires is_causal=True.");
    }
    // 返回 false 表示不满足要求
    return false;
  }
  // 满足要求返回 true
  return true;
}

bool check_for_nested_inputs(sdp_params const& params, bool debug) {
  // 检查输入是否包含嵌套结构
  if (has_for_nested_inputs(params)) {
    // 如果有嵌套输入，且 debug 模式打开，输出警告信息
    if (debug) {
      TORCH_WARN("CuDNN currently does not support nested inputs.");
    }
    // 返回 false 表示不支持嵌套输入
    return false;
  }
  // 不是嵌套输入返回 true
  return true;
}

bool check_dtypes_low_precision(sdp_params const& params, bool debug) {
  // 获取当前 CUDA 设备的属性
  auto dprop = at::cuda::getCurrentDeviceProperties();
  // 根据设备主版本号选择不同的数据类型数组
  if (dprop->major >= 8) {
    constexpr auto sm80_dtypes =
        array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
    return check_tensor_dtype(params, sm80_dtypes, debug);
  } else {
    constexpr auto default_dtypes = array_of<at::ScalarType>(at::kHalf);
    return check_tensor_dtype(params, default_dtypes, debug);
  }
}

bool check_runtime_enabled_cudnn(sdp_params const& params, bool debug) {
  // 静态标志位和布尔变量，用于仅执行一次的初始化
  static c10::once_flag supported_flag;
  static bool supported = false;
  // 使用 call_once 确保只初始化一次
  c10::call_once(supported_flag, []() {
    // 检查环境变量是否设置为支持 CuDNN SDPA
    supported = (c10::utils::check_env("TORCH_CUDNN_SDPA_ENABLED") == true);
  });
  // 如果不支持，且 debug 模式打开，输出警告信息
  if (!supported) {
    if (debug) {
      TORCH_WARN(
          "The CuDNN backend needs to be enabled by setting the enviornment variable `TORCH_CUDNN_SDPA_ENABLED=1`");
    }
    // 返回 false 表示未启用 CuDNN SDPA
    return false;
  }
  // 支持的话返回 true
  return true;
}

bool check_runtime_disabled_cudnn(sdp_params const& params, bool debug) {
  // 检查全局上下文，判断用户是否显式禁用了 CuDNN SDP 内核
  if (!at::globalContext().userEnabledCuDNNSDP()) {
    // 如果禁用了，且 debug 模式打开，输出警告信息
    if (debug) {
      TORCH_WARN("CuDNN attention has been runtime disabled.");
    }
    // 返回 false 表示 CuDNN SDP 内核已被禁用
    return false;
  }
  // 未禁用返回 true
  return true;
}

bool check_cudnn_requires_grad(sdp_params const& params, bool debug) {
  // 检查输入是否需要梯度
  if (input_requires_grad(params)) {
    // 如果需要梯度，且 debug 模式打开，输出警告信息
    if (debug) {
      TORCH_WARN("CuDNN does not currently support inputs with requires_grad=True.");
    }
    // 返回 false 表示不支持需要梯度的输入
    return false;
  }
  // 不需要梯度返回 true
  return true;
}
// 判断是否可以使用 cuDNN 注意力机制，根据参数和调试标志决定
bool can_use_cudnn_attention(const sdp_params& params, bool debug) {

  // 定义门函数，用于确定是否可以运行闪存内核
  // 在迁移到 C++20 后，替换为 std::to_array
  constexpr auto general_constraints =
      array_of<bool (*)(sdp_params const&, bool)>(
          check_runtime_enabled_cudnn,
          check_runtime_disabled_cudnn,
          check_cudnn_hardware_support,
          check_all_tensors_on_device,
          check_cudnn_tensor_shapes,
          check_cudnn_layout,
          // check_is_causal,
          check_for_nested_inputs,
          check_cudnn_requires_grad,
          check_dtypes_low_precision);
  // 遍历通用约束条件，如果有任何一个约束条件不满足，则返回 false
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  // 所有约束条件满足，则返回 true
  return true;
}

// 判断是否可以使用闪存注意力机制，根据参数和调试标志决定
bool can_use_flash_attention(sdp_params const& params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  // 如果未定义 USE_FLASH_ATTENTION 宏，则给出警告并返回 false
  TORCH_WARN_ONCE(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif

  // 定义门函数，用于确定是否可以运行闪存内核
  // 在迁移到 C++20 后，替换为 std::to_array
  constexpr auto general_constraints = array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_flash,
      check_all_tensors_on_device,
      check_tensor_shapes,
      check_for_attn_mask,
      check_head_dim_size_flash,
      check_flash_attention_hardware_support,
      check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89,
      check_flash_causal_non_square_seqlens,
      check_dtypes_low_precision);
  // 遍历通用约束条件，如果有任何一个约束条件不满足，则返回 false
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  // 如果具有嵌套输入，则进行额外的嵌套约束条件检查
  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_nested,
        check_head_dim_size_flash_nested,
        check_for_seq_len_0_nested_tensor);
    // 遍历嵌套约束条件，如果有任何一个约束条件不满足，则返回 false
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

  // 如果仅具有稠密输入，则进行额外的稠密约束条件检查
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>);
    // 遍历稠密约束条件，如果有任何一个约束条件不满足，则返回 false
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  // 所有约束条件满足，则返回 true
  return true;
}

// 判断是否可以使用内存高效的注意力机制，根据参数和调试标志决定
bool can_use_mem_efficient_attention(sdp_params const& params, bool debug) {
#ifndef USE_MEM_EFF_ATTENTION
  // 如果未定义 USE_MEM_EFF_ATTENTION 宏，则给出警告并返回 false
  TORCH_WARN_ONCE(!debug, "Torch was not compiled with memory efficient attention.");
  return false;
#endif
  // 特定于内存高效注意力机制的约束条件
  constexpr auto greater_than_or_equal_sm80_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);
  constexpr auto less_than_sm80_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat);
#ifdef USE_ROCM
  // 如果使用 ROCM，则定义内存高效数据类型为半精度、单精度和 BF16
  constexpr auto aotriton_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);
#endif

  // 定义一般约束函数数组，用于确定是否可以运行内存高效的核函数
  constexpr auto general_constraints = array_of<bool (*)(sdp_params const&, bool)>(
      // 检查运行时禁用的内存高效特性
      check_runtime_disabled_mem_efficient,
      // 检查所有张量是否在设备上
      check_all_tensors_on_device,
      // 检查设备是否支持内存高效特性
      check_mem_efficient_hardware_support,
      // 检查张量形状
      check_tensor_shapes,
      // 检查头维度大小对于内存高效是否合适
      check_head_dim_size_mem_efficient);
  // 对于每个约束函数，如果有一个约束不满足，则返回 false
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  // 如果输入包含嵌套张量
  if (has_for_nested_inputs(params)) {
#ifdef USE_ROCM
    // 如果使用 ROCM，则不支持内存高效注意力中的嵌套张量，发出警告并返回 false
    TORCH_WARN_ONCE(false, "[ROCM] no support for nested tensors in memory efficient attention.");
    return false;
#endif
    // 定义嵌套约束函数数组
    constexpr auto nested_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        // 检查是否需要梯度并且包含嵌套张量
        check_requires_grad_and_nested,
        // 检查批处理大小对于嵌套张量
        check_batch_size_nested,
        // 检查序列长度为 0 的嵌套张量
        check_for_seq_len_0_nested_tensor);
    // 对于每个嵌套约束函数，如果有一个约束不满足，则返回 false
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

  // 如果输入只包含密集张量
  if (has_only_dense_inputs(params)) {
    // 定义密集约束函数数组
    constexpr auto dense_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        // 检查批处理大小和头数对于密集张量
        check_batch_size_and_num_heads_dense,
        // 检查非零序列长度对于密集张量
        check_nonzero_sequence_lengths_dense,
        // 检查最后维度步幅是否为 1 对于密集张量，忽略单维度
        check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim=*/>);
    // 对于每个密集约束函数，如果有一个约束不满足，则返回 false
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

#ifdef USE_ROCM
  // 如果使用 ROCM，检查张量数据类型是否满足内存高效数据类型要求
  return check_tensor_dtype(params, aotriton_mem_efficient_dtypes, debug);
#else
  // 否则获取当前 CUDA 设备属性
  auto dprop = at::cuda::getCurrentDeviceProperties();
  // 如果设备主版本号大于等于 8，则检查张量数据类型是否满足 SM80 及以上的内存高效数据类型要求
  if (dprop->major >= 8) {
    return check_tensor_dtype(params, greater_than_or_equal_sm80_mem_efficient_dtypes, debug);
  }
#endif
  // 默认情况下，检查张量数据类型是否满足 SM80 以下的内存高效数据类型要求
  return check_tensor_dtype(params, less_than_sm80_mem_efficient_dtypes, debug);
}

// 选择 SDP 后端类型的函数
SDPBackend select_sdp_backend(sdp_params const& kernel_params) {
  // 该函数定义了不同 SDP 后端的优先顺序
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  // 如果没有启用任何 SDP 后端，则返回错误
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP() && !ctx.userEnabledCuDNNSDP()) {
    return SDPBackend::error;
  }
  // 获取理想的内核顺序
  const auto ordering = priority_order(kernel_params);

  // 当 debug 为真时，通过 TORCHCHECK 检查条件是否为真，所以我们取反 debug 来打印语句
  bool print_debug = false;
  for (auto& backend : ordering) {
  // 根据给定的后端选择适当的SDP后端类型
  switch (backend) {
    case SDPBackend::cudnn_attention:
      // 如果可以使用CudNN注意力机制，则选择CudNN后端
      if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) {
        return SDPBackend::cudnn_attention;
      }
      break;
    case SDPBackend::flash_attention:
      // 如果可以使用Flash注意力机制，则选择Flash后端
      if (sdp::can_use_flash_attention(kernel_params, print_debug)) {
        return SDPBackend::flash_attention;
      }
      break;
    case SDPBackend::efficient_attention:
      // 如果可以使用内存高效的注意力机制，则选择内存高效后端
      if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) {
        return SDPBackend::efficient_attention;
      }
      break;
    case SDPBackend::math:
      // 如果用户启用了数学SDP，则选择数学后端
      if (ctx.userEnabledMathSDP()) {
        return SDPBackend::math;
      }
      break;
    default:
      // 若后端类型无效，则抛出错误
      TORCH_CHECK(false, "Invalid backend");
  }
}
// 若程序运行到此处，说明两件事情发生了：
// 1. use_flash_attention 或 use_mem_efficient 没有满足运行的约束条件
// 2. 用户显式禁用了数学核心
// 因此，我们使用调试模式重新运行核心检查以打印未选择该核心的原因

print_debug = true;
// 输出警告，说明为什么未选择内存高效核心
TORCH_WARN("Memory efficient kernel not used because:");
sdp::can_use_mem_efficient_attention(kernel_params, print_debug);
// 输出警告，说明为什么未选择Flash注意力核心
TORCH_WARN("Flash attention kernel not used because:");
sdp::can_use_flash_attention(kernel_params, print_debug);
// 输出警告，说明为什么未选择CudNN注意力核心
TORCH_WARN("CuDNN attention kernel not used because:");
sdp::can_use_cudnn_attention(kernel_params, print_debug);
// 最终检查是否打印了调试信息，如果是，则终止执行并返回错误后端
TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
return SDPBackend::error;
// 结束 sdp 命名空间，此处为代码文件的末尾
} // namespace sdp

// 检查是否存在长度为1的嵌套张量
bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug) {
  // 当调用此函数时，确保参数 params.query 是嵌套张量且维度为4
  if (!params.query.is_nested()) {
    return true; // 如果不是嵌套张量，直接返回 true
  }

  // 获取嵌套张量的实现对象
  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  // 获取嵌套张量中的尺寸信息
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>(); // 获取尺寸数据的指针
  const int64_t n_tensors = params.query.size(0); // 获取张量的数量
  const int64_t size_tensor_stride = sizes.stride(0); // 获取尺寸张量的步长

  // 在形如 [batch, heads, {seq_len}, dim] 的上下文中进行调用
  for (const auto i : c10::irange(n_tensors)) {
    // 检查每个张量的第二个尺寸是否小于等于1
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        // 如果启用了调试模式，发出警告，指出不支持长度小于等于1的序列
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false; // 如果有任何一个张量长度小于等于1，则返回 false
    }
  }

  return true; // 所有张量的长度都大于1，则返回 true
}
```