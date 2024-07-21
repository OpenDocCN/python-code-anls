# `.\pytorch\aten\src\ATen\native\Convolution.cpp`

```
// 定义宏，用于在 Torch 中仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 核心 Tensor 类头文件
#include <ATen/core/Tensor.h>

// 包含 ATen 配置头文件
#include <ATen/Config.h>

// 包含 ATen 并行处理头文件
#include <ATen/Parallel.h>

// 包含 ATen 张量运算符头文件
#include <ATen/TensorOperators.h>

// 包含 ATen 三维卷积头文件
#include <ATen/native/ConvolutionMM3d.h>

// 包含 ATen 卷积工具头文件
#include <ATen/native/ConvUtils.h>

// 包含 ATen 池化头文件
#include <ATen/native/Pool.h>

// 包含 ATen CPU 端深度卷积核头文件
#include <ATen/native/cpu/DepthwiseConvKernel.h>

// 包含 ATen 实用参数工具头文件
#include <ATen/native/utils/ParamUtils.h>

// 包含 ATen XNNPACK 引擎头文件
#include <ATen/native/xnnpack/Engine.h>

// 包含 C10 核心梯度模式头文件
#include <c10/core/GradMode.h>

// 包含 C10 实用累加头文件
#include <c10/util/accumulate.h>

// 包含 C10 实用范围头文件
#include <c10/util/irange.h>

// 包含 C10 宏定义头文件
#include <c10/macros/Macros.h>

// 包含数学库限制头文件
#include <limits>

// 包含实用性头文件
#include <utility>

// 如果未定义每个操作符的头文件，则包含 ATen 函数头文件，否则包含 ATen 排列头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/permute.h>
#endif

// 如果启用 NNPACK，则包含 NNPACK 头文件
#if AT_NNPACK_ENABLED()
#include <nnpack.h>
#endif

// 如果启用 MKLDNN，则包含 MKLDNN 实用工具头文件
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/Utils.h>
#endif

// 如果未定义每个操作符的头文件，则包含 ATen 函数头文件和本地函数头文件，否则包含 ATen 卷积深度2D头文件等
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv_depthwise2d.h>
#include <ATen/ops/_convolution.h>
#include <ATen/ops/_convolution_double_backward_native.h>
#include <ATen/ops/_convolution_mode.h>
#include <ATen/ops/_convolution_mode_native.h>
#include <ATen/ops/_convolution_native.h>
#include <ATen/ops/_mps_convolution.h>
#include <ATen/ops/_mps_convolution_transpose.h>
#include <ATen/ops/_nnpack_available.h>
#include <ATen/ops/_nnpack_spatial_convolution.h>
#include <ATen/ops/_slow_conv2d_backward.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv1d_native.h>
#include <ATen/ops/conv2d_native.h>
#include <ATen/ops/conv3d_native.h>
#include <ATen/ops/conv_depthwise3d.h>
#include <ATen/ops/conv_transpose1d_native.h>
#include <ATen/ops/conv_transpose2d_native.h>
#include <ATen/ops/conv_transpose3d_native.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward_native.h>
#include <ATen/ops/convolution_backward_overrideable.h>
#include <ATen/ops/convolution_backward_overrideable_native.h>
#include <ATen/ops/convolution_native.h>
#include <ATen/ops/convolution_overrideable.h>
#include <ATen/ops/convolution_overrideable_native.h>
#include <ATen/ops/cudnn_convolution.h>
#include <ATen/ops/cudnn_convolution_transpose.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/miopen_convolution.h>
#include <ATen/ops/miopen_convolution_transpose.h>
#include <ATen/ops/miopen_depthwise_convolution.h>
#include <ATen/ops/mkldnn_convolution.h>
#include <ATen/ops/mps_convolution_backward.h>
#include <ATen/ops/mps_convolution_transpose_backward.h>
#include <ATen/ops/slow_conv3d.h>
#include <ATen/ops/slow_conv_dilated2d.h>
#include <ATen/ops/slow_conv_dilated3d.h>
#include <ATen/ops/slow_conv_transpose2d.h>
#include <ATen/ops/slow_conv_transpose3d.h>
#include <ATen/ops/thnn_conv2d.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

// 定义最大 MIOPEN 维度
constexpr int MIOPEN_DIM_MAX = 5;

// 命名空间开始：ATen 本地实现
namespace at::native {
// 定义一个静态布尔变量，用于控制是否进行空缓存的基准测试
static bool conv_benchmark_empty_cache = true;

// 检查是否需要激活快速深度可分卷积的FP16 CUDNN内核
template <typename T>
bool check_cudnn_depthwise_workload(const at::Tensor& input, T stride) {
  // 获取输入张量的维度信息，并使用模板类型T将其转换为整数
  auto w = at::symint::size<T>(input, 3);  // w和h维度相同
  auto ch = at::symint::size<T>(input, 1); // 通道数
  auto bs = at::symint::size<T>(input, 0); // 批量大小

  // 如果步长为1
  if (stride==1) {
    // 如果宽度大于等于7
    if (w >= 7) {
      // 对所有批量大小和通道数生效
      if (w >= 112) {
        return true;
      }

      // 大通道数
      if (ch >= 1024) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (w >= 56) {
          return true;
        } else if (bs >= 32) {
          return true;
        }
      }

      // 特定批量大小
      if (bs >= 128) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (ch >= 512) {
          return true;
        } else if (ch >= 64) {
          if (w >= 14) {
            return true;
          }
        } else if ((ch >= 32) && (w >=28)) {
          return true;
        }
      } else if (bs >= 64) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 32) && (w >= 28)) {
          return true;
        }
      } else if (bs >= 32) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 128) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 16) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        }
        if ((ch >= 256) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 8) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if ((ch >= 64) && (w >= 56)) {
          return true;
        }
      }
    }
  } else if (stride==2) {
    // 如果步长为2，并且通道数小于256，则返回false
    if (ch < 256) {
      return false;
    }
    # 检查 w 是否大于等于 7
    if (w >= 7) {
      # 检查 bs 是否大于等于 128
      if (bs >= 128) {
        # NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        # 检查 ch 是否大于等于 1024
        if (ch >= 1024) {
          # 如果满足条件，返回 true
          return true;
        } else if ((ch >= 512) && (w >= 14)) {
          # 否则，如果 ch 大于等于 512 并且 w 大于等于 14，返回 true
          return true;
        } else if (w >= 28) {
          # 否则，如果 w 大于等于 28，返回 true
          return true;
        }
      } else if (bs >= 64) {
        # 否则，如果 bs 大于等于 64
        # NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 14)) {
          # 如果 ch 大于等于 512 并且 w 大于等于 14，返回 true
          return true;
        } else if (w >= 28) {
          # 否则，如果 w 大于等于 28，返回 true
          return true;
        }
      } else if (bs >= 32) {
        # 否则，如果 bs 大于等于 32
        # NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 14)) {
          # 如果 ch 大于等于 1024 并且 w 大于等于 14，返回 true
          return true;
        } else if (w >= 28) {
          # 否则，如果 w 大于等于 28，返回 true
          return true;
        }
      } else if (bs >= 16) {
        # 否则，如果 bs 大于等于 16
        # NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 28)) {
          # 如果 ch 大于等于 512 并且 w 大于等于 28，返回 true
          return true;
        } else if (w >= 56) {
          # 否则，如果 w 大于等于 56，返回 true
          return true;
        }
      } else if (bs >= 8) {
        # 否则，如果 bs 大于等于 8
        # NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 28)) {
          # 如果 ch 大于等于 1024 并且 w 大于等于 28，返回 true
          return true;
        } else if (w >= 56) {
          # 否则，如果 w 大于等于 56，返回 true
          return true;
        }
      } else if (bs >= 1) {
        # 否则，如果 bs 大于等于 1
        if ((ch >= 512) && (w >=112)) {
          # 如果 ch 大于等于 512 并且 w 大于等于 112，返回 true
          return true;
        }
      }
    }
  }
  # 如果以上条件都不满足，则返回 false
  return false;
// 结构体模板，用于存储卷积参数配置
template <typename T>
struct ConvParams {
  // 存储步长（stride）、填充（padding）、扩展（dilation）、是否反卷积（transposed）的向量
  std::vector<T> stride;
  std::vector<T> padding;
  std::vector<T> dilation;
  bool transposed;         // 是否反卷积
  std::vector<T> output_padding;  // 输出填充
  T groups;                // 分组
  bool benchmark;          // 是否启用基准测试
  bool deterministic;      // 是否确定性操作
  bool cudnn_enabled;      // 是否启用 cuDNN 加速
  bool allow_tf32;         // 是否允许 TF32 格式

  // 检查是否有步长不为1的情况
  bool is_strided() const {
    bool is_strided = false;
    for (auto s : stride) {
      is_strided |= (s != 1);
    }
    return is_strided;
  }

  // 检查是否有扩展（dilation）不为1的情况
  bool is_dilated() const {
    bool is_dilated = false;
    for (auto d : dilation) {
      is_dilated |= (d != 1);
    }
    return is_dilated;
  }

  // 检查是否有填充（padding）
  bool is_padded() const {
    bool is_padded = false;
    // 对于每个填充参数，如果有任何一个不为0，则认为有填充
    for (auto p : padding) {
      is_padded |= (p != 0);
    }
    return is_padded;
  }
}
    // 遍历 padding 数组中的每个元素
    for (auto p : padding) {
      // 使用位运算检查是否存在非零的 padding 值
      is_padded |= (p != 0);
    }
    // 返回是否存在非零的 padding 值的布尔结果
    return is_padded;
  }

  bool is_output_padding_neg() const {
    bool is_non_neg = false;
    // 遍历 output_padding 数组中的每个元素的引用
    for (const auto& p : output_padding) {
      // 使用位运算检查是否存在小于零的 output_padding 值
      is_non_neg |= (p < 0);
    }
    // 返回是否存在小于零的 output_padding 值的布尔结果
    return is_non_neg;
  }

  bool is_output_padding_big() const {
    bool is_big = false;
    // 使用范围遍历 output_padding 数组的索引
    for (auto i: c10::irange(output_padding.size())) {
      // 使用位运算检查是否存在大于或等于 stride 对应位置的 output_padding 值
      is_big |= (output_padding[i] >= stride[i]);
    }
    // 返回是否存在大于或等于 stride 对应位置的 output_padding 值的布尔结果
    return is_big;
  }

  bool is_padding_neg() const {
    bool is_non_neg = false;
    // 遍历 padding 数组中的每个元素的引用
    for (const auto& p : padding) {
      // 使用位运算检查是否存在小于零的 padding 值
      is_non_neg |= (p < 0);
    }
    // 返回是否存在小于零的 padding 值的布尔结果
    return is_non_neg;
  }

  bool is_dilation_neg() const {
    bool is_non_neg = false;
    // 遍历 dilation 数组中的每个元素的引用
    for (const auto& p : dilation) {
      // 使用位运算检查是否存在小于零的 dilation 值
      is_non_neg |= (p < 0);
    }
    // 返回是否存在小于零的 dilation 值的布尔结果
    return is_non_neg;
  }

  bool is_stride_nonpos() const {
    bool is_nonpos = false;
    // 遍历 stride 数组中的每个元素
    for (auto s : stride) {
      // 使用位运算检查是否存在小于或等于零的 stride 值
      is_nonpos |= (s <= 0);
    }
    // 返回是否存在小于或等于零的 stride 值的布尔结果
    return is_nonpos;
  }

  void view1d_as_2d() {
    // 如果 stride 数组的大小为 1
    if (stride.size() == 1) {
      // 在数组开头插入一个值为 1 的元素
      stride.insert(stride.begin(), 1);
      // 在 padding 数组开头插入一个值为 0 的元素
      padding.insert(padding.begin(), 0);
      // 在 dilation 数组开头插入一个值为 1 的元素
      dilation.insert(dilation.begin(), 1);
      // 在 output_padding 数组开头插入一个值为 0 的元素
      output_padding.insert(output_padding.begin(), 0);
    }
  }

  bool use_cpu_depthwise3x3_winograd(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias) const {
#if defined(__ARM_NEON__)
    // 当前仅支持在 float 类型的 4 维张量上进行 3x3 深度卷积。
    return (input.ndimension() == 4) &&  // 输入张量必须是 4 维
           (at::symint::size<T>(input, 1) == groups) &&  // 第二维大小必须等于 groups 参数
           (weight.ndimension() == 4 ) &&  // 权重张量必须是 4 维
           (at::symint::size<T>(weight, 0) % at::symint::size<T>(input, 1) == 0) &&  // 权重张量第一维大小必须是输入张量第二维大小的倍数
           (at::symint::size<T>(weight, 1) == 1) &&  // 权重张量第二维大小必须为 1
           (at::symint::size<T>(weight, 2) == 3) &&  // 权重张量第三维大小必须为 3
           (at::symint::size<T>(weight, 3) == 3) &&  // 权重张量第四维大小必须为 3
           (input.device().is_cpu()) &&  // 输入张量必须在 CPU 上
           (input.scalar_type() == at::kFloat) &&  // 输入张量数据类型必须是 float
           input.is_contiguous() &&  // 输入张量必须是连续的
           (weight.device().is_cpu()) &&  // 权重张量必须在 CPU 上
           (weight.scalar_type() == at::kFloat) &&  // 权重张量数据类型必须是 float
           weight.is_contiguous() &&  // 权重张量必须是连续的
           (!bias.has_value() || bias->is_contiguous()) &&  // 如果存在偏置，偏置张量必须是连续的
           !is_strided() &&  // 不允许张量是 strided 的
           !is_dilated() &&  // 不允许张量是 dilated 的
           !transposed;  // 不允许张量是转置的
#else
    return false;
#endif
  }

  bool needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const {
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    auto numel_input = at::symint::numel<T>(input);
    // 空输入张量
    if (numel_input == 0) {
      return false;
    }
    // 无法通过分割批处理维度将输入大小减少到 int 范围内
    auto n = at::symint::size<T>(input, 0);
    if (numel_input / n > int_max) {
      return true;
    }
    // 无法通过分割批处理维度将输出大小减少到 int 范围内
    T outsize = 1;
    if (transposed) {
      auto o = conv_input_size(at::symint::sizes<T>(input), at::symint::sizes<T>(weight), padding, output_padding, stride, dilation, groups);
      outsize = c10::multiply_integers(o.begin() + 1, o.end());
    } else {
      auto o = conv_output_size(at::symint::sizes<T>(input), at::symint::sizes<T>(weight), padding, stride, dilation);
      outsize = c10::multiply_integers(o.begin() + 1, o.end());
    }
    return outsize > int_max;
  }

  bool use_cudnn(const at::Tensor& input, const at::Tensor& weight) const {
    // 注意 [Mobile check segfaults]
    // cudnn 和 miopen 在移动端不可用，而 T102591915 / T110194934 表明
    // compiledWithCuDNN() 检查有时会导致段错误（尽管我无法想象原因）
#if !defined(C10_MOBILE)
    if (needs_64bit_indexing_no_split(input, weight)) {
      return false;
    }
    if (!detail::getCUDAHooks().compiledWithCuDNN()) {
      return false;
    }
    if (!input.is_cuda() || !cudnn_enabled) {
      return false;
    }
    if (input.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kBFloat16) {
      if (!(detail::getCUDAHooks().supportsBFloat16ConvolutionWithCuDNNv8() && at::native::cudnnv8_enabled_check_debug())) {
        return false;
      }
    }
#endif
    return true;
  }
    # 检查是否推荐使用连续存储的内存格式来进行 cuDNN 卷积计算
    if (cudnn_conv_suggest_memory_format(input, weight) == at::MemoryFormat::Contiguous) {
      
      // 如果是 channels_last 卷积，并且需要确定性结果，则跳过膨胀检查
      if (deterministic && is_dilated()) {
        // 当前情况下，cuDNN 尚不完全支持确定性膨胀卷积
        return false;
      }
      
      // 如果存在膨胀操作，检查当前系统是否支持 cuDNN 的膨胀卷积，并且输出填充不大
      if (is_dilated()) {
        return detail::getCUDAHooks().supportsDilatedConvolutionWithCuDNN() && !is_output_padding_big();
      }
    }
    
    // 如果不是连续存储的内存格式推荐，或者前述条件不满足，则检查输出填充是否不大
    return !is_output_padding_big();
  // 如果不符合前面条件，则返回 false
  else
    return false;
#endif
  }

  // 使用 cudnn 进行 FP16 深度卷积
  bool use_cudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const  {
    // 如果推荐的内存格式不是连续的，并且可以使用 cudnn，则始终使用 cudnn_depthwise
    if (cudnn_conv_suggest_memory_format(input, weight) != at::MemoryFormat::Contiguous && use_cudnn(input, weight)) {
      return true;
    }
    // 如果当前环境支持 cudnn 的深度卷积
    if (detail::getCUDAHooks().supportsDepthwiseConvolutionWithCuDNN()) {
      // 获取 cudnn 的版本号
      long cudnn_version = detail::getCUDAHooks().versionCuDNN();
      // 如果 cudnn 版本大于等于 8200
      if (cudnn_version >= 8200) {
        // 检查条件是否满足，包括使用 cudnn、输入和权重都是半精度、是深度卷积、输入维度是4、不是扩展卷积、stride 符合条件、通道数大于等于32
        bool kernel_cond =  (use_cudnn(input, weight) &&
                             input.scalar_type() == kHalf && // 只适用于半精度
                             weight.scalar_type() == kHalf &&
                             is_depthwise(input, weight) &&
                             input.ndimension() == 4 &&   // TODO: 5-D contiguous depthwise is not supported yet, need benchmarks
                             !is_dilated() && // 不支持扩展卷积
                             (stride[0] == stride[1] || at::symint::size<T>(input, 2) == 1) && // 正方形或者1维
                             at::symint::size<T>(input, 1) >= 32); // 至少支持32个通道）
        // 如果条件满足，检查 cudnn 深度卷积工作负载
        if (kernel_cond) {
          return check_cudnn_depthwise_workload_with_filter<T>(input, stride[1], weight);
        }
      }
      // 如果 cudnn 版本在 7600 到 8200 之间，代码保持不变
      bool kernel_cond =  (cudnn_version >= 7600 &&
                           use_cudnn(input, weight) &&
                           input.scalar_type() == kHalf && // 只适用于半精度
                           weight.scalar_type() == kHalf &&
                           is_depthwise(input, weight) &&
                           input.ndimension() == 4 &&   // TODO: 5-D contiguous depthwise is not supported yet, need benchmarks
                           at::symint::size<T>(weight, 2) == at::symint::size<T>(weight, 3) && // 只有方形的卷积核
                           at::symint::size<T>(input, 2) >= 7 && // 最小宽度/高度为7
                           !is_dilated() && // 不支持扩展卷积
                           stride[0] == stride[1] && // 相等的步幅
                           ((at::symint::size<T>(weight, 3) == 3) || (at::symint::size<T>(weight, 3) == 1)) &&
                           at::symint::size<T>(input, 1) >= 32); // 至少支持32个通道）
      // 如果条件满足，检查 cudnn 深度卷积工作负载
      if (kernel_cond) {
        return check_cudnn_depthwise_workload<T>(input, stride[0]);
      } else {
        return false;
      }
    } else {
      // 如果不支持 cudnn 的深度卷积，则返回 false
      return false;
    }
  }

  // 使用 miopen 进行卷积计算
  bool use_miopen(const at::Tensor& input, const at::Tensor& weight, bool bias_defined) const  {
    // 如果需要使用64位索引且不能拆分，则返回 false
    if (needs_64bit_indexing_no_split(input, weight)) {
      return false;
    }
    // 检查输入张量的标量类型是否为 float、half 或 bfloat16，并且当前环境编译使用了 MIOpen 库
    // 并且输入张量位于 CUDA 设备上，并且张量维度不超过 MIOpen 所支持的最大维度
    // 并且如果存在分组卷积且存在扩张操作，则不使用 MIOpen，因为 MIOpen 不支持这种情况
    // 并且当前环境启用了 cuDNN 库
    return ((input.scalar_type() == at::kFloat) || (input.scalar_type() == at::kHalf) || (input.scalar_type() == at::kBFloat16))
           && detail::getCUDAHooks().compiledWithMIOpen()
           && input.is_cuda()
           && input.dim() <= MIOPEN_DIM_MAX
           && !(groups > 1 && is_dilated()) // MIOpen currently does not support dilation with groups of size > 1
           && cudnn_enabled
           ;
  }
  bool use_mkldnn(const at::Tensor& input, const at::Tensor& weight) const  {
#if AT_MKLDNN_ENABLED()
    // 检查是否启用了MKLDNN，并且用户未禁用MKLDNN，否则返回false
    if (!at::globalContext().userEnabledMkldnn()) {
      return false;
    }
    // 如果输入被转置并且输出填充较大，则返回false
    if (transposed && is_output_padding_big()) {
      return false;
    }
    // 如果输入张量在CPU上，并且（标量类型为kBFloat16且满足MKLDNN的BF16设备检查条件，或者标量类型为kHalf且满足MKLDNN的FP16设备检查条件），则返回true
    if (input.device().is_cpu() &&
        ((input.scalar_type() == at::kBFloat16 && mkldnn_bf16_device_check()) ||
         (input.scalar_type() == at::kHalf && mkldnn_fp16_device_check()))) {
      return true;
    }
    // 否则，如果输入是MKLDNN张量或者以下条件满足，则返回true：
    // - 输入在CPU上且标量类型为kFloat
    // - 对于1x1过滤器，多线程情况下MKLDNN比THNN更快，但单线程情况下THNN更快
    // - 满足一定条件的步幅、扩展或张量尺寸
    // - 线程数大于1时
    return (input.is_mkldnn()) || // input is mkldnn Tensor
      (input.device().is_cpu() &&
       input.scalar_type() == kFloat && // only on CPU Float Tensors
       // For 1x1 filters, MKLDNN is faster than THNN when multi-threaded,
       // but THNN is faster when single-threaded.
       (is_strided() || is_dilated() || at::symint::size<T>(input, 0) >= 16 ||
        at::symint::size<T>(weight, -1) != 1 || at::symint::size<T>(weight, -2) != 1 || at::get_num_threads() > 1) &&
       (groups > 1
        || (at::symint::size<T>(weight, -1) > 3 && at::symint::size<T>(weight, -2) > 3)
        || at::symint::size<T>(input, 0) > 1
        || at::symint::size<T>(input, 0)*at::symint::size<T>(input, 1)*at::symint::size<T>(input, 2)*at::symint::size<T>(input, 3) > 20480) // for some case, native is faster
        );

#endif
    // 如果未启用MKLDNN，则返回false
    return false;
  }
  // 检查是否可以使用NNPACK进行计算
  bool use_nnpack(const at::Tensor& input, const at::Tensor& weight) const  {
#if AT_NNPACK_ENABLED()
    // 检查用户是否启用了NNPACK，并且NNPACK可用，且输入张量在CPU上且标量类型为kFloat，且未扩展，未转置，且是四维张量（NCHW格式）
    return at::globalContext().userEnabledNNPACK() &&
           at::_nnpack_available() &&
           input.device().is_cpu() &&
           input.scalar_type() == kFloat && // only on CPU Float Tensors
           !is_dilated() && // or dilation
           !transposed &&   // or transposed tensors
           input.ndimension() == 4 && // must be in NCHW format
           weight.ndimension() == 4 &&
           (at::symint::size<T>(weight, 2) < 17) && (at::symint::size<T>(weight, 3) < 17) && // NNPACK only supports kernels up to 16x16
           (padding[0] < at::symint::size<T>(weight, 2)) && (padding[1] < at::symint::size<T>(weight, 3)) // NNPACK only supports padding < kernel_size. See https://github.com/pytorch/pytorch/issues/90142.
#if !defined(C10_MOBILE)
           && at::symint::size<T>(input, 0) >= 16 // ensure large enough batch size to ensure perf, tuneable
#endif
       ;
#endif
    // 如果未启用NNPACK，则返回false
    return false;
  }
  // 检查是否可以使用XNNPACK进行计算
  bool use_xnnpack(const at::Tensor& input, const at::Tensor& weight,
                   const at::OptionalArrayRef<T> bias_sizes_opt) const {
#if defined(C10_MOBILE)
    // 如果未转置，则检查输入是否符合特定条件，并调用XNNPACK的卷积函数进行检查
    if (!transposed) {
      // NB: for the call here, it MATTERS that we are templated. If you
      // untemplate this to always use SymInt, the function
      // xnnpack_use_convolution2d will always return false
      return (at::symint::size<T>(input, 1) == groups) &&
              xnnpack_use_convolution2d(
                  input,
                  weight,
                  bias_sizes_opt,
                  padding,
                  stride,
                  dilation,
                  groups,
                  transposed);
    }
#endif
    return false;
  }

  bool use_mps(const at::Tensor& input, const at::Tensor& weight) const {
    // 这些检查需要进行扩展。目前我们对 MPS 的检查非常有限。
#ifdef USE_MPS
    // 如果需要64位索引且不分割，则返回false
    if (needs_64bit_indexing_no_split(input, weight)) {
      return false;
    }
    // 如果输入张量不是MPS张量，则返回false
    if (!input.is_mps()) {
      return false;
    }
    // 否则返回true
    return true;
#else
    // 如果未定义USE_MPS，则直接返回false
    return false;
#endif
  }

  // 当前仅支持深度可分卷积的情况，其中groups == nInputPlane且nInputPlane == nOutputPlane（由于缺少深度可分乘数）
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const  {
    // 输入张量在CUDA上，并且未转置
    return input.is_cuda() &&
           !transposed &&
           // 输入张量维度为4或5
           (input.ndimension() == 4 || input.ndimension() == 5) &&
           // 输入张量的第二个维度大小等于groups
           at::symint::size<T>(input, 1) == groups &&
           // groups大于1，如果只有一个组则没有意义
           groups > 1 &&
           // 权重张量的第一个维度大小必须是输入张量第一个维度大小的倍数，用于输出通道数是输入通道数的倍数
           at::symint::size<T>(weight, 0) % at::symint::size<T>(input, 1) == 0;
  }
};

// 下面是一系列函数的定义和注册，用于不同类型的卷积操作的后向传播
DEFINE_DISPATCH(conv_depthwise2d_backward_stub);
DEFINE_DISPATCH(conv_depthwise3d_backward_stub);
DEFINE_DISPATCH(cudnn_convolution_backward_stub);
DEFINE_DISPATCH(cudnn_convolution_transpose_backward_stub);
DEFINE_DISPATCH(slow_conv_transpose3d_backward_stub);
DEFINE_DISPATCH(convolution_depthwise3x3_winograd_stub);
DEFINE_DISPATCH(miopen_convolution_backward_stub);
DEFINE_DISPATCH(miopen_convolution_transpose_backward_stub);
DEFINE_DISPATCH(miopen_depthwise_convolution_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_transpose_stub);
DEFINE_DISPATCH(mkldnn_convolution_transpose_backward_stub);
DEFINE_DISPATCH(slow_conv_dilated2d_backward_stub);
DEFINE_DISPATCH(slow_conv_dilated3d_backward_stub);
DEFINE_DISPATCH(slow_conv_transpose2d_backward_stub);
REGISTER_NO_CPU_DISPATCH(conv_depthwise2d_backward_stub);
REGISTER_NO_CPU_DISPATCH(conv_depthwise3d_backward_stub);
REGISTER_NO_CPU_DISPATCH(cudnn_convolution_backward_stub);
REGISTER_NO_CPU_DISPATCH(cudnn_convolution_transpose_backward_stub);
REGISTER_NO_CPU_DISPATCH(miopen_convolution_backward_stub);
REGISTER_NO_CPU_DISPATCH(miopen_convolution_transpose_backward_stub);
REGISTER_NO_CPU_DISPATCH(miopen_depthwise_convolution_backward_stub);

// 重载输出流操作符<<，用于打印ConvParams<T>对象的信息
template <typename T>
std::ostream& operator<<(std::ostream & out, const ConvParams<T>& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << ArrayRef<T>{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << ArrayRef<T>{params.output_padding}
      << "  groups = " << params.groups
      << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic
      << "  cudnn_enabled = " << params.cudnn_enabled
      << "  allow_tf32 = " << params.allow_tf32
      << "}";
  return out;
}

template <typename T>
# 检查前向传播的形状，验证输入参数和权重参数的一致性
static void check_shape_forward(const at::Tensor& input,
                                const c10::ArrayRef<T>& weight_sizes, const at::Tensor& bias,
                                const ConvParams<T>& params) {
  # 获取输入张量的维度数
  int64_t k = input.ndimension();
  # 获取权重尺寸的维度数
  int64_t weight_dim = weight_sizes.size();
  # 获取分组数
  auto groups = params.groups;
  # 获取填充值
  const auto& padding = params.padding;
  # 获取扩张值
  const auto& dilation = params.dilation;
  # 获取是否是转置卷积
  bool transposed = params.transposed;

  # 检查是否存在负填充
  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  # 检查是否存在负输出填充
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  # 检查是否存在非正的步幅
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");
  # 检查是否存在负的扩张
  TORCH_CHECK(!params.is_dilation_neg(), "dilation should be greater than zero");

  # 检查输入张量的维度是否与期望的权重维度一致
  TORCH_CHECK(weight_dim == k,
           "Expected ", weight_dim, "-dimensional input for ", weight_dim,
           "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
           at::symint::sizes<T>(input), " instead");
  # 检查权重的第一个维度是否大于等于分组数
  TORCH_CHECK(weight_sizes[0] >= groups,
           "Given groups=", groups, ", expected weight to be at least ", groups,
           " at dimension 0, but got weight of size ", weight_sizes, " instead");
  # 检查权重的第一个维度是否能被分组数整除
  TORCH_CHECK(weight_sizes[0] % groups == 0,
           "Given groups=", groups, ", expected weight to be divisible by ",
           groups, " at dimension 0, but got weight of size [", weight_sizes,
           "] instead");

  if (!transposed) {
    # 初始化输入形状和核心形状的向量
    std::vector<T> input_shape;
    std::vector<T> kernel_shape;
    bool kernel_size_correct = true;

    # 检查输入张量的第二维度是否与期望的权重通道数一致
    TORCH_CHECK(at::symint::size<T>(input, 1) == (weight_sizes[1] * groups),
                "Given groups=", groups, ", weight of size ", weight_sizes,
                ", expected input", at::symint::sizes<T>(input), " to have ",
                (weight_sizes[1] * groups), " channels, but got ", at::symint::size<T>(input, 1),
                " channels instead");

    # 检查是否定义了偏置或者偏置的维度与权重的第一个维度一致
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && at::symint::size<T>(bias, 0) == weight_sizes[0]),
             "Given weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
             ", but got bias of size ", at::symint::sizes<T>(bias), " instead");

    # 遍历除去前两个维度外的所有维度
    for (const auto i : c10::irange(2, k)) {
      # 计算考虑扩张后的新核心大小
      input_shape.push_back(at::symint::size<T>(input, i) + 2 * padding[i-2]);
      # 计算核心的新形状并记录日志
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      # 检查核心大小是否正确
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    # 检查输入形状和核心形状的一致性
    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");
    // 如果卷积核大小不正确
    if (!kernel_size_correct) {
      // 创建用于存储错误信息的字符串流
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      // 用于分隔元素的字符串
      std::string separator = "";

      // 遍历输入形状列表，将每个维度的大小添加到输入字符串流中
      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";  // 更新分隔符为 " x "
      }

      // 抛出错误，包括输入形状和卷积核大小信息
      AT_ERROR("Calculated padded input size per channel: (", input_ss.str(), "). "
               "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    // 检查输入的通道数是否与期望的权重大小匹配
    TORCH_CHECK(at::symint::size<T>(input, 1) == weight_sizes[0],
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected input", at::symint::sizes<T>(input), " to have ", weight_sizes[0],
             " channels, but got ", at::symint::size<T>(input, 1), " channels instead");
    // 检查偏置项的维度是否正确，如果存在的话
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && at::symint::size<T>(bias, 0) == weight_sizes[1] * groups),
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
             ", but got bias of size ", at::symint::sizes<T>(bias), " instead");
  }


注释：
}

// 定义静态函数 `check_shape_backward`，用于检查反向传播时输入形状的有效性
template <typename T>
static void check_shape_backward(
    const at::Tensor& input,                   // 输入张量
    const c10::ArrayRef<T>& weight_sizes,      // 权重尺寸的引用数组
    const ConvParams<T>& params) {             // ConvParams 类型的参数对象

  // 调用 `check_shape_forward` 函数，检查前向传播时输入的形状
  check_shape_forward<T>(input, weight_sizes, /*bias=*/ Tensor(), params);
}

// 给定输入张量和预期的空间维度数量，检查输入形状是否有效，并返回带有批次维度的输入
//
// Args:
//     input (Tensor): 输入张量
//     num_spatial_dims (int): 预期输入的空间维度数量
//     func_name (string): 用于生成无效输入的友好错误消息的函数名称
//
// Returns:
//     std::tuple<Tensor, bool>: 包含以下内容的元组
//         batched_input (Tensor): 带有批次维度的输入张量
//         is_batched (bool): 指示原始输入是否已经带有批次维度
static std::tuple<Tensor, bool> batchify(
    const Tensor& input,                      // 输入张量
    const int64_t num_spatial_dims,           // 空间维度的数量
    const std::string& func_name) {           // 函数名称字符串

  // 假设 NTs 总是带有批次维度
  if (input.is_nested()) {
    return std::make_tuple(input, true);
  }

  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = (input.dim() == dim_count_batch);

  // 使用 TORCH_CHECK 检查输入的维度是否符合预期
  TORCH_CHECK(input.dim() == dim_count_no_batch || is_batched,
      "Expected ", dim_count_no_batch, "D (unbatched) or ", dim_count_batch,
      "D (batched) input to ", func_name, ", but got input of size: ", input.sizes());

  // 返回带有批次维度的输入张量或者将未带批次维度的输入张量增加批次维度后返回
  return std::make_tuple(is_batched ? input : input.unsqueeze(0), is_batched);
}

// 检查输入张量与参数张量的类型是否相同
static void check_input_same_type_as_parameters(
    const Tensor& input,                      // 输入张量
    const Tensor& weight,                     // 权重张量
    const Tensor& bias) {                     // 偏置张量

  // 使用 TORCH_CHECK 检查输入张量与权重张量的类型是否相同
  TORCH_CHECK(input.options().type_equal(weight.options()),
      "Input type (", input.toString(), ") and weight type (", weight.toString(),
      ") should be the same");

  // 如果偏置张量已定义，则使用 TORCH_CHECK 检查输入张量与偏置张量的类型是否相同
  TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (", input.toString(), ") and bias type (", bias.toString(),
      ") should be the same");
}

// 重载函数 `check_input_same_type_as_parameters`，不带偏置张量参数
static void check_input_same_type_as_parameters(
    const Tensor& input,                      // 输入张量
    const Tensor& weight) {                   // 权重张量

  // 调用带偏置参数的 `check_input_same_type_as_parameters` 函数
  check_input_same_type_as_parameters(input, weight, /*bias=*/ Tensor());
}

// 如果 MKLDNN 可用，检查输入张量与参数张量的类型是否相同
#if AT_MKLDNN_ENABLED()
static void check_input_same_type_as_parameters(
    const Tensor& input,                      // 输入张量
    const Tensor& weight,                     // 权重张量
    const Tensor& bias,                       // 偏置张量
    const ConvBackend backend) {              // ConvBackend 枚举类型的参数

  // 如果使用 MKLDNN 或 MKLDNNTranspose，则检查输入张量与权重张量的类型是否相同
  if (backend == ConvBackend::Mkldnn || backend == ConvBackend::MkldnnTranspose) {
    TORCH_CHECK(input.options().type_equal(weight.options())
        || (input.is_mkldnn() && weight.device().is_cpu() && weight.scalar_type() == kFloat),
        "Input type (", input.toString(), ") and weight type (", weight.toString(),
        ") should be the same or input should be a MKLDNN tensor and weight is a dense tensor");
    # 如果没有偏置项或者偏置项与输入张量类型相同，或者输入张量是 MKLDNN 张量且偏置是密集张量，检查通过
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options()))
        || (input.is_mkldnn() && bias.device().is_cpu() && bias.scalar_type() == kFloat),
        "Input type (", input.toString(), ") and bias type (", bias.toString(),
        ") should be the same or input should be a MKLDNN tensor and bias is a dense tensor");
  } else {
    # 否则，检查输入张量与权重和偏置项的类型是否一致
    check_input_same_type_as_parameters(input, weight, bias);
  }
}

#endif

// 定义静态函数 view4d，接收一个 Tensor 参数，返回一个 Tensor
static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  // 使用 TORCH_CHECK 检查 tensor 的维度是否为 3，如果不是则抛出错误信息
  TORCH_CHECK(tensor.ndimension() == 3,
           "expected 3D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  // 返回在第2维上增加一个维度后的 tensor
  return tensor.unsqueeze(2);
}

// 定义静态函数 view3d，接收一个 Tensor 参数，返回一个 Tensor
static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  // 使用 TORCH_CHECK 检查 tensor 的维度是否为 4，如果不是则抛出错误信息
  TORCH_CHECK(tensor.ndimension() == 4,
           "expected 4D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  // 返回在第2维上压缩一个维度后的 tensor
  return tensor.squeeze(2);
}

// 定义 subtensor 函数，接收一个 tensor 引用和三个整数参数，返回一个 Tensor
static at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  // 如果 tensor 未定义，则返回一个未定义的 Tensor
  if (!tensor.defined()) {
    return at::Tensor();
  }
  // 获取 tensor 建议的内存格式
  const auto memory_format = tensor.suggest_memory_format();
  // 计算在指定维度上的子张量，并保证其内存格式与原始 tensor 一致
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous(memory_format);
}

// 定义一个匿名命名空间
namespace {

// 定义函数 complex_to_real，接收一个 Tensor 参数，返回一对 Tensor
std::pair<Tensor, Tensor> complex_to_real(const Tensor& inp) {
  // 将输入 inp 视作实部和虚部组成的复数张量
  auto inp_view_as_complex = at::view_as_real(inp);
  // 获取 inp_view_as_complex 的最后一个维度
  auto dim_i = inp_view_as_complex.dim() - 1;
  // 选择 inp_view_as_complex 的第一个和第二个通道作为实部和虚部
  auto i_r = inp_view_as_complex.select(dim_i, 0);
  auto i_i = inp_view_as_complex.select(dim_i, 1);
  // 返回实部和虚部构成的一对 Tensor
  return std::make_pair(i_r, i_i);
}

// 定义函数 complex_convolution，接收多个 Tensor 参数和一些标量参数，返回一个 Tensor
at::Tensor complex_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    SymIntArrayRef stride,
    SymIntArrayRef padding,
    SymIntArrayRef dilation,
    bool transposed,
    SymIntArrayRef output_padding,
    c10::SymInt groups) {
  // 检查输入的类型是否与参数的类型相同
  check_input_same_type_as_parameters(input, weight, bias);
  // 将输入 input 视作解析共轭后的实部和虚部
  auto [i_r, i_i] = complex_to_real(input.resolve_conj());
  // 将权重 weight 视作解析共轭后的实部和虚部
  auto [w_r, w_i] = complex_to_real(weight.resolve_conj());

  // [NOTE] Complex Convolution
  // 复数卷积的计算过程说明
  // conv(W, x, b) = conv(Wr, xr, br) - conv(Wi, xi, 0) + i(conv(Wi, xr, bi) + conv(Wr, xi, 0))
  // 使用高斯技巧进行分解和计算
  // 其中 W, x 和 b 都是复数输入。
  // 定义变量 a, b, c 来存储不同部分的卷积结果
  Tensor a, b, c;
  // 如果没有定义偏置 bias
  if (!bias.defined()) {
    // 计算实部部分的卷积 a
    a = at::convolution_symint(i_r, w_r, bias, stride, padding, dilation, transposed, output_padding, groups);
    // 计算虚部部分的卷积 b
    b = at::convolution_symint(i_i, w_i, bias, stride, padding, dilation, transposed, output_padding, groups);
    // 计算实部加虚部的卷积 c
    c = at::convolution_symint(i_r + i_i, w_r + w_i, bias, stride, padding, dilation, transposed, output_padding, groups);
  } else {
    // 如果定义了偏置 bias
    auto [b_r, b_i] = complex_to_real(bias.resolve_conj());
    // 计算实部部分的卷积 a
    a = at::convolution_symint(i_r, w_r, b_r, stride, padding, dilation, transposed, output_padding, groups);
    // 计算虚部部分的卷积 b
    b = at::convolution_symint(i_i, w_i, Tensor(), stride, padding, dilation, transposed, output_padding, groups);
    // 计算实部加虚部的卷积 c
    c = at::convolution_symint(i_r + i_i, w_r + w_i, b_r + b_i, stride, padding, dilation, transposed, output_padding, groups);
  }

  // 定义复数单位 i
  auto i = c10::Scalar(c10::complex<double>(0, 1));
  // 返回复数卷积的最终结果
  return a - b + i * (c - a - b);
}

// 定义函数 complex_convolution_mode，接收多个 Tensor 参数和一些标量参数，返回一个 Tensor
at::Tensor complex_convolution_mode(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymIntArrayRef stride,
    c10::string_view padding,                      // 接收填充参数
    c10::SymIntArrayRef dilation,                 // 接收膨胀参数
    c10::SymInt groups) {                         // 接收分组参数
  auto bias = bias_opt.value_or(Tensor());        // 如果存在偏置，使用其值；否则使用空张量
  check_input_same_type_as_parameters(input, weight, bias);  // 检查输入、权重、偏置的数据类型是否一致
  auto [i_r, i_i] = complex_to_real(input.resolve_conj());  // 将输入复数张量解析为实部和虚部
  auto [w_r, w_i] = complex_to_real(weight.resolve_conj()); // 将权重复数张量解析为实部和虚部

  // 查看 [NOTE] 复数卷积
  Tensor a, b, c;
  if (!bias.defined()) {                         // 如果偏置未定义
    a = at::_convolution_mode_symint(i_r, w_r, bias, stride, padding, dilation, groups);  // 计算实部的卷积
    b = at::_convolution_mode_symint(i_i, w_i, bias, stride, padding, dilation, groups);  // 计算虚部的卷积
    c = at::_convolution_mode_symint(i_r + i_i, w_r + w_i, bias, stride, padding, dilation, groups);  // 计算复数部分的卷积
  } else {
    auto [b_r, b_i] = complex_to_real(bias.resolve_conj());  // 将偏置复数张量解析为实部和虚部
    a = at::_convolution_mode_symint(i_r, w_r, b_r, stride, padding, dilation, groups);  // 计算实部的卷积
    b = at::_convolution_mode_symint(i_i, w_i, Tensor(), stride, padding, dilation, groups);  // 计算虚部的卷积，偏置为空张量
    c = at::_convolution_mode_symint(i_r + i_i, w_r + w_i, b_r + b_i, stride, padding, dilation, groups);  // 计算复数部分的卷积
  }

  auto i = c10::Scalar(c10::complex<double>(0, 1));  // 创建一个复数单位 i
  return a - b + i * (c - a - b);  // 返回复数卷积结果
}
} // namespace
    output = at::convolution_symint(input, weight, bias, stride, padding, dilation, false, {{0, 0, 0}}, groups);
    
    调用 PyTorch 的对称整数卷积函数 `at::convolution_symint`，并传入以下参数：
    - `input`: 输入张量
    - `weight`: 卷积核张量
    - `bias`: 偏置张量
    - `stride`: 卷积步长
    - `padding`: 卷积填充
    - `dilation`: 卷积膨胀率
    - `false`: 表示不进行转置卷积
    - `{{0, 0, 0}}`: 控制对称整数卷积的一组参数，具体含义依赖于实现
    - `groups`: 卷积分组数
    
    
    return is_batched ? std::move(output) : output.squeeze(0);
    
    根据 `is_batched` 变量的值决定返回的结果：
    - 如果 `is_batched` 为真，则移动 `output` 的所有权并返回，否则对 `output` 在第0维进行压缩（squeeze）并返回。
# 静态方法，实现了“same”填充的卷积操作
static Tensor convolution_same(
    const Tensor &input, const Tensor &weight, const Tensor &bias,
    SymIntArrayRef stride, SymIntArrayRef dilation, c10::SymInt groups) {

  # 获取权重张量的维度数
  auto k = weight.dim();
  # 检查权重张量维度至少为3，否则引发错误
  TORCH_CHECK(k > 2, "weight should have at least three dimensions");
  # 检查分组数量大于0，不支持非正数分组
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");
  # 计算输入张量的维度（去除批次和通道维度）
  auto dim = static_cast<size_t>(k - 2);
  # 获取权重张量的符号化大小
  auto weight_sizes = weight.sym_sizes();
  # 获取输入张量的符号化大小
  auto input_sizes = input.sym_sizes();
  # 检查权重张量和输入张量的维度匹配
  TORCH_CHECK(k == input.dim(),
              "Expected ", k, "-dimensional input for ",
              k, "-dimensional weight", weight_sizes, ", but got ",
              input.dim(), "-dimensional input of size ",
              input.sizes(), " instead");
  # 检查步幅的大小是否与维度匹配或可以广播
  TORCH_CHECK(stride.size() == dim || stride.size() == 1U,
              "stride cannot broadcast to ", dim, " dimensions");
  # 检查膨胀的大小是否与维度匹配或可以广播
  TORCH_CHECK(dilation.size() == dim || dilation.size() == 1U,
              "dilation cannot broadcast to ", dim, " dimensions");
  # 遍历步幅，如果有大于1的步幅，则不支持"padding='same'"
  for (auto i: c10::irange(stride.size())) {
    TORCH_CHECK(stride[i] == 1, "padding='same' is not supported for strided convolutions");
  }

  // 计算正确的填充值
  SymDimVector padding_l, padding_r;
  bool symmetric_padding = true;
  for (auto i: c10::irange(dim)) {
    auto s = stride.size() == 1 ? stride[0] : stride[i];
    auto d = dilation.size() == 1 ? dilation[0] : dilation[i];
    auto pad = pooling_same_mode_padding_lr(
        input_sizes[i + 2], weight_sizes[i + 2], s, d);
    padding_l.push_back(pad.first);
    padding_r.push_back(pad.second);
    if (pad.first != pad.second) {
      symmetric_padding = false;
    }
  }

  # 如果填充是对称的，则使用所有后端原生支持的对称填充
  if (symmetric_padding) {
    SymDimVector output_padding(static_cast<size_t>(dim));
    return at::convolution_symint(input, weight, bias, stride, padding_l, dilation,
                               false, output_padding, groups);
  }

  # 如果填充不对称，发出警告并处理非对称填充
  TORCH_WARN_ONCE("Using padding='same' with even kernel lengths and odd dilation may"
                  " require a zero-padded copy of the input be created");
  SmallVector<c10::SymInt, kDimVectorStaticSize * 2> pad_nd(static_cast<size_t>(2 * dim));
  for (auto i: c10::irange(dim)) {
    # 计算填充的差值，以实现对称填充
    auto delta_pad = padding_r[i] - padding_l[i];
    # F.pad 从最后一个维度到第一个维度进行填充
    auto pad_idx = 2 * (dim - 1 - i);
    if (delta_pad > 0) {
      pad_nd[pad_idx + 1] = delta_pad;
    } else {
      pad_nd[pad_idx] = delta_pad;
      padding_l[i] = padding_r[i];
    }
  }
  # 对输入进行常量填充，以实现非对称填充
  auto padded_input = at::constant_pad_nd_symint(input, pad_nd, 0);
  SymDimVector output_padding(static_cast<size_t>(dim));
  # 执行对称整数卷积操作
  return at::convolution_symint(padded_input, weight, bias, stride, padding_l,
                                dilation, false, output_padding, groups);
}
  // 将可能的可选张量借用为常规张量，用于避免可能为空的情况
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 解引用得到常规张量作为偏置项
  const Tensor& bias = *bias_maybe_owned;

  // 根据填充方式进行卷积操作的选择
  if (padding == "same") {
    // 执行“same”填充方式的卷积操作
    return at::native::convolution_same(
        input, weight, bias, stride, dilation, groups);
  } else if (padding == "valid") {
    // 执行“valid”填充方式的卷积操作
    return at::convolution_symint(
        input, weight, bias, stride, {{0}}, dilation, false, {{0}}, groups);
  }
  // 若填充方式既不是“same”也不是“valid”，则抛出错误
  TORCH_CHECK(false, "Invalid padding string: '", padding, "'");
// 定义一个对输入进行1维对称整数填充卷积操作的函数，返回输出张量
at::Tensor conv1d_padding_symint(
    // 输入张量、卷积核张量、可选的偏置张量、步长、填充方式、扩展率、分组数作为参数
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias,
    c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation,
    c10::SymInt groups) {
  // 对输入张量进行批处理，返回处理后的张量及是否进行了批处理的标志
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 1, "conv1d");
  // 定义输出张量
  Tensor output;
  // 如果输入张量的数据类型是复数类型
  if (at::isComplexType(input_.scalar_type())) {
    // 调用复数卷积模式处理函数，生成输出张量
    output = complex_convolution_mode(input, weight, bias, stride, std::move(padding), dilation, groups);
  } else {
    // 调用整数对称卷积模式处理函数，生成输出张量
    output = at::_convolution_mode_symint(input, weight, bias, stride, std::move(padding), dilation, groups);
  }
  // 如果进行了批处理，则移动输出张量；否则将第0维度压缩为标量后返回
  return is_batched ? std::move(output) : output.squeeze(0);
}

// 定义一个对输入进行2维对称整数填充卷积操作的函数，返回输出张量
at::Tensor conv2d_padding_symint(
    // 输入张量、卷积核张量、可选的偏置张量、步长、填充方式、扩展率、分组数作为参数
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias,
    c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation,
    c10::SymInt groups) {
  // 对输入张量进行批处理，返回处理后的张量及是否进行了批处理的标志
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 2, "conv2d");
  // 定义输出张量
  Tensor output;
  // 如果输入张量的数据类型是复数类型
  if (at::isComplexType(input_.scalar_type())) {
    // 调用复数卷积模式处理函数，生成输出张量
    output = complex_convolution_mode(input, weight, bias, stride, std::move(padding), dilation, groups);
  } else {
    // 调用整数对称卷积模式处理函数，生成输出张量
    output = at::_convolution_mode_symint(input, weight, bias, stride, std::move(padding), dilation, groups);
  }
  // 如果进行了批处理，则移动输出张量；否则将第0维度压缩为标量后返回
  return is_batched ? std::move(output) : output.squeeze(0);
}

// 定义一个对输入进行3维对称整数填充卷积操作的函数，返回输出张量
at::Tensor conv3d_padding_symint(
    // 输入张量、卷积核张量、可选的偏置张量、步长、填充方式、扩展率、分组数作为参数
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias,
    c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation,
    c10::SymInt groups) {
  // 对输入张量进行批处理，返回处理后的张量及是否进行了批处理的标志
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 3, "conv3d");
  // 定义输出张量
  Tensor output;
  // 如果输入张量的数据类型是复数类型
  if (at::isComplexType(input_.scalar_type())) {
    // 调用复数卷积模式处理函数，生成输出张量
    output = complex_convolution_mode(input, weight, bias, stride, std::move(padding), dilation, groups);
  } else {
    // 调用整数对称卷积模式处理函数，生成输出张量
    output = at::_convolution_mode_symint(input, weight, bias, stride, std::move(padding), dilation, groups);
  }
  // 如果进行了批处理，则移动输出张量；否则将第0维度压缩为标量后返回
  return is_batched ? std::move(output) : output.squeeze(0);
}

// 定义一个对输入进行1维对称整数填充转置卷积操作的函数，返回输出张量
at::Tensor conv_transpose1d_symint(
    // 输入张量、卷积核张量、可选的偏置张量、步长、填充、输出填充、分组数、扩展率作为参数
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
  // 使用 borrow_from_optional_tensor 从可选的偏置张量获取对应的引用
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 对输入张量进行批处理，返回处理后的张量及是否进行了批处理的标志
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 1, "conv_transpose1d");
  // 定义输出张量
  Tensor output;
  // 如果输入张量的数据类型是复数类型
  if (at::isComplexType(input_.scalar_type())) {
    // 调用复数卷积处理函数，生成输出张量
    output = complex_convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  } else {
    // 调用整数对称卷积处理函数，生成输出张量
    output = at::convolution_symint(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  }
  // 如果进行了批处理，则移动输出张量；否则将第0维度压缩为标量后返回
  return is_batched ? std::move(output) : output.squeeze(0);
}

// 定义一个对输入进行2维对称整数填充转置卷积操作的函数，返回输出张量
at::Tensor conv_transpose2d_symint(
    // 输入张量、卷积核张量、可选的偏置张量、步长、填充、输出填充、分组数、扩展率作为参数
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
    // 声明一个函数，接受输入张量 input_、权重张量 weight、可选的偏置张量 bias_opt、
    // 对称整数数组 stride、padding、output_padding、dilation 和对称整数 groups 作为参数
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
      // 查看注释：为了处理可选张量的包装层
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      // 解包可能拥有的偏置张量，得到具体的偏置张量 bias
      const Tensor& bias = *bias_maybe_owned;
    
      // 将输入张量 input_ 进行批处理，返回批处理后的张量 input 和一个标志 is_batched
      auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 2, "conv_transpose2d");
      // 声明输出张量 output
      Tensor output;
      // 如果输入张量 input_ 是复数类型
      if (at::isComplexType(input_.scalar_type())) {
        // 进行复数卷积操作，得到输出张量 output
        output = complex_convolution(
          input, weight, bias, stride, padding, dilation, true, output_padding, groups);
      } else {
        // 进行对称整数卷积操作，得到输出张量 output
        output = at::convolution_symint(
          input, weight, bias, stride, padding, dilation, true, output_padding, groups);
      }
      // 如果进行了批处理，则将输出张量 output 移除批处理维度后返回，否则直接返回输出张量 output
      return is_batched ? std::move(output) : output.squeeze(0);
    }
}

// 对称整数卷积转置操作函数
at::Tensor conv_transpose3d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
  // 查看注释：用于处理可选张量的包装器移除
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 强制解引用以获取实际的偏置张量
  const Tensor& bias = *bias_maybe_owned;

  // 将输入张量批量化处理，标记是否为批量输入
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 3, "conv_transpose3d");
  Tensor output;
  // 如果输入张量的数据类型为复数类型，则执行复数卷积操作
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  } else {
    // 否则执行对称整数卷积操作
    output = at::convolution_symint(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  }
  // 如果输入张量是批量输入，则移动输出张量的维度
  return is_batched ? std::move(output) : output.squeeze(0);
}

// 标准卷积操作函数
at::Tensor convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  // 查看注释：用于处理可选张量的包装器移除
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 强制解引用以获取实际的偏置张量
  const Tensor& bias = *bias_maybe_owned;

  // 获取全局上下文
  auto& ctx = at::globalContext();
  // 查看注释：启用确定性操作
  bool deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  // 调用底层卷积函数，并返回结果张量
  return at::_convolution(input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups,
                          ctx.benchmarkCuDNN(), deterministic, ctx.userEnabledCuDNN(), ctx.allowTF32CuDNN());
}

// 可重写的卷积操作函数，用于选择卷积后端
at::Tensor convolution_overrideable(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  // 抛出错误，指示未实现该函数，建议使用 TORCH_LIBRARY_IMPL 进行函数重写
  TORCH_CHECK_NOT_IMPLEMENTED(false, "convolution_overrideable not implemented. You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL to override this function ");
}

// 选择卷积后端的函数，基于输入和参数进行选择
// 此重载用于卷积内部但不对外公开
// 注意：前向传播提供偏置张量，而反向传播提供一个布尔值，指示是否定义了偏置。这样做是为了通过避免保存完整的偏置张量来节省内存。
template <typename T>
ConvBackend _select_conv_backend(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const at::OptionalArrayRef<T> bias_sizes_opt,
    const bool need_backward,
    // 按照给定的参数计算卷积操作的后端选择
    const ConvParams<T>& params) {
    // 检查输入是否为空，如果是，则根据输入是否是 MKLDNN 类型返回对应的 ConvBackend 类型
    if (at::symint::size<T>(input, 0) == 0 || at::symint::size<T>(input, 1) == 0) {
      return input.is_mkldnn() ? ConvBackend::MkldnnEmpty : ConvBackend::Empty;
    } else if (at::symint::numel<T>(input) == 0) {
      // 如果输入元素数量为零，则抛出错误，指明只支持零批次或零通道的输入
      TORCH_CHECK(false, "Only zero batch or zero channel inputs are supported, but got input shape: ", at::symint::sizes<T>(input));
    }
    
    // 检查是否为深度可分离卷积，并根据不同的条件选择对应的后端类型
    if (params.is_depthwise(input, weight)) {
      if (params.use_cudnn_depthwise(input, weight)) {
        return ConvBackend::Cudnn;
      } else if (params.use_miopen(input, weight, bias_sizes_opt.has_value())) {
        return ConvBackend::MiopenDepthwise;
      } else {
        // 根据输入的维度选择 CUDA 的深度可分离卷积类型
        if (input.ndimension() == 4) {
          return ConvBackend::CudaDepthwise2d;
        } else if (input.ndimension() == 5) {
          return ConvBackend::CudaDepthwise3d;
        } else {
          // 不支持的情况
        }
      }
    } else if (params.use_cudnn(input, weight)) {
      // 使用 cuDNN 的卷积操作，根据是否转置返回对应的后端类型
      if (params.transposed) {
        return ConvBackend::CudnnTranspose;
      } else {
        return ConvBackend::Cudnn;
      }
    } else if (params.use_miopen(input, weight, bias_sizes_opt.has_value())) {
      // 使用 MIOpen 的卷积操作，根据是否转置返回对应的后端类型
      if (params.transposed) {
        return ConvBackend::MiopenTranspose;
      } else {
        return ConvBackend::Miopen;
      }
    } else if (params.use_mkldnn(input, weight)) {
      // 使用 MKL-DNN 的卷积操作，根据是否转置返回对应的后端类型
      if (params.transposed) {
        return ConvBackend::MkldnnTranspose;
      } else {
        return ConvBackend::Mkldnn;
      }
    } else if (!need_backward && params.use_xnnpack(input, weight, bias_sizes_opt)) {
      // 如果不需要反向传播，并且使用 XNNPACK，则返回 XNNPACK 2D 的后端类型
      // 使用预打包的卷积是首选，但 XNNPACK 对于 NHWC 仍然是最快的选择
      return ConvBackend::Xnnpack2d;
    // 3x3 深度可分离卷积的实现仅限推断
    } else if (!need_backward && params.use_cpu_depthwise3x3_winograd(input, weight, bias)) {
      return ConvBackend::Winograd3x3Depthwise;
    } else if (
        !params.transposed && (input.ndimension() == 5) &&
        (input.device().is_cpu()) &&
        !params.is_dilated()) {
      // 用于分组卷积 3D 的快速路径
      return ConvBackend::Slow3d;
    } else if (input.device().is_cpu() || input.is_cuda()) {
      // 对于不支持分组的后端，根据是否转置和输入的维度选择对应的后端类型
      if (params.transposed) {
        if (input.ndimension() == 4) {
          return ConvBackend::SlowTranspose2d;
        } else if (input.ndimension() == 5) {
          return ConvBackend::SlowTranspose3d;
        } else {
          // 不支持的情况
        }
    } else {  /* Not transposed */
      // 检查输入是否为四维张量
      if (input.ndimension() == 4) {
        // 检查是否使用了扩展卷积
        if (params.is_dilated()) {
          // 返回适合的慢速扩展二维卷积后端
          return ConvBackend::SlowDilated2d;
        } else {  /* dim == 4, non-dilated */
          // 如果没有扩展且维度为四，检查是否可以使用 NNPACK 加速
          if (params.use_nnpack(input, weight)) {
            // 返回适合的 NNPACK 空间卷积后端
            return ConvBackend::NnpackSpatial;
          } else {
            /* CPU 实现在非扩展情况下具有专门的矩阵乘积内核 */
            // 返回适合的慢速二维卷积后端
            return ConvBackend::Slow2d;
          }
        }
      } else if (input.ndimension() == 5 && (input.is_cuda() || params.is_dilated())) {
        // 维度为五且使用 CUDA 或扩展卷积，返回适合的慢速扩展三维卷积后端
        return ConvBackend::SlowDilated3d;
      } else if (input.ndimension() == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU 实现在非扩展情况下具有专门的矩阵乘积内核 */
        // 返回适合的慢速三维卷积后端
        return ConvBackend::Slow3d;
      } else {
        // 不支持的情况
        // unsupported
      }
    }
  } else if (params.use_mps(input, weight)) {
    // 检查是否使用了 MPS
    if (params.transposed) {
      // 返回适合的 MPS 转置卷积后端
      return ConvBackend::MpsTranspose;
    } else {
      // 返回适合的 MPS 卷积后端
      return ConvBackend::Mps;
    }
  } else {
    // 当输入的后端具有非源码实现时才会到达这里
    // 返回可以覆盖的卷积后端
    return ConvBackend::Overrideable;
  }

  // 如果没有找到适合的后端，报错
  // Error out if no suitable backend was found.
  AT_ERROR("unsupported ConvNd parameters");
// 选择基于输入和参数的卷积后端。
ConvBackend select_conv_backend(
    // 输入张量
    const Tensor& input_r,
    // 权重张量
    const Tensor& weight_r,
    // 可选的偏置张量
    const std::optional<Tensor>& bias_opt,
    // 步长
    SymIntArrayRef stride_,
    // 填充
    SymIntArrayRef padding_,
    // 空洞
    SymIntArrayRef dilation_,
    // 是否转置
    bool transposed_,
    // 输出填充
    SymIntArrayRef output_padding_,
    // 分组
    c10::SymInt groups_,
    // 可选的偏置大小
    const at::OptionalSymIntArrayRef bias_sizes_opt) {
  // 从可选的偏置中借用张量
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 引用偏置张量
  const Tensor& bias = *bias_maybe_owned;

  // 获取全局上下文
  auto& ctx = at::globalContext();
  // 获取权重张量的维度数
  auto k = weight_r.ndimension();
  // 计算空间维度
  int64_t dim = k - 2;
  // 创建卷积参数对象
  ConvParams<c10::SymInt> params;
  // 扩展步长参数（如果需要）
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  // 扩展填充参数（如果需要）
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  // 扩展空洞参数（如果需要）
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  // 设置是否转置
  params.transposed = transposed_;
  // 扩展输出填充参数（如果需要）
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  // 设置分组数
  params.groups = groups_;
  // 设置是否使用 CuDNN 的基准模式
  params.benchmark = ctx.benchmarkCuDNN();
  // 设置是否确定性计算（CuDNN）
  params.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  // 设置是否启用 CuDNN
  params.cudnn_enabled = ctx.userEnabledCuDNN();
  // 设置是否允许 TF32（CuDNN）
  params.allow_tf32 = ctx.allowTF32CuDNN();

  // 复制输入和权重张量
  auto input = input_r;
  auto weight = weight_r;
  // 检查输入形状是否合法
  check_shape_forward(input, weight.sym_sizes(), bias, params);

  // 将1维扩展为2维
  // 仅适用于不原生支持1维空间输入的后端
  if (k == 3 && !input.is_mkldnn() && !input.is_xpu()) {
    // 避免意外通过 NHWC 处理置换的3维输入
    input = input.contiguous();
    // 将1维视为2维
    params.view1d_as_2d();
    // 将输入视为4维
    input = view4d(input);
    // 将权重视为4维
    weight = view4d(weight);
  }

  // 如果定义了偏置，则获取其大小作为可选参数
  auto bias_sizes = bias.defined() ? std::optional<SymIntArrayRef>(bias.sym_sizes()) : bias_sizes_opt;
  // 判断是否需要反向传播
  bool need_backward = GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
  // 选择卷积后端
  return _select_conv_backend(input, weight, bias, bias_sizes, need_backward, params);
}

// 由于向后兼容性原因，提供一个不需要 bias_opt 参数的副本
static ConvBackend select_conv_backend(
    // 输入张量
    const Tensor& input,
    // 权重张量
    const Tensor& weight,
    // 可选的偏置大小
    const at::OptionalIntArrayRef bias_sizes_opt,
    // 是否需要反向传播
    const bool need_backward,
    // 卷积参数
    const ConvParams<int64_t>& params) {
  // 调用选择卷积后端函数
  return _select_conv_backend(input, weight, {}, bias_sizes_opt, need_backward, params);
}

// 非分组的卷积后端实现
static at::Tensor _convolution_nogroup_backend(
    // 输入张量
    const Tensor& input,
    // 权重张量
    const Tensor& weight,
    // 偏置张量
    const Tensor& bias,
    // 卷积后端
    const ConvBackend backend,
    // 卷积参数
    const ConvParams<int64_t>& params) {
  // 获取卷积核大小
  auto kernel_size = weight.sizes().slice(2);
  // 根据后端类型执行不同的操作
  switch(backend) {
    // 使用 NNPACK 空间卷积
    case ConvBackend::NnpackSpatial:
      // 如果支持 NNPACK，调用 NNPACK 空间卷积函数
#if AT_NNPACK_ENABLED()
      return at::_nnpack_spatial_convolution(input, weight, bias, params.padding, params.stride);
#else
      // 如果 PyTorch 编译时没有支持 NNPACK，抛出错误
      TORCH_INTERNAL_ASSERT(false, "NnpackSpatial backend was selected in PyTorch compiled without nnpack support");
#endif
    # 对于慢速 2D 卷积后端，调用对应的 Torch 函数 at::thnn_conv2d
    case ConvBackend::Slow2d:
      return at::thnn_conv2d(input, weight, kernel_size, bias, params.stride, params.padding);
    # 对于慢速带扩张 2D 卷积后端，调用对应的 Torch 函数 at::slow_conv_dilated2d
    case ConvBackend::SlowDilated2d:
      return at::slow_conv_dilated2d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
    # 对于慢速带扩张 3D 卷积后端，调用对应的 Torch 函数 at::slow_conv_dilated3d
    case ConvBackend::SlowDilated3d:
      return at::slow_conv_dilated3d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
    # 对于慢速转置 2D 卷积后端，调用对应的 Torch 函数 at::slow_conv_transpose2d
    case ConvBackend::SlowTranspose2d:
      return at::slow_conv_transpose2d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
    # 对于慢速转置 3D 卷积后端，调用对应的 Torch 函数 at::slow_conv_transpose3d
    case ConvBackend::SlowTranspose3d:
      return at::slow_conv_transpose3d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
    # 默认情况下，若遇到不支持的卷积后端，抛出错误信息
    default:
      TORCH_CHECK(false, "Unsupported conv nogroup backend encountered");
}

// 结束一个静态内联函数的定义

static inline std::vector<int64_t> calc_output_size(
    const Tensor& input,
    const Tensor& weight,
    const ConvParams<int64_t>& params) {
  // 计算输出尺寸，根据是否转置选择不同的计算方式
  std::vector<int64_t> output_size = params.transposed ?
    conv_input_size(input.sizes(), weight.sizes(), params.padding, params.output_padding,
        params.stride, params.dilation, params.groups) :
    conv_output_size(input.sizes(), weight.sizes(), params.padding, params.stride, params.dilation);

  // 处理输入通道数为零的情况
  if (input.size(input_channels_dim) == 0) {
    // 将输出通道数置为零
    output_size[output_channels_dim] = 0;
  }
  return output_size;
}

static inline at::MemoryFormat determine_backend_memory_format(
    const Tensor& input,
    const Tensor& weight,
    const ConvBackend backend) {
  // 初始化后端内存格式为连续格式
  at::MemoryFormat backend_memory_format = at::MemoryFormat::Contiguous;
  auto k = weight.ndimension();
#if !defined(C10_MOBILE)
  // 检查是否编译了 CuDNN 并根据后端选择推荐的内存格式
  switch(backend) {
    case ConvBackend::Cudnn:
    case ConvBackend::CudnnTranspose:
      if (detail::getCUDAHooks().compiledWithCuDNN()) {
        backend_memory_format = cudnn_conv_suggest_memory_format(input, weight);
      }
      break;
    // 对于 MIOpen 和其变体，检查是否编译了 MIOpen 并根据输入和权重使用通道最后的内存格式
    case ConvBackend::Miopen:
    case ConvBackend::MiopenDepthwise:
    case ConvBackend::MiopenTranspose:
      if (detail::getCUDAHooks().compiledWithMIOpen() && miopen_conv_use_channels_last(input, weight)) {
        TORCH_INTERNAL_ASSERT((k == 4 || k == 5),
            "Expected 4D or 5D input for miopen memory format selection in determine_backend_memory_format()");
        backend_memory_format = (k == 5) ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
      }
      break;
    // 对于 MKLDNN 和其转置操作，检查是否使用通道最后的内存格式
    case ConvBackend::Mkldnn:
    case ConvBackend::MkldnnTranspose:
      if (mkldnn_conv_use_channels_last(input, weight)) {
        backend_memory_format = (k == 5) ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
      }
      break;
    // 对于慢速实现的 2D 卷积，检查是否使用通道最后的内存格式
    case ConvBackend::Slow2d:
    case ConvBackend::SlowDilated2d:
    case ConvBackend::SlowTranspose2d:
      if (thnn_conv_use_channels_last(input, weight)) {
        backend_memory_format = at::MemoryFormat::ChannelsLast;
      }
      break;
    // 对于可以覆盖的后端，检查是否使用通道最后的内存格式
    case ConvBackend::Overrideable:
      if (xpu_conv_use_channels_last(input, weight)) {
        backend_memory_format = (k == 5) ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
      }
      break;
    // 默认情况下使用连续的内存格式
    default:
      backend_memory_format = at::MemoryFormat::Contiguous;
  }
#endif
  // 返回推断出的后端内存格式
  return backend_memory_format;
}

// 包装函数，用于决定后端的内存格式
at::MemoryFormat _determine_backend_memory_format(
    const Tensor& input,
    const Tensor& weight,
    const ConvBackend backend)  {
  return determine_backend_memory_format(input, weight, backend);
}

// 执行卷积操作的函数，根据输入参数选择合适的后端内存格式
at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const std::optional<Tensor>& bias_r_opt,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
  // 使用 borrow_from_optional_tensor 函数从可能为空的 bias_r_opt 变量中获取有效的 Tensor 引用
  c10::MaybeOwned<Tensor> bias_r_maybe_owned = at::borrow_from_optional_tensor(bias_r_opt);
  // 通过解引用获取真正的 bias_r 引用
  const Tensor& bias_r = *bias_r_maybe_owned;

  // 复制 input_r 到 input 变量
  auto input = input_r;
  // 复制 weight_r 到 weight 变量
  auto weight = weight_r;
  // 复制 bias_r 到 bias 变量
  auto bias = bias_r;
  // 获取 weight 的维度数量
  auto k = weight.ndimension();
  // 获取 weight 的尺寸
  c10::IntArrayRef weight_sizes = weight.sizes();
  // 计算维度，减去两个
  int64_t dim = k - 2;

  // 检查 weight 的维度数是否大于 0
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
  // 检查 groups_ 是否大于 0
  TORCH_CHECK(groups_ > 0, "non-positive groups is not supported");

  // 创建 ConvParams<int64_t> 对象 params 并设置其属性
  ConvParams<int64_t> params;
  // 如果需要，根据维度扩展 stride_
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  // 如果需要，根据维度扩展 padding_
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  // 如果需要，根据维度扩展 dilation_
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  // 设置是否转置的标志位
  params.transposed = transposed_;
  // 如果需要，根据维度扩展 output_padding_
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  // 设置 groups_ 属性
  params.groups = groups_;
  // 设置是否使用 benchmark 模式的标志位
  params.benchmark = benchmark;
  // 设置是否使用 deterministic 模式的标志位
  params.deterministic = deterministic;
  // 设置是否启用 cuDNN 的标志位
  params.cudnn_enabled = cudnn_enabled;
  // 设置是否允许使用 TF32 的标志位
  params.allow_tf32 = allow_tf32;

  // 检查输入数据的形状是否符合预期，并抛出错误信息如果不符合
  check_shape_forward(input, weight_sizes, bias, params);

  // 如果 weight 的维度为 3，且 input 不是 mkldnn 或 xpu，将其视为 1 维转换成 2 维
  if (k == 3 && !input.is_mkldnn() && !input.is_xpu()) {
    // 确保 input 是连续的
    input = input.contiguous();
    // 将输入视为 1 维并转换为 2 维
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  // 根据需要选择合适的后端实现
  auto bias_sizes_opt = bias.defined() ? std::optional<IntArrayRef>(bias.sizes()) : c10::nullopt;
  // 确定是否需要计算梯度
  bool need_backward = GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
  // 选择卷积的后端实现方式
  ConvBackend backend = _select_conv_backend(input, weight, bias, c10::OptionalIntArrayRef(bias_sizes_opt), need_backward, params);
  // 确定后端实现的内存格式
  at::MemoryFormat backend_memory_format = determine_backend_memory_format(input, weight, backend);

  // 调用相应的后端实现
  Tensor output;
  // 获取 kernel_size
  auto kernel_size = weight.sizes().slice(2);
  switch (backend) {
    // 对不同的后端实现进行分支处理
    case ConvBackend::CudaDepthwise2d:
      // 调用 CUDA 深度可分离卷积的实现函数
      output = at::_conv_depthwise2d(input.contiguous(), weight, kernel_size, bias,
          params.stride, params.padding, params.dilation);
      break;
    case ConvBackend::CudaDepthwise3d:
      // 调用 CUDA 3D 深度可分离卷积的实现函数
      output = at::conv_depthwise3d(input.contiguous(), weight, kernel_size, bias,
          params.stride, params.padding, params.dilation);
      break;
    case ConvBackend::Cudnn:
      // 检查输入与参数的数据类型是否一致
      check_input_same_type_as_parameters(input, weight, bias);
      // 调用 CuDNN 提供的卷积操作，计算输出
      output = at::cudnn_convolution(
          input.contiguous(backend_memory_format), weight, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
      // 如果定义了偏置，则加上偏置
      if (bias.defined()) {
        output.add_(reshape_bias(input.dim(), bias));
      }
      break;
    case ConvBackend::CudnnTranspose:
      // 检查输入与参数的数据类型是否一致
      check_input_same_type_as_parameters(input, weight, bias);
      // 调用 CuDNN 提供的转置卷积操作，计算输出
      output = at::cudnn_convolution_transpose(
          input.contiguous(backend_memory_format), weight, params.padding, params.output_padding,
          params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
      // 如果定义了偏置，则加上偏置
      if (bias.defined()) {
        output.add_(reshape_bias(input.dim(), bias));
      }
      break;
    case ConvBackend::Empty:
    {
      Tensor weight_view;
      // 使用 permute 和 clone 来避免在非连续情况下 at::_unsafe_view(weight, -1) 失败的问题，
      // 当视图大小与输入张量的大小和步幅不兼容时。
      if(weight.is_contiguous()) {
        weight_view = at::_unsafe_view(weight, -1);
      } else if (weight.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        weight_view = at::_unsafe_view(at::permute(weight, {0, 2, 3, 1}), -1);
      } else if (weight.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
        weight_view = at::_unsafe_view(at::permute(weight, {0, 2, 3, 4, 1}), -1);
      } else {
        weight_view = at::_unsafe_view(weight.clone(at::MemoryFormat::Contiguous), -1);
      }

      // 根据输入大小选择合适的权重视图进行计算输出
      output = (input.size(1) == 0) ? (input.view(-1) * weight_view) : (input * weight_view[0]);
      // 如果定义了偏置，则加上偏置
      if (bias.defined()) {
        output.add_(bias[0]);
      }
      // 重新调整输出的大小，以匹配计算的输出尺寸
      output = output.view(calc_output_size(input, weight, params));
      break;
    }
    case ConvBackend::Miopen:
      // 检查输入与参数的数据类型是否一致
      check_input_same_type_as_parameters(input, weight, bias);
      // 调用 MIOpen 提供的卷积操作，计算输出
      output = at::miopen_convolution(
          input.contiguous(backend_memory_format), weight, bias, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic);
      break;
    case ConvBackend::MiopenDepthwise:
      // 调用 MIOpen 提供的深度可分离卷积操作，计算输出
      output = at::miopen_depthwise_convolution(
          input.contiguous(backend_memory_format), weight, bias, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic);
      break;
    case ConvBackend::MiopenTranspose:
      // 检查输入与参数的数据类型是否一致
      check_input_same_type_as_parameters(input, weight, bias);
      // 调用 MIOpen 提供的转置卷积操作，计算输出
      output = at::miopen_convolution_transpose(
          input.contiguous(backend_memory_format), weight, bias, params.padding, params.output_padding,
          params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
      break;
    case ConvBackend::Mkldnn:
#if AT_MKLDNN_ENABLED()
      # 检查输入张量与参数张量的数据类型是否相同，使用的后端是什么
      check_input_same_type_as_parameters(input, weight, bias, backend);
      # 如果输入张量不是 mkldnn 张量，则需要保证其连续性，使用指定的后端内存格式
      if (!input.is_mkldnn()) {
        input = input.contiguous(backend_memory_format);
        weight = weight.contiguous(backend_memory_format);
        # 如果有定义偏置，则也需要保证其连续性
        bias = bias.defined() ? bias.contiguous() : bias;
      }
      # 调用 mkldnn_convolution 函数进行 MKLDNN 卷积操作
      output = at::mkldnn_convolution(
          input, weight, bias, params.padding, params.stride, params.dilation, params.groups);
#else
      # 如果不支持 MKLDNN，应该永远不会执行到这里，否则抛出异常
      TORCH_INTERNAL_ASSERT(false, "Mkldnn backend was selected in PyTorch compiled without mkldnn support");
#endif
      break;
    case ConvBackend::MkldnnTranspose:
#if AT_MKLDNN_ENABLED()
      # 检查输入张量与参数张量的数据类型是否相同，使用的后端是什么
      check_input_same_type_as_parameters(input, weight, bias, backend);
      # 如果输入张量不是 mkldnn 张量，则需要保证其连续性，使用指定的后端内存格式
      if (!input.is_mkldnn()) {
        input = input.contiguous(backend_memory_format);
        weight = weight.contiguous(backend_memory_format);
        # 如果有定义偏置，则也需要保证其连续性
        bias = bias.defined() ? bias.contiguous() : bias;
      }
      # 调用 mkldnn_convolution_transpose_stub 函数进行 MKLDNN 转置卷积操作
      output = mkldnn_convolution_transpose_stub(input.device().type(),
          input, weight, bias, params.padding, params.output_padding, params.stride, params.dilation, params.groups);
#else
      # 如果不支持 MKLDNN，应该永远不会执行到这里，否则抛出异常
      TORCH_INTERNAL_ASSERT(false, "Mkldnn backend was selected in PyTorch compiled without mkldnn support");
#endif
      break;
    case ConvBackend::MkldnnEmpty:
#if AT_MKLDNN_ENABLED()
      # 调用 empty_mkldnn 函数创建一个 MKLDNN 空张量
      output = empty_mkldnn(
          calc_output_size(input, weight, params), optTypeMetaToScalarType(input.options().dtype_opt()),
          input.options().layout_opt(), input.options().device_opt(), input.options().pinned_memory_opt());
#else
      # 如果不支持 MKLDNN，应该永远不会执行到这里，否则抛出异常
      TORCH_INTERNAL_ASSERT(false, "Mkldnn backend was selected in PyTorch compiled without mkldnn support");
#endif
      break;
    case ConvBackend::Overrideable:
      # 调用 convoution_overrideable 函数进行可覆盖的卷积操作
      output = at::convolution_overrideable(
          input, weight, bias, params.stride, params.padding, params.dilation, params.transposed,
          params.output_padding, params.groups);
      break;
    case ConvBackend::Slow3d:
      # 调用 slow_conv3d 函数进行 3D 慢速卷积操作
      output = at::slow_conv3d(input, weight, kernel_size, bias, params.stride, params.padding);
      break;
    case ConvBackend::Winograd3x3Depthwise:
      # 调用 convolution_depthwise3x3_winograd_stub 函数进行 Winograd 算法的深度卷积操作
      output = convolution_depthwise3x3_winograd_stub(
          input.device().type(), input, weight, bias, params.stride, params.padding, params.groups);
      break;
    case ConvBackend::Xnnpack2d:
      # 调用 xnnpack::convolution2d 函数进行 XNNPACK 2D 卷积操作
      output = xnnpack::convolution2d(
          input, weight, bias, params.padding, params.stride, params.dilation, params.groups);
      break;
    // 处理不原生支持 groups > 1 的后端
    case ConvBackend::NnpackSpatial:
    case ConvBackend::Slow2d:
    case ConvBackend::SlowDilated2d:
    case ConvBackend::SlowDilated3d:
    case ConvBackend::SlowTranspose2d:
    // 当选择 ConvBackend::SlowTranspose3d 时执行以下代码块
    case ConvBackend::SlowTranspose3d:
      // 确保输入张量按照指定的后端内存格式是连续的
      input = input.contiguous(backend_memory_format);
      // 确保权重张量按照指定的后端内存格式是连续的
      weight = weight.contiguous(backend_memory_format);
      // 如果卷积操作中没有分组（groups == 1）
      if (params.groups == 1) {
        // 执行无分组卷积操作，并将结果赋给输出张量
        output = _convolution_nogroup_backend(input, weight, bias, backend, params);
      } else {
        // 如果卷积操作中存在分组
        std::vector<Tensor> outputs(params.groups);
        // 对每个分组进行循环
        for (const auto g : c10::irange(params.groups)) {
          // 从输入张量中提取当前分组的子张量
          auto input_g = subtensor(input, 1, params.groups, g);
          // 从权重张量中提取当前分组的子张量
          auto weight_g = subtensor(weight, 0, params.groups, g);
          // 从偏置张量中提取当前分组的子张量
          auto bias_g = subtensor(bias, 0, params.groups, g);
          // 执行无分组卷积操作，并将结果存储在输出向量的当前分组位置
          outputs[g] = _convolution_nogroup_backend(input_g, weight_g, bias_g, backend, params);
        }
        // 将所有分组的结果张量按照维度1进行拼接，得到最终的输出张量
        output = at::cat(outputs, 1);
      }
      // 结束当前 case 分支的执行
      break;
    // 当选择 ConvBackend::Mps 时继续执行下一个 case 分支
    case ConvBackend::Mps:
#ifdef USE_MPS
      // 如果使用 MPS 后端，则进行以下检查和操作
      TORCH_CHECK(input.options().type_equal(weight.options()),
               "Input type (", input.toString(), ") and weight type (", weight.toString(),
               ") should be the same");
      TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
               "Input type (", input.toString(), ") and bias type (", bias.toString(),
               ") should be the same");

      // 使用 MPS 后端进行卷积操作
      output = at::_mps_convolution(input.contiguous(), weight, bias.defined() ? bias.contiguous() : bias,
                                     params.padding, params.stride, params.dilation,
                                     params.groups);
#else
      // 如果未使用 MPS 后端，报告错误，因为不支持 MPS 后端
      TORCH_INTERNAL_ASSERT(false, "MPS backend was selected in PyTorch without support");
#endif
      break;
    case ConvBackend::MpsTranspose:
#ifdef USE_MPS
      // 如果使用 MPS 后端，则进行以下检查和操作
      TORCH_CHECK(input.options().type_equal(weight.options()),
               "Input type (", input.toString(), ") and weight type (", weight.toString(),
               ") should be the same");
      TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
               "Input type (", input.toString(), ") and bias type (", bias.toString(),
               ") should be the same");

      // 使用 MPS 后端进行转置卷积操作
      output = at::_mps_convolution_transpose(
          input.contiguous(backend_memory_format), weight,
          params.padding, params.output_padding,
          params.stride, params.dilation, params.groups);
      
      // 如果定义了偏置，则将偏置重新整形后添加到输出中
      if (bias.defined()) {
        output.add_(reshape_bias(input.dim(), bias));
      }
#else
      // 如果未使用 MPS 后端，报告错误，因为不支持 MPS 后端
      TORCH_INTERNAL_ASSERT(false, "MPS backend was selected in PyTorch without support");
#endif
      break;
  }

  // 如果卷积核大小为 3，且输入张量不是 MKLDNN 或 XPU 类型，则对输出进行 3D 视图调整
  if (k == 3 && !input.is_mkldnn() && !input.is_xpu()) {
    output = view3d(output);
  }

  // 返回经过卷积操作后的输出张量
  return output;
}

// _convolution 函数的封装，处理输入参数并调用具体的卷积操作函数
at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const std::optional<Tensor>& bias_r_opt,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled)
{
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的偏置张量中获取有效的引用
  c10::MaybeOwned<Tensor> bias_r_maybe_owned = at::borrow_from_optional_tensor(bias_r_opt);
  const Tensor& bias_r = *bias_r_maybe_owned;

  // 调用具体的卷积函数，并返回计算结果
  return at::_convolution(input_r, weight_r, bias_r, stride_, padding_, dilation_, transposed_, output_padding_, groups_, benchmark, deterministic, cudnn_enabled, at::globalContext().allowTF32CuDNN());
}
// 定义一个函数，用于处理反向卷积的计算，返回三个张量：输入梯度，权重梯度，偏置梯度
std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
        // 输入参数：梯度输出，输入张量，权重张量，步幅，填充，扩展，是否转置，输出填充，分组数，输出掩码
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool, 3> output_mask) {
   // 检查函数未实现，抛出错误信息
   TORCH_CHECK_NOT_IMPLEMENTED(false, "convolution_backward_overrideable: You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL to override this function ");
   // 返回空张量元组作为占位符
  return std::tuple<Tensor, Tensor, Tensor>(
          at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT),
          at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT),
          at::empty({}));
}

// 定义一个静态函数，用于从变量中截取子变量
static Tensor subvariable(const Tensor& var, int dim, int groups, int g) {
  // 计算每组的大小
  int64_t n = var.sizes()[dim] / groups;
  // 根据给定的维度和组索引 g，从 var 中截取子变量
  auto result = var.narrow(dim, n * g, n);
  // 返回截取的结果
  return result;
}

// 定义一个函数，用于处理双向卷积的反向传播
std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward( const std::optional<Tensor>& ggI_opt, const std::optional<Tensor>& ggW_r_opt, const std::optional<Tensor>& ggb_opt,
    const Tensor& gO_r, const Tensor& weight_r, const Tensor& input,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    std::array<bool, 3> output_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选张量 ggI_opt 中借用或者创建 ggI
  c10::MaybeOwned<Tensor> ggI_maybe_owned = at::borrow_from_optional_tensor(ggI_opt);
  const Tensor& ggI = *ggI_maybe_owned;
  // 如果 ggW_r_opt 存在则取其值，否则创建空张量
  const Tensor& ggW_r = c10::value_or_else(ggW_r_opt, [] {return Tensor();});
  // 如果 ggb_opt 存在则取其值，否则创建空张量
  const Tensor& ggb = c10::value_or_else(ggb_opt, [] {return Tensor();});

  // 复制传入的张量，以便在函数中修改它们
  auto ggW = ggW_r;
  auto gO = gO_r;
  auto weight = weight_r;

  // 计算权重的维度
  int64_t dim = weight.ndimension() - 2;
  // 构建卷积参数结构体
  ConvParams<int64_t> params;
  // 根据需要扩展步幅、填充、扩展参数
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  
  // TODO: hacky way of inferring the groups number for grouped Conv3D
  // 如果不是转置卷积且输入维度大于4，根据权重和输入通道数自动推断分组数
  // 避免当通道数为0时出现未定义行为；对于这种情况，参数不被使用。
  if (!params.transposed && input.dim() > 4) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    // 如果权重大小大于0，则计算输入通道数除以权重通道数
    params.groups = (weight.size(1) > 0) ? input.size(1) / weight.size(1) : -1;
  } else {
    // 否则使用传入的分组数
    params.groups = groups_;
  }

  // 计算 ggO = conv(ggI, w) + conv(i, ggW) + ggb
  Tensor ggO;
  if (input.numel() != 0) {
    if (ggI.defined()) {
      // 检查 ggI 是否已定义，即是否存在梯度输入
      if (weight.is_cuda()) {
        // 如果权重在 CUDA 上，则需要进行内存连续性操作
        weight = weight.contiguous();
      }
      // 执行卷积操作，计算 ggO，使用 ggI 和权重进行卷积
      ggO = at::convolution(ggI, weight, Tensor(), params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups);
    }

    if (ggW.defined()) {
      // 检查 ggW 是否已定义，即是否存在梯度权重
      if (ggW.is_cuda()) {
        // 如果梯度权重在 CUDA 上，则需要进行内存连续性操作
        ggW = ggW.contiguous();
      }
      // 执行卷积操作，计算 ggW_term，使用输入 input 和 ggW 进行卷积
      auto ggW_term = at::convolution(input, ggW, Tensor(), params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups);
      if (ggO.defined()) {
        // 如果 ggO 已定义，则将 ggW_term 加到 ggO 上
        ggO = ggO + ggW_term;
      } else {
        // 如果 ggO 未定义，则将 ggW_term 赋给 ggO
        ggO = ggW_term;
      }
    }
  }

  if (ggb.defined()) {
    // 检查 ggb 是否已定义，即是否存在梯度偏置
    // 将 ggb 视为 (1, ggb.size(0), 1, 1...) 的形状

    // 扩展 ggb
    std::vector<int64_t> new_size(gO.ndimension(), 1);
    new_size[1] = ggb.sizes()[0];
    auto ggb_contiguous = ggb.contiguous();
    auto ggb_view = ggb_contiguous.view(new_size);

    // 扩展 ggb_view 到 gO 的形状
    auto ggb_expanded = ggb_view.expand(gO.sizes());

    if (ggO.defined()) {
      // 如果 ggO 已定义，则将 ggb_expanded 加到 ggO 上
      ggO = ggO + ggb_expanded;
    } else {
      // 如果 ggO 未定义，则将 ggb_expanded 赋给 ggO
      ggO = ggb_expanded;
    }
  }

  // 计算梯度权重 gW = conv(ggI, gO)
  Tensor gW;
  if (ggI.defined()) {
    // 检查 ggI 是否已定义，即是否存在梯度输入

    // 使用正确的填充修改参数
    ConvParams<int64_t> gw_conv_params(params);

    // 将组数设置为 1，因为组是单独处理的
    auto groups = gw_conv_params.groups;
    gw_conv_params.groups = 1;
    // 交换 dilation 和 stride
    std::swap(gw_conv_params.dilation, gw_conv_params.stride);

    // 转置 gO 和 ggI 以便在批处理上累积
    auto gOt = gO.transpose(0, 1);
    auto ggIt = ggI.transpose(0, 1);

    Tensor gWt;
    // 计算卷积
    // 检查输入张量是否非空
    if (input.numel() != 0) {
      // 如果 groups 等于 1
      if (groups == 1) {

        // 如果 gOt 张量在 GPU 上，需要确保其连续性
        if (gOt.is_cuda()) {
          gOt = gOt.contiguous();
        }
        // 计算卷积操作
        // 如果是转置卷积，设置卷积参数的转置为 false，计算 gWt
        if (params.transposed) {
          gw_conv_params.transposed = false;
          gWt = at::convolution(gOt, ggIt, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups);
        } else {
          // 否则直接计算 gWt
          gWt = at::convolution(ggIt, gOt, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups);
        }
      } else {
        // 如果 groups 大于 1，创建一个 gWt 的张量列表
        std::vector<Tensor> gWt_list(groups);
        // 对每个分组进行处理
        for (const auto g : c10::irange(groups)) {
          // 对 ggIt 和 gOt 进行分组子变量的处理
          auto ggIt_g = subvariable(ggIt, 0, groups, g);
          auto gOt_g = subvariable(gOt, 0, groups, g);
          // 如果 gOt_g 在 GPU 上，需要确保其连续性
          if (gOt_g.is_cuda()) {
            gOt_g = gOt_g.contiguous();
          }

          // 计算卷积操作
          // 如果是转置卷积，设置卷积参数的转置为 false，计算 gWt_list[g]
          if (params.transposed) {
            gw_conv_params.transposed = false;
            gWt_list[g] = at::convolution(gOt_g, ggIt_g, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups);
          } else {
            // 否则直接计算 gWt_list[g]
            gWt_list[g] = at::convolution(ggIt_g, gOt_g, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups);
          }
        }

        // 将 gWt_list 中的张量沿着第一个维度拼接，得到 gWt
        gWt = at::cat(gWt_list, 1);
      }

      // 将 gWt 转置，使其与输入和输出通道数匹配
      gW = gWt.transpose(0, 1);

      // 缩窄 gW，仅保留相关部分
      // 之所以采用这种方式而不是缩窄输入本身，是因为 ConvForward 内核不支持不对称填充。
      auto gW_size = gW.sizes();
      auto w_size = weight.sizes();
      for (const auto i : c10::irange(2, gW_size.size())) {
        if (gW_size[i] > w_size[i]) {
            gW = gW.narrow(i, 0, w_size[i]);
            gW_size = gW.sizes();
        }
      }
    }
  }

  // 计算 gI = convT(gO, ggW) 如果非转置
  //         gI = conv(gO, ggW)  如果转置
  Tensor gI;
  // 再次检查输入张量是否非空
  if (input.numel() != 0) {
    if (ggW.defined()) {
      // 定义卷积参数对象，使用与输入参数相关的值
      ConvParams<int64_t> gi_conv_params(params);
      // 如果需要转置卷积，则设置参数中的转置标志为真
      gi_conv_params.transposed = !params.transposed;

      if (params.transposed) {
        // 如果是转置卷积且输入张量在 CUDA 上，则要求其连续性
        if (gO.is_cuda()) {
          gO = gO.contiguous();
        }
        // 执行转置卷积操作，计算梯度输入张量 gI
        gI = at::convolution(gO, ggW, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups);

        // 缩小 gI 以仅保留相关部分
        // 采用这种方法是因为不支持负的 output_padding
        // TODO: 弄清楚是否可以缩小 gO 并节省一些计算，而不是缩小已计算的 gI
        auto gI_size = gI.sizes();
        auto i_size = input.sizes();
        for (const auto i : c10::irange(2, gI_size.size())) {
          if (gI_size[i] > i_size[i]) {
            gI = gI.narrow(i, 0, i_size[i]);
            gI_size = gI.sizes();
          }
        }
      } else {
        // 计算输出填充
        // TODO: 弄清楚为什么需要计算这个...
        auto kernel_size = weight.sizes().slice(2);
        auto input_shape = input.sizes().slice(2);
        auto grad_output_shape = gO.sizes().slice(2);

        for (const auto i : c10::irange(kernel_size.size())) {
          // 检查是否整个输入已被使用
          auto expected_input_shape = (kernel_size[i] - 1) * gi_conv_params.dilation[i]
            - 2 * gi_conv_params.padding[i]
            + (gi_conv_params.stride[i] * (grad_output_shape[i] - 1) + 1);
          if (expected_input_shape != input_shape[i]) {
            // 根据预期的输入形状调整输出填充
            gi_conv_params.output_padding[i] = input_shape[i] - expected_input_shape;
          }
        }

        // 如果输入张量在 CUDA 上，则要求其连续性
        if (gO.is_cuda()) {
          gO = gO.contiguous();
        }

        // 执行卷积操作，计算梯度输入张量 gI
        gI = at::convolution(gO, ggW, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups);
      }
    }
  }

  // 返回三元组，包含计算得到的梯度输出张量 ggO，梯度输入张量 gI，以及梯度权重 ggW
  return std::tuple<Tensor,Tensor,Tensor>{ggO, gI, gW};
}

// 后向传播函数，处理卷积操作的梯度计算，根据输出掩码设置计算输入、权重和偏置的梯度。
// 支持1D、2D或3D空间卷积，当前要求输入张量至少包含一个批次维度。
//
// 参数：
//   grad_output: 形状为(N, C_out, L_out)、(N, C_out, H_out, W_out)或(N, C_out, D_out, H_out, W_out)的张量，代表梯度输出
//   input: 形状为(N, C_in, L_in)、(N, C_in, H_in, W_in)或(N, C_in, D_in, H_in, W_in)的张量，代表输入
//   weight: 形状为(C_out, C_in // groups, *kernel_size)的张量，*kernel_size的维度必须与输入空间维度匹配
//   output_mask: 一个包含3个布尔值的数组，控制输出的梯度计算
//   backend: 表示卷积后向计算的后端类型
//   params: ConvParams<int64_t>类型的对象，包含卷积操作的参数如步长、填充等
static std::tuple<at::Tensor, at::Tensor, at::Tensor> _convolution_backward_nogroup_backend(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const std::array<bool, 3> output_mask,
    const ConvBackend backend,
    const ConvParams<int64_t>& params) {
  // 获取卷积核大小
  auto kernel_size = weight.sizes().slice(2);
  
  // 根据后端类型选择相应的卷积后向计算函数
  switch(backend) {
    case ConvBackend::Slow2d:
      return at::_slow_conv2d_backward(
        grad_output, input, weight, kernel_size, params.stride, params.padding, output_mask);
    // 注意：nnpack后向计算不支持步进卷积，应使用慢速实现
    case ConvBackend::NnpackSpatial:
    case ConvBackend::SlowDilated2d:
      return slow_conv_dilated2d_backward_stub(
        input.device().type(),
        grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation, output_mask);
    case ConvBackend::SlowDilated3d:
      return slow_conv_dilated3d_backward_stub(
        input.device().type(),
        grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation, output_mask);
    case ConvBackend::SlowTranspose2d:
      return slow_conv_transpose2d_backward_stub(
        input.device().type(), grad_output, input, weight, kernel_size, params.stride, params.padding,
        params.output_padding, params.dilation, output_mask);
    case ConvBackend::SlowTranspose3d:
      return slow_conv_transpose3d_backward_stub(
        input.device().type(), grad_output, input, weight, kernel_size, params.stride, params.padding,
        params.output_padding, params.dilation, output_mask);
    default:
      // 若遇到不支持的卷积后端类型，抛出错误
      TORCH_CHECK(false, "Unsupported conv nogroup backend encountered");
  }
}
// dilation: 单个值或与输入空间维度数量匹配的数组
// transposed: 布尔值，指示卷积是否为转置操作
// output_padding: 单个值或维度与输入空间维度数量匹配；仅在 transposed 为 true 时支持
// groups: 分组卷积的组数
// output_mask: 三维布尔数组，指定输入、权重、偏置的梯度计算顺序
std::tuple<Tensor, Tensor, Tensor> convolution_backward(
    const Tensor& grad_output_, const Tensor& input_, const Tensor& weight_,
    const at::OptionalIntArrayRef bias_sizes_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  auto grad_output = grad_output_;
  auto input = input_;
  auto weight = weight_;

  auto k = weight.ndimension(); // 获取权重张量的维度数
  int64_t dim = k - 2; // 计算空间维度数，通常是 k 减去 2

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions"); // 检查权重张量至少有三个维度

  auto& ctx = at::globalContext(); // 获取全局上下文
  ConvParams<int64_t> params; // 创建卷积参数对象
  params.stride = expand_param_if_needed(stride, "stride", dim); // 根据需要扩展步幅参数到与空间维度匹配
  params.padding = expand_param_if_needed(padding, "padding", dim); // 根据需要扩展填充参数到与空间维度匹配
  params.dilation = expand_param_if_needed(dilation, "dilation", dim); // 根据需要扩展膨胀参数到与空间维度匹配
  params.transposed = transposed; // 设置卷积是否为转置操作
  params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim); // 根据需要扩展输出填充参数到与空间维度匹配
  params.groups = groups; // 设置卷积分组数
  params.benchmark = ctx.benchmarkCuDNN(); // 设置是否使用 CuDNN 的基准模式
  params.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms(); // 设置是否使用确定性 CuDNN 算法
  params.cudnn_enabled = ctx.userEnabledCuDNN(); // 设置是否用户启用了 CuDNN
  params.allow_tf32 = ctx.allowTF32CuDNN(); // 设置是否允许使用 CuDNN 的 TF32 模式

  // 验证输入参数的形状
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of dimensions, but got: ",
      input.dim(), " and ", grad_output.dim());

  // 如果卷积不是转置操作，则不支持 output_padding
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
        params.output_padding);
    }
  }

  // 扩展 1 维到 2 维
  // 仅对不原生支持 1 维空间输入的后端执行此操作
  if (k == 3 && !input.is_mkldnn() && !input.is_xpu()) {
    // 避免意外通过 NHWC 进行置换的 3 维输入
    input = input.contiguous();
    params.view1d_as_2d();
    grad_output = view4d(grad_output);
    input = view4d(input);
    // 将权重张量转换为四维视图
    weight = view4d(weight);
  }

  // 选择适当的后端来使用。
  ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, /*need_backward=*/ true, params);
  // 确定后端使用的内存格式
  at::MemoryFormat backend_memory_format = determine_backend_memory_format(input, weight, backend);

  // 调用后端计算梯度。
  Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  // 提取卷积核大小
  auto kernel_size = weight.sizes().slice(2);
  switch(backend) {
    case ConvBackend::CudaDepthwise2d:
    {
      // 为深度可分离二维卷积计算反向传播
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) =
        conv_depthwise2d_backward_stub(input.device().type(), grad_output, input,
          weight, kernel_size, params.stride, params.padding, params.dilation, input_weight_output_mask);
      break;
    }
    case ConvBackend::CudaDepthwise3d:
      // 检查输入是否是五维张量
      TORCH_CHECK(input.ndimension() == 5);
      // 为深度可分离三维卷积计算反向传播
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        conv_depthwise3d_backward_stub(
          input.device().type(), grad_output, input, weight, kernel_size, params.stride,
          params.padding, params.dilation, output_mask);
      break;
    case ConvBackend::Cudnn:
    {
      // 检查输入张量类型与参数是否相同
      check_input_same_type_as_parameters(input, weight);
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      // 使用cuDNN库计算卷积反向传播
      std::tie(backend_grad_input, backend_grad_weight) = cudnn_convolution_backward_stub(
          input.device().type(),
          // 仅在反向计算需要时使输入连续
          output_mask[1] ? input.contiguous(backend_memory_format) : input,
          grad_output, weight, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32,
          input_weight_output_mask);
      break;
    }
    case ConvBackend::Mps:
    {
#ifdef USE_MPS
      // 检查输入和参数的类型是否相同
      check_input_same_type_as_parameters(input, weight);
      // 如果使用 MPS 后端，调用 MPS 卷积反向传播函数
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        at::mps_convolution_backward(input, grad_output, weight, params.padding,
          params.stride, params.dilation, params.groups, output_mask);
#else
      // 如果未启用 MPS 后端，则断言报错，因为不支持 MPS 后端
      TORCH_INTERNAL_ASSERT(false, "MPS backend was selected in PyTorch without support");
#endif
      // 结束当前 case
      break;
    }
    case ConvBackend::MpsTranspose:
    {
#ifdef USE_MPS
      // 检查输入和参数的类型是否相同
      check_input_same_type_as_parameters(input, weight);
      // 根据需要，在反向计算时仅在必要时使输入连续
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      // 调用 MPS 转置卷积反向传播函数
      std::tie(backend_grad_input, backend_grad_weight) = at::mps_convolution_transpose_backward(
        output_mask[1] ? input.contiguous(backend_memory_format) : input,
        grad_output, weight, params.padding, params.output_padding,
        params.stride, params.dilation, params.groups, input_weight_output_mask);
#else
      // 如果未启用 MPS 后端，则断言报错，因为不支持 MPS 后端
      TORCH_INTERNAL_ASSERT(false, "MPS backend was selected in PyTorch without support");
#endif
      // 结束当前 case
      break;
    }
    case ConvBackend::CudnnTranspose:
    {
      // 检查输入和参数的类型是否相同
      check_input_same_type_as_parameters(input, weight);
      // 根据需要，在反向计算时仅在必要时使输入连续
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      // 调用 cuDNN 转置卷积反向传播函数
      std::tie(backend_grad_input, backend_grad_weight) = cudnn_convolution_transpose_backward_stub(
        input.device().type(),
        output_mask[1] ? input.contiguous(backend_memory_format) : input,
        grad_output, weight, params.padding, params.output_padding,
        params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32,
        input_weight_output_mask);
      // 结束当前 case
      break;
    }
    case ConvBackend::Empty:
      // 如果输出掩码的第一个元素为真，则将 backend_grad_input 初始化为与输入相同形状的零张量
      if (output_mask[0]) {
        backend_grad_input = at::zeros_like(input);
      }
      // 如果输出掩码的第二个元素为真，则将 backend_grad_weight 初始化为与参数相同形状的零张量
      if (output_mask[1]) {
        backend_grad_weight = at::zeros_like(weight);
      }
      // 如果输出掩码的第三个元素为真，则将 backend_grad_bias 初始化为与指定大小和参数相同选项的零张量
      if (output_mask[2]) {
        backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
      }
      // 结束当前 case
      break;
    case ConvBackend::MkldnnEmpty:
#if AT_MKLDNN_ENABLED()
      // 如果 AT_MKLDNN_ENABLED 宏被启用
      if (output_mask[0]) {
        // 如果输出掩码的第一个位置为真
        if (input.is_mkldnn()) {
          // 如果输入是 MKLDNN 张量
          backend_grad_input = empty_mkldnn(input.sizes(), optTypeMetaToScalarType(input.options().dtype_opt()),
              input.options().layout_opt(), input.options().device_opt(), input.options().pinned_memory_opt());
          // 创建一个空的 MKLDNN 张量作为梯度输入，类型与输入的选项一致，并清零
          backend_grad_input.zero_();
        } else {
          // 如果输入不是 MKLDNN 张量
          backend_grad_input = at::zeros_like(input);
          // 使用 input 的形状创建一个与之相同形状的零张量作为梯度输入
        }
      }
      if (output_mask[1]) {
        // 如果输出掩码的第二个位置为真
        // 在训练过程中，MKLDNN 后端不支持 MKLDNN 权重
        backend_grad_weight = at::zeros_like(weight);
        // 使用 weight 的形状创建一个与之相同形状的零张量作为梯度权重
      }
      if (output_mask[2]) {
        // 如果输出掩码的第三个位置为真
        // 在训练过程中，MKLDNN 后端不支持 MKLDNN 偏置
        backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
        // 使用指定大小和权重选项创建一个零张量作为梯度偏置
      }
#else
      // 如果未启用 MKLDNN 支持
      TORCH_INTERNAL_ASSERT(false, "Mkldnn backend was selected in PyTorch compiled without mkldnn support");
      // 报告错误：在未编译支持 MKLDNN 的 PyTorch 中选择了 MKLDNN 后端
#endif
      // 结束条件语句

    case ConvBackend::Miopen:
      // 如果选择的后端是 Miopen
      check_input_same_type_as_parameters(input, weight);
      // 检查输入与参数类型是否相同
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        miopen_convolution_backward_stub(
          input.device().type(),
          input.contiguous(backend_memory_format), grad_output, weight, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic, output_mask);
      // 执行 Miopen 卷积反向传播函数，并得到梯度输入、梯度权重和梯度偏置

    case ConvBackend::MiopenDepthwise:
      // 如果选择的后端是 MiopenDepthwise
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
          miopen_depthwise_convolution_backward_stub(
            input.device().type(),
            input.contiguous(backend_memory_format), grad_output, weight, params.padding, params.stride,
            params.dilation, params.groups, params.benchmark, params.deterministic, output_mask);
      // 执行 Miopen 深度可分离卷积反向传播函数，并得到梯度输入、梯度权重和梯度偏置

    case ConvBackend::MiopenTranspose:
      // 如果选择的后端是 MiopenTranspose
      check_input_same_type_as_parameters(input, weight);
      // 检查输入与参数类型是否相同
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        miopen_convolution_transpose_backward_stub(
          input.device().type(),
          input.contiguous(backend_memory_format), grad_output, weight, params.padding, params.output_padding,
          params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, output_mask);
      // 执行 Miopen 转置卷积反向传播函数，并得到梯度输入、梯度权重和梯度偏置

    case ConvBackend::Mkldnn:
      // 如果选择的后端是 Mkldnn
      TORCH_CHECK(!weight.is_mkldnn(),
          "The MKLDNN backend does not support weight as an MKLDNN tensor during training");
      // 检查权重是否为 MKLDNN 张量，MKLDNN 后端在训练过程中不支持权重为 MKLDNN 张量
      if (!input.is_mkldnn()) {
        // 如果输入不是 MKLDNN 张量
        input = input.contiguous(backend_memory_format);
        // 强制输入为指定格式的连续张量
        weight = weight.contiguous(backend_memory_format);
        // 强制权重为指定格式的连续张量
      }
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        mkldnn_convolution_backward_stub(input.device().type(), input, grad_output, weight, params.padding,
          params.stride, params.dilation, params.groups, output_mask);
      // 执行 MKLDNN 卷积反向传播函数，并得到梯度输入、梯度权重和梯度偏置
      break;
    // 当选择的卷积后端为MkldnnTranspose时执行以下代码块
    case ConvBackend::MkldnnTranspose:
      // 检查权重是否不是MKLDNN张量，MKLDNN后端不支持在训练期间作为MKLDNN张量的权重
      TORCH_CHECK(!weight.is_mkldnn(),
          "The MKLDNN backend does not support weight as an MKLDNN tensor during training");
      // 如果输入不是MKLDNN张量，则需要将其转换为连续的指定内存格式
      if (!input.is_mkldnn()) {
        input = input.contiguous(backend_memory_format);
        weight = weight.contiguous(backend_memory_format);
      }
      // 调用MKLDNN后端的反向传播函数，计算梯度：输入、权重、偏置
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        mkldnn_convolution_transpose_backward_stub(input.device().type(), input, grad_output, weight, params.padding,
        params.output_padding, params.stride, params.dilation, params.groups, output_mask);
      break;
    // 当选择的卷积后端为Overrideable时执行以下代码块
    case ConvBackend::Overrideable:
      // 只有在输入是后端带有外部实现时才会执行到这里
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        at::convolution_backward_overrideable(grad_output, input, weight, params.stride, params.padding,
          params.dilation, params.transposed, params.output_padding, params.groups, output_mask);
      break;
    // 当选择的卷积后端为Slow3d时执行以下代码块
    case ConvBackend::Slow3d:
      // 注意，当前没有CUDA实现这个内核
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        slow_conv3d_backward_cpu(
            grad_output, input, weight, kernel_size,
            params.stride, params.padding, output_mask);
      break;
    // 处理不原生支持groups > 1的后端情况
    // 当选择的卷积后端为NnpackSpatial、Slow2d、SlowDilated2d、SlowDilated3d、SlowTranspose2d、SlowTranspose3d时执行以下代码块
    case ConvBackend::NnpackSpatial:
    case ConvBackend::Slow2d:
    case ConvBackend::SlowDilated2d:
    case ConvBackend::SlowDilated3d:
    case ConvBackend::SlowTranspose2d:
    case ConvBackend::SlowTranspose3d:
    {
      // 将输入和权重张量转换为指定的内存格式
      input = input.contiguous(backend_memory_format);
      weight = weight.contiguous(backend_memory_format);
      
      // 根据分组数量决定使用不同的反卷积算法
      if (params.groups == 1) {
        // 对于不分组的情况，使用特定的反卷积算法计算梯度
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
          _convolution_backward_nogroup_backend(
            grad_output, input, weight, output_mask, backend, params);
      } else {
        // 对于分组卷积的情况，分别计算每个组的梯度
        std::vector<Tensor> backend_grad_inputs(params.groups);
        std::vector<Tensor> backend_grad_weights(params.groups);
        std::vector<Tensor> backend_grad_biases(params.groups);
        for (int g = 0; g < params.groups; ++g) {
          // 提取当前组的输出梯度、输入和权重
          auto grad_output_g = subtensor(grad_output, 1, params.groups, g);
          auto input_g = subtensor(input, 1, params.groups, g);
          auto weight_g = subtensor(weight, 0, params.groups, g);
          
          // 使用特定的反卷积算法计算当前组的梯度
          std::tie(backend_grad_inputs[g], backend_grad_weights[g], backend_grad_biases[g]) =
            _convolution_backward_nogroup_backend(
              grad_output_g, input_g, weight_g, output_mask, backend, params);
        }
        
        // 根据输出掩码组装整体的梯度
        if (output_mask[0]) {
          backend_grad_input = at::cat(backend_grad_inputs, 1);
        }
        if (output_mask[1]) {
          backend_grad_weight = at::cat(backend_grad_weights, 0);
        }
        if (output_mask[2]) {
          backend_grad_bias = at::cat(backend_grad_biases, 0);
        }
      }
      // 结束当前代码块，执行后续操作
      break;
    }
    // 对于不支持反向传播的后端，抛出错误信息
    case ConvBackend::Winograd3x3Depthwise:
      TORCH_CHECK(false, "Backward is not supported for depthwise 3x3 winograd");
      break;
    case ConvBackend::Xnnpack2d:
      TORCH_CHECK(false, "Backward is not supported for xnnpack");
      break;
    }
    
    // 如果输出掩码表明需要，将二维输入转换为一维，适应不支持一维空间输入的后端
    if (output_mask[0]) {
      if (k == 3 && !input.is_mkldnn() && !input.is_xpu()) {
        backend_grad_input = view3d(backend_grad_input);
      }
    }
    if (output_mask[1]) {
      if (k == 3 && !input.is_mkldnn() && !input.is_xpu()) {
        backend_grad_weight = view3d(backend_grad_weight);
      }
    }
    // 如果输出掩码表明需要，计算偏置梯度，对于不支持的后端在此计算
    if (output_mask[2]) {
      if (!backend_grad_bias.defined()) {
        backend_grad_bias = grad_output.sum((dim == 3) ? IntArrayRef{0, 2, 3, 4} : IntArrayRef{0, 2, 3});
      }
    }
    
    // 返回计算得到的梯度结果元组
    return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

// 设置卷积基准空缓存的函数，接受一个布尔值参数用以启用或禁用
void _cudnn_set_conv_benchmark_empty_cache(bool enable) {
  // 将输入的布尔值赋给全局变量 conv_benchmark_empty_cache
  conv_benchmark_empty_cache = enable;
}

// 获取卷积基准空缓存的当前状态，返回一个布尔值
bool _cudnn_get_conv_benchmark_empty_cache() {
  // 返回当前全局变量 conv_benchmark_empty_cache 的值
  return conv_benchmark_empty_cache;
}

// 结束 at::native 命名空间的声明
} // namespace at::native
```