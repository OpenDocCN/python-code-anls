# `.\pytorch\aten\src\ATen\native\vulkan\ops\Convolution.h`

```py
// 针对头文件的条件编译，仅在使用 Vulkan API 时包含本文件
#pragma once

// 如果定义了 USE_VULKAN_API，则包含 Vulkan 相关头文件
#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>

// 声明命名空间 at::native::vulkan::ops
namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 定义枚举 Conv2dMethod，表示不同的卷积方法
enum Conv2dMethod {
  Conv2dDepthwise,        // 深度可分离卷积
  Conv2dPointwise,        // 点卷积
  Conv2dSlidingWindow,    // 滑动窗口卷积
};

// 声明命名空间 conv2d，包含卷积相关函数
namespace conv2d {

// 重新排列深度可分离卷积的权重张量
Tensor rearrange_weights_dw(const Tensor& weight_in);
// 重新排列二维卷积的权重张量，tconv 表示是否转置卷积
Tensor rearrange_weights_2d(const Tensor& weight_in, bool tconv);
// 重新排列偏置项，根据权重张量和是否转置卷积
Tensor rearrange_bias(
    const std::optional<Tensor>& bias_in,
    const at::Tensor& weight_in,
    bool tconv);

} // namespace conv2d

// 声明命名空间 qconv2d_vk，用于 Vulkan 加速的量化卷积操作
namespace qconv2d_vk {

// 结构体 QParams，描述量化卷积的参数
struct QParams final {
  api::utils::uvec3 out_extents;        // 输出张量的尺寸
  int32_t ic4;                          // 输入通道数，每四个通道分组
  api::utils::ivec4 sizes_2d;            // 二维尺寸
  float output_scale;                    // 输出缩放因子
  float input_scale;                     // 输入缩放因子
  int32_t output_zero_point;             // 输出零点
  int32_t input_zero_point;              // 输入零点
  float weight_scale;                    // 权重缩放因子
  float bias_scale;                      // 偏置项缩放因子
  int32_t weight_zero_point;             // 权重零点
  int32_t bias_zero_point;               // 偏置项零点
  api::utils::ivec2 kernel_size;         // 卷积核尺寸
  api::utils::ivec2 stride;              // 步长
  api::utils::ivec2 padding;             // 填充
  api::utils::ivec2 dilate;              // 膨胀
  api::utils::vec2 clamp;                // 夹紧范围
  api::utils::ivec4 src_filter;          // 源过滤器
};

} // namespace qconv2d_vk

// 类 Conv2dPackedContext，继承自 VulkanPackedContext 和 torch::jit::CustomClassHolder
class Conv2dPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;      // 未打包的列表
  api::ShaderInfo compute_shader_{};     // 计算着色器信息

 public:
  // 构造函数，初始化卷积上下文
  Conv2dPackedContext(
      const Tensor& weight,
      const std::optional<Tensor>& bias,
      const IntArrayRef stride_arg,
      const IntArrayRef padding_arg,
      const IntArrayRef dilation_arg,
      const bool transposed,
      const bool quantized,
      const IntArrayRef output_padding_arg,
      const int64_t groups,
      const std::optional<Scalar>& output_min = c10::nullopt,
      const std::optional<Scalar>& output_max = c10::nullopt);

  /*
   * 为未打包列表中的每个索引分配一个名称。
   */
  struct Unpacked final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;
    static constexpr uint32_t Stride = 2u;
    static constexpr uint32_t Padding = 3u;
    static constexpr uint32_t Dilation = 4u;
    static constexpr uint32_t isTransposed = 5u;
    static constexpr uint32_t isQuantized = 6u;
    static constexpr uint32_t OutputPadding = 7u;
    static constexpr uint32_t Groups = 8u;
    static constexpr uint32_t OutputMin = 9u;
    static constexpr uint32_t OutputMax = 10u;

    static constexpr uint32_t NumArgs = 11u;
  };

  /*
   * 为已打包列表中的每个索引分配一个名称。
   */
  struct Packed final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;
    static constexpr uint32_t OverlayRegion = 2u;
    static constexpr uint32_t Stride = 3u;
    static constexpr uint32_t Padding = 4u;
    static constexpr uint32_t OutputPadding = 5u;
    static constexpr uint32_t Dilation = 6u;
    static constexpr uint32_t isTransposed = 7u;
    static constexpr uint32_t isQuantized = 8u;
    static constexpr uint32_t Groups = 9u;
    // 定义静态常量，表示输出范围的最小值和最大值
    static constexpr uint32_t OutputMin = 10u;
    static constexpr uint32_t OutputMax = 11u;
    // 定义静态常量，表示卷积方法的标识
    static constexpr uint32_t ConvMethod = 12u;
    // 定义静态常量，表示权重大小的标识
    static constexpr uint32_t WeightSizes = 13u;

    // 定义静态常量，表示参数的数量
    static constexpr uint32_t NumArgs = 14u;
  };

  // 定义 Conv2dPackedContext 类的静态成员变量 pack
  static Conv2dPackedContext pack(c10::impl::GenericList);

  // 实现 unpack 函数，返回已解压的元素列表
  const c10::impl::GenericList unpack() const override {
    // 检查 unpacked_ 中是否有元素，如果没有则抛出错误信息
    TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

    // 返回解压后的元素列表
    return unpacked_;
  }

  // 返回 compute_shader_ 引用，用于计算着色器信息
  inline api::ShaderInfo& compute_shader() {
    return compute_shader_;
  }
};

// 创建 Conv2dPackedContext 对象的函数，用于卷积操作
c10::intrusive_ptr<Conv2dPackedContext> create_conv2d_context(
    Tensor&& weight,  // 卷积核权重张量（移动语义）
    std::optional<Tensor>&& bias,  // 可选的卷积偏置张量（移动语义）
    std::vector<int64_t>&& stride,  // 步长向量（移动语义）
    std::vector<int64_t>&& padding,  // 填充向量（移动语义）
    std::vector<int64_t>&& dilation,  // 膨胀向量（移动语义）
    const int64_t groups,  // 组数
    const std::optional<Scalar>& output_min = c10::nullopt,  // 输出最小值（默认空）
    const std::optional<Scalar>& output_max = c10::nullopt);  // 输出最大值（默认空）

// 运行 Conv2dPackedContext 对象的卷积操作函数
Tensor run_conv2d_context(
    const Tensor& input,  // 输入张量
    const c10::intrusive_ptr<Conv2dPackedContext>& context);  // Conv2dPackedContext 上下文对象

// 创建 Transposed Conv2dPackedContext 对象的函数，用于转置卷积操作
c10::intrusive_ptr<Conv2dPackedContext> create_tconv2d_context(
    Tensor&& weight,  // 卷积核权重张量（移动语义）
    std::optional<Tensor>&& bias,  // 可选的卷积偏置张量（移动语义）
    std::vector<int64_t>&& stride,  // 步长向量（移动语义）
    std::vector<int64_t>&& padding,  // 填充向量（移动语义）
    std::vector<int64_t>&& output_padding,  // 输出填充向量（移动语义）
    std::vector<int64_t>&& dilation,  // 膨胀向量（移动语义）
    const int64_t groups,  // 组数
    const std::optional<Scalar>& output_min = c10::nullopt,  // 输出最小值（默认空）
    const std::optional<Scalar>& output_max = c10::nullopt);  // 输出最大值（默认空）

// 运行 Transposed Conv2dPackedContext 对象的转置卷积操作函数
Tensor run_tconv2d_context(
    const Tensor& input,  // 输入张量
    const c10::intrusive_ptr<Conv2dPackedContext>& context);  // Conv2dPackedContext 上下文对象

// 创建量化 Conv2dPackedContext 对象的函数，用于量化卷积操作
c10::intrusive_ptr<Conv2dPackedContext> create_qconv2d_context(
    Tensor&& weight,  // 卷积核权重张量（移动语义）
    std::optional<Tensor>&& bias,  // 可选的卷积偏置张量（移动语义）
    std::vector<int64_t>&& stride,  // 步长向量（移动语义）
    std::vector<int64_t>&& padding,  // 填充向量（移动语义）
    std::vector<int64_t>&& dilation,  // 膨胀向量（移动语义）
    const int64_t groups,  // 组数
    const std::optional<Scalar>& output_min = c10::nullopt,  // 输出最小值（默认空）
    const std::optional<Scalar>& output_max = c10::nullopt);  // 输出最大值（默认空）

// 运行量化 Conv2dPackedContext 对象的卷积操作函数
Tensor run_qconv2d_context(
    const Tensor& input_arg,  // 输入张量
    double scale,  // 缩放因子
    int64_t zero_point,  // 零点
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context);  // Conv2dPackedContext 上下文对象

// 创建量化 Transposed Conv2dPackedContext 对象的函数，用于量化转置卷积操作
c10::intrusive_ptr<Conv2dPackedContext> create_qtconv2d_context(
    Tensor&& weight,  // 卷积核权重张量（移动语义）
    std::optional<Tensor>&& bias,  // 可选的卷积偏置张量（移动语义）
    std::vector<int64_t>&& stride,  // 步长向量（移动语义）
    std::vector<int64_t>&& padding,  // 填充向量（移动语义）
    std::vector<int64_t>&& output_padding,  // 输出填充向量（移动语义）
    std::vector<int64_t>&& dilation,  // 膨胀向量（移动语义）
    const int64_t groups,  // 组数
    const std::optional<Scalar>& output_min = c10::nullopt,  // 输出最小值（默认空）
    const std::optional<Scalar>& output_max = c10::nullopt);  // 输出最大值（默认空）

// 用于向后兼容的类，继承自 torch::jit::CustomClassHolder，实现了 Conv2dOpContext
class Conv2dOpContext final : public torch::jit::CustomClassHolder {
 public:
  // 创建 Conv2dOpContext 对象的静态方法
  static Conv2dOpContext create(
      const Tensor& weight,  // 卷积核权重张量
      const std::optional<Tensor>& bias,  // 可选的卷积偏置张量
      IntArrayRef stride,  // 步长数组引用
      IntArrayRef padding,  // 填充数组引用
      IntArrayRef dilation,  // 膨胀数组引用
      bool transposed,  // 是否转置
      IntArrayRef output_padding,  // 输出填充数组引用
      int64_t groups,  // 组数
      const std::optional<Scalar>& output_min = c10::nullopt,  // 输出最小值（默认空）
      const std::optional<Scalar>& output_max = c10::nullopt);  // 输出最大值（默认空）

  // 定义状态元组类型
  using State = std::tuple<
      Tensor,
      std::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      std::optional<Scalar>,
      std::optional<Scalar>>;

  // 运行卷积操作的方法
  Tensor run(const Tensor& input) const;

  // 解包状态的方法
  State unpack() const;

 private:
  // 显式构造函数，接受 Conv2dPackedContext 参数
  explicit Conv2dOpContext(Conv2dPackedContext conv_context);
  Conv2dPackedContext conv_context_;  // Conv2dPackedContext 对象
};

// 运行 Conv2dOpContext 对象的卷积操作函数，包括输出值的范围限制
Tensor conv2d_clamp_run(
    const Tensor& input,  // 输入张量
    const c10::intrusive_ptr<Conv2dOpContext>& context);  // Conv2dOpContext 上下文对象
/*
 * Declaration of a function `conv2d_clamp_prepack` that pre-packages weights and biases for a 2D convolution operation.
 * This function returns a smart pointer to `Conv2dOpContext`.
 */
c10::intrusive_ptr<Conv2dOpContext> conv2d_clamp_prepack(
    Tensor&& weight,                                     // Input tensor for weights
    std::optional<Tensor>&& bias,                        // Optional input tensor for biases
    std::vector<int64_t>&& stride,                       // Vector specifying the stride dimensions
    std::vector<int64_t>&& padding,                      // Vector specifying the padding dimensions
    std::vector<int64_t>&& dilation,                     // Vector specifying the dilation dimensions
    const int64_t groups,                                // Number of groups for grouped convolution
    const std::optional<Scalar>& output_min,             // Optional minimum output value
    const std::optional<Scalar>& output_max);            // Optional maximum output value

/*
 * Definition of the `Conv1dPackedContext` class which is a context holder for packed convolution operations.
 * It inherits from `VulkanPackedContext` and `torch::jit::CustomClassHolder`.
 */
class Conv1dPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;                     // List for storing unpacked data
  api::ShaderInfo compute_shader_;                      // Shader information for compute operations

 public:
  /*
   * Constructor for `Conv1dPackedContext`.
   * Initializes with weight, bias, stride, padding, dilation, and groups.
   */
  Conv1dPackedContext(
      const Tensor& weight,                             // Input tensor for weights
      const std::optional<Tensor>& bias,                // Optional input tensor for biases
      const IntArrayRef stride_arg,                     // Reference to array specifying stride dimensions
      const IntArrayRef padding_arg,                    // Reference to array specifying padding dimensions
      const IntArrayRef dilation_arg,                   // Reference to array specifying dilation dimensions
      const int64_t groups);                            // Number of groups for grouped convolution

  /*
   * Static struct defining symbolic names for indices in the unpacked list.
   */
  struct Unpacked final {
    static constexpr uint32_t Weight = 0u;              // Index for weight tensor
    static constexpr uint32_t Bias = 1u;                // Index for bias tensor
    static constexpr uint32_t Stride = 2u;              // Index for stride dimensions
    static constexpr uint32_t Padding = 3u;             // Index for padding dimensions
    static constexpr uint32_t Dilation = 4u;            // Index for dilation dimensions
    static constexpr uint32_t Groups = 5u;              // Index for groups
    static constexpr uint32_t NumArgs = 6u;             // Total number of arguments
  };

  /*
   * Static struct defining symbolic names for indices in the packed list.
   */
  struct Packed final {
    static constexpr uint32_t Weight = 0u;              // Index for packed weight tensor
    static constexpr uint32_t Bias = 1u;                // Index for packed bias tensor
    static constexpr uint32_t Stride = 2u;              // Index for packed stride dimensions
    static constexpr uint32_t Padding = 3u;             // Index for packed padding dimensions
    static constexpr uint32_t Dilation = 4u;            // Index for packed dilation dimensions
    static constexpr uint32_t Groups = 5u;              // Index for packed groups
    static constexpr uint32_t WeightSizes = 6u;         // Index for packed weight sizes
    static constexpr uint32_t NumArgs = 7u;             // Total number of packed arguments
  };

  /*
   * Static method to pack data into `Conv1dPackedContext` from a generic list.
   */
  static Conv1dPackedContext pack(c10::impl::GenericList);

  /*
   * Method to unpack data from `Conv1dPackedContext`.
   * Returns a generic list containing unpacked data.
   * Throws an error if `unpacked_` is empty.
   */
  const c10::impl::GenericList unpack() const override;

  /*
   * Inline method to access compute shader information.
   * Returns a reference to `compute_shader_`.
   */
  inline api::ShaderInfo& compute_shader();
};

/*
 * Declaration of a function `create_conv1d_context` that creates a context for 1D convolution operations.
 * Returns a smart pointer to `Conv1dPackedContext`.
 */
c10::intrusive_ptr<Conv1dPackedContext> create_conv1d_context(
    Tensor&& weight,                                     // Input tensor for weights
    std::optional<Tensor>&& bias,                        // Optional input tensor for biases
    std::vector<int64_t>&& stride,                       // Vector specifying the stride dimensions
    std::vector<int64_t>&& padding,                      // Vector specifying the padding dimensions
    std::vector<int64_t>&& dilation,                     // Vector specifying the dilation dimensions
    const int64_t groups);                               // Number of groups for grouped convolution

/*
 * Declaration of a function `run_conv1d_context` that executes a 1D convolution operation using a pre-packed context.
 * Takes input tensor `input` and a smart pointer to `Conv1dPackedContext`.
 * Returns the result tensor.
 */
Tensor run_conv1d_context(
    const Tensor& input,                                 // Input tensor for convolution operation
    const c10::intrusive_ptr<Conv1dPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```