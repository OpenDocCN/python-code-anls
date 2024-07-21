# `.\pytorch\aten\src\ATen\native\vulkan\ops\BinaryOp.cpp`

```
// 引入头文件，ATen 是 PyTorch 中的张量库，Vulkan 是一个图形和计算 API
#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>
#include <vector>

// 命名空间声明开始
namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 使用 Vulkan 相关的 API 函数
using namespace api::utils;

// 执行与标量的二元操作，返回处理后的张量
Tensor binary_op_scalar(
    const Tensor& self_arg,                   // 输入张量 self
    const Scalar& other,                      // 标量 other
    const std::optional<Scalar>& alpha_arg,   // 可选的标量 alpha
    const api::ShaderInfo& shader_descriptor) {  // Vulkan 着色器描述信息

  // 获取 Vulkan 的上下文环境
  api::Context* const context = api::context();

  // 如果 self_arg 是 Vulkan 张量，则直接使用；否则转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  // 创建 Vulkan 张量 v_output，保留了 v_self 的大小和数据类型
  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  // 计算 other 的浮点值，如果 alpha_arg 存在则乘以 alpha_arg 的浮点值
  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();

  // 定义 Vulkan 着色器中的 Block 结构体，包含了张量的尺寸和 other 的值
  const struct Block final {
    uvec3 extents;  // 张量的尺寸
    int fill0;      // 填充字段，未使用
    float other;    // other 的浮点值
  } block{
      v_self.extents(),  // 使用 v_self 的尺寸
      0,                 // 填充字段填 0
      other_val,         // other 的计算值
  };

  // 创建 UniformParamsBuffer 对象，用于传递参数给 Vulkan 着色器
  api::UniformParamsBuffer params(context, block);

  // 创建 PipelineBarrier 对象，表示计算管线的障碍
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 环境
  context->submit_compute_job(
      shader_descriptor,                          // 着色器描述符
      pipeline_barrier,                           // 管线障碍
      v_output.extents(),                         // 全局工作组大小
      adaptive_work_group_size(v_output.extents()),// 局部工作组大小
      VK_NULL_HANDLE,                             // 使用默认的 fence handle
      v_output.image(                             // 输出张量的映像
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());                          // 参数缓冲区

  // 将 Vulkan 张量 v_output 转换为普通张量并返回
  return convert(v_output);
}

// 预处理 other_arg 张量，将非 Vulkan 张量转换为 Vulkan 张量
Tensor binary_op_preprocess_other_arg(const Tensor& other_arg) {
  // 类似 binary_op_scalar 函数，将已知的整数类型（但不包括量化类型）张量转换为浮点数

  // 将输入的 other_arg 复制到 other 中
  Tensor other = other_arg;

  // 如果 other 不是 Vulkan 张量
  if (!other.is_vulkan()) {
    // 根据其标量类型执行相应的类型转换
    switch (other.scalar_type()) {
      case at::kByte:
      case at::kChar:
      case at::kShort:
      case at::kInt:
      case at::kLong:
      case at::kDouble:
        other = other.to(kFloat);  // 转换为 float 类型
        break;
      case at::kFloat:
        // 如果已经是 float 类型，则不进行任何操作
        break;
      default:
        // 抛出异常，指示不支持的类型
        TORCH_CHECK(
            false,
            "binary_op_tensor, doesn't support type %s",
            other.scalar_type());
        break;
    }
    // 将转换后的 other 张量转换为 Vulkan 张量
    other = other.vulkan();
  }

  // 返回预处理后的 other 张量
  return other;
}

// 继续定义 binary_op_scalar_ 函数，但未提供完整的实现
Tensor& binary_op_scalar_(
    Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    // 检查 self_arg 是否为 Vulkan 张量，否则抛出错误信息
    TORCH_CHECK(
        self_arg.is_vulkan(),
        "Vulkan: In-place operator is only supported on Vulkan tensors.");
    
    // 获取当前上下文对象指针
    api::Context* const context = api::context();
    
    // 将 self_arg 转换为 Vulkan 张量对象 v_self
    vTensor& v_self = convert(self_arg);
    
    // 计算 other_val，若存在 alpha_arg，则将 other 转换为 float 后乘以 alpha_arg 转换为 float 后的值，否则直接转换 other 为 float
    const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>() : other.to<float>();
    
    // 定义名为 Block 的结构体实例 block，包含成员 extents（表示张量的尺寸）、fill0（填充值，此处为0）、other（前面计算得到的 other_val）
    const struct Block final {
      uvec3 extents;
      int fill0;
      float other;
    } block{
        v_self.extents(),
        0,
        other_val,
    };
    
    // 使用 block 创建 UniformParamsBuffer 对象 params，传入当前上下文 context
    api::UniformParamsBuffer params(context, block);
    
    // 创建空的 PipelineBarrier 对象 pipeline_barrier
    api::PipelineBarrier pipeline_barrier{};
    
    // 提交计算作业到 Vulkan 上下文，传入以下参数：
    context->submit_compute_job(
        // shader descriptor，描述计算任务使用的着色器信息
        shader_descriptor,
        // pipeline barrier，指定流水线屏障，此处为空
        pipeline_barrier,
        // global work group size，全局工作组大小为 v_self 的尺寸
        v_self.extents(),
        // local work group size，本地工作组大小根据 v_self 的尺寸自适应确定
        adaptive_work_group_size(v_self.extents()),
        // fence handle，栅栏句柄为空
        VK_NULL_HANDLE,
        // shader arguments，通过 v_self 的图像数据和指定访问权限设置 shader 参数
        v_self.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        // params buffer，传入 params 对象的缓冲区
        params.buffer());
    
    // 返回原始的 self_arg
    return self_arg;
// 定义二元操作函数，接受两个张量、可选的标量乘数和着色器信息作为参数，并返回一个张量
Tensor binary_op_tensor(
    const Tensor& self_arg,                     // 第一个输入张量的引用
    const Tensor& other_arg,                    // 第二个输入张量的引用
    const std::optional<Scalar>& alpha_arg,     // 可选的标量乘数
    const api::ShaderInfo& shader_descriptor) { // 着色器描述信息

  utils::is_broadcastable(self_arg, other_arg);  // 检查两个张量是否可以广播

  api::Context* const context = api::context(); // 获取当前上下文环境的指针

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan(); // 如果第一个输入张量是 Vulkan 张量，则使用它，否则转换为 Vulkan 张量
  const vTensor& v_self = convert(self); // 将 Vulkan 张量转换为 vTensor 格式

  Tensor other = binary_op_preprocess_other_arg(other_arg); // 预处理第二个输入张量
  const vTensor& v_other = convert(other); // 将第二个输入张量转换为 vTensor 格式

  // 创建输出张量 v_output，使用当前上下文、广播后的大小和第一个输入张量的数据类型
  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      v_self.dtype(),
  };

  // 获取标量乘数 alpha 的值，如果未提供则默认为 1.0
  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;

  // 定义用于着色器执行的数据块结构体
  const struct Block final {
    uvec4 output_tensor_size;    // 输出张量的尺寸
    uvec4 input_tensor_size;     // 输入张量 self 的尺寸
    uvec4 other_tensor_size;     // 输入张量 other 的尺寸
    float alpha;                 // 标量乘数 alpha
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},

      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
  };

  // 创建用于传递给着色器的参数缓冲区
  api::UniformParamsBuffer params(context, block);

  // 创建流水线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到上下文环境中
  context->submit_compute_job(
      // 着色器描述信息
      shader_descriptor,
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 自适应本地工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 围栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 vTensor 格式的输出张量转换为标准 Tensor 格式并返回
  return convert(v_output);
}
    // 检查是否可以广播操作
    utils::is_broadcastable(self_arg, other_arg);
    // 获取当前上下文
    api::Context* const context = api::context();

    // 根据 self_arg 是否为 Vulkan 张量选择适当的 Tensor 对象
    const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
    // 将 self 转换为 Vulkan 张量的视图
    const vTensor& v_self = convert(self);
    // 根据 other_arg 是否为 Vulkan 张量选择适当的 Tensor 对象
    const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
    // 将 other 转换为 Vulkan 张量的视图
    const vTensor& v_other = convert(other);

    // 断言输入的 Vulkan 张量是否是量化的
    TORCH_CHECK(v_self.is_quantized(), "Input tensor is not quantized");
    TORCH_CHECK(v_other.is_quantized(), "Input tensor is not quantized");

    // 创建一个输出 Vulkan 张量 v_output，用于量化的计算
    vTensor v_output{
        context,
        utils::broadcast_size(self_arg, other_arg),
        scale,
        zero_point,
        api::kQUInt8,
    };

    // 获取 self 和 other 的缩放因子和零点偏移
    const double scale1 = v_self.get_scale();
    const double scale2 = v_other.get_scale();
    const int64_t zero_point1 = v_self.get_zero_point();
    const int64_t zero_point2 = v_other.get_zero_point();

    // 定义一个结构体 block，包含各种计算所需的参数
    const struct Block final {
      uvec3 extents;
      uint32_t channelSize;
      uvec3 input1Extents;
      uint32_t channelBatchSize1;
      uvec3 input2Extents;
      uint32_t channelBatchSize2;
      float scale1;
      float scale2;
      int32_t zeroPoint1;
      int32_t zeroPoint2;
      float scale;
      float fill1;
      int32_t zeroPoint;
      int32_t fill2;
    } block{
        v_output.extents(),
        get_dim<Dim4D::Channel>(v_output),
        v_self.extents(),
        get_dim<Dim4D::Channel>(self) * get_dim<Dim4D::Batch>(self),
        v_other.extents(),
        get_dim<Dim4D::Channel>(other) * get_dim<Dim4D::Batch>(other),
        safe_downcast<float>(scale1),
        safe_downcast<float>(scale2),
        safe_downcast<int32_t>(zero_point1),
        safe_downcast<int32_t>(zero_point2),
        safe_downcast<float>(scale),
        0.0f,
        safe_downcast<int32_t>(zero_point),
        0u,
    };

    // 创建一个 UniformParamsBuffer 对象 params，用于存储 block 的数据
    api::UniformParamsBuffer params(context, block);
    // 创建一个 PipelineBarrier 对象 pipeline_barrier
    api::PipelineBarrier pipeline_barrier{};

    // 使用 context 提交计算作业，执行量化操作
    context->submit_compute_job(
        // 着色器描述符
        shader_descriptor,
        // 管线屏障
        pipeline_barrier,
        // 全局工作组大小
        v_output.extents(),
        // 本地工作组大小
        adaptive_work_group_size(v_output.extents()),
        // 栅栏句柄
        VK_NULL_HANDLE,
        // 着色器参数
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // 参数缓冲区
        params.buffer());

    // 将输出 Vulkan 张量 v_output 转换为量化后的结果并返回
    return convert_quantized(v_output);
}

Tensor& binary_op_tensor_(
    Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  // 检查输入张量的维度是否满足 Vulkan 中的要求
  TORCH_CHECK(
      get_dim<Dim4D::Batch>(self_arg) >= get_dim<Dim4D::Batch>(other_arg) &&
          get_dim<Dim4D::Channel>(self_arg) >=
              get_dim<Dim4D::Channel>(other_arg) &&
          get_dim<Dim4D::Height>(self_arg) >=
              get_dim<Dim4D::Height>(other_arg) &&
          get_dim<Dim4D::Width>(self_arg) >= get_dim<Dim4D::Width>(other_arg),
      "Dimensions of input tensor to Vulkan in-place binary elementwise op "
      "must be less than or equal the dimensions of the underlying tensor.");

  // 检查张量是否可以进行广播操作
  utils::is_broadcastable(self_arg, other_arg);

  // 检查张量是否为 Vulkan 张量
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  // 获取当前的 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入张量转换为 Vulkan 张量
  vTensor& v_self = convert(self_arg);

  // 预处理其他输入张量
  Tensor other = binary_op_preprocess_other_arg(other_arg);

  // 将其他输入张量转换为 Vulkan 张量
  const vTensor& v_other = convert(other);

  // 获取 alpha 值，如果未提供则默认为 1.0
  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  // 定义包含张量尺寸和 alpha 值的 Block 结构体
  const struct Block final {
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha 值
      safe_downcast<float>(alpha),
  };

  // 创建 UniformParamsBuffer 对象，用于存储参数
  api::UniformParamsBuffer params(context, block);
  // 创建 PipelineBarrier 对象，用于控制管线
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 上下文
  context->submit_compute_job(
      // shader 描述符
      shader_descriptor,
      // 管线障碍
      pipeline_barrier,
      // 全局工作组大小
      v_self.extents(),
      // 本地工作组大小
      adaptive_work_group_size(v_self.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // shader 参数
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 返回原始的 self_arg 引用
  return self_arg;
}

Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  // 调用标量二元操作函数，返回新张量
  return binary_op_scalar(
      self_arg, other, std::optional<Scalar>(alpha), VK_KERNEL(add_scalar));
}

Tensor& add_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  // 调用标量二元操作函数，返回原始 self 引用
  return binary_op_scalar_(
      self, other, std::optional<Scalar>(alpha), VK_KERNEL(add_scalar_inplace));
}

Tensor quantized_add(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  // 调用量化二元操作函数，返回新张量
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_add));
}
# 执行量化的张量减法操作，返回结果张量
Tensor quantized_sub(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_sub));
}

# 执行量化的张量乘法操作，返回结果张量
Tensor quantized_mul(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_mul));
}

# 执行量化的张量除法操作，返回结果张量
Tensor quantized_div(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_div));
}

# 执行张量加法操作，返回结果张量
Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg, other_arg, std::optional<Scalar>(alpha), VK_KERNEL(add));
}

# 执行张量原地加法操作，修改并返回自身张量
Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(alpha), VK_KERNEL(add_inplace));
}

# 执行张量减去标量操作，返回结果张量
Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar));
}

# 执行张量原地减去标量操作，修改并返回自身张量
Tensor& sub_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar_inplace));
}

# 执行张量减法操作，返回结果张量
Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg, other_arg, std::optional<Scalar>(alpha), VK_KERNEL(sub));
}

# 执行张量原地减法操作，修改并返回自身张量
Tensor& sub_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(alpha), VK_KERNEL(sub_inplace));
}

# 执行张量乘以标量操作，返回结果张量
Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg, other, std::optional<Scalar>(), VK_KERNEL(mul_scalar));
}

# 执行张量原地乘以标量操作，修改并返回自身张量
Tensor& mul_scalar_(
    Tensor& self,
    const Scalar& other) {
  return binary_op_scalar_(
      self, other, std::optional<Scalar>(), VK_KERNEL(mul_scalar_inplace));
}

# 执行张量乘法操作，返回结果张量
Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg, other_arg, std::optional<Scalar>(), VK_KERNEL(mul));
}

# 执行张量原地乘法操作，修改并返回自身张量
Tensor& mul_tensor_(
    Tensor& self,
    const Tensor& other_arg) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(), VK_KERNEL(mul_inplace));
}

# 执行张量除以标量操作，返回结果张量
Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar));
}
// 实现对张量进行原地除以标量的操作
Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  // 调用 binary_op_scalar_ 函数，对 self 张量进行标量除法操作，使用 1.0 / other.to<float>() 作为除数
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar_inplace));
}

// 对两个张量进行除法操作
Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  // 调用 binary_op_tensor 函数，对两个张量进行除法操作，使用 VK_KERNEL(div) 内核
  return binary_op_tensor(
      self_arg, other_arg, std::optional<Scalar>(), VK_KERNEL(div));
}

// 对张量进行原地除以另一个张量的操作
Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  // 调用 binary_op_tensor_ 函数，对 self 张量进行原地除法操作，使用 VK_KERNEL(div_inplace) 内核
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(), VK_KERNEL(div_inplace));
}

// 对张量进行幂次方操作
Tensor pow(const Tensor& self, const Tensor& other) {
  // 调用 binary_op_tensor 函数，对两个张量进行幂次方操作，使用 VK_KERNEL(pow) 内核
  return binary_op_tensor(self, other, std::optional<Scalar>(), VK_KERNEL(pow));
}

// 对张量进行原地幂次方操作
Tensor& pow_(Tensor& self, const Tensor& other) {
  // 调用 binary_op_tensor_ 函数，对 self 张量进行原地幂次方操作，使用 VK_KERNEL(pow_inplace) 内核
  return binary_op_tensor_(
      self, other, std::optional<Scalar>(), VK_KERNEL(pow_inplace));
}

// 对张量和标量进行幂次方操作
Tensor pow_tensor_scalar(const Tensor& self, const Scalar& other) {
  // 调用 binary_op_scalar 函数，对张量和标量进行幂次方操作，使用 VK_KERNEL(pow_tensor_scalar) 内核
  return binary_op_scalar(
      self, other, std::optional<Scalar>(), VK_KERNEL(pow_tensor_scalar));
}

// 对张量进行原地幂次方和标量操作
Tensor& pow_tensor_scalar_(Tensor& self, const Scalar& other) {
  // 调用 binary_op_scalar_ 函数，对 self 张量进行原地幂次方和标量操作，使用 VK_KERNEL(pow_tensor_scalar_inplace) 内核
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_tensor_scalar_inplace));
}

// 对标量和张量进行幂次方操作
Tensor pow_scalar_tensor(const Scalar& self, const Tensor& other) {
  // 调用 binary_op_scalar 函数，对标量和张量进行幂次方操作，使用 VK_KERNEL(pow_scalar_tensor) 内核
  return binary_op_scalar(
      other, self, std::optional<Scalar>(), VK_KERNEL(pow_scalar_tensor));
}

// 对张量进行标量地除以操作
Tensor floor_divide_scalar(const Tensor& self, const Scalar& other) {
  // 检查除数是否为零，避免除以零的情况
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar: can't divide by zero");
  // 调用 binary_op_scalar 函数，对张量进行标量地除以操作，使用 1.0 / other.to<float>() 作为除数
  return binary_op_scalar(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar));
}

// 对张量进行原地标量地除以操作
Tensor& floor_divide_scalar_(Tensor& self, const Scalar& other) {
  // 检查除数是否为零，避免除以零的情况
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar_: can't divide by zero");
  // 调用 binary_op_scalar_ 函数，对 self 张量进行原地标量地除以操作，使用 1.0 / other.to<float>() 作为除数
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar_inplace));
}

// 使用 Vulkan API 实现对张量进行地板除法操作
Tensor floor_divide_tensor(const Tensor& self, const Tensor& other) {
  // 调用 binary_op_tensor 函数，对张量进行地板除法操作，使用 VK_KERNEL(floor_divide) 内核
  return binary_op_tensor(
      self, other, std::optional<Scalar>(), VK_KERNEL(floor_divide));
}

// 使用 Vulkan API 实现对张量进行原地地板除法操作
Tensor& floor_divide_tensor_(Tensor& self, const Tensor& other_arg) {
  // 调用 binary_op_tensor_ 函数，对 self 张量进行原地地板除法操作，使用 VK_KERNEL(floor_divide_inplace) 内核
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide_inplace));
}

#ifdef USE_VULKAN_API
// 实现 Torch 库函数的 Vulkan 后端，注册各种操作的实现
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
    // 注册 aten::add.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
    // 注册 aten::add_.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
    // 注册 aten::add.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
    // 注册 aten::add_.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
    // 注册 aten::sub.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
    // 注册 aten::sub_.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
    // 注册 aten::sub.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
    // 注册 aten::sub_.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
    // 注册 aten::mul.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
    // 注册 aten::mul_.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
    // 注册 aten::mul.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
    // 注册 aten::mul_.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
    // 注册 aten::div.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
    // 注册 aten::div_.Scalar 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
    // 注册 aten::div.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
    // 注册 aten::div_.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
    // 注册 aten::pow.Tensor_Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::pow.Tensor_Tensor"), TORCH_FN(pow));
    // 注册 aten::pow_.Tensor 操作的 Vulkan 后端实现
    m.impl(TORCH_SELECTIVE_NAME("aten::pow_.Tensor"), TORCH_FN(pow_));
    // 注册 aten::pow.Tensor_Scalar 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::pow.Tensor_Scalar"),
        TORCH_FN(pow_tensor_scalar));
    // 注册 aten::pow_.Scalar 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::pow_.Scalar"), TORCH_FN(pow_tensor_scalar_));
    // 注册 aten::pow.Scalar 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::pow.Scalar"), TORCH_FN(pow_scalar_tensor));
    // 注册 aten::floor_divide.Scalar 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::floor_divide.Scalar"),
        TORCH_FN(floor_divide_scalar));
    // 注册 aten::floor_divide_.Scalar 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::floor_divide_.Scalar"),
        TORCH_FN(floor_divide_scalar_));
    // 注册 aten::floor_divide 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::floor_divide"),
        TORCH_FN(floor_divide_tensor));
    // 注册 aten::floor_divide_.Tensor 操作的 Vulkan 后端实现
    m.impl(
        TORCH_SELECTIVE_NAME("aten::floor_divide_.Tensor"),
        TORCH_FN(floor_divide_tensor_));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```