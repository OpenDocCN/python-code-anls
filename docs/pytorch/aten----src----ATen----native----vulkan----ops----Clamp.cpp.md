# `.\pytorch\aten\src\ATen\native\vulkan\ops\Clamp.cpp`

```
# 包含 Vulkan 操作的常用头文件
#include <ATen/native/vulkan/ops/Common.h>
# 包含 Torch 库的头文件
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

# 实现 Vulkan 下的 _clamp 函数，对输入张量进行数值范围限制
Tensor _clamp(
    const Tensor& self_arg,                   // 输入张量
    const std::optional<Scalar>& min,         // 可选的最小值
    const std::optional<Scalar>& max,         // 可选的最大值
    const api::ShaderInfo& shader_descriptor  // Vulkan 着色器描述信息
) {
  # 至少 'min' 或 'max' 中的一个必须不为 None
  TORCH_CHECK(min || max, "At least one of 'min' or 'max' must not be None");

  # 获取 Vulkan API 上下文
  api::Context* const context = api::context();

  # 如果输入张量已经在 Vulkan 内存中，则直接使用；否则转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self_arg);

  # 创建输出 Vulkan 张量，保持输入张量的大小和数据类型
  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };
  # 如果输入张量是量化的，则设置输出张量为量化类型，并传递量化相关信息
  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  # 创建用于传递给 Vulkan 着色器的统一参数缓冲区
  api::UniformParamsBuffer params;

  # 如果输入张量是量化的，计算量化的最小和最大值，以供 Vulkan 着色器使用
  if (v_self.is_quantized()) {
    float mini = min
        ? roundevenf(min->to<float>() / float(v_self.get_scale())) +
            float(v_self.get_zero_point())
        : -std::numeric_limits<float>::infinity();
    float maxi = max
        ? roundevenf(max->to<float>() / float(v_self.get_scale())) +
            float(v_self.get_zero_point())
        : std::numeric_limits<float>::infinity();
    # 定义用于 Vulkan 着色器的参数块结构体
    const struct Block final {
      uvec3 extents;
      uint32_t align;
      vec2 clamp;
    } block{
        v_output.extents(),
        0u,
        {mini, maxi},
    };
    # 创建 Vulkan 着色器参数缓冲区
    params = api::UniformParamsBuffer(context, block);
  } else {
    # 如果输入张量不是量化的，直接传递给 Vulkan 着色器的参数块结构体
    const struct Block final {
      uvec3 extents;
      uint32_t align;
      vec2 clamp;
    } block{
        v_output.extents(),
        0u,
        {
            min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
            max ? max->to<float>() : std::numeric_limits<float>::infinity(),
        },
    };
    # 创建 Vulkan 着色器参数缓冲区
    params = api::UniformParamsBuffer(context, block);
  }

  # 定义 Vulkan 管线障碍对象
  api::PipelineBarrier pipeline_barrier{};

  # 提交计算任务给 Vulkan API
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管线障碍对象
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  # 将 Vulkan 张量转换为 Torch 张量并返回
  return convert(v_output);
}

# 对外暴露的 clamp 函数，调用 _clamp 函数，并指定着色器描述符
Tensor clamp(
    const Tensor& self_arg,                   // 输入张量
    const std::optional<Scalar>& min,         // 可选的最小值
    const std::optional<Scalar>& max          // 可选的最大值
) {
  return _clamp(self_arg, min, max, VK_KERNEL(clamp));
}

# Vulkan 下的 _clamp_ 函数，用于原位数值范围限制
Tensor& _clamp_(
    Tensor& self_arg,                         // 输入输出张量
    const std::optional<Scalar>& min,         // 可选的最小值
    const std::optional<Scalar>& max,         // 可选的最大值
    // 检查 'min' 或 'max' 中至少有一个不为 None
    TORCH_CHECK(min || max, "At least one of 'min' or 'max' must not be None");
    
    // 检查 self_arg 是否为 Vulkan 张量，只有在 Vulkan 环境下才支持原地 clamp 操作
    TORCH_CHECK(
        self_arg.is_vulkan(),
        "Vulkan: In-place clamp is only supported on Vulkan tensors.");
    
    // 获取当前的 Vulkan 渲染上下文
    api::Context* const context = api::context();
    
    // 根据 self_arg 的类型，获取对应的 Vulkan 张量
    const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
    // 将 Tensor 转换为 vTensor
    vTensor& v_self = convert(self);
    
    // 创建 UniformParamsBuffer 对象，用于存储 uniform 参数
    api::UniformParamsBuffer params;
    
    // 如果 v_self 是量化的张量
    if (v_self.is_quantized()) {
      // 计算 mini 和 maxi 的值，根据量化参数进行调整
      float mini = min
          ? roundevenf(min->to<float>() / float(v_self.get_scale())) +
              float(v_self.get_zero_point())
          : -std::numeric_limits<float>::infinity();
      float maxi = max
          ? roundevenf(max->to<float>() / float(v_self.get_scale())) +
              float(v_self.get_zero_point())
          : std::numeric_limits<float>::infinity();
    
      // 定义 UniformParamsBuffer 所需的结构体 Block
      const struct Block final {
        uvec3 extents;  // 张量的尺寸
        uint32_t align;  // 对齐参数
        vec2 clamp;      // clamp 范围
      } block{
          v_self.extents(),
          0u,
          {mini, maxi},
      };
    
      // 使用 Block 创建 UniformParamsBuffer 对象
      params = api::UniformParamsBuffer(context, block);
    } else {
      // 如果 v_self 不是量化的张量，则直接使用给定的 min 和 max 值
      const struct Block final {
        uvec3 extents;  // 张量的尺寸
        uint32_t align;  // 对齐参数
        vec2 clamp;      // clamp 范围
      } block{
          v_self.extents(),
          0u,
          {
              min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
              max ? max->to<float>() : std::numeric_limits<float>::infinity(),
          },
      };
    
      // 使用 Block 创建 UniformParamsBuffer 对象
      params = api::UniformParamsBuffer(context, block);
    }
    
    // 创建空的 PipelineBarrier 对象
    api::PipelineBarrier pipeline_barrier{};
    
    // 提交计算任务到 Vulkan 渲染上下文中
    context->submit_compute_job(
        // shader descriptor，指定计算任务的着色器描述符
        shader_descriptor,
        // pipeline barrier，管线屏障
        pipeline_barrier,
        // global work group size，全局工作组大小
        v_self.extents(),
        // local work group size，本地工作组大小，根据张量尺寸自动调整
        adaptive_work_group_size(v_self.extents()),
        // fence handle，屏障句柄
        VK_NULL_HANDLE,
        // shader arguments，着色器参数，读写访问张量内存
        v_self.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        // params buffer，参数缓冲区，传递 UniformParamsBuffer 对象的缓冲区
        params.buffer());
    
    // 返回 self_arg，表示原地 clamp 操作完成后的张量
    return self_arg;
}

// 使用指定阈值和值对张量进行阈值化操作
Tensor threshold(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  return _clamp(self, threshold, value, VK_KERNEL(threshold));
}

// 对张量进行原位(clamp)操作，限制在指定的最小和最大值范围内
Tensor& clamp_(
    Tensor& self,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  return _clamp_(self, min, max, VK_KERNEL(clamp_));
}

// 使用指定的着色器描述符对张量进行激活操作
Tensor activation(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  // 如果张量在 Vulkan 上下文中，则直接使用，否则转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  // 创建 Vulkan 张量作为输出，具有相同的大小和数据类型
  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  // 定义块结构，描述计算作业的工作组范围
  const struct Block final {
    uvec3 extents;
    uint32_t _;
  } block{
      v_output.extents(),
      0u,
  };

  // 创建用于统一参数的缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建管道屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 上下文中
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管道屏障
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
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 张量转换为正常张量并返回
  return convert(v_output);
}

// 使用指定的着色器描述符对张量进行原位激活操作
Tensor& activation_(
    Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  // 检查张量是否在 Vulkan 上下文中
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  // 转换张量为 Vulkan 张量
  vTensor& v_self = convert(self_arg);

  // 定义块结构，描述计算作业的工作组范围
  const struct Block final {
    uvec3 extents;
    uint32_t _;
  } block{
      v_self.extents(),
      0u,
  };

  // 创建用于统一参数的缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建管道屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 上下文中
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      v_self.extents(),
      // 本地工作组大小
      adaptive_work_group_size(v_self.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // 参数缓冲区
      params.buffer());

  // 返回原始张量
  return self_arg;
}

// 使用指定的最小和最大值对张量进行硬切线(hardtanh)操作
Tensor hardtanh(const Tensor& self, const Scalar& min, const Scalar& max) {
  return ops::_clamp(self, min, max, VK_KERNEL(clamp));
}

// 使用指定的最小和最大值对张量进行原位硬切线(hardtanh)操作
Tensor& hardtanh_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  return ops::_clamp_(self, min, max, VK_KERNEL(clamp_));
}
Tensor relu(const Tensor& self) {
  return (
      (self.scalar_type() == at::kQUInt8)  // 如果张量是无符号8位整数类型
          ? ops::_clamp(                   // 调用 ops::_clamp 函数，用于张量的截断操作
                self, 0, c10::nullopt, VK_KERNEL(quantized_clamp_quint8))
          : ((self.scalar_type() == at::kQInt8)  // 如果张量是有符号8位整数类型
                 ? ops::_clamp(                  // 调用 ops::_clamp 函数，用于张量的截断操作
                       self, 0, c10::nullopt, VK_KERNEL(quantized_clamp_qint8))
                 : ops::_clamp(self, 0, c10::nullopt, VK_KERNEL(clamp))));  // 否则调用通常的 ops::_clamp 函数
}

Tensor& relu_(Tensor& self) {
  return (
      (self.scalar_type() == at::kQUInt8)  // 如果张量是无符号8位整数类型
          ? ops::_clamp_(                  // 调用 inplace 版本的 ops::_clamp_ 函数，用于张量的截断操作
                self, 0, c10::nullopt, VK_KERNEL(quantized_clamp_quint8_))
          : ((self.scalar_type() == at::kQInt8)  // 如果张量是有符号8位整数类型
                 ? ops::_clamp_(                 // 调用 inplace 版本的 ops::_clamp_ 函数，用于张量的截断操作
                       self, 0, c10::nullopt, VK_KERNEL(quantized_clamp_qint8_))
                 : ops::_clamp_(self, 0, c10::nullopt, VK_KERNEL(clamp_))));  // 否则调用通常的 inplace 版本的 ops::_clamp_ 函数
}

Tensor hardswish(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardswish));  // 调用 ops::activation 函数，使用硬切线函数进行激活
}

Tensor& hardswish_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardswish_));  // 调用 inplace 版本的 ops::activation_ 函数，使用硬切线函数进行激活
}

Tensor hardsigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardsigmoid));  // 调用 ops::activation 函数，使用硬 sigmoid 函数进行激活
}

Tensor& hardsigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardsigmoid_));  // 调用 inplace 版本的 ops::activation_ 函数，使用硬 sigmoid 函数进行激活
}

Tensor activation_scalar(
    const Tensor& self_arg,
    const std::vector<Scalar>& scalar_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();  // 获取当前的 Vulkan 环境上下文

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();  // 如果输入张量是 Vulkan 张量，则直接使用，否则转换为 Vulkan 张量
  const vTensor& v_self = convert(self);  // 将张量转换为 Vulkan 张量

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };  // 创建一个 Vulkan 张量 v_output，与输入张量的大小和数据类型相同

  api::UniformParamsBuffer params;  // 创建一个用于存储 uniform 参数的缓冲区

  if (v_self.is_quantized()) {  // 如果输入张量是量化的
    v_output.set_is_quantized();  // 设置输出 Vulkan 张量为量化类型
    v_output.set_scale(v_self.get_scale());  // 设置输出 Vulkan 张量的量化比例
    v_output.set_zero_point(v_self.get_zero_point());  // 设置输出 Vulkan 张量的量化零点
  }

  if (scalar_arg.size() == 1) {  // 如果标量参数数组大小为1
    if (v_self.is_quantized()) {  // 如果输入张量是量化的
      const struct Block final {  // 定义一个结构体 Block，用于存储 uniform 参数
        uvec3 extents;  // 张量的尺寸
        uint32_t _;  // 保留字段
        float scalar_value;  // 标量值
        float scale;  // 量化比例
        int zero_point;  // 量化零点
      } block{  // 初始化 Block 结构体
          v_output.extents(),
          0u,
          scalar_arg[0].to<float>(),  // 将第一个标量参数转换为 float 类型
          safe_downcast<float>(v_self.get_scale()),  // 获取输入张量的量化比例并安全地转换为 float
          safe_downcast<int32_t>(v_self.get_zero_point()),  // 获取输入张量的量化零点并安全地转换为 int32_t
      };
      params = api::UniformParamsBuffer(context, block);  // 使用 Block 结构体创建 uniform 参数缓冲区
    } else {
      const struct Block final {  // 定义一个结构体 Block，用于存储 uniform 参数
        uvec3 extents;  // 张量的尺寸
        uint32_t _;  // 保留字段
        float scalar_value;  // 标量值
      } block{  // 初始化 Block 结构体
          v_output.extents(),
          0u,
          scalar_arg[0].to<float>(),  // 将第一个标量参数转换为 float 类型
      };
      params = api::UniformParamsBuffer(context, block);  // 使用 Block 结构体创建 uniform 参数缓冲区
    }
  } else {  // 如果标量参数数组大小不为1
    const struct Block final {  // 定义一个结构体 Block，用于存储 uniform 参数
      uvec3 extents;  // 张量的尺寸
      uint32_t _;  // 保留字段
      float scalar_value1;  // 第一个标量值
      float scalar_value2;  // 第二个标量值
    } block{  // 初始化 Block 结构体
        v_output.extents(),
        0u,
        scalar_arg[0].to<float>(),  // 将第一个标量参数转换为 float 类型
        scalar_arg[1].to<float>(),  // 将第二个标量参数转换为 float 类型
    };
    // 使用 api 命名空间中的 UniformParamsBuffer 类创建一个名为 params 的对象，
    // 该对象基于给定的 context 和 block 进行初始化
    params = api::UniformParamsBuffer(context, block);
  }

  // 创建一个名为 pipeline_barrier 的 PipelineBarrier 对象，用于控制流水线的执行顺序和内存访问
  api::PipelineBarrier pipeline_barrier{};

  // 在 context 上提交计算作业
  context->submit_compute_job(
      // 使用 shader_descriptor 作为着色器描述符
      shader_descriptor,
      // 使用 pipeline_barrier 作为流水线屏障
      pipeline_barrier,
      // 设置全局工作组大小为 v_output 的尺寸
      v_output.extents(),
      // 使用自适应工作组大小函数计算局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 使用 VK_NULL_HANDLE 表示不需要 fence handle
      VK_NULL_HANDLE,
      // 设置 shader arguments，写入 v_output 图像
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // 设置 shader arguments，读取 v_self 图像
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 设置 params 对象的缓冲区作为 shader arguments
      params.buffer());

  // 返回将 v_output 转换后的结果
  return convert(v_output);
}

Tensor& activation_scalar_(
    Tensor& self_arg,
    const std::vector<Scalar>& scalar_arg,
    const api::ShaderInfo& shader_descriptor) {
  // 检查是否为 Vulkan 张量，否则抛出错误
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  // 获取当前的 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入的 PyTorch 张量转换为 Vulkan 张量
  vTensor& v_self = convert(self_arg);

  // 创建 UniformParamsBuffer 对象，用于存储 uniform 参数
  api::UniformParamsBuffer params;

  // 根据标量参数的数量选择不同的代码路径
  if (scalar_arg.size() == 1) {
    if (v_self.is_quantized()) {
      // 定义并初始化用于量化张量的 uniform 参数块
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
        float scale;
        int zero_point;
      } block{
          v_self.extents(),
          0u,
          scalar_arg[0].to<float>(),
          safe_downcast<float>(v_self.get_scale()),
          safe_downcast<int32_t>(v_self.get_zero_point()),
      };
      // 创建 UniformParamsBuffer 对象
      params = api::UniformParamsBuffer(context, block);
    } else {
      // 定义并初始化非量化张量的 uniform 参数块
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block{
          v_self.extents(),
          0u,
          scalar_arg[0].to<float>(),
      };
      // 创建 UniformParamsBuffer 对象
      params = api::UniformParamsBuffer(context, block);
    }
  } else {
    // 定义并初始化包含两个标量值的 uniform 参数块
    const struct Block final {
      uvec3 extents;
      uint32_t _;
      float scalar_value1;
      float scalar_value2;
    } block{
        v_self.extents(),
        0u,
        scalar_arg[0].to<float>(),
        scalar_arg[1].to<float>(),
    };
    // 创建 UniformParamsBuffer 对象
    params = api::UniformParamsBuffer(context, block);
  }

  // 创建空的 PipelineBarrier 对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算任务到 Vulkan 上下文
  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  // 返回原始的自身张量
  return self_arg;
}

Tensor gelu(const Tensor& self, c10::string_view approximate) {
  // 检查近似类型是否为 "tanh"，否则抛出错误
  TORCH_CHECK(
      approximate == "tanh", "Vulkan: gelu only supported for tanh type");
  
  // 计算常量 kBetaVec 的值
  Scalar kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5;
  // 创建标量向量，并将 kBetaVec 添加到向量中
  std::vector<Scalar> scalar;
  scalar.push_back(kBetaVec);

  // 根据输入张量的数据类型选择不同的 Vulkan 核心函数并调用 activation_scalar
  if (self.scalar_type() == at::kQUInt8) {
    return ops::activation_scalar(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_quint8));
  }

  if (self.scalar_type() == at::kQInt8) {
    return ops::activation_scalar(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_qint8));
  }

  // 默认情况下调用通用的 gelu_tanh Vulkan 核心函数
  return ops::activation_scalar(self, scalar, VK_KERNEL(gelu_tanh));
}
Tensor& gelu_(Tensor& self, c10::string_view approximate) {
  // 检查是否使用了正确的近似方法
  TORCH_CHECK(
      approximate == "tanh", "Vulkan: gelu only supported for tanh type");
  // 计算常量 kBetaVec
  Scalar kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5;
  // 创建 Scalar 向量并添加 kBetaVec
  std::vector<Scalar> scalar;
  scalar.push_back(kBetaVec);

  // 如果张量是 kQUInt8 类型，则调用对应的 Vulkan 操作函数
  if (self.scalar_type() == at::kQUInt8) {
    return ops::activation_scalar_(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_quint8_));
  }

  // 如果张量是 kQInt8 类型，则调用对应的 Vulkan 操作函数
  if (self.scalar_type() == at::kQInt8) {
    return ops::activation_scalar_(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_qint8_));
  }

  // 否则调用默认的 Vulkan gelu_tanh_ 操作函数
  return ops::activation_scalar_(self, scalar, VK_KERNEL(gelu_tanh_));
}

Tensor hardshrink(const Tensor& self_arg, const Scalar& lambd) {
  // 计算 lambd 的绝对值
  float abs_lambd = std::abs(lambd.to<float>());
  // 创建 Scalar 向量并添加 abs_lambd
  std::vector<Scalar> scalar;
  scalar.push_back(abs_lambd);
  // 调用 Vulkan 操作函数执行 hardshrink 激活函数
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(hardshrink));
}

Tensor& hardshrink_(Tensor& self, const Scalar& lambd) {
  // 计算 lambd 的绝对值
  float abs_lambd = std::abs(lambd.to<float>());
  // 创建 Scalar 向量并添加 abs_lambd
  std::vector<Scalar> scalar;
  scalar.push_back(abs_lambd);
  // 原地执行 Vulkan 操作函数，应用 hardshrink 激活函数
  return ops::activation_scalar_(self, scalar, VK_KERNEL(hardshrink_));
}

Tensor leaky_relu(const Tensor& self_arg, const Scalar& negative_slope) {
  // 创建 Scalar 向量并添加 negative_slope
  std::vector<Scalar> scalar;
  scalar.push_back(negative_slope);
  // 调用 Vulkan 操作函数执行 leaky_relu 激活函数
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(leaky_relu));
}

Tensor& leaky_relu_(Tensor& self, const Scalar& negative_slope) {
  // 创建 Scalar 向量并添加 negative_slope
  std::vector<Scalar> scalar;
  scalar.push_back(negative_slope);
  // 原地执行 Vulkan 操作函数，应用 leaky_relu 激活函数
  return ops::activation_scalar_(self, scalar, VK_KERNEL(leaky_relu_));
}

Tensor sigmoid(const Tensor& self) {
  // 调用 Vulkan 操作函数执行 sigmoid 激活函数
  return ops::activation(self, VK_KERNEL(sigmoid));
}

Tensor& sigmoid_(Tensor& self) {
  // 原地执行 Vulkan 操作函数，应用 sigmoid 激活函数
  return ops::activation_(self, VK_KERNEL(sigmoid_));
}

Tensor tanh(const Tensor& self) {
  // 调用 Vulkan 操作函数执行 tanh 激活函数
  return ops::activation(self, VK_KERNEL(tanh));
}

Tensor& tanh_(Tensor& self) {
  // 原地执行 Vulkan 操作函数，应用 tanh 激活函数
  return ops::activation_(self, VK_KERNEL(tanh_));
}

Tensor abs(const Tensor& self) {
  // 调用 Vulkan 操作函数执行 abs 激活函数
  return ops::activation(self, VK_KERNEL(abs));
}

Tensor& abs_(Tensor& self) {
  // 原地执行 Vulkan 操作函数，应用 abs 激活函数
  return ops::activation_(self, VK_KERNEL(abs_));
}

#ifdef USE_VULKAN_API
// 定义 TORCH_LIBRARY_IMPL 宏，将 aten 操作注册到 Vulkan 后端
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
    // 注册 aten::clamp 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::clamp"), TORCH_FN(clamp));
    // 注册 aten::clamp_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::clamp_"), TORCH_FN(clamp_));
    // 注册 aten::gelu 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::gelu"), gelu);
    // 注册 aten::gelu_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::gelu_"), gelu_);
    // 注册 aten::hardsigmoid 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid"), hardsigmoid);
    // 注册 aten::hardsigmoid_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid_"), hardsigmoid_);
    // 注册 aten::hardshrink 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink"), hardshrink);
    // 注册 aten::hardshrink_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink_"), hardshrink_);
    // 注册 aten::hardswish 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardswish"), hardswish);
    // 注册 aten::hardswish_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardswish_"), hardswish_);
    // 注册 aten::hardtanh 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh"), hardtanh);
    // 注册 aten::hardtanh_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh_"), hardtanh_);
    // 注册 aten::leaky_relu 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu"), leaky_relu);
    // 注册 aten::leaky_relu_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu_"), leaky_relu_);
    // 注册 aten::sigmoid 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid"), sigmoid);
    // 注册 aten::sigmoid_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid_"), sigmoid_);
    // 注册 aten::tanh 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::tanh"), tanh);
    // 注册 aten::tanh_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::tanh_"), tanh_);
    // 注册 aten::abs 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::abs"), abs);
    // 注册 aten::abs_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::abs_"), abs_);
    // 注册 aten::relu 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::relu"), relu);
    // 注册 aten::relu_ 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::relu_"), relu_);
    // 注册 aten::threshold 操作的 Vulkan 实现
    m.impl(TORCH_SELECTIVE_NAME("aten::threshold"), threshold);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```