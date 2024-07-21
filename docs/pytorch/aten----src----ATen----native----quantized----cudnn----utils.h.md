# `.\pytorch\aten\src\ATen\native\quantized\cudnn\utils.h`

```
#pragma once
/*
This file contains some of the auxiliary functions used by both Conv.cpp & Linear.cpp (introduced in a later PR)
*/

#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/cudnn/Types.h>
#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <c10/core/QScheme.h>
#include <c10/util/ArrayRef.h>

// Disable warnings related to suggesting overrides
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

// Conditionally include headers based on whether AT_PER_OPERATOR_HEADERS is defined
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

// Definition of a structure inheriting LinearPackedParamsBase for packed linear weights using CuDNN
struct PackedLinearWeightCudnn : public LinearPackedParamsBase {
  // Constructor for initializing packed linear weights using CuDNN
  PackedLinearWeightCudnn(
      at::Tensor orig_weight,
      std::optional<at::Tensor> bias,
      c10::QScheme q_scheme)
      : orig_weight(std::move(orig_weight)),
        bias_(std::move(bias)),
        q_scheme(std::move(q_scheme)) {}

  // Override method to apply packed linear weights with optional ReLU fusion
  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  // Override method to apply packed linear weights with ReLU fusion
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  // Not implemented override method for applying dynamically
  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false) override {
    throw std::runtime_error(
    "apply_dynamic is not implemented for this packed "
    "parameter type");
  }

  // Not implemented override method for applying dynamically with ReLU fusion
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false) override {
    throw std::runtime_error(
    "apply_dynamic_relu is not implemented for this packed "
    "parameter type");
  }

  // Method to unpack packed parameters and retrieve bias
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // Override method to retrieve bias
  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  // Static method to prepack linear weights
  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

 private:
  at::Tensor orig_weight;                   // Original weight tensor
  std::optional<at::Tensor> bias_;          // Optional bias tensor
  c10::QScheme q_scheme;                    // Quantization scheme

  // Template method for applying packed linear weights with or without ReLU fusion
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  // Helper method for applying packed linear weights with or without ReLU fusion
  template <bool ReluFused>
  void apply_impl_helper(
      const at::Tensor& quantized_output,
      const at::Tensor& input,
      double output_scale);
};

// Template specialization for kSpatialDim = 2
template <int kSpatialDim = 2>
// 继承自 ConvPackedParamsBase<kSpatialDim>，表示封装的 CUDNN 卷积权重参数结构体
struct PackedConvWeightCudnn : public ConvPackedParamsBase<kSpatialDim> {
  // 构造函数，初始化 CUDNN 卷积权重参数
  PackedConvWeightCudnn(
      at::Tensor orig_weight,                             // 原始权重张量
      std::optional<at::Tensor> bias,                     // 可选的偏置张量
      torch::List<int64_t> stride,                        // 步长列表
      torch::List<int64_t> padding,                       // 填充列表
      torch::List<int64_t> output_padding,                // 输出填充列表
      torch::List<int64_t> dilation,                      // 膨胀列表
      int64_t groups,                                     // 分组数量
      bool transpose,                                     // 是否转置
      c10::QScheme q_scheme,                              // 量化方案
      int64_t output_channels)                            // 输出通道数
      : maybe_padded_weight_(std::move(orig_weight)),     // 可能填充的权重张量
        bias_(std::move(bias)),                           // 偏置张量
        stride_(std::move(stride)),                       // 步长列表
        padding_(std::move(padding)),                     // 填充列表
        output_padding_(std::move(output_padding)),       // 输出填充列表
        dilation_(std::move(dilation)),                   // 膨胀列表
        groups_(groups),                                  // 分组数量
        transpose_(transpose),                            // 是否转置
        q_scheme_(q_scheme),                              // 量化方案
        num_unpadded_output_channels_(output_channels) {} // 未填充的输出通道数需要存储

  // 应用卷积操作到输入张量，返回输出张量
  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用带 ReLU 激活函数的卷积操作到输入张量，返回输出张量
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用动态量化卷积操作到输入张量，目前未实现，抛出错误信息
  at::Tensor apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) override {
    TORCH_CHECK(false, "apply_dynamic is currently not reported");
  }

  // 应用带 ReLU 激活函数的动态量化卷积操作到输入张量，目前未实现，抛出错误信息
  at::Tensor apply_dynamic_relu(
    const at::Tensor& input,
    bool reduce_range) {
    TORCH_CHECK(false, "apply_dynamic_relu is currently not reported");
  }

  // 解压当前封装的卷积参数，返回权重张量和可选的偏置张量
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // 静态方法，预封装卷积参数，返回 ConvPackedParamsBase<kSpatialDim> 指针
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  // 获取偏置数据的指针，返回常量浮点数指针
  const float* GetBiasData(at::Tensor* bias);

  // 返回步长列表
  torch::List<int64_t> stride() const override {
    return stride_;
  }

  // 返回填充列表
  torch::List<int64_t> padding() const override {
    return padding_;
  }

  // 返回输出填充列表
  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  // 返回膨胀列表
  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  // 返回分组数量
  int64_t groups() const override {
    return groups_;
  }

  // 返回是否转置
  bool transpose() const override {
  // 返回 transpose_ 成员变量
  return transpose_;
}

private:
  // cudnn v8.4.0 要求 int8 类型的卷积权重张量的输入和输出通道数必须是4的倍数。
  // 如果不是，我们需要显式地对其进行填充，使其成为4的倍数，因为 cudnn 目前不支持填充，因此命名为 "maybe"_padded_weight。
  // TODO: 如果 cudnn 在其操作符中启用填充，我们可以在我们这边移除填充，并将这个变量重命名为 orig_weight_
  at::Tensor maybe_padded_weight_;
  std::optional<at::Tensor> bias_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  c10::QScheme q_scheme_;
  int64_t num_unpadded_output_channels_;

  template <bool ReluFused>
  // 应用实现函数模板，用于执行卷积操作
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  // 辅助函数模板，用于执行与 ReLU 合并的卷积操作
  void apply_impl_helper(
      const at::Tensor& quantized_output,
      const at::Tensor& input,
      double output_scale);
};

namespace cudnn_utils {
namespace {

// 当cuDNN启用对逐点乘法操作的按值传递支持后，可以移除此函数。
// 目前我们需要此函数的唯一原因是，在卷积、线性和加法操作中使用广播标量乘法，
// 而cuDNN要求标量是一个与我们要乘到的张量具有相同维数（num_dim）的标量张量。
at::Tensor getRequantMultiplierTensor(double requant_multiplier, uint8_t num_dim) {
  // 创建一个大小为num_dim的小型整数向量，每个元素初始化为1
  at::SmallVector<int64_t, 4> requantize_multiplier_tensor_size(num_dim, 1);
  // 创建一个空张量，大小为requantize_multiplier_tensor_size，位于CUDA设备上，数据类型为float
  at::Tensor requantize_multiplier_tensor = at::empty(requantize_multiplier_tensor_size, at::device(at::kCUDA).dtype(at::kFloat));
  // 将张量填充为requant_multiplier
  requantize_multiplier_tensor.fill_(requant_multiplier);
  // 返回填充后的张量
  return requantize_multiplier_tensor;
}

// 获取张量t的对齐方式（以字节为单位）
uint8_t getAlignment(const at::Tensor &t) {
  // 初始对齐度为1字节
  uint8_t alignment = 1;
  // 获取张量数据指针的地址
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  // 逐渐增加对齐度，直到找到第一个不能被整除的地址
  for (; alignment < 16; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  // 如果对齐度大于等于16或未找到可用的对齐度，则返回当前对齐度
  return alignment;
}

// 获取张量描述符，支持虚拟张量（默认为非虚拟）
cudnn_frontend::Tensor getTensorDescriptor(const at::Tensor &t, int64_t id, uint8_t alignment, bool is_virtual = false) {
  // 获取张量形状和步长
  auto shape = t.sizes();
  auto strides = t.strides();
  // 如果设置为虚拟张量，使用cudnn_frontend::TensorBuilder创建虚拟张量描述符
  if (is_virtual) {
    return cudnn_frontend::TensorBuilder()
      .setDim(shape.size(), shape.data())
      .setStrides(strides.size(), strides.data())
      .setId(id)
      .setAlignment(alignment)
      .setVirtual()
      .setDataType(at::native::getCudnnDataType(t))
      .build();
  }
  // 否则，创建非虚拟张量描述符
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(at::native::getCudnnDataType(t))
    .build();
}

// 获取张量描述符，支持虚拟张量（默认为非虚拟）
cudnn_frontend::Tensor getTensorDescriptor(const c10::IntArrayRef& shape, const c10::IntArrayRef& strides, cudnnDataType_t cudnn_dtype, int64_t id, uint8_t alignment, bool is_virtual = false) {
  // 如果设置为虚拟张量，使用cudnn_frontend::TensorBuilder创建虚拟张量描述符
  if (is_virtual) {
    return cudnn_frontend::TensorBuilder()
      .setDim(shape.size(), shape.data())
      .setStrides(strides.size(), strides.data())
      .setId(id)
      .setAlignment(alignment)
      .setVirtual()
      .setDataType(cudnn_dtype)
      .build();
  }
  // 否则，创建非虚拟张量描述符
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(cudnn_dtype)
    .build();
}
// 根据输入数据类型获取点乘操作的描述符
cudnn_frontend::PointWiseDesc_v8 getPointWiseMulDescriptor(cudnnDataType_t dataType) {
  // 使用 PointWiseDescBuilder 创建点乘描述符，设置点乘模式为 CUDNN_POINTWISE_MUL，并指定数学精度为输入的数据类型
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_MUL)
    .setMathPrecision(dataType)
    .build();
}

// 根据输入数据类型获取点加操作的描述符
cudnn_frontend::PointWiseDesc_v8 getPointWiseAddDescriptor(cudnnDataType_t dataType) {
  // 使用 PointWiseDescBuilder 创建点加描述符，设置点加模式为 CUDNN_POINTWISE_ADD，并指定数学精度为输入的数据类型
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD)
    .setMathPrecision(dataType)
    .build();
}

// 根据输入数据类型获取ReLU点乘操作的描述符
cudnn_frontend::PointWiseDesc_v8 getPointWiseReluDescriptor(cudnnDataType_t dataType) {
  // 使用 PointWiseDescBuilder 创建ReLU点乘描述符，设置ReLU点乘模式为 CUDNN_POINTWISE_RELU_FWD，并指定数学精度为输入的数据类型
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_FWD)
    .setMathPrecision(dataType)
    .build();
}

// 根据特定条件过滤引擎配置列表
void filterEngineConfigs(
  cudnn_frontend::EngineConfigList &from,
  cudnn_frontend::EngineConfigList &to,
  bool deterministic, bool allow_tf32, c10::ScalarType scalar_type)
{
  // 定义一个lambda函数filter，根据给定的条件筛选引擎描述符
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      // 如果需要确定性，检查引擎描述符是否有CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC数值注释
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) return true;
    }
    // 根据标量类型和允许的TF32标志，检查引擎描述符是否符合条件
    if (scalar_type == at::kFloat || scalar_type == at::kChar || !allow_tf32) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) return true;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
    }
    return false;
  };
  // 使用filter函数从源引擎配置列表from筛选出符合条件的配置，并存入目标引擎配置列表to
  cudnn_frontend::filter(from, to, filter);
}

// 从启发式策略获取执行计划，否则返回默认的执行计划
cudnn_frontend::ExecutionPlan get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
  // 使用 EngineHeuristicsBuilder 创建启发式引擎，设置操作图和启发模式为即时模式
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
    .setOperationGraph(opGraph)
    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
    .build();

  // 获取启发式引擎配置的数量
  auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

  // 尝试使用启发式引擎配置中的引擎配置，并选择第一个可用的执行计划
  for (auto& ecfg : engine_config) {
    try {
      // 使用 ExecutionPlanBuilder 创建执行计划，设置句柄、引擎配置和操作图标签
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle_)
        .setEngineConfig(ecfg, opGraph.getTag())
        .build();
      return plan;
    } catch (cudnn_frontend::cudnnException& e) {
      continue;
    }
  }

  {
    // 如果无法找到可用的执行计划，创建一个默认引擎并返回
    auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
  }
}
    // 使用 cudnn_frontend 库的 EngineConfigBuilder 创建引擎配置对象，设置引擎参数为 engine
    auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
    // 创建一个引擎配置对象后，如果需要，可以打印其描述信息
    // std::cout << engine_config.describe() << std::endl;

    // 使用 cudnn_frontend 库的 ExecutionPlanBuilder 创建执行计划构建器对象，
    // 设置计划的句柄为 handle_，引擎配置为刚刚创建的 engine_config
    return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();
  }
}
} // anonymous
} // cudnn_utils

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
```