# `.\pytorch\aten\src\ATen\native\quantized\cudnn\Linear.cpp`

```py
#ifdef USE_CUDA
// 如果定义了 USE_CUDA 宏，则包含 AT_CUDNN_ENABLED 宏的定义
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()
// 如果 AT_CUDNN_ENABLED 宏为真，则包含以下头文件

#include <c10/util/ArrayRef.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/TensorUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cudnn_frontend.h>
#include <torch/library.h>

#include <iostream>
#include <unordered_map>

// 声明一个函数 register_linear_params
int register_linear_params();

// TODO: there is a table from input dtype and weight dtype to operator dtype,
// we can derive the operator dtype based on input dtype

// 定义一个函数，根据输入数据类型获取线性层描述符
cudnn_frontend::MatMulDesc_v8 getLinearDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::MatMulDescBuilder()
    .setMathPrecision(dataType)
    .build();
}

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
namespace {
// 匿名命名空间，定义线性参数结构体和相关常量

// 我们当前将最大输入维度数设置为5，如有必要可以增加
constexpr uint8_t max_num_input_dim = 5;
struct LinearParams {
  c10::DeviceIndex device_id;        // 设备 ID
  cudnnDataType_t dataType;          // 数据类型
  int input_size[max_num_input_dim]; // 输入大小数组
  uint8_t input_dim;                 // 输入维度
  at::MemoryFormat memory_format;    // 内存格式
  int64_t weight_size[2];            // 权重大小数组
  bool deterministic;                // 是否确定性计算
  bool allow_tf32;                   // 是否允许 TF32 计算
};

struct CacheKey {
  LinearParams params;               // 线性参数
  uint8_t input_alignment;           // 输入对齐
  uint8_t weight_alignment;          // 权重对齐
  uint8_t output_alignment;          // 输出对齐
  // default to -1 when no bias
  int8_t bias_alignment;             // 偏置对齐，默认为 -1（无偏置）
};

// 设置线性参数函数，填充 LinearParams 结构体
void setLinearParams(
    LinearParams* params, const at::Tensor& input, const at::Tensor& weight,
    bool deterministic, bool allow_tf32) {
  // 操作数据类型需要对于 int8 矩阵乘法是 int32，但可以将输出张量的数据类型设置为 int32 或 fp32
  memset(params, 0, sizeof(LinearParams));  // 初始化参数结构体
  params->device_id = at::cuda::current_device();  // 当前设备 ID
  params->dataType = CUDNN_DATA_INT32;     // 数据类型设为 int32
  params->input_dim = input.dim();         // 输入张量的维度
  params->memory_format = input.suggest_memory_format();  // 建议的内存格式
  for (int i = 0; i < params->input_dim; ++i) {
    params->input_size[i] = input.sizes()[i];  // 填充输入大小数组
  }
  for (int i = 0; i < 2; ++i) {
    params->weight_size[i] = weight.sizes()[i];  // 填充权重大小数组
  }
  params->deterministic = deterministic;    // 确定性计算
  params->allow_tf32 = allow_tf32;          // 是否允许 TF32 计算
}

// 声明一个无序映射表，用于缓存执行计划
std::unordered_map<CacheKey, cudnn_frontend::ExecutionPlan, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;
}

// TODO: we can use cudnn_frontend::ExecutionPlanCache when it supports caching
// multiple operators
// reference: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/conv_sample.cpp#L293
//static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

// currently we only support int8 symmetric (zero_point = 0 for inputs and output) quantized linear op
// 我们实现 relu(act_int8 * transpose(w_int8) + [bias_fp32/(act_scale * w_scale)] * (act_scale * w_scale / out_scale)
// 这需要 5 个 cudnn 操作（1 个矩阵乘法，2 个乘法，1 个加法和 1 个 relu 操作）
// matmul 操作：linear_op
// 乘法操作：rhs_mult_op, requant_op
// 加法操作：add_op
// relu 操作：relu_op
template <bool kReluFused>
void PackedLinearWeightCudnn::apply_impl_helper(const at::Tensor& quantized_output, const at::Tensor& input, double output_scale) {
  if (quantized_output.numel() == 0) {
    return;
  }
  auto act_scale = input.q_scale();  // 获取输入张量的量化比例因子
  auto weight_scale = orig_weight.q_scale();  // 获取原始权重的量化比例因子
  auto requantize_multiplier = act_scale * weight_scale / output_scale;  // 计算重新量化的乘数
  at::Tensor requantize_multiplier_tensor = cudnn_utils::getRequantMultiplierTensor(requantize_multiplier, quantized_output.dim());  // 获取重新量化乘数的张量表示
  std::optional<at::Tensor> bias_multiplier_tensor;
  std::optional<at::Tensor> broadcasted_bias;
  if (bias_.has_value()) {
    // 输入的偏置是一个 1-D 张量，其大小与 quantized_output 的最后一个维度大小相同
    // 我们需要添加尾部维度以正确广播偏置，否则 broadcast_to 将失败。
    // 尾部维度的数量是 quantized_output.dim() - 2。我们还在开头添加一个维度以增强清晰度
    std::vector<int64_t> new_size(quantized_output.dim(), 1);
    new_size.back() = bias_.value().size(0);
    broadcasted_bias = bias_.value().clone().reshape(new_size);  // 克隆并重塑偏置张量
    broadcasted_bias.value() = broadcasted_bias.value().broadcast_to(quantized_output.sizes()).contiguous();  // 广播到 quantized_output 的尺寸并保持连续性
    bias_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat));  // 创建一个在 CUDA 设备上的空张量，数据类型为 float
    auto bias_multiplier = 1.0 / (act_scale * weight_scale);
    bias_multiplier_tensor.value().fill_(bias_multiplier);  // 填充张量为 bias_multiplier
  }

  cudnnHandle_t handle = at::native::getCudnnHandle();  // 获取 cudnn 句柄
  CacheKey key;
  // 这里需要 memset 是因为 CacheKey 隐式添加了填充，这可能导致未初始化的填充值
  // 在哈希时使用（参见 at::native::ParamsHash 的定义）。没有 memset，可能会出现这样的情况，
  // 两个 CacheKey 对象具有相同的用户定义参数，但填充值不同，导致不同的哈希输出。
  memset(&key, 0, sizeof(key));  // 将 key 内存清零
  bool deterministic{true};
  bool allow_tf32{false};
  setLinearParams(&key.params, input, orig_weight, deterministic, allow_tf32);  // 设置线性参数到 key.params 中

  key.input_alignment = cudnn_utils::getAlignment(input);  // 获取输入张量的对齐方式
  key.output_alignment = cudnn_utils::getAlignment(quantized_output);  // 获取输出量化张量的对齐方式
  key.weight_alignment = cudnn_utils::getAlignment(orig_weight);  // 获取权重张量的对齐方式
  if (bias_.has_value()) {
    key.bias_alignment = cudnn_utils::getAlignment(broadcasted_bias.value());  // 获取偏置张量的对齐方式
  } else {
    // 如果没有偏置，则将 key.bias_alignment 置为默认值 0
    key.bias_alignment = 0;
  }
}
    key.bias_alignment = -1;
  }
  key.kReluFused = kReluFused;
  // 设置偏置对齐为-1

  // 矩阵乘法操作是 input * transpose(weight)，因此我们将使用转置后的权重
  auto weight_transposed = transpose(orig_weight, 0, 1);
  // cudnn 需要张量至少是3维的。weight_transposed 目前是2维的。我们将创建一个3维视图
  // 通过添加一个前导虚拟维度（cudnn 期望前导维度是虚拟的维度）
  std::vector<int64_t> new_sizes(3, 1);
  new_sizes.back() = weight_transposed.size(1);
  new_sizes[1] = weight_transposed.size(0);
  weight_transposed = weight_transposed.view(new_sizes);
  // 创建新的3维视图以适应 cudnn 的要求

  auto run = [&](const cudnn_frontend::ExecutionPlan& plan_desc) {
    auto workspace_size = plan_desc.getWorkspaceSize();
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    // 分配 cudnn 执行所需的工作空间
    at::SmallVector<void *, 8> data_ptrs;
    at::SmallVector<int64_t, 8> uids;
    data_ptrs = {input.data_ptr<int8_t>(), weight_transposed.data_ptr<int8_t>(),
                 requantize_multiplier_tensor.data_ptr(), quantized_output.data_ptr<int8_t>()};
    uids = {'x', 'w', 's', 'r'};
    if (bias_.has_value()) {
      data_ptrs.insert(data_ptrs.end(), {broadcasted_bias.value().data_ptr(), bias_multiplier_tensor.value().data_ptr(),
                                         broadcasted_bias.value().data_ptr(), broadcasted_bias.value().data_ptr()});
      uids.insert(uids.end(), {'b', 'c', 'd', 'n'});
    }
    // 设置数据指针和唯一标识符以构建 cudnn 的 VariantPack

    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
      .setDataPointers(uids.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    // 构建 VariantPack 对象

    auto variant_pack_desc = variantPack.get_raw_desc();
    // 获取 VariantPack 的原始描述

    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc.get_raw_desc(), variant_pack_desc));
    // 使用 cudnn 执行后端来执行计划描述和 VariantPack 描述
  };

  auto search = execution_plan_cache.find(key);
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ExecutionPlan plan_desc = search->second;
    run(plan_desc);
    // 如果找到缓存中的执行计划，执行该计划
    return;
  }

  // linear_op computes act_int8 * tranpose(w_int8) (matrix multiplication)
  // where act_int8 and w_int8 are the input and weight variables, resp.
  // output is a fp32 tensor
  auto linear_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
      .setaMatDesc(cudnn_utils::getTensorDescriptor(input.sizes(), input.strides(), CUDNN_DATA_INT8, 'x', key.input_alignment))
      .setbMatDesc(cudnn_utils::getTensorDescriptor(weight_transposed.sizes(), weight_transposed.strides(), CUDNN_DATA_INT8, 'w', key.weight_alignment))
      // for virtual tensors, the alignment is not used, so we can just put an arbitrary value here, e.g., key.output_alignment
      .setcMatDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'y', key.output_alignment, true))
      .setmatmulDesc(getLinearDescriptor(key.params.dataType))
      .build();
  // std::cout << "operator:" << linear_op.describe() << std::endl;

  std::optional<cudnn_frontend::Operation> bias_mult_op;
  std::optional<cudnn_frontend::Operation> sum_linear_bias_op;
  if (bias_.has_value()) {
    // we can't directly assign bias_mult_op because operator= is deleted for cudnn_frontend::Operation;
    // alternatively, I think we can use std::unique_ptr and dynamically allocate these builder ops
    // but here, we chose to do it statically. std::optional<T>::emplace() enables this approach

    // bias_mult_op computes bias_fp32 / (act_scale * w_scale) or bias_fp32 * (1 / (act_scale * w_scale))
    // where bias_multiplier = (1 / (act_scale * w_scale))
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    bias_mult_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'b', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setbDesc(cudnn_utils::getTensorDescriptor(bias_multiplier_tensor.value(), 'c', cudnn_utils::getAlignment(bias_multiplier_tensor.value())))
      // TODO: I think we should be able to make this a virtual tensor, but we would need cudnn to support
      // setbdesc(ManagedOpaqueDescriptor const &raw_tensor) first
      .setyDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(bias_multiplier_tensor.value())))
      .build());

    // computes (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)])
    // where the 1st and 2nd summands is output of linear op and broadcasted_bias, resp.
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    sum_linear_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_ADD_DESCRIPTOR)
      .setaDesc(cudnn_utils::getTensorDescriptor(linear_op.getOutputTensor()))
      .setbDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'b', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setcDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'y', key.output_alignment, true))
      .setaddDesc(getAddDescriptor(key.params.dataType))
      .build());
    sum_linear_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(linear_op.getOutputTensor())
      // TODO: 在当前版本的 cudnn（8.4.0）中，需要为广播偏置项（broadcasted_bias）在 uid-data_ptr 对中增加一个额外的条目。
      // 如果不增加，某些测试用例会失败。NVIDIA 目前正在调查此问题。
      // 当此问题解决后，我们可以将 'n' 改回 'd'，并在上述的 variant pack 中移除额外的条目。
      .setbDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'n', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'e', key.output_alignment, true))
      .setpwDesc(cudnn_utils::getPointWiseAddDescriptor(at::native::getCudnnDataType(broadcasted_bias.value())))
      .build());
  }

  // relu_op 计算 relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)] 或 relu(act_int8 * w_int8)（如果没有偏置）。
  // 输出是一个 fp32 张量
  std::optional<cudnn_frontend::Operation> relu_op;
  std::shared_ptr<cudnn_frontend::OpaqueBackendPointer> tensor2requant_ptr = bias_.has_value() ? sum_linear_bias_op.value().getOutputTensor() : linear_op.getOutputTensor();
  if (kReluFused) {
    // 在这里使用原位操作，输出被分配给输入
    relu_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(tensor2requant_ptr)
      // 对于虚拟张量，对齐方式不被使用，因此可以在这里放置一个任意值，例如，key.output_alignment
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'f', key.output_alignment, true))
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(CUDNN_DATA_FLOAT))
      .build());
  }

  // requant_op 计算 relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
  // 或 relu(act_int8 * w_int8) / (out_scale / (act_scale * w_scale)))（如果没有偏置）。
  // 输出是一个 fp32 张量
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : tensor2requant_ptr)
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 's', cudnn_utils::getAlignment(requantize_multiplier_tensor)))
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'r', key.output_alignment))
    .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(requantize_multiplier_tensor)))
    .build();
  // // std::cout << "operator:" << requant_op.describe() << std::endl;

  std::vector<cudnn_frontend::Operation const *> ops{&linear_op};
  if (bias_.has_value()) {


这段代码是关于使用 cudnn 前端库进行张量操作描述的示例。
    // 将偏置乘法操作的指针添加到操作列表
    ops.emplace_back(&(bias_mult_op.value()));
    // 将线性偏置求和操作的指针添加到操作列表
    ops.emplace_back(&(sum_linear_bias_op.value()));

  // 如果启用了融合ReLU操作
  if (kReluFused) {
    // 将ReLU操作的指针添加到操作列表
    ops.emplace_back(&(relu_op.value()));
  }

  // 将重新量化操作的指针添加到操作列表
  ops.emplace_back(&requant_op);

  // 构建操作图
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(ops.size(), ops.data())
      .build();
  // 创建操作图后的一些后续处理，例如打印操作图的描述信息（注释掉的代码行）
  // std::cout << "opGraph: " << opGraph.describe() << std::endl;

  // 构建引擎推断
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
      .build();

  // 构建引擎后备列表
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)
                    .setOperation(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                    .build();

  // 获取引擎配置和后备列表
  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  auto& fallback_list = fallback.getFallbackList();

  // 过滤引擎配置，根据给定的条件（确定性、允许TF32、数据类型）
  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_utils::filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, at::kChar);
  cudnn_utils::filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, at::kChar);

  // 遍历筛选后的引擎配置
  for (auto &cfg : engine_configs) {
    try {
      // 创建执行计划
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();
      // 运行执行计划
      run(plan);
      // 将执行计划缓存起来
      execution_plan_cache.emplace(key, plan);
      // 成功找到引擎并执行，直接返回
      return;
    } catch (cudnn_frontend::cudnnException &e) {
      // 捕获cudnn前端异常并输出错误信息
      std::cout << "cudnn error:" << e.what() << std::endl;
    } catch(c10::CuDNNError &e) {
      // 捕获c10 CuDNN错误并输出错误信息
      std::cout << "other error" << e.what() << std::endl;
    }
  }

  // 如果找不到合适的引擎配置，抛出错误信息
  TORCH_CHECK(false, "Unable to find an engine to execute this computation Quantized Linear Cudnn");
}

// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
// Numerics are the same as conv (see aten/src/ATen/native/quantized/Conv.cpp):
template <bool kReluFused>
// 实现函数 apply_impl，用于在 CUDA 上应用量化线性层权重
at::Tensor PackedLinearWeightCudnn::apply_impl(
    const at::Tensor& act,                         // 输入的激活张量
    double output_scale,                           // 输出的缩放因子
    int64_t output_zero_point) {                   // 输出的零点
  std::vector<int64_t> original_output_shape{act.sizes().vec()}; // 保存原始输出形状的 2D 向量
  original_output_shape.back() = orig_weight.size(0); // 设置最后一个维度为原始权重的输出通道数
  // cudnn 需要张量至少是 3D 的。我们会在量化输出前添加一个虚拟维度
  std::vector<int64_t> output_shape(3, 1);         // 输出的形状向量，初始为 3 个维度，都是 1
  output_shape[1] = original_output_shape[0];       // 第二个维度是原始输出的第一个维度
  output_shape[2] = original_output_shape[1];       // 第三个维度是原始输出的第二个维度
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8), // 在 CUDA 上创建一个量化 int8 张量
      output_scale,                                // 输出的缩放因子
      output_zero_point);                          // 输出的零点
  // cudnn 需要张量至少是 3D 的。act 目前是 2D 的。我们会创建一个 3D 视图
  std::vector<int64_t> new_sizes(3, 1);            // 新形状的大小向量，初始为 3 个维度，都是 1
  // cudnn 需要前导维度是虚拟维度
  new_sizes.back() = act.sizes().back();           // 最后一个维度是 act 的最后一个维度
  new_sizes[1] = act.size(0);                      // 第二个维度是 act 的第一个维度
  apply_impl_helper<kReluFused>(
      quantized_output,                            // 量化输出张量
      act.view(new_sizes),                         // act 的重新视图
      output_scale);                               // 输出的缩放因子
  return quantized_output.view(original_output_shape); // 返回原始输出形状的视图
}

// 应用量化线性层权重，不包含 ReLU 激活
at::Tensor PackedLinearWeightCudnn::apply(
    at::Tensor input,                              // 输入张量
    double output_scale,                           // 输出的缩放因子
    int64_t output_zero_point) {                   // 输出的零点
  return apply_impl<false>(input, output_scale, output_zero_point); // 调用 apply_impl，不包含 ReLU
}

// 应用量化线性层权重，并包含 ReLU 激活
at::Tensor PackedLinearWeightCudnn::apply_relu(
    at::Tensor input,                              // 输入张量
    double output_scale,                           // 输出的缩放因子
    int64_t output_zero_point) {                   // 输出的零点
  return apply_impl<true>(input, output_scale, output_zero_point); // 调用 apply_impl，包含 ReLU
}

namespace at {
namespace native {
namespace {

template <bool kReluFused>
// 定义 QLinearInt8 类，实现对量化线性层的操作
class QLinearInt8 final {
 public:
  // 运行函数，对输入的激活张量应用量化线性层的权重
  static at::Tensor run(
      at::Tensor act,                               // 输入的激活张量
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight, // 线性层参数的包装指针
      double output_scale,                          // 输出的缩放因子
      int64_t output_zero_point) {                  // 输出的零点
    // TODO: 检查所有的零点都是零/所有张量都是对称量化的
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point); // 如果包含 ReLU，则调用带 ReLU 的应用函数
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);     // 否则，调用不带 ReLU 的应用函数
    }
  }
};

// 注册 CUDA 上的量化线性操作库实现
TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  register_linear_params();                        // 注册线性参数
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear"), QLinearInt8<false>::run); // 实现不包含 ReLU 的量化线性操作
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu"), QLinearInt8<true>::run); // 实现包含 ReLU 的量化线性操作
}

} // namespace
} // namespace native
} // namespace at

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
```