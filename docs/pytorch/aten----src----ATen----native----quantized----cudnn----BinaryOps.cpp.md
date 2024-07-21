# `.\pytorch\aten\src\ATen\native\quantized\cudnn\BinaryOps.cpp`

```py
#ifdef USE_CUDA
// 如果定义了 USE_CUDA，则包含 AT_CUDNN_ENABLED 的定义
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()
// 如果 AT_CUDNN_ENABLED() 返回 true，则继续包含以下头文件

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/TensorUtils.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/_empty_affine_quantized.h>
#endif

#include <unordered_map>

namespace at {
namespace native {
namespace {
constexpr uint8_t max_num_input_dim = 5;
// 定义了结构体 AddParams，用于保存加法操作的参数
struct AddParams {
  c10::DeviceIndex device_id;  // 设备 ID
  int input_a_size[max_num_input_dim];  // 输入张量 A 的各维度大小
  int input_b_size[max_num_input_dim];  // 输入张量 B 的各维度大小
  uint8_t input_dim; // 输入张量的维度，假定 A 和 B 的维度相同
  at::MemoryFormat memory_format;  // 建议的内存格式
  bool deterministic;  // 是否确定性操作
  bool allow_tf32;  // 是否允许 TF32 数据类型
};
// 定义了结构体 CacheKey，用作加法操作缓存的键
struct CacheKey {
  AddParams params;  // 加法操作的参数
  uint8_t input_a_alignment;  // 输入张量 A 的对齐方式
  uint8_t input_b_alignment;  // 输入张量 B 的对齐方式
  uint8_t output_alignment;  // 输出张量的对齐方式
  bool kReluFused;  // 是否融合了 ReLU 操作
};
// 设置 AddParams 结构体的参数，用于描述加法操作的输入张量信息
void setAddParams(
    AddParams* params, const at::Tensor& input_a, const at::Tensor& input_b,
    bool deterministic, bool allow_tf32) {
  memset(params, 0, sizeof(AddParams));  // 清空 params 结构体的内存
  params->device_id = at::cuda::current_device();  // 获取当前 CUDA 设备 ID
  params->input_dim = input_a.dim();  // 获取输入张量 A 的维度
  params->memory_format = input_a.suggest_memory_format();  // 建议的内存格式
  // 遍历各维度，获取输入张量 A 和 B 的大小
  for (int i = 0; i < params->input_dim; ++i) {
    params->input_a_size[i] = input_a.sizes()[i];
    params->input_b_size[i] = input_b.sizes()[i];
  }
  params->deterministic = deterministic;  // 设置是否确定性操作
  params->allow_tf32 = allow_tf32;  // 设置是否允许 TF32 数据类型
}
// FIXME: 通过重用 Conv_v7.cpp 中的基准缓存使其线程安全
// 目前将最大输入维度设置为 5
// 如有必要可以增加该值
std::unordered_map<CacheKey, cudnn_frontend::ManagedOpaqueDescriptor, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;

// TODO: 这部分代码也出现在 BinaryOps.cpp 和 quantized/cpu/ 中的其他 cpp 文件中。我们考虑后续将其移入 quantized/ 目录下的一个实用工具文件中。
// 检查输入张量 qa 和 qb 的合法性
inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is supported in Add.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization scheme.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
}

// 目前仅支持 int8 对称量化加法（输入和输出的 zero_point 均为 0）
// 实现了带有 ReLU 的加法操作
// 这需要 4 个 cudnn 操作（2 个乘法、1 个加法和 1 个 ReLU 操作）
// Multiplication ops: rhs_mult_op, requant_op
// Addition op: add_op
// Relu op: relu_op

// 定义模板函数 add，可以选择是否融合 ReLU 操作
template <bool kReluFused = false>
Tensor add(Tensor qa, Tensor qb, double output_scale, int64_t output_zero_point) {
  // 如果 qa 的元素个数为 0，返回一个空 Tensor
  if (qa.numel() == 0) {
    return Tensor{};
  }
  // 检查输入的两个 Tensor 的形状是否相同
  TORCH_CHECK(qa.sizes() == qb.sizes(), "Quantized cudnn add currently expects both input tensors to be the same shape");

  // 对输入的 Tensor 进行进一步的检查
  check_inputs(qa, qb);

  // 如果输入的 Tensor 维度小于 3，则在前面添加虚拟维度，以满足 cudnn 对至少 3D 的要求
  auto orig_sizes = qa.sizes().vec();
  if (qa.dim() < 3) {
    std::vector<int64_t> new_sizes(3, 1);
    // cudnn 要求前导维度是虚拟维度
    new_sizes.back() = qa.sizes().back();
    if (qa.dim() == 2) {
      new_sizes[1] = qa.size(0);
    }
    // 修改 Tensor 的视图以匹配新的维度
    qa = qa.view(new_sizes);
    qb = qb.view(new_sizes);
  } else if (qa.dim() == 4) {
    // 如果输入的 Tensor 维度为 4，则使用 ChannelsLast 内存格式进行连续化
    qa = qa.contiguous(c10::MemoryFormat::ChannelsLast);
    qb = qb.contiguous(c10::MemoryFormat::ChannelsLast);
  }

  // 确定内存格式，如果维度为 4，则使用 ChannelsLast，否则使用 Contiguous
  auto memory_format = qa.dim() == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  // 创建一个新的 Float 类型的 Tensor，用于存储加法操作的输出
  at::Tensor add_output = at::empty(qa.sizes(), at::device(at::kCUDA).dtype(at::kFloat), memory_format);

  // 创建一个新的 QInt8 类型的 Tensor，用于存储量化后的输出
  at::Tensor quantized_output = at::_empty_affine_quantized(qa.sizes(), at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
                                                            output_scale, output_zero_point, memory_format);

  // 计算重新量化的乘法因子
  double requantize_multiplier = qa.q_scale() / output_scale;
  // 创建一个 Tensor，存储重新量化的乘法因子
  at::Tensor requantize_multiplier_tensor = cudnn_utils::getRequantMultiplierTensor(requantize_multiplier, quantized_output.dim());

  // 创建一个 Tensor，存储右操作数的乘法因子
  at::Tensor rhs_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), memory_format);
  rhs_multiplier_tensor.fill_(qb.q_scale() / qa.q_scale());

  // 获取 cudnn 的句柄
  cudnnHandle_t handle = at::native::getCudnnHandle();

  // 创建缓存键对象
  CacheKey key;
  // 因为 CacheKey 中可能存在填充的未初始化值，这里需要使用 memset 进行初始化，
  // 以确保哈希输出的一致性和正确性
  memset(&key, 0, sizeof(key));
  // 设置添加操作的参数到缓存键中
  bool deterministic{true};
  bool allow_tf32{false};
  setAddParams(&key.params, qa, qb, deterministic, allow_tf32);
  key.kReluFused = kReluFused;
  key.input_a_alignment = cudnn_utils::getAlignment(qa);
  key.input_b_alignment = cudnn_utils::getAlignment(qb);
  key.output_alignment = cudnn_utils::getAlignment(add_output);

  // 定义一个 lambda 函数用于运行 cudnn 前端计划描述
  auto run = [&](cudnn_frontend::ManagedOpaqueDescriptor plan_desc) {
    auto workspace_size = 0;
    // 创建一个字节类型的 Tensor 作为工作空间
    auto workspace = at::empty({workspace_size}, qa.options().dtype(at::kByte));
    // 创建两个空的指针向量，用于存储数据指针和UID
    std::vector<void *> data_ptrs;
    std::vector<int64_t> uids;
    // 预留空间以存储8个元素，以避免在添加元素时重新分配内存
    data_ptrs.reserve(8);
    uids.reserve(8);
    // 初始化data_ptrs向量，包含qb、rhs_multiplier_tensor等的数据指针
    data_ptrs = {qb.data_ptr<int8_t>(), rhs_multiplier_tensor.data_ptr(), add_output.data_ptr(),
                 qa.data_ptr<int8_t>(), add_output.data_ptr(), requantize_multiplier_tensor.data_ptr(),
                 quantized_output.data_ptr<int8_t>()};
    // 初始化uids向量，包含'b'、'm'、'c'、'a'、'p'、'r'、'q'等UID
    uids = {'b', 'm', 'c', 'a', 'p', 'r', 'q'};
    // 如果启用了ReLU融合，添加额外的数据指针和UID
    if (kReluFused) {
        data_ptrs.emplace_back(add_output.data_ptr());
        uids.emplace_back('f');
    }

    // 创建VariantPack对象，用于封装数据和UID，设置工作空间指针和数据信息
    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace.data_ptr())
      .setDataPointers(uids.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    // 获取VariantPack的原始描述符
    auto variant_pack_desc = variantPack.get_raw_desc();
    // 执行后端操作，使用CUDNN后端执行引擎处理操作计划
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc->get_backend_descriptor(), variant_pack_desc));
  };

  // 在执行计划缓存中查找是否已经存在对应的执行计划
  auto search = execution_plan_cache.find(key);
  // 如果找到了对应的执行计划，直接使用缓存中的计划描述符运行操作
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ManagedOpaqueDescriptor plan_desc = search->second;
    run(plan_desc);
    // 返回量化输出视图，恢复原始尺寸
    return quantized_output.view(orig_sizes);
  }

  // 创建乘法操作描述符，计算qb_int8 * ( qb_scale/qa_scale )
  auto rhs_mult_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(qb.sizes(), qb.strides(), CUDNN_DATA_INT8, 'b', key.input_b_alignment))
      .setbDesc(cudnn_utils::getTensorDescriptor(rhs_multiplier_tensor, 'm', cudnn_utils::getAlignment(rhs_multiplier_tensor)))
      .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'c', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(add_output)))
      .build();

  // 创建加法操作描述符，计算qa_int8 + qb_int8 * ( qb_scale/qa_scale )
  // add_output用于累加，是一个fp32张量
  auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(rhs_mult_op.getOutputTensor())
      .setbDesc(cudnn_utils::getTensorDescriptor(qa.sizes(), qa.strides(), CUDNN_DATA_INT8, 'a', key.input_a_alignment))
      .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'p', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseAddDescriptor(at::native::getCudnnDataType(add_output)))
      .build();

  // 创建ReLU操作描述符，计算relu( (qa_int8 + qb_int8 * ( qb_scale/qa_scale ) ) )
  // 输出是一个fp32张量
  std::optional<cudnn_frontend::Operation> relu_op;
  if (kReluFused) {
    // 在此处使用原地操作，将输出分配给输入
    relu_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
        .setxDesc(add_op.getOutputTensor())
        .setbDesc(cudnn_utils::getTensorDescriptor(add_output, 'f', key.output_alignment))
        .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'f', key.output_alignment))
        .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(at::native::getCudnnDataType(add_output)))
        .build();
  }
    relu_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(add_op.getOutputTensor())  // 设置输入张量描述为 add_op 的输出张量
      .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'f', key.output_alignment))  // 设置输入张量描述为 add_output 的张量描述
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(at::native::getCudnnDataType(add_output)))  // 设置逐点运算的 ReLU 描述
      .build());  // 构建并添加操作到 relu_op 中

  // requant_op computes
  // (a_int8 + b_int8 * ( b_scale/a_scale) ) * a_scale / out_scale
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : add_op.getOutputTensor())  // 根据 kReluFused 条件设置输入张量描述
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 'r', cudnn_utils::getAlignment(requantize_multiplier_tensor)))  // 设置输入张量描述为 requantize_multiplier_tensor 的张量描述
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'q', cudnn_utils::getAlignment(quantized_output)))  // 设置输出张量描述为 quantized_output 的张量描述
    .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(requantize_multiplier_tensor)))  // 设置逐点乘法的描述
    .build();  // 构建 requant_op

  std::vector<cudnn_frontend::Operation const *> ops{&rhs_mult_op, &add_op};
  if (kReluFused) {
    ops.emplace_back(&(relu_op.value()));  // 如果 kReluFused 为真，则添加 relu_op 的值到操作列表中
  }
  ops.emplace_back(&requant_op);  // 添加 requant_op 到操作列表中

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)  // 设置操作图的句柄
      .setOperationGraph(ops.size(), ops.data())  // 设置操作图的操作列表
      .build();  // 构建操作图

  // 创建引擎启发器
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)  // 设置引擎启发器的操作图
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)  // 设置启发模式为即时模式
      .build();  // 构建启发器

  // 创建引擎回退列表
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)  // 设置引擎回退列表的操作图
                    .setOperation(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)  // 设置回退列表的操作类型
                    .build();  // 构建回退列表

  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());  // 获取引擎配置列表
  auto& fallback_list = fallback.getFallbackList();  // 获取回退列表

  // 创建过滤后的引擎配置列表
  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_utils::filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, at::kChar);  // 根据条件过滤引擎配置
  cudnn_utils::filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, at::kChar);  // 根据条件过滤回退列表
  for (auto &cfg : engine_configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)  // 设置执行计划的句柄
        .setEngineConfig(cfg)  // 设置执行计划的引擎配置
        .build();  // 构建执行计划
      auto plan_desc = plan.get_desc();  // 获取执行计划的描述
      run(plan_desc);  // 执行计划
      execution_plan_cache[key] = plan_desc;  // 将执行计划缓存起来
      return quantized_output.view(orig_sizes);  // 返回量化输出的视图
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << std::endl;} catch(c10::CuDNNError &e) { std::cout << "other error" << e.what() << std::endl;}
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation in Quantized Add Cudnn");
}

// 定义 TORCH_LIBRARY_IMPL 宏，实现 quantized 模块的 CUDA 版本
TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
    // 注册 quantized::add 函数的 CUDA 实现，不包含 ReLU 合并
    m.impl(TORCH_SELECTIVE_NAME("quantized::add"), TORCH_FN(add</*ReLUFused=*/false>));
    // 注册 quantized::add_relu 函数的 CUDA 实现，包含 ReLU 合并
    m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"), TORCH_FN(add</*ReLUFused=*/true>));
}

// 结束 quantized 命名空间
} // namespace quantized

// 结束 native 命名空间
} // namespace native

// 结束 at 命名空间
} // namespace at

// 如果 AT_CUDNN_ENABLED 宏已定义，则结束此代码块
#endif  // AT_CUDNN_ENABLED

// 如果 USE_CUDA 宏已定义，则结束此代码块
#endif  // USE_CUDA
```