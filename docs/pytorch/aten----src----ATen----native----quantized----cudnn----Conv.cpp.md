# `.\pytorch\aten\src\ATen\native\quantized\cudnn\Conv.cpp`

```py
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // CUDAConfig.h文件中定义了AT_CUDNN_ENABLED宏

#if AT_CUDNN_ENABLED()
#include <c10/util/ArrayRef.h>  // ArrayRef类的头文件

#include <ATen/ATen.h>  // ATen库的头文件
#include <ATen/cuda/Exceptions.h>  // CUDA异常处理的头文件
#include <ATen/cudnn/Handle.h>  // cuDNN句柄的头文件
#include <ATen/native/cudnn/ConvShared.h>  // cuDNN卷积共享函数的头文件
#include <ATen/native/quantized/cudnn/utils.h>  // 量化cuDNN工具函数的头文件
#include <ATen/native/quantized/ConvUtils.h>  // 量化卷积工具函数的头文件
#include <ATen/native/quantized/PackedParams.h>  // 打包参数的头文件
#include <ATen/native/utils/ParamsHash.h>  // 参数哈希工具函数的头文件
#include <ATen/TensorUtils.h>  // Tensor工具函数的头文件
#include <c10/cuda/CUDACachingAllocator.h>  // CUDA缓存分配器的头文件
#include <cudnn_frontend.h>  // cuDNN前端的头文件
#include <torch/library.h>  // Torch库的头文件

#include <iostream>  // 标准输入输出流的头文件
#include <unordered_map>  // 无序映射容器的头文件
#include <vector>  // 向量容器的头文件

template <int kSpatialDim = 2>
int register_conv_params();  // 注册卷积参数的模板函数声明

extern template int register_conv_params<2>();  // 对模板函数实例化的外部声明
extern template int register_conv_params<3>();  // 对模板函数实例化的外部声明

// TODO: 根据输入数据类型和权重数据类型确定运算数据类型的映射表
// 根据输入数据类型、填充、步长、扩展率创建卷积描述符
cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dataType, c10::IntArrayRef padding, c10::IntArrayRef stride, c10::IntArrayRef dilation) {
  uint64_t convDim = stride.size();  // 获取卷积维度
  return cudnn_frontend::ConvDescBuilder()
    .setDataType(dataType)  // 设置数据类型
    .setMathMode(CUDNN_CROSS_CORRELATION)  // 设置数学模式为交叉相关
    .setNDims(convDim)  // 设置卷积维度数
    .setStrides(convDim, stride.data())  // 设置步长
    .setPrePadding(convDim, padding.data())  // 设置前填充
    .setPostPadding(convDim, padding.data())  // 设置后填充
    .setDilation(convDim, dilation.data())  // 设置扩展率
    .build();  // 构建并返回卷积描述符
}

// FIXME: 通过重用Conv_v7.cpp中的基准缓存使其线程安全
namespace {
struct CacheKey {
  at::native::ConvolutionParams params;  // 卷积参数结构体
  uint8_t input_alignment;  // 输入对齐
  uint8_t weight_alignment;  // 权重对齐
  uint8_t output_alignment;  // 输出对齐
  // 默认情况下没有偏置时为-1
  int8_t bias_alignment;  // 偏置对齐
  bool kReluFused;  // 是否融合ReLU
};
// 执行计划缓存，映射为参数哈希和参数相等比较的无序映射
std::unordered_map<CacheKey, cudnn_frontend::ExecutionPlan, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;
} // 匿名命名空间

// TODO: 当支持缓存多个操作符时可以使用cudnn_frontend::ExecutionPlanCache
// 参考链接：https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/conv_sample.cpp#L293
// static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

// 参数quantized_output是一个量化张量
template <int kSpatialDim>
template <bool kReluFused>
void PackedConvWeightCudnn<kSpatialDim>::apply_impl_helper(const at::Tensor& quantized_output, const at::Tensor& input, double output_scale) {
  auto act_scale = input.q_scale();  // 获取输入的量化比例
  auto weight_scale = maybe_padded_weight_.q_scale();  // 获取可能填充的权重的量化比例
  auto requantize_multiplier = act_scale * weight_scale / output_scale;  // 重新量化乘数
  // 获取重新量化乘数的张量
  at::Tensor requantize_multiplier_tensor = cudnn_utils::getRequantMultiplierTensor(requantize_multiplier, kSpatialDim + 2);

  std::optional<at::Tensor> bias_multiplier_tensor;
  std::optional<at::Tensor> broadcasted_bias;
  if (bias_.has_value()) {
    // 输入偏置是一个大小与quantized_output的第二个维度相同的1-D张量。
    // 我们需要在尾部添加维度，以正确广播偏置，否则 broadcast_to 将失败。
    // 尾部维度的数量是 quantized_output.dim() - 2，因此 broadcast_bias 的新尺寸变为 quantized_output.dim() - 2 + 1。
    // 对于前导维度不需要进行任何操作。
    std::vector<int64_t> new_size(quantized_output.dim() - 1, 1);
    new_size[0] = bias_.value().size(0);
    // 将偏置重塑为新尺寸的张量
    broadcasted_bias = bias_.value().reshape(new_size);
    // 将其广播到 quantized_output 的尺寸
    broadcasted_bias.value() = broadcasted_bias.value().broadcast_to(quantized_output.sizes());
    // 设置内存格式为 ChannelsLast
    broadcasted_bias.value() = broadcasted_bias.value().to(c10::MemoryFormat::ChannelsLast);
    // 创建一个和 quantized_output 尺寸相同的 tensor 作为 bias 的乘子
    bias_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
    auto bias_multiplier = 1.0 / (act_scale * weight_scale);
    // 填充 bias_multiplier_tensor
    bias_multiplier_tensor.value().fill_(bias_multiplier);
  }

  // 获取当前的 cudnn 句柄
  cudnnHandle_t handle = at::native::getCudnnHandle();
  // 初始化缓存键的内存，因为 CacheKey 会添加隐式的填充，这可能导致未初始化的填充值
  // 用于哈希计算，防止不同填充值导致相同参数的不同哈希输出。
  memset(&key, 0, sizeof(key));
  bool deterministic{true};
  bool allow_tf32{false};
  // 提取 padding、stride、dilation 的向量表示
  auto padding_vec = padding_.vec();
  auto stride_vec = stride_.vec();
  auto dilation_vec = dilation_.vec();
  // 设置卷积参数到 key.params 中
  setConvolutionParams(&key.params, input, maybe_padded_weight_, padding_vec, stride_vec, dilation_vec, groups_, deterministic, allow_tf32, input.suggest_memory_format());

  // 对于 int8 卷积，操作数数据类型必须为 int32，但是输出张量的数据类型可以是 int32 或 fp32
  key.params.dataType = CUDNN_DATA_INT32;
  // 获取输入和输出张量的对齐方式
  key.input_alignment = cudnn_utils::getAlignment(input);
  key.output_alignment = cudnn_utils::getAlignment(quantized_output);
  key.weight_alignment = cudnn_utils::getAlignment(maybe_padded_weight_);
  // 如果有偏置，则获取偏置的对齐方式
  if (bias_.has_value()) {
    key.bias_alignment = cudnn_utils::getAlignment(broadcasted_bias.value());
  } else {
    key.bias_alignment = -1;
  }
  // 设置是否融合 ReLU 标志
  key.kReluFused = kReluFused;

  // 定义一个 lambda 函数用于执行 cudnn 前端的执行计划
  auto run = [&](const cudnn_frontend::ExecutionPlan& plan_desc) {
    // 获取所需的工作空间大小
    auto workspace_size = plan_desc.getWorkspaceSize();
    // 分配 CUDA 工作空间内存
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    // 将数据指针和 UID 放入容器中
    at::SmallVector<void *, 7> data_ptrs;
    at::SmallVector<int64_t, 7> uids;
    data_ptrs = {input.data_ptr<int8_t>(), maybe_padded_weight_.data_ptr<int8_t>(),
                 requantize_multiplier_tensor.data_ptr(), quantized_output.data_ptr<int8_t>()};
    uids = {'x', 'w', 's', 'r'};
    // 检查是否存在偏置项
    if (bias_.has_value()) {
      // 如果存在偏置项，将偏置项数据指针、偏置乘数张量数据指针、以及偏置项数据指针加入数据指针列表
      data_ptrs.insert(data_ptrs.end(), {broadcasted_bias.value().data_ptr(), bias_multiplier_tensor.value().data_ptr(),
                                         broadcasted_bias.value().data_ptr()});
      // 将'b', 'c', 'd'作为唯一标识符插入uids列表
      uids.insert(uids.end(), {'b', 'c', 'd'});
    }
    // 创建CUDNN前端的VariantPackBuilder对象，设置工作空间指针、数据指针和唯一标识符
    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
      .setDataPointers(uids.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    // 获取VariantPack对象的原始描述符
    auto variant_pack_desc = variantPack.get_raw_desc();
    // 执行CUDNN后端操作，检查执行结果
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc.get_raw_desc(), variant_pack_desc));
  };

  // 在执行计划缓存中查找指定键的执行计划
  auto search = execution_plan_cache.find(key);
  // 如果找到了执行计划，则从缓存中获取该计划描述符并执行
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ExecutionPlan plan_desc = search->second;
    run(plan_desc);
    return;
  }
  // conv_op计算act_fp32 * w_fp32的卷积操作（矩阵乘法）
  // 其中act_fp32和w_fp32分别为输入和权重变量，输出为一个fp32张量
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(input.sizes(), input.strides(), CUDNN_DATA_INT8, 'x', key.input_alignment))
      // 对于虚拟张量，不使用对齐参数，因此可以在这里放置一个任意的值，例如key.output_alignment
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'y', key.output_alignment, true))
      .setwDesc(cudnn_utils::getTensorDescriptor(maybe_padded_weight_.sizes(), maybe_padded_weight_.strides(), CUDNN_DATA_INT8, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding_vec, stride_vec, dilation_vec))
      .build();
  // std::cout << "operator:" << conv_op.describe() << std::endl;

  // 初始化可选的bias_mult_op和sum_conv_bias_op操作对象
  std::optional<cudnn_frontend::Operation> bias_mult_op;
  std::optional<cudnn_frontend::Operation> sum_conv_bias_op;
  // 如果存在偏置项，则创建bias_mult_op操作对象
  if (bias_.has_value()) {
    // bias_mult_op计算bias_fp32 / (act_scale * w_scale)或bias_fp32 * (1 / (act_scale * w_scale))
    // 其中bias_multiplier = (1 / (act_scale * w_scale))
    // 输出为一个fp32张量，在此处使用就地操作，输出赋值给输入
    // 我们无法直接赋值给bias_mult_op，因为cudnn_frontend::Operation的operator=被删除；
    // 替代方法是使用std::unique_ptr并动态分配这些构建器操作对象，但这里我们选择静态方式。
    // std::optional<T>::emplace()允许使用这种方法
    // 在 bias_mult_op 中添加一个 cudnn_frontend 操作，创建一个 pointwise 描述符的操作
    bias_mult_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      // 设置输入描述符为 broadcasted_bias.value() 的张量描述符，标记为 'b'，并使用对齐函数确定对齐方式
      .setxDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'b', cudnn_utils::getAlignment(broadcasted_bias.value())))
      // 设置输入描述符为 bias_multiplier_tensor.value() 的张量描述符，标记为 'c'，并使用对齐函数确定对齐方式
      .setbDesc(cudnn_utils::getTensorDescriptor(bias_multiplier_tensor.value(), 'c', cudnn_utils::getAlignment(bias_multiplier_tensor.value())))
      // 设置输出描述符为 broadcasted_bias.value() 的张量描述符，标记为 'd'，并使用对齐函数确定对齐方式
      .setyDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      // 设置点乘操作描述符，使用 bias_multiplier_tensor.value() 的 cudnn 数据类型
      .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(bias_multiplier_tensor.value())))
      // 构建操作
      .build());

    // 计算 (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)])
    // 其中第一项和第二项分别是 conv_op 和 broadcasted_bias 的输出
    // 输出是一个 fp32 张量
    // 在这里使用 inplace 操作，将输出分配给输入
    sum_conv_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      // 设置输入描述符为 conv_op 的输出张量描述符
      .setxDesc(conv_op.getOutputTensor())
      // 设置输入描述符为 broadcasted_bias.value() 的张量描述符，标记为 'd'，并使用对齐函数确定对齐方式
      // 对于虚拟张量，对齐不适用，因此可以放置一个任意值，例如 key.output_alignment
      .setbDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      // 设置输出描述符为 quantized_output 的张量描述符，标记为 'e'，使用 key.output_alignment 确定对齐方式，允许是虚拟张量
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'e', key.output_alignment, true))
      // 设置点加操作描述符，使用 broadcasted_bias.value() 的 cudnn 数据类型
      .setpwDesc(cudnn_utils::getPointWiseAddDescriptor(at::native::getCudnnDataType(broadcasted_bias.value())))
      // 构建操作
      .build());
  }

  // relu_op 计算 relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]
  // 或者如果没有 bias，则计算 relu(act_int8 * w_int8)
  // 输出是一个 fp32 张量
  std::optional<cudnn_frontend::Operation> relu_op;
  // 如果 kReluFused 为真
  if (kReluFused) {
    // 在这里使用 inplace 操作，将输出分配给输入
    relu_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      // 设置输入描述符为 tensor2requant_ptr 的张量描述符
      // 对于虚拟张量，对齐不适用，因此可以放置一个任意值，例如 key.output_alignment
      .setxDesc(tensor2requant_ptr)
      // 设置输出描述符为 quantized_output 的张量描述符，标记为 'f'，使用 key.output_alignment 确定对齐方式，允许是虚拟张量
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'f', key.output_alignment, true))
      // 设置 relu 操作描述符，使用 CUDNN_DATA_FLOAT 的数据类型
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(CUDNN_DATA_FLOAT))
      // 构建操作
      .build());
  }

  // requant_op 计算 relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
  // 或者如果没有 bias，则计算 relu(act_int8 * w_int8) / (out_scale / (act_scale * w_scale))
  // 输出是一个 fp32 张量
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : tensor2requant_ptr)
    // 设置输入张量描述符，根据 kReluFused 的条件选择使用 ReLU 输出张量或指定的 tensor2requant_ptr
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 's', cudnn_utils::getAlignment(requantize_multiplier_tensor)))
    // 设置偏置张量描述符，使用 requantize_multiplier_tensor 创建描述符，并指定 's' 对齐策略
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'r', key.output_alignment))
    // 设置输出张量描述符，使用 quantized_output 的尺寸和步长，数据类型为 CUDNN_DATA_INT8，'r' 表示对齐策略，使用 key.output_alignment
    .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(requantize_multiplier_tensor)))
    // 设置点乘描述符，根据 requantize_multiplier_tensor 的数据类型确定描述符
    .build();
  // 构建操作描述符

  std::vector<cudnn_frontend::Operation const *> ops{&conv_op};
  // 创建操作指针向量，包含指向 conv_op 的指针
  if (bias_.has_value()) {
    ops.emplace_back(&(bias_mult_op.value()));
    // 如果存在偏置值，将 bias_mult_op 加入操作指针向量
    ops.emplace_back(&(sum_conv_bias_op.value()));
    // 将 sum_conv_bias_op 加入操作指针向量
  }
  if (kReluFused) {
    ops.emplace_back(&(relu_op.value()));
    // 如果 kReluFused 为真，将 relu_op 加入操作指针向量
  }
  ops.emplace_back(&requant_op);
  // 将 requant_op 加入操作指针向量

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      // 设置操作图的句柄
      .setOperationGraph(ops.size(), ops.data())
      // 设置操作图的操作数量和操作指针数组
      .build();
  // 构建操作图

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)
      // 设置启发式引擎构建器的操作图
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
      // 设置启发式模式为即时模式
      .build();
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)
                    .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                    .build();

  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  // 获取启发式引擎的配置列表
  auto& fallback_list = fallback.getFallbackList();
  // 获取回退列表

  cudnn_frontend::EngineConfigList filtered_configs;
  // 创建筛选后的引擎配置列表
  cudnn_utils::filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, at::kChar);
  // 筛选出符合条件的引擎配置到 filtered_configs 中，根据 deterministic、allow_tf32 和数据类型 at::kChar

  cudnn_utils::filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, at::kChar);
  // 将回退列表中符合条件的引擎配置也加入到 filtered_configs 中

  for (auto &cfg : engine_configs) {
    // 遍历筛选后的引擎配置列表
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        // 设置执行计划的句柄
        .setEngineConfig(cfg)
        // 设置执行计划使用的引擎配置
        .build();
      // 构建执行计划
      run(plan);
      // 执行计划
      execution_plan_cache.emplace(key, plan);
      // 将执行计划缓存起来，使用 key 作为键
      return;
      // 返回结束函数执行
    } catch (cudnn_frontend::cudnnException &e) {
      std::cout << "cudnn error:" << e.what() << std::endl;
      // 捕获并打印 cudnn 前端异常
    } catch(c10::CuDNNError &e) {
      std::cout << "other error" << e.what() << std::endl;
      // 捕获并打印 c10 CuDNN 错误
    }
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation in Quantized Conv2D Cudnn");
  // 若未找到可执行该计算的引擎，则抛出异常
// 输出张量将是一个截断的 int8 张量
// act 和 weight 都将是 int8 张量
/*
数值计算：
out_fp32 = conv_fp32(act_fp32, w_fp32, …)
                    = act_fp32 * w_fp32 + bias_fp32
act_int8 = act_fp32 / act_scale + act_zero_point
w_int8 = w_fp32 / w_scale + w_zero_point
out_int8 = out_fp32 / out_scale + out_zero_point
out_int8 = (act_fp32 * w_fp32 + [bias_fp32]) / out_scale + out_zero_point
              = (act_int8 - act_zero_point) * act_scale * (w_int8 - w_zero_point) * w_scale / out_scale + out_zero_point + [bias_fp32 / out_scale]
             = (act_int8 * w_int8 - act_int8 * w_zero_point - act_zero_point * w_int8 + act_zero_point * w_zero_point) * act_scale * w_scale / out_scale + out_zero_point + [bias_fp32 / out_scale]
             = (如果 act 和 weight 都是对称量化的，int8，则 act_zero_point = w_zero_point = 0)
             = (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) * act_scale * w_scale / out_scale
             = (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
             = 重新量化((act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]), out_scale / (act_scale * w_scale))
*/
template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeightCudnn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  // 确定批量大小
  const auto batch_size = kSpatialDim == 2 ? act.size(0) : 1;
  // 确定输入通道数
  const auto num_input_channels = act.size(kSpatialDim - 1);
  // 获取输入张量的高度和宽度
  const auto H = act.size(kSpatialDim);
  const auto W = act.size(kSpatialDim + 1);
  // 确定输出通道数
  const auto num_output_channels = maybe_padded_weight_.size(0); // 输出通道数
  // 确定卷积核大小
  std::vector<int64_t> kernel_size = {maybe_padded_weight_.size(2), maybe_padded_weight_.size(3)};
  // 创建输出形状
  auto output_shape = at::native::quantized::MakeConvOutputShape<kSpatialDim>(batch_size, num_output_channels, {H, W},
  kernel_size, stride_, padding_, dilation_);
  // 创建量化的输出张量
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      output_scale,
      output_zero_point,
      at::MemoryFormat::ChannelsLast);

  // cudnn v8.4.0 要求 conv2d 的 int8 激活张量的输入通道数必须是 4 的倍数。如果不是，需要我们手动填充到 4 的倍数，
  // 因为 cudnn 目前不支持填充操作。
  // TODO: 当 cudnn 在他们的运算符中启用填充时，我们可以移除这里的填充；目前仅在 groups=1（未分组卷积）情况下支持填充
  auto act_maybe_padded = act;
  if (num_input_channels % 4 != 0) {
    int8_t num_slices = 4 - num_input_channels % 4; // 需要填充的片段数
    // ...
  }
    // 对输入张量进行零填充以适应指定的维度
    act_maybe_padded = at::pad(act, {0, 0, 0, 0, 0, num_slices, 0, 0}, "constant", 0);
  }
  
  // 应用带融合ReLU的操作到量化输出张量上，转换为ChannelsLast内存格式
  apply_impl_helper<kReluFused>(
      quantized_output, act_maybe_padded.to(c10::MemoryFormat::ChannelsLast), output_scale);

  // 如果输出通道数被填充，则需要返回被切片的张量
  if (num_unpadded_output_channels_ != maybe_padded_weight_.size(0)) {
    // 返回从第1维开始切片的量化输出张量，范围是[0, num_unpadded_output_channels_)
    return quantized_output.slice(1, 0, num_unpadded_output_channels_);
  }
  // 否则直接返回量化输出张量
  return quantized_output;
}

template <int kSpatialDim>
at::Tensor PackedConvWeightCudnn<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  // 调用 apply_impl 函数，不使用 ReLU 激活函数
  return apply_impl<false>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightCudnn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  // 调用 apply_impl 函数，使用 ReLU 激活函数
  return apply_impl<true>(input, output_scale, output_zero_point);
}

template at::Tensor PackedConvWeightCudnn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightCudnn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

namespace at::native {
namespace {

template <bool kReluFused>
class QConv1dInt8 final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    at::Tensor output;
    // 目前我们使用 conv2d 内核来执行 conv1d，通过将输入和权重张量变为4维而不是3维。我们增加一个大小为1的虚拟宽度维度
    // N, C, L -> N, C, 1, L
    act = act.unsqueeze(-2);
    if (kReluFused) {
      // 如果启用了 ReLU 融合，则调用带有 ReLU 的 packed_weight 的 apply 函数
      output = packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      // 否则调用普通的 packed_weight 的 apply 函数
      output = packed_weight->apply(act, output_scale, output_zero_point);
    }
    // N, C, 1, L -> N, C, L
    // 去除虚拟的宽度维度，返回最终输出张量
    return output.squeeze_(-2);
  }
};

template <int kSpatialDim, bool kReluFused>
class QConvInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    // 检查 kSpatialDim 是否为1或2，因为这是 quantized cudnn conv2d 操作的预期维度
    TORCH_CHECK(kSpatialDim == 1 || kSpatialDim == 2, "Error in quantized cudnn conv2d operator: "
                "Expected kSpatialDim == 1 || kSpatialDim == 2; received kSpatialDim=", kSpatialDim);
    // TODO: 检查所有 zero_point 是否为零/所有张量是否对称量化
    if (kReluFused) {
      // 如果启用了 ReLU 融合，则调用带有 ReLU 的 packed_weight 的 apply 函数
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      // 否则调用普通的 packed_weight 的 apply 函数
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};
TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
    // 定义 Torch 库的实现，命名为 quantized，使用 QuantizedCUDA，注册到 m
    // 对于 quantized::conv1d，使用 QConv1dInt8<false>::run 函数实现
    // QConv1dInt8<false>::run 不使用新的变体来处理打包的权重，与 QuantizedCPU 的 conv1d 保持一致
    m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d"), QConv1dInt8<false>::run);
    // 对于 quantized::conv1d_relu，使用 QConv1dInt8<true>::run 函数实现
    // QConv1dInt8<true>::run 也不使用新的变体来处理打包的权重
    m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_relu"), QConv1dInt8<true>::run);
    // 对于 quantized::conv2d.new，使用 QConvInt8<2, false>::run 函数实现
    // QConvInt8<2, false>::run 使用新的变体来处理打包的权重
    m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d.new"), QConvInt8<2, false>::run);
    // 对于 quantized::conv2d_relu.new，使用 QConvInt8<2, true>::run 函数实现
    // QConvInt8<2, true>::run 使用新的变体来处理打包的权重
    m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"), QConvInt8<2, true>::run);
}
// 匿名命名空间结束
} // anonymous namespace
// 命名空间 at::native 结束

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
```