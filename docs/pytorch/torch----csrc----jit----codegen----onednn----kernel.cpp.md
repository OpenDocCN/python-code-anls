# `.\pytorch\torch\csrc\jit\codegen\onednn\kernel.cpp`

```
// 包含 Torch 和 OneDNN 之间的头文件
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/kernel.h>

// 包含 ATen 核心库中的功能模块和 Torch 的日志记录支持
#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>

// 命名空间定义，将代码置于 Torch JIT 编译器中的 OneDNN 后端的命名空间下
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 使用 OneDNN 图形定义命名空间中的数据类型
using namespace dnnl::graph;
using data_type = dnnl::graph::logical_tensor::data_type;

// 构造函数：初始化 LLGA 内核对象
LlgaKernel::LlgaKernel(const Node* fusionNode)
    : fusionNode_(fusionNode),
      graph_(fusionNode->g(attr::Subgraph)),
      nGraphInputs_(graph_->inputs().size()),
      nOutputs_(graph_->outputs().size()),
      debugName_(genDebugName()) {
  // TODO: This is a workaround to recreate the partitions here.
  // The ideal way is to use the partition serialization API (not available from
  // LLGA now) to carry a serialized string representation from graph rewrite
  // and deserialize it here.
  
  // 使用 LLGA 图形助手创建 LLGA 图形帮助对象，获取分区信息
  auto llgaGraphHelper = LlgaGraphHelper(graph_);
  auto partitions = llgaGraphHelper.getPartitions();
  // 获取 LLGA 图形助手对象中的张量 ID 到值的映射
  tensorIdToValue_ = llgaGraphHelper.getTensorIdToValue();
  // 检查 LLGA 子图中分区的数量是否为1
  TORCH_CHECK(
      partitions.size() == 1,
      "LLGA subgraph should contain only one partition");
  // 设置当前对象的分区为第一个分区
  partition_ = partitions[0];
  // 获取分区的输入端口数量
  nPartitionInputs_ = partition_.get_input_ports().size();
  
#ifdef GRAPH_DEBUG_ENABLED
  // 若启用了图形调试，输出初始化信息及图形的字符串表示
  GRAPH_DEBUG("Initialized ", debugName(), "\n", graph_->toString());
#endif
}

// 判断是否使用不透明布局的方法
bool LlgaKernel::useOpaqueLayout(size_t offset) const {
  // 使用 LLGA 节点包装器对象来判断是否使用给定偏移量的不透明布局
  return LlgaNodeWrapper(fusionNode_).useOpaqueLayout(offset);
}

// 初始化常量输入值
void LlgaKernel::initializeConstantInputs() {
  // 遍历分区的输入端口列表
  for (auto& lt : partition_.get_input_ports()) {
    auto inputId = lt.get_id();
    // 检查输入端口是否已经初始化，如果未初始化则进行初始化
    if (initializedInputIds_.find(inputId) == initializedInputIds_.end()) {
      // 检查张量 ID 是否在映射中存在
      TORCH_CHECK(
          tensorIdToValue_.count(inputId) > 0,
          "inputs with inputId ",
          inputId,
          " is missing");
      // 获取张量值的指针
      auto* value = tensorIdToValue_[inputId];
      // 检查值是否为常量张量
      TORCH_CHECK(
          value->node()->kind() == prim::Constant &&
              value->type()->cast<TensorType>(),
          "inputs with inputId ",
          inputId,
          " should be a Constant tensor");
      // 将常量值添加到列表中
      constantValues_.emplace_back(value);
      // 将常量张量添加到常量输入列表中
      auto const_tensor = toIValue(value)->toTensor();
      constantInputs_.emplace_back(const_tensor);
    }
  }
}

// 初始化张量 ID 到出现次数的映射关系
std::map<size_t, int64_t> LlgaKernel::initializeTensorIdToOccurence() const {
  // 创建张量 ID 到出现次数的映射关系的对象
  std::map<size_t, int64_t> tensorIdToOccurence;
  // 遍历分区的输入端口列表
  for (auto& lt : partition_.get_input_ports()) {
    auto inputId = lt.get_id();
    // 查找并更新张量 ID 的出现次数
    std::map<size_t, int64_t>::iterator it(tensorIdToOccurence.find(inputId));
    if (it != tensorIdToOccurence.end()) {
      it->second++;
    } else {
      tensorIdToOccurence[inputId] = 1;
    }
  }
  return tensorIdToOccurence;
}

// 初始化输入规格的方法
ArgSpecs LlgaKernel::initializeInputSpecs(const TensorArgs& inputs) {
  // 创建输入规格对象
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nPartitionInputs_);
  // 输出调试信息：初始化图形输入的逻辑张量
  GRAPH_DEBUG("Initializing graph input logical tensors");
  // 初始化张量 ID 到出现次数的映射
  std::map<size_t, int64_t> tensorIdToOccurence =
      initializeTensorIdToOccurence();
  // 遍历图形的输入范围
  for (const auto i : c10::irange(nGraphInputs_)) {
    // 为当前图中的每个输入创建参数规范，并补充张量信息
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    // 将已初始化的输入张量ID插入集合中
    initializedInputIds_.insert(spec.tid());
    // 获取当前参数规范对应的张量ID在映射表中的出现次数
    int64_t occurence = tensorIdToOccurence[spec.tid()];
    // 将当前参数规范按照其出现次数插入输入规范列表中
    inputSpecs.insert(inputSpecs.end(), occurence, spec);
    // 将当前输入索引按照其出现次数插入运行参数索引列表中
    runArgsIdx_.insert(runArgsIdx_.end(), occurence, i);
  }
  // 输出调试信息：初始化常量输入张量
  GRAPH_DEBUG("Initializing constant input tensors");
  // 初始化常量输入张量
  initializeConstantInputs();

  // 检查输入规范列表长度与分区输入总数是否一致，否则报错
  TORCH_CHECK(
      inputSpecs.size() + constantValues_.size() ==
          static_cast<size_t>(nPartitionInputs_),
      "Partition inputs are missing");
  // 输出调试信息：将常量输入逻辑张量连接到图输入逻辑张量
  GRAPH_DEBUG(
      "Concatenating constant input logical tensors to graph input "
      "logical tensors");
  // 遍历常量值列表，为每个常量值创建参数规范并加入输入规范列表和常量逻辑张量列表中
  for (Value* constant_value : constantValues_) {
    ArgSpec constantInputSpec(constant_value);
    inputSpecs.emplace_back(constantInputSpec);
    constantLogicalTensors_.emplace_back(constantInputSpec.logical_tensor());
  }
  // 返回完整的输入规范列表
  return inputSpecs;
}

ArgSpecs LlgaKernel::initializeOutputSpecs() const {
  // 创建一个空的输出参数规格列表
  ArgSpecs outputSpecs;
  // 预留足够的空间以容纳 nOutputs_ 个输出参数
  outputSpecs.reserve(nOutputs_);
  // 遍历所有输出的数量
  for (const auto i : c10::irange(nOutputs_)) {
    // 创建一个输出参数规格对象
    auto spec = ArgSpec(graph_->outputs()[i]);
    // 如果使用不透明布局，则将规格设置为任意类型
    if (useOpaqueLayout(i)) {
      spec = spec.any();
    }
    // 将输出参数规格对象添加到输出参数列表中
    outputSpecs.emplace_back(spec);
  }
  // 返回填充好的输出参数规格列表
  return outputSpecs;
}

std::tuple<RunArgs, RunArgs> LlgaKernel::prepareRunArgs(
    const TensorArgs& inputs,
    TensorArgs& outputs) const {
  // 创建运行输入和输出参数的容器
  RunArgs runInputs, runOutputs;
  // 计算输入参数的数量
  auto numInputs = runArgsIdx_.size();
  // 遍历所有输入参数的索引
  for (const auto i : c10::irange(numInputs)) {
    // 获取当前输入参数的规格
    auto spec = inputSpecs_[i];
    // 获取输入数据张量
    auto input = inputs[runArgsIdx_[i]];
    // 将当前输入参数添加到运行输入参数列表中
    runInputs.push_back(
        {spec.logical_tensor(), Engine::getEngine(), input.data_ptr()});
  }
  // 计算常量输入参数的数量
  auto numConstantInputs = constantInputs_.size();
  // 遍历所有常量输入参数
  for (size_t i = 0; i < numConstantInputs; i++) {
    // 常量输入参数的规格索引位于图输入参数之后
    auto constantInputSpecIdx = nGraphInputs_ + i;
    // 获取常量输入参数的规格
    auto constantInputSpec = inputSpecs_[constantInputSpecIdx];
    // 将当前常量输入参数添加到运行输入参数列表中
    runInputs.push_back(
        {constantLogicalTensors_[i],
         Engine::getEngine(),
         constantInputs_[i].data_ptr()});
  }

  // 遍历所有输出参数的数量
  for (const auto i : c10::irange(nOutputs_)) {
    // 获取当前输出参数的规格
    auto spec = outputSpecs_[i];
    // 根据规格创建张量选项
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    // 如果规格要求重用输入张量
    if (spec.reuses_input_tensor()) {
#ifdef GRAPH_DEBUG_ENABLED
      // 如果启用了图调试模式，输出相关调试信息
      GRAPH_DEBUG("inplace computation - input tensor would be reused");
#endif
      // 获取要重用的输入张量
      auto inputTensor = inputs[spec.get_input_tensor_index()];
      // 如果输入张量是 MKLDNN 张量
      if (inputTensor.is_mkldnn()) {
        // 获取数据类型
        auto dataType = spec.dtype();
        // 如果不使用不透明布局
        if (C10_UNLIKELY(!useOpaqueLayout(i))) {
#ifdef GRAPH_DEBUG_ENABLED
          // 如果启用了图调试模式，输出相关调试信息
          GRAPH_DEBUG("rewrap tensors");
#endif
          // 将输入张量转换为 LlgaTensorImpl 类型
          auto llgaImpl =
              static_cast<LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
          // 根据数据类型进行处理
          switch (dataType) {
            // 对于浮点数和半精度浮点数类型
            case data_type::f32:
            case data_type::bf16:
              // 将 LlgaTensorImpl 转换为 ATen 张量
              inputTensor = LlgaTensorImpl::llga_to_aten_tensor(llgaImpl);
              break;
            // 对于整型数据类型
            case data_type::s32:
            default:
              // 抛出异常，表示无效的数据类型
              TORCH_CHECK(
                  false, "Invalid data type ", static_cast<size_t>(dataType));
          }
        }
        // 将处理后的输入张量添加到输出列表中
        outputs.push_back(inputTensor);
        // 将处理后的输入张量添加到运行输出参数列表中
        runOutputs.push_back(
            {spec.logical_tensor(),
             Engine::getEngine(),
             inputTensor.data_ptr()});
        // 返回填充好的运行输入和输出参数列表
        return std::make_tuple(runInputs, runOutputs);
      }
    }
    // 如果使用不透明布局来处理第i个张量
    if (useOpaqueLayout(i)) {
        // 使用 LlgaTensorImpl 包装张量在分区之间，这样我们可以绕过守卫检查，
        // 因为步幅会与预期的不同。
#ifdef GRAPH_DEBUG_ENABLED
      // 如果开启了图形调试，记录调试信息到日志
      GRAPH_DEBUG("Between two oneDNN Graph partitions");
#endif
      // 使用指定规格和选项创建一个空的 LLGA 张量
      auto tensor = empty_llga(spec, opt);
      // 将创建的张量添加到输出列表中
      outputs.push_back(tensor);
      // 将 LLGA 张量转换为 ATen 张量并添加到运行时输出列表中
      runOutputs.push_back(llga_from_aten_tensor(tensor));
    } else {
#ifdef GRAPH_DEBUG_ENABLED
      // 如果开启了图形调试，记录调试信息到日志
      GRAPH_DEBUG("Neither opaque to PyTorch nor inplace-computation");
#endif
      // 根据指定规格创建一个空的 ATen 张量
      auto tensor = at::empty_strided(spec.sizes(), spec.strides(), opt);
      // 将创建的张量添加到输出列表中
      outputs.push_back(tensor);
      // 将张量的逻辑视图、引擎和数据指针封装为运行时输出的元组并添加到列表中
      runOutputs.push_back(
          {spec.logical_tensor(), Engine::getEngine(), tensor.data_ptr()});
    }
  }

  // 返回输入和输出的元组，用于运行时调用
  return std::make_tuple(runInputs, runOutputs);
}

compiled_partition LlgaKernel::compile(const partition& partition) {
  // 将输入规格列表转换为逻辑张量列表
  auto inputs = fmap(inputSpecs_, toLogicalTensor);
  // 将输出规格列表转换为逻辑张量列表
  auto outputs = fmap(outputSpecs_, toLogicalTensor);
  // 调用指定分区对象的编译方法，得到编译结果
  auto compilation = partition.compile(inputs, outputs, Engine::getEngine());

  // 根据编译结果更新输出规格列表中的逻辑张量信息
  for (const auto i : c10::irange(nOutputs_)) {
    auto tid = outputSpecs_[i].tid();
    outputSpecs_[i] = compilation.query_logical_tensor(tid);
  }

  // 根据可用的就地计算选项，构建输出 ID 到输入偏移量的静态映射
  for (auto&& option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    // 在输入规格列表中查找与输入 ID 匹配的规格
    auto inputSpecIter =
        std::find_if(inputSpecs_.begin(), inputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == inputId;
        });
    // 检查找到的输入规格是否存在
    TORCH_CHECK(inputSpecIter != inputSpecs_.end(), "In-place input not found");
    // 计算找到的输入规格在列表中的偏移量
    auto inputOffset = inputSpecIter - inputSpecs_.begin();
    // 在输出规格列表中查找与输出 ID 匹配的规格
    auto outputSpecIter =
        std::find_if(outputSpecs_.begin(), outputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == outputId;
        });
    // 计算找到的输出规格在列表中的偏移量
    auto outputOffset = outputSpecIter - outputSpecs_.begin();
    // 标记找到的输出规格支持就地计算
    outputSpecs_[outputOffset].set_compute_inplace();
    // 设置找到的输出规格的输入张量索引
    outputSpecs_[outputOffset].set_input_tensor_index(inputOffset);
  }

  // 返回编译结果对象
  return compilation;
}

void LlgaKernel::run(Stack& stack) {
#ifdef GRAPH_DEBUG_ENABLED
  // 如果开启了图形调试，记录调试信息到日志，包括调试名称
  GRAPH_DEBUG("In ", debugName(), "\n");
#endif

  // 从堆栈中获取输入值
  auto stackInputs = last(stack, nGraphInputs_);
  // 将输入值列表映射为张量类型的值，同时检查类型是否为张量
  auto inputs = fmap(stackInputs, [&](const IValue& v) {
    TORCH_CHECK(
        v.isTensor(), "Stack values for LLGA partition must be Tensor type");
    // 返回一个张量对象的 Tensor 表示形式
    return v.toTensor();
  });

  // 即使在并发线程情况下，内核也只会初始化一次。
  // TODO: 尝试不使用原子锁
  c10::call_once(
      initialized_flag,
      [&](const TensorArgs& inputs) {
        // 调试信息：初始化输入逻辑张量
        GRAPH_DEBUG("Initializing input logical tensors");
        // 使用输入参数初始化输入规格
        inputSpecs_ = initializeInputSpecs(inputs);
        // 调试信息：初始化输出逻辑张量
        GRAPH_DEBUG("Initializing output logical tensors");
        // 初始化输出规格
        outputSpecs_ = initializeOutputSpecs();
        // 调试信息：编译分区
        GRAPH_DEBUG("Compiling partition");
        // 编译给定分区
        compilation_ = compile(partition_);
        // 标记为已初始化
        is_initialized_ = true;
      },
      inputs);
#ifdef GRAPH_DEBUG_ENABLED
  // 如果启用了图形调试，则打印调试信息：准备运行时张量
  GRAPH_DEBUG("Preparing runtime tensors");
#endif

// 创建一个 TensorArgs 对象用于存储输出张量
TensorArgs outputs;

// 调用 prepareRunArgs 函数，准备运行时的输入和输出参数
auto [runInputs, runOutputs] = prepareRunArgs(inputs, outputs);

#ifdef GRAPH_DEBUG_ENABLED
  // 如果启用了图形调试，则打印调试信息：执行分区
  GRAPH_DEBUG("Executing partition");
#endif

// 调用 compilation_ 对象的 execute 方法，执行编译后的计算图分区
compilation_.execute(Stream::getStream(), runInputs, runOutputs);

#ifdef GRAPH_DEBUG_ENABLED
  // 如果启用了图形调试，则打印调试信息：分区执行完毕
  GRAPH_DEBUG("Partition executed");
#endif

// 更新堆栈内容，丢弃 nGraphInputs_ 个输入
drop(stack, nGraphInputs_);

// 遍历 outputs 中的每个元素，并将其逐个推送到堆栈中
for (auto& o : outputs)
  push_one(stack, std::move(o));

#ifdef GRAPH_DEBUG_ENABLED
  // 如果启用了图形调试，则打印调试信息：堆栈已更新
  GRAPH_DEBUG("Stack updated");
#endif
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```