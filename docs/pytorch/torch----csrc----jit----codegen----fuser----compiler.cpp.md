# `.\pytorch\torch\csrc\jit\codegen\fuser\compiler.cpp`

```py
// 引入Torch库中的头文件，用于混合编译器的相关功能
#include <torch/csrc/jit/codegen/fuser/compiler.h>

// 引入ATen库中的头文件
#include <ATen/ATen.h>
// 引入ATen核心类型相关的头文件
#include <ATen/core/jit_type.h>
// 引入C10异常处理相关的头文件
#include <c10/util/Exception.h>
// 引入C10中的范围迭代器相关的头文件
#include <c10/util/irange.h>
// 引入Torch库中的Fuser代码生成器的头文件
#include <torch/csrc/jit/codegen/fuser/codegen.h>
// 引入Torch库中Fuser接口定义的头文件
#include <torch/csrc/jit/codegen/fuser/interface.h>
// 引入Torch库中的内核缓存管理的头文件
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
// 引入Torch库中的张量描述相关的头文件
#include <torch/csrc/jit/codegen/fuser/tensor_desc.h>
// 引入Torch库中的IR节点的头文件
#include <torch/csrc/jit/ir/ir.h>
// 引入Torch库中的规范化处理相关的头文件
#include <torch/csrc/jit/passes/canonicalize.h>
// 引入Torch库中的形状分析相关的头文件
#include <torch/csrc/jit/passes/shape_analysis.h>
// 引入Torch库中运行时操作符相关的头文件
#include <torch/csrc/jit/runtime/operator.h>

// 引入C++标准库中的原子操作相关的头文件
#include <atomic>
// 引入C++标准库中输入输出流相关的头文件
#include <iostream>
// 引入C++标准库中智能指针相关的头文件
#include <memory>
// 引入C++标准库中字符串流相关的头文件
#include <sstream>
// 引入C++标准库中异常处理相关的头文件
#include <stdexcept>
// 引入C++标准库中字符串处理相关的头文件
#include <string>
// 引入C++标准库中元组相关的头文件
#include <tuple>
// 引入C++标准库中无序集合相关的头文件
#include <unordered_set>
// 引入C++标准库中实用工具相关的头文件
#include <utility>

// 定义匿名命名空间，用于实现私有的函数和变量
namespace {
    // 静态变量，用于控制Fusion后端锁
    std::mutex& fusionBackendLock() {
        static std::mutex fusion_backends_lock_{};
        return fusion_backends_lock_;
    }
} // namespace

namespace torch {
namespace jit {
namespace fuser {

// 静态函数，获取融合后端的映射表
static std::unordered_map<at::Device::Type, FusedKernelConstructor>&
getFusionBackends() {
    // 静态变量，存储融合后端的构造器映射表
    static std::unordered_map<at::Device::Type, FusedKernelConstructor>
        fusion_backends;
    return fusion_backends;
}

// 注册融合后端的函数，将构造器映射到对应的后端设备类型
void registerFusionBackend(
    at::Device::Type backend_type,
    FusedKernelConstructor ctor) {
  // 获取融合后端锁，确保线程安全
  std::lock_guard<std::mutex> guard(fusionBackendLock());
  // 将构造器注册到映射表中
  getFusionBackends()[backend_type] = std::move(ctor);
}

// 判断是否存在指定后端类型的融合后端
bool hasFusionBackend(at::Device::Type backend_type) {
  // 获取融合后端锁，确保线程安全
  std::lock_guard<std::mutex> guard(fusionBackendLock());
  // 判断映射表中是否存在指定后端类型的构造器
  return getFusionBackends().count(backend_type);
}

// 获取指定后端类型的融合构造器
static const FusedKernelConstructor& getConstructor(
    at::Device::Type backend_type) {
  // 获取融合后端锁，确保线程安全
  std::lock_guard<std::mutex> guard(fusionBackendLock());
  // 返回指定后端类型对应的融合构造器
  return getFusionBackends().at(backend_type);
}

// 计数器，记录编译的核函数数量，用于调试和生成任意的核函数名称
static std::atomic<size_t> next_kernel_id{0};
// 调试标志，用于控制融合操作的调试输出，默认为-1，表示未初始化
static int debug_fusion{-1};

// 返回编译的核函数数量
size_t nCompiledKernels() {
  return next_kernel_id.load();
}

// 获取融合操作的调试标志
int debugFuser() {
  if (debug_fusion < 0) {
    // 从环境变量中获取PYTORCH_FUSION_DEBUG的值，设置调试标志
    const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
    debug_fusion = debug_env ? atoi(debug_env) : 0;
  }
  return debug_fusion;
}

// 如果给定的节点被某个Chunk节点使用一次，则返回该节点，否则返回nullptr
static const Node* usedInFusedChunk(const Value* input) {
  // 获取输入值的使用情况
  const auto& uses = input->uses();
  if (uses.size() == 1) {
    const Node* user = uses[0].user;
    // 如果使用节点是ConstantChunk类型，则返回该节点
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  // 没有符合条件的使用情况，返回nullptr
  return nullptr;
}

// 设置输入张量的Chunk描述符
static void setInputChunkDescriptors(KernelSpec& spec) {
  // Chunk描述符数量与张量输入数量相同
  // 同时我们知道张量输入位于融合组输入的前面
  spec.inputChunks().reserve(spec.nTensorInputs());
  // 遍历张量输入的范围
  for (const auto i : c10::irange(spec.nTensorInputs())) {
    // 获取规范的图输入
    const Value* input = spec.graph()->inputs()[i];
    // ...
    // 如果调用 usedInFusedChunk(input) 返回非空指针，将其赋给常量指针 chunk
    if (const Node* chunk = usedInFusedChunk(input)) {
      // 在 spec 的 inputChunks 中添加一个新的元组，包含 chunk 的 attr::chunks 和 attr::dim 属性值
      spec.inputChunks().emplace_back(
          chunk->i(attr::chunks), chunk->i(attr::dim));
    } else {
      // 如果 usedInFusedChunk(input) 返回空指针，添加一个默认的元组 (1, 0) 到 spec 的 inputChunks 中
      spec.inputChunks().emplace_back(1, 0);
    }
  }
// 执行深度优先搜索（DFS）遍历，找出影响给定输出值的所有输入值
static std::vector<int64_t> getInputDependencies(const Value* output) {
  // 使用输出值初始化队列，该队列用于存储待处理的值
  std::vector<const Value*> queue{output};
  // 使用无序集合存储输入值，确保唯一性
  std::unordered_set<const Value*> inputs;
  // 使用无序集合记录已处理过的值，避免重复处理
  std::unordered_set<const Value*> seen;
  // 进行 BFS 遍历
  while (!queue.empty()) {
    // 取出队列中的最后一个值
    const Value* val = queue.back();
    queue.pop_back();
    // 获取值所属的节点
    const Node* producer = val->node();
    // 如果节点是 prim::Param 类型且值类型是 TensorType，则将其视为输入
    if (producer->kind() == prim::Param &&
        val->type()->isSubtypeOf(*TensorType::get())) {
      inputs.insert(val);
      continue;
    }
    // 遍历节点的所有输入值
    for (const Value* input : producer->inputs()) {
      // 如果输入值是第一次遇到，则加入队列准备处理
      if (seen.insert(input).second) {
        queue.push_back(input);
      }
    }
  }

  // 将 Value* 转换为图输入列表中的偏移量
  std::vector<int64_t> offsets;
  offsets.reserve(inputs.size());
  for (const Value* input : inputs) {
    offsets.push_back(input->offset());
  }

  // 对偏移量进行排序
  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

// 设置输入广播组
static void setInputBroadcastGroups(KernelSpec& spec) {
  // 使用自定义哈希函数的无序集合存储广播组
  std::unordered_set<std::vector<int64_t>, c10::hash<std::vector<int64_t>>>
      broadcast_groups;
  // 遍历图的输出值
  for (const Value* output : (spec.graph())->outputs()) {
    // 如果输出节点是 prim::FusedConcat 类型，则遍历其所有输入
    if (output->node()->kind() == prim::FusedConcat) {
      for (const Value* concat_input : output->node()->inputs()) {
        // 将每个连接输入的依赖添加到广播组中
        broadcast_groups.insert(getInputDependencies(concat_input));
      }
    } else {
      // 否则直接将输出值的依赖添加到广播组中
      broadcast_groups.insert(getInputDependencies(output));
    }
  }
  // 将广播组复制到 KernelSpec 的输入广播组中
  std::copy(
      broadcast_groups.begin(),
      broadcast_groups.end(),
      std::back_inserter(spec.inputBroadcastGroups()));
}

// 执行“预先”编译，其中存储已知但形状未知
// 目前识别如何扩展所有张量，使所有中间张量具有相同的形状，简化代码生成。
// 使用逻辑属性识别广播组和分块，不使用形状信息。
// 特别地，张量总是可以扩展到它们或其后代涉及的逐点操作的输出。
// 注意：由于连接和分块，逻辑略显复杂。
static void upfrontCompilation(KernelSpec& spec) {
  // 设置输入广播组
  setInputBroadcastGroups(spec);
  // 设置输入分块描述符
  setInputChunkDescriptors(spec);
}
// 注册融合操作，返回融合操作的唯一标识符
int64_t registerFusion(const Node* fusion_group) {
  // 标准化融合图以供缓存使用
  auto graph = normalizeGraphForCache(fusion_group->g(attr::Subgraph));

  // 如果存在预先存在的规范，则不重新注册融合操作
  const auto maybe_spec = lookupGraph(graph);
  if (maybe_spec) {
    return (*maybe_spec)->key();
  }

  // 无条件创建并注册融合操作
  // 这是为了支持全局禁用融合的标志：如果某人在无融合模式下运行了一些代码，
  // 然后在启用融合的模式下运行了一些代码，那么第二次从缓存返回的规范应该是有效的规范
  const auto key = store(graph);
  const auto maybe_retrieved_spec = retrieve(key);
  AT_ASSERT(maybe_retrieved_spec);
  upfrontCompilation(**maybe_retrieved_spec);

  return key;
}

// 编译融合内核
std::shared_ptr<FusedKernel> compileKernel(
    const KernelSpec& spec,
    const ArgSpec& arg_spec,
    const std::vector<int64_t>& map_size,
    const at::Device device) {
  const std::vector<TensorDesc>& input_desc = arg_spec.descs();

  // 复制规范图以便操作
  auto graph = spec.graph()->copy();

  // 对每个输入描述进行处理，设置输入节点的类型信息
  for (const auto i : c10::irange(input_desc.size())) {
    const auto& desc = input_desc[i];

    // TODO: 在切换到 ProfilingGraphExecutor 之前无法消除对 TensorType 的使用，
    // 因此我们需要运行 PropagateInputShapes 来处理
    graph->inputs()[i]->setType(TensorType::create(
        desc.scalar_type,
        device,
        {desc.nDim()},
        false)); // TODO: nDim 是不好的，因为它被合并了
  }

  // 传播输入形状变化
  PropagateInputShapes(graph);

  // 创建分块和扁平化输入描述
  std::vector<PartitionDesc> chunk_desc;
  std::vector<std::pair<const Value*, const std::optional<TensorDesc>>>
      flat_inputs;
  {
    size_t input_index = 0;
    for (const auto& p : graph->inputs()) {
      if (p->type()->isSubtypeOf(*FloatType::get())) {
        flat_inputs.emplace_back(p, c10::nullopt);
      }
      if (!p->type()->isSubtypeOf(*TensorType::get())) {
        continue;
      }
      // 如果节点用于融合块，则创建相关的分块描述和扁平化输入
      if (const Node* chunk = usedInFusedChunk(p)) {
        int64_t dim = chunk->i(attr::dim);
        int64_t chunks = chunk->i(attr::chunks);
        chunk_desc.emplace_back(input_desc[input_index++], chunks, dim);
        for (const auto* o : chunk->outputs()) {
          flat_inputs.emplace_back(o, *chunk_desc.back().subTensorDesc());
        }
      } else {
        chunk_desc.emplace_back();
        flat_inputs.emplace_back(p, input_desc[input_index++]);
      }
    }
  }

  // 创建输出、连接和扁平化输出描述
  std::vector<TensorDesc> output_desc;
  std::vector<PartitionDesc> concat_desc;
  std::vector<std::pair<const Value*, const TensorDesc>> flat_outputs;
  for (const Value* o : graph->outputs()) {
    // 创建输出描述
    std::vector<int64_t> sizes = map_size;
    if (o->node()->kind() == prim::FusedConcat) {
      sizes.at(o->node()->i(attr::dim)) *= o->node()->inputs().size();
    }
    // 获取张量的标量类型
    auto scalar_type = o->type()->expectRef<TensorType>().scalarType();
    // 断言标量类型存在
    TORCH_INTERNAL_ASSERT(scalar_type);
    // 创建连续张量类型
    auto type = TensorType::createContiguous(*scalar_type, device, sizes);
    // 将创建的输出描述添加到输出描述向量中
    output_desc.emplace_back(type);
    // 获取刚添加的输出描述的引用
    const auto& desc = output_desc.back();

    // 创建连接(concat)和扁平化(flattened)的输出描述（依赖于输出描述）
    if (o->node()->kind() != prim::FusedConcat) {
      // 如果节点类型不是融合连接，则在连接描述向量中添加空项
      concat_desc.emplace_back();
      // 在扁平化输出向量中添加（操作指针，输出描述）
      flat_outputs.emplace_back(o, desc);
    } else {
      // 如果节点类型是融合连接
      const auto cat = o->node();
      // 在连接描述向量中添加（输出描述，连接输入的数量，连接的维度）
      concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
      // 遍历连接节点的输入
      for (const auto& c : cat->inputs()) {
        // 在扁平化输出向量中添加（连接节点，子张量描述）
        flat_outputs.emplace_back(c, *concat_desc.back().subTensorDesc());
      }
    }
  }

  // 检查设备是否使用 CUDA
  const bool use_cuda = device.is_cuda();
  // 创建内核名称，格式为 "kernel_" + 下一个内核 ID 的字符串表示
  const std::string name = "kernel_" + std::to_string(next_kernel_id++);
  // 生成内核代码
  std::string code =
      generateKernel(name, *graph, flat_inputs, flat_outputs, use_cuda);
  // 获取内核构造函数
  const FusedKernelConstructor& kernel_ctor =
      getConstructor(use_cuda ? DeviceType::CUDA : DeviceType::CPU);
  // 返回内核构造函数的调用结果
  return kernel_ctor(
      device.index(),
      name,
      code,
      input_desc,
      output_desc,
      chunk_desc,
      concat_desc,
      spec.hasRandom());
} // End of namespace torch
} // End of namespace jit
} // End of namespace fuser
```