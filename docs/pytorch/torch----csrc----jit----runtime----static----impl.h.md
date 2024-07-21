# `.\pytorch\torch\csrc\jit\runtime\static\impl.h`

```
#pragma once
// 预处理指令：指定头文件只被包含一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 IValue 头文件
#include <ATen/core/symbol.h>
// 包含 ATen 库中的 Symbol 头文件
#include <c10/core/CPUAllocator.h>
// 包含 c10 库中的 CPUAllocator 头文件
#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义头文件
#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef 头文件
#include <c10/util/FbcodeMaps.h>
// 包含 c10 库中的 FbcodeMaps 头文件
#include <torch/csrc/jit/api/module.h>
// 包含 PyTorch JIT 库中的 module.h 头文件
#include <torch/csrc/jit/ir/graph_node_list.h>
// 包含 PyTorch JIT 库中的 graph_node_list.h 头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 PyTorch JIT 库中的 ir.h 头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
// 包含 PyTorch JIT 库中的 constant_propagation.h 头文件
#include <torch/csrc/jit/passes/freeze_module.h>
// 包含 PyTorch JIT 库中的 freeze_module.h 头文件
#include <torch/csrc/jit/passes/inliner.h>
// 包含 PyTorch JIT 库中的 inliner.h 头文件
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>
// 包含 PyTorch JIT 库中的 ProcessedNodeInputs.h 头文件
#include <torch/custom_class.h>
// 包含 PyTorch 自定义类相关的头文件
#include <limits>
// 包含标准库中的 limits 头文件

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
// 如果定义了 FBCODE_CAFFE2，包含 folly 库中的 F14Map 头文件
#include <folly/container/F14Set.h>
// 如果定义了 FBCODE_CAFFE2，包含 folly 库中的 F14Set 头文件
#endif

namespace torch::jit {
// 进入 torch::jit 命名空间

TORCH_API bool canEnableStaticRuntime(
    const std::shared_ptr<torch::jit::Graph>& graph);
// 声明 canEnableStaticRuntime 函数，用于判断是否可以启用静态运行时

TORCH_API std::string dumpValueSet(
    const c10::FastSet<const Value*>& value_set,
    const char* set_name = "");
// 声明 dumpValueSet 函数，用于将值集合转储为字符串

TORCH_API inline bool doesNotHeapAllocateWhenStoredInIValue(const Type& type) {
  // 内联函数定义，用于检查特定类型在存储于 IValue 中时是否不分配堆内存
  switch (type.kind()) {
    // 根据类型的种类进行判断
    case TypeKind::NoneType:
    case TypeKind::IntType:
    case TypeKind::FloatType:
    case TypeKind::BoolType:
    case TypeKind::DeviceObjType:
    case TypeKind::StreamObjType:
      return true;
      // 如果是以上列出的类型，返回 true
    default:
      return false;
      // 其他情况返回 false
  }
}

TORCH_API inline c10::Symbol getStaticRuntimeMetadataSymbol() {
  // 内联函数定义，用于获取静态运行时元数据的符号
  return Symbol::attr("static_runtime::metadata");
  // 返回特定的符号对象
}

TORCH_API inline bool borrowsOutputs(c10::Symbol kind) {
  // 内联函数定义，用于判断特定符号是否借用输出
  static const std::array<c10::Symbol, 4> symbols_with_borrowed_outputs = {
      c10::Symbol::fromQualString("static_runtime::select_tensor"),
      c10::Symbol::fromQualString("static_runtime::dict_unpack"),
      c10::Symbol::fromQualString("static_runtime::VarTupleUnpack"),
      c10::Symbol::fromQualString("prim::IfThenElse"),
  };
  // 静态数组，包含借用输出的符号列表
  return std::find(
             symbols_with_borrowed_outputs.begin(),
             symbols_with_borrowed_outputs.end(),
             kind) != symbols_with_borrowed_outputs.end();
  // 返回是否找到特定符号在借用输出符号列表中的结果
}

// Group values used by `graph` into three categories:
// 将由图 `graph` 使用的值分成三类：

// - output_aliases:
//     values that are either outputs or contain aliases of outputs
//     输出别名：
//     输出值或包含输出别名的值

// - external_aliases:
//     values that are inputs, constants, or their aliases.
//     The output aliases that end up here are as a result of aliasDb failing to
//     recognize them as outputs due to collection object (e.g., Tuple) aliasing
//     inputs.
//     外部别名：
//     输入值、常量或它们的别名。
//     最终出现在这里的输出别名是由于别名数据库未能识别它们为输出而导致的，原因是集合对象（例如 Tuple）别名输入。

// Values that dont't show up in output_aliases or external_aliases are created
// and consumed within the graph.
// 不在输出别名或外部别名中显示的值是在图中创建和消耗的。

class ValueGroup {
// ValueGroup 类的定义

 public:
  explicit ValueGroup() = default;
  // 显式默认构造函数

  void init(const Block& block, const AliasDb& db);
  // 初始化函数，用于初始化 ValueGroup 实例

  bool isExternalAlias(const Value* value) const {
    return external_aliases_.find(value) != external_aliases_.end();
  }
  // 检查值是否是外部别名的函数

  bool isOutputAlias(const Value* value) const {
    return output_aliases_.find(value) != output_aliases_.end();
  }
  // 检查值是否是输出别名的函数

  bool isAlwaysAlive(const Value* value) const {
  // 检查值是否总是存活的函数（未完整展示）
    return isExternalAlias(value) || isOutputAlias(value);
  }


    // 返回是否是外部别名或输出别名的逻辑或结果



  std::string toString() const {
    return c10::str(
        dumpValueSet(output_aliases_, "ValueGroup::output_aliases_"),
        "\n",
        dumpValueSet(external_aliases_, "ValueGroup::external_aliases_"));
  }


  // 返回该对象的字符串表示形式
  std::string toString() const {
    // 使用 c10::str 函数将输出别名和外部别名的集合转换为字符串，以换行符分隔
    return c10::str(
        dumpValueSet(output_aliases_, "ValueGroup::output_aliases_"), // 转储输出别名集合
        "\n",                                                          // 换行符
        dumpValueSet(external_aliases_, "ValueGroup::external_aliases_")); // 转储外部别名集合
  }



 private:
  c10::FastSet<const Value*> output_aliases_;
  c10::FastSet<const Value*> external_aliases_;


  // 私有成员变量，存储输出别名和外部别名的集合
  c10::FastSet<const Value*> output_aliases_;
  c10::FastSet<const Value*> external_aliases_;
};

// 管理张量生命周期的类
class TORCH_API ManagedTensorRanges {
 public:
  // 默认构造函数
  ManagedTensorRanges() = default;
  
  // 构造函数，初始化管理的张量范围
  ManagedTensorRanges(
      Block& block,                          // 区块引用
      const AliasDb& alias_db,               // 别名数据库引用
      const c10::FastSet<const Value*>& managed_tensor_values);  // 管理的张量值集合

  // 如果为真，表示该节点是至少一个管理张量的最后使用点。
  // 在此节点后可用的张量值将在availableTensorValuesAfterNode(node)中返回。
  bool nodeFreesManagedTensors(Node* node) const;

  // 返回节点后可用的张量值列表
  const std::vector<const Value*>& availableTensorValuesAfterNode(
      Node* node) const;

  // 用于测试。如果v1和v2都是可变类型，并且它们的生命周期重叠，则返回true。
  bool lifetimesOverlap(const Value* v1, const Value* v2) const;

 private:
  // 生命周期结构体，表示值的生命周期起始和结束位置
  struct Lifetime {
    Lifetime(size_t start_, size_t end_) : start(start_), end(end_) {}
    size_t start;   // 生命周期开始位置
    size_t end;     // 生命周期结束位置
  };

  // 获取值的生命周期，如果不跟踪该值的生命周期则返回nullptr
  Lifetime* getLifetime(const Value* value);

  // 获取值的生命周期（const版本）
  const Lifetime* getLifetime(const Value* value) const;

  // 收集所有具有跟踪生命周期的输入值
  // 如果值是图输入或不可变类型（至少有一个可变类型的容器是可变的），则可能不会跟踪其生命周期
  std::vector<const Value*> collectValuesWithTrackedLifetimes(
      at::ArrayRef<const Value*> values);

  // 将节点映射到在该节点之后可重用的管理张量集合
  c10::FastMap<Node*, std::vector<const Value*>> node_to_newly_free_tensors_{};

  // 将每个值映射到其生命周期（起始节点索引，结束节点索引）
  c10::FastMap<const Value*, Lifetime> value_lifetimes_{};
};
struct TORCH_API StaticModuleOptions {
  // 启用 out 变体允许静态运行时进行内存规划
  bool enable_out_variant{true};
  // 重用张量存储，对于生命周期不重叠的张量以减少内存占用（enable_out_variant 必须为 true）
  bool optimize_memory{true};
  // 批量分配图输出张量的张量存储，其中存储在静态运行时外部被释放（enable_out_variant 必须为 true）
  bool manage_output_tensors{false};
  // 控制 ReplaceWithCopy pass，该 pass 替换那些有时与其输出别名的操作为总是进行复制的 out 变体（以便输出能参与内存规划）
  // 由于替换为复制操作发生在 TensorExpr 融合之后，结果图不符合 fuser 的假设。
  // 因此，即使该标志被打开，如果启用了 TensorExpr 融合，ReplaceWithCopy pass 也不会被执行。
  bool use_copy_variants{true};
  // 控制 ReplaceWithMaybeCopy pass，该 pass 替换那些有时与其输出别名的操作为包含 out 变体的子图。
  // 由于与 use_copy_variants 相同的原因，即使该标志被打开，如果启用了 TensorExpr 融合，ReplaceWithMaybeCopy pass 也不会被执行。
  bool use_maybe_copy_variants{true};
  // 启用在模型加载时对操作进行 TensorExpr 融合
  bool enable_tensorexpr_fusion{false};
};

/*
  负责将 StaticRuntime 元数据插入 IR 节点中。StaticRuntimeMetadata 扩展了 CustomClassHolder，
  可以被转换为 IValue 并附加到 IR 节点上。
  在 prim::fork 运算符存在的情况下，这对于将父图元数据传递给分叉图是必要的。
*/
class TORCH_API StaticRuntimeMetadata : public torch::CustomClassHolder {
 public:
  explicit StaticRuntimeMetadata(const StaticModuleOptions& opts)
      : opts_(opts) {}

  const StaticModuleOptions& get_opts() {
    return opts_;
  }

 private:
  StaticModuleOptions opts_;
};

/// 静态运行时支持两种执行模式。
///
/// 模式 1：单线程执行，除了内部操作并行化外无并行性
/// 对于此模式，可以执行以下操作之一：
/// @code
///   // m 是一个 TorchScript 模块
///   auto module = StaticModule(m, opts);
///   auto output = module(args, kwargs);
/// @endcode
///
/// 或者
///
/// @code
///   // g 是一个 TorchScript 图
///   auto module = StaticModule(g, opts);
///   auto output = module(args, kwargs);
/// @endcode
///
/// 模式 2：类似于数据并行性，同时在不同线程上为不同输入运行相同的模型
/// 每个模型应有一个 StaticModule，并且每个运行线程应有一个 StaticRuntime 实例。
/// 为避免在运行时创建 StaticRuntime，可以使用同步栈（如 boost::lockfree::stack）缓存所有 StaticRuntime 实例。
/// @code
///   // 初始化
///   // 创建一个指向 StaticModule 对象的 shared_ptr，并使用 m 和 opts 进行初始化
///   auto module = std::make_shared<StaticModule>(m, opts);
///
///   // 创建一个大小为 128 的锁定自由栈，用于存储 StaticRuntime 的 shared_ptr
///   // fixed_sized<true> 表示栈的大小是固定的
///   boost::lockfree::stack<std::shared_ptr<StaticRuntime>,
///     boost::lockfree::fixed_sized<true>> pool(128);
///
///   // 推理
///   std::shared_ptr<StaticRuntime> runtime = nullptr;
///   pool.pop(runtime);  // 从栈中弹出一个 runtime 的 shared_ptr
///   if (!runtime) {
///     // 如果栈中没有可用的 runtime，则创建一个新的 SharedRuntime 对象
///     // 该对象通过复制 *module 初始化，负责自己的内存管理
///     runtime = std::make_shared<StaticRuntime>(*module);
///   }
///   // 调用 runtime 对象进行推理，args 和 kwargs 是传入的参数
///   auto output = runtime(args, kwargs);
///   pool.push(runtime);  // 将 runtime 的 shared_ptr 推回栈中
/// @endcode
///
class MemoryPlanner;
class StaticNodeInfo;
class ProcessedNode;
class StaticRuntime;

using SROperator = std::function<void(ProcessedNode*)>;

#ifdef FBCODE_CAFFE2
struct TORCH_API SROperatorObserver {
  using OperatorCallback = void (*)(const Node*);
  OperatorCallback startCb = nullptr;
  OperatorCallback endCb = nullptr;

  static void setCurrentThreadObserver(SROperatorObserver* observer);
  static SROperatorObserver* getCurrentThreadObserver();
  static void onStart(const Node* name);
  static void onEnd(const Node* name);
};
#endif

class TORCH_API ProcessedFunction {
 public:
  ProcessedFunction(
      Node* node,
      bool enable_out_variant,
      bool check_memory_overlap);

  enum class Kind : uint8_t {
    kOutVariant,
    kNativeFunction,
    kInterpreterFallback,
  };

  // 调用函数对象 f_ 处理 ProcessedNode，用于执行处理函数
  void run(ProcessedNode* pnode) const {
    return f_(pnode);
  }

  // 返回函数对象的种类 Kind
  Kind kind() const {
    return kind_;
  }

  // 返回是否检查内存重叠的标志
  bool checkMemoryOverlap() const {
    return check_memory_overlap_;
  }

  // 返回输出数量 num_outputs_
  size_t num_outputs() const {
    return num_outputs_;
  }

 private:
  SROperator f_;  // 函数对象，用于处理 ProcessedNode
  Kind kind_{ProcessedFunction::Kind::kOutVariant};  // 函数对象的种类，默认为 kOutVariant
  bool check_memory_overlap_{false};  // 是否检查内存重叠的标志，默认为 false
  size_t num_outputs_{0};  // 输出数量，默认为 0
};

// 一个 `BlockInfo` 实例存储每个 `BlockRunner` 需要访问的所有共享状态。
// 大多数信息是只读的，且在线程之间共享。
// - 每个 `BlockInfo` 对应图中的一个块。
// - 每个 `BlockInfo` 可能被多个块运行器使用（当有多个线程时）。
// - 所有的 `BlockInfo` 存储在 `StaticModule` 的向量中，并在 `StaticModule` 构造期间初始化。
// - 存储的大多数信息用于初始化块的内存规划器。
class BlockInfo {
 public:
  // 构造函数，初始化一个 BlockInfo 实例，指定输入索引和对应的块
  BlockInfo(uint32_t input_idx, Block& block);

  // 设置节点信息和节点是否有输出变体的快速映射
  void set_nodes(
      std::vector<StaticNodeInfo> nodes,
      const c10::FastMap<Node*, bool>& node_has_out_variant);

  // 返回节点信息的向量 nodes_
  const std::vector<StaticNodeInfo>& nodes() const {
    return nodes_;
  }

  // 返回节点数量
  size_t num_nodes() const;

  // 返回输入数量，即块的输入个数
  size_t num_inputs() const {
    return block_.inputs().size();
  }

  // 返回输出数量，即块的输出个数
  size_t num_outputs() const {
    return block_.outputs().size();
  }

  // 返回块的节点列表
  graph_node_list node_ptrs() const {
    return block_.nodes();
  }

  // 设置输出索引
  void set_output_indices(std::vector<uint16_t> indices) {
  // 将输入参数 indices 移动到 output_indices_ 中
  output_indices_ = std::move(indices);
}

// 返回 block 的输出索引列表
const std::vector<uint16_t>& block_output_indices() const {
  return output_indices_;
}

// 返回 block 的输入索引
auto block_inputs_idx() const {
  return input_idx_;
}

// 检查给定节点是否是可优化的容器类型节点
bool node_is_optimizable_container_type(const Node* node) const {
  return node_is_optimizable_container_type_.find(node) !=
      node_is_optimizable_container_type_.end();
}

// 检查给定值是否是托管的张量值
bool value_is_managed_tensor(const Value* value) const {
  return managed_tensor_values_.find(value) != managed_tensor_values_.end();
}

// 检查给定值是否是泄漏的容器值
bool value_is_leaked_container(const Value* value) const {
  return leaked_values_.find(value) != leaked_values_.end();
}

// 返回值分组对象的引用
const ValueGroup& value_group() const {
  return value_group_;
}

// 返回托管张量范围对象的引用
const ManagedTensorRanges& managed_tensor_ranges() const {
  return managed_tensor_ranges_;
}

// 使用给定的别名数据库初始化值分组对象
void init_value_group(const AliasDb& alias_db) {
  value_group_.init(block_, alias_db);
}

// 准备内存规划器所需的数据，但未提供具体实现
void prepare_for_memory_planner(
    const AliasDb& alias_db,
    const StaticModuleOptions& opt);

// 返回托管输出张量值集合的引用
const auto& managed_output_tensor_values() const {
  return managed_output_tensor_values_;
}

// 返回托管张量值集合的引用
const auto& managed_tensor_values() const {
  return managed_tensor_values_;
}

// 返回泄漏值集合的引用
const auto& leaked_values() const {
  return leaked_values_;
}
};

// 静态模块类定义
class TORCH_API StaticModule {
 public:
  // 构造函数：从图形对象创建静态模块
  explicit StaticModule(
      std::shared_ptr<torch::jit::Graph> g,
      const StaticModuleOptions& opts = StaticModuleOptions(),
      std::vector<IValue> sample_inputs = {});

  // 构造函数：从 Torch 模块创建静态模块
  explicit StaticModule(
      const torch::jit::Module& m,
      bool is_frozen = false,
      const StaticModuleOptions& opts = StaticModuleOptions(),
      std::vector<IValue> sample_inputs = {});

 private:
  // 私有构造函数：从图形和模块对象对创建静态模块
  explicit StaticModule(
      std::pair<std::shared_ptr<torch::jit::Graph>, std::optional<Module>>
          graph_and_module,
      const StaticModuleOptions& opts);

 public:
  // 关键字参数映射类型定义
  using KeywordArgs = std::unordered_map<std::string, c10::IValue>;

  // 函数调用运算符重载：接受常规参数和关键字参数，返回计算结果
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs());

  // 函数调用运算符重载：接受右值引用参数和关键字参数，返回计算结果
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs());

  // 返回图形对象的常量引用
  const Graph& graph() const {
    return *graph_;
  }

  // 返回模块对象的常量引用，需要确保模块对象存在
  const Module& module() const {
    DCHECK(module_.has_value());
    return *module_;
  }

  // 返回静态模块的选项对象的常量引用
  const StaticModuleOptions& opts() const;

  // 返回输入参数的数量
  size_t num_inputs() const;

  // 返回输出参数的数量
  size_t num_outputs() const;

  // 返回常量值的数量
  size_t num_constants() const {
    return constants_.size();
  }

  // 返回中间值的数量
  size_t num_intermediate_values() const {
    return num_intermediate_values_;
  }

  // 返回所有值的总数量（输入参数、常量和中间值的总和）
  size_t total_num_values() const {
    return num_inputs() + num_constants() + num_intermediate_values();
  }

  // 返回输出索引的常量引用
  C10_NODISCARD const std::vector<uint16_t>& output_indices() const {
    return output_indices_;
  }

  // 返回常量值数组的常量引用
  const std::vector<IValue>& constants() const {
    return constants_;
  }

  // 返回特定块的块信息引用
  const BlockInfo& block_info(Block* block) const {
    return block_infos_.at(block);
  }

  // 返回根块的常量指针
  Block* root_block() const {
    return graph_->block();
  }

 private:
  friend class StaticRuntime;
  friend class BlockRunner;

 public:
  // 返回节点数量，通过累加所有块的节点数量实现
  auto num_nodes() const {
    return std::accumulate(
        block_infos_.begin(),
        block_infos_.end(),
        0,
        [](size_t sum, const auto& block_and_info) {
          auto& block_info = block_and_info.second;
          return sum + block_info.num_nodes();
        });
  }

  // 查找具有指定种类的节点，用于测试目的
  C10_NODISCARD Node* findNodeWithKindForTesting(const std::string& kind) const;

  // 返回函数模式的可选引用
  const std::optional<c10::FunctionSchema>& schema() const {
    return schema_;
  }

  // 判断是否第一个输入参数是自身
  bool first_input_is_self() const {
    return module_.has_value();
  }

  // 返回静态运行时对象的引用
  StaticRuntime& runtime();

  // 查看共享值数组的大小
  size_t value_buffer_size() const {
  // 返回 value_buffer_size_ 的当前值
  return value_buffer_size_;
}

private:
  // 递归准备 BlockInfo 数组。
  // - 使用 value_to_index 填充每个中间值的索引
  // - 返回已处理的 Value* 数量，包括子块。
  size_t prepareBlockInfo(
      Block* block,
      const size_t start_idx,
      c10::FastMap<const Value*, uint32_t>& value_to_index);

  void prepareFunctionsAndConstants(
      Block* block,
      const AliasDb& alias_db,
      c10::FastMap<const Value*, uint32_t>& value_to_index);

  // 递归遍历图并在 prim::fork 节点上附加 SR 元数据作为附加属性
  void attachNodeMetadata(Block* block);

  // 递归处理子块并填充 ProcessedNodes 数组
  // 返回（处理的节点数，处理的块数）
  size_t prepareStaticNodeInfos(
      Block* block,
      const c10::FastMap<const Value*, uint32_t>& value_to_index,
      const AliasDb& alias_db,
      size_t node_idx = 0);

  // 初始化内存规划器所需的各种属性。
  // 在构造函数的末尾调用。
  void prepareForMemoryPlanner();

  StaticModuleOptions opts_;
  // 存储在 IR 节点中的元数据作为属性
  at::intrusive_ptr<jit::StaticRuntimeMetadata> sr_metadata_;
  std::shared_ptr<torch::jit::Graph> graph_;
  std::optional<torch::jit::Module> module_;
  std::optional<c10::FunctionSchema> schema_;
  std::unique_ptr<StaticRuntime> cached_runtime_;

  // 创建新的 StaticRuntime 实例的簿记
  // IValue 表（由 prim::Constant 节点定义）
  std::vector<IValue> constants_;
  // 对应于每个 ProcessedNode 要调用的函数
  std::vector<ProcessedFunction> functions_{};
  // 从中创建每个 StaticRuntime 实例的预处理节点列表
  std::vector<StaticNodeInfo> nodes_;
  // 图输出在单个值数组中的索引
  std::vector<uint16_t> output_indices_;

  size_t num_intermediate_values_ = 0;

  // 如果 module_ != nullopt，则包括 self。
  // 注意，即使 schema 包含 `self` 参数，num_inputs_ 可能为 0。
  // 在这种情况下，`self` 在图中未使用，但 schema 仍然包含它以与 JIT 解释器保持一致。
  size_t num_inputs_;
  // 参见 `BlockInfo` 定义。块按深度优先顺序存储。
  c10::FastMap<Block*, BlockInfo> block_infos_;
  // value_buffer_size_ 的大小
  size_t value_buffer_size_ = 0;
};

// `BlockRunner` 包含核心运行时逻辑。每个 `BlockRunner` 对应于图中的一个块，具有自己的内存规划器。
// `StaticRuntime` 在构造时会初始化所有 `BlockRunner`。
// 每个 `BlockRunner` 仅直接执行其块中的节点。具有子块（如 `prim::If`）的特殊操作可能在其 `ProcessedNode` 中存储 `BlockRunner`；
// 这些子块在操作的实现中执行。
// `StaticRuntime` 存储一个所有 `BlockRunner` 共享的 IValues 向量。
// 此向量用于存储所有常量、输入和中间张量。
class TORCH_API BlockRunner {
 public:
  // 构造函数，初始化 `BlockRunner` 对象
  BlockRunner(
      const StaticModule& sm,         // 静态模块引用
      IValue* values,                 // IValue 类型的值数组
      Block* block,                   // 块对象指针
      torch::jit::TaskLauncher* launcher,  // TaskLauncher 对象指针
      bool is_root_block = false);    // 是否为根块的标志

  // 移动构造函数
  BlockRunner(BlockRunner&&) noexcept;

  // 禁用赋值运算符
  BlockRunner& operator=(BlockRunner&&) = delete;

  // 析构函数
  ~BlockRunner();

  // 运算符重载，用于同步执行操作
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,   // 输入参数向量
      const KeywordArgs& kwargs = KeywordArgs());  // 关键字参数

  // 运算符重载，用于异步执行操作
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,         // 输入参数向量（移动语义）
      const KeywordArgs& kwargs = KeywordArgs());  // 关键字参数

  // 异步运行操作，返回 Future 对象指针
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      const std::vector<c10::IValue>& args,   // 输入参数向量
      const KeywordArgs& kwargs);             // 关键字参数

  // 异步运行操作，返回 Future 对象指针（移动语义）
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      std::vector<c10::IValue>&& args,        // 输入参数向量（移动语义）
      const KeywordArgs& kwargs);             // 关键字参数

  // 对操作进行基准测试
  void benchmark(
      const std::vector<std::vector<c10::IValue>>& args_list,   // 输入参数向量的向量
      const std::vector<KeywordArgs>& kwargs_list,              // 关键字参数的向量
      const uint32_t warmup_runs,                               // 预热运行次数
      const uint32_t main_runs,                                 // 主要运行次数
      bool print_per_node_time = false,                         // 是否打印每个节点的时间
      bool generate_ai_pep_output = false);                     // 是否生成 AI PEP 输出

  // 每个操作的个体度量指标结构体
  struct IndividualMetrics {
    float setup_time{0.0};                    // 设置时间
    float memory_alloc_time{0.0};             // 内存分配时间
    float memory_dealloc_time{0.0};           // 内存释放时间
    float output_dealloc_time{0.0};           // 输出释放时间
    float first_iter_time{0.0};               // 首次迭代时间
    float total_time{0.0};                    // 总时间
    size_t out_nodes_count{0};                // 输出节点计数
    size_t total_nodes_count{0};              // 总节点计数
    std::vector<float> time_per_node;         // 每个节点的时间
    std::unordered_map<std::string, float> time_per_node_type;     // 每种节点类型的时间
    std::unordered_map<std::string, float> percent_per_node_type;  // 每种节点类型的百分比
    std::unordered_map<std::string, int> instances_per_node_type;  // 每种节点类型的实例数
    std::unordered_set<std::string> out_nodes;                      // 输出节点集合
    std::unordered_set<std::string> native_nodes;                   // 本地节点集合
  };

  // 测量每个操作的个体度量指标
  IndividualMetrics benchmark_individual_ops(
      const std::vector<std::vector<c10::IValue>>& args_list,   // 输入参数向量的向量
      const std::vector<KeywordArgs>& kwargs_list,              // 关键字参数的向量
      const uint32_t warmup_runs,                               // 预热运行次数
      const uint32_t main_runs);                                // 主要运行次数

  // 获取输入值（可读写）
  IValue& Input(uint32_t i) {
    TORCH_DCHECK_LT(i, block_info_.num_inputs());                // 确保索引 i 小于输入数量
    return values_[i + block_info_.block_inputs_idx()];          // 返回对应索引的值
  }

  // 获取输出值（只读），写入过程在 ProcessedNodes 内进行
  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < outputs_.size());                                // 断言确保索引 i 小于输出数量
    ...
  // 返回第i个输出的指针
  return *outputs_[i];
}

const std::vector<IValue*> outputs() const {
  // 返回输出向量的副本
  return outputs_;
}

const std::vector<ProcessedNode>& nodes() const {
  // 返回节点向量的常量引用
  return nodes_;
}

std::vector<ProcessedNode>& nodes() {
  // 返回节点向量的非常量引用，可用于修改节点
  return nodes_;
}

graph_node_list node_ptrs() const {
  // 返回块信息中节点指针的列表
  return block_info_.node_ptrs();
}

const Graph& graph() const {
  // 返回静态模块的图
  return static_module_.graph();
}

const MemoryPlanner* get_memory_planner() const {
  // 返回内存规划器的指针
  return planner_.get();
}

bool check_for_memory_leak(
    bool output_returned = true,
    bool recurse_on_sub_blocks = false);

// 警告：释放管理的输出张量。接收静态运行时管理的张量的客户端需要非常小心，
// 在所有输出张量的引用都消失后调用 `StaticRuntime::deallocateOutputTensors`。
void deallocateOutputTensors();

bool checkOutputTensorMemoryLeaks();

bool isManagedOutputTensor(const IValue& ivalue) const;
bool isManagedOutputTensorValue(const Value* value) const;

void disableManageOutputTensors();

// 如果我们无法在第一次迭代中构造内存规划器，则使用此回退路径。
// 重要提示：此处不应该抛出任何异常！！！
// 此函数可以从 `noexcept` 析构函数中调用（隐式），这意味着如果有任何异常逃逸，
// 将调用 `std::terminate`。即使 resetMemory 和 ~Deallocator 不是 `noexcept(false)`，
// 当调用 ~Deallocator 时，堆栈可能已经在展开，因此仍存在调用 std::terminate 的危险。
void resetMemory() noexcept;

private:
// 在析构时调用内存规划器的清理代码的辅助对象。
class Deallocator {
 public:
  explicit Deallocator(BlockRunner& block_runner)
      : block_runner_(block_runner) {}

  Deallocator(Deallocator&&) = default;
  Deallocator(const Deallocator&) = default;
  Deallocator& operator=(const Deallocator&) = delete;
  Deallocator& operator=(Deallocator&&) = delete;
  ~Deallocator();

  void setFinished() {
    // 设置完成标志
    finished_ = true;
  }

 private:
  // 执行清理操作的内部实现
  void cleanupImpl();

  // 完成标志，表示清理是否已完成
  bool finished_ = false;
    // 定义一个引用成员变量 block_runner_，类型为 BlockRunner&
    BlockRunner& block_runner_;
    
    // 定义一个模板函数 run_impl，接受一个 IValueList 类型的参数 args 和一个关键字参数 kwargs，返回一个 c10::IValue 类型的值
    template <typename IValueList>
    c10::IValue run_impl(IValueList&& args, const KeywordArgs& kwargs);
    
    // 定义一个模板函数 run_impl_record_functions，接受一个 IValueList 类型的参数 args 和一个关键字参数 kwargs，
    // 返回一个 c10::IValue 类型的值，用于记录函数运行过程中的函数
    template <typename IValueList>
    c10::IValue run_impl_record_functions(
        IValueList&& args,
        const KeywordArgs& kwargs);
    
    // 定义一个模板函数 run_impl_async，接受一个 IValueList 类型的参数 args 和一个关键字参数 kwargs，
    // 返回一个 intrusive_ptr<c10::ivalue::Future> 类型的指针，用于异步运行
    template <typename IValueList>
    c10::intrusive_ptr<c10::ivalue::Future> run_impl_async(
        IValueList&& args,
        const KeywordArgs& kwargs);
    
    // 定义一个模板函数 run_impl_record_functions_async，接受一个 IValueList 类型的参数 args 和一个关键字参数 kwargs，
    // 返回一个 intrusive_ptr<c10::ivalue::Future> 类型的指针，用于异步记录函数运行过程中的函数
    template <typename IValueList>
    c10::intrusive_ptr<c10::ivalue::Future> run_impl_record_functions_async(
        IValueList&& args,
        const KeywordArgs& kwargs);
    
    // 辅助方法，将输入的 args 和 kwargs 复制到 inputs_ 中
    template <typename IValueList>
    void set_inputs(IValueList&& args, const KeywordArgs& kwargs);
    
    // 将第 idx 个输入设置为 args[idx]，使用右值引用。由 set_inputs 调用，根据重载进行复制或移动操作
    void set_arg(const size_t idx, std::vector<IValue>&& args);
    
    // 将第 idx 个输入设置为 args[idx]，使用左值引用。由 set_inputs 调用，进行复制操作
    void set_arg(const size_t idx, const std::vector<IValue>& args);
    
    // 将第 idx 个输入设置为 arg，始终进行复制操作。用于处理关键字参数 kwargs
    void set_arg(const size_t idx, const IValue& arg);
    
    // 快速检查并校正与 ProcessedNode n 中的 tensor_ival 的内存重叠
    bool fast_check_and_correct_overlap_with(
        ProcessedNode& n,
        c10::IValue& tensor_ival);
    
    // 验证并校正 ProcessedNode n 的内存重叠情况
    void verify_and_correct_memory_overlap(ProcessedNode& n);
    
    // 清理输入 IValues 的所有权引用，使用 noexcept 标记表示不抛出异常
    void clean_up_input_ivalues() noexcept {
      // 迭代所有输入的索引，从 inputs_begin_ 开始
      for (const auto idx : c10::irange(block_info_.num_inputs())) {
        // 将 values_ 数组中对应位置的元素设为默认构造的 IValue，即清空所有输入的值
        values_[idx + inputs_begin_] = IValue();
      }
    }
  }
}

void clean_up_intermediate_ivalues() noexcept;

IValue move_outputs_to_tuple(uint32_t num_outputs);

void create_memory_planner();

float benchmark_model(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const uint32_t warmup_runs,
    const uint32_t main_runs);

void display_nodes(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs);

const StaticModule& static_module_;
const BlockInfo& block_info_;

const bool is_root_block_;
// 缓存这个值，避免多次调用 static_module_.first_input_is_self()
const bool first_input_is_self_;
// 这个块在共享的 values_ 数组中输入的起始索引
const uint16_t inputs_begin_;

bool manage_output_tensors_enabled_ = false;
std::unique_ptr<MemoryPlanner> planner_;
// [共享的 values_ 数组]
// ProcessedNode 使用偏移量引用它们在这个数组中的输入和输出，
// 这样可以节省内存。
// 所有的 BlockRunner 共享同一个数组。布局如下：
// [常量][block_0][block_1]...[block_N]
// 注意，所有块的常量都汇总在一起放置在最前面。
// 块的顺序是深度优先的。
// 每个块进一步分为输入和中间值：
// [block_i] = [inputs_i][intermediates_i]
// 每个 BlockRunner 知道它的输入从哪里开始。每个 ProcessedNode
// 知道如何在这个数组中找到它的输出和输入的索引。
IValue* values_;

std::vector<IValue*> outputs_;
std::vector<ProcessedNode> nodes_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// StaticNodeInfo 类用于存储静态节点的信息，包括节点指针、处理函数、节点输入、输出偏移量
class TORCH_API StaticNodeInfo {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，初始化静态节点信息
  StaticNodeInfo(
      Node* n,  // 节点指针
      ProcessedFunction* fn,  // 处理函数指针
      ProcessedNodeInputs inputs,  // 节点输入
      uint16_t outputs_offset);  // 输出偏移量

  Node* node() const {  // 返回节点指针
    return node_;
  }

  size_t num_outputs() const {  // 返回节点输出数量
    DCHECK(fn_ != nullptr);  // 断言处理函数不为空
    return fn_->num_outputs();  // 返回处理函数的输出数量
  }

  bool has_out_variant() const {  // 判断是否具有输出变体
    return fn_->kind() == ProcessedFunction::Kind::kOutVariant;  // 判断处理函数种类是否为输出变体
  }

 private:
  friend class ProcessedNode;

  Node* node_;  // 节点指针
  const ProcessedFunction* fn_;  // 处理函数指针
  ProcessedNodeInputs inputs_;  // 节点输入
  uint16_t outputs_offset_;  // 输出偏移量
};

// BlockInfo 类的内联函数，返回节点数量
inline size_t BlockInfo::num_nodes() const {
  return nodes_.size();  // 返回节点列表的大小
}

/*
  ProcessedNodeMetadata 类封装了 ProcessedNode 的元数据，根据操作的性质，
  ProcessedNode 可以有以下几种元数据可能性：
  - prim::If/prim::Loop 操作包含 block_runners_ 作为其元数据
  - prim::fork 操作包含 TaskLauncher (std::function)，负责执行分叉子图
*/
class TORCH_API ProcessedNodeMetadata {
 public:
  ProcessedNodeMetadata(
      std::vector<BlockRunner> runners,  // 块运行器向量
      torch::jit::TaskLauncher* launcher)  // 任务启动器指针
      : block_runners_(std::move(runners)), launcher_(launcher) {}  // 构造函数，初始化元数据

  ProcessedNodeMetadata() : launcher_(nullptr) {}  // 默认构造函数，初始化启动器为空指针

  // deleted copy ctor/assignment as standard containers (vector) always
  // have copy constructors, but their instantiation is not well-formed
  // if the contained type (BlockRunner) is not copyable
  ProcessedNodeMetadata(const ProcessedNodeMetadata&) = delete;  // 删除复制构造函数
  ProcessedNodeMetadata& operator=(const ProcessedNodeMetadata&) = delete;  // 删除复制赋值运算符

  std::vector<BlockRunner>& block_runners() {  // 返回块运行器向量的引用
    return block_runners_;
  }

  void set_block_runners(std::vector<BlockRunner> runners) {  // 设置块运行器向量
    block_runners_ = std::move(runners);
  }

  void set_launcher(torch::jit::TaskLauncher* launcher) {  // 设置任务启动器指针
    launcher_ = launcher;
  }

  torch::jit::TaskLauncher* launcher() {  // 返回任务启动器指针
    return launcher_;
  }

 private:
  std::vector<BlockRunner> block_runners_;  // 块运行器向量
  torch::jit::TaskLauncher* launcher_;  // 任务启动器指针
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 定义一个名为 ProcessedNode 的类，用于表示处理过的节点
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 默认构造函数，使用默认参数初始化，禁止 Lint 工具检查
  ProcessedNode() = default;

  // 构造函数，从 StaticNodeInfo 和 IValue* 对象构造 ProcessedNode
  ProcessedNode(const StaticNodeInfo& other, IValue* values)
      : node_(other.node_),
        fn_(other.fn_),
        inputs_(other.inputs_),
        outputs_offset_(other.outputs_offset_),
        values_(values),
        metadata_(nullptr) {}

  // 移动构造函数，默认生成，用于支持对象的移动语义
  // 复制构造函数禁用，不允许对象的复制
  ProcessedNode(ProcessedNode&&) = default;
  
  // 赋值操作符重载，支持对象的移动赋值
  ProcessedNode& operator=(ProcessedNode&&) = default;
  // 复制赋值操作符禁用，不允许对象的复制赋值
  ProcessedNode(const ProcessedNode&) = delete;
  ProcessedNode& operator=(const ProcessedNode&) = delete;

  // 返回节点指针 node_
  Node* node() const {
    return node_;
  }

  // 返回第 i 个输入的值，是只读的
  // 使用 values_ 数组根据 inputs_ 的索引来获取对应的 IValue 对象
  C10_NODISCARD const IValue& Input(uint32_t i) const {
    return values_[inputs_[i]];
  }

  // 返回第 i 个输出的值，可以进行读写操作
  // 使用 values_ 数组根据 outputs_offset_ 的偏移量和索引来获取对应的 IValue 对象
  IValue& Output(uint32_t i) {
    DCHECK(i < num_outputs());
    return values_[outputs_offset_ + i];
  }

  // 返回第 i 个输出的值，是只读的
  // 使用 values_ 数组根据 outputs_offset_ 的偏移量和索引来获取对应的 IValue 对象
  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < num_outputs());
    return values_[outputs_offset_ + i];
  }

  // 返回节点的输出数量
  uint32_t num_outputs() const {
    DCHECK(fn_ != nullptr);
    return static_cast<uint32_t>(fn_->num_outputs());
  }

  // 返回节点的输出的 ArrayRef，用于访问输出的 IValue 数组
  C10_NODISCARD c10::ArrayRef<const IValue> outputs() const {
    return c10::ArrayRef<const IValue>(
        values_ + outputs_offset_, num_outputs());
  }

  // 返回节点的输入数量
  C10_NODISCARD uint16_t num_inputs() const {
    return inputs_.size();
  }

  // 返回节点的输入作为 std::vector<IValue> 的副本
  std::vector<IValue> inputs_ivalue_vec() const;

  // 检查节点是否包含 out variant
  bool has_out_variant() const {
    return fn_->kind() == ProcessedFunction::Kind::kOutVariant;
  }

  // 检查节点是否具有本地函数
  bool has_native() const {
    return fn_->kind() == ProcessedFunction::Kind::kNativeFunction;
  }

  // 如果没有禁用每个操作的分析，则返回操作名称
  // 返回节点的操作名称字符串
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* get_op_name() const {
    return node_->kind().toQualString();
  }
#endif

  // 检查节点输出是否存在内存重叠
  bool check_outputs_for_memory_overlap() const {
    return fn_->checkMemoryOverlap();
  }

  // 标记节点的输出存在内存重叠
  void set_outputs_memory_overlap_detected() {
    overlap_detected_ = true;
  }

  // 返回节点的输出是否检测到内存重叠
  bool outputs_memory_overlap_detected() {
    return overlap_detected_;
  }

  // 检查并校正节点输出与输入张量之间的内存重叠
  bool check_and_correct_overlap_with(
      const at::Tensor& input,
      c10::IValue& output);
  void verify_and_correct_memory_overlap();

  // 设置节点的值数组
  void set_values(IValue* values) {
    DCHECK(values_ == nullptr);
    values_ = values;
  }

  // 返回第 i 个输出的索引，使用于 IValue 数组的访问
  C10_NODISCARD uint16_t output_ivalue_index(uint16_t i) const {
    DCHECK(i < num_outputs());
    return outputs_offset_ + i;
  }

  // 用于调试模式，验证节点之间没有内存重叠
  bool verify_no_memory_overlap(bool force_check = false) const;

  // 返回指向 ProcessedNodeMetadata 的指针，如果没有则返回 nullptr
  // 返回指向 ProcessedNodeMetadata 对象的指针，如果没有则返回 nullptr
  ProcessedNodeMetadata* metadata() {
    return metadata_.get();
  }

  // 为 ProcessedNode 的元数据附加 block_runner
  // 将 block_runner 添加到 ProcessedNode 的元数据中
  void set_metadata(std::vector<BlockRunner> block_runners) {
    // 如果 metadata_ 指针为空，创建一个新的 ProcessedNodeMetadata 对象
    if (metadata_ == nullptr) {
      metadata_ = std::make_unique<ProcessedNodeMetadata>();
    }
    // 将 block_runners 移动到 metadata_ 指针指向的对象中
    metadata_->set_block_runners(std::move(block_runners));
  }

  // 将 TaskLauncher 对象附加到 ProcessedNode 的 metadata 中
  void set_metadata(torch::jit::TaskLauncher* launcher) {
    // 如果 metadata_ 指针为空，创建一个新的 ProcessedNodeMetadata 对象
    if (metadata_ == nullptr) {
      metadata_ = std::make_unique<ProcessedNodeMetadata>();
    }
    // 将 TaskLauncher 对象设置到 metadata_ 指针指向的对象中
    metadata_->set_launcher(launcher);
  }

 private:
  // 声明一个私有函数 verify_outputs_dont_overlap_each_other，用于检查输出是否重叠
  C10_NODISCARD bool verify_outputs_dont_overlap_each_other() const;

  // 声明一个私有函数 verify_inputs_dont_overlap_outputs，用于检查输入和输出是否重叠
  C10_NODISCARD bool verify_inputs_dont_overlap_outputs(bool force_check) const;

  Node* node_;  // 指向 Node 对象的指针
  const ProcessedFunction* fn_;  // 指向 const ProcessedFunction 对象的指针
  ProcessedNodeInputs inputs_;  // ProcessedNodeInputs 对象
  uint16_t outputs_offset_;  // 用于表示输出的偏移量的无符号 16 位整数
  bool overlap_detected_{false};  // 表示是否检测到重叠的布尔值，默认为 false
  IValue* values_ = nullptr; // unowned  // 指向 IValue 对象的指针，未拥有所有权
  // ProcessedNode 的元数据
  // 1. prim::If/Loop 节点包含子块作为元数据
  // 2. prim::fork 节点包含自定义执行器用于异步执行
  std::unique_ptr<ProcessedNodeMetadata> metadata_;  // 指向 ProcessedNodeMetadata 对象的独占指针
};

// `StaticRuntime` is the owner of the array of IValues (used for constants,
// inputs, and intermediate tensors) that all `BlockRunner`s share.
// Upon construction, it initializes all block runners. `operator()` simply
// forwards the inputs to the top-level block runner. Each `StaticRuntime`
// instance corresponds to one `StaticModule`. Multiple `StaticRuntime`
// instances can be created; this is useful for multi-threaded execution, since
// `operator()` is not thread-safe.
// `StaticRuntime` 类负责管理一组 IValue 数组，这些数组用于存储常量、输入和中间张量，所有的 `BlockRunner` 共享这些数据。
// 在构造时，它初始化所有的块运行器。`operator()` 简单地将输入转发给顶层块运行器。
// 每个 `StaticRuntime` 实例对应一个 `StaticModule`。可以创建多个 `StaticRuntime` 实例；这对于多线程执行很有用，因为 `operator()` 不是线程安全的。

class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(const StaticModule& sm);

  using KeywordArgs = std::unordered_map<std::string, c10::IValue>;

  // Executes the static module with given arguments and keyword arguments.
  // 同步执行静态模块，使用给定的参数和关键字参数。
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs());

  // Moves and executes the static module with given arguments and keyword arguments.
  // 移动并执行静态模块，使用给定的参数和关键字参数。
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs());

  // Runs the static module asynchronously.
  // 在调用线程上内联执行图，或在任务启动器上异步执行。
  // 如果未指定自定义的任务启动器，则在 inter-op 线程池上执行。
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs(),
      torch::jit::TaskLauncher taskLauncher = at::launch);

  // Moves and runs the static module asynchronously.
  // 移动并异步执行静态模块，使用给定的参数和关键字参数。
  // 如果未指定自定义的任务启动器，则在 inter-op 线程池上执行。
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs(),
      torch::jit::TaskLauncher taskLauncher = at::launch);

  // Checks for memory leaks related to the execution of the static module.
  // 检查与静态模块执行相关的内存泄漏。
  bool check_for_memory_leak(bool output_returned = true);

  // Checks for memory leaks in output tensors.
  // 检查输出张量的内存泄漏。
  bool checkOutputTensorMemoryLeaks();

  // Deallocates output tensors managed by the static runtime.
  // 释放静态运行时管理的输出张量。
  void deallocateOutputTensors();

  // Checks if an IValue is a managed output tensor.
  // 检查一个 IValue 是否是管理的输出张量。
  bool isManagedOutputTensor(const IValue& ivalue) const;

  // Disables the management of output tensors by the static runtime.
  // 禁用静态运行时对输出张量的管理。
  void disableManageOutputTensors();

  // Gets the top-level memory planner for testing purposes.
  // 获取顶层内存规划器，用于测试目的。
  const MemoryPlanner* get_memory_planner() const;

  // Performs benchmarking of the static module.
  // 执行静态模块的基准测试。
  void benchmark(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const uint32_t warmup_runs,
      const uint32_t main_runs,
      bool print_per_node_time = false,
      bool generate_ai_pep_output = false);

  // Performs benchmarking of individual operations within the static module.
  // 执行静态模块中各个操作的基准测试。
  IndividualMetrics benchmark_individual_ops(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const int warmup_runs,
      const int main_runs);

 private:
  // An array of IValues with unchanging size/data ptr.
  // 不变大小和数据指针的 IValues 数组。
  class IValueArray {
   public:
    IValueArray() = default;
    explicit IValueArray(size_t size) : array_(allocate(size)), size_(size) {}

    // Returns the pointer to the data array.
    // 返回数据数组的指针。
    IValue* data() const {
      return array_.get();
    }

    // Returns the size of the data array.
    // 返回数据数组的大小。
    size_t size() const {
      return size_;
    }

   private:
    // Allocates memory for the data array.
    // 为数据数组分配内存。
    std::unique_ptr<IValue[]> array_;
    size_t size_;
    // 分配指定大小的 IValue 数组的静态方法
    static std::unique_ptr<IValue[]> allocate(size_t size) {
      // 如果 size 大于 0，使用 make_unique 创建一个大小为 size 的 IValue 数组的智能指针
      if (size) {
        return std::make_unique<IValue[]>(size);
      }
      // 如果 size 为 0，返回空指针
      return nullptr;
    }
    
    // 用于存储 IValue 数组的智能指针，初始值为 nullptr
    std::unique_ptr<IValue[]> array_ = nullptr;
    
    // 存储数组大小的变量，初始值为 0
    size_t size_ = 0;
    };
    
    // 用于执行异步操作的 BlockRunner 对象的智能指针
    std::unique_ptr<BlockRunner> block_;
    
    // 用于在图中执行异步操作的 Torch JIT 任务启动器
    torch::jit::TaskLauncher async_task_launcher_;
    
    // 存储 IValue 数组的对象，可能用于传递给异步任务
    IValueArray values_;
};

// 结束 torch::jit 命名空间的定义
} // namespace torch::jit
```