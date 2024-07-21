# `.\pytorch\torch\csrc\autograd\function.h`

```
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <torch/csrc/autograd/anomaly_mode.h>
// 包含异常模式相关的头文件
#include <torch/csrc/autograd/edge.h>
// 包含边缘相关的头文件
#include <torch/csrc/autograd/grad_mode.h>
// 包含梯度模式相关的头文件
#include <torch/csrc/autograd/graph_task.h>
// 包含图任务相关的头文件
#include <torch/csrc/autograd/input_metadata.h>
// 包含输入元数据相关的头文件
#include <torch/csrc/autograd/saved_variable.h>
// 包含保存变量相关的头文件
#include <torch/csrc/autograd/variable.h>
// 包含变量相关的头文件
#include <torch/csrc/utils/python_stub.h>
// 包含 Python 存根相关的头文件
#include <torch/csrc/utils/variadic.h>
// 包含可变参数相关的头文件

#include <ATen/SequenceNumber.h>
// 包含序列号相关的头文件
#include <ATen/core/Tensor.h>
// 包含张量相关的核心头文件
#include <ATen/record_function.h>
// 包含记录函数相关的头文件
#include <c10/util/Exception.h>
// 包含异常相关的头文件
#include <c10/util/irange.h>
// 包含整数范围相关的头文件

#include <algorithm>
// 包含算法相关的头文件
#include <cstdint>
// 包含固定宽度整数类型相关的头文件
#include <initializer_list>
// 包含初始化列表相关的头文件
#include <memory>
// 包含内存管理相关的头文件
#include <string>
// 包含字符串相关的头文件
#include <utility>
// 包含实用工具相关的头文件
#include <vector>
// 包含向量相关的头文件

namespace torch::autograd {
// 进入 torch::autograd 命名空间

struct Edge;
// 声明结构体 Edge
struct FunctionPostHook;
// 声明结构体 FunctionPostHook
struct FunctionPreHook;
// 声明结构体 FunctionPreHook

using tensor_list = std::vector<at::Tensor>;
// 定义 tensor_list 为 at::Tensor 的向量类型
using variable_list = std::vector<Variable>;
// 定义 variable_list 为 Variable 的向量类型
using edge_list = std::vector<Edge>;
// 定义 edge_list 为 Edge 的向量类型
using saved_variable_list = std::vector<SavedVariable>;
// 定义 saved_variable_list 为 SavedVariable 的向量类型
using IndexRange = std::pair<size_t, size_t>;
// 定义 IndexRange 为 size_t 类型的一对值
using torch::dynamo::autograd::CompiledNodeArgs;
// 使用 torch::dynamo::autograd 命名空间中的 CompiledNodeArgs
using torch::dynamo::autograd::SwapSavedVariables;
// 使用 torch::dynamo::autograd 命名空间中的 SwapSavedVariables

// Custom deleter to prevent stack overflows.
// 自定义删除器以防止堆栈溢出
TORCH_API void deleteNode(Node* function);

// Guard that sets and restores the evaluating node
// 设置和恢复评估节点的保护
class NodeGuard {
 public:
  explicit NodeGuard(std::shared_ptr<Node> node);
  // 显式构造函数，接受一个节点的共享指针作为参数
  ~NodeGuard();
  // 析构函数

 private:
  std::shared_ptr<Node> last_evaluating_node_;
  // 最后一个评估的节点的共享指针
};

// Return the Node currently being evaluated (if any)
// 返回当前正在评估的节点（如果有的话）
// This is only set during the backward pass while a Node is being
// executed.
// 这只在反向传播过程中设置，当一个节点被执行时
TORCH_API std::shared_ptr<Node> get_current_node();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                               Node
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A `Node` is an abstract class that represents an operation taking zero
// or more input `Variable`s and producing zero or more output `Variable`s. All
// functions in PyTorch's autograd machinery derive from this class and
// override its `apply` method. Instances of such subclasses will then be
// invokable via the call operator.
//
//                    Nodes in the Autograd Graph
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// When viewing the autograd system as a graph, `Node`s are the vertices or
// nodes, connected to each other via (directed) `Edge`s, which themselves are
// represented via (`Node`, input_nr) pairs. `Variable`s are the outputs to
// and inputs of `Node`s, and travel between these edges during execution
// of the graph. When two or more `Edge`s (from different sources) point at the
// same input to a `Node`, the values produced along all of these edges are
// implicitly summed prior to being forwarded to the target `Node`.
//
//                              Hierarchy
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// `Node` 是一个抽象类，表示接受零个或多个输入 `Variable` 并产生零个或多个输出 `Variable` 的操作。
// PyTorch 的自动求导机制中的所有函数都从这个类派生，并重写其 `apply` 方法。这些子类的实例可以通过调用运算符进行调用。
//
// 在自动求导系统中，将其视为图时，`Node` 是图中的顶点或节点，通过（有向）`Edge` 连接到彼此，
// `Edge` 本身由（`Node`，input_nr）对表示。`Variable` 是 `Node` 的输出和输入，在图的执行过程中通过这些边传输。
// 当两个或多个来自不同源的 `Edge` 指向 `Node` 的同一输入时，沿着这些边产生的值在转发到目标 `Node` 之前会被隐式求和。
//
//                              Hierarchy
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 层次结构
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Subclasses usually represent differentiable functions as well as their
// gradient operators. Note, however, that due to the very general definition
// of a `Node` taking *zero* or more inputs and producing *zero* or more
// outputs, uses of `Node`s are flexible and extend beyond purely
// mathematical operations. For example, the `AccumulateGrad` function is a
// *sink*: it takes one input, but produces no outputs, instead accumulating
// the input as a side effect. At the other extreme, the `GraphRoot` function
// receives no inputs from other functions, but produces multiple outputs.
//
//                              Interface
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The most important method on `Node` is the call operator, which takes in
// a list of variables and produces a list of variables. The precise size of
// these lists can be determined with `num_inputs()` and `num_outputs()`.
// `Node`s are stitched together via their `next_edge` interface, which let
// you manipulate the set of outgoing edges of a `Node`. You can add an
// edge with `add_next_edge()`, retrieve an edge with `next_edge(index)` and
// iterate over them via the `next_edges()` method. Other methods exist for
// integration with the JIT and other parts of PyTorch. Every `Node` has a
// *sequence number* that increases monotonically in the order of `Node`
// construction. It can be retrieved via the `sequence_nr()` method. Note that
// this sequence number is *thread local*. This means that when `Node`s
// `A`, `B` and `C` are created consecutively in the same thread, their
// sequence numbers will be ordered `A` < `B` < `C`. If, however, `A` and `B`
// are created in one thread and `C` is created in a new thread, there are *no
// guarantees* w.r.t. the ordering of `C` relative to `A` or `B`.
// See NOTE [ Sequence Number] for more details on the usages of sequence
// number.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 结构体 `Node` 定义
struct TORCH_API Node : std::enable_shared_from_this<Node> {
 public:
  /// Construct a new `Node` with the given `next_edges`
  // 使用给定的 `next_edges` 构造一个新的 `Node`
  explicit Node(uint64_t sequence_nr, edge_list&& next_edges = edge_list())
      : sequence_nr_(sequence_nr), next_edges_(std::move(next_edges)) {
    // 对于每个 `next_edges` 中的边，更新拓扑顺序号
    for (const Edge& edge : next_edges_) {
      update_topological_nr(edge);
    }

    // 如果异常模式已启用
    if (AnomalyMode::is_enabled()) {
      // 存储当前调用栈的元数据
      metadata()->store_stack();

      // 如果异常模式启用且图已构建，则将当前评估的节点分配为该节点的父节点
      // 父节点是创建此节点的节点
      assign_parent();
    }

    // 存储前向操作的线程 ID
    // 参见 NOTE [ Sequence Numbers ]
    // 获取当前线程的 ID 并赋值给 thread_id_
    thread_id_ = at::RecordFunction::currentThreadId();
  }

  // 使用给定的下一边缘列表构造节点对象
  explicit Node(edge_list&& next_edges = edge_list())
      : Node(
            /*sequence_nr=*/at::sequence_number::get_and_increment(),
            std::move(next_edges)) {}

  /// Nodes are neither copyable nor moveable.
  // 禁用节点对象的拷贝构造函数和移动构造函数
  Node(const Node& other) = delete;
  Node(Node&& other) = delete;
  // 禁用节点对象的拷贝赋值运算符和移动赋值运算符
  Node& operator=(const Node& other) = delete;
  Node& operator=(Node&& other) = delete;
  // 默认虚析构函数
  virtual ~Node() = default;

  // 返回节点对象的 shared_ptr
  std::shared_ptr<Node> getptr() {
    return shared_from_this();
  }
  /// Evaluates the function on the given inputs and returns the result of the
  /// function call.
  // 对给定的输入进行函数求值，并返回函数调用的结果
  variable_list operator()(variable_list&& inputs) {
    // 在命名张量的第一个迭代中，自动求导忽略名称并操作无名称的张量。
    // 长期来看，自动求导应该支持名称。
    at::NoNamesGuard no_names_guard;
#ifdef USE_ROCM
    // 在 ROCm 使用下，跟踪 rocblas 的反向传播
    at::ROCmBackwardPassGuard in_backward;
#endif

    // 获取步骤回调函数，如果不为空，则使用记录域为 BACKWARD_FUNCTION 的回调函数
    auto step_callbacks =
        at::getStepCallbacksUnlessEmpty(at::RecordScope::BACKWARD_FUNCTION);
    if (C10_UNLIKELY(step_callbacks.has_value())) {
      // 记录函数进入，关联前向传播函数的序列号和线程 ID
      at::RecordFunction guard(std::move(*step_callbacks));
      guard.setForwardThreadId(thread_id_);
      // 如果需要输入，则将输入转换为 c10::IValue 数组，并在函数调用前记录函数
      if (guard.needsInputs()) {
        std::vector<c10::IValue> inputs_vec(inputs.begin(), inputs.end());
        guard.before(
            name(),
            c10::ArrayRef<const c10::IValue>(
                inputs_vec.data(), inputs_vec.size()),
            static_cast<int64_t>(sequence_nr()));
      } else {
        // 否则，只记录函数名称和序列号
        guard.before(name(), static_cast<int64_t>(sequence_nr()));
      }
      // 应用函数并返回结果
      return apply(std::move(inputs));
    } else {
      // 如果没有回调函数，直接应用函数并返回结果
      return apply(std::move(inputs));
    }
  }

  // Graph Connectivity API
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Inputs. NOTE: inputs of the grad_fn correspond to Tensor outputs of the
  // forward function.

  // Marker for expected undefined input
  // 预期未定义输入的标记
  struct undefined_input {};

  /// Adds the type and shape metadata for a new input. Returns the index of
  /// of the new input.
  // 添加新输入的类型和形状元数据。返回新输入的索引。
  uint32_t add_input_metadata(
      const at::TensorOptions& options,
      c10::SymIntArrayRef shape,
      bool is_tensor_subclass,
      bool is_nested) noexcept {
    // 获取当前输入的索引，创建并添加元数据形状
    uint32_t input_nr = input_metadata_.size();
    auto meta_shape = MetadataShape{std::in_place_type<SymIntSmallVec>, shape};
    input_metadata_.emplace_back(
        options, meta_shape, is_tensor_subclass, is_nested);
    return input_nr;
  }

  // 添加输入的元数据信息，基于给定的 Tensor 对象。
  uint32_t add_input_metadata(const at::Tensor& t) noexcept {
    uint32_t input_nr = input_metadata_.size();
    input_metadata_.emplace_back(t);
    return input_nr;
  }

  /// Adds a placeholder for an input that will not be used.
  // 添加一个用于未使用的输入的占位符。
  uint32_t add_input_metadata(undefined_input u) noexcept {
    uint32_t input_nr = input_metadata_.size();
    input_metadata_.emplace_back();
    return input_nr;
  }

  // 返回输入的数量。
  uint32_t num_inputs() const noexcept {
    return input_metadata_.size();
  }

  // 获取指定索引位置的输入元数据。
  const InputMetadata& input_metadata(size_t index) const {
    return input_metadata_[index];
  }

  // 危险操作：非线程安全，调用者必须使用锁来保护。
  // 获取可变的输入元数据，通过指定的索引。
  InputMetadata& mutable_input_metadata(size_t index) {
    return input_metadata_[index];
  }

  /**
   * Note: Function Streams
   * A function's stream (for a given device type) is the stream of the first
   * element of its input buffer on a device of that type.
   *
   * If all elements are on the same device they MUST share a stream. If
   * elements are on different devices (across multiple GPUs, for example)
   * they may have different streams.
   */
  // 函数流注意事项：函数在给定设备类型上的流是该类型设备上输入缓冲区的第一个元素的流。
  // 如果所有元素都在同一设备上，则它们必须共享一个流。如果元素在不同设备上（例如跨多个 GPU），则它们可能有不同的流。
  std::optional<c10::Stream> stream() {
    // 获取加速器的设备类型
    auto opt_device_type = at::getAccelerator();
    // 如果设备类型选项没有值，返回空的optional
    if (!opt_device_type.has_value()) {
      return c10::nullopt;
    }
    // 遍历输入元数据列表，查找与设备类型选项匹配的元数据
    for (const auto& metadata : input_metadata_) {
      // 如果找到匹配的设备类型，则返回对应的流信息
      if (metadata.device().type() == opt_device_type.value())
        return metadata.stream();
    }

    // 如果未找到匹配的设备类型，返回空的optional
    return c10::nullopt;
  }

  // 清空输入元数据列表
  void clear_input_metadata() {
    input_metadata_.clear();
  }

  // 更新拓扑编号（Topological Number）
  void update_topological_nr(const Edge& edge) {
    // 断言当前节点没有父节点，否则抛出错误信息
    TORCH_INTERNAL_ASSERT(
        !has_parent_,
        "Cannot update a node's topological_nr after it already has a parent."
        " If we allow this, we can no longer guarantee that a parent's"
        " topo_nr is always greater than those of all its children")
    // 获取边所关联的节点
    Node* node = edge.function.get();
    // 如果节点存在
    if (node) {
      // 获取节点的拓扑编号
      auto topo_nr = node->topological_nr();
      // 如果当前节点的拓扑编号小于等于获取的节点的拓扑编号，则更新当前节点的拓扑编号
      if (topological_nr_ <= topo_nr) {
        topological_nr_ = topo_nr + 1;
      }
    }
  }

  // 设置指定位置的下一条边
  void set_next_edge(size_t index, Edge edge) {
    // 更新当前节点的拓扑编号
    update_topological_nr(edge);
    // 将指定位置的下一条边设置为给定的边
    next_edges_[index] = std::move(edge);
  }

  // 添加一条下一条边
  void add_next_edge(Edge edge) {
    // 更新当前节点的拓扑编号
    update_topological_nr(edge);
    // 在当前节点的下一条边列表末尾添加一条边
    next_edges_.emplace_back(std::move(edge));
  }

  // 设置所有下一条边
  void set_next_edges(edge_list&& next_edges) {
    // 将给定的下一条边列表移动赋值给当前节点的下一条边列表
    next_edges_ = std::move(next_edges);
    // 遍历所有新的下一条边，更新当前节点的拓扑编号
    for (const auto& next_edge : next_edges_) {
      update_topological_nr(next_edge);
    }
  }

  // 返回指定位置的下一条边的常量引用
  const Edge& next_edge(size_t index) const noexcept {
    return next_edges_[index];
  }

  // 返回当前节点的所有下一条边的常量引用
  const edge_list& next_edges() const noexcept {
    return next_edges_;
  }

  // 返回当前节点的所有下一条边的引用
  edge_list& next_edges() noexcept {
    return next_edges_;
  }

  // 返回当前节点的下一条边的数量
  uint32_t num_outputs() const noexcept {
    return next_edges_.size();
  }

  // 杂项方法
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// 注意 [序列号]
  ///
  /// 序列号在自动求导中有两个主要用途：
  ///
  /// 1) 帮助确定节点在引擎中的执行优先级。
  ///    其他条件相等时，具有较高优先级数字的节点首先执行。
  ///    因此，稍后执行的操作对应的节点将首先在反向传播中执行。
  ///    一个注意事项是，我们通过将其序列号显式设置为UINT64_MAX，优先处理AccumulateGrad节点。
  /// 2) 此 `Node` 的序列号与其创建的线程ID一起作为分析器的唯一标识符。
  ///    这个目的是帮助用户（以及可能的程序）解释分析器输出，以将反向节点与其前向操作对应起来。
  ///    我们需要同时拥有序列号和线程ID来标识一个节点，因为序列号是线程局部的，即在新线程中从零开始计数。
  uint64_t sequence_nr() const noexcept {
    return sequence_nr_;
  }

  // 设置节点的序列号
  void set_sequence_nr(uint64_t sequence_nr) {
  sequence_nr_ = sequence_nr;
}

// NOTE [ Topological Number ]
//
// topological_nr is used to prune branches in the DAG during autograd
// discovery as maintaining topological_nr helps us check in O(1) if there
// does NOT exist a directed path between two nodes.
//
// The topological order number of this `Node` representing the length of the
// longest possible path from this Node to any leaf node. If you are leaf
// node, aka AccumulateGrad, this will be zero. This value has the property
// that For every pair of nodes X, Y in G, existence of a directed path from X
// to Y implies topo_nr(X) > topo_nr(Y). The converse is not true, however, so
// we cannot prove existence of a path from X to Y, only non-existence.
//
// One assumption we make when using topo_nr is that once a node
// has been used, i.e., has a parent node, its own topo_nr does not change
// we have added some checks with the `has_parent_` field to enforce this.
//
// What NOT to do:
//
//   1) 2 -> 1 -> 0               In this diagram we label nodes with their
//   topo_nr.
//      2 -> 1 -> 0               We have two simple graphs that can each
//                                arise from
//                                `t.exp().exp()`, for example.
//   2)        2 -> 1 -> 0
//            /
//      2 -> 1 -> 0               We add 2 as a next edge to 1 even though 1
//      already
//                                has a parent.
//   3)        2 -> 1 -> 0
//            /
//      2 -> 3 -> 0               2 < 3, yet there exists a path from 2 to 3!
//
// Sets `has_parent_` to true and returns the current topological order number.
uint64_t topological_nr() const noexcept {
  has_parent_ = true;
  return topological_nr_;
}

// assigning a node as a parent to this node
void assign_parent();

/// Id of the thread that created Node
uint64_t thread_id() const noexcept {
  return thread_id_;
}

/// Returns the name of the dynamic type of the function, for debugging.
virtual std::string name() const;

/// The difference between functions `should_compute_output` and
/// `task_should_compute_output`:
/// - `should_compute_output` should only be used during graph construction
/// and takes into account only requires_grad information
/// - `task_should_compute_output` should only be called during the backward
/// pass (unless called directly through grad_fn) and takes into account the
/// current graph task.  Specifically, the autograd engine trims unnecessary
/// edges when `inputs` are specified, and during backward untrimmed nodes
/// left on the graph can/should check `task_should_compute_output` to see if
/// any outgoing edges have been trimmed by the engine. If that is the case,
/// gradient computation wrt those edges can be omitted.
///
/// Returns true if the particular output edge is active, and that particular
/// output of this function should be computed.
bool should_compute_output(size_t output_edge_index) const {
    // 检查输出边索引是否小于输出总数，否则抛出错误信息
    TORCH_CHECK(output_edge_index < num_outputs(), "Index out of range");
    // 返回指定输出边索引对应的下一边是否有效
    return next_edges_[output_edge_index].is_valid();
  }

  /// 如果任何一个范围中的输出边是活跃的，则返回true。
  bool should_compute_output(std::initializer_list<IndexRange> idxs) const {
    // 使用任意范围中的任何一个来检查是否有活跃的输出边
    return std::any_of(idxs.begin(), idxs.end(), [this](IndexRange range) {
      // 遍历指定范围内的所有索引
      for (const auto i : c10::irange(range.first, range.second)) {
        // 如果当前索引处的输出边应该计算，则返回true
        if (should_compute_output(i))
          return true;
      }
      return false;
    });
  }

  /// 与上述 `should_compute_output` 函数相同，但还会检查当前图任务中是否需要此边。
  bool task_should_compute_output(size_t output_edge_index) const {
    // 检查输出边索引是否小于输出总数，否则抛出错误信息
    TORCH_CHECK(output_edge_index < num_outputs(), "Index out of range");
    // 获取指定输出边索引的下一边
    const auto& next = next_edges_[output_edge_index];
    // 如果下一边有效
    if (next.is_valid()) {
      // 获取当前图任务的执行信息
      const auto exec_info = get_current_graph_task_exec_info();
      // 如果执行信息存在且非空
      if (exec_info && !exec_info->empty()) {
        // 查找当前下一函数是否在执行信息中
        auto it = exec_info->find(next.function.get());
        // 如果未找到或者不应该执行，则返回false，表示当前图任务不需要此边
        if (it == exec_info->end() || !it->second.should_execute()) {
          return false; // 此边在当前图任务中不需要
        }
      }
      return true; // 此边在当前图任务中需要计算
    }
    return false; // 下一边无效，不需要计算
  }

  /// 返回true，如果任何一个范围中的输出边是活跃的，并且应该在当前图任务中计算。
  bool task_should_compute_output(
      std::initializer_list<IndexRange> idxs) const {
    // 使用任意范围中的任何一个来检查是否有活跃的输出边，并且在当前图任务中应该计算
    return std::any_of(idxs.begin(), idxs.end(), [this](IndexRange range) {
      // 遍历指定范围内的所有索引
      for (const auto i : c10::irange(range.first, range.second)) {
        // 如果当前索引处的输出边在当前图任务中应该计算，则返回true
        if (task_should_compute_output(i))
          return true;
      }
      return false;
    });
  }

  /// 返回为此节点存储的 `PyObject`（用于Python交互）。
  PyObject* pyobj() const noexcept {
    return pyobj_;
  }

  /// 设置为此节点存储的 `PyObject`（用于Python交互）。
  void set_pyobj(PyObject* pyobj) noexcept {
    pyobj_ = pyobj;
  }

  /// 返回存储在此节点的异常元数据。
  /// 如果不存在，则创建一个新的空元数据。
  AnomalyMetadata* metadata() noexcept;

  // Hook API
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// 添加一个后置钩子，并返回此钩子的唯一键（作为原始指针）。
  uintptr_t add_post_hook(std::unique_ptr<FunctionPostHook>&& post_hook) {
    // 将后置钩子添加到后置钩子列表中
    post_hooks_.emplace_back(std::move(post_hook));
    // 使用后置钩子的原始指针作为唯一键，用于识别和删除此钩子
    return reinterpret_cast<std::uintptr_t>(post_hooks_.back().get());
  }

  /// 返回后置钩子列表的常量引用。
  const std::vector<std::unique_ptr<FunctionPostHook>>& post_hooks()
      const noexcept {
    return post_hooks_;
  }

  /// 根据键删除匹配的后置钩子。
  bool del_post_hook(const uintptr_t& key) {
  for (auto it = post_hooks_.begin(); it != post_hooks_.end(); ++it) {
    // 遍历 post_hooks_ 容器中的每个元素
    if (key == reinterpret_cast<std::uintptr_t>(it->get())) {
      // 检查当前元素的指针是否与给定的 key 相匹配
      post_hooks_.erase(it);
      // 如果匹配，则从 post_hooks_ 中删除该元素
      return true;
      // 返回 true，表示删除成功
    }
  }
  // 如果未找到匹配的元素，返回 false
  return false;
}

std::vector<std::unique_ptr<FunctionPostHook>>& post_hooks() noexcept {
  // 返回 post_hooks_ 成员变量，用于访问后处理钩子的列表
  return post_hooks_;
}

void add_pre_hook(std::unique_ptr<FunctionPreHook>&& pre_hook) {
  // 向 pre_hooks_ 中添加一个前处理钩子
  pre_hooks_.emplace_back(std::move(pre_hook));
}

void add_tensor_pre_hook(std::unique_ptr<FunctionPreHook>&& pre_hook) {
  // 向 tensor_pre_hooks_ 中添加一个张量前处理钩子
  tensor_pre_hooks_.emplace_back(std::move(pre_hook));
}

void add_retains_grad_hook(
    std::unique_ptr<FunctionPreHook>&& pre_hook,
    size_t output_idx) {
  // 将一个保留梯度前处理钩子与指定的输出索引关联存储
  retains_grad_hooks_[output_idx] = std::move(pre_hook);
}

std::unique_ptr<FunctionPreHook> pop_retains_grad_hook(size_t output_idx) {
  // 从 retains_grad_hooks_ 中弹出并返回与指定输出索引相关联的保留梯度前处理钩子
  auto ret = std::move(retains_grad_hooks_[output_idx]);
  retains_grad_hooks_.erase(output_idx);
  return ret;
}

const std::vector<std::unique_ptr<FunctionPreHook>>& pre_hooks()
    const noexcept {
  // 返回 pre_hooks_ 成员变量，用于访问前处理钩子的常量引用列表
  return pre_hooks_;
}

std::vector<std::unique_ptr<FunctionPreHook>>& pre_hooks() noexcept {
  // 返回 pre_hooks_ 成员变量，用于访问前处理钩子的非常量引用列表
  return pre_hooks_;
}

virtual std::vector<std::unique_ptr<FunctionPreHook>>&
tensor_pre_hooks() noexcept {
  // 虚函数重载，返回 tensor_pre_hooks_ 成员变量，用于访问张量前处理钩子的列表
  return tensor_pre_hooks_;
}

virtual std::unique_ptr<PostAccumulateGradHook>&
tensor_post_acc_grad_hooks() noexcept {
  // 虚函数重载，返回静态的空指针，用于访问张量后累积梯度钩子
  static std::unique_ptr<PostAccumulateGradHook> empty = nullptr;
  return empty;
}

std::unordered_map<size_t, std::unique_ptr<FunctionPreHook>>&
retains_grad_hooks() noexcept {
  // 返回 retains_grad_hooks_ 成员变量，用于访问保留梯度前处理钩子的无序映射
  return retains_grad_hooks_;
}

// Customization Points for Subclasses
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Releases saved variables if the operation won't be reused.
virtual void release_variables() {
  // 虚函数，子类可以实现以释放保存的变量，如果操作不会被重用
}

/// Called before an apply if `release_variables()` is going to be called.
/// Allows larger ops like `InterpreterAutogradFunction` to incrementally
/// release variables as they run.
virtual void will_release_variables() {
  // 虚函数，子类可以实现以在调用 apply() 之前执行一些操作，例如逐步释放变量
}

/// Returns true if this function is traceable. An op is traceable if all
/// operations happening within `apply()` are performed on autograd
/// `Variables` (i.e. apply mostly instantiates and applies other functions).
virtual bool is_traceable() {
  // 虚函数，子类可以实现以判断该函数是否可追踪
  // 如果 apply() 中的所有操作都在 autograd 变量上进行，则返回 true
}
  // 返回 false，表示该函数返回一个布尔值为假
  return false;
}

/// `Node` 在向后传递状态透明时，如果状态仅由（已保存的）变量组成，并且仅包含某种方式参数化操作的非变量对象，
/// 该方式定义了图结构，同时向后函数是可追踪的。特别地，参数化不得依赖于任何 `Variable` 的数据。
/// TODO: 可能可以处理向后不可追踪但状态传递被视为透明的情况。这可能取决于 saved_variable_list 是否可变。
/// 注意：仅当 is_traceable() 返回 false 时，此值才重要。
virtual bool passes_state_transparently() {
  // 返回 false，表示此函数不会透明地传递状态
  return false;
}

// see [Note: Compiled Autograd]
// 由编译自动求导使用
//   1) 提取张量/symint 参数
//   2) 收集节点信息以供特化和缓存
// 子类中的实现应调用 args.collect()，其中包括所有节点属性。这些函数仅在向后过程中调用。
virtual void compiled_args(CompiledNodeArgs& args) {
  // 抛出运行时错误，指示未实现 compiled_args 方法，附带当前节点名称。
  throw std::runtime_error(
      std::string("compiled_args not implemented: ") + name());
}

// 由编译自动求导使用以不同的保存张量调用 apply()
// 实现应在所有属性上调用 saved.before()，然后调用 apply()，最后按相同顺序在所有属性上调用 saved.after()。
virtual variable_list apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
};

/// 结构体 `TraceableFunction` 继承自 `Node`，表示可追踪函数
struct TraceableFunction : public Node {
  using Node::Node;
  // 重写 `is_traceable()` 方法，始终返回 true
  bool is_traceable() final {
    return true;
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                       Associated Free Nodes
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
// `collect_next_edges` 的实现结构体，用于收集下一个函数列表
struct MakeNextFunctionList : IterArgs<MakeNextFunctionList> {
  edge_list next_edges;  // 下一个边的列表
  using IterArgs<MakeNextFunctionList>::operator();
  // 处理变量 `Variable` 的重载 `operator()`
  void operator()(const Variable& variable) {
    if (variable.defined()) {
      next_edges.emplace_back(impl::gradient_edge(variable));  // 收集梯度边
    } else {
      next_edges.emplace_back();  // 空边
    }
  }
  // 处理指针类型的变量 `Variable`
  void operator()(const Variable* variable) {
    operator()(*variable);  // 调用上述的 `operator()`
  }
  // 处理可选的变量 `Variable`
  void operator()(const std::optional<Variable>& variable) {
    if (variable.has_value()) {
      operator()(*variable);  // 调用上述的 `operator()`
    } else {
      next_edges.emplace_back();  // 空边
    }
  }
};
} // namespace detail

/// 创建一个 `Edge`，连接给定的 `variable` 和假设是该变量的梯度函数的 `function`
/// 这设置了 `variable` 的 `grad_fn` 属性。该函数假定 `Variable` 是梯度函数的新输入，
/// 其 `input_nr` 等于 `function->num_inputs()`。此外，它会增加 `Node` 的输入数量。
/// 大致相当于 `variable.set_gradient_edge(function,
/// function->add_input_metadata(variable.dispatch_type(), variable.sizes()))`。
/// 如果不想增加 `Node` 的 `num_inputs`，可以直接使用 `set_gradient_edge`。
inline void create_gradient_edge(
    Variable& variable,
    std::shared_ptr<Node> function) {
  // 在移动之前复制。
  const auto input_nr = function->add_input_metadata(variable);
  impl::set_gradient_edge(variable, {std::move(function), input_nr});  // 设置梯度边
}

/// 如果列表中的任何变量需要梯度，则返回 true。
inline bool any_variable_requires_grad(const variable_list& variables) {
  return std::any_of(
      variables.begin(), variables.end(), [](const Variable& variable) {
        return variable.defined() && variable.requires_grad();  // 检查是否定义并需要梯度
      });
}

/// 返回给定变量或变量元组的下一个边。
template <typename... Variables>
edge_list collect_next_edges(Variables&&... variables) {
  detail::MakeNextFunctionList make;  // 创建收集下一个函数列表的对象
  make.apply(std::forward<Variables>(variables)...);  // 应用参数列表到收集器上
  return std::move(make.next_edges);  // 返回收集到的下一个边列表
}

struct TypeAndSize {
  TypeAndSize() : options(at::TensorOptions()) {}  // 默认构造函数，初始化选项
  /* implicit */
  TypeAndSize(const at::Tensor& t)
      : sym_sizes(t.sym_sizes().vec()), options(t.options()) {}  // 从张量构造函数，初始化符号大小和选项

  at::Tensor zeros();  // 返回一个零张量

  std::vector<c10::SymInt> sym_sizes;  // 符号大小的向量
  at::TensorOptions options;  // 张量选项
};
} // 结束命名空间 torch::autograd
```