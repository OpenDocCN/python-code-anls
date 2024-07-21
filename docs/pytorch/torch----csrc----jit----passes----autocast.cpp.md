# `.\pytorch\torch\csrc\jit\passes\autocast.cpp`

```py
// 包含 Torch 自动混合精度自动转换相关头文件
#include <torch/csrc/jit/passes/autocast.h>

// 包含 Torch 自动混合精度模式相关头文件
#include <ATen/autocast_mode.h>
// 包含 C10 标量类型定义
#include <c10/core/ScalarType.h>
// 包含 C10 异常处理相关头文件
#include <c10/util/Exception.h>
// 包含 C10 可选类型相关头文件
#include <c10/util/Optional.h>
// 包含 Torch IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 日志相关头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch 量化辅助函数相关头文件
#include <torch/csrc/jit/passes/quantization/helper.h>

// 包含标准库头文件
#include <stack>
#include <unordered_set>
#include <vector>

// 定义 Torch JIT 命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于限定作用域
namespace {

// 自动混合精度是否启用的标志，默认启用
bool autocast_enabled = true;

// 自动混合精度上下文结构体
struct AutocastContext {
  bool gpu_enabled = false;            // GPU 是否启用自动混合精度
  bool cpu_enabled = false;            // CPU 是否启用自动混合精度
  c10::ScalarType gpu_scalar_type = c10::ScalarType::Undefined;  // GPU 标量类型
  c10::ScalarType cpu_scalar_type = c10::ScalarType::Undefined;  // CPU 标量类型

  // 转换为 bool 类型，判断是否启用自动混合精度
  operator bool() const {
    return gpu_enabled || cpu_enabled;
  }
};

// 自动混合精度作用域结构体
struct AutocastScope {
  Value* instance = nullptr;          // JIT 值的实例
  AutocastContext context;            // 自动混合精度上下文

  // 堆栈操作，用于保存父上下文
  void stack(const AutocastContext& parent_context) {}
};

// 判断值是否为自动混合精度节点
bool isAutocastNode(Value* value) {
  const auto class_name = getModuleName(value);
  return class_name.has_value() &&
      (*class_name == "__torch__.torch.cuda.amp.autocast_mode.autocast" ||
       *class_name == "__torch__.torch.cpu.amp.autocast_mode.autocast" ||
       *class_name == "__torch__.torch.amp.autocast_mode.autocast");
}

// 解析自动混合精度节点
//
// 如果存在自动混合精度实例，则返回该实例
//
// 模式示例（在 autocast.__init__() 内联后执行）：
//
// %4 : bool = prim::Constant[value=1]()
// %5 : __torch__.torch.cuda.amp.autocast_mode.autocast = prim::CreateObject()
//  = prim::SetAttr[name="_enabled"](%5, %4)
//
// 注意：
//  1. 不能保证自动混合精度实例与 prim::Enter() 节点位于同一块中
//  2. `prim::SetAttr` 必须在同一块中的 `prim::CreateObject()` 后面，但中间可能有其他节点
//
std::optional<AutocastScope> parseAutocast(
    Value* value,
    const AutocastContext& context) {
  if (!isAutocastNode(value)) {
    // 非自动混合精度节点...
    return c10::nullopt;
  }
  if (value->node()->kind() == prim::CreateObject) {
    AutocastScope scope;
    scope.instance = value;
    scope.context = context;
    std::optional<bool> enabled;
    std::string device;
    c10::ScalarType dtype = c10::ScalarType::Undefined;
    // 遍历Autocast对象的每个使用情况
    for (Use use : value->uses()) {
      // TODO: support runtime flag
      // 检查是否为`prim::SetAttr`操作，且属性名为"_enabled"
      if (use.user->kind() == prim::SetAttr &&
          use.user->s(attr::name) == "_enabled") {
        // 找到`prim::SetAttr[name="_enabled"]`，获取其常量值
        auto ret = constant_as<bool>(use.user->input(1));
        // 检查返回值是否有效，如果有效则设置enabled值
        TORCH_CHECK(
            ret.has_value(), "Autocast _enabled argument must be a constant");
        enabled = ret.value();
      } else if (
          use.user->kind() == prim::SetAttr &&
          use.user->s(attr::name) == "device") {
        // 找到`prim::SetAttr[name="device"]`，获取其常量值
        auto ret = constant_as<std::string>(use.user->input(1));
        // 检查返回值是否有效，如果有效则设置device值
        TORCH_CHECK(
            ret.has_value(), "Autocast device argument must be a constant");
        device = ret.value();
      } else if (
          use.user->kind() == prim::SetAttr &&
          use.user->s(attr::name) == "fast_dtype") {
        // 找到`prim::SetAttr[name="fast_dtype"]`，获取其常量值
        auto ret = constant_as<c10::ScalarType>(use.user->input(1));
        // 如果返回值有效，则设置dtype值
        if (ret.has_value()) {
          dtype = ret.value();
        }
      }
    }
    // 检查是否成功获取_enabled属性
    TORCH_CHECK(enabled.has_value(), "Autocast missing _enabled attribute");
    // 检查是否成功获取device属性
    TORCH_CHECK(!device.empty(), "Autocast missing device attribute");
    // 如果dtype为Undefined，则根据device获取默认的dtype
    if (dtype == c10::ScalarType::Undefined) {
      dtype = at::autocast::get_autocast_dtype(c10::Device(device).type());
    }
    // 检查dtype是否有效
    TORCH_CHECK(
        dtype != c10::ScalarType::Undefined,
        "Autocast has invalid fast_dtype attribute");
    // 根据device类型设置相应的Autocast上下文信息
    if (device == "cuda") {
      scope.context.gpu_enabled = enabled.value();
      scope.context.gpu_scalar_type = dtype;
    } else if (device == "cpu") {
      scope.context.cpu_enabled = enabled.value();
      scope.context.cpu_scalar_type = dtype;
    } else {
      // 如果device类型不是"cuda"或"cpu"，抛出内部断言错误
      TORCH_INTERNAL_ASSERT(
          false, "unrecognized device for autocast pass: ", device);
    }
    // 返回更新后的Autocast上下文
    return scope;
  } else {
    // 如果Autocast表达式不是简单静态的形式，抛出错误信息
    // 我们仅支持简单静态的Autocast表达式，例如：
    //    autocast_on = autocast(enabled=True)
    //    autocast_off = autocast(enabled=False)
    //    with autocast_on if condition else autocast_off:
    //        ...
    //
    // TODO: 提供更好的错误信息
    //
    AT_ERROR("Unsupported autocast syntax");
  }

  // 如果未能返回更新后的Autocast上下文，返回空值
  return c10::nullopt;
}

// 将节点的输入张量转换为最宽的类型
void castTensorInputs(
    Node* node,
    Symbol cast_op,
    const AutocastContext& context) {
  // 如果上下文为空，直接返回
  if (!context) {
    return;
  }

  // 获取节点所在的计算图
  const auto graph = node->owningGraph();

  // 用于存储已经转换过类型的输入张量集合
  std::unordered_set<Value*> casted_inputs;
  // 需要保持输入顺序，否则追踪会失败
  // 对插入的类型转换操作进行合理性检查
  std::vector<Value*> casted_inputs_ordered;
  // 遍历节点的输入
  for (auto input : node->inputs()) {
    // TODO: 更新cast_op的签名以使用动态上下文标志
    auto input_tensor_type = input->type()->cast<TensorType>();
    // 如果输入是张量类型且不是cast_op类型的节点
    if (input_tensor_type && input->node()->kind() != cast_op) {
      auto has_inserted = casted_inputs.insert(input);
      // 如果成功插入新元素到集合中，则添加到有序列表中
      if (has_inserted.second) {
        casted_inputs_ordered.push_back(input);
      }
    }
  }

  // 设置插入点为当前节点
  WithInsertPoint insert_point(node);

  // 遍历已排序的转换过类型的输入张量
  for (auto input : casted_inputs_ordered) {
    // 根据cast_op的不同进行不同的操作
    if (cast_op == aten::_autocast_to_full_precision) {
      // 插入全精度转换操作，并替换原始输入
      const auto new_input = graph->insert(
          cast_op,
          {input,
           graph->insertConstant(IValue(context.gpu_enabled)),
           graph->insertConstant(IValue(context.cpu_enabled))});
      node->replaceInputWith(input, new_input);
    } else if (cast_op == aten::_autocast_to_reduced_precision) {
      // 插入降低精度转换操作，并替换原始输入
      const auto new_input = graph->insert(
          cast_op,
          {input,
           graph->insertConstant(IValue(context.gpu_enabled)),
           graph->insertConstant(IValue(context.cpu_enabled)),
           graph->insertConstant(IValue(context.gpu_scalar_type)),
           graph->insertConstant(IValue(context.cpu_scalar_type))});
      node->replaceInputWith(input, new_input);
    } else {
      // 报告错误，未识别的cast_op符号
      TORCH_INTERNAL_ASSERT(
          false, "unrecognized cast_op symbol: ", cast_op.toQualString());
    }
  }
}

// 检查节点是否具有显式的dtype参数
bool hasExplicitDtypeArgument(Node* node) {
  if (node->hasNamedInput("dtype")) {
    // 获取名为dtype的输入参数
    Value* dtype_arg = node->namedInput("dtype");
    // 如果dtype参数的类型不是NoneType，则返回true
    return dtype_arg->type()->kind() != TypeKind::NoneType;
  }
  // 否则返回false
  return false;
}

// 将节点的输入转换为最宽的类型
void castInputsToWidestType(Node* node, const AutocastContext& context) {
  // 如果上下文为空，直接返回
  if (!context) {
    return;
  }
  // 确定最宽的类型（实际上只是查找任何float32类型的输入）
  //
  // TODO: 重新审视这一点（是否需要考虑float64类型？）
  //
  for (auto input : node->inputs()) {
    if (auto tensor_type = input->type()->cast<TensorType>()) {
      const auto dtype = tensor_type->scalarType();
      // 如果没有指定dtype或者dtype为Float类型
      if (!dtype.has_value() || *dtype == at::ScalarType::Float) {
        // 调用castTensorInputs函数进行全精度转换，并返回
        castTensorInputs(node, aten::_autocast_to_full_precision, context);
        return;
      }
    }
  }
}

// 用户可以调用torch.is_autocast_enabled()或is_autocast_cpu_enabled()来确定是否启用了自动转换。
// 对于JIT脚本函数，如果启用了急切自动转换或JIT自动转换，则实际上需要返回true。
//
// 在启用JIT自动转换的情况下，我们将
//    %x : bool = aten::is_autocast_enabled()
// 替换为常量"True"。
//
// More context on eager vs JIT autocasting:
//
// Autocasting actually has two settings: eager autocasting, and JIT
// autocasting. Eager autocasting is the thread-local setting that turns on
// the relevant bit in the dispatcher settings. JIT autocasting is the pass
// implemented in this file, which makes changes to the graph to insert casting
// ops in order to achieve the same behavior as eager autocasting.
//
// If eager autocasting is enabled at the time when a JIT-scripted function is
// invoked, then autocasting will occur regardless of what the JIT-autocasting
// settings are.
void updateAutocastEnabledCheck(Node* node, bool is_jit_enabled) {
  if (!is_jit_enabled) {
    return;
  }

  auto graph = node->owningGraph();

  // Set the insertion point for new operations to be inserted after 'node'
  WithInsertPoint insert_point(node);

  // Replace the output of 'node' with a constant value 'true'
  Value* true_constant = graph->insertConstant(IValue(true));
  node->output()->replaceAllUsesWith(true_constant);

  // Destroy 'node', removing it from the graph
  node->destroy();
}

// [Note: implicit type promotion in Autocast]
//
// Casting policy below mostly follows pytorch/aten/src/ATen/autocast.cpp, with
// a few exceptions, e.g. `aten::add`, which is needed to be put to promotion
// list for JIT autocast.
// The reason is that in eager amp, some binary ops promote inputs implicitly
// inside the operation, e.g. `aten::add` with fp16 & fp32 inputs would both be
// casted to fp32. In backward, autograd would cast dgrad to match their
// scalar_type in forward graph. So inputs with mismatched scalar_type would
// get the different dgrad.
// While in JIT, autodiff doesn't do this, so implicit cast is not visible to
// autodiff and backward dgrad for mismatched inputs would ended up with dgrads
// in the same scalar_type. This has caused downstream operations, which
// expects dgrad to be the same scalar type to throw mismatch error.
//
// TODO: Use the list from AMP eager directly
void handleBlock(Block* block, AutocastContext initial_state) {
  std::stack<AutocastScope> autocast_stack;

  std::optional<bool> incompatible_amp = c10::nullopt;

  // Function to retrieve the current autocast state from the stack
  auto current_state = [&] {
    return autocast_stack.empty() ? initial_state
                                  : autocast_stack.top().context;
  };

  // Iterate through all nodes in the block
  for (Node* node : block->nodes()) {
    // Process sub-blocks, if any
    for (Block* sub_block : node->blocks()) {
      handleBlock(sub_block, current_state());
    }
  }

  // Assert that there are no unbalanced transitions in the autocast stack
  TORCH_INTERNAL_ASSERT(autocast_stack.empty());
}

// End of namespace
} // namespace

// Set the global autocasting mode and return the previous value
bool setAutocastMode(bool value) {
  auto old_value = autocast_enabled;
  autocast_enabled = value;
  return old_value;
}

// Retrieve the current global autocasting mode
bool autocastEnabled() {
  return autocast_enabled;
}

// Apply autocasting to the given graph, dumping its state before the operation
void Autocast(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("\nBefore Autocast: ", graph);
  // Check if autocasting is enabled globally
  if (autocastEnabled()) {
    # 创建一个 AutocastContext 对象，并初始化其属性
    AutocastContext init = {
        # 获取当前 CUDA 是否启用自动类型转换的状态
        at::autocast::is_autocast_enabled(at::kCUDA),
        # 获取当前 CPU 是否启用自动类型转换的状态
        at::autocast::is_autocast_enabled(at::kCPU),
        # 获取 CUDA 上的自动类型转换的默认数据类型
        at::autocast::get_autocast_dtype(at::kCUDA),
        # 获取 CPU 上的自动类型转换的默认数据类型
        at::autocast::get_autocast_dtype(at::kCPU)
    };
    # 使用初始化后的 AutocastContext 对象处理图的块
    handleBlock(graph->block(), init);
    # 打印输出调试信息，显示自动类型转换后的图状态
    GRAPH_DUMP("\nAfter Autocast: ", graph);
}

} // namespace jit
} // namespace torch
```