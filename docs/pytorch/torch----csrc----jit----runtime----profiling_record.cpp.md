# `.\pytorch\torch\csrc\jit\runtime\profiling_record.cpp`

```py
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch::jit {

// 声明 ProfileRegistry 类用于注册和管理节点的分析功能
namespace {

class ProfileRegistry {
 public:
  // 获取全局的 ProfileRegistry 单例对象
  static ProfileRegistry* getRegistry() {
    static ProfileRegistry profile_registry_;
    return &profile_registry_;
  }

  // 注册节点分析函数，以便后续节点可以进行分析
  void registerProfileNode(const std::function<bool(const Node*)>& func) {
    std::lock_guard<std::mutex> guard(mutex_);
    registry_funcs_.push_back(func);
  }

  // 判断节点是否需要进行分析
  bool shouldProfileNode(const Node* node) {
    std::lock_guard<std::mutex> guard(mutex_);
    // 对于可微分图，需要特定的分析信息，例如 requires_grad 属性
    if (isDifferentiable(node)) {
      return true;
    }
    // 遍历注册的所有分析函数，判断是否需要对当前节点进行分析
    for (const auto& func : registry_funcs_) {
      if (func(node)) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<std::function<bool(const Node*)>> registry_funcs_;
  std::mutex mutex_;
};

} // namespace

// 注册一个节点分析函数到全局的 ProfileRegistry 中
void RegisterProfilingNode(const std::function<bool(const Node*)>& func) {
  ProfileRegistry::getRegistry()->registerProfileNode(func);
}

// 绑定符号形状到具体的张量尺寸，用于符号执行的形状绑定
bool ShapeSymbolTable::bindSymbolicShapes(
    at::IntArrayRef new_sizes,
    const c10::SymbolicShape& sym_shapes) {
  // 如果符号形状的 rank 未定义，返回 true
  if (!sym_shapes.rank().has_value()) {
    return true;
  }
  // 如果符号形状的 rank 与新尺寸的大小不匹配，返回 false
  if (*sym_shapes.rank() != new_sizes.size()) {
    return false;
  }
  // 遍历新尺寸的每一个维度，绑定静态符号形状到具体的尺寸值
  for (const auto i : c10::irange(new_sizes.size())) {
    auto symbol = (*sym_shapes.sizes())[i];
    // 如果符号不是静态的，跳过
    if (!symbol.is_static()) {
      continue;
    }

    // 如果符号尚未绑定，进行绑定操作
    if (!isBound(symbol)) {
      assign(symbol, new_sizes[i]);
      continue;
    }

    // 如果符号已经绑定且值不匹配，返回 false
    if (getValue(symbol) != new_sizes[i]) {
      return false;
    }
  }
  return true;
}

// ProfilingRecord 类的构造函数，用于记录分析的图和分析运行次数
ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(getNumProfiledRuns()) {}

// 创建一个 ProfileOp 节点，用于图的分析
ProfileOp* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOp(profiled_graph_.get(), fp);

  // 添加输入值到 ProfileOp 节点
  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

// 创建一个 ProfileIValueOp 节点，用于值的分析
ProfileIValueOp* ProfilingRecord::createProfileIValueNode(Value* in_val) {
  auto pn = new ProfileIValueOp(this->profiled_graph_.get(), nullptr);
  pn->addInput(in_val);
  auto pno = pn->addOutput();
  pno->setType(in_val->type());
  return pn;
}

// 创建一个 ProfileIValueOp 节点，用于值的分析，支持多输入
ProfileIValueOp* ProfilingRecord::createProfileIValueNode(
    ArrayRef<Value*> inputs) {
  auto pn = new ProfileIValueOp(this->profiled_graph_.get(), nullptr);
  for (auto inp : inputs) {
    pn->addInput(inp);
    auto pno = pn->addOutput();
    pno->setType(inp->type());
  }
  return pn;
}

} // namespace torch::jit


这段代码包含了一些与 TorchScript JIT 相关的类和函数，用于图的分析和节点的性能分析。
// 命名空间，用于封装不具有全局作用域的功能函数或变量
namespace {
// 检查给定类型是否为 Optional[Tensor] 类型
bool isOptionalTensorType(const TypePtr& type) {
  // 如果类型不是 OptionalType，则返回 false
  if (type->kind() != c10::TypeKind::OptionalType) {
    return false;
  }
  // 获取 OptionalType 的元素类型，检查其是否为 TensorType
  const auto& kind = type->expectRef<OptionalType>().getElementType()->kind();
  return kind == c10::TypeKind::TensorType;
}
} // namespace

// 插入分析节点。
//
// prim::profile 节点用于分析 Tensor 和 Optional[Tensor]。
//
// 存储两个字段：
// 1. attr::seen_none，一个整数，初始值为0，如果分析的值曾为 `None` 则设为1
// 2. attr::profiled_type，表示在分析期间观察到的所有非空输入的最具体的 Tensor 类型
void ProfilingRecord::insertShapeProfile(
    Node* n,
    size_t offset,
    const TypePtr& input_type) {
  // 获取节点 n 的第 offset 个输入值
  Value* i = n->input(offset);
  // 创建一个 profile 节点，分析输入值 i
  auto pn = createProfileNode(nullptr, {i});
  // 为 profile 节点添加一个输出值
  auto pno = pn->addOutput();
  // 设置 profile 节点的属性 attr::profiled_type，指定为 TensorType::get()
  pn->ty_(attr::profiled_type, TensorType::get());
  // 设置 profile 节点的属性 attr::seen_none，初始化为0
  pn->i_(attr::seen_none, 0);
  
  // 如果输入类型是 Optional[Tensor]，则设置输出节点 pno 的类型为 OptionalType(TensorType::get())
  if (isOptionalTensorType(input_type)) {
    pno->setType(OptionalType::create(TensorType::get()));
  } 
  // 如果输入类型是 TensorType，则设置输出节点 pno 的类型为 TensorType::get()
  else if (input_type->kind() == c10::TypeKind::TensorType) {
    pno->setType(TensorType::get());
  } 
  // 如果输入类型既不是 Optional[Tensor] 也不是 TensorType，则断言错误并显示错误消息
  else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Trying to profile an unsupported type (neither Tensor or Optional[Tensor]): ",
        input_type->str());
  }

  // 定义一个函数 shape_profiler，用于处理 profile 结果
  std::function<void(Stack&)> shape_profiler = [this, pn, pno](Stack& stack) {
    int64_t frame_id = 0;
    // 从堆栈中弹出帧 ID
    pop(stack, frame_id);
    IValue v;
    // 从堆栈中弹出值 v
    pop(stack, v);

    TensorTypePtr new_tensor_type = nullptr;
    // 如果值 v 是 Tensor 类型，则获取当前执行环境中的 Tensor 类型
    if (v.isTensor()) {
      auto& t = v.toTensor();
      new_tensor_type = tensorTypeInCurrentExecutionContext(t);
    }

    // 如果值 v 是 Tensor 或者 None 类型
    if (v.isTensor() || v.isNone()) {
      // 加锁以访问互斥量 this->mutex_
      std::lock_guard<std::mutex> lock(this->mutex_);
      // 如果正在进行分析
      if (profiling_count_ > 0) {
        // 调试信息，显示当前运行中的帧 ID，正在注解的节点名称，以及新的 Tensor 类型
        GRAPH_DEBUG(
            "In run ",
            frame_id,
            " annotating %",
            pno->debugName(),
            " with ",
            *new_tensor_type);

        // 如果新的 Tensor 类型不为空
        if (new_tensor_type != nullptr) {
          // 如果 profile 节点已经观察到过 Tensor
          if (pn->hasSeenTensor()) {
            // 获取已存在的 Tensor 类型
            const auto& existing_tensor_type =
                pn->ty(attr::profiled_type)->expectRef<TensorType>();
            // 调试信息，显示已存在的 Tensor 类型
            GRAPH_DEBUG(
                "Existing type for %",
                pno->debugName(),
                ": ",
                existing_tensor_type);
            // 合并新的 Tensor 类型和已存在的 Tensor 类型
            auto merged_type = new_tensor_type->merge(existing_tensor_type);
            // 调试信息，显示合并后的 Tensor 类型
            GRAPH_DEBUG(
                "Merged type for %", pno->debugName(), ": ", *merged_type);
            // 更新 profile 节点的属性 attr::profiled_type 为合并后的类型
            pn->ty_(attr::profiled_type, std::move(merged_type));
          } else {
            // 如果 profile 节点还未观察到 Tensor，则标记已观察到 Tensor，并设置属性 attr::profiled_type
            pn->setHasSeenTensor(true);
            pn->ty_(attr::profiled_type, std::move(new_tensor_type));
          }
        }
        // 如果值 v 是 None，则设置 profile 节点的属性 attr::seen_none 为1
        if (v.isNone()) {
          pn->i_(attr::seen_none, 1);
        }
      }
    }
    // 将值 v 推回堆栈中
    push(stack, v);
  };

  // 设置 profile 节点的回调函数为 shape_profiler
  pn->setCallback(shape_profiler);
  // 将 profile 节点插入到节点 n 之前
  pn->insertBefore(n);
  // 替换节点 n 的第 offset 个输入为 profile 节点的输出
  n->replaceInput(offset, pn->output());
}
// 检查节点是否需要进行输入参数的分析
static bool needsProfiledInputs(Node* n) {
  // 如果节点在TensorExpr中受支持，则需要进行输入参数的分析
  if (tensorexpr::isSupported(n)) {
    return true;
  }

  switch (n->kind()) {
    // 对于以下Autograd节点和Peephole操作，需要进行输入参数的分析
    case prim::AutogradAdd:
    case prim::AutogradAnyNonZero:
    case prim::AutogradAllNonZero:
    case prim::AutogradAllZero:
    case prim::AutogradZero:
    case aten::dim:
    case aten::size:
    case aten::expand:
    case prim::dtype:
    case prim::device:
    case prim::is_cuda:
    case aten::is_floating_point:
    case aten::type_as:
    // 对于特定测试`test_lstm_gates_permutations_cuda`，需要进行输入参数的分析
    case aten::t:
    case aten::mm:
      return true;
    default:
      // 使用ProfileRegistry检查节点是否需要进行输入参数的分析
      return ProfileRegistry::getRegistry()->shouldProfileNode(n);
  }
}

// 检查节点是否需要进行输出参数的分析
static bool needsProfiledOutput(Node* n) {
  // 如果节点在TensorExpr中受支持，则需要进行输出参数的分析
  if (tensorexpr::isSupported(n)) {
    return true;
  }

  switch (n->kind()) {
    // 对于以下Autograd节点，需要进行输出参数的分析
    case prim::AutogradAdd:
    case prim::AutogradZero:
      return true;
    default:
      // 使用ProfileRegistry检查节点是否需要进行输出参数的分析
      return ProfileRegistry::getRegistry()->shouldProfileNode(n);
  }
}

// 从给定的块中移除分析计数器节点
void ProfilingRecord::removeProfileCounter(Block* b) {
  for (auto it = b->nodes().rbegin(); it != b->nodes().rend();) {
    auto n = *it;
    // 如果节点为profile类型且没有输入，则销毁该节点
    if (n->kind() == prim::profile && n->inputs().empty()) {
      it.destroyCurrent();
      // 因为只有一个计数器节点，所以可以直接返回
      return;
    } else {
      it++;
    }
  }
}

// 在给定的块中插入形状分析节点
void ProfilingRecord::instrumentBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (const auto offset : c10::irange(n->inputs().size())) {
      auto i = n->input(offset);
      // 如果节点的输入或输出需要进行参数分析且类型为TensorType或OptionalTensorType，则插入形状分析节点
      if ((needsProfiledInputs(n) || needsProfiledOutput(i->node()))) {
        if (i->type()->kind() == c10::TypeKind::TensorType ||
            isOptionalTensorType(i->type())) {
          insertShapeProfile(n, offset, i->type());
        }
      }
    }

    // 递归处理节点的子块
    for (auto b : n->blocks()) {
      instrumentBlock(b);
    }
  }

  // 在块的返回节点上插入形状分析节点，以便消除更多的保护逻辑
  for (size_t offset = 0; offset < block->return_node()->inputs().size();
       offset++) {
    auto i = block->return_node()->input(offset);
    if (i->type()->isSubtypeOf(*TensorType::get()) ||
        isOptionalTensorType(i->type())) {
      insertShapeProfile(block->return_node(), offset, i->type());
    }
  }
}

// 从给定的块中移除分析节点
void ProfilingRecord::removeProfilingNodes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    // 如果节点是profile或profile_ivalue类型，则替换其输出使用，并销毁节点
    if (it->kind() == prim::profile || it->kind() == prim::profile_ivalue) {
      it->output()->replaceAllUsesWith(it->input());
      it.destroyCurrent();
    } else {
      // 递归处理节点的子块
      for (Block* ib : it->blocks()) {
        removeProfilingNodes(ib);
      }
    }
  }
}

// 检查记录是否已准备好进行分析
bool ProfilingRecord::ready() const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  // 如果分析计数为0，则认为记录已准备好
  return profiling_count_ == 0;
}
std::unique_ptr<ProfilingRecord> ProfilingRecord::instrumentGraph(
    const std::shared_ptr<Graph>& graph) {
  // 复制输入图形，创建一个新的图形副本
  auto new_g = graph->copy();

  // 创建一个 ProfilingRecord 对象，使用新图形副本作为参数
  auto pr = std::unique_ptr<ProfilingRecord>(new ProfilingRecord(new_g));
  
  // 获取指向原始 ProfilingRecord 对象的裸指针
  auto raw_pr = pr.get();
  
  // 取消对输入图形副本的输入数据进行性能分析
  unprofileGraphInputs(new_g);
  
  // 取消对输入图形副本的整个块进行性能分析
  unprofileBlock(new_g->block());
  
  // 在输入图形副本的块上进行性能分析
  pr->instrumentBlock(new_g->block());

  // 定义一个计数器函数对象，用于处理栈数据
  std::function<void(Stack&)> counter = [raw_pr](Stack& stack) {
    int64_t frame_id = 0;
    pop(stack, frame_id);

    // 使用互斥锁保护原始 ProfilingRecord 对象的访问
    std::lock_guard<std::mutex> lock(raw_pr->mutex_);

    // 如果还有性能分析次数剩余，则递减计数
    if (raw_pr->profiling_count_ > 0) {
      raw_pr->profiling_count_--;
    }
  };

  // 创建一个性能分析节点，并将其添加到新图形副本中
  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);

  // 打印调试信息，显示被仪器化的新图形副本
  GRAPH_DUMP("Instrumented Graph: ", new_g);

  // 返回创建的 ProfilingRecord 对象的唯一指针
  return pr;
}

} // namespace torch::jit
```