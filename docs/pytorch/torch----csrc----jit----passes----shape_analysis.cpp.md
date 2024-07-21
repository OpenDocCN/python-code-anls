# `.\pytorch\torch\csrc\jit\passes\shape_analysis.cpp`

```py
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/utils/op_registry.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/symbol.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_strided.h>
#endif

#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

// 合并两组值的类型信息，并更新输出值的类型
bool mergeTypes(
    ArrayRef<Value*> lhs,        // 左侧值的数组
    ArrayRef<Value*> rhs,        // 右侧值的数组
    ArrayRef<Value*> outputs) {  // 输出值的数组
  AT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
  bool changed = false;
  for (const auto i : c10::irange(lhs.size())) {
    auto old_output_type = outputs[i]->type();
    auto new_type =
        unifyTypes(lhs[i]->type(), rhs[i]->type(), /*default_to_union=*/true);  // 统一左右值的类型
    AT_ASSERT(new_type);  // 确保新类型有效
    outputs[i]->setType(*new_type);  // 更新输出值的类型
    if (*old_output_type != *outputs[i]->type())  // 检查类型是否改变
      changed = true;
  }
  return changed;  // 返回类型是否改变的标志
}

// 将源值数组的类型应用到目标值数组
static void applyTypes(ArrayRef<Value*> src, ArrayRef<Value*> dst) {
  AT_ASSERT(src.size() == dst.size());
  for (const auto i : c10::irange(src.size())) {
    dst[i]->setType(src[i]->type());  // 应用源值的类型到目标值
  }
}

// 在当前块中传播属性信息
void PropertyPropBase::propagateBlock(Block* block, bool insert_expands) {
  for (Node* node : block->nodes()) {  // 遍历块中的每个节点
    try {
      propagateNode(node, insert_expands);  // 传播节点的属性信息
    } catch (propagation_error& e) {
      setUnshapedType(node);  // 如果传播错误，则设置节点为未形状化类型
    } catch (std::exception& e) {
      throw ErrorReport(node->sourceRange())
          << ExceptionMessage(e)
          << "\nThe above operation failed shape propagation in this context";  // 抛出异常报告传播失败信息
    }
  }
}

// 处理 If 节点
void PropertyPropBase::processIf(Node* node) {
  auto then_block = node->blocks().at(0);  // 获取 If 节点的第一个块
  auto else_block = node->blocks().at(1);  // 获取 If 节点的第二个块
  propagateBlock(then_block);  // 传播 Then 块的属性信息
  propagateBlock(else_block);  // 传播 Else 块的属性信息
  mergeTypes(then_block->outputs(), else_block->outputs(), node->outputs());  // 合并 Then 和 Else 块的输出类型信息到节点的输出中
}

// 处理 Loop 节点
void PropertyPropBase::processLoop(Node* node) {
  LoopView loop(node);  // 创建 LoopView 对象以便访问循环信息
  // 传播计数器类型
  loop.currentTripCount()->setType(loop.maxTripCount()->type());
  applyTypes(loop.carriedInputs(), loop.bodyCarriedInputs());  // 应用循环输入值的类型到循环体输入值中

  do {
    propagateBlock(loop.bodyBlock(), /*insert_expands=*/false);  // 传播循环体的属性信息，不插入扩展操作
    // 注意：此时插入扩展操作是不安全的，我们不知道...
    // 当类型稳定时，mergeTypes 函数的参数可能会改变，因此需要在此处循环直到稳定为止
    } while (mergeTypes(
        loop.bodyCarriedInputs(),     // 获取循环体内部传入的变量列表
        loop.bodyCarriedOutputs(),    // 获取循环体内部传出的变量列表
        loop.bodyCarriedInputs()));   // 将循环体内部传入的变量列表作为参数，调用 mergeTypes 函数
    
    // 当类型已经稳定时，可以执行插入 expands 操作
    propagateBlock(loop.bodyBlock(), /*insert_expands=*/true);  // 在循环体的代码块中执行数据展开操作
    
    // 应用循环体内部传入的变量列表到循环体外部传出的变量列表
    applyTypes(loop.bodyCarriedInputs(), loop.carriedOutputs());  // 将循环体内部传入的变量列表应用到循环体外部传出的变量列表
}

// 结束一个类的定义，这里没有类名，可能是在全局命名空间中定义的成员函数

void PropertyPropBase::setUnshapedType(Value* o) {
  // 设置给定值的未成形类型，即将其类型设置为其未成形类型
  o->setType(unshapedType(o->type()));
}

void PropertyPropBase::setUnshapedType(Node* node) {
  // 遍历给定节点的所有输出，并设置它们的未成形类型
  for (auto o : node->outputs()) {
    setUnshapedType(o);
  }
}

namespace prim {
using namespace ::c10::prim;
}

// 定义一个宏，用于检查条件，如果条件不满足则抛出传播错误异常
#define SHAPE_ASSERT(cond) \
  if (!(cond))             \
  throw propagation_error()

namespace {

bool isValidArgumentForRunning(Value* v) {
  // 允许常量作为有效参数
  if (toIValue(v))
    return true;
  // 对于张量类型的值，验证其是否是有效的数值类型
  if (TensorTypePtr tt = v->type()->cast<TensorType>()) {
    if (!tt->scalarType()) {
      return false;
    }
    return !at::isIntegralType(*tt->scalarType(), /*includeBool=*/false);
  }
  // 对于浮点数类型的值，视为有效参数
  return v->type()->isSubtypeOf(*FloatType::get());
}

bool isValidReturnForRunning(Value* v) {
  // 验证返回值是否是张量类型或者数字类型的子类型
  return v->type()->isSubtypeOf(*TensorType::get()) ||
      v->type()->isSubtypeOf(*NumberType::get());
}

bool containsTensorType(const TypePtr& t) {
  // 检查类型是否包含张量类型
  auto n_contained = t->containedTypes().size();
  if (n_contained == 1) {
    return t->containedTypes().at(0)->isSubtypeOf(*TensorType::get());
  } else if (n_contained > 1) {
    // 如果包含多个类型，则递归检查是否有任何一个包含张量类型
    return std::any_of(
        t->containedTypes().begin(),
        t->containedTypes().end(),
        containsTensorType);
  }
  return false;
}

// 针对具有张量类型的节点，提取其张量类型
// 如果任何张量类型的节点没有已知形状，则返回空值
std::optional<std::vector<TensorTypePtr>> gatherTensorTypes(
    Node* node,
    bool complete = false) {
  std::vector<TensorTypePtr> tensor_types;

  auto schema_opt = node->maybeSchema();
  if (!schema_opt) {
    return c10::nullopt;
  }
  auto& schema = *schema_opt;
  auto& args = schema.arguments();
  // 对于包含可变参数的原语，无法处理，返回空值
  if (schema.is_vararg()) {
    return c10::nullopt;
  }
  for (const auto i : c10::irange(args.size())) {
    if (args[i].type()->isSubtypeOf(*ListType::ofTensors())) {
      return c10::nullopt;
    } else if (args[i].type()->isSubtypeOf(*TensorType::get())) {
      if (auto type = node->input(i)->type()->cast<TensorType>()) {
        if (complete && !type->isComplete()) {
          return c10::nullopt;
        }
        tensor_types.push_back(type);
      } else {
        return c10::nullopt;
      }
    } else /* 非张量类型 */ {
      continue;
    }
  }
  return tensor_types;
}

int64_t wrapDim(int64_t dim, at::IntArrayRef sizes) {
  // 处理负数维度，将其转换为非负数
  if (dim < 0) {
    dim += (int64_t)sizes.size();
  }
  return dim;
}

c10::ScalarType unionScalarTypes(
    c10::ScalarType original,
    c10::ScalarType next) {
  // 合并两个标量类型，返回它们的最宽松的类型
  if (original == c10::ScalarType::Undefined) {
    return next;
  } else {
    return c10::promoteTypes(original, next);
  }
}

// 用于对张量操作的结果类型进行提升，使用新的类型提升逻辑
// 详情请参阅tensor_attributes.rst文档
// 此函数不处理具有标量参数的算术操作的情况（当
// 返回用于算术操作的升级类型的可选类型
std::optional<c10::ScalarType> getPromotedTypeForArithmeticOp(Node* node) {
  // 初始化两个标量类型为未定义
  c10::ScalarType dimmed = c10::ScalarType::Undefined;
  c10::ScalarType zerodim = c10::ScalarType::Undefined;
  // 对于二元算术操作，超过2个参数被视为alpha版本
  for (const auto i : c10::irange(2)) {
    // 获取输入节点的Tensor类型
    auto dtt = node->inputs()[i]->type()->expect<TensorType>();
    auto inputDtype = dtt->scalarType();
    // 如果类型为空或者标量类型为空，返回空值
    if (!dtt || !inputDtype) {
      return c10::nullopt;
    }
    // 如果张量有维度且维度大于0
    if (dtt->dim() && *dtt->dim() > 0) {
      // 合并标量类型
      dimmed = unionScalarTypes(dimmed, *inputDtype);
    } else if (!isFloatingType(dimmed)) {
      // 如果没有维度，且当前标量类型不是浮点类型
      zerodim = unionScalarTypes(zerodim, *inputDtype);
    }
  }
  // 如果有维度的张量已经是最高类别，不需要检查零维张量
  if (isFloatingType(dimmed)) {
    return dimmed;
  }
  // 整数张量 * 零维浮点数张量 -> 浮点数张量
  if (isIntegralType(dimmed, false) && isFloatingType(zerodim)) {
    return zerodim;
  }
  // 布尔张量 * 非布尔标量 -> 非布尔张量
  if (c10::ScalarType::Bool == dimmed &&
      c10::ScalarType::Undefined != zerodim) {
    return zerodim;
  }
  // 维度张量的类型通常优先于零维张量，如果不是因为类别升级而升级。例如：
  // 整数张量 * 长整数 -> 整数张量
  if (c10::ScalarType::Undefined != dimmed) {
    return dimmed;
  }

  // 没有有维度的张量。例如零维张量 + 零维张量。
  return zerodim;
}

// ShapePropagator 类，继承自 PropertyPropBase
class ShapePropagator : public PropertyPropBase {
 public:
  // 构造函数，初始化 ShapePropagator 对象
  explicit ShapePropagator(const std::shared_ptr<Graph>& graph)
      : PropertyPropBase(graph), aliasDb_(graph) {
    collectResizeSet(graph->block());
  }

 private:
  // 存储被调整大小的别名集合
  ValueSet resized_alias_set;
  // 图的别名数据库
  const AliasDb aliasDb_;

  // 检查节点是否调整输入大小
  bool resizesInput(Node* n) {
    // 定义可以调整大小的操作符号集合
    static std::unordered_set<Symbol> resize_ops{
        aten::resize_,
        aten::resize_as_,
        aten::copy_,
        aten::set_,
        aten::unsqueeze_,
        aten::t_,
        aten::transpose_,
    };

    if (resize_ops.count(n->kind()))
      return true;

    if (!n->maybeSchema())
      return false;

    // 操作符使用结果并写入输入 "out"
    if (auto out_arg_index = n->schema().argumentIndexWithName("out")) {
      auto arg = n->schema().arguments().at(*out_arg_index);
      return arg.kwarg_only() && arg.type()->isSubtypeOf(*TensorType::get());
    }
    return false;
  }

  // 收集调整大小的集合
  void collectResizeSet(Block* block) {
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        collectResizeSet(b);
      }
      if (resizesInput(n)) {
        for (const auto input : n->inputs()) {
          // 如果写入到别名数据库中的别名
          if (aliasDb_.writesToAlias(n, {input})) {
            resized_alias_set.insert(input);
          }
        }
      }
    }
  }

  // 获取值的代表值
  IValue representativeValue(Value* v) {
    TypePtr type_ = v->type();
    // 如果值实际上是常量，直接使用它！
    // 如果 v 可以转换为 IValue 对象，则返回其解引用值
    if (auto iv = toIValue(v)) {
      return *iv;
    }
    // 如果 type_ 可以转换为 TensorTypePtr 对象
    if (TensorTypePtr type = type_->cast<TensorType>()) {
      // 如果该 TensorType 对象是完整的
      if (type->isComplete()) {
        // 在 type->device() 指定的设备上创建一个空张量，使用给定的大小和步长
        at::DeviceGuard device_guard(*type->device());
        return at::empty_strided(
                   *type->sizes().concrete_sizes(),
                   *type->strides().concrete_sizes(),
                   at::TensorOptions(*type->device())
                       .dtype(*type->scalarType()))
            .zero_();
      }
      // 如果不是完整的 TensorType 对象，则通过 fallthrough 继续执行
    } else if (type_->isSubtypeOf(*FloatType::get())) {
      // 如果 type_ 是 FloatType 的子类型，则返回浮点数 0.0
      return 0.f;
    }
    // 如果程序运行到这里，说明 isValidArgumentForRunning 应该已经阻止了这种情况
    // 构造一个异常信息字符串流，报告无法为 type_ 创建代表值的错误
    std::stringstream ss;
    ss << "unable to create representative value for: " << type_->str()
       << ". File a bug report";
    throw std::runtime_error(ss.str());
  }

  // 对两个张量类型进行广播操作
  void broadcastBinary(
      Node* node,
      std::vector<TensorTypePtr>& types,
      size_t idx1,
      size_t idx2) {
    // 推断两个输入张量的期望大小
    auto expected_size = at::infer_size(
        *types[idx1]->sizes().concrete_sizes(),
        *types[idx2]->sizes().concrete_sizes());
    // 广播函数，根据输入索引进行处理
    auto broadcast = [&](size_t input_idx) {
      TensorTypePtr input_type = types.at(input_idx);
      // 如果输入类型的大小与期望大小不同，则进行广播操作
      if (input_type->sizes() == expected_size)
        return;
      auto graph = node->owningGraph();
      // 在当前节点之前创建一个 expand 操作节点，用于广播操作
      WithInsertPoint point_guard{node};
      Node* expand = graph
                         ->create(
                             aten::expand,
                             {node->inputs().at(input_idx),
                              graph->insertConstant(expected_size),
                              graph->insertConstant(false)})
                         ->insertBefore(node);
      propagateNode(expand);
      // 替换节点的输入为 expand 节点的输出
      node->replaceInput(input_idx, expand->output());
    };
    // 分别对 idx1 和 idx2 进行广播操作
    broadcast(idx1);
    broadcast(idx2);
    // 更新 types 中的第一个和第二个元素为相应的 TensorType
    types[0] = node->inputs().at(idx1)->type()->expect<TensorType>();
    types[1] = node->inputs().at(idx2)->type()->expect<TensorType>();
  }

  // 定义一个操作符集合，表示不能通过运行来传播其形状
  OperatorSet cannot_propagate_shape_by_running_it = {
      "aten::inverse(Tensor self) -> Tensor",
  };

  // 检查此节点是否依赖于先前已经发生变化的值
  std::unordered_map<Node*, bool> dependsOnMutationMemo_;
  bool dependsOnMutation(Node* node) {
    // 如果已经记录了节点是否依赖于变化，则直接返回相应值
    if (dependsOnMutationMemo_.count(node) != 0) {
      return dependsOnMutationMemo_[node];
    }

    // 如果别名分析数据库表明该节点有写入操作
    if (aliasDb_.hasWriters(node)) {
      // 表明节点依赖于变化的值，不能在隔离状态下运行
      dependsOnMutationMemo_[node] = true;
      return true;
    }

    // 递归检查输入的生产者节点，如果可变值通过纯函数被使用
    // 例如：
    //   a += 1
    //   c = a + b
    //   d = c + 1
    // 需要进行此检查
    // 表明节点不依赖于变化的值，可以在隔离状态下运行
    return false;
  }
    // 初始化一个布尔变量 `depends`，用于标记当前节点 `node` 是否依赖于任何可能的变异操作
    auto depends = false;

    // 遍历当前节点 `node` 的所有输入节点 `input`，检查每个输入节点是否依赖于变异操作
    for (auto input : node->inputs()) {
      depends |= dependsOnMutation(input->node());
    }

    // 将当前节点 `node` 是否依赖于变异操作的结果存储在 `dependsOnMutationMemo_` 中
    dependsOnMutationMemo_[node] = depends;

    // 返回当前节点 `node` 是否依赖于变异操作的布尔值
    return depends;
  }

  // 判断节点 `node` 是否可以通过运行来传播形状信息
  bool canPropagateShapeByRunningIt(Node* node) {
    // 如果节点 `node` 属于不能通过运行传播形状信息的特定类型，则直接返回 false
    if (node->isMemberOf(cannot_propagate_shape_by_running_it)) {
      return false;
    }

    // 如果节点 `node` 依赖于变异操作，则返回 false
    if (dependsOnMutation(node)) {
      return false;
    }

    // 检查节点 `node` 的所有输入是否都是运行的有效参数
    bool valid_args = std::all_of(
        node->inputs().begin(),
        node->inputs().end(),
        isValidArgumentForRunning);
    if (!valid_args)
      return false;

    // 检查节点 `node` 的所有输出是否都是运行的有效返回值
    bool valid_returns = std::all_of(
        node->outputs().begin(),
        node->outputs().end(),
        isValidReturnForRunning);
    if (!valid_returns)
      return false;

    // 如果以上条件都满足，则可以通过运行传播形状信息，返回 true
    return true;
  }

  // 判断节点 `node` 的输出是否不包含 Tensor 类型
  // 如果输出中没有 Tensor 类型，则不需要传播形状信息
  bool DoesntRefineOutputs(Node* node) {
    auto outputs = node->outputs();
    for (auto& out : outputs) {
      // 如果输出 `out` 包含 Tensor 类型，则返回 false
      if (containsTensorType(out->type())) {
        return false;
      }
    }
    // 如果所有输出都不包含 Tensor 类型，则返回 true
    return true;
  }

  // 根据运行节点 `node` 的操作 `op` 来传播节点的形状信息
  bool PropagateShapeOnNodeByRunningIt(Node* node, Operation op = nullptr) {
    // 如果节点 `node` 不能通过运行传播形状信息，则直接返回 false
    if (!canPropagateShapeByRunningIt(node))
      return false;

    // 如果未提供操作 `op`，则获取节点 `node` 的操作
    if (!op)
      op = node->getOperation();

    // 准备一个堆栈 `stack`，用于保存节点 `node` 的输入的代表值
    Stack stack;

    // 将节点 `node` 的所有输入的代表值推入堆栈 `stack` 中
    for (auto input : node->inputs()) {
      stack.push_back(representativeValue(input));
    }

    // XXX: 目前不捕获操作 `op` 的任何异常。这是为了在编辑代码时发现可能的错误，
    // 但最终不应该再重视，因为此阶段应在模式检查之前。
    op(stack);

    // 断言堆栈 `stack` 的大小与节点 `node` 的输出数目相等
    AT_ASSERT(stack.size() == node->outputs().size());

    // 遍历堆栈 `stack` 中的每个元素
    for (const auto i : c10::irange(stack.size())) {
      // 检查节点 `node` 的输出中的第 `i` 个元素的类型是否为 TensorType
      auto tensor_type = node->outputs()[i]->type()->cast<TensorType>();
      if (stack[i].isTensor() && tensor_type) {
        // 如果堆栈中的第 `i` 个元素是 Tensor 类型，并且输出的类型是 TensorType
        // 保持原始梯度属性信息，更新输出节点的类型为最新的 TensorType
        auto tensor_grad = tensor_type->requiresGrad();
        node->outputs()[i]->setType(TensorType::create(stack[i].toTensor())
                                        ->withRequiresGrad(tensor_grad));
      }
    }

    // 成功传播形状信息，返回 true
    return true;
  }

  // 传播 Cat 节点的形状信息
  void PropagateCatShape(Node* cat_node) {
    static const auto propagate_complete =
        [](Node* node, at::ArrayRef<Value*> tensors) -> bool {
      // 从输入的张量数组中提取每个张量的 TensorType，并放入 input_types 中
      auto input_types =
          fmap(tensors, [](Value* v) { return v->type()->cast<TensorType>(); });
      
      // 检查所有输入张量的类型是否都是完整的 TensorType
      if (!std::all_of(
              input_types.begin(),
              input_types.end(),
              [](const TensorTypePtr& tp) {
                return tp != nullptr && tp->isComplete();
              })) {
        return false;
      }
      
      // 检查节点是否是常数维度
      if (!node->is_constant(attr::dim))
        return false;
      
      // 获取第一个输入张量的尺寸，并计算出节点指定的维度
      std::vector<int64_t> sizes = *input_types[0]->sizes().concrete_sizes();
      const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      const int64_t ndim = (int64_t)sizes.size();

      // 如果节点指定的维度超出范围，则返回 false
      if (dim < 0 || dim >= ndim)
        return false;

      // 将节点指定维度的尺寸设为 0，并更新所有输入张量对应维度的尺寸
      sizes[dim] = 0;
      for (auto& tp : input_types) {
        auto tp_sizes = tp->sizes().concrete_sizes().value();
        // 如果输入张量的尺寸维度不一致，则返回 false
        if (sizes.size() != tp_sizes.size())
          return false;
        for (const auto i : c10::irange(ndim)) {
          // 检查除了节点指定的维度外的其他维度尺寸是否匹配
          if (sizes[i] != tp_sizes[i] && i != dim) {
            return false;
          }
        }
        // 更新节点指定维度的尺寸累加值
        sizes[dim] += tp_sizes[dim];
      }
      
      // 将节点输出的类型设置为第一个输入张量类型，并更新尺寸
      node->output()->setType(input_types[0]->withSizes(sizes));
      return true;
    };

    static const auto propagate = [](Node* node,
                                     at::ArrayRef<Value*> tensors) -> bool {
      // 遍历输入张量数组，如果存在张量类型，则将节点输出类型设置为该类型的 dimensionedOnly
      for (Value* v : tensors) {
        if (auto type = v->type()->cast<TensorType>()) {
          node->output()->setType(type->dimensionedOnly());
          return true;
        }
      }
      return false;
    };

    // 获取输入节点的 list_node
    auto list_node =
        ((cat_node->kind() == prim::FusedConcat)
             ? cat_node
             : cat_node->namedInput(attr::tensors)->node());

    // 如果 list_node 是 ListConstruct 或者 cat_node 是 FusedConcat
    if (list_node->kind() == prim::ListConstruct ||
        cat_node->kind() == prim::FusedConcat) {
      auto tensors = list_node->inputs();

      // 如果 tensors 非空
      if (!tensors.empty()) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        // 尝试使用 propagate_complete 函数进行传播，如果成功则返回
        if (propagate_complete(cat_node, tensors)) {
          return;
        } else if (propagate(cat_node, tensors)) {
          // 否则尝试使用 propagate 函数进行传播，如果成功则返回
          return;
        }
      }
    }
    
    // 如果以上传播均未成功，则将节点的输出类型设置为未定义形状
    setUnshapedType(cat_node);
  }

  void propagateTorchTensorShape(Node* node) {
    auto input_type = node->inputs().at(0)->type();

    // 初始化维度计数器和输入基本类型
    size_t dims = 0;
    auto input_base_type = input_type;
    auto list_type = input_type->cast<ListType>();
    
    // 循环直到找到非 ListType 的基本类型
    while (list_type) {
      dims++;
      input_base_type = list_type->getElementType();
      list_type = input_base_type->cast<ListType>();
    }

    // 尝试从 JitType 转换为标量类型
    at::optional<at::ScalarType> default_type =
        tryScalarTypeFromJitType(*input_base_type);
    
    // 如果节点的 schema 中有 'dtype' 参数
    if (auto grad_index = node->schema().argumentIndexWithName("dtype")) {
      auto inp = toIValue(node->inputs().at(*grad_index));
      // 如果 'dtype' 参数为 None，则返回
      if (inp == c10::nullopt) {
        return;
      } else if (!inp->isNone()) {
        // 否则将 'dtype' 参数的值作为默认类型
        default_type = inp->toScalarType();
      }
    }

    // 设置默认设备为 CPU
    at::Device default_device = at::kCPU;
    // 如果节点的参数中包含名为 "device" 的参数索引
    if (auto device_index = node->schema().argumentIndexWithName("device")) {
      // 将索引对应的输入节点转换为 IValue 类型
      auto inp = toIValue(node->inputs().at(*device_index));
      // 如果转换失败或者结果为空值，则直接返回
      if (inp == c10::nullopt) {
        return;
      } else if (!inp->isNone()) {
        // 如果输入不是 None 类型，则将其转换为设备类型并赋值给 default_device
        default_device = inp->toDevice();
      }
    }
    // 设置节点的输出类型为 TensorType，指定默认类型、设备、维度信息，并且不要求梯度信息
    node->output()->setType(TensorType::create(
        default_type, default_device, dims, /*requires_grad=*/c10::nullopt));
  }

  // 如果有任何值被设置为未定形类型（UnshapedType），返回 true
  bool setUnshapedTypeIfAliasResizedSet(at::ArrayRef<Value*> vs) {
    bool in_resize = false;
    // 遍历给定的值列表
    for (auto v : vs) {
      // 如果当前值可能与 resized_alias_set 中的任何值别名，则设置其为未定形类型
      if (aliasDb_.mayAlias(ValueSet{v}, resized_alias_set)) {
        setUnshapedType(v);
        in_resize = true;
      }
    }
    return in_resize;
  }

  // 重写 propagateNode 方法，用于节点的形状传播
  void propagateNode(Node* node, bool insert_expands = true) override {
    // 对于像 resize_ 这样的操作会改变输入张量的大小
    // 因为分析是流不变的，所以将任何可能与重设大小的张量别名的张量设置为基本张量类型（不包含大小信息）
    if (setUnshapedTypeIfAliasResizedSet(node->inputs())) {
      return setUnshapedType(node);
    }

    // 对于不需要类型信息且具有复杂模式的操作，处理后直接返回
    // 根据节点的副作用判断是否直接返回
    if (node->hasSideEffects()) {
      return;
    }

    // 对于特定的操作（如 aten::cat 或 prim::FusedConcat），调用 PropagateCatShape 进行形状传播
    if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor") ||
        node->kind() == prim::FusedConcat) {
      return PropagateCatShape(node);
    }

    // 收集节点的张量类型信息，如果成功获取完整的类型信息，则进行完整的形状传播
    if (auto maybe_complete_types =
            gatherTensorTypes(node, /*complete=*/true)) {
      if (PropagateCompleteShapeOnNode(
              node, insert_expands, std::move(*maybe_complete_types))) {
        return;
      }
    }

    // 根据节点的张量形状进行传播
    if (PropagateTensorShapeOnNode(node, insert_expands)) {
      return;
    }

    // 如果节点不会改变输出的细节，则直接返回
    if (DoesntRefineOutputs(node)) {
      return;
    }

    // 通过运行节点来传播形状
    if (PropagateShapeOnNodeByRunningIt(node)) {
      return;
    }
    // 否则，将节点设置为未定形类型
    return setUnshapedType(node);
  }

  // 确定列表类型值的大小，如果可以确定，则返回其大小
  static std::optional<size_t> determineListSize(Value* list) {
    AT_ASSERT(list->type()->cast<ListType>());
    // 如果列表是常量且可以转换为 List<int64_t>，则返回其大小
    if (auto shape = constant_as<c10::List<int64_t>>(list)) {
      return shape->size();
    }
    // 否则，获取列表构造节点的输入大小作为列表大小
    auto input_node = list->node();
    if (input_node->kind() == prim::ListConstruct) {
      return input_node->inputs().size();
    }
    return c10::nullopt;
  }

  // 是否可以尝试运行该操作
  // 如果输入是一个常量，则假设输入是有效的，可以尝试运行它。
  // 否则：
  // 整型类型的输入通常表示我们正在对张量进行索引，因此我们在形状传播中应特殊处理这些操作。
  // 此外，将零表示张量传递给整数除法操作会导致除以零错误。
  // 输出必须是张量或基本类型。
  // 我们将对张量调用inferTypeFrom，而忽略基本类型。
  // 但是，我们允许基本类型返回，因为我们希望支持混合的基本类型/张量输出。

  bool PropagateTensorShapeOnNode(Node* node, bool insert_expands) {
    static const auto broadcast =
        [](std::vector<TensorTypePtr>& tensor_types,
           std::optional<at::ScalarType> t) -> TensorTypePtr {
      if (tensor_types.size() == 1) {
        return tensor_types[0]->dimensionedOnly()->withScalarType(t);
      }
      AT_ASSERT(!tensor_types.empty());
      auto any_type = tensor_types[0];
      auto max_dims = any_type->dim();
      for (auto& type : tensor_types) {
        if (!max_dims || !type->dim()) {
          max_dims = c10::nullopt;
        } else {
          max_dims = std::max(*max_dims, *type->dim());
        }
      }
      return TensorType::create(
          t,
          any_type->device(),
          max_dims,
          /*requires_grad=*/c10::nullopt);
    };

    using type_vec_t = std::vector<TensorTypePtr>;
    // 预期公式将返回一个长度等于节点的张量输出数量的向量，或者一个空向量表示传播失败。
    using formula_t = std::function<type_vec_t(Node*)>;
    static std::mutex shape_formulas_mutex;
    static std::vector<std::pair<OperatorSet, formula_t>> shape_formulas;
    struct register_formula_for {
      register_formula_for(OperatorSet operators, formula_t formula) {
        std::unique_lock<std::mutex> lock{shape_formulas_mutex};
        shape_formulas.emplace_back(std::move(operators), std::move(formula));
      }
    };

    // 要求：
    //   dims           : 保持不变
    //   scalar type    : 保持不变
    //   device         : 保持不变
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // 此外：
    //   - 第一个输入应为唯一的张量输入
    // 要求：
    //   dims           : 保持不变
    //   scalar type    : 保持不变，但是复数映射到浮点数
    //   device         : 保持不变
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // 此外：
    //   - 第一个输入应为唯一的张量输入
    // 声明一个静态的注册函数对象，用于处理简单的一元操作，将复数类型转换为浮点数类型
    static const register_formula_for simple_unary_ops_complex_to_float{
        {
            "aten::abs(Tensor self) -> Tensor",  // 注册处理绝对值操作的函数
        },
        [](Node* node) -> type_vec_t {  // Lambda 函数，处理节点的类型推断
          auto input_type = node->input(0)->type()->cast<TensorType>();  // 获取输入节点的张量类型

          // 如果输入张量的标量类型存在
          if (input_type->scalarType()) {
            const auto scalar_type = *(input_type->scalarType());  // 获取标量类型
            // 如果是复数类型
            if (isComplexType(scalar_type)) {
              const auto out_type = c10::toRealValueType(scalar_type);  // 转换为实部类型
              return type_vec_t{
                  input_type->dimensionedOnly()->withScalarType(out_type)};  // 返回带有实部类型的张量
            }
          }

          return input_type ? type_vec_t{input_type->dimensionedOnly()}  // 返回维度已知的输入张量类型
                            : type_vec_t{};  // 如果没有输入类型，返回空类型向量
        }};

    // Requirements:
    //   dims           : broadcast all tensor args  // 要求广播所有张量参数
    //   scalar type    : promoted from input dtypes  // 标量类型从输入数据类型提升
    //   device         : always matching and preserved  // 设备始终匹配且保留
    //   tensor inputs  : *  // 张量输入：任意个数
    //   tensor outputs : 1  // 张量输出：一个

    // 声明一个静态的注册函数对象，用于处理广播的算术运算操作
    static const register_formula_for broadcasting_ops_arithmetic{
        {
            // Tensor-Tensor operators
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",  // 注册张量相加操作
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",  // 注册张量相减操作
            "aten::mul(Tensor self, Tensor other) -> Tensor",  // 注册张量相乘操作
            "aten::div(Tensor self, Tensor other) -> Tensor",  // 注册张量相除操作
        },
        [](Node* node) -> type_vec_t {  // Lambda 函数，处理节点的类型推断
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {  // 收集节点的张量类型
            AT_ASSERT(maybe_tensor_types->size() >= 2);  // 断言至少有两个张量类型
            auto dtype = getPromotedTypeForArithmeticOp(node);  // 获取算术运算操作的提升类型
            return {broadcast(*maybe_tensor_types, dtype)};  // 返回广播后的张量类型向量
          }
          return {};  // 如果没有收集到张量类型，返回空类型向量
        }};
    
    // Requirements:
    //   dims           : broadcast all tensor args  // 要求广播所有张量参数
    //   scalar type    : always matching and preserved  // 标量类型始终匹配且保留
    //   device         : always matching and preserved  // 设备始终匹配且保留
    //   tensor inputs  : *  // 张量输入：任意个数
    //   tensor outputs : 1  // 张量输出：一个
    // 定义静态常量，用于注册广播操作的公式
    static const register_formula_for broadcasting_ops{
        // 广播操作函数签名列表
        {
            "aten::pow(Tensor self, Tensor exponent) -> Tensor",
            "aten::fmod(Tensor self, Tensor other) -> Tensor",
            "aten::remainder(Tensor self, Tensor other) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor",
            "aten::max(Tensor self, Tensor other) -> Tensor",
            "aten::min(Tensor self, Tensor other) -> Tensor",
            "aten::__and__(Tensor self, Tensor other) -> Tensor",
            "aten::__or__(Tensor self, Tensor other) -> Tensor",
            "aten::__xor__(Tensor self, Tensor other) -> Tensor",
            "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__iand__(Tensor self, Tensor other) -> Tensor",
            "aten::__ior__(Tensor self, Tensor other) -> Tensor",
            "aten::__ixor__(Tensor self, Tensor other) -> Tensor",
            "aten::__ilshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__irshift__(Tensor self, Tensor other) -> Tensor",

            // 只有 Tensor-Tensor 重载的操作
            "aten::atan2(Tensor self, Tensor other) -> Tensor",
        },
        // 匿名函数，接受一个节点指针，返回一个类型向量
        [](Node* node) -> type_vec_t {
            // 调用 gatherTensorTypes 函数收集节点的 tensor 类型
            if (auto maybe_tensor_types = gatherTensorTypes(node)) {
                // 断言 tensor 类型至少有两个
                AT_ASSERT(maybe_tensor_types->size() >= 2);
                // 获取第一个和第二个 tensor 的标量类型
                auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
                auto second_scalar_type = (*maybe_tensor_types)[1]->scalarType();
                // 如果标量类型不存在，返回空向量
                if (!first_scalar_type || !second_scalar_type) {
                    return {};
                }
                // 初始化参数类型为第一个 tensor
                size_t arg_for_type = 0;
                // 如果通过 promoteTypes 函数升级第一个和第二个 tensor 的类型结果不是第一个 tensor 的类型
                if (c10::promoteTypes(*first_scalar_type, *second_scalar_type) !=
                    first_scalar_type) {
                    arg_for_type = 1;
                }
                // 获取最终的标量类型
                auto t = (*maybe_tensor_types)[arg_for_type]->scalarType();
                // 返回广播后的类型向量
                return {broadcast(*maybe_tensor_types, *t)};
            }
            // 如果无法收集到 tensor 类型，返回空向量
            return {};
        }};

    // 定义静态常量，用于注册融合累积二元操作的公式
    static const register_formula_for fused_accum_binary_ops{
        // 融合累积二元操作函数签名列表
        {
            // 非二元操作
            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
        },
        // 匿名函数，接受一个节点指针，返回一个类型向量
        [](Node* node) -> type_vec_t {
            // 调用 gatherTensorTypes 函数收集节点的 tensor 类型
            if (auto maybe_tensor_types = gatherTensorTypes(node)) {
                // 获取第一个 tensor 的标量类型
                auto dtype = (*maybe_tensor_types)[0]->scalarType();
                // 如果标量类型不存在，返回空向量
                if (!dtype) {
                    return {};
                }
                // 返回广播后的类型向量，使用第一个 tensor 的标量类型
                return {broadcast(*maybe_tensor_types, *dtype)};
            }
            // 如果无法收集到 tensor 类型，返回空向量
            return {};
        }};
    static const register_formula_for broadcasting_tensor_scalar_ops_arithmetic{
        {
            // 定义注册的公式集合，用于处理张量与标量的算术运算
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            "aten::div(Tensor self, Scalar other) -> Tensor",
        },
        // 注册处理函数，根据节点处理张量类型
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            // 获取第一个张量的标量类型
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            // 尝试获取第二个输入节点的标量类型
            auto second_scalar_type =
                tryScalarTypeFromJitType(*node->inputs()[1]->type());
            if (!first_scalar_type || !second_scalar_type) {
              return {};  // 如果标量类型无法获取，则返回空
            }
            // 如果第一个标量类型为整数且第二个标量类型为浮点数，选择默认的数据类型
            if (isIntegralType(*first_scalar_type, false) &&
                isFloatingType(*second_scalar_type)) {
              auto default_dtype =
                  at::typeMetaToScalarType(caffe2::get_default_dtype());
              return {broadcast(*maybe_tensor_types, default_dtype)};  // 进行广播操作
            }
            // 如果第一个标量类型为布尔型且第二个标量类型不是布尔型，提升类型
            if (c10::ScalarType::Bool == *first_scalar_type &&
                c10::ScalarType::Bool != *second_scalar_type) {
              auto result_type =
                  c10::promoteTypes(*first_scalar_type, *second_scalar_type);
              return {broadcast(*maybe_tensor_types, result_type)};  // 进行广播操作
            }
            // 返回广播操作后的类型结果
            return {broadcast(*maybe_tensor_types, first_scalar_type)};
          }
          return {};  // 如果无法获取张量类型，则返回空
        }};

    // NB: we always take the scalar type of the Tensor
    static const register_formula_for broadcasting_tensor_scalar_ops{
        {
            // 定义注册的公式集合，用于处理张量与标量的各种运算
            "aten::pow(Tensor self, Scalar exponent) -> Tensor",
            "aten::fmod(Tensor self, Scalar other) -> Tensor",
            "aten::remainder(Tensor self, Scalar other) -> Tensor",
            "aten::pow(Scalar self, Tensor exponent) -> Tensor",
            "aten::__and__(Tensor self, Scalar other) -> Tensor",
            "aten::__or__(Tensor self, Scalar other) -> Tensor",
            "aten::__xor__(Tensor self, Scalar other) -> Tensor",
            "aten::__lshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__rshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__iand__(Tensor self, Scalar other) -> Tensor",
            "aten::__ior__(Tensor self, Scalar other) -> Tensor",
            "aten::__ixor__(Tensor self, Scalar other) -> Tensor",
            "aten::__ilshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__irshift__(Tensor self, Scalar other) -> Tensor",
        },
        // 注册处理函数，根据节点处理张量类型
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            // 返回广播操作后的类型结果
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[0]->scalarType())};
          }
          return {};  // 如果无法获取张量类型，则返回空
        }};

    // aten::where is special in that its return type is the second argument's
    // 注册用于处理 `aten::where` 操作的公式
    static const register_formula_for where_op{
        {
            "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          // 收集节点输入张量的类型信息
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            // 返回广播后的张量类型向量
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[1]->scalarType())};
          }
          return {};
        }};

    // 定义一个 lambda 函数，用于获取节点中任意张量类型
    static const auto any_tensor_type = [](Node* node) -> TensorTypePtr {
      for (Value* input : node->inputs()) {
        if (auto type = input->type()->cast<TensorType>()) {
          // 如果张量类型具有维度信息，则返回该类型
          if (type->dim().has_value()) {
            return type;
          }
        }
      }
      return nullptr;
    };

    // 注册严格匹配二元操作的公式
    // 要求：
    //   dims           : 始终匹配和保留
    //   scalar type    : 始终匹配和保留
    //   device         : 始终匹配和保留
    //   tensor inputs  : 2
    //   tensor outputs : 1
    static const register_formula_for binary_ops_strict_match{
        {
            "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::mm(Tensor self, Tensor mat2) -> Tensor",
            "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          // 如果存在任意张量类型，则返回该类型向量
          if (auto type = any_tensor_type(node)) {
            return {std::move(type)};
          }
          return {};
        }};

    // 注册比较操作的公式
    // 要求：
    //   dims           : 所有张量参数都要广播
    //   scalar type    : byte/uint8
    //   device         : 始终匹配和保留
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for comparison_ops{
        {
            "aten::lt(Tensor self, Tensor other) -> Tensor",
            "aten::le(Tensor self, Tensor other) -> Tensor",
            "aten::gt(Tensor self, Tensor other) -> Tensor",
            "aten::ge(Tensor self, Tensor other) -> Tensor",
            "aten::eq(Tensor self, Tensor other) -> Tensor",
            "aten::ne(Tensor self, Tensor other) -> Tensor",
            "aten::lt(Tensor self, Scalar other) -> Tensor",
            "aten::le(Tensor self, Scalar other) -> Tensor",
            "aten::gt(Tensor self, Scalar other) -> Tensor",
            "aten::ge(Tensor self, Scalar other) -> Tensor",
            "aten::eq(Tensor self, Scalar other) -> Tensor",
            "aten::ne(Tensor self, Scalar other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          // 收集节点的张量类型信息并进行广播，返回布尔类型的张量向量
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(*maybe_tensor_types, at::kBool)};
          }
          return {};
        }};
    // 定义静态常量 register_formula_for，注册用于处理 nn_ops_first_input_formula 的公式
    static const register_formula_for nn_ops_first_input_formula{
        // 使用 nn_ops_first_input_preserving() 函数注册保持第一个输入的操作
        *nn_ops_first_input_preserving(), [](Node* node) -> type_vec_t {
          // 检查节点的第一个输入是否为 TensorType 类型
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            // 返回包含只有维度信息的 TensorType 类型的向量
            return {type->dimensionedOnly()};
          }
          // 如果第一个输入不是 TensorType 类型，则返回空向量
          return {};
        }};
    
    // Requirements:
    //   dims           : 0
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for all_reduce_ops{
        // 注册用于处理 all_reduce_ops 的公式，针对指定的操作字符串
        {
            "aten::det(Tensor self) -> Tensor",
            "aten::logdet(Tensor self) -> Tensor",
            "aten::max(Tensor self) -> Tensor",
            "aten::min(Tensor self) -> Tensor",
            "aten::median(Tensor self) -> Tensor",
            "aten::nanmedian(Tensor self) -> Tensor",
            "aten::norm(Tensor self, Scalar p) -> Tensor",
            "aten::std(Tensor self, bool unbiased) -> Tensor",
            "aten::trace(Tensor self) -> Tensor",
            "aten::var(Tensor self, bool unbiased) -> Tensor",
            "aten::all(Tensor self) -> Tensor",
            "aten::any(Tensor self) -> Tensor",
        },
        // 匿名函数，处理节点以返回类型向量
        [](Node* node) -> type_vec_t {
          // 检查节点的第一个输入是否为 TensorType 类型
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            // 返回包含只有维度信息的 TensorType 类型的向量，将维度设为 0
            return {type->withDim(0)};
          }
          // 如果第一个输入不是 TensorType 类型，则返回空向量
          return {};
        }};
    
    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for reduce_ops_with_opt_dtype{
        // 注册用于处理 reduce_ops_with_opt_dtype 的公式，仅对指定的操作字符串生效
        {"aten::mean(Tensor self, *, int? dtype) -> Tensor"},
        // 匿名函数，处理节点以返回类型向量
        [](Node* node) -> type_vec_t {
          // 获取节点的 dtype 属性值
          at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
          // 检查节点的第一个输入是否为 TensorType 类型
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto ret = type->withDim(0); // 返回 TensorType 类型，设置维度为 0
            // 如果节点的 dtype 属性有值且不为 None，则设置返回类型的标量类型为指定的 dtype
            if (maybe_dtype_option && !maybe_dtype_option->isNone()) {
              return {ret->withScalarType(maybe_dtype_option->toScalarType())};
            } else {
              // 否则返回保持标量类型的 TensorType 类型
              return {std::move(ret)};
            }
          }
          // 如果第一个输入不是 TensorType 类型，则返回空向量
          return {};
        }};
    static const register_formula_for
        all_reduce_ops_with_integer_upcast_and_dtype{
            {
                "aten::sum(Tensor self, *, int? dtype) -> Tensor",
                "aten::prod(Tensor self, *, int? dtype) -> Tensor",
            },
            [](Node* node) -> type_vec_t {
              // 检查节点的第一个输入是否为张量类型
              if (auto type = node->input(0)->type()->cast<TensorType>()) {
                // 将张量类型的维度设置为0
                type = type->withDim(0);
                // 获取节点的dtype属性，如果存在且非空，则设置类型的标量类型为dtype指定的类型
                at::optional<IValue> maybe_dtype_option =
                    node->get(attr::dtype);
                if (maybe_dtype_option && !maybe_dtype_option->isNone()) {
                  return {
                      type->withScalarType(maybe_dtype_option->toScalarType())};
                }
                // 如果类型已经有标量类型，则根据标量类型判断是否需要转换为浮点类型
                if (type->scalarType()) {
                  return {
                      at::isFloatingType(*type->scalarType())
                          ? std::move(type)
                          : type->withScalarType(at::kLong)};
                } else {
                  // 否则直接返回类型
                  return {std::move(type)};
                }
              }
              // 如果输入不是张量类型，则返回空类型向量
              return {};
            }};

    static const auto reduce_op_handler = [](Node* node,
                                             int64_t num_reduced_dim = 0,
                                             bool upcast_integer = false,
                                             std::optional<IValue> opt_dtype =
                                                 c10::nullopt) -> type_vec_t {
      // 检查节点的第一个输入是否为张量类型
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        // 如果张量类型的标量类型或维度未定义，则返回空类型向量
        if (!type->scalarType() || !type->dim()) {
          return {};
        }
        // 如果提供了dtype选项且非空，则根据dtype设置类型的标量类型
        if (opt_dtype && !opt_dtype->isNone()) {
          type = type->withScalarType(opt_dtype->toScalarType());
        } else if (upcast_integer && !at::isFloatingType(*type->scalarType())) {
          // 如果需要向上转型整数并且当前标量类型不是浮点类型，则设置标量类型为长整型
          type = type->withScalarType(at::kLong);
        }
        // 如果张量维度大于等于要减少的维度数，并且要减少的维度数大于0，则减少相应维度
        if (static_cast<int64_t>(*type->dim()) >= num_reduced_dim &&
            num_reduced_dim > 0) {
          return {type->withDim(*type->dim() - num_reduced_dim)};
        } else {
          // 否则直接返回类型
          return {std::move(type)};
        }
      }
      // 如果输入不是张量类型，则返回空类型向量
      return {};
    };

    static const auto multidim_reduce_with_keepdim =
        [](Node* node,
           int64_t num_reduced_dim,
           bool upcast_integer) -> type_vec_t {
      // 获取keepdim属性的布尔值
      auto maybe_keepdim = node->get<bool>(attr::keepdim);
      if (!maybe_keepdim)
        // 如果keepdim属性不存在，则返回空类型向量
        return {};
      // 根据keepdim属性的值调用reduce_op_handler处理节点
      return reduce_op_handler(
          node, *maybe_keepdim ? 0 : num_reduced_dim, upcast_integer);
    };

    // Requirements:
    //   dims           : 如果dim为None，则为0，否则如果keepdim == false则保持不变，否则减小1
    //   scalar type    : 保持不变
    //   device         : 保持不变
    //   tensor inputs  : 1个张量输入
    //   outputs        : 1个张量输出
    // Additionally:
    //   - 第一个输入应该是唯一的张量输入
    //   - 必须有一个布尔类型的keepdim参数
    // 定义静态常量注册表达式，用于处理 argmin 和 argmax 操作
    static const register_formula_for argminmax{
        {
            "aten::argmax(Tensor self, int? dim, bool keepdim) -> Tensor",
            "aten::argmin(Tensor self, int? dim, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
            // 检查第一个输入是否为 Tensor 类型
            if (auto type = node->input(0)->type()->cast<TensorType>()) {
                // 如果第二个输入 dim 的类型为 NoneType，则返回结果类型为保留第一维度
                if (node->input(1)->type()->kind() == c10::TypeKind::NoneType) {
                    return {type->withDim(0)};
                } else {
                    // 否则，执行多维度保留或者不保留操作
                    return multidim_reduce_with_keepdim(
                        node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
                }
            }
            // 如果条件不满足，返回空类型向量
            return {};
        }};
    
    // Requirements:
    //   dims           : 如果 keepdim == false 则保留，否则减小1维
    //   scalar type    : 对于第一个输出保留，对于第二个为 byte/uint8
    //   output 若存在设备 : 保留张量输入 : 1个张量
    //   输出 : 1个或2个
    // Additionally:
    //   - 第一个输入应为唯一的张量输入
    //   - 有一个 bool 类型的 keepdim 参数
    static const register_formula_for dim_reduce_ops{
        {
            "aten::all(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::any(Tensor self, int dim, bool keepdim) -> Tensor",
    
            // 返回索引作为第二个输出的操作
            "aten::kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::max(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::min(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::median(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::nanmedian(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::mode(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
        },
        [](Node* node) -> type_vec_t {
            // 注意：虽然这个函数通常用于只有一个输出的操作，我们会在下面修复它的返回结果
            auto output_types = multidim_reduce_with_keepdim(
                node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
            // 如果输出类型非空且节点输出大小为2，则添加额外的 long 类型输出
            if (!output_types.empty() && node->outputs().size() == 2) {
                output_types.push_back(
                    output_types.back()->withScalarType(at::kLong));
            }
            return output_types;
        }};
    
    // Requirements:
    //   dims           : 如果 keepdim == false 则保留，否则减小1维
    //   scalar type    : 如果指定了 dtype，则保留；如果是浮点数，则保留，否则为 long/int64
    //   device 若存在 : 保留张量输入 : 1个张量输出 : 1个张量
    // Additionally:
    //   - 第一个输入应为唯一的张量输入
    //   - 有一个 bool 类型的 keepdim 参数
    // 定义静态常量对象 `dim_reduce_ops_with_integer_upcast`
    static const register_formula_for dim_reduce_ops_with_integer_upcast{
        {
            "aten::prod(Tensor self, int dim, bool keepdim, *, int? dtype) -> Tensor",  // 注册的操作名及其签名
        },
        [](Node* node) -> type_vec_t {
            auto maybe_keepdim = node->get<bool>(attr::keepdim);  // 获取是否保持维度的信息
            at::optional<IValue> opt_dtype = node->get(attr::dtype);  // 获取数据类型的可选值
            return reduce_op_handler(
                node,
                /*num_reduce_dim=*/maybe_keepdim ? 0 : 1,  // 计算需要减少的维度数量，如果保持维度则为0，否则为1
                /*integer_upcast=*/true,  // 标志是否进行整数升级
                std::move(opt_dtype));  // 移动数据类型的可选值作为参数
        }};
    
    // 注释：
    // Requirements:
    //   dims           : 保持不变
    //   scalar type    : 如果指定了 dtype，则为 dtype；如果是浮点型则保持不变，否则为 long/int64
    //   device         : 保持不变
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - 第一个输入应该是唯一的张量输入
    
    // 定义静态常量对象 `dim_reduce_ops_dtype`
    static const register_formula_for dim_reduce_ops_dtype{
        {"aten::cumprod(Tensor self, int dim, *, int? dtype) -> Tensor",
         "aten::cumsum(Tensor self, int dim, *, int? dtype) -> Tensor",
         "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"},  // 注册的操作名及其签名
        [](Node* node) -> type_vec_t {
            at::optional<IValue> opt_dtype = node->get(attr::dtype);  // 获取数据类型的可选值
            return reduce_op_handler(
                node,
                /*num_reduce_dim=*/0,  // 减少的维度数量为0
                /*integer_upcast=*/true,  // 标志是否进行整数升级
                std::move(opt_dtype));  // 移动数据类型的可选值作为参数
        }};
    
    // 注释：
    // Requirements:
    //   dims           : 保持不变
    //   scalar type    : 如果指定了 dtype，则为 dtype，否则保持不变
    //   device         : 保持不变
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - 有 bool keepdim 和 int[] dim 参数
    
    // 定义静态常量对象 `register_softmax`
    static const register_formula_for register_softmax{
        {"aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"},  // 注册的操作名及其签名
        [](Node* node) -> type_vec_t {
            at::optional<IValue> opt_dtype = node->get(attr::dtype);  // 获取数据类型的可选值
            return reduce_op_handler(
                node,
                /*num_reduced_dim=*/0,  // 减少的维度数量为0
                /*upcast_integer=*/false,  // 标志是否进行整数升级
                std::move(opt_dtype));  // 移动数据类型的可选值作为参数
        }};
    
    // 注释：
    // Requirements:
    //   dims           : 保持不变
    //   scalar type    : 如果指定了 dtype，则为 dtype，否则保持不变
    //   device         : 保持不变
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - 有 bool keepdim 和 int[] dim 参数
    // 定义一个 lambda 函数 factory_with_ndim，接受一个节点 node、维度 dim 和默认数据类型 default_dtype 作为参数，并返回一个类型向量 type_vec_t
    static const auto factory_with_ndim =
        [](Node* node, int dim, at::ScalarType default_dtype) -> type_vec_t {
      
      // 尝试获取节点的 layout 属性值
      at::optional<IValue> maybe_layout_option = node->get(attr::layout);
      // 如果 layout 属性不存在，则返回空类型向量
      if (!maybe_layout_option)
        return {};

      // 尝试获取节点的 device 属性值
      at::optional<IValue> maybe_device_option = node->get(attr::device);
      // 如果 device 属性不存在，则返回空类型向量
      if (!maybe_device_option)
        return {};
      
      // 根据 device 属性值确定设备类型，若为 None 则设为 CPU
      auto device =
          (maybe_device_option->isNone() ? at::kCPU
                                         : maybe_device_option->toDevice());

      // 尝试获取节点的 dtype 属性值
      at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
      // 如果 dtype 属性不存在，则返回空类型向量
      if (!maybe_dtype_option)
        return {};

      // 根据 dtype 属性值确定数据类型，若为 None 则设为 default_dtype
      auto dtype =
          (maybe_dtype_option->isNone() ? default_dtype
                                        : maybe_dtype_option->toScalarType());

      // 创建一个包含单个元素的类型向量，元素为一个 TensorType 对象，指定数据类型、设备、维度及不需要梯度信息
      return {TensorType::create(
          dtype, device, dim, /*requires_grad=*/c10::nullopt)};
    };

    // 定义一个 lambda 函数 factory_like_with_ndim，接受一个节点 node 和维度 dim 作为参数，并返回一个类型向量 type_vec_t
    static const auto factory_like_with_ndim = [](Node* node,
                                                  int dim) -> type_vec_t {
      // 获取节点的第一个输入的类型，期望是 TensorType
      auto tt = node->input(0)->type()->expect<TensorType>();
      // 获取输入的数据类型和设备
      auto in_type = tt->scalarType();
      auto in_dev = tt->device();

      // 尝试获取节点的 layout 属性值
      at::optional<IValue> maybe_layout_option = node->get(attr::layout);
      // 如果 layout 属性不存在，则返回空类型向量
      if (!maybe_layout_option)
        return {};

      // 尝试获取节点的 device 属性值
      at::optional<IValue> maybe_device_option = node->get(attr::device);
      // 如果 device 属性不存在，则返回空类型向量
      if (!maybe_device_option)
        return {};

      // 如果 device 属性值不为 None，则更新输入的设备类型
      if (!maybe_device_option->isNone()) {
        in_dev = maybe_device_option->toDevice();
      }

      // 尝试获取节点的 dtype 属性值
      at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
      // 如果 dtype 属性不存在，则返回空类型向量
      if (!maybe_dtype_option)
        return {};

      // 如果 dtype 属性值不为 None，则更新输入的数据类型
      if (!maybe_dtype_option->isNone()) {
        in_type = maybe_dtype_option->toScalarType();
      }

      // 创建一个包含单个元素的类型向量，元素为一个 TensorType 对象，指定数据类型、设备、维度及不需要梯度信息
      return {TensorType::create(
          in_type, in_dev, dim, /*requires_grad=*/c10::nullopt)};
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has ScalarType dtype, Layout layout and Device device arguments
    // 定义静态常量注册公式，用于处理带有选项的类似工厂函数
    static const register_formula_for like_factories_with_options{
        {
            // 第一个工厂函数：返回一个与输入张量相同大小和类型的空张量
            "aten::empty_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第二个工厂函数：返回一个与输入张量相同大小和类型的张量，且每个元素填充为指定值
            "aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第三个工厂函数：返回一个与输入张量相同大小和类型的全1张量
            "aten::ones_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第四个工厂函数：返回一个与输入张量相同大小和类型的随机值张量
            "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第五个工厂函数：返回一个与输入张量相同大小和类型的随机整数张量，范围在[0, high)
            "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第六个工厂函数：返回一个与输入张量相同大小和类型的随机整数张量，范围在[low, high)
            "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第七个工厂函数：返回一个与输入张量相同大小和类型的随机正态分布张量
            "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            // 第八个工厂函数：返回一个与输入张量相同大小和类型的全0张量
            "aten::zeros_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
        },
        // Lambda 函数定义：根据输入节点的维度信息调用相应的工厂函数
        [](Node* node) -> type_vec_t {
            if (auto type = node->namedInput(attr::self)->type()->cast<TensorType>()) {
                // 如果输入张量类型有定义维度
                if (type->dim()) {
                    // 调用 factory_like_with_ndim 函数，传入节点和张量的维度数
                    return factory_like_with_ndim(node, (int)*type->dim());
                }
            }
            // 如果没有符合条件的情况，返回空向量
            return {};
        }
    };
    
    // 要求：
    //   dims           : 等于 size 中元素的数量
    //   scalar type    : 等于 dtype 的值
    //   device         : 等于 device 的值
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // 此外：
    //   - 必须有 int[] size, ScalarType dtype, Layout layout 和 Device device
    //   的参数
    static const register_formula_for size_factories_with_options{
        {
            // 注册公式，处理空张量的情况，支持不同选项
            "aten::empty(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory, MemoryFormat? memory_format=contiguous_format) -> Tensor",
            // 注册公式，创建指定大小并填充指定值的张量
            "aten::full(int[] size, Scalar fill_value, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            // 注册公式，创建指定大小并填充为1的张量
            "aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            // 注册公式，创建指定大小并填充随机值的张量
            "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            // 注册公式，创建指定大小并填充从标准正态分布中抽取的随机数的张量
            "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            // 注册公式，创建指定大小并填充为0的张量
            "aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
            // 如果节点具有大小属性，返回相应的张量类型工厂
            if (auto maybe_size = node->get<c10::List<int64_t>>(attr::size)) {
                return factory_with_ndim(node, (int)maybe_size->size(), at::kDouble);
            }
            return {};
        }};

    static const register_formula_for randint{
        {
            // 注册公式，创建指定范围内随机整数的张量
            "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            // 注册公式，创建指定范围内随机整数的张量
            "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
            // 如果节点具有大小属性，返回相应的张量类型工厂
            if (auto maybe_size = node->get<c10::List<int64_t>>(attr::size)) {
                return factory_with_ndim(node, (int)maybe_size->size(), at::kLong);
            }
            return {};
        }};

    static const auto get_cast_scalar_type = [](Node* node) -> at::ScalarType {
        // 根据节点类型返回对应的标量类型
        switch (node->kind()) {
            case aten::_cast_Byte:
                return at::kByte;
            case aten::_cast_Char:
                return at::kChar;
            case aten::_cast_Double:
                return at::kDouble;
            case aten::_cast_Float:
                return at::kFloat;
            case aten::_cast_Half:
                return at::kHalf;
            case aten::_cast_Int:
                return at::kInt;
            case aten::_cast_Long:
                return at::kLong;
            case aten::_cast_Short:
                return at::kShort;
            default:
                // 如果节点类型未知，抛出断言错误
                AT_ASSERTM(
                    false,
                    "unknown node kind in get_cast_scalar_type: ",
                    node->kind().toQualString());
        }
    };
    // 定义静态常量 register_formula_for，注册了多个张量类型转换操作的公式
    static const register_formula_for cast_ops{
        {
            "aten::_cast_Byte(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Char(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Double(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Float(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Half(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Int(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Long(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Short(Tensor self, bool non_blocking) -> Tensor",
        },
        // 匿名函数，根据节点类型推断输出类型的处理逻辑
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            return {type->withScalarType(get_cast_scalar_type(node))};
          }
          return {};
        }};

    // 遍历 shape_formulas 中的条目，尝试匹配节点的成员操作符
    for (auto& entry : shape_formulas) {
      if (node->isMemberOf(entry.first)) {
        // 调用 entry.second 处理节点，获取类型信息
        auto types = entry.second(node);
        if (types.empty()) {
          return false;
        } else {
          // 确保输出节点的类型与计算结果一致
          auto outputs = node->outputs();
          AT_ASSERT(types.size() == outputs.size());
          for (const auto i : c10::irange(types.size())) {
            AT_ASSERT(outputs[i]->type()->isSubtypeOf(*TensorType::get()));
            outputs[i]->setType(types[i]);
          }
          return true;
        }
      }
    }

    // 对于一些特定节点，执行形状推断，仅需部分输入类型信息
    const auto input_type = [node](size_t index) {
      auto result = node->input(index)->type()->cast<TensorType>();
      if (result) {
        result = result->dimensionedOnly();
      }
      return result;
    };

    // 处理特定的节点匹配情况，设置输出节点的类型
    if (node->matches(
            "aten::masked_select(Tensor self, Tensor mask) -> Tensor")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (node->matches("aten::detach(Tensor(a) self) -> Tensor(a)")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type->withRequiresGrad(false));
        return true;
      }
    } else if (
        node->matches(
            "aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        if (type->scalarType() == at::kHalf) {
          type = type->withScalarType(at::kFloat);
        }
        type = type->withDim(1);
        node->outputs()[0]->setType(type);
        node->outputs()[1]->setType(std::move(type));
        return true;
      }
    } else if (node->matches(
                   "aten::dot(Tensor self, Tensor tensor) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    }
    } else if (
        node->matches("aten::mv(Tensor self, Tensor vec) -> Tensor") ||
        node->matches(
            "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta, Scalar alpha) -> Tensor")) {
      // 检查节点是否匹配 mv 或 addmv 操作
      if (auto type = any_tensor_type(node)) {
        // 设置节点输出类型为包含一维的张量类型
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (
        node->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      // 检查节点是否匹配 addmm、addbmm 或 addr 操作
      if (auto type = any_tensor_type(node)) {
        // 设置节点输出类型为包含二维的张量类型
        node->output()->setType(type->withDim(2));
        return true;
      }
    } else if (
        node->matches(
            "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      // 检查节点是否匹配 baddbmm 操作
      if (auto type = any_tensor_type(node)) {
        // 设置节点输出类型为包含三维的张量类型
        node->output()->setType(type->withDim(3));
        return true;
      }
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // 对于 index_select 操作，处理特殊情况：当 self.dim() == 0 时的输出类型
      if (type && index_type && type->dim()) {
        if (*type->dim() == 0) {
          node->output()->setType(type->withDim(index_type->dim()));
        } else {
          node->output()->setType(std::move(type));
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // 处理 gather 操作中的特殊情况：当 self 是一维且 index 是零维时的输出类型
      if (type && index_type && index_type->dim()) {
        if (*index_type->dim() == 0) {
          node->output()->setType(type->withDim(0));
        } else {
          node->output()->setType(std::move(type));
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")) {
      auto weight_type = input_type(0);
      auto indices_type = input_type(1);
      // 处理 embedding 操作的输出类型，根据 indices 的维度确定
      if (weight_type && indices_type && indices_type->dim()) {
        node->output()->setType(weight_type->withDim(*indices_type->dim() + 1));
        return true;
      }
    // 如果节点匹配 bilinear 函数的签名，处理输入类型并设置输出类型
    } else if (
        node->matches(
            "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor")) {
      // 获取第一个输入的类型并设置为输出类型
      if (auto type = input_type(0)) {
        node->output()->setType(std::move(type));
        return true;
      }
      // 获取第二个输入的类型并设置为输出类型
      if (auto type = input_type(1)) {
        node->output()->setType(std::move(type));
        return true;
      }
    } else if (
        // 如果节点匹配 dist 函数的签名，处理节点的输出类型
        node->matches(
            "aten::dist(Tensor self, Tensor other, Scalar p) -> Tensor")) {
      // 获取节点的任意张量类型并设置为输出类型，带有零维度信息
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    }

    // 以下代码实现了需要所有张量输入的类型信息，并且只有一个输出的公式
    std::vector<TensorTypePtr> tensor_types;
    // reshape_prop 是一个 lambda 函数，用于处理节点的形状重塑属性
    static const auto reshape_prop =
        [](Node* node,
           Symbol shape_input,
           const std::vector<TensorTypePtr>& tensor_types) -> TensorTypePtr {
      // 确定命名输入的列表大小
      if (auto list_size = determineListSize(node->namedInput(shape_input))) {
        // 返回第一个张量类型，但带有确定的维度大小
        return tensor_types.at(0)->withDim(*list_size);
      }
      return nullptr;
    };
    // gatherTensorTypes 函数收集节点的张量类型信息
    if (auto maybe_tensor_types = gatherTensorTypes(node)) {
      tensor_types = std::move(*maybe_tensor_types);
    } else {
      return false;
    }
    // 如果节点只有一个输出
    if (node->outputs().size() == 1) {
      // 获取单一输出的类型并设置为节点的输出类型
      if (auto type = getSingleOutputType()) {
        node->output()->setType(std::move(type));
        return true;
      }
    }
    // 默认返回 false
    return false;
  }

  // PropagateCompleteShapeOnNode 函数用于节点上的完整形状传播
  bool PropagateCompleteShapeOnNode(
      Node* node,
      bool insert_expands,
      std::vector<TensorTypePtr> tensor_types) {
    // 对于昂贵的操作，直接编码其形状传播，否则回退到运行操作的虚拟版本
    if (node->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Tensor other) -> Tensor")) {
      // 这些节点内部处理不同形状的张量，不需要插入显式的扩展节点
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (node->matches(
                   "aten::div(Tensor self, Tensor other) -> Tensor")) {
      // "div" 函数内部处理不同形状的张量，不需要插入显式的扩展节点
      // 注意，这个函数可以合并到上面一个函数中，但由于 "div" 函数可能因为整数除以零不安全，所以单独处理为运行 "mul" 操作来模拟执行
      auto op = getOperatorForLiteral(
                    "aten::mul(Tensor self, Tensor other) -> Tensor")
                    ->getOperation();
      return PropagateShapeOnNodeByRunningIt(node, std::move(op));
    // 如果节点匹配 aten::pow(Tensor self, Scalar exponent) -> Tensor
    } else if (node->matches(
                   "aten::pow(Tensor self, Scalar exponent) -> Tensor")) {
      // 设置节点输出类型为 tensor_types 的第一个类型
      node->output()->setType(tensor_types.at(0));
      // 返回 true 表示成功处理节点
      return true;
    // 如果节点匹配以下操作：aten::add, aten::sub, aten::div, aten::mul
    } else if (
        node->matches(
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches("aten::div(Tensor self, Scalar other) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Scalar other) -> Tensor")) {
      // 获取第一个标量类型
      auto first_scalar_type = (tensor_types)[0]->scalarType();
      // 尝试获取第二个标量类型
      auto second_scalar_type =
          tryScalarTypeFromJitType(*node->inputs()[1]->type());
      // 如果任一标量类型为空，则返回 false
      if (!first_scalar_type || !second_scalar_type) {
        return false;
      }
      // 如果第一个标量类型是整数类型且第二个标量类型是浮点类型
      if (isIntegralType(*first_scalar_type, false) &&
          isFloatingType(*second_scalar_type)) {
        // 获取默认数据类型
        auto default_dtype =
            at::typeMetaToScalarType(caffe2::get_default_dtype());
        // 使用默认数据类型创建新的类型
        auto type = tensor_types[0]->withScalarType(default_dtype);
        // 设置节点输出类型为新创建的类型
        node->output()->setType(std::move(type));
        // 返回 true 表示成功处理节点
        return true;
      }
      // 如果第一个标量类型是布尔类型且第二个标量类型不是布尔类型
      if (c10::ScalarType::Bool == *first_scalar_type &&
          c10::ScalarType::Bool != *second_scalar_type) {
        // 通过提升类型来获取结果类型
        auto result_type =
            c10::promoteTypes(*first_scalar_type, *second_scalar_type);
        // 使用结果类型创建新的类型
        auto type = tensor_types[0]->withScalarType(result_type);
        // 设置节点输出类型为新创建的类型
        node->output()->setType(std::move(type));
        // 返回 true 表示成功处理节点
        return true;
      }
      // 使用第一个标量类型创建新的类型
      auto type = tensor_types[0]->withScalarType(first_scalar_type);
      // 设置节点输出类型为新创建的类型
      node->output()->setType(std::move(type));
      // 返回 true 表示成功处理节点
      return true;
    // 如果 insert_expands 为真且节点匹配以下操作：aten::pow, aten::min, aten::max 等等
    } else if (
        insert_expands &&
        (node->matches("aten::pow(Tensor self, Tensor exponent) -> Tensor") ||
         node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::le(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::gt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ge(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::eq(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ne(Tensor self, Tensor other) -> Tensor"))) {
      // 二进制广播操作
      // 注意：我们不以任何其他方式处理节点（请注意缺少的 return！），因为标量情况下的类型转换逻辑非常复杂。
      // 直接运行它们通常更好。
      // 对节点进行广播处理
      broadcastBinary(node, tensor_types, 0, 1);
      // 通过运行节点来传播形状
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (
        node->matches(
            "aten::logit(Tensor self, float? eps = None) -> Tensor") ||
        node->matches("aten::neg(Tensor self) -> Tensor") ||
        node->matches("aten::sigmoid(Tensor self) -> Tensor") ||
        node->matches("aten::tanh(Tensor self) -> Tensor")) {
      // 如果节点匹配以下任一操作：logit、neg、sigmoid、tanh，则执行以下操作
      node->output()->setType(tensor_types.at(0)->contiguous());
      // 设置节点输出类型为第一个张量类型的连续形式
      return true;
    } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      // 如果节点匹配 matmul 操作，则执行以下操作
      auto lhs_type = tensor_types.at(0);
      auto rhs_type = tensor_types.at(1);
      auto lhs_sizes = lhs_type->sizes().concrete_sizes().value();
      auto rhs_sizes = rhs_type->sizes().concrete_sizes().value();
      SHAPE_ASSERT(
          *lhs_type->sizes().size() == 2 && *rhs_type->sizes().size() == 2);
      // 确保张量类型的尺寸均为二维
      node->output()->setType(TensorType::createContiguous(
          *lhs_type->scalarType(),
          *lhs_type->device(),
          at::IntArrayRef{lhs_sizes[0], rhs_sizes[1]}));
      // 设置节点输出类型为二维张量，大小为 (lhs_sizes[0], rhs_sizes[1])
      return true;
    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      // 如果节点匹配转置操作，则执行以下操作
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto strides = tp->strides().concrete_sizes().value();
      SHAPE_ASSERT(sizes.size() == 2);
      // 确保张量类型的尺寸为二维
      std::swap(sizes.at(0), sizes.at(1));
      std::swap(strides.at(0), strides.at(1));
      // 交换尺寸和步幅的第一和第二维度
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      // 设置节点输出类型为具有交换后尺寸和步幅的张量类型
      return true;
    } else if (
        node->matches(
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::length})) {
      // 如果节点匹配 narrow 操作，并且包含维度和长度的常量输入，则执行以下操作
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t length = node->get<int64_t>(attr::length).value();
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      // 确保维度值在有效范围内
      sizes.at(dim) = length;
      // 设置指定维度的大小为指定长度
      node->output()->setType(
          tp->withSizesStrides(sizes, tp->strides().concrete_sizes().value()));
      // 设置节点输出类型为具有更新尺寸和步幅的张量类型
      return true;
    } else if (node->matches(
                   "aten::sum(Tensor self, *, int? dtype) -> Tensor")) {
      // 如果节点匹配 sum 操作，则执行以下操作
      node->output()->setType(tensor_types.at(0)->withSizes({}));
      // 设置节点输出类型为与第一个张量类型相同，但尺寸为空的张量类型
      return true;
    } else if (
        node->matches(
            "aten::sum(Tensor self, int[]? dim, bool keepdim, *, int? dtype) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::keepdim})) {
      // 获取第一个张量类型
      auto& tp = tensor_types.at(0);
      // 获取张量的具体尺寸
      auto sizes = tp->sizes().concrete_sizes().value();
      // 获取节点的维度参数
      auto dims = node->get<c10::List<int64_t>>(attr::dim).value();
      // 获取节点的 keepdim 参数
      bool keepdim = node->get<bool>(attr::keepdim).value();
      // 反转维度列表
      std::reverse(dims.begin(), dims.end());
      // 遍历每一个维度
      for (int64_t dim : dims) {
        // 断言维度在有效范围内
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        // 根据 keepdim 参数处理尺寸
        if (keepdim) {
          sizes.at(dim) = 1;
        } else {
          sizes.erase(sizes.begin() + dim);
        }
      }
      // 设置节点输出的张量类型
      node->output()->setType(tp->withSizes(sizes));
      // 返回处理成功
      return true;
    } else if (node->matches(
                   "aten::squeeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      // 获取第一个张量类型
      auto& tp = tensor_types.at(0);
      // 获取张量的具体尺寸
      auto sizes = tp->sizes().concrete_sizes().value();
      // 获取张量的步长
      auto strides = tp->strides().concrete_sizes().value();
      // 获取节点的维度参数
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      // 断言维度在有效范围内
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      // 如果维度是 1，则压缩维度
      if (sizes.at(dim) == 1) {
        sizes.erase(sizes.begin() + dim);
        strides.erase(strides.begin() + dim);
      }
      // 设置节点输出的张量类型
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      // 返回处理成功
      return true;
    } else if (node->matches(
                   "aten::unsqueeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      // 获取第一个张量类型
      auto& tp = tensor_types.at(0);
      // 获取张量的具体尺寸
      auto sizes = tp->sizes().concrete_sizes().value();
      // 获取张量的步长
      auto strides = tp->strides().concrete_sizes().value();
      // 获取节点的维度参数
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      // 断言维度在有效范围内
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
      // 计算新的步长
      int64_t new_stride = dim >= static_cast<int64_t>(sizes.size())
          ? 1
          : sizes.at(dim) * strides.at(dim);
      // 在指定维度插入新的尺寸和步长
      sizes.insert(sizes.begin() + dim, 1);
      strides.insert(strides.begin() + dim, new_stride);
      // 设置节点输出的张量类型
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      // 返回处理成功
      return true;
    // 如果节点匹配 "aten::view(Tensor self, int[] size) -> Tensor"，其中 size 是常量输入参数
    } else if (node->matches(
                   "aten::view(Tensor self, int[] size) -> Tensor",
                   /*const_inputs=*/attr::size)) {
      // 获取 size 参数的具体值列表
      auto sizes = node->get<c10::List<int64_t>>(attr::size).value();
      // 推断变量初始化为 false
      bool inferred = false;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t inferred_idx;
      // 计算 size 的乘积
      int64_t size_product = 1;
      // 遍历 size 列表
      for (const auto i : c10::irange(sizes.size())) {
        // 如果 size 中有值为 -1，则进行推断处理
        if (sizes.get(i) == -1) {
          // 如果已经推断过，则抛出推断错误异常
          if (inferred)
            throw propagation_error();
          // 设置推断标志为 true，并记录推断的索引位置
          inferred = true;
          inferred_idx = i;
        } else {
          // 否则，计算 size 的乘积
          size_product *= sizes.get(i);
        }
      }

      // 如果进行了推断处理
      if (inferred) {
        // 确保 size_product 不为零
        SHAPE_ASSERT(size_product != 0);
        // 计算张量元素总数
        size_t numel = 1;
        auto concrete_sizes =
            tensor_types.at(0)->sizes().concrete_sizes().value();
        for (int64_t s : concrete_sizes)
          numel *= s;
        // 推断出的大小值
        int64_t inferred_size = numel / size_product;
        // 更新推断位置的 size 值
        sizes[inferred_idx] = inferred_size;
      }
      // 更新节点输出类型，带上新的 size 信息
      node->output()->setType(tensor_types.at(0)->withSizes(sizes.vec()));
      // 返回处理成功标志
      return true;
    } else if (node->matches(
                   // 如果节点匹配 "aten::type_as(Tensor self, Tensor other) -> Tensor"
                   "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      // 如果两个张量的标量类型相同
      if (tensor_types.at(0)->scalarType() ==
          tensor_types.at(1)->scalarType()) {
        // 输出张量类型设为输入 self 张量的类型
        node->output()->setType(node->namedInput(attr::self)->type());
      } else {
        // 否则，由于是复制操作，结果张量将是连续的
        // 输出张量类型设为与第一个张量大小相同的第二个张量类型
        node->output()->setType(tensor_types.at(1)->withSizes(
            tensor_types.at(0)->sizes().concrete_sizes().value()));
      }
      // 返回处理成功标志
      return true;
    } else if (
        // 如果节点匹配 "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor"，其中 size 是常量输入参数
        node->matches(
            "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            /*const_inputs=*/attr::size)) {
      // 获取第一个张量的类型
      auto tp = tensor_types.at(0);
      // 推断张量扩展的几何形状大小和步长
      auto sizesAndStrides = at::inferExpandGeometry_dimvector(
          tp->sizes().concrete_sizes().value(),
          tp->strides().concrete_sizes().value(),
          node->get<c10::List<int64_t>>(attr::size).value().vec());
      // 更新节点输出类型，带上新的大小和步长信息
      node->output()->setType(
          tp->withSizesStrides(sizesAndStrides.sizes, sizesAndStrides.strides));
      // 返回处理成功标志
      return true;
    } else if (
        // 如果节点匹配 "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor"，其中 dim 是常量输入参数
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
            /*const_inputs=*/attr::dim)) {
      // 获取第一个和第二个张量的类型
      auto ten = tensor_types.at(0);
      auto index = tensor_types.at(1);
      // 获取 dim 参数值
      int64_t dim = node->get<int64_t>(attr::dim).value();
      // 断言 index 张量的维度为 1
      SHAPE_ASSERT(*index->sizes().size() == 1);
      // 断言 dim 在合理范围内，并更新第一个张量对应维度的大小
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < ten->sizes().size());
      std::vector<int64_t> sizes = ten->sizes().concrete_sizes().value();
      sizes[dim] = index->sizes()[0].value();
      // 更新节点输出类型，带上新的大小信息
      node->output()->setType(ten->withSizes(sizes));
      // 返回处理成功标志
      return true;
    // 如果节点匹配 "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]"
    // 则执行以下操作
    } else if (node->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      // 获取输入张量的类型信息
      auto input_type = tensor_types.at(0);
      // 获取输入张量的具体尺寸
      auto sizes = input_type->sizes().concrete_sizes().value();
      // 获取输入张量的具体步长
      auto strides = input_type->strides().concrete_sizes().value();
      // 获取操作中指定的维度
      int64_t dim = node->get<int64_t>(attr::dim).value();
      // 获取操作中指定的分块数
      int64_t chunks = node->get<int64_t>(attr::chunks).value();
      // 将指定维度的尺寸除以分块数，以便进行分块操作
      sizes[dim] /= chunks;
      // 更新所有输出张量的类型，修改其尺寸和步长
      for (Value* output : node->outputs()) {
        output->setType(input_type->withSizesStrides(sizes, strides));
      }
      // 如果分块操作不均匀，处理余下的部分
      if (*input_type->sizes()[dim] % chunks != 0) {
        sizes[dim] = *input_type->sizes()[dim] % chunks;
        // 更新最后一个输出张量的类型，以反映剩余的尺寸
        node->outputs().back()->setType(
            input_type->withSizesStrides(sizes, strides));
      }
      // 操作成功，返回 true
      return true;
    } else if (node->kind() == ::c10::onnx::Shape) {
      // 处理 ONNX 中的 Shape 操作节点
      SHAPE_ASSERT(node->inputs().size() == 1 && node->outputs().size() == 1);
      // 获取输入张量的维度数，并转换为 int64_t 类型
      std::vector<int64_t> dim_vec = {
          (int64_t)*tensor_types.at(0)->sizes().size()};
      // 创建一个表示维度的 IntArrayRef
      at::IntArrayRef dims(dim_vec);
      // 设置节点输出张量的类型，使其成为一个连续的 Long 类型张量
      node->output()->setType(
          TensorType::createContiguous(at::kLong, at::kCPU, dims));
      // 操作成功，返回 true
      return true;
    } else if (node->kind() == ::c10::onnx::Reshape) {
      // 对应 ONNX 中的 Reshape 操作节点，设置为未定义形状的类型
      setUnshapedType(node);
      // 操作成功，返回 true
      return true;
    }
    // 对于无法识别的节点类型，设置为未定义形状的类型
    setUnshapedType(node);
    // 操作失败，返回 false
    return false;
  }
} // anonymous namespace
```