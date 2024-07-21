# `.\pytorch\torch\csrc\jit\passes\utils\check_alias_annotation.cpp`

```py
// 包含 Torch 的 JIT 模块中的头文件，用于别名注释的检查
#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>

// 包含 Torch 的 JIT 模块中的头文件，用于常量传播的优化
#include <torch/csrc/jit/passes/constant_propagation.h>

// 包含 Torch 的 JIT 模块中的头文件，用于操作的规范化
#include <torch/csrc/jit/passes/normalize_ops.h>

// 包含 Torch 的 JIT 运行时操作符的头文件
#include <torch/csrc/jit/runtime/operator.h>

// 包含 C10 库中的范围迭代工具
#include <c10/util/irange.h>

// 定义 Torch 命名空间
namespace torch {
// 定义 Torch JIT 命名空间
namespace jit {
// 定义匿名命名空间，用于内部函数和结构体
namespace {

// 深度复制 IValue 类型数据的函数
IValue deepCopy(const IValue& self) {
  // 基本类型可以直接复制
  if (!self.isPtrType()) {
    return self;
  }

  // Tensor 类型需要特殊处理，因为复制会创建别名
  if (self.isTensor()) {
    return IValue(self.toTensor().clone(at::MemoryFormat::Preserve));
  }

  // IValue 列表需要递归深度复制其内容
  if (self.isList()) {
    // NOLINTNEXTLINE(performance-move-const-arg)
    auto source = std::move(self).toList();
    auto newList = c10::impl::GenericList(source.elementType());
    newList.reserve(source.size());
    // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
    for (const IValue& value : source) {
      newList.push_back(deepCopy(value));
    }
    return newList;
  }

  // 普通列表可以进行复制赋值
  if (self.isIntList()) {
    return IValue(self.toIntList().copy());
  } else if (self.isDoubleList()) {
    return IValue(self.toDoubleList().copy());
  } else if (self.isComplexDoubleList()) {
    return IValue(self.toComplexDoubleList().copy());
  } else if (self.isBoolList()) {
    return IValue(self.toBoolList().copy());
  } else if (self.isString()) {
    return IValue(self.toStringRef());
  }

  // 如果未来添加了更多在 ATen 操作中使用的引用类型，需要在这里添加相应处理
  AT_ASSERT(false);
}

// 深度复制 Stack 类型数据的函数
Stack deepCopy(const Stack& stack) {
  Stack ret;
  ret.reserve(stack.size());
  for (const auto& v : stack) {
    ret.push_back(deepCopy(v));
  }
  return ret;
}

// 检查两个 IValue 是否深度相等的函数
bool deepEquals(const IValue& lhs, const IValue& rhs) {
  if (lhs.isTensor() && rhs.isTensor()) {
    return lhs.toTensor().equal(rhs.toTensor());
  }

  if (lhs.isTensorList() && rhs.isTensorList()) {
    const auto a = lhs.toTensorList();
    const auto b = rhs.toTensorList();
    if (a.size() != b.size()) {
      return false;
    }
    for (auto i = decltype(a.size()){0}; i < a.size(); ++i) {
      if (!a[i].equal(b[i])) {
        return false;
      }
    }
    return true;
  }

  // 其他情况直接比较值是否相等
  return lhs == rhs;
}

// 别名和 IValue 结合的结构体，包含别名信息和对应的 IValue
struct AliasAndIValue {
  AliasAndIValue(const at::AliasInfo* aliasInfo, IValue iValue)
      : aliasInfo(aliasInfo), iValue(std::move(iValue)) {}

  const at::AliasInfo* aliasInfo; // 别名信息的指针
  const IValue iValue; // 对应的 IValue
};

// 检查输入 Stack 中的每个 IValue 是否互不别名
void checkInputPreconditions(const Stack& inputs) {
  for (const auto i : c10::irange(inputs.size())) {
    for (const auto j : c10::irange(inputs.size())) {
      if (i == j) {
        continue;
      }
      const auto& lhs = inputs.at(i);
      const auto& rhs = inputs.at(j);
      AT_ASSERT(!lhs.isAliasOf(rhs)); // 断言：输入的两个 IValue 不能相互别名
    }
  }
}

// 检查两个 IValue 是否别名，如果是，则它们必须共享一个别名集合
void checkAliases(
    // 对于每一个输出项，检查是否与任何输入项存在别名关系，确保它们共享同一个别名集合
    for (const auto& output : outputs) {
        // 遍历所有输入项
        for (const auto& input : inputs) {
            // 如果当前输出项与某个输入项存在别名关系
            if (output.iValue.isAliasOf(input.iValue)) {
                // 获取输入项和输出项的别名信息
                const auto* inputSet = input.aliasInfo;
                const auto* outputSet = output.aliasInfo;
                // 断言输入项和输出项的别名信息不为空
                AT_ASSERT(inputSet && outputSet);
                // 标志位，用于检测是否找到共享的前置别名集合
                bool found = false;
                // 遍历输入项的所有前置别名集合
                for (const auto& set : inputSet->beforeSets()) {
                    // 如果输出项的前置别名集合中包含当前遍历到的集合
                    if (outputSet->beforeSets().count(set)) {
                        // 标记为找到共享的前置别名集合，并跳出循环
                        found = true;
                        break;
                    }
                }
                // 断言必须找到共享的前置别名集合
                AT_ASSERT(found);
            }
        }
    }
}

// 如果没有指定写入输入值，那么它必须没有改变
void checkWrites(
    const std::vector<AliasAndIValue>& inputs,
    const std::vector<IValue>& deepCopiedInputs) {
  // 确保输入和深拷贝输入的数量相同
  AT_ASSERT(inputs.size() == deepCopiedInputs.size());
  // 遍历所有输入
  for (const auto i : c10::irange(inputs.size())) {
    const auto& input = inputs[i];
    const auto& deepCopiedInput = deepCopiedInputs[i];
    // 如果输入没有别名信息或者不是写操作
    if (!input.aliasInfo || !input.aliasInfo->isWrite()) {
      // 断言输入的值与深拷贝后的输入值深度相等
      AT_ASSERT(deepEquals(input.iValue, deepCopiedInput));
    }
  }
}

// 查找图中与给定未限定操作名匹配的节点
const Node* findNodeForOp(
    const Graph& g,
    const std::string& unqualifiedOpName) {
  // 构造操作符名字符串
  const auto opName = Symbol::fromQualString("aten::" + unqualifiedOpName);
  // 遍历图中的所有节点
  for (const auto* node : g.nodes()) {
    // 如果节点的类型与操作名相同，返回该节点
    if (node->kind() == opName) {
      return node;
    }
  }

  // 检查别名操作符名称
  const auto aliasOp = torch::jit::getOperatorAliasMap().find(opName);
  // 如果找到别名映射，继续查找别名映射的节点
  if (aliasOp != torch::jit::getOperatorAliasMap().end()) {
    for (const auto* node : g.nodes()) {
      if (node->kind() == aliasOp->second) {
        return node;
      }
    }
  }

  // 理想情况下，图中应该只有一个具有张量输出的 ATen 操作符
  // 以增强 checkAliasAnnotation 的鲁棒性，使用该节点作为最后的解析
  for (const auto* node : g.nodes()) {
    // 如果节点没有操作符，跳过
    if (!node->maybeOperator()) {
      continue;
    }
    // 如果操作符不是 C10 操作，跳过
    if (!node->getOperator().isC10Op()) {
      continue;
    }

    // 检查节点的所有输出
    for (const auto* output : node->outputs()) {
      // 如果输出类型是张量类型，返回该节点
      if (output->type()->kind() == TypeKind::TensorType) {
        return node;
      }
    }
  }

  // 如果以上情况都不满足，断言失败
  AT_ASSERT(false);
}

// 处理需要手动传播常量的几种特殊情况
// TODO(suo): 应该能将这些内容移动到常量传播中
std::optional<IValue> toIValueProp(const Value* v) {
  // 如果节点类型是 prim::ListConstruct
  if (v->node()->kind() == prim::ListConstruct) {
    std::vector<IValue> genericList;
    // 遍历 ListConstruct 的所有输入
    for (auto input : v->node()->inputs()) {
      // 将每个输入转换为 IValue
      if (auto elem = toIValue(input)) {
        genericList.push_back(*elem);
      } else {
        // 如果列表元素中有一个不是常量，返回空
        return c10::nullopt;
      }
    }

    // 根据 ListConstruct 的返回类型特化列表
    auto listType = v->node()->output()->type();
    auto containedType = listType->containedTypes().at(0);
    if (containedType == IntType::get()) {
      return IValue(
          fmap(genericList, [](const IValue& v) { return v.toInt(); }));
    } else if (containedType == FloatType::get()) {
      return IValue(
          fmap(genericList, [](const IValue& v) { return v.toDouble(); }));
    } else if (containedType->isSubtypeOf(*TensorType::get())) {
      return IValue(
          fmap(genericList, [](const IValue& v) { return v.toTensor(); }));
    } else {
      return c10::nullopt;
    }
  }

  // 如果节点类型是 aten::Float
  if (v->node()->kind() == aten::Float) {
    // 如果节点的输入是常量，运行节点
    if (auto maybe_stack = runNodeIfInputsAreConstant(v->node())) {
      return maybe_stack->at(0);
    }
  }
  // 返回空值
  return c10::nullopt;
}
// 定义函数，判断是否应忽略特定的节点，如 batch_norm 和 instance_norm
bool shouldIgnoreNode(const Node* n) {
  // 根据节点的操作种类进行判断
  switch (n->kind()) {
    case aten::batch_norm:
    case aten::instance_norm:
      return true;  // 如果是 batch_norm 或 instance_norm 操作，则应忽略
    default:
      return false; // 其他情况不忽略
  }
}

// 命名空间开始

// 检查操作的别名注解
void checkAliasAnnotation(
    const std::shared_ptr<Graph>& graph,
    std::vector<IValue> pythonInputs,
    const std::string& unqualifiedOpName) {
  // 找到与指定操作名称对应的节点
  const auto node = findNodeForOp(*graph, unqualifiedOpName);
  if (shouldIgnoreNode(node)) {
    return;  // 如果节点应被忽略，则直接返回
  }

  // 准备用作操作输入的堆栈
  Stack stack;
  for (const auto input : node->inputs()) {
    if (input->node() == graph->param_node()) {
      // 这个值在 Python 中作为输入传递
      push(stack, pythonInputs.at(input->offset()));
    } else {
      // 这是一个生成的常量，需要进行评估
      auto inputValue = toIValue(input);
      if (!inputValue) {
        inputValue = toIValueProp(input);
      }

      if (inputValue) {
        push(stack, *inputValue);
      } else {
        AT_ASSERT(input->type()->kind() == TypeKind::OptionalType);
        push(stack, IValue());
      }
    }
  }

  // 前提条件：输入之间不应有别名。因此，如果发现别名，则是操作创建的。
  checkInputPreconditions(stack);

  // 获取节点的模式
  const auto& schema = node->schema();

  // 准备要检查的输入别名和值
  std::vector<AliasAndIValue> inputsToCheck;
  for (const auto i : c10::irange(schema.arguments().size())) {
    inputsToCheck.emplace_back(
        schema.arguments().at(i).alias_info(), stack.at(i));
  }

  // 保存输入的深拷贝，以便检查是否意外写入了输入
  const auto inputsDeepCopy = deepCopy(stack);

  // 运行操作
  node->getOperation()(stack);

  // 获取输出并移动堆栈
  const auto outputs = std::move(stack);

  // 准备要检查的输出别名和值
  std::vector<AliasAndIValue> outputsToCheck;
  for (const auto i : c10::irange(schema.returns().size())) {
    outputsToCheck.emplace_back(
        schema.returns().at(i).alias_info(), outputs.at(i));
  }

  // 检查是否正确注释了任何创建的别名
  checkAliases(inputsToCheck, outputsToCheck);

  // 检查是否有意外写入的情况
  checkWrites(inputsToCheck, inputsDeepCopy);
}

// 命名空间结束
```