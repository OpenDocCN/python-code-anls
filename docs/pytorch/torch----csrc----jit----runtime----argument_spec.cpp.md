# `.\pytorch\torch\csrc\jit\runtime\argument_spec.cpp`

```py
// 引入头文件，包括C++标准库和Torch相关头文件
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/argument_spec.h>

// 引入输出流处理库
#include <iostream>

// Torch命名空间
namespace torch::jit {

// 实现ArgumentSpecCreator类的scan函数，用于生成优化指令序列
void ArgumentSpecCreator::scan(
    const TypePtr& typ,                      // 类型指针，表示当前处理的类型
    size_t depth,                            // 当前递归深度
    const WrittenSlots& written_slots) {     // 记录已写入槽位的集合
  auto finishAggregate = [&](size_t pos) {
    // 完成聚合类型处理的闭包函数
    // 如果没有特殊化的张量或可选项，生成跳过整个聚合类型的指令
    bool any_spec = std::any_of(
        instructions_.begin() + pos, instructions_.end(), [](Inst i) {
          return i == SPECIALIZE_TENSOR || i == SPECIALIZE_OPTIONAL ||
              i == SPECIALIZE_OPTIONAL_TENSOR;
        });
    if (!any_spec) {
      instructions_[pos] = SKIP;
      instructions_.resize(pos + 1);
    } else {
      instructions_.emplace_back(LEAVE);
    }
  };

  // 简单虚拟机扫描指令序列时，限制递归深度，超过则跳过处理
  if (depth >= ARG_SPEC_DEPTH_LIMIT) {
    instructions_.emplace_back(SKIP);
  }

  // 根据类型不同进行处理和特殊化
  if (typ->isSubtypeOf(*TensorType::get())) {
    // 处理张量类型
    num_tensors_++;
    instructions_.emplace_back(SPECIALIZE_TENSOR);
  } else if (typ->isSubtypeOf(*OptionalType::ofTensor())) {
    // 处理可选张量类型
    num_tensors_++;
    num_optionals_++;
    instructions_.emplace_back(SPECIALIZE_OPTIONAL_TENSOR);
  } else if (typ->kind() == TypeKind::OptionalType) {
    // 处理可选类型
    num_optionals_++;
    instructions_.emplace_back(SPECIALIZE_OPTIONAL);
  } else if (auto tup = typ->cast<TupleType>()) {
    // 处理元组类型
    size_t pos = instructions_.size();
    instructions_.emplace_back(ENTER_TUPLE);
    for (const auto& elem : tup->containedTypes()) {
      // 递归处理元组中的每个元素
      scan(elem, depth + 1, written_slots);
    }
    finishAggregate(pos);
  } else if (auto cls = typ->cast<ClassType>()) {
    // 处理类类型
    size_t pos = instructions_.size();
    instructions_.emplace_back(ENTER_OBJECT);
    for (size_t i = 0; i < cls->numAttributes(); ++i) {
      auto key =
          cls->name()->qualifiedName() + cls->getAttributes().at(i).getName();
      // 只有未写入槽位的属性才能安全特化
      if (!written_slots.count(key)) {
        scan(cls->containedTypes().at(i), depth + 1, written_slots);
      } else {
        instructions_.emplace_back(SKIP);
      }
    }
    finishAggregate(pos);
  } else {
    // 其他类型均视为跳过处理
    instructions_.emplace_back(SKIP);
  }
};

// 静态函数，扫描类的节点块，记录已写入槽位的信息
static void scanWrittenSlots(
    Block* block,                            // 节点块指针
    ArgumentSpecCreator::WrittenSlots& written_slots) {  // 记录写入槽位的集合
  for (Node* n : block->nodes()) {
    # 如果当前节点是 SetAttr 类型
    if (n->kind() == prim::SetAttr) {
        # 获取当前节点的第一个输入节点的类型，如果是 ClassType 类型
        if (auto cls = n->inputs().at(0)->type()->cast<ClassType>()) {
            # 将类名与属性名合并后加入到 written_slots 集合中
            written_slots.insert(cls->name()->qualifiedName() + n->s(attr::name));
        }
    }
    # 遍历当前节点的每一个子块
    for (Block* subblock : n->blocks()) {
        # 递归调用 scanWrittenSlots 函数，处理当前子块
        scanWrittenSlots(subblock, written_slots);
    }
    # 如果当前节点具有 Subgraph 属性
    if (n->hasAttribute(attr::Subgraph)) {
        # 获取当前节点的 Subgraph 属性对应的子块，然后递归处理
        scanWrittenSlots(n->g(attr::Subgraph)->block(), written_slots);
    }
}

# ArgumentSpecCreator 类的构造函数，接受一个 Graph 对象的引用作为参数
ArgumentSpecCreator::ArgumentSpecCreator(Graph& graph)
    : num_inputs_(graph.inputs().size()) {
  # 创建一个 WrittenSlots 对象，用于跟踪写入的槽位
  WrittenSlots written_slots;
  # 扫描图中基本块的写入槽位
  scanWrittenSlots(graph.block(), written_slots);
  # 遍历图的输入值，对每个输入值执行扫描操作
  for (Value* input : graph.inputs()) {
    scan(input->type(), 0, written_slots);
  }
}

# 打印 ArgumentSpecCreator 对象的指令列表到标准输出
void ArgumentSpecCreator::dump() const {
  # 遍历指令列表 instructions_
  for (Inst inst : instructions_) {
    # 根据指令类型输出相应的字符串表示
    switch (inst) {
      case LEAVE:
        std::cout << "] ";
        break;
      case ENTER_TUPLE:
        std::cout << "Tuple[";
        break;
      case ENTER_OBJECT:
        std::cout << "Object[";
        break;
      case SKIP:
        std::cout << "Skip ";
        break;
      case SPECIALIZE_TENSOR:
        std::cout << "SpecializeTensor ";
        break;
      case SPECIALIZE_OPTIONAL_TENSOR:
        std::cout << "SpecializeOptionalTensor ";
        break;
      case SPECIALIZE_OPTIONAL:
        std::cout << "SpecializeOptional ";
        break;
    }
  }
  # 输出换行符表示指令列表打印结束
  std::cout << "\n";
}

# 根据给定参数创建 ArgumentSpec 对象
ArgumentSpec ArgumentSpecCreator::create(bool with_grad, const Stack& input)
    const {
  # 使用 num_tensors_ 和 num_optionals_ 创建 ArgumentSpec 对象 spec
  ArgumentSpec spec(num_tensors_, num_optionals_);
  # 定义一个 IValue 指针数组 stack，用于存储 IValue 列表的栈
  # 初始化栈顶指向输入列表的开头
  stack[0] = last(input, num_inputs_).begin();
  # stack_top 表示栈顶的偏移量，初始为 0
  size_t stack_top = 0;
  # 遍历指令列表 instructions_
  for (Inst inst : instructions_) {
    switch (inst) {
      case SPECIALIZE_OPTIONAL_TENSOR: {
        // 处理一个可选的张量，并添加到参数规范中
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        auto& arg = *stack[stack_top]++;
        spec.addOptional(arg);
        if (!arg.isNone()) {
          spec.addTensor(arg, with_grad);
        }
      } break;
      case SPECIALIZE_TENSOR:
        // 处理一个张量，并添加到参数规范中
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        spec.addTensor(*stack[stack_top]++, with_grad);
        break;
      case SPECIALIZE_OPTIONAL:
        // 处理一个非张量的可选项，并添加到参数规范中
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        spec.addOptional(*stack[stack_top]++);
        break;
      case ENTER_TUPLE: {
        // 处理元组
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        const IValue* iv = stack[stack_top]++;
        AT_ASSERT(iv->isTuple(), "Expected Tuple but got ", iv->tagKind());
        auto p = *reinterpret_cast<const at::ivalue::Tuple* const*>(iv);
        auto tup_ptr = &p->elements()[0];
        // 将元组元素的列表推送到堆栈
        stack[++stack_top] = tup_ptr;
      } break;
      case ENTER_OBJECT: {
        // 处理对象
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        const IValue* iv = stack[stack_top]++;
        AT_ASSERT(iv->isObject(), "Expected Object but got ", iv->tagKind());
        auto obj_ptr = &iv->toObjectRef().slots()[0];
        // 将对象元素的列表推送到堆栈
        stack[++stack_top] = obj_ptr;
      } break;
      case SKIP:
        // 消耗并跳过一个元素
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        stack[stack_top]++;
        break;
      case LEAVE:
        // 减少堆栈顶部的索引，退出当前作用域
        --stack_top;
        break;
    }
  }
  // 返回处理后的参数规范
  return spec;
}

// 对于给定图形的每个输入，根据 ArgumentSpec 推断并返回最详细的类型。
void ArgumentSpecCreator::specializeTypes(
    Graph& graph,
    const ArgumentSpec& spec) const {
  // 收集所有输入节点的类型
  auto input_types =
      fmap(graph.inputs(), [](Value* input) { return input->type(); });
  
  // 结果堆栈，用于存储推断的类型信息
  std::vector<std::vector<TypePtr>> result_stack;
  result_stack.emplace_back();  // 初始化结果堆栈

  // 输入堆栈，用于跟踪当前处理的输入类型
  std::vector<const TypePtr*> input_stack = {input_types.data()};

  // 聚合创建器，用于构造复合类型的函数
  std::vector<std::function<TypePtr()>> aggregate_creators;

  // 记录已特化张量和可选项的偏移量
  size_t tensor_arg_spec_offset =
      0; // 目前已见过的特化张量数量
  size_t optional_arg_spec_offset =
      0; // 目前已见过的特化可选项数量

  // 遍历指令列表执行推断类型的过程
  for (Inst inst : instructions_) {
    switch (inst) {
      case SPECIALIZE_OPTIONAL_TENSOR: {
        // 获取当前输入类型，并根据 spec 判断是否存在特化的可选项张量
        auto& input_type = *input_stack.back()++;
        auto is_present = spec.isPresent(optional_arg_spec_offset++);
        if (!is_present) {
          // 如果不存在特化，则直接使用当前输入类型
          result_stack.back().emplace_back(input_type);
          break;
        }
        // 否则，获取特化的张量并将其类型添加到结果堆栈
        auto& arg = spec.tensorAt(tensor_arg_spec_offset++);
        AT_ASSERT(arg.defined());  // 断言特化的张量已定义
        result_stack.back().emplace_back(arg.toType());
      } break;
      case SPECIALIZE_TENSOR: {
        // 跳过当前输入类型，处理特化的张量情况
        input_stack.back()++;
        auto& arg = spec.tensorAt(tensor_arg_spec_offset++);
        if (!arg.defined()) {
          // 如果特化的张量未定义，则创建一个未定义状态的张量类型
          result_stack.back().emplace_back(TensorType::get()->withUndefined());
        } else {
          // 否则，使用特化的张量类型
          result_stack.back().emplace_back(arg.toType());
        }
      } break;
      case SPECIALIZE_OPTIONAL: {
        // 处理特化的可选项情况
        auto is_present = spec.isPresent(optional_arg_spec_offset++);
        auto ot = (*input_stack.back()++)->expect<OptionalType>();
        if (!is_present) {
          // 如果不存在特化，则保留当前可选项类型
          result_stack.back().emplace_back(ot);
        } else {
          // 否则，使用可选项的元素类型
          result_stack.back().emplace_back(ot->getElementType());
        }
      } break;
      case ENTER_TUPLE: {
        // 处理进入元组类型的情况
        auto tup = (*input_stack.back()++)->expect<TupleType>();
        // 将元组的元素类型添加到输入堆栈，并为元组类型创建一个新的结果堆栈
        input_stack.emplace_back(tup->elements().data());
        result_stack.emplace_back();
        // 添加创建元组类型的函数到聚合创建器中
        aggregate_creators.emplace_back(
            [&] { return TupleType::create(result_stack.back()); });
      } break;
      case ENTER_OBJECT: {
        // 处理进入对象类型的情况
        auto cls = (*input_stack.back()++)->expect<ClassType>();
        // 将类包含的类型添加到输入堆栈，并为对象类型创建一个新的结果堆栈
        input_stack.emplace_back(cls->containedTypes().data());
        result_stack.emplace_back();
        // 添加创建对象类型的函数到聚合创建器中
        aggregate_creators.emplace_back(
            [&result_stack, cls] { return cls->refine(result_stack.back()); });
      } break;
      case SKIP:
        // 跳过当前输入类型，将其直接添加到结果堆栈
        result_stack.back().emplace_back(*input_stack.back()++);
        break;
      case LEAVE:
        // 处理离开聚合类型的情况
        TypePtr result = aggregate_creators.back()();
        result_stack.pop_back();
        aggregate_creators.pop_back();
        input_stack.pop_back();
        // 将聚合类型的创建结果添加到当前结果堆栈中
        result_stack.back().emplace_back(std::move(result));
        break;
    }
  }
  // 确保结果栈中只有一个元素，即最终的计算结果
  AT_ASSERT(result_stack.size() == 1);
  // FIXME: 仅在输入中执行此操作，因此我们只捕获图的输入，而不是元组或对象中的可选项。
  // 要使其工作，我们需要详细调查输入的使用情况，以更改访问/解包。
  // 获取计算图的输入
  auto inputs = graph.inputs();
  // 遍历所有输入
  for (const auto i : c10::irange(inputs.size())) {
    // 获取结果栈顶部第i个元素的类型
    auto t = result_stack.back()[i];
    // 检查是否是可选类型
    if (auto ot = t->cast<OptionalType>()) {
      // 如果可选输入未在上面特化，它应该是 None
      // 因此我们在这里断开连接并用常量替换其使用
      // 在图的开始位置插入一个新节点
      WithInsertPoint guard(*graph.nodes().begin());
      // 插入一个空常量节点
      auto c = graph.insertConstant({});
      // 用常量替换输入的所有使用
      inputs[i]->replaceAllUsesWith(c);
    } else {
      // 如果不是可选类型，则将输入的类型设置为t
      inputs[i]->setType(t);
    }
  }
}

} // namespace torch::jit


}


注释：

// 关闭 torch::jit 命名空间
}
```