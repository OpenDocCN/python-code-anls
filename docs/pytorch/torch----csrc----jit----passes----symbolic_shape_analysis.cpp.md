# `.\pytorch\torch\csrc\jit\passes\symbolic_shape_analysis.cpp`

```
/*
XXX: this is still in prototype phase and has much work left to do, including
but not limited to:
- Refactor APIs
- Add decent coverage of common ops
- Add shape analysis pass on Graph that handles Loops
- Allow concurrent reads to the operator map
- Supporting returning partially evaluated shape compute graph
*/

// 声明一个静态变量，用于控制符号形状分析的测试模式
static bool symbolic_shape_analysis_test_mode = false;

namespace torch {
namespace jit {

// 这个结构类似于 c10::SymbolicShape，但不同之处在于：
// - 可能是符号维度
// - 可能是常量整数
// - 也可能既不是符号维度也不是常量整数，例如对于像 view 这样的操作接受负值的情况
// 维护这些区别允许我们优化掉与值 < 0 的比较（符号形状始终 >= 0）
// 例如，对于图中的调用 graph(%y: Tensor(SS(-1), 10, 10), %inp: int):
//   %five: int = prim::Constant[value=5]()
//   %zero: int = prim::Constant[value=0]()
//   %1 : int = aten::size(%y, %zero)
//   %2 : int[] = prim::ListConstruct(%five, %1, %inp)
//   %y.2: Tensor(5, SS(-1), (New Symbolic Shape)) = aten::view(%y, %2)
//
// x.view([5, y.size(0), inp])
// 将具有等于 [5, SS(-1), c10::nullopt] 的输入

struct ShapeArg
    : public std::
          pair<std::optional<c10::ShapeSymbol>, std::optional<int64_t>> {
  using pair::pair;

  // 创建一个未知整数的 ShapeArg 对象
  static ShapeArg unknownInteger() {
    return ShapeArg();
  }

  // 创建一个具有整数值的 ShapeArg 对象
  ShapeArg(int64_t int_value) {
    this->first = c10::nullopt;
    this->second = int_value;
  }

  // 创建一个具有符号形状的 ShapeArg 对象
  ShapeArg(c10::ShapeSymbol ss) {

    this->first = ss;
    this->second = c10::nullopt;
  }
  
  // 结束注释
    // 如果 ShapeSymbol 是静态的
    if (ss.is_static()) {
      // 将第一个成员设置为无值（null）
      this->first = c10::nullopt;
      // 将第二个成员设置为 ShapeSymbol 的值
      this->second = ss.value();
    } else {
      // 如果 ShapeSymbol 不是静态的，则将第一个成员设置为 ShapeSymbol
      this->first = ss;
      // 将第二个成员设置为无值（null）
      this->second = c10::nullopt;
    }
  }

  // 返回第二个成员，这里是一个 optional<int64_t> 类型的值
  std::optional<int64_t> asConstantInt() const {
    return this->second;
  }

  // 返回第一个成员，这里是一个 optional<c10::ShapeSymbol> 类型的值
  std::optional<c10::ShapeSymbol> asShapeSymbol() const {
    return this->first;
  }

 private:
  // 私有构造函数，默认将第一个和第二个成员都设置为无值（null）
  ShapeArg() {
    this->first = c10::nullopt;
    this->second = c10::nullopt;
  }
};

// 重载操作符<<，用于打印ShapeArg对象到输出流
static std::ostream& operator<<(std::ostream& out, const ShapeArg& sa) {
  // 如果ShapeArg对象是常量整数，输出其值
  if (auto val = sa.asConstantInt()) {
    out << *val;
  } 
  // 如果ShapeArg对象是形状符号，输出其符号
  else if (auto ss = sa.asShapeSymbol()) {
    out << *ss;
  } 
  // 否则输出"UNK"表示未知
  else {
    out << "UNK";
  }
  return out;
}

// 表示形状参数的结构体ShapeArguments
struct ShapeArguments {
  // SymbolicShape的超集，额外支持未知、非符号值

 public:
  // 构造函数，接受SymbolicShape对象作为参数
  ShapeArguments(const c10::SymbolicShape& ss) {
    // 检查SymbolicShape对象是否具有维度信息
    has_dim_ = ss.rank().has_value();
    if (has_dim_) {
      // 遍历每个维度，将其加入maybe_shape_symbols_
      for (size_t i = 0; i < *ss.rank(); ++i) {
        maybe_shape_symbols_.emplace_back(ss.at(i));
      }
    }
  }

  // 构造函数，接受形状参数对象的向量作为参数
  ShapeArguments(std::vector<ShapeArg> ss)
      : has_dim_(true), maybe_shape_symbols_(std::move(ss)) {}

  // 返回是否具有维度信息
  bool has_dim() const {
    return has_dim_;
  }

  // 返回形状参数的长度
  int64_t len() const {
    // 如果没有维度信息，抛出内部断言错误
    TORCH_INTERNAL_ASSERT(has_dim_, "ShapeArguments has no known dim")
    return (int64_t)maybe_shape_symbols_.size();
  }

  // 返回指定索引处的形状参数
  const ShapeArg at(size_t i) const {
    // 如果没有维度信息，抛出内部断言错误
    TORCH_INTERNAL_ASSERT(has_dim_, "ShapeArguments has no known dim")
    return maybe_shape_symbols_.at(i);
  }

 private:
  bool has_dim_;                // 是否具有维度信息的标志
  std::vector<ShapeArg> maybe_shape_symbols_;  // 形状参数的向量
};

// 重载操作符<<，用于打印ShapeArguments对象到输出流
static std::ostream& operator<<(std::ostream& os, const ShapeArguments& sa) {
  // 如果没有维度信息，输出"(UNKNOWN DIM)"
  if (!sa.has_dim()) {
    os << "(UNKNOWN DIM)";
    return os;
  }

  // 输出形状参数
  os << "(";
  for (const auto i : c10::irange(sa.len())) {
    os << sa.at(i);
  }
  os << ")";

  return os;
}

// 设置符号形状分析测试模式的函数
bool setSymbolicShapeAnalysisTestMode(bool value) {
  bool old_value = symbolic_shape_analysis_test_mode;  // 保存旧值
  symbolic_shape_analysis_test_mode = value;  // 设置新值
  return old_value;  // 返回旧值
}

// 返回符号形状分析测试模式是否启用的函数
bool symbolicShapeAnalysisTestModeEnabled() {
  return symbolic_shape_analysis_test_mode;  // 返回当前模式
}

// SSArgument的类型别名，可以是ShapeArguments或IValue
using SSArgument = std::variant<ShapeArguments, IValue>;

// 重载操作符<<，用于打印SSArgument对象到输出流
static std::ostream& operator<<(std::ostream& out, const SSArgument& sa) {
  // 如果是IValue类型，直接输出其值
  if (const IValue* iv = std::get_if<IValue>(&sa)) {
    out << *iv;
  } 
  // 否则输出ShapeArguments对象
  else {
    out << std::get<ShapeArguments>(sa);
  }
  return out;
}

// 匿名命名空间，定义一些辅助函数

// 判断类型是否为整数列表的函数
bool isListOfInts(const TypePtr& type) {
  return type->cast<ListType>() &&
      type->cast<ListType>()->getElementType()->cast<IntType>();
}

// 判断类型是否为整数列表的列表的函数
bool isListOfListOfInts(const TypePtr& type) {
  // 允许List[Optional[List[Int]]]
  if (!type->cast<ListType>()) {
    return false;
  }
  TypePtr element_type = type->cast<ListType>()->getElementType();
  if (element_type->cast<OptionalType>()) {
    element_type = element_type->cast<OptionalType>()->getElementType();
  }
  return isListOfInts(element_type);
}

// 判断类型是否为张量列表的函数
bool isListOfTensors(const TypePtr& type) {
  return type->cast<ListType>() &&
      type->cast<ListType>()->getElementType()->cast<TensorType>();
}

// 规范化索引值的函数
std::optional<size_t> normIndex(int64_t index, size_t len) {
  if (index < 0) {
    index = index + len;
  }
  // 如果索引在有效范围内，返回索引值，否则返回空
  if (index >= 0 && index < static_cast<int64_t>(len)) {
    return index;
  } else {
    return c10::nullopt;
  }
}
// 执行图形清理传递操作，以优化图形表示的计算图
bool shapeGraphCleanupPasses(std::shared_ptr<Graph> graph) {
  // TODO: lower simple tuples ?
  // 移除列表变异操作，若有变化则更新标志位
  bool made_change = RemoveListMutation(graph);
  // 展开常量循环，若有变化则更新标志位
  made_change |= UnrollConstantLoops(graph);
  // 常量传播优化，若有变化则更新标志位
  made_change |= ConstantPropagation(graph);
  // 针对非张量进行目视优化，若有变化则更新标志位
  made_change |= PeepholeOptimizeNonTensor(graph);
  // 针对列表习语进行目视优化，若有变化则更新标志位
  made_change |= PeepholeOptimizeListIdioms(graph, /*refine_list_len*/ true);
  // 优化整数值，若有变化则更新标志位
  made_change |= RefineIntegerValues(graph);
  // 再次进行常量传播优化，若有变化则更新标志位
  made_change |= ConstantPropagation(graph);
  // 常量池化操作，无返回值
  ConstantPooling(graph);
  // 消除公共子表达式，若有变化则更新标志位
  made_change |= EliminateCommonSubexpression(graph);
  // 消除死代码
  EliminateDeadCode(graph);
  // 返回是否进行了修改
  return made_change;
}

// 替换值为 IValue
void replaceWithIValue(Value* v, IValue val) {
  // 在节点块的起始位置插入替换操作
  WithInsertPoint guard(*v->node()->owningBlock()->nodes().begin());
  // 替换值为常量
  v->replaceAllUsesWith(v->owningGraph()->insertConstant(val));
}

// 提取列表的符号形状
c10::SymbolicShape extractListShape(
    Value* list,
    std::unordered_map<Value*, int64_t>& symbolic_shape_values,
    const AliasDb& db) {
  // 如果列表节点是常量
  if (list->node()->kind() == prim::Constant) {
    // 转换为整数向量，并返回符号形状
    auto int_list = toIValue(list)->toIntVector();
    return c10::SymbolicShape(int_list);
  }
  // 需要列表构造或不写入的常量输出以分析输出形状
  if (list->node()->kind() != prim::ListConstruct || db.hasWriters(list)) {
    // 输出调试信息，无法提取形状
    GRAPH_DEBUG("Could not extract shape");
    return c10::SymbolicShape();
  }
  // 获取列表构造节点
  Node* list_construct = list->node();
  // 输出形状的可选整数向量
  std::vector<std::optional<int64_t>> output_shape;
  // 遍历列表构造节点的输入
  for (Value* input : list_construct->inputs()) {
    // 如果输入在符号形状值中存在
    if (symbolic_shape_values.count(input)) {
      // 添加符号形状值
      output_shape.emplace_back(symbolic_shape_values[input]);
    } else {
      // 将常量作为整数添加到输出形状
      output_shape.push_back(constant_as<int64_t>(input));
    }
  }
  // 返回符号形状
  return c10::SymbolicShape(output_shape);
}
  struct SymbolicShapeOpAnalyzer {
    // 用于存储符号形状操作分析器的计算图
    std::shared_ptr<Graph> shape_compute_graph_;
    // 指向函数模式的指针，描述了分析器操作的函数签名
    const FunctionSchema* schema_;
    // 存储输入参数列表的向量
    std::vector<SSArgument> inputs_;

    // 对于具有 JIT 图的情况，
    // 如果类型已知，用可选类型替换其组件类型。
    // 对于已知的 IValue，不需要执行此操作。
    void refineInputUnionTypes(const Node* parent_graph_node) {
      // 遍历形状计算图的输入
      for (size_t op_in_index = 0;
           op_in_index < shape_compute_graph_->inputs().size();
           op_in_index++) {
        // 获取父图节点的输入类型
        auto type = parent_graph_node->input(op_in_index)->type();
        // 如果当前输入是 OptionalType 类型
        if (auto opt_type = shape_compute_graph_->inputs()
                                .at(op_in_index)
                                ->type()
                                ->cast<OptionalType>()) {
          // None 类型将在后续常量替换中处理
          if (!type->cast<OptionalType>() &&
              !NoneType::get()->isSubtypeOf(*type)) {
            // 设置输入类型为 OptionalType 的元素类型
            shape_compute_graph_->inputs()
                .at(op_in_index)
                ->setType(opt_type->getElementType());
          }
        }
        // 如果当前输入是 NumberType 类型
        else if (shape_compute_graph_->inputs()
                       .at(op_in_index)
                       ->type()
                       ->cast<NumberType>()) {
          // 设置输入类型为父图节点的输入类型
          shape_compute_graph_->inputs().at(op_in_index)->setType(type);
        }
      }
    }

    // 在形状传播步骤中处理非常量值
    void substituteConstantInputs() {
      // 如果形状计算图的输入为空，则直接返回
      if (shape_compute_graph_->inputs().empty()) {
        return;
      }

      // 是否已经看到过张量列表
      bool seen_tensor_list = false;

      // 操作输入索引
      size_t op_in_index = 0;
    while (op_in_index < shape_compute_graph_->inputs().size()) {
      // 获取当前操作在形状计算图中的输入变量
      Value* graph_in_var = shape_compute_graph_->inputs().at(op_in_index);
      // 如果输入变量不是 List[List[int]] 类型，则跳过处理
      if (!isListOfListOfInts(graph_in_var->type())) {
        op_in_index++;
        continue;
      }

      // 修改包含 _node 的图形，使其不使用张量构造

      // 当我们部分评估类似 cat(tensor[]) 的张量列表时，会遇到一些问题：
      // - 优化列表长度调用的优化: len(tensors)
      // - 解析列表访问到张量的符号大小对应的列表元素
      // 我们可以通过替换部分评估的 cat([x, y]) 来解决这两个问题：
      // def cat(tensors: List[List[int]], dim: int)
      //    主体
      // 替换为
      // def cat(x, y, dim: int)
      //     tensors = [x, y]
      //     主体
      TORCH_INTERNAL_ASSERT(
          !seen_tensor_list,
          "SSA doesn't handle case with multiple tensor lists")
      seen_tensor_list = true;

      // 计算需要的长度，输入的总长度减去（模式参数数量 - 1）
      uint64_t li_length = inputs_.size() - (schema_->arguments().size() - 1);
      std::vector<Value*> li_inputs;

      // 获取元素类型，假设 graph_in_var 是 ListType 类型
      TypePtr element_type =
          graph_in_var->type()->cast<ListType>()->getElementType();
      // 在形状计算图中插入新的输入节点
      for (size_t j = op_in_index; j < op_in_index + li_length; ++j) {
        auto new_inp = shape_compute_graph_->insertInput(op_in_index + j);
        new_inp->setType(element_type);
        li_inputs.push_back(new_inp);
      }
      // 在形状计算图中插入新的列表节点
      WithInsertPoint guard(*shape_compute_graph_->block()->nodes().begin());
      auto new_li = shape_compute_graph_->insertNode(
          shape_compute_graph_->createList(element_type, li_inputs));
      // 用新列表节点替换原始输入变量的使用
      graph_in_var->replaceAllUsesWith(new_li->output());
      // 删除已替换的输入
      shape_compute_graph_->eraseInput(op_in_index + li_length);
    }

    // 断言：形状计算图的输入数量不应超过 inputs_ 的数量
    TORCH_INTERNAL_ASSERT(
        shape_compute_graph_->inputs().size() <= inputs_.size(),
        "Shape Compute Graph expected to have less inputs than actual inputs");

    // 遍历形状计算图的输入，并根据情况替换为常量值或设置类型
    for (size_t op_in_index = 0;
         op_in_index < shape_compute_graph_->inputs().size();
         op_in_index++) {
      // 获取输入参数
      SSArgument& argument = inputs_[op_in_index];
      // 获取形状计算图的输入变量
      Value* graph_in_var = shape_compute_graph_->inputs().at(op_in_index);

      // 如果当前参数是常量，则替换输入变量为常量值
      if (IValue* cur_val = std::get_if<IValue>(&argument)) {
        GRAPH_DEBUG("Substituting constant input ", *cur_val);
        replaceWithIValue(graph_in_var, *cur_val);
      } else {
        // 如果当前参数是 ShapeArguments 类型，并且包含维度信息，则设置输入变量类型为 ListType::ofInts()
        auto cur_arg = std::get<ShapeArguments>(argument);
        if (cur_arg.has_dim()) {
          graph_in_var->setType(ListType::ofInts());
        }
      }
    }
  }

  // 替换符号属性的函数
  void substituteSymbolicProperties(
      std::unordered_map<Value*, int64_t>* symbolic_shape_values) {
    // clang-format off
    // 这里我们迭代地将节点输入张量的属性替换到形状计算图中。
    // 如果张量具有固定长度或固定
    // 声明一个无序映射表，用于存储符号形状值到其对应的值集合（Value*）的映射关系
    std::unordered_map<int64_t, std::vector<Value*>> symbolic_shape_map;

    // 使用内部断言确保输入的数量不少于形状计算图的输入数量，否则抛出异常消息 "Missing Arg for Shape Graph"
    TORCH_INTERNAL_ASSERT(
        inputs_.size() >= shape_compute_graph_->inputs().size(),
        "Missing Arg for Shape Graph");

    // 结束了函数定义的大括号，该注释看起来是不完整的或者是错误的注释
    }
  }

  // 合并符号形状集合的函数定义，参数 symbolic_set 表示一组相等的 Value* 对象集合
  void mergeSymbolicShapeSets(const std::vector<Value*>& symbolic_set) {
    // 在此处优化图形，将集合中的值替换为其他主导值
    // 在以下示例中，a、b 和 c 都属于同一个符号集合：
    // if cond:
    //    a = li[0]
    //    b = li[1]
    //    return [a, b]
    // else:
    //    c = li[0]
    //    return [c, c]
    // 我们可以用 `a` 替换 `b`，因为 `b` 被 `a` 主导，
    // 但是我们不能用另一个主导值替换 `c`

    // 尽管有更高效的计算方法，但每个符号集合中的值数量通常很少，这种方法运行起来成本低廉
    for (const auto i : c10::irange(symbolic_set.size())) {
      Value* v = symbolic_set[i];
      Value* dominating_value = v;
      for (const auto& sym_set : symbolic_set) {
        // 如果当前的主导值被某个符号集合的值所主导，则更新为该符号集合的值
        if (dominating_value->node()->isDominatedBy(sym_set->node())) {
          dominating_value = sym_set;
        }
      }
      // 如果找到了比原始值 `v` 更优的主导值 `dominating_value`，则用 `dominating_value` 替换 `v` 的所有用法
      if (dominating_value != v) {
        v->replaceAllUsesWith(dominating_value);
      }
    }
  }

  // 在图中传播形状的函数定义，返回推断的符号形状的向量
  std::vector<c10::SymbolicShape> propagateShapesInGraph() {
    // 标志是否进行了修改
    bool made_change = true;
    // 最大尝试次数
    constexpr size_t MAX_ATTEMPTS = 8;
    for (unsigned attempt_num = 0; made_change && attempt_num < MAX_ATTEMPTS;
         attempt_num++) {
      // 对图形进行形状计算前的替换操作，仅在最终形状提取时使用
      GRAPH_DUMP("Before substitution: ", shape_compute_graph_);
      // 替换符号形状属性为具体数值
      substituteSymbolicProperties(/*symbolic_shape_values*/ nullptr);
      // 输出替换后的图形状态
      GRAPH_DUMP("Before Opt: ", shape_compute_graph_);
      // 执行图形清理操作，返回是否有改变
      made_change = shapeGraphCleanupPasses(shape_compute_graph_);
    }
    // 符号形状属性的映射表
    std::unordered_map<Value*, int64_t> symbolic_shape_values;
    // 替换符号形状属性为具体数值
    substituteSymbolicProperties(&symbolic_shape_values);
    // 输出部分评估完成的图形状态
    GRAPH_DUMP("Done with partial evaluation", shape_compute_graph_);

    // 提取输出形状，基于符号形状属性映射表
    return extractOutputShape(symbolic_shape_values);
  }

  std::vector<c10::SymbolicShape> extractOutputShape(
      std::unordered_map<Value*, int64_t>& symbolic_shape_values) {
    // 断言输出的形状图节点数量与返回值数量一致
    TORCH_INTERNAL_ASSERT(
        shape_compute_graph_->outputs().size() == schema_->returns().size());
    // 结果向量
    auto res = std::vector<c10::SymbolicShape>();
    // 创建别名分析数据库
    AliasDb db(shape_compute_graph_);
    // 遍历输出节点
    for (size_t i = 0; i < shape_compute_graph_->outputs().size(); ++i) {
      auto output = shape_compute_graph_->outputs().at(i);
      auto type = output->type();
      // 断言输出节点是整数列表类型
      TORCH_INTERNAL_ASSERT(isListOfInts(type));
      // 提取列表形状，基于符号形状属性映射表和别名分析数据库
      c10::SymbolicShape ss =
          extractListShape(output, symbolic_shape_values, db);
      // 调试输出提取的输出形状
      GRAPH_DEBUG("Extracted Output: ", ss);
      // 将提取的形状添加到结果向量中
      res.push_back(ss);
    }
    // 返回提取的所有输出形状
    return res;
  }

 public:
  SymbolicShapeOpAnalyzer(const FunctionSchema* schema) : schema_(schema) {
    // 初始化形状计算图为 nullptr，如果没有提供函数模式，直接返回
    shape_compute_graph_ = nullptr;
    if (!schema_) {
      return;
    }
    // 获取给定函数模式的形状计算图
    auto maybe_graph = shapeComputeGraphForSchema(*schema_);
    // 如果未能获取形状计算图，直接返回
    if (!maybe_graph) {
      return;
    }
    // 复制获取的形状计算图作为当前形状计算图
    shape_compute_graph_ = (*maybe_graph)->copy();
  }

  SymbolicShapeOpAnalyzer(
      const FunctionSchema* schema,
      std::shared_ptr<Graph> graph)
      : schema_(schema) {
    // 复制给定的图作为当前形状计算图
    shape_compute_graph_ = graph->copy();
  }

  std::optional<std::vector<c10::SymbolicShape>> run(
      std::vector<SSArgument>& inputs) {
    // 如果形状计算图为空，返回空的可选向量
    if (!shape_compute_graph_) {
      return c10::nullopt;
    }
    // 保存输入参数
    inputs_ = inputs;
    // 替换常量输入
    substituteConstantInputs();
    // 调试输出输入参数
    GRAPH_DEBUG(inputs_)
    // 在图中传播形状信息，返回结果
    return propagateShapesInGraph();
  }

  std::shared_ptr<Graph> getShapeComputeGraph() {
    // 返回当前的形状计算图
    return shape_compute_graph_;
  }
};

SSArgument tensorShapeArg(Value* tensor_v) {
  auto tt = tensor_v->type()->expect<TensorType>();
  c10::SymbolicShape symbolic_shapes = tt->symbolic_sizes();

  // 如果符号形状完整且不处于符号形状分析测试模式，则返回具体形状大小
  if (symbolic_shapes.isComplete() && !symbolic_shape_analysis_test_mode) {
    return IValue(tt->sizes().concrete_sizes());
  }
  // 如果可以转换为 IValue，则返回其大小
  if (toIValue(tensor_v)) {
    auto size = constant_as<at::Tensor>(tensor_v)->sizes();
    if (!symbolic_shape_analysis_test_mode) {
      return IValue(size);
    } else {
      return c10::SymbolicShape(size);
    }
  }
  // 否则返回符号形状
  return symbolic_shapes;
}

std::vector<SSArgument> getNodeInputShapes(Node* n, const AliasDb& db) {
  // TODO: 修复整数列表的实现，并提取形状变化，否则此处已完成
  // 注意：形状计算图可能比其节点对应的输入少，以允许共享单个一元定义
  // 因此迭代形状输入数量
  // 我们使张量输入列表变为可变参数，这导致节点索引和其对应的图索引之间存在偏移

  std::vector<SSArgument> input_shapes = std::vector<SSArgument>();

  for (size_t node_index = 0; node_index < n->inputs().size(); ++node_index) {
    auto type = n->input(node_index)->type();

    if (type->castRaw<TensorType>()) {
      // 如果是张量类型，则获取其形状信息并加入输入形状列表
      input_shapes.push_back(tensorShapeArg(n->input(node_index)));
      continue;
    }
    if (isListOfTensors(type)) {
      // 等待更多用例以决定最佳的泛化方法
      if (n->input(node_index)->node()->kind() == prim::Constant) {
        // 如果是常量节点，将张量向量的大小作为列表加入输入形状列表
        auto ival = toIValue(n->input(node_index));
        for (const auto& ten : ival->toTensorVector()) {
          input_shapes.emplace_back(c10::List<int64_t>(ten.sizes()));
        }
      } else if (
          n->input(node_index)->node()->kind() == prim::ListConstruct &&
          !db.hasWriters(n->input(node_index))) {
        // 如果是列表构造节点且没有写操作，则将列表元素的形状加入输入形状列表
        auto li_construct_node = n->input(node_index)->node();
        for (size_t j = 0; j < li_construct_node->inputs().size(); ++j) {
          input_shapes.push_back(tensorShapeArg(li_construct_node->input(j)));
        }
      } else {
        // 否则抛出内部断言错误，不应该到达这里
        TORCH_INTERNAL_ASSERT(false, "Unhandled List, we shouldn't get here");
      }
      continue;
    }
    // 如果可以转换为 IValue，则将其作为输入形状的一部分加入列表
    if (auto ival = toIValue(n->input(node_index))) {
      input_shapes.emplace_back(*ival);
      continue;
    }
    // 检查节点类型是否为 ListType，并且列表元素类型是否为 IntType
    if (type->cast<ListType>() &&
        type->cast<ListType>()->getElementType()->cast<IntType>()) {
      // 获取输入节点的源节点
      auto input_src_node = n->input(node_index)->node();
      // 检查输入节点是否为 prim::ListConstruct 类型，并且没有写入操作
      if (input_src_node->kind() == prim::ListConstruct &&
          !db.hasWriters(n->input(node_index))) {
        // 对于常见的图形模式，如 x.view(y.size()) 或 x.view(1, 10, y.size(0), y.size(1))
        // 我们希望从 y 传播符号维度和具体大小到 z。
        // 为此，我们尝试将符号维度或具体大小与整数列表输入关联起来，这些输入来自常量或 y.size() 或 y.size(0) 的构造函数
        auto list_construct = n->input(node_index)->node();
        std::vector<ShapeArg> shape;
        // 遍历列表构造节点的输入
        for (Value* v : list_construct->inputs()) {
          // 如果当前值为常量整数
          if (auto constant = constant_as<int64_t>(v)) {
            shape.emplace_back(*constant);
          } else if (v->node()->kind() == aten::size) {
            // 如果当前值为 aten::size 函数调用
            auto const_index = constant_as<int64_t>(v->node()->input(1));
            auto tt = v->node()->input(0)->type()->expect<TensorType>();
            auto ss = tt->symbolic_sizes();
            // 如果尺寸符号没有秩或常量索引不可用，表示未知符号维度
            if (!ss.rank() || !const_index) {
              shape.emplace_back(at::ShapeSymbol::newSymbol());
              continue;
            }
            // 规范化索引并获取符号维度
            auto norm_index = normIndex(*const_index, *ss.rank());
            if (!norm_index) {
              shape.emplace_back(at::ShapeSymbol::newSymbol());
              continue;
            }
            shape.emplace_back(ss[*norm_index]);
          } else {
            // 其他情况视为未知整数
            shape.emplace_back(ShapeArg::unknownInteger());
          }
        }
        // 将解析的形状参数作为 ShapeArguments 存入 input_shapes
        input_shapes.emplace_back(ShapeArguments(shape));
        continue;
      }
      // 如果输入节点为 aten::size 函数调用，并且没有写入操作
      if (input_src_node->kind() == aten::size &&
          !db.hasWriters(n->input(node_index))) {
        // 获取 size 函数的输入 tensor
        auto ten_inp = input_src_node->input();
        auto ss = ten_inp->type()->expect<TensorType>()->symbolic_sizes();
        // 将 tensor 的符号维度作为 ShapeArguments 存入 input_shapes
        input_shapes.emplace_back(ss);
        continue;
      }
    }
    // 如果以上条件都不满足，则标记为无法处理的输入，并将空的符号形状作为默认值存入 input_shapes
    GRAPH_DEBUG(
        "Unhandled input: ",
        n->kind().toDisplayString(),
        " arg num: ",
        node_index);
    input_shapes.emplace_back(c10::SymbolicShape());
  }
  // 断言输入形状的数量不少于节点的输入数量
  TORCH_INTERNAL_ASSERT(
      input_shapes.size() >= n->inputs().size(),
      "input_shapes size: ",
      input_shapes.size(),
      " n inputs size: ",
      n->inputs().size());
  // 返回所有解析的输入形状
  return input_shapes;
}

void applyOutputShapeToGraph(
    Node* node,
    const std::vector<c10::SymbolicShape>& output_shapes) {
  // 断言：节点的输出数量与输出形状的数量必须相等
  TORCH_INTERNAL_ASSERT(
      node->outputs().size() == output_shapes.size(),
      "Output shape size mismatch");
  // 遍历输出形状列表，将每个节点输出的类型设置为带有符号形状的张量类型
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    auto& ss = output_shapes.at(i);
    node->output(i)->setType(
        node->output(i)->type()->expect<TensorType>()->withSymbolicShapes(ss));
  }
}

std::shared_ptr<Graph> PropagateShapesWithShapeFunction(
    Node* n,
    const AliasDb& db) {
  // 获取节点的函数模式
  const FunctionSchema* func_schema = n->maybeSchema();
  if (!func_schema) {
    return nullptr;
  }
  // 使用函数模式创建符号形状操作分析器
  auto op_analyzer = SymbolicShapeOpAnalyzer(func_schema);
  // 如果无法获取形状计算图，返回空指针
  if (!op_analyzer.getShapeComputeGraph()) {
    return nullptr;
  }
  // 获取节点的输入形状
  auto input_shapes = getNodeInputShapes(n, db);
  // 优化输入联合类型
  op_analyzer.refineInputUnionTypes(n);

  // 运行符号形状操作分析器，并获取输出形状
  if (auto output_shapes = op_analyzer.run(input_shapes)) {
    // 应用输出形状到节点的计算图中
    applyOutputShapeToGraph(n, *output_shapes);
  }

  // 返回形状计算图
  return op_analyzer.getShapeComputeGraph();
}

c10::SymbolicShape combine_bounds(
    c10::SymbolicShape& lower_bound,
    c10::SymbolicShape& upper_bound) {
  // TODO: At some point we might want to add support for dynamic dims
  // 断言：下界和上界的秩必须相等
  TORCH_INTERNAL_ASSERT(lower_bound.rank() == upper_bound.rank());
  // 如果下界的秩为 null，返回空的符号形状
  if (lower_bound.rank() == c10::nullopt) {
    return c10::SymbolicShape();
  }
  // 合并边界的符号形状
  std::vector<c10::ShapeSymbol> merged_shapes;
  for (const auto i : c10::irange(*lower_bound.rank())) {
    // TODO: Merge equivalent expressions (not needed for current use case)
    // 如果下界和上界相等，添加到合并形状中；否则添加新的符号形状
    if (lower_bound[i] == upper_bound[i]) {
      merged_shapes.push_back(lower_bound[i]);
    } else {
      merged_shapes.push_back(c10::ShapeSymbol::newSymbol());
    }
  }
  // 返回合并后的符号形状
  return c10::SymbolicShape(std::move(merged_shapes));
}

struct SymbolicShapeGraphAnalyzer {
  SymbolicShapeGraphAnalyzer(
      std::shared_ptr<Graph>& graph,
      Node* beg,
      Node* end)
      : graph_(graph), beg_(beg), end_(end) {
    // 断言：起始节点和结束节点必须属于同一代码块，并且结束节点必须在起始节点之后
    TORCH_INTERNAL_ASSERT(
        beg_->owningBlock() == end_->owningBlock() && end_->isAfter(beg_));
  }

  // 运行符号形状图分析
  std::optional<ShapeComputeGraphMapping> run() {
    // 使用图形创建别名数据库
    AliasDb db(graph_);
    // 传播形状并收集部分评估的形状计算图
    std::unordered_map<Node*, std::shared_ptr<Graph>> partial_evaluated_graphs =
        propagateShapesAndGatherPartialEvalShapeGraphs(db);

    // 创建新的计算图，用于计算所有未知静态形状的符号形状
    auto stitched_shape_compute_graph = std::make_shared<Graph>();
    // 我们希望建立一个计算图，它计算所有在 [beg, end) 区间内的符号形状，必须在起始节点之前执行。
    // TODO: 不要求张量的维度预先设置？
    // 迭代器循环，遍历从 beg_ 到 end_ 的所有元素
    for (auto it = beg_->iterator(); it != end_->iterator(); it++) {
      // 获取当前元素
      auto curr = *it;
      // 如果当前元素的类型为常量，继续下一次迭代
      if (curr->kind() == prim::Constant) {
        continue;
      }
      // 如果当前元素的类型为 ListConstruct
      // 检查其输出是否全部由 aten::cat 操作使用
      if (curr->kind() == prim::ListConstruct) {
        // 获取当前元素的使用列表
        auto uses = curr->output()->uses();
        // 如果不是所有使用都是 aten::cat 操作，则返回空值
        if (!std::all_of(uses.begin(), uses.end(), [](const Use& use) {
              return use.user->kind() == aten::cat;
            })) {
          // 输出调试信息，指出非 cat 操作使用的 ListConstruct
          GRAPH_DEBUG("Non cat list use ", getHeader(curr));
          return c10::nullopt;
        }
        continue;
      }

      // 如果当前元素不在部分评估图中
      if (!partial_evaluated_graphs.count(curr)) {
        // 输出调试信息，指出缺少对应评估图的情况
        GRAPH_DEBUG("No graph ", getHeader(curr));
        return c10::nullopt;
      }

      // 获取当前元素的所有输出
      auto outputs = curr->outputs();
      // 对每个输出进行检查
      for (Value* v : outputs) {
        // 尝试将输出类型转换为 TensorType
        auto tt = v->type()->cast<TensorType>();
        // 如果类型转换失败，则输出调试信息，指出非张量节点
        if (!tt) {
          GRAPH_DEBUG("Non tensor node", getHeader(curr));
          return c10::nullopt;
        }
        // 获取张量类型的符号大小
        auto symbolic_sizes = tt->symbolic_sizes();
        // 如果张量的符号大小的秩不存在
        if (!symbolic_sizes.rank()) {
          // 输出调试信息，指出输出缺乏秩的情况
          GRAPH_DEBUG("No rank on output ", getHeader(curr));
          return c10::nullopt;
        }
      }
      // 获取当前元素的部分评估图
      auto partial_eval_graph = partial_evaluated_graphs[curr];
      // 将部分评估图的形状与整体形状计算图进行合并
      joinPartialEvaluatedShapeGraphToLargeShapeGraph(
          curr, partial_eval_graph, stitched_shape_compute_graph);
    }

    // 定义最大迭代次数和迭代标志
    size_t MAX_ITER = 8;
    bool made_change = true;
    size_t i = 0;
    // 当迭代次数小于最大迭代次数且仍有改变时，执行清理形状图的操作
    while (i < MAX_ITER && made_change) {
      i++;
      made_change = shapeGraphCleanupPasses(stitched_shape_compute_graph);
    }

    // 对于任何重复的输出，其符号形状必须相等
    // 获取首次生成的符号形状，并获取等效的符号形状
    std::unordered_map<int64_t, int64_t> discovered_sym_shape_equalities;
    // 输出到符号形状维度的映射
    std::unordered_map<Value*, int64_t> graph_output_to_symbolic_shape_dim;
    // 要删除的索引列表
    std::vector<size_t> erase_indices;

    // 遍历形状计算图的所有输出
    for (size_t i = 0; i < stitched_shape_compute_graph->outputs().size();
         ++i) {
      // 获取当前输出
      Value* output = stitched_shape_compute_graph->outputs().at(i);
      // 如果该输出已包含，当前符号形状必须与现有索引处的符号形状相等
      if (graph_output_to_symbolic_shape_dim.count(output)) {
        // 获取当前符号形状和现有符号形状
        auto curr_sym_shape = output_index_to_symbolic_shape_[i];
        auto existing_sym_shape = graph_output_to_symbolic_shape_dim[output];
        // 记录已发现的符号形状相等关系
        discovered_sym_shape_equalities[curr_sym_shape] = existing_sym_shape;
        // 将当前索引添加到要删除的索引列表中
        erase_indices.push_back(i);
      } else {
        // 将当前输出与其符号形状维度映射起来
        graph_output_to_symbolic_shape_dim[output] =
            output_index_to_symbolic_shape_[i];
      }
    }
    // 反向遍历要删除的索引列表，从形状计算图中擦除输出
    for (int64_t i = erase_indices.size() - 1; i >= 0; i--) {
      stitched_shape_compute_graph->eraseOutput(erase_indices[i]);
    }
    // 遍历 stitched_shape_compute_graph 的所有输入
    for (size_t i = 0; i < stitched_shape_compute_graph->inputs().size();) {
      // 检查当前输入是否没有被使用
      if (!stitched_shape_compute_graph->inputs().at(i)->hasUses()) {
        // 如果没有被使用，从 enclosing_graph_value_to_shape_graph_input_ 中删除对应项
        enclosing_graph_value_to_shape_graph_input_.erase(
            stitched_shape_compute_graph->inputs().at(i));
        // 同时从 stitched_shape_compute_graph 中删除该输入
        stitched_shape_compute_graph->eraseInput(i);
      } else {
        // 如果有被使用，则继续处理下一个输入
        ++i;
      }
    }

    // 使用 discovered_sym_shape_equalities 更新图形
    updateGraphWithSymbolicShapeEqualities(discovered_sym_shape_equalities);
    
    // 返回一个 ShapeComputeGraphMapping 对象，包括修改后的 stitched_shape_compute_graph、
    // enclosing_graph_value_to_shape_graph_input_ 和 graph_output_to_symbolic_shape_dim
    return ShapeComputeGraphMapping(
        std::move(stitched_shape_compute_graph),
        enclosing_graph_value_to_shape_graph_input_,
        std::move(graph_output_to_symbolic_shape_dim));
  }

  void updateGraphWithSymbolicShapeEqualities(
      std::unordered_map<int64_t, int64_t>& sym_shape_equalities) {
    // 遍历 beg_ 到 end_ 之间的所有节点
    for (auto it = beg_->iterator(); it != end_->iterator(); it++) {
      auto curr = *it;
      // 遍历当前节点的所有输出
      for (size_t i = 0; i < curr->outputs().size(); ++i) {
        auto output = curr->output(i);
        auto tt = output->type()->cast<TensorType>();
        // 检查输出是否是 TensorType，并且具有符号大小信息
        if (!tt || !tt->symbolic_sizes().rank()) {
          continue;
        }
        bool changed = false;
        // 获取当前符号大小信息的向量
        std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();
        // 根据 sym_shape_equalities 更新符号大小信息
        auto new_sizes =
            c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
              auto value = shape.value();
              // 如果存在对应的等式，更新大小信息
              if (sym_shape_equalities.count(value)) {
                changed = true;
                return sym_shape_equalities[value];
              }
              // 否则保持原来的大小信息
              return value;
            });
        // 如果大小信息发生了变化，则更新输出的类型
        if (changed) {
          output->setType(
              tt->withSymbolicShapes(c10::SymbolicShape(new_sizes)));
        }
      }
    }
  }

  void registerStitchedComputeOutput(
      std::shared_ptr<Graph> stitched_shape_compute_graph,
      Value* output,
      int64_t symbolic_shape) {
    // 注册输出到 stitched_shape_compute_graph
    stitched_shape_compute_graph->registerOutput(output);
    // 记录输出索引与符号形状的映射关系
    output_index_to_symbolic_shape_
        [stitched_shape_compute_graph->outputs().size() - 1] = symbolic_shape;
    // 记录符号形状值到图形输出的映射关系
    symbolic_shape_value_to_graph_output_[symbolic_shape] =
        stitched_shape_compute_graph->outputs().at(
            stitched_shape_compute_graph->outputs().size() - 1);
  }

  void joinPartialEvaluatedShapeGraphToLargeShapeGraph(
      Node* curr,
      std::shared_ptr<Graph> partial_eval_graph,
      std::shared_ptr<Graph> stitched_shape_compute_graph) {
    // 通过逐步组合部分评估的节点形状图来构建大形状计算图

    // 我们需要维护两个映射关系：
    // 一个是从封闭图中的非张量输入到大形状计算图中它们等效映射的映射关系，
    // 另一个是从符号形状维度到新节点输出的映射关系

    // 当我们添加一个新的张量节点时，我们做两件事：
    // 1：记录从张量节点输出到其在部分评估图中的形状的映射
    // 2：添加每个我们拥有的符号形状维度

    // 这里不需要具体解释代码内容，因为这段代码是对整体逻辑的描述和说明
    // 尚未添加到大型形状计算图的输出

    // 在我们完成拼接所有部分评估的图之后，可以清理图并移除不需要的完整形状作为输出，
    // 只保留计算运行时符号维度值的计算
    // 只保留计算运行时符号维度值的计算

    // 创建一个存储节点输入的向量
    std::vector<Value*> node_inputs;
    // TODO: 泛化逻辑
    if (curr->kind() == aten::cat) {
      // 断言当前操作的第一个输入节点是 prim::ListConstruct 类型
      TORCH_INTERNAL_ASSERT(
          curr->input(0)->node()->kind() == prim::ListConstruct);
      // 将 curr 的第一个输入节点的所有输入添加到 node_inputs 中
      for (Value* v : curr->input(0)->node()->inputs()) {
        node_inputs.push_back(v);
      }
      // 将 curr 的名为 "dim" 的命名输入添加到 node_inputs 中
      node_inputs.push_back(curr->namedInput("dim"));
    } else {
      // 将部分评估图的所有输入添加到 node_inputs 中
      for (size_t i = 0; i < partial_eval_graph->inputs().size(); ++i) {
        node_inputs.push_back(curr->input(i));
      }
    }

    // 创建存储部分评估输入的向量
    std::vector<Value*> partial_eval_inputs;
    for (size_t i = 0; i < node_inputs.size(); ++i) {
      auto node_input = node_inputs[i];
      // 查找当前节点输入在外围图值到形状图输入的映射中是否存在
      auto existing_graph_mapping =
          enclosing_graph_value_to_shape_graph_input_.find(node_input);
      if (existing_graph_mapping !=
          enclosing_graph_value_to_shape_graph_input_.end()) {
        // 如果存在映射，则将对应的形状图输入添加到部分评估输入中
        partial_eval_inputs.push_back(existing_graph_mapping->second);
      } else {
        // 否则，创建一个新的形状图输入，并将其加入到映射中
        Value* shape_graph_input =
            stitched_shape_compute_graph->addInput()->copyMetadata(
                partial_eval_graph->inputs().at(i));
        enclosing_graph_value_to_shape_graph_input_[node_input] =
            shape_graph_input;
        partial_eval_inputs.push_back(shape_graph_input);
      }
      // 确保我们创建的图中所有符号维度在部分评估图中计算
      if (auto tt = node_input->type()->cast<TensorType>()) {
        if (!tt->symbolic_sizes().rank()) {
          continue;
        }
        auto rank = *tt->symbolic_sizes().rank();
        for (size_t j = 0; j < rank; ++j) {
          auto shape = tt->symbolic_sizes()[j];
          if (shape.is_static() ||
              symbolic_shape_value_to_graph_output_.count(shape.value())) {
            continue;
          }
          // 获取节点输入对应的形状图输入
          auto input = enclosing_graph_value_to_shape_graph_input_[node_input];
          // 在插入点处插入常量 j
          WithInsertPoint guard(stitched_shape_compute_graph->block());
          auto index = stitched_shape_compute_graph->insertConstant(
              static_cast<int64_t>(j));
          // 插入 aten::__getitem__ 操作
          auto li_index = stitched_shape_compute_graph->insert(
              aten::__getitem__, {input, index});
          // 注册拼接计算输出
          registerStitchedComputeOutput(
              stitched_shape_compute_graph, li_index, shape.value());
        }
      }
    }

    // 设置插入点保护，以确保在正确的块中插入新节点
    WithInsertPoint guard(stitched_shape_compute_graph->block());
    // 插入形状计算图到部分评估图的图形
    std::unordered_map<Value*, Value*> value_map;
    insertGraph(
        *stitched_shape_compute_graph,
        *partial_eval_graph,
        partial_eval_inputs,
        value_map);
    for (size_t i = 0; i < curr->outputs().size(); ++i) {
      // 获取当前节点的第 i 个输出对应的新值
      Value* new_list_output = value_map[partial_eval_graph->outputs().at(i)];
      // 将当前节点的输出和新值映射关系存入字典中
      enclosing_graph_value_to_shape_graph_input_[curr->output(i)] =
          new_list_output;

      // 断言新值对应的节点是列表构造或常量
      TORCH_INTERNAL_ASSERT(
          new_list_output->node()->kind() == prim::ListConstruct ||
          new_list_output->node()->kind() == prim::Constant);
      // 断言新值没有被使用过
      TORCH_INTERNAL_ASSERT(!new_list_output->node()->hasUses());

      // 获取当前节点输出的符号化尺寸
      auto symbolic_sizes =
          curr->output(i)->type()->expect<TensorType>()->symbolic_sizes();
      // 断言符号化尺寸的维度是确定的
      TORCH_INTERNAL_ASSERT(symbolic_sizes.rank());

      // 遍历符号化尺寸的每一个维度
      for (size_t i = 0; i < *symbolic_sizes.rank(); i++) {
        // 如果当前维度是静态的，则继续下一个维度
        if (symbolic_sizes[i].is_static()) {
          continue;
        }
        // 获取符号化形状的值
        int64_t symbolic_shape = symbolic_sizes[i].value();
        // 如果已经注册过符号化形状的计算输出，则继续下一个维度
        if (symbolic_shape_value_to_graph_output_.count(symbolic_shape)) {
          continue;
        }
        // 注册符号化形状的计算输出到计算图中
        registerStitchedComputeOutput(
            stitched_shape_compute_graph,
            new_list_output->node()->input(i),
            symbolic_shape);
      }
    }
  }

  // 传播形状并收集部分评估形状图
  std::unordered_map<Node*, std::shared_ptr<Graph>>
  propagateShapesAndGatherPartialEvalShapeGraphs(AliasDb& db) {
    // 存储部分评估的形状图的映射关系
    std::unordered_map<Node*, std::shared_ptr<Graph>> partial_evaluated_graphs;
    // 遍历开始节点到结束节点之间的每一个节点
    for (auto it = beg_->iterator(); it != end_->iterator(); it++) {
      auto curr = *it;
      // 使用形状函数传播形状，并得到部分评估的形状图
      if (auto maybe_graph = PropagateShapesWithShapeFunction(curr, db)) {
        // 将当前节点及其对应的形状图存入映射关系中
        partial_evaluated_graphs[curr] = maybe_graph;
      }
    }
    // 返回部分评估的形状图的映射关系
    return partial_evaluated_graphs;
  }

  // 外部图中值到形状图输入的映射
  std::unordered_map<Value*, Value*>
      enclosing_graph_value_to_shape_graph_input_;
  // 符号化形状值到计算图输出的映射
  std::unordered_map<int64_t, Value*> symbolic_shape_value_to_graph_output_;
  // 输出索引到符号化形状的映射
  std::unordered_map<size_t, int64_t> output_index_to_symbolic_shape_;

  // 引用的计算图
  std::shared_ptr<Graph>& graph_;
  // 开始节点
  Node* beg_;
  // 结束节点
  Node* end_;
};

// 在给定块上传播形状信息，使用别名数据库进行别名分析
void PropagateShapesOnBlock(Block* b, const AliasDb& db) {
  // 遍历块中的每个节点
  for (Node* n : b->nodes()) {
    // 如果节点类型为 prim::If，处理条件分支
    if (n->kind() == prim::If) {
      // 使用 IfView 获取条件节点的视图
      IfView if_v(n);
      // 递归地传播形状信息到 if 条件成立分支
      PropagateShapesOnBlock(if_v.thenBlock(), db);
      // 递归地传播形状信息到 if 条件不成立分支
      PropagateShapesOnBlock(if_v.elseBlock(), db);
      // 合并 if 条件分支的输出形状信息
      mergeTypes(if_v.thenOutputs(), if_v.elseOutputs(), if_v.outputs());
    } 
    // 如果节点可能有架构，使用形状函数传播形状信息
    else if (n->maybeSchema()) {
      PropagateShapesWithShapeFunction(n, db);
    } 
    // 如果节点类型为 prim::TupleConstruct，重新设置输出类型
    else if (n->kind() == prim::TupleConstruct) {
      auto orig_type = n->output()->type()->expect<TupleType>();
      auto new_types = fmap(n->inputs(), [](Value* v) { return v->type(); });
      n->output()->setType(
          orig_type->createWithContained(std::move(new_types)));
    }
  }
}

// 在图上传播形状信息，使用别名数据库进行别名分析
void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  PropagateShapesOnBlock(graph->block(), db);
}

// 传播形状信息并构建大型形状计算图
std::optional<ShapeComputeGraphMapping>
PropagateShapesAndBuildLargeShapeComputeGraph(
    std::shared_ptr<Graph>& graph,
    Node* beg,
    Node* end) {
  return SymbolicShapeGraphAnalyzer(graph, beg, end).run();
}

// 计算操作符的符号形状
TORCH_API std::optional<std::vector<c10::SymbolicShape>>
calculateSymbolicShapesOnOp(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& inputs) {
  // 获取给定模式的边界图
  auto bounded_graphs = boundedGraphsForSchema(*schema);
  // 检查是否支持形状计算
  auto has_shape_compute = shapeComputeGraphForSchema(*schema) != c10::nullopt;
  // 如果不支持形状计算且没有边界图，则返回空
  if (!has_shape_compute && bounded_graphs == c10::nullopt) {
    // 避免为不支持的模式进行不必要的工作
    return c10::nullopt;
  }

  // 检查是否有缓存的形状函数
  if (auto cached_ret_vec = get_cached_shape_function(schema, inputs)) {
    return cached_ret_vec;
  }

  std::vector<SSArgument> ssa_args;
  // 将输入参数转换为 SSA 参数
  for (auto& arg : inputs) {
    if (const IValue* ival = std::get_if<IValue>(&arg)) {
      ssa_args.emplace_back(*ival);
    } else {
      const c10::SymbolicShape* ss = std::get_if<c10::SymbolicShape>(&arg);
      ssa_args.emplace_back(ShapeArguments(*ss));
    }
  }

  // 处理边界形状选项
  if (bounded_graphs) {
    // 分析下界的符号形状操作
    auto lower_bound =
        SymbolicShapeOpAnalyzer(schema, bounded_graphs->lower_bound);
    auto lower_bound_res = lower_bound.run(ssa_args);
    // 分析上界的符号形状操作
    auto upper_bound =
        SymbolicShapeOpAnalyzer(schema, bounded_graphs->upper_bound);
    auto upper_bound_res = upper_bound.run(ssa_args);
    // 组合这些值
    if (lower_bound_res.has_value() && upper_bound_res.has_value()) {
      TORCH_INTERNAL_ASSERT(lower_bound_res->size() == upper_bound_res->size());
      auto merged_res = std::vector<c10::SymbolicShape>();
      for (size_t i = 0; i < lower_bound_res->size(); i++) {
        merged_res.push_back(
            combine_bounds(lower_bound_res->at(i), upper_bound_res->at(i)));
      }
      // 缓存形状函数结果
      cache_shape_function(schema, inputs, merged_res);
      return merged_res;
    }
    return c10::nullopt;
  }

  // 如果没有边界图，则进行常规符号形状操作分析
  auto op_analyzer = SymbolicShapeOpAnalyzer(schema);
  auto res = op_analyzer.run(ssa_args);
  if (res.has_value()) {
    // 返回符号形状结果
    return res;
  }

  // 如果没有有效的结果，则返回空
  return c10::nullopt;
}
    cache_shape_function(schema, inputs, res.value());


# 调用函数 cache_shape_function，缓存计算结果的形状函数
cache_shape_function(schema, inputs, res.value());
}

// 结束命名空间 jit
} // namespace jit

// 结束命名空间 torch
} // namespace torch
```