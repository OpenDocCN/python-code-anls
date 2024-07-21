# `.\pytorch\torch\csrc\jit\runtime\operator.cpp`

```
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/edit_distance.h>

#include <queue>
#include <utility>
#include <vector>

namespace torch::jit {

namespace {
// OperatorMap用于存储操作符映射关系，将符号(Symbol)映射到其对应的操作符列表
using OperatorMap =
    std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>>;

// OperatorRegistry结构体负责注册和管理操作符及其对应的函数模式
struct OperatorRegistry {
 private:
  std::mutex lock;  // 锁，用于线程安全地操作注册和查询过程
  OperatorMap operators;  // 存储操作符的映射表
  // to_register列表存储需要注册但尚未解析其模式的操作符
  std::vector<std::shared_ptr<Operator>> to_register;
  // operators_by_sig和operators_by_sig_literal用于实现lookupByLiteral功能，
  // 提高函数模式匹配效率
  std::unordered_map<std::string, std::shared_ptr<Operator>> operators_by_sig;
  std::unordered_map<const char*, std::shared_ptr<Operator>>
      operators_by_sig_literal;

  // 注册待注册的操作符
  // XXX - 调用者必须持有锁
  void registerPendingOperators() {
    for (const auto& op : to_register) {
      // 获取操作符的符号，并将其加入到operators映射中
      Symbol sym = Symbol::fromQualString(op->schema().name());
      operators[sym].push_back(op);
      // 将操作符按其规范化模式字符串加入operators_by_sig映射中
      operators_by_sig[canonicalSchemaString(op->schema())] = op;
    }
    // 清空待注册列表
    to_register.clear();
  }

 public:
  // 注册操作符
  void registerOperator(Operator&& op) {
    std::lock_guard<std::mutex> guard(lock);  // 加锁
#ifdef C10_MOBILE
    // 断言确保操作符名称未注册过，避免重复注册
    TORCH_INTERNAL_ASSERT(
        0 == registered_operator_names.count(op.schema().operator_name()),
        "Tried to register operator \"",
        op.schema(),
        "\" to JIT but the operator name was already registered before. Please add or change the overload name.");
    // 将操作符名称加入已注册名称集合中
    registered_operator_names.insert(op.schema().operator_name());
#endif
    // 将操作符移动构造到to_register列表中
    to_register.push_back(std::make_shared<Operator>(std::move(op)));
  }

  // 反注册操作符
  void deregisterOperator(const FunctionSchema& schema) {
    // 获取操作符的符号和规范化模式字符串
    Symbol sym = Symbol::fromQualString(schema.name());
    auto sig = canonicalSchemaString(schema);

    std::lock_guard<std::mutex> guard(lock);  // 加锁
#ifdef C10_MOBILE
    // 如果定义了 C10_MOBILE 宏，则执行以下操作
    TORCH_INTERNAL_ASSERT(
        1 == registered_operator_names.count(schema.operator_name()),
        "Tried to remove operator ",
        schema,
        " from JIT but it wasn't found.");
    // 检查注册的运算符名称中是否存在指定的运算符名，确保只有一个，否则断言失败
    registered_operator_names.erase(schema.operator_name());
#endif

    // 尝试首先从待注册的运算符列表中移除
    auto pending_it = to_register.begin();
    while (pending_it != to_register.end() && (*pending_it)->schema() != schema)
      ++pending_it;

    if (pending_it != to_register.end()) {
      // 如果找到了待注册列表中的运算符，将其移除并返回
      to_register.erase(pending_it);
      return;
    }

    // 从签名映射中移除运算符
    auto sig_it = operators_by_sig.find(sig);
    if (sig_it == operators_by_sig.end()) {
      // 如果在签名映射中找不到对应的签名，则直接返回
      return;
    }

    operators_by_sig.erase(sig_it);

    // 从符号映射中移除运算符
    auto op_it = operators.find(sym);
    TORCH_CHECK(
        op_it != operators.end(),
        "operator with signature ",
        sig,
        " is missing from symbol registry");
    // 检查运算符是否在符号映射中，如果不存在则抛出错误信息

    auto& op_vec = op_it->second;
    auto it = op_vec.begin();
    while (it != op_vec.end() && (*it)->schema() != schema)
      ++it;
    if (it != op_vec.end()) {
      // 如果找到了运算符在向量中的位置，则移除它
      op_vec.erase(it);
    }
    if (op_vec.empty()) {
      // 如果移除运算符后该符号对应的向量为空，则从符号映射中移除该符号
      operators.erase(op_it);
    }
  }

  // 根据字面量名称查找运算符
  const std::shared_ptr<Operator>& lookupByLiteral(const char* name) {
    std::lock_guard<std::mutex> guard(lock);
    // 锁定互斥量以保证线程安全
    registerPendingOperators();
    // 注册待处理运算符
    auto it = operators_by_sig_literal.find(name);
    if (it == operators_by_sig_literal.end()) {
      // 如果在字面量名称映射中找不到该名称，则尝试从标准化模式字符串中查找对应的运算符
      auto op_ptr_it =
          operators_by_sig.find(canonicalSchemaString(parseSchema(name)));
      // 调试代码，用于在匹配失败时输出所有已知运算符
#if 0
      if (op_ptr_it == operators_by_sig.end()) {
        for (auto & entry : operators_by_sig) {
          std::cout << entry.first << std::endl;
        }
      }
#endif
      TORCH_CHECK(
          op_ptr_it != operators_by_sig.end(),
          "Couldn't find an operator for ",
          name,
          ". Do you have to update a set of hardcoded JIT ops?");
      // 如果找到了匹配的运算符，则将其添加到字面量名称映射中
      it = operators_by_sig_literal.emplace_hint(it, name, op_ptr_it->second);
    }
    return it->second;
  }

  // 根据符号名称获取所有运算符
  const std::vector<std::shared_ptr<Operator>>& getOperators(Symbol name) {
    std::lock_guard<std::mutex> guard(lock);
    // 锁定互斥量以保证线程安全
    registerPendingOperators();
    // 注册待处理运算符
    static std::vector<std::shared_ptr<Operator>> empty;
    auto it = operators.find(name);
    if (it != operators.end())
      return it->second;
    // 如果找到了指定符号对应的运算符向量，则返回之
    return empty;
    // 否则返回空向量
  }

  // 查找与输入运算符相似的所有运算符
  std::vector<Symbol> findSimilarOperators(Symbol input_op) {
    std::lock_guard<std::mutex> guard(lock);
    // 锁定互斥量以保证线程安全
    registerPendingOperators();
    // 注册待处理运算符

    using EntryPair = std::pair<int64_t, Symbol>;
    auto cmp = [](const EntryPair& lhs, const EntryPair& rhs) {
      return lhs.first > rhs.first;
    };

    std::priority_queue<EntryPair, std::vector<EntryPair>, decltype(cmp)>
        rankings(cmp);
    // 使用优先级队列对运算符进行排序，依据编辑距离和符号进行比较
    static constexpr size_t MAX_EDIT_DIST = 2u;
    // 设置最大编辑距离为2
    for (const auto& op : operators) {
      // 遍历 operators 容器中的每个操作符对象
      auto edit_dist = ComputeEditDistance(
          input_op.toQualString(), op.first.toQualString(), MAX_EDIT_DIST);
      // 计算输入操作符 input_op 和当前遍历的操作符 op 的编辑距离
      if (edit_dist <= MAX_EDIT_DIST) {
        // 如果编辑距离小于等于最大编辑距离 MAX_EDIT_DIST
        // 将该操作符 op 插入到 rankings 中，使用编辑距离作为键
        rankings.emplace(edit_dist, op.first);
      }
    }
    // 初始化返回的符号向量
    std::vector<Symbol> ret;
    // 当 rankings 非空时持续执行循环
    while (!rankings.empty()) {
      // 将排名最高（编辑距离最小）的操作符加入返回的符号向量 ret 中
      ret.push_back(rankings.top().second);
      // 弹出当前编辑距离最小的操作符
      rankings.pop();
    }
    // 返回填充好的符号向量 ret
    return ret;
  }

  const std::vector<std::shared_ptr<Operator>> getAllOperators() {
    // 使用互斥锁保护，确保线程安全
    std::lock_guard<std::mutex> guard(lock);
    // 注册尚未注册的操作符
    registerPendingOperators();
    // 初始化存储所有操作符的共享指针向量 values
    std::vector<std::shared_ptr<Operator>> values;
    // 清空 values 容器，以便重新填充
    values.clear();
    // 遍历 operators 容器中的所有键值对
    for (auto& kv : operators) {
      // 将每个键（操作符名称）对应的值（操作符对象的共享指针）插入 values 向量的末尾
      values.insert(values.end(), kv.second.begin(), kv.second.end());
    }
    // 返回包含所有操作符的共享指针向量 values
    return values;
  }
};

OperatorRegistry& getRegistry() {
  // 获取静态的操作符注册表实例
  static OperatorRegistry r;
  // 返回静态的操作符注册表实例
  return r;
}

}

} // anonymous namespace

bool aliasAnalysisHasSpecialCaseFor(Symbol symbol) {
  using namespace at;
  // 警告：通过添加到这个列表中，表明你已经在 AliasDb::analyze 中为未计划的节点添加了一个情况
  const static std::unordered_set<Symbol> handled = {
      prim::If,
      prim::Loop,
      prim::FusionGroup,
      prim::CudaFusionGroup,
      prim::oneDNNFusionGroup,
      prim::DifferentiableGraph,
      prim::TensorExprGroup,
      prim::TensorExprDynamicGroup,
      prim::StaticSubgraph,
      prim::FunctionalGraph,
      prim::Constant,
      prim::Uninitialized,
      prim::DictConstruct,
      prim::ListConstruct,
      prim::TupleConstruct,
      prim::AutogradZero,
      prim::FusedConcat,
      prim::GradOf,
      prim::MMTreeReduce,
      prim::MMBatchSide,
      prim::BroadcastSizes,
      prim::ChunkSizes,
      prim::Closure,
      prim::TupleUnpack,
      prim::TupleIndex,
      prim::TupleSlice,
      prim::ListUnpack,
      prim::PythonOp,
      prim::ConstantChunk,
      prim::BroadcastingChunk,
      prim::MKLDNNGroup,
      prim::ConstantMKLDNNTensor,
      prim::BroadcastMKLDNNTensors,
      prim::fork,
      prim::awaitable,
      prim::awaitable_nowait,
      prim::awaitable_wait,
      prim::CreateObject,
      prim::AutogradAdd,
      prim::GetAttr,
      prim::SetAttr,
      prim::profile,
      prim::profile_ivalue,
      prim::TypeCheck,
      prim::RequiresGradCheck,
      prim::Print,
      prim::CallFunction,
      prim::CallMethod,
      aten::wait,
      prim::isinstance,
      prim::unchecked_cast,
      prim::tolist,
      prim::rpc_async,
      prim::rpc_sync,
      prim::rpc_remote,
      prim::Enter,
      prim::Exit,
      prim::FallbackGraph,
  };

  // 不应由别名分析使用的操作符
  const static std::unordered_set<Symbol> purposefully_not_handled = {
      prim::Load,
      prim::Store,
      prim::Drop,
      at::onnx::Reshape,
      at::onnx::Shape,
      prim::AutogradAdd,
  };

  // 检查操作符是否在已处理集合或者有意不处理集合中
  return handled.count(symbol) || purposefully_not_handled.count(symbol);
}

void registerOperator(Operator&& op) {
  if (op.schema().is_varret()) {
    // 从限定字符串创建符号
    Symbol s = Symbol::fromQualString(op.schema().name());
    // 如果打印机没有特殊情况处理该符号，则报错
    if (!printerHasSpecialCaseFor(s)) {
      AT_ERROR(
          "Missing special case in python printer for non-schematized"
          " operator ",
          op.schema().name(),
          ". File a bug to add a case for this operator.\n");
    }
    // 如果别名分析中有特殊情况处理该符号，并且分析类型是保守的，则报错
    if (aliasAnalysisHasSpecialCaseFor(s) &&
        op.aliasAnalysisKind() == AliasAnalysisKind::CONSERVATIVE) {
      AT_ERROR(
          "Conflict in special casing in alias analysis for non-schematized"
          " operator ",
          op.schema().name(),
          ". File a bug to add a case for this operator.\n");
    }
    if (aliasAnalysisHasSpecialCaseFor(s) &&
        op.aliasAnalysisKind() == AliasAnalysisKind::FROM_SCHEMA) {
      // 检查是否针对当前操作符（op）有特殊的别名分析情况，并且该操作符的别名分析类型是从模式中确定的
      AT_ERROR(
          "The operator ",
          op.schema().name(),
          " is special cased and cannot use explicit alias analysis.");
      // 抛出错误，指出该操作符由于特殊情况无法使用显式别名分析
    }
  }
  // 将操作符(op)注册到注册表中
  getRegistry().registerOperator(std::move(op));
}

// 取消注册给定函数模式的操作符
void deregisterOperator(const FunctionSchema& schema) {
  // 获取注册表并调用其取消注册操作符的方法
  getRegistry().deregisterOperator(schema);
}

// 返回所有操作符的共享指针向量
const std::vector<std::shared_ptr<Operator>> getAllOperators() {
  // 获取注册表并调用其获取所有操作符的方法
  return getRegistry().getAllOperators();
}

// 返回给定符号名称的所有操作符的共享指针向量的引用
const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name) {
  // 获取注册表并调用其根据符号名称获取操作符的方法
  return getRegistry().getOperators(name);
}

// 返回按符号名称排序的所有操作符的共享指针向量
std::vector<std::shared_ptr<Operator>> getAllSortedOperatorsFor(Symbol name) {
  const auto& unsortedOps = getAllOperatorsFor(name);
  // 根据是否为C10操作符进行排序，以确保顺序一致性
  std::vector<std::shared_ptr<Operator>> sortedOps;
  sortedOps.reserve(unsortedOps.size());
  std::copy_if(
      unsortedOps.begin(),
      unsortedOps.end(),
      std::back_inserter(sortedOps),
      [](const std::shared_ptr<Operator>& op) { return op->isC10Op(); });
  std::copy_if(
      unsortedOps.begin(),
      unsortedOps.end(),
      std::back_inserter(sortedOps),
      [](const std::shared_ptr<Operator>& op) { return !op->isC10Op(); });
  return sortedOps;
}

// 查找给定完整操作符名称的操作符
std::shared_ptr<Operator> findOperatorFor(const c10::OperatorName& full_name) {
  for (const auto& op :
       getRegistry().getOperators(Symbol::fromQualString(full_name.name))) {
    if (op->schema().overload_name() == full_name.overload_name) {
      return op;
    }
  }
  return nullptr;
}

// 查找与输入操作符符号相似的符号
std::vector<Symbol> findSimilarOperators(Symbol input_op) {
  return getRegistry().findSimilarOperators(input_op);
}

// 根据字面值查找操作符的共享指针
std::shared_ptr<Operator> getOperatorForLiteral(const char* signature) {
  return getRegistry().lookupByLiteral(signature);
}

// 返回给定函数模式的规范化字符串表示形式
std::string canonicalSchemaString(const FunctionSchema& schema) {
  std::string out = schema.name();
  out.push_back('(');

  bool seen_kwarg_only = false;
  for (const auto i : c10::irange(schema.arguments().size())) {
    if (i > 0) {
      out += ", ";
    }
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out += "*, ";
      seen_kwarg_only = true;
    }
    const auto& arg = schema.arguments()[i];
    out += arg.type()->str();
    out.push_back(' ');
    out += arg.name();
  }

  out += ") -> ";
  if (schema.returns().size() == 1) {
    out += schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out.push_back('(');
    for (const auto i : c10::irange(schema.returns().size())) {
      if (i > 0) {
        out += ", ";
      }
      out += schema.returns()[i].type()->str();
    }
    out.push_back(')');
  }
  return out;
}

} // namespace torch::jit


这些注释按照要求对每行代码进行了解释，描述了其具体作用和功能。
```