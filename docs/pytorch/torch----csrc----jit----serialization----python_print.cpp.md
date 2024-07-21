# `.\pytorch\torch\csrc\jit\serialization\python_print.cpp`

```
// 包含 Torch 序列化和打印相关的头文件
#include <torch/csrc/jit/serialization/python_print.h>

// 包含标准库的算法相关头文件
#include <algorithm>

// 包含 ATen 库的核心 IValue 和 QualifiedName 相关头文件
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>

// 包含 C10 库的异常处理、字符串工具和范围迭代相关头文件
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>

// 包含 Caffe2 序列化版本相关头文件
#include <caffe2/serialize/versions.h>

// 包含 Torch JIT API 的函数实现和模块相关头文件
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>

// 包含 Torch JIT 前端错误报告和版本化符号相关头文件
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/versioned_symbols.h>

// 包含 Torch JIT IR 属性和 IR 相关头文件
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>

// 包含 Torch JIT 运算符升级器版本映射相关头文件
#include <torch/csrc/jit/operator_upgraders/version_map.h>

// 包含 Torch JIT 资源保护相关头文件
#include <torch/csrc/jit/resource_guard.h>

// 包含 Torch JIT 运行时计算必要参数相关头文件
#include <torch/csrc/jit/runtime/calculate_necessary_args.h>

// 包含 Torch JIT 序列化类型名称唯一化相关头文件
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

// 使用 c10 命名空间下的 QualifiedName 类型
using c10::QualifiedName;

// Torch JIT 命名空间
namespace torch::jit {

// 静态函数：判断字符是否是有效的标识符字符
static bool isValidIdentifierChar(char c, size_t pos) {
  return islower(c) || isupper(c) || c == '_' || (pos > 0 && isdigit(c));
}

// 静态函数：判断字符串是否是有效的标识符
static bool isValidIdentifier(const std::string& name) {
  if (name.empty())
    return false;
  for (const auto i : c10::irange(name.size())) {
    if (!isValidIdentifierChar(name[i], i))
      return false;
  }
  return true;
}

// 一些名称是有效的标识符但是不能使用，因为它们是关键字或输出中使用的命名空间
const static std::unordered_set<std::string> reserved_names = {
    // 在解析环境中的标识符
    "_", // 避免混淆的未命名 _
    "as",
    "aten",
    "attribute",
    "CONSTANTS",
    "fork",
    "getattr",
    "inf",
    "nan",
    "infj",
    "nanj",
    "ops",
    "__torch__",
    // Python 关键字
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "with",
    "while",
    "with",
    "yield",
    "uninitialized",
    "unchecked_cast",
};

// 辅助函数：避免重复添加类类型到依赖表
void PrintDepsTable::add(const c10::NamedTypePtr& type) {
  // 尽管在下面进行线性搜索，我们不想做无用的工作，只尝试插入每个实例一次
  if (!non_unique_.insert(type).second) {
    return;
  }
  // 需要进行实际的相等比较，而不是指针相等。这是因为对于某些类型（例如 FunctionType），可能有多个 TypePtr 表示相同的底层对象。
  // TODO: 应该使用更高效的方法替换这里的线性搜索
  auto it = std::find_if(
      table_.cbegin(), table_.cend(), [&](const c10::NamedTypePtr& dep) {
        return *dep == *type;
      });

  if (it == table_.cend()) {
    table_.push_back(type);
  }
}
struct PythonPrintImpl {
  using SourceRangeStack = std::vector<SourceRange>;
  // 初始化一个源代码范围栈，包含一个空的默认 SourceRange
  SourceRangeStack source_range_stack_ = {SourceRange()};

  struct WithSourceRange {
    explicit WithSourceRange(SourceRangeStack* stack, Node* n) : stack(stack) {
      // 断言栈不为空
      TORCH_INTERNAL_ASSERT(stack);
      // 如果能找到生成源范围，则将其推入栈中；否则推入节点的原始源范围
      if (auto gen_source = n->sourceRange().findSourceRangeThatGenerated()) {
        stack->push_back(std::move(gen_source.value()));
      } else {
        stack->push_back(n->sourceRange());
      }
    }

    ~WithSourceRange() {
      // 弹出栈顶的源范围
      stack->pop_back();
    }

    SourceRangeStack* stack;
  };

  class TaggedStringStream {
   public:
    // 构造函数，初始化时传入源代码范围栈的指针
    TaggedStringStream(const SourceRangeStack* srs) : srs_(srs) {}

    // 插入字符串操作符重载，用于向流中插入字符串
    TaggedStringStream& operator<<(const std::string& s) {
      // 避免在相同偏移量出现冗余条目，例如在 printValueList 中，当 begin 和 end 都是空字符串时
      if (s.empty()) {
        return *this;
      }

      // 如果当前范围栈为空或者最后一个范围与当前流的最后源范围不同，则添加新的标记范围
      if (ranges_.empty() || ranges_.back().range != srs_->back()) {
        ranges_.emplace_back((size_t)oss_.tellp(), srs_->back());
      }
      // 向字符串流中插入字符串
      oss_ << s;
      return *this;
    }

    // 流合并操作符重载，用于合并两个流
    TaggedStringStream& operator<<(const TaggedStringStream& rhs) {
      // 遍历 rhs 中的所有范围，如果当前范围栈为空或者最后一个范围与 rhs 的范围不同，则添加新的标记范围
      for (const auto& range : rhs.ranges_) {
        if (ranges_.empty() || ranges_.back().range != range.range) {
          ranges_.emplace_back((size_t)oss_.tellp() + range.bytes, range.range);
        }
      }
      // 合并 rhs 的字符串流到当前流中
      oss_ << rhs.oss_.str();
      return *this;
    }

    // 防止输出 TaggedStringStream 地址的操作符重载
    TaggedStringStream& operator<<(
        const std::shared_ptr<TaggedStringStream>& rhs) {
      (*this) << *rhs;
      return *this;
    }

    // 通用类型插入操作符重载，用于插入任意类型到流中
    template <typename T>
    TaggedStringStream& operator<<(const T& t) {
      // 如果当前范围栈为空或者最后一个范围与当前流的最后源范围不同，则添加新的标记范围
      if (ranges_.empty() || ranges_.back().range != srs_->back()) {
        ranges_.emplace_back((size_t)oss_.tellp(), srs_->back());
      }
      // 向字符串流中插入数据
      oss_ << t;
      return *this;
    }

    // 返回流的字符串表示
    std::string str() const {
      return oss_.str();
    }

    // 返回所有标记范围的向量
    const std::vector<TaggedRange>& ranges() const {
      return ranges_;
    }

   private:
    std::ostringstream oss_;  // 字符串流对象
    std::vector<TaggedRange> ranges_;  // 标记范围的向量
    const SourceRangeStack* srs_;  // 源代码范围栈的指针
  const SourceRangeStack* srs_;

  // scanValue, scanNode, scanBlock:
  // 决定是否可以安全地省略临时变量的输出，并将表达式内联到其使用处
  // 只有在以下情况下才会执行这种优化：
  // (1) 变量是常量，或者
  // (2) 临时变量没有命名，只有一个输出，被使用了一次，
  //     并且在表达式树被重新解析时，它们的顺序是一致的。
  // 最后一种情况可以通过以下方式检查：
  // 在解析器中，我们对表达式树进行从左到右的后序遍历（先生成子节点，再生成操作符）。
  // 这个顺序的反向是树的右到左的前序遍历。通过对节点的输入进行右到左的前序遍历，
  // 同时向后扫描已生成节点的列表，我们可以看到它们是否与将节点作为表达式解析时的顺序一致。
  // 只有在它们一致时，我们才将它们折叠成一个内联表达式。

  // 归纳步骤是，右侧最远的输入应该由当前节点的前一个节点产生，如果它们是按照树的顺序排列的。

  bool canInline(Value* v) {
    Node* n = v->node();
    // 如果节点有多个输出值，则无法进行内联优化
    if (n->outputs().size() != 1)
      return false;
    // 如果值被多次使用，则需要保留变量，无法进行内联优化
    if (v->uses().size() != 1)
      return false;
    auto use = v->uses().at(0);
    // 如果值有调试名称，并且它的使用不是直接作为块的结束返回，则无法进行内联优化
    if (v->hasDebugName() && use.user->kind() != prim::Return)
      return false;
    // 不尝试内联控制块
    if (!n->blocks().empty())
      return false;
    // 如果值是循环传递的输入，需要保留变量，以确保条件或迭代计数的顺序不会出错
    if (use.user->kind() == prim::Loop && use.offset >= 2)
      return false;

    // 子图可能会多次使用该值，因此禁用内联优化
    if (use.user->kind() == prim::fork || use.user->kind() == prim::rpc_async ||
        use.user->kind() == prim::rpc_sync ||
        use.user->kind() == prim::rpc_remote)
      return false;

    // isinstance 出现在 if 表达式中会导致类型细化发生，
    // 但我们已经处理了细化并插入了强制转换表达式。
    // 通过不将其内联到 if 条件中，我们防止它再次发生。
    if (v->node()->kind() == prim::isinstance) {
      return false;
    }

    return true;
  }

  // block_point 是反向线性扫描中当前节点的位置
  // v 是树遍历中当前值，可能与 block_point 的输出匹配
  Node* scanValue(Node* block_point, Value* v) {
    Node* n = v->node();
    // ...
  }
    // 确保节点是常量或者不在输出内联集合中
    AT_ASSERT(n->kind() == prim::Constant || output_inline_.count(n) == 0);

    // 如果节点处于块点位置，并且可以内联，则递归地检查是否可以内联该节点的输入
    if (n == block_point &&
        canInline(v)) { // 节点必须位于典型树遍历的预期位置
      block_point = scanNode(block_point); // 递归扫描该块点
      output_inline_.insert(n); // 将该节点添加到输出内联集合中
    } else if (n->kind() == prim::Constant) {
      // 常量节点总是可以内联的，在解析时我们将对它们进行去重，并放置在函数顶部
      output_inline_.insert(n); // 将该常量节点添加到输出内联集合中
    }
    return block_point; // 返回块点位置
  }

  Node* previousNonConstant(Node* n) {
    do {
      n = n->prev(); // 获取上一个非常量节点
    } while (n->kind() == prim::Constant); // 跳过所有常量节点
    return n; // 返回第一个非常量节点
  }

  Node* scanNode(Node* n) {
    // 如果节点已确定为内联，则不必扫描
    if (output_inline_.count(n)) {
      return n;
    }
    for (auto b : n->blocks()) {
      scanBlock(b); // 扫描节点所含的所有块
    }
    Node* block_point = previousNonConstant(n); // 获取节点前一个非常量节点
    for (auto it = n->inputs().rbegin(), end = n->inputs().rend(); it != end;
         ++it) {
      block_point = scanValue(block_point, *it); // 递归扫描节点的输入值
    }
    return block_point; // 返回块点位置
  }

  void scanBlock(Block* b) {
    scanNode(b->return_node()); // 扫描块的返回节点
    for (auto node : b->nodes().reverse()) {
      scanNode(node); // 逆序扫描块中的所有节点
    }
  }

  size_t getOrAddConstant(at::IValue val) {
    // XXX - N^2 警告。这段代码与ConstantPool做完全相同的事情，
    // 因为它不对张量的任何信息进行哈希处理，可能在某些时候需要优化，使用哈希算法。
    if (val.isTensor()) {
      auto& t = val.toTensor();
      for (const auto i : c10::irange(constant_table_.size())) {
        if (!constant_table_[i].isTensor()) {
          continue;
        }
        auto& t2 = constant_table_[i].toTensor();
        if (t.options().type_equal(t2.options()) && t.equal(t2)) {
          return i; // 如果找到相同的张量，返回其索引
        }
      }
    }
    constant_table_.emplace_back(std::move(val)); // 否则将新的常量值添加到常量表中
    return constant_table_.size() - 1; // 返回新常量值在表中的索引
  }

  std::unordered_set<Node*> seen_constants;
  void buildConstantList(Node* n, std::vector<Node*>& constants) {
    for (auto input : n->inputs()) {
      if (input->node()->kind() == prim::Constant &&
          seen_constants.count(input->node()) == 0) {
        constants.push_back(input->node()); // 将未见过的常量节点添加到常量列表中
        seen_constants.insert(input->node()); // 标记该常量节点已经见过
      }
    }
    for (auto b : n->blocks()) {
      buildConstantList(b, constants); // 递归构建块内的常量列表
    }
    buildConstantList(n->return_node(), constants); // 构建返回节点的常量列表
  }

  void buildConstantList(Block* b, std::vector<Node*>& constants) {
    for (auto n : b->nodes()) {
      buildConstantList(n, constants); // 构建块中节点的常量列表
    }
    buildConstantList(b->return_node(), constants); // 构建块的返回节点的常量列表
  }

  // 生成一个在调试名称和已使用名称上唯一的新名称
  std::unordered_map<std::string, size_t> next_id;

  std::string genNameImpl(
      const std::string& candidate,
      std::unordered_set<std::string>& used) {
    // 将候选名称初始化为指定的名称
    std::string name = candidate;
    // 检查名称是否已经被使用或者是保留名称，如果是，则生成一个新的名称
    while (used.count(name) || reserved_names.count(name)) {
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      // 使用 NOLINT 注释来忽略性能警告，这里执行候选名称与数字后缀的字符串连接
      name = candidate + std::to_string(next_id[name]++);
    }
    // 将已使用的名称加入集合中
    used.insert(name);
    // 返回生成的唯一名称
    return name;
    }
    
    // 生成符合标识符要求的有效名称，使其成为有效的 C++ 标识符
    static std::string makeValidIdentifier(const std::string& candidate) {
      std::stringstream ss;
      // 如果候选名称为空或者以数字开头，则在标识符前加下划线
      if (candidate.empty() || isdigit(candidate[0]))
        ss << "_";
      // 遍历候选名称中的每个字符，将不符合标识符要求的字符替换为下划线
      for (char c : candidate) {
        if (isupper(c) || islower(c) || isdigit(c) || c == '_')
          ss << c;
        else
          ss << '_';
      }
      // 返回生成的有效标识符名称
      return ss.str();
    }
    
    // 为给定的值生成唯一名称，如果该值有调试名称，则使用调试名称；否则生成一个默认名称
    std::string genUniqueNameFor(Value* v) {
      return genName(
          v->hasDebugName() ? makeValidIdentifier(v->debugNameBase()) : "_");
    }
    
    // 将值(Value)映射到其在每次使用时应该打印的方式的表格
    std::unordered_map<Value*, std::shared_ptr<TaggedStringStream>> expr_table_;
    // 将值(Value)映射到其标识符引用的字符串表示形式
    std::unordered_map<Value*, std::string> ident_refs_;
    
    // 获取给定值的使用方式(TaggedStringStream)，优先考虑标识符引用(ident_refs_)，其次是表达式引用(expr_table_)
    std::shared_ptr<TaggedStringStream> useOf(Value* v) const {
      if (ident_refs_.count(v)) {
        // 如果标识符引用表中存在该值，则创建一个新的标记字符串流，并复制引用的字符串
        auto rv = std::make_shared<TaggedStringStream>(&source_range_stack_);
        (*rv) << ident_refs_.at(v);
        return rv;
      }
      // 如果表达式引用表中存在该值，则直接返回表中存储的字符串流
      if (expr_table_.count(v)) {
        return expr_table_.at(v);
      }
      // 如果既不在标识符引用表中，也不在表达式引用表中，抛出断言错误
      TORCH_INTERNAL_ASSERT(
          false,
          "Value (debug name: \"",
          v->debugName(),
          "\") was not present in either expressions table or ident refs table");
    }
    
    // 将给定值(Value)关联的字符串赋值给标识符引用表(ident_refs_)
    void assignValue(Value* v, const std::string& s) {
      ident_refs_[v] = s;
    }
    
    // 将给定值(Value)关联的字符串流赋值给表达式引用表(expr_table_)
    void assignValue(Value* v, std::shared_ptr<TaggedStringStream> s) {
      expr_table_[v] = std::move(s);
    }
    
    // 将值(Value) w 关联的字符串流或标识符赋值给值(Value) v
    void assignValue(Value* v, Value* w) {
      assignValue(v, useOf(w));
    }
    
    // 将给定的值数组中的每个值关联到其唯一名称
    void assignValuesToTheirUniqueNames(at::ArrayRef<Value*> values) {
      for (auto v : values) {
        assignValue(v, genUniqueNameFor(v));
      }
    }
    
    // 当前缩进级别的计数器
    size_t level = 0;
    // 缩进到当前缩进级别
    TaggedStringStream& indent() {
      for (const auto i : c10::irange(level)) {
        (void)i; // 抑制未使用变量警告
        body_ << "  ";
      }
      // 返回缩进后的字符串流
      return body_;
    }
    
    // ResourceGuard 结构，用于在其生命周期中增加缩进级别
    ResourceGuard WithIndented() {
      level++;
      // ...
    }
    return ResourceGuard([this] { level--; });
  }

  template <class T0, class T1, class F>
  void zipWith(at::ArrayRef<T0> list_a, at::ArrayRef<T1> list_b, F action)
      const {
    auto it_a = list_a.begin();
    auto it_b = list_b.begin();

    if (list_a.size() != list_b.size()) {
      AT_ERROR("Python printer expected 2 lists of same size");
    }

    // 遍历两个数组并对每一对元素执行指定的操作函数
    for (; it_a != list_a.end(); ++it_a, ++it_b) {
      action(*it_a, *it_b);
    }
  }

  void printValueList(
      TaggedStringStream& stmt,
      at::ArrayRef<Value*> list,
      const char* begin = "",
      const char* end = "") {
    // 打印值列表的起始部分
    stmt << begin;
    auto delimiter = "";
    // 遍历值列表并打印每个值及其之间的分隔符
    for (auto* value : list) {
      stmt << delimiter;
      stmt << useOf(value);  // 使用自定义函数将值转换为字符串形式添加到语句流中
      delimiter = ", ";
    }
    // 打印值列表的结束部分
    stmt << end;
  }

  void printValueIndex(TaggedStringStream& stmt, at::ArrayRef<Value*> inputs) {
    // 获取输入值的名称字符串
    const std::string val_name = useOf(inputs[0])->str();
    // 如果名称是有效标识符直接打印，否则用括号括起来再打印
    if (isValidIdentifier(val_name)) {
      stmt << val_name;
    } else {
      stmt << "(" << val_name << ")";
    }
    stmt << "[";  // 打印索引操作符的起始部分
    stmt << useOf(inputs[1]);  // 打印索引值
    stmt << "]";  // 打印索引操作符的结束部分
  }

  void printDict(
      TaggedStringStream& stmt,
      at::ArrayRef<Value*> key_value_pairs,
      const char* begin = "{",
      const char* end = "}") {
    // 打印字典的起始部分
    stmt << begin;
    auto delimiter = "";
    // 遍历键值对数组，打印每对键值及其之间的分隔符
    for (size_t i = 0; i < key_value_pairs.size(); i += 2) {
      stmt << delimiter;
      auto key = key_value_pairs[i];
      auto value = key_value_pairs[i + 1];

      // 打印键值对，格式为 key: value
      stmt << useOf(key) << ": " << useOf(value);

      delimiter = ", ";
    }
    // 打印字典的结束部分
    stmt << end;
  }

  void printAssignment(at::ArrayRef<Value*> lhs, at::ArrayRef<Value*> rhs) {
    if (lhs.empty()) {
      return;
    }
    indent();  // 缩进输出
    printValueList(body_, lhs);  // 打印左侧值列表
    // 如果左侧只有一个值且不是元组解包语句，需要保留 Union/Optional 类型注解
    if (lhs.size() == 1) {
      Value* v = lhs.at(0);
      if (!annotated_unions_.count(v) && !expr_table_.count(v) &&
          (v->type()->kind() == UnionType::Kind ||
           v->type()->kind() == OptionalType::Kind)) {
        body_ << " : " << v->type()->annotation_str();  // 打印类型注解
        annotated_unions_.insert(v);
      }
    }
    body_ << " = ";  // 打印赋值符号
    printValueList(body_, rhs);  // 打印右侧值列表
    body_ << "\n";  // 换行
  }

  bool requiresAnnotation(Value* lhs, Value* rhs) {
    // 如果左侧值类型是 Union 或 Optional，需要添加注解
    if (lhs->type()->kind() == UnionType::Kind ||
        lhs->type()->kind() == OptionalType::Kind) {
      return annotated_unions_.insert(lhs).second;
    } else {
      return *lhs->type() != *rhs->type();  // 检查左右值类型是否相同
    }
  }

  void printAnnotatedAssignment(
      at::ArrayRef<Value*> lhs,
      at::ArrayRef<Value*> rhs) {
    for (const auto i : c10::irange(lhs.size())) {
      // 遍历 lhs 容器中的每个元素
      indent();
      // 调用缩进函数，准备输出
      body_ << useOf(lhs[i]);
      // 将 lhs[i] 的使用情况输出到 body_
      if (requiresAnnotation(lhs[i], rhs[i])) {
        // 如果 lhs[i] 和 rhs[i] 需要注释
        body_ << ": " << lhs[i]->type()->annotation_str(type_printer_);
        // 在 body_ 中添加类型注释字符串
      }
      body_ << " = " << useOf(rhs[i]) << "\n";
      // 将 rhs[i] 的使用情况输出到 body_，并换行
    }
  }

  void printIf(IfView stmt) {
    // 打印 If 语句的条件部分
    assignValuesToTheirUniqueNames(stmt.outputs());
    // 为 If 语句的输出值分配唯一名称
    indent() << "if " << useOf(stmt.cond()) << ":\n";
    // 输出 If 语句的条件表达式
    {
      auto guard = WithIndented();
      // 创建一个缩进的代码块
      // 打印 If 语句的 then 分支内容
      printBlock(stmt.thenBlock(), !stmt.outputs().empty());
      // 如果输出不为空，则打印分配语句
      printAssignment(stmt.outputs(), stmt.thenOutputs());
      // 打印输出值的分配语句
    }
    indent() << "else:\n";
    // 输出 Else 分支
    {
      auto guard = WithIndented();
      // 创建一个缩进的代码块
      // 打印 If 语句的 else 分支内容
      printBlock(stmt.elseBlock(), !stmt.outputs().empty());
      // 如果输出不为空，则打印分配语句
      printAssignment(stmt.outputs(), stmt.elseOutputs());
      // 打印输出值的分配语句
    }
  }

  void printLoop(LoopView stmt) {
    // 处理循环传递的依赖关系，将初始值分配给节点的输出
    auto loop_type = stmt.loopType();
    if (loop_type == LoopView::ModifiedLoop) {
      throw ErrorReport(stmt.node()->sourceRange())
          << "loop cannot be printed as python "
          << "because it has gone through an optimization "
          << "that combined while and for loops. File a bug";
    }

    bool emit_as_for_loop = loop_type == LoopView::For;
    // 确定循环类型是否为 for 循环

    assignValuesToTheirUniqueNames(stmt.carriedOutputs());
    // 为循环传递的输出值分配唯一名称
    // 为循环传递的输入值添加别名
    zipWith(
        stmt.bodyCarriedInputs(), // 从1开始忽略循环次数
        stmt.carriedOutputs(),
        [&](Value* block_input, Value* node_output) {
          assignValue(block_input, node_output);
        });
    // 使用 zipWith 函数将循环体的输入值与输出值逐一配对并分配

    // 打印循环节点输出值初始赋值 = 循环节点输入值
    printAnnotatedAssignment(stmt.carriedOutputs(), stmt.carriedInputs());

    assignValuesToTheirUniqueNames(stmt.currentTripCount());
    // 为当前循环次数分配唯一名称
    // 循环头部
    if (emit_as_for_loop) {
      indent();
      body_ << "for " << useOf(stmt.currentTripCount()) << " in range("
            << useOf(stmt.maxTripCount()) << "):\n";
      // 打印 for 循环头部
    } else {
      // 注意：由于这是一个 while 循环，trip_count_in_block 没有使用，因为我们将 Value* 重新用作循环条件的替代品
      printAssignment(stmt.currentTripCount(), stmt.inputCond());
      // 打印 while 循环的条件语句
      indent();
      body_ << "while " << useOf(stmt.currentTripCount()) << ":\n";
      // 打印 while 循环头部
    }
    // 循环体
    // 打印循环体内容
    {
      ResourceGuard indent = WithIndented();
      // 更新块的输出为下一个循环迭代的输入
      // 在 for 循环中跳过对新条件的赋值，因为条件始终为 True
      size_t offset = emit_as_for_loop ? 1 : 0;
      // 获取循环体的块对象
      auto body_block = stmt.bodyBlock();
      // 获取循环体传递的块输入（跳过偏移量）
      ArrayRef<Value*> loop_carried_block_inputs =
          body_block->inputs().slice(offset);
      // 打印循环体的块内容，如果有输入则打印
      printBlock(body_block, !loop_carried_block_inputs.empty());
      // 打印赋值语句，将块输出赋值给输入
      printAssignment(
          loop_carried_block_inputs, body_block->outputs().slice(offset));
    }
    
    bool isLongLine(const std::string& str) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      // 判断字符串长度加上级别乘以 2 是否大于等于 40
      return str.size() + level * 2 >= 40;
    }
    
    bool isLongInline(Node* node) {
      // 检查节点是否在内联输出中，且使用的行是否过长
      return output_inline_.count(node) &&
          isLongLine(useOf(node->output())->str());
    }
    
    bool isNonConstantInline(Value* input) {
      // 检查输入是否为非常量并且在输出内联中
      return input->node()->kind() != prim::Constant &&
          output_inline_.count(input->node());
    }
    
    // [inlines 重排]
    // 我们内联所有语义上合法的内容，但有时这些行会变得太长。在这种情况下，我们需要断开行，
    // 并确保在长输入之前取消所有先前的内联：
    //   r = foo(x.add_(b), some_long + expression)
    //  错误！
    //   _0 = some_long + expression
    //   r = foo(x.add_(b), _0) # 错误！_0 在修改 add_ 之前运行
    // 合法！
    //   _0 = x.add_(b)
    //   _1 = some_long + expression
    //   r = foo(_0, _1)
    void splitLongInlines(Value* v) {
      std::vector<Value*> to_split_reversed;
      Use u = v->uses().at(0);
      // 扫描长内联内容并记录需要分割的值
      scanLongInlines(u.user, u.offset, to_split_reversed);
      for (auto it = to_split_reversed.rbegin(), end = to_split_reversed.rend();
           it != end;
           ++it) {
        // 打印输出定义
        printOutputDefinition((*it)->node(), *useOf(*it));
      }
    }
    
    void scanLongInlines(
        Node* user,
        int64_t offset,
        std::vector<Value*>& to_split_reversed) {
      auto it = visited_split_inline_uses_.find(user);
      bool present = it != visited_split_inline_uses_.end();
      // 逆向扫描内联使用的节点，并记录需要分割的值
      for (int64_t i = offset; i >= (present ? it->second + 1 : 0); --i) {
        Value* prev_arg = user->input(i);
        if (isNonConstantInline(prev_arg)) {
          to_split_reversed.push_back(prev_arg);
        }
      }
      visited_split_inline_uses_[user] = offset;
      if (!present && output_inline_.count(user)) {
        Use u = user->output()->uses().at(0);
        // 由于实际使用仍在进行中，因此扫描内联时将偏移量减一
        scanLongInlines(u.user, int64_t(u.offset) - 1, to_split_reversed);
      }
    }
    
    template <typename T>
    void printOutputDefinition(Node* node, const T& expr) {
      assignValuesToTheirUniqueNames(node->outputs());
      indent();
      // 打印输出
      if (!node->outputs().empty()) {
        printValueList(body_, node->outputs());
        body_ << " = ";
      }
    body_ << expr << "\n";
  }

  // 递归检查包含类型，查找任何类依赖关系
  void registerClassDependencies(const TypePtr& type) {
    // 如果类型是类类型，则将其添加到依赖表中
    if (const auto classType = type->cast<ClassType>()) {
      deps_table_.add(classType);
    } 
    // 如果类型是元组类型且具有名称，则将其添加到依赖表中
    else if (const auto tupleType = type->cast<TupleType>()) {
      if (tupleType->name()) {
        deps_table_.add(tupleType);
      }
    } 
    // 如果类型是接口类型，则将其添加到依赖表中
    else if (const auto interfaceType = type->cast<InterfaceType>()) {
      deps_table_.add(interfaceType);
    } 
    // 如果类型是枚举类型，则将其添加到依赖表中
    else if (const auto enumType = type->cast<EnumType>()) {
      deps_table_.add(enumType);
    }
    // 递归处理包含的每个类型
    for (const auto& containedType : type->containedTypes()) {
      registerClassDependencies(containedType);
    }
  }

  // 检查节点的类型依赖
  void scanTypeDependencies(Node* node) {
    // 检查输入节点的类依赖
    for (const auto input : node->inputs()) {
      registerClassDependencies(input->type());
    }
    // 检查输出节点的类依赖
    for (const auto output : node->outputs()) {
      registerClassDependencies(output->type());
    }
    // 检查节点的属性名称和类型
    for (const auto& name : node->attributeNames()) {
      switch (node->kindOf(name)) {
        // 如果属性类型是单一类型，则检查其类依赖
        case AttributeKind::ty:
          registerClassDependencies(node->ty(name));
          break;
        // 如果属性类型是多个类型，则逐个检查其类依赖
        case AttributeKind::tys:
          for (const TypePtr& t : node->tys(name)) {
            registerClassDependencies(t);
          }
          break;
        default:
          // 对于其他属性类型，不执行任何操作
          // （即不涉及类依赖）
          break;
      }
    }
  }

  // 检查节点的版本信息
  void checkVersion(Node* node) {
    // 获取节点的模式（如果存在）
    if (auto schema = node->maybeSchema()) {
      auto schema_name = getFullSchemaName(*schema);
      // 查找节点模式在操作符版本映射表中的条目
      auto version_entry = get_operator_version_map().find(schema_name);
      // 如果找到了对应的版本条目
      if (version_entry != get_operator_version_map().end()) {
        const auto& entry = version_entry->second;
        // 获取最新的版本号
        uint64_t current_version = entry[entry.size() - 1].bumped_at_version;
        // 获取节点种类对应的最低版本号
        uint64_t legacy_version_map_version =
            get_min_version_for_kind(node->kind());

        // 如果设置了版本计算器标志
        if (get_version_calculator_flag()) {
          // 选择最大的版本号作为节点的最小版本号
          min_version_ = std::max(min_version_, current_version);
        } else {
          // 否则，根据情况选择适当的版本号作为节点的最小版本号
          if (legacy_version_map_version != 0) {
            min_version_ = std::max(min_version_, legacy_version_map_version);
          } else {
            min_version_ = std::max(min_version_, current_version);
          }
        }
      }
    }
  }

  // 打印节点的信息
  void printNode(Node* node, bool print_const) {
    // 保留节点的源码范围
    WithSourceRange guard(&source_range_stack_, node);
    // 扫描节点的类型依赖
    scanTypeDependencies(node);
    // 检查节点的版本信息
    checkVersion(node);
    // 如果不打印常量且节点是常量类型，则直接返回
    if (!print_const && node->kind() == prim::Constant)
      return;
    // （未完整的代码块，缺少了一些可能的功能）
  }

  // 静态函数：检查给定的 IValue 是否包含非 ASCII 字符串
  static bool containsNonASCIIString(const IValue& val) {
    bool hasNonASCII = false;
    // （未完整的代码块，缺少了一些可能的功能）
    // 定义一个lambda函数checkSubvalue，用于检查IValue对象是否包含非ASCII字符
    auto checkSubvalue = [&hasNonASCII](const IValue& val) {
      // 如果val是字符串类型
      if (val.isString()) {
        // 遍历字符串中的每个字符，以signed char类型遍历
        for (signed char c : val.toStringRef()) {
          // 如果字符小于0，则说明是非ASCII字符
          if (c < 0) {
            hasNonASCII = true;
            return true;
          }
        }
      }
      // 返回是否存在非ASCII字符的标志
      return false;
    };

    // 使用val的visit方法调用checkSubvalue lambda函数，检查是否存在非ASCII字符
    val.visit(checkSubvalue);
    // 返回hasNonASCII标志，指示是否存在非ASCII字符
    return hasNonASCII;
  }

  // 打印常量值的字符串表示到stmt流中
  void printConstant(TaggedStringStream& stmt, const IValue& v) {
    // 定义一个lambda函数customFormatter，用于自定义格式化输出IValue
    const auto customFormatter = [&](std::ostream& ss, const IValue& v) {
      // 如果v是Tensor类型或者包含非ASCII字符串或者是对象类型
      if (v.isTensor() || containsNonASCIIString(v) || v.isObject()) {
        // 断言v不是c10::Type类型的模块
        TORCH_INTERNAL_ASSERT(!v.type<c10::Type>()->is_module());
        // 输出常量名"CONSTANTS.c"后跟常量在系统中的索引
        ss << "CONSTANTS.c" << getOrAddConstant(v);
        return true;
      }

      // 获取v的类型
      auto type = v.type();
      // 如果是动态类型，则获取其fallback类型
      if (auto dyn = type->castRaw<c10::DynamicType>()) {
        type = dyn->fallback();
      }
      // 如果v是元组类型并且具有schema信息
      if (v.isTuple() && type->expectRef<TupleType>().schema()) {
        // 输出命名元组的构造函数，然后继续打印元组的其余部分
        ss << type->expectRef<TupleType>().annotation_str(type_printer_);
      }
      return false;
    };

    // 创建一个stringstream对象ss，将v的字符串表示形式按照customFormatter格式化后写入ss
    std::stringstream ss;
    v.repr(ss, customFormatter);
    // 将stringstream对象ss的内容追加到stmt流中
    stmt << ss.str();
  }

  // 打印操作符名称到stmt流中
  void printOpName(TaggedStringStream& stmt, Symbol kind) {
    // 定义一个静态的unordered_map override_symbols，用于重写特定操作符的序列化输出
    const static std::unordered_map<Symbol, std::string> override_symbols = {
        {aten::backward, "torch.autograd.backward"},
        {aten::grad, "torch.autograd.grad"},
    };
    // 如果kind在override_symbols中
    if (override_symbols.find(kind) != override_symbols.end()) {
      // 将override_symbols中kind对应的值追加到stmt流中
      stmt << override_symbols.at(kind);
    } else if (kind.is_aten()) {
      // 对于aten命名空间下的操作符，以"torch."开头输出其名称
      stmt << "torch." << kind.toUnqualString();
    } else {
      // 对于其他命名空间下的操作符，按照"ops.<namespace>.<op_name>"的格式输出
      stmt << "ops." << kind.ns().toUnqualString() << "."
           << kind.toUnqualString();
    }
  }

  // 打印节点的右侧值的字符串表示形式到stmt流中
  void printRHS(TaggedStringStream& stmt, Node* node) {
    // 空函数，未实现具体的打印逻辑
    }
  }

  // 打印代码块的内容到TaggedStringStream流中，处理是否有其他语句的标志
  TaggedStringStream& printBlock(Block* root, bool block_has_other_statements) {
    // 对Python中的空'pass'语法进行检查，以确定代码块是否为空
    // 但并非所有块中的内容都是语句
    // 此处将实现对代码块内容的打印
    // 如果当前代码块没有其他语句，并且根节点的子节点列表为空，则执行下面的代码块
    if (!block_has_other_statements &&
        root->nodes().begin() == root->nodes().end()) {
      // 缩进，生成Python中的pass语句
      indent();
      // 将"pass"语句追加到函数体字符串流中
      body_ << "pass\n";
    }
    // 遍历根节点的所有子节点
    for (auto* node : root->nodes()) {
      // 打印节点内容到函数体字符串流中，不打印常量
      printNode(node, /*print_const=*/false);
    }
    // 返回生成的函数体字符串流
    return body_;
  }

  template <typename dtype>
  IValue createBroadList(dtype value, const int64_t& N) {
    // 创建一个C++的List容器，用于存放dtype类型的元素
    c10::List<dtype> repeated;
    // 预留N个空间，以提高插入效率
    repeated.reserve(N);
    // 循环N次，向List容器中添加value元素
    for (const auto i : c10::irange(N)) {
      // 使用(void)i来抑制未使用变量的警告
      (void)i; // Suppress unused variable warning
      // 向List容器中添加value元素
      repeated.push_back(value);
    }
    // 返回填充后的List容器
    return repeated;
  }

  void printDefaultValue(
      const Argument& arg,
      TaggedStringStream& stmt,
      const IValue& value) {
    // 向字符串流中添加等号字符"="
    stmt << "=";
    // 处理广播列表（broadcasting lists）
    if (arg.type()->kind() == ListType::Kind &&
        (value.isInt() || value.isDouble() || value.isBool())) {
      // 断言arg参数的广播列表（broadcasting list）的大小不为空
      TORCH_INTERNAL_ASSERT(arg.N(), "expected broadcastinglist");
      // 根据value的类型，分别创建对应类型的广播列表，并打印到字符串流中
      if (value.isInt()) {
        printConstant(stmt, createBroadList<int64_t>(value.toInt(), *arg.N()));
      } else if (value.isBool()) {
        printConstant(stmt, createBroadList<bool>(value.toBool(), *arg.N()));
      } else if (value.isDouble()) {
        printConstant(
            stmt, createBroadList<double>(value.toDouble(), *arg.N()));
      }
    } else {
      // 否则，直接打印value的常量值到字符串流中
      printConstant(stmt, value);
    }
  }

  void printBody(Block* body) {
    // 存储当前函数体中的常量节点
    std::vector<Node*> constants;
    // 构建常量节点列表
    buildConstantList(body, constants);

    // 扫描函数体内的块，用于处理局部名称冲突
    scanBlock(body);
    {
      auto guard = WithIndented();
      // 打印初始常量表（大多数常量直接内联到使用它们的地方，但是有些像长字符串会被单独输出）
      for (Node* n : constants) {
        // 打印常量节点到函数体字符串流中，打印常量
        printNode(n, /*print_const=*/true);
      }
      // 打印函数体块内容
      printBlock(body, !body->return_node()->inputs().empty());
      // 打印函数的返回节点到函数体字符串流中，不打印常量
      printNode(body->return_node(), /*print_const=*/false);
    }
  }

 public:
  void printFunction(
      const Function& func,
      bool print_first_argument_type = true) {
    // 获取函数的schema
    const FunctionSchema& schema = func.getSchema();
    // 获取函数的计算图
    Graph& graph = *toGraphFunction(func).graph();
    // 清空已使用名称的集合，每个图可以重复使用本地名称
    used_names_.clear();

    // 设置源码范围，用于生成代码的位置标记
    WithSourceRange guard(&source_range_stack_, graph.param_node());

    // 缩进，开始生成Python函数定义
    indent();
    // 将函数名和参数列表追加到函数体字符串流中
    body_ << "def " << func.name() << "(";
    // 迭代函数的输入参数列表
    auto param_it = graph.inputs().begin();
    for (const Argument& arg : schema.arguments()) {
      // 遍历模式(schema)中的每个参数Argument对象
      registerClassDependencies(arg.type());
      // 注册参数类型的类依赖关系

      std::string arg_name = genName(arg.name());
      // 生成参数名的字符串表示

      if (param_it == graph.inputs().begin()) {
        // 如果是参数列表中的第一个参数
        // 第一个参数在上下文中可以省略类型，根据print_first_argument_type标志来确定
        body_ << arg_name;
        if (print_first_argument_type) {
          // 如果需要打印第一个参数的类型
          body_ << ": " << arg.type()->annotation_str(type_printer_);
          annotated_unions_.insert(*param_it);
        }
      } else {
        // 如果不是第一个参数，需要换行并缩进
        body_ << ",\n    " << arg_name << ": "
              << arg.type()->annotation_str(type_printer_);
        annotated_unions_.insert(*param_it);
      }

      if (arg.default_value()) {
        // 如果参数有默认值，则打印默认值
        printDefaultValue(arg, body_, *arg.default_value());
      }

      assignValue(*param_it++, arg_name);
      // 将参数值赋给param_it指向的位置，并递增param_it
    }

    const auto& returnType = schema.returns().at(0).type();
    // 获取返回类型
    body_ << ") -> " << returnType->annotation_str(type_printer_) << ":\n";
    // 打印函数的返回类型
    registerClassDependencies(returnType);
    // 注册返回类型的类依赖关系

    printBody(graph.block());
    // 打印函数体内容
  }

  void printMethod(const Function& func) {
    // 打印函数的方法
    printFunction(func, /*print_first_argument_type=*/false);
    // 调用printFunction打印函数，不打印第一个参数的类型
  }

  PythonPrintImpl(
      std::vector<at::IValue>& constant_table,
      PrintDepsTable& deps_table,
      c10::TypePrinter type_printer,
      bool enforce_importable)
      : body_(&source_range_stack_),
        constant_table_(constant_table),
        deps_table_(deps_table),
        type_printer_(std::move(type_printer)),
        enforce_importable_(enforce_importable) {
    // PythonPrintImpl类的构造函数，初始化成员变量
  }

  void printClass(const ClassTypePtr& classType) {
    // 打印类的方法
    for (auto& method : classType->methods()) {
      // 遍历类的每个方法
      if (!method->isGraphFunction()) {
        // 如果方法不是图函数，表示这个类是一个定制绑定的C++类，不进行序列化
        return;
      }
    }

    bool is_module = classType->is_module();
    // 判断类是否是模块类
    body_ << "class " << classType->name()->name();
    // 打印类名

    if (is_module) {
      body_ << "(Module)";
      // 如果是模块类，打印为继承自Module
    }

    body_ << ":\n";
    // 打印类定义的开始
  }

  void printNamedType(const c10::NamedTypePtr& type) {
    // 打印命名类型的方法
    if (auto functionType = type->cast<FunctionType>()) {
      // 如果类型可以转换为函数类型
      printFunction(*functionType->function());
      // 调用printFunction打印函数
    } else if (auto classType = type->cast<ClassType>()) {
      // 如果类型可以转换为类类型
      printClass(classType);
      // 调用printClass打印类
    } else if (auto tupleType = type->cast<TupleType>()) {
      // 如果类型可以转换为元组类型
      TORCH_INTERNAL_ASSERT(tupleType->schema());
      body_ << "class " << tupleType->name()->name();
      // 打印元组类名

      body_ << "(NamedTuple):\n";
      // 打印继承自NamedTuple的定义开始
      {
        const auto guard = WithIndented();
        // 使用WithIndented类保证缩进
        for (const auto& attr : tupleType->schema()->arguments()) {
          // 遍历元组中的每个属性
          TORCH_INTERNAL_ASSERT(attr.type());
          // 断言属性类型存在
          indent();
          // 缩进
          body_ << attr.name() << " : "
                << attr.type()->annotation_str(type_printer_) << "\n";
          // 打印属性名和类型注解
        }
      }
    }
  }
    } else if (auto interfaceType = type->cast<InterfaceType>()) {
      // 检查是否可以将类型转换为接口类型，并声明一个自动变量interfaceType
      body_ << "class " << interfaceType->name()->name();
      // 在body_中添加接口类型的类定义，包括类名
      if (interfaceType->is_module()) {
        // 如果接口类型是模块接口，添加继承自ModuleInterface的声明
        body_ << "(ModuleInterface):\n";
      } else {
        // 否则，添加继承自Interface的声明
        body_ << "(Interface):\n";
      }
      {
        // 进入一个作用域，使用WithIndented确保每行都正确缩进
        auto guard = WithIndented();
        // 遍历接口类型的所有方法
        for (const FunctionSchema& method : interfaceType->methods()) {
          indent();
          // 添加方法定义，包括方法名和第一个参数self
          body_ << "def " << method.name() << "(self";
          // 断言方法参数非空且第一个参数为self
          TORCH_INTERNAL_ASSERT(
              !method.arguments().empty() &&
              method.arguments().at(0).name() == "self");
          // 遍历方法的其他参数
          for (const Argument& arg :
               at::ArrayRef<Argument>(method.arguments()).slice(1)) {
            const auto& arg_type = arg.type();
            // 注册类依赖关系，可能是引用其他类型
            registerClassDependencies(arg_type);
            // 添加方法参数的声明，包括参数名和类型的注解字符串
            body_ << ", " << arg.name() << ": "
                  << arg_type->annotation_str(type_printer_);
          }
          // 添加方法的返回类型声明
          auto return_type = method.returns().at(0).type();
          registerClassDependencies(return_type);
          body_ << ") -> " << return_type->annotation_str(type_printer_)
                << ":\n";
          indent();
          // 添加方法的占位符实现
          body_ << "  pass\n";
        }
      }
    } else if (auto enumType = type->cast<EnumType>()) {
      // 检查是否可以将类型转换为枚举类型，并声明一个自动变量enumType
      body_ << "class " << enumType->qualifiedClassName().name() << "(Enum):\n";

      std::string value_wrapper = "";
      // 根据枚举值类型确定值的包装方式
      if (enumType->getValueType() == StringType::get()) {
        value_wrapper = "\"";
      }

      {
        // 进入一个作用域，使用WithIndented确保每行都正确缩进
        auto guard = WithIndented();
        // 遍历枚举类型的所有枚举名称和对应值
        for (const auto& name_value : enumType->enumNamesValues()) {
          indent();
          // 添加枚举名称和对应值的声明
          body_ << name_value.first << " = " << value_wrapper
                << name_value.second << value_wrapper << "\n";
        }
      }
    } else {
      // 如果不是接口类型也不是枚举类型，抛出未处理的命名类型错误
      TORCH_INTERNAL_ASSERT(false, "Unhandled NamedType");
  }
}

~PythonPrintImpl() = default;

TaggedStringStream body_;
// 当打印这个节点时，是否可以安全地将其内联输出（即不需要分配临时变量）
std::unordered_set<Node*> output_inline_;

// 见 [reordering of inlines]
// 用于跟踪内联语句的部分，我们已经扫描过以分割长行，以避免重新访问它们，导致 n^2 的行为。
// 存储了已经扫描过的节点中最大的输入偏移量。
std::unordered_map<Node*, int64_t> visited_split_inline_uses_;

// 当前函数中正在使用的有效标识符集合
std::unordered_set<std::string> used_names_;

// 常量写入此表中，并按照 CONSTANTS.cN 命名，其中 N 是此表中的索引。
std::vector<at::IValue>& constant_table_;

// 所有已使用的 NamedTypes（类、函数、NamedTuples）写入此表。
PrintDepsTable& deps_table_;

// 我们需要保留 Union/Optional 类型注释，但应该仅在变量声明时打印注释（不在后续用法上）。
// 此集合跟踪我们已经用注释打印过的 Value*s。
std::unordered_set<Value*> annotated_unions_;

// 给定一个命名类型，返回正确的字符串以打印它的函数。
c10::TypePrinter type_printer_;

// 当我们打印这个对象时，如果结果输出无法重新解析，是否应报错？
bool enforce_importable_;

// 支持所有打印操作的最低版本
uint64_t min_version_ = caffe2::serialize::kMinSupportedFileFormatVersion;
};

PythonPrint::PythonPrint(
    std::vector<at::IValue>& constant_table,
    PrintDepsTable& deps_table,
    c10::TypePrinter type_printer,
    bool enforce_importable)
    : pImpl(std::make_shared<PythonPrintImpl>(
          constant_table,
          deps_table,
          std::move(type_printer),
          enforce_importable)) {}
// PythonPrint 类的构造函数，初始化 PythonPrintImpl 对象 pImpl

void PythonPrint::printNamedType(const c10::NamedTypePtr& type) {
  pImpl->printNamedType(type);
}
// 调用 pImpl 对象的 printNamedType 方法，打印给定命名类型的信息

void PythonPrint::printFunction(const Function& func) {
  pImpl->printFunction(func);
}
// 调用 pImpl 对象的 printFunction 方法，打印给定函数的信息

void PythonPrint::printMethod(const Function& func) {
  pImpl->printMethod(func);
}
// 调用 pImpl 对象的 printMethod 方法，打印给定方法的信息

std::string PythonPrint::str() const {
  return pImpl->body_.str();
}
// 返回 pImpl 对象的 body_ 成员的字符串表示

const SourceRangeRecords& PythonPrint::ranges() const {
  return pImpl->body_.ranges();
}
// 返回 pImpl 对象的 body_ 成员的源代码范围记录

uint64_t PythonPrint::minVersion() const {
  return pImpl->min_version_;
}
// 返回 pImpl 对象的 min_version_ 成员，表示最小版本号

static std::vector<IValue> traverseIValueAndGetObjects(const IValue& ivalue) {
  std::vector<IValue> result;
  std::vector<IValue> stack;
  stack.emplace_back(ivalue);
  while (!stack.empty()) {
    IValue head = stack.back();
    stack.pop_back();
    if (head.isObject()) {
      result.push_back(head);
      auto obj = head.toObject();
      ClassTypePtr type = obj->type();
      if (type->hasMethod("__getstate__")) {
        Function& getstate = type->getMethod("__getstate__");
        stack.emplace_back(getstate({obj}));
      } else {
        for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
          stack.emplace_back(obj->getSlot(i));
        }
      }
    } else if (ivalue.isGenericDict()) {
      for (const auto& kv : ivalue.toGenericDict()) {
        // skip key because key cannot be an object
        stack.emplace_back(kv.value());
      }
    } else if (ivalue.isList()) {
      for (const auto& v : ivalue.toList()) {
        stack.emplace_back(v);
      }
    } else if (ivalue.isTuple()) {
      for (const auto& v : ivalue.toTuple()->elements()) {
        stack.emplace_back(v);
      }
    }
  }
  return result;
}
// 递归遍历 IValue 并收集对象类型的实例，返回结果向量

static std::optional<std::string> printType(
    const c10::Type& type,
    torch::jit::TypeNameUniquer& type_name_uniquer) {
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str(
        [&](auto&& t) { return printType(t, type_name_uniquer); });
  }
  auto namedType = type.cast<c10::NamedType>();
  if (namedType && namedType->name()) {
    return type_name_uniquer.getUniqueName(namedType).qualifiedName();
  }
  return c10::nullopt;
}
// 打印给定类型的字符串表示，返回可选的字符串，使用类型名唯一化器以确保名称唯一

void jitModuleToPythonCodeAndConstants(
    const Module& module,
    ExtraFilesMap* jit_sources, // output
    std::vector<IValue>* constants // output
) {
  // 使用 traverseIValueAndGetObjects 函数遍历 module 的 IValue 表示，获取其中的对象列表
  std::vector<IValue> objects = traverseIValueAndGetObjects(module._ivalue());
  // 创建一个无序集合 visited，用于记录已访问的 c10::QualifiedName
  std::unordered_set<c10::QualifiedName> visited;
  // 创建 PrintDepsTable 类的对象 class_deps，用于打印依赖关系表
  PrintDepsTable class_deps;
  // 创建 TypeNameUniquer 类的对象 uniquer，用于确保类型名的唯一性
  TypeNameUniquer uniquer;
  // 定义 lambda 函数 type_printer，用于打印 c10::Type 的类型信息，并传入 uniquer 以确保类型名的唯一性
  auto type_printer = [&](const c10::Type& t) { return printType(t, uniquer); };

  // 根据前缀进行分组，每个前缀代表一个文件
  std::unordered_map<std::string, PythonPrint> grouped_by_prefix;
  // 遍历 objects 中的每个 IValue 对象
  for (const IValue& obj : objects) {
    // 将 IValue 对象转换为 ObjectPtr 智能指针
    ObjectPtr obj_ptr = obj.toObject();
    // 获取对象的 ClassTypePtr 类型
    ClassTypePtr class_type = obj_ptr->type();
    // 将 class_type 添加到 class_deps 中，记录类的依赖关系
    class_deps.add(class_type);
  }

  // 遍历 class_deps 中的每个类型
  for (size_t i = 0; i < class_deps.size(); ++i) {
    // 注意: PythonPrint 可能会扩展 class_deps，因此需要重新检查 size()
    auto type = class_deps[i];
    // 获取类型的唯一名称
    auto qualname = uniquer.getUniqueName(type);
    // 获取名称的前缀
    std::string qualifier = qualname.prefix();
    // 查找 grouped_by_prefix 中是否已存在该前缀的条目
    auto pp_iter = grouped_by_prefix.find(qualifier);
    if (pp_iter == grouped_by_prefix.end()) {
      // 如果不存在，则插入新条目，创建 PythonPrint 对象，并传入相关参数
      pp_iter = grouped_by_prefix
                    .emplace(
                        qualifier,
                        PythonPrint(
                            *constants,
                            class_deps,
                            type_printer,
                            /*enforce_importable=*/true))
                    .first;
    }
    // 打印该类型的命名类型
    pp_iter->second.printNamedType(type);
  }

  // 将 grouped_by_prefix 中的结果放入 jit_sources 中
  for (const auto& kv : grouped_by_prefix) {
    (*jit_sources)[kv.first] = kv.second.str();
  }
}

} // namespace torch::jit
```