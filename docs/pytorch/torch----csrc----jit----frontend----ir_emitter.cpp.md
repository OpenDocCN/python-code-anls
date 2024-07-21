# `.\pytorch\torch\csrc\jit\frontend\ir_emitter.cpp`

```
// 包含 Torch JIT 前端的 IR 生成器头文件
#include <torch/csrc/jit/frontend/ir_emitter.h>
// 包含 Torch JIT 前端的树视图头文件
#include <torch/csrc/jit/frontend/tree_views.h>

// 包含 C10 库的异常处理工具头文件
#include <c10/util/Exception.h>
// 包含 C10 库的字符串工具头文件
#include <c10/util/StringUtil.h>
// 包含 C10 库的整数范围迭代器头文件
#include <c10/util/irange.h>
// 包含 Caffe2 序列化版本头文件
#include <caffe2/serialize/versions.h>
// 包含 Torch JIT API 的函数实现头文件
#include <torch/csrc/jit/api/function_impl.h>
// 包含 Torch JIT 前端的规范化修改循环头文件
#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
// 包含 Torch JIT 前端的转换为 SSA 形式头文件
#include <torch/csrc/jit/frontend/convert_to_ssa.h>
// 包含 Torch JIT 前端的词法分析器头文件
#include <torch/csrc/jit/frontend/lexer.h>
// 包含 Torch JIT 前端的解析器头文件
#include <torch/csrc/jit/frontend/parser.h>
// 包含 Torch JIT 前端的模式匹配头文件
#include <torch/csrc/jit/frontend/schema_matching.h>
// 包含 Torch JIT 前端的脚本类型解析器头文件
#include <torch/csrc/jit/frontend/script_type_parser.h>
// 包含 Torch JIT IR 的头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 通过警告注释的传递过程头文件
#include <torch/csrc/jit/passes/annotate_warns.h>
// 包含 Torch JIT 的规范化传递过程头文件
#include <torch/csrc/jit/passes/canonicalize.h>
// 包含 Torch JIT 的常量池化传递过程头文件
#include <torch/csrc/jit/passes/constant_pooling.h>
// 包含 Torch JIT 的常量传播传递过程头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
// 包含 Torch JIT 的死代码消除传递过程头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含 Torch JIT 的内联分支闭包传递过程头文件
#include <torch/csrc/jit/passes/inline_forked_closures.h>
// 包含 Torch JIT 的内联传递过程头文件
#include <torch/csrc/jit/passes/inliner.h>
// 包含 Torch JIT 的闭包提升传递过程头文件
#include <torch/csrc/jit/passes/lift_closures.h>
// 包含 Torch JIT 的元组降低传递过程头文件
#include <torch/csrc/jit/passes/lower_tuples.h>
// 包含 Torch JIT 的操作规范化传递过程头文件
#include <torch/csrc/jit/passes/normalize_ops.h>
// 包含 Torch JIT 的旧操作符替换传递过程头文件
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
// 包含 Torch JIT 运行时图迭代器头文件
#include <torch/csrc/jit/runtime/graph_iterator.h>
// 包含 Torch JIT 运行时解释器头文件
#include <torch/csrc/jit/runtime/interpreter.h>
// 包含 Torch JIT 运行时操作符头文件
#include <torch/csrc/jit/runtime/operator.h>
// 包含 Torch JIT 运行时切片索引调整头文件
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
// 包含 Torch JIT 用于测试的钩子头文件
#include <torch/csrc/jit/testing/hooks_for_testing.h>

// 包含 Torch JIT IR 的常量定义头文件
#include <torch/csrc/jit/ir/constants.h>

// 包含 C10 库的可选类型头文件
#include <c10/util/Optional.h>
// 包含 C10 库的哈希计算工具头文件
#include <c10/util/hash.h>

// 包含 ATen 核心库的内部化字符串头文件
#include <ATen/core/interned_strings.h>
// 包含 ATen 核心库的 JIT 类型头文件
#include <ATen/core/jit_type.h>
// 包含 Torch JIT 前端的错误报告头文件
#include <torch/csrc/jit/frontend/error_report.h>

// 包含标准库的原子操作头文件
#include <atomic>
// 包含标准库的整数极限头文件
#include <climits>
// 包含标准库的集合头文件
#include <set>
// 包含标准库的栈头文件
#include <stack>

// 定义匿名命名空间，用于实现本地函数和变量的私有性
namespace {
// 根据文件大小报告源位置是否启用的函数
bool reportSourceLocation(size_t file_size) {
  // 如果文件大小小于 512KB，则启用源位置报告
  if (file_size < 512 * 1024) {
    return true;
  }
  // 否则，根据环境变量 PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION 来决定是否启用
  const char* enable_env = std::getenv("PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION");
  bool flag = true;
  // 环境变量未设置或者设置为 "0", "FALSE", "false" 时，禁用源位置报告
  if (enable_env == nullptr || std::strcmp(enable_env, "0") == 0 ||
      std::strcmp(enable_env, "FALSE") == 0 ||
      std::strcmp(enable_env, "false") == 0) {
    flag = false;
  }
  return flag;
}
} // namespace

// Torch JIT 命名空间
namespace torch::jit {

// 使用字符串键和函数引用值的哈希映射类型定义
using FunctionTable = std::unordered_map<std::string, Function&>;
// 使用字符串键和 SugerredValuePtr 值的哈希映射类型定义
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
// 使用字符串键和 TypePtr 值的哈希映射类型定义
using TypeTable = std::unordered_map<std::string, TypePtr>;
// 使用字符串键和 Const 值的哈希映射类型定义
using AttributeMap = std::unordered_map<std::string, Const>;
// 使用字符串键和 vector<Const> 值的哈希映射类型定义
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

// 表示细化信息的结构体，包含标识符和类型指针
struct Refinement {
  // 构造函数，初始化标识符和类型
  Refinement(std::string identifier, TypePtr type)
      : identifier_(std::move(identifier)), type_(std::move(type)) {}
  // 返回细化信息的标识符
  const std::string& identifier() const {
    return identifier_;
  }
  // 返回细化信息的类型指针
  TypePtr type() const {
    return type_;
  }

 private:
  std::string identifier_; // 细化信息的标识符
  TypePtr type_;           // 细化信息的类型指针
};

} // namespace torch::jit
struct RefinementSet {
  // 当像 x is None 这样的比较被执行时，我们将类型细化与其真值和假值关联。
  // 如果一个布尔值有关联的细化，在 if 语句的条件中使用时，真值和假值的细化被插入对应的块中。

  // 定义一个类型别名，用于存储细化信息的向量
  using Refinements = std::vector<Refinement>;

  // 构造函数，接受真值和假值的细化作为参数
  RefinementSet(Refinements true_refinements, Refinements false_refinements)
      : true_refinements_(std::move(true_refinements)),
        false_refinements_(std::move(false_refinements)) {}

  // 构造函数，接受单个细化作为参数，并将其放入真值细化中
  RefinementSet(Refinement single) : RefinementSet({std::move(single)}, {}) {}

  // 构造函数，接受单个真值和单个假值细化作为参数
  RefinementSet(Refinement single_true, Refinement single_false)
      : RefinementSet(
            Refinements({std::move(single_true)}),
            Refinements({std::move(single_false)})) {}

  // 默认构造函数，创建一个空的细化集合
  RefinementSet() = default; // empty

  // AND 操作，返回与另一个细化集合的交集和并集
  RefinementSet And(const RefinementSet& rhs) const {
    return RefinementSet(
        unionSet(true_refinements_, rhs.true_refinements_),
        intersectSet(false_refinements_, rhs.false_refinements_));
  }

  // OR 操作，返回与另一个细化集合的交集和并集
  RefinementSet Or(const RefinementSet& rhs) const {
    return RefinementSet(
        intersectSet(true_refinements_, rhs.true_refinements_),
        unionSet(false_refinements_, rhs.false_refinements_));
  }

  // NOT 操作，返回真值和假值细化的交换
  RefinementSet Not() const {
    return RefinementSet(false_refinements_, true_refinements_);
  }

  // 返回当前真值细化集合
  const std::vector<Refinement> activeRefinements() const {
    return true_refinements_;
  }

 private:
  // 静态方法，用于判断两个细化是否是同一个变量
  static bool sameVar(const Refinement& a, const Refinement& b) {
    return a.identifier() == b.identifier();
  }

  // 静态方法，返回两个细化集合的并集
  static Refinements unionSet(const Refinements& a, const Refinements& b) {
    Refinements result = a;
    for (const Refinement& r : b) {
      auto it =
          std::find_if(result.begin(), result.end(), [&](const Refinement& e) {
            return e.identifier() == r.identifier();
          });
      if (it == result.end()) {
        result.push_back(r);
      } else if (*it->type() != *r.type()) {
        // 仅保留完全匹配的细化类型
        result.erase(it);
      }
    }
    return result;
  }

  // 静态方法，返回两个细化集合的交集
  static Refinements intersectSet(const Refinements& a, const Refinements& b) {
    Refinements result;
    // 省略了交集计算的部分代码
    return result;
  }

  // 成员变量，存储真值和假值的细化集合
  Refinements true_refinements_;
  Refinements false_refinements_;
};
    // 遍历容器 a 中的每个 Refinement 对象 r
    for (const Refinement& r : a) {
      // 在容器 b 中查找与 r 具有相同标识符的 Refinement 对象
      auto it = std::find_if(b.begin(), b.end(), [&](const Refinement& e) {
        return e.identifier() == r.identifier();
      });
      // 如果找到匹配的 Refinement 对象并且类型也相同，则将 r 添加到结果容器 result 中
      if (it != b.end() && r.type() == it->type()) {
        result.push_back(r);
      }
    }
    // 返回存储匹配结果的容器 result
    return result;
  }

  // 声明存储真实和假设 Refinement 对象的容器 true_refinements_ 和 false_refinements_
  Refinements true_refinements_;
  Refinements false_refinements_;
};

// CondValue 结构体，用于表示条件值
struct CondValue {
  // 构造函数，接受一个值指针、细化集合和静态条件的可选值
  CondValue(
      Value* value,
      RefinementSet refinements,
      std::optional<bool> static_if)
      : value_(value),
        refinements_(std::move(refinements)),
        static_if_(static_if) {}
  
  // 构造函数，接受图、源范围、静态值和细化集合
  CondValue(
      Graph& g,
      const SourceRange& loc,
      bool static_value,
      RefinementSet refinements)
      : value_(g.insertConstant(static_value, loc)),
        refinements_(std::move(refinements)),
        static_if_(static_value) {}

  // 返回存储的值指针
  Value* value() const {
    return value_;
  }

  // 返回细化集合的常量引用
  const RefinementSet& refinements() const {
    return refinements_;
  }

  // 返回静态条件的可选值
  std::optional<bool> staticIf() const {
    return static_if_;
  }

 private:
  Value* value_;                   // 存储的值指针
  RefinementSet refinements_;      // 细化集合
  std::optional<bool> static_if_;  // 静态条件的可选值，表示是否生成静态 if 语句
                                    // 当表达式触发静态 if 行为时存在
                                    // 不等同于 value_ 是常量，value_ 可能是常量
                                    // 但生成它的表达式未触发静态 if 行为，例如使用赋给常量的变量
};

// NoneStatus 枚举，表示值可以为 None 的状态
enum NoneStatus { ALWAYS, MAYBE, NEVER };

// 返回值 v 是否可以为 None
static NoneStatus canBeNone(Value* v) {
  if (v->node()->mustBeNone()) {  // 如果节点必须为 None
    return ALWAYS;
  }
  if (v->type()->kind() == OptionalType::Kind ||  // 如果类型为可选类型
      (v->type()->kind() == UnionType::Kind &&
       v->type()->expect<UnionType>()->canHoldType(*NoneType::get()))) {  // 或者可以包含 None 类型的联合类型
    return MAYBE;
  }
  return NEVER;
}

// 将 SugaredValuePtr 转换为简单值的函数
static Value* asSimple(const SugaredValuePtr& value) {
  if (SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {  // 如果是 SimpleValue 类型
    return sv->getValue();  // 返回其值
  }
  return nullptr;  // 否则返回空指针
}

// 创建魔术方法的辅助函数
static std::shared_ptr<MagicMethod> makeMagic(
    const std::string& name,
    SugaredValuePtr base) {
  return std::make_shared<MagicMethod>(name, base);  // 返回一个以 name 和 base 为参数的 MagicMethod 共享指针
}

// 环境数据结构，用于将变量绑定解糖为显式作用域的辅助结构
//
// Environment 跟踪两个表，一个用于非第一类值，一个用于第一类值的类型。
// 当第一类值在环境中设置时，我们发出一个 prim::Store 指令，将变量名设置为适当的类型；
// 当引用第一类值时，我们发出一个 prim::Load 指令，生成相应类型的值。
//
// 示例：
// a = 1
// print(a)
// 转换为：
// = prim::Store[name="a"](%a.1)
// %a : int = prim::Load[name="a"]()
// prim::Print(%a)
//
// 从前端解析变量绑定时，我们一直使用显式作用域语言，这些控制结构本身不引入作用域。
// 定义 Environment 结构体，表示程序执行环境
struct Environment {
  // 构造函数，初始化环境
  Environment(
      GraphFunction& method,        // 传入的方法
      ResolverPtr resolver,         // 解析器指针
      Block* b,                     // 当前块
      std::shared_ptr<Environment> next = nullptr)
      : method(method),             // 初始化方法
        resolver(std::move(resolver)),  // 移动赋值解析器
        b(b),                       // 初始化当前块
        next(std::move(next)) {}   // 移动赋值下一个环境指针

  // 引用 GraphFunction 对象，用于表示方法
  GraphFunction& method;
  // 智能指针，指向 Resolver 对象，用于解析
  ResolverPtr resolver;
  // 无序映射，存储错误信息的函数
  std::unordered_map<std::string, std::function<std::string()>> error_messages;
  // 指向当前块的指针
  Block* b;

  // 智能指针，指向下一个 Environment 对象
  std::shared_ptr<Environment> next;

  // 在最底层的环境中设置变量类型错误，并存储相应的错误消息
  void setVariableTypeError(
      const std::string& name,     // 变量名
      std::function<std::string()> msg) {  // 返回错误消息的函数
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();  // 获取链表中的最后一个环境
    }
    runner->error_messages[name] = std::move(msg);  // 存储错误消息
  }

  // 查找指定变量名是否存在类型错误消息
  std::optional<std::string> findVariableTypeError(const std::string& name) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();  // 获取链表中的最后一个环境
    }
    auto msg = runner->error_messages.find(name);  // 查找错误消息
    if (msg != runner->error_messages.end()) {
      return msg->second();  // 返回错误消息
    } else {
      return c10::nullopt;    // 如果不存在则返回空值
    }
  }

  // 向当前块中插入加载操作，并返回对应的 SugaredValuePtr
  SugaredValuePtr insertLoad(const std::string& name, const TypePtr& type) {
    auto g = b->owningGraph();              // 获取当前块所属的图对象
    auto load = g->insertNode(g->createLoad(name, type));  // 插入加载节点
    if (meaningfulName(name)) {
      load->output()->setDebugName(name);  // 设置调试名称
    }
    return std::make_shared<SimpleValue>(load->output());  // 返回加载的值的封装
  }

  // 插入存储操作到当前块中，记录变量的位置信息和类型
  void insertStore(
      const std::string& name,     // 变量名
      const SourceRange& loc,      // 变量的源范围
      Value* v,                    // 值
      TypePtr type) {              // 类型指针
    auto g = b->owningGraph();     // 获取当前块所属的图对象
    g->insertNode(g->createStore(name, v))->setSourceRange(loc);  // 插入存储节点
    type_table[name] = std::move(type);  // 记录类型信息
  }

  // 在当前帧中查找指定名称的值
  SugaredValuePtr findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);  // 查找值表中是否存在该变量名
    if (it != value_table.end()) {
      return it->second;    // 返回对应的值
    }
    auto it2 = type_table.find(name);  // 查找类型表中是否存在该变量名
    if (it2 != type_table.end()) {
      return insertLoad(name, it2->second);  // 插入加载操作并返回值的封装
    }
    return nullptr;   // 如果都不存在则返回空指针
  }

  // 在父级环境中查找指定名称的值
  SugaredValuePtr findInParentFrame(const std::string& name) {
    return next ? next->findInAnyFrame(name) : nullptr;  // 如果存在父级环境则递归查找
  }

  // 设置指定名称的变量类型
  void setType(const std::string& name, TypePtr type) {
    type_table[name] = std::move(type);  // 存储变量的类型信息
  }

  // 在任意环境中查找指定名称的值
  SugaredValuePtr findInAnyFrame(const std::string& name) {
    // 这个函数在代码截断处省略了，后续应补充完整
    // 使用自身作为起始点，沿着链表形式的链表迭代
    for (auto runner = this; runner; runner = runner->next.get()) {
      // 在当前节点中查找指定名称的变量
      if (auto r = runner->findInThisFrame(name)) {
        // 如果找到变量，则返回该变量的指针
        return r;
      }
    }
    // 如果未找到变量，则返回空指针
    return nullptr;
  }

  // 返回当前对象持有的块指针
  Block* block() {
    return b;
  }

  // 设置变量的值，将其封装成简单值后存储
  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    // 调用带有简单值参数的 setSugaredVar 函数
    setSugaredVar(
        loc,
        name,
        std::make_shared<SimpleValue>(value),
        /*annotated_type=*/nullptr);
  }

  // 设置包含糖值（SugaredValue）的变量
  void setSugaredVar(
      const SourceRange& loc,
      const std::string& name,
      SugaredValuePtr value,
      TypePtr annotated_type) {
    // 将糖值转换为简单值
    Value* as_simple_value = asSimple(value);
    // 如果转换后的简单值存在，且没有调试名称，并且名称具有实际意义，
    // 且简单值的节点所属的块与当前对象的块相同
    if (as_simple_value && !as_simple_value->hasDebugName() &&
        meaningfulName(name) &&
        as_simple_value->node()->owningBlock() == block()) {
      // 设置简单值的调试名称为变量名称
      as_simple_value->setDebugName(name);
    }
    // 防止涉及任何糖值的重新赋值
    // 对于类似以下的重新赋值：
    // a = ...
    // while ...
    //   a = ..
    // 需要使 'a' 在图中成为一流对象，因为其值取决于控制流
    // （即控制流中的值）
    // 查找父作用域中是否存在同名变量，并返回其指针
    if (auto parent = findInParentFrame(name)) {
      // 如果存在变量注释类型但尚未定义，则抛出错误报告
      if (annotated_type) {
        throw ErrorReport(loc)
            << "Attempting to declare and annotate the type of variable '"
            << name << "' but it is already defined in an outer block";
      }
      // 如果不是简单值类型，则抛出错误报告
      if (!as_simple_value) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' to a value of type "
            << value->kind() << " because " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      // 将父变量转换为简单值类型，若无法转换则抛出错误报告
      Value* simple_parent = asSimple(parent);
      if (!simple_parent) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' because it has type "
            << value->kind() << " and " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }

      // 获取父变量的未整形类型
      auto parent_type = unshapedType(simple_parent->type());
      // 尝试将值转换为父变量类型，若转换失败则抛出错误报告
      as_simple_value = tryConvertToType(
          loc,
          *b->owningGraph(),
          parent_type,
          as_simple_value,
          /*allow_conversions=*/true);
      std::stringstream why_not;
      // 如果值的类型不是父变量类型的子类型，则抛出详细错误报告
      if (!as_simple_value->type()->isSubtypeOfExt(*parent_type, &why_not)) {
        auto error = ErrorReport(loc);
        error << "Variable '" << name << "' previously had type "
              << simple_parent->type()->repr_str()
              << " but is now being assigned to a value of type "
              << as_simple_value->type()->repr_str();

        // 特别处理尝试分配给张量列表的错误消息
        if (simple_parent->type()->kind() == TypeKind::ListType &&
            as_simple_value->type()->kind() == TypeKind::ListType) {
          error << "\nEmpty lists default to List[Tensor]. Add a variable "
                   "annotation to the assignment to create an empty list "
                   "of another type (torch.jit.annotate(List[T, []]) where T "
                   "is the type of elements in the list for Python 2)";
        }
        error << "\n" << why_not.str();
        throw error;
      }
    }
    // 如果存在简单值，则继续处理
    if (as_simple_value) {
      // 如果存在变量注释类型并且值的类型不是注释类型的子类型，则抛出错误报告
      if (annotated_type &&
          !as_simple_value->type()->isSubtypeOf(*annotated_type)) {
        throw ErrorReport(loc)
            << "Variable '" << name << "' is annotated with type "
            << annotated_type->repr_str()
            << " but is being assigned to a value of type "
            << as_simple_value->type()->repr_str();
      }
      // 确定值存储的类型，并插入存储
      auto value_store_type =
          annotated_type ? annotated_type : as_simple_value->type();
      insertStore(name, loc, as_simple_value, value_store_type);
    } else {
      // 否则将值存储在值表中
      value_table[name] = std::move(value);
    }
  }

  // 返回变量的糖化值对象指针，根据标识符查找
  SugaredValuePtr getSugaredVar(const Ident& ident, bool required = true) {
    return getSugaredVar(ident.name(), ident.range());
  }

  // 返回变量对象指针，根据标识符查找
  Value* getVar(const Ident& ident) {
  // 返回标识符对应的糖化值，作为值对象
  return getSugaredVar(ident)->asValue(ident.range(), method);
}

// 抛出变量未找到错误的函数
void throwVarNotFoundError(
    const std::string& ident,
    const SourceRange& range) {
  // 检查是否由于类型不匹配未在if语句中发出该值。如果是，则打印更详细的错误消息
  if (auto msg = findVariableTypeError(ident)) {
    throw ErrorReport(range) << *msg << "and was used here";
  }
  throw ErrorReport(range) << "undefined value " << ident;
}

// 获取标识符对应的糖化值指针，带有范围和可选的必需标志
SugaredValuePtr getSugaredVar(
    const std::string& ident,
    const SourceRange& range,
    bool required = true) {
  auto retval = findInAnyFrame(ident);

  // 如果未找到值，尝试解析标识符的类型并生成对应的命名元组构造器
  if (!retval) {
    if (auto type = resolver->resolveType(ident, range)) {
      if (auto tuple_type = type->cast<TupleType>()) {
        retval = std::make_shared<NamedTupleConstructor>(tuple_type);
      }
    }
  }

  // 如果仍未找到值，尝试解析标识符的值并获取其解析值
  if (!retval) {
    retval = resolver->resolveValue(ident, method, range);
  }

  // 如果仍未找到值，并且需要必需的值，则抛出变量未找到错误
  if (!retval && required) {
    throwVarNotFoundError(ident, range);
  }

  return retval;
}

// 获取标识符对应的值对象指针
Value* getVar(const std::string& ident, const SourceRange& range) {
  return getSugaredVar(ident, range)->asValue(range, method);
}

// 移除标识符对应的变量，可以选择检查是否已移除
void removeVar(const Ident& ident, bool check_if_removed = false) {
  bool removed = false;

  // 在当前及其父级作用域中移除标识符对应的条目
  for (auto runner = this; runner; runner = runner->next.get()) {
    auto a = runner->value_table.erase(ident.name());
    auto b = runner->type_table.erase(ident.name());
    removed = a || b;
  }

  // 如果需要检查并且未移除标识符，则抛出变量未找到错误
  if (check_if_removed && !removed) {
    throwVarNotFoundError(ident.name(), ident.range());
  }
}

// 返回已定义变量的名称列表
std::vector<std::string> definedVariables() {
  std::vector<std::string> result;
  for (auto& kv : type_table) {
    result.push_back(kv.first);
  }
  return result;
}
};

// 静态函数模板：将常量值材料化到图中
template <class T, class Hash>
static Value* materializeConstant(
    T val,  // 要材料化的常量值
    Graph& graph,  // 当前图对象的引用
    const SourceRange& r,  // 源代码范围
    std::unordered_map<T, Value*, Hash>& map) {  // 存储常量值到图中的映射
  // 查找是否已经存在这个常量值对应的 Value
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;  // 如果找到，则返回已存在的 Value
  }

  // 在图的第一个节点之前设置插入点
  WithInsertPoint guard(graph.block()->nodes().front());
  // 在图中插入新的常量值，并返回新的 Value
  auto new_constant = graph.insertConstant(val, r);
  map[val] = new_constant;  // 将新的常量值映射到新的 Value

  return new_constant;  // 返回新的 Value
}

// 内联函数：检查类型是否为支持的列表元素类型
inline bool isSupportedListElementType(const TypePtr& type) {
  return type->isSubtypeOf(*TensorType::get()) ||  // 检查是否为张量类型的子类型
      type->isSubtypeOf(*NumberType::get());  // 检查是否为数字类型的子类型
}

// 用于记录正在处理的函数的信息的结构体
// 支持闭包，因此需要一个此信息的堆栈
struct DefContext {
  TypePtr declared_return_type_;  // 声明的返回类型，如果没有类型注释则为 nullptr
  TypePtr merged_return_type_;  // 合并后的返回类型，如果还未看到 Return 则为 nullptr
};

// 循环状态的枚举类
enum class LoopStatus { NOT_IN_LOOP, IN_LOOP, IN_UNROLLED_LOOP };

// 用于管理循环状态的结构体
struct WithLoopStatus {
  // 构造函数：保存之前的循环状态，并设置新的循环状态
  WithLoopStatus(LoopStatus* prev, LoopStatus new_status) {
    prev_value_ = *prev;  // 保存之前的循环状态
    prev_ptr_ = prev;  // 保存循环状态指针
    *prev = new_status;  // 设置新的循环状态
  }
  
  // 析构函数：在退出作用域时恢复之前的循环状态
  ~WithLoopStatus() {
    *prev_ptr_ = prev_value_;  // 恢复之前的循环状态
  }

 private:
  LoopStatus* prev_ptr_;  // 指向之前循环状态的指针
  LoopStatus prev_value_;  // 保存的之前的循环状态的值
};

// 用于将函数定义转换为 IR 的结构体
struct to_ir {
  // 构造函数：接收函数定义、解析器、self 指针和正在构建的方法
  to_ir(
      const Def& def,
      ResolverPtr resolver_,
      const Self* self,
      GraphFunction& method) // 正在构建的方法
      : method(method),
        graph(method.graph()),
        resolver(std::move(resolver_)),
        typeParser_(resolver),
        environment_stack(nullptr) {
    AT_ASSERT(resolver);  // 确保解析器不为空
    pushFrame(graph->block(), /*starts_def=*/true);

    // 类型注释不包括显式类型化的 "self" 参数，因此在方法中使用 self 时，参数数量会比类型注释少一个
    if (self && def.decl().params().empty()) {
      throw ErrorReport(def.decl().params().range())
          << "methods must have a self argument";  // 抛出错误，方法必须有 self 参数
    }
    method.setSchema(emitDef(def, self, graph->block()));  // 设置方法的模式

    // 替换旧的操作符为其有效的升级器，以适应使用旧操作符模式的情况
    ReplaceOldOperatorsWithUpgraders(graph);

    // 注意顺序：SSA 转换必须在闭包和分支的提升之前进行，
    // 这样闭包在其原始图中转换为 SSA，且准备好内联到分支闭包中
    ConvertToSSA(graph);

    // 将具有迭代器和条件体的循环转换为 Python 识别的 while 循环
    // 旨在导出和运行此传递以避免抖动，与 SSA 转换一样，只需要运行一次
    CanonicalizeModifiedLoops(graph);

    // 将操作符转换为规范化形式
    // 对图形进行归一化操作
    NormalizeOps(graph);

    // 运行清理传递的操作，对图形进行必要的优化和修正
    runCleanupPasses(graph);
  }

 private:
  GraphFunction& method;
  std::shared_ptr<Graph> graph;
  ResolverPtr resolver;
  // 整数常量的哈希映射，用于快速查找整数值
  std::unordered_map<int64_t, Value*, std::hash<int64_t>> integral_constants;
  // 浮点数常量的哈希映射，用于快速查找浮点数值
  std::unordered_map<double, Value*, std::hash<double>> fp_constants;
  // 复数常量的哈希映射，用于快速查找复数值
  std::unordered_map<
      c10::complex<double>,
      Value*,
      c10::hash<c10::complex<double>>>
      complex_constants;
  // 保存所有的出口块，用于控制流分析
  std::unordered_set<Block*> exit_blocks;
  ScriptTypeParser typeParser_;
  // 循环状态标识，默认为不在循环中
  LoopStatus loop_status_ = LoopStatus::NOT_IN_LOOP;

  // 环境栈的头部，指向最近的封闭作用域
  std::shared_ptr<Environment> environment_stack;
  std::vector<DefContext> def_stack_;
  size_t temp_name_count_ = 0;
  
  // 创建一个临时变量名，确保名称唯一性
  std::string createTempName(const std::string& prefix) {
    return prefix + std::to_string(temp_name_count_++);
  }

  // 推入一个新的执行帧到环境栈
  void pushFrame(Block* b, bool starts_def = false) {
    if (starts_def) {
      def_stack_.emplace_back();
    }
    environment_stack =
        std::make_shared<Environment>(method, resolver, b, environment_stack);
  }

  // 弹出当前的执行帧，并返回上一个执行帧
  std::shared_ptr<Environment> popFrame(bool ends_def = false) {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    if (ends_def) {
      def_stack_.pop_back();
    }
    return old_frame;
  }

  // 处理函数可能没有返回值的情况，在函数最后添加隐式的 None 返回
  void handleMaybeNoReturn(const Def& def, Block* block) {
    auto decl_ret = def_stack_.back().declared_return_type_;
    if (exit_blocks.count(block) == 0) {
      // 如果函数块没有返回语句，则在末尾添加一个隐式的 None 返回
      auto decl_ret = def_stack_.back().declared_return_type_;
      if (decl_ret && decl_ret != NoneType::get()) {
        throw ErrorReport(def.range())
            << "Function was not annotated as having type None, but does not "
            << "return along all paths";
      }
      WithInsertPoint b(*block->nodes().end());
      emitReturn(Return::create(
          def.range(), Expr(Compound::create(TK_NONE, def.range(), {}))));
    } else {
      // 如果函数块存在返回，但没有具体返回值，则接受声明的返回类型或设置为 None
      if (def_stack_.back().merged_return_type_ == nullptr) {
        def_stack_.back().merged_return_type_ =
            decl_ret != nullptr ? decl_ret : NoneType::get();
      }
    }
  }

  // 发出函数定义，并返回其函数签名
  FunctionSchema emitDef(const Def& def, const Self* self, Block* block) {
    auto schema = typeParser_.parseSchemaFromDef(def, bool(self));
    // 如果函数只有一个返回值，记录其声明的返回类型
    if (schema.returns().size() == 1) {
      def_stack_.back().declared_return_type_ = schema.returns().at(0).type();
    }
    // 发出形式参数并返回参数列表
    std::vector<Argument> arguments =
        emitFormalArguments(def, self, schema, block);

    // 发出函数体的语句列表
    auto stmts_list = def.statements();
    emitStatements(stmts_list.begin(), stmts_list.end());
    // 处理可能没有返回值的函数调用，使用给定的定义和代码块
    handleMaybeNoReturn(def, block);
    
    // 根据函数定义的范围、函数模式和代码块生成输出，放入返回值向量中
    std::vector<Argument> returns = {emitOutput(def.range(), schema, block)};
    
    // 返回一个包含函数名称、空字符串、移动后的参数向量和返回值向量的元组
    return {def.name().name(), "", std::move(arguments), std::move(returns)};
    }
    
    // 查看 [setstate type] 部分的类型定义
    static TypePtr getTypeForSetStateArg(const Def& def, const Self* self) {
      // 断言确保 self 对象存在
      TORCH_CHECK(self, "Expected __setstate__ to have a `self` argument");
    
      // 获取类类型的 __getstate__ 方法
      auto getstate = self->getClassType()->findMethod("__getstate__");
    
      // 如果找不到 __getstate__ 方法，则抛出错误报告
      if (!getstate) {
        throw ErrorReport(def.range())
            << "`__setstate__` defined but not `__getstate__`. "
            << "You must have both defined on a ScriptModule "
            << "to customize serialization.\n"
            << "Did you forget to use `@torch.jit.export`?";
      }
    
      // 确保 __getstate__ 方法已定义
      getstate->ensure_defined();
    
      // 返回 __getstate__ 方法的返回类型
      return self->getClassType()
          ->getMethod("__getstate__")
          .getSchema()
          .returns()
          .at(0)
          .type();
    }
    
    // 查看 [setstate type] 部分的类型推导条件
    static bool shouldDeriveSetStateType(
        const Def& def,
        const FunctionSchema& schema) {
      // 检查所有参数是否都是推断类型
      const bool noTypeAnnotations = std::all_of(
          schema.arguments().begin(),
          schema.arguments().end(),
          [](const Argument& arg) { return arg.is_inferred_type(); });
    
      // 如果函数名是 "__setstate__" 并且没有类型注解，则应该进行类型推导
      bool shouldInfer = def.name().name() == "__setstate__" && noTypeAnnotations;
      if (!shouldInfer) {
        return false;
      }
    
      // 对 __setstate__ 函数进行基本验证，确保其格式正确
      TORCH_INTERNAL_ASSERT(def.name().name() == "__setstate__");
      const auto numDeclParams = def.decl().params().size();
    
      // 如果声明的参数数量不是 2，则抛出错误报告
      if (numDeclParams != 2) {
        throw ErrorReport(def.range())
            << "Expected 2 arguments for `__setstate__`, got: " << numDeclParams;
      }
    
      // 返回应该进行类型推导的标志
      return true;
    }
    
    // 生成正式参数的表达式，用于给定的函数定义、self 指针、函数模式和代码块
    std::vector<Argument> emitFormalArguments(
        const Def& def,
        const Self* self,
        const FunctionSchema& schema,
        Block* block) {
      std::vector<Argument> arguments; // 用于存储参数的向量
    
      // 输入参数处理
      auto it = def.decl().params().begin();
      auto end = def.decl().params().end();
      auto expected_annotation_size = def.decl().params().size();
    
      // 如果存在 self 指针，则期望的类型注解数量减少一个
      if (self) {
        expected_annotation_size--;
      }
    
      // 如果函数参数的类型注解数量与期望的数量不匹配，则抛出错误报告
      if (schema.arguments().size() != expected_annotation_size) {
        throw ErrorReport(def.decl().params().range())
            << "Number of type annotations for"
            << " function parameters (" << schema.arguments().size() << ")"
            << " does not match the number of parameters on the function ("
            << expected_annotation_size << ")!";
      }
    
      // 如果存在 self 指针，则处理第一个参数
      if (self) {
        AT_ASSERT(it != end);
        const auto& name = (*it).ident().name();
        Value* new_input = block->addInput()->setDebugName(name);
        environment_stack->setSugaredVar(
            (*it).ident().range(),
            name,
            self->makeSugared(new_input),
            /*annotated_type=*/nullptr);
        arguments.emplace_back(name, new_input->type());
        ++it;
      }
    
      // 这里是 [setstate type] 部分的开始
    // __setstate__ is special, because if the user leaves it un-annotated we
    // will derive the type for `state` from the output type of __getstate__.
    // This is necessary so that we can allow submodules to appear in `state`.
    // 检查是否应该推导 __setstate__ 的类型，如果需要，从 __getstate__ 的输出类型推导 `state` 的类型
    bool shouldDeriveType = shouldDeriveSetStateType(def, schema);
    // 初始化参数注释索引
    size_t arg_annotation_idx = 0;
    // 遍历语句列表
    for (; it != end; ++it) {
      auto& name = (*it).ident().name();
      // 将输入添加到图中
      Value* new_input = block->addInput();
      // 如果名称有意义，则设置调试名称
      if (meaningfulName(name)) {
        new_input->setDebugName(name);
      }
      // 记录模式的类型并在 Value* 上设置类型
      auto arg = schema.arguments().at(arg_annotation_idx++);
      // 如果需要推导类型，则从 getTypeForSetStateArg 推导状态参数的类型
      if (shouldDeriveType) {
        TORCH_INTERNAL_ASSERT(schema.arguments().size() == 1);
        const auto& inferredStateType = getTypeForSetStateArg(def, self);
        arg = arg.cloneWithType(inferredStateType);
      }

      // 将参数添加到 arguments 列表中
      arguments.push_back(arg);
      // 设置 new_input 的类型
      new_input->setType(arguments.back().type());

      // 注意：在调用 setVar 之前设置 new_input 的类型，以便 Store 操作能正确推断类型
      // 在环境栈中设置变量，使用标识符的范围和名称来关联 new_input
      environment_stack->setVar((*it).ident().range(), name, new_input);
    }
    // 返回构建好的参数列表
    return arguments;
  }

  Argument emitOutput(
      const SourceRange& range,
      const FunctionSchema& schema,
      Block* block) {
    // handleMaybeNoReturn 确保 merged_return_type_ 始终被设置
    auto ret_type = def_stack_.back().merged_return_type_;
    TORCH_INTERNAL_ASSERT(ret_type);

    // 在 ConvertToSSA pass 中，prim::ReturnStmts 会被降低级别，以便正确设置返回值
    // 目前我们有一个正确类型的占位符返回值，这对于正确类型化闭包和图形很重要
    auto placeholder_return =
        graph->insertNode(graph->createUninitialized(ret_type))->output();
    // 注册占位符返回值作为块的输出
    block->registerOutput(placeholder_return);
    // 返回空参数名和函数定义栈顶部的合并返回类型
    return Argument("", def_stack_.back().merged_return_type_);
  }

  void emitStatements(const List<Stmt>& statements) {
  // 调用 emitStatements 函数处理 statements 容器中的语句，并返回结果
  return emitStatements(statements.begin(), statements.end());
}

// XXX: 目前闭包并没有通用实现，仅用作特殊任务的中间形式，如定义梯度或分叉函数。
//
// 还有几个未完成的方面使它们不能通用使用：
// 1. 我们没有一个类型、ivalue、操作符来表示 prim::Closure，因此 closure_node 的类型为 None。
// 2. 尚未为其编写导出逻辑，因此无法导出或打印为 Python 代码。
// 3. 没有阻止闭包内已存在变量赋值的机制，对这些变量的更改将被忽略。
// 4. frontend.py 中没有解析支持，这是有意为之，因为这可以防止人们意外使用该功能。
//
// 该函数在图中留下类似以下结构的内容：
//
//   %2 : None = prim::Closure()
//     block0():
//       %1 : Tensor = prim::DoSomething(%0)
//       -> (%1)
//
// 需要单独的处理来消除此闭包并替换为实际可执行内容（参见 liftClosure 和 inlineForkedClosure）。
std::shared_ptr<ClosureValue> emitClosure(
    const std::function<void(Block*)>& emit_body) {
  // 在图中插入一个 prim::Closure 节点，并声明一个输出
  Node* closure_node = graph->insertNode(graph->create(prim::Closure, 1));
  // 目前它还不是一个真实的对象，因此将其类型设置为 None
  closure_node->output()->setType(NoneType::get());
  // 为闭包节点添加一个新的块
  Block* block = closure_node->addBlock();
  // 设定循环状态为 NOT_IN_LOOP
  WithLoopStatus loop_guard(&loop_status_, LoopStatus::NOT_IN_LOOP);
  {
    // 在当前块设置插入点
    WithInsertPoint guard(block);
    // 推入帧栈，并标记为定义的起始
    pushFrame(block, /*starts_def=*/true);
    // 在块中生成闭包体
    emit_body(block);
    // 弹出帧栈，并标记为定义结束
    popFrame(/*ends_def=*/true);
  }
  // 返回一个表示闭包值的 shared_ptr
  return std::make_shared<ClosureValue>(closure_node->output());
}

void emitClosure(const Def& def) {
  // 当闭包块设置为环境时调用
  auto emit_body = [&](Block* closure_block) {
    // 发射定义，忽略模式返回，目前不会创建闭包的方法
    emitDef(
        def,
        nullptr,
        closure_block);
  };
  // 生成闭包体，并获取闭包值
  auto closure_value = emitClosure(emit_body);
  // 在环境堆栈中设置 sugared 变量
  environment_stack->setSugaredVar(
      def.name().range(),
      def.name().name(),
      closure_value,
      /*annotated_type=*/nullptr);
}

void checkBreakContinue(
    const SourceRange& loc,
    const std::string& stmt_name) {
  // 如果当前不在循环中，则抛出语法错误
  if (loop_status_ == LoopStatus::NOT_IN_LOOP) {
    throw ErrorReport(loc) << "SyntaxError: '" << stmt_name << "'"
                           << " outside loop";
  } 
  // 如果当前在未展开的循环中，则不支持在循环体内使用 break 或 continue
  else if (loop_status_ == LoopStatus::IN_UNROLLED_LOOP) {
    throw ErrorReport(loc)
        << "Because we emit iteration over modulelists or tuples as "
           "unrolled loops, we do not support break or continue inside the body of these loops";
  }
}

void emitBreak(const Break& stmt) {
    // 检查是否在循环语句中使用了 break，进行相应的错误检查和处理
    checkBreakContinue(stmt.range(), "break");
    // 创建一个 BreakStmt 节点，并设置其源代码范围
    auto break_node =
        graph->create(prim::BreakStmt, {}, 0)->setSourceRange(stmt.range());
    // 将节点插入到计算图中
    graph->insertNode(break_node);
  }

  void emitContinue(const Continue& stmt) {
    // 检查是否在循环语句中使用了 continue，进行相应的错误检查和处理
    checkBreakContinue(stmt.range(), "continue");
    // 创建一个 ContinueStmt 节点，并设置其源代码范围
    auto continue_node =
        graph->create(prim::ContinueStmt, {}, 0)->setSourceRange(stmt.range());
    // 将节点插入到计算图中
    graph->insertNode(continue_node);
  }

  void emitDelete(const Delete& stmt) {
    // 遍历删除语句中的目标对象
    for (const auto& target : stmt.targets()) {
      // 如果目标是下标操作
      if (target.kind() == TK_SUBSCRIPT) {
        Subscript subscript(target);
        const List<Expr>& subscript_exprs = subscript.subscript_exprs();
        // 如果下标表达式是切片表达式，抛出异常
        if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
          throw ErrorReport(target.range())
              << "del statements only support deletion at a single index, "
                 "slicing is not supported"
                 " (see https://github.com/pytorch/pytorch/issues/31430)";
        }
        // 生成下标表达式对应的值的处理后的对象
        const SugaredValuePtr sv = emitSugaredExpr(subscript.value(), 1);
        // 获取值的源代码范围
        const SourceRange& val_range = subscript.value().range();
        // 生成下标表达式的索引值
        Value* idx = emitExpr(subscript_exprs[0]);
        // 生成值的处理后对象
        Value* val = sv->asValue(val_range, method);

        // 如果值是类实例，调用其定义在 __delitem__ 方法中的特定类型实现
        if (auto cls = val->type()->cast<ClassType>()) {
          // 如果类没有定义 __delitem__ 方法，抛出异常
          if (!cls->findMethod("__delitem__")) {
            throw ErrorReport(target.range())
                << "Class does not define __delitem__";
          }

          // 使用 MethodValue 调用方法处理递归删除
          MethodValue(val, "__delitem__")
              .call(stmt.range(), method, {idx}, {}, 0);
        } else {
          // 否则，创建一个删除节点，并设置其源代码范围
          auto node = graph->create(aten::Delete, {val, idx}, 0)
                          ->setSourceRange(target.range());
          // 将节点插入到计算图中
          graph->insertNode(node);
        }
      } else if (target.kind() == TK_VAR) {
        // 如果目标是变量，从环境栈中移除该变量
        Var var(target);
        environment_stack->removeVar(var.name(), /*check_if_removed=*/true);
      } else {
        // 否则，抛出不支持的删除语句类型的异常
        throw ErrorReport(target.range())
            << "del statements are only supported for deleting"
               " list and dict items and variables";
      }
    }
  }

  void emitReturn(const Return& stmt) {
    // 获取当前函数声明的返回类型，如果没有注解则为 nullptr
    TypePtr declared_return_type =
        def_stack_.back().declared_return_type_; // nullptr if not annotated
    // 生成实际返回值的处理后的对象
    auto actual_return = emitExpr(stmt.expr(), declared_return_type);

    // 如果结果类型有注解，则每个返回值必须转换为该类型
    if (declared_return_type) {
      // 如果声明了返回类型，则进行类型检查和转换
      // 这个保护条件跳过从 None -> Tensor 的隐式转换，否则如果忘记在返回 Tensor 的函数中写返回语句，
      // 将导致 None 转换为 Tensor。
      if (!(actual_return->type()->isSubtypeOf(*TensorType::get()) &&
            actual_return->type()->isSubtypeOf(*NoneType::get()))) {
        // 尝试将实际返回值转换为声明的返回类型
        actual_return = tryConvertToType(
            stmt.range(),
            *graph,
            declared_return_type,
            actual_return,
            /*allow_conversions=*/true);
      }
      // 检查实际返回值是否符合声明的返回类型
      if (!actual_return->type()->isSubtypeOf(*declared_return_type)) {
        throw ErrorReport(stmt.range())
            << "Return value was annotated as having type "
            << declared_return_type->repr_str() << " but is actually of type "
            << actual_return->type()->repr_str();
      }
    } else {
      // 如果未声明返回类型，则使用当前作用域中的合并返回类型
      declared_return_type = def_stack_.back().merged_return_type_;
      if (!declared_return_type) {
        // 如果没有合并的返回类型，则使用实际返回值的类型
        declared_return_type = actual_return->type();
      }
      // 合并当前返回值的类型和之前的返回类型
      auto merged_return_type =
          unifyTypes(declared_return_type, actual_return->type());
      if (!merged_return_type) {
        throw ErrorReport(stmt.range())
            << "Previous return statement returned a value of type "
            << declared_return_type->repr_str()
            << " but this return statement returns a value of type "
            << actual_return->type()->repr_str();
      }
      declared_return_type = merged_return_type.value();
    }
    // 确保声明的返回类型不为空
    AT_ASSERT(declared_return_type);

    // 如果声明的返回类型是 Any，并且实际返回值的类型不是 Any，则将结果转换为 Any 类型，
    // 以便在不同代码路径（如 if 的不同分支，循环的主体和包含的作用域）之间进行类型统一。
    if (declared_return_type == AnyType::get() &&
        actual_return->type() != AnyType::get()) {
      actual_return =
          graph->insertUncheckedCast(actual_return, declared_return_type);
    }

    // 在计算图中插入 ReturnStmt 节点，表示返回实际的返回值
    graph->insertNode(graph->create(prim::ReturnStmt, {actual_return}, 0));
    // 将当前块添加到退出块的集合中
    exit_blocks.insert(environment_stack->block());
  }
    // 迭代遍历给定的语句列表
    for (; begin != end; ++begin) {
      // 获取当前语句并更新错误报告的调用堆栈中的待处理范围
      auto stmt = *begin;
      ErrorReport::CallStack::update_pending_range(stmt.range());

      // 根据语句类型进行分支处理
      switch (stmt.kind()) {
        case TK_IF:
          // 如果是 IF 语句，则生成对应的代码
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          // 如果是 WHILE 语句，则生成对应的代码
          emitWhile(While(stmt));
          break;
        case TK_FOR:
          // 如果是 FOR 语句，则生成对应的代码
          emitFor(For(stmt));
          break;
        case TK_ASSIGN:
          // 如果是赋值语句，则生成对应的代码
          emitAssignment(Assign(stmt));
          break;
        case TK_AUG_ASSIGN:
          // 如果是增强赋值语句，则生成对应的代码
          emitAugAssignment(AugAssign(stmt));
          break;
        case TK_EXPR_STMT: {
          // 如果是表达式语句，则获取表达式并生成对应的代码
          auto expr = ExprStmt(stmt).expr();
          emitSugaredExpr(expr, 0);
        } break;
        case TK_RAISE:
          // 如果是 RAISE 语句，则生成对应的代码
          emitRaise(Raise(stmt));
          break;
        case TK_ASSERT:
          // 如果是 ASSERT 语句，则生成对应的代码
          emitAssert(Assert(stmt));
          break;
        case TK_RETURN: {
          // 如果是 RETURN 语句，则生成对应的代码
          emitReturn(Return(stmt));
        } break;
        case TK_CONTINUE: {
          // 如果是 CONTINUE 语句，则生成对应的代码
          emitContinue(Continue(stmt));
        } break;
        case TK_BREAK: {
          // 如果是 BREAK 语句，则生成对应的代码
          emitBreak(Break(stmt));
        } break;
        case TK_PASS:
          // PASS 语句不生成任何代码，直接跳过
          break;
        case TK_DEF:
          // 如果是函数定义语句，则生成对应的闭包代码
          emitClosure(Def(stmt));
          break;
        case TK_DELETE:
          // 如果是 DELETE 语句，则生成对应的代码
          emitDelete(Delete(stmt));
          break;
        case TK_WITH:
          // 如果是 WITH 语句，则生成对应的代码
          emitWith(With(stmt));
          break;
        default:
          // 如果是未知的语句类型，抛出错误报告
          throw ErrorReport(stmt)
              << "Unrecognized statement kind " << kindToString(stmt.kind());
      }

      // 如果在当前块中发现了退出语句，则停止处理后续语句
      if (exit_blocks.count(environment_stack->block()))
        return;
    }
  }

  // 查找并返回指定表达式 lhs 和 rhs 中是否存在 "is None" 的细化集合
  RefinementSet findIsNoneRefinements(
      const Expr& lhs,
      Value* lhs_value,
      const Expr& rhs,
      Value* rhs_value,
      int tok) {
    if (rhs.kind() != TK_NONE && lhs.kind() == TK_NONE) {
      // 将 'None is var' 转换为 'var is None'，并递归调用以处理
      return findIsNoneRefinements(rhs, rhs_value, lhs, lhs_value, tok);
    }
    if (rhs.kind() != TK_NONE || lhs.kind() != TK_VAR) {
      // 如果 rhs 不是 None 或者 lhs 不是变量，则返回空集合
      return {};
    }
    // 对于语句形式为 var {is, is not} None，获取变量名并构造细化集合
    const std::string& name = Var(lhs).name().name();
    // 尽管理论上可以将 'x is None' 特化为 x 的类型为 NoneType，
    // 但我们之前并未这样做。这样做会导致 None 类型在所有加载的模型中传播。
    // unwrap_optional 的处理在这些情况下将失败，因为导出未预期输入会是未注释的 None。
    // 要启用此功能，我们需要 (1) 实现一个真正的类型转换操作符 annotated(T, X)，
    // 它保留在图中并执行类型转换，以及 (2) 只在加载新图时启用此 OPTIONAL_NONE，
    // 因为它与旧图不兼容。
    // 创建并返回一个 OPTIONAL_NONE 类型的细化对象
    // Refinement none(name, RefinementKind::OPTIONAL_NONE);
    // 如果 lhs_value 是 OptionalType 类型的指针
    if (const auto optional_type = lhs_value->type()->cast<OptionalType>()) {
      // 创建一个 Refinement 对象，表示变量名和 OptionalType 中元素类型的关系
      Refinement present(name, optional_type->getElementType());
      // 如果是 TK_IS，则返回一个 RefinementSet，包含 present，表示存在此类型的变量
      if (tok == TK_IS) {
        return RefinementSet({}, {present});
      } else { // TK_ISNOT
        // 如果是 TK_ISNOT，则返回一个 RefinementSet，包含 present，表示不存在此类型的变量
        return RefinementSet({present}, {});
      }
    }
    // 如果 lhs_value 是 UnionType 类型的指针
    if (const auto union_type = lhs_value->type()->cast<UnionType>()) {
      // 准备要从 UnionType 中排除的类型，这里排除 NoneType
      std::vector<TypePtr> to_subtract{NoneType::get()};
      // 尝试从 UnionType 中排除指定类型后剩余的类型
      std::optional<TypePtr> remaining =
          union_type->subtractTypeSet(to_subtract);
      // 存储所有 present 的 Refinement 对象
      std::vector<Refinement> all_present;
      if (remaining) {
        // 如果有剩余类型，则创建一个 Refinement 对象表示此类型的存在
        Refinement present{name, *remaining};
        all_present.push_back(std::move(present));
      }
      // 根据 token 类型返回相应的 RefinementSet
      if (tok == TK_IS) {
        // 如果是 TK_IS，则返回一个 RefinementSet，表示存在这些类型的变量
        return RefinementSet({}, all_present);
      } else { // TK_ISNOT
        // 如果是 TK_ISNOT，则返回一个 RefinementSet，表示不存在这些类型的变量
        return RefinementSet(all_present, {});
      }
    }
    // 如果不是 OptionalType 或 UnionType，返回一个空的 RefinementSet
    return RefinementSet();
  }

  // 根据条件表达式发出条件值
  CondValue emitCondExpr(const Expr& expr) {
    // 留空
    // 实际实现应该根据条件表达式生成相应的条件值
    // 但这里只是展示注释的形式，没有实际代码
    // 实际代码可能包括对表达式进行求值，并生成条件值
    // 略去详细的实现细节
    // 返回一个 CondValue 类型的对象
    }
  }

  // 发出单一的 if 分支
  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt>& branch,
      const RefinementSet& refinements) {
    // 将当前块推入堆栈
    pushFrame(b);
    // 设置当前块为插入点
    WithInsertPoint guard(b);
    // 在当前分支上插入细化条件
    insertRefinements(branch.range(), refinements);
    // 发出分支中的语句
    emitStatements(branch);
    // 弹出当前块并返回环境
    return popFrame();
  }

  // 创建一个节点
  Node* create(Symbol kind, const SourceRange& loc, size_t n_outputs) {
    // 在图中创建一个特定类型的节点，并设置其源范围
    return graph->create(kind, n_outputs)->setSourceRange(loc);
  }

  // 发出三元 if 表达式
  Value* emitTernaryIf(
      const TernaryIf& expr,
      const TypePtr& type_hint = nullptr) {
    // 发出条件表达式的条件值
    CondValue cond_value = emitCondExpr(expr.cond());
    // 如果条件表达式是静态值，则编译 `if` 语句，只发出 true 或 false 分支
    if (cond_value.staticIf()) {
      if (*cond_value.staticIf()) {
        // 如果条件为 true，则发出 true 分支表达式
        return emitExpr(expr.true_expr(), type_hint);
      } else {
        // 如果条件为 false，则发出 false 分支表达式
        return emitExpr(expr.false_expr(), type_hint);
      }
    }
    // 如果条件不是静态值，则发出完整的三元 if 表达式
    auto true_expr = [&] { return emitExpr(expr.true_expr(), type_hint); };
    auto false_expr = [&] { return emitExpr(expr.false_expr(), type_hint); };
    return emitIfExpr(expr.range(), cond_value, true_expr, false_expr);
  }

  // 模板函数，根据类型提示进行细化或填充候选类型向量
  template <class F1, class F2, class F3>
  void refineAndSetUnionTypeHintOrPopulateCandidatesVector(
      const TypePtr& type_hint,
      TypePtr* refined_type_hint_ptr,
      std::vector<TypePtr>* all_candidates,
      const std::string& match_repr,
      const Expr& src,
      const F1& type_match,
      const F2& do_if_match,
      const F3& do_if_anytype,
      bool is_dict_constructor = false) {
    if (auto union_type_hint = (*refined_type_hint_ptr)->cast<UnionType>()) {
      // 检查是否存在 Union 类型注解，并获取其包含的类型列表
      std::vector<TypePtr> candidate_types;

      // 将符合条件的 List 类型添加到 candidate_types 中
      std::copy_if(
          union_type_hint->containedTypes().begin(),
          union_type_hint->containedTypes().end(),
          std::back_inserter(candidate_types),
          [&](TypePtr type_ptr) { return type_match(type_ptr); });

      // 如果不是字典构造器且候选类型列表为空，则抛出错误报告
      if (!is_dict_constructor && candidate_types.empty()) {
        throw ErrorReport(src)
            << "Expected an Union type annotation "
            << "with an inner " << match_repr << " type, but got "
            << (*refined_type_hint_ptr)->repr_str();
      } else if (candidate_types.size() == 1) {
        // 如果 Union 只包含了一个符合条件的容器类型，则将 refined_type_hint_ptr 重置为该类型
        (*refined_type_hint_ptr) = candidate_types[0];
      } else {
        // 如果 Union 包含多种符合条件的容器类型，则将所有的候选类型赋给 all_candidates
        (*all_candidates) = std::move(candidate_types);
      }
    } else if (
        auto optional_type_hint =
            (*refined_type_hint_ptr)->cast<OptionalType>()) {
      // 如果存在 Optional 类型注解，则将 refined_type_hint_ptr 重置为其元素类型
      (*refined_type_hint_ptr) = optional_type_hint->getElementType();
    }

    // 处理像 `dict([(x, y), (a, b)])` 这样的构造器，直接返回，不进行后续检查
    if (is_dict_constructor) {
      return;
    }

    // 如果所有的候选类型为空，则根据 refined_type_hint_ptr 的类型进行相应操作
    if (all_candidates->empty()) {
      if (type_match(*refined_type_hint_ptr)) {
        // 如果 refined_type_hint_ptr 符合条件，则执行对应的操作
        do_if_match();
      } else if ((*refined_type_hint_ptr)->kind() == AnyType::Kind) {
        // 如果 refined_type_hint_ptr 是 AnyType 类型，则执行对应的操作
        do_if_anytype();
      } else {
        // 抛出错误报告，指出预期的注解类型与实际类型不符
        throw ErrorReport(src)
            << "Expected an annotation of type " << match_repr << " but got "
            << type_hint->repr_str();
      }
    }
  }

  void refineAndSetListTypeHintFromCandidatesVector(
      const std::vector<TypePtr>& all_candidates,
      const TypePtr& type_hint,
      TypePtr* refined_type_hint_ptr,
      const TypePtr& unified_elem_type,
      const Expr& src) {
    // 初始化最大元素类型为 nullptr
    TypePtr greatest_elem_type = nullptr;

    // 遍历所有的候选类型
    std::for_each(
        all_candidates.begin(), all_candidates.end(), [&](TypePtr candidate) {
          // 获取候选类型的列表元素类型
          auto candidate_elem_type =
              candidate->expect<ListType>()->getElementType();
          // 如果统一的元素类型是候选元素类型的子类型，则更新最大元素类型
          if (unified_elem_type->isSubtypeOf(candidate_elem_type)) {
            if (!greatest_elem_type) {
              greatest_elem_type = candidate_elem_type;
            } else {
              greatest_elem_type =
                  *(unifyTypes(greatest_elem_type, candidate_elem_type));
            }
          }
        });
    // 如果没有最大元素类型
    if (!greatest_elem_type) {
      // 创建一个字符串流，用于构建候选类型的表示字符串
      std::stringstream vector_repr;
      // 遍历所有候选类型
      for (size_t i = 0; i < all_candidates.size(); ++i) {
        // 如果不是第一个元素，并且候选类型数量大于2，则添加逗号分隔符
        if (i > 0 && all_candidates.size() > 2) {
          vector_repr << ", ";
        }
        // 如果不是第一个元素，并且是最后一个候选类型，则添加 " or "
        if (i != 0 && i == all_candidates.size() - 1) {
          vector_repr << " or ";
        }
        // 添加当前候选类型的表示字符串到 vector_repr
        vector_repr << all_candidates[i]->repr_str();
      }
      // 抛出错误报告，描述联合类型注解不能容纳给定列表元素的类型
      throw ErrorReport(src)
          << "Union type annotation `" << type_hint->repr_str() << "` can hold "
          << vector_repr.str() << ", but none of "
          << "those types match the types of the given list "
          << "elements, which were unified to "
          << unified_elem_type->repr_str();
    } else {
      // 更新 refined_type_hint_ptr 指向的类型为 ListType，其元素类型为 greatest_elem_type
      (*refined_type_hint_ptr) = ListType::create(greatest_elem_type);
      ;
    }
  }

  // 从候选类型向量中细化并设置字典类型注解
  void refineAndSetDictTypeHintFromCandidatesVector(
      const std::vector<TypePtr>& all_candidates,
      const TypePtr& type_hint,
      TypePtr* refined_type_hint_ptr,
      const TypePtr& known_key_type,
      const TypePtr& known_value_type,
      const Expr& src) {
    // 初始化候选键类型和值类型
    TypePtr candidate_key_type = nullptr;
    TypePtr candidate_value_type = nullptr;
    TypePtr candidate = nullptr;

    // 遍历所有候选类型
    for (const auto& current_candidate : all_candidates) {
      // 获取当前候选类型的键和值类型
      auto current_key_type =
          current_candidate->expect<DictType>()->getKeyType();
      auto current_value_type =
          current_candidate->expect<DictType>()->getValueType();

      // 如果已知的键类型是当前键类型的子类型，并且已知的值类型是当前值类型的子类型
      if (known_key_type->isSubtypeOf(current_key_type) &&
          known_value_type->isSubtypeOf(current_value_type)) {
        // 如果没有候选类型或者当前候选类型的键值对类型更具体，则更新候选类型
        if (!candidate ||
            (candidate_key_type->isSubtypeOf(current_key_type) &&
             candidate_value_type->isSubtypeOf(current_value_type))) {
          candidate_key_type = current_key_type;
          candidate_value_type = current_value_type;
          candidate = current_candidate;
        }
      }
    }

    // 如果没有找到合适的候选类型
    if (!candidate) {
      // 创建一个字符串流，用于构建候选类型的表示字符串
      std::stringstream vector_repr;
      // 遍历所有候选类型
      for (size_t i = 0; i < all_candidates.size(); ++i) {
        // 如果不是第一个元素，并且候选类型数量大于2，则添加逗号分隔符
        if (i > 0 && all_candidates.size() > 2) {
          vector_repr << ", ";
        }
        // 如果不是第一个元素，并且是最后一个候选类型，则添加 " or "
        if (i != 0 && i == all_candidates.size() - 1) {
          vector_repr << " or ";
        }
        // 添加当前候选类型的表示字符串到 vector_repr
        vector_repr << all_candidates[i]->repr_str();
      }
      // 抛出错误报告，描述联合类型注解不能容纳给定键和值的类型
      throw ErrorReport(src)
          << "Union type annotation `" << type_hint->repr_str() << "` can hold "
          << vector_repr.str() << ", but none of "
          << "those dict types can hold the types of the given"
          << " keys and values, which were unified to Dict["
          << known_key_type->repr_str() << ", " << known_value_type->repr_str();
    } else {
      // 更新 refined_type_hint_ptr 指向的类型为找到的候选类型
      (*refined_type_hint_ptr) = candidate;
    }
  }

  // 发出列表推导式的值，给定列表推导式和类型注解
  Value* emitListComprehension(const ListComp& lc, const TypePtr& type_hint) {
    // 获取列表推导式的位置信息
    const auto loc = lc.range();
    // 创建目标列表，包含列表推导式的目标表达式
    const auto targets_list = List<Expr>::create(lc.range(), {lc.target()});
    // 创建迭代器列表，包含列表推导式的迭代器表达式
    const auto itrs = List<Expr>::create(lc.range(), {lc.iter()});
    // 创建一个值为 prim::ListConstruct 的新节点，并将其输出值设定为列表类型的值
    Value* list_value = graph->insertNode(graph->create(prim::ListConstruct, 1))
                            ->output()
                            ->setType(ListType::ofTensors());

    // 复制 type_hint 作为 refined_type_hint
    TypePtr refined_type_hint = type_hint;
    // 创建一个空的候选类型列表
    std::vector<TypePtr> all_candidates = {};

    // 如果 refined_type_hint 存在
    if (refined_type_hint) {
      // 定义一个 lambda 函数 do_if_type_match，用于设置 list_value 的类型为 refined_type_hint
      auto do_if_type_match = [&]() { list_value->setType(refined_type_hint); };

      // 定义一个 lambda 函数 type_match，用于检查类型是否为 AnyListType::get() 的子类型
      auto type_match = [&](const TypePtr& t) {
        return t->isSubtypeOf(AnyListType::get());
      };

      // 调用 refineAndSetUnionTypeHintOrPopulateCandidatesVector 函数，根据条件设置 refined_type_hint 或填充候选类型列表
      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "List",
          lc,
          type_match,
          do_if_type_match,
          do_if_type_match);
    }

    // 初始化 seen_first_elem 为 false

    // 创建一个新节点 n，类型为 prim::ComprehensionScope，范围为 lc.range()，深度为 0
    Node* n =
        graph->insertNode(create(prim::ComprehensionScope, lc.range(), 0));
    // 为节点 n 添加一个新的 block 作为 comprehension_block
    auto* comprehension_block = n->addBlock();
    // 将 comprehension_block 推入堆栈，作为当前帧的一部分
    pushFrame(comprehension_block);
    // 设置插入点为 comprehension_block 的最后一条语句之后
    WithInsertPoint guard(comprehension_block);
    // 调用 emitFor 函数，对 targets_list 和 itrs 进行迭代，生成循环体 emit_body
    emitFor(targets_list, itrs, loc, emit_body);
    // 弹出当前帧
    popFrame();
    // 返回 list_value 作为函数的结果
    return list_value;
  }

  // 根据给定的 DictComp 和类型提示 type_hint，生成字典推导式的值
  Value* emitDictComprehension(const DictComp& dc, const TypePtr& type_hint) {
    // 获取字典推导式的范围
    const auto loc = dc.range();
    // 创建一个包含 dc.target() 的 targets_list
    const auto targets_list = List<Expr>::create(dc.range(), {dc.target()});
    // 创建一个包含 dc.iter() 的 itrs
    const auto itrs = List<Expr>::create(dc.range(), {dc.iter()});

    // 创建一个新节点，值为 prim::DictConstruct 的字典值
    Value* dict_value =
        graph->insertNode(graph->create(prim::DictConstruct, 1))->output();

    // 将 dict_value 的默认类型设置为 Dict[str, Tensor]
    dict_value->setType(DictType::create(StringType::get(), TensorType::get()));

    // 复制 type_hint 作为 refined_type_hint
    TypePtr refined_type_hint = type_hint;
    // 如果 type_hint 存在并且是联合类型，则将 annotated_union_type 设置为 type_hint，否则为 nullptr
    TypePtr annotated_union_type =
        type_hint && type_hint->isUnionType() ? type_hint : nullptr;

    // 创建一个空的候选类型列表
    std::vector<TypePtr> all_candidates = {};

    // 如果 refined_type_hint 存在
    if (refined_type_hint) {
      // 定义一个 lambda 函数 type_match，用于检查类型是否为 DictType::Kind
      auto type_match = [&](const TypePtr& t) {
        return t->kind() == DictType::Kind;
      };

      // 定义一个 lambda 函数 do_if_match，用于设置 dict_value 的类型为 refined_type_hint
      auto do_if_match = [&]() { dict_value->setType(refined_type_hint); };

      // 调用 refineAndSetUnionTypeHintOrPopulateCandidatesVector 函数，根据条件设置 refined_type_hint 或填充候选类型列表
      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "Dict",
          dc,
          type_match,
          do_if_match,
          do_if_match);
    }

    // 初始化 first_generated_key_type 和 first_generated_value_type 为 nullptr

    // 创建一个新节点 n，类型为 prim::ComprehensionScope，范围为 dc.range()，深度为 0
    Node* n =
        graph->insertNode(create(prim::ComprehensionScope, dc.range(), 0));
    // 为节点 n 添加一个新的 block 作为 comprehension_block
    auto* comprehension_block = n->addBlock();
    // 将 comprehension_block 推入堆栈，作为当前帧的一部分
    pushFrame(comprehension_block);
    // 设置插入点为 comprehension_block 的最后一条语句之后
    WithInsertPoint guard(comprehension_block);
    // 调用 emitFor 函数，对 targets_list 和 itrs 进行迭代，生成循环体 emit_body
    emitFor(targets_list, itrs, loc, emit_body);
    // 弹出当前帧
    popFrame();
    // 如果 annotated_union_type 不为空，则进行下面的操作
    if (annotated_union_type) {
      // 在计算图中插入一个节点，节点操作为 prim::unchecked_cast，输入为 dict_value
      Node* n = graph->insertNode(graph->create(prim::unchecked_cast, {dict_value}));
      // 设置节点的输出类型为 annotated_union_type，并移动该类型对象的所有权
      n->output()->setType(std::move(annotated_union_type));
      // 更新 dict_value 为节点 n 的输出值
      dict_value = n->output();
    }
    
    // 返回 dict_value
    return dict_value;
    }
    
    // 插入子类型细化信息
    void insertRefinements(const SourceRange& loc, const RefinementSet& ref) {
      // 遍历 ref 中的每一个活跃细化对象 r
      for (const Refinement& r : ref.activeRefinements()) {
        // 在环境栈中查找标识符 r.identifier() 对应的值 v
        Value* v = environment_stack->getVar(r.identifier(), loc);
        // 在计算图中插入一个 unchecked_cast 操作，将值 v 转换为类型 r.type()，返回新的值 new_v
        Value* new_v = graph->insertUncheckedCast(v, r.type());
        // 在环境栈中设置标识符 r.identifier() 对应的值为 new_v
        environment_stack->setVar(loc, r.identifier(), new_v);
      }
    }
    
    // 发出短路逻辑运算
    CondValue emitShortCircuitLogical(
        const SourceRange& loc,
        const Expr& first_expr,
        const Expr& second_expr,
        bool is_or) {
      // 发出第一个表达式的条件值
      CondValue lhs = emitCondExpr(first_expr);
    
      // 插入常量表达式以便于优化，如果是 OR 运算则为 true，否则为 false
      auto get_const_expr = [&] { return graph->insertConstant(is_or, loc); };
    
      // 定义一个可选类型 rhs 存储第二个表达式的条件值
      std::optional<CondValue> rhs;
      // 获取第二个表达式的条件值
      auto get_continue_expr = [&] {
        rhs = emitCondExpr(second_expr);
        return rhs->value();
      };
    
      // 定义新结果的值
      Value* new_result;
      // 定义一个可选类型 refinements 存储细化集合
      std::optional<RefinementSet> refinements;
      // 定义一个可选类型 static_if 存储静态 if 信息
      std::optional<bool> static_if;
    
      // 如果是 OR 运算
      if (is_or) {
        // 使用 emitIfExpr 发出 If 表达式，如果 lhs 为真则返回 get_const_expr 结果，否则返回 get_continue_expr 结果
        new_result = emitIfExpr(loc, lhs, get_const_expr, get_continue_expr);
        // 计算 refinements，为 lhs 和 rhs 的细化集合的并集
        refinements = lhs.refinements().Or(rhs->refinements());
        // 如果 lhs 或 rhs 任一具有静态 if 信息，则 static_if 为 true
        if ((lhs.staticIf() && *lhs.staticIf()) ||
            (rhs->staticIf() && *rhs->staticIf())) {
          static_if = true;
        } else if (lhs.staticIf() && rhs->staticIf()) {
          static_if = *lhs.staticIf() || *rhs->staticIf();
        }
      } else { // 如果是 AND 运算
        // 使用 emitIfExpr 发出 If 表达式，如果 lhs 为真则返回 get_continue_expr 结果，否则返回 get_const_expr 结果
        new_result = emitIfExpr(loc, lhs, get_continue_expr, get_const_expr);
        // 计算 refinements，为 lhs 和 rhs 的细化集合的交集
        refinements = lhs.refinements().And(rhs->refinements());
        // 如果 lhs 和 rhs 同时具有静态 if 信息，则 static_if 为 true
        if (((lhs.staticIf() && !*lhs.staticIf()) ||
             (rhs->staticIf() && !*rhs.staticIf()))) {
          static_if = false;
        } else if (lhs.staticIf() && rhs->staticIf()) {
          static_if = *lhs.staticIf() && *rhs.staticIf();
        }
      }
      
      // 返回条件值对象 CondValue，包含 new_result、refinements 和 static_if
      return CondValue(new_result, std::move(*refinements), static_if);
    }
    
    // 发出 If 表达式
    Value* emitIfExpr(
        const SourceRange& range,
        const CondValue& cond_value,
        const std::function<Value*()>& true_expr,
        const std::function<Value*()>& false_expr) {
      // 在计算图中插入一个 If 节点，范围为 range，初始输入数量为 0
      Node* n = graph->insertNode(create(prim::If, range, 0));
      // 将条件值的值作为 If 节点的输入
      n->addInput(cond_value.value());
    // 创建一个新的基本块用于存放 if 语句的真实执行路径
    auto* true_block = n->addBlock();
    // 创建一个新的基本块用于存放 if 语句的假执行路径
    auto* false_block = n->addBlock();

    // 定义一个 lambda 函数 emit_if_expr，用于生成 if 表达式的代码块
    auto emit_if_expr = [this, &range](
                            Block* b,
                            const RefinementSet& refinements,
                            const std::function<Value*()>& expr_value) {
      pushFrame(b); // 将当前块推入帧栈
      WithInsertPoint guard(b); // 设置插入点为当前块
      insertRefinements(range, refinements); // 插入细化信息
      Value* out_val = expr_value(); // 执行表达式并获取结果
      b->registerOutput(out_val); // 在当前块中注册输出值
      popFrame(); // 弹出当前帧栈
    };

    // 生成真实执行路径的代码块
    emit_if_expr(true_block, cond_value.refinements(), true_expr);
    // 生成假执行路径的代码块
    emit_if_expr(false_block, cond_value.refinements().Not(), false_expr);

    // 获取真实执行路径的输出类型
    auto true_type = true_block->outputs().at(0)->type();
    // 获取假执行路径的输出类型
    auto false_type = false_block->outputs().at(0)->type();
    // 统一两个路径的输出类型
    auto unified = unifyTypes(true_type, false_type);
    // 如果无法统一输出类型，则抛出错误
    if (!unified) {
      throw ErrorReport(range)
          << "if-expression's true branch has type " << true_type->repr_str()
          << " but false branch has type " << false_type->repr_str();
    }

    // 添加操作的输出
    auto expr_value = n->addOutput()->setType(*unified); // 结果值

    // 返回生成的表达式值
    return expr_value;
  }
  Value* emitToBool(const SourceRange& loc, Value* v) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* out;
    try {
      auto bool_cast = environment_stack->getSugaredVar("bool", loc);
      // 调用环境栈中的 bool 类型转换方法，并获取简单的结果
      out = asSimple(bool_cast->call(loc, method, {v}, {}, 0));
    } catch (...) {
      throw ErrorReport(loc) << "Could not cast value of type "
                             << v->type()->repr_str() << " to bool";
    }
    // 检查结果值是否为布尔类型
    if (!out->type()->isSubtypeOf(*BoolType::get())) {
      throw ErrorReport(loc)
          << "expected a bool expression for condition but found "
          << out->type()->repr_str();
    }
    // 返回转换后的布尔值
    return out;
  }

  void emitIfElseBlocks(
      const SourceRange& loc,
      const CondValue& cond_value,
      const List<Stmt>& trueBranch,
      const List<Stmt>& falseBranch) {
    // 这是一个静态 if 语句，即在静态条件下只包含真或假执行路径的子集。
    // 用于元编程模块，例如，当子模块不存在时，可以使用 None 检查以确保
    // 不编译到 None 检查的访问，从而避免出错。
    if (cond_value.staticIf()) {
      // 如果条件为静态真，则插入真实执行路径的细化信息并生成真实执行路径的代码
      if (*cond_value.staticIf()) {
        insertRefinements(loc, cond_value.refinements());
        emitStatements(trueBranch);
      } else {
        // 否则插入假执行路径的细化信息并生成假执行路径的代码
        insertRefinements(loc, cond_value.refinements().Not());
        emitStatements(falseBranch);
      }
      return;
    }

    // 否则创建一个新的 if 节点并添加输入值
    Node* n = graph->insertNode(create(prim::If, loc, 0));
    n->addInput(cond_value.value());
    // 添加真实执行路径的基本块
    auto* true_block = n->addBlock();
    // 添加假执行路径的基本块
    auto* false_block = n->addBlock();

    // 一次性生成两个基本块，以获取所有变异值的联合
    // 调用 emitSingleIfBranch 函数，生成 true 分支的代码，返回保存的结果
    auto save_true =
        emitSingleIfBranch(true_block, trueBranch, cond_value.refinements());
    // 调用 emitSingleIfBranch 函数，生成 false 分支的代码，返回保存的结果
    auto save_false = emitSingleIfBranch(
        false_block, falseBranch, cond_value.refinements().Not());

    // 检查 true 分支和 false 分支是否都是出口块，如果是，则将当前块加入出口块集合
    bool true_exits = exit_blocks.count(true_block);
    bool false_exits = exit_blocks.count(false_block);
    if (true_exits && false_exits) {
      exit_blocks.insert(n->owningBlock());
    }

    // 在 Python 中，每个在 if 语句中赋值的变量都会逃出 if 语句的作用域，
    // 所有变量都在函数作用域内。Script 是 Python 的子集：我们认为变量在 if 语句的所有路径中都有定义。
    // 如果一个变量只在一个分支中定义，则保存错误以防稍后使用。
    // ordered set，因为我们希望图形输出是确定性的。
    std::set<std::string> mutated_variables;

    // 当访问 true 或 false 环境时，需要设置插入点，以便 prim::Load 插入到正确的块中。
    // 如果变量只在一个分支中定义，则保存错误以防稍后使用。
    for (auto& v : save_true->definedVariables()) {
      {
        // 设置插入点到 false 分支
        WithInsertPoint insert(false_block);
        // 如果在任何帧中查找到 save_false 中的变量 v，或者 false 分支是出口，则将变量插入到 mutated_variables 中
        if (save_false->findInAnyFrame(v) || false_exits) {
          mutated_variables.insert(v);
        } else {
          // 如果在 false 分支中找不到变量 v，则报告错误
          if (reportSourceLocation(loc.source()->size())) {
            ErrorReport error(loc);
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              error << v << " is not defined in the false branch";
              return error.what();
            });
          } else {
            // 如果源文件过大而消除了源信息，则给出相应提示
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              std::stringstream ss;
              ss << v << " is not defined in the false branch. "
                 << "The source info is eliminated due to the source file is too large. "
                 << "To get it back, please set PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION=1 "
                 << "as env var";
              return ss.str();
            });
          }
        }
      }
    }
    // 遍历 save_false 对象的所有定义变量
    for (auto& v : save_false->definedVariables()) {
      {
        // 设置插入点为 true_block
        WithInsertPoint insert(true_block);
        // 如果在 save_true 或者 true_exits 中找到变量 v
        if (save_true->findInAnyFrame(v) || true_exits) {
          // 将变量 v 加入到 mutated_variables 集合中
          mutated_variables.insert(v);
        } else {
          // 如果未找到变量 v，则执行以下分支
          // 检查是否需要报告源位置信息
          if (reportSourceLocation(loc.source()->size())) {
            // 报告错误信息，指定源位置 loc
            ErrorReport error(loc);
            // 设置变量 v 的类型错误信息，返回一个函数描述错误信息
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              error << v << " is not defined in the true branch";
              return error.what();
            });
          } else {
            // 设置变量 v 的类型错误信息，返回一个函数描述错误信息，源信息因文件过大而被消除
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              std::stringstream ss;
              ss << v << " is not defined in the false branch. "
                 << "The source info is eliminated due to the source file is too large. "
                 << "To get it back, please set PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION=1 "
                 << "as env var";
              return ss.str();
            });
          }
        }
      }
    }

    // 注册每个块中的输出结果
    }
  }

  // 发出检查对象是否具有属性的操作
  CondValue emitHasAttr(const Expr& objExpr, const Expr& attrExpr) {
    // 发出对 objExpr 的含糊表达式，深度为 1
    auto obj = emitSugaredExpr(objExpr, 1);
    // 如果 attrExpr 不是字符串字面量，抛出错误报告
    if (attrExpr.kind() != TK_STRINGLITERAL) {
      throw ErrorReport(attrExpr)
          << "hasattr's second argument must be a string literal";
    }
    // 获取字符串字面量的文本内容作为属性名
    const std::string& name = StringLiteral(attrExpr).text();
    // 检查 obj 是否具有指定属性名的属性
    const bool hasAttr = obj->hasAttr(objExpr.range(), method, name);
    // 返回一个条件值对象，表示是否具有属性
    return CondValue(*graph, objExpr.range(), hasAttr, {});
  }

  // 发出检查对象是否是某个类的实例的操作
  CondValue emitIsInstance(const Expr& obj, const Expr& classinfo) {
    // 发出 obj 表达式并获取其值
    Value* lhs_val = emitExpr(obj);
    // 创建左侧和右侧类型向量
    std::vector<TypePtr> lhs_types;
    std::vector<TypePtr> rhs_types;

    // 用于收集右侧类型的函数
    std::function<void(const Expr&)> gather_rhs = [&](const Expr& expr) {
      // 如果表达式类型为元组字面量，则遍历其中的每个表达式
      if (expr.kind() == TK_TUPLE_LITERAL) {
        for (Expr e : TupleLiteral(expr).inputs()) {
          gather_rhs(e);
        }
        return;
      }
      // 解析表达式并将其类型添加到 rhs_types 中
      TypePtr type = typeParser_.parseTypeFromExpr(expr);
      rhs_types.emplace_back(type);
    };

    // 将左侧值的类型添加到 lhs_types 中
    lhs_types.push_back(lhs_val->type());
    // 收集 classinfo 的右侧类型
    gather_rhs(classinfo);

    // 标准化向量以便进行联合操作
    standardizeVectorForUnion(&lhs_types);
    standardizeVectorForUnion(&rhs_types);

    // 创建 RefinementSet 对象
    RefinementSet refinement;

    // 初始化变量
    TypePtr unified_true = nullptr;
    TypePtr unified_false = nullptr;

    // 初始化类型向量
    std::vector<TypePtr> isinstance_types;
    std::vector<TypePtr> not_isinstance_types;

    // 初始化精炼向量
    std::vector<Refinement> true_refinements;
    std::vector<Refinement> false_refinements;

    // 是否所有左侧类型为某些右侧类型的子类型
    bool all_lhs_subtype_some_rhs = true;

    // 我们可以丢弃任何我们静态知道是不可能的 rhs 类型。
    // 例如，如果我们有:
    //
    //    def fn(x: Optional[str]):
    //        if isinstance(x, (List[str], str, int)):
    //            ...
    //
    // 那么在真分支中 `x` 将是 `str`，在假分支中将是 `None`，
    // 而不是真分支中是 `(List[str], str, int)`，假分支中是 `None`
    // 的情况
    // 遍历 lhs_types 中的每一个类型指针 lhs_type
    for (const TypePtr& lhs_type : lhs_types) {
      // 检查 lhs_type 是否为 AnyType::get()，即是否为任意类型
      if (lhs_type == AnyType::get()) {
        // 如果是任意类型，则将 rhs_types 中的所有类型添加到 isinstance_types 中
        isinstance_types.insert(
            isinstance_types.end(), rhs_types.begin(), rhs_types.end());
        // 将 AnyType::get() 添加到 not_isinstance_types 中作为特例
        not_isinstance_types.emplace_back(AnyType::get());
        // 特例情况：如果 isinstance_types 中除了 AnyType::get() 还有其他类型，则 all_lhs_subtype_some_rhs 置为 false
        if (isinstance_types.size() != 1 ||
            isinstance_types[0] != AnyType::get()) {
          all_lhs_subtype_some_rhs = false;
        }
        break;  // 跳出循环，因为已经确定 lhs_type 是任意类型
      }

      // 定义 lambda 函数 get_smaller_type，用于获取两个类型指针中更小的一个
      auto get_smaller_type = [&](const TypePtr& t1,
                                  const TypePtr& t2) -> TypePtr {
        // 如果 t1 是 t2 的子类型，则返回 t1
        if (t1->isSubtypeOf(*t2)) {
          return t1;
        // 如果 t2 是 t1 的子类型，则返回 t2
        } else if (t2->isSubtypeOf(*t1)) {
          return t2;
        // 否则返回空指针表示两者无法比较
        } else {
          return nullptr;
        }
      };

      // 初始化 found_refinement 为空指针
      TypePtr found_refinement = nullptr;
      // 遍历 rhs_types 中的每一个类型指针 rhs_type
      for (const TypePtr& rhs_type : rhs_types) {
        // 获取 lhs_type 和 rhs_type 中更小的类型
        TypePtr maybe_smaller_type = get_smaller_type(lhs_type, rhs_type);
        // 如果没有更小的类型，继续下一个循环
        if (!maybe_smaller_type) {
          continue;
        // 如果 maybe_smaller_type 是 lhs_type
        } else if (*maybe_smaller_type == *lhs_type) {
          // 处理类似于 lhs = `List[str]` 和 rhs = `list` 的情况，找到更精确的 lhs_type
          found_refinement = lhs_type;
        // 如果 maybe_smaller_type 是 rhs_type
        } else if (*maybe_smaller_type == *rhs_type) {
          // 找到最窄的可能类型
          found_refinement = found_refinement
              ? *(unifyTypes(found_refinement, rhs_type))  // 如果已有 found_refinement，则尝试统一它们
              : rhs_type;  // 否则将 rhs_type 设为 found_refinement
        }
      }

      // 如果 found_refinement 不为空
      if (found_refinement) {
        // 如果 found_refinement 和 lhs_type 相同
        if (*found_refinement == *lhs_type) {
          // 更新 all_lhs_subtype_some_rhs 为 true
          all_lhs_subtype_some_rhs &= true;
        }
        // 将 found_refinement 添加到 isinstance_types 中
        isinstance_types.push_back(found_refinement);
      } else {
        // 如果无法将 lhs_type 视为 rhs 的子类型（或无法“精炼”为自身，如上述的 `List[str]` 和 `list` 情况），则将 lhs_type 添加到 not_isinstance_types 中
        not_isinstance_types.push_back(lhs_type);
        // 将 all_lhs_subtype_some_rhs 设为 false
        all_lhs_subtype_some_rhs = false;
      }
    }

    // 用于 `unifyTypeList` 的 std::stringstream，命名为 nowhere
    std::stringstream nowhere;

    // 获取 true 分支和 false 分支的单一类型
    if (!isinstance_types.empty()) {
      // 调用 unifyTypeList 函数，将 isinstance_types 统一为一个类型
      unified_true =
          *unifyTypeList(isinstance_types, nowhere, /*default_to_union=*/true);
    }
    // 如果 obj 的种类为 TK_VAR 并且 unified_true 不为空
    if (obj.kind() == TK_VAR && unified_true) {
      // 获取变量的标识符
      std::string ident = Var(obj).name().name();
      // 更新 true_refinements，添加一个新的 Refinement 对象
      true_refinements = {Refinement(ident, unified_true)};
    }

    // 获取 true 分支和 false 分支的单一类型
    if (!not_isinstance_types.empty()) {
      // 调用 unifyTypeList 函数，将 not_isinstance_types 统一为一个类型
      unified_false = *unifyTypeList(
          not_isinstance_types, nowhere, /*default_to_union=*/true);
    }
    // 如果 obj 的种类为 TK_VAR 并且 unified_false 不为空
    if (obj.kind() == TK_VAR && unified_false) {
      // 获取变量的标识符
      std::string ident = Var(obj).name().name();
      // 更新 false_refinements，添加一个新的 Refinement 对象
      false_refinements = {Refinement(ident, unified_false)};
    }
    // 创建一个RefinementSet对象，使用给定的true_refinements和false_refinements
    refinement = RefinementSet(true_refinements, false_refinements);

    // 检查isinstance_types是否为空，判断是否为静态假
    bool is_statically_false = isinstance_types.empty();

    // 如果语句在静态上下文中为真
    if (all_lhs_subtype_some_rhs) {
      // 返回一个CondValue对象，表示条件为真的情况，包括范围、真值标记和refinement的移动语义
      return CondValue(*graph, obj.range(), true, std::move(refinement));
    }

    // 如果在静态上下文中为假
    if (is_statically_false) {
      // 返回一个CondValue对象，表示条件为假的情况，包括范围、假值标记和refinement的移动语义
      return CondValue(*graph, obj.range(), false, std::move(refinement));
    }

    // 在运行时检查可能的真/假情况，需要一个实际的操作
    // 插入一个IsInstance节点到计算图中，使用lhs_val和rhs_types，并获取其输出
    Value* result =
        graph->insertNode(graph->createIsInstance(lhs_val, rhs_types))
            ->output();
    
    // 返回一个CondValue对象，表示运行时条件结果的情况，包括结果值、refinement的移动语义和空的可选值
    return CondValue(result, std::move(refinement), c10::nullopt);
  }

  // 发射一个If语句
  void emitIf(const If& stmt) {
    // 获取条件表达式
    Expr cond = stmt.cond();
    // 发射条件表达式并获取其CondValue
    CondValue cond_value = emitCondExpr(cond);
    // 发射If-Else块，根据条件值分别处理true分支和false分支
    emitIfElseBlocks(
        stmt.range(), cond_value, stmt.trueBranch(), stmt.falseBranch());
  }

  // *********************** Loop Operators ************************************
  // 发射一个循环操作符，形式如下：
  // Loop(max_trip_count)
  // block0(loop_counter) {
  //   <body>
  // }
  // block1 {
  //   <loop condition>
  //   -> (condition)
  // }
  // 对于for循环，会有一个空的循环条件块，并且条件被设置为true。
  // 在转换为SSA过程中，循环条件将被正确地内联，同时添加输入和输出，使循环符合语义规范
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
  void emitLoopCommon(
      const SourceRange& range,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iter_val,
      std::optional<List<Expr>> targets,
      std::optional<Expr> cond) {
    // 初始化max_trip_count_val为nullptr
    Value* max_trip_count_val = nullptr;
    // 如果迭代值不为空，则计算其长度
    if (iter_val != nullptr) {
      max_trip_count_val = iter_val->len(range, method);
    } else {
      // 否则，将max_trip_count_val设置为int64_t的最大值常量
      max_trip_count_val = materializeConstant(
          std::numeric_limits<int64_t>::max(),
          *graph,
          range,
          integral_constants);
    }

    // 在计算图中插入一个Loop节点
    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    auto* body_block = n->addBlock();
    {
      Block* condition_block = n->addBlock();
      pushFrame(condition_block);
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      Value* out;
      if (cond) {
        // 如果有条件，则在条件块中插入对应的条件表达式
        WithInsertPoint insert(condition_block);
        out = emitToBool(cond.value().range(), emitExpr(cond.value()));
      } else {
        // 否则，在当前节点插入一个常量true
        WithInsertPoint insert(n);
        out = graph->insertConstant(true, range);
      }
      // 将条件块的输出注册为out
      condition_block->registerOutput(out);
      popFrame();
    }
    // 将max_trip_count_val作为Loop节点的输入
    n->addInput(max_trip_count_val);

    // 设置循环状态为IN_LOOP
    WithLoopStatus loop_guard(&loop_status_, LoopStatus::IN_LOOP);
    // 将body_block的第一个输入添加为迭代次数的值，类型为IntType
    Value* trip_count =
        body_block->addInput()->setType(IntType::get()); // Iteration num
    {
      pushFrame(body_block);
      // 将当前的代码块压入堆栈，准备执行循环体
    
      WithInsertPoint guard(body_block);
      // 设置当前插入点为指定的代码块的起始位置
    
      // 如果 FOR 循环的迭代器和目标表达式均已提供，则生成目标变量的赋值操作
      if (iter_val != nullptr && targets) {
        // 获取当前元素的值，并将其转换为 Value 类型
        Value* cur_elem = iter_val->getitem(range, method, trip_count)
                              ->asValue(range, method);
        // 将其包装为 SimpleValue 类型的 SugaredValuePtr
        SugaredValuePtr sv = std::make_shared<SimpleValue>(cur_elem);
    
        // 获取目标表达式列表
        List<Expr> target_exprs = targets.value();
        // 验证赋值左侧表达式的有效性
        validateAssignLhsExpr(target_exprs, range);
    
        // 如果目标表达式数量大于 1，表示左侧有多个变量需要解构赋值
        // 我们创建一个 TupleLiteral 来包装这些目标表达式
        if (target_exprs.size() > 1) {
          Expr tl = TupleLiteral::create(range, target_exprs);
          target_exprs = List<Expr>::create(range, {tl});
        }
    
        // 发出表达式赋值的操作
        emitExprsAssign(target_exprs, {sv}, range, /*n_binders=*/1);
      }
    
      // 发出循环体的代码
      emit_body();
      // 弹出当前代码块的堆栈
      popFrame();
    }
    
    void emitUnrolledLoop(
        const SourceRange& loc,
        const std::function<void()>& emit_body,
        const SugaredValuePtr& iterable,
        const List<Expr>& targets) {
      // 获取迭代器的静态长度
      auto static_len = iterable->staticLen();
      // 断言迭代器应具有静态长度
      TORCH_INTERNAL_ASSERT(
          static_len, "Unrolled loop iter should have static length");
      // 获取迭代器的长度
      int64_t len = *static_len;
    
      // 设置循环状态为 IN_UNROLLED_LOOP，以便在循环中追踪其状态
      WithLoopStatus loop_guard(&loop_status_, LoopStatus::IN_UNROLLED_LOOP);
    
      // 遍历长度为 len 的迭代器
      for (const auto i : c10::irange(len)) {
        // 根据索引创建常量值
        auto index =
            materializeConstant(i, *method.graph(), loc, integral_constants);
        // 获取迭代器中索引处的值
        auto sugared_value = iterable->getitem(loc, method, index);
        // 发出表达式赋值的操作
        emitExprsAssign(
            targets, {sugared_value}, targets.range(), /*n_binders=*/1);
        // 发出循环体的代码
        emit_body();
      }
    }
    
    void emitFor(
        const List<Expr>& targets,
        const List<Expr>& itrs,
        const SourceRange& loc,
        const std::function<void()>& emit_body) {
      // 如果迭代器数量不为 1，则抛出错误
      if (itrs.size() != 1) {
        throw ErrorReport(loc) << "List of iterables is not supported currently";
      }
    
      // 发出迭代器表达式的操作，并获取其对应的 SugaredValuePtr
      SugaredValuePtr sv = emitSugaredExpr(itrs[0], 1);
      // 调用迭代器的 iter 方法，获取可迭代对象
      SugaredValuePtr iterable = sv->iter(loc, method);
    
      // 如果迭代器不支持展开循环，则使用普通循环处理
      if (!iterable->shouldEmitUnrolled()) {
        emitLoopCommon(loc, emit_body, iterable, targets, {});
      } else {
        // 否则，使用展开循环处理
        emitUnrolledLoop(loc, emit_body, iterable, targets);
      }
    }
  // 为给定的 for 循环语句生成代码
  void emitFor(const Targets& targets, const Iters& itrs, const Range& range, const std::function<void()>& emit_body) {
    // 实现 for 循环的常见代码逻辑
    emitLoopCommon(range, emit_body, nullptr, {}, nullptr);
  }

  // 为给定的 while 循环语句生成代码
  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();  // 获取循环条件表达式
    auto emit_body = [&]() { emitStatements(stmt.body()); };  // 定义一个 lambda 函数，用于生成循环体代码
    emitLoopCommon(stmt.range(), emit_body, nullptr, {}, cond);  // 调用通用的循环生成函数
  }

  // 为给定的 with 语句生成代码
  void emitWith(const With& stmt) {
    auto targets = stmt.targets();  // 获取 with 语句中的目标列表

    // 维护一个进入对象的堆栈，以确保正确的退出顺序
    std::stack<Value*> entered;

    for (const auto& target : targets) {
      Expr e = target.target();  // 获取目标表达式

      auto* rhs = emitExpr(e);  // 生成目标表达式的代码，并获取其结果值
      auto* n = graph->insertNode(graph->create(prim::Enter, {rhs}));  // 插入一个进入节点到计算图中
      entered.push(rhs);  // 将生成的结果值入栈

      // 检查目标表达式返回的对象是否是类对象
      if (rhs->type()->kind() != TypeKind::ClassType) {
        throw ErrorReport(e.range())
            << "With item expression must return an object";  // 抛出错误，要求目标表达式返回一个对象
      }

      auto rhsClass = rhs->type()->expect<ClassType>();  // 获取类对象的类型信息
      auto* enterMethod = rhsClass->findMethod("__enter__");  // 查找 __enter__ 方法
      auto* exitMethod = rhsClass->findMethod("__exit__");  // 查找 __exit__ 方法

      // 检查是否找到了 __enter__ 和 __exit__ 方法
      if (!enterMethod || !exitMethod) {
        throw ErrorReport(e.range())
            << "Object returned by with item expression does not define __enter__ and __exit__ methods";  // 抛出错误，要求对象定义了 __enter__ 和 __exit__ 方法
      }

      // 检查 __enter__ 方法的参数和返回值的结构
      auto& enterSchema = enterMethod->getSchema();
      if (enterSchema.arguments().size() != 1) {
        throw ErrorReport(e.range())
            << "__enter__ must have only one argument and one return value";  // 抛出错误，要求 __enter__ 方法只有一个参数和一个返回值
      }

      // 检查 __exit__ 方法的参数结构
      auto& exitSchema = exitMethod->getSchema();
      if (exitSchema.arguments().size() != 4) {
        throw ErrorReport(e.range()) << "__exit__ must have four arguments";  // 抛出错误，要求 __exit__ 方法有四个参数
      } else {
        // 检查 __exit__ 方法的参数类型是否为 AnyType
        for (unsigned i = 1; i < 4; ++i) {
          if (exitSchema.arguments().at(i).type() != AnyType::get()) {
            throw ErrorReport(e.range())
                << "argument " << i
                << " of __exit__ must have Any type; TorchScript does not currently support passing exception type, value, or traceback to the __exit__ function.";  // 抛出错误，指出 __exit__ 方法的参数必须是 AnyType 类型
          }
        }
      }

      // 将进入节点的输出类型设置为 __enter__ 方法的返回类型
      n->output(0)->setType(enterSchema.returns().at(0).type());

      // 如果目标有变量名，则将 __enter__() 的返回值赋给变量
      if (target.var().present()) {
        Var i = target.var().get();
        environment_stack->setVar(i.range(), i.name().name(), n->output(0));  // 设置环境栈中的变量
      }
    }

    emitStatements(stmt.body());  // 生成 with 语句体的代码

    // 插入所有对应的 prim::Exit 节点
    while (!entered.empty()) {
      auto* input = entered.top();
      entered.pop();
      auto* n = graph->create(prim::Exit);  // 创建 prim::Exit 节点
      graph->insertNode(n);  // 将节点插入计算图中
      n->addInput(input);  // 将进入节点的输出作为 prim::Exit 节点的输入
    }
  }

  // 目前不支持将异常赋给变量
  void emitRaise(const Raise& raise) {
    // 调用函数 `emitSugaredExpr` 处理 `raise.expr()` 并返回一个包装过的表达式对象
    auto sv = emitSugaredExpr(raise.expr(), 1);
    // 初始化两个指针，用于存储错误信息和限定类名
    Value* error_message = nullptr;
    Value* qualified_class_name = nullptr;

    // 检查 `sv` 是否是 `ExceptionMessageValue` 类型的实例
    if (auto exception_instance =
            std::dynamic_pointer_cast<ExceptionMessageValue>(sv)) {
      // 典型情况，抛出异常类的实例，例如 `raise RuntimeError("error")`
      error_message = exception_instance->getValue();
      qualified_class_name = exception_instance->getQualifiedClassName();
    } else if (
        auto exception_class = std::dynamic_pointer_cast<ExceptionValue>(sv)) {
      // 抛出一个裸露的异常，例如 `raise RuntimeError`
      error_message = insertConstant(*graph, "", raise.range());
    } else {
      // `raise` 后面没有异常实例（例如 `raise "error"` 而不是 `raise RuntimeError("error")`）
      throw ErrorReport(raise.range())
          << "exceptions must derive from BaseException";
    }

    // 如果 `error_message` 的类型不是字符串类型的子类型，则将其转换为字符串类型
    if (!error_message->type()->isSubtypeOf(*StringType::get())) {
      error_message = graph->insert(aten::str, {error_message});
    }

    // 在计算图中插入 `prim::RaiseException` 操作，抛出异常
    graph->insert(
        prim::RaiseException,
        {error_message, qualified_class_name},
        {},
        raise.range());
    // 将当前块加入到退出块集合中
    exit_blocks.insert(environment_stack->block());
  }

  // 将断言表达式作为一个 if 分支来发出，这样断言可以重用消息
  void emitAssert(const Assert& stmt) {
    // 发出条件表达式并得到条件值
    CondValue cond_value = emitCondExpr(stmt.test());
    // 创建一个空的 `true_branch`，因为断言失败时不执行额外操作
    List<Stmt> true_branch = List<Stmt>::create(stmt.range(), {});
    // 创建一个 `AssertionError("the_message")` 的调用
    auto message = (stmt.msg().present())
        ? stmt.msg().get()
        : StringLiteral::create(stmt.range(), "");
    auto callee = Var::create(
        stmt.range(), Ident::create(stmt.range(), "AssertionError"));
    auto apply = Apply::create(
        stmt.range(),
        callee,
        List<Expr>::create(stmt.range(), {message}),
        List<Attribute>::create(stmt.range(), {}));

    // 创建一个包含 `Raise` 节点的 `false_branch`，用于在断言失败时抛出异常
    List<Stmt> false_branch =
        List<Stmt>::create(stmt.range(), {Raise::create(stmt.range(), apply)});
    // 发出 if-else 块来处理断言
    emitIfElseBlocks(stmt.range(), cond_value, true_branch, false_branch);
  }

  // 验证赋值语句的左值表达式是否有效
  // 1) 所有的左值表达式必须是 Var、Tuple 或 Starred 节点
  // 2) 左值表达式中最多只能有一个 Starred 节点
  // 3) 当有一个 Starred 节点时，必须存在另一个非 Starred 的左值表达式
  bool validateAssignLhsExpr(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    // 遍历 lhs 中的每一个赋值目标
    for (const auto& assignee : lhs) {
      // 检查赋值目标的类型，计算普通赋值目标的数量
      if (assignee.kind() == TK_VAR || assignee.kind() == TK_SUBSCRIPT ||
          assignee.kind() == TK_TUPLE_LITERAL || assignee.kind() == '.') {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        // 计算星号表达式的数量
        num_starred++;
      } else {
        // 如果赋值目标类型不支持抛出错误
        throw ErrorReport(assignee) << "lhs of assignment must be a variable, "
                                    << "subscript, or starred expression";
      }
    }

    // 检查是否有超过一个星号表达式，若是则抛出错误
    if (num_starred > 1) {
      throw ErrorReport(r)
          << "Only one starred expression is allowed on the lhs";
    }

    // 检查如果有星号表达式但没有普通赋值目标时抛出错误
    if (num_starred > 0 && num_normal_assign == 0) {
      throw ErrorReport(r) << "A Starred expression may only appear on the "
                           << "lhs within the presence of another non-starred"
                           << " expression";
    }

    // 返回星号表达式的数量作为结果
    return num_starred;
  }

  // 根据增强赋值语句获取对应的内置操作符
  // 如果 RHS 是张量，则返回相应的 ATen 原地操作
  // 如果是标量列表，则返回相应的列表增强操作
  Symbol getAugOp(const AugAssign& stmt, const TypePtr& type) {
    // 判断是否使用原地操作符
    bool use_inplace_op = type->isSubtypeOf(*TensorType::get()) ||
        type->kind() == TypeKind::ListType;
    // 根据不同的增强赋值操作符返回对应的操作符 Symbol
    switch (stmt.aug_op()) {
      case '+':
        return use_inplace_op ? aten::add_ : aten::add;
      case '-':
        return use_inplace_op ? aten::sub_ : aten::sub;
      case '/':
        return use_inplace_op ? aten::div_ : aten::div;
      case '*':
        return use_inplace_op ? aten::mul_ : aten::mul;
      case '%':
        return use_inplace_op ? aten::fmod_ : aten::fmod;
      case '|':
        return use_inplace_op ? aten::bitwise_or : aten::__or__;
      case '&':
        return use_inplace_op ? aten::bitwise_and : aten::__and__;
      case '^':
        return use_inplace_op ? aten::bitwise_xor : aten::__xor__;
      case TK_LSHIFT:
        // NOLINTNEXTLINE(bugprone-branch-clone)
        return use_inplace_op ? aten::__lshift__ : aten::__lshift__;
      case TK_RSHIFT:
        return use_inplace_op ? aten::__irshift__ : aten::__rshift__;
      case TK_POW:
        return aten::pow;
      default:
        // 如果增强赋值操作符未知，抛出错误
        throw ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
    }
  }

  // 获取一对 <原地魔术方法名, 非原地魔术方法名>
  // 如果原地方法不存在，则调用非原地方法
  std::pair<std::string, std::string> getAugMagicMethod(const AugAssign& stmt) {
  // 根据 stmt.aug_op() 的不同操作符，返回对应的魔术方法名对
  switch (stmt.aug_op()) {
    case '+':
      return std::make_pair(std::string("__iadd__"), std::string("__add__"));
    case '-':
      return std::make_pair(std::string("__isub__"), std::string("__sub__"));
    case '/':
      return std::make_pair(
          std::string("__itruediv__"), std::string("__truediv__"));
    case '*':
      return std::make_pair(std::string("__imul__"), std::string("__mul__"));
    case '%':
      return std::make_pair(std::string("__imod__"), std::string("__mod__"));
    default:
      // 如果操作符未知，则抛出错误报告
      throw ErrorReport(stmt)
          << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
  }
}

// 生成针对类参数或模块缓冲区的增量赋值节点，如 `+=`
void emitAugAssignment(const AugAssign& stmt) {
  switch (stmt.lhs().kind()) {
    case TK_VAR: {
      // 如果左侧表达式为变量，调用处理变量的增量赋值函数
      emitAugAssignmentToVar(stmt);
    } break;
    case '.': {
      // 如果左侧表达式为点操作符，调用处理选择变量的增量赋值函数
      emitAugAssignmentToSelectVar(stmt);
    } break;
    case TK_SUBSCRIPT: {
      // 如果左侧表达式为下标操作符，调用处理下标的增量赋值函数
      emitAugAssignmentToSubscript(stmt);
    } break;
    default:
      // 如果左侧表达式不符合预期，则抛出错误报告
      throw ErrorReport(stmt.lhs())
          << "unexpected expression on "
          << "left-hand side of augmented assignment";
  }
}

// 当类参数或模块缓冲区发生变化时，处理选择表达式作为左侧表达式的增量赋值
//
// 例如：
// class A(Module):
//  def __init__():
//    self.register_buffer("running_var", torch.zeros(1))
//
//  def forward():
//    self.num_batches += 1
void emitAugAssignmentToSelectVar(const AugAssign& stmt) {
  const auto lhs = Select(stmt.lhs());
  // 生成左侧选择表达式的糖化变量
  auto lhsSugaredVar = emitSugaredExpr(lhs.value(), 1);
  // 获取选择表达式的值
  const auto lhsValue =
      lhsSugaredVar->attr(lhs.range(), method, lhs.selector().name())
          ->asValue(lhs.range(), method);
  // 调用辅助函数处理增量赋值操作，并获取结果
  auto result = emitAugAssignmentHelper(stmt, lhsValue);
  // 设置选择表达式的属性为结果值
  lhsSugaredVar->setAttr(stmt.range(), method, lhs.selector().name(), result);
}

// 处理变量作为左侧表达式的增量赋值
void emitAugAssignmentToVar(const AugAssign& stmt) {
  const auto lhs = Var(stmt.lhs());
  // 生成左侧变量的表达式
  auto lhsValue = emitExpr(lhs);
  // 调用辅助函数处理增量赋值操作，并获取结果
  auto result = emitAugAssignmentHelper(stmt, lhsValue);
  // 在环境栈中设置变量的值
  environment_stack->setVar(lhs.range(), lhs.name().name(), result);
}

// 辅助函数：处理增量赋值操作的逻辑
Value* emitAugAssignmentHelper(const AugAssign& stmt, Value* lhs) {
    // 检查左操作数的类型是否为类类型
    if (lhs->type()->kind() == TypeKind::ClassType) {
      // 调用 `__iadd__` 方法，以便在类类型中进行原地更新
      // 参考 Python 文档：https://docs.python.org/3/reference/datamodel.html#object.__iadd__
      std::string in_place_method_name;
      std::string out_of_place_method_name;
      std::tie(in_place_method_name, out_of_place_method_name) =
          getAugMagicMethod(stmt);
      // 生成右操作数的值
      const auto rhs = emitExpr(stmt.rhs());

      // 确定是使用 __iadd__ 还是 __add__ 方法（仅当 __iadd__ 方法不存在时使用 __add__ 方法）
      auto type = lhs->type()->expect<ClassType>();
      std::string magic_method_name;
      if (type->findMethod(in_place_method_name)) {
        magic_method_name = in_place_method_name;
      } else if (type->findMethod(out_of_place_method_name)) {
        magic_method_name = out_of_place_method_name;
      } else {
        // 如果类未定义相应的方法，则抛出错误
        throw ErrorReport(stmt.range())
            << "Cannot emit inplace op on " << type->repr_str()
            << " since it does not define an " << in_place_method_name << " or "
            << out_of_place_method_name << " method";
      }

      // 执行原地增量赋值操作，等效于 x += y 被翻译为 x = x.__iadd__(y) 或者 x = x.__add__(y)（如果 __iadd__ 不存在）
      return MethodValue(lhs, magic_method_name)
          .call(stmt.range(), method, {rhs}, {}, 0)
          ->asValue(stmt.range(), method);
    } else {
      // 对于非类类型的左操作数，处理一般的增量赋值操作
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()))
                           .value(*method.graph());
      // 发出内置调用来处理增量操作
      return emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, lhs->type()),
          /*args=*/{lhs, rhs},
          /*kwargs=*/{},
          /*self=*/c10::nullopt);
    }
  }

  void emitAugAssignmentGeneric(
      const AugAssign& stmt,
      const Subscript& lhs,
      Value* sliceable) {
    // 获取要增量赋值的索引
    const auto subscriptExprs = lhs.subscript_exprs();
    const TypePtr type = sliceable->type();
    // 如果表达式被切片，抛出错误，因为目前不支持切片赋值
    if (subscriptExprs.size() != 1) {
      throw ErrorReport(subscriptExprs)
          << "Sliced expression not yet supported for " << type->repr_str()
          << " augmented assignment. "
          << "File a bug if you want this";
    }

    TypePtr elemType = nullptr;
    // 检查容器类型，确定元素类型
    if (const ListTypePtr listType = type->cast<ListType>()) {
      elemType = listType->getElementType();
    } else if (const DictTypePtr dictType = type->cast<DictType>()) {
      elemType = dictType->getKeyType();
    }

    // 如果无法确定元素类型，抛出错误
    if (elemType == nullptr) {
      throw ErrorReport(lhs)
          << type->repr_str() << " does not support augmented assignment.";
    }
    // 生成索引表达式的值
    const auto idxValue = emitExpr(subscriptExprs[0]);
    // 创建容器参数
    const auto containerArg =
        NamedValue(lhs.value().range(), type->str(), sliceable);
    // 创建索引参数
    const auto idxArg = NamedValue(subscriptExprs.range(), "idx", idxValue);
    // 创建值参数
    const auto valueArg =
        NamedValue(stmt.rhs().range(), "value", emitExpr(stmt.rhs()));
  // 获取 __getitem__ 的调用节点，用于获取容器中的元素
  const auto getItem = graph->insert(
      aten::__getitem__, {containerArg, idxArg}, {}, stmt.range());
  // 获取经过增强操作后的元素，如 += 或 -= 操作
  const auto augmentedItem = graph->insert(
      getAugOp(stmt, elemType), {getItem, valueArg}, {}, stmt.range());
  // 插入 _set_item 节点，用于在容器中设置经过增强操作后的元素
  graph->insert(
      aten::_set_item,
      {containerArg, idxArg, augmentedItem},
      {},
      stmt.range());
}

void emitAugAssignmentToSubscript(const AugAssign& stmt) {
  // 处理左侧列表值
  const auto lhs = Subscript(stmt.lhs());
  // 发射表达式以获取可切片对象
  const auto sliceable = emitExpr(lhs.value());

  if (sliceable->type()->isSubtypeOf(*TensorType::get())) {
    // 如果是张量，则完全评估切片操作并发射原地赋值操作
    std::vector<Value*> tensorIndices;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* sliced;
    // 发射整数和切片索引，获取切片的值和张量索引
    std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
        lhs.range(), sliceable, lhs.subscript_exprs());

    const auto slicedArg = NamedValue(stmt.lhs().range(), "self", sliced);
    const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
    if (tensorIndices.empty()) {
      // 常见情况：只尝试使用整数和切片进行索引。发射正确的增强赋值操作到切片的值
      emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, sliceable->type()),
          {rhs},
          {},
          slicedArg);
    } else {
      // 特殊情况：尝试进行“高级索引”。将此表达式降级为使用张量索引的 index 和 index_put_ 操作
      const auto indices = graph
                               ->insertNode(graph->createList(
                                   OptionalType::ofTensor(), tensorIndices))
                               ->output();
      const auto indexed =
          graph->insert(aten::index, {slicedArg, indices}, {}, stmt.range());
      const auto augmented = emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, sliceable->type()),
          {rhs},
          {},
          indexed);
      graph->insert(
          aten::index_put_,
          {slicedArg, indices, augmented},
          {},
          stmt.range());
    }
  } else {
    // 如果不是张量，则进行通用的增强赋值处理
    emitAugAssignmentGeneric(stmt, lhs, sliceable);
  }
}

NamedValue emitValueToTensor(
    const NamedValue& value,
    const NamedValue& matchTypeOf) {
  // 添加将 int/float/complex/bool/number 类型隐式转换为张量的功能
  // 用于在 emitSubscriptAssign 中将 `tensor(...)[x] = 99` 转换为 `tensor(...)[x] = tensor(99)`
  // 类似于 python_variable_indexing.cpp 中的 valueToTensor 行为
  const auto kind = value.type()->kind();
    // 如果类型是数值类型、整数类型、布尔类型、浮点数类型或复数类型
    if (kind == c10::TypeKind::NumberType || kind == c10::TypeKind::IntType ||
        kind == c10::TypeKind::BoolType || kind == c10::TypeKind::FloatType ||
        kind == c10::TypeKind::ComplexType) {
      // 插入 dtype 操作到图中，使用 matchTypeOf 作为参数
      auto dtype = graph->insert(prim::dtype, {matchTypeOf}, {});
      // 插入 device 操作到图中，使用 matchTypeOf 作为参数
      auto device = graph->insert(prim::device, {matchTypeOf}, {});
      // 插入 aten::tensor 操作到图中，使用 value 作为输入，并带有 dtype 和 device 参数
      auto converted = graph->insert(
          aten::tensor,
          {value},
          {NamedValue("dtype", dtype), NamedValue("device", device)});
      // 返回转换后的 NamedValue
      return NamedValue(value.loc(), converted);
    }

    // 返回原始的 value
    return value;
  }

  // 发出类似 `foo[0] = bar` 的修改赋值语句
  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const Expr& rhs) {
    // 调用重载的 emitSubscriptAssign 函数，将 rhs 编译成 NamedValue 并发出赋值
    emitSubscriptAssign(stmtRange, lhs, NamedValue(rhs.range(), emitExpr(rhs)));
  }

  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const NamedValue& rhs) {
    // 首先检查基础值
    auto sliceable = emitExpr(lhs.value());

    // 如果是张量，则将 RHS 数据复制到其中
    if (sliceable->type()->isSubtypeOf(*TensorType::get())) {
      // 处理多维切片：首先发出 int/slice 索引
      // TODO: Python 等效代码有特殊的 copy_to，用于广播以匹配 NumPy 语义（参见 PR#4853）。
      // 我们无法在不知道张量大小的情况下复制该代码，因此该代码应该移动到 aten 函数中
      auto [sliced, tensorIndices] = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(lhs.range(), sliced);

      // RHS 必须是张量，隐式转换为 int/float/complex/bool
      const auto convertedRhs = emitValueToTensor(rhs, slicedArg);

      if (tensorIndices.empty()) {
        // 常见情况：仅尝试使用 int 和 slice 进行索引。将 RHS 复制到结果张量中。
        graph->insert(aten::copy_, {slicedArg, convertedRhs}, {}, stmtRange);
      } else {
        // 特殊情况：尝试使用张量进行 "高级索引"。使用 Tensor?[] 的 tensorindices 调度到 `aten::index_put_`
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();

        graph->insert(
            aten::index_put_,
            {slicedArg, indices, convertedRhs},
            {},
            stmtRange);
      }
      // 否则，这是列表或 classtype。调度到 aten::_set_item 以同时选择和赋值
    } else {
      // 获取左侧表达式的下标表达式列表
      const auto subscript = lhs.subscript_exprs();
      // 如果下标表达式的数量不为1或第一个下标表达式是切片表达式，则抛出错误
      if (subscript.size() != 1 || subscript[0].kind() == TK_SLICE_EXPR) {
        throw ErrorReport(subscript)
            << "Sliced expression not yet supported for"
            << " subscripted assignment. "
            << "File a bug if you want this";
      }
      // 如果sliceable对象的类型是AnyTupleType的子类型，则抛出不支持下标赋值的错误
      if (sliceable->type()->isSubtypeOf(*AnyTupleType::get())) {
        throw ErrorReport(lhs) << sliceable->type()->repr_str()
                               << " does not support subscripted assignment";
      }

      // 准备函数调用的参数列表
      std::vector<NamedValue> args;
      // 添加self参数，指向sliceable对象
      args.emplace_back(lhs.value().range(), "self", sliceable);
      // 添加idx参数，使用第一个下标表达式的编译后表达式
      args.emplace_back(
          lhs.subscript_exprs().range(), "idx", emitExpr(subscript[0]));
      // 添加rhs作为参数
      args.push_back(rhs);
      // 调用__setitem__函数或方法
      makeMagic(
          "__setitem__",
          std::make_shared<BuiltinFunction>(aten::_set_item, at::nullopt))
          ->call(stmtRange, method, args, {}, 0);
    }
  }

  void emitTupleAssign(const TupleLiteral& tl, const Expr& rhs) {
    // 获取元组字面量的绑定器数量
    size_t n_binders = tl.inputs().size();
    // 验证赋值左侧表达式是否支持星号解包
    bool starred_unpack = validateAssignLhsExpr(tl.inputs(), tl.range());
    // 如果支持星号解包，则减少绑定器数量
    if (starred_unpack)
      n_binders--;
    // 对rhs表达式进行编译，并获取其糖值
    auto output = emitSugaredExpr(rhs, n_binders);
    // 执行元组赋值操作
    emitTupleAssign(tl, output, rhs.range(), n_binders, starred_unpack);
  }

  void emitTupleAssign(
      const TupleLiteral& tl,
      const SugaredValuePtr& rhs_output,
      const SourceRange& rhs_loc,
      size_t n_binders,
      bool starred_unpack) {
    // 将rhs_output作为元组处理，获取多个输出值
    auto outputs = rhs_output->asTuple(
        rhs_loc,
        method,
        starred_unpack ? c10::nullopt : std::optional<size_t>{n_binders});
    // 检查输出数量是否符合预期
    if (outputs.size() < n_binders) {
      throw ErrorReport(tl)
          << "need " << (starred_unpack ? "at least " : "") << n_binders
          << " values to unpack but found only " << outputs.size();
    }
    // 如果输出数量超过预期且未使用星号解包，则抛出错误
    if (outputs.size() > n_binders && !starred_unpack) {
      throw ErrorReport(tl) << "too many values to unpack: need " << n_binders
                            << " but found " << outputs.size();
    }

    // 执行表达式赋值操作
    emitExprsAssign(tl.inputs(), outputs, rhs_loc, n_binders);
  }

  void emitExprsAssign(
      const List<Expr>& lhs_exprs,
      const at::ArrayRef<SugaredValuePtr> outputs,
      const SourceRange& rhs_loc,
      size_t n_binders) {
    // 初始化计数器
    int i = 0;
  // 遍历左侧表达式列表中的每个赋值目标
  for (auto assignee : lhs_exprs) {
    // 根据赋值目标的类型进行不同的处理
    switch (assignee.kind()) {
      case TK_SUBSCRIPT:
        // 发射下标赋值操作，将右侧表达式的值赋给左侧的子脚本
        emitSubscriptAssign(
            rhs_loc,
            Subscript(assignee),
            NamedValue(rhs_loc, outputs.at(i)->asValue(rhs_loc, method)));
        i++;
        break;
      case TK_VAR:
        // 在环境堆栈中设置变量的糖化值
        environment_stack->setSugaredVar(
            assignee.range(),
            Var(assignee).name().name(),
            outputs.at(i),
            /*annotated_type=*/nullptr);
        i++;
        break;
      case TK_STARRED: {
        // 处理星号解包的情况
        auto var = Starred(assignee).expr();
        // 如果星号解包的目标不是变量，则抛出错误
        if (var.kind() != TK_VAR) {
          throw ErrorReport(var) << "Cannot pack a tuple into a non-variable";
        }
        size_t n_matched = outputs.size() - n_binders;
        ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
        // 生成值的数组，用于创建元组
        auto values = fmap(
            outputs_ref.slice(i, n_matched),
            [&](const std::shared_ptr<SugaredValue>& v) {
              return v->asValue(assignee.range(), method);
            });
        // 在图中插入创建元组的节点，并获取其输出
        auto tup = graph->insertNode(graph->createTuple(values))->output();
        // 在环境堆栈中设置变量的值
        environment_stack->setVar(var.range(), Var(var).name().name(), tup);
        i += n_matched;
      } break;
      case TK_TUPLE_LITERAL: {
        // 递归地处理元组字面值的赋值操作
        TupleLiteral sub_tl = TupleLiteral(assignee);
        size_t sub_n_binders = sub_tl.inputs().size();
        // 验证左侧表达式是否合法
        bool sub_starred_unpack =
            validateAssignLhsExpr(sub_tl.inputs(), sub_tl.range());
        if (sub_starred_unpack)
          sub_n_binders--;
        // 发射元组赋值操作
        emitTupleAssign(
            sub_tl,
            outputs.at(i),
            rhs_loc,
            sub_n_binders,
            sub_starred_unpack);
        i++;
      } break;
      case '.': {
        // 发射属性选择赋值操作
        emitSelectAssign(assignee, outputs.at(i), rhs_loc);
        i++;
      } break;
      default:
        // 抛出意外表达式的错误
        throw ErrorReport(assignee)
            << "unexpected expression on the left-hand side";
    }
  }
}

// 发射赋值语句的主函数
void emitAssignment(const Assign& stmt) {
  // 如果只有一个左侧表达式，直接发射单个赋值操作
  if (stmt.lhs_list().size() == 1) {
    return emitSingleAssignment(stmt);
  }
  // 多个赋值目标及不支持的注释类型在Python中不支持
  TORCH_INTERNAL_ASSERT(stmt.lhs_list().size() > 1 && !stmt.type().present());
  // 对于表达式 a = b = expr() 的语义是 expr() 只发射一次，然后从左到右进行赋值
  // 创建临时变量名
  const auto tmp_name = createTempName("$tmp_assign_");
  // 在环境堆栈中设置变量的糖化值
  environment_stack->setSugaredVar(
      stmt.rhs().range(),
      tmp_name,
      emitSugaredExpr(stmt.rhs().get(), 1),
      /*annotated_type=*/nullptr);
  // 创建标识符
  auto ident = Var::create(
      stmt.rhs().range(), Ident::create(stmt.rhs().range(), tmp_name));
   cpp
// 循环处理语句的左值列表，对每个表达式生成单一赋值语句并发射
for (auto expr : stmt.lhs_list()) {
  // 创建一个单一赋值语句，包括赋值语句的范围、左值表达式列表、右值表达式（可能为空）、语句自身的范围（可能为空）
  emitSingleAssignment(Assign::create(
      stmt.range(),
      List<Expr>::create(expr.range(), {expr}),
      Maybe<Expr>::create(stmt.rhs().range(), ident),
      Maybe<Expr>::create(stmt.range())));
}

void emitSingleAssignment(const Assign& stmt) {
  // 如果右值不存在，抛出错误报告
  if (!stmt.rhs().present()) {
    throw ErrorReport(stmt.range())
        << "For an assignment, expected an expression on the right-hand side";
  }
  // 获取右值表达式
  const Expr& rhs = stmt.rhs().get();

  // 根据左值的类型进行不同处理
  switch (stmt.lhs().kind()) {
    case TK_VAR: {
      // 如果左值是变量，则创建变量对象
      auto v = Var(stmt.lhs());
      TypePtr type = nullptr;
      // 如果指定了类型，则从表达式中解析类型
      if (stmt.type().present()) {
        type = typeParser_.parseTypeFromExpr(stmt.type().get());
      }
      // 生成右值表达式的糖化后值
      auto rhs_sugared_val = emitSugaredExpr(rhs, 1, type);

      // BC HACK 开始
      //
      // 对于旧的序列化量化 RNN 模块，将 quantized::linear_prepack 转换为 quantized::linear_prepack_legacy。
      // 我们将 linear_prepack 更改为返回 TorchBind 类而不是 cpp_custom_type_hack 张量，
      // 但旧的序列化模型与 type_hack 版本紧密耦合。如果这里仍创建一个张量，则 quantized_lstm.legacy
      // 重载可以在 forward_impl() 中启动，并且模块仍然可以正常运行。
      if (method.qualname() ==
          "__torch__.torch.nn.quantized.dynamic.modules.rnn.PackedParameter.__setstate__") {
        if (auto sv =
                std::dynamic_pointer_cast<SimpleValue>(rhs_sugared_val)) {
          Node* rhs_node = sv->getValue()->node();
          if (rhs_node->kind() ==
              Symbol::fromQualString("quantized::linear_prepack")) {
            std::vector<NamedValue> inputs;
            for (Value* i : rhs_node->inputs()) {
              inputs.emplace_back(i);
            }
            Value* new_val = rhs_node->owningGraph()->insert(
                Symbol::fromQualString("quantized::linear_prepack_legacy"),
                inputs,
                {},
                rhs_node->sourceRange());
            rhs_sugared_val = std::make_shared<SimpleValue>(new_val);
          }
        }
      }
      // BC HACK 结束

      // 设置环境栈中的糖化变量
      environment_stack->setSugaredVar(
          v.range(),
          v.name().name(),
          std::move(rhs_sugared_val),
          /*annotated_type=*/type);
    } break;
    case TK_TUPLE_LITERAL:
      // 如果左值是元组字面值，则生成元组赋值
      emitTupleAssign(TupleLiteral(stmt.lhs()), rhs);
      break;
    case '.':
      // 如果左值是点      emitTupleAssign(TupleLiteral(stmt.lhs()), rhs);
        break;
      case '.':
        // 如果左侧表达式是属性选择
        emitSelectAssign(stmt);
        break;
      case TK_SUBSCRIPT:
        // 如果左侧表达式是下标访问
        emitSubscriptAssign(stmt.range(), Subscript(stmt.lhs()), rhs);
        break;
      default:
        // 其他情况报错
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on left-hand side of assignment";
    }
  }

  void emitSelectAssign(const Assign& stmt) {
    // 如果语句的右手边不存在，则抛出错误报告
    if (!stmt.rhs().present()) {
      throw ErrorReport(stmt.range()) << "Expected RHS for assignment";
    }

    // 初始化类型提示为 nullptr
    TypePtr type_hint = nullptr;
    // 如果语句包含类型信息，则解析并设置类型提示
    if (stmt.type().present()) {
      type_hint = typeParser_.parseTypeFromExpr(stmt.type().get());
    }

    // 获取左手边表达式的选择器
    const auto lhs = Select(stmt.lhs());
    // 生成左手边表达式的糖化值
    auto lhsObject = emitSugaredExpr(lhs.value(), 1);
    // 生成右手边表达式的糖化值，作为值表达式
    const auto rhsValue = emitSugaredExpr(stmt.rhs().get(), 1, type_hint)
                              ->asValue(stmt.rhs().range(), method);
    // 设置左手边对象的属性，将右手边值赋给左手边选择器名称
    lhsObject->setAttr(stmt.range(), method, lhs.selector().name(), rhsValue);
  }

  // 发出选择赋值操作
  void emitSelectAssign(
      const Expr& lhs,
      SugaredValuePtr rhs,
      const SourceRange& loc) {
    // 获取左手边表达式的选择器
    const auto lhs_select = Select(lhs);
    // 生成左手边表达式的糖化值
    auto lhs_sv = emitSugaredExpr(lhs_select.value(), 1);
    // 将右手边糖化值作为值表达式
    const auto rhs_value = rhs->asValue(loc, method);
    // 设置左手边对象的属性，将右手边值赋给左手边选择器名称
    lhs_sv->setAttr(loc, method, lhs_select.selector().name(), rhs_value);
  }

  // 获取节点类型
  NodeKind getNodeKind(int kind, int ninputs) {
    // 根据不同的操作符种类返回对应的节点类型
    switch (kind) {
      case '+':
        return aten::add;
      case '-':
        return aten::sub;
      case TK_UNARY_MINUS:
        return aten::neg;
      case '*':
        return aten::mul;
      case TK_POW:
        return aten::pow;
      case '@':
        return aten::matmul;
      case TK_STARRED:
        return prim::Starred;
      case '/':
        return aten::div;
      case '%':
        return aten::remainder;
      case TK_NE:
        return aten::ne;
      case TK_EQ:
        return aten::eq;
      case '<':
        return aten::lt;
      case '>':
        return aten::gt;
      case TK_LE:
        return aten::le;
      case TK_GE:
        return aten::ge;
      case TK_AND:
        return aten::__and__;
      case TK_OR:
        return aten::__or__;
      case TK_IS:
        return aten::__is__;
      case TK_ISNOT:
        return aten::__isnot__;
      case TK_NOT:
        return aten::__not__;
      case TK_FLOOR_DIV:
        return aten::floordiv;
      case TK_LSHIFT:
        return aten::__lshift__;
      case TK_RSHIFT:
        return aten::__rshift__;
      case '&':
        return aten::__and__;
      case '|':
        return aten::__or__;
      case '^':
        return aten::__xor__;
      case TK_IN:
        return aten::__contains__;
      default:
        // 如果操作符种类未知，则抛出运行时错误
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }

  // 获取操作符重载的字符串表示
  std::string getOperatorOverload(int kind, int ninputs) {
  switch (kind) {
    case '+':
      return "__add__";
    // 如果操作符是加法，返回对应的特殊方法名 "__add__"
    case '-':
      return "__sub__";
    // 如果操作符是减法，返回对应的特殊方法名 "__sub__"
    case TK_UNARY_MINUS:
      return "__neg__";
    // 如果操作符是一元减号，返回对应的特殊方法名 "__neg__"
    case '~':
      return "__invert__";
    // 如果操作符是按位取反，返回对应的特殊方法名 "__invert__"
    case '*':
      return "__mul__";
    // 如果操作符是乘法，返回对应的特殊方法名 "__mul__"
    case TK_POW:
      return "__pow__";
    // 如果操作符是幂运算，返回对应的特殊方法名 "__pow__"
    case '/':
      return "__truediv__";
    // 如果操作符是除法，返回对应的特殊方法名 "__truediv__"
    case '%':
      return "__mod__";
    // 如果操作符是取模运算，返回对应的特殊方法名 "__mod__"
    case TK_NE:
      return "__ne__";
    // 如果操作符是不等于，返回对应的特殊方法名 "__ne__"
    case TK_EQ:
      return "__eq__";
    // 如果操作符是等于，返回对应的特殊方法名 "__eq__"
    case '<':
      return "__lt__";
    // 如果操作符是小于，返回对应的特殊方法名 "__lt__"
    case '>':
      return "__gt__";
    // 如果操作符是大于，返回对应的特殊方法名 "__gt__"
    case TK_LE:
      return "__le__";
    // 如果操作符是小于等于，返回对应的特殊方法名 "__le__"
    case TK_GE:
      return "__ge__";
    // 如果操作符是大于等于，返回对应的特殊方法名 "__ge__"
    case '&':
      return "__and__";
    // 如果操作符是按位与，返回对应的特殊方法名 "__and__"
    case '|':
      return "__or__";
    // 如果操作符是按位或，返回对应的特殊方法名 "__or__"
    case '^':
      return "__xor__";
    // 如果操作符是按位异或，返回对应的特殊方法名 "__xor__"
    case TK_IN:
      return "__contains__";
    // 如果操作符是成员关系判断，返回对应的特殊方法名 "__contains__"
    case TK_LSHIFT:
      return "__lshift__";
    // 如果操作符是左移位，返回对应的特殊方法名 "__lshift__"
    case TK_RSHIFT:
      return "__rshift__";
    // 如果操作符是右移位，返回对应的特殊方法名 "__rshift__"
    default:
      throw std::runtime_error("unknown kind " + std::to_string(kind));
    // 如果操作符不在预期的范围内，抛出运行时错误，提示未知的操作符类型
  }

  std::vector<NamedValue> getNamedValues(
      const TreeList& trees,
      bool maybe_unpack) {
    std::vector<NamedValue> values;
    // 遍历树列表中的每个树节点
    for (const auto& tree : trees) {
      // 如果允许解包，并且当前树节点是星号表达式
      if (maybe_unpack && tree->kind() == TK_STARRED) {
        auto starred = Starred(tree);
        // 生成星号表达式的糖化表达式，并转换为元组形式的条目
        auto entries = emitSugaredExpr(starred.expr(), 1)
                           ->asTuple(starred.range(), method);
        // 将每个条目转换为命名值并添加到结果向量中
        for (const auto& entry : entries) {
          values.emplace_back(
              tree->range(), entry->asValue(starred.range(), method));
        }
      } else {
        // 否则，将树节点转换为表达式并添加到结果向量中
        values.emplace_back(tree->range(), emitExpr(Expr(tree)));
      }
    }
    return values;
    // 返回生成的命名值向量
  }
  std::vector<NamedValue> getNamedValues(
      const List<Expr>& trees,
      bool maybe_unpack) {
    // 将表达式列表转换为树列表，然后调用上述函数
    return getNamedValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<Value*> getValues(const TreeList& trees, bool maybe_unpack) {
    // 将树列表转换为命名值向量，然后转换为值指针向量
    return toValues(*graph, getNamedValues(trees, maybe_unpack));
  }
  std::vector<Value*> getValues(const List<Expr>& trees, bool maybe_unpack) {
    // 将表达式列表转换为树列表，然后调用上述函数
    return getValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<NamedValue> emitAttributes(const List<Attribute>& attributes) {
    // 遍历属性列表，为每个属性生成命名值并返回命名值向量
    return fmap(attributes, [&](const Attribute& attr) {
      return NamedValue(
          attr.range(), attr.name().name(), emitExpr(attr.value()));
    });
  }

  void checkApplyNumInputs(Apply& apply, size_t expected_inputs) {
    const SourceRange& loc = apply.range();
    // 检查应用程序调用的输入数量是否与预期数量不匹配
    if (apply.inputs().size() != expected_inputs) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " expected exactly "
          << expected_inputs << " arguments but found "
          << apply.inputs().size();
    }
    // 如果应用程序调用有关键字参数，抛出错误，指出不支持关键字参数
    if (!apply.attributes().empty()) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " takes no keyword arguments";
    }
  }
  }
}

void checkApplyNumInputsRange(
    Apply& apply,
    size_t min_expected_inputs,
    size_t max_expected_inputs) {
  // 获取调用的源代码范围
  const SourceRange& loc = apply.range();
  // 获取调用的位置参数个数
  size_t position_arg_size = apply.inputs().size();
  // 检查位置参数个数是否在期望范围内
  if (position_arg_size < min_expected_inputs ||
      position_arg_size > max_expected_inputs) {
    // 抛出错误报告，指出调用的函数期望的参数个数范围
    throw ErrorReport(loc)
        << Var(apply.callee()).name().name()
        << " expected to have number of arguments between "
        << min_expected_inputs << " and " << max_expected_inputs
        << " but found " << position_arg_size;
  }
  // 如果调用包含关键字参数，则抛出错误报告
  if (!apply.attributes().empty()) {
    throw ErrorReport(loc)
        << Var(apply.callee()).name().name() << " takes no keyword arguments";
  }
}

std::shared_ptr<SugaredValue> emitApplyExpr(
    Apply& apply,
    size_t n_binders,
    const TypePtr& type_hint = nullptr) {
  // 生成调用表达式对应的求值器
  auto sv = emitSugaredExpr(apply.callee(), 1);
  // 获取调用的源代码范围
  auto loc = apply.callee().range();
  // 如果是特殊形式的调用，则使用特殊形式的处理函数处理
  if (auto special_form = dynamic_cast<SpecialFormValue*>(sv.get())) {
    return emitApplySpecialForm(special_form->form(), apply, sv, type_hint);
  }
  // 获取调用的位置参数和关键字参数的值
  auto args = getNamedValues(apply.inputs(), true);
  auto kwargs = emitAttributes(apply.attributes());
  // 返回调用表达式的求值结果
  return sv->call(loc, method, args, kwargs, n_binders);
}

// 处理看起来像应用语句的表达式，但其参数具有特殊的求值规则
// 当添加新情况时，仅在无法使用标准的SugaredValue::call函数表达时添加特殊形式
std::shared_ptr<SugaredValue> emitApplySpecialForm(
    Symbol form,
    Apply& apply,
    std::shared_ptr<SugaredValue> sv,
    const TypePtr& type_hint = nullptr) {
  // 空实现，需要根据特殊形式的实际处理逻辑来填充
}

std::shared_ptr<SugaredValue> emitApplySpecialFormForList(
    Apply& apply,
    const TypePtr& type_hint = nullptr) {
  // 如果应用没有输入参数，则生成一个空列表节点
  if (apply.inputs().empty()) {
    TypePtr type = type_hint ? type_hint : ListType::ofTensors();
    if (!type->cast<ListType>()) {
      throw ErrorReport(apply.range())
          << "Expected list type annotation for list(), found "
          << type_hint->repr_str();
    }
    // 创建一个包含空列表的简单值对象
    return std::make_shared<SimpleValue>(
        graph
            ->insertNode(graph->createList(
                type->expectRef<ListType>().getElementType(), {}))
            ->output());
  }
  // 处理 list(iter) 的特殊形式，展开成 [_elem for _elem in iter]
  checkApplyNumInputs(apply, 1);
  // 获取迭代器输入参数的求值结果
  auto iter_input = emitSugaredExpr(apply.inputs()[0], 1);

  // aten::list 内置操作已注册为适用于列表和字符串输入
  // 分派到内置操作，以避免现有用例的性能减慢
    // 检查是否可以将 iter_input 转换为 Simple 类型
    if (auto simple = asSimple(iter_input)) {
      // 如果 iter_input 是 ListType 或 StringType 类型，则生成相应的 SimpleValue
      if (simple->type()->cast<ListType>() ||
          simple->type()->cast<StringType>()) {
        // 调用内置函数 emitBuiltinCall 生成 SimpleValue 对象
        return std::make_shared<SimpleValue>(emitBuiltinCall(
            apply.range(), *method.graph(), aten::list, {simple}, {}));
      }
    }

    // 创建临时迭代器名称，用于环境变量栈的设置
    const std::string& iter_name = createTempName("$_iter");
    environment_stack->setSugaredVar(
        apply.range(),
        iter_name,
        iter_input,
        /*annotated_type=*/nullptr);

    // 创建临时元素名称，用于 ListComp 的创建
    const std::string& elem_name = createTempName("$_elem");
    // 创建表示元素的标识符 Var
    auto ident =
        Var::create(apply.range(), Ident::create(apply.range(), elem_name));
    // 创建表示迭代器的 Var
    auto iter =
        Var::create(apply.range(), Ident::create(apply.range(), iter_name));
    // 创建 ListComp 对象 lc，用于列表推导式的生成
    auto lc = ListComp::create(apply.range(), ident, ident, iter);
    // 调用 emitListComprehension 生成 SimpleValue 对象，表示列表推导式的结果
    return std::make_shared<SimpleValue>(emitListComprehension(lc, type_hint));
  }

  // 生成特殊形式字典的应用表达式的处理函数
  std::shared_ptr<SugaredValue> emitApplySpecialFormForDict(
      Apply& apply,
      const TypePtr& type_hint = nullptr) {
    // 检查类型分配错误的辅助函数
    auto check_type_assignment_error = [&](const TypePtr& key_type,
                                           const TypePtr& value_type,
                                           const TypePtr& annotated_dict_type) {
      std::stringstream ss;
      std::stringstream err;

      // 获取注释字典类型的键和值类型
      auto annotated_k_type =
          annotated_dict_type->expect<DictType>()->getKeyType();
      auto annotated_v_type =
          annotated_dict_type->expect<DictType>()->getValueType();

      // 检查生成的键类型是否符合预期的类型
      const auto is_key_subtype = key_type == annotated_k_type;
      // 检查生成的值类型是否是注释值类型的子类型
      const auto is_value_subtype =
          value_type->isSubtypeOfExt(annotated_v_type, &ss);

      // 如果键类型不匹配，则记录错误信息
      if (!is_key_subtype) {
        err << "Generated key type " << key_type->repr_str()
            << " did not match the annotated key type, which was "
            << annotated_k_type->repr_str() << "\n";
      }

      // 如果值类型不匹配，则记录错误信息
      if (!is_value_subtype) {
        err << "Generated value type " << value_type->repr_str()
            << " did not match the annotated value type, which was "
            << annotated_v_type->repr_str() << "\n"
            << ss.str();
      }

      // 如果存在类型不匹配的情况，则抛出错误报告
      if (!is_key_subtype || !is_value_subtype) {
        throw ErrorReport(apply) << err.str();
      }
    };

    // 添加关键字参数的处理函数
    auto add_kwargs = [&](Value* dc_value) {
      // 创建 self 参数，表示字典的实例
      NamedValue self = NamedValue(apply.range(), "self", dc_value);
      // 遍历应用的所有属性
      for (const auto& kwarg : apply.attributes()) {
        // 创建表示属性名称的 StringLiteral
        auto name = StringLiteral::create(kwarg.range(), kwarg.name().name());
        // 生成属性名称对应的表达式
        auto k = emitExpr(name);
        // 生成属性值对应的表达式
        auto v = emitExpr(kwarg.value());
        // 创建命名值对象，表示属性名称和属性值
        NamedValue input_k = NamedValue(kwarg.range(), "", k);
        NamedValue input_v = NamedValue(kwarg.range(), "", v);

        // 检查类型分配错误
        check_type_assignment_error(k->type(), v->type(), dc_value->type());

        // 调用内置函数 emitBuiltinCall，实现属性设置操作
        emitBuiltinCall(
            kwarg.range(),
            *graph,
            aten::_set_item,
            {self, input_k, input_v},
            {});
      }
    };
    // 定义一个 lambda 函数 treat_as_empty_container，用于检查是否应将输入视为空容器
    auto treat_as_empty_container = [&]() {
      // 如果 apply.inputs() 为空且 apply.attributes() 不为空，则返回 true
      if (apply.inputs().empty() && !apply.attributes().empty()) {
        return true;
      }
      // 如果 apply.inputs() 非空且第一个输入的类型为 TK_DICT_LITERAL，则继续判断
      if (!apply.inputs().empty() &&
          apply.inputs()[0].kind() == TK_DICT_LITERAL) {
        // 将第一个输入解析为 DictLiteral 对象
        auto dict_lit = DictLiteral(apply.inputs()[0]);
        // 如果字典字面量的键和值都为空，则返回 true
        return dict_lit.key_inputs().empty() && dict_lit.value_inputs().empty();
      }
      // 如果 apply.inputs() 非空且第一个输入的类型为 TK_LIST_LITERAL，则继续判断
      if (!apply.inputs().empty() &&
          apply.inputs()[0].kind() == TK_LIST_LITERAL) {
        // 将第一个输入解析为 ListLiteral 对象
        auto list_lit = ListLiteral(apply.inputs()[0]);
        // 如果列表字面量的内容为空，则返回 true
        return list_lit.inputs().empty();
      }
      // 默认情况下返回 false
      return false;
    };

    // 根据 type_hint 是否为 UnionType，决定 annotated_union_type 的取值
    TypePtr annotated_union_type =
        type_hint && type_hint->isUnionType() ? type_hint : nullptr;

    // 定义一个 lambda 函数 add_union_cast，用于向结果添加 Union 类型的转换
    auto add_union_cast = [&](Value* result) {
      // 在图中插入一个节点，类型为 prim::unchecked_cast，输入为 result
      Node* n =
          graph->insertNode(graph->create(prim::unchecked_cast, {result}));
      // 设置节点输出的类型为 annotated_union_type
      n->output()->setType(std::move(annotated_union_type));
      // 更新 result 为节点的输出
      result = n->output();
    };

    // 初始化 refined_type_hint 为 type_hint
    TypePtr refined_type_hint = type_hint;

    // 初始化 all_candidates 为空的 TypePtr 向量
    std::vector<TypePtr> all_candidates = {};

    // 定义一个 lambda 函数 type_match，用于检查类型是否为 DictType::Kind
    auto type_match = [&](const TypePtr& t) {
      return t->kind() == DictType::Kind;
    };

    // 如果 type_hint 存在且其类型不为 DictType::Kind，则进行类型细化或者填充候选向量
    if (type_hint && type_hint->kind() != DictType::Kind) {
      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "Dict",
          apply,
          type_match,
          [] {},
          [] {},
          /*is_dict_constructor=*/true);
    }

    // 如果 all_candidates 不为空，则抛出错误报告，说明存在多个候选的 Dict 类型
    if (!all_candidates.empty()) {
      throw ErrorReport(apply)
          << "There are multiple candidate "
          << "Dict types in the Union type annotation `"
          << type_hint->repr_str()
          << "`, and full type inference is not yet supported for the "
          << "`dict()` constructor.";
    }

    // 如果可能，将当前结果强制转换为 Dict 类型，并手动添加 kwargs
    // 这不仅是最简单的解决方案，还适用于诸如 `dict(dict([1, 2, 3]))` 或 `dict(x)`（其中 `x` 是先前定义的变量）的情况
    // 如果 apply 对象的输入不为空
    if (!apply.inputs().empty()) {
      // 尝试生成输入表达式的 SugaredValue
      std::shared_ptr<SugaredValue> iter_input;
      try {
        // 尝试使用类型提示生成 SugaredValue
        iter_input = emitSugaredExpr(apply.inputs()[0], 1, type_hint);
      } catch (const ErrorReport&) {
        // 如果失败，则不使用类型提示生成 SugaredValue
        iter_input = emitSugaredExpr(apply.inputs()[0], 1);
      }
      // 如果生成的 SugaredValue 是一个简单值
      if (auto simple = asSimple(iter_input)) {
        // 如果这个简单值的类型是字典类型
        if (simple->type()->cast<DictType>()) {
          // 调用内置函数创建字典
          auto dc_value = emitBuiltinCall(
              apply.range(), *method.graph(), aten::dict, {simple}, {});
          // 添加关键字参数
          add_kwargs(dc_value);
          // 如果有联合类型注解，添加联合转换
          if (annotated_union_type) {
            add_union_cast(dc_value);
          }
          // 返回一个包装了 dc_value 的 SimpleValue
          return std::make_shared<SimpleValue>(dc_value);
        }
      }
    }

    // 如果被视为空容器的条件成立
    if (treat_as_empty_container()) {
      // 创建一个空的表达式列表
      auto expr_list = List<Expr>::create(apply.range(), {});
      // 替换 apply 对象为一个只有 callee 和 attributes 的调用对象
      apply = Apply::create(
          apply.range(), apply.callee(), expr_list, apply.attributes());
    }

    // 如果 apply 对象的输入为空并且其属性也为空
    if (apply.inputs().empty() && apply.attributes().empty()) {
      // 如果没有精炼的类型提示
      if (!refined_type_hint) {
        // 创建一个精炼的字典类型提示
        refined_type_hint =
            DictType::create(StringType::get(), TensorType::get());
      } else if (!all_candidates.empty()) {
        // 如果存在多个候选的字典类型，抛出错误
        throw ErrorReport(apply.range())
            << "Cannot determine the type "
            << "of an empty dict given the Union annotation `"
            << type_hint->repr_str() << "`, which contains multiple "
            << "candidate Dict types ";
      }

      // 检查精炼的类型提示是否为字典类型
      TORCH_CHECK(
          refined_type_hint->kind() == DictType::Kind,
          "Expected a type annotation "
          "of Dict for dict constructor dict(), got ",
          type_hint->str());

      // 创建一个空字典并返回其 SimpleValue
      return std::make_shared<SimpleValue>(
          graph
              ->insertNode(graph->createDict(
                  refined_type_hint->expect<DictType>()->getKeyType(),
                  refined_type_hint->expect<DictType>()->getValueType(),
                  {},
                  {}))
              ->output());
    }

    // 特殊情况处理字典推导式
    if (!apply.inputs().empty() && apply.inputs()[0].kind() == TK_DICT_COMP) {
      // 解析字典推导式并生成其值
      auto dc = DictComp(apply.inputs()[0]);
      auto dc_value = emitDictComprehension(dc, refined_type_hint);
      // 添加关键字参数
      add_kwargs(dc_value);
      // 返回一个包装了 dc_value 的 SimpleValue
      return std::make_shared<SimpleValue>(dc_value);
    }
    // 如果 apply 的输入非空，并且第一个输入是字典字面量
    // 则将字典字面量转换为元组列表，以便使用现有的构造函数
    if (!apply.inputs().empty() &&
        apply.inputs()[0].kind() == TK_DICT_LITERAL) {
      // 从 apply 的输入中获取字典字面量
      auto dict_lit = DictLiteral(apply.inputs()[0]);
      // 准备存储转换后的元组列表
      std::vector<Expr> zipped;
      zipped.reserve(dict_lit.key_inputs().size());
      // 断言键值对数量相等
      TORCH_INTERNAL_ASSERT(
          dict_lit.key_inputs().size() == dict_lit.value_inputs().size());
      // 遍历字典字面量的键和值，创建元组并存储到 zipped 中
      for (auto key_it = dict_lit.key_inputs().begin(),
                val_it = dict_lit.value_inputs().begin();
           key_it != dict_lit.key_inputs().end();
           ++key_it, ++val_it) {
        auto tuple_inputs =
            List<Expr>::create(apply.range(), {*key_it, *val_it});
        auto tuple = TupleLiteral::create(apply.range(), tuple_inputs);
        zipped.push_back(tuple);
      }
      // 创建包含 zipped 的列表表达式
      auto ll_values = List<Expr>::create(apply.range(), zipped);
      auto ll = ListLiteral::create(apply.range(), ll_values);
      auto expr_list = List<Expr>::create(apply.range(), {ll});
      // 将 apply 节点更新为包含元组列表的新 Apply 节点
      apply = Apply::create(
          apply.range(), apply.callee(), expr_list, apply.attributes());
    }

    // 如果有关键字参数要包含，将采用类似的方法标准化 Apply 节点
    if (!apply.attributes().empty() &&
        (apply.inputs().empty() ||
         apply.inputs()[0].kind() == TK_LIST_LITERAL)) {
      // 准备存储所有表达式的向量
      std::vector<Expr> exprs;
      // 如果 apply 的输入非空，获取输入中的元组列表
      if (!apply.inputs().empty()) {
        auto tuple_list = ListLiteral(apply.inputs()[0]).inputs();
        // 将每个元组添加到 exprs 中
        for (const auto& tuple : tuple_list) {
          exprs.push_back(tuple);
        }
      }
      // 创建每个关键字参数的元组并添加到 exprs 中
      for (const auto& attr : apply.attributes()) {
        auto k = StringLiteral::create(apply.range(), attr.name().name());
        auto v = attr.value();
        auto tuple_inputs = List<Expr>::create(apply.range(), {k, v});
        auto tuple = TupleLiteral::create(apply.range(), tuple_inputs);
        exprs.push_back(tuple);
      }
      // 创建包含 exprs 的列表表达式
      auto expr_list = List<Expr>::create(apply.range(), {exprs});
      auto ll = ListLiteral::create(apply.range(), expr_list);
      auto new_inputs = List<Expr>::create(apply.range(), {ll});
      auto new_kwargs = List<Attribute>::create(apply.range(), {});
      // 更新 apply 节点为包含新输入和空的关键字参数列表的新 Apply 节点
      apply =
          Apply::create(apply.range(), apply.callee(), new_inputs, new_kwargs);
    }

    // 检查 apply 的输入数量是否为 1
    checkApplyNumInputs(apply, 1);

    // 对 apply 的第一个输入表达式进行语法糖处理
    auto iter_input = emitSugaredExpr(apply.inputs()[0], 1);

    // 创建临时变量名 $_iter 和 $_key
    const std::string& iter_name = createTempName("$_iter");
    const std::string& key_name = createTempName("$_key");
    // 创建一个临时的变量名，用于存储值表达式的名称
    const std::string& value_name = createTempName("$_value");

    // 创建一个代表键的变量，使用指定的键名
    auto key =
        Var::create(apply.range(), Ident::create(apply.range(), key_name));
    
    // 创建一个代表值的变量，使用生成的临时值名
    auto value =
        Var::create(apply.range(), Ident::create(apply.range(), value_name));
    
    // 创建一个元组字面量，包含之前创建的键值对变量
    auto target = TupleLiteral::create(
        apply.range(), List<Expr>::create(apply.range(), {key, value}));
    
    // 创建一个迭代变量，使用指定的迭代器名
    auto iter =
        Var::create(apply.range(), Ident::create(apply.range(), iter_name));

    // 在当前环境栈中设置变量的糖化表示
    environment_stack->setSugaredVar(
        apply.range(),
        iter_name,
        iter_input,
        /*annotated_type=*/nullptr);

    // 创建一个字典推导表达式，使用之前创建的键、值、目标和迭代变量
    auto dc = DictComp::create(apply.range(), key, value, target, iter);
    
    // 发出字典推导表达式，生成结果并传入细化的类型提示
    auto result = emitDictComprehension(dc, refined_type_hint);
    
    // 将关键字参数添加到结果中
    add_kwargs(result);

    // 如果存在注解的联合类型，则添加一个联合转换到结果中
    if (annotated_union_type) {
      add_union_cast(result);
    }

    // 创建并返回一个包含结果的简单值的共享指针
    return std::make_shared<SimpleValue>(result);
  }

  // 发出表达式，返回值的类型为 Value*
  // 如果提供了类型提示，将使用它进行类型检查
  Value* emitExpr(const Expr& tree, const TypePtr& type_hint = nullptr) {
    // 推入调用的源范围，以防编译此函数时触发错误
    ErrorReport::CallStack::update_pending_range(tree.range());
    
    // 发出糖化表达式，返回结果值
    Value* out_val =
        emitSugaredExpr(tree, 1, type_hint)->asValue(tree.range(), method);
    
    // 如果类型提示为 AnyType::get() 且结果值类型不是 AnyType::get()，
    // 则插入一个未经检查的类型转换
    if (type_hint == AnyType::get() && out_val->type() != AnyType::get()) {
      out_val = graph->insertUncheckedCast(out_val, type_hint);
    }
    
    // 返回生成的值
    return out_val;
  }

  // 反转比较操作符的节点种类
  NodeKind reverseComparision(NodeKind kind) {
    if (kind == aten::lt) {
      return aten::gt;
    } else if (kind == aten::le) {
      return aten::ge;
    } else if (kind == aten::gt) {
      return aten::lt;
    } else if (kind == aten::ge) {
      return aten::le;
    }
    
    // 如果节点种类不支持反转，则抛出运行时错误
    throw std::runtime_error(
        "reverseComparision: unsupported NodeKind. File a bug");
  }

  // 处理任何可能生成 SugaredValue 的表达式
  // 只返回单个 Value* 的表达式在 emitSimpleExpr 中处理
  // 如果存在类型提示，则调用者负责检查结果是否符合类型提示
  std::shared_ptr<SugaredValue> emitSugaredExpr(
      const Expr& tree,
      size_t n_binders,
      const TypePtr& type_hint = nullptr) {
    // 根据树节点的类型进行分支处理
    switch (tree.kind()) {
      case TK_VAR: {
        // 如果节点类型是变量（TK_VAR），获取变量名并返回相应的封装变量
        return environment_stack->getSugaredVar(Var(tree).name());
      }
      case '.': {
        // 如果节点类型是属性访问（'.'），获取属性选择器并生成对应的封装表达式
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
      }
      case TK_APPLY: {
        // 如果节点类型是函数应用（TK_APPLY），生成相应的函数应用表达式
        auto apply = Apply(tree);
        return emitApplyExpr(apply, n_binders, type_hint);
      } break;
      case TK_SUBSCRIPT: {
        // 如果节点类型是下标访问（TK_SUBSCRIPT），生成相应的下标访问表达式
        return emitSubscript(Subscript(tree), type_hint);
      } break;
      default:
        // 对于其他类型的节点，生成简单值表达式并返回
        return std::make_shared<SimpleValue>(emitSimpleExpr(tree, type_hint));
    }
  }

  // 发出一元操作表达式
  Value* emitUnaryOp(
      const TreeRef& tree,
      const std::string& magicMethod,
      const c10::Symbol& opSymbol) {
    // 获取操作的输入树节点
    const auto& inputs = tree->trees();
    // 获取命名值，并排除可能的解包
    auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
    // 创建对应的魔术方法，并调用它，返回简单值表达式
    auto val =
        asSimple(makeMagic(
                     magicMethod,
                     std::make_shared<BuiltinFunction>(opSymbol, at::nullopt))
                     ->call(tree->range(), method, named_values, {}, 0));

    // 如果生成的值不是由给定的操作符生成的，则直接返回该值
    if (val->node()->kind() != opSymbol) {
      return val;
    }

    // 如果输入都是常量，则尝试常量折叠操作
    auto maybe_out_stack = runNodeIfInputsAreConstant(val->node());
    if (!maybe_out_stack) {
      return val;
    }
    TORCH_INTERNAL_ASSERT(maybe_out_stack->size() == 1);
    // 将常量折叠后的结果插入图中，并返回该值
    return graph->insertConstant(maybe_out_stack->at(0), tree->range());
  }

  /**
   * 发出 fork 表达式，形如:
   *   torch.jit.fork(forked, *args, **kwargs)
   */
  std::shared_ptr<SugaredValue> emitForkExpr(
      SourceRange loc,
      const std::shared_ptr<SugaredValue>& forked,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs) {
    auto g = method.graph();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* fork_node;
    TypePtr out_type;

    // 插入 forkClosure 节点到图中，表示创建一个新的并行执行块
    fork_node = g->insertNode(method.graph()->create(prim::forkClosure, 1))
                    ->setSourceRange(loc);

    // 通过生成闭包来创建 fork，将闭包的输出设置为 fork 的输入
    // 如果闭包不存在，则创建一个新的闭包
    // (此处省略的代码会进一步创建闭包并将其输出连接到 fork_node)
    {
      // 在 fork_node 中插入代码
      WithInsertPoint insert(fork_node);
      // 检查 forked 是否为 ClosureValue 类型的指针
      if (ClosureValue* sv = dynamic_cast<ClosureValue*>(forked.get())) {
        // 将闭包值转换为普通值
        Value* closure_output = sv->asValue(loc, method);
        // 获取闭包块并注册输出类型
        Block* closure_block = closure_output->node()->blocks().at(0);
        TORCH_INTERNAL_ASSERT(closure_block->outputs().size() == 1);
        // 获取闭包块的输出类型
        out_type = closure_block->outputs().at(0)->type();
        // 将闭包输出作为 fork_node 的输入
        fork_node->addInput(closure_output);
      } else {
        // 定义一个 lambda 函数用于发射闭包体
        auto emit_closure_body = [&](Block* closure_block) {
          // 调用 forked 对象的方法，获取返回的简单值
          auto fn_sugared_output = forked->call(loc, method, args, kwargs, 1);
          auto fn_simple_output = fn_sugared_output->asValue(loc, method);
          // 注册闭包块的输出
          closure_block->registerOutput(fn_simple_output);
          // 设置输出类型为简单值的类型
          out_type = fn_simple_output->type();
        };
        // 发射闭包体并获取闭包值
        auto closure_value = emitClosure(emit_closure_body);
        // 将闭包值的简单值形式作为 fork_node 的输入
        fork_node->addInput(closure_value->asValue(loc, method));
      }
    }
    // 设置 fork_node 的输出类型为 FutureType
    Value* node_output =
        fork_node->output()->setType(FutureType::create(out_type));
    // 返回一个包含 node_output 的 SimpleValue 指针
    return std::make_shared<SimpleValue>(node_output);
  }

  // 发射可等待表达式的函数
  std::shared_ptr<SugaredValue> emitAwaitableExpr(
      SourceRange loc,
      const std::shared_ptr<SugaredValue>& awaited,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs) {
    auto g = method.graph();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    TypePtr out_type;

    // 插入一个 awaitableClosure 节点到图中
    auto await_node =
        g->insertNode(method.graph()->create(prim::awaitableClosure, 1))
            ->setSourceRange(loc);

    {
      // 在 await_node 中插入代码
      WithInsertPoint insert(await_node);
      // 检查 awaited 是否为 ClosureValue 类型的指针
      if (auto sv = dynamic_cast<ClosureValue*>(awaited.get())) {
        // 将闭包值转换为普通值
        Value* closure_output = sv->asValue(loc, method);
        // 获取闭包块并注册输出类型
        Block* closure_block = closure_output->node()->blocks().at(0);
        TORCH_INTERNAL_ASSERT(closure_block->outputs().size() == 1);
        // 获取闭包块的输出类型
        out_type = closure_block->outputs().at(0)->type();
        // 将闭包输出作为 await_node 的输入
        await_node->addInput(closure_output);
      } else {
        // 定义一个 lambda 函数用于发射闭包体
        auto emit_closure_body = [&](Block* closure_block) {
          // 调用 awaited 对象的方法，获取返回的简单值
          auto fn_sugared_output = awaited->call(loc, method, args, kwargs, 1);
          auto fn_simple_output = fn_sugared_output->asValue(loc, method);
          // 注册闭包块的输出
          closure_block->registerOutput(fn_simple_output);
          // 设置输出类型为简单值的类型
          out_type = fn_simple_output->type();
        };
        // 发射闭包体并获取闭包值
        auto closure_value = emitClosure(emit_closure_body);
        // 将闭包值的简单值形式作为 await_node 的输入
        await_node->addInput(closure_value->asValue(loc, method));
      }
    }
    // 设置 await_node 的输出类型为 AwaitType
    Value* node_output =
        await_node->output()->setType(AwaitType::create(out_type));
    // 返回一个包含 node_output 的 SimpleValue 指针
    return std::make_shared<SimpleValue>(node_output);
  }

  std::shared_ptr<SugaredValue> emitRpcExpr(const Apply& apply, Symbol rpc_op) {
    // TODO: This is a temporary apporoach to enable calling user fucntion
    // through RPC in TorchScript,
    // Ideally, function value in JIT IR is first-class citizen and
    // The RPC C++ entry API can take c10::Function directly.
    size_t rpcMinInputs = 2;
    size_t rpcMaxInputs = 5; // NOLINT
    // 将 RPC 操作符转换为非限定字符串形式的操作名
    std::string op_name = rpc_op.toUnqualString();
    
    // 检查应用的输入数量是否在预期范围内，如果不在范围内则抛出错误报告
    if (apply.inputs().size() < rpcMinInputs ||
        apply.inputs().size() > rpcMaxInputs) {
      throw ErrorReport(apply)
          << "Possible forms of call to " << op_name << "(..) are\n"
          << op_name
          << "(dst_worker_name, user_callable, args, kwargs, timeout)\n"
          << op_name << "(dst_worker_name, user_callable, args, kwargs)\n"
          << op_name << "(dst_worker_name, user_callable, args)\n"
          << op_name << "(dst_worker_name, user_callable)\n"
          << "Now the number of arguments is " << apply.inputs().size();
    }
    
    // 检查应用的属性是否为空，如果不为空则抛出错误报告
    if (!apply.attributes().empty()) {
      throw ErrorReport(apply)
          << op_name << "(dst_worker_name, user_callable, args, kwargs)"
          << "does not support kwargs yet";
    }
    
    // TODO: Make rpc_op(..) support taking kwargs,
    // like rpc_async(to="worker1", func=my_func, args=(), kwargs={})
    
    // 获取应用的输入树列表，根据第一个和第二个输入生成目标工作器名称值和用户可调用值
    auto& input_trees = apply.inputs().tree()->trees();
    Value* dst_worker_name_value = emitExpr(Expr(input_trees[0]));
    std::shared_ptr<SugaredValue> user_callable_sugared_value =
        emitSugaredExpr(Expr(input_trees[1]), 1);
    
    // 检查用户可调用值是否为函数类型，如果不是则抛出错误
    TORCH_CHECK(
        user_callable_sugared_value->kind() == "function",
        "user_callable should be a FunctionValue, it's now a ",
        user_callable_sugared_value->kind())
    
    // 使用静态强制类型转换获取用户可调用函数值
    std::shared_ptr<FunctionValue> user_callable_function_value =
        std::static_pointer_cast<FunctionValue>(user_callable_sugared_value);
    
    // 如果 `kwargs` 是空字典，则允许用户不传递 `kwargs`。
    // 如果 `args` 和 `kwargs` 分别是空元组和空字典，则允许用户不传递 `args` 和 `kwargs`。
    
    // 从输入树列表中获取除了前两个元素外的其他元素，形成参数、关键字参数和超时参数的树列表
    TreeList args_kwargs_timeout_trees(
        input_trees.begin() + 2, input_trees.end());
    
    // 获取用户可调用函数的调用者列表，并确保只有一个调用者
    const auto& callablePtrs = user_callable_function_value->callees();
    TORCH_INTERNAL_ASSERT(
        callablePtrs.size() == 1,
        "User-provided callable size should be 1. Now it's",
        callablePtrs.size())
    
    // 获取可调用指针的函数模式和应用的范围
    Function* callablePtr = callablePtrs.at(0);
    const auto& functionSchema = callablePtr->getSchema();
    const SourceRange& loc = apply.range();
    auto graphPtr = method.graph();
    
    // 匹配函数模式的命名值列表
    std::vector<NamedValue> args;
    std::vector<NamedValue> kwargs;
    // 获取参数和关键字参数作为 `NamedValue`。
    // 类似于 `getNamedValues(..)` 和 `emitAttributes(..)`.
    if (!args_kwargs_timeout_trees.empty()) {
      // 如果 args_kwargs_timeout_trees 不为空，则执行以下操作

      // 从第一个树 args_kwargs_timeout_trees[0] 中解开参数，该树已知是一个元组
      auto& args_tree = args_kwargs_timeout_trees[0];
      auto entry_sugared_values = emitSugaredExpr(Expr(args_tree), 1)
                                      ->asTuple(args_tree->range(), method);
      
      // 将解开的参数添加到 args 中
      args.reserve(entry_sugared_values.size());
      for (const auto& entrie_sugared_value : entry_sugared_values) {
        args.emplace_back(
            args_tree->range(),
            entrie_sugared_value->asValue(args_tree->range(), method));
      }
      
      // 注意：无法对 kwargs 进行模式匹配检查，因为 RPC API 是 rpc_op(to, user_callable, args, kwargs)，
      // 用户可以构造 kwargs = {"first" + "_arg" : 1}。
      // 注意键是在运行时确定的。我们无法在编译时进行，除非有一天 RPC API 是 rpc_op(to, user_callable, arg_0, arg_1, kwarg_0="foo", kwarg_1="bar")
    }

    // 匹配函数的 schema
    matchSchema(functionSchema, loc, *graphPtr, args, kwargs);

    // 将 QualifiedName 作为常量输入插入到图中
    const auto& qualname = callablePtr->qualname();
    IValue userCallableQualNameIValue(qualname.qualifiedName());
    Value* userCallableQualNameValue =
        graphPtr->insertConstant(userCallableQualNameIValue, loc);

    // 将相应的 RPC 节点插入到图中
    Node* rpc_node =
        graphPtr->insertNode(graphPtr->create(rpc_op, 1))->setSourceRange(loc);
    {
      WithInsertPoint insert(rpc_node);
      rpc_node->addInput(dst_worker_name_value);
      rpc_node->addInput(userCallableQualNameValue);

      // 将 args_kwargs_timeout_trees 中的每个树作为输入添加到 RPC 节点中
      for (const auto& tree : args_kwargs_timeout_trees) {
        rpc_node->addInput(emitExpr(Expr(tree)));
      }
    }
    Value* rpc_node_output = rpc_node->output();

    // 从 FunctionSchema 和对应的 rpc_op 设置输出类型
    const std::vector<Argument>& returns = functionSchema.returns();
    TORCH_INTERNAL_ASSERT(returns.size() == 1);
    TypePtr output_type = nullptr;
    if (rpc_op == prim::rpc_async) {
      // rpc_async 返回 functionSchema 返回类型的 FutureType
      output_type = FutureType::create(returns[0].type());
    } else if (rpc_op == prim::rpc_sync) {
      // rpc_sync 返回 functionSchema 返回类型
      output_type = returns[0].type();
    } else if (rpc_op == prim::rpc_remote) {
      // rpc_remote 返回 functionSchema 返回类型的 RRefType
      output_type = RRefType::create(returns[0].type());
    } else {
      throw ErrorReport(apply)
          << rpc_op.toDisplayString() << " is not supported in TorchScript!'";
    }
    rpc_node_output->setType(output_type);

    // 返回一个指向 rpc_node_output 的 SimpleValue 共享指针
    return std::make_shared<SimpleValue>(rpc_node_output);
  }

  // 发出二元操作的值
  Value* emitBinaryOp(const TreeRef& tree) {
    const auto& inputs = tree->trees();
    auto kind = getNodeKind(tree->kind(), inputs.size());
    auto overload = getOperatorOverload(tree->kind(), inputs.size());
    // 获取带有命名的输入值，不解包元组
    auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);

    // 如果操作符是 `in`，则参数顺序相反（被检查对象在第二个位置）
    if (tree->kind() == TK_IN) {
      // 交换命名值数组中第一个和第二个元素的位置
      std::iter_swap(named_values.begin() + 0, named_values.begin() + 1);
    }

    // 如果这是添加两个元组的操作，在这里处理
    // 原因是我们不能在注册 custom aten::add 时指定元组的长度
    if (named_values[0].type()->kind() == TupleType::Kind &&
        named_values[1].type()->kind() == TupleType::Kind &&
        kind == aten::add) {
      // 创建第一个和第二个元组的解包
      auto first_tuple = createTupleUnpack(named_values[0].value(*graph)).vec();
      auto second_tuple =
          createTupleUnpack(named_values[1].value(*graph)).vec();
      // 将第二个元组的元素插入到第一个元组的末尾
      first_tuple.insert(
          first_tuple.end(), second_tuple.begin(), second_tuple.end());
      // 创建一个新的元组节点，插入到图中，并返回其输出
      return graph->insertNode(graph->createTuple(first_tuple))->output();
    }

    // 转换成简单形式，并调用 makeMagic(overload, ...) 来创建内置函数
    return asSimple(
        makeMagic(
            overload, std::make_shared<BuiltinFunction>(kind, at::nullopt))
            ->call(tree->range(), method, named_values, {}, 0));
  }

  // 生成列表字面量的值
  Value* emitListLiteral(ListLiteral ll, const TypePtr& type_hint) {
    // 获取列表字面量的值，可能需要解包
    auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);

    // 如果值为空且没有类型提示，则创建一个空列表字面量节点
    if (values.empty() && type_hint == nullptr) {
      auto node = graph->insertNode(graph->create(prim::EmptyListLiteral));
      node->output()->setType(ListType::ofTensors());
      return node->output();
    }

    // 推断列表的元素类型，默认为 Tensor
    TypePtr inferred_elem_type = TensorType::get();

    TypePtr refined_type_hint = type_hint;

    // 如果 `type_hint` 是 Union/Optional 类型，则存储原始的 UnionType
    TypePtr annotated_union_type =
        refined_type_hint && refined_type_hint->isUnionType()
        ? refined_type_hint
        : nullptr;
    // 当我们有包含多个列表的联合类型提示时使用这个变量
    std::vector<TypePtr> all_candidates = {};
    
    if (refined_type_hint) {
      // 如果有细化的类型提示，则执行以下操作
      auto do_if_type_match = [&]() {
        // 获取列表类型提示并推断其元素类型
        auto list_type_hint = refined_type_hint->cast<ListType>();
        inferred_elem_type = list_type_hint->getElementType();
      };
    
      // 检查给定的类型是否是 AnyListType 的子类型
      auto type_match = [&](const TypePtr& t) {
        return t->isSubtypeOf(AnyListType::get());
      };
    
      // 细化并设置联合类型提示，或填充候选类型向量
      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "List",
          ll,
          type_match,
          do_if_type_match,
          do_if_type_match);
    
      // 如果候选类型向量非空且值为空，则抛出错误报告
      if (!all_candidates.empty() && values.empty()) {
        throw ErrorReport(ll)
            << "Cannot assign an empty list to a "
            << "variable annotated to be type " << refined_type_hint->repr_str()
            << " because there are multiple possible List "
            << "type candidates in the Union annotation";
      }
    }
    // 如果 values 不为空
    if (!values.empty()) {
      // 使用 fmap 函数将 values 中每个 Value* 映射为其类型，并存储在 types 中
      auto types = fmap(values, [](const Value* v) { return v->type(); });

      // 创建一个未使用的 stringstream 对象 nowhere，用于 unifyTypeList 函数
      std::stringstream nowhere; // never used

      // 如果 refined_type_hint 存在且其类型是 ListType，则取出其元素类型作为 elem_type_hint
      // 否则 elem_type_hint 为 nullptr
      const auto elem_type_hint =
          refined_type_hint && refined_type_hint->kind() == ListType::Kind
          ? refined_type_hint->cast<ListType>()->getElementType()
          : nullptr;

      // 调用 unifyTypeList 函数，尝试统一 types 中的类型，使用默认的联合类型选项，elem_type_hint 作为提示
      std::optional<TypePtr> unified_elem_type = unifyTypeList(
          types, nowhere, /*default_to_union=*/true, elem_type_hint);

      // 如果 refined_type_hint 不存在，并且 unified_elem_type 类型为 UnionType
      if (!refined_type_hint &&
          (*unified_elem_type)->kind() == UnionType::Kind) {
        // 发出警告，说明列表包含多种类型，需要在使用之前添加类型断言
        TORCH_WARN(
            "List consists of heterogeneous types, which means",
            " that it has been typed as containing ",
            (*unified_elem_type)->repr_str(),
            ". To use any of the "
            "values in this List, it will be necessary to add an "
            "`assert isinstance` statement before first use to trigger "
            "type refinement.\n",
            ll.range().str());
      }

      // 如果 all_candidates 为空，并且 refined_type_hint 存在，并且 unified_elem_type 不是 inferred_elem_type 的子类型
      if (all_candidates.empty() && refined_type_hint &&
          !(*unified_elem_type)->isSubtypeOf(*inferred_elem_type)) {
        // 抛出错误报告，说明列表的类型注释与给定列表元素的类型不匹配
        throw ErrorReport(ll)
            << "List type annotation `" << refined_type_hint->repr_str()
            << "` did not match the types of the given list elements,"
            << " which were unified to " << (*unified_elem_type)->repr_str();
      }

      // 如果 all_candidates 不为空
      if (!all_candidates.empty()) {
        // 调用 refineAndSetListTypeHintFromCandidatesVector 函数，从候选向量中提炼和设置列表类型提示
        refineAndSetListTypeHintFromCandidatesVector(
            all_candidates,
            type_hint,
            &refined_type_hint,
            *unified_elem_type,
            ll);
        // 将 inferred_elem_type 设置为经过提炼的 refined_type_hint 的元素类型
        inferred_elem_type =
            refined_type_hint->expect<ListType>()->getElementType();
      }

      // 只有当 refined_type_hint 不存在时，才设置 inferred_elem_type 为 unified_elem_type
      if (!refined_type_hint) {
        inferred_elem_type = *unified_elem_type;
      }
    }

    // 在图中插入一个节点，创建一个列表，使用 inferred_elem_type 和 values
    Node* result =
        graph->insertNode(graph->createList(inferred_elem_type, values));

    // 如果存在 annotated_union_type
    if (annotated_union_type) {
      // 插入一个节点，执行 unchecked_cast 操作，目标是 result 的输出
      Node* n = graph->insertNode(
          graph->create(prim::unchecked_cast, {result->output()}));
      // 设置节点 n 的输出类型为 annotated_union_type
      n->output()->setType(std::move(annotated_union_type));
      result = n;
    }

    // 返回 result 节点的输出作为函数结果
    return result->output();
  }

  // 函数定义 emitDictLiteral，接收一个字典字面值 dl 和类型提示 type_hint
  Value* emitDictLiteral(DictLiteral dl, const TypePtr& type_hint) {
    // 获取字典字面值的键输入的树结构的树列表
    auto key_trees = dl.key_inputs().tree()->trees();
    // 获取字典字面值的值输入的树结构的树列表
    auto value_trees = dl.value_inputs().tree()->trees();

    // 断言键树和值树的数量相同
    AT_ASSERT(key_trees.size() == value_trees.size());

    // 创建空的键和值列表，rhs_value_type 用于存储右侧值的类型
    std::vector<Value*> keys, values;
    TypePtr rhs_value_type;
    // 遍历 key_trees.size() 范围内的索引 i
    for (const auto i : c10::irange(key_trees.size())) {
      // 将每个 key_trees[i] 和 value_trees[i] 转换为表达式对象，并添加到对应的 vectors 中
      keys.push_back(emitExpr(Expr(key_trees[i])));
      values.push_back(emitExpr(Expr(value_trees[i])));

      // 如果当前索引 i 为 0，设置 rhs_value_type 为第一个值的类型
      if (i == 0) {
        rhs_value_type = values[i]->type();
      } else {
        // 如果前一个键的类型与当前键的类型不同，抛出错误报告
        if (keys[i - 1]->type()->kind() != keys[i]->type()->kind()) {
          throw ErrorReport(key_trees[i])
              << "Dict keys must contain "
              << "only a single type. Expected: "
              << keys[i - 1]->type()->repr_str() << " but found "
              << keys[i]->type()->repr_str() << " instead";
        }
        // 否则，统一当前值的类型到 rhs_value_type
        rhs_value_type = *(unifyTypes(
            rhs_value_type, values[i]->type(), /*default_to_union=*/true));
      }
    }

    // 复制 type_hint 到 refined_type_hint
    TypePtr refined_type_hint = type_hint;

    // 如果 type_hint 存在且是联合类型，将其赋值给 annotated_union_type；否则 annotated_union_type 设为 nullptr
    TypePtr annotated_union_type =
        type_hint && type_hint->isUnionType() ? type_hint : nullptr;

    // 创建空的候选类型数组
    std::vector<TypePtr> all_candidates = {};

    // 默认的 refined_type_hint 设置器函数
    auto default_refined_type_hint_setter = [&]() {
      // 如果 keys 为空，设置 refined_type_hint 为包含字符串类型和张量类型的字典
      if (keys.empty()) {
        refined_type_hint =
            DictType::create(StringType::get(), TensorType::get());
      } else {
        // 否则，设置 refined_type_hint 为包含 keys 的类型和 rhs_value_type 的字典
        refined_type_hint =
            DictType::create(keys.at(0)->type(), rhs_value_type);
        // 如果 rhs_value_type 是联合类型，发出警告
        if (rhs_value_type->kind() == UnionType::Kind) {
          TORCH_WARN(
              "Dict values consist of heterogeneous types, which means",
              " that the dict has been typed as containing ",
              refined_type_hint->repr_str(),
              ". To use any of the values in this Dict, it will be "
              "necessary to add an `assert isinstance` statement before "
              "first use to trigger type refinement.\n",
              dl.range().str());
        }
      }
    };

    // 如果 type_hint 存在
    if (type_hint) {
      // 定义类型匹配函数，用于匹配字典类型
      auto type_match = [&](const TypePtr& t) {
        return t->kind() == DictType::Kind;
      };

      // 根据 type_hint 进行类型细化或填充候选类型数组
      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "Dict",
          dl,
          type_match,
          [] {},
          default_refined_type_hint_setter);

      // 如果候选类型数组不为空且 values 为空，抛出错误报告
      if (!all_candidates.empty() && values.empty()) {
        throw ErrorReport(dl)
            << "Cannot assign an empty dict to a "
            << "variable annotated to be type " << type_hint->repr_str()
            << " because there are multiple possible Dict "
            << "type candidates in the Union annotation";
      }
    } else {
      // 否则，调用默认的 refined_type_hint 设置器函数
      default_refined_type_hint_setter();
    }

    // 必须要么已经具有特定的键/值类型，要么有一组可能的候选类型
    TORCH_INTERNAL_ASSERT(!all_candidates.empty() || refined_type_hint);
    // 如果值不为空
    if (!values.empty()) {
      // 如果候选集不为空
      if (!all_candidates.empty()) {
        // 根据候选集精细化和设置字典类型提示
        refineAndSetDictTypeHintFromCandidatesVector(
            all_candidates,
            type_hint,
            &refined_type_hint,
            keys[0]->type(),
            rhs_value_type,
            dl);
      }

      // 检查精细化后的字典键类型是否与给定的键类型一致
      if (refined_type_hint->expect<DictType>()->getKeyType() !=
          keys.at(0)->type()) {
        throw ErrorReport(dl)
            << "Type annotation was inferred to be "
            << refined_type_hint->repr_str()
            << "but the type of keys given by the dict literal is "
            << keys.at(0)->type()->repr_str();
      }

      // 检查字典值类型是否符合预期的类型注解
      if (!rhs_value_type->isSubtypeOf(
              refined_type_hint->expect<DictType>()->getValueType())) {
        throw ErrorReport(dl)
            << "Type annotation was inferred to be `"
            << refined_type_hint->repr_str()
            << "`, but the type of values given by the dict literal is "
            << rhs_value_type->repr_str();
      }
    }

    // 根据精细化后的类型提示创建字典节点
    Node* result = graph->insertNode(graph->createDict(
        refined_type_hint->expect<DictType>()->getKeyType(),
        refined_type_hint->expect<DictType>()->getValueType(),
        keys,
        values));
    
    // 如果存在联合类型注解
    if (annotated_union_type) {
      // 进行未经检查的类型转换操作
      Node* n = graph->insertNode(
          graph->create(prim::unchecked_cast, {result->output()}));
      // 设置节点输出的类型为注解的联合类型
      n->output()->setType(std::move(annotated_union_type));
      result = n;
    }

    // 返回结果节点的输出值
    return result->output();
  }

  // 发射简单表达式
  Value* emitSimpleExpr(const TreeRef& tree, TypePtr type_hint = nullptr) {
    // 空实现，待实现
    }
  }

  // 发射常量
  Value* emitConst(const Const& c) {
    // 如果常量是浮点数，返回浮点数的材料化常量
    if (c.isFloatingPoint())
      return materializeConstant(
          c.asFloatingPoint(), *graph, c.range(), fp_constants);
    // 如果常量是复数，返回复数的材料化常量
    else if (c.isComplex())
      return materializeConstant(
          c.asComplex(), *graph, c.range(), complex_constants);
    // 否则，返回整数的材料化常量
    else
      return materializeConstant(
          c.asIntegral(), *graph, c.range(), integral_constants);
  }

  // 发射字符串字面量
  Value* emitStringLiteral(const StringLiteral& c) {
    // 插入字符串常量并返回其值
    return insertConstant(*graph, c.text(), c.range());
  }

  // 发射选择操作：tensor[i] -> tensor.select(dim, i)
  // 对索引操作进行解糖处理
  Value* emitSelect(
      const SourceRange& loc,
      Value* input,
      Value* dim,
      Value* index) {
    // 调用内置函数调用以执行选择操作
    return emitBuiltinCall(loc, *graph, aten::select, {input, dim, index}, {});
  }

  // 发射切片操作
  Value* emitSliceOp(
      const SourceRange& loc,
      Value* sliceable,
      Value* dim,
      Value* start,
      Value* end,
      Value* step) {
    // 准备参数列表
    std::vector<NamedValue> args;
    args.reserve(5);
    args.emplace_back(loc, "self", sliceable);

    // 如果存在维度参数
    // XXX: 如果列表切片变得更复杂或不再使用aten::slice，应该将其与此函数分离。
    if (dim) {
      // 断言切片对象的类型是张量类型的子类型
      AT_ASSERT(sliceable->type()->isSubtypeOf(*TensorType::get()));

      args.emplace_back(dim);
    } else {
      // 断言切片对象的类型不是张量类型的子类型
      AT_ASSERT(!sliceable->type()->isSubtypeOf(*TensorType::get()));
    }
    // 如果 sliceable 是 TupleType 类型的指针
    if (sliceable->type()->cast<TupleType>()) {
      // 创建一个存储可选命名值的向量，用于处理元组切片
      std::vector<at::optional<NamedValue>> tuple_args;
      // 因为只处理元组切片，暂时保持元组参数分开
      tuple_args.reserve(3);

      // 根据情况将 start 添加到 tuple_args 中，如果不存在则添加空值
      start ? tuple_args.emplace_back(start)
            : tuple_args.emplace_back(c10::nullopt);
      // 根据情况将 end 添加到 tuple_args 中，如果不存在则添加空值
      end ? tuple_args.emplace_back(end)
          : tuple_args.emplace_back(c10::nullopt);
      // 根据情况将 step 添加到 tuple_args 中，如果不存在则添加空值
      step ? tuple_args.emplace_back(step)
           : tuple_args.emplace_back(c10::nullopt);

      // 调用 emitTupleSlice 函数处理元组切片，并返回结果
      return emitTupleSlice(loc, args[0], tuple_args);
    }

    // 处理类似 x[0:2]. x[0:2:] 的情况，Python 中已经处理了 x[0:2:]
    if (!step) {
      // 如果 step 为空，则插入一个常量值 1
      step = graph->insertConstant(1, loc);
    }

    // 将 start、end、step 添加到 args 中
    args.emplace_back(loc, "start", start);
    args.emplace_back(loc, "end", end);
    args.emplace_back(loc, "step", step);

    // 调用 emitBuiltinCall 函数生成内置函数调用并返回结果
    return emitBuiltinCall(loc, *graph, aten::slice, args, {});
  }

  // 将切片索引进行解析：tensor[begin:end] -> tensor.slice(dim, begin, end, 1)
  Value* emitSlice(
      const SourceRange& loc,
      Value* input,
      Value* dim, // 仅用于张量切片
      const SliceExpr& slice) {
    Value* start = nullptr;
    Value* end = nullptr;
    Value* step = nullptr;

    // 如果切片有起始值，则生成表达式并赋给 start
    if (slice.start().present()) {
      start = emitExpr(Expr(slice.start().get()));
    }
    // 如果切片有结束值，则生成表达式并赋给 end
    if (slice.end().present()) {
      end = emitExpr(Expr(slice.end().get()));
    }
    // 如果切片有步长值，则生成表达式并赋给 step
    if (slice.step().present()) {
      step = emitExpr(Expr(slice.step().get()));
    }

    // 调用 emitSliceOp 函数生成切片操作并返回结果
    return emitSliceOp(loc, input, dim, start, end, step);
  }

  // 发出 unsqueeze 操作
  Value* emitUnsqueeze(const SourceRange& loc, Value* input, Value* dim_val) {
    // 调用 emitBuiltinCall 函数生成 unsqueeze 内置函数调用并返回结果
    return emitBuiltinCall(loc, *graph, aten::unsqueeze, {input, dim_val}, {});
  }

  // 发出索引操作
  Value* emitIndex(
      const SourceRange& loc,
      Value* input,
      at::ArrayRef<Value*> indices) {
    // 注意：aten::index 的索引应该是 List[Optional[Tensor]] 类型，以支持例如 t[:, :, 1] 这样的情况
    auto* index =
        // 创建一个带有可选 Tensor 类型的列表节点
        graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
            ->output();
    // 调用 emitBuiltinCall 函数生成 index 内置函数调用并返回结果
    return emitBuiltinCall(loc, *graph, aten::index, {input, index}, {});
  }

  // 发出包含整数和切片索引的多维切片操作
  // 返回：
  // - Value*: 经过整数和切片索引后的输入
  // - vector<Value*>: 未应用索引的张量 Value* 列表，在切片后的 sliceable 不是张量时应为 NULL
  std::pair<Value*, std::vector<Value*>> emitIntAndSliceIndexing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    // 总体上，为了处理除张量以外的索引，我们需要处理几种不同的情况。例如，对于 x[1:3, None, 4]，每种不同的索引类型（切片、None 和整数）导致不同的处理方式
    // ...
    // 定义一个存储张量索引的向量
    std::vector<Value*> tensor_indices;

    // 定义一个 lambda 函数，用于插入维度值
    auto insert_value_for_dim = [&](int64_t dim) {
      return graph->insertConstant(dim, loc);
    };
    
    // 存储子表达式的维度和表达式的向量
    std::vector<int64_t> dims(subscript_exprs.size());
    std::vector<std::optional<Value*>> exprs(
        subscript_exprs.size(), c10::nullopt);

    // 遍历子表达式，直到遇到省略号
    size_t idx = 0;
    int64_t dim = 0;
    for (; idx < subscript_exprs.size(); idx++) {
      auto subscript_expr = subscript_exprs[idx];
      if (subscript_expr.kind() == TK_DOTS) {
        break;
      }
      dim = handle_indexing(subscript_expr, idx, dim, /*is_reverse=*/false);
    }
    
    // 从右向左遍历子表达式，直到遇到省略号
    int64_t rdim = -1;
    for (size_t rev_idx = subscript_exprs.size() - 1; rev_idx > idx;
         rev_idx--) {
      auto subscript_expr = subscript_exprs[rev_idx];
      if (subscript_expr.kind() == TK_DOTS) {
        throw ErrorReport(loc)
            << "An index can only have a single ellipsis ('...')";
      }
      rdim =
          handle_indexing(subscript_expr, rev_idx, rdim, /*is_reverse=*/true);
    }
    // 对表达式列表进行迭代，exprs.size() 是表达式列表的大小
    for (const auto i : c10::irange(exprs.size())) {
      // 如果表达式未定义
      if (!exprs[i].has_value()) {
        // 如果子表达式的类型是 TK_SLICE_EXPR，则生成对应的切片操作
        if (subscript_exprs[i].kind() == TK_SLICE_EXPR) {
          sliceable = emitSlice(
              loc,
              sliceable,
              insert_value_for_dim(dims[i]),
              SliceExpr(subscript_exprs[i]));
          continue; // 继续下一个迭代
        }
    
        // 如果子表达式的类型是 TK_DOTS，则跳过当前迭代
        if (subscript_exprs[i].kind() == TK_DOTS) {
          continue;
        }
    
        // 对子表达式进行发射处理，获取其 Sugar 表示
        auto subscript_sv = emitSugaredExpr(subscript_exprs[i], 1);
        // 如果子表达式的 Sugar 表示是 SliceValue 类型
        if (const auto slice_value =
                dynamic_cast<SliceValue*>(subscript_sv.get())) {
          // 生成对应的切片操作
          sliceable = emitSliceOp(
              loc,
              sliceable,
              insert_value_for_dim(dims[i]),
              slice_value->start(),
              slice_value->stop(),
              slice_value->step());
        }
    
        continue; // 继续下一个迭代
      }
    
      // 获取当前表达式
      auto expr = exprs[i].value();
      // 如果表达式的类型是 NoneType 类型的子类型
      if (expr->type()->isSubtypeOf(*NoneType::get())) {
        // 生成对应的 unsqueeze 操作
        sliceable =
            emitUnsqueeze(loc, sliceable, insert_value_for_dim(dims[i]));
      } else if (expr->type() == IntType::get()) {
        // 如果表达式的类型是 IntType 类型
        // 生成对应的 select 操作
        sliceable =
            emitSelect(loc, sliceable, insert_value_for_dim(dims[i]), expr);
      } else if (expr->type()->isSubtypeOf(*OptionalType::ofTensor())) {
        // 如果表达式的类型是 OptionalType 的 Tensor 类型的子类型
        // 调整 tensor_indices 的大小，并赋值对应维度的表达式
        tensor_indices.resize(dims[i] + 1);
        tensor_indices[dims[i]] = expr;
      } else {
        // 如果表达式的类型无法支持，则抛出内部断言错误
        TORCH_INTERNAL_ASSERT(
            false, "Trying to process index type that we don't support.");
      }
    }
    
    // 遍历 tensor_indices 中的每个索引
    for (auto& index : tensor_indices) {
      // 如果索引为空指针，则插入一个 None 节点并赋值给索引
      if (index == nullptr) {
        index = graph->insertNode(graph->createNone())->output();
      }
    }
    
    // 返回切片后的结果 sliceable 和索引列表 tensor_indices 的 pair
    return std::make_pair(sliceable, tensor_indices);
    }
    
    
    
    // 将多维切片转换为 slice/select/index/unsqueeze 调用序列
    // 
    // XXX: 用户代码中的错误信息没有优雅的报告方式。
    // 假设有人这样做：
    //   @torch.jit.script
    //   def fn(x):
    //       return x[0, 1]
    //   fn(torch.randn(5))
    // 因为我们将其解析为两个 aten::select 操作，错误消息将抱怨 aten::select 操作失败，
    // 而不是 "没有足够的维度来索引"。
    //
    // 策略是首先在一次遍历中对整数和切片进行切片选择，然后在结果上应用 at::index。
    // 在应用了切片/选择后称之为 `sliced` 的张量后调用张量。tensor_indices 应该与 sliced.dim() 大小相同：
    // - 如果我们不应该在维度 i 上对 `sliced` 进行索引，则 tensor_indices[i] = NULL
    // - 如果我们应该使用张量 t 在维度 i 上对 `sliced` 进行索引，则 tensor_indices[i] = t。
    Value* emitMultidimSlicing(
        const SourceRange& loc,
        Value* sliceable,
        const List<Expr>& subscript_exprs) {
    // 如果 sliceable 不是 TensorType 的子类型，则抛出错误
    if (!sliceable->type()->isSubtypeOf(*TensorType::get())) {
      throw ErrorReport(loc)
          << "Unsupported operation: attempted to use multidimensional "
          << "indexing on a non-tensor type";
    }

    // 创建一个空的 tensor_indices 向量，准备用于存储索引操作后的结果
    std::vector<Value*> tensor_indices;

    // 调用 emitIntAndSliceIndexing 函数，处理整数和切片索引操作，并更新 sliceable 和 tensor_indices
    std::tie(sliceable, tensor_indices) =
        emitIntAndSliceIndexing(loc, sliceable, subscript_exprs);

    // 如果 tensor_indices 为空，返回 sliceable 对象，暂时不支持 mutability
    if (tensor_indices.empty()) {
      // XXX: 当我们支持 mutability 时可能需要调用 at::alias
      return sliceable;
    }

    // 调用 emitIndex 函数，生成索引操作后的结果，并返回
    return emitIndex(loc, sliceable, tensor_indices);
  }

  // 将切片语法糖 tensor[begin:end] 转换为 tensor.slice(begin, end) 的语法
  Value* emitBasicSlice(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    AT_ASSERT(subscript_exprs.size() == 1);
    AT_ASSERT(subscript_exprs[0].kind() == TK_SLICE_EXPR);
    auto slice_exp = SliceExpr(subscript_exprs[0]);
    Value* maybe_dim = nullptr;
    // 如果 sliceable 是 tensor 类型，指定默认的维度为 0
    if (sliceable->type()->isSubtypeOf(*TensorType::get())) {
      maybe_dim = graph->insertConstant(0, loc);
    }
    // 调用 emitSlice 函数，生成切片操作的结果，并返回
    return emitSlice(loc, sliceable, maybe_dim, slice_exp);
  }

  // 获取调整后的元组索引
  int64_t getAdjTupleIndex(
      const SourceRange& loc,
      const TupleTypePtr& tuple_type,
      int64_t input_index,
      bool allow_out_of_bounds) {
    // 将输入索引调整为正数，简化运行时逻辑
    int64_t adj_index = input_index;
    int64_t tuple_len = tuple_type->elements().size();
    // 如果输入索引为负数，转换为相对于元组长度的正数索引
    if (input_index < 0) {
      adj_index = tuple_len + input_index;
    }
    // 如果不允许超出边界并且调整后的索引超出元组长度范围，抛出错误
    if (!allow_out_of_bounds && (adj_index >= tuple_len || adj_index < 0)) {
      throw ErrorReport(loc) << "Tuple index out of range. Tuple is length "
                             << tuple_len << " and index is " << input_index;
    }
    // 返回调整后的索引
    return adj_index;
  }

  // 当一个列表在模块中标记为 const 时，会被转换为元组
  // 索引一个只包含一个类型的元组是很常见的情况
  // 因为索引操作通常在循环中完成，我们不希望在每次迭代时将元组转换为列表，以避免额外开销
  Value* emitTupleIndex(
      const SourceRange& loc,
      Value* tuple_val,
      Value* idx_val) {
    auto tuple_typ = tuple_val->type()->cast<TupleType>();
    auto elems = tuple_typ->elements();
    TypePtr output_type;
    // 确保 idx_val 是 IntType 类型的整数
    if (idx_val->type() != IntType::get()) {
      throw ErrorReport(loc) << "tuple index must be an integer";
    }
    // 尝试将 idx_val 转换为 IValue
    auto idx = toIValue(idx_val);
    if (!idx) {
      // 如果 idx 为 null，并且元素为空或无法将 tuple_typ 转换为 ListType 的元素类型，抛出错误
      if (elems.empty() ||
          !convertibleToList(tuple_typ, ListType::create(elems[0]))) {
        throw ErrorReport(loc)
            << "Cannot index into a " << tuple_typ->repr_str()
            << " with a non-integer literal because we cannot resolve the output type";
      }
      output_type = elems[0];
  std::shared_ptr<SugaredValue> emitSubscript(
      const Subscript& subscript,
      TypePtr type_hint = nullptr) {
    // 提取子表达式的求值结果
    const SugaredValuePtr sv = emitSugaredExpr(subscript.value(), 1);
    // 获取下标表达式列表
    const List<Expr>& subscript_exprs = subscript.subscript_exprs();
    // 获取整个下标操作的源代码范围
    const SourceRange& range = subscript.range();
    // 获取被索引值的源代码范围
    const SourceRange& val_range = subscript.value().range();

    // 如果下标表达式数量不为1，则执行多维切片操作
    if (subscript_exprs.size() != 1) {
      // 返回一个简单值的共享指针，表示多维切片的结果
      return std::make_shared<SimpleValue>(emitMultidimSlicing(
          range, sv->asValue(val_range, method), subscript_exprs));
    }

    // 如果下标表达式数量为1，执行单一维度切片操作
    } else {
      // 获取调整后的切片索引
      auto adj_index = getAdjTupleIndex(
          loc, tuple_typ, idx->toInt(), /*allow_out_of_bounds*/ false);
      // 确定输出类型为调整后的索引处的元素类型
      output_type = elems[adj_index];
    }
    // 在计算图中插入节点，创建元组切片操作，并返回其输出
    return graph
        ->insertNode(graph->createTupleSlice(
            tuple_val.value(*graph), beg, step_size, num_values))
        ->output();
  }
    // 如果第一个下标表达式的类型是 TK_SLICE_EXPR
    if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
      // TODO @wconstab 使用符号而非字符串比较进行重构
      // 检查当前 sv 的类型是否为 "module"
      if (sv->kind() == "module") {
        // 对于 Sequential/ModuleList，目前不支持切片操作，但对于元组支持切片，因此可以将其转换为模块的元组以支持切片操作。
        auto s_tuple_val =
            sv->asTupleValue(val_range, method)->asValue(val_range, method);
        // 获取切片表达式的具体信息
        const SliceExpr& slice = SliceExpr(subscript_exprs[0]);
        // 准备存储切片参数的向量
        std::vector<at::optional<NamedValue>> tuple_args;
        tuple_args.reserve(3);
        // 如果切片有起始值，则创建起始值的命名参数
        if (slice.start().present()) {
          auto begin = NamedValue(
              val_range, "begin", emitExpr(Expr(slice.start().get())));
          tuple_args.emplace_back(begin);
        } else {
          tuple_args.emplace_back(c10::nullopt);
        }

        // 如果切片有终止值，则创建终止值的命名参数
        if (slice.end().present()) {
          auto end =
              NamedValue(val_range, "end", emitExpr(Expr(slice.end().get())));
          tuple_args.emplace_back(end);
        } else {
          tuple_args.emplace_back(c10::nullopt);
        }

        // 如果切片有步长值，则创建步长值的命名参数
        if (slice.step().present()) {
          auto step =
              NamedValue(val_range, "step", emitExpr(Expr(slice.step().get())));
          tuple_args.emplace_back(step);
        } else {
          tuple_args.emplace_back(c10::nullopt);
        }

        // 发出元组切片的代码并获取结果
        auto tupleSliceValue =
            emitTupleSlice(val_range, s_tuple_val, tuple_args);
        // 返回一个包含元组切片值的 SimpleValue 智能指针
        return std::make_shared<SimpleValue>(tupleSliceValue);
      } else {
        // 如果 sv 不是 "module" 类型，则发出基本切片操作并返回结果
        return std::make_shared<SimpleValue>(emitBasicSlice(
            range, sv->asValue(val_range, method), subscript_exprs));
      }
    // 处理 else 分支，即 subscript_exprs 的大小为 1 时的情况
    } else {
      // 断言 subscript_exprs 的大小为 1
      AT_ASSERT(subscript_exprs.size() == 1);
      // 将 sv 转换为 Value* 类型
      Value* sliceable = sv->asValue(val_range, method);

      // 如果 subscript 表达式是 Python 的 Slice 对象
      // 则将其转换为对应的 SugaredValue
      auto subscript_sv = emitSugaredExpr(subscript_exprs[0], 1);
      if (const auto slice_value =
              dynamic_cast<SliceValue*>(subscript_sv.get())) {
        Value* dim = nullptr;
        // 如果 sliceable 是 Tensor 类型，则需要额外的 dim 输入
        if (sliceable->type()->isSubtypeOf(*TensorType::get())) {
          // 插入常量 0 作为 dim
          dim = method.graph()->insertConstant(0, val_range);
        }

        // 发出 Slice 操作并获取结果值
        Value* sliced = emitSliceOp(
            val_range,
            sliceable,
            dim,
            slice_value->start(),
            slice_value->stop(),
            slice_value->step());
        // 返回封装后的 SimpleValue 对象
        return std::make_shared<SimpleValue>(sliced);
      }

      // 如果 subscript 不是 Slice 对象，则必须可转换为普通值
      // 对 gather 语法糖 foo[i] 进行解糖
      Value* idx = subscript_sv->asValue(val_range, method);
      if (sliceable->type()->cast<TupleType>()) {
        // 如果 sliceable 是 Tuple 类型，则返回 tuple 的索引值
        return std::make_shared<SimpleValue>(
            emitTupleIndex(range, sv->asValue(val_range, method), idx));
      } else if (sliceable->type()->isSubtypeOf(*TensorType::get())) {
        // 如果 sliceable 是 Tensor 类型，则进行多维切片操作
        return std::make_shared<SimpleValue>(
            emitMultidimSlicing(range, sliceable, subscript_exprs));
      } else {
        // 否则调用 getitem 方法获取项
        return sv->getitem(range, method, idx, std::move(type_hint));
      }
    }
  }
};

// FunctionResolver类继承自Resolver类，用于解析函数相关的值
struct FunctionResolver : public Resolver {
  // 构造函数，初始化其他解析器和函数表
  explicit FunctionResolver(
      Resolver* otherResolver,
      const std::unordered_map<std::string, Function*>& functionTable)
      : otherResolver_(otherResolver), functionTable_(functionTable) {}

  // 解析值的方法，根据名称在函数表中查找函数并返回其对应的FunctionValue
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    auto it = functionTable_.find(name);
    if (it != functionTable_.end()) {
      return std::make_shared<FunctionValue>(it->second);
    }
    // 如果在函数表中找不到，则委托给其他解析器处理
    return otherResolver_->resolveValue(name, m, loc);
  }

  // 解析类型的方法，委托给其他解析器处理
  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return otherResolver_->resolveType(name, loc);
  }

 private:
  Resolver* otherResolver_;  // 其他解析器对象指针
  const std::unordered_map<std::string, Function*>& functionTable_;  // 函数表，映射函数名称到函数对象的指针
};

// CompilationUnit类的构造函数，接受源代码字符串，并使用nativeResolver()来定义函数图形
CompilationUnit::CompilationUnit(const std::string& source)
    : CompilationUnit() {
  // 使用nativeResolver()定义函数图形
  define(c10::nullopt, source, nativeResolver(), nullptr);
}

// CompilationUnit类中表示一个属性对的结构体，包含从编译属性中获取的getter和setter函数对
struct CompilationUnit::PropertyPair
    : public std::pair<std::unique_ptr<Function>, std::unique_ptr<Function>> {
  PropertyPair(
      std::unique_ptr<Function> getter,
      std::unique_ptr<Function> setter) {
    TORCH_INTERNAL_ASSERT(getter, "Property pair must have defined getter")
    this->first = std::move(getter);  // 移动getter函数对象
    this->second = std::move(setter);  // 移动setter函数对象
  }

  // 获取getter函数的引用
  std::unique_ptr<Function>& getGetter() {
    return this->first;
  }

  // 获取setter函数的引用
  std::unique_ptr<Function>& getSetter() {
    return this->second;
  }
};

// CompilationUnit类中定义属性的方法，返回一个属性对，包含编译得到的getter和setter函数
CompilationUnit::PropertyPair CompilationUnit::define_property(
    const std::optional<c10::QualifiedName>& prefix,
    const Property& prop,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,
    bool shouldMangle) const {
  // 断言self必须被定义，因为属性是类和模块的特性
  TORCH_INTERNAL_ASSERT(self);

  // 编译getter函数
  std::unique_ptr<Function> getter_fn = define(
      prefix, prop.getter(), resolver, self, function_table, shouldMangle);

  // 如果存在setter函数，则编译它
  std::unique_ptr<Function> setter_fn = nullptr;
  if (prop.setter().present()) {
    setter_fn = define(
        prefix,
        prop.setter().get(),
        resolver,
        self,
        function_table,
        shouldMangle);
  }

  // 将属性添加到类类型定义中
  self->getClassType()->addProperty(
      prop.name().name(), getter_fn.get(), setter_fn.get());

  // 返回包含getter和setter函数的属性对
  return PropertyPair(std::move(getter_fn), std::move(setter_fn));
}

// CompilationUnit类中定义函数的方法，返回一个唯一指针指向定义的函数对象
std::unique_ptr<Function> CompilationUnit::define(
    const std::optional<QualifiedName>& prefix,
    const Def& def,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,  // 输入：一个字符串到函数指针的无序映射表
    bool shouldMangle,  // 输入：一个布尔值，指示是否应该对函数名称进行编码
    CompilationUnit::FunctionType type,  // 输入：一个枚举类型，表示函数的类型
    std::optional<size_t> operator_set_version) const {  // 输入：一个可选的操作集版本号
  TORCH_INTERNAL_ASSERT(resolver);  // 断言：确保 resolver 不为空
  auto _resolver = resolver;  // 复制 resolver 到 _resolver
  if (!self) {  // 如果 self 为空指针
    // 如果 self 未定义，那么这些函数是方法，并不进入全局命名空间，否则它们将一起定义，因此将它们添加到函数表中，以便方法可以相互看到
    _resolver =
        std::make_shared<FunctionResolver>(resolver.get(), function_table);  // 创建一个新的函数解析器，继承现有 resolver 和 function_table
  }
  auto creator = [def, _resolver, self](GraphFunction& method) {
    // 存储函数名，以便在编译此函数时可以引用它，如果出现错误
    std::string call_name = method.qualname().name();
    if (self) {  // 如果 self 不为空指针
      auto atoms = method.qualname().atoms();
      // 至少应该有一个 ClassName.method_name
      TORCH_INTERNAL_ASSERT(atoms.size() >= 2);  // 断言：确保 atoms 的大小至少为 2
      call_name = atoms.at(atoms.size() - 2) + "." + atoms.at(atoms.size() - 1);  // 构造方法调用名称
    }
    ErrorReport::CallStack call(call_name, def.range());  // 创建一个错误报告的函数调用堆栈
    to_ir(def, _resolver, self, method);  // 将定义转换为中间表示(IR)
  };
  auto name = prefix ? QualifiedName(*prefix, def.name().name())  // 如果前缀不为空，则使用前缀和函数名构造 QualifiedName
                     : QualifiedName(def.name().name());  // 否则只使用函数名构造 QualifiedName
  if (shouldMangle) {  // 如果 shouldMangle 为真
    // 如果 shouldMangle 被设置，如果已经存在同名函数，我们应该为该函数生成一个唯一的名称
    if (find_function(name)) {  // 如果在函数表中找到了同名函数
      name = mangle(name);  // 对函数名进行编码处理
    }
  }

  auto graph = std::make_shared<Graph>();  // 创建一个新的图对象
  graph->set_op_version(operator_set_version);  // 设置图对象的操作版本号

  auto fn = std::make_unique<GraphFunction>(std::move(name), graph, creator);  // 创建一个新的图函数对象
  if (self) {  // 如果 self 不为空指针
    // 将此函数注册为 self 类型的方法
    if (type == CompilationUnit::FunctionType::Hook) {  // 如果函数类型是 Hook
      self->getClassType()->addForwardHook(fn.get());  // 将函数添加为前向钩子
    } else if (type == CompilationUnit::FunctionType::PreHook) {  // 如果函数类型是 PreHook
      self->getClassType()->addForwardPreHook(fn.get());  // 将函数添加为前向预处理钩子
    } else {  // 否则
      self->getClassType()->addMethod(fn.get());  // 将函数添加为方法
    }
  }
  return fn;  // 返回创建的函数对象指针
}
}

std::vector<Function*> CompilationUnit::define(
    const std::optional<c10::QualifiedName>& prefix,
    const std::vector<Property>& properties,
    const std::vector<ResolverPtr>& propResolvers,
    const std::vector<Def>& definitions,
    const std::vector<ResolverPtr>& defResolvers,
    const Self* self,
    bool shouldMangle,
    std::optional<size_t> operator_set_version) {
  // 确保定义的属性和属性解析器数量相等
  TORCH_INTERNAL_ASSERT(definitions.size() == defResolvers.size());
  // 确保属性和属性解析器数量相等
  TORCH_INTERNAL_ASSERT(properties.size() == propResolvers.size());
  // 存储所有函数的指针
  std::vector<Function*> functions;
  // 函数名到函数指针的映射表
  std::unordered_map<std::string, Function*> function_table;

  // 记录函数到表格、functions，并使用 register_function 注册函数
  // 这段代码会多次使用，因此使用 lambda 函数避免重复代码
  auto record_function = [&](std::unique_ptr<Function> fn) {
    // 将函数添加到函数表中
    function_table[fn->name()] = fn.get();
    // 将函数指针添加到 functions 中
    functions.emplace_back(fn.get());
    // 调用当前对象的 register_function 方法注册函数
    this->register_function(std::move(fn));
  };

  // 遍历属性数组
  for (const auto i : c10::irange(properties.size())) {
    // 定义属性的 getter 和 setter 函数
    PropertyPair property_fns = define_property(
        prefix,
        properties[i],
        propResolvers[i],
        self,
        function_table,
        shouldMangle);

    // 获取 getter 和 setter 函数的引用
    auto& getter_fn = property_fns.getGetter();
    auto& setter_fn = property_fns.getSetter();

    // 记录 getter 函数
    record_function(std::move(getter_fn));

    // 如果存在 setter 函数，则记录 setter 函数
    if (setter_fn) {
      record_function(std::move(setter_fn));
    }
  }

  // 遍历定义数组
  for (const auto i : c10::irange(definitions.size())) {
    // 定义函数
    auto fn = define(
        prefix,
        definitions[i],
        defResolvers[i],
        self,
        function_table,
        shouldMangle,
        CompilationUnit::FunctionType::Method,
        operator_set_version);

    // 记录函数
    record_function(std::move(fn));
  }

  // 确保首先编译 `__init__` 函数，因为它可以确定其他方法可用的属性。
  // 因此需要相应地重新排序定义。
  for (auto& kv : function_table) {
    if (kv.first == "__init__") {
      // 确保 `__init__` 函数已定义
      kv.second->ensure_defined();
    }
  }

  // 确保所有函数都已定义
  for (Function* function : functions) {
    function->ensure_defined();
  }

  // 返回所有定义的函数列表
  return functions;
}

void CompilationUnit::define_hooks(
    const std::optional<c10::QualifiedName>& prefix,
    const std::vector<Def>& hookDefs,
    const std::vector<ResolverPtr>& hookResolvers,
    const std::vector<Def>& preHookDefs,
    const std::vector<ResolverPtr>& preHookResolvers,
    const Self* self,
    bool shouldMangle) {
  // 确保 hookDefs 和 hookResolvers 数量相等
  TORCH_INTERNAL_ASSERT(hookDefs.size() == hookResolvers.size());
  // 确保 preHookDefs 和 preHookResolvers 数量相等
  TORCH_INTERNAL_ASSERT(preHookDefs.size() == preHookResolvers.size());
  // 存储所有函数的指针
  std::vector<Function*> functions;
  // 函数名到函数指针的映射表
  std::unordered_map<std::string, Function*> function_table;

  // 检查钩子函数是否存在命名冲突和重新定义
  auto check_collisions = [&](const Def& hook) -> Function* {
    // 获取函数名称
    auto name = prefix ? QualifiedName(*prefix, hook.name().name()).name()
                       : QualifiedName(hook.name().name()).name();
    // 检查该模块是否已经定义了这个钩子函数
  auto found_hook = function_table.find(name);
  // 在函数表中查找给定名称的钩子函数
  auto existing_hook =
      found_hook != function_table.end() ? found_hook->second : nullptr;
  // 如果找到，获取对应的钩子函数；否则设置为nullptr

  // 检查如果钩子名称已经在模块中定义为方法，则抛出错误
  if (existing_hook == nullptr) {
    TORCH_CHECK(
        self->getClassType()->findMethod(name) == nullptr &&
            self->getClassType()->findHook(name) == nullptr,
        "Can't define hook: ",
        name,
        " on class: ",
        self->getClassType()->repr_str(),
        " because a method or hook with that name already exists.");
  }
  // 返回现有的钩子函数或nullptr

  return existing_hook;
};

// 用于构建钩子函数模式的辅助函数
auto build_schema = [&](const Def& hook_def,
                        const ResolverPtr& hook_res) -> FunctionSchema {
  ScriptTypeParser typeParser(hook_res);
  // 从钩子定义中解析函数模式
  FunctionSchema schema =
      typeParser.parseSchemaFromDef(hook_def, true /* skip_self*/);
  // 需要将self作为第一个参数添加，因为我们跳过了它
  std::vector<Argument> arguments;
  arguments.emplace_back(
      hook_def.decl().params()[0].ident().name(), self->getClassType());
  arguments.insert(
      arguments.end(), schema.arguments().begin(), schema.arguments().end());
  // 返回更新后的函数模式
  return schema.cloneWithArguments(arguments);
};

// 定义钩子函数
for (const auto i : c10::irange(hookDefs.size())) {
  // 检查是否已经定义了该钩子函数
  auto existing_fn = check_collisions(hookDefs[i]);
  if (existing_fn != nullptr) {
    // 将现有的钩子函数再次添加到类类型中，以便调用
    self->getClassType()->addForwardHook(existing_fn);
    continue;
  }
  // 定义新的钩子函数
  auto fn = define(
      prefix,
      hookDefs[i],
      hookResolvers[i],
      self,
      function_table,
      shouldMangle,
      CompilationUnit::FunctionType::Hook);

  function_table[fn->name()] = fn.get();
  functions.emplace_back(fn.get());
  this->register_function(std::move(fn));
  // 检查并注册钩子函数的模式
  self->getClassType()->checkForwardHookSchema(
      i, build_schema(hookDefs[i], hookResolvers[i]));
  // 确保函数定义完成
  functions.back()->ensure_defined();
}

// 定义预先钩子函数
for (const auto i : c10::irange(preHookDefs.size())) {
  // 检查是否已经定义了该预先钩子函数
  auto existing_fn = check_collisions(preHookDefs[i]);
  if (existing_fn != nullptr) {
    // 将现有的预先钩子函数再次添加到类类型中，以便调用
    self->getClassType()->addForwardPreHook(existing_fn);
    continue;
  }
  // 定义新的预先钩子函数
  auto fn = define(
      prefix,
      preHookDefs[i],
      preHookResolvers[i],
      self,
      function_table,
      shouldMangle,
      CompilationUnit::FunctionType::PreHook);

  function_table[fn->name()] = fn.get();
  functions.emplace_back(fn.get());
  this->register_function(std::move(fn));
  // 检查并注册预先钩子函数的模式
  self->getClassType()->checkForwardPreHookSchema(
      i, build_schema(preHookDefs[i], preHookResolvers[i]));
  // 确保函数定义完成
  functions.back()->ensure_defined();
}
}

// CompilationUnit 类的 define 方法，用于解析给定源代码中的函数定义并返回一个函数指针向量
std::vector<Function*> CompilationUnit::define(
    const std::optional<QualifiedName>& prefix,
    const std::string& source,
    const ResolverPtr& resolver,
    const Self* self) {
  // 创建一个解析器对象 p，用于解析传入的源代码
  Parser p(std::make_shared<Source>(source, "<string>", 1));
  // 存储解析得到的函数定义
  std::vector<Def> definitions;
  // 存储解析器对象
  std::vector<ResolverPtr> resolvers;
  // 循环解析源代码，直到遇到文件末尾
  while (p.lexer().cur().kind != TK_EOF) {
    // 解析函数定义，并将其添加到 definitions 向量中
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.push_back(def);
    // 将 resolver 添加到 resolvers 向量中
    resolvers.push_back(resolver);
  }
  // 调用另一个 define 方法，传递函数定义、解析器等参数，并返回结果
  return define(
      prefix,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      self);
}

// 静态函数，用于从图中删除空列表字面量节点
static void eraseListLiterals(std::shared_ptr<Graph>& graph) {
  // 创建一个深度优先图节点迭代器对象 it，用于遍历图中节点
  DepthFirstGraphNodeIterator it(graph);

  // 开始迭代图中的节点
  for (auto next_node = it.next(); next_node != nullptr;) {
    Node* node = next_node;
    next_node = it.next();

    // 如果节点类型是 prim::EmptyListLiteral
    if (node->kind() == prim::EmptyListLiteral) {
      // 如果节点有使用者
      if (node->hasUses()) {
        // 断言节点的输出类型是张量列表的子类型
        TORCH_INTERNAL_ASSERT(
            node->output()->type()->isSubtypeOf(ListType::ofTensors()));

        // 创建一个空列表节点 li，并插入到当前节点之前，替换所有使用当前节点的地方
        auto li = graph->createList(TensorType::get(), {});
        li->insertBefore(node);
        node->replaceAllUsesWith(li);
      }
      // 销毁当前节点
      node->destroy();
    }
  }
}

// 清理图中的不必要元素，例如插入但未使用的元组
void runCleanupPasses(std::shared_ptr<Graph>& to_clean) {
  // 提升闭包
  liftClosures(to_clean);
  // 内联分叉闭包
  inlineForkedClosures(to_clean);

  // 如果启用了内联模式，对整个图进行内联处理
  if (getInlineEverythingMode()) {
    Inline(*to_clean);
  }

  // 清理阶段的临时存在，移除空列表字面量
  eraseListLiterals(to_clean);

  // 移除插入的但未使用的元组
  LowerSimpleTuples(to_clean);

  // 完全常量传播，如果可以证明输入在图中任何地方未被修改，则运行操作
  ConstantPropagationImmutableTypes(to_clean);

  // 常量池化，必须在常量传播之后进行，以便池化新的常量
  ConstantPooling(to_clean);

  // 规范化输出，用于 jitter
  CanonicalizeOutputs(to_clean);

  // 对 aten::warns 进行注释，确保每个都有唯一的 ID，模仿 Python 的行为
  AnnotateWarns(to_clean);
}

// 判断给定名称是否为有意义的名称，根据特定规则进行判断
bool meaningfulName(const std::string& name) {
  // 如果名称为空，则返回 false
  if (name.empty())
    return false;
  // 如果名称以 '$' 开头，则返回 false
  if (name[0] == '$')
    return false;
  // 如果名称以 '_' 开头，但后面的字符都是数字，则返回 false
  if (name[0] != '_')
    return true;
  for (const auto i : c10::irange(1, name.size())) {
    // 如果名称中间有非数字字符，则返回 true
    if (!isdigit(name[i]))
      return true;
  }
  // 其他情况返回 false
  return false;
}
// 定义 CompilationUnit 类的 define_interface 方法，用于定义接口
void CompilationUnit::define_interface(
    const c10::QualifiedName& qualifiedName,  // 接口的完全限定名称
    const ClassDef& classDef,  // 类定义的结构
    ResolverPtr rcb,  // 解析器指针
    bool is_module) {  // 是否为模块接口

  // 创建 ScriptTypeParser 对象，并使用给定的解析器指针
  ScriptTypeParser typeParser(std::move(rcb));

  // 创建接口类型对象 iface，使用给定的限定名称和是否为模块接口标志
  InterfaceTypePtr iface =
      InterfaceType::create(c10::QualifiedName(qualifiedName), is_module);

  // 遍历类定义中的每个语句
  for (const Stmt& stmt : classDef.body()) {
    // 检查语句类型是否为方法定义
    if (stmt.kind() != TK_DEF) {
      throw ErrorReport(stmt)
          << "interface declarations can only contain method definitions";
    }

    // 解析方法定义
    auto method_def = Def(stmt);

    // 检查方法声明中是否有返回类型注解
    if (!method_def.decl().return_type().present()) {
      throw ErrorReport(method_def)
          << "interface declarations must have a return type annotated.";
    }

    // 解析方法的函数模式
    FunctionSchema schema =
        typeParser.parseSchemaFromDef(method_def, /* skip_self*/ true);

    // 准备方法参数，将 self 参数作为第一个参数添加，因为我们跳过了它
    std::vector<Argument> arguments;
    arguments.emplace_back(method_def.decl().params()[0].ident().name(), iface);
    arguments.insert(
        arguments.end(), schema.arguments().begin(), schema.arguments().end());

    // 向接口对象添加方法，使用克隆的方法模式和参数
    iface->addMethod(schema.cloneWithArguments(std::move(arguments)));

    // 检查方法体的语句，确保除了最后一个元素外，其余都是字符串字面量或者 "pass"
    auto stmts_size = method_def.statements().size();
    for (size_t i = 0; i < stmts_size - 1; i++) {
      auto cur_statement = method_def.statements()[i];
      if (cur_statement.kind() == TK_EXPR_STMT) {
        auto expr = ExprStmt(cur_statement).expr();
        // 如果表达式不是字符串字面量，则抛出错误
        if (expr.kind() != TK_STRINGLITERAL) {
          throw ErrorReport(method_def.range())
              << "interfaces declarations should only contain a single 'pass' statement.";
        }
      }
      // 如果遇到 "pass"，则停止检查
      if (cur_statement.kind() == TK_PASS) {
        this->register_type(iface);
        return;
      }
    }

    // 检查最后一个语句是否为 "pass"
    if (method_def.statements()[stmts_size - 1].kind() != TK_PASS) {
      throw ErrorReport(method_def.range())
          << "interfaces declarations should contain 'pass' statement.";
    }
  }

  // 注册接口类型
  this->register_type(iface);
}

} // namespace torch::jit
```