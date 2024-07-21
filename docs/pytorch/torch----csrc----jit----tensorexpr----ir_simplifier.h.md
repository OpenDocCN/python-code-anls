# `.\pytorch\torch\csrc\jit\tensorexpr\ir_simplifier.h`

```
#pragma once

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <utility>

/* IR Simplification
 *
 * Simplifies expressions in two stages:
 *  1. Recursively traverse the map combining similar operations into Terms
 * (interacted via Multiplication) and Polynomials (interacted via Addition). We
 * reorder the components of each Term or Polynomial into a consistent order to
 * allow combination or cancelling of like terms.
 *  2. Once the format of the tree is minimal, expand each Term into a sequence
 * of Muls, and each Polynomial into a sequence of Adds.
 */

namespace torch {
namespace jit {
namespace tensorexpr {

// A bunch of helpers for determine the Dtype of the output of a multi argument
// Term or Polynomial.
template <class ExprType>
Dtype promoteTypesVec(ExprPtr s, std::vector<ExprType>& v) {
  Dtype t = s->dtype(); // 获取输入表达式的数据类型
  bool first = true; // 设置初始值为 true

  // 遍历表达式列表 v 中的每个表达式 e
  for (const auto& e : v) {
    if (first) {
      t = Dtype(t.scalar_type(), e->dtype().lanes()); // 若是第一个表达式，则初始化 t
      first = false; // 标记已经不是第一个表达式了
    }
    t = promoteTypes(t, e->dtype()); // 调用 promoteTypes 函数推导表达式的数据类型
  }
  return t; // 返回推导出的数据类型
}

template <class ExprType>
Dtype promoteTypesVec(std::vector<ExprType>& v) {
  if (v.empty()) {
    throw malformed_input("empty list of types"); // 如果表达式列表为空，则抛出异常
  }

  Dtype t = v[0]->dtype(); // 获取第一个表达式的数据类型
  // 遍历表达式列表 v 中的每个表达式 e
  for (const auto& e : v) {
    t = promoteTypes(t, e->dtype()); // 调用 promoteTypes 函数推导表达式的数据类型
  }
  return t; // 返回推导出的数据类型
}

template <class ExprType>
Dtype promoteTypesMap(
    ExprPtr s,
    std::unordered_map<SimplifierHashType, ExprType>& m) {
  Dtype t = s->dtype(); // 获取输入表达式的数据类型
  bool first = true;
  // 遍历哈希表 m 中的每对键值对 (key, value)
  for (auto& e : m) {
    if (first) {
      t = Dtype(t.scalar_type(), e.second->dtype().lanes()); // 若是第一个表达式，则初始化 t
      first = false; // 标记已经不是第一个表达式了
    }
    t = promoteTypes(t, e.second->dtype()); // 调用 promoteTypes 函数推导表达式的数据类型
  }
  return t; // 返回推导出的数据类型
}

template <class ExprType>
Dtype promoteTypesVar(ExprType e) {
  return e->dtype(); // 返回表达式 e 的数据类型
}

template <class ExprType, class... Args>
Dtype promoteTypesVar(ExprType e, Args... es) {
  Dtype lhs = e->dtype(); // 获取第一个表达式的数据类型
  Dtype rhs = promoteTypesVar(es...); // 递归调用，获取其余表达式的数据类型
  if (e->isConstant()) {
    lhs = Dtype(lhs.scalar_type(), rhs.lanes()); // 如果第一个表达式是常量，则匹配 rhs 的 lanes
  }

  return promoteTypes(lhs, rhs); // 调用 promoteTypes 函数推导表达式的数据类型
}

// Uses the evaluator to fold an Expression with constant terms.
// E.g. evaluateOp(Add(3, 4)) => 7.
// Expr v must not have any unbound Vars.
inline ExprPtr evaluateOp(ExprPtr v) {
  ExprHandle handle(v); // 使用表达式 v 创建表达式句柄
  ExprEval<SimpleIREvaluator> eval(handle); // 使用 SimpleIREvaluator 对表达式进行评估

  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                 \
  case ScalarType::Name: {                                    \
    Type val = eval.value<Type>();                            \
    return getImmediateByType(v->dtype().scalar_type(), val); \
  }
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE); // 针对不同的标量类型进行处理
#undef TYPE_CASE
    default:
      // 如果程序运行到这里，表示遇到了不支持的数据类型
      LOG(FATAL) << "Unsupported datatype: " << v->dtype();
      // 记录错误信息后，返回空指针
      return nullptr;
  }
  // 默认情况下，返回空指针
  return nullptr;
}

// Term 类表示通过乘法组合的表达式组。
// 例如：product(scalar, *variables)。
class Term : public ExprNode<Term> {
 public:
  template <class... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Term 类的构造函数，接受一个哈希提供器和一系列表达式作为参数。
  Term(HashProvider& hasher, ExprPtr s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    // 检查标量表达式是否是常数。
    CHECK(s->isConstant());
    // 将所有传入的表达式作为组件添加到 Term 中，并进行排序。
    addComponent(ts...);
    sort();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Term 类的构造函数，接受一个哈希提供器、标量表达式和一个表达式向量作为参数。
  Term(HashProvider& hasher, ExprPtr s, std::vector<ExprPtr> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    // 对表达式向量进行排序。
    sort();
  }

  // Convenience constructor from a map of hash -> var, used when merging Terms.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Term 类的构造函数，接受一个哈希提供器、标量表达式和一个哈希映射作为参数。
  Term(
      HashProvider& hasher,
      ExprPtr s,
      std::unordered_map<SimplifierHashType, ExprPtr> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    // 将哈希映射中的表达式作为组件添加到 Term 中，并进行排序。
    for (auto& p : varmap) {
      addComponent(p.second);
    }
    sort();
  }

  // 返回 Term 的标量表达式。
  ExprPtr scalar() const {
    return scalar_;
  }
  // 返回 Term 的变量表达式向量。
  const std::vector<ExprPtr>& variables() const {
    return variables_;
  }
  // 返回 Term 的哈希提供器。
  HashProvider& hasher() const {
    return hasher_;
  }

  // 计算 Term 的变量部分的哈希值，用于确定是否可以与另一个 Term 合并。
  SimplifierHashType hashVars() const;

 private:
  std::vector<ExprPtr> variables_;
  ExprPtr scalar_;
  HashProvider& hasher_;

  // 添加空参数的私有成员函数。
  void addComponent() {}
  // 添加单个表达式参数的私有成员函数。
  void addComponent(ExprPtr e) {
    variables_.push_back(std::move(e));
  }
  // 添加多个表达式参数的私有成员函数。
  template <class... Es>
  void addComponent(ExprPtr e, Es&&... es) {
    addComponent(std::move(e));
    addComponent(std::forward<Es>(es)...);
  }

  // 按照哈希值对组件进行排序，以规范化组件的顺序。
  void sort();
};

// Polynomial 类表示通过加法组合的表达式组。
// 例如：sum(*variables, scalar)。
// 本应该称为 Expression，但是出现了命名冲突…
class Polynomial : public ExprNode<Polynomial> {
 public:
  template <class... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Polynomial 类的构造函数，接受一个哈希提供器和一系列表达式作为参数。
  Polynomial(HashProvider& hasher, ExprPtr s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    // 检查标量表达式是否是常数。
    CHECK(s->isConstant());
    // 将所有传入的表达式作为项添加到 Polynomial 中，并进行排序。
    addTerm(ts...);
    sort();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Polynomial 类的构造函数，接受一个哈希提供器、标量表达式和一个 Term 向量作为参数。
  Polynomial(HashProvider& hasher, ExprPtr s, std::vector<TermPtr> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    // 对 Term 向量进行排序。
    sort();
  }
  // 调用排序函数对多项式对象进行排序
  sort();
}

// 辅助构造函数，用于创建没有标量部分的项列表。
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
Polynomial(HashProvider& hasher, std::vector<TermPtr> terms)
    : ExprNodeBase(promoteTypesVec(terms)),
      variables_(std::move(terms)),
      scalar_(getImmediateByType(dtype(), 0)),
      hasher_(hasher) {
  // 调用排序函数对多项式对象进行排序
  sort();
}

// 方便的构造函数，用于哈希映射的哈希值到变量的映射，用于合并多项式。
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
Polynomial(
    HashProvider& hasher,
    ExprPtr s,
    std::unordered_map<SimplifierHashType, TermPtr> varmap)
    : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
  // 将哈希映射中的每个项添加到多项式对象中
  for (auto& p : varmap) {
    addTerm(p.second);
  }
  // 调用排序函数对多项式对象进行排序
  sort();
}

// 返回多项式对象的标量部分
ExprPtr scalar() const {
  return scalar_;
}

// 返回多项式对象的变量列表
const std::vector<TermPtr>& variables() const {
  return variables_;
}

// 返回多项式对象使用的哈希提供器
HashProvider& hasher() const {
  return hasher_;
}

// 计算多项式对象变量的哈希值
SimplifierHashType hashVars() const;

private:
std::vector<TermPtr> variables_; // 存储多项式对象的变量列表
ExprPtr scalar_; // 存储多项式对象的标量部分
HashProvider& hasher_; // 多项式对象使用的哈希提供器

// 添加一个项到多项式对象的变量列表中
void addTerm(TermPtr t) {
  variables_.push_back(std::move(t));
}

// 添加多个项到多项式对象的变量列表中
template <class... Ts>
void addTerm(TermPtr t, Ts&&... ts) {
  addTerm(std::move(t));
  addTerm(std::forward<Ts>(ts)...);
}

// 根据哈希值对多项式对象的变量列表进行排序，以规范化项的顺序
void sort();
};

// RoundOff 类的定义，继承自 BinaryOpNode<RoundOff>，表示一个四舍五入操作节点
class RoundOff : public BinaryOpNode<RoundOff> {
 public:
  // 构造函数，接受两个表达式指针作为参数
  RoundOff(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kOther) {}
};

// MaxTerm 类的定义，继承自 ExprNode<MaxTerm>
class MaxTerm : public ExprNode<MaxTerm> {
 public:
  // 可变参数模板构造函数，接受 HashProvider 引用、标量表达式指针、布尔值和多个表达式参数
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MaxTerm(HashProvider& hasher, ExprPtr s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...); // 将参数添加到 variables_ 向量中
    uniquefy(); // 唯一化处理
  }

  // 另一构造函数，接受 HashProvider 引用、标量表达式指针、布尔值和表达式向量作为参数
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MaxTerm(HashProvider& hasher, ExprPtr s, bool p, std::vector<ExprPtr> v)
      : ExprNodeBase(s ? promoteTypesVec(s, v) : promoteTypesVec(v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    uniquefy(); // 唯一化处理
  }

  // 返回 propagate_nans_ 成员变量的值
  bool propagate_nans() const {
    return propagate_nans_;
  }

  // 返回 scalar_ 成员变量的值
  ExprPtr scalar() const {
    return scalar_;
  }

  // 返回 variables_ 成员变量的常量引用
  const std::vector<ExprPtr>& variables() const {
    return variables_;
  }

  // 返回 hasher_ 成员变量的引用
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<ExprPtr> variables_; // 表达式指针的向量，存储多个表达式
  ExprPtr scalar_; // 标量表达式指针
  HashProvider& hasher_; // HashProvider 的引用
  bool propagate_nans_; // 布尔值，指示是否传播 NaN

  // 添加表达式到 variables_ 向量中的私有成员函数
  void addComponent() {}

  // 添加单个表达式到 variables_ 向量中的私有成员函数
  void addComponent(ExprPtr e) {
    variables_.push_back(std::move(e));
  }

  // 可变参数模板函数，添加多个表达式到 variables_ 向量中的私有成员函数
  template <class... Es>
  void addComponent(ExprPtr e, Es&&... es) {
    addComponent(std::move(e)); // 添加当前表达式
    addComponent(std::forward<Es>(es)...); // 递归添加剩余表达式
  }

  // 使用其哈希值对项进行唯一化处理的私有成员函数
  void uniquefy();
};

// MinTerm 类的定义，继承自 ExprNode<MinTerm>
class MinTerm : public ExprNode<MinTerm> {
 public:
  // 可变参数模板构造函数，接受 HashProvider 引用、标量表达式指针、布尔值和多个表达式参数
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MinTerm(HashProvider& hasher, ExprPtr s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...); // 将参数添加到 variables_ 向量中
    uniquefy(); // 唯一化处理
  }

  // 另一构造函数，接受 HashProvider 引用、标量表达式指针、布尔值和表达式向量作为参数
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MinTerm(HashProvider& hasher, ExprPtr s, bool p, std::vector<ExprPtr> v)
      : ExprNodeBase(s ? promoteTypesVec(s, v) : promoteTypesVec(v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    uniquefy(); // 唯一化处理
  }

  // 返回 propagate_nans_ 成员变量的值
  bool propagate_nans() const {
    return propagate_nans_;
  }

  // 返回 scalar_ 成员变量的值
  ExprPtr scalar() const {
    return scalar_;
  }

  // 返回 variables_ 成员变量的常量引用
  const std::vector<ExprPtr>& variables() const {
    return variables_;
  }

  // 返回 hasher_ 成员变量的引用
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<ExprPtr> variables_; // 表达式指针的向量，存储多个表达式
  ExprPtr scalar_; // 标量表达式指针
  HashProvider& hasher_; // HashProvider 的引用
  bool propagate_nans_; // 布尔值，指示是否传播 NaN

  // 添加表达式到 variables_ 向量中的私有成员函数
  void addComponent() {}

  // 添加单个表达式到 variables_ 向量中的私有成员函数
  void addComponent(ExprPtr e) {
    variables_.push_back(std::move(e));
  }

  // 可变参数模板函数，添加多个表达式到 variables_ 向量中的私有成员函数
  template <class... Es>
  void addComponent(ExprPtr e, Es&&... es) {
    addComponent(std::move(e)); // 添加当前表达式
    addComponent(std::forward<Es>(es)...); // 递归添加剩余表达式
  }

  // 使用其哈希值对项进行唯一化处理的私有成员函数
  void uniquefy();
};
// 使用 VarBoundInfo 类型定义变量到边界信息的映射，用于表示变量的边界信息
using VarBoundInfo = std::unordered_map<VarPtr, analysis::Bound>;

// SimplifierUnderContext 类继承自 IRMutator 类，用于上下文相关的IR简化
class TORCH_API SimplifierUnderContext : public IRMutator {
 public:
  ~SimplifierUnderContext() override = default;
  // 重写 mutate 方法以处理 For 循环节点的变换
  StmtPtr mutate(ForPtr v) override;

  // 重写 mutate 方法以处理除法表达式节点的变换
  ExprPtr mutate(DivPtr v) override;

  // 重写 mutate 方法以处理取模表达式节点的变换
  ExprPtr mutate(ModPtr v) override;

  // 重写 mutate 方法以处理比较选择表达式节点的变换
  ExprPtr mutate(CompareSelectPtr v) override;

  // 重写 mutate 方法以处理条件选择表达式节点的变换
  ExprPtr mutate(IfThenElsePtr v) override;

 protected:
  // 根据表达式获取循环边界信息
  bool getLoopBoundInfo(const ExprPtr& expr, analysis::Bound* loop_bound_info);

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  HashProvider hasher_; // 哈希提供器对象，用于计算哈希值
  VarBoundInfo var_bound_info_; // 存储变量边界信息的映射
};

// PolynomialBase 类继承自 IRMutator 类，用于多项式基类的操作
class TORCH_API PolynomialBase : public IRMutator {
 public:
  ~PolynomialBase() override = default;

  // 重写 mutate 方法以处理代码块节点的变换
  StmtPtr mutate(BlockPtr v) override;

  // 重写 mutate 方法以处理条件语句节点的变换
  StmtPtr mutate(CondPtr v) override;

  // 重写 mutate 方法以处理 For 循环节点的变换
  StmtPtr mutate(ForPtr v) override;

  // 对多项式进行因式分解，通过标量组件的最大公约数
  TermPtr factorizePolynomial(PolynomialPtr poly);

  // 返回哈希提供器的引用
  HashProvider& hasher() {
    return hasher_;
  }

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  HashProvider hasher_; // 哈希提供器对象，用于计算哈希值
};

// 简化 IR，通过合并共同项上的算术表达式
// 多项式转换器类，继承自 PolynomialBase 类
class TORCH_API PolynomialTransformer : public PolynomialBase {
 public:
  // 使用 using 关键字，继承自 PolynomialBase 的 mutate 方法
  using PolynomialBase::mutate;
  
  // 将术语插入到提供的映射中，处理哈希冲突时将术语与现有术语组合并更新映射
  void addOrUpdateTerm(
      std::unordered_map<SimplifierHashType, TermPtr>& varmap,
      TermPtr term);

  // 添加多项式表达式，合并表示相同变量的术语
  ExprPtr addPolynomials(PolynomialPtr lhs, PolynomialPtr rhs);

  // 向提供的多项式中插入新术语。如果新术语与现有术语具有公共变量，则将其合并
  ExprPtr insertTerm(PolynomialPtr poly, TermPtr term);

  // 合并和简化加法操作
  ExprPtr mutate(AddPtr v) override;

  // 减去一个术语从另一个术语中，必要时进行取消操作
  ExprPtr subTerms(TermPtr lhs, TermPtr rhs, bool negated);

  // 减去右侧多项式从左侧多项式中，尽可能取消相同的项
  ExprPtr subPolynomials(PolynomialPtr lhs, PolynomialPtr rhs);

  // 合并和简化减法操作
  ExprPtr mutate(SubPtr v) override;

  // 将两个术语相乘，通常创建一个新术语，其变量列表连接在一起
  TermPtr mulTerms(TermPtr lhs, TermPtr rhs);

  // 将多项式与术语相乘
  ExprPtr polyByTerm(PolynomialPtr poly, TermPtr term);

  // 匹配一个舍入模式并创建 RoundOff（四舍五入）（如果找到的话）
  ExprPtr isRoundOff(ExprPtr lhs, ExprPtr rhs);

  // 将新组件插入到术语中，可能进行简化
  ExprPtr insertIntoTerm(TermPtr term, ExprPtr expr);

  // 合并和简化乘法操作
  ExprPtr mutate(MulPtr v) override;

  // 继承自基类，重写除法操作的变异
  ExprPtr mutate(DivPtr v) override;

  // 继承自基类，重写取模操作的变异
  ExprPtr mutate(ModPtr v) override;

  // 继承自基类，重写按位与操作的变异
  ExprPtr mutate(AndPtr v) override;

  // 继承自基类，重写按位异或操作的变异
  ExprPtr mutate(XorPtr v) override;

  // 继承自基类，重写左移位操作的变异
  ExprPtr mutate(LshiftPtr v) override;

  // 继承自基类，重写右移位操作的变异
  ExprPtr mutate(RshiftPtr v) override;

  // 继承自基类，重写取最大值操作的变异
  ExprPtr mutate(MaxPtr v) override;

  // 继承自基类，重写取最小值操作的变异
  ExprPtr mutate(MinPtr v) override;

  // 继承自基类，重写比较选择操作的变异
  ExprPtr mutate(CompareSelectPtr v) override;

  // 继承自基类，重写内部函数操作的变异
  ExprPtr mutate(IntrinsicsPtr v) override;

  // 继承自基类，重写类型转换操作的变异
  ExprPtr mutate(CastPtr v) override;

  // 继承自基类，重写条件语句操作的变异
  ExprPtr mutate(IfThenElsePtr v) override;

  // 静态方法，简化表达式的计算结果，返回简化后的表达式
  static ExprPtr simplify(ExprPtr e);

  // 静态方法，简化表达式句柄的计算结果，返回简化后的表达式句柄
  static ExprHandle simplify(const ExprHandle& e);

  // 静态方法，简化语句的计算结果，返回简化后的语句
  static StmtPtr simplify(StmtPtr e);
};

// 术语扩展器类，继承自 PolynomialBase 类，将术语和多项式表达式展开为基本操作，进行一些简单的因式分解和重排序
class TORCH_API TermExpander : public PolynomialBase {
  PolynomialTransformer* simplifier_;
  std::set<VarPtr> eliminated_allocations_;

 public:
  // 使用 using 关键字，继承自 PolynomialBase 的 mutate 方法
  using PolynomialBase::mutate;
  
  // 构造函数，初始化术语转换器指针
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TermExpander(PolynomialTransformer* simplifier) : simplifier_(simplifier) {}
  
  // 检查安全性的方法
  bool check_safe() {
    // 返回一个布尔值，指示 eliminated_allocations_ 是否为空
    return eliminated_allocations_.empty();
  }

  // 将 Terms 展开为一系列的乘法表达式。
  ExprPtr mutate(TermPtr v) override;

  // 将 Polynomials 展开为一系列的加法表达式。
  ExprPtr mutate(PolynomialPtr v) override;

  // 将 MaxTerms 展开为一系列的最大值操作。
  ExprPtr mutate(MaxTermPtr v) override;

  // 将 MinTerms 展开为一系列的最小值操作。
  ExprPtr mutate(MinTermPtr v) override;

  // 将 RoundOff 展开为其组成部分：Mul(Div(lhs, rhs), rhs)。
  ExprPtr mutate(RoundOffPtr v) override;

  // 消除长度为零的分配操作。
  StmtPtr mutate(AllocatePtr v) override;
  StmtPtr mutate(FreePtr v) override;

  // 覆盖以启用条件融合。
  BlockPtr fuseConditions(BlockPtr v);

  // 融合同步线程的语句块。
  StmtPtr fuseSyncThreads(BlockPtr block);

  // 对 Block 进行变异，可能涉及条件融合和同步线程融合。
  StmtPtr mutate(BlockPtr v) override;
};

// 结束类定义的闭合括号

class TORCH_API IRSimplifier {
 public:
  // 声明静态方法，接受一个 StmtPtr 参数并返回一个 StmtPtr
  static StmtPtr simplify(StmtPtr s);
  // 声明静态方法，接受一个 ExprPtr 参数并返回一个 ExprPtr
  static ExprPtr simplify(ExprPtr e);
  // 声明静态方法，接受一个 ExprHandle 引用参数并返回一个 ExprHandle
  static ExprHandle simplify(const ExprHandle& e) {
    // 调用上面声明的 simplify 方法处理 ExprHandle 中的节点，并用处理后的节点构造一个新的 ExprHandle 返回
    return ExprHandle(simplify(e.node()));
  }
};

// 对缓冲区进行扁平化处理，并对扁平化后的维度执行简化器
ExprPtr buf_flat_size(BufPtr v);
// 如果表达式 A 和 B 可以简化为相同的表达式，则返回 true
TORCH_API bool exprEquals(ExprPtr A, ExprPtr B);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```