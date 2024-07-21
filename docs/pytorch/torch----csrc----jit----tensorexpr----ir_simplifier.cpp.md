# `.\pytorch\torch\csrc\jit\tensorexpr\ir_simplifier.cpp`

```
// 引入头文件，用于 JIT 日志、张量表达式边界重叠、IR 打印和简化器功能
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

// 引入 C++ 标准库中的实用工具
#include <utility>

// 定义命名空间 torch::jit::tensorexpr
namespace torch::jit::tensorexpr {

// 创建一个新的具有给定类型、左操作数和右操作数的表达式
inline ExprPtr newBinaryOpOfType(
    IRNodeType expr_type,
    ExprPtr lhs,
    ExprPtr rhs,
    bool option) {
  // 根据不同的表达式类型选择合适的操作符节点进行创建
  switch (expr_type) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case IRNodeType::kAdd:
      return alloc<Add>(lhs, rhs);  // 创建加法节点
    case IRNodeType::kSub:
      return alloc<Sub>(lhs, rhs);  // 创建减法节点
    case IRNodeType::kMul:
      return alloc<Mul>(lhs, rhs);  // 创建乘法节点
    case IRNodeType::kDiv:
      return alloc<Div>(lhs, rhs);  // 创建除法节点
    case IRNodeType::kMod:
      return alloc<Mod>(lhs, rhs);  // 创建取模节点
    case IRNodeType::kMax:
      return alloc<Max>(lhs, rhs, option);  // 创建最大值节点
    case IRNodeType::kMin:
      return alloc<Min>(lhs, rhs, option);  // 创建最小值节点
    case IRNodeType::kAnd:
      return alloc<And>(lhs, rhs);  // 创建按位与节点
    case IRNodeType::kXor:
      return alloc<Xor>(lhs, rhs);  // 创建按位异或节点
    case IRNodeType::kLshift:
      return alloc<Lshift>(lhs, rhs);  // 创建左移节点
    case IRNodeType::kRshift:
      return alloc<Rshift>(lhs, rhs);  // 创建右移节点
    default:
      LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);  // 记录并报告不支持的表达式类型
      return nullptr;
  }
}

// 修改二元操作节点的函数，使用 IRMutator 进行修改
template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static ExprPtr mutateBinaryOp(
    NodePtr<Op> v,
    IRMutator* mutator,
    bool option = false) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  
  // 对左右操作数分别应用 mutator 进行修改
  ExprPtr lhs_new = lhs->accept_mutator(mutator);
  ExprPtr rhs_new = rhs->accept_mutator(mutator);

  ExprPtr node = v;

  // 如果左右操作数有变化，则创建新的二元操作节点
  if (lhs != lhs_new || rhs != rhs_new) {
    node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
  }

  // 只有当左右操作数都是常数时才能进行折叠
  if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
    return node;
  }

  // 对节点进行求值
  return evaluateOp(node);
}

// 简单的递归求最大公约数函数模板
template <typename T>
T gcd(T a, T b) {
  if (b == 0) {
    return a;
  }
  return gcd(b, a % b);
}

// 判断表达式是否是多条数据的基本元素（如广播或者斜坡）
static bool isMultilanePrimitive(ExprPtr e) {
  return to<Broadcast>(e) || to<Ramp>(e);
}

// Term 类的 hashVars 函数实现
SimplifierHashType Term::hashVars() const {
  SimplifierHashType hash;
  
  // 遍历变量列表并计算哈希值
  for (const auto& v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }

  return hash;
}

// Term 类的 sort 函数实现
void Term::sort() {
  // 浮点数操作时需要保持操作顺序的重要性
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");  // 报错，不支持重新排序浮点数操作
  }
  
  // 用于存储表达式到字符串表示的缓存
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  
  // 使用 Lambda 函数对 variables_ 中的表达式进行排序
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);  // 将表达式转换为字符串并存储在缓存中
    }
    // 按字符串表示进行比较排序
    return str_repr_cache[a] < str_repr_cache[b];
  });
}
    // 如果缓存中不存在字符 b 的字符串表示，将其添加到缓存中
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    // 比较字符 a 和字符 b 的字符串表示在缓存中的顺序
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

// 计算多项式对象的哈希值，基于其中的变量
SimplifierHashType Polynomial::hashVars() const {
  SimplifierHashType hash;
  // 遍历多项式对象的变量集合，使用哈希函数计算哈希值
  for (const auto& v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }
  return hash; // 返回计算得到的哈希值
}

// 对多项式对象的变量进行排序操作
void Polynomial::sort() {
  // 如果数据类型是浮点数，则抛出逻辑错误异常
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  // 用于存储表达式指针及其字符串表示的缓存
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  // 使用 Lambda 表达式对变量集合进行排序，排序规则是按照字符串表示的字典序
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

// 对最大项对象的变量进行去重操作
void MaxTerm::uniquefy() {
  // 使用 Lambda 表达式对变量集合进行排序，排序规则是根据哈希值
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    return hasher_.hash(a) < hasher_.hash(b);
  });
  // 使用 std::unique 去除重复的变量
  auto it = std::unique(
      variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));

  // 去除重复后，再次对变量集合进行排序，以保证稳定性
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

// 对最小项对象的变量进行去重操作
void MinTerm::uniquefy() {
  // 使用 Lambda 表达式对变量集合进行排序，排序规则是根据哈希值
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    return hasher_.hash(a) < hasher_.hash(b);
  });
  // 使用 std::unique 去除重复的变量
  auto it = std::unique(
      variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));

  // 去除重复后，再次对变量集合进行排序，以保证稳定性
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

// 处理广播/斜坡组合的优化情况
template <class Op>
ExprPtr combineMultilane(ExprPtr lhs, ExprPtr rhs) {
  if (BroadcastPtr bc = to<Broadcast>(lhs)) {
    if (BroadcastPtr bcother = to<Broadcast>(rhs)) {
      // 如果两个广播对象的通道数不匹配，则抛出输入异常
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      // 合并两个广播对象，生成新的广播对象
      ExprPtr ret = alloc<Broadcast>(
          alloc<Op>(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }
    # 如果 rhs 能够转换为 RampPtr 类型，并将结果存储在 r 中
    if (RampPtr r = to<Ramp>(rhs)) {
      # 检查当前对象 bc 的 lanes 数量是否与 r 的 lanes 数量相同
      if (bc->lanes() != r->lanes()) {
        # 抛出异常，指示多通道 lane 不匹配的错误
        throw malformed_input("multilane lane mismatch");
      }

      # 分配一个新的 Ramp 对象 ret，其 base 值为 bc->value() 和 r->base() 的运算结果，
      # stride 为 r->stride()，lanes 为 r->lanes()
      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(bc->value(), r->base()), r->stride(), r->lanes());
      # 返回构造的新对象 ret
      return ret;
    }
  } else if (RampPtr ramp = to<Ramp>(lhs)) {
    # 如果 lhs 能够转换为 RampPtr 类型，并将结果存储在 ramp 中
    if (RampPtr rother = to<Ramp>(rhs)) {
      # 检查 ramp 和 rother 的 lanes 数量是否相同
      if (ramp->lanes() != rother->lanes()) {
        # 抛出异常，指示多通道 lane 不匹配的错误
        throw malformed_input("multilane lane mismatch");
      }

      # 分配一个新的 Ramp 对象 ret，其 base 值为 ramp->base() 和 rother->base() 的运算结果，
      # stride 为 ramp->stride() 和 rother->stride() 的运算结果，lanes 为 ramp->lanes()
      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(ramp->base(), rother->base()),
          alloc<Op>(ramp->stride(), rother->stride()),
          ramp->lanes());
      # 返回构造的新对象 ret
      return ret;
    }

    # 如果 rhs 能够转换为 BroadcastPtr 类型，并将结果存储在 bc 中
    if (BroadcastPtr bc = to<Broadcast>(rhs)) {
      # 检查 ramp 和 bc 的 lanes 数量是否相同
      if (ramp->lanes() != bc->lanes()) {
        # 抛出异常，指示多通道 lane 不匹配的错误
        throw malformed_input("multilane lane mismatch");
      }
      # 分配一个新的 Ramp 对象 ret，其 base 值为 ramp->base() 和 bc->value() 的运算结果，
      # stride 为 ramp->stride()，lanes 为 ramp->lanes()
      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(ramp->base(), bc->value()), ramp->stride(), ramp->lanes());
      # 返回构造的新对象 ret
      return ret;
    }
  }

  # 如果以上所有条件都不满足，则返回空指针 nullptr
  return nullptr;
}

// 处理广播/斜坡 * 广播/斜坡的优化情况
static ExprPtr mulMultilane(ExprPtr lhs, ExprPtr rhs) {
  // 如果 lhs 是 Broadcast 类型
  if (BroadcastPtr bc = to<Broadcast>(lhs)) {
    // 如果 rhs 也是 Broadcast 类型
    if (BroadcastPtr bcother = to<Broadcast>(rhs)) {
      // 检查广播的通道数是否匹配
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      // 创建一个新的 Broadcast 表达式，值为 lhs 值与 rhs 值的乘积，通道数与 bc 相同
      ExprPtr ret = alloc<Broadcast>(
          alloc<Mul>(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    // 如果 rhs 是 Ramp 类型
    if (RampPtr r = to<Ramp>(rhs)) {
      // 检查广播的通道数与斜坡的通道数是否匹配
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      // 创建一个新的 Ramp 表达式，基数为 lhs 值与 rhs 基数的乘积，步长为 lhs 值与 rhs 步长的乘积，通道数与 r 相同
      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(bc->value(), r->base()),
          alloc<Mul>(bc->value(), r->stride()),
          r->lanes());
      return ret;
    }
  } else if (RampPtr ramp = to<Ramp>(lhs)) {  // 如果 lhs 是 Ramp 类型
    // 如果 rhs 也是 Ramp 类型
    if (RampPtr r = to<Ramp>(rhs)) {
      // 检查斜坡的通道数是否匹配
      if (ramp->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      // 创建一个新的 Ramp 表达式，基数为 lhs 基数与 rhs 基数的乘积，步长为 lhs 步长与 rhs 步长的乘积，通道数与 r 相同
      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(ramp->base(), r->base()),
          alloc<Mul>(ramp->stride(), r->stride()),
          r->lanes());
      return ret;
    }

    // 如果 rhs 是 Broadcast 类型
    if (BroadcastPtr bc = to<Broadcast>(rhs)) {
      // 检查斜坡的通道数与广播的通道数是否匹配
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      // 创建一个新的 Ramp 表达式，基数为 lhs 基数与 rhs 值的乘积，步长为 lhs 步长与 rhs 值的乘积，通道数与 ramp 相同
      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(bc->value(), ramp->base()),
          alloc<Mul>(bc->value(), ramp->stride()),
          ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

void PolynomialTransformer::addOrUpdateTerm(
    std::unordered_map<SimplifierHashType, TermPtr>& varmap,
    TermPtr term) {
  // 计算术语的哈希值
  SimplifierHashType hash = term->hashVars();
  // 尝试将术语插入到变量映射中
  auto insertRes = varmap.emplace(hash, term);
  // 如果插入失败，说明映射中已存在相同变量的术语
  if (insertRes.second == false) {
    // 获取映射中已存在的术语
    TermPtr lt = insertRes.first->second;
    // 计算两个术语标量部分的和
    ExprPtr termScalar = evaluateOp(alloc<Add>(lt->scalar(), term->scalar()));

    // 如果术语被抵消，从映射中删除它
    if (immediateEquals(termScalar, 0)) {
      varmap.erase(hash);
      return;
    }

    // 更新映射中的术语为新的术语，保留相同的哈希值和变量
    varmap[hash] = alloc<Term>(hasher_, termScalar, lt->variables());
  }
}

ExprPtr PolynomialTransformer::addPolynomials(
    PolynomialPtr lhs,
    PolynomialPtr rhs) {
  // 简化公共部分
  // 这里的关键是变量的哈希值，而不是术语的哈希值，因为我们希望结合具有相同变量但不同标量组件的术语
  std::unordered_map<SimplifierHashType, TermPtr> varmap;

  // 处理左多项式的变量
  for (const auto& lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }
  // 处理右多项式的变量
  for (const auto& rt : rhs->variables()) {
    addOrUpdateTerm(varmap, rt);
  }

  // 计算新的标量部分
  ExprPtr newScalar = evaluateOp(alloc<Add>(lhs->scalar(), rhs->scalar()));
  // 创建新的多项式对象，使用哈希器和变量映射
  return alloc<Polynomial>(hasher_, newScalar, varmap);
}

// 将新术语插入到提供的多项式中。如果新术语与现有术语有共同变量，则进行合并。
ExprPtr PolynomialTransformer::insertTerm(PolynomialPtr poly, TermPtr term) {
  // 计算插入项的变量哈希值
  SimplifierHashType tHash = term->hashVars();
  // 存储新的变量项
  std::vector<TermPtr> newVars;

  bool found = false;
  // 遍历多项式中的变量项
  for (const auto& v : poly->variables()) {
    // 检查变量项的哈希值是否与插入项相同
    if (v->hashVars() == tHash) {
      // 对标量进行求值运算，尝试消除相同变量项
      ExprPtr newScalar = evaluateOp(alloc<Add>(term->scalar(), v->scalar()));
      found = true;
      // 如果运算结果为零，则跳过这个项
      if (immediateEquals(newScalar, 0)) {
        continue;
      }
      // 创建一个新的项，并添加到新变量列表中
      auto term = alloc<Term>(hasher_, newScalar, v->variables());
      newVars.push_back(term);
    } else {
      // 如果变量项的哈希值不同，直接添加到新变量列表中
      newVars.push_back(v);
    }
  }

  // 如果未找到相同的变量项，将插入项添加到新变量列表中
  if (!found) {
    newVars.push_back(term);
  }

  // 如果新变量列表为空，返回多项式的标量部分
  if (newVars.empty()) {
    return poly->scalar();
  }

  // 创建一个新的多项式对象，并返回
  auto Poly = alloc<Polynomial>(hasher_, poly->scalar(), newVars);
  return Poly;
}

ExprPtr PolynomialTransformer::mutate(AddPtr v) {
  // 对左右操作数分别应用变换器
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // 常数折叠
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    // 对两个常数进行求和运算，并返回结果
    ExprPtr result = evaluateOp(alloc<Add>(lhs_new, rhs_new));
    return result;
  }

  // 多行折叠
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = combineMultilane<Add>(lhs_new, rhs_new)) {
      // 如果可以合并多行，则返回合并后的结果
      return ret->accept_mutator(this);
    }
  }

  // 如果左操作数为常数
  ExprPtr scalar = nullptr;
  ExprPtr variable = nullptr;
  if (lhs_new->isConstant()) {
    scalar = evaluateOp(lhs_new);
    variable = rhs_new;
  } else if (rhs_new->isConstant()) {
    scalar = evaluateOp(rhs_new);
    variable = lhs_new;
  }

  // 如果存在标量并且为零，则短路并返回另一侧的变量
  if (scalar && immediateEquals(scalar, 0)) {
    auto c = alloc<Cast>(v->dtype(), variable);
    return c->accept_mutator(this);
  }

  // 如果这是浮点数加法，返回新的加法表达式
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Add>(lhs_new, rhs_new);
  }

  // 尝试将左右操作数转换为多项式对象
  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  // 如果左右操作数都是多项式对象，则将它们相加
  if (lhsPoly && rhsPoly) {
    return addPolynomials(lhsPoly, rhsPoly);
  }

  // 尝试将左操作数转换为多项式，右操作数转换为项，并插入项到多项式中
  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  if (lhsPoly && rhsTerm) {
    return insertTerm(lhsPoly, rhsTerm);
  }

  if (rhsPoly && lhsTerm) {
    return insertTerm(rhsPoly, lhsTerm);
  }

  // 如果左右操作数都是项，并且引用相同的变量，则合并它们
  if (lhsTerm && rhsTerm) {
    if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
      // 对标量进行求和运算
      ExprPtr newScalar =
          evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar()));

      // 如果项相互抵消，则返回零
      if (immediateEquals(newScalar, 0)) {
        return newScalar->accept_mutator(this);
      }

      // 创建一个新的项并返回
      return alloc<Term>(hasher_, newScalar, lhsTerm->variables());
    }

    // 否则，创建一个新的多项式，其中包含这两个项
    // （不包含标量，因为它们是零）
    return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);


// 使用给定的哈希器、常量零值和左右两个项创建一个新的多项式对象并返回
return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);



  }

  // Adds are commutative.
  PolynomialPtr poly = lhsPoly ? lhsPoly : rhsPoly;


// 多项式的加法是交换律的
PolynomialPtr poly = lhsPoly ? lhsPoly : rhsPoly;
// 如果左操作数的多项式存在则选用它，否则选用右操作数的多项式



  // Add to Polynomial->scalar().
  if (scalar && poly) {
    ExprPtr newScalar = evaluateOp(alloc<Add>(scalar, poly->scalar()));
    return alloc<Polynomial>(hasher_, newScalar, poly->variables());
  }


// 将标量添加到多项式的标量部分
if (scalar && poly) {
    // 计算新的标量值并创建一个新的多项式对象
    ExprPtr newScalar = evaluateOp(alloc<Add>(scalar, poly->scalar()));
    return alloc<Polynomial>(hasher_, newScalar, poly->variables());
}



  // Simple Polynomial with a scalar and Term.
  TermPtr term = lhsTerm ? lhsTerm : rhsTerm;
  if (scalar && term) {
    return alloc<Polynomial>(hasher_, scalar, term);
  }


// 包含标量和项的简单多项式
TermPtr term = lhsTerm ? lhsTerm : rhsTerm;
if (scalar && term) {
    // 创建一个新的多项式对象
    return alloc<Polynomial>(hasher_, scalar, term);
}



  // Simple Term with a scalar and variable type.
  if (scalar) {
    return alloc<Polynomial>(
        hasher_, scalar, alloc<Term>(hasher_, immLike(v, 1), variable));
  }


// 包含标量和变量类型的简单项
if (scalar) {
    // 创建一个只包含一个项的新多项式对象
    return alloc<Polynomial>(
        hasher_, scalar, alloc<Term>(hasher_, immLike(v, 1), variable));
}



  // If LHS is neither Term not Polynomial, wrap it in a Term.
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
  }


// 如果左操作数既不是项也不是多项式，则将其包装在一个项中
if (!lhsTerm && !lhsPoly) {
    lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
}



  // Same for RHS.
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = alloc<Term>(hasher_, immLike(v, 1), rhs_new);
  }


// 对于右操作数也是一样的处理方式
if (!rhsTerm && !rhsPoly) {
    rhsTerm = alloc<Term>(hasher_, immLike(v, 1), rhs_new);
}



  // If we now have a poly and a term, we can insert.
  if (poly) {
    return insertTerm(poly, lhsTerm ? lhsTerm : rhsTerm);
  }


// 如果现在有多项式和项，则可以进行插入操作
if (poly) {
    // 插入项到多项式中并返回新的多项式对象
    return insertTerm(poly, lhsTerm ? lhsTerm : rhsTerm);
}



  if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
    return alloc<Term>(
        hasher_,
        evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar())),
        lhsTerm->variables());
  }


// 如果左项和右项的变量哈希相同，则创建一个新的项对象
if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
    return alloc<Term>(
        hasher_,
        evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar())),
        lhsTerm->variables());
}



  // If all else fails we have a new Polynomial with two new variable Terms.
  return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);


// 如果所有条件都不符合，则创建一个新的多项式对象，包含常量零值和两个新的变量项
return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
}

ExprPtr PolynomialTransformer::subTerms(
    TermPtr lhs,
    TermPtr rhs,
    bool negated) {
  // 如果 RHS 没有被否定，则对其进行否定操作。
  if (!negated) {
    // 创建一个值为 -1 的立即表达式 minusOne
    ExprPtr minusOne = immLike(rhs, -1);
    // 计算 rhs 乘以 -1 的结果，得到 negateScalar
    ExprPtr negateScalar = evaluateOp(alloc<Mul>(minusOne, rhs->scalar()));
    // 将 rhs 的符号取反，并生成一个新的 Term 对象
    rhs = alloc<Term>(hasher_, negateScalar, rhs->variables());
  }

  // 如果 lhs 和 rhs 的变量哈希值相等
  if (lhs->hashVars() == rhs->hashVars()) {
    // 计算新的标量 newScalar，为 lhs 的标量加上 rhs 的标量
    ExprPtr newScalar = evaluateOp(alloc<Add>(lhs->scalar(), rhs->scalar()));

    // 如果项相互抵消，返回零表达式
    if (immediateEquals(newScalar, 0)) {
      return newScalar;
    }

    // 返回一个新的 Term 对象，包含新的标量和 lhs 的变量
    return alloc<Term>(hasher_, newScalar, lhs->variables());
  }

  // 返回一个新的 Polynomial 对象，包含指定类型的零值、lhs 和 rhs
  return alloc<Polynomial>(
      hasher_,
      getImmediateByType(promoteTypes(lhs->dtype(), rhs->dtype()), 0),
      lhs,
      rhs);
}

// Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
// possible.
ExprPtr PolynomialTransformer::subPolynomials(
    PolynomialPtr lhs,
    PolynomialPtr rhs) {
  // 简化公共组件
  // 关键在于变量的哈希值，而不是项的哈希值，因为我们希望结合具有相同变量但不同标量部分的项。
  std::unordered_map<SimplifierHashType, TermPtr> varmap;

  // 遍历 lhs 的变量列表
  for (const auto& lt : lhs->variables()) {
    // 向 varmap 中添加或更新项
    addOrUpdateTerm(varmap, lt);
  }

  // 遍历 rhs 的变量列表
  for (const auto& rt : rhs->variables()) {
    // 多项式将其项相加，因此对 rhs 的项进行取反操作
    ExprPtr negated = evaluateOp(alloc<Mul>(immLike(rt, -1), rt->scalar()));
    // 创建一个新的 Term 对象，将取反后的结果加入 varmap
    TermPtr newRHS = alloc<Term>(hasher_, negated, rt->variables());
    addOrUpdateTerm(varmap, newRHS);
  }

  // 计算新的标量 newScalar，为 lhs 的标量减去 rhs 的标量
  ExprPtr newScalar = evaluateOp(alloc<Sub>(lhs->scalar(), rhs->scalar()));

  // 如果 varmap 为空，表示所有变量项相互抵消，直接返回新的标量
  if (varmap.empty()) {
    return newScalar;
  }

  // 如果 newScalar 为零且变量项只有零个或一个，则不包装它
  if (immediateEquals(newScalar, 0)) {
    if (varmap.empty()) {
      return nullptr;
    }
    if (varmap.size() == 1) {
      return varmap.begin()->second;
    }
  }

  // 将新的变量项包装在一个 Polynomial 对象中，并返回
  return alloc<Polynomial>(hasher_, newScalar, varmap);
}

ExprPtr PolynomialTransformer::mutate(SubPtr v) {
  // 对 lhs 和 rhs 分别使用当前的变换器进行变异
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // 常量折叠
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    // 执行减法操作，并返回结果
    ExprPtr result = evaluateOp(alloc<Sub>(lhs_new, rhs_new));
    return result;
  }

  // 多行折叠
  if (isMultilanePrimitive(lhs_new)) {
    // 如果可以结合多行操作，则返回变换器对其进行变异后的结果
    if (auto ret = combineMultilane<Sub>(lhs_new, rhs_new)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return ret->accept_mutator(this);
    }
  }

  // 如果 rhs_new 是常量且其值为零
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 0)) {
    // 创建一个 Cast 表达式，并返回结果
    auto c = alloc<Cast>(v->dtype(), lhs_new);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c->accept_mutator(this);
  }



  // 如果这是一个浮点数的减法操作，操作顺序非常重要，我们不希望合并操作。
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    // 返回一个新的 Sub 对象，将左右表达式作为参数
    return alloc<Sub>(lhs_new, rhs_new);
  }



  // 将左右表达式分别转换为多项式类型
  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  // 如果左右表达式均为多项式
  if (lhsPoly && rhsPoly) {
    // 执行多项式的减法操作
    auto ret = subPolynomials(lhsPoly, rhsPoly);
    if (!ret) {
      // 如果完全抵消，则返回一个与 v 类型相同且值为 0 的立即数表达式
      return immLike(v, 0);
    }
    // 返回多项式减法的结果
    return ret;
  }



  // 将左右表达式分别转换为项类型
  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  // 如果左边为多项式，右边为项
  if (lhsPoly && rhsTerm) {
    // 对项取负
    ExprPtr negate =
        evaluateOp(alloc<Mul>(immLike(rhsTerm, -1), rhsTerm->scalar()));
    // 创建一个新的项，对其进行插入操作
    TermPtr newTerm = alloc<Term>(hasher_, negate, rhsTerm->variables());
    return insertTerm(lhsPoly, newTerm);
  }



  // 如果右边为多项式，左边为项
  if (rhsPoly && lhsTerm) {
    // 对多项式的每一部分取负
    ExprPtr minusOne = immLike(lhsTerm, -1);
    ExprPtr negateScalar = evaluateOp(alloc<Mul>(minusOne, rhsPoly->scalar()));

    // 对多项式的每个项取负
    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    // 创建一个新的多项式并插入项
    PolynomialPtr newPoly = alloc<Polynomial>(hasher_, negateScalar, variables);
    return insertTerm(newPoly, lhsTerm);
  }



  // 如果左右表达式均为项
  if (lhsTerm && rhsTerm) {
    // 执行项之间的减法操作
    return subTerms(lhsTerm, rhsTerm, false);
  }



  // 判断左右表达式是否为常数
  bool lhsScalar = lhs_new->isConstant();
  bool rhsScalar = rhs_new->isConstant();

  // 如果左边为多项式，右边为常数
  if (lhsPoly && rhsScalar) {
    // 简单路径，直接对多项式的常数部分进行减法操作
    ExprPtr newScalar = evaluateOp(alloc<Sub>(lhsPoly->scalar(), rhs_new));
    return alloc<Polynomial>(hasher_, newScalar, lhsPoly->variables());
  }



  // 如果左边为常数，右边为多项式
  if (lhsScalar && rhsPoly) {
    // 对多项式的常数部分进行减法操作
    ExprPtr newScalar = evaluateOp(alloc<Sub>(lhs_new, rhsPoly->scalar()));

    // 对多项式的每个项取负
    ExprPtr minusOne = immLike(rhsPoly, -1);
    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    // 创建一个新的多项式并插入项
    return alloc<Polynomial>(hasher_, newScalar, variables);
  }



  // 如果左边为项，右边为常数
  if (lhsTerm && rhsScalar) {
    // 对常数取负
    ExprPtr negate = evaluateOp(alloc<Mul>(immLike(rhs_new, -1), rhs_new));
    // 创建一个新的多项式，仅包含项
    return alloc<Polynomial>(hasher_, negate, lhsTerm);
  }



  // 如果左边为常数，右边为项
  if (lhsScalar && rhsTerm) {
    // 对右边的项取负
    ExprPtr negate = evaluateOp(
        alloc<Mul>(immLike(rhsTerm->scalar(), -1), rhsTerm->scalar()));

    // 创建一个新的多项式，包含常数和负的项
    return alloc<Polynomial>(
        hasher_, lhs_new, alloc<Term>(hasher_, negate, rhsTerm->variables()));
  }



  // 简单的情况，左边为常数且右边为变量类型
  if (lhsScalar) {
  // 创建一个被取反的项。
  return alloc<Polynomial>(
      hasher_, lhs_new, alloc<Term>(hasher_, immLike(v, -1), rhs_new));
}

if (rhsScalar) {
  // 取反标量。
  ExprPtr negate = evaluateOp(alloc<Mul>(immLike(rhs_new, -1), rhs_new));
  return alloc<Polynomial>(
      hasher_, negate, alloc<Term>(hasher_, immLike(v, 1), lhs_new));
}

// 没有标量...
if (!lhsTerm && !lhsPoly) {
  // 如果左侧项和多项式均不存在，则创建左侧项。
  lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
}

bool createdRHSnegated = false;
if (!rhsTerm && !rhsPoly) {
  // 如果右侧项和多项式均不存在，则创建右侧取反的项。
  rhsTerm = alloc<Term>(hasher_, immLike(v, -1), rhs_new);
  createdRHSnegated = true;
}

if (lhsTerm && rhsTerm) {
  // 如果左右两侧均存在项，则执行项的减法操作。
  return subTerms(lhsTerm, rhsTerm, createdRHSnegated);
}

// 将包装后的项插入左侧多项式。
if (lhsPoly) {
  CHECK(rhsTerm);
  return insertTerm(lhsPoly, rhsTerm);
}

// 将包装后的项插入取反后的右侧多项式。
if (rhsPoly) {
  CHECK(lhsTerm);
  ExprPtr minusOne = immLike(rhsPoly, -1);
  ExprPtr newScalar = evaluateOp(alloc<Mul>(minusOne, rhsPoly->scalar()));

  // 取反右侧多项式中的每一项。
  std::vector<TermPtr> variables;
  for (const auto& t : rhsPoly->variables()) {
    ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
    variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
  }

  auto poly = alloc<Polynomial>(hasher_, newScalar, variables);
  return insertTerm(poly, lhsTerm);
}

// 返回一个多项式，结果是对左右项进行加法或减法操作的结果。
return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
// 多项式转换器类的方法：将两个项相乘，通常会创建一个新的项，其中变量列表被连接起来。
TermPtr PolynomialTransformer::mulTerms(TermPtr lhs, TermPtr rhs) {
  // 使用评估操作处理左右项的标量乘积
  ExprPtr scalar = evaluateOp(alloc<Mul>(lhs->scalar(), rhs->scalar()));
  // 如果标量乘积为0，则返回空指针
  if (immediateEquals(scalar, 0)) {
    return nullptr;
  }

  // 可以重新排序变量，因为浮点运算不会被放入项中。
  std::vector<ExprPtr> variables;
  std::vector<ExprPtr> multilaneVariables;
  // 目前不处理指数。
  for (const auto& c : lhs->variables()) {
    // 如果变量是多行原始值，则放入多行变量向量中；否则放入变量向量中。
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }
  for (const auto& c : rhs->variables()) {
    // 如果变量是多行原始值，则放入多行变量向量中；否则放入变量向量中。
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }

  // 合并所有多行变量：
  ExprPtr lastNode{nullptr};
  for (const auto& node : multilaneVariables) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      // 如果可以将上一个节点和当前节点相乘，得到新节点，则使用变异器进行处理；
      // 否则将上一个节点添加到变量向量中，并更新为当前节点。
      if (auto next = mulMultilane(lastNode, node)) {
        lastNode = next->accept_mutator(this);
      } else {
        variables.push_back(lastNode);
        lastNode = node;
      }
    }
  }
  // 如果存在最后一个节点，则将其添加到变量向量中。
  if (lastNode) {
    variables.push_back(lastNode);
  }

  // 返回新的项对象，包含哈希、标量和变量向量。
  return alloc<Term>(hasher_, scalar, variables);
}

// 多项式转换器类的方法：将多项式与项相乘。
ExprPtr PolynomialTransformer::polyByTerm(PolynomialPtr poly, TermPtr term) {
  // poly * term
  //    = (poly_terms + poly_scalar) * term
  //    = poly_terms * term + poly_scalar * term

  // 首先，将多项式中的所有项与输入项相乘。
  std::vector<TermPtr> newTerms;
  for (const auto& var : poly->variables()) {
    // 调用mulTerms方法，将多项式中的每个项与输入项相乘。
    TermPtr newTerm = mulTerms(var, term);
    // 如果乘积项不为空，则添加到新项向量中。
    if (newTerm) {
      newTerms.push_back(newTerm);
    }
  }

  // 如果多项式中的标量不为0，则将其与输入项的标量相乘。
  // 如果输入项中没有变量，则该乘积成为结果多项式的标量。
  // 如果输入项中有变量，则创建一个新的项并加入结果多项式。
  if (!immediateEquals(poly->scalar(), 0)) {
    ExprPtr scalar = evaluateOp(alloc<Mul>(poly->scalar(), term->scalar()));
    if (term->variables().empty()) {
      return alloc<Polynomial>(hasher_, scalar, newTerms);
    }
    newTerms.push_back(alloc<Term>(hasher_, scalar, term->variables()));
  }

  // 结果多项式中只有在输入项没有任何变量且输入多项式有非零标量时才有标量。
  // 上述情况在上面已处理，所以此处结果多项式不包含标量。
  return alloc<Polynomial>(hasher_, std::move(newTerms));
}

// 多项式转换器类的方法：判断这两个表达式相乘是否会产生舍入操作。
// 例如：LHS = (x/y),  RHS = y => (x / y) * y => RoundOff(x, y).
ExprPtr PolynomialTransformer::isRoundOff(ExprPtr lhs, ExprPtr rhs) {
  DivPtr div{nullptr};
  ExprPtr other{nullptr};

  // 如果LHS是除法表达式，则将其转换为Div指针。
  if ((div = to<Div>(lhs))) {
    other = rhs;
  } else if ((div = to<Div>(rhs))) {
    other = lhs;
  } else {
    return nullptr;
  }


    // 如果 rhs 是一个 Term 对象，则将其赋值给 other
    other = rhs;
  } else if ((div = to<Div>(rhs))) {
    // 如果 rhs 是一个 Div 对象，则将其赋值给 div，并将 lhs 赋值给 other
    other = lhs;
  } else {
    // 如果 rhs 不是 Term 也不是 Div 对象，则返回空指针
    return nullptr;
  }



  ExprPtr denom = div->rhs();

  if (TermPtr denomTerm = to<Term>(denom)) {
    if (immediateEquals(denomTerm->scalar(), 1) &&
        denomTerm->variables().size() == 1) {
      denom = denomTerm->variables()[0];
    }
  }


  // 获取 div 的右操作数，并赋值给 denom
  ExprPtr denom = div->rhs();

  // 如果 denom 是 Term 对象
  if (TermPtr denomTerm = to<Term>(denom)) {
    // 如果 denomTerm 的标量为 1，并且变量数量为 1
    if (immediateEquals(denomTerm->scalar(), 1) &&
        denomTerm->variables().size() == 1) {
      // 将 denom 替换为 denomTerm 的唯一变量
      denom = denomTerm->variables()[0];
    }
  }



  if (hasher_.hash(denom) == hasher_.hash(other)) {
    // 如果 denom 和 other 的哈希值相同，则返回一个 RoundOff 对象
    // 使用 div 的左操作数和右操作数作为参数
    return alloc<RoundOff>(div->lhs(), div->rhs());
  }


  // 如果 denom 和 other 的哈希值相同，则认为是一个 RoundOff 对象
  if (hasher_.hash(denom) == hasher_.hash(other)) {
    // 使用 div 的左操作数和右操作数创建一个 RoundOff 对象，并返回
    return alloc<RoundOff>(div->lhs(), div->rhs());
  }



  if (denom->isConstant() && other->isConstant()) {
    if (immediateEquals(denom, 0) || immediateEquals(other, 0)) {
      // 如果 denom 或 other 是 0，则返回空指针
      return nullptr;
    }
    // 如果 denom 和 other 都是标量，尝试找到一个公因子
    if (immediateEquals(evaluateOp(alloc<Mod>(other, denom)), 0)) {
      // 如果 other 对 denom 取模为 0，则计算 other 除以 denom 的结果
      ExprPtr scalar = evaluateOp(alloc<Div>(other, denom));
      // 计算 other 除以 scalar 的结果
      ExprPtr newDenom = evaluateOp(alloc<Div>(other, scalar));
      // 返回一个新的 Term 对象，使用 hasher_ 计算哈希值
      // scalar 作为标量，newDenom 作为 RoundOff 对象的右操作数
      return alloc<Term>(
          hasher_, scalar, alloc<RoundOff>(div->lhs(), newDenom));
    }
  }


  // 如果 denom 和 other 都是常量
  if (denom->isConstant() && other->isConstant()) {
    // 如果 denom 或 other 是 0，则返回空指针
    if (immediateEquals(denom, 0) || immediateEquals(other, 0)) {
      return nullptr;
    }
    // 如果 denom 和 other 都是标量，尝试找到一个公因子
    if (immediateEquals(evaluateOp(alloc<Mod>(other, denom)), 0)) {
      // 如果 other 对 denom 取模为 0，则计算 other 除以 denom 的结果
      ExprPtr scalar = evaluateOp(alloc<Div>(other, denom));
      // 计算 other 除以 scalar 的结果
      ExprPtr newDenom = evaluateOp(alloc<Div>(other, scalar));
      // 返回一个新的 Term 对象，使用 hasher_ 计算哈希值
      // scalar 作为标量，newDenom 作为 RoundOff 对象的右操作数
      return alloc<Term>(
          hasher_, scalar, alloc<RoundOff>(div->lhs(), newDenom));
    }
  }



  return nullptr;


  // 如果上述条件均不满足，则返回空指针
  return nullptr;
}

// 在一个项中插入一个新的组件，寻找简化的机会。
ExprPtr PolynomialTransformer::insertIntoTerm(TermPtr term, ExprPtr expr) {
  std::vector<ExprPtr> vars;  // 创建一个表达式指针的向量，用于存储项中的变量

  // 搜索 RoundOff。
  bool merged{false};  // 标志变量，表示是否进行了合并操作
  for (const auto& component : term->variables()) {  // 遍历项中的每一个组件
    if (auto roundoff = isRoundOff(component, expr)) {  // 判断是否存在 RoundOff
      vars.push_back(roundoff);  // 将找到的 RoundOff 添加到 vars 中
      merged = true;  // 设置合并标志为 true
    } else {
      vars.push_back(component);  // 否则将原始组件添加到 vars 中
    }
  }

  if (!merged) {  // 如果没有进行合并操作
    vars.push_back(expr);  // 直接将新的表达式添加到 vars 中
  }

  if (vars.size() == 1 && immediateEquals(term->scalar(), 1)) {  // 如果 vars 中只有一个元素且标量为 1
    return vars[0];  // 直接返回这个单独的变量表达式
  }

  return alloc<Term>(hasher_, term->scalar(), vars);  // 否则，使用 alloc 创建一个新的 Term 对象并返回
}

ExprPtr PolynomialTransformer::mutate(MulPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);  // 对左子表达式进行变异操作
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);  // 对右子表达式进行变异操作

  // 常量折叠。
  if (lhs_new->isConstant() && rhs_new->isConstant()) {  // 如果左右子表达式都是常量
    return evaluateOp(alloc<Mul>(lhs_new, rhs_new));  // 对它们进行乘法运算，并返回结果的表达式指针
  }

  // 多通道折叠。
  if (isMultilanePrimitive(lhs_new)) {  // 如果左子表达式是多通道基元
    if (auto ret = mulMultilane(lhs_new, rhs_new)) {  // 尝试对多通道进行乘法运算
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return ret->accept_mutator(this);  // 返回变异后的结果表达式指针
    }
  }

  // 顺序无关紧要。
  ExprPtr scalar = nullptr;
  ExprPtr variable = nullptr;
  if (lhs_new->isConstant()) {  // 如果左子表达式是常量
    scalar = lhs_new;  // 将其设为标量
    variable = rhs_new;  // 将右子表达式设为变量
  } else if (rhs_new->isConstant()) {  // 如果右子表达式是常量
    scalar = rhs_new;  // 将其设为标量
    variable = lhs_new;  // 将左子表达式设为变量
  }

  // 处理特殊情况乘以 1，即使是 Nan/Inf 也是安全的浮点运算。
  if (scalar && immediateEquals(scalar, 1)) {  // 如果标量为 1
    auto c = alloc<Cast>(v->dtype(), variable);  // 创建一个类型转换表达式
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c->accept_mutator(this);  // 返回变异后的类型转换表达式指针
  }

  // 如果这是浮点数乘法，操作顺序很重要，我们不希望合并操作。
  if (lhs_new->dtype().is_floating_point() || rhs_new->dtype().is_floating_point()) {
    return alloc<Mul>(lhs_new, rhs_new);  // 返回浮点数乘法表达式指针
  }

  // 处理乘以 0 的特殊情况。
  if (scalar && immediateEquals(scalar, 0)) {  // 如果标量为 0
    return immLike(v, 0);  // 返回与 v 类似的零表达式
  }

  // 捕获舍入的情况（Div(A/B) * B）。
  if (auto ret = isRoundOff(lhs_new, rhs_new)) {  // 判断是否是舍入操作
    return ret;  // 返回舍入操作的结果表达式指针
  } else if (auto ret = isRoundOff(v->lhs(), v->rhs())) {  // 否则，检查左右子表达式是否是舍入操作
    // 我们可以通过因式分解 Div 的方式打破 Round + Mod 模式，因此需要再次简化。
    return ret->accept_mutator(this);  // 返回简化后的表达式指针
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);  // 将左子表达式转换为多项式指针
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);  // 将右子表达式转换为多项式指针

  if (lhsPoly && rhsPoly) {  // 如果左右子表达式都是多项式
    // 这会扩展为更多项，通常无法修复，因此最好保持这些表达式作为乘法。
    return alloc<Mul>(lhsPoly, rhsPoly);  // 返回多项式乘法表达式指针
  }

  TermPtr lhsTerm = to<Term>(lhs_new);  // 将左子表达式转换为项指针
  TermPtr rhsTerm = to<Term>(rhs_new);  // 将右子表达式转换为项指针

  if (lhsPoly && rhsTerm) {  // 如果左子表达式是多项式且右子表达式是项
    return polyByTerm(lhsPoly, rhsTerm);  // 返回多项式乘以项的表达式指针
  }

  if (rhsPoly && lhsTerm) {  // 如果右子表达式是多项式且左子表达式是项
    // 在这种情况下，我们应该是 polyByTerm(lhsTerm, rhsPoly)，可能是笔误
    // 我们先假设是 polyByTerm 函数返回一个表达式指针
    return polyByTerm(lhsTerm, rhsPoly);  // 返回项乘以多项式的表达式指针
  }
  // 返回一个由 rhsPoly 乘以 lhsTerm 得到的结果。
  return polyByTerm(rhsPoly, lhsTerm);
}

if (lhsTerm && rhsTerm) {
  // 如果 lhsTerm 和 rhsTerm 都存在，则返回它们的乘积。
  return mulTerms(lhsTerm, rhsTerm);
}

if (scalar && lhsTerm) {
  // 如果存在标量 scalar 和左项项 lhsTerm，则将 scalar 乘以 lhsTerm 的标量部分，然后创建一个新的 Term 对象。
  ExprPtr newScalar = evaluateOp(alloc<Mul>(scalar, lhsTerm->scalar()));
  return alloc<Term>(hasher_, newScalar, lhsTerm->variables());
}

if (scalar && rhsTerm) {
  // 如果存在标量 scalar 和右项项 rhsTerm，则将 scalar 乘以 rhsTerm 的标量部分，然后创建一个新的 Term 对象。
  ExprPtr newScalar = evaluateOp(alloc<Mul>(scalar, rhsTerm->scalar()));
  return alloc<Term>(hasher_, newScalar, rhsTerm->variables());
}

// 如果存在标量 scalar 和左项多项式 lhsPoly，则将 scalar 包装为 Term 对象，然后使用 polyByTerm 处理。
if (scalar && lhsPoly) {
  return polyByTerm(lhsPoly, alloc<Term>(hasher_, scalar));
}
// 如果存在标量 scalar 和右项多项式 rhsPoly，则将 scalar 包装为 Term 对象，然后使用 polyByTerm 处理。
if (scalar && rhsPoly) {
  return polyByTerm(rhsPoly, alloc<Term>(hasher_, scalar));
}

// 对于简单的标量项，创建一个新的 Term 对象。
if (scalar) {
  return alloc<Term>(hasher_, scalar, variable);
}

// 如果存在左项多项式 lhsPoly，则创建一个包含 rhs_new 的 Term 对象，并使用 polyByTerm 处理。
if (lhsPoly) {
  auto term = alloc<Term>(hasher_, immLike(rhs_new, 1), rhs_new);
  return polyByTerm(lhsPoly, term);
}
// 如果存在右项多项式 rhsPoly，则创建一个包含 lhs_new 的 Term 对象，并使用 polyByTerm 处理。
if (rhsPoly) {
  auto term = alloc<Term>(hasher_, immLike(lhs_new, 1), lhs_new);
  return polyByTerm(rhsPoly, term);
}

// 如果存在左项项 lhsTerm，则将 rhs_new 插入到 lhsTerm 的变量列表中。
if (lhsTerm) {
  return insertIntoTerm(lhsTerm, rhs_new);
}
// 如果存在右项项 rhsTerm，则将 lhs_new 插入到 rhsTerm 的变量列表中。
if (rhsTerm) {
  return insertIntoTerm(rhsTerm, lhs_new);
}

// 如果既没有多项式也没有项，创建一个新的 Term 对象，包含变量 lhs_new 和 rhs_new。
return alloc<Term>(hasher_, immLike(v, 1), lhs_new, rhs_new);
}

// 函数：因式分解除法表达式
static ExprPtr factorizeDivision(ExprPtr lhs_new, ExprPtr rhs_new) {
  // 如果左右操作数有任意一个为空指针，返回空指针
  if (!lhs_new || !rhs_new) {
    return nullptr;
  }

  // 判断左操作数是否为常数，若是则赋给leftScalar，否则置为nullptr
  ExprPtr leftScalar = lhs_new->isConstant() ? lhs_new : nullptr;
  // 判断右操作数是否为常数，若是则赋给rightScalar，否则置为nullptr
  ExprPtr rightScalar = rhs_new->isConstant() ? rhs_new : nullptr;

  // 尝试将左操作数转换为Term类型，获取其标量
  auto lhsTerm = to<Term>(lhs_new);
  // 尝试将右操作数转换为Term类型，获取其标量
  auto rhsTerm = to<Term>(rhs_new);
  
  // 如果左操作数是Term类型，获取其标量赋给leftScalar
  if (lhsTerm) {
    leftScalar = lhsTerm->scalar();
  }

  // 如果右操作数是Term类型，获取其标量赋给rightScalar
  if (rhsTerm) {
    rightScalar = rhsTerm->scalar();
  }

  // 如果左右操作数中有任意一个不是常数，返回空指针
  if (!leftScalar || !rightScalar) {
    return nullptr;
  }

  // 将左右操作数的标量值转换为长整型
  long left = immediateAs<long>(leftScalar);
  long right = immediateAs<long>(rightScalar);

  // 计算左右操作数的最大公约数
  long GCD = gcd<long>(left, right);
  // 如果最大公约数小于等于1，返回空指针
  if (GCD <= 1) {
    return nullptr;
  }

  // 对左右操作数进行除以最大公约数的运算，生成新的标量表达式
  leftScalar = evaluateOp(alloc<Div>(leftScalar, immLike(leftScalar, GCD)));
  rightScalar = evaluateOp(alloc<Div>(rightScalar, immLike(rightScalar, GCD)));

  // 如果左操作数是Term类型，更新lhs_new为包含新标量的Term对象，否则为新的leftScalar
  if (lhsTerm) {
    lhs_new = alloc<Term>(lhsTerm->hasher(), leftScalar, lhsTerm->variables());
  } else {
    lhs_new = leftScalar;
  }

  // 如果右操作数是Term类型，更新rhs_new为包含新标量的Term对象，否则为新的rightScalar
  if (rhsTerm) {
    rhs_new = alloc<Term>(rhsTerm->hasher(), rightScalar, rhsTerm->variables());
  } else {
    rhs_new = rightScalar;
  }

  // 返回新的除法表达式节点
  return alloc<Div>(lhs_new, rhs_new);
}

// 函数：对除法表达式进行变异
ExprPtr PolynomialTransformer::mutate(DivPtr v) {
  // 对左右操作数分别调用当前变异器进行变异处理
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // 常数折叠优化
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    // 对常数操作数进行除法运算并返回结果
    return evaluateOp(alloc<Div>(lhs_new, rhs_new));
  }

  // 如果左右操作数中有任意一个为浮点数类型，直接返回原始的除法表达式节点
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Div>(lhs_new, rhs_new);
  }

  // 如果左操作数为常数且值为0，返回左操作数
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return lhs_new;
  }

  // 如果右操作数为常数且值为1，返回左操作数
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    return lhs_new;
  }

  // 尝试对除法表达式进行因式分解
  if (auto ret = factorizeDivision(lhs_new, rhs_new)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    // 对因式分解后的结果再次调用当前变异器进行变异处理并返回
    return ret->accept_mutator(this);
  }

  // 返回包含原始左右操作数的新的除法表达式节点
  return alloc<Div>(lhs_new, rhs_new);
}

// 函数：对取模表达式进行变异
ExprPtr PolynomialTransformer::mutate(ModPtr v) {
  // 对左右操作数分别调用当前变异器进行变异处理
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // 常数折叠优化
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    // 对常数操作数进行取模运算并返回结果
    return evaluateOp(alloc<Mod>(lhs_new, rhs_new));
  }

  // 如果左操作数为常数且值为0，返回左操作数
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return lhs_new;
  }

  // 返回包含原始左右操作数的新的取模表达式节点
  return alloc<Mod>(lhs_new, rhs_new);
}
    return lhs_new;
  }

  // 如果 rhs_new 是常量且等于 1，则结果为 0.
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    // 返回一个与 v 类型相同的常量 0
    return immLike(v, 0);
  }

  // 如果 lhs_new 和 rhs_new 相等，则结果为 0.
  if (hasher_.hash(lhs_new) == hasher_.hash(rhs_new)) {
    // 返回一个与 v 类型相同的常量 0
    return immLike(v, 0);
  }

  // 尝试将 lhs_new 转换为 Term 类型
  TermPtr lhsTerm = to<Term>(lhs_new);
  if (!lhsTerm) {
    // 如果无法转换为 Term 类型，则尝试转换为 Polynomial 类型
    PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
    if (lhsPoly) {
      // 如果能够因式分解多项式，则继续优化处理
      lhsTerm = factorizePolynomial(lhsPoly);
    }
  }

  if (lhsTerm) {
    // 如果 rhs_new 是常量且 lhsTerm 的标量与 rhs_new 取模等于 0，则结果为 0.
    if (rhs_new->isConstant() &&
        immediateEquals(
            evaluateOp(alloc<Mod>(lhsTerm->scalar(), rhs_new)), 0)) {
      // 返回一个与 v 类型相同的常量 0
      return immLike(v, 0);
    }

    // 如果 lhsTerm 包含的变量与 rhs_new 的变量相同，则结果为 0.
    for (const auto& component : lhsTerm->variables()) {
      if (hasher_.hash(component) == hasher_.hash(rhs_new)) {
        // 返回一个与 v 类型相同的常量 0
        return immLike(v, 0);
      }
    }

    // 如果 lhsTerm 和 rhsTerm 都是 Term 类型，并且 rhsTerm 的变量集合包含在 lhsTerm 的变量集合中，
    // 且 lhsTerm 的标量对 rhsTerm 取模等于 0，则结果为 0.
    TermPtr rhsTerm = to<Term>(rhs_new);
    if (rhsTerm) {
      auto& lVars = lhsTerm->variables();
      auto& rVars = rhsTerm->variables();
      size_t rLeft = rVars.size();

      auto rIt = rVars.begin();

      for (auto lIt = lVars.begin(); lIt != lVars.end() && !rVars.empty();
           ++lIt) {
        auto lHash = hasher_.hash(*lIt);
        for (; rIt != rVars.end(); ++rIt) {
          auto rHash = hasher_.hash(*rIt);
          if (lHash == rHash) {
            --rLeft;
            break;
          } else if (lHash < rHash) {
            break;
          }
        }
      }

      if (rLeft == 0 &&
          immediateEquals(
              evaluateOp(alloc<Mod>(lhsTerm->scalar(), rhsTerm->scalar())),
              0)) {
        // 返回一个与 v 类型相同的常量 0
        return immLike(v, 0);
      }
    }
  }

  // 如果以上条件都不满足，则返回 lhs_new 和 rhs_new 取模的结果
  return alloc<Mod>(lhs_new, rhs_new);
}

namespace {

// Combines two MinTerm / MaxTerm expressions into one.
// The first type on the template refers to the operation type (Min or Max),
// and the second type refers to the corresponding term type (MinTerm or MaxTerm).
template <class Op, class OpTerm>
ExprPtr combineMinMaxTerms(
    ExprPtr lhs,                          // 左操作数表达式指针
    ExprPtr rhs,                          // 右操作数表达式指针
    bool propagate_nans,                  // 是否传播 NaN
    HashProvider& hasher) {               // 哈希提供器的引用
  auto combine_scalars = [&](ExprPtr c1, ExprPtr c2) -> ExprPtr {  // 组合标量值的lambda函数
    if (c1 && c2) {
      return evaluateOp(alloc<Op>(c1, c2, propagate_nans));       // 如果两个操作数都存在，则执行操作并返回结果
    }
    if (c1) {
      return c1;                                                  // 如果只有一个操作数存在，则返回该操作数
    }
    return c2;                                                     
  };

  auto combine_opterms = [&](NodePtr<OpTerm> m1, NodePtr<OpTerm> m2) {  // 组合 MinTerm 或 MaxTerm 的lambda函数
    ExprPtr scalar = combine_scalars(m1->scalar(), m2->scalar());       // 组合标量值
    std::vector<ExprPtr> variables;
    for (const auto& v : m1->variables()) {
      variables.push_back(v);                                          // 将 m1 的变量复制到 variables
    }
    for (const auto& v : m2->variables()) {
      variables.push_back(v);                                          // 将 m2 的变量复制到 variables
    }
    return alloc<OpTerm>(hasher, scalar, propagate_nans, std::move(variables));  // 创建新的 OpTerm 对象
  };

  auto add_expr_to_opterm = [&](ExprPtr expr, NodePtr<OpTerm> opterm) {  // 将表达式添加到 OpTerm 的lambda函数
    ExprPtr scalar = nullptr;
    std::vector<ExprPtr> variables;
    if (opterm) {
      scalar = opterm->scalar();
      variables = opterm->variables();                                   // 如果 opterm 存在，则复制其标量和变量
    }
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (expr->isConstant()) {
      scalar = combine_scalars(scalar, expr);                            // 如果表达式是常数，则组合标量值
    } else {
      variables.push_back(expr);                                         // 否则将表达式添加到变量列表中
    }
    return alloc<OpTerm>(hasher, scalar, propagate_nans, std::move(variables));  // 创建新的 OpTerm 对象
  };

  auto lhs_opterm = to<OpTerm>(lhs);    // 尝试将 lhs 转换为 OpTerm
  auto rhs_opterm = to<OpTerm>(rhs);    // 尝试将 rhs 转换为 OpTerm
  if (lhs_opterm && lhs_opterm->propagate_nans() != propagate_nans) {
    return alloc<Op>(lhs, rhs, propagate_nans);  // 如果 lhs 是 OpTerm 且 NaN 传播属性不同，则返回新的 Op 对象
  }
  if (rhs_opterm && rhs_opterm->propagate_nans() != propagate_nans) {
    return alloc<Op>(lhs, rhs, propagate_nans);  // 如果 rhs 是 OpTerm 且 NaN 传播属性不同，则返回新的 Op 对象
  }

  if (lhs_opterm && rhs_opterm) {
    return combine_opterms(lhs_opterm, rhs_opterm);  // 如果 lhs 和 rhs 都是 OpTerm，则组合它们
  } else if (lhs_opterm) {
    return add_expr_to_opterm(rhs, lhs_opterm);      // 如果只有 lhs 是 OpTerm，则将 rhs 添加到 lhs 中
  } else if (rhs_opterm) {
    return add_expr_to_opterm(lhs, rhs_opterm);      // 如果只有 rhs 是 OpTerm，则将 lhs 添加到 rhs 中
  }
  return add_expr_to_opterm(rhs, add_expr_to_opterm(lhs, nullptr));  // 否则将 lhs 和 rhs 都添加到新的 OpTerm 中
}

// Returns true if op is one of the 2 operands in opterm and also returns
// the other operand of opterm in other_op.
template <class OpTerm>
bool isOperandInMinMaxTerm(
    NodePtr<OpTerm> opterm,               // MinTerm 或 MaxTerm 节点指针
    ExprPtr op,                          // 操作数表达式指针
    HashProvider& hasher,                // 哈希提供器的引用
    ExprPtr* other_op) {                 // 其他操作数表达式指针的指针
  if (opterm->variables().size() != 2) {
    return false;                       // 如果 opterm 的变量数不等于 2，则返回 false
  }
  auto lhs = opterm->variables()[0];    // 获取 opterm 的第一个变量
  auto rhs = opterm->variables()[1];    // 获取 opterm 的第二个变量
  auto op_hash = hasher.hash(op);       // 计算操作数的哈希值
  if (hasher.hash(lhs) == op_hash) {
    *other_op = rhs;                    // 如果操作数的哈希等于 lhs 的哈希，则将 rhs 赋给 other_op
    return true;
  } else if (hasher.hash(rhs) == op_hash) {
    *other_op = lhs;                    // 如果操作数的哈希等于 rhs 的哈希，则将 lhs 赋给 other_op
    return true;
  }
  return false;                         // 否则返回 false
};

// Simplifies the nested min-max pattern like:
//   * Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
//   * Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
// This function is called while processing the outer Min / Max ops.
// 当前函数用于简化嵌套的最大-最小操作，检查给定的 lhs 和 rhs 中是否包含特定的术语表达式。
// 模板的第一个类型必须是外部操作对应的术语类型（例如 MaxTerm），第二个类型必须是期望的内部操作对应的术语类型（例如 MinTerm）。
template <class OpTerm, class OtherOpTerm>
bool simplifyNestedMinMax(
    ExprPtr lhs,                         // 左操作数表达式指针
    ExprPtr rhs,                         // 右操作数表达式指针
    bool propagate_nans,                 // 是否传播 NaN 值的布尔值
    HashProvider& hasher,                // 哈希提供者的引用
    ExprPtr* new_op) {                   // 输出参数，用于存储简化后的新操作表达式指针

  auto lhs_opterm = to<OtherOpTerm>(lhs); // 尝试将 lhs 转换为 OtherOpTerm 类型
  auto rhs_opterm = to<OtherOpTerm>(rhs); // 尝试将 rhs 转换为 OtherOpTerm 类型
  
  // 检查 lhs 和 rhs 是否都成功转换为 OtherOpTerm，并且满足传播 NaN 值的条件
  if (lhs_opterm && rhs_opterm &&
      lhs_opterm->propagate_nans() == propagate_nans &&
      rhs_opterm->propagate_nans() == propagate_nans) {
    
    // 如果 lhs 和 rhs 都不是标量，并且变量的数量都是 2
    if (!lhs_opterm->scalar() && !rhs_opterm->scalar()) {
      if (lhs_opterm->variables().size() == 2 &&
          rhs_opterm->variables().size() == 2) {
        
        auto rhs_v1 = rhs_opterm->variables()[0]; // 获取 rhs 的第一个变量
        auto rhs_v2 = rhs_opterm->variables()[1]; // 获取 rhs 的第二个变量
        
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ExprPtr new_op_lhs;
        
        // 检查 lhs_opterm 中是否存在 rhs_v1 变量的操作，并返回新的左操作数表达式指针
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v1, hasher, &new_op_lhs)) {
          // 创建 OpTerm 类型的内部操作，传入哈希提供者、传播 NaN 值、新的左操作数表达式指针和 rhs_v2
          auto inner_op = alloc<OpTerm>(
              hasher, nullptr, propagate_nans, new_op_lhs, rhs_v2);
          // 创建 OtherOpTerm 类型的新操作，传入哈希提供者、传播 NaN 值、rhs_v1 和内部操作
          *new_op = alloc<OtherOpTerm>(
              hasher, nullptr, propagate_nans, rhs_v1, inner_op);
          return true;
        }
        
        // 检查 lhs_opterm 中是否存在 rhs_v2 变量的操作，并返回新的左操作数表达式指针
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v2, hasher, &new_op_lhs)) {
          // 创建 OpTerm 类型的内部操作，传入哈希提供者、传播 NaN 值、新的左操作数表达式指针和 rhs_v1
          auto inner_op = alloc<OpTerm>(
              hasher, nullptr, propagate_nans, new_op_lhs, rhs_v1);
          // 创建 OtherOpTerm 类型的新操作，传入哈希提供者、传播 NaN 值、rhs_v2 和内部操作
          *new_op = alloc<OtherOpTerm>(
              hasher, nullptr, propagate_nans, rhs_v2, inner_op);
          return true;
        }
      }
    }
  }
  // 如果没有成功简化嵌套的最大-最小操作，则返回 false
  return false;
}

namespace {

// 对 MaxPtr 类型进行变异，返回变异后的表达式指针
ExprPtr PolynomialTransformer::mutate(MaxPtr v) {
  // 分别对左右操作数进行变异
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // 常数折叠
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Max>(lhs_new, rhs_new, v->propagate_nans()));
  }

  // 如果差值为常数，则返回适当的操作数
  ExprPtr diff = alloc<Sub>(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) > 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // 简化嵌套的最大-最小操作
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr new_op;
  if (simplifyNestedMinMax<MaxTerm, MinTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  // 合并最大操作的术语
  return combineMinMaxTerms<Max, MaxTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}
ExprPtr PolynomialTransformer::mutate(MinPtr v) {
  // 获取新的左操作数和右操作数
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // 常数折叠。
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    // 如果左右操作数均为常数，则返回计算后的最小值表达式
    return evaluateOp(alloc<Min>(lhs_new, rhs_new, v->propagate_nans()));
  }

  // 如果差值为常数，返回相应的操作数。
  ExprPtr diff = alloc<Sub>(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    // 如果差值为常数，根据其值确定返回哪个操作数
    if (immediateAs<int>(diff) < 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr new_op;
  // 如果可以简化嵌套的最小最大值操作，则返回简化后的新操作
  if (simplifyNestedMinMax<MinTerm, MaxTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  // 合并最小最大值项并返回结果
  return combineMinMaxTerms<Min, MinTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

ExprPtr PolynomialTransformer::mutate(CompareSelectPtr v) {
  // 获取新的左操作数、右操作数以及真假分支
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);
  ExprPtr true_branch = v->ret_val1()->accept_mutator(this);
  ExprPtr false_branch = v->ret_val2()->accept_mutator(this);

  // 常数折叠。
  if (lhs_new->isConstant() && rhs_new->isConstant() &&
      true_branch->isConstant() && false_branch->isConstant()) {
    // 如果所有操作数均为常数，则返回计算后的比较选择表达式
    ExprPtr v_new = alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
    return evaluateOp(v_new);
  }

  // 如果比较涉及浮点数，不尝试简化差值，因为无法正确处理 NaN。
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    // 如果操作数中存在浮点数，则返回比较选择表达式
    return alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
  }

  // 如果差值为常数，可以确定结果。
  ExprPtr diff = alloc<Sub>(rhs_new, lhs_new);
  diff = diff->accept_mutator(this);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  if (!diff->isConstant()) {
    // 如果差值不是常数，则返回比较选择表达式
    return alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        v->compare_select_op(),
        v->bias());
  }

  // 根据比较选择操作返回相应的分支
  bool equal = immediateEquals(diff, 0);
  bool lhsSmaller = !equal && !immediateIsNegative(diff);

  switch (v->compare_select_op()) {
    case CompareSelectOperation::kEQ:
      return equal ? true_branch : false_branch;
    case CompareSelectOperation::kGT:
      return (lhsSmaller || equal) ? false_branch : true_branch;
    case CompareSelectOperation::kGE:
      return lhsSmaller ? false_branch : true_branch;
    case CompareSelectOperation::kLT:
      return lhsSmaller ? true_branch : false_branch;
  // 根据比较选择操作符进行条件选择
  case CompareSelectOperation::kLE:
    // 如果左操作数小于等于右操作数或者它们相等，则选择 true_branch，否则选择 false_branch
    return (lhsSmaller || equal) ? true_branch : false_branch;

  // 根据比较选择操作符进行条件选择
  case CompareSelectOperation::kNE:
    // 如果左右操作数不相等，则选择 false_branch，否则选择 true_branch
    return equal ? false_branch : true_branch;
}

// 在正常情况下不应该执行到这里，添加一个保险的返回语句
return alloc<CompareSelect>(
    lhs_new,
    rhs_new,
    true_branch,
    false_branch,
    v->compare_select_op(),
    v->bias());
}

ExprPtr PolynomialTransformer::mutate(IntrinsicsPtr v) {
  // 初始化一个新的参数列表和变量用于跟踪是否有变化和是否所有子表达式都是常量
  std::vector<ExprPtr> new_params;
  bool changed = false;
  bool allConstant = true;
  // 遍历给定的内置函数表达式的参数列表
  for (const auto& p : v->params()) {
    // 对每个参数应用变异器，生成新的子表达式
    ExprPtr new_child = p->accept_mutator(this);
    new_params.push_back(new_child);

    // 检查是否有参数发生了变化
    changed |= p != new_child;
    // 检查新的子表达式是否都是常量
    allConstant &= new_child->isConstant();
  }

  // 如果有参数发生了变化，则重新分配内置函数表达式节点
  ExprPtr node = v;
  if (changed) {
    node = alloc<Intrinsics>(v->op_type(), new_params);
  }

  // 如果不是所有子表达式都是常量或者原始表达式不是纯表达式，则返回新节点
  if (!allConstant || !v->isPure()) {
    return node;
  }

  // 如果所有参数都是常量且原始表达式是纯表达式，则需要进行评估操作
  // 准备用于评估的参数列表
  std::vector<ExprPtr> const_params;
  changed = false;
  for (const auto& p : new_params) {
    // 如果参数的标量类型是浮点数，则保留该参数
    if (p->dtype().scalar_type() == ScalarType::Float) {
      const_params.push_back(p);
    } else {
      // 否则进行类型转换为浮点数
      const_params.push_back(
          alloc<Cast>(Dtype(ScalarType::Float, p->dtype().lanes()), p));
      changed = true;
    }
  }

  // 如果有类型转换发生，则重新分配内置函数表达式节点
  if (changed) {
    node = alloc<Intrinsics>(v->op_type(), const_params);
  }
  // 返回最终评估的结果节点
  return evaluateOp(node);
}

ExprPtr PolynomialTransformer::mutate(CastPtr v) {
  // 获取源值并应用变异器
  ExprPtr node = v->src_value()->accept_mutator(this);
  // 如果源值是常量，则对转换操作进行评估
  if (node->isConstant()) {
    return evaluateOp(alloc<Cast>(v->dtype(), node));
  }

  // 如果目标类型和源值类型相同，则直接返回源值
  if (v->dtype() == node->dtype()) {
    return node;
  }

  // 否则进行类型转换
  return alloc<Cast>(v->dtype(), node);
}

ExprPtr PolynomialTransformer::mutate(IfThenElsePtr v) {
  // 获取条件、真值和假值，并分别应用变异器
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();
  ExprPtr condition_new = condition->accept_mutator(this);
  ExprPtr true_value_new = true_value->accept_mutator(this);
  ExprPtr false_value_new = false_value->accept_mutator(this);

  // 如果条件是常量，则根据条件值选择真值或假值
  if (condition_new->isConstant()) {
    if (!immediateEquals(condition_new, 0)) {
      return true_value_new;
    } else {
      return false_value_new;
    }
  }

  // 如果真值和假值哈希相同，则返回其中一个
  if (hasher_.hash(true_value_new) == hasher_.hash(false_value_new)) {
    return true_value_new;
  }

  // 如果没有变异器应用于条件、真值和假值，则返回原始 IfThenElse 节点
  if (condition == condition_new && true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  // 否则，创建一个新的 IfThenElse 节点
  return alloc<IfThenElse>(condition_new, true_value_new, false_value_new);
}

ExprPtr PolynomialTransformer::mutate(AndPtr v) {
  // 对于 And 节点，将其视为二元操作符进行处理
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(XorPtr v) {
  // 对于 Xor 节点，将其视为二元操作符进行处理
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(LshiftPtr v) {
  // 对于 Lshift 节点，将其视为二元操作符进行处理
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(RshiftPtr v) {
  // 对于 Rshift 节点，将其视为二元操作符进行处理
  return mutateBinaryOp(v, this);
}
// 在多项式基类中，根据给定条件指针 v 进行变异操作，返回变异后的语句指针
StmtPtr PolynomialBase::mutate(CondPtr v) {
  // 获取条件表达式、真实语句和假语句的旧版本
  ExprPtr cond_old = v->condition();
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();

  // 对条件表达式进行变异，得到新的条件表达式
  ExprPtr cond_new = cond_old->accept_mutator(this);
  // 如果旧的真实语句存在，则对其进行变异；否则保持不变
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  // 如果旧的假语句存在，则对其进行变异；否则保持不变
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;

  // 如果条件是常量，则可以根据条件直接选择正确的分支
  if (cond_new->isConstant()) {
    if (!immediateEquals(cond_new, 0)) {
      // 如果条件不为0，返回变异后的真实语句
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return true_new;
    } else {
      // 如果条件为0，返回变异后的假语句
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return false_new;
    }
  }

  // 如果真实语句和假语句相同，则不需要进行条件判断
  if (true_new && false_new &&
      hasher_.hash(true_new) == hasher_.hash(false_new)) {
    return true_new;
  }

  // 尝试将变异后的真实语句和假语句转换为块类型
  BlockPtr true_block = to<Block>(true_new);
  BlockPtr false_block = to<Block>(false_new);
  // 检查真实语句是否为空或者是否只包含空语句
  bool true_empty = !true_new || (true_block && true_block->nstmts() == 0);
  // 检查假语句是否为空或者是否只包含空语句
  bool false_empty = !false_new || (false_block && false_block->nstmts() == 0);

  // 如果真实语句和假语句都为空，则返回一个空块
  if (true_empty && false_empty) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }
  // 如果条件表达式发生变化，则更新条件指针的条件
  if (cond_old != cond_new) {
    v->set_condition(cond_new);
  }
  // 如果真实语句发生变化，则更新条件指针的真实语句
  if (true_old != true_new) {
    v->set_true_stmt(true_new);
  }
  // 如果假语句发生变化，则更新条件指针的假语句
  if (false_old != false_new) {
    v->set_false_stmt(false_new);
  }
  // 返回更新后的条件指针
  return v;
}

// 处理循环条件重排序的静态函数，接受一个循环指针和条件指针作为参数
static StmtPtr handleForCondReordering(ForPtr loop, CondPtr cond) {
  // 如果条件指针的假语句不为空，则直接返回空指针
  if (cond->false_stmt()) {
    return nullptr;
  }

  // 查找条件表达式中涉及的变量
  auto condition_vars = VarFinder::find(cond->condition());
  // 遍历条件表达式中的变量
  for (const auto& v : condition_vars) {
    // 如果条件依赖于在循环体中被修改的变量，则不安全，返回空指针
    if (ModifiesVarChecker::check(loop, v)) {
      return nullptr;
    }
  }

  // 克隆一个新的循环对象，并用条件指针的真实语句替换其循环体
  ForPtr new_f = loop->cloneWithNewBody(Stmt::clone(cond->true_stmt()));
  return cond->cloneWithNewBody(new_f);
}

// 在多项式基类中，根据给定循环指针 v 进行变异操作
StmtPtr PolynomialBase::mutate(ForPtr v) {
  // 获取循环变量、起始值、终止值、循环体和循环选项的表达式或指针
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  // 对循环变量进行变异，得到新的表达式
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  // 对起始值和终止值进行变异，得到新的表达式
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  // 循环体保持不变
  StmtPtr body_new = body;

  // 计算循环次数表达式：终止值减去起始值
  ExprPtr loops = alloc<Sub>(stop_new, start_new);
  loops = loops->accept_mutator(this);
  // 如果循环选项是默认的，并且循环次数是常量
  if (loop_options.isDefault() && loops->isConstant()) {
    // 如果循环次数为0，返回一个空块
    if (immediateEquals(loops, 0)) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    }
    // 如果循环次数为1，用起始值替换循环体中的循环变量，并返回变异后的循环体
    else if (immediateEquals(loops, 1)) {
      body_new = Substitute(body, {{var_new, start_new}});
      body_new = body_new->accept_mutator(this);
      return body_new;
    }
  }

  // 对循环体进行变异，得到新的循环体
  body_new = body_new->accept_mutator(this);
  // 如果新的循环体为空，则返回一个空块
  if (!body_new) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }

  // 如果新的循环体是块类型
  if (auto block = to<Block>(body_new)) {
    # 如果块中没有语句
    if (block->nstmts() == 0) {
      # 返回一个空的块对象，使用 alloc 函数分配内存
      return alloc<Block>(std::vector<StmtPtr>({}));
    }

    # 如果块中只有一个语句
    if (block->nstmts() == 1) {
      # 如果这个语句是条件语句
      if (auto cond = to<Cond>(block->front())) {
        # 对条件语句进行重新排序处理
        StmtPtr reordered = handleForCondReordering(v, cond);
        # 如果重新排序成功，则接受变异器处理后的结果
        if (reordered) {
          return reordered->accept_mutator(this);
        }
      }
    }
  }

  # 更新变量 var
  if (var != var_new) {
    v->set_var(var_new);
  }
  # 更新起始值 start
  if (start != start_new) {
    v->set_start(start_new);
  }
  # 更新终止值 stop
  if (stop != stop_new) {
    v->set_stop(stop_new);
  }
  # 更新循环体 body
  if (body != body_new) {
    v->set_body(body_new);
  }
  # 返回更新后的对象 v
  return v;
}



StmtPtr PolynomialBase::mutate(BlockPtr v) {
  std::vector<StmtPtr> stmts;
  // Flatten sub-blocks:
  bool stmts_changed = false;
  // 遍历给定的块（v），对其中的语句进行处理
  for (const StmtPtr& stmt : *v) {
    // 使用当前对象作为 mutator 来处理当前语句（stmt）
    StmtPtr stmt_new = stmt->accept_mutator(this);
    // 检查语句是否发生了改变
    stmts_changed |= stmt != stmt_new;
    // 如果新语句为空指针，则继续处理下一个语句
    if (stmt_new == nullptr) {
      continue;
    }

    // 如果新语句是一个块（subBlock）
    if (auto subBlock = to<Block>(stmt_new)) {
      // 遍历块中的语句，注意避免迭代器失效
      for (Block::iterator I = subBlock->begin(), E = subBlock->end();
           I != E;) {
        // 获取当前语句，并从块中移除
        StmtPtr s = *(I++);
        subBlock->remove_stmt(s);
        // 将移除的语句添加到 stmts 中
        stmts.push_back(s);
      }
      stmts_changed = true;
    } else {
      // 将处理后的语句添加到 stmts 中
      stmts.push_back(stmt_new);
    }
  }
  // 如果语句发生了改变，则更新块（v）中的语句
  if (stmts_changed) {
    v->set_stmts(stmts);
  }
  // 返回处理后的块（v）
  return v;
}

// TermExpander

ExprPtr TermExpander::mutate(TermPtr v) {
  // 处理 Term 中的标量部分
  ExprPtr newScalar = v->scalar()->accept_mutator(this);
  // 如果标量部分为 0，则直接返回
  if (immediateEquals(newScalar, 0)) {
    return newScalar;
  }

  std::vector<ExprPtr> vars;
  std::vector<ExprPtr> multilaneVars;

  // 假设可以在此处重新排序，因为不会合并浮动项。
  ExprPtr lastNode{nullptr};
  // 遍历 Term 中的变量部分
  for (const auto& var : v->variables()) {
    // 对每个变量使用当前对象作为 mutator 来处理
    ExprPtr node = var->accept_mutator(this);
    // 如果变量节点是一个乘法表达式（Mul）
    if (MulPtr mul = to<Mul>(node)) {
      // 如果子表达式可以解析为一个乘法，将其提升到当前项中
      if (isMultilanePrimitive(mul->lhs())) {
        multilaneVars.push_back(mul->lhs());
      } else {
        vars.push_back(mul->lhs());
      }

      if (isMultilanePrimitive(mul->rhs())) {
        multilaneVars.push_back(mul->rhs());
      } else {
        vars.push_back(mul->rhs());
      }
    } else {
      // 如果变量节点是一个多行表达式，将其添加到多行变量集合中
      if (isMultilanePrimitive(node)) {
        multilaneVars.push_back(node);
      } else {
        vars.push_back(node);
      }
    }
  }

  // 处理多行变量集合
  for (const auto& node : multilaneVars) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      // 对多行变量进行乘法处理
      lastNode = mulMultilane(lastNode, node);
      // 先简化，然后重新扩展
      lastNode = lastNode->accept_mutator(simplifier_);
      lastNode = lastNode->accept_mutator(this);
    }
  }

  // 处理普通变量集合
  for (const auto& node : vars) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      // 分配一个新的乘法表达式
      lastNode = alloc<Mul>(lastNode, node);
    }
  }

  // 如果标量部分不为 1，则将其乘到最后的表达式中
  if (!immediateEquals(newScalar, 1)) {
    if (lastNode) {
      // 如果存在上一个节点（lastNode），执行以下逻辑：
      
      // 获取当前标量节点（v）的数据类型
      auto termDtype = v->scalar()->dtype();
      // 获取上一个节点（lastNode）的数据类型
      auto lastNodeDtype = lastNode->dtype();
      
      // 检查当前节点和上一个节点的数据类型是否相同
      if (termDtype != lastNodeDtype) {
        // 如果数据类型不同，进行类型转换操作
        
        // 将当前标量节点（v）转换为一个表达式指针（castV）
        ExprPtr castV = v->scalar();
        
        // 处理数据类型中的 lane 不匹配情况
        if (termDtype.lanes() != lastNodeDtype.lanes()) {
          // 如果 lane 数量不同，使用广播操作调整标量节点的形状
          castV = alloc<Broadcast>(v->scalar(), lastNodeDtype.lanes());
        }
        
        // 处理标量类型不匹配情况
        if (termDtype.scalar_type() != lastNodeDtype.scalar_type()) {
          // 将标量节点（castV）转换为与上一个节点相同的数据类型
          castV = alloc<Cast>(lastNode->dtype(), castV);
          
          // 对于标量，可以进一步简化类型转换操作
          if (lastNodeDtype.lanes() == 1) {
            // 如果上一个节点是标量，则对转换后的节点进行评估操作
            castV = evaluateOp(castV);
          }
        }
        
        // 使用转换后的标量节点（castV）与上一个节点进行乘法操作
        lastNode = alloc<Mul>(castV, lastNode);
      } else {
        // 如果数据类型相同，直接使用当前标量节点（v）与上一个节点进行乘法操作
        lastNode = alloc<Mul>(v->scalar(), lastNode);
      }
    } else {
      // 如果不存在上一个节点（lastNode），直接将当前标量节点（v）赋给上一个节点
      lastNode = v->scalar();
    }
  }

  // 返回最后处理的节点（lastNode）
  return lastNode;
}

// 返回多项式中所有项（包括标量项）的最大公约数的立即数表示。
// 如果最大公约数不足以节省工作（例如为1），则返回nullptr。
static ExprPtr polyGCD(PolynomialPtr poly) {
  ExprPtr scalar = poly->scalar(); // 获取多项式的标量项
  const std::vector<TermPtr>& variables = poly->variables(); // 获取多项式的变量项列表

  // 只有在节省完整操作时才进行因式分解，例如，不值得将6x + 4y分解为2 * (3x + 2y)，因为没有节省工作。
  int opsSaved = 1; // 默认保存标量项的操作次数。
  long GCD = std::abs(immediateAs<long>(scalar)); // 计算标量项的绝对值作为初始最大公约数
  for (const auto& t : variables) {
    long termScalar = std::abs(immediateAs<long>(t->scalar())); // 计算每个变量项的标量项的绝对值
    long newGCD = gcd(std::max(GCD, termScalar), std::min(GCD, termScalar)); // 计算当前项与之前最大公约数的最大公约数
    if (newGCD == 1) { // 如果新的最大公约数为1，表示不值得继续计算，返回nullptr
      return nullptr;
    }

    if (GCD != newGCD) { // 如果发现新的最大公约数与之前的不同，则说明不是所有项的公约数，需要单独计算
      opsSaved = 0; // 不再保存操作
      GCD = newGCD; // 更新最大公约数
    }

    if (GCD == termScalar) { // 如果当前项的标量项与最大公约数相同，增加操作次数
      opsSaved++;
    }
  }

  if (opsSaved == 0) { // 如果没有节省操作，返回nullptr
    return nullptr;
  }

  if (GCD == 0) { // 如果最大公约数为0，返回nullptr
    return nullptr;
  }

  // 不值得，可以是Sub。
  if (GCD == -1 && opsSaved == 1) { // 如果最大公约数为-1且只保存了一次操作，返回nullptr
    return nullptr;
  }

  return immLike(poly, GCD); // 返回与多项式类型相似的最大公约数表达式
}

// ModRound 是一个除模乘操作，其中除数在div中，乘数在mul中，二者相同且不等于1。
// 在 ModRound x/y%z*y*c（其中c为常数）中，'scalar' 表示c，'denominator' 表示x，'divisor' 表示y，'mod_divisor' 表示z。
class ModRound {
 public:
  ModRound(ExprPtr scalar, ExprPtr denom, ExprPtr divisor, ExprPtr mod_divisor)
      : scalar(std::move(scalar)),
        denom(std::move(denom)),
        divisor(std::move(divisor)),
        mod_divisor(std::move(mod_divisor)) {}
  ExprPtr scalar;
  ExprPtr denom;
  ExprPtr divisor;
  ExprPtr mod_divisor;
};

static std::optional<class ModRound> isModRound(TermPtr e) {
  DivPtr div{nullptr};
  ModPtr mod{nullptr};
  ExprPtr denom{nullptr};
  ExprPtr divisor{nullptr};
  ExprPtr mod_divisor{nullptr};
  ExprPtr multiplier = e->scalar();
  ExprPtr scalar{nullptr};
  ExprPtr other{nullptr};

  for (const auto& m : e->variables()) {
    if (m->expr_type() == IRNodeType::kMod) { // 检查变量类型是否为 Mod
      // TODO: 目前仅识别一个变量为 Mod 的项；如果需要处理如 (t/(x%2 * y) % z) * (x%2 *y) 这样的项，可以扩展这一逻辑。
      if (!mod) {
        mod = to<Mod>(m); // 将变量转换为 Mod 类型
      } else {
        return c10::nullopt; // 如果已经有一个 Mod 变量，返回空
      }
    }
  } else {
    // 处理在乘以标量和变量之前的特殊情况。
    if (multiplier->isConstant()) {
      // 处理先处理通道不匹配的情况。
      if (multiplier->dtype().lanes() != m->dtype().lanes()) {
        // 如果通道数不匹配，则使用广播方式扩展标量。
        multiplier = alloc<Broadcast>(multiplier, m->dtype().lanes());
      }
      // 处理标量类型不匹配的情况。
      if (multiplier->dtype().scalar_type() != m->dtype().scalar_type()) {
        // 如果标量类型不匹配，则进行类型转换。
        multiplier = alloc<Cast>(m->dtype(), multiplier);
        // 如果 m 的通道数为 1，则对 multiplier 进行评估操作。
        if (m->dtype().lanes() == 1) {
          multiplier = evaluateOp(multiplier);
        }
      }
    }

    // 将所有非 mod 变量视为乘数的一部分。
    multiplier = alloc<Mul>(multiplier, m);
  }
}
// 简化乘数表达式。
multiplier = IRSimplifier::simplify(multiplier);

if (!mod) {
  // 如果 mod 为空指针，则返回空的 optional 对象。
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return c10::nullopt;
}

// 简化 mod 表达式的右操作数。
mod_divisor = IRSimplifier::simplify(mod->rhs());
other = mod->lhs();

// 如果 other 不是 Div 类型，则返回空的 optional 对象。
if (!(div = to<Div>(other))) {
  return c10::nullopt;
}

// 简化除法表达式的右操作数作为除数。
divisor = IRSimplifier::simplify(div->rhs());
other = div->lhs();

// 简化除法表达式的左操作数作为分母。
denom = IRSimplifier::simplify(other);

// 拒绝除数和乘数不相等的情况。
HashProvider& hasher = e->hasher();
if (hasher.hash(divisor) != hasher.hash(multiplier)) {
  // TODO: 目前如果除数和乘数不是常量，我们不会提取公因数。这种提取在 IRSimplifier.simplify 中不支持（例如 x*2/x -> 2），
  // 因为 x 可能为 0。未来工作可以扩展除法的两个版本：1）对必须严格简化的客户进行除法和2）我们引入的转换中使用的除法，
  // 可以简化而不考虑 0，例如 Div_nonzero。第二种除法仅用于促进我们的转换。
  if (divisor->isConstant() && multiplier->isConstant()) {
    // 如果两者都是标量，我们可能能够找到一个公因数。
    if (immediateEquals(evaluateOp(alloc<Mod>(multiplier, divisor)), 0)) {
      // 公因数成为术语的“标量”，例如在 t/3%7*6 中，divisor=multiplier=3，标量=2。
      ExprPtr c = evaluateOp(alloc<Div>(multiplier, divisor));
      scalar = c;
    } else if (immediateEquals(
                   evaluateOp(alloc<Mod>(divisor, multiplier)), 0)) {
      // 公因数成为“denom”的一部分，例如在 t/14%7*2 中，divisor=multiplier=2，denom=t/7。
      ExprPtr c = evaluateOp(alloc<Div>(divisor, multiplier));
      divisor = multiplier;
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      denom = IRSimplifier::simplify(alloc<Div>(other, c));
    } else {
      return c10::nullopt;
    }
  } else {
    return c10::nullopt;
  }
}

// 拒绝除数等于 1 的情况。这样的情况被视为 Mods。
if (divisor->isConstant() && immediateEquals(divisor, 1)) {
    // 如果条件不满足，返回空的optional对象
    return c10::nullopt;
  }

  // 如果scalar为空，则使用immLike函数创建一个值为1的imm变量
  if (!scalar) {
    scalar = immLike(multiplier, 1);
  }

  // 返回使用ModRound函数计算得到的结果
  return ModRound(scalar, denom, divisor, mod_divisor);
}

// 结束了一个代码块，可能是函数或类的结尾

// 在多项式中搜索可以合并的项
// (1) Round + Mod 模式：(x/y) * y + x % y => RoundOff(x,y) + Mod(x, y) => x
// (2) Mod round + Mod 模式：(x/y % z)*y + x%y => ModRound(x, y, z) + Mod(x,
// y) => x % (y*z)
static ExprPtr simplifyRoundModPattern(PolynomialPtr poly) {
  // 初始化存储不同类型项的向量
  std::vector<TermPtr> rounds;
  std::vector<TermPtr> mods;
  std::vector<TermPtr> mod_rounds;
  std::vector<TermPtr> others;

  // 将多项式中的项分组为 Mod、ModRound 和 RoundOff 操作，以便检查
  for (const auto& c : poly->variables()) {
    if (c->variables().size() > 1) {
      // 判断是否为 ModRound
      if (auto a = isModRound(c)) {
        mod_rounds.push_back(c);
      } else {
        others.push_back(c);
      }
      continue;
    }

    ExprPtr e = c->variables()[0];

    // 判断是否为 RoundOff
    if (to<RoundOff>(e)) {
      rounds.push_back(c);
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    } else if (e->expr_type() == IRNodeType::kMod) {
      // 判断是否为 ModRound
      if (auto a = isModRound(c)) {
        mod_rounds.push_back(c);
      } else {
        mods.push_back(c);
      }
    } else {
      others.push_back(c);
    }
  }

  // 如果没有 RoundOff 或 ModRound，或者没有 Mod，无法继续
  if ((rounds.empty() && mod_rounds.empty()) || mods.empty()) {
    return nullptr;
  }

  HashProvider& hasher = poly->hasher();
  bool didAnything = false;
  std::vector<TermPtr> mods_merged;
  bool repeat = true;
  // 反复合并项，直到没有 Mod 或无法进一步合并
  while (!mods.empty() && repeat) {
    repeat = false;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    } // end of for-loop

    // 将新生成的 Mods 添加到下一轮迭代的合并机会中
    if (!mods_merged.empty()) {
      mods.insert(mods.end(), mods_merged.begin(), mods_merged.end());
      mods_merged.clear();
    }

  } // end of while-loop

  // 如果没有进行任何更改，直接退出
  if (!didAnything) {
    return nullptr;
  }

  // 保留剩余的 ModRound 和 RoundOff
  if (!mod_rounds.empty()) {
    others.insert(others.end(), mod_rounds.begin(), mod_rounds.end());
  }

  if (!rounds.empty()) {
    others.insert(others.end(), rounds.begin(), rounds.end());
  }

  // 创建一个新的多项式结构
  return alloc<Polynomial>(hasher, poly->scalar(), others);
}

// 通过标量组件的最大公约数（GCD）对项进行简单因式分解
TermPtr PolynomialBase::factorizePolynomial(PolynomialPtr poly) {
  ExprPtr scalar = poly->scalar();
  const std::vector<TermPtr>& variables = poly->variables();

  // 计算项的最大公约数（GCD）
  ExprPtr GCD = polyGCD(poly);

  // 如果没有最大公约数（GCD），意味着是0或1，无法进行因式分解
  if (!GCD) {
    return nullptr;
  }

  // 创建新的多项式项
  std::vector<TermPtr> newPolyTerms;
  newPolyTerms.reserve(variables.size());
  for (const auto& t : variables) {
    // 使用标量除以 GCD 创建新的项
    // 此处应有更多代码，但在提供的示例中被截断了
    // 将新项添加到新多项式的末尾
    newPolyTerms.push_back(alloc<Term>(
        poly->hasher(),                           // 使用原多项式的哈希器创建新项
        evaluateOp(alloc<Div>(t->scalar(), GCD)), // 计算 t->scalar() 与 GCD 的商，并作为新项的系数
        t->variables()));                         // 继承 t 的变量列表作为新项的变量

  }

  // 创建新多项式对象，并使用新项列表和标量值的 GCD 作为参数
  PolynomialPtr newPoly = alloc<Polynomial>(
      poly->hasher(),                           // 使用原多项式的哈希器创建新多项式
      evaluateOp(alloc<Div>(scalar, GCD)),       // 计算标量值 scalar 与 GCD 的商，并作为新多项式的常数项系数
      newPolyTerms);                             // 使用新项列表初始化新多项式的项集合

  // 创建并返回一个新的项，使用原多项式的哈希器、GCD 作为系数、和新创建的多项式作为变量
  return alloc<Term>(poly->hasher(), GCD, newPoly);
}

ExprPtr TermExpander::mutate(PolynomialPtr v) {
    // 如果多项式中不包含变量，则直接返回其标量部分
    if (v->variables().empty()) {
        return v->scalar();
    }

    // 如果能够对多项式进行模式化简化，则进行简化后继续展开
    if (ExprPtr simplified = simplifyRoundModPattern(v)) {
        return simplified->accept_mutator(this);
    }

    // 如果能够对多项式进行因式分解，则进行因式分解后继续展开
    if (ExprPtr factorized = factorizePolynomial(v)) {
        return factorized->accept_mutator(this);
    }

    // 存储需要加法合并的项和需要减法合并的项
    std::vector<TermPtr> addTerms;
    std::vector<TermPtr> subTerms;

    // 获取多项式中的变量列表，并进行排序
    auto vars = v->variables();
    std::unordered_map<ExprPtr, std::string> str_repr_cache;
    std::sort(vars.begin(), vars.end(), [&](ExprPtr a, ExprPtr b) {
        if (!str_repr_cache.count(a)) {
            str_repr_cache[a] = std::to_string(a);
        }
        if (!str_repr_cache.count(b)) {
            str_repr_cache[b] = std::to_string(b);
        }
        return str_repr_cache.at(a) < str_repr_cache.at(b);
    });

    // 将变量分为需要加的项和需要减的项
    for (const auto& node : vars) {
        if (immediateIsNegative(node->scalar())) {
            subTerms.push_back(node);
        } else if (!immediateEquals(node->scalar(), 0)) {
            addTerms.push_back(node);
        }
        // 跳过标量为零的项
    }

    // 最后构造的节点
    ExprPtr lastNode{nullptr};

    // 处理需要加的项
    for (const auto& node : addTerms) {
        ExprPtr simpleNode = node->accept_mutator(this);

        if (lastNode == nullptr) {
            lastNode = simpleNode;
            continue;
        }

        if (isMultilanePrimitive(simpleNode)) {
            auto ret = combineMultilane<Add>(lastNode, simpleNode);
            if (ret) {
                // 首先简化结果，然后展开
                lastNode = ret->accept_mutator(simplifier_);
                lastNode = lastNode->accept_mutator(this);
                continue;
            }
        }

        lastNode = alloc<Add>(lastNode, simpleNode);
    }

    // 如果没有加法项，则标量应该放在第一位
    // 例如：1 - x.
    bool scalarWritten = false;
    if (lastNode == nullptr) {
        auto scalarNode = v->scalar()->accept_mutator(simplifier_);

        if (!immediateEquals(scalarNode, 0)) {
            lastNode = scalarNode;
            scalarWritten = true;
        }
    }

    // 处理需要减的项
    for (const auto& node : subTerms) {
        // 如果是第一个节点，且标量为0，则仍然可以是第一个节点
        if (lastNode == nullptr) {
            lastNode = node->accept_mutator(this);
            continue;
        }

        // 反转术语以使其为正数，因为我们将要减去它
        ExprPtr negated =
            evaluateOp(alloc<Mul>(immLike(node->scalar(), -1), node->scalar()));
        TermPtr newRHS = alloc<Term>(node->hasher(), negated, node->variables());
        lastNode = alloc<Sub>(lastNode, newRHS->accept_mutator(this));
    }

    // 如果已经写入了标量或者多项式标量为0，则直接返回
    if (scalarWritten || immediateEquals(v->scalar(), 0)) {
        if (!lastNode) {
            return immLike(v, 0);
        }
        return lastNode;
    }

    // 如果多项式标量为负数，则对其进行取反并相减
    if (immediateIsNegative(v->scalar())) {
        ExprPtr negated =
            evaluateOp(alloc<Mul>(immLike(lastNode, -1), v->scalar()));
    # 如果条件成立，为最后一个节点分配一个 Sub 对象，用于存储 evaluateOp(negated) 的结果
    lastNode = alloc<Sub>(lastNode, evaluateOp(negated));
  } else {
    # 如果条件不成立，确保不会进行标量类型的强制转换。
    # NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (v->scalar()->dtype() != lastNode->dtype()) {
      # 如果 v 的标量类型与 lastNode 的类型不同，为最后一个节点分配一个 Add 对象，
      # 其中包含一个 Cast 对象，用于将 v 的标量类型转换为 lastNode 的类型，然后再进行 evaluateOp
      lastNode = alloc<Add>(
          lastNode, evaluateOp(alloc<Cast>(lastNode->dtype(), v->scalar())));
    } else {
      # 如果 v 的标量类型与 lastNode 的类型相同，直接为最后一个节点分配一个 Add 对象，包含 v 的标量
      lastNode = alloc<Add>(lastNode, v->scalar());
    }
  }

  # 返回最后一个节点，可能是 Sub 或 Add 对象
  return lastNode;
}

// 实现 TermExpander 类中的 mutate 方法，用于处理 MaxTermPtr 类型的对象
ExprPtr TermExpander::mutate(MaxTermPtr v) {
  // 获取 MaxTermPtr 对象中的变量列表
  auto& variables = v->variables();
  // 如果变量列表为空
  if (variables.empty()) {
    // 如果 MaxTermPtr 对象不是标量，则抛出逻辑错误异常
    if (!v->scalar()) {
      throw std::logic_error("empty maxterm op");
    }
    // 返回 MaxTermPtr 对象的标量值
    return v->scalar();
  }
  // 声明一个表达式指针 max
  ExprPtr max;
  // 如果 MaxTermPtr 对象是标量
  if (v->scalar()) {
    // 创建一个 Max 对象，其中包含第一个变量、标量值和 NaN 传播标志
    max = alloc<Max>(variables[0], v->scalar(), v->propagate_nans());
  } else {
    // 否则直接将第一个变量赋给 max
    max = variables[0];
  }
  // 遍历变量列表中的其他变量，依次与 max 进行 Max 操作
  for (size_t i = 1; i < variables.size(); i++) {
    max = alloc<Max>(max, variables[i], v->propagate_nans());
  }
  // 对 max 执行接受变异器操作，并返回结果
  return max->accept_mutator(this);
}

// 实现 TermExpander 类中的 mutate 方法，用于处理 MinTermPtr 类型的对象
ExprPtr TermExpander::mutate(MinTermPtr v) {
  // 获取 MinTermPtr 对象中的变量列表
  auto& variables = v->variables();
  // 如果变量列表为空
  if (variables.empty()) {
    // 如果 MinTermPtr 对象不是标量，则抛出逻辑错误异常
    if (!v->scalar()) {
      throw std::logic_error("empty minterm op");
    }
    // 返回 MinTermPtr 对象的标量值
    return v->scalar();
  }
  // 声明一个表达式指针 min
  ExprPtr min;
  // 如果 MinTermPtr 对象是标量
  if (v->scalar()) {
    // 创建一个 Min 对象，其中包含第一个变量、标量值和 NaN 传播标志
    min = alloc<Min>(variables[0], v->scalar(), v->propagate_nans());
  } else {
    // 否则直接将第一个变量赋给 min
    min = variables[0];
  }
  // 遍历变量列表中的其他变量，依次与 min 进行 Min 操作
  for (size_t i = 1; i < variables.size(); i++) {
    min = alloc<Min>(min, variables[i], v->propagate_nans());
  }
  // 对 min 执行接受变异器操作，并返回结果
  return min->accept_mutator(this);
}

// 实现 TermExpander 类中的 mutate 方法，用于处理 RoundOffPtr 类型的对象
// 将 RoundOff(x, y) 扩展为 Term(1, Div(x, y), y)，以便后续扩展为 Mul(Div(x, y), y)
ExprPtr TermExpander::mutate(RoundOffPtr v) {
  // 创建一个 Term 对象，包含 hasher、立即值为 1、Div(x, y) 和 y
  TermPtr term = alloc<Term>(
      simplifier_->hasher(),
      immLike(v, 1),
      alloc<Div>(v->lhs(), v->rhs()),
      v->rhs());
  // 对 term 执行接受变异器操作，并返回结果
  return term->accept_mutator(this);
}

// 计算缓冲区的扁平尺寸，即所有维度的乘积
ExprPtr buf_flat_size(BufPtr v) {
  // 获取缓冲区的维度列表
  std::vector<ExprPtr> dims = v->dims();
  // 如果维度列表为空，返回立即数 1
  if (dims.empty()) {
    return alloc<LongImm>(1);
  }
  // 初始化扁平尺寸为第一个维度的立即数值
  ExprPtr flattened = immLike(dims[0], 1);
  // 遍历维度列表中的每个维度，计算它们的乘积
  for (auto& dim : dims) {
    flattened = alloc<Mul>(flattened, dim);
  }
  // 简化扁平尺寸表达式
  flattened = IRSimplifier::simplify(flattened);
  // 返回简化后的扁平尺寸表达式
  return flattened;
}

// 实现 TermExpander 类中的 mutate 方法，用于处理 AllocatePtr 类型的对象
StmtPtr TermExpander::mutate(AllocatePtr v) {
  // 获取 AllocatePtr 对象中的缓冲区
  BufPtr buf = v->buf();
  // 将缓冲区对象应用变异器，获取新的缓冲区对象 buf_new
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  // 断言新的缓冲区对象不为空
  TORCH_INTERNAL_ASSERT(
      buf_new,
      buildErrorMessage("TermExpander mutation produced null for Buf."));
  // 计算新缓冲区的扁平尺寸
  ExprPtr flattened = buf_flat_size(buf_new);

  // 如果扁平尺寸为常数且为零，将缓冲区标记为已消除分配，并返回空指针
  if (flattened->isConstant() && immediateEquals(flattened, 0)) {
    eliminated_allocations_.insert(buf_new->base_handle());
    return nullptr;
  }

  // 如果原始缓冲区与新缓冲区不同，更新 AllocatePtr 对象中的缓冲区
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 返回更新后的 AllocatePtr 对象
  return v;
}

// 实现 TermExpander 类中的 mutate 方法，用于处理 FreePtr 类型的对象
StmtPtr TermExpander::mutate(FreePtr v) {
  // 获取 FreePtr 对象中的缓冲区
  BufPtr buf = v->buf();
  // 将缓冲区对象应用变异器，获取新的缓冲区对象 buf_new
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  // 断言新的缓冲区对象不为空
  TORCH_INTERNAL_ASSERT(
      buf_new,
      buildErrorMessage("TermExpander mutation produced null for Buf."));

  // 如果新缓冲区对象的基础句柄已被标记为已消除分配，从已消除分配集合中移除该句柄并返回空指针
  if (eliminated_allocations_.count(buf_new->base_handle())) {
    eliminated_allocations_.erase(buf_new->base_handle());
    return nullptr;
  }

  // 如果原始缓冲区与新缓冲区不同，更新 FreePtr 对象中的缓冲区
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 返回更新后的 FreePtr 对象
  return v;
}
    v->set_buf(buf_new);

设置指针变量 `v` 的 `buf` 成员为 `buf_new`。


  }

返回变量 `v`，函数结束。
// Combines adjacent Cond nodes with identical conditions.
BlockPtr TermExpander::fuseConditions(BlockPtr v) {
  std::vector<StmtPtr> stmts; // 用于存储处理后的语句列表
  bool did_anything = false; // 标记是否进行了任何操作
  CondPtr prev_cond = nullptr; // 用于存储前一个条件节点的指针

  for (const auto& s : *v) { // 遍历传入的块指针v中的每个语句
    CondPtr cond = to<Cond>(s); // 尝试将当前语句转换为条件节点

    if (!cond) { // 如果当前语句不是条件节点
      prev_cond = nullptr; // 清空前一个条件节点指针
      stmts.push_back(s); // 将当前语句直接添加到结果列表中
      continue; // 继续处理下一个语句
    }

    // 如果前一个语句是条件节点且条件相同，则进行条件合并
    if (!prev_cond ||
        hasher_.hash(prev_cond->condition()) !=
            hasher_.hash(cond->condition())) {
      prev_cond = cond; // 更新前一个条件节点指针为当前条件节点
      stmts.push_back(s); // 将当前语句添加到结果列表中
      continue; // 继续处理下一个语句
    }

    // 合并两个条件节点，将第二个条件节点的体部分追加到第一个条件节点的体部分
    BlockPtr true_block = alloc<Block>(std::vector<StmtPtr>({}));
    BlockPtr false_block = alloc<Block>(std::vector<StmtPtr>({}));

    if (prev_cond->true_stmt()) {
      true_block->splice(true_block->end(), prev_cond->true_stmt());
    }

    if (cond->true_stmt()) {
      true_block->splice(true_block->end(), cond->true_stmt());
    }

    if (prev_cond->false_stmt()) {
      false_block->splice(false_block->end(), prev_cond->false_stmt());
    }

    if (cond->false_stmt()) {
      false_block->splice(false_block->end(), cond->false_stmt());
    }

    // 如果条件体为空，则置为nullptr，避免展开此条件节点
    if (true_block->empty()) {
      true_block = nullptr;
    }

    if (false_block->empty()) {
      false_block = nullptr;
    }

    // 克隆新的条件节点，并使用新的体部分，然后接受变异器进行后续处理
    StmtPtr new_cond = prev_cond->cloneWithNewBodies(true_block, false_block)
                           ->accept_mutator(this);
    prev_cond = to<Cond>(new_cond); // 更新前一个条件节点为新的条件节点

    // 移除最后一个语句，因为已经用新的条件节点替代了前两个条件节点
    stmts.pop_back();
    stmts.push_back(new_cond); // 将新的条件节点添加到结果列表中
    did_anything = true; // 标记已经进行了合并操作
  }

  if (!did_anything) { // 如果没有进行任何合并操作，则直接返回原始块指针v
    return v;
  }

  // 清理父节点关系
  for (const auto& s : stmts) {
    if (s->get_parent() == v) {
      v->remove_stmt(s); // 移除块v中不再需要的语句
    }
  }

  return alloc<Block>(stmts); // 返回合并后的新块指针
}

// 合并同步线程节点，只保留第一个顶层块
StmtPtr TermExpander::fuseSyncThreads(BlockPtr block) {
  bool first = block->get_parent() == nullptr; // 判断是否为顶层块
  SyncThreadsPtr last = nullptr; // 用于存储上一个同步线程节点
  std::vector<StmtPtr> stmts; // 用于存储处理后的语句列表
  bool did_anything = false; // 标记是否进行了任何操作

  for (const auto& s : *block) { // 遍历传入块中的每个语句
    SyncThreadsPtr sync = to<SyncThreads>(s); // 尝试将当前语句转换为同步线程节点

    if (!sync) { // 如果当前语句不是同步线程节点
      first = false; // 更新first标志
      last = nullptr; // 清空上一个同步线程节点指针
      stmts.push_back(s); // 将当前语句直接添加到结果列表中
      continue; // 继续处理下一个语句
    }

    if (first || last) { // 如果是第一个或上一个不为空的同步线程节点
      did_anything = true; // 标记已经进行了操作
      continue; // 直接继续处理下一个语句
    }

    last = sync; // 更新上一个同步线程节点指针为当前同步线程节点
    first = false; // 更新first标志
    stmts.push_back(s); // 将当前语句添加到结果列表中
  }

  if (last) { // 如果最后一个同步线程节点不为空
    stmts.pop_back(); // 移除最后一个语句
    did_anything = true; // 标记已经进行了操作
  }

  if (!did_anything) { // 如果没有进行任何操作，则直接返回原始块指针block
    return block;
  }

  // 清理父节点关系
  for (const auto& s : stmts) {
    if (s->get_parent() == block) {
      block->remove_stmt(s); // 移除块block中不再需要的语句
    }
  }

  return alloc<Block>(std::vector<StmtPtr>({stmts})); // 返回处理后的新块指针
}
StmtPtr TermExpander::mutate(BlockPtr v) {
  // 调用父类的mutate方法对给定的块进行多项式展开
  StmtPtr new_stmt = PolynomialBase::mutate(v);
  // 将返回的语句转换为块类型
  BlockPtr new_block = to<Block>(new_stmt);
  if (!new_block) {
    // 如果转换失败，返回原始的语句
    return new_stmt;
  }

  // 调用fuseConditions函数尝试合并条件，如果无法合并则返回原始的块
  new_block = fuseConditions(new_block);
  /// 同样地，调用fuseSyncThreads函数尝试合并同步线程
  return fuseSyncThreads(new_block);
}

// SimplifierUnderContext
//
// 这个函数记录了for循环语句中索引变量的边界信息（范围信息）。
// 边界信息将在后续简化表达式时使用，用于处理索引变量。
StmtPtr SimplifierUnderContext::mutate(ForPtr v) {
  // 提取for循环的索引变量、起始值、终止值和循环体
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  // 使用当前mutator处理新的索引变量表达式
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body;

  // 在处理for循环语句之前保存边界信息
  //
  // 同一变量可能出现在嵌套在if语句中的for语句中，我们需要在for语句后恢复其边界信息。
  //
  // 示例：
  // if (i>=0 && i<5) {
  //   for (i=0; i<3; i++){
  //     A[i] = ...
  //   }
  //   x = (i+20) / 5;
  //}
  // 在if语句内部，i的范围是[0, 5)，如果我们能在for语句后恢复这个边界信息，
  // 我们就可以使用它来简化赋值语句x = (i+20)/5为x = 4。
  bool has_bounds = false;
  analysis::Bound bound_old;
  VarPtr var_key = to<Var>(var);
  auto got = var_bound_info_.find(var_key);
  if (got != var_bound_info_.end()) {
    has_bounds = true;
    bound_old = got->second;
  }
  // 设置索引变量的边界信息
  const analysis::Bound bound_new(start_new, stop_new);
  var_bound_info_[var_key] = bound_new;

  // 计算迭代次数，并使用当前mutator处理结果
  ExprPtr iters = alloc<Sub>(stop_new, start_new);
  iters = iters->accept_mutator(this);
  // 如果循环选项是默认的，并且迭代次数是常数
  if (loop_options.isDefault() && iters->isConstant()) {
    if (immediateEquals(iters, 0)) {
      // 如果迭代次数为0，返回空块
      return alloc<Block>(std::vector<StmtPtr>({}));
    } else if (immediateEquals(iters, 1)) {
      // 替换循环体中的变量，并使用当前mutator处理结果
      body_new = Substitute(body, {{var_new, start_new}});
      body_new = body_new->accept_mutator(this);

      // 擦除索引变量的边界信息或者恢复旧的边界信息
      if (has_bounds) {
        var_bound_info_[var_key] = bound_old;
      } else {
        var_bound_info_.erase(var_key);
      }

      return body_new;
    }
  }

  // 使用当前mutator处理新的循环体
  body_new = body_new->accept_mutator(this);

  // 擦除索引变量的边界信息或者恢复旧的边界信息
  if (has_bounds) {
    var_bound_info_[var_key] = bound_old;
  } else {
    var_bound_info_.erase(var_key);
  }

  if (!body_new) {
    // 如果处理后的循环体为空，返回空块
    return alloc<Block>(std::vector<StmtPtr>({}));
  }

  if (auto block = to<Block>(body_new)) {
    if (block->nstmts() == 0) {
      // 如果处理后的块没有语句，返回空块
      return alloc<Block>(std::vector<StmtPtr>({}));
    }
    // 如果块不为空，继续处理
    // 如果当前基本块只包含一个语句
    if (block->nstmts() == 1) {
      // 如果该语句是一个条件语句（if-stmt），尝试将分支操作移到循环外部
      if (auto cond = to<Cond>(block->front())) {
        // 调用函数处理条件重排序，返回重排序后的语句
        StmtPtr reordered = handleForCondReordering(v, cond);
        // 如果成功重排序
        if (reordered) {
          // 对重排序后的语句进行变异操作，并返回变异后的结果
          return reordered->accept_mutator(this);
        }
      }
    }

  // 检查变量是否需要更新
  if (var != var_new) {
    // 更新变量
    v->set_var(var_new);
  }
  // 检查起始值是否需要更新
  if (start != start_new) {
    // 更新起始值
    v->set_start(start_new);
  }
  // 检查终止条件是否需要更新
  if (stop != stop_new) {
    // 更新终止条件
    v->set_stop(stop_new);
  }
  // 检查循环体是否需要更新
  if (body != body_new) {
    // 更新循环体
    v->set_body(body_new);
  }

  // 返回更新后的循环语句
  return v;
}

// 结束当前的函数定义
// 通过分配法则简化除法操作，适用于以下情况：
// 1) (i + x) / n => x/n，如果：
//   a) n 是正整数常数；
//   b) i 是 for 循环语句的索引变量，并且其范围是 [0, n) 的子集；
//   c) x 是常数，并且 i 的范围的结束值小于 n - x%n；
//   TODO: 从要求中去除 d)，因为简化公式仍然适用于 x 是负整数的情况。在整数除法中，除法的结果
//   被转换为整数，使用 `floor` 函数返回不大于 X 的最大整数。例如，-1/6 返回 -1。但当前，
//   Pytorch 和 NNC 都执行错误的整数除法：(-1)/6 = 0。根据当前的整数除法实现，x 必须是非负数。d) x 是非负数
//
// 2) (i + j*n) / n => j，如果：
//   a) n 是正整数常数；
//   b) i 是 for 循环语句的索引变量，并且其范围是 [0, n) 的子集；
//   c) j 是整数变量；
//   TODO: 从要求中去除 d)，因为简化公式仍然适用于 j 是负整数的情况。在整数除法中，除法的结果
//   被转换为整数，使用 `floor` 函数返回不大于 X 的最大整数。例如，-1/6 返回 -1。但当前，
//   Pytorch 和 NNC 都执行错误的整数除法：(-1)/6 = 0。根据当前的整数除法实现，x 必须是非负数。d) j 是非负数
static ExprPtr distributeDiv(
    ExprPtr lhs,
    ExprPtr rhs,
    VarBoundInfo var_bound_info) {
  if (!lhs || !rhs) {
    return nullptr;
  }
  // 如果不是整数除法则返回空指针
  if (lhs->dtype().is_floating_point() || rhs->dtype().is_floating_point()) {
    return nullptr;
  }

  // 确定 n：一个正整数常数
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (!rhsScalar) {
    return nullptr;
  }
  // 检查是否 rhsScalar > 0
  ExprPtr check_n_value = IRSimplifier::simplify(
      alloc<CompareSelect>(rhsScalar, immLike(rhsScalar, 0), kGT));
  if (!immediateEquals(check_n_value, 1)) {
    return nullptr;
  }

  auto lhsAdd = to<Add>(lhs);
  if (!lhsAdd) {
    return nullptr;
  }
  ExprPtr lhsAdd1 = lhsAdd->lhs();
  ExprPtr lhsAdd2 = lhsAdd->rhs();

  // 确定索引变量 'i'
  VarPtr var_key = to<Var>(lhsAdd1);
  ExprPtr main = lhsAdd2;
  if (var_key == nullptr) {
    var_key = to<Var>(lhsAdd2);
    main = lhsAdd1;
  }

  if (var_key == nullptr) {
    return nullptr;
  }

  // 查找变量的边界信息
  auto got = var_bound_info.find(var_key);
  if (got == var_bound_info.end()) {
    // 返回空指针，表示未找到符合条件的表达式
    return nullptr;
  }

  // 检查 'i' 的范围
  auto start = got->second.start;
  // 开放上界，即 end 比范围内最大值大一
  auto end = got->second.end;
  // 简化并检查起始位置的条件表达式
  ExprPtr check_start = IRSimplifier::simplify(
      alloc<CompareSelect>(start, immLike(start, 0), kGE));
  // 简化并检查结束位置的条件表达式
  ExprPtr check_end =
      IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
  // 如果起始或结束条件不是常量，或者不满足条件 (check_start == 1) 或 (check_end == 1)，返回空指针
  if (!check_start->isConstant() || !check_end->isConstant() ||
      !immediateEquals(check_start, 1) || !immediateEquals(check_end, 1)) {
    return nullptr;
  }

  // 简化并返回主表达式与右侧标量的除法结果
  ExprPtr ret = IRSimplifier::simplify(alloc<Div>(main, rhsScalar));

  // 简化类型1的表达式: '(i+x)/n' => 'x/n'
  ExprPtr sign_check =
      IRSimplifier::simplify(alloc<CompareSelect>(main, immLike(main, 0), kGE));
  ExprPtr main_mod = IRSimplifier::simplify(alloc<Mod>(main, rhsScalar));
  ExprPtr mod_check = IRSimplifier::simplify(
      alloc<CompareSelect>(alloc<Add>(main_mod, end), rhsScalar, kLE));
  // 如果主表达式大于等于零且 (main_mod + end) 小于等于 rhsScalar，则返回 ret
  if (sign_check->isConstant() && immediateEquals(sign_check, 1) &&
      mod_check->isConstant() && immediateEquals(mod_check, 1)) {
    return ret;
  }

  // 简化类型2的表达式: '(i+j*n)/n' => 'j'
  auto ret_var = to<Var>(ret);
  // FIXME: 允许任何整数类型。
  if (ret_var && ret_var->dtype() == kInt) {
    // 获取变量 j 的范围信息
    auto got = var_bound_info.find(ret_var);
    // 如果未找到 j 的范围信息，则返回空指针
    if (got == var_bound_info.end()) {
      return nullptr;
    }

    // 检查 j 是否为非负数
    sign_check = IRSimplifier::simplify(alloc<CompareSelect>(
        got->second.start, immLike(got->second.start, 0), kGE));
    // 如果 j 是非负数，则返回 ret_var
    if (sign_check->isConstant() && immediateEquals(sign_check, 1)) {
      return ret_var;
    }
  }

  // 默认情况下返回空指针
  return nullptr;
// 简化使用分配律的模运算，适用于以下情况：
// 1) (i + x) % n => i + x%n，如果
//   a) n 是正整数常量；
//   b) i 是 for 循环语句的索引变量，并且其范围是 [0, n) 的子集；
//   c) x 是常量，并且 i 的范围的结束值小于 n - x%n；
//   TODO: 移除 d) 的要求，因为简化公式仍然适用于 x 是负整数的情况。在整数除法中，结果使用 `floor` 函数转换为整数，返回不大于 X 的最大整数。
//   例如，-1/6 返回 -1。但是当前，Pytorch 和 NNC 都执行不正确的整数除法：(-1)/6 = 0。根据当前的整数除法实现，x 必须为非负数。d) x 不是负数。
//
// 2) (i + j*n) % n => i，如果
//   a) n 是正整数常量；
//   b) i 是 for 循环语句的索引变量，并且其范围是 [0, n) 的子集；
//   c) j 是整数变量；
//   TODO: 移除 d) 的要求，因为简化公式仍然适用于 j 是负整数的情况。在整数除法中，结果使用 `floor` 函数转换为整数，返回不大于 X 的最大整数。
//   例如，-1/6 返回 -1。但是当前，Pytorch 和 NNC 都执行不正确的整数除法：(-1)/6 = 0。根据当前的整数除法实现，j 必须为非负数。d) j 不是负数。
static ExprPtr distributeMod(
    ExprPtr lhs,                         // 左侧表达式
    ExprPtr rhs,                         // 右侧表达式
    VarBoundInfo var_bound_info) {       // 变量边界信息映射

  if (!lhs || !rhs) {
    return nullptr;                     // 如果左右表达式为空，返回空指针
  }

  // 如果左右表达式为浮点数类型，则返回空指针
  if (lhs->dtype().is_floating_point() || rhs->dtype().is_floating_point()) {
    return nullptr;
  }

  // 确定 n: 正整数常量
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;  // 右侧表达式是否为常量
  if (!rhsScalar) {
    return nullptr;
  }

  // 确认 n 的值为正常数
  ExprPtr check_n_value = IRSimplifier::simplify(
      alloc<CompareSelect>(rhsScalar, immLike(rhsScalar, 0), kGT));
  if (!immediateEquals(check_n_value, 1)) {
    return nullptr;
  }

  auto lhsAdd = to<Add>(lhs);
  if (!lhsAdd) {
    return nullptr;
  }

  // 检查左侧表达式是否为加法表达式，同时确保左侧加法表达式和右侧标量不为空
  if (!lhsAdd || !rhsScalar) {
    return nullptr;
  }

  ExprPtr lhsAdd1 = lhsAdd->lhs();  // 左侧加法表达式的左子表达式
  ExprPtr lhsAdd2 = lhsAdd->rhs();  // 左侧加法表达式的右子表达式

  // 确定索引变量 'i'
  VarPtr var_key = to<Var>(lhsAdd1);
  ExprPtr main = lhsAdd2;
  if (var_key == nullptr) {
    var_key = to<Var>(lhsAdd2);
    main = lhsAdd1;
  }
  if (var_key == nullptr) {
    return nullptr;
  }

  // 查找索引变量在变量边界信息映射中的记录
  auto got = var_bound_info.find(var_key);
  if (got == var_bound_info.end()) {
    // 如果未找到索引变量的边界信息，返回空指针
    // 返回空指针，表示没有找到符合条件的表达式
    return nullptr;
  }

  // 检查变量 'i' 的范围边界
  auto start = got->second.start;
  // 开区间的上界，即 'end' 比范围内最大值多一
  auto end = got->second.end;
  // 简化检查起始点是否为常数表达式
  ExprPtr check_start = IRSimplifier::simplify(
      alloc<CompareSelect>(start, immLike(start, 0), kGE));
  // 简化检查结束点是否为常数表达式
  ExprPtr check_end =
      IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
  // 如果起始点或结束点不是常数，或者不满足条件，则返回空指针
  if (!check_start->isConstant() || !check_end->isConstant() ||
      !immediateEquals(check_start, 1) || !immediateEquals(check_end, 1)) {
    return nullptr;
  }

  // 简化类型 1) 表达式: '(i+x)%n' => 'i+x%n'
  ExprPtr sign_check =
      IRSimplifier::simplify(alloc<CompareSelect>(main, immLike(main, 0), kGE));
  // 简化主要模运算表达式
  ExprPtr main_mod = IRSimplifier::simplify(alloc<Mod>(main, rhsScalar));
  // 简化模运算后的结果检查
  ExprPtr mod_check = IRSimplifier::simplify(
      alloc<CompareSelect>(alloc<Add>(main_mod, end), rhsScalar, kLE));
  // 如果满足模式化简条件，则返回修正的表达式
  if (sign_check->isConstant() && immediateEquals(sign_check, 1) &&
      mod_check->isConstant() && immediateEquals(mod_check, 1)) {
    return alloc<Add>(var_key, main_mod);
  }

  // 简化类型 2) 表达式: '(i+j*n)%n' => 'i'
  ExprPtr main_div = IRSimplifier::simplify(alloc<Div>(main, rhsScalar));
  auto j_var = to<Var>(main_div);
  // FIXME: 允许任何整数类型。
  if (j_var && j_var->dtype() == kInt) {
    // 检索变量 j 的范围信息
    auto got = var_bound_info.find(j_var);
    // 如果未找到 j 的范围信息，则返回空指针
    if (got == var_bound_info.end()) {
      return nullptr;
    }

    // 检查 j 是否为非负数
    sign_check = IRSimplifier::simplify(alloc<CompareSelect>(
        got->second.start, immLike(got->second.start, 0), kGE));
    // 如果 j 是非负数，则返回变量键 var_key
    if (sign_check->isConstant() && immediateEquals(sign_check, 1)) {
      return var_key;
    }
  }

  // 默认返回空指针，表示未满足任何条件
  return nullptr;
ExprPtr SimplifierUnderContext::mutate(DivPtr v) {
  // 提取除法表达式的左右操作数
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();

  // 创建一个字符串流对象，用于日志输出
  std::ostringstream oss;

  // 尝试应用分配除法优化，如果成功，返回优化后的表达式
  if (auto ret = distributeDiv(lhs, rhs, var_bound_info_)) {
    GRAPH_DEBUG("SimplifierUnderContext: ", *v, " => ", *ret);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return ret->accept_mutator(this);
  }

  // 对于形如 i / N 的表达式，如果 i 的值范围是 [0, N) 的子集，并且 N 是整数常量，返回 0
  auto lhsVar = to<Var>(lhs);
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (lhsVar && rhsScalar && !rhsScalar->dtype().is_floating_point()) {
    auto got = var_bound_info_.find(lhsVar);
    if (got != var_bound_info_.end()) {
      auto start = got->second.start;
      auto end = got->second.end;
      // 简化起始条件
      ExprPtr check_start = IRSimplifier::simplify(
          alloc<CompareSelect>(start, immLike(start, 0), kGE));
      // 简化结束条件
      ExprPtr check_end =
          IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
      // 如果起始条件和结束条件都是常量且为真，则返回 0
      if (check_start->isConstant() && check_end->isConstant() &&
          immediateEquals(check_start, 1) && immediateEquals(check_end, 1)) {
        GRAPH_DEBUG(
            "SimplifierUnderContext: ", *v, " => ", *immLike(lhsVar, 0));
        return immLike(lhsVar, 0);
      }
    }
  }

  // 对左右操作数分别应用变异器进行简化
  ExprPtr lhs_new = lhs->accept_mutator(this);
  ExprPtr rhs_new = rhs->accept_mutator(this);
  // 如果左右操作数没有变化，则返回原始除法表达式
  if (lhs == lhs_new && rhs == rhs_new) {
    return v;
  }
  // 否则返回一个新的除法表达式
  return alloc<Div>(lhs_new, rhs_new);
}

ExprPtr SimplifierUnderContext::mutate(IfThenElsePtr v) {
  // 提取条件、真值和假值表达式
  ExprPtr condition = v->condition();
  ExprPtr true_val = v->true_value();
  ExprPtr false_val = v->false_value();

  // 简化条件、真值和假值表达式
  auto simplified_condition =
      IRSimplifier::simplify(condition->accept_mutator(this));
  auto simplified_true_val =
      IRSimplifier::simplify(true_val->accept_mutator(this));
  auto simplified_false_val =
      IRSimplifier::simplify(false_val->accept_mutator(this));
  
  // 如果条件表达式是常量，根据条件值返回真值或假值
  if (simplified_condition->isConstant()) {
    return immediateAs<int>(simplified_condition) ? simplified_true_val
                                                  : simplified_false_val;
  }

  // 如果条件、真值和假值都没有改变，则返回原始 IfThenElse 表达式
  bool nothing_changed = (simplified_condition == condition) &&
      (simplified_true_val == true_val) && (simplified_false_val == false_val);
  return nothing_changed
      ? v
      : alloc<IfThenElse>(
            simplified_condition, simplified_true_val, simplified_false_val);
}
ExprPtr SimplifierUnderContext::mutate(CompareSelectPtr v) {
  // 输出调试信息，显示原始的 CompareSelect 对象
  GRAPH_DEBUG("(SimplifierUnderContext) Original: ", std::to_string(v));

  // 获取 CompareSelect 对象的各个表达式成员
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr ret1 = v->ret_val1();
  ExprPtr ret2 = v->ret_val2();

  // 对每个表达式应用 IRSimplifier 进行简化
  auto simplified_lhs = IRSimplifier::simplify(lhs->accept_mutator(this));
  auto simplified_rhs = IRSimplifier::simplify(rhs->accept_mutator(this));
  auto simplified_ret1 = IRSimplifier::simplify(ret1->accept_mutator(this));
  auto simplified_ret2 = IRSimplifier::simplify(ret2->accept_mutator(this));

  // 初始化简化后的 CompareSelect 表达式
  ExprPtr simplified_cmp_select_expr = nullptr;
  // 如果所有表达式都没有改变，则直接使用原始的 CompareSelect 对象
  if ((simplified_lhs == lhs) && (simplified_rhs == rhs) &&
      (simplified_ret1 == ret1) && (simplified_ret2 == ret2)) {
    simplified_cmp_select_expr = v;
  } else {
    // 否则，根据简化后的表达式创建新的 CompareSelect 对象
    simplified_cmp_select_expr = alloc<CompareSelect>(
        simplified_lhs,
        simplified_rhs,
        simplified_ret1,
        simplified_ret2,
        v->compare_select_op(),
        v->bias());
  }

  // 输出调试信息，显示简化后的 CompareSelect 对象
  GRAPH_DEBUG(
      "(SimplifierUnderContext) after simplify: ",
      std::to_string(simplified_cmp_select_expr));

  // 分析左右表达式的边界信息
  analysis::Bound lhs_bound;
  analysis::Bound rhs_bound;
  auto lhs_has_bound = getLoopBoundInfo(simplified_lhs, &lhs_bound);
  auto rhs_has_bound = getLoopBoundInfo(simplified_rhs, &rhs_bound);
  // 如果左右表达式中有任意一个没有边界信息，则直接返回简化后的 CompareSelect 对象
  if (!lhs_has_bound || !rhs_has_bound) {
    GRAPH_DEBUG(
        "(SimplifierUnderContext) Final: ",
        std::to_string(simplified_cmp_select_expr));
    return simplified_cmp_select_expr;
  }

  // 比较左右表达式的边界，并得到比较结果
  analysis::CmpEvalResult cmp_res =
      analysis::compareBound(lhs_bound, rhs_bound, v->compare_select_op());

  // 如果比较结果是确定的，则返回简化后的 ret1 或 ret2
  // 否则根据比较结果返回简化后的 CompareSelect 对象
  auto ret_expr = (cmp_res == analysis::CmpEvalResult::True)
      ? simplified_ret1
      : ((cmp_res == analysis::CmpEvalResult::False)
             ? simplified_ret2
             : simplified_cmp_select_expr);
  // 输出调试信息，显示最终返回的表达式
  GRAPH_DEBUG("(SimplifierUnderContext) Final: ", std::to_string(ret_expr));
  return ret_expr;
}

ExprPtr SimplifierUnderContext::mutate(ModPtr v) {
  // 获取 Mod 表达式的左右操作数
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();

  std::ostringstream oss;
  // 尝试分配 Mod 表达式，如果成功则返回分配后的表达式
  if (auto ret = distributeMod(lhs, rhs, var_bound_info_)) {
    // 输出调试信息，显示 Mod 表达式经过分配后的结果
    GRAPH_DEBUG("SimplifierUnderContext: ", *v, " => ", *ret);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return ret->accept_mutator(this);
  }

  // 如果无法分配 Mod 表达式，则检查是否可以简化为 lhs，前提是 rhs 是整数常量
  // 并且 lhs 是变量
  auto lhsVar = to<Var>(lhs);
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (lhsVar && rhsScalar && !rhsScalar->dtype().is_floating_point()) {
    auto got = var_bound_info_.find(lhsVar);
    // ...
    // 此处省略了一部分代码，在上下文中继续处理
    // 检查变量是否存在于变量边界信息中
    if (got != var_bound_info_.end()) {
      // 获取变量的起始和结束表达式
      auto start = got->second.start;
      auto end = got->second.end;
      
      // 对起始位置进行简化并生成检查表达式
      ExprPtr check_start = IRSimplifier::simplify(
          alloc<CompareSelect>(start, immLike(start, 0), kGE));
      
      // 对结束位置进行简化并生成检查表达式
      ExprPtr check_end =
          IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
      
      // 检查起始和结束位置是否为常量并且为1，如果是，则进行简化处理
      if (check_start->isConstant() && check_end->isConstant() &&
          immediateEquals(check_start, 1) && immediateEquals(check_end, 1)) {
        // 输出调试信息并返回左手边变量
        GRAPH_DEBUG("SimplifierUnderContext: ", *v, " => ", *lhsVar);
        return lhsVar;
      }
    }
  
    // 对左手边表达式进行变异处理
    ExprPtr lhs_new = lhs->accept_mutator(this);
    
    // 对右手边表达式进行变异处理
    ExprPtr rhs_new = rhs->accept_mutator(this);
    
    // 如果左右手边表达式未发生变化，则直接返回原始表达式
    if (lhs == lhs_new && rhs == rhs_new) {
      return v;
    }
    
    // 返回修改后的求模操作表达式
    return alloc<Mod>(lhs_new, rhs_new);
} // 结束 SimplifierUnderContext 命名空间

// 获取循环边界信息的方法
bool SimplifierUnderContext::getLoopBoundInfo(
    const ExprPtr& expr,
    analysis::Bound* loop_bound_info) {
  // 如果表达式为空，返回 false
  if (expr == nullptr)
    return false;

  // 如果表达式是常量
  if (expr->isConstant()) {
    // 设置循环边界信息的开始和结束为表达式本身
    loop_bound_info->start = expr;
    loop_bound_info->end = expr;
    return true;
  }

  // 尝试将表达式转换为变量指针
  VarPtr var_key = to<Var>(expr);
  if (var_key == nullptr) {
    return false;
  }

  // 在变量边界信息的映射中查找该变量
  auto got = var_bound_info_.find(var_key);
  if (got == var_bound_info_.end()) {
    return false;
  }

  // 设置循环边界信息的开始为找到的起始边界
  loop_bound_info->start = got->second.start;
  // 添加一个 TODO 注释，说明需要将范围的边界信息（开放/闭合）添加到 Bound 结构体中
  // 目前 VarBoundInfo 是来自于 for 循环语句，其中范围的结束是开放的
  // 但是我们假设范围的开始和结束总是闭合的，因此我们显式地将开放的边界转换为闭合的
  //   [for-start, for-stop) => [for-start, for-stop - 1]
  loop_bound_info->end = IRSimplifier::simplify(
      alloc<Sub>(got->second.end, immLike(got->second.end, 1)));
  return true;
}

// 比较两个表达式是否相等的方法
bool exprEquals(ExprPtr A, ExprPtr B) {
  try {
    // 使用 IRSimplifier 进行简化差异表达式 A - B
    ExprPtr diff = IRSimplifier::simplify(alloc<Sub>(A, B));
    // 如果差异表达式不是常量，返回 false
    if (!diff->isConstant()) {
      return false;
    }
    // 检查差异表达式是否等于 0
    return immediateEquals(diff, 0);
  } catch (std::exception& e) {
    return false;
  }
}

// 对表达式进行简化的方法
ExprPtr IRSimplifier::simplify(ExprPtr e) {
  // 打印调试信息，显示原始表达式内容
  GRAPH_DEBUG("(Simplifier) Original: ", std::to_string(e));
  // 创建 SimplifierUnderContext 实例
  SimplifierUnderContext ctxsimplifier;
  // 使用上下文简化器对表达式进行变异
  e = e->accept_mutator(&ctxsimplifier);

  // 创建 PolynomialTransformer 实例
  PolynomialTransformer simplifier;
  // 使用多项式变换器对表达式进行变异
  e = e->accept_mutator(&simplifier);

  // 创建 TermExpander 实例，以扩展 IR 中可能存在的项
  TermExpander expander(&simplifier);
  // 使用项扩展器对表达式进行变异
  e = e->accept_mutator(&expander);
  // 检查项扩展器是否安全
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  if (!expander.check_safe()) {
    throw malformed_input("eliminated null Allocation without free");
  }

  // 打印调试信息，显示简化后的表达式内容
  GRAPH_DEBUG("(Simplifier) Simplified: ", std::to_string(e));
  return e;
}

// 对语句进行简化的方法
StmtPtr IRSimplifier::simplify(StmtPtr s) {
  // 打印调试信息，显示原始语句内容
  GRAPH_DEBUG("(Simplifier) Original: ", std::to_string(s));
  // 创建 SimplifierUnderContext 实例
  SimplifierUnderContext ctxsimplifier;
  // 使用上下文简化器对语句进行变异
  s = s->accept_mutator(&ctxsimplifier);

  // 创建 PolynomialTransformer 实例
  PolynomialTransformer simplifier;
  // 使用多项式变换器对语句进行变异
  s = s->accept_mutator(&simplifier);

  // 如果语句为空，返回空指针
  if (s == nullptr) {
    GRAPH_DEBUG("(Simplifier) Simplified: NULL");
    return nullptr;
  }

  // 创建 TermExpander 实例，以扩展 IR 中可能存在的项
  TermExpander expander(&simplifier);
  // 使用项扩展器对语句进行变异
  s = s->accept_mutator(&expander);
  // 检查项扩展器是否安全
  if (!expander.check_safe()) {
    throw malformed_input("eliminated null Allocation without free");
  }

  // 打印调试信息，显示简化后的语句内容
  GRAPH_DEBUG("(Simplifier) Simplified: ", std::to_string(s));
  return s;
}

} // namespace torch::jit::tensorexpr
```