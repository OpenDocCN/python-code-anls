# `.\pytorch\torch\csrc\jit\tensorexpr\hash_provider.cpp`

```
`
#include <torch/csrc/jit/tensorexpr/hash_provider.h>  // 引入头文件以使用HashProvider类和相关定义

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>  // 引入头文件以使用IR简化器

#include <c10/util/irange.h>  // 引入头文件以使用irange函数

namespace torch::jit::tensorexpr {  // 定义torch::jit::tensorexpr命名空间

// 比较运算符重载，用于判断SimplifierHashType对象是否相等
bool SimplifierHashType::operator==(const SimplifierHashType& other) const {
  return _h == other._h;  // 比较_hash成员变量是否相等
}

// 比较运算符重载，用于判断SimplifierHashType对象是否不相等
bool SimplifierHashType::operator!=(const SimplifierHashType& other) const {
  return _h != other._h;  // 比较_hash成员变量是否不相等
}

// 比较运算符重载，用于判断SimplifierHashType对象是否小于另一个对象
bool SimplifierHashType::operator<(const SimplifierHashType& other) const {
  return _h < other._h;  // 比较_hash成员变量是否小于另一个对象的_hash成员变量
}

// 比较运算符重载，用于判断SimplifierHashType对象与size_t类型的值是否相等
bool SimplifierHashType::operator==(const size_t other) const {
  return _h == other;  // 比较_hash成员变量是否等于给定的size_t值
}

// 比较运算符重载，用于判断SimplifierHashType对象与size_t类型的值是否不相等
bool SimplifierHashType::operator!=(const size_t other) const {
  return _h != other;  // 比较_hash成员变量是否不等于给定的size_t值
}

// HashProvider类的visit方法，处理AddPtr类型的节点
void HashProvider::visit(AddPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问AddPtr节点的左子节点
  v->rhs()->accept(this);  // 访问AddPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "+", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理SubPtr类型的节点
void HashProvider::visit(SubPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问SubPtr节点的左子节点
  v->rhs()->accept(this);  // 访问SubPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "-", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理MulPtr类型的节点
void HashProvider::visit(MulPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问MulPtr节点的左子节点
  v->rhs()->accept(this);  // 访问MulPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "*", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理DivPtr类型的节点
void HashProvider::visit(DivPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问DivPtr节点的左子节点
  v->rhs()->accept(this);  // 访问DivPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "/", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理ModPtr类型的节点
void HashProvider::visit(ModPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问ModPtr节点的左子节点
  v->rhs()->accept(this);  // 访问ModPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "%", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理RoundOffPtr类型的节点
void HashProvider::visit(RoundOffPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问RoundOffPtr节点的左子节点
  v->rhs()->accept(this);  // 访问RoundOffPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "rof", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理MaxPtr类型的节点
void HashProvider::visit(MaxPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问MaxPtr节点的左子节点
  v->rhs()->accept(this);  // 访问MaxPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "Mx", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理MinPtr类型的节点
void HashProvider::visit(MinPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问MinPtr节点的左子节点
  v->rhs()->accept(this);  // 访问MinPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "Mn", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理AndPtr类型的节点
void HashProvider::visit(AndPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问AndPtr节点的左子节点
  v->rhs()->accept(this);  // 访问AndPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "&", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理OrPtr类型的节点
void HashProvider::visit(OrPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问OrPtr节点的左子节点
  v->rhs()->accept(this);  // 访问OrPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "|", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理XorPtr类型的节点
void HashProvider::visit(XorPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问XorPtr节点的左子节点
  v->rhs()->accept(this);  // 访问XorPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "^", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理LshiftPtr类型的节点
void HashProvider::visit(LshiftPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问LshiftPtr节点的左子节点
  v->rhs()->accept(this);  // 访问LshiftPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), "<<", hashOf(v->rhs())));  // 计算哈希值并存储
}

// HashProvider类的visit方法，处理RshiftPtr类型的节点
void HashProvider::visit(RshiftPtr v) {
  CACHE_GUARD();  // 执行缓存保护宏，确保线程安全
  v->lhs()->accept(this);  // 访问RshiftPtr节点的左子节点
  v->rhs()->accept(this);  // 访问RshiftPtr节点的右子节点
  putHash(v, hash_combine(hashOf(v->lhs()), ">>", hashOf(v->rhs())));  // 计算哈希值并存储
}
void HashProvider::visit(CompareSelectPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->lhs()->accept(this);  // 访问左操作数并计算哈希值
  v->rhs()->accept(this);  // 访问右操作数并计算哈希值
  v->ret_val1()->accept(this);  // 访问第一个返回值并计算哈希值
  v->ret_val2()->accept(this);  // 访问第二个返回值并计算哈希值
  putHash(
      v,
      hash_combine(
          hashOf(v->lhs()),  // 左操作数的哈希值
          (int)v->compare_select_op(),  // 比较选择操作的哈希值（强制转换为整数）
          hashOf(v->rhs()),  // 右操作数的哈希值
          hashOf(v->ret_val1()),  // 第一个返回值的哈希值
          hashOf(v->ret_val2())));  // 第二个返回值的哈希值
}

void HashProvider::visit(CastPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->src_value()->accept(this);  // 访问源值并计算哈希值
  putHash(v, hash_combine("cast", v->dtype(), hashOf(v->src_value())));  // 计算类型转换的哈希值
}

void HashProvider::visit(VarPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  putHash(v, hash_combine("var", name_manager_.get_unique_name(v)));  // 计算变量的哈希值
}

void HashProvider::visit(RampPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->base()->accept(this);  // 访问基础值并计算哈希值
  v->stride()->accept(this);  // 访问步长并计算哈希值
  putHash(
      v,
      hash_combine("ramp", hashOf(v->base()), hashOf(v->stride()), v->lanes()));  // 计算 Ramp 对象的哈希值
}

void HashProvider::visit(LoadPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->base_handle()->accept(this);  // 访问基地址并计算哈希值
  SimplifierHashType indices_hash;
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);  // 访问索引表达式并计算哈希值
    indices_hash = hash_combine(indices_hash, hashOf(ind));  // 组合所有索引表达式的哈希值
  }
  putHash(v, hash_combine("load", hashOf(v->base_handle()), indices_hash));  // 计算 Load 对象的哈希值
}

void HashProvider::visit(StorePtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->base_handle()->accept(this);  // 访问基地址并计算哈希值
  SimplifierHashType indices_hash;
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);  // 访问索引表达式并计算哈希值
    indices_hash = hash_combine(indices_hash, hashOf(ind));  // 组合所有索引表达式的哈希值
  }
  v->value()->accept(this);  // 访问存储值并计算哈希值
  putHash(
      v,
      hash_combine(
          "store", hashOf(v->base_handle()), indices_hash, hashOf(v->value())));  // 计算 Store 对象的哈希值
}

void HashProvider::visit(BlockPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  SimplifierHashType hash;

  for (const StmtPtr& s : *v) {
    s->accept(this);  // 访问块中的每个语句并计算哈希值
    hash = hash_combine(hash, hashOf(s));  // 组合每个语句的哈希值
  }
  putHash(v, hash);  // 计算 Block 对象的哈希值
}

void HashProvider::visit(ForPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->var()->accept(this);  // 访问循环变量并计算哈希值
  v->start()->accept(this);  // 访问起始值并计算哈希值
  v->stop()->accept(this);  // 访问终止值并计算哈希值

  SimplifierHashType hash = hash_combine(
      "for", hashOf(v->var()), hashOf(v->start()), hashOf(v->stop()));  // 计算 For 循环对象的哈希值
  hash = hash_combine(hash, v->loop_options().ToString());  // 添加循环选项的哈希值
  if (v->body()) {
    v->body()->accept(this);  // 访问循环体并计算哈希值
    hash = hash_combine(hash, hashOf(v->body()));  // 添加循环体的哈希值
  }

  putHash(v, hash);  // 计算整个 For 循环对象的最终哈希值
}

void HashProvider::visit(BroadcastPtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->value()->accept(this);  // 访问广播值并计算哈希值
  putHash(v, hash_combine("broadcast", hashOf(v->value()), v->lanes()));  // 计算 Broadcast 对象的哈希值
}

void HashProvider::visit(IfThenElsePtr v) {
  CACHE_GUARD();  // 启用缓存保护机制
  v->condition()->accept(this);  // 访问条件表达式并计算哈希值
  v->true_value()->accept(this);  // 访问真值并计算哈希值
  v->false_value()->accept(this);  // 访问假值并计算哈希值

  putHash(
      v,
      hash_combine(
          "ifthenelse",
          hashOf(v->condition()),  // 条件表达式的哈希值
          hashOf(v->true_value()),  // 真值的哈希值
          hashOf(v->false_value())));  // 假值的哈希值
}
// 访问 IntrinsicsPtr 类型对象的哈希计算函数
void HashProvider::visit(IntrinsicsPtr v) {
  CACHE_GUARD(); // 缓存保护

  // 如果操作类型是 kRand，则使用 rand() 函数返回值作为哈希值
  // 这里的 rand() 每次调用都返回不同的值，不能生成确定性哈希，但这是我们目前能做到的最好处理方式。
  if (v->op_type() == kRand) {
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
    putHash(v, (SimplifierHashType)rand());
    return;
  }

  // 计算函数名称的哈希值作为初始哈希
  SimplifierHashType hash(te_hash(v->func_name()));

  // 遍历参数列表，计算每个参数的哈希值并组合到主哈希中
  for (const auto i : c10::irange(v->nparams())) {
    v->param(i)->accept(this);
    hash = hash_combine(hash, hashOf(v->param(i)));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

// 访问 AllocatePtr 类型对象的哈希计算函数
void HashProvider::visit(AllocatePtr v) {
  CACHE_GUARD(); // 缓存保护

  // 访问缓冲变量并计算其哈希值
  VarPtr buffer_var = v->buffer_var();
  buffer_var->accept(this);

  // 计算 "allocate"、缓冲变量哈希和数据类型哈希的组合哈希值
  SimplifierHashType hash =
      hash_combine("allocate", hashOf(buffer_var), v->dtype());

  // 遍历维度表达式列表，计算每个维度表达式的哈希值并组合到主哈希中
  std::vector<ExprPtr> dims = v->dims();
  for (const ExprPtr& dim : dims) {
    dim->accept(this);
    hash = hash_combine(hash, hashOf(dim));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

// 访问 FreePtr 类型对象的哈希计算函数
void HashProvider::visit(FreePtr v) {
  CACHE_GUARD(); // 缓存保护

  // 访问缓冲变量并计算其哈希值
  VarPtr buffer_var = v->buffer_var();
  buffer_var->accept(this);

  // 计算 "free" 和缓冲变量哈希的组合哈希值
  putHash(v, hash_combine("free", hashOf(buffer_var)));
}

// 访问 CondPtr 类型对象的哈希计算函数
void HashProvider::visit(CondPtr v) {
  CACHE_GUARD(); // 缓存保护

  // 访问条件表达式，并计算其哈希值
  ExprPtr condition = v->condition();
  condition->accept(this);

  // 初始化主哈希值为条件表达式的哈希值
  SimplifierHashType hash = hash_combine("cond", hashOf(condition));

  // 如果存在 true 分支，计算其哈希值并与主哈希组合
  StmtPtr true_stmt = v->true_stmt();
  if (true_stmt) {
    true_stmt->accept(this);
    hash = hash_combine(hash, hashOf(true_stmt));
  }

  // 如果存在 false 分支，计算其哈希值并与主哈希组合
  StmtPtr false_stmt = v->false_stmt();
  if (false_stmt) {
    false_stmt->accept(this);
    hash = hash_combine(hash, hashOf(false_stmt));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

// 访问 TermPtr 类型对象的哈希计算函数
void HashProvider::visit(TermPtr v) {
  CACHE_GUARD(); // 缓存保护

  // 访问标量并计算其哈希值
  v->scalar()->accept(this);

  // 计算 "term" 和标量哈希的组合哈希值
  SimplifierHashType hash = hash_combine("term", hashOf(v->scalar()));

  // 遍历变量列表，计算每个变量的哈希值并组合到主哈希中
  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

// 访问 PolynomialPtr 类型对象的哈希计算函数
void HashProvider::visit(PolynomialPtr v) {
  CACHE_GUARD(); // 缓存保护

  // 访问标量并计算其哈希值
  v->scalar()->accept(this);

  // 计算 "term" 和标量哈希的组合哈希值
  SimplifierHashType hash = hash_combine("term", hashOf(v->scalar()));

  // 遍历变量列表，计算每个变量的哈希值并组合到主哈希中
  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

// 访问 MaxTermPtr 类型对象的哈希计算函数
void HashProvider::visit(MaxTermPtr v) {
  CACHE_GUARD(); // 缓存保护

  // 初始化主哈希值为 "maxterm"
  SimplifierHashType hash = hash_combine("maxterm");

  // 如果存在标量，计算其哈希值并与主哈希组合
  if (v->scalar()) {
    v->scalar()->accept(this);
    hash = hash_combine(hash, hashOf(v->scalar()));
  }

  // 遍历变量列表，计算每个变量的哈希值并组合到主哈希中
  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

// 访问 MinTermPtr 类型对象的哈希计算函数
void HashProvider::visit(MinTermPtr v) {
  CACHE_GUARD(); // 缓存保护

  // 初始化主哈希值为 "minterm"
  SimplifierHashType hash = hash_combine("minterm");

  // 如果存在标量，计算其哈希值并与主哈希组合
  if (v->scalar()) {
    v->scalar()->accept(this);
    hash = hash_combine(hash, hashOf(v->scalar()));
  }

  // 遍历变量列表，计算每个变量的哈希值并组合到主哈希中
  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  // 将最终的哈希值与对象关联存储
  putHash(v, hash);
}

} // namespace torch::jit::tensorexpr
```