# `.\pytorch\torch\csrc\jit\tensorexpr\ir_cloner.cpp`

```
// 包含 Torch 库中的头文件，用于张量表达式的 IR 克隆
#include <torch/csrc/jit/tensorexpr/ir_cloner.h>

// 包含 Torch 库中的张量表达式 IR 相关头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

// 包含 C++ 标准库中的实用工具，例如 irange
#include <c10/util/irange.h>

// 定义了 torch::jit::tensorexpr 命名空间
namespace torch::jit::tensorexpr {

// 模板函数，用于处理二元操作的变异
template <
    typename Op,
    // 启用 SFINAE 技术，只有当 detail::bin_op_deducer 返回 void 时才有效
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static ExprPtr mutate_binary_op(
    NodePtr<Op> v,
    IRCloner* cloner,
    bool option = false) {
  // 用克隆器对左操作数进行变异
  ExprPtr lhs_new = v->lhs()->accept_mutator(cloner);
  // 用克隆器对右操作数进行变异
  ExprPtr rhs_new = v->rhs()->accept_mutator(cloner);
  // 获取表达式的类型
  IRNodeType expr_type = v->expr_type();
  // 根据表达式类型进行处理
  switch (expr_type) {
    // 加法表达式
    case IRNodeType::kAdd:
      return alloc<Add>(lhs_new, rhs_new);
    // 减法表达式
    case IRNodeType::kSub:
      return alloc<Sub>(lhs_new, rhs_new);
    // 乘法表达式
    case IRNodeType::kMul:
      return alloc<Mul>(lhs_new, rhs_new);
    // 除法表达式
    case IRNodeType::kDiv:
      return alloc<Div>(lhs_new, rhs_new);
    // 取模表达式
    case IRNodeType::kMod:
      return alloc<Mod>(lhs_new, rhs_new);
    // 最大值表达式
    case IRNodeType::kMax:
      return alloc<Max>(lhs_new, rhs_new, option);
    // 最小值表达式
    case IRNodeType::kMin:
      return alloc<Min>(lhs_new, rhs_new, option);
    // 逻辑与表达式
    case IRNodeType::kAnd:
      return alloc<And>(lhs_new, rhs_new);
    // 逻辑或表达式
    case IRNodeType::kOr:
      return alloc<Or>(lhs_new, rhs_new);
    // 逻辑异或表达式
    case IRNodeType::kXor:
      return alloc<Xor>(lhs_new, rhs_new);
    // 左移表达式
    case IRNodeType::kLshift:
      return alloc<Lshift>(lhs_new, rhs_new);
    // 右移表达式
    case IRNodeType::kRshift:
      return alloc<Rshift>(lhs_new, rhs_new);
    // 默认情况，抛出未实现的低级错误
    default:
      throw unimplemented_lowering(v);
  }
}

// 以下是 IRCloner 类中不同类型二元操作的具体变异实现

// 加法操作的变异
ExprPtr IRCloner::mutate(AddPtr v) {
  return mutate_binary_op(v, this);
}

// 减法操作的变异
ExprPtr IRCloner::mutate(SubPtr v) {
  return mutate_binary_op(v, this);
}

// 乘法操作的变异
ExprPtr IRCloner::mutate(MulPtr v) {
  return mutate_binary_op(v, this);
}

// 除法操作的变异
ExprPtr IRCloner::mutate(DivPtr v) {
  return mutate_binary_op(v, this);
}

// 取模操作的变异
ExprPtr IRCloner::mutate(ModPtr v) {
  return mutate_binary_op(v, this);
}

// 逻辑与操作的变异
ExprPtr IRCloner::mutate(AndPtr v) {
  return mutate_binary_op(v, this);
}

// 逻辑或操作的变异
ExprPtr IRCloner::mutate(OrPtr v) {
  return mutate_binary_op(v, this);
}

// 逻辑异或操作的变异
ExprPtr IRCloner::mutate(XorPtr v) {
  return mutate_binary_op(v, this);
}

// 左移操作的变异
ExprPtr IRCloner::mutate(LshiftPtr v) {
  return mutate_binary_op(v, this);
}

// 右移操作的变异
ExprPtr IRCloner::mutate(RshiftPtr v) {
  return mutate_binary_op(v, this);
}

// 最大值操作的变异
ExprPtr IRCloner::mutate(MaxPtr v) {
  // 调用二元操作变异函数，同时传递是否传播 NaN 的选项
  return mutate_binary_op(v, this, v->propagate_nans());
}

// 最小值操作的变异
ExprPtr IRCloner::mutate(MinPtr v) {
  // 调用二元操作变异函数，同时传递是否传播 NaN 的选项
  return mutate_binary_op(v, this, v->propagate_nans());
}

} // namespace torch::jit::tensorexpr
ExprPtr IRCloner::mutate(CompareSelectPtr v) {
  // 使用当前对象的lhs()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  // 使用当前对象的rhs()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);
  // 使用当前对象的ret_val1()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr retval1_new = v->ret_val1()->accept_mutator(this);
  // 使用当前对象的ret_val2()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr retval2_new = v->ret_val2()->accept_mutator(this);
  // 返回一个新创建的CompareSelect对象，包括新的表达式指针和当前对象的其它属性
  return alloc<CompareSelect>(
      lhs_new,
      rhs_new,
      retval1_new,
      retval2_new,
      v->compare_select_op(),
      v->bias());
}

// NOLINTNEXTLINE
// 为每种标量类型定义一个对应的mutate方法，实现对不同类型对象的克隆
#define IMM_MUTATE_DEFINE(_1, Name)          \
  ExprPtr IRCloner::mutate(Name##ImmPtr v) { \
    return v;                                \
  }
// 对所有标量类型和Bool、Half、BFloat16类型应用IMM_MUTATE_DEFINE宏定义的mutate方法
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

ExprPtr IRCloner::mutate(CastPtr v) {
  // 使用当前对象的src_value()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr src_value_new = v->src_value()->accept_mutator(this);
  // 返回一个新创建的Cast对象，包括新的表达式指针和当前对象的数据类型
  return alloc<Cast>(v->dtype(), src_value_new);
}

ExprPtr IRCloner::mutate(BitCastPtr v) {
  // 使用当前对象的src_value()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr src_value_new = v->src_value()->accept_mutator(this);
  // 返回一个新创建的BitCast对象，包括新的表达式指针和当前对象的数据类型
  return alloc<BitCast>(v->dtype(), src_value_new);
}

ExprPtr IRCloner::mutate(RampPtr v) {
  // 使用当前对象的base()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr base_new = v->base()->accept_mutator(this);
  // 使用当前对象的stride()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr stride_new = v->stride()->accept_mutator(this);
  // 返回一个新创建的Ramp对象，包括新的表达式指针和当前对象的lanes数
  return alloc<Ramp>(base_new, stride_new, v->lanes());
}

ExprPtr IRCloner::mutate(LoadPtr v) {
  // 创建一个新的表达式指针数组indices_new，保留v->indices()大小的空间
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  // 遍历当前对象的indices()数组，每个元素调用accept_mutator(this)，并添加到indices_new
  for (const ExprPtr& ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  // 使用当前对象的buf()方法调用accept_mutator(this)，并转换为BufPtr类型
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  // 返回一个新创建的Load对象，包括新的表达式指针和当前对象的数据类型、buf指针
  return alloc<Load>(v->dtype(), buf_new, indices_new);
}

// 不对Var进行克隆，因为原始IR和克隆IR共享底层变量
ExprPtr IRCloner::mutate(VarPtr v) {
  // 直接返回当前对象的指针，不进行克隆
  return v;
}

// 不对Buf进行克隆，因为原始IR和克隆IR共享底层Buf
// 尽管Buf具有表达式作为dims和initializers，这是克隆在这一点上的预期使用
//
// TODO: 如果Buf需要被克隆，需要重新审视这一点。
ExprPtr IRCloner::mutate(BufPtr v) {
  // 直接返回当前对象的指针，不进行克隆
  return v;
}

ExprPtr IRCloner::mutate(BroadcastPtr v) {
  // 获取广播的lanes数
  int lanes = v->lanes();
  // 使用当前对象的value()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr value_new = v->value()->accept_mutator(this);
  // 返回一个新创建的Broadcast对象，包括新的表达式指针和lanes数
  return alloc<Broadcast>(value_new, lanes);
}

ExprPtr IRCloner::mutate(IfThenElsePtr v) {
  // 使用当前对象的condition()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr condition_new = v->condition()->accept_mutator(this);
  // 使用当前对象的true_value()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr true_value_new = v->true_value()->accept_mutator(this);
  // 使用当前对象的false_value()方法调用accept_mutator(this)，返回一个新的表达式指针
  ExprPtr false_value_new = v->false_value()->accept_mutator(this);

  // 返回一个新创建的IfThenElse对象，包括新的表达式指针
  return alloc<IfThenElse>(condition_new, true_value_new, false_value_new);
}

ExprPtr IRCloner::mutate(IntrinsicsPtr v) {
  // 创建一个新的表达式指针数组params_new，保留v->nparams()大小的空间
  std::vector<ExprPtr> params_new;
  params_new.reserve(v->nparams());
  // 遍历当前对象的params()数组，每个元素调用accept_mutator(this)，并添加到params_new
  for (const auto& param : v->params()) {
    params_new.push_back(param->accept_mutator(this));
  }
  // 返回一个新创建的Intrinsics对象，包括新的表达式指针和当前对象的操作类型、数据类型
  return alloc<Intrinsics>(v->op_type(), v->dtype(), params_new);
}
ExprPtr IRCloner::mutate(TermPtr v) {
  // 对给定的 Term 指针进行变异，处理其标量部分
  ExprPtr scalar_new = v->scalar()->accept_mutator(this);

  // 处理 Term 中的变量部分，使用迭代器遍历每一个变量并进行变异处理
  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (const auto& t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }

  // 创建并返回一个新的 Term 对象，保持其哈希值不变，但更新标量和变量部分
  return alloc<Term>(v->hasher(), scalar_new, variables_new);
}

ExprPtr IRCloner::mutate(PolynomialPtr v) {
  // 对给定的 Polynomial 指针进行变异，处理其标量部分
  ExprPtr scalar_new = v->scalar()->accept_mutator(this);

  // 处理 Polynomial 中的 Term 变量部分，使用迭代器遍历每一个 Term 并进行变异处理
  std::vector<TermPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (const auto& t : v->variables()) {
    variables_new.push_back(static_to<Term>(t->accept_mutator(this)));
  }

  // 创建并返回一个新的 Polynomial 对象，保持其哈希值不变，但更新标量和变量部分
  return alloc<Polynomial>(v->hasher(), scalar_new, variables_new);
}

ExprPtr IRCloner::mutate(RoundOffPtr v) {
  // 对给定的 RoundOff 指针进行变异，处理其左右表达式部分
  return alloc<RoundOff>(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

ExprPtr IRCloner::mutate(MaxTermPtr v) {
  // 处理 MaxTerm 中的标量部分，如果存在则进行变异处理
  ExprPtr scalar_new =
      v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  // 处理 MaxTerm 中的变量部分，使用迭代器遍历每一个变量并进行变异处理
  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (const auto& t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }

  // 创建并返回一个新的 MaxTerm 对象，保持其哈希值不变，但更新标量和变量部分及 NaN 传播属性
  return alloc<MaxTerm>(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new);
}

ExprPtr IRCloner::mutate(MinTermPtr v) {
  // 处理 MinTerm 中的标量部分，如果存在则进行变异处理
  ExprPtr scalar_new =
      v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  // 处理 MinTerm 中的变量部分，使用迭代器遍历每一个变量并进行变异处理
  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (const auto& t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }

  // 创建并返回一个新的 MinTerm 对象，保持其哈希值不变，但更新标量和变量部分及 NaN 传播属性
  return alloc<MinTerm>(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new);
}

ExprPtr IRCloner::mutate(ReduceOpPtr v) {
  // 处理 ReduceOp 中的主体表达式部分，进行变异处理
  ExprPtr body_new = v->body()->accept_mutator(this);

  // 处理 ReduceOp 中的减少参数部分，使用迭代器遍历每一个参数并进行变异处理
  std::vector<VarPtr> reduce_args_new;
  reduce_args_new.reserve(v->reduce_args().size());
  for (const auto& r : v->reduce_args()) {
    reduce_args_new.push_back(static_to<Var>(r->accept_mutator(this)));
  }

  // 创建并返回一个新的 ReduceOp 对象，更新其主体表达式和减少参数部分
  return alloc<ReduceOp>(body_new, reduce_args_new, v->reducer());
}

StmtPtr IRCloner::mutate(ForPtr v) {
  // 处理 For 循环中的起始、终止、主体部分，分别进行变异处理
  auto start_new = v->start()->accept_mutator(this);
  auto stop_new = v->stop()->accept_mutator(this);
  auto body_new = v->body()->accept_mutator(this);

  // 创建并返回一个新的 For 循环对象，更新其起始、终止、主体和循环选项
  return alloc<For>(v->var(), start_new, stop_new, body_new, v->loop_options());
}

StmtPtr IRCloner::mutate(BlockPtr v) {
  // 处理 Block 中的语句列表，使用迭代器遍历每一条语句并进行变异处理
  std::vector<StmtPtr> stmts_new;
  stmts_new.reserve(v->nstmts());
  for (const StmtPtr& stmt : *v) {
    stmts_new.push_back(stmt->accept_mutator(this));
  }

  // 创建并返回一个新的 Block 对象，更新其语句列表
  return alloc<Block>(stmts_new);
}

StmtPtr IRCloner::mutate(StorePtr v) {
  // 处理 Store 操作中的索引列表，使用迭代器遍历每一个索引并进行变异处理
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (const auto& ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }

  // 处理 Store 操作中的值表达式，进行变异处理
  auto value_new = v->value()->accept_mutator(this);

  // 处理 Store 操作中的缓冲区对象，进行变异处理并转换为 Buf 类型
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));

  // 创建并返回一个新的 Store 对象，更新其缓冲区、索引列表和值表达式
  return alloc<Store>(buf_new, indices_new, value_new);
}
StmtPtr IRCloner::mutate(AtomicAddPtr v) {
  // 创建一个新的空向量用于存储新的索引表达式
  std::vector<ExprPtr> indices_new;
  // 预留空间以避免重新分配
  indices_new.reserve(v->indices().size());
  // 遍历原子加操作的索引表达式，并对每个表达式进行变异操作
  for (const auto& ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  // 对原子加操作的值表达式进行变异操作
  auto value_new = v->value()->accept_mutator(this);
  // 将原子加操作的缓冲区表达式转换为缓冲区对象，并对其进行变异操作
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  // 创建并返回一个新的原子加语句对象，使用变异后的缓冲区、索引和值
  return alloc<AtomicAdd>(buf_new, indices_new, value_new);
}

StmtPtr IRCloner::mutate(AllocatePtr v) {
  // 将分配语句的缓冲区表达式转换为缓冲区对象，并对其进行变异操作
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  // 创建并返回一个新的分配语句对象，使用变异后的缓冲区
  return alloc<Allocate>(buf_new);
}

StmtPtr IRCloner::mutate(FreePtr v) {
  // 将释放语句的缓冲区表达式转换为缓冲区对象，并对其进行变异操作
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  // 创建并返回一个新的释放语句对象，使用变异后的缓冲区
  return alloc<Free>(buf_new);
}

StmtPtr IRCloner::mutate(SyncThreadsPtr v) {
  // 创建并返回一个新的线程同步语句对象
  return alloc<SyncThreads>();
}

StmtPtr IRCloner::mutate(ExternalCallPtr v) {
  // 将外部调用语句的缓冲区表达式转换为缓冲区对象，并对其进行变异操作
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));

  // 创建一个新的空向量用于存储新的缓冲区参数表达式
  std::vector<BufPtr> buf_args_new;
  // 预留空间以避免重新分配
  buf_args_new.reserve(v->buf_args().size());
  // 遍历原始缓冲区参数表达式，并对每个表达式进行变异操作
  for (const BufPtr& buf_arg : v->buf_args()) {
    buf_args_new.push_back(to<Buf>(buf_arg->accept_mutator(this)));
  }

  // 创建一个新的空向量用于存储新的参数表达式
  std::vector<ExprPtr> args_new;
  // 预留空间以避免重新分配
  args_new.reserve(v->args().size());
  // 遍历原始参数表达式，并对每个表达式进行变异操作
  for (const ExprPtr& arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  // 创建并返回一个新的外部调用语句对象，使用变异后的缓冲区、函数名、缓冲区参数和参数
  return alloc<ExternalCall>(buf_new, v->func_name(), buf_args_new, args_new);
}

StmtPtr IRCloner::mutate(ExternalCallWithAllocPtr v) {
  // 创建一个新的空向量用于存储新的输出缓冲区参数表达式
  std::vector<BufPtr> buf_out_args_new;
  // 预留空间以避免重新分配
  buf_out_args_new.reserve(v->buf_out_args().size());
  // 遍历原始输出缓冲区参数表达式，并对每个表达式进行变异操作
  for (const auto& buf_out_arg : v->buf_out_args()) {
    buf_out_args_new.push_back(to<Buf>(buf_out_arg->accept_mutator(this)));
  }

  // 创建一个新的空向量用于存储新的缓冲区参数表达式
  std::vector<BufPtr> buf_args_new;
  // 预留空间以避免重新分配
  buf_args_new.reserve(v->buf_args().size());
  // 遍历原始缓冲区参数表达式，并对每个表达式进行变异操作
  for (const auto& buf_arg : v->buf_args()) {
    buf_args_new.push_back(to<Buf>(buf_arg->accept_mutator(this)));
  }

  // 创建一个新的空向量用于存储新的参数表达式
  std::vector<ExprPtr> args_new;
  // 预留空间以避免重新分配
  args_new.reserve(v->args().size());
  // 遍历原始参数表达式，并对每个表达式进行变异操作
  for (const auto& arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  // 创建并返回一个新的带分配的外部调用语句对象，使用函数名、输出缓冲区参数、缓冲区参数和参数
  return alloc<ExternalCallWithAlloc>(v->func_name(), buf_out_args_new, buf_args_new, args_new);
}

StmtPtr IRCloner::mutate(LetPtr v) {
  // 对赋值语句的值表达式进行变异操作
  auto value_new = v->value()->accept_mutator(this);
  // 创建并返回一个新的赋值语句对象，使用变异后的变量名和值表达式
  return alloc<Let>(v->var(), value_new);
}

StmtPtr IRCloner::mutate(CondPtr v) {
  // 对条件语句的条件表达式进行变异操作
  auto condition_new = v->condition()->accept_mutator(this);
  // 复制原始的真语句和假语句
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();
  // 如果真语句存在，则对其进行变异操作；否则保持不变
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  // 如果假语句存在，则对其进行变异操作；否则保持不变
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;
  // 创建并返回一个新的条件语句对象，使用变异后的条件、真语句和假语句
  return alloc<Cond>(condition_new, true_new, false_new);
}

StmtPtr Stmt::clone(StmtPtr s) {
  // 创建一个 IRCloner 实例用于克隆操作
  IRCloner cloner;
  // 使用 IRCloner 对象进行变异操作，克隆给定的语句
  StmtPtr cloned = s->accept_mutator(&cloner);
  // 设置克隆语句的父节点为空
  set_parent(cloned, nullptr);
  // 返回克隆后的语句对象
  return cloned;
}

ExprPtr Expr::clone(ExprPtr e) {
  // 创建一个 IRCloner 实例用于克隆操作
  IRCloner cloner;
  // 使用 IRCloner 对象进行变异操作，克隆给定的表达式
  return e->accept_mutator(&cloner);
}

} // namespace torch::jit::tensorexpr
```