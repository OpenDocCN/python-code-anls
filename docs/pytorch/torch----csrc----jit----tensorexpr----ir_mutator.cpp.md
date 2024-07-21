# `.\pytorch\torch\csrc\jit\tensorexpr\ir_mutator.cpp`

```py
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
// 引入 IRMutator 类的声明，用于修改和变异 IR（Intermediate Representation）

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
// 引入其他相关的头文件，包括表达式求值、IR 表达式、IR 简化和归约

#include <c10/util/irange.h>
// 引入 c10 库中的 irange.h 头文件，用于整数范围迭代器

namespace torch::jit::tensorexpr {

template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static ExprPtr mutate_binary_op(
    NodePtr<Op> v,
    IRMutator* mutator,
    bool option = false) {
  ExprPtr lhs = v->lhs();
  // 获取二元操作的左操作数表达式
  ExprPtr rhs = v->rhs();
  // 获取二元操作的右操作数表达式
  ExprPtr lhs_new = lhs->accept_mutator(mutator);
  // 使用 mutator 变异左操作数表达式
  ExprPtr rhs_new = rhs->accept_mutator(mutator);
  // 使用 mutator 变异右操作数表达式
  if (lhs != lhs_new) {
    v->set_lhs(lhs_new);
    // 如果左操作数有变化，则更新操作数为变异后的表达式
  }
  if (rhs != rhs_new) {
    v->set_rhs(rhs_new);
    // 如果右操作数有变化，则更新操作数为变异后的表达式
  }
  Dtype dtype_new =
      BinaryOpDtype(lhs_new->dtype(), rhs_new->dtype(), ScalarType::Undefined);
  // 计算新的操作数据类型，使用左右操作数表达式的数据类型
  if (dtype_new != v->dtype()) {
    v->set_dtype(dtype_new);
    // 如果新的数据类型不同于当前的数据类型，则更新操作的数据类型
  }
  return v;
  // 返回更新后的操作对象指针
}

ExprPtr IRMutator::mutate(AddPtr v) {
  return mutate_binary_op(v, this);
  // 对 AddPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(SubPtr v) {
  return mutate_binary_op(v, this);
  // 对 SubPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(MulPtr v) {
  return mutate_binary_op(v, this);
  // 对 MulPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(DivPtr v) {
  return mutate_binary_op(v, this);
  // 对 DivPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(ModPtr v) {
  return mutate_binary_op(v, this);
  // 对 ModPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(AndPtr v) {
  return mutate_binary_op(v, this);
  // 对 AndPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(OrPtr v) {
  return mutate_binary_op(v, this);
  // 对 OrPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(XorPtr v) {
  return mutate_binary_op(v, this);
  // 对 XorPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(LshiftPtr v) {
  return mutate_binary_op(v, this);
  // 对 LshiftPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(RshiftPtr v) {
  return mutate_binary_op(v, this);
  // 对 RshiftPtr 类型的操作对象进行变异处理并返回结果
}

ExprPtr IRMutator::mutate(MaxPtr v) {
  return mutate_binary_op(v, this, v->propagate_nans());
  // 对 MaxPtr 类型的操作对象进行变异处理并返回结果，根据 propagate_nans() 方法决定是否传播 NaN
}

ExprPtr IRMutator::mutate(MinPtr v) {
  return mutate_binary_op(v, this, v->propagate_nans());
  // 对 MinPtr 类型的操作对象进行变异处理并返回结果，根据 propagate_nans() 方法决定是否传播 NaN
}

ExprPtr IRMutator::mutate(CompareSelectPtr v) {
  ExprPtr lhs = v->lhs();
  // 获取 CompareSelectPtr 类型操作对象的左操作数表达式
  ExprPtr rhs = v->rhs();
  // 获取 CompareSelectPtr 类型操作对象的右操作数表达式
  ExprPtr ret_val1 = v->ret_val1();
  // 获取 CompareSelectPtr 类型操作对象的第一个返回值表达式
  ExprPtr ret_val2 = v->ret_val2();
  // 获取 CompareSelectPtr 类型操作对象的第二个返回值表达式
  ExprPtr lhs_new = lhs->accept_mutator(this);
  // 使用当前 mutator 对左操作数表达式进行变异处理
  ExprPtr rhs_new = rhs->accept_mutator(this);
  // 使用当前 mutator 对右操作数表达式进行变异处理
  ExprPtr ret_val1_new = ret_val1->accept_mutator(this);
  // 使用当前 mutator 对第一个返回值表达式进行变异处理
  ExprPtr ret_val2_new = ret_val2->accept_mutator(this);
  // 使用当前 mutator 对第二个返回值表达式进行变异处理
  if (lhs != lhs_new) {
    v->set_lhs(lhs_new);
    // 如果左操作数有变化，则更新操作对象的左操作数
  }
  if (rhs != rhs_new) {
    v->set_rhs(rhs_new);
    // 如果右操作数有变化，则更新操作对象的右操作数
  }
  if (ret_val1 != ret_val1_new) {
    v->set_ret_val1(ret_val1_new);
    // 如果第一个返回值有变化，则更新操作对象的第一个返回值
  }
  if (ret_val2 != ret_val2_new) {
    v->set_ret_val2(ret_val2_new);
    // 如果第二个返回值有变化，则更新操作对象的第二个返回值
  }
  return v;
  // 返回更新后的 CompareSelectPtr 操作对象指针
}

// NOLINTNEXTLINE
#define IMM_MUTATE_DEFINE(_1, Name)           \
  ExprPtr IRMutator::mutate(Name##ImmPtr v) { \
    return v;                                 \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE
// 使用宏定义遍历标量类型（scalar types），对每种类型的 NameImmPtr 类型操作对象进行变异处理并返回结果

} // namespace torch::jit::tensorexpr
// 结束 torch::jit::tensorexpr 命名空间
ExprPtr IRMutator::mutate(CastPtr v) {
  // 获取转换操作的源值表达式
  ExprPtr src_value = v->src_value();
  // 对源值表达式进行变异操作
  ExprPtr src_value_new = src_value->accept_mutator(this);
  // 如果源值表达式有变化，则更新转换操作的源值
  if (src_value != src_value_new) {
    v->set_src_value(src_value_new);
  }
  // 返回更新后的转换操作
  return v;
}

ExprPtr IRMutator::mutate(BitCastPtr v) {
  // 获取位转换操作的源值表达式
  ExprPtr src_value = v->src_value();
  // 对源值表达式进行变异操作
  ExprPtr src_value_new = src_value->accept_mutator(this);
  // 如果源值表达式有变化，则更新位转换操作的源值
  if (src_value != src_value_new) {
    v->set_src_value(src_value_new);
  }
  // 返回更新后的位转换操作
  return v;
}

ExprPtr IRMutator::mutate(VarPtr v) {
  // 变异操作中不需要对变量表达式进行修改，直接返回原变量表达式
  return v;
}

ExprPtr IRMutator::mutate(RampPtr v) {
  // 获取 Ramp 操作的基础值和步长
  ExprPtr base = v->base();
  ExprPtr stride = v->stride();
  // 对基础值和步长进行变异操作
  ExprPtr base_new = base->accept_mutator(this);
  ExprPtr stride_new = stride->accept_mutator(this);
  // 如果基础值有变化，则更新 Ramp 操作的基础值
  if (base != base_new) {
    v->set_base(base_new);
  }
  // 如果步长有变化，则更新 Ramp 操作的步长
  if (stride != stride_new) {
    v->set_stride(stride_new);
  }
  // 返回更新后的 Ramp 操作
  return v;
}

ExprPtr IRMutator::mutate(LoadPtr v) {
  // 获取 Load 操作的缓冲区
  BufPtr buf = v->buf();

  // 检查 Load 操作的索引是否有任何变化
  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (const ExprPtr& ind : v->indices()) {
    // 对每个索引表达式进行变异操作
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  // 获取更新后的缓冲区表达式
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));

  // 如果缓冲区表达式有变化，则更新 Load 操作的缓冲区
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 如果任何索引有变化，则更新 Load 操作的索引列表
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  // 返回更新后的 Load 操作
  return v;
}

ExprPtr IRMutator::mutate(BufPtr v) {
  // 获取缓冲区操作的基础变量
  VarPtr var = v->base_handle();
  // 对基础变量进行变异操作
  VarPtr var_new = to<Var>(var->accept_mutator(this));
  // 如果新的基础变量为空，则返回空指针
  if (!var_new) {
    return nullptr;
  }

  // 检查维度是否有变化
  bool dims_changed = false;
  std::vector<ExprPtr> dims_old = v->dims();
  std::vector<ExprPtr> dims_new(dims_old.size());
  for (const auto i : c10::irange(dims_old.size())) {
    // 对每个维度表达式进行变异操作
    dims_new[i] = dims_old[i]->accept_mutator(this);
    dims_changed |= (dims_new[i] != dims_old[i]);
  }

  // 如果基础变量有变化，则更新缓冲区操作的基础变量
  if (var != var_new) {
    v->set_base_handle(var_new);
  }
  // 如果维度有变化，则更新缓冲区操作的维度列表
  if (dims_changed) {
    v->set_dims(dims_new);
  }

  // 获取缓冲区操作的 qscale 和 qzero 值，并对它们进行变异操作
  ExprPtr qscale = v->qscale();
  if (qscale) {
    ExprPtr qscale_new = qscale->accept_mutator(this);
    if (qscale != qscale_new) {
      v->set_qscale(qscale_new);
    }
  }

  ExprPtr qzero = v->qzero();
  if (qzero) {
    ExprPtr qzero_new = qzero->accept_mutator(this);
    if (qzero != qzero_new) {
      v->set_qzero(qzero_new);
    }
  }

  // 返回更新后的缓冲区操作
  return v;
}

ExprPtr IRMutator::mutate(BroadcastPtr v) {
  // 获取广播操作的值表达式
  ExprPtr value = v->value();
  // 对值表达式进行变异操作
  ExprPtr value_new = value->accept_mutator(this);
  // 如果值表达式有变化，则更新广播操作的值
  if (value != value_new) {
    v->set_value(value_new);
  }
  // 返回更新后的广播操作
  return v;
}

ExprPtr IRMutator::mutate(IfThenElsePtr v) {
  // 获取条件、真值和假值表达式
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();
  // 对条件、真值和假值表达式进行变异操作
  ExprPtr condition_new = condition->accept_mutator(this);
  ExprPtr true_value_new = true_value->accept_mutator(this);
  ExprPtr false_value_new = false_value->accept_mutator(this);

  // 如果条件表达式有变化，则更新 IfThenElse 操作的条件
  if (condition != condition_new) {
    v->set_condition(condition_new);
  }
  // 如果条件值发生变化，则更新变量 v 的条件值
  if (true_value != true_value_new) {
    // 设置变量 v 的真值为新值 true_value_new
    v->set_true_value(true_value_new);
  }
  // 如果假值发生变化，则更新变量 v 的假值
  if (false_value != false_value_new) {
    // 设置变量 v 的假值为新值 false_value_new
    v->set_false_value(false_value_new);
  }
  // 返回更新后的变量 v
  return v;
}

# 定义 IRMutator 类的 mutate 方法，用于处理 IntrinsicsPtr 类型的表达式
ExprPtr IRMutator::mutate(IntrinsicsPtr v) {
  # 创建与参数个数相同的表达式指针数组
  std::vector<ExprPtr> params(v->nparams());
  # 标志是否有任何变化
  bool any_change = false;
  # 遍历每个参数
  for (int i = 0; i < v->nparams(); i++) {
    # 获取当前参数的表达式指针
    ExprPtr value = v->param(i);
    # 对参数表达式进行变异操作
    ExprPtr value_new = value->accept_mutator(this);
    # 检查是否有变化
    if (value != value_new) {
      any_change = true;
    }
    # 更新参数数组
    params[i] = value_new;
  }
  # 如果有任何变化，则设置新的参数列表
  if (any_change) {
    v->set_params(params);
  }
  # 返回更新后的表达式指针
  return v;
}

# 定义 IRMutator 类的 mutate 方法，用于处理 TermPtr 类型的表达式
ExprPtr IRMutator::mutate(TermPtr v) {
  # 变异标量部分的表达式指针
  ExprPtr newScalar = v->scalar()->accept_mutator(this);

  # 创建变量表达式指针数组
  std::vector<ExprPtr> variables;
  # 遍历每个变量表达式，并进行变异操作
  for (const auto& t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  # 返回新的 Term 对象
  return alloc<Term>(v->hasher(), newScalar, variables);
}

# 定义 IRMutator 类的 mutate 方法，用于处理 PolynomialPtr 类型的表达式
ExprPtr IRMutator::mutate(PolynomialPtr v) {
  # 变异标量部分的表达式指针
  ExprPtr newScalar = v->scalar()->accept_mutator(this);

  # 创建 TermPtr 类型的变量数组
  std::vector<TermPtr> variables;
  # 遍历每个变量表达式，并进行变异操作
  for (const auto& t : v->variables()) {
    variables.push_back(static_to<Term>(t->accept_mutator(this)));
  }
  # 返回新的 Polynomial 对象
  return alloc<Polynomial>(v->hasher(), newScalar, variables);
}

# 定义 IRMutator 类的 mutate 方法，用于处理 RoundOffPtr 类型的表达式
ExprPtr IRMutator::mutate(RoundOffPtr v) {
  # 对左右操作数进行变异操作，生成新的 RoundOff 对象
  return alloc<RoundOff>(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

# 定义 IRMutator 类的 mutate 方法，用于处理 MaxTermPtr 类型的表达式
ExprPtr IRMutator::mutate(MaxTermPtr v) {
  # 变异标量部分的表达式指针
  ExprPtr newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  # 创建变量表达式指针数组
  std::vector<ExprPtr> variables;
  # 遍历每个变量表达式，并进行变异操作
  for (const auto& t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  # 返回新的 MaxTerm 对象
  return alloc<MaxTerm>(v->hasher(), newScalar, v->propagate_nans(), variables);
}

# 定义 IRMutator 类的 mutate 方法，用于处理 MinTermPtr 类型的表达式
ExprPtr IRMutator::mutate(MinTermPtr v) {
  # 变异标量部分的表达式指针
  ExprPtr newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  # 创建变量表达式指针数组
  std::vector<ExprPtr> variables;
  # 遍历每个变量表达式，并进行变异操作
  for (const auto& t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  # 返回新的 MinTerm 对象
  return alloc<MinTerm>(v->hasher(), newScalar, v->propagate_nans(), variables);
}

# 定义 IRMutator 类的 mutate 方法，用于处理 ReduceOpPtr 类型的表达式
ExprPtr IRMutator::mutate(ReduceOpPtr v) {
  # 对体部表达式进行变异操作
  ExprPtr body_new = v->body()->accept_mutator(this);

  # 创建新的 VarPtr 类型数组
  std::vector<VarPtr> new_reduce_args;
  # 遍历每个减少参数，并进行变异操作
  for (const auto& r : v->reduce_args()) {
    new_reduce_args.push_back(static_to<Var>(r->accept_mutator(this)));
  }

  # 返回新的 ReduceOp 对象
  return alloc<ReduceOp>(body_new, new_reduce_args, v->reducer());
}

# 定义 IRMutator 类的 mutate 方法，用于处理 ForPtr 类型的语句
StmtPtr IRMutator::mutate(ForPtr v) {
  # 获取循环变量、开始、结束表达式指针和主体语句指针
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();

  # 对循环变量、开始、结束和主体进行变异操作
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body->accept_mutator(this);

  # 如果主体语句为空，则返回空指针
  if (!body_new) {
    return nullptr;
  }

  # 如果主体语句发生了变化，则设置新的主体语句
  if (body != body_new) {
    v->set_body(body_new);
  }

  # 如果循环变量发生了变化，则设置新的循环变量
  if (var != var_new) {
    v->set_var(var_new);
  }

  # 如果开始表达式发生了变化，则设置新的开始表达式
  if (start != start_new) {
    v->set_start(start_new);
  }

  # 如果结束表达式发生了变化，则设置新的结束表达式
  if (stop != stop_new) {
    v->set_stop(stop_new);
  }

  # 返回更新后的 For 对象
  return v;
}
// 对给定的基本块进行变异操作，返回变异后的基本块指针
StmtPtr IRMutator::mutate(BlockPtr v) {
  // 标记是否有任何变化
  bool any_change = false;

  // 存储变异后的语句列表
  std::vector<StmtPtr> stmts;
  // 遍历基本块中的每个语句
  for (const StmtPtr& stmt : *v) {
    // 调用接受变异器的方法对语句进行变异
    StmtPtr stmt_new = stmt->accept_mutator(this);
    // 检查语句是否发生变化
    if (stmt != stmt_new) {
      any_change = true;
    } else {
      // 如果语句未变化，克隆一个新的语句对象
      stmt_new = Stmt::clone(stmt);
    }
    // 如果变异后的语句不为空，加入到语句列表中
    if (stmt_new) {
      stmts.push_back(stmt_new);
    }
  }
  // 如果有语句发生变化，更新基本块的语句列表
  if (any_change) {
    v->set_stmts(stmts);
  }
  // 返回变异后的基本块指针
  return v;
}

// 对 Store 类型的操作进行变异处理
StmtPtr IRMutator::mutate(StorePtr v) {
  // 获取存储操作关联的缓冲区
  BufPtr buf = v->buf();

  // 标记索引是否有任何变化
  bool any_index_changed = false;
  // 存储变异后的索引列表
  std::vector<ExprPtr> indices_new;
  // 遍历存储操作的索引
  for (const ExprPtr& ind : v->indices()) {
    // 调用接受变异器的方法对索引进行变异
    ExprPtr new_ind = ind->accept_mutator(this);
    // 检查索引是否发生变化
    if (new_ind != ind) {
      any_index_changed = true;
    }
    // 将变异后的索引加入列表
    indices_new.push_back(new_ind);
  }
  // 获取存储操作的值
  ExprPtr value = v->value();
  // 对缓冲区和值进行变异处理
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  ExprPtr value_new = value->accept_mutator(this);

  // 如果缓冲区发生变化，更新存储操作中的缓冲区
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 如果索引发生变化，更新存储操作中的索引列表
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  // 如果值发生变化，更新存储操作中的值
  if (value != value_new) {
    v->set_value(value_new);
  }
  // 返回变异后的存储操作指针
  return v;
}

// 对 AtomicAdd 类型的操作进行变异处理
StmtPtr IRMutator::mutate(AtomicAddPtr v) {
  // 获取原子加操作关联的缓冲区
  BufPtr buf = v->buf();

  // 标记索引是否有任何变化
  bool any_index_changed = false;
  // 存储变异后的索引列表
  std::vector<ExprPtr> indices_new;
  // 遍历原子加操作的索引
  for (const ExprPtr& ind : v->indices()) {
    // 调用接受变异器的方法对索引进行变异
    ExprPtr new_ind = ind->accept_mutator(this);
    // 检查索引是否发生变化
    if (new_ind != ind) {
      any_index_changed = true;
    }
    // 将变异后的索引加入列表
    indices_new.push_back(new_ind);
  }
  // 获取原子加操作的值
  ExprPtr value = v->value();
  // 对缓冲区和值进行变异处理
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  ExprPtr value_new = value->accept_mutator(this);

  // 如果缓冲区发生变化，更新原子加操作中的缓冲区
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 如果索引发生变化，更新原子加操作中的索引列表
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  // 如果值发生变化，更新原子加操作中的值
  if (value != value_new) {
    v->set_value(value_new);
  }
  // 返回变异后的原子加操作指针
  return v;
}

// 对 SyncThreadsPtr 类型的操作进行变异处理
StmtPtr IRMutator::mutate(SyncThreadsPtr v) {
  // 返回分配的同步线程对象
  return alloc<SyncThreads>();
}

// 对 ExternalCallPtr 类型的操作进行变异处理
StmtPtr IRMutator::mutate(ExternalCallPtr v) {
  // 获取外部调用操作关联的缓冲区
  BufPtr buf = v->buf();
  // 对缓冲区进行变异处理
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  // 断言确保新的缓冲区非空
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));

  // 标记缓冲区参数是否有任何变化
  bool buf_args_changed = false;
  // 存储变异后的缓冲区参数列表
  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  // 遍历外部调用操作的缓冲区参数
  for (const BufPtr& buf_arg : v->buf_args()) {
    // 调用接受变异器的方法对缓冲区参数进行变异
    BufPtr buf_arg_new = to<Buf>(buf_arg->accept_mutator(this));
    // 断言确保新的缓冲区参数非空
    TORCH_INTERNAL_ASSERT(
        buf_arg_new, buildErrorMessage("IRMutator produced null for Buf."));
    // 将变异后的缓冲区参数加入列表
    buf_args_new.push_back(buf_arg_new);
    // 检查是否有缓冲区参数发生变化
    buf_args_changed |= buf_arg_new != buf_arg;
  }

  // 标记参数是否有任何变化
  bool args_changed = false;
  // 存储变异后的参数列表
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  // 遍历外部调用操作的参数
  for (const ExprPtr& arg : v->args()) {
    // 调用接受变异器的方法对参数进行变异
    ExprPtr arg_new = arg->accept_mutator(this);
    // 将变异后的参数加入列表
    args_new.push_back(arg_new);
    // 检查是否有参数发生变化
    args_changed |= arg_new != arg;
  }

  // 如果缓冲区发生变化，更新外部调用操作中的缓冲区
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 如果缓冲区参数发生变化，更新外部调用操作中的缓冲区参数列表
  if (buf_args_changed) {
    v->set_buf_args(buf_args_new);
  }
  // 如果参数发生变化，更新外部调用操作中的参数列表
  if (args_changed) {
    v->set_args(args_new);
  }
  // 返回变异后的外部调用操作指针
  return v;
}
// 对 ExternalCallWithAllocPtr 类型对象进行变异操作，返回变异后的对象
StmtPtr IRMutator::mutate(ExternalCallWithAllocPtr v) {
  // 标志位，指示 buf_out_args 是否有变化
  bool buf_out_args_changed = false;
  // 存储新的 buf_out_args
  std::vector<BufPtr> buf_out_args_new;
  buf_out_args_new.reserve(v->buf_out_args().size());
  // 遍历原 buf_out_args
  for (const auto& buf_out_arg : v->buf_out_args()) {
    // 对每个 buf_out_arg 进行变异
    BufPtr buf_out_arg_new = to<Buf>(buf_out_arg->accept_mutator(this));
    // 断言，确保变异后的 buf_out_arg_new 不为 null
    TORCH_INTERNAL_ASSERT(
        buf_out_arg_new, buildErrorMessage("IRMutator produced null for Buf."));
    // 将变异后的 buf_out_arg_new 添加到新的列表中
    buf_out_args_new.push_back(buf_out_arg_new);
    // 更新 buf_out_args_changed 标志位
    buf_out_args_changed |= buf_out_arg_new != buf_out_arg;
  }

  // 类似处理 buf_args
  bool buf_args_changed = false;
  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (const auto& buf_arg : v->buf_args()) {
    BufPtr buf_arg_new = to<Buf>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(
        buf_arg_new, buildErrorMessage("IRMutator produced null for Buf."));
    buf_args_new.push_back(buf_arg_new);
    buf_args_changed |= buf_arg_new != buf_arg;
  }

  // 类似处理普通的 args
  bool args_changed = false;
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (const auto& arg : v->args()) {
    ExprPtr arg_new = arg->accept_mutator(this);
    args_new.push_back(arg_new);
    args_changed |= arg_new != arg;
  }

  // 根据变异情况更新对象的 buf_out_args、buf_args 和 args
  if (buf_out_args_changed) {
    v->set_buf_out_args(buf_out_args_new);
  }
  if (buf_args_changed) {
    v->set_buf_args(buf_args_new);
  }
  if (args_changed) {
    v->set_args(args_new);
  }
  // 返回变异后的对象
  return v;
}

// 对 AllocatePtr 类型对象进行变异操作，返回变异后的对象
StmtPtr IRMutator::mutate(AllocatePtr v) {
  // 获取对象的 buf
  BufPtr buf = v->buf();
  // 对 buf 进行变异
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  // 断言，确保变异后的 buf_new 不为 null
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));
  // 如果 buf 发生了变化，则更新对象的 buf
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 返回变异后的对象
  return v;
}

// 对 FreePtr 类型对象进行变异操作，返回变异后的对象
StmtPtr IRMutator::mutate(FreePtr v) {
  // 获取对象的 buf
  BufPtr buf = v->buf();
  // 对 buf 进行变异
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  // 断言，确保变异后的 buf_new 不为 null
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));
  // 如果 buf 发生了变化，则更新对象的 buf
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  // 返回变异后的对象
  return v;
}

// 对 FreeExtPtr 类型对象进行变异操作，返回变异后的对象
StmtPtr IRMutator::mutate(FreeExtPtr v) {
  // 标志位，指示 bufs 是否有变化
  bool bufs_changed = false;
  // 存储新的 bufs
  std::vector<BufPtr> bufs_new;
  bufs_new.reserve(v->bufs().size());
  // 遍历原 bufs
  for (const auto& buf : v->bufs()) {
    // 对每个 buf 进行变异
    BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
    // 断言，确保变异后的 buf_new 不为 null
    TORCH_INTERNAL_ASSERT(
        buf_new, buildErrorMessage("IRMutator produced null for Buf."));
    // 将变异后的 buf_new 添加到新的列表中
    bufs_new.push_back(buf_new);
    // 更新 bufs_changed 标志位
    bufs_changed |= buf_new != buf;
  }

  // 根据变异情况更新对象的 bufs
  if (bufs_changed) {
    v->set_bufs(bufs_new);
  }
  // 返回变异后的对象
  return v;
}
// 对于给定的 PlacementAllocatePtr v，使用 IRMutator 对象来变异处理。
StmtPtr IRMutator::mutate(PlacementAllocatePtr v) {
  // 获取 v 对应的缓冲区指针 buf
  BufPtr buf = v->buf();
  // 使用当前 mutator 对 buf 进行变异处理，并获取新的缓冲区指针 buf_new
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  // 断言新的 buf_new 非空，否则抛出错误信息
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));
  // 将 v 的 buf 设置为新的 buf_new
  v->set_buf(buf_new);

  // 获取 v 中要重用的缓冲区指针 buf_to_reuse
  BufPtr buf_to_reuse = v->buf_to_reuse();
  // 使用当前 mutator 对 buf_to_reuse 进行变异处理，并获取新的 buf_to_reuse_new
  BufPtr buf_to_reuse_new = to<Buf>(buf_to_reuse->accept_mutator(this));
  // 断言新的 buf_to_reuse_new 非空，否则抛出错误信息
  TORCH_INTERNAL_ASSERT(
      buf_to_reuse_new, buildErrorMessage("IRMutator produced null for Buf."));
  // 将 v 的 buf_to_reuse 设置为新的 buf_to_reuse_new
  v->set_buf_to_reuse(buf_to_reuse_new);

  // 返回变异后的对象 v
  return v;
}

// 对于给定的 LetPtr v，使用 IRMutator 对象来变异处理。
StmtPtr IRMutator::mutate(LetPtr v) {
  // 获取 v 的变量指针 var_old
  VarPtr var_old = v->var();
  // 使用当前 mutator 对 var_old 进行变异处理，并获取新的 var_new
  VarPtr var_new = to<Var>(var_old->accept_mutator(this));

  // 获取 v 的值表达式指针 val_old
  ExprPtr val_old = v->value();
  // 使用当前 mutator 对 val_old 进行变异处理，并获取新的 val_new
  ExprPtr val_new = val_old->accept_mutator(this);

  // 如果 var_old 与 var_new 不同，则将 v 的 var 设置为新的 var_new
  if (var_old != var_new) {
    v->set_var(var_new);
  }
  // 如果 val_old 与 val_new 不同，则将 v 的 value 设置为新的 val_new
  if (val_old != val_new) {
    v->set_val(val_new);
  }
  // 返回变异后的对象 v
  return v;
}

// 对于给定的 CondPtr v，使用 IRMutator 对象来变异处理。
StmtPtr IRMutator::mutate(CondPtr v) {
  // 获取 v 的条件表达式指针 cond_old
  ExprPtr cond_old = v->condition();
  // 获取 v 的 true 分支语句指针 true_old 和 false 分支语句指针 false_old
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();

  // 使用当前 mutator 对 cond_old 进行变异处理，并获取新的 cond_new
  ExprPtr cond_new = cond_old->accept_mutator(this);
  // 如果 true_old 不为空，则使用当前 mutator 对 true_old 进行变异处理，并获取新的 true_new
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  // 如果 false_old 不为空，则使用当前 mutator 对 false_old 进行变异处理，并获取新的 false_new
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;

  // 如果 cond_old 与 cond_new 不同，则将 v 的 condition 设置为新的 cond_new
  if (cond_old != cond_new) {
    v->set_condition(cond_new);
  }

  // 如果 true_old 与 true_new 不同，则将 v 的 true_stmt 设置为新的 true_new
  if (true_old != true_new) {
    v->set_true_stmt(true_new);
  }

  // 如果 false_old 与 false_new 不同，则将 v 的 false_stmt 设置为新的 false_new
  if (false_old != false_new) {
    v->set_false_stmt(false_new);
  }

  // 返回变异后的对象 v
  return v;
}
```