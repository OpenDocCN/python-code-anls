# `.\pytorch\torch\csrc\jit\tensorexpr\ir_visitor.cpp`

```py
// 包含TensorExpr库中的头文件，用于访问IR的结构和操作
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

// 包含TensorExpr库中其他相关的IR头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 包含C++标准库头文件
#include <c10/util/irange.h>

// 定义了命名空间torch::jit::tensorexpr
namespace torch::jit::tensorexpr {

// 模板函数，用于访问二元操作符的IR节点
template <
    typename Op,
    // 当Op的推导结果为void类型时启用该函数模板
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static void visit_binary_op(NodePtr<Op> v, IRVisitor* visitor) {
  // 访问二元操作符的左操作数节点
  v->lhs()->accept(visitor);
  // 访问二元操作符的右操作数节点
  v->rhs()->accept(visitor);
}

// 实现IRVisitor类的成员函数visit，用于访问Add节点的IR结构
void IRVisitor::visit(AddPtr v) {
  // 调用通用的二元操作访问函数visit_binary_op，传入当前节点和访问者对象
  visit_binary_op(v, this);
}

// 以下各函数实现类似于visit(AddPtr v)函数，用于访问对应类型的IR节点

void IRVisitor::visit(SubPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(MulPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(DivPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(ModPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(MaxPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(MinPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(AndPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(OrPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(XorPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(LshiftPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(RshiftPtr v) {
  visit_binary_op(v, this);
}

// 实现visit函数，用于访问CompareSelect节点的IR结构
void IRVisitor::visit(CompareSelectPtr v) {
  // 访问CompareSelect节点的左操作数节点
  v->lhs()->accept(this);
  // 访问CompareSelect节点的右操作数节点
  v->rhs()->accept(this);
  // 访问CompareSelect节点的第一个返回值节点
  v->ret_val1()->accept(this);
  // 访问CompareSelect节点的第二个返回值节点
  v->ret_val2()->accept(this);
}

// 宏定义IMM_VISIT展开为visit函数，用于访问各种类型的立即数节点
#define IMM_VISIT(Type, Name) \
  void IRVisitor::visit(Name##ImmPtr v) {}

// 用于展开各种标量类型（Scalar types）的立即数访问函数
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT);

// 取消IMM_VISIT宏定义
#undef IMM_VISIT

// 实现visit函数，用于访问Cast节点的IR结构
void IRVisitor::visit(CastPtr v) {
  // 访问Cast节点的源数值节点
  v->src_value()->accept(this);
}

// 实现visit函数，用于访问BitCast节点的IR结构
void IRVisitor::visit(BitCastPtr v) {
  // 访问BitCast节点的源数值节点
  v->src_value()->accept(this);
}

// 实现visit函数，用于访问Var节点的IR结构
void IRVisitor::visit(VarPtr v) {
  // Var节点没有子节点，无需进一步访问
}

// 实现visit函数，用于访问Ramp节点的IR结构
void IRVisitor::visit(RampPtr v) {
  // 访问Ramp节点的基础值节点
  v->base()->accept(this);
  // 访问Ramp节点的步长节点
  v->stride()->accept(this);
}

// 实现visit函数，用于访问Load节点的IR结构
void IRVisitor::visit(LoadPtr v) {
  // 访问Load节点的缓冲区节点
  v->buf()->accept(this);
  // 遍历访问Load节点的所有索引节点
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
}

// 实现visit函数，用于访问Buf节点的IR结构
void IRVisitor::visit(BufPtr v) {
  // 访问Buf节点的基础句柄节点
  v->base_handle()->accept(this);
  // 如果存在量化比例节点，访问该节点
  if (v->qscale()) {
    v->qscale()->accept(this);
  }
  // 如果存在量化零点节点，访问该节点
  if (v->qzero()) {
    v->qzero()->accept(this);
  }
}

// 实现visit函数，用于访问Store节点的IR结构
void IRVisitor::visit(StorePtr v) {
  // 访问Store节点的缓冲区节点
  v->buf()->accept(this);
  // 遍历访问Store节点的所有索引节点
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
  // 访问Store节点的数值节点
  v->value()->accept(this);
}

// 实现visit函数，用于访问AtomicAdd节点的IR结构
void IRVisitor::visit(AtomicAddPtr v) {
  // 访问AtomicAdd节点的缓冲区节点
  v->buf()->accept(this);
  // 遍历访问AtomicAdd节点的所有索引节点
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
  // 访问AtomicAdd节点的数值节点
  v->value()->accept(this);
}

// 实现visit函数，用于访问SyncThreads节点的IR结构
void IRVisitor::visit(SyncThreadsPtr v) {
  // SyncThreads节点无需访问子节点
}

// 实现visit函数，用于访问ExternalCall节点的IR结构
void IRVisitor::visit(ExternalCallPtr v) {
  // 访问ExternalCall节点的缓冲区节点
  v->buf()->accept(this);
  // 遍历访问ExternalCall节点的所有缓冲区参数节点
  for (const BufPtr& buf_arg : v->buf_args()) {
    buf_arg->accept(this);
  }
  // 遍历访问ExternalCall节点的所有参数节点
  for (const ExprPtr& arg : v->args()) {
    arg->accept(this);
  }
}

// 结束命名空间torch::jit::tensorexpr
}
void IRVisitor::visit(ExternalCallWithAllocPtr v) {
  // 遍历外部调用的输出缓冲区参数，并对每个参数执行访问操作
  for (const auto& buf_out_arg : v->buf_out_args()) {
    buf_out_arg->accept(this);
  }
  // 遍历外部调用的缓冲区参数，并对每个参数执行访问操作
  for (const auto& buf_arg : v->buf_args()) {
    buf_arg->accept(this);
  }
  // 遍历外部调用的常规参数，并对每个参数执行访问操作
  for (const auto& arg : v->args()) {
    arg->accept(this);
  }
}

void IRVisitor::visit(FreeExtPtr v) {
  // 遍历 FreeExtPtr 对象的缓冲区列表，并对每个缓冲区执行访问操作
  for (const auto& buf : v->bufs()) {
    buf->accept(this);
  }
}

void IRVisitor::visit(BlockPtr v) {
  // 遍历 BlockPtr 对象包含的语句列表，并对每个语句执行访问操作
  for (const StmtPtr& s : *v) {
    s->accept(this);
  }
}

void IRVisitor::visit(ForPtr v) {
  // 访问 For 循环中的循环变量
  v->var()->accept(this);
  // 访问 For 循环中的起始值表达式
  v->start()->accept(this);
  // 访问 For 循环中的终止值表达式
  v->stop()->accept(this);
  // 如果存在循环体，则对循环体执行访问操作
  if (v->body()) {
    v->body()->accept(this);
  }
}

void IRVisitor::visit(BroadcastPtr v) {
  // 访问 BroadcastPtr 对象中的值表达式
  v->value()->accept(this);
}

void IRVisitor::visit(IfThenElsePtr v) {
  // 访问 IfThenElsePtr 对象中的条件表达式
  v->condition()->accept(this);
  // 访问 IfThenElsePtr 对象中的真值分支
  v->true_value()->accept(this);
  // 访问 IfThenElsePtr 对象中的假值分支
  v->false_value()->accept(this);
}

void IRVisitor::visit(IntrinsicsPtr v) {
  // 遍历 IntrinsicsPtr 对象的参数列表，并对每个参数执行访问操作
  for (const auto i : c10::irange(v->nparams())) {
    v->param(i)->accept(this);
  }
}

void IRVisitor::visit(AllocatePtr v) {
  // 访问 AllocatePtr 对象中的缓冲区变量
  v->buffer_var()->accept(this);
  // 遍历 AllocatePtr 对象中的维度列表，并对每个维度执行访问操作
  std::vector<ExprPtr> dims = v->dims();
  for (const ExprPtr& dim : dims) {
    dim->accept(this);
  }
}

void IRVisitor::visit(FreePtr v) {
  // 访问 FreePtr 对象中的缓冲区变量
  v->buffer_var()->accept(this);
}

void IRVisitor::visit(PlacementAllocatePtr v) {
  // 访问 PlacementAllocatePtr 对象中的 buf 成员
  v->buf()->accept(this);
  // 访问 PlacementAllocatePtr 对象中的 buf_to_reuse 成员
  v->buf_to_reuse()->accept(this);
}

void IRVisitor::visit(LetPtr v) {
  // 访问 LetPtr 对象中的变量
  v->var()->accept(this);
  // 访问 LetPtr 对象中的值表达式
  v->value()->accept(this);
}

void IRVisitor::visit(CondPtr v) {
  // 访问 CondPtr 对象中的条件表达式
  ExprPtr condition = v->condition();
  condition->accept(this);
  // 如果存在真值分支，则对其执行访问操作
  StmtPtr true_stmt = v->true_stmt();
  if (true_stmt) {
    true_stmt->accept(this);
  }
  // 如果存在假值分支，则对其执行访问操作
  StmtPtr false_stmt = v->false_stmt();
  if (false_stmt) {
    false_stmt->accept(this);
  }
}

void IRVisitor::visit(TermPtr v) {
  // 访问 TermPtr 对象中的标量表达式
  v->scalar()->accept(this);
  // 遍历 TermPtr 对象中的变量列表，并对每个变量执行访问操作
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(PolynomialPtr v) {
  // 访问 PolynomialPtr 对象中的标量表达式
  v->scalar()->accept(this);
  // 遍历 PolynomialPtr 对象中的变量列表，并对每个变量执行访问操作
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(RoundOffPtr v) {
  // 访问 RoundOffPtr 对象中的左操作数表达式
  v->lhs()->accept(this);
  // 访问 RoundOffPtr 对象中的右操作数表达式
  v->rhs()->accept(this);
}

void IRVisitor::visit(MaxTermPtr v) {
  // 如果 MaxTermPtr 对象中存在标量表达式，则对其执行访问操作
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  // 遍历 MaxTermPtr 对象中的变量列表，并对每个变量执行访问操作
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(MinTermPtr v) {
  // 如果 MinTermPtr 对象中存在标量表达式，则对其执行访问操作
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  // 遍历 MinTermPtr 对象中的变量列表，并对每个变量执行访问操作
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(ReduceOpPtr v) {
  // 访问 ReduceOpPtr 对象中的主体表达式
  v->body()->accept(this);
  // 遍历 ReduceOpPtr 对象中的 reduce_args 列表，并对每个参数执行访问操作
  for (const auto& r : v->reduce_args()) {
    r->accept(this);
  }
}

} // namespace torch::jit::tensorexpr
```