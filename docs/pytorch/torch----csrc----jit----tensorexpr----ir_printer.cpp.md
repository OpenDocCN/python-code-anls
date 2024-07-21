# `.\pytorch\torch\csrc\jit\tensorexpr\ir_printer.cpp`

```
// 包含头文件 ir_printer.h，这是 TensorExpr 库中用于打印 IR 的头文件
#include <torch/csrc/jit/tensorexpr/ir_printer.h>

// 包含用于 IR 简化的头文件 ir_simplifier.h
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

// 包含用于计算中的归约操作的头文件 reduction.h
#include <torch/csrc/jit/tensorexpr/reduction.h>

// 包含表示张量的头文件 tensor.h
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 包含 C++ 的 range 实用工具 irange.h
#include <c10/util/irange.h>

// 包含标准输入输出流的头文件 iostream
#include <iostream>

// 使用 tensorexpr 命名空间
namespace torch::jit::tensorexpr {

// 将 Dtype 类型转换为对应的 C++ 字符串表示
std::string IRPrinter::dtypeToCppString(const Dtype& dtype) {
  return dtype.ToCppString();
}

// 打印表达式 ExprHandle 的方法
void IRPrinter::print(ExprHandle expr) {
  expr.node()->accept(this);
}

// 打印表达式 Expr 的方法
void IRPrinter::print(Expr& expr) {
  expr.accept(this);
}

// 打印语句 Stmt 的方法
void IRPrinter::print(Stmt& stmt) {
  stmt.accept(this);
}

// 将 CompareSelectOperation 枚举转换为对应的字符串表示
std::string IRPrinter::to_string(CompareSelectOperation op) {
  switch (op) {
    case CompareSelectOperation::kEQ:
      return "==";
    case CompareSelectOperation::kNE:
      return "!=";
    case CompareSelectOperation::kGT:
      return ">";
    case CompareSelectOperation::kGE:
      return ">=";
    case CompareSelectOperation::kLT:
      return "<";
    case CompareSelectOperation::kLE:
      return "<=";
    default:
      // 如果操作符不在预期的范围内，则抛出运行时错误
      throw std::runtime_error("invalid compare select operator");
  }
}

// 访问二元操作的模板方法，根据运算符的优先级决定是否包含括号
template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
void visitBinaryOp(
    NodePtr<Op> v,
    const std::string& op_str,
    IRPrinter* printer,
    bool parens = true) {
  std::ostream& os = printer->os();
  // 获取当前节点、左节点和右节点的运算符优先级
  int self_prec = getPrecedence(v->expr_type());
  int lhs_prec = getPrecedence(v->lhs()->expr_type());
  int rhs_prec = getPrecedence(v->rhs()->expr_type());

  // 根据左节点的优先级决定是否加左括号
  if (lhs_prec >= self_prec) {
    os << "(";
  }
  // 访问并打印左节点
  v->lhs()->accept(printer);
  // 根据左节点的优先级决定是否加右括号
  if (lhs_prec >= self_prec) {
    os << ")";
  }

  // 输出操作符和空格
  os << " " << op_str << " ";

  // 根据右节点的优先级决定是否加左括号
  if (rhs_prec >= self_prec) {
    os << "(";
  }
  // 访问并打印右节点
  v->rhs()->accept(printer);
  // 根据右节点的优先级决定是否加右括号
  if (rhs_prec >= self_prec) {
    os << ")";
  }
}

// 访问 AddPtr 类型节点的方法，调用 visitBinaryOp 方法打印加法表达式
void IRPrinter::visit(AddPtr v) {
  visitBinaryOp(v, "+", this);
}

// 访问 SubPtr 类型节点的方法，调用 visitBinaryOp 方法打印减法表达式
void IRPrinter::visit(SubPtr v) {
  visitBinaryOp(v, "-", this);
}

// 访问 MulPtr 类型节点的方法，调用 visitBinaryOp 方法打印乘法表达式
void IRPrinter::visit(MulPtr v) {
  visitBinaryOp(v, "*", this);
}

// 访问 DivPtr 类型节点的方法，调用 visitBinaryOp 方法打印除法表达式
void IRPrinter::visit(DivPtr v) {
  visitBinaryOp(v, "/", this);
}

// 访问 AndPtr 类型节点的方法，调用 visitBinaryOp 方法打印按位与表达式
void IRPrinter::visit(AndPtr v) {
  visitBinaryOp(v, "&", this);
}

// 访问 OrPtr 类型节点的方法，调用 visitBinaryOp 方法打印按位或表达式
void IRPrinter::visit(OrPtr v) {
  visitBinaryOp(v, "|", this);
}

// 访问 XorPtr 类型节点的方法，调用 visitBinaryOp 方法打印按位异或表达式
void IRPrinter::visit(XorPtr v) {
  visitBinaryOp(v, "^", this);
}

// 访问 LshiftPtr 类型节点的方法，调用 visitBinaryOp 方法打印左移表达式
void IRPrinter::visit(LshiftPtr v) {
  visitBinaryOp(v, "<<", this);
}

// 访问 RshiftPtr 类型节点的方法，调用 visitBinaryOp 方法打印右移表达式
void IRPrinter::visit(RshiftPtr v) {
  visitBinaryOp(v, ">>", this);
}

// 访问 ModPtr 类型节点的方法，根据数据类型决定打印取模表达式或调用 mod 函数
void IRPrinter::visit(ModPtr v) {
  if (v->dtype().is_integral()) {
    visitBinaryOp(v, "%", this);
  } else if (v->dtype().is_floating_point()) {
    os() << "mod(" << *v->lhs() << ", " << *v->rhs() << ")";
  } else {
    // 如果数据类型不支持，则抛出运行时错误
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
}
void IRPrinter::visit(MaxPtr v) {
  // 打印 "Max(" 到输出流
  os() << "Max(";
  // 访问 v 指针的左操作数，并递归调用 accept 方法
  v->lhs()->accept(this);
  // 打印 ", " 到输出流
  os() << ", ";
  // 访问 v 指针的右操作数，并递归调用 accept 方法
  v->rhs()->accept(this);
  // 打印 ", " 和 propagate_nans() 的无符号整数结果到输出流
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(MinPtr v) {
  // 打印 "Min(" 到输出流
  os() << "Min(";
  // 访问 v 指针的左操作数，并递归调用 accept 方法
  v->lhs()->accept(this);
  // 打印 ", " 到输出流
  os() << ", ";
  // 访问 v 指针的右操作数，并递归调用 accept 方法
  v->rhs()->accept(this);
  // 打印 ", " 和 propagate_nans() 的无符号整数结果到输出流
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(CompareSelectPtr v) {
  // 获取 CompareSelect 操作类型
  CompareSelectOperation cmp_op = v->compare_select_op();
  // 获取当前表达式类型的优先级
  int self_prec = getPrecedence(v->expr_type());
  // 获取左操作数表达式类型的优先级
  int lhs_prec = getPrecedence(v->lhs()->expr_type());
  // 获取右操作数表达式类型的优先级
  int rhs_prec = getPrecedence(v->rhs()->expr_type());

  // 如果左操作数的优先级大于等于当前表达式的优先级，则在输出流中加上括号
  if (lhs_prec >= self_prec) {
    os() << "(";
  }
  // 递归访问左操作数，并打印到输出流
  v->lhs()->accept(this);
  // 如果左操作数的优先级大于等于当前表达式的优先级，则在输出流中加上括号
  if (lhs_prec >= self_prec) {
    os() << ")";
  }

  // 打印比较操作的字符串表示到输出流
  os() << to_string(cmp_op);

  // 如果右操作数的优先级大于等于当前表达式的优先级，则在输出流中加上括号
  if (rhs_prec >= self_prec) {
    os() << "(";
  }
  // 递归访问右操作数，并打印到输出流
  v->rhs()->accept(this);
  // 如果右操作数的优先级大于等于当前表达式的优先级，则在输出流中加上括号
  if (rhs_prec >= self_prec) {
    os() << ")";
  }
  // 打印 " ? " 到输出流
  os() << " ? ";

  // 定义一个 lambda 函数，用于带括号地访问表达式，并打印到输出流
  auto withParens = [&](ExprPtr e) {
    auto prec = getPrecedence(e->expr_type());
    // 如果表达式的优先级大于等于当前表达式的优先级，则在输出流中加上括号
    if (prec >= self_prec) {
      os() << "(";
    }
    // 递归访问表达式，并打印到输出流
    e->accept(this);
    // 如果表达式的优先级大于等于当前表达式的优先级，则在输出流中加上括号
    if (prec >= self_prec) {
      os() << ")";
    }
  };
  // 调用带括号的访问函数，访问 v 的第一个返回值表达式，并打印到输出流
  withParens(v->ret_val1());
  // 打印 " : " 到输出流
  os() << " : ";
  // 调用带括号的访问函数，访问 v 的第二个返回值表达式，并打印到输出流
  withParens(v->ret_val2());
}

static void formatFPSuffix(std::ostream& os, double v) {
  // 如果 v 是整数，打印 ".0" 到输出流；否则不打印
  os << (v == std::ceil(v) ? ".0" : "");
}

template <typename T>
static void formatFPSuffix(std::ostream& os, T v) {
  // 如果 v 是整数，打印 ".f" 到输出流；否则打印 "f"
  os << (v == std::ceil(v) ? ".f" : "f");
}

template <
    typename T,
    std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static void formatImm(std::ostream& os, T v) {
  // 定义浮点数格式的输出函数，根据值的类型进行输出
  const int precision = 16;
  // 如果 v 是 NaN，打印 "NAN" 到输出流
  if (std::isnan(v)) {
    os << "NAN";
  // 如果 v 是正无穷大，打印 "POS_INFINITY" 到输出流；如果是负无穷大，打印 "NEG_INFINITY"
  } else if (std::isinf(v)) {
    os << (v > 0 ? "POS_INFINITY" : "NEG_INFINITY");
  // 否则，按照指定精度打印 v 的值，并调用 formatFPSuffix 函数添加后缀
  } else {
    os << std::setprecision(precision) << v;
    formatFPSuffix(os, v);
  }
}

static void formatIntSuffix(std::ostream& os, int64_t v) {
  // 打印 "ll" 到输出流
  os << "ll";
}

template <typename T>
static void formatIntSuffix(std::ostream& os, T v) {}

template <
    typename T,
    std::enable_if_t<!std::is_floating_point<T>::value>* = nullptr>
static void formatImm(std::ostream& os, T v) {
  // 定义整数格式的输出函数，根据值的类型进行输出
  os << +v;  // 打印 v 的值到输出流
  formatIntSuffix(os, v);  // 根据类型调用 formatIntSuffix 添加后缀
}

// NOLINTNEXTLINE
#define IMM_PRINT_VISIT(Type, Name)       \
  void IRPrinter::visit(Name##ImmPtr v) { \
    formatImm(os(), v->value());          // 调用 formatImm 输出 Name##ImmPtr 类型对象的值
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT);  // 对所有标量类型生成 IMM_PRINT_VISIT 宏定义
#undef IMM_PRINT_VISIT

void IRPrinter::visit(CastPtr v) {
  auto dtype = v->dtype();
  // 打印类型转换的目标类型到输出流
  os() << dtypeToCppString(dtype) << "(";
  // 递归访问源值表达式，并打印到输出流
  v->src_value()->accept(this);
  // 打印 ")"
  os() << ")";
}

void IRPrinter::visit(BitCastPtr v) {
  auto dtype = v->dtype();
  // 打印 "BitCast<" 和目标类型的 C++ 字符串形式到输出流
  os() << "BitCast<" << dtype.ToCppString() << ">(";
  // 递归访问源值表达式，并打印到输出流
  v->src_value()->accept(this);
  // 打印 ")"
  os() << ")";
}

void IRPrinter::visit(VarPtr v) {
  // 打印变量名的唯一标识符到输出流
  os() << name_manager_.get_unique_name(v);
}

void IRPrinter::visit(BufPtr v) {
  auto dtype = v->dtype();
  // 打印缓冲区的基础句柄到输出流
  os() << *v->base_handle();
  // 打印 "(dtype=" 和缓冲区数据类型的 C++ 字符串形式到输出流
  os() << "(dtype=" << dtypeToCppString(dtype);
  // 如果存在 qscale，打印 ": qscale="
    os() << ", qscale=";

# 输出流中添加文本 ", qscale="，用于指示接下来将输出量化比例的信息


    v->qscale()->accept(this);

# 调用对象v的qscale()方法，将其结果接受并输出到当前访问者对象中

  }
  if (v->qscale()) {

# 检查对象v的qscale()方法返回值是否存在，如果存在执行以下代码块

    os() << ", qzero=";
    v->qzero()->accept(this);

# 输出流中添加文本 ", qzero="，用于指示接下来将输出量化零点的信息，并将对象v的qzero()方法结果接受并输出到当前访问者对象中

  }
  os() << ", sizes=[";

# 输出流中添加文本 ", sizes=["，用于指示接下来将输出尺寸信息的开始

  size_t i = 0;
  for (const ExprPtr& s : v->dims()) {

# 初始化计数器i为0，遍历对象v的dims()方法返回的尺寸表达式列表，并将每个表达式s接受并输出到当前访问者对象中

    if (i++) {
      os() << ", ";
    }
    s->accept(this);

# 如果计数器i不为0，输出流中添加文本 ", "，然后将尺寸表达式s接受并输出到当前访问者对象中

  }
  os() << "]";

# 输出流中添加文本 "]"，用于指示尺寸信息的结束

  os() << ", strides=[";

# 输出流中添加文本 ", strides=["，用于指示接下来将输出步长信息的开始

  i = 0;
  for (const ExprPtr& s : v->strides()) {

# 将计数器i重置为0，遍历对象v的strides()方法返回的步长表达式列表，并将每个表达式s接受并输出到当前访问者对象中

    if (i++) {
      os() << ", ";
    }
    s->accept(this);

# 如果计数器i不为0，输出流中添加文本 ", "，然后将步长表达式s接受并输出到当前访问者对象中

  }
  os() << "]";

# 输出流中添加文本 "]"，用于指示步长信息的结束

  os() << ")";

# 输出流中添加文本 ")"，用于指示当前输出信息的结束
void IRPrinter::visit(RampPtr v) {
  // 打印 Ramp 的信息，包括基址、步长和长度
  os() << "Ramp(" << *v->base() << ", " << *v->stride() << ", " << v->lanes()
       << ")";
}

void IRPrinter::visit(LoadPtr v) {
  // TODO: support the mask case
  // 如果索引为空，则打印基地址
  if (v->indices().empty()) {
    os() << *v->base_handle();
  } else {
    // 否则打印带有索引的基地址
    os() << *v->base_handle() << "[";
    size_t i = 0;
    // 遍历索引并打印
    for (const ExprPtr& ind : v->indices()) {
      if (i++) {
        os() << ", ";
      }
      ind->accept(this);
    }
    // 如果索引为空，则打印默认索引 0
    if (v->indices().empty()) {
      os() << "0";
    }
    os() << "]";
  }
}

void IRPrinter::visit(BroadcastPtr v) {
  // 打印 Broadcast 的信息，包括值和长度
  os() << "Broadcast(" << *v->value() << ", " << v->lanes() << ")";
}

void IRPrinter::visit(IfThenElsePtr v) {
  // 打印 IfThenElse 的条件、真值和假值
  os() << "IfThenElse(" << *v->condition() << ", " << *v->true_value() << ", "
       << *v->false_value() << ")";
}

void IRPrinter::visit(IntrinsicsPtr v) {
  // 打印内置函数的名称和参数列表
  os() << v->func_name() << "(";
  for (const auto i : c10::irange(v->nparams())) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void IRPrinter::visit(TermPtr v) {
  // 打印 Term 的信息，包括标量和变量列表
  os() << "Term(";
  v->scalar()->accept(this);
  for (const auto& t : v->variables()) {
    os() << ",";
    t->accept(this);
  }
  os() << ")";
}

void IRPrinter::visit(PolynomialPtr v) {
  bool first = true;
  // 打印 Polynomial 的信息，包括变量列表和标量
  os() << "Polynomial(";
  for (const auto& t : v->variables()) {
    if (!first) {
      os() << " + ";
    }
    first = false;
    t->accept(this);
  }

  if (!first) {
    os() << " + ";
  }
  v->scalar()->accept(this);
  os() << ")";
}

void IRPrinter::visit(RoundOffPtr v) {
  // 打印 RoundOff 的信息，包括左操作数和右操作数
  os() << "RoundOff(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ")";
}

void IRPrinter::visit(MaxTermPtr v) {
  // 打印 MaxTerm 的信息，包括标量和变量列表
  os() << "MaxTerm(";
  if (v->scalar()) {
    v->scalar()->accept(this);
    os() << ", ";
  }
  for (size_t i = 0; i < v->variables().size(); ++i) {
    v->variables()[i]->accept(this);
    if (i < v->variables().size() - 1) {
      os() << ", ";
    }
  }
  os() << ")";
}

void IRPrinter::visit(MinTermPtr v) {
  // 打印 MinTerm 的信息，包括标量和变量列表
  os() << "MinTerm(";
  if (v->scalar()) {
    v->scalar()->accept(this);
    os() << ", ";
  }
  for (size_t i = 0; i < v->variables().size(); ++i) {
    v->variables()[i]->accept(this);
    if (i < v->variables().size() - 1) {
      os() << ", ";
    }
  }
  os() << ")";
}

void IRPrinter::visit(ReduceOpPtr v) {
  // 打印 ReduceOp 的信息，包括主体和约简参数列表
  os() << "ReduceOp(";
  os() << *v->body() << ", ";

  bool first = true;
  os() << "reduce_args={";
  for (const auto& d : v->reduce_args()) {
    if (!first) {
      os() << ", ";
    }
    os() << *d;
    first = false;
  }
  os() << "})";
}

void IRPrinter::visit(StorePtr v) {
  // TODO: handle the mask
  // 如果索引为空，则打印基地址、赋值操作和分号
  if (v->indices().empty()) {
    os() << *v->base_handle() << " = " << *v->value() << ";";

  // 如果索引不为空，则打印带有索引的基地址、赋值操作和分号
  } else {
    os() << *v->base_handle() << "[";
    size_t i = 0;
    for (const ExprPtr& ind : v->indices()) {
      if (i++) {
        os() << ", ";
      }
      ind->accept(this);
    }
    if (v->indices().empty()) {
      os() << "0";
    }
    os() << "] = " << *v->value() << ";";
  }
}
    // 结束函数的执行，立即返回，不返回任何值
    return;
  }

  // 输出流中打印变量 v 的基础句柄的内容，并追加 "["
  os() << *v->base_handle() << "[";
  // 初始化索引计数器 i 为 0
  size_t i = 0;
  // 遍历变量 v 的索引集合
  for (const ExprPtr& ind : v->indices()) {
    // 如果不是第一个索引，则在输出流中追加 ", "
    if (i++) {
      os() << ", ";
    }
    // 调用索引表达式对象的 accept 方法进行访问
    ind->accept(this);
  }
  // 如果变量 v 的索引集合为空，则在输出流中打印 "0"
  if (v->indices().empty()) {
    os() << "0";
  }
  // 输出流中追加 "] = " 并打印变量 v 的值
  os() << "] = " << *v->value() << ";";
void IRPrinter::visit(ForPtr v) {
  // 获取循环的迭代变量
  VarPtr var = v->var();
  // 创建变量处理器
  VarHandle vv(var);
  // 输出循环的起始部分
  os() << "for (" << dtypeToCppString(var->dtype()) << " " << vv << " = "
       << ExprHandle(v->start()) << "; " << vv << " < " << ExprHandle(v->stop())
       << "; " << vv << "++) ";
  // 获取循环的选项并输出
  std::string loop_options_str = v->loop_options().ToString();
  if (!loop_options_str.empty()) {
    os() << " /* " << loop_options_str << " */";
  }
  // 输出循环体
  if (v->body()) {
    os() << *v->body();
  } else {
    os() << "{}";
  }
}

void IRPrinter::visit(BlockPtr v) {
  // 输出块的起始部分
  os() << "{\n";
  // 增加缩进
  indent_++;

  // 遍历块中的每一个语句并输出
  for (const StmtPtr& s : *v) {
    emitIndent();  // 输出当前缩进
    os() << *s << "\n";  // 输出语句
  }

  // 减少缩进
  indent_--;
  // 输出块的结束部分，并确保格式正确
  emitIndent();
  os() << "}";
}

void IRPrinter::visit(AllocatePtr v) {
  // 输出分配操作的信息，包括缓冲变量和数据类型
  os() << "Allocate(" << *v->buffer_var()
       << "); // dtype=" << dtypeToCppString(v->dtype());
  // 输出维度信息
  os() << ", dims=[";
  const std::vector<ExprPtr>& dims = v->dims();
  for (const auto i : c10::irange(dims.size())) {
    if (i != 0) {
      os() << ", ";
    }
    os() << *dims[i];
  }
  os() << "]";
}

void IRPrinter::visit(FreePtr v) {
  // 输出释放操作的信息，包括缓冲变量
  os() << "Free(" << *v->buffer_var() << ");";
}

void IRPrinter::visit(FreeExtPtr v) {
  // 输出扩展的释放操作的信息，包括所有缓冲变量
  os() << "FreeExt(bufs={";
  int i = 0;
  for (const auto& buf : v->bufs()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf;
  }

  os() << "});";
}

void IRPrinter::visit(PlacementAllocatePtr v) {
  // 输出别名分配操作的信息，包括源缓冲和目标缓冲
  os() << "Alias(" << *v->buf()->base_handle() << ","
       << *v->buf_to_reuse()->base_handle() << ");";
}

void IRPrinter::visit(LetPtr v) {
  // 输出变量声明和初始化的信息，包括数据类型、变量名和初始值
  os() << dtypeToCppString(v->var()->dtype()) << " " << *v->var();
  os() << " = " << *v->value() << ";";
}

void IRPrinter::visit(CondPtr v) {
  // 输出条件语句的信息，包括条件表达式和对应的真假分支
  ExprPtr cond = v->condition();
  StmtPtr true_stmt = v->true_stmt();
  StmtPtr false_stmt = v->false_stmt();
  if (!true_stmt) {
    os() << "if (!" << *cond << ") ";
    os() << *false_stmt;
  } else {
    os() << "if (" << *cond << ") ";
    os() << *true_stmt;
    if (false_stmt) {
      os() << " else ";
      os() << *false_stmt;
    }
  }
}

void IRPrinter::visit(AtomicAddPtr v) {
  // 输出原子加操作的信息，包括基础句柄和索引
  os() << "atomicAdd(&" << *v->base_handle() << "[";
  size_t i = 0;
  for (const ExprPtr& ind : v->indices()) {
    if (i++) {
      os() << ", ";
    }
    ind->accept(this);  // 访问并输出索引表达式
  }
  if (v->indices().empty()) {
    os() << "0";
  }
  os() << "], " << *v->value() << ");";
}

void IRPrinter::visit(SyncThreadsPtr v) {
  // 输出线程同步操作的信息
  os() << "__syncthreads();";
}

void IRPrinter::visit(ExternalCallPtr v) {
  // 输出外部调用的信息，包括缓冲参数和普通参数
  os() << *v->buf() << " = " << v->func_name() << "(";

  os() << "buf_args={";
  int i = 0;
  for (const BufPtr& buf_arg : v->buf_args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf_arg;
  }

  os() << "}, args={";
  i = 0;
  for (const ExprPtr& arg : v->args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *arg;
  }
  os() << "})";
}

void IRPrinter::visit(ExternalCallWithAllocPtr v) {
  // 输出带有分配的外部调用的信息，包括输出缓冲参数
  int i = 0;
  for (const auto& buf_out_arg : v->buf_out_args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    // 输出每个输出缓冲参数
    os() << *buf_out_arg;
  }
  // 其余信息由 IRPrinter 类的其余部分处理
}
  // 将 buf_out_arg 的内容输出到 os 流中
  os() << *buf_out_arg;
}

// 输出函数名称及其参数到 os 流中
os() << " := " << v->func_name() << "(";

// 输出 buf_args 的内容到 os 流中
os() << "buf_args={";
i = 0;
for (const auto& buf_arg : v->buf_args()) {
  if (i++ > 0) {
    os() << ", ";
  }
  os() << *buf_arg;
}

// 输出 args 的内容到 os 流中
os() << "}, args={";
i = 0;
for (const auto& arg : v->args()) {
  if (i++ > 0) {
    os() << ", ";
  }
  os() << *arg;
}
// 输出函数调用的结尾括号到 os 流中
os() << "})";
} // 结束 torch::jit::tensorexpr 命名空间

void IRPrinter::emitIndent() {
  // 输出当前缩进，用空格填充，数量为当前缩进值的两倍
  os() << std::setw(2 * indent_) << "";
}

std::ostream& operator<<(std::ostream& stream, const ExprHandle& expr) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  ExprHandle& mutable_expr = const_cast<ExprHandle&>(expr);
  // 如果流是 PrinterStream 类型，则调用表达式节点的接受方法打印到流中
  if (printer_stream != nullptr) {
    mutable_expr.node()->accept(printer_stream->printer());
  } else {
    // 否则创建 IRPrinter 对象 p，并打印可变表达式 mutable_expr
    IRPrinter p(stream);
    p.print(mutable_expr);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Expr& expr) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  Expr& mutable_expr = const_cast<Expr&>(expr);
  // 如果流是 PrinterStream 类型，则调用表达式的接受方法打印到流中
  if (printer_stream != nullptr) {
    mutable_expr.accept(printer_stream->printer());
  } else {
    // 否则创建 IRPrinter 对象 p，并打印可变表达式 mutable_expr
    IRPrinter p(stream);
    p.print(mutable_expr);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Stmt& stmt) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  Stmt& mutable_stmt = const_cast<Stmt&>(stmt);
  // 如果流是 PrinterStream 类型，则调用语句的接受方法打印到流中
  if (printer_stream != nullptr) {
    mutable_stmt.accept(printer_stream->printer());
  } else {
    // 否则创建 IRPrinter 对象 p，并打印可变语句 mutable_stmt
    IRPrinter p(stream);
    p.print(mutable_stmt);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Tensor& t) {
  // 将 Tensor 对象 t 转换为字符串并输出到流中
  stream << std::to_string(t);
  return stream;
}

void print(ExprPtr expr) {
  // 如果表达式非空，则创建 IRPrinter 对象 p，并打印表达式
  if (expr) {
    IRPrinter p(std::cout);
    p.print(*expr);
  } else {
    // 否则输出 (null expr)
    std::cout << "(null expr)";
  }
  std::cout << "\n";
}

void print(StmtPtr stmt) {
  // 如果语句非空，则创建 IRPrinter 对象 p，并打印语句
  if (stmt) {
    IRPrinter p(std::cout);
    p.print(*stmt);
  } else {
    // 否则输出 (null stmt)
    std::cout << "(null stmt)\n";
  }
}

void print(const Tensor& t) {
  // 输出 Tensor 对象 t 的字符串表示
  std::cout << std::to_string(t);
}

} // 结束 std 命名空间
```