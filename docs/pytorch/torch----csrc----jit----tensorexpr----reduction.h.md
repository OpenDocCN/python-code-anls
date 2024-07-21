# `.\pytorch\torch\csrc\jit\tensorexpr\reduction.h`

```py
#pragma once
// 引入TensorExpr的相关头文件
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <functional>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

// 使用ParameterList作为变量句柄的常量向量
using ParameterList = const std::vector<VarHandle>;
// ReduceInteraction是一个函数类型，描述了如何互动以及积累每个值的方式
using ReduceInteraction = std::function<ExprHandle(ExprHandle, ExprHandle)>;

// Reducer类是一个用户接口，描述了特定的减少操作
// 它有三个组件：初始化值、与累积值互动的方式以及获取当前要减少的值的方法
// 当循环变量已知时，它会实现为ReduceOp
class TORCH_API Reducer {
 public:
  // 构造函数，初始化Reducer对象
  Reducer(ExprHandle init, ReduceInteraction& interaction)
      : init_(init.node()), interaction_(interaction) {}

  // 模板构造函数，用于不同类型的interaction
  template <typename RI>
  Reducer(ExprHandle init, RI interaction)
      : init_(init.node()), interaction_(std::move(interaction)) {}

  // 返回初始化表达式节点
  ExprPtr initializer() const {
    return init_;
  }

  // 重载操作符，用于构造ReduceOp
  ExprHandle operator()(
      BufHandle result_buf,
      ExprHandle body,
      const std::vector<ExprHandle>& output,
      const std::vector<VarHandle>& inner) const;

  // 重载操作符，用于构造ReduceOp
  ReduceOpPtr operator()(
      BufPtr result_buf,
      ExprPtr body,
      const std::vector<ExprPtr>& output,
      const std::vector<VarPtr>& inner) const;

  // 重载操作符，用于构造ReduceOp
  ExprHandle operator()(
      BufHandle result_buf,
      BufHandle acc_buf,
      ExprHandle body,
      const std::vector<ExprHandle>& output,
      const std::vector<VarHandle>& inner) const;

  // 处理具有多种参数形式的Body函数的多态性
  static ExprHandle getReduceBody(
      const std::function<ExprHandle(ParameterList&)>& func,
      const std::vector<VarHandle>& vars) {
    return func(vars);
  }

  // 处理具有单个参数形式的Body函数的多态性
  static ExprHandle getReduceBody(
      const std::function<ExprHandle(const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 1) {
      throw malformed_input("mismatch between reduce body and arg size (1)");
    }

    return func(vars[0]);
  }

  // 处理具有两个参数形式的Body函数的多态性
  static ExprHandle getReduceBody(
      const std::function<ExprHandle(const VarHandle&, const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 2) {
      throw malformed_input("mismatch between reduce body and arg size (2)");
    }
    return func(vars[0], vars[1]);
  }

  // 处理具有三个参数形式的Body函数的多态性
  static ExprHandle getReduceBody(
      const std::function<
          ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
          func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 3) {
      throw malformed_input("mismatch between reduce body and arg size (3)");
    }


注意：这里只给出了部分代码的注释，根据需要继续完善其余部分的注释。
  // 调用函数 func 处理给定的变量，并返回结果
  return func(vars[0], vars[1], vars[2], vars[3]);
}

static ExprHandle getReduceBody(
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& func,
    const std::vector<VarHandle>& vars) {
  // 检查变量数量是否为4，否则抛出异常
  if (vars.size() != 4) {
    throw malformed_input("mismatch between reduce body and arg size (4)");
  }
  // 调用函数 func 处理给定的变量，并返回结果
  return func(vars[0], vars[1], vars[2], vars[3]);
}

// 完成 reduce 操作符，通过应用交互函数到累加器和体表达式
static ExprPtr complete(
    BufPtr accumulator,
    ReduceInteraction interaction,
    ExprHandle body,
    const std::vector<ExprPtr>& output_args,
    const std::vector<VarPtr>& reduce_args) {
  // 创建表达式处理 accum
  ExprHandle accum =
      ExprHandle(alloc<Load>(body.dtype(), accumulator, output_args));
  // 调用交互函数 interaction 处理 accum 和 body，返回结果
  auto e = interaction(std::move(accum), std::move(body));
  // 返回表达式节点 e
  return e.node();
}

// 完成 reduce 操作符，通过应用交互函数到累加器和体表达式
static ExprHandle complete(
    BufHandle accumulator,
    ReduceInteraction interaction,
    ExprHandle body,
    const std::vector<ExprHandle>& output_args,
    const std::vector<VarHandle>& reduce_args) {
  // 创建表达式处理 accum
  ExprHandle accum = Load::make(body.dtype(), accumulator, output_args);
  // 调用交互函数 interaction 处理 accum 和 body，返回结果
  auto e = interaction(std::move(accum), std::move(body));
  // 返回表达式 e
  return e;
}
};

// 表示一个归约操作（如求和、取最大值）分解为其组成部分的表达式：初始化、累积变量、获取要归约的值和交互。
//
// 这意味着在循环嵌套中扩展，而不是生成代码。
class TORCH_API ReduceOp : public ExprNode<ReduceOp> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ReduceOp(
      ExprPtr body,                            // 归约操作的主体表达式
      std::vector<VarPtr> reduce_args,         // 归约轴关联的变量集合
      const Reducer& reducer)                  // 原始的归约器工厂
      : ExprNodeBase(body->dtype()),           // 表达式节点基类初始化
        body_(body),                           // 主体表达式
        reduce_args_(std::move(reduce_args)),  // 归约轴变量集合
        reducer_(reducer) {                    // 归约器
    result_buf_ = nullptr;                     // 初始化结果缓冲区为 nullptr
    acc_buf_ = nullptr;                        // 初始化累积缓冲区为 nullptr
    ri_operand_ = nullptr;                     // 初始化交互操作数为 nullptr
  }

  ReduceOp(
      ExprPtr body,
      std::vector<VarPtr> reduce_args,
      BufPtr result_buf,
      BufPtr acc_buf,
      ExprPtr ri_operand,
      const Reducer& reducer)
      : ExprNodeBase(body->dtype()),
        body_(body),
        reduce_args_(std::move(reduce_args)),
        result_buf_(std::move(result_buf)),
        acc_buf_(std::move(acc_buf)),
        ri_operand_(std::move(ri_operand)),
        reducer_(reducer) {}

  static ExprHandle make(
      ExprHandle body,
      std::vector<VarHandle> reduce_args,
      const Reducer& reducer);

  static ExprHandle make(
      ExprHandle body,
      std::vector<VarHandle> reduce_args,
      BufHandle result_buf,
      BufHandle acc_buf,
      ExprHandle ri_operand,
      const Reducer& reducer);

  // 返回获取要归约的值的主体表达式。
  ExprPtr body() const {
    return body_;
  }

  // 返回可以创建 ReduceOp 的原始归约器工厂。
  const Reducer& reducer() const {
    return reducer_;
  }

  // 返回与归约轴关联的变量集合。
  const std::vector<VarPtr>& reduce_args() const {
    return reduce_args_;
  }

  // 设置累积缓冲区。
  void setAccBuf(BufHandle acc_buf) {
    acc_buf_ = acc_buf.node();
  }

  // 返回累积缓冲区。
  BufPtr getAccBuf() {
    return acc_buf_;
  }

  // 设置结果缓冲区。
  void setResultBuf(BufHandle buf) {
    result_buf_ = buf.node();
  }

  // 返回结果缓冲区。
  BufPtr getResultBuf() {
    return result_buf_;
  }

  // 设置交互操作数。
  void setRiOperand(ExprHandle ri_operand) {
    ri_operand_ = ri_operand.node();
  }

  // 返回交互操作数。
  ExprPtr getRiOperand() {
    return ri_operand_;
  }

 private:
  // body_ = reducer_->interaction_(result_buf_, ri_operand_)
  ExprPtr body_;                             // 主体表达式
  std::vector<VarPtr> reduce_args_;           // 归约轴关联的变量集合

  BufPtr result_buf_;                         // 结果缓冲区
  BufPtr acc_buf_;                            // 累积缓冲区
  ExprPtr ri_operand_;                        // 交互操作数

  const Reducer reducer_;                     // 归约器
};

// 表示求和操作的归约器子类。
class Sum : public Reducer {
 public:
  Sum()
      : Reducer(ExprHandle(0), [](ExprHandle a, ExprHandle b) {  // 初始化求和归约器
          return a + b;                         // 返回两个表达式相加的结果
        }) {}
};

// 返回指定类型的最大值表达式。
inline ExprHandle maximumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::max());  // 返回特定类型的最大值表达式
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    # 如果遇到未支持的数据类型，抛出异常 unsupported_dtype()
    default:
      throw unsupported_dtype();
  }
  # 返回一个空的表达式句柄
  return ExprHandle();
}

// 内联函数，返回指定类型的最小值表达式
inline ExprHandle minimumVal(ScalarType type) {
  // 根据类型使用 switch 语句返回相应的最小值表达式
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::min());
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    // 默认情况下抛出不支持的数据类型异常
    default:
      throw unsupported_dtype();
  }
}

// Maximum 类，继承自 Reducer
class Maximum : public Reducer {
 public:
  // 构造函数，根据数据类型初始化最小值
  Maximum(Dtype dtype)
      : Reducer(
            minimumVal(dtype.scalar_type()),  // 调用 minimumVal 函数获取最小值表达式作为初始值
            [](ExprHandle a, ExprHandle b) { return Max::make(a, b, true); }) {}  // Lambda 函数，定义最大值操作
  // 构造函数，使用指定的初始值进行初始化
  Maximum(ExprHandle initializer)
      : Reducer(initializer, [](ExprHandle a, ExprHandle b) {
          return Max::make(a, b, true);  // Lambda 函数，定义最大值操作
        }) {}
};

// Minimum 类，继承自 Reducer
class Minimum : public Reducer {
 public:
  // 构造函数，根据数据类型初始化最大值
  Minimum(Dtype dtype)
      : Reducer(
            maximumVal(dtype.scalar_type()),  // 调用 maximumVal 函数获取最大值表达式作为初始值
            [](ExprHandle a, ExprHandle b) { return Min::make(a, b, true); }) {}  // Lambda 函数，定义最小值操作
  // 构造函数，使用指定的初始值进行初始化
  Minimum(ExprHandle initializer)
      : Reducer(initializer, [](ExprHandle a, ExprHandle b) {
          return Min::make(a, b, true);  // Lambda 函数，定义最小值操作
        }) {}
};

// ReductionExpander 类，继承自 IRMutator
class ReductionExpander : public IRMutator {
 public:
  // expand 函数，接受一个语句指针并进行扩展处理
  StmtPtr expand(StmtPtr s) {
    return s->accept_mutator(this);  // 调用 accept_mutator 方法处理语句
  }

  // mutate 函数，重载 IRMutator 中的 mutate 方法，处理 ReduceOpPtr 类型对象
  ExprPtr mutate(ReduceOpPtr v) override {
    return v->body();  // 返回 ReduceOpPtr 对象的 body
  }
};

// namespace tensorexpr
} // namespace jit
} // namespace torch
```