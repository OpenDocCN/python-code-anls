# `.\pytorch\torch\csrc\jit\tensorexpr\reduction.cpp`

```
// 包含头文件 <torch/csrc/jit/tensorexpr/reduction.h>
// 包含头文件 <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 定义命名空间 torch::jit::tensorexpr
namespace torch::jit::tensorexpr {

// 实现 Reducer 类的 operator() 函数，返回一个表达式句柄 ExprHandle
ExprHandle Reducer::operator()(
    BufHandle result_buf,
    ExprHandle body,
    const std::vector<ExprHandle>& output,
    const std::vector<VarHandle>& inner) const {
  // 调用 ReduceOp::make 函数，构造一个 ReduceOp 对象并返回
  return ReduceOp::make(
      complete(result_buf, interaction_, body, output, inner), inner, *this);
}

// 实现 Reducer 类的 operator() 函数，返回一个 ReduceOpPtr 智能指针
ReduceOpPtr Reducer::operator()(
    BufPtr result_buf,
    ExprPtr body,
    const std::vector<ExprPtr>& output,
    const std::vector<VarPtr>& inner) const {
  // 调用 alloc<ReduceOp> 函数，分配一个 ReduceOp 对象并返回其智能指针
  return alloc<ReduceOp>(
      complete(result_buf, interaction_, ExprHandle(body), output, inner),
      inner,
      *this);
}

// 实现 Reducer 类的 operator() 函数，返回一个表达式句柄 ExprHandle
ExprHandle Reducer::operator()(
    BufHandle result_buf,
    BufHandle acc_buf,
    ExprHandle body,
    const std::vector<ExprHandle>& output,
    const std::vector<VarHandle>& inner) const {
  // 调用 ReduceOp::make 函数，构造一个 ReduceOp 对象并返回
  return ReduceOp::make(
      complete(result_buf, interaction_, body, output, inner),
      inner,
      result_buf,
      acc_buf,
      body,
      *this);
}

// 定义 ReduceOp 类的静态函数 make，返回一个表达式句柄 ExprHandle
ExprHandle ReduceOp::make(
    ExprHandle body,
    std::vector<VarHandle> reduce_args,
    const Reducer& reducer) {
  // 调用 alloc<ReduceOp> 函数，分配一个 ReduceOp 对象并返回其表达式句柄
  return ExprHandle(alloc<ReduceOp>(
      body.node(), VarHandleVectorToVarVector(reduce_args), reducer));
}

// 定义 ReduceOp 类的静态函数 make，返回一个表达式句柄 ExprHandle
ExprHandle ReduceOp::make(
    ExprHandle body,
    std::vector<VarHandle> reduce_args,
    BufHandle result_buf,
    BufHandle acc_buf,
    ExprHandle ri_operand,
    const Reducer& reducer) {
  // 调用 alloc<ReduceOp> 函数，分配一个 ReduceOp 对象并返回其表达式句柄
  return ExprHandle(alloc<ReduceOp>(
      body.node(),
      VarHandleVectorToVarVector(reduce_args),
      result_buf.node(),
      acc_buf.node(),
      ri_operand.node(),
      reducer));
}

} // namespace torch::jit::tensorexpr
```