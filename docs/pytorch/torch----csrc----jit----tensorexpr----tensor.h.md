# `.\pytorch\torch\csrc\jit\tensorexpr\tensor.h`

```py
#pragma once

#include <torch/csrc/Export.h>  // 引入 Torch 导出宏
#include <functional>           // 引入函数对象库
#include <utility>              // 引入实用工具库
#include <vector>               // 引入向量容器库

#include <torch/csrc/jit/tensorexpr/expr.h>       // 引入表达式头文件
#include <torch/csrc/jit/tensorexpr/reduction.h>  // 引入约简操作头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API Tensor {
 public:
  // 构造函数，接受缓冲区指针、变量指针向量和表达式指针，初始化成员变量
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Tensor(BufPtr buf, const std::vector<VarPtr>& args, ExprPtr body)
      : buf_(std::move(buf)) {
    // 构造语句对象，并赋值给成员变量
    stmt_ = constructStmt(args, std::move(body), {}, {});
  }

  // 构造函数，接受缓冲区句柄、变量句柄向量和表达式句柄，调用上述构造函数
  Tensor(BufHandle buf, const std::vector<VarHandle>& args, ExprHandle body)
      : Tensor(buf.node(), VarHandleVectorToVarVector(args), body.node()) {}

  // 构造函数，接受缓冲区指针、变量指针向量、约简维度向量和约简变量向量，初始化成员变量
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Tensor(
      BufPtr buf,
      const std::vector<VarPtr>& args,
      const std::vector<ExprPtr>& reduce_dims,
      const std::vector<VarPtr>& reduce_args,
      ExprPtr body)
      : buf_(std::move(buf)) {
    // 构造语句对象，并赋值给成员变量
    stmt_ = constructStmt(args, std::move(body), reduce_dims, reduce_args);
  }

  // 构造函数，接受缓冲区句柄、变量句柄向量、约简维度句柄向量和约简变量句柄向量，调用上述构造函数
  Tensor(
      BufHandle buf,
      const std::vector<VarHandle>& args,
      const std::vector<ExprHandle>& reduce_dims,
      const std::vector<VarHandle>& reduce_args,
      ExprHandle body)
      : Tensor(
            buf.node(),
            VarHandleVectorToVarVector(args),
            ExprHandleVectorToExprVector(reduce_dims),
            VarHandleVectorToVarVector(reduce_args),
            body.node()) {}

  // 构造函数，接受缓冲区指针和语句指针，初始化成员变量
  Tensor(BufPtr buf, StmtPtr stmt)
      : buf_(std::move(buf)), stmt_(std::move(stmt)) {}

  // 返回成员变量 buf_
  BufPtr buf() const {
    return buf_;
  }

  // 返回成员变量 stmt_
  StmtPtr stmt() const {
    return stmt_;
  }

  // 成员函数模板，根据参数向量加载表达式
  template <typename T>
  inline ExprHandle load(const std::vector<T>& args) const;
  
  // 成员函数模板，根据变长参数加载表达式
  template <typename... Ts>
  inline ExprHandle load(const Ts&... ts) const;

 private:
  // 私有成员函数，根据参数构造语句对象并返回其指针
  StmtPtr constructStmt(
      const std::vector<VarPtr>& args,
      ExprPtr body,
      const std::vector<ExprPtr>& reduce_dims,
      const std::vector<VarPtr>& reduce_args) const;

  // 成员变量，缓冲区指针
  BufPtr buf_;
  // 成员变量，语句指针
  StmtPtr stmt_;
};

// 创建计算张量的函数，接受函数名称、维度表达式向量、可选步长向量和表达式函数对象
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&)>& body_func);

// 创建计算张量的函数，接受函数名称、维度表达式向量和表达式函数对象
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&)>& body_func);

// 创建计算张量的函数，接受函数名称、维度表达式向量、可选步长向量和双参数表达式函数对象
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func);

// 创建计算张量的函数，接受函数名称、维度表达式向量和双参数表达式函数对象
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func);

// 创建计算张量的函数，接受函数名称、维度表达式向量、可选步长向量和多参数表达式函数对象
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
    # 定义一个常量，其类型为 std::function，表示这是一个函数对象的类型
    # 函数接受三个参数，分别是 VarHandle 类型的引用，表示这些参数是变量句柄对象
    # 函数返回一个 ExprHandle 对象，表示函数返回一个表达式句柄对象
    # 函数对象的引用被命名为 body_func，该函数对象在后续使用中用于处理三个 VarHandle 参数
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func);
// 定义一个名为 Compute 的函数模板，用于计算张量，接受函数名、维度列表和计算主体函数作为参数
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func);

// 定义另一个名为 Compute 的函数模板，用于计算张量，接受函数名、维度列表、可选的步长、以及计算主体函数作为参数
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func);

// 定义 Compute 函数的另一个重载，用于计算张量，接受函数名、维度列表和计算主体函数作为参数
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func);

// 定义另一个名为 Compute 的函数模板，用于计算张量，接受函数名、维度列表、可选的步长、以及计算主体函数作为参数
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);

// 定义 Compute 函数的重载，用于计算张量，接受函数名、维度列表和计算主体函数作为参数
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);

// 创建名为 create_index_vars 的内联函数，用于根据给定的维度列表创建变量列表
inline std::vector<VarHandle> create_index_vars(
    const std::vector<ExprHandle>& dims) {
  // 创建一个变量列表 vars，预留足够的空间以容纳所有维度
  std::vector<VarHandle> vars;
  vars.reserve(dims.size());
  // 遍历维度列表 dims，为每个维度创建一个变量，并将其添加到 vars 中
  for (const ExprHandle& dim : dims) {
    vars.emplace_back(alloc<Var>(
        "i", dim.dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  // 返回创建的变量列表 vars
  return vars;
}

// 定义一个名为 Reduce 的模板函数，用于处理 Reducer 和提供值的 body_func 的归约操作
template <typename InitFunc, typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    const InitFunc& init_func,
    const BodyFunc& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  // 创建维度索引变量列表 vars
  std::vector<VarHandle> vars = create_index_vars(dims);
  // 创建归约维度索引变量列表 reduce_vars
  std::vector<VarHandle> reduce_vars = create_index_vars(reduce_dims);

  // 如果 reduce_vars 为空，则不是归约操作，而是简单的复制
  if (reduce_vars.empty()) {
    // 生成归约主体表达式 body
    ExprHandle body = Reducer::getReduceBody(body_func, vars);
    // 创建结果缓冲区 func_result
    BufHandle func_result = Buf::make(
        func_name, dims, body.dtype(), c10::nullopt, std::move(strides));
    // 返回张量对象，包含结果缓冲区、变量列表 vars 和主体表达式 body
    return Tensor(std::move(func_result), vars, std::move(body));
  }

  // 创建所有变量的列表 all_vars
  std::vector<VarHandle> all_vars;
  all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  all_vars.insert(all_vars.end(), reduce_vars.begin(), reduce_vars.end());

  // 生成归约主体表达式 body
  ExprHandle body = Reducer::getReduceBody(body_func, all_vars);
  // 将 vars 的子集作为输出参数
  std::vector<ExprHandle> output_args(vars.begin(), vars.end());
  // 初始化表达式 init_expr
  ExprHandle init_expr = Cast::make(body.dtype(), init_func(vars));
  // 创建结果缓冲区 func_result
  BufHandle func_result = Buf::make(func_name, dims, body.dtype(), init_expr);

  // 执行归约操作，并生成归约操作表达式 reduce_op
  ExprHandle reduce_op = reducer(func_result, body, output_args, reduce_vars);
  // 如果主体表达式的数据类型为 kBFloat16
  if (body.dtype() == kBFloat16) {
    // 使用 init_func 对变量 vars 进行初始化，并将结果转换为 kFloat 类型的表达式
    ExprHandle init_expr_acc = Cast::make(kFloat, init_func(vars));

    // 创建一个名为 func_name + "_acc" 的缓冲区，数据类型为 kFloat，初始化表达式为 init_expr_acc
    BufHandle func_result_acc =
        Buf::make(func_name + "_acc", dims, kFloat, init_expr_acc);

    // 调用 reducer 函数生成 reduce_op，用于对 func_result 进行归约操作
    reduce_op = reducer(
        func_result,
        std::move(func_result_acc),
        std::move(body),
        output_args,
        reduce_vars);
  }

  // 创建一个 Tensor 对象 t，用于存储归约后的结果
  Tensor t = Tensor(
      std::move(func_result),   // 归约后的结果数据
      vars,                     // 变量列表
      reduce_dims,              // 归约维度
      reduce_vars,              // 归约变量
      std::move(reduce_op));    // 归约操作
  // 返回 Tensor 对象 t 作为函数结果
  return t;
// 定义一个模板函数 Reduce，接受 InitFunc 和 BodyFunc 两个模板参数
template <typename InitFunc, typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const InitFunc& init_func,  // 初始化函数对象，用于初始化缩减操作的结果
    const BodyFunc& body_func,  // 主体函数对象，定义了缩减操作的具体计算逻辑
    const std::vector<ExprHandle>& reduce_dims) {  // 缩减的维度向量
  return Reduce<InitFunc, BodyFunc>(  // 调用 Reduce 函数的重载版本，传递所有参数
      func_name,
      dims,
      c10::nullopt,
      reducer,
      init_func,
      body_func,
      reduce_dims);
}

// 定义一个模板函数 Reduce，接受 BodyFunc 一个模板参数
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    std::optional<std::vector<ExprHandle>> strides,  // 可选的步长向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const BodyFunc& body_func,  // 主体函数对象，定义了缩减操作的具体计算逻辑
    const std::vector<ExprHandle>& reduce_dims) {  // 缩减的维度向量
  return Reduce(  // 调用 Reduce 函数的重载版本，传递所有参数
      func_name,
      dims,
      strides,
      reducer,
      [&](ParameterList p) { return ExprHandle(reducer.initializer()); },  // 使用 Reducer 的初始化器初始化缩减结果
      body_func,
      reduce_dims);
}

// 定义一个模板函数 Reduce，接受 BodyFunc 一个模板参数
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const BodyFunc& body_func,  // 主体函数对象，定义了缩减操作的具体计算逻辑
    const std::vector<ExprHandle>& reduce_dims) {  // 缩减的维度向量
  return Reduce<BodyFunc>(  // 调用 Reduce 函数的重载版本，传递所有参数
      func_name, dims, c10::nullopt, reducer, body_func, reduce_dims);
}

// 为支持使用内联 lambda 函数作为 body_func 的重载版本
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    std::optional<std::vector<ExprHandle>> strides,  // 可选的步长向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const BodyFunc&& body_func,  // 右值引用的 lambda 函数作为主体函数对象
    const std::vector<ExprHandle>& reduce_dims) {  // 缩减的维度向量
  return Reduce(func_name, dims, strides, reducer, body_func, reduce_dims);  // 调用 Reduce 函数的重载版本，传递所有参数
}

// 为支持使用内联 lambda 函数作为 body_func 的重载版本
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const BodyFunc&& body_func,  // 右值引用的 lambda 函数作为主体函数对象
    const std::vector<ExprHandle>& reduce_dims) {  // 缩减的维度向量
  return Reduce(func_name, dims, c10::nullopt, reducer, body_func, reduce_dims);  // 调用 Reduce 函数的重载版本，传递所有参数
}

// TORCH_API 标识的 Reduce 函数重载，接受缓冲区对象作为参数
TORCH_API Tensor Reduce(
    const std::string& name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    std::optional<std::vector<ExprHandle>> strides,  // 可选的步长向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const BufHandle& buffer,  // 缓冲区句柄对象
    const std::vector<ExprHandle>& reduce_dims);

// TORCH_API 标识的 Reduce 函数重载，接受缓冲区对象作为参数
TORCH_API Tensor Reduce(
    const std::string& name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    const BufHandle& buffer,  // 缓冲区句柄对象
    const std::vector<ExprHandle>& reduce_dims);

// TORCH_API 标识的 Reduce 函数重载，用于所有维度为已计算张量的常见情况
TORCH_API Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    std::optional<std::vector<ExprHandle>> strides,  // 可选的步长向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    Tensor tensor,  // 已计算张量对象
    const std::vector<ExprHandle>& reduce_dims);  // 缩减的维度向量

// TORCH_API 标识的 Reduce 函数重载，用于所有维度为已计算张量的常见情况
TORCH_API Tensor Reduce(
    const std::string& func_name,  // 函数名称
    const std::vector<ExprHandle>& dims,  // 维度表达式的向量
    const Reducer& reducer,  // Reducer 对象，用于指定如何进行数据缩减操作
    Tensor tensor,  // 已计算张量对象
    const std::vector<ExprHandle>& reduce_dims);

// 模板函数 Reduce 的可变参数模板形式的声明
template <typename... Ts>
// 返回一个表达式句柄，表示从当前 Tensor 对象对应的缓冲区中加载数据
inline ExprHandle Tensor::load(const Ts&... ts) const {
  // 使用传入的参数 ts 创建一个表达式句柄的向量
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  // 调用 Load::make 方法创建一个 Load 对象，加载当前 Tensor 对象对应缓冲区的数据
  return Load::make(BufHandle(this->buf()), params);
}

template <typename T>
// 返回一个表达式句柄，表示从当前 Tensor 对象对应的缓冲区中加载数据
inline ExprHandle Tensor::load(const std::vector<T>& args) const {
  // 使用传入的参数 args 创建一个表达式句柄的向量
  std::vector<ExprHandle> params(args.begin(), args.end());
  // 调用 Load::make 方法创建一个 Load 对象，加载当前 Tensor 对象对应缓冲区的数据
  return Load::make(BufHandle(this->buf()), params);
}

template <typename... Ts>
// 返回一个表达式句柄，表示从当前 BufHandle 对象对应的缓冲区中加载数据
inline ExprHandle BufHandle::load(const Ts&... ts) const {
  // 使用传入的参数 ts 创建一个表达式句柄的向量
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  // 调用 alloc<Load> 方法创建一个 Load 对象，加载当前 BufHandle 对象对应缓冲区的数据
  return ExprHandle(alloc<Load>(node(), ExprHandleVectorToExprVector(params)));
}

template <typename T>
// 返回一个表达式句柄，表示从当前 BufHandle 对象对应的缓冲区中加载数据
inline ExprHandle BufHandle::load(const std::vector<T>& args) const {
  // 使用传入的参数 args 创建一个表达式句柄的向量
  std::vector<ExprHandle> params(args.begin(), args.end());
  // 调用 alloc<Load> 方法创建一个 Load 对象，加载当前 BufHandle 对象对应缓冲区的数据
  return ExprHandle(alloc<Load>(node(), ExprHandleVectorToExprVector(params)));
}

// 返回一个表达式句柄，表示从当前 BufHandle 对象对应的缓冲区中加载数据
inline ExprHandle BufHandle::load(const std::vector<ExprHandle>& args) const {
  // 调用模板方法 load<ExprHandle>，传入参数 args，加载当前 BufHandle 对象对应缓冲区的数据
  return this->template load<ExprHandle>(args);
}
```