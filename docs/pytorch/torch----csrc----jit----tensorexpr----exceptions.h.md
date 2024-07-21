# `.\pytorch\torch\csrc\jit\tensorexpr\exceptions.h`

```py
// 预处理指令，用于确保头文件只包含一次
#pragma once

// 引入 Torch 库的导出头文件
#include <torch/csrc/Export.h>

// 引入 TensorExpr 的前向声明头文件
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

// 引入 C++ 标准库中的字符串流和异常处理功能
#include <sstream>
#include <stdexcept>

// 声明命名空间 torch::jit::tensorexpr 中的类 Expr 和 Stmt
namespace torch {
namespace jit {
namespace tensorexpr {
class Expr;
class Stmt;
} // namespace tensorexpr
} // namespace jit
} // namespace torch

// 声明在 std 命名空间中的函数 to_string，用于将 ExprPtr 和 StmtPtr 转换为字符串
namespace std {
// 使用 Torch API 导出的函数声明，将 ExprPtr 转换为字符串
TORCH_API std::string to_string(const torch::jit::tensorexpr::ExprPtr);
// 使用 Torch API 导出的函数声明，将 StmtPtr 转换为字符串
TORCH_API std::string to_string(const torch::jit::tensorexpr::StmtPtr);
} // namespace std

// 声明命名空间 torch::jit::tensorexpr 中的异常类 unsupported_dtype
namespace torch {
namespace jit {
namespace tensorexpr {

// 继承自 std::runtime_error 的异常类 unsupported_dtype，用于表示不支持的数据类型异常
class unsupported_dtype : public std::runtime_error {
 public:
  // 默认构造函数，抛出默认错误消息 "UNSUPPORTED DTYPE"
  explicit unsupported_dtype() : std::runtime_error("UNSUPPORTED DTYPE") {}
  // 构造函数，接受自定义错误消息，格式为 "UNSUPPORTED DTYPE: <err>"
  explicit unsupported_dtype(const std::string& err)
      : std::runtime_error("UNSUPPORTED DTYPE: " + err) {}
};

// 声明命名空间 torch::jit::tensorexpr 中的异常类 out_of_range_index
class out_of_range_index : public std::runtime_error {
 public:
  // 默认构造函数，抛出默认错误消息 "OUT OF RANGE INDEX"
  explicit out_of_range_index() : std::runtime_error("OUT OF RANGE INDEX") {}
  // 构造函数，接受自定义错误消息，格式为 "OUT OF RANGE INDEX: <err>"
  explicit out_of_range_index(const std::string& err)
      : std::runtime_error("OUT OF RANGE INDEX: " + err) {}
};

// 声明命名空间 torch::jit::tensorexpr 中的异常类 unimplemented_lowering
class unimplemented_lowering : public std::runtime_error {
 public:
  // 默认构造函数，抛出默认错误消息 "UNIMPLEMENTED LOWERING"
  explicit unimplemented_lowering()
      : std::runtime_error("UNIMPLEMENTED LOWERING") {}
  // 构造函数，接受 ExprPtr 类型参数，格式为 "UNIMPLEMENTED LOWERING: <expr>"
  explicit unimplemented_lowering(ExprPtr expr)
      : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(expr)) {}
  // 构造函数，接受 StmtPtr 类型参数，格式为 "UNIMPLEMENTED LOWERING: <stmt>"
  explicit unimplemented_lowering(StmtPtr stmt)
      : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(stmt)) {}
};

// 声明命名空间 torch::jit::tensorexpr 中的异常类 malformed_input
class malformed_input : public std::runtime_error {
 public:
  // 默认构造函数，抛出默认错误消息 "MALFORMED INPUT"
  explicit malformed_input() : std::runtime_error("MALFORMED INPUT") {}
  // 构造函数，接受自定义错误消息，格式为 "MALFORMED INPUT: <err>"
  explicit malformed_input(const std::string& err)
      : std::runtime_error("MALFORMED INPUT: " + err) {}
  // 构造函数，接受 ExprPtr 类型参数，格式为 "MALFORMED INPUT: <expr>"
  explicit malformed_input(ExprPtr expr)
      : std::runtime_error("MALFORMED INPUT: " + std::to_string(expr)) {}
  // 构造函数，接受自定义错误消息和 ExprPtr 类型参数，格式为 "MALFORMED INPUT: <err> - <expr>"
  explicit malformed_input(const std::string& err, ExprPtr expr)
      : std::runtime_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(expr)) {}
  // 构造函数，接受 StmtPtr 类型参数，格式为 "MALFORMED INPUT: <stmt>"
  explicit malformed_input(StmtPtr stmt)
      : std::runtime_error("MALFORMED INPUT: " + std::to_string(stmt)) {}
  // 构造函数，接受自定义错误消息和 StmtPtr 类型参数，格式为 "MALFORMED INPUT: <err> - <stmt>"
  explicit malformed_input(const std::string& err, StmtPtr stmt)
      : std::runtime_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(stmt)) {}
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
class malformed_ir : public std::runtime_error {
 public:
  // 默认构造函数，抛出一个包含"MALFORMED IR"信息的异常
  explicit malformed_ir() : std::runtime_error("MALFORMED IR") {}
  
  // 构造函数，根据给定的错误信息err，抛出异常，信息为"MALFORMED IR: " + err
  explicit malformed_ir(const std::string& err)
      : std::runtime_error("MALFORMED IR: " + err) {}
  
  // 构造函数，根据给定的表达式expr，抛出异常，信息为"MALFORMED IR: " + std::to_string(expr)
  explicit malformed_ir(ExprPtr expr)
      : std::runtime_error("MALFORMED IR: " + std::to_string(expr)) {}
  
  // 构造函数，根据给定的错误信息err和表达式expr，抛出异常，信息为"MALFORMED IR: " + err + " - " + std::to_string(expr)
  explicit malformed_ir(const std::string& err, ExprPtr expr)
      : std::runtime_error(
            "MALFORMED IR: " + err + " - " + std::to_string(expr)) {}
  
  // 构造函数，根据给定的语句stmt，抛出异常，信息为"MALFORMED IR: " + std::to_string(stmt)
  explicit malformed_ir(StmtPtr stmt)
      : std::runtime_error("MALFORMED IR: " + std::to_string(stmt)) {}
  
  // 构造函数，根据给定的错误信息err和语句stmt，抛出异常，信息为"MALFORMED IR: " + err + " - " + std::to_string(stmt)
  explicit malformed_ir(const std::string& err, StmtPtr stmt)
      : std::runtime_error(
            "MALFORMED IR: " + err + " - " + std::to_string(stmt)) {}
};

// 定义了一个名为buildErrorMessage的函数，返回类型为std::string，参数为s，默认为空字符串
TORCH_API std::string buildErrorMessage(const std::string& s = "");

// 结束命名空间torch::jit::tensorexpr::torch
} // namespace tensorexpr
} // namespace jit
} // namespace torch
```