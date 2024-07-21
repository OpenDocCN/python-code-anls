# `.\pytorch\torch\csrc\jit\tensorexpr\ir_verifier.h`

```py
#pragma once

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 前向声明各种类，以避免循环依赖问题
class Expr;                  // 表达式类
class ExprHandle;            // 表达式句柄类
class Mod;                   // 取模运算类
class And;                   // 逻辑与运算类
class Or;                    // 逻辑或运算类
class Xor;                   // 异或运算类
class Lshift;                // 左移运算类
class Rshift;                // 右移运算类
class CompareSelect;         // 比较选择类
class Ramp;                  // 坡道类
class Load;                  // 加载类
class IfThenElse;            // 条件分支类
class Intrinsics;            // 内置函数类

class Stmt;                  // 语句类
class ExternalCall;          // 外部调用类
class Store;                 // 存储类
class For;                   // 循环类
class Block;                 // 块类

// IRVerifier 类，用于验证中间表示(IR)的正确性，继承自 IRVisitor 类
class TORCH_API IRVerifier : public IRVisitor {
 public:
  IRVerifier() = default;

  // 下面的函数重载用于访问不同类型的 IR 节点
  void visit(ModPtr v) override;                   // 访问 ModPtr 类型节点
  void visit(AndPtr v) override;                   // 访问 AndPtr 类型节点
  void visit(OrPtr v) override;                    // 访问 OrPtr 类型节点
  void visit(XorPtr v) override;                   // 访问 XorPtr 类型节点
  void visit(LshiftPtr v) override;                // 访问 LshiftPtr 类型节点
  void visit(RshiftPtr v) override;                // 访问 RshiftPtr 类型节点
  void visit(CompareSelectPtr v) override;         // 访问 CompareSelectPtr 类型节点
  void visit(RampPtr v) override;                  // 访问 RampPtr 类型节点
  void visit(LoadPtr v) override;                  // 访问 LoadPtr 类型节点
  void visit(IfThenElsePtr v) override;            // 访问 IfThenElsePtr 类型节点
  void visit(IntrinsicsPtr v) override;            // 访问 IntrinsicsPtr 类型节点

  void visit(ExternalCallPtr v) override;          // 访问 ExternalCallPtr 类型节点
  void visit(StorePtr v) override;                 // 访问 StorePtr 类型节点
  void visit(ForPtr v) override;                   // 访问 ForPtr 类型节点
  void visit(BlockPtr v) override;                 // 访问 BlockPtr 类型节点
};

// 验证 StmtPtr 类型的 IR 节点
TORCH_API void verify(StmtPtr);
// 验证 ExprPtr 类型的 IR 节点
TORCH_API void verify(ExprPtr);
// 验证 ExprHandle 类型的 IR 节点
TORCH_API void verify(ExprHandle);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```