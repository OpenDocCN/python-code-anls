# `.\pytorch\torch\csrc\jit\tensorexpr\ir_visitor.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/ScalarType.h>
// 包含 C10 库中的标量类型定义

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏定义

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
// 包含 Torch 的 JIT Tensor Expression 模块的前向声明头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRVisitor {
// 定义 IRVisitor 类，用于访问 IR 表达式

 public:
  virtual ~IRVisitor() = default;
  // 虚析构函数，用于多态销毁对象

  virtual void visit(AddPtr v);
  // 纯虚函数，用于访问 AddPtr 对象

  virtual void visit(SubPtr v);
  // 纯虚函数，用于访问 SubPtr 对象

  virtual void visit(MulPtr v);
  // 纯虚函数，用于访问 MulPtr 对象

  virtual void visit(DivPtr v);
  // 纯虚函数，用于访问 DivPtr 对象

  virtual void visit(ModPtr v);
  // 纯虚函数，用于访问 ModPtr 对象

  virtual void visit(MaxPtr v);
  // 纯虚函数，用于访问 MaxPtr 对象

  virtual void visit(MinPtr v);
  // 纯虚函数，用于访问 MinPtr 对象

  virtual void visit(AndPtr v);
  // 纯虚函数，用于访问 AndPtr 对象

  virtual void visit(OrPtr v);
  // 纯虚函数，用于访问 OrPtr 对象

  virtual void visit(XorPtr v);
  // 纯虚函数，用于访问 XorPtr 对象

  virtual void visit(LshiftPtr v);
  // 纯虚函数，用于访问 LshiftPtr 对象

  virtual void visit(RshiftPtr v);
  // 纯虚函数，用于访问 RshiftPtr 对象

  virtual void visit(CompareSelectPtr v);
  // 纯虚函数，用于访问 CompareSelectPtr 对象

#define IMM_PRINT_VISIT(Type, Name) virtual void visit(Name##ImmPtr v);
  // 宏定义，用于生成对各种类型的立即数的访问函数声明

  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
  // 循环展开，对所有标量类型生成立即数访问函数声明
#undef IMM_PRINT_VISIT

  virtual void visit(CastPtr v);
  // 纯虚函数，用于访问 CastPtr 对象

  virtual void visit(BitCastPtr v);
  // 纯虚函数，用于访问 BitCastPtr 对象

  virtual void visit(VarPtr v);
  // 纯虚函数，用于访问 VarPtr 对象

  virtual void visit(BufPtr v);
  // 纯虚函数，用于访问 BufPtr 对象

  virtual void visit(RampPtr v);
  // 纯虚函数，用于访问 RampPtr 对象

  virtual void visit(LoadPtr v);
  // 纯虚函数，用于访问 LoadPtr 对象

  virtual void visit(ForPtr v);
  // 纯虚函数，用于访问 ForPtr 对象

  virtual void visit(BlockPtr v);
  // 纯虚函数，用于访问 BlockPtr 对象

  virtual void visit(StorePtr v);
  // 纯虚函数，用于访问 StorePtr 对象

  virtual void visit(BroadcastPtr v);
  // 纯虚函数，用于访问 BroadcastPtr 对象

  virtual void visit(IfThenElsePtr v);
  // 纯虚函数，用于访问 IfThenElsePtr 对象

  virtual void visit(IntrinsicsPtr v);
  // 纯虚函数，用于访问 IntrinsicsPtr 对象

  virtual void visit(AllocatePtr v);
  // 纯虚函数，用于访问 AllocatePtr 对象

  virtual void visit(FreePtr v);
  // 纯虚函数，用于访问 FreePtr 对象

  virtual void visit(FreeExtPtr v);
  // 纯虚函数，用于访问 FreeExtPtr 对象

  virtual void visit(PlacementAllocatePtr v);
  // 纯虚函数，用于访问 PlacementAllocatePtr 对象

  virtual void visit(LetPtr v);
  // 纯虚函数，用于访问 LetPtr 对象

  virtual void visit(CondPtr v);
  // 纯虚函数，用于访问 CondPtr 对象

  virtual void visit(TermPtr v);
  // 纯虚函数，用于访问 TermPtr 对象

  virtual void visit(PolynomialPtr v);
  // 纯虚函数，用于访问 PolynomialPtr 对象

  virtual void visit(RoundOffPtr v);
  // 纯虚函数，用于访问 RoundOffPtr 对象

  virtual void visit(MaxTermPtr v);
  // 纯虚函数，用于访问 MaxTermPtr 对象

  virtual void visit(MinTermPtr v);
  // 纯虚函数，用于访问 MinTermPtr 对象

  virtual void visit(ReduceOpPtr v);
  // 纯虚函数，用于访问 ReduceOpPtr 对象

  virtual void visit(AtomicAddPtr v);
  // 纯虚函数，用于访问 AtomicAddPtr 对象

  virtual void visit(SyncThreadsPtr v);
  // 纯虚函数，用于访问 SyncThreadsPtr 对象

  virtual void visit(ExternalCallPtr v);
  // 纯虚函数，用于访问 ExternalCallPtr 对象

  virtual void visit(ExternalCallWithAllocPtr v);
  // 纯虚函数，用于访问 ExternalCallWithAllocPtr 对象
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```