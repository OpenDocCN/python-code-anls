# `.\pytorch\torch\csrc\jit\tensorexpr\ir_mutator.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <c10/core/ScalarType.h>
// 包含 c10 库中的 ScalarType 头文件

#include <torch/csrc/Export.h>
// 包含 torch 库中的 Export 头文件

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
// 包含 torch 库中 jit 模块下 tensorexpr 子模块的 fwd_decls 头文件

#include <vector>
// 包含 C++ 标准库中的 vector 头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRMutator {
// 定义了 IRMutator 类，作为所有 IR 变异器的基类，TORCH_API 是一个导出符号的宏

 public:
  virtual ~IRMutator() = default;
  // 虚析构函数，用于多态释放资源

  virtual ExprPtr mutate(AddPtr v);
  // 虚函数，接收 AddPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(SubPtr v);
  // 虚函数，接收 SubPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(MulPtr v);
  // 虚函数，接收 MulPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(DivPtr v);
  // 虚函数，接收 DivPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(ModPtr v);
  // 虚函数，接收 ModPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(MaxPtr v);
  // 虚函数，接收 MaxPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(MinPtr v);
  // 虚函数，接收 MinPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(AndPtr v);
  // 虚函数，接收 AndPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(OrPtr v);
  // 虚函数，接收 OrPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(XorPtr v);
  // 虚函数，接收 XorPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(LshiftPtr v);
  // 虚函数，接收 LshiftPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(RshiftPtr v);
  // 虚函数，接收 RshiftPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(CompareSelectPtr v);
  // 虚函数，接收 CompareSelectPtr 类型参数 v，并返回 ExprPtr 类型对象

#define IMM_MUTATE_DECLARE(Type, Name) virtual ExprPtr mutate(Name##ImmPtr v);
  // 宏定义，用于声明针对具体类型的 mutate 函数
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE

  virtual ExprPtr mutate(CastPtr v);
  // 虚函数，接收 CastPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(BitCastPtr v);
  // 虚函数，接收 BitCastPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(VarPtr v);
  // 虚函数，接收 VarPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(BufPtr v);
  // 虚函数，接收 BufPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(RampPtr v);
  // 虚函数，接收 RampPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(LoadPtr v);
  // 虚函数，接收 LoadPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(BroadcastPtr v);
  // 虚函数，接收 BroadcastPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(IfThenElsePtr v);
  // 虚函数，接收 IfThenElsePtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(IntrinsicsPtr v);
  // 虚函数，接收 IntrinsicsPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(TermPtr v);
  // 虚函数，接收 TermPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(PolynomialPtr v);
  // 虚函数，接收 PolynomialPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(RoundOffPtr v);
  // 虚函数，接收 RoundOffPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(MaxTermPtr v);
  // 虚函数，接收 MaxTermPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(MinTermPtr v);
  // 虚函数，接收 MinTermPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual ExprPtr mutate(ReduceOpPtr v);
  // 虚函数，接收 ReduceOpPtr 类型参数 v，并返回 ExprPtr 类型对象

  virtual StmtPtr mutate(ForPtr v);
  // 虚函数，接收 ForPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(BlockPtr v);
  // 虚函数，接收 BlockPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(StorePtr v);
  // 虚函数，接收 StorePtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(AtomicAddPtr v);
  // 虚函数，接收 AtomicAddPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(SyncThreadsPtr v);
  // 虚函数，接收 SyncThreadsPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(ExternalCallPtr v);
  // 虚函数，接收 ExternalCallPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(ExternalCallWithAllocPtr v);
  // 虚函数，接收 ExternalCallWithAllocPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(AllocatePtr v);
  // 虚函数，接收 AllocatePtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(FreePtr v);
  // 虚函数，接收 FreePtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(FreeExtPtr v);
  // 虚函数，接收 FreeExtPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(PlacementAllocatePtr v);
  // 虚函数，接收 PlacementAllocatePtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(LetPtr v);
  // 虚函数，接收 LetPtr 类型参数 v，并返回 StmtPtr 类型对象

  virtual StmtPtr mutate(CondPtr v);
  // 虚函数，接收 CondPtr 类型参数 v，并返回 StmtPtr 类型对象
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```