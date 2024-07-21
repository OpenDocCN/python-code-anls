# `.\pytorch\torch\csrc\jit\tensorexpr\ir_cloner.h`

```py
#pragma once
#include <c10/core/ScalarType.h>  // 包含C10库中的标量类型定义
#include <torch/csrc/Export.h>   // 包含Torch导出相关的头文件
#include <vector>                // 包含STL库中的向量容器

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>  // 包含Torch张量表达式中的IRMutator类定义

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRCloner : public IRMutator {  // IRCloner类继承自IRMutator类，用于复制IR节点
 public:
  ~IRCloner() override = default;  // 默认析构函数

  ExprPtr mutate(AddPtr v) override;     // 重写父类虚函数，处理加法节点
  ExprPtr mutate(SubPtr v) override;     // 处理减法节点
  ExprPtr mutate(MulPtr v) override;     // 处理乘法节点
  ExprPtr mutate(DivPtr v) override;     // 处理除法节点
  ExprPtr mutate(ModPtr v) override;     // 处理取模节点
  ExprPtr mutate(MaxPtr v) override;     // 处理取最大值节点
  ExprPtr mutate(MinPtr v) override;     // 处理取最小值节点
  ExprPtr mutate(AndPtr v) override;     // 处理按位与节点
  ExprPtr mutate(OrPtr v) override;      // 处理按位或节点
  ExprPtr mutate(XorPtr v) override;     // 处理按位异或节点
  ExprPtr mutate(LshiftPtr v) override;  // 处理左移节点
  ExprPtr mutate(RshiftPtr v) override;  // 处理右移节点
  ExprPtr mutate(CompareSelectPtr v) override;  // 处理比较选择节点

#define IMM_MUTATE_DECLARE(Type, Name) ExprPtr mutate(Name##ImmPtr v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);  // 宏定义处理不同标量类型的立即数节点
#undef IMM_MUTATE_DECLARE

  ExprPtr mutate(CastPtr v) override;       // 处理类型转换节点
  ExprPtr mutate(BitCastPtr v) override;    // 处理位转换节点
  ExprPtr mutate(VarPtr v) override;        // 处理变量节点
  ExprPtr mutate(BufPtr v) override;        // 处理缓冲区节点
  ExprPtr mutate(RampPtr v) override;       // 处理分段线性函数节点
  ExprPtr mutate(LoadPtr v) override;       // 处理加载节点
  ExprPtr mutate(BroadcastPtr v) override;  // 处理广播节点
  ExprPtr mutate(IfThenElsePtr v) override; // 处理条件选择节点
  ExprPtr mutate(IntrinsicsPtr v) override; // 处理内部函数节点

  ExprPtr mutate(TermPtr v) override;         // 处理项节点
  ExprPtr mutate(PolynomialPtr v) override;   // 处理多项式节点
  ExprPtr mutate(RoundOffPtr v) override;     // 处理舍入节点
  ExprPtr mutate(MaxTermPtr v) override;      // 处理最大项节点
  ExprPtr mutate(MinTermPtr v) override;      // 处理最小项节点

  ExprPtr mutate(ReduceOpPtr v) override;     // 处理归约操作节点

  StmtPtr mutate(ForPtr v) override;          // 处理for循环节点
  StmtPtr mutate(BlockPtr v) override;        // 处理代码块节点
  StmtPtr mutate(StorePtr v) override;        // 处理存储节点
  StmtPtr mutate(AtomicAddPtr v) override;    // 处理原子加节点
  StmtPtr mutate(SyncThreadsPtr v) override;  // 处理线程同步节点
  StmtPtr mutate(ExternalCallPtr v) override; // 处理外部函数调用节点
  StmtPtr mutate(ExternalCallWithAllocPtr v) override;  // 处理带分配的外部函数调用节点

  StmtPtr mutate(AllocatePtr v) override;     // 处理分配内存节点
  StmtPtr mutate(FreePtr v) override;         // 处理释放内存节点
  StmtPtr mutate(LetPtr v) override;          // 处理变量赋值节点
  StmtPtr mutate(CondPtr v) override;         // 处理条件语句节点
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```