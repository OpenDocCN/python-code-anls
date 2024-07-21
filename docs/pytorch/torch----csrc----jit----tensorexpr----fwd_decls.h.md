# `.\pytorch\torch\csrc\jit\tensorexpr\fwd_decls.h`

```
/*
#pragma once  // 只包含一次该头文件的保护预处理指令

#include <c10/core/ScalarType.h>  // 包含 c10 库中的 ScalarType 头文件
#include <memory>  // 包含标准库中的 memory 头文件，用于智能指针管理

namespace torch {  // 命名空间 torch
namespace jit {  // 命名空间 jit
namespace tensorexpr {  // 命名空间 tensorexpr

template <typename Node>  // 定义模板，用于声明 NodePtr 为 Node 的 shared_ptr
using NodePtr = std::shared_ptr<Node>;

template <typename To, typename From>
NodePtr<To> to(NodePtr<From> x) {  // 将动态类型转换为 To 类型的智能指针
  return std::dynamic_pointer_cast<To>(x);
}

template <typename To, typename From>
NodePtr<To> static_to(NodePtr<From> x) {  // 将静态类型转换为 To 类型的智能指针
  return std::static_pointer_cast<To>(x);
}

template <typename Node, typename... Args>
NodePtr<Node> alloc(Args&&... args) {  // 分配并构造 Node 类型的对象的智能指针
  return std::make_shared<Node>(std::forward<Args>(args)...);
}

class Buf;  // 前置声明类 Buf
class Expr;  // 前置声明类 Expr
class Stmt;  // 前置声明类 Stmt
class Var;  // 前置声明类 Var

using BufPtr = NodePtr<Buf>;  // 定义 Buf 类型的智能指针
using ExprPtr = NodePtr<Expr>;  // 定义 Expr 类型的智能指针
using StmtPtr = NodePtr<Stmt>;  // 定义 Stmt 类型的智能指针
using VarPtr = NodePtr<Var>;  // 定义 Var 类型的智能指针

class ExprHandle;  // 前置声明类 ExprHandle
class VarHandle;  // 前置声明类 VarHandle
class BufHandle;  // 前置声明类 BufHandle

// 各种操作符类的前置声明
class Add;
class And;
class BitCast;
class Broadcast;
class Cast;
class CompareSelect;
class Div;
class IfThenElse;
class Intrinsics;
class Let;
class Load;
class Lshift;
class Max;
class MaxTerm;
class Min;
class MinTerm;
class Mod;
class Mul;
class Or;
class Polynomial;
class Ramp;
class ReduceOp;
class RoundOff;
class Rshift;
class Store;
class Sub;
class Term;
class Xor;

// 定义各种操作符类的智能指针类型
using AddPtr = NodePtr<Add>;
using AndPtr = NodePtr<And>;
using BitCastPtr = NodePtr<BitCast>;
using BroadcastPtr = NodePtr<Broadcast>;
using CastPtr = NodePtr<Cast>;
using CompareSelectPtr = NodePtr<CompareSelect>;
using DivPtr = NodePtr<Div>;
using IfThenElsePtr = NodePtr<IfThenElse>;
using IntrinsicsPtr = NodePtr<Intrinsics>;
using LetPtr = NodePtr<Let>;
using LoadPtr = NodePtr<Load>;
using LshiftPtr = NodePtr<Lshift>;
using MaxPtr = NodePtr<Max>;
using MaxTermPtr = NodePtr<MaxTerm>;
using MinPtr = NodePtr<Min>;
using MinTermPtr = NodePtr<MinTerm>;
using ModPtr = NodePtr<Mod>;
using MulPtr = NodePtr<Mul>;
using OrPtr = NodePtr<Or>;
using PolynomialPtr = NodePtr<Polynomial>;
using RampPtr = NodePtr<Ramp>;
using ReduceOpPtr = NodePtr<ReduceOp>;
using RoundOffPtr = NodePtr<RoundOff>;
using RshiftPtr = NodePtr<Rshift>;
using StorePtr = NodePtr<Store>;
using SubPtr = NodePtr<Sub>;
using TermPtr = NodePtr<Term>;
using XorPtr = NodePtr<Xor>;

// 前置声明类 Allocate 等
class Allocate;
class AtomicAdd;
class Block;
class Cond;
class ExternalCall;
class ExternalCallWithAlloc;
class For;
class Free;
class FreeExt;
class PlacementAllocate;
class SyncThreads;

// 定义类 Allocate 等的智能指针类型
using AllocatePtr = NodePtr<Allocate>;
using AtomicAddPtr = NodePtr<AtomicAdd>;
using BlockPtr = NodePtr<Block>;
using CondPtr = NodePtr<Cond>;
using ExternalCallPtr = NodePtr<ExternalCall>;
using ExternalCallWithAllocPtr = NodePtr<ExternalCallWithAlloc>;
using ForPtr = NodePtr<For>;
using FreePtr = NodePtr<Free>;
using FreeExtPtr = NodePtr<FreeExt>;
using PlacementAllocatePtr = NodePtr<PlacementAllocate>;
using SyncThreadsPtr = NodePtr<SyncThreads>;

// 宏定义 IMM_DECLARE(Type, Name)，用于声明 IMM 类型和智能指针类型
#define IMM_DECLARE(Type, Name) \
  class Name##Imm;  \
  using Name##ImmPtr = NodePtr<Name##Imm>;

// 对所有标量类型和 Bool、Half、BFloat16 使用 IMM_DECLARE 宏
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_DECLARE);

// 取消宏定义 IMM_DECLARE
#undef IMM_DECLARE

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
*/
} // namespace jit
} // namespace torch
```