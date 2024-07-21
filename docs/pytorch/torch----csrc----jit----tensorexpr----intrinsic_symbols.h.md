# `.\pytorch\torch\csrc\jit\tensorexpr\intrinsic_symbols.h`

```py
#pragma once
#ifdef TORCH_ENABLE_LLVM
#include <c10/util/ArrayRef.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义了一个结构体 SymbolAddress，用于存储符号名和地址的对应关系
struct SymbolAddress {
  const char* symbol;  // 符号名，C风格字符串
  void* address;       // 符号对应的地址指针

  // 结构体构造函数，初始化符号名和地址
  SymbolAddress(const char* sym, void* addr) : symbol(sym), address(addr) {}
};

// 声明一个函数 getIntrinsicSymbols，返回一个 ArrayRef，包含 SymbolAddress 结构体的数组引用
c10::ArrayRef<SymbolAddress> getIntrinsicSymbols();

} // namespace tensorexpr
} // namespace jit
} // namespace torch
#endif // TORCH_ENABLE_LLVM
```