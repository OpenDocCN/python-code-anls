# `.\pytorch\aten\src\ATen\templates\LazyIr.h`

```py
// 指令：#pragma once，确保本文件只被编译一次
#pragma once

// 此文件包含自动生成的 LazyTensor IR 节点
${lazy_ir_sysinc}
${lazy_ir_inc}

// 命名空间引入声明开始
${namespace_prologue}

// 引入 at 命名空间中的 operator<<
using at::operator<<;

// kNullValue 用于在节点具有 Optional<Value> 输入为 nullopt 时贡献静态哈希值。
// 区分 HASH(nullopt, something) 和 HASH(something, nullopt) 是很重要的，
// 在哈希函数中按参数顺序使用 kNullValue 可以实现这一目的。
static const torch::lazy::Value kNullValue = torch::lazy::Value();

// IR 声明
${ir_declarations}

// 命名空间引入声明结束
${namespace_epilogue}
```