# `.\pytorch\torch\csrc\jit\mobile\promoted_prim_ops.h`

```
#pragma once
#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/mobile/register_ops_common_utils.h>

namespace torch {
namespace jit {

// 声明函数 void tupleIndex(Stack& stack);
void tupleIndex(Stack& stack);

// 声明函数 void raiseException(Stack& stack);
void raiseException(Stack& stack);

// 声明函数 void is(Stack& stack);
void is(Stack& stack);

// 声明函数 void unInitialized(Stack& stack);
void unInitialized(Stack& stack);

// 声明函数 void isNot(Stack& stack);
void isNot(Stack& stack);

// 声明函数 void aten_format(Stack& stack);
void aten_format(Stack& stack);

// 声明函数 void size(Stack& stack);
void size(Stack& stack);

// 声明函数 void sym_size(Stack& stack);
void sym_size(Stack& stack);

// 声明函数 void sym_size_int(Stack& stack);
void sym_size_int(Stack& stack);

// 声明函数 void sym_stride_int(Stack& stack);
void sym_stride_int(Stack& stack);

// 声明函数 void sym_numel(Stack& stack);
void sym_numel(Stack& stack);

// 声明函数 void sym_storage_offset(Stack& stack);
void sym_storage_offset(Stack& stack);

// 声明函数 void sym_stride(Stack& stack);
void sym_stride(Stack& stack);

// 声明函数 void device(Stack& stack);
void device(Stack& stack);

// 声明函数 void device_with_index(Stack& stack);
void device_with_index(Stack& stack);

// 声明函数 void dtype(Stack& stack);
void dtype(Stack& stack);

// 声明函数 void layout(Stack& stack);
void layout(Stack& stack);

// 声明函数 void toPrimDType(Stack& stack);
void toPrimDType(Stack& stack);

// 声明函数 void dim(Stack& stack);
void dim(Stack& stack);

// 声明函数 void _not(Stack& stack);
void _not(Stack& stack);

// 声明函数 void boolTensor(Stack& stack);
void boolTensor(Stack& stack);

// 声明函数 void toList(Stack& stack);
void toList(Stack& stack);

// 声明函数 void numToTensorScalar(Stack& stack);
void numToTensorScalar(Stack& stack);

// 声明函数 void isCuda(Stack& stack);
void isCuda(Stack& stack);

// 声明函数 void numToTensorBool(Stack& stack);
void numToTensorBool(Stack& stack);

// 声明函数 void dictIndex(Stack& stack);
void dictIndex(Stack& stack);

// 声明函数 void raiseExceptionWithMessage(Stack& stack);
void raiseExceptionWithMessage(Stack& stack);

} // namespace jit
} // namespace torch
```