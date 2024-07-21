# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\mm_pattern.py`

```py
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 模块
import torch
# 导入 torch._inductor 模块（这是一个私有模块）
import torch._inductor

# 从 torch.ops 中导入 aten 和 prims
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块中导入多个类和函数
from torch._inductor.pattern_matcher import (
   Arg,                        # 参数类
   CallFunction,               # 调用函数模式
   CallFunctionVarArgs,        # 调用函数（可变参数）模式
   CallMethod,                 # 调用方法模式
   CallMethodVarArgs,          # 调用方法（可变参数）模式
   CallModule,                 # 调用模块模式
   CallModuleVarArgs,          # 调用模块（可变参数）模式
   ExclusiveKeywordArg,        # 排他性关键字参数模式
   Ignored,                    # 忽略参数模式
   KeywordArg,                 # 关键字参数模式
   ListOf,                     # 列表参数模式
   MultiOutputPattern,         # 多输出模式
   PatternExpr,                # 模式表达式
   RepeatedExpr,               # 重复表达式模式
   _TargetArgsExpr,            # 目标参数表达式模式
   _TargetExpr,                # 目标表达式模式
   _TargetExprVarArgs,         # 目标表达式（可变参数）模式
)

# 定义 mm_default 模式，调用 aten.mm.default 函数，指定两个关键字参数
mm_default = CallFunction(aten.mm.default, KeywordArg('mat1'), KeywordArg('mat2'))

# 定义 permute_default 模式，调用 aten.permute.default 函数，其中一个参数被忽略
permute_default = CallFunction(aten.permute.default, KeywordArg('mat2'), Ignored())

# 定义 mm_default_1 模式，调用 aten.mm.default 函数，指定一个关键字参数和 permute_default 模式作为另一个参数
mm_default_1 = CallFunction(aten.mm.default, KeywordArg('tangents_1'), permute_default)

# 定义 permute_default_1 模式，调用 aten.permute.default 函数，其中一个参数被忽略
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('mat1'), Ignored())

# 定义 mm_default_2 模式，调用 aten.mm.default 函数，指定 permute_default_1 模式和一个关键字参数
mm_default_2 = CallFunction(aten.mm.default, permute_default_1, KeywordArg('tangents_1'))

# 定义 mm_pattern_training 模式，使用 MultiOutputPattern 构造函数，包含多个 mm_default 等模式
mm_pattern_training = MultiOutputPattern([mm_default,
  mm_default_1,
  mm_default_2
])

# 定义 mm_pattern_inference 模式，调用 aten.mm.default 函数，指定两个关键字参数和 _users=0
mm_pattern_inference = CallFunction(aten.mm.default, KeywordArg('mat1'), KeywordArg('mat2'), _users=0)
```