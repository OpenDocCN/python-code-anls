# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\addmm_pattern.py`

```py
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 库和 torch._inductor 模块
import torch
import torch._inductor

# 从 torch.ops 中导入 aten 和 prims
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块中导入一系列类
from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)

# 创建 addmm_default 模式匹配对象
addmm_default = CallFunction(aten.addmm.default, KeywordArg('input'), KeywordArg('mat1'), KeywordArg('mat2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'))

# 创建 mul_Scalar 模式匹配对象
mul_Scalar = CallFunction(aten.mul.Scalar, KeywordArg('tangents_1'), KeywordArg('beta'))

# 创建 sum_dim_IntList 模式匹配对象
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Scalar, Ignored(), True)

# 创建 view_default 模式匹配对象
view_default = CallFunction(aten.view.default, sum_dim_IntList, Ignored())

# 创建 permute_default 模式匹配对象
permute_default = CallFunction(aten.permute.default, KeywordArg('mat2'), Ignored())

# 创建 mm_default 模式匹配对象
mm_default = CallFunction(aten.mm.default, KeywordArg('tangents_1'), permute_default)

# 创建 mul_Scalar_1 模式匹配对象
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mm_default, KeywordArg('alpha'))

# 创建 permute_default_1 模式匹配对象
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('mat1'), Ignored())

# 创建 mm_default_1 模式匹配对象
mm_default_1 = CallFunction(aten.mm.default, permute_default_1, KeywordArg('tangents_1'))

# 创建 mul_Scalar_2 模式匹配对象
mul_Scalar_2 = CallFunction(aten.mul.Scalar, mm_default_1, KeywordArg('alpha'))

# 创建 addmm_pattern_training 多输出模式匹配对象
addmm_pattern_training = MultiOutputPattern([addmm_default,
  view_default,
  mul_Scalar_1,
  mul_Scalar_2,
  None,
  None
])

# 创建 addmm_pattern_inference 模式匹配对象
addmm_pattern_inference = CallFunction(aten.addmm.default, KeywordArg('input'), KeywordArg('mat1'), KeywordArg('mat2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'), _users=0)
```