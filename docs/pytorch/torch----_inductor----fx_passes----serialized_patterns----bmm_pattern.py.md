# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\bmm_pattern.py`

```
# mypy: ignore-errors
# 禁止 mypy 报告类型错误

# noqa: F401, E501
# 禁止 Flake8 报告“未使用的导入”和“行过长”错误
# 这是自动生成的文件，请勿手动修改。
# 若要重新生成，请运行：
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor

# 导入 torch 操作符的别名
aten = torch.ops.aten
prims = torch.ops.prims

# 导入模式匹配所需的类
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

# 创建默认的 bmm 模式
bmm_default = CallFunction(aten.bmm.default, KeywordArg('mat1'), KeywordArg('mat2'))

# 创建默认的 permute 模式
permute_default = CallFunction(aten.permute.default, KeywordArg('mat2'), Ignored())

# 创建第一个 bmm 默认模式
bmm_default_1 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default)

# 创建第二个 permute 默认模式
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('mat1'), Ignored())

# 创建第二个 bmm 默认模式
bmm_default_2 = CallFunction(aten.bmm.default, permute_default_1, KeywordArg('tangents_1'))

# 创建用于训练的 bmm 模式
bmm_pattern_training = MultiOutputPattern([bmm_default,
  bmm_default_1,
  bmm_default_2
])

# 创建用于推断的 bmm 模式
bmm_pattern_inference = CallFunction(aten.bmm.default, KeywordArg('mat1'), KeywordArg('mat2'), _users=0)
```