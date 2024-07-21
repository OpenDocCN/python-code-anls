# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_8.py`

```
# mypy: ignore-errors
# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入torch模块
import torch
# 导入torch的内部模块torch._inductor
import torch._inductor

# 定义torch.ops.aten为aten，torch.ops.prims为prims
aten = torch.ops.aten
prims = torch.ops.prims

# 从torch._inductor.pattern_matcher模块导入以下符号
from torch._inductor.pattern_matcher import (
   Arg,  # 单个参数模式
   CallFunction,  # 调用函数的模式
   CallFunctionVarArgs,  # 调用带可变参数的函数的模式
   CallMethod,  # 调用方法的模式
   CallMethodVarArgs,  # 调用带可变参数的方法的模式
   CallModule,  # 调用模块的模式
   CallModuleVarArgs,  # 调用带可变参数的模块的模式
   ExclusiveKeywordArg,  # 独占关键字参数的模式
   Ignored,  # 表示忽略的模式
   KeywordArg,  # 关键字参数的模式
   ListOf,  # 列表形式的模式
   MultiOutputPattern,  # 多输出模式
   PatternExpr,  # 模式表达式
   RepeatedExpr,  # 重复表达式
   _TargetArgsExpr,  # 目标参数表达式
   _TargetExpr,  # 目标表达式
   _TargetExprVarArgs,  # 带可变参数的目标表达式
)

# 定义permute_default为调用aten.permute.default函数的模式，带有'query'关键字参数和Ignored占位符参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 定义expand_default为调用aten.expand.default函数的模式，参数为permute_default和Ignored占位符参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 定义clone_default为调用aten.clone.default函数的模式，参数为expand_default和memory_format=torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 定义view_default为调用aten.view.default函数的模式，参数为clone_default、Ignored占位符参数和_users=2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 后续模式定义与上述类似，具体模式及其参数依次为：
# permute_default_1, permute_default_2, expand_default_1, clone_default_1, view_default_1, bmm_default, view_default_2,
# div_Tensor, amax_default, sub_Tensor, exp_default, sum_dim_IntList, div_Tensor_1, convert_element_type_default,
# expand_default_2, view_default_3, permute_default_3, expand_default_3, clone_default_2, view_default_4, bmm_default_1,
# view_default_5, neg_default, view_default_6
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_4 和 Ignored() 作为参数
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default_6 和 permute_default_4 作为参数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，使用 bmm_default_2 和 Ignored() 作为参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, bmm_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 convert_element_type_default_1 和 Ignored() 作为参数
view_default_7 = CallFunction(aten.view.default, convert_element_type_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，使用 view_default_7 和 Ignored() 作为参数
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，使用 convert_element_type_default_2、div_Tensor_1 和 _users=2 作为参数
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_2, div_Tensor_1, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，使用 mul_Tensor、Ignored() 和 True 作为参数
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)

# 创建一个 CallFunction 对象，调用 prims.fma.default 函数，使用 neg_default、sum_dim_IntList_1 和 mul_Tensor 作为参数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，使用 fma_default 和 Ignored() 作为参数
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 div_Tensor_2 和 Ignored() 作为参数，并设置 _users=2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_1 和 Ignored() 作为参数
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default_8 和 permute_default_5 作为参数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_3 和 Ignored() 作为参数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_9 和 Ignored() 作为参数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default 和 Ignored() 作为参数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 permute_default_7 和 view_default_8 作为参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_4 和 Ignored() 作为参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_10 和 Ignored() 作为参数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 permute_default_8 和 Ignored() 作为参数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_3 和 Ignored() 作为参数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 permute_default_10 和 view_default_6 作为参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_5 和 Ignored() 作为参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_11 和 Ignored() 作为参数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，包含 view_default_5、permute_default_6、permute_default_9 和 permute_default_11
_sfdp_pattern_8_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11
])

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 KeywordArg('query') 和 Ignored() 作为参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 permute_default 和 Ignored() 作为参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，使用 expand_default 和 memory_format=torch.contiguous_format 作为参数
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 clone_default 和 Ignored() 作为参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 KeywordArg('key') 和 Ignored() 作为参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 permute_default_1 和 Ignored() 作为参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 permute_default_2 和 Ignored() 作为参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，使用 expand_default_1 和 memory_format=torch.contiguous_format 作为参数
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 clone_default_1 和 Ignored() 作为参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default 和 view_default_1 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 bmm_default，忽略其他参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored(), _users=2)
# 创建一个调用张量除法操作的函数调用，输入为 view_default_2，忽略其他参数，有两个用户（使用者）

amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
# 创建一个调用取张量最大值操作的函数调用，输入为 div_Tensor，忽略其他参数，启用 keepdim=True

sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
# 创建一个调用张量减法操作的函数调用，输入为 div_Tensor 和 amax_default

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建一个调用指数函数操作的函数调用，输入为 sub_Tensor，有两个用户（使用者）

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建一个调用按维度求和操作的函数调用，输入为 exp_default，忽略其他参数，启用 keepdim=True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 创建一个调用张量除法操作的函数调用，输入为 exp_default 和 sum_dim_IntList

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 创建一个调用转换元素类型操作的函数调用，输入为 div_Tensor_1，忽略其他参数

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())
# 创建一个调用扩展操作的函数调用，输入为 convert_element_type_default，忽略其他参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 expand_default_2，忽略其他参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 创建一个调用排列操作的函数调用，使用默认选项，关键字参数 'value'，忽略其他参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 创建一个调用扩展操作的函数调用，输入为 permute_default_3，忽略其他参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 创建一个调用克隆操作的函数调用，输入为 expand_default_3，内存格式为 torch.contiguous_format

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 clone_default_2，忽略其他参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 创建一个调用批量矩阵乘操作的函数调用，输入为 view_default_3 和 view_default_4

_sfdp_pattern_8_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 bmm_default_1，忽略其他参数，没有用户（使用者）


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 创建一个调用排列操作的函数调用，使用默认选项，关键字参数 'query'，忽略其他参数

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建一个调用扩展操作的函数调用，输入为 permute_default，忽略其他参数

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建一个调用克隆操作的函数调用，输入为 expand_default，内存格式为 torch.contiguous_format

view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 clone_default，忽略其他参数，有两个用户（使用者）

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个调用排列操作的函数调用，使用默认选项，关键字参数 'key'，忽略其他参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 创建一个调用排列操作的函数调用，使用默认选项，输入为 permute_default_1，忽略其他参数

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建一个调用扩展操作的函数调用，输入为 permute_default_2，忽略其他参数

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 创建一个调用克隆操作的函数调用，输入为 expand_default_1，内存格式为 torch.contiguous_format

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 clone_default_1，忽略其他参数，有两个用户（使用者）

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 创建一个调用批量矩阵乘操作的函数调用，输入为 view_default 和 view_default_1

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 创建一个调用视图操作的函数调用，使用默认选项，输入为 bmm_default，忽略其他参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 创建一个调用张量除法操作的函数调用，输入为 view_default_2，忽略其他参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 创建一个调用转换元素类型操作的函数调用，输入为 div_Tensor，忽略其他参数，有两个用户（使用者）

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 创建一个调用取张量最大值操作的函数调用，输入为 convert_element_type_default，忽略其他参数，启用 keepdim=True

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 创建一个调用张量减法操作的函数调用，输入为 convert_element_type_default 和 amax_default

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建一个调用指数函数操作的函数调用，输入为 sub_Tensor，有两个用户（使用者）

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建一个调用按维度求和操作的函数调用，输入为 exp_default，忽略其他参数，启用 keepdim=True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 创建一个调用张量除法操作的函数调用，输入为 exp_default 和 sum_dim_IntList，有三个用户（使用者）

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 创建一个调用转换元素类型操作的函数调用，输入为 div_Tensor_1，忽略其他参数
# 调用名为 `aten.expand.default` 的函数，并传递参数 `convert_element_type_default_1` 和 `Ignored()`，将结果赋给 `expand_default_2`
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 调用名为 `aten.view.default` 的函数，并传递参数 `expand_default_2`、`Ignored()`，将结果赋给 `view_default_3`，同时指定 `_users` 为 2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 调用名为 `aten.permute.default` 的函数，并传递参数 `KeywordArg('value')` 和 `Ignored()`，将结果赋给 `permute_default_3`
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用名为 `aten.expand.default` 的函数，并传递参数 `permute_default_3` 和 `Ignored()`，将结果赋给 `expand_default_3`
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用名为 `aten.clone.default` 的函数，并传递参数 `expand_default_3` 和 `memory_format=torch.contiguous_format`，将结果赋给 `clone_default_2`
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 调用名为 `aten.view.default` 的函数，并传递参数 `clone_default_2`、`Ignored()`，将结果赋给 `view_default_4`，同时指定 `_users` 为 2
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 调用名为 `aten.bmm.default` 的函数，并传递参数 `view_default_3` 和 `view_default_4`，将结果赋给 `bmm_default_1`
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用名为 `aten.view.default` 的函数，并传递参数 `bmm_default_1`、`Ignored()`，将结果赋给 `view_default_5`
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 调用名为 `aten.neg.default` 的函数，并传递参数 `div_Tensor_1`，将结果赋给 `neg_default`
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 调用名为 `aten.view.default` 的函数，并传递参数 `KeywordArg('tangents_1')` 和 `Ignored()`，将结果赋给 `view_default_6`，同时指定 `_users` 为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 调用名为 `aten.permute.default` 的函数，并传递参数 `view_default_4` 和 `Ignored()`，将结果赋给 `permute_default_4`
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 调用名为 `aten.bmm.default` 的函数，并传递参数 `view_default_6` 和 `permute_default_4`，将结果赋给 `bmm_default_2`
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 调用名为 `aten.view.default` 的函数，并传递参数 `bmm_default_2`、`Ignored()`，将结果赋给 `view_default_7`
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 调用名为 `prims.convert_element_type.default` 的函数，并传递参数 `view_default_7` 和 `Ignored()`，将结果赋给 `convert_element_type_default_2`
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())

# 调用名为 `aten.mul.Tensor` 的函数，并传递参数 `convert_element_type_default_2`、`div_Tensor_1`，将结果赋给 `mul_Tensor`，同时指定 `_users` 为 2
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_2, div_Tensor_1, _users=2)

# 调用名为 `aten.sum.dim_IntList` 的函数，并传递参数 `mul_Tensor`、`Ignored()`、`True`，将结果赋给 `sum_dim_IntList_1`
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)

# 调用名为 `prims.fma.default` 的函数，并传递参数 `neg_default`、`sum_dim_IntList_1`、`mul_Tensor`，将结果赋给 `fma_default`
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)

# 调用名为 `prims.convert_element_type.default` 的函数，并传递参数 `fma_default` 和 `Ignored()`，将结果赋给 `convert_element_type_default_3`
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 调用名为 `aten.div.Tensor` 的函数，并传递参数 `convert_element_type_default_3` 和 `Ignored()`，将结果赋给 `div_Tensor_2`
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_3, Ignored())

# 调用名为 `aten.view.default` 的函数，并传递参数 `div_Tensor_2`、`Ignored()`，将结果赋给 `view_default_8`，同时指定 `_users` 为 2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 调用名为 `aten.permute.default` 的函数，并传递参数 `view_default_1` 和 `Ignored()`，将结果赋给 `permute_default_5`
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 调用名为 `aten.bmm.default` 的函数，并传递参数 `view_default_8` 和 `permute_default_5`，将结果赋给 `bmm_default_3`
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 调用名为 `aten.view.default` 的函数，并传递参数 `bmm_default_3`、`Ignored()`，将结果赋给 `view_default_9`
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 调用名为 `aten.permute.default` 的函数，并传递参数 `view_default_9` 和 `Ignored()`，将结果赋给 `permute_default_6`
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 调用名为 `aten.permute.default` 的函数，并传递参数 `view_default` 和 `Ignored()`，将结果赋给 `permute_default_7`
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 调用名为 `aten.bmm.default` 的函数，并传递参数 `permute_default_7` 和 `view_default_8`，将结果赋给 `bmm_default_4`
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 调用名为 `aten.view.default` 的函数，并传递参数 `bmm_default_4`、`Ignored()`，将结果赋给 `view_default_10`
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 调用名为 `aten.permute.default` 的函数，并传递参数 `view_default_10` 和 `Ignored()`，将结果赋给 `permute_default_8`
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 调用名为 `aten.permute.default` 的函数，并传递参数 `permute_default_8` 和 `Ignored()`，将结果赋给 `permute_default_9`
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 调用名为 `aten.permute.default` 的函数，并
# 定义并初始化 permute_default 变量，调用 aten.permute.default 函数，设置 'query' 关键字参数，忽略第二个参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 定义并初始化 expand_default 变量，调用 aten.expand.default 函数，传入 permute_default 变量作为第一个参数，忽略第二个参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 定义并初始化 clone_default 变量，调用 aten.clone.default 函数，传入 expand_default 变量作为第一个参数，设置内存格式为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 定义并初始化 view_default 变量，调用 aten.view.default 函数，传入 clone_default 变量作为第一个参数，忽略第二个参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 定义并初始化 permute_default_1 变量，调用 aten.permute.default 函数，设置 'key' 关键字参数，忽略第二个参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 定义并初始化 permute_default_2 变量，调用 aten.permute.default 函数，传入 permute_default_1 变量作为第一个参数，忽略第二个参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 定义并初始化 expand_default_1 变量，调用 aten.expand.default 函数，传入 permute_default_2 变量作为第一个参数，忽略第二个参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 定义并初始化 clone_default_1 变量，调用 aten.clone.default 函数，传入 expand_default_1 变量作为第一个参数，设置内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 定义并初始化 view_default_1 变量，调用 aten.view.default 函数，传入 clone_default_1 变量作为第一个参数，忽略第二个参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 定义并初始化 bmm_default 变量，调用 aten.bmm.default 函数，传入 view_default 和 view_default_1 变量作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 定义并初始化 view_default_2 变量，调用 aten.view.default 函数，传入 bmm_default 变量作为第一个参数，忽略第二个参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 定义并初始化 div_Tensor 变量，调用 aten.div.Tensor 函数，传入 view_default_2 变量作为第一个参数，忽略第二个参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())

# 定义并初始化 convert_element_type_default 变量，调用 prims.convert_element_type.default 函数，传入 div_Tensor 变量作为第一个参数，忽略第二个和第三个参数，指定 _users 参数为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 定义并初始化 amax_default 变量，调用 aten.amax.default 函数，传入 convert_element_type_default 变量作为第一个参数，忽略第二个和第三个参数，指定 keepdim 参数为 True
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 定义并初始化 sub_Tensor 变量，调用 aten.sub.Tensor 函数，传入 convert_element_type_default 和 amax_default 变量作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 定义并初始化 exp_default 变量，调用 aten.exp.default 函数，传入 sub_Tensor 变量作为第一个参数，指定 _users 参数为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 定义并初始化 sum_dim_IntList 变量，调用 aten.sum.dim_IntList 函数，传入 exp_default 变量作为第一个参数，忽略第二个和第三个参数，指定 keepdim 参数为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 定义并初始化 div_Tensor_1 变量，调用 aten.div.Tensor 函数，传入 exp_default 和 sum_dim_IntList 变量作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 定义并初始化 convert_element_type_default_1 变量，调用 prims.convert_element_type.default 函数，传入 div_Tensor_1 变量作为第一个参数，忽略第二个参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 定义并初始化 expand_default_2 变量，调用 aten.expand.default 函数，传入 convert_element_type_default_1 变量作为第一个参数，忽略第二个参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 定义并初始化 view_default_3 变量，调用 aten.view.default 函数，传入 expand_default_2 变量作为第一个参数，忽略第二个参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 定义并初始化 permute_default_3 变量，调用 aten.permute.default 函数，设置 'value' 关键字参数，忽略第二个参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 定义并初始化 expand_default_3 变量，调用 aten.expand.default 函数，传入 permute_default_3 变量作为第一个参数，忽略第二个参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 定义并初始化 clone_default_2 变量，调用 aten.clone.default 函数，传入 expand_default_3 变量作为第一个参数，设置内存格式为 torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 定义并初始化 view_default_4 变量，调用 aten.view.default 函数，传入 clone_default_2 变量作为第一个参数，忽略第二个参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 定义并初始化 bmm_default_1 变量，调用 aten.bmm.default 函数，传入 view_default_3 和 view_default_4 变量作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 定义并初始化 _sfdp_pattern_8_half_inference 变量，调用 aten.view.default 函数，传入 bmm_default_1 变量作为第一个参数，忽略第二个参数，并指定 _users 参数为 0
_sfdp_pattern_8_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```