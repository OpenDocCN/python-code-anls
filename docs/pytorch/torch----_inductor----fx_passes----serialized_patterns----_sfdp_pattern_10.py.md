# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_10.py`

```py
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 库和 torch._inductor 模块
import torch
import torch._inductor

# 设置简称以便后续调用
aten = torch.ops.aten
prims = torch.ops.prims

# 导入 pattern_matcher 模块中的各种类和函数
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

# 创建 permute_default 模式对象
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建 div_Tensor 模式对象
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())

# 创建 expand_default 模式对象
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())

# 创建 clone_default 模式对象
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建 view_default 模式对象
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建 permute_default_1 模式对象
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建 permute_default_2 模式对象
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建 expand_default_1 模式对象
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建 clone_default_1 模式对象
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建 view_default_1 模式对象
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 创建 bmm_default 模式对象
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建 view_default_2 模式对象
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored(), _users=2)

# 创建 amax_default 模式对象
amax_default = CallFunction(aten.amax.default, view_default_2, Ignored(), True)

# 创建 sub_Tensor 模式对象
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_2, amax_default)

# 创建 exp_default 模式对象
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建 sum_dim_IntList 模式对象
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建 div_Tensor_1 模式对象
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建 convert_element_type_default 模式对象
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建 expand_default_2 模式对象
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 创建 view_default_3 模式对象
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 创建 permute_default_3 模式对象
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建 expand_default_3 模式对象
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建 clone_default_2 模式对象
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建 view_default_4 模式对象
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建 bmm_default_1 模式对象
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建 view_default_5 模式对象
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建 neg_default 模式对象
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建 view_default_6 模式对象
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 KeywordArg('query') 和 Ignored() 作为参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，传入 permute_default 和 Ignored() 作为参数
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，传入 div_Tensor 和 Ignored() 作为参数
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，传入 expand_default 和 memory_format=torch.contiguous_format 作为参数
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 clone_default 和 Ignored() 作为参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 KeywordArg('key') 和 Ignored() 作为参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 permute_default_1 和 Ignored() 作为参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，传入 permute_default_2 和 Ignored() 作为参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，传入 expand_default_1 和 memory_format=torch.contiguous_format 作为参数
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 clone_default_1 和 Ignored() 作为参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并传入 view_default 和 view_default_1 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 bmm_default 和 Ignored() 作为参数，设置 _users=2
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.amax.default 函数，并传入 view_default_2 和 Ignored() 作为参数，设置 True 为第三个参数
amax_default = CallFunction(aten.amax.default, view_default_2, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.sub.Tensor 函数，并传入 view_default_2 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_2, amax_default)

# 创建一个 CallFunction 对象，调用 aten.exp.default 函数，并传入 sub_Tensor 作为参数，设置 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，并传入 exp_default 和 Ignored() 作为参数，设置 True 为第三个参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并传入 exp_default 和 sum_dim_IntList 作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，并传入 div_Tensor_1 和 Ignored() 作为参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 convert_element_type_default 和 Ignored() 作为参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 expand_default_2 和 Ignored() 作为参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并传入 KeywordArg('value') 和 Ignored() 作为参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 permute_default_3 和 Ignored() 作为参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并传入 expand_default_3 和 memory_format=torch.contiguous_format 作为参数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 clone_default_2 和 Ignored() 作为参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并传入 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 bmm_default_1 和 Ignored() 作为参数，设置 _users=0
_sfdp_pattern_10_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)


# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并传入 KeywordArg('query') 和 Ignored() 作为参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并传入 permute_default 和 Ignored() 作为参数
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 div_Tensor 和 Ignored() 作为参数
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并传入 expand_default 和 memory_format=torch.contiguous_format 作为参数
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 clone_default 和 Ignored() 作为参数，设置 _users=2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并传入 KeywordArg('key') 和 Ignored() 作为参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并传入 permute_default_1 和 Ignored() 作为参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 permute_default_2 和 Ignored() 作为参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并传入 expand_default_1 和 memory_format=torch.contiguous_format 作为参数
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 clone_default_1 和 Ignored() 作为参数，设置 _users=2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并传入 view_default 和 view_default_1 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 bmm_default 和 Ignored() 作为参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，并传入 view_default_2 和 Ignored() 作为参数，设置 _users=2
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.amax.default 函数，并传入 convert_element_type_default 和 Ignored() 作为参数，设置 True 为第三个参数
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.sub.Tensor 函数，并传入 convert_element_type_default 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个 CallFunction 对象，调用 aten.exp.default 函数，并传入 sub_Tensor 作为参数，设置 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，并传入 exp_default 和 Ignored() 作为参数，设置 True 为第三个参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并传入 exp_default 和 sum_dim_IntList 作为参数，设置 _users=3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，并传入 div_Tensor_1 和 Ignored() 作为参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 Torch 的 expand 函数，对 convert_element_type_default_1 进行扩展操作，并忽略其他参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用 Torch 的 view 函数，对 expand_default_2 进行视图操作，并指定两个用户

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 Torch 的 permute 函数，对指定参数进行置换操作，并忽略其他参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 Torch 的 expand 函数，对 permute_default_3 进行扩展操作，并忽略其他参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用 Torch 的 clone 函数，对 expand_default_3 进行克隆操作，使用连续内存格式

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 调用 Torch 的 view 函数，对 clone_default_2 进行视图操作，并指定两个用户

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 Torch 的 bmm 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法操作

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 Torch 的 view 函数，对 bmm_default_1 进行视图操作，并忽略其他参数

neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 调用 Torch 的 neg 函数，对 div_Tensor_1 进行取负操作

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 Torch 的 view 函数，对指定参数 'tangents_1' 进行视图操作，并指定两个用户

permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_4 进行置换操作，并忽略其他参数

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 调用 Torch 的 bmm 函数，对 view_default_6 和 permute_default_4 进行批量矩阵乘法操作

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 Torch 的 view 函数，对 bmm_default_2 进行视图操作，并忽略其他参数

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
# 调用 Torch 的 convert_element_type 函数，对 view_default_7 进行类型转换操作，并忽略其他参数

mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_2, div_Tensor_1, _users=2)
# 调用 Torch 的 mul 函数，对 convert_element_type_default_2 和 div_Tensor_1 进行张量乘法操作，并指定两个用户

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
# 调用 Torch 的 sum 函数，对 mul_Tensor 进行维度求和操作，并指定忽略参数和保留维度

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
# 调用 Torch 的 fma 函数，对 neg_default、sum_dim_IntList_1 和 mul_Tensor 进行 FMA（fused multiply-add）操作

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用 Torch 的 convert_element_type 函数，对 fma_default 进行类型转换操作，并忽略其他参数

view_default_8 = CallFunction(aten.view.default, convert_element_type_default_3, Ignored(), _users=2)
# 调用 Torch 的 view 函数，对 convert_element_type_default_3 进行视图操作，并指定两个用户

permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_1 进行置换操作，并忽略其他参数

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 调用 Torch 的 bmm 函数，对 view_default_8 和 permute_default_5 进行批量矩阵乘法操作

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 Torch 的 view 函数，对 bmm_default_3 进行视图操作，并忽略其他参数

div_Tensor_2 = CallFunction(aten.div.Tensor, view_default_9, Ignored())
# 调用 Torch 的 div 函数，对 view_default_9 进行张量除法操作，并忽略其他参数

permute_default_6 = CallFunction(aten.permute.default, div_Tensor_2, Ignored())
# 调用 Torch 的 permute 函数，对 div_Tensor_2 进行置换操作，并忽略其他参数

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 Torch 的 permute 函数，对 view_default 进行置换操作，并忽略其他参数

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用 Torch 的 bmm 函数，对 permute_default_7 和 view_default_8 进行批量矩阵乘法操作

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 Torch 的 view 函数，对 bmm_default_4 进行视图操作，并忽略其他参数

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_10 进行置换操作，并忽略其他参数

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 调用 Torch 的 permute 函数，对 permute_default_8 进行置换操作，并忽略其他参数

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_3 进行置换操作，并忽略其他参数

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 调用 Torch 的 bmm 函数，对 permute_default_10 和 view_default_6 进行批量矩阵乘法操作

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 Torch 的 view 函数，对 bmm_default_5 进行视图操作，并忽略其他参数

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_11 进行置换操作，并忽略其他参数

_sfdp_pattern_10_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11
])
# 创建一个多输出模式，包含 view_default_5、permute_default_6、permute_default_9 和 permute_default_11
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 torch 的 permute 操作，使用 'query' 关键字参数，其他参数忽略

div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())
# 调用 torch 的 div 操作，第一个参数是 permute_default 变量，第二个参数忽略

expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
# 调用 torch 的 expand 操作，第一个参数是 div_Tensor 变量，第二个参数忽略

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用 torch 的 clone 操作，第一个参数是 expand_default 变量，使用 torch.contiguous_format 作为内存格式

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 调用 torch 的 view 操作，第一个参数是 clone_default 变量，第二个参数忽略

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 torch 的 permute 操作，使用 'key' 关键字参数，其他参数忽略

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 torch 的 permute 操作，第一个参数是 permute_default_1 变量，第二个参数忽略

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 torch 的 expand 操作，第一个参数是 permute_default_2 变量，第二个参数忽略

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用 torch 的 clone 操作，第一个参数是 expand_default_1 变量，使用 torch.contiguous_format 作为内存格式

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 调用 torch 的 view 操作，第一个参数是 clone_default_1 变量，第二个参数忽略

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 torch 的 bmm 操作，第一个和第二个参数分别是 view_default 和 view_default_1 变量

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 torch 的 view 操作，第一个参数是 bmm_default 变量，第二个参数忽略

convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)
# 调用 prims.convert_element_type.default 操作，第一个参数是 view_default_2 变量，第二个参数忽略，_users 设置为 2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 torch 的 amax 操作，第一个参数是 convert_element_type_default 变量，第二个参数忽略，第三个参数设置为 True

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 torch 的 sub 操作，第一个参数是 convert_element_type_default 变量，第二个参数是 amax_default 变量

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 torch 的 exp 操作，第一个参数是 sub_Tensor 变量，_users 设置为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 torch 的 sum 操作，第一个参数是 exp_default 变量，第二个参数忽略，第三个参数设置为 True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 torch 的 div 操作，第一个参数是 exp_default 变量，第二个参数是 sum_dim_IntList 变量

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 调用 prims.convert_element_type.default 操作，第一个参数是 div_Tensor_1 变量，第二个参数忽略

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 torch 的 expand 操作，第一个参数是 convert_element_type_default_1 变量，第二个参数忽略

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 torch 的 view 操作，第一个参数是 expand_default_2 变量，第二个参数忽略

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 torch 的 permute 操作，使用 'value' 关键字参数，其他参数忽略

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 torch 的 expand 操作，第一个参数是 permute_default_3 变量，第二个参数忽略

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用 torch 的 clone 操作，第一个参数是 expand_default_3 变量，使用 torch.contiguous_format 作为内存格式

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 调用 torch 的 view 操作，第一个参数是 clone_default_2 变量，第二个参数忽略

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 torch 的 bmm 操作，第一个和第二个参数分别是 view_default_3 和 view_default_4 变量

_sfdp_pattern_10_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 torch 的 view 操作，第一个参数是 bmm_default_1 变量，第二个参数忽略，_users 设置为 0
```