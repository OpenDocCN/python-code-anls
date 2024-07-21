# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_7.py`

```
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 和 torch._inductor 模块
import torch
import torch._inductor

# 定义 aten 和 prims 变量，用于访问 torch.ops 中的操作
aten = torch.ops.aten
prims = torch.ops.prims

# 导入 torch._inductor.pattern_matcher 模块中的多个类和函数
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

# 定义 rand_default 变量，调用 aten.rand.default 函数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 定义 gt_Scalar 变量，调用 aten.gt.Scalar 函数，传入 rand_default 作为参数，并指定一个关键字参数 'dropout_p'
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 定义 permute_default 变量，调用 aten.permute.default 函数，指定一个关键字参数 'query'
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 定义 expand_default 变量，调用 aten.expand.default 函数，参数为 permute_default
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 定义 clone_default 变量，调用 aten.clone.default 函数，参数为 expand_default，并设置内存格式为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 定义 view_default 变量，调用 aten.view.default 函数，参数为 clone_default，并忽略部分参数，同时指定 _users 为 2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 定义 permute_default_1 变量，调用 aten.permute.default 函数，指定一个关键字参数 'key'
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 定义 permute_default_2 变量，调用 aten.permute.default 函数，参数为 permute_default_1，并忽略其他参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 定义 expand_default_1 变量，调用 aten.expand.default 函数，参数为 permute_default_2，并忽略其他参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 定义 clone_default_1 变量，调用 aten.clone.default 函数，参数为 expand_default_1，并设置内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 定义 view_default_1 变量，调用 aten.view.default 函数，参数为 clone_default_1，并忽略部分参数，同时指定 _users 为 2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 定义 bmm_default 变量，调用 aten.bmm.default 函数，参数为 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 定义 view_default_2 变量，调用 aten.view.default 函数，参数为 bmm_default，并忽略部分参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 定义 div_Tensor 变量，调用 aten.div.Tensor 函数，参数为 view_default_2，并忽略部分参数，同时指定 _users 为 2
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored(), _users=2)

# 定义 amax_default 变量，调用 aten.amax.default 函数，参数为 div_Tensor，并忽略部分参数，同时指定为 True
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)

# 定义 sub_Tensor 变量，调用 aten.sub.Tensor 函数，参数为 div_Tensor 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)

# 定义 exp_default 变量，调用 aten.exp.default 函数，参数为 sub_Tensor，并指定 _users 为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 定义 sum_dim_IntList 变量，调用 aten.sum.dim_IntList 函数，参数为 exp_default，并忽略部分参数，同时指定为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 定义 div_Tensor_1 变量，调用 aten.div.Tensor 函数，参数为 exp_default 和 sum_dim_IntList，并指定 _users 为 3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 定义 mul_Tensor 变量，调用 aten.mul.Tensor 函数，参数为 gt_Scalar 和 div_Tensor_1
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)

# 定义 mul_Tensor_1 变量，调用 aten.mul.Tensor 函数，参数为 mul_Tensor 和 Ignored()
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 定义 convert_element_type_default 变量，调用 prims.convert_element_type.default 函数，参数为 mul_Tensor_1 和 Ignored()
convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())

# 定义 expand_default_2 变量，调用 aten.expand.default 函数，参数为 convert_element_type_default 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 定义 view_default_3 变量，调用 aten.view.default 函数，参数为 expand_default_2，并忽略部分参数，同时指定 _users 为 2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 定义 permute_default_3 变量，调用 aten.permute.default 函数，指定一个关键字参数 'value'
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 定义 expand_default_3 变量，调用 aten.expand.default 函数，参数为 permute_default_3 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 定义 clone_default_2 变量，调用 aten.clone.default 函数，参数为 expand_default_3，并设置内存格式为 torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数，对 clone_default_2 进行视图操作，标记用户为 2

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的默认批量矩阵乘法函数，对 view_default_3 和 view_default_4 进行操作

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的默认视图函数，对 bmm_default_1 进行视图操作，未指定具体用途

neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 调用 PyTorch 的默认取负函数，对 div_Tensor_1 进行操作

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数，对关键字参数 'tangents_1' 进行视图操作，标记用户为 2

permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default_4 进行排列操作

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 调用 PyTorch 的默认批量矩阵乘法函数，对 view_default_6 和 permute_default_4 进行操作

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, bmm_default_2, Ignored())
# 调用 prims 模块的默认类型转换函数，对 bmm_default_2 进行操作

view_default_7 = CallFunction(aten.view.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的默认视图函数，对 convert_element_type_default_1 进行视图操作，未指定具体用途

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
# 调用 prims 模块的默认类型转换函数，对 view_default_7 进行操作

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用 prims 模块的默认类型转换函数，对 gt_Scalar 进行操作

mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
# 调用 PyTorch 的张量乘法函数，对 convert_element_type_default_3 进行操作

mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default_2, mul_Tensor_2)
# 调用 PyTorch 的张量乘法函数，对 convert_element_type_default_2 和 mul_Tensor_2 进行操作

mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
# 调用 PyTorch 的张量乘法函数，对 mul_Tensor_3 和 div_Tensor_1 进行操作，标记用户为 2

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 调用 PyTorch 的按指定维度求和函数，对 mul_Tensor_4 进行操作，保持维度

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用 prims 模块的默认 FMA 函数，对 neg_default、sum_dim_IntList_1 和 mul_Tensor_4 进行操作

div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, Ignored())
# 调用 PyTorch 的张量除法函数，对 fma_default 进行操作

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数，对 div_Tensor_2 进行视图操作，标记用户为 2

permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default_1 进行排列操作

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 调用 PyTorch 的默认批量矩阵乘法函数，对 view_default_8 和 permute_default_5 进行操作

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 的默认视图函数，对 bmm_default_3 进行视图操作，未指定具体用途

permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default_9 进行排列操作

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default 进行排列操作

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用 PyTorch 的默认批量矩阵乘法函数，对 permute_default_7 和 view_default_8 进行操作

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 PyTorch 的默认视图函数，对 bmm_default_4 进行视图操作，未指定具体用途

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default_10 进行排列操作

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 调用 PyTorch 的默认排列函数，对 permute_default_8 进行排列操作

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default_3 进行排列操作

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 调用 PyTorch 的默认批量矩阵乘法函数，对 permute_default_10 和 view_default_6 进行操作

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 PyTorch 的默认视图函数，对 bmm_default_5 进行视图操作，未指定具体用途

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 调用 PyTorch 的默认排列函数，对 view_default_11 进行排列操作

_sfdp_pattern_7_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None
])
# 创建一个多输出模式对象，包含 view_default_5、permute_default_6、permute_default_9、permute_default_11 和 None

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的默认排列函数，对关键字参数 'query' 进行排列操作

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的默认扩展函数，对 permute_default 进行扩展操作
# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并使用指定的参数 expand_default 和 memory_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 clone_default 和一个忽略参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并使用指定的参数 'key' 和一个忽略参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并使用指定的参数 permute_default_1 和一个忽略参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并使用指定的参数 permute_default_2 和一个忽略参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并使用指定的参数 expand_default_1 和 memory_format=torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 clone_default_1 和一个忽略参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并使用指定的参数 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 bmm_default 和一个忽略参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并使用指定的参数 view_default_2 和一个忽略参数，_users 设为 2
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored(), _users=2)
# 创建一个 CallFunction 对象，调用 aten.amax.default 函数，并使用指定的参数 div_Tensor 和一个忽略参数，最后一个参数设为 True
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
# 创建一个 CallFunction 对象，调用 aten.sub.Tensor 函数，并使用指定的参数 div_Tensor 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
# 创建一个 CallFunction 对象，调用 aten.exp.default 函数，并使用指定的参数 sub_Tensor，_users 设为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，并使用指定的参数 exp_default 和一个忽略参数，最后一个参数设为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并使用指定的参数 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，并使用指定的参数 div_Tensor_1 和一个忽略参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并使用指定的参数 convert_element_type_default 和一个忽略参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 expand_default_2 和一个忽略参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并使用指定的参数 'value' 和一个忽略参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并使用指定的参数 permute_default_3 和一个忽略参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并使用指定的参数 expand_default_3 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 clone_default_2 和一个忽略参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并使用指定的参数 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 bmm_default_1 和一个忽略参数，_users 设为 0
_sfdp_pattern_7_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)

# 创建一个 CallFunction 对象，调用 aten.rand.default 函数，并使用多个忽略参数和设定的 pin_memory=False
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 创建一个 CallFunction 对象，调用 aten.gt.Scalar 函数，并使用指定的参数 rand_default 和 'dropout_p'
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并使用指定的参数 'query' 和一个忽略参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并使用指定的参数 permute_default 和一个忽略参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并使用指定的参数 expand_default 和 memory_format=torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并使用指定的参数 clone_default 和一个忽略参数，_users 设为 2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并使用指定的参数 'key' 和一个忽略参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并使用指定的参数 permute_default_1 和一个忽略参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并使用指定的参数 permute_default_2 和一个忽略参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，并使用指定的参数 expand_default_1 和 memory_format=torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor_1)
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default_2, mul_Tensor_2)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())


注释：


# 调用 aten.view.default 函数，参数为 clone_default_1，Ignored()，_users=2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 调用 aten.bmm.default 函数，参数为 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用 aten.view.default 函数，参数为 bmm_default 和 Ignored()
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用 aten.div.Tensor 函数，参数为 view_default_2 和 Ignored()
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())

# 调用 prims.convert_element_type.default 函数，参数为 div_Tensor 和 Ignored()，_users=2
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 调用 aten.amax.default 函数，参数为 convert_element_type_default，Ignored()，True
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 调用 aten.sub.Tensor 函数，参数为 convert_element_type_default 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 调用 aten.exp.default 函数，参数为 sub_Tensor，_users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用 aten.sum.dim_IntList 函数，参数为 exp_default，Ignored()，True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用 aten.div.Tensor 函数，参数为 exp_default，sum_dim_IntList，_users=3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 调用 aten.mul.Tensor 函数，参数为 gt_Scalar 和 div_Tensor_1
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)

# 调用 aten.mul.Tensor 函数，参数为 mul_Tensor 和 Ignored()
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 调用 prims.convert_element_type.default 函数，参数为 mul_Tensor_1 和 Ignored()
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())

# 调用 aten.expand.default 函数，参数为 convert_element_type_default_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 调用 aten.view.default 函数，参数为 expand_default_2，Ignored()，_users=2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 调用 aten.permute.default 函数，参数为 KeywordArg('value')，Ignored()
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用 aten.expand.default 函数，参数为 permute_default_3 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用 aten.clone.default 函数，参数为 expand_default_3 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 调用 aten.view.default 函数，参数为 clone_default_2，Ignored()，_users=2
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 调用 aten.bmm.default 函数，参数为 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用 aten.view.default 函数，参数为 bmm_default_1 和 Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 调用 aten.neg.default 函数，参数为 div_Tensor_1
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 调用 aten.view.default 函数，参数为 KeywordArg('tangents_1')，Ignored()，_users=2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 调用 aten.permute.default 函数，参数为 view_default_4 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 调用 aten.bmm.default 函数，参数为 view_default_6 和 permute_default_4
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 调用 aten.view.default 函数，参数为 bmm_default_2 和 Ignored()
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 调用 prims.convert_element_type.default 函数，参数为 view_default_7 和 Ignored()
convert_element_type_default_2 =
# 定义变量 div_Tensor_2，调用函数 aten.div.Tensor，执行默认的元素类型转换，参数 Ignored() 表示未指定
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_4, Ignored())

# 定义变量 view_default_8，调用函数 aten.view.default，对 div_Tensor_2 进行默认视图操作，参数 Ignored() 表示未指定
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 定义变量 permute_default_5，调用函数 aten.permute.default，对 view_default_1 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 定义变量 bmm_default_3，调用函数 aten.bmm.default，执行 view_default_8 和 permute_default_5 的批量矩阵乘操作
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 定义变量 view_default_9，调用函数 aten.view.default，对 bmm_default_3 进行默认视图操作，参数 Ignored() 表示未指定
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 定义变量 permute_default_6，调用函数 aten.permute.default，对 view_default_9 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 定义变量 permute_default_7，调用函数 aten.permute.default，对 view_default 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 定义变量 bmm_default_4，调用函数 aten.bmm.default，执行 permute_default_7 和 view_default_8 的批量矩阵乘操作
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 定义变量 view_default_10，调用函数 aten.view.default，对 bmm_default_4 进行默认视图操作，参数 Ignored() 表示未指定
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 定义变量 permute_default_8，调用函数 aten.permute.default，对 view_default_10 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 定义变量 permute_default_9，调用函数 aten.permute.default，对 permute_default_8 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 定义变量 permute_default_10，调用函数 aten.permute.default，对 view_default_3 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 定义变量 bmm_default_5，调用函数 aten.bmm.default，执行 permute_default_10 和 view_default_6 的批量矩阵乘操作
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 定义变量 view_default_11，调用函数 aten.view.default，对 bmm_default_5 进行默认视图操作，参数 Ignored() 表示未指定
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 定义变量 permute_default_11，调用函数 aten.permute.default，对 view_default_11 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 定义变量 _sfdp_pattern_7_half_training，调用 MultiOutputPattern 构造多输出模式对象，包含 view_default_5, permute_default_6,
# permute_default_9, permute_default_11 四个元素，最后一个元素为 None
_sfdp_pattern_7_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None
])

# 定义变量 permute_default，调用函数 aten.permute.default，对参数 'query' 进行默认置换操作，参数 Ignored() 表示未指定
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 定义变量 expand_default，调用函数 aten.expand.default，对 permute_default 进行默认扩展操作，参数 Ignored() 表示未指定
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 定义变量 clone_default，调用函数 aten.clone.default，对 expand_default 进行默认克隆操作，使用内存格式为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 定义变量 view_default，调用函数 aten.view.default，对 clone_default 进行默认视图操作，参数 Ignored() 表示未指定
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 定义变量 permute_default_1，调用函数 aten.permute.default，对参数 'key' 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 定义变量 permute_default_2，调用函数 aten.permute.default，对 permute_default_1 进行默认置换操作，参数 Ignored() 表示未指定
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 定义变量 expand_default_1，调用函数 aten.expand.default，对 permute_default_2 进行默认扩展操作，参数 Ignored() 表示未指定
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 定义变量 clone_default_1，调用函数 aten.clone.default，对 expand_default_1 进行默认克隆操作，使用内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 定义变量 view_default_1，调用函数 aten.view.default，对 clone_default_1 进行默认视图操作，参数 Ignored() 表示未指定
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 定义变量 bmm_default，调用函数 aten.bmm.default，执行 view_default 和 view_default_1 的批量矩阵乘操作
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 定义变量 view_default_2，调用函数 aten.view.default，对 bmm_default 进行默认视图操作，参数 Ignored() 表示未指定
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 定义变量 div_Tensor，调用函数 aten.div.Tensor，执行 view_default_2 的张量除法操作，参数 Ignored() 表示未指定
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())

# 定义变量 convert_element_type_default，调用函数 prims.convert_element_type.default，
# 对 div_Tensor 进行默认元素类型转换操作，参数 Ignored() 表示未指定，_users=2 表示有两个使用者
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 定义变量 amax_default，调用函数 aten.amax.default，计算 convert_element_type_default 的最大值，参数 Ignored() 表示未指定，True 表示返回索引
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 定义变量 sub_Tensor，调用函数 aten.sub.Tensor，执行 convert_element
# 创建一个调用函数对象，调用 torch 中的 aten.expand.default 函数，传入 convert_element_type_default_1 和 Ignored() 作为参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建一个调用函数对象，调用 torch 中的 aten.view.default 函数，传入 expand_default_2 和 Ignored() 作为参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个调用函数对象，调用 torch 中的 aten.permute.default 函数，使用 KeywordArg('value') 作为关键字参数，Ignored() 作为第二个参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 torch 中的 aten.expand.default 函数，传入 permute_default_3 和 Ignored() 作为参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 torch 中的 aten.clone.default 函数，传入 expand_default_3 和 memory_format=torch.contiguous_format 作为参数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch 中的 aten.view.default 函数，传入 clone_default_2 和 Ignored() 作为参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个调用函数对象，调用 torch 中的 aten.bmm.default 函数，传入 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 torch 中的 aten.view.default 函数，传入 bmm_default_1 和 Ignored() 作为参数，
# 同时 _users=0 表示此对象没有用户，可能是暂时不被使用的中间结果
_sfdp_pattern_7_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```