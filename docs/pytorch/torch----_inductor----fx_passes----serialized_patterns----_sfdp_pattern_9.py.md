# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_9.py`

```py
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 库
import torch
# 导入 torch._inductor 模块
import torch._inductor

# 从 torch.ops 中导入 aten 和 prims
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 中导入多个模式匹配相关类
from torch._inductor.pattern_matcher import (
   Arg,  # 单个参数
   CallFunction,  # 调用函数模式
   CallFunctionVarArgs,  # 调用函数模式（可变参数）
   CallMethod,  # 调用对象方法模式
   CallMethodVarArgs,  # 调用对象方法模式（可变参数）
   CallModule,  # 调用模块模式
   CallModuleVarArgs,  # 调用模块模式（可变参数）
   ExclusiveKeywordArg,  # 排他关键字参数模式
   Ignored,  # 忽略模式
   KeywordArg,  # 关键字参数模式
   ListOf,  # 列表模式
   MultiOutputPattern,  # 多输出模式
   PatternExpr,  # 模式表达式
   RepeatedExpr,  # 重复表达式模式
   _TargetArgsExpr,  # 目标参数表达式
   _TargetExpr,  # 目标表达式
   _TargetExprVarArgs,  # 目标表达式（可变参数）
)

# 定义调用函数模式对象 rand_default
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 定义调用函数模式对象 gt_Scalar
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 定义调用函数模式对象 permute_default
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 定义调用函数模式对象 div_Tensor
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())
# 定义调用函数模式对象 expand_default
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
# 定义调用函数模式对象 clone_default
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 定义调用函数模式对象 view_default
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 定义调用函数模式对象 permute_default_1
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 定义调用函数模式对象 permute_default_2
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 定义调用函数模式对象 expand_default_1
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 定义调用函数模式对象 clone_default_1
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 定义调用函数模式对象 view_default_1
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 定义调用函数模式对象 bmm_default
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 定义调用函数模式对象 view_default_2
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored(), _users=2)
# 定义调用函数模式对象 amax_default
amax_default = CallFunction(aten.amax.default, view_default_2, Ignored(), True)
# 定义调用函数模式对象 sub_Tensor
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_2, amax_default)
# 定义调用函数模式对象 exp_default
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义调用函数模式对象 sum_dim_IntList
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义调用函数模式对象 div_Tensor_1
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 定义调用函数模式对象 mul_Tensor
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
# 定义调用函数模式对象 mul_Tensor_1
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 定义调用函数模式对象 convert_element_type_default
convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())
# 定义调用函数模式对象 expand_default_2
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())
# 定义调用函数模式对象 view_default_3
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 定义调用函数模式对象 permute_default_3
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 定义调用函数模式对象 expand_default_3
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 定义调用函数模式对象 clone_default_2
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 clone_default_2 和 Ignored()
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_1 和 Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.neg.default 函数，传入参数 div_Tensor_1
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 KeywordArg('tangents_1') 和 Ignored()
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_4 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_6 和 permute_default_4
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 bmm_default_2
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, bmm_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 convert_element_type_default_1 和 Ignored()
view_default_7 = CallFunction(aten.view.default, convert_element_type_default_1, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 view_default_7 和 Ignored()
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 gt_Scalar 和 Ignored()
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 convert_element_type_default_3 和 Ignored()
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 convert_element_type_default_2 和 mul_Tensor_2
mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default_2, mul_Tensor_2)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 mul_Tensor_3、div_Tensor_1 和 _users=2
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，传入参数 mul_Tensor_4 和 Ignored()，并设置 keepdim=True
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 创建一个调用函数对象，调用 prims.fma.default 函数，传入参数 neg_default、sum_dim_IntList_1 和 mul_Tensor_4
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 fma_default 和 Ignored()
view_default_8 = CallFunction(aten.view.default, fma_default, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_1 和 Ignored()
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_8 和 permute_default_5
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_3 和 Ignored()
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传入参数 view_default_9 和 Ignored()
div_Tensor_2 = CallFunction(aten.div.Tensor, view_default_9, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 div_Tensor_2 和 Ignored()
permute_default_6 = CallFunction(aten.permute.default, div_Tensor_2, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default 和 Ignored()
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 permute_default_7 和 view_default_8
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_4 和 Ignored()
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_10 和 Ignored()
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 permute_default_8 和 Ignored()
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_3 和 Ignored()
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 permute_default_10 和 view_default_6
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_5 和 Ignored()
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_11 和 Ignored()
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，包含多个函数调用对象作为其成员
_sfdp_pattern_9_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None
])

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 KeywordArg('query') 和 Ignored()
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传入参数 permute_default 和 Ignored()
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传入参数 div_Tensor 和 Ignored()
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建一个调用函数对象，调用 ATen 库中的 clone.default 函数，使用 expand_default 结果作为参数，并指定内存格式为连续格式

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 clone_default 结果作为参数，并忽略其他参数

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 permute.default 函数，使用关键字参数 'key'，并忽略其他参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 创建一个调用函数对象，再次调用 ATen 库中的 permute.default 函数，使用 permute_default_1 结果作为参数，并忽略其他参数

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 expand.default 函数，使用 permute_default_2 结果作为参数，并忽略其他参数

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 创建一个调用函数对象，调用 ATen 库中的 clone.default 函数，使用 expand_default_1 结果作为参数，并指定内存格式为连续格式

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 clone_default_1 结果作为参数，并忽略其他参数

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 创建一个调用函数对象，调用 ATen 库中的 bmm.default 函数，使用 view_default 和 view_default_1 结果作为参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored(), _users=2)
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 bmm_default 结果作为参数，并忽略部分参数，_users 设置为 2

amax_default = CallFunction(aten.amax.default, view_default_2, Ignored(), True)
# 创建一个调用函数对象，调用 ATen 库中的 amax.default 函数，使用 view_default_2 结果作为参数，并忽略部分参数，最后一个参数设置为 True

sub_Tensor = CallFunction(aten.sub.Tensor, view_default_2, amax_default)
# 创建一个调用函数对象，调用 ATen 库中的 sub.Tensor 函数，使用 view_default_2 和 amax_default 结果作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建一个调用函数对象，调用 ATen 库中的 exp.default 函数，使用 sub_Tensor 结果作为参数，并设置 _users 为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建一个调用函数对象，调用 ATen 库中的 sum.dim_IntList 函数，使用 exp_default 结果作为参数，并忽略部分参数，最后一个参数设置为 True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 创建一个调用函数对象，调用 ATen 库中的 div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 结果作为参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，使用 div_Tensor_1 结果作为参数，并忽略部分参数

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 expand.default 函数，使用 convert_element_type_default 结果作为参数，并忽略其他参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 expand_default_2 结果作为参数，并忽略其他参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 permute.default 函数，使用关键字参数 'value'，并忽略其他参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 expand.default 函数，使用 permute_default_3 结果作为参数，并忽略其他参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 创建一个调用函数对象，调用 ATen 库中的 clone.default 函数，使用 expand_default_3 结果作为参数，并指定内存格式为连续格式

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 clone_default_2 结果作为参数，并忽略其他参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 创建一个调用函数对象，调用 ATen 库中的 bmm.default 函数，使用 view_default_3 和 view_default_4 结果作为参数

_sfdp_pattern_9_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 bmm_default_1 结果作为参数，并忽略部分参数，_users 设置为 0

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 创建一个调用函数对象，调用 ATen 库中的 rand.default 函数，使用默认参数，并设置 pin_memory 为 False

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 创建一个调用函数对象，调用 ATen 库中的 gt.Scalar 函数，使用 rand_default 和关键字参数 'dropout_p' 作为参数，并设置 _users 为 2

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 permute.default 函数，使用关键字参数 'query'，并忽略其他参数

div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 div.Tensor 函数，使用 permute_default 结果作为参数，并忽略其他参数

expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 expand.default 函数，使用 div_Tensor 结果作为参数，并忽略其他参数

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建一个调用函数对象，调用 ATen 库中的 clone.default 函数，使用 expand_default 结果作为参数，并指定内存格式为连续格式

view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 创建一个调用函数对象，调用 ATen 库中的 view.default 函数，使用 clone_default 结果作为参数，并忽略部分参数，_users 设置为 2

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 permute.default 函数，使用关键字参数 'key'，并忽略其他参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 创建一个调用函数对象，再次调用 ATen 库中的 permute.default 函数，使用 permute_default_1 结果作为参数，并忽略其他参数

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建一个调用函数对象，调用 ATen 库中的 expand.default 函数，使用 permute_default_2 结果作为参数，并忽
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)
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
view_default_8 = CallFunction(aten.view.default, convert_element_type_default_4, Ignored(), _users=2)


注释：


# 调用 torch 的 view 函数，对 clone_default_1 进行默认视图操作，用户数为 2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 调用 torch 的 bmm 函数，对 view_default 和 view_default_1 进行批量矩阵乘法操作
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用 torch 的 view 函数，对 bmm_default 进行默认视图操作
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用 prims 的 convert_element_type 函数，进行默认类型转换操作，用户数为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)

# 调用 torch 的 amax 函数，计算 convert_element_type_default 的最大值
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 调用 torch 的 sub 函数，对 convert_element_type_default 和 amax_default 进行张量减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 调用 torch 的 exp 函数，对 sub_Tensor 进行指数运算，用户数为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用 torch 的 sum 函数，对 exp_default 进行指定维度的求和操作
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用 torch 的 div 函数，对 exp_default 和 sum_dim_IntList 进行张量除法操作，用户数为 3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 调用 torch 的 mul 函数，对 gt_Scalar 和 div_Tensor_1 进行张量乘法操作
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)

# 调用 torch 的 mul 函数，对 mul_Tensor 和忽略参数进行张量乘法操作
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 调用 prims 的 convert_element_type 函数，进行默认类型转换操作
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())

# 调用 torch 的 expand 函数，对 convert_element_type_default_1 进行默认展开操作
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 调用 torch 的 view 函数，对 expand_default_2 进行默认视图操作，用户数为 2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 调用 torch 的 permute 函数，对关键字参数 'value' 和忽略参数进行置换操作
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用 torch 的 expand 函数，对 permute_default_3 进行默认展开操作
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用 torch 的 clone 函数，对 expand_default_3 进行默认克隆操作，使用连续格式的内存
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 调用 torch 的 view 函数，对 clone_default_2 进行默认视图操作，用户数为 2
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 调用 torch 的 bmm 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法操作
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用 torch 的 view 函数，对 bmm_default_1 进行默认视图操作
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 调用 torch 的 neg 函数，对 div_Tensor_1 进行取负操作
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 调用 torch 的 view 函数，对关键字参数 'tangents_1' 和忽略参数进行默认视图操作，用户数为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 调用 torch 的 permute 函数，对 view_default_4 和忽略参数进行置换操作
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 调用 torch 的 bmm 函数，对 view_default_6 和 permute_default_4 进行批量矩阵乘法操作
bmm_default_2 = CallFunction(aten.bmm.default, view_default
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default_1 进行维度重排，结果赋给 permute_default_5

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 使用默认参数调用 torch.bmm 函数，对 view_default_8 和 permute_default_5 进行批量矩阵乘法，结果赋给 bmm_default_3

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default_3 进行张量视图操作，结果赋给 view_default_9

div_Tensor_2 = CallFunction(aten.div.Tensor, view_default_9, Ignored())
# 使用默认参数调用 torch.div 函数，对 view_default_9 进行张量除法操作，结果赋给 div_Tensor_2

permute_default_6 = CallFunction(aten.permute.default, div_Tensor_2, Ignored())
# 使用默认参数调用 torch.permute 函数，对 div_Tensor_2 进行维度重排，结果赋给 permute_default_6

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default 进行维度重排，结果赋给 permute_default_7

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 使用默认参数调用 torch.bmm 函数，对 permute_default_7 和 view_default_8 进行批量矩阵乘法，结果赋给 bmm_default_4

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default_4 进行张量视图操作，结果赋给 view_default_10

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default_10 进行维度重排，结果赋给 permute_default_8

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 使用默认参数调用 torch.permute 函数，对 permute_default_8 进行维度重排，结果赋给 permute_default_9

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default_3 进行维度重排，结果赋给 permute_default_10

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 使用默认参数调用 torch.bmm 函数，对 permute_default_10 和 view_default_6 进行批量矩阵乘法，结果赋给 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default_5 进行张量视图操作，结果赋给 view_default_11

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default_11 进行维度重排，结果赋给 permute_default_11

_sfdp_pattern_9_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None
])
# 使用 view_default_5, permute_default_6, permute_default_9, permute_default_11 和 None 创建 MultiOutputPattern 对象，并赋给 _sfdp_pattern_9_half_training

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 使用默认参数调用 torch.permute 函数，以关键字参数 'query' 调用，结果赋给 permute_default

div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())
# 使用默认参数调用 torch.div 函数，对 permute_default 进行张量除法操作，结果赋给 div_Tensor

expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
# 使用默认参数调用 torch.expand 函数，对 div_Tensor 进行张量扩展操作，结果赋给 expand_default

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 使用默认参数调用 torch.clone 函数，对 expand_default 进行张量克隆操作，并指定内存格式为 torch.contiguous_format，结果赋给 clone_default

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 使用默认参数调用 torch.view 函数，对 clone_default 进行张量视图操作，结果赋给 view_default

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 使用默认参数调用 torch.permute 函数，以关键字参数 'key' 调用，结果赋给 permute_default_1

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 使用默认参数调用 torch.permute 函数，对 permute_default_1 进行维度重排，结果赋给 permute_default_2

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 使用默认参数调用 torch.expand 函数，对 permute_default_2 进行张量扩展操作，结果赋给 expand_default_1

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 使用默认参数调用 torch.clone 函数，对 expand_default_1 进行张量克隆操作，并指定内存格式为 torch.contiguous_format，结果赋给 clone_default_1

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 使用默认参数调用 torch.view 函数，对 clone_default_1 进行张量视图操作，结果赋给 view_default_1

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 使用默认参数调用 torch.bmm 函数，对 view_default 和 view_default_1 进行批量矩阵乘法，结果赋给 bmm_default

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default 进行张量视图操作，结果赋给 view_default_2

convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)
# 使用默认参数调用 prims.convert_element_type.default 函数，对 view_default_2 进行元素类型转换操作，结果赋给 convert_element_type_default

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 使用默认参数调用 torch.amax 函数，对 convert_element_type_default 进行张量最大值计算，结果赋给 amax_default

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 使用默认参数调用 torch.sub 函数，对 convert_element_type_default 和 amax_default 进行张量减法操作，结果赋给 sub_Tensor

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 使用默认参数调用 torch.exp 函数，对 sub_Tensor 进行指数函数操作，结果赋给 exp_default

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 使用默认参数调用 torch.sum.dim_IntList 函数，对 exp_default 进行指定维度求和操作，结果赋给 sum_dim_IntList

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 使用默认参数调用 torch.div 函数，对 exp_default 进行张量除法操作，结果赋给 div_Tensor_1

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 使用默认参数调用 prims.convert_element_type.default 函数，对 div_Tensor_1 进行元素类型转换操作，结果赋给 convert_element_type_default_1

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored
# 调用 aten.view.default 函数，传入 expand_default_2 和 Ignored() 作为参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 调用 aten.permute.default 函数，传入 KeywordArg('value') 和 Ignored() 作为参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用 aten.expand.default 函数，传入 permute_default_3 和 Ignored() 作为参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用 aten.clone.default 函数，传入 expand_default_3 和 memory_format=torch.contiguous_format 作为参数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 调用 aten.view.default 函数，传入 clone_default_2 和 Ignored() 作为参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 调用 aten.bmm.default 函数，传入 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用 aten.view.default 函数，传入 bmm_default_1、Ignored() 和 _users=0 作为参数
_sfdp_pattern_9_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```