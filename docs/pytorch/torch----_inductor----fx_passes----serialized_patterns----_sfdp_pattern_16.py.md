# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_16.py`

```
# mypy: ignore-errors
# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 库
import torch
# 导入 torch._inductor 模块
import torch._inductor

# 获取 torch.ops.aten 别名
aten = torch.ops.aten
# 获取 torch.ops.prims 别名
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块导入以下符号
from torch._inductor.pattern_matcher import (
   Arg,  # 单个参数
   CallFunction,  # 调用函数模式
   CallFunctionVarArgs,  # 调用函数模式（可变参数）
   CallMethod,  # 调用方法模式
   CallMethodVarArgs,  # 调用方法模式（可变参数）
   CallModule,  # 调用模块方法模式
   CallModuleVarArgs,  # 调用模块方法模式（可变参数）
   ExclusiveKeywordArg,  # 独占关键字参数模式
   Ignored,  # 忽略的参数模式
   KeywordArg,  # 关键字参数模式
   ListOf,  # 列表类型模式
   MultiOutputPattern,  # 多输出模式
   PatternExpr,  # 模式表达式
   RepeatedExpr,  # 重复表达式
   _TargetArgsExpr,  # 目标参数表达式
   _TargetExpr,  # 目标表达式
   _TargetExprVarArgs,  # 目标表达式（可变参数）
)

# 定义 rand_default 模式
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 定义 gt_Scalar 模式
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 定义 permute_default 模式
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 定义 expand_default 模式
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 定义 clone_default 模式
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 定义 view_default 模式
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 定义 permute_default_1 模式
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 定义 permute_default_2 模式
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 定义 expand_default_1 模式
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 定义 clone_default_1 模式
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 定义 view_default_1 模式
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 定义 bmm_default 模式
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 定义 view_default_2 模式
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 定义 div_Tensor 模式
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 定义 add_Tensor 模式
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
# 定义 amax_default 模式
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 定义 sub_Tensor 模式
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 定义 exp_default 模式
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义 sum_dim_IntList 模式
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义 div_Tensor_1 模式
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 定义 mul_Tensor 模式
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
# 定义 mul_Tensor_1 模式
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 定义 expand_default_2 模式
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 定义 view_default_3 模式
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 定义 permute_default_3 模式
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 定义 expand_default_3 模式
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 定义 clone_default_2 模式
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 定义 view_default_4 模式
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 view_default_3 和 view_default_4 作为参数，将结果赋给 bmm_default_1
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_1 和 Ignored() 作为参数，将结果赋给 view_default_5
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.neg.default 函数，传入 div_Tensor_1 作为参数，将结果赋给 neg_default
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 KeywordArg('tangents_1'), Ignored(), _users=2 作为参数，将结果赋给 view_default_6
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_4 和 Ignored() 作为参数，将结果赋给 permute_default_4
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 view_default_6 和 permute_default_4 作为参数，将结果赋给 bmm_default_2
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_2 和 Ignored() 作为参数，将结果赋给 view_default_7
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，传入 gt_Scalar 和 Ignored() 作为参数，将结果赋给 convert_element_type_default
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，传入 convert_element_type_default 和 Ignored() 作为参数，将结果赋给 mul_Tensor_2
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，传入 view_default_7 和 mul_Tensor_2 作为参数，将结果赋给 mul_Tensor_3
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，传入 mul_Tensor_3、div_Tensor_1 和 _users=2 作为参数，将结果赋给 mul_Tensor_4
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，传入 mul_Tensor_4、Ignored() 和 True 作为参数，将结果赋给 sum_dim_IntList_1
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 创建一个 CallFunction 对象，调用 prims.fma.default 函数，传入 neg_default、sum_dim_IntList_1 和 mul_Tensor_4 作为参数，将结果赋给 fma_default
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，传入 fma_default 和 KeywordArg('inv_scale') 作为参数，将结果赋给 div_Tensor_2
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, KeywordArg('inv_scale'))

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 div_Tensor_2、Ignored() 和 _users=2 作为参数，将结果赋给 view_default_8
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_1 和 Ignored() 作为参数，将结果赋给 permute_default_5
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 view_default_8 和 permute_default_5 作为参数，将结果赋给 bmm_default_3
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_3 和 Ignored() 作为参数，将结果赋给 view_default_9
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_9 和 Ignored() 作为参数，将结果赋给 permute_default_6
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default 和 Ignored() 作为参数，将结果赋给 permute_default_7
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 permute_default_7 和 view_default_8 作为参数，将结果赋给 bmm_default_4
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_4 和 Ignored() 作为参数，将结果赋给 view_default_10
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_10 和 Ignored() 作为参数，将结果赋给 permute_default_8
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 permute_default_8 和 Ignored() 作为参数，将结果赋给 permute_default_9
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_3 和 Ignored() 作为参数，将结果赋给 permute_default_10
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 permute_default_10 和 view_default_6 作为参数，将结果赋给 bmm_default_5
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_5 和 Ignored() 作为参数，将结果赋给 view_default_11
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_11 和 Ignored() 作为参数，将结果赋给 permute_default_11
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，包含 view_default_5、permute_default_6、permute_default_9、permute_default_11 等作为输出的一部分
_sfdp_pattern_16_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 KeywordArg('query') 和 Ignored() 作为参数，将结果赋给 permute_default
permute_default = CallFunction
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 Torch 的 expand 函数，默认模式，对 permute_default_2 进行扩展操作

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用 Torch 的 clone 函数，默认模式，克隆 expand_default_1，使用连续内存格式

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 调用 Torch 的 view 函数，默认模式，对 clone_default_1 进行视图变换

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 Torch 的 bmm 函数，默认模式，对 view_default 和 view_default_1 进行批量矩阵乘法运算

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 Torch 的 view 函数，默认模式，对 bmm_default 进行视图变换

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 Torch 的 div 函数，Tensor 模式，对 view_default_2 进行除法操作，指定 'inv_scale' 为关键参数

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
# 调用 Torch 的 add 函数，Tensor 模式，对 div_Tensor 进行加法操作，指定 'attn_mask' 为关键参数，并设置 _users 参数为 2

amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 调用 Torch 的 amax 函数，默认模式，对 add_Tensor 进行最大值计算，设置最后一个参数为 True

sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 调用 Torch 的 sub 函数，Tensor 模式，对 add_Tensor 和 amax_default 进行减法操作

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 Torch 的 exp 函数，默认模式，对 sub_Tensor 进行指数运算，设置 _users 参数为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 Torch 的 sum 函数，dim_IntList 模式，对 exp_default 进行按维度求和操作，设置最后一个参数为 True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 Torch 的 div 函数，Tensor 模式，对 exp_default 和 sum_dim_IntList 进行除法操作

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 调用 Torch 的 expand 函数，默认模式，对 div_Tensor_1 进行扩展操作

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 Torch 的 view 函数，默认模式，对 expand_default_2 进行视图变换

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 Torch 的 permute 函数，默认模式，指定 'value' 为关键参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 Torch 的 expand 函数，默认模式，对 permute_default_3 进行扩展操作

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用 Torch 的 clone 函数，默认模式，克隆 expand_default_3，使用连续内存格式

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 调用 Torch 的 view 函数，默认模式，对 clone_default_2 进行视图变换

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 Torch 的 bmm 函数，默认模式，对 view_default_3 和 view_default_4 进行批量矩阵乘法运算

_sfdp_pattern_16_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 Torch 的 view 函数，默认模式，对 bmm_default_1 进行视图变换，设置 _users 参数为 0


rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 Torch 的 rand 函数，默认模式，生成随机张量，忽略其他参数设置，不使用固定内存

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用 Torch 的 gt 函数，Scalar 模式，对 rand_default 进行大于比较，指定 'dropout_p' 为关键参数，并设置 _users 参数为 2

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 Torch 的 permute 函数，默认模式，指定 'query' 为关键参数

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 Torch 的 expand 函数，默认模式，对 permute_default 进行扩展操作

view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 调用 Torch 的 view 函数，默认模式，对 expand_default 进行视图变换，并设置 _users 参数为 2

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 Torch 的 permute 函数，默认模式，指定 'key' 为关键参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 Torch 的 permute 函数，默认模式，对 permute_default_1 进行再次排列操作

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 Torch 的 expand 函数，默认模式，对 permute_default_2 进行扩展操作

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 调用 Torch 的 view 函数，默认模式，对 expand_default_1 进行视图变换，并设置 _users 参数为 2

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 Torch 的 bmm 函数，默认模式，对 view_default 和 view_default_1 进行批量矩阵乘法运算

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 Torch 的 view 函数，默认模式，对 bmm_default 进行视图变换

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 Torch 的 div 函数，Tensor 模式，对 view_default_2 进行除法操作，指定 'inv_scale' 为关键参数

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
# 调用 Torch 的 add 函数，Tensor 模式，对 div_Tensor 进行加法操作，指定 'attn_mask' 为关键参数，并设置 _users 参数为 2

amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 调用 Torch 的 amax 函数，默认模式，对 add_Tensor 进行最大值计算，设置最后一个参数为 True

sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 调用 Torch 的 sub 函数，Tensor 模式，对 add_Tensor 和 amax_default 进行减法操作

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 Torch 的 exp 函数，默认模式，对 sub_Tensor 进行指数运算，设置 _users 参数为 2
# 调用函数 CallFunction，并传入 aten.sum.dim_IntList 作为参数 exp_default、Ignored()、True，并将结果赋给 sum_dim_IntList
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用函数 CallFunction，并传入 aten.div.Tensor 作为参数 exp_default、sum_dim_IntList，并将结果赋给 div_Tensor_1，_users=3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 调用函数 CallFunction，并传入 aten.mul.Tensor 作为参数 gt_Scalar、div_Tensor_1，并将结果赋给 mul_Tensor
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
# 调用函数 CallFunction，并传入 aten.mul.Tensor 作为参数 mul_Tensor、Ignored()，并将结果赋给 mul_Tensor_1
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 调用函数 CallFunction，并传入 aten.expand.default 作为参数 mul_Tensor_1、Ignored()，并将结果赋给 expand_default_2
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 expand_default_2、Ignored()，_users=2，并将结果赋给 view_default_3
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用函数 CallFunction，并传入 aten.permute.default 作为参数 KeywordArg('value')、Ignored()，并将结果赋给 permute_default_3
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用函数 CallFunction，并传入 aten.expand.default 作为参数 permute_default_3、Ignored()，并将结果赋给 expand_default_3
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 expand_default_3、Ignored()，_users=2，并将结果赋给 view_default_4
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用函数 CallFunction，并传入 aten.bmm.default 作为参数 view_default_3、view_default_4，并将结果赋给 bmm_default_1
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 bmm_default_1、Ignored()，并将结果赋给 view_default_5
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用函数 CallFunction，并传入 aten.neg.default 作为参数 div_Tensor_1，并将结果赋给 neg_default
neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 KeywordArg('tangents_1')、Ignored()，_users=2，并将结果赋给 view_default_6
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用函数 CallFunction，并传入 aten.permute.default 作为参数 view_default_4、Ignored()，并将结果赋给 permute_default_4
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用函数 CallFunction，并传入 aten.bmm.default 作为参数 view_default_6、permute_default_4，并将结果赋给 bmm_default_2
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 bmm_default_2、Ignored()，并将结果赋给 view_default_7
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用函数 CallFunction，并传入 prims.convert_element_type.default 作为参数 gt_Scalar、Ignored()，并将结果赋给 convert_element_type_default
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用函数 CallFunction，并传入 aten.mul.Tensor 作为参数 convert_element_type_default、Ignored()，并将结果赋给 mul_Tensor_2
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
# 调用函数 CallFunction，并传入 aten.mul.Tensor 作为参数 view_default_7、mul_Tensor_2，并将结果赋给 mul_Tensor_3
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 调用函数 CallFunction，并传入 aten.mul.Tensor 作为参数 mul_Tensor_3、div_Tensor_1，_users=2，并将结果赋给 mul_Tensor_4
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
# 调用函数 CallFunction，并传入 aten.sum.dim_IntList 作为参数 mul_Tensor_4、Ignored()、True，并将结果赋给 sum_dim_IntList_1
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 调用函数 CallFunction，并传入 prims.fma.default 作为参数 neg_default、sum_dim_IntList_1、mul_Tensor_4，并将结果赋给 fma_default
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用函数 CallFunction，并传入 aten.div.Tensor 作为参数 fma_default、KeywordArg('inv_scale')，并将结果赋给 div_Tensor_2
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, KeywordArg('inv_scale'))
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 div_Tensor_2、Ignored()，_users=2，并将结果赋给 view_default_8
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用函数 CallFunction，并传入 aten.permute.default 作为参数 view_default_1、Ignored()，并将结果赋给 permute_default_5
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用函数 CallFunction，并传入 aten.bmm.default 作为参数 view_default_8、permute_default_5，并将结果赋给 bmm_default_3
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 bmm_default_3、Ignored()，并将结果赋给 view_default_9
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用函数 CallFunction，并传入 aten.permute.default 作为参数 view_default_9、Ignored()，并将结果赋给 permute_default_6
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 调用函数 CallFunction，并传入 aten.permute.default 作为参数 view_default、Ignored()，并将结果赋给 permute_default_7
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用函数 CallFunction，并传入 aten.bmm.default 作为参数 permute_default_7、view_default_8，并将结果赋给 bmm_default_4
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用函数 CallFunction，并传入 aten.view.default 作为参数 bmm_default_4、Ignored()，并将结果赋给 view_default_10
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用函数 CallFunction，并传入 aten.permute.default 作为参数 view_default_10、Ignored()，并将结果赋给 permute_default_8
permute_default_
# 创建一个调用函数对象，使用 aten.permute.default 函数对 view_default_11 进行置换操作
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个多输出模式对象，包含多个元素，其中包括若干个 permute_default_X 对象和几个空值
_sfdp_pattern_16_bs1_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])

# 创建一个调用函数对象，使用 aten.permute.default 函数对关键字参数 'query' 进行置换操作
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，使用 aten.expand.default 函数对 permute_default 进行扩展操作
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，使用 aten.view.default 函数对 expand_default 进行视图变换操作
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 创建一个调用函数对象，使用 aten.permute.default 函数对关键字参数 'key' 进行置换操作
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，使用 aten.permute.default 函数对 permute_default_1 进行置换操作
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，使用 aten.expand.default 函数对 permute_default_2 进行扩展操作
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，使用 aten.view.default 函数对 expand_default_1 进行视图变换操作
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 创建一个调用函数对象，使用 aten.bmm.default 函数对 view_default 和 view_default_1 进行批次矩阵乘法
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，使用 aten.view.default 函数对 bmm_default 进行视图变换操作
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，使用 aten.div.Tensor 函数对 view_default_2 进行张量除法
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，使用 aten.add.Tensor 函数对 div_Tensor 进行张量加法
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)

# 创建一个调用函数对象，使用 aten.amax.default 函数对 add_Tensor 进行张量最大值计算
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 创建一个调用函数对象，使用 aten.sub.Tensor 函数对 add_Tensor 和 amax_default 进行张量减法
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 创建一个调用函数对象，使用 aten.exp.default 函数对 sub_Tensor 进行指数运算
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，使用 aten.sum.dim_IntList 函数对 exp_default 进行指定维度的张量求和
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，使用 aten.div.Tensor 函数对 exp_default 和 sum_dim_IntList 进行张量除法
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，使用 aten.expand.default 函数对 div_Tensor_1 进行扩展操作
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())

# 创建一个调用函数对象，使用 aten.view.default 函数对 expand_default_2 进行视图变换操作
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个调用函数对象，使用 aten.permute.default 函数对关键字参数 'value' 进行置换操作
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，使用 aten.expand.default 函数对 permute_default_3 进行扩展操作
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，使用 aten.view.default 函数对 expand_default_3 进行视图变换操作
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建一个调用函数对象，使用 aten.bmm.default 函数对 view_default_3 和 view_default_4 进行批次矩阵乘法
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，使用 aten.view.default 函数对 bmm_default_1 进行视图变换操作，设置 _users=0
_sfdp_pattern_16_bs1_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)

# 创建一个调用函数对象，使用 aten.rand.default 函数生成一个随机张量
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，使用 aten.gt.Scalar 函数对 rand_default 和关键字参数 'dropout_p' 进行大于比较操作
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 创建一个调用函数对象，使用 aten.clone.default 函数对 expand_default 进行克隆操作，使用连续内存格式
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，使用 aten.view.default 函数对 clone_default 进行视图变换操作，设置 _users=2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建一个调用函数对象，使用 aten.permute.default 函数对关键字参数 'key' 进行置换操作
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，使用 aten.permute.default 函数对 permute_default_1 进行置换操作
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，使用 aten.expand.default 函数对 permute_default_2 进行扩展操作
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建一个调用函数对象，调用 aten.clone.default 函数，传入参数 expand_default_1，并设置内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 clone_default_1 和 Ignored()，并设置 _users 参数为 2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default 和 Ignored()
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传入参数 view_default_2 和 KeywordArg('inv_scale')
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 aten.add.Tensor 函数，传入参数 div_Tensor 和 KeywordArg('attn_mask')
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 add_Tensor 和 Ignored()，设置 _users 参数为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.amax.default 函数，传入参数 convert_element_type_default 和 Ignored()，设置 True 参数为 True
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.sub.Tensor 函数，传入参数 convert_element_type_default 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个调用函数对象，调用 aten.exp.default 函数，传入参数 sub_Tensor，设置 _users 参数为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，传入参数 exp_default 和 Ignored()，设置 True 参数为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传入参数 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 div_Tensor_1 和 Ignored()，设置 _users 参数为 2
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 gt_Scalar 和 convert_element_type_default_1
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 mul_Tensor 和 Ignored()
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传入参数 mul_Tensor_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 expand_default_2 和 Ignored()，设置 _users 参数为 2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 KeywordArg('value') 和 Ignored()
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传入参数 permute_default_3 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，传入参数 expand_default_3，并设置内存格式为 torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 clone_default_2 和 Ignored()，设置 _users 参数为 2
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_1 和 Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 convert_element_type_default_1 和 Ignored()，设置 _users 参数为 2
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.neg.default 函数，传入参数 convert_element_type_default_2
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 KeywordArg('tangents_1') 和 Ignored()，设置 _users 参数为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_4 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_6 和 permute_default_4
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_2 和 Ignored()
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 gt_Scalar 和 Ignored()
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 convert_element_type_default_3 和 Ignored()
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 view_default_7 和 mul_Tensor_2
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入参数 mul_Tensor_3 和 Ignored()
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
# 使用 torch.Tensor 的乘法运算函数对 convert_element_type_default_4 和 convert_element_type_default_2 进行元素级别的乘法操作，结果赋给 mul_Tensor_4

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 对 mul_Tensor_4 进行指定维度的求和操作，将忽略一些参数，确保结果是一个张量

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用 fma 函数，进行 Fused Multiply-Add 操作，其中包括 neg_default，sum_dim_IntList_1 和 mul_Tensor_4

convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 使用默认的类型转换函数将 fma_default 转换为指定的类型

div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_5, KeywordArg('inv_scale'))
# 使用 torch.Tensor 的除法运算函数对 convert_element_type_default_5 进行除法操作，除数为 inv_scale

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 使用默认的视图函数对 div_Tensor_2 进行视图变换，忽略一些参数，并标记用户数为 2

permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 使用默认的维度重排函数对 view_default_1 进行维度重排操作

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 使用 torch.bmm 函数进行批量矩阵乘法操作，其中包括 view_default_8 和 permute_default_5

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 使用默认的视图函数对 bmm_default_3 进行视图变换

permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 使用默认的维度重排函数对 view_default_9 进行维度重排操作

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 使用默认的维度重排函数对 view_default 进行维度重排操作

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 使用 torch.bmm 函数进行批量矩阵乘法操作，其中包括 permute_default_7 和 view_default_8

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 使用默认的视图函数对 bmm_default_4 进行视图变换

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 使用默认的维度重排函数对 view_default_10 进行维度重排操作

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 使用默认的维度重排函数对 permute_default_8 进行维度重排操作

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 使用默认的维度重排函数对 view_default_3 进行维度重排操作

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 使用 torch.bmm 函数进行批量矩阵乘法操作，其中包括 permute_default_10 和 view_default_6

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 使用默认的视图函数对 bmm_default_5 进行视图变换

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 使用默认的维度重排函数对 view_default_11 进行维度重排操作

_sfdp_pattern_16_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])
# 创建一个多输出模式对象，其中包含视图变换和维度重排操作的结果

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 使用默认的维度重排函数对指定的 query 进行维度重排操作

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 使用默认的扩展函数对 permute_default 进行张量扩展操作

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 使用默认的克隆函数对 expand_default 进行张量克隆操作，并指定内存格式为连续格式

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 使用默认的视图函数对 clone_default 进行视图变换

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 使用默认的维度重排函数对指定的 key 进行维度重排操作

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 使用默认的维度重排函数对 permute_default_1 进行维度重排操作

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 使用默认的扩展函数对 permute_default_2 进行张量扩展操作

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 使用默认的克隆函数对 expand_default_1 进行张量克隆操作，并指定内存格式为连续格式

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 使用默认的视图函数对 clone_default_1 进行视图变换

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 使用 torch.bmm 函数进行批量矩阵乘法操作，其中包括 view_default 和 view_default_1

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 使用默认的视图函数对 bmm_default 进行视图变换

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 使用 torch.Tensor 的除法运算函数对 view_default_2 进行除法操作，除数为 inv_scale

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))
# 使用 torch.Tensor 的加法运算函数对 div_Tensor 进行加法操作，加数为 attn_mask
convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)
# 调用 prims.convert_element_type.default 函数，将 add_Tensor 转换为默认类型，忽略其他参数，用户数为 2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 aten.amax.default 函数，使用 convert_element_type_default 作为参数，忽略其他参数，设置 True 标志

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 aten.sub.Tensor 函数，使用 convert_element_type_default 和 amax_default 作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 aten.exp.default 函数，将 sub_Tensor 作为参数，用户数为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 aten.sum.dim_IntList 函数，使用 exp_default 作为参数，忽略其他参数，设置 True 标志

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为参数

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 调用 prims.convert_element_type.default 函数，将 div_Tensor_1 转换为默认类型，忽略其他参数

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 aten.expand.default 函数，使用 convert_element_type_default_1 作为参数，忽略其他参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 aten.view.default 函数，使用 expand_default_2 作为参数，忽略其他参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 aten.permute.default 函数，使用 'value' 关键字参数，忽略其他参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 aten.expand.default 函数，使用 permute_default_3 作为参数，忽略其他参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用 aten.clone.default 函数，使用 expand_default_3 作为参数，设置内存格式为 torch.contiguous_format

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 调用 aten.view.default 函数，使用 clone_default_2 作为参数，忽略其他参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 aten.bmm.default 函数，使用 view_default_3 和 view_default_4 作为参数

_sfdp_pattern_16_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 aten.view.default 函数，使用 bmm_default_1 作为参数，忽略其他参数，用户数为 0


rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 aten.rand.default 函数，忽略参数 Ignored()，dtype，device，pin_memory 设置为 False

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用 aten.gt.Scalar 函数，使用 rand_default 作为参数，关键字参数 'dropout_p'，用户数为 2

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 aten.permute.default 函数，使用 'query' 关键字参数，忽略其他参数

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 aten.expand.default 函数，使用 permute_default 作为参数，忽略其他参数

view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 调用 aten.view.default 函数，使用 expand_default 作为参数，忽略其他参数，用户数为 2

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 aten.permute.default 函数，使用 'key' 关键字参数，忽略其他参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 aten.permute.default 函数，使用 permute_default_1 作为参数，忽略其他参数

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 aten.expand.default 函数，使用 permute_default_2 作为参数，忽略其他参数

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 调用 aten.view.default 函数，使用 expand_default_1 作为参数，忽略其他参数，用户数为 2

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 aten.bmm.default 函数，使用 view_default 和 view_default_1 作为参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 aten.view.default 函数，使用 bmm_default 作为参数，忽略其他参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 aten.div.Tensor 函数，使用 view_default_2 作为参数，并设置关键字参数 'inv_scale'

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))
# 调用 aten.add.Tensor 函数，使用 div_Tensor 作为参数，并设置关键字参数 'attn_mask'

convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)
# 调用 prims.convert_element_type.default 函数，将 add_Tensor 转换为默认类型，忽略其他参数，用户数为 2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 aten.amax.default 函数，使用 convert_element_type_default 作为参数，忽略其他参数，设置 True 标志

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 aten.sub.Tensor 函数，使用 convert_element_type_default 和 amax_default 作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 aten.exp.default 函数，将 sub_Tensor 作为参数，用户数为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 aten.sum.dim_IntList 函数，使用 exp_default 作为参数，忽略其他参数，设置 True 标志

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 使用 prims.convert_element_type.default 函数将 div_Tensor_1 转换为默认类型
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
# 使用 aten.mul.Tensor 函数计算 gt_Scalar 与 convert_element_type_default_1 的张量乘法
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 继续使用 aten.mul.Tensor 函数计算 mul_Tensor 与 Ignored() 的张量乘法
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 使用 aten.expand.default 函数扩展 mul_Tensor_1 的维度
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 使用 aten.view.default 函数对 expand_default_2 进行视图变换，_users 参数指定用户数为 2
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 使用 aten.permute.default 函数对 KeywordArg('value') 进行置换操作
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 使用 aten.expand.default 函数扩展 permute_default_3 的维度
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 使用 aten.view.default 函数对 expand_default_3 进行视图变换，_users 参数指定用户数为 2
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 使用 aten.bmm.default 函数对 view_default_3 和 view_default_4 进行批量矩阵乘法
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 使用 aten.view.default 函数对 bmm_default_1 进行视图变换
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 使用 prims.convert_element_type.default 函数将 convert_element_type_default_1 转换为默认类型
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 使用 aten.neg.default 函数计算 convert_element_type_default_2 的负值
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 使用 aten.view.default 函数对 KeywordArg('tangents_1') 进行视图变换，_users 参数指定用户数为 2
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 使用 aten.permute.default 函数对 view_default_4 进行置换操作
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 使用 aten.bmm.default 函数对 view_default_6 和 permute_default_4 进行批量矩阵乘法
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 使用 aten.view.default 函数对 bmm_default_2 进行视图变换
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 使用 prims.convert_element_type.default 函数将 gt_Scalar 转换为默认类型
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
# 使用 aten.mul.Tensor 函数计算 convert_element_type_default_3 与 Ignored() 的张量乘法
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 继续使用 aten.mul.Tensor 函数计算 view_default_7 与 mul_Tensor_2 的张量乘法
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
# 使用 prims.convert_element_type.default 函数将 mul_Tensor_3 转换为默认类型
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
# 使用 aten.mul.Tensor 函数计算 convert_element_type_default_4 与 convert_element_type_default_2 的张量乘法，_users 参数指定用户数为 2
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 使用 aten.sum.dim_IntList 函数对 mul_Tensor_4 沿指定维度进行求和，True 表示保持维度
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 使用 prims.fma.default 函数计算 neg_default 与 sum_dim_IntList_1 的乘加操作，结果与 mul_Tensor_4 相乘
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 使用 prims.convert_element_type.default 函数将 fma_default 转换为默认类型
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_5, KeywordArg('inv_scale'))
# 使用 aten.div.Tensor 函数计算 convert_element_type_default_5 与 KeywordArg('inv_scale') 的张量除法
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 使用 aten.view.default 函数对 div_Tensor_2 进行视图变换，_users 参数指定用户数为 2
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 使用 aten.permute.default 函数对 view_default_1 进行置换操作
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 使用 aten.bmm.default 函数对 view_default_8 和 permute_default_5 进行批量矩阵乘法
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 使用 aten.view.default 函数对 bmm_default_3 进行视图变换
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 使用 aten.permute.default 函数对 view_default_9 进行置换操作
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 使用 aten.permute.default 函数对 view_default 进行置换操作
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 使用 aten.bmm.default 函数对 permute_default_7 和 view_default_8 进行批量矩阵乘法
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 使用 aten.view.default 函数对 bmm_default_4 进行视图变换
# 调用 torch.Tensor.permute 方法对 view_default_10 进行维度重排操作，保存结果至 permute_default_8
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 再次调用 torch.Tensor.permute 方法对 permute_default_8 进行维度重排操作，保存结果至 permute_default_9
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 调用 torch.Tensor.permute 方法对 view_default_3 进行维度重排操作，保存结果至 permute_default_10
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 调用 torch.Tensor.bmm 方法执行矩阵乘法操作，传入 permute_default_10 和 view_default_6，保存结果至 bmm_default_5
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 调用 torch.Tensor.view 方法对 bmm_default_5 进行视图变换，保存结果至 view_default_11
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 再次调用 torch.Tensor.permute 方法对 view_default_11 进行维度重排操作，保存结果至 permute_default_11
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，其中包含了 view_default_5, permute_default_6,
# permute_default_9, permute_default_11 这四个元素，并且有三个 None 占位符
_sfdp_pattern_16_half_bs1_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])


# 调用 torch.Tensor.permute 方法对输入的 'query' 进行维度重排操作，保存结果至 permute_default
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 调用 torch.Tensor.expand 方法对 permute_default 进行维度扩展操作，保存结果至 expand_default
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 调用 torch.Tensor.view 方法对 expand_default 进行视图变换，保存结果至 view_default
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 再次调用 torch.Tensor.permute 方法对输入的 'key' 进行维度重排操作，保存结果至 permute_default_1
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 再次调用 torch.Tensor.permute 方法对 permute_default_1 进行维度重排操作，保存结果至 permute_default_2
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 调用 torch.Tensor.expand 方法对 permute_default_2 进行维度扩展操作，保存结果至 expand_default_1
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 调用 torch.Tensor.view 方法对 expand_default_1 进行视图变换，保存结果至 view_default_1
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 调用 torch.Tensor.bmm 方法执行矩阵乘法操作，传入 view_default 和 view_default_1，保存结果至 bmm_default
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用 torch.Tensor.view 方法对 bmm_default 进行视图变换，保存结果至 view_default_2
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用 torch.Tensor.div 方法对 view_default_2 进行除法操作，传入 'inv_scale' 作为关键字参数，保存结果至 div_Tensor
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 调用 torch.Tensor.add 方法对 div_Tensor 进行加法操作，传入 'attn_mask' 作为关键字参数，保存结果至 add_Tensor
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))

# 调用 prims.convert_element_type 方法执行元素类型转换，传入 add_Tensor，忽略其它参数，保存结果至 convert_element_type_default
convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)

# 调用 torch.Tensor.amax 方法对 convert_element_type_default 进行计算最大值操作，忽略其它参数，保存结果至 amax_default
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 调用 torch.Tensor.sub 方法对 convert_element_type_default 和 amax_default 进行减法操作，保存结果至 sub_Tensor
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 调用 torch.Tensor.exp 方法对 sub_Tensor 进行指数运算，忽略其它参数，保存结果至 exp_default
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用 torch.Tensor.sum 方法对 exp_default 进行求和操作，传入维度列表 IntList，忽略其它参数，保存结果至 sum_dim_IntList
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用 torch.Tensor.div 方法对 exp_default 进行除法操作，传入 sum_dim_IntList 作为除数，保存结果至 div_Tensor_1
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 再次调用 prims.convert_element_type 方法执行元素类型转换，传入 div_Tensor_1，忽略其它参数，保存结果至 convert_element_type_default_1
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 调用 torch.Tensor.expand 方法对 convert_element_type_default_1 进行维度扩展操作，保存结果至 expand_default_2
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 调用 torch.Tensor.view 方法对 expand_default_2 进行视图变换，保存结果至 view_default_3
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 调用 torch.Tensor.permute 方法对输入的 'value' 进行维度重排操作，保存结果至 permute_default_3
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用 torch.Tensor.expand 方法对 permute_default_3 进行维度扩展操作，保存结果至 expand_default_3
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用 torch.Tensor.view 方法对 expand_default_3 进行视图变换，保存结果至 view_default_4
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 调用 torch.Tensor.bmm 方法执行矩阵乘法操作，传入 view_default_3 和 view_default_4，保存结果至 bmm_default_1
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用 torch.Tensor.view 方法对 bmm_default_1 进行视图变换，忽略其它参数，保存结果至 _sfdp_pattern_16_half_bs1_inference
_sfdp_pattern_16_half_bs1_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)


# 调用 torch.rand 方法生成随机张量，忽略其它参数，保存结果至 rand_default
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 调用 torch.gt 方法执行大于比较操作，传入 rand_default 和 'dropout_p' 作为关键字参数，保存结果至 gt_Scalar
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 创建一个调用函数对象，调用 torch 库中 aten.permute.default 函数，使用 'query' 作为关键字参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.expand.default 函数，传入 permute_default 结果和一个忽略的参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.clone.default 函数，传入 expand_default 结果、内存格式为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，传入 clone_default 结果、一个忽略的参数、并设置 _users=2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.permute.default 函数，使用 'key' 作为关键字参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.permute.default 函数，传入 permute_default_1 结果和一个忽略的参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.expand.default 函数，传入 permute_default_2 结果和一个忽略的参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.clone.default 函数，传入 expand_default_1 结果、内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，传入 clone_default_1 结果、一个忽略的参数、并设置 _users=2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.bmm.default 函数，传入 view_default 和 view_default_1 结果
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，传入 bmm_default 结果和一个忽略的参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.div.Tensor 函数，传入 view_default_2 结果和关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 torch 库中 aten.add.Tensor 函数，传入 div_Tensor 结果、关键字参数 'attn_mask'、并设置 _users=2
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.amax.default 函数，传入 add_Tensor 结果、一个忽略的参数、并设置 True
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 创建一个调用函数对象，调用 torch 库中 aten.sub.Tensor 函数，传入 add_Tensor 结果和 amax_default 结果
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 创建一个调用函数对象，调用 torch 库中 aten.exp.default 函数，传入 sub_Tensor 结果、并设置 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.sum.dim_IntList 函数，传入 exp_default 结果、一个忽略的参数、并设置 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 torch 库中 aten.div.Tensor 函数，传入 exp_default 和 sum_dim_IntList 结果、并设置 _users=3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建一个调用函数对象，调用 torch 库中 aten.mul.Tensor 函数，传入 gt_Scalar 和 div_Tensor_1 结果
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)

# 创建一个调用函数对象，调用 torch 库中 aten.mul.Tensor 函数，传入 mul_Tensor 结果和一个忽略的参数
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传入 mul_Tensor_1 结果和一个忽略的参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.expand.default 函数，传入 convert_element_type_default 结果和一个忽略的参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，传入 expand_default_2 结果、一个忽略的参数、并设置 _users=2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.permute.default 函数，使用 'value' 作为关键字参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.expand.default 函数，传入 permute_default_3 结果和一个忽略的参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.clone.default 函数，传入 expand_default_3 结果、内存格式为 torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，传入 clone_default_2 结果、一个忽略的参数、并设置 _users=2
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.bmm.default 函数，传入 view_default_3 和 view_default_4 结果
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，传入 bmm_default_1 结果和一个忽略的参数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.neg.default 函数，传入 div_Tensor_1 结果
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个调用函数对象，调用 torch 库中 aten.view.default 函数，使用 'tangents_1' 作为关键字参数，传入一个忽略的参数、并设置 _users=2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch 库中 aten.permute.default 函数，传入 view_default_4 结果和一个忽略的参数
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用函数对象，调用 torch 库中 aten.bmm.default 函数，传入 view_default_6 和 permute_default_4 结果
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用函数对象，调用 torch 库中 prims.convert_element_type.default 函数，传入 view_default_7 结果和一个忽
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用函数 prims.convert_element_type.default，将 gt_Scalar 和 Ignored() 作为参数传入，返回结果赋给 convert_element_type_default_2

mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_2, Ignored())
# 调用 PyTorch 的 aten.mul.Tensor 函数，以 convert_element_type_default_2 和 Ignored() 为参数，返回结果赋给 mul_Tensor_2

mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default_1, mul_Tensor_2)
# 再次调用 PyTorch 的 aten.mul.Tensor 函数，以 convert_element_type_default_1 和 mul_Tensor_2 为参数，返回结果赋给 mul_Tensor_3

mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
# 调用 PyTorch 的 aten.mul.Tensor 函数，以 mul_Tensor_3、div_Tensor_1 和 _users=2 为参数，返回结果赋给 mul_Tensor_4

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 调用 PyTorch 的 aten.sum.dim_IntList 函数，以 mul_Tensor_4、Ignored() 和 True 为参数，返回结果赋给 sum_dim_IntList_1

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用函数 prims.fma.default，将 neg_default、sum_dim_IntList_1 和 mul_Tensor_4 作为参数传入，返回结果赋给 fma_default

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 再次调用函数 prims.convert_element_type.default，以 fma_default 和 Ignored() 为参数传入，返回结果赋给 convert_element_type_default_3

div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_3, KeywordArg('inv_scale'))
# 调用 PyTorch 的 aten.div.Tensor 函数，以 convert_element_type_default_3 和 KeywordArg('inv_scale') 为参数，返回结果赋给 div_Tensor_2

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 PyTorch 的 aten.view.default 函数，以 div_Tensor_2、Ignored() 和 _users=2 为参数，返回结果赋给 view_default_8

permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 view_default_1 和 Ignored() 为参数，返回结果赋给 permute_default_5

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 调用 PyTorch 的 aten.bmm.default 函数，以 view_default_8 和 permute_default_5 为参数，返回结果赋给 bmm_default_3

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，以 bmm_default_3 和 Ignored() 为参数，返回结果赋给 view_default_9

permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 view_default_9 和 Ignored() 为参数，返回结果赋给 permute_default_6

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 view_default 和 Ignored() 为参数，返回结果赋给 permute_default_7

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用 PyTorch 的 aten.bmm.default 函数，以 permute_default_7 和 view_default_8 为参数，返回结果赋给 bmm_default_4

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，以 bmm_default_4 和 Ignored() 为参数，返回结果赋给 view_default_10

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 view_default_10 和 Ignored() 为参数，返回结果赋给 permute_default_8

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 permute_default_8 和 Ignored() 为参数，返回结果赋给 permute_default_9

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 view_default_3 和 Ignored() 为参数，返回结果赋给 permute_default_10

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 调用 PyTorch 的 aten.bmm.default 函数，以 permute_default_10 和 view_default_6 为参数，返回结果赋给 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，以 bmm_default_5 和 Ignored() 为参数，返回结果赋给 view_default_11

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 view_default_11 和 Ignored() 为参数，返回结果赋给 permute_default_11

_sfdp_pattern_16_half_mask_fp32_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])
# 创建一个 MultiOutputPattern 对象，包含了 view_default_5、permute_default_6、permute_default_9 和 permute_default_11 作为其输出模式的一部分

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 KeywordArg('query') 和 Ignored() 为参数，返回结果赋给 permute_default

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，以 permute_default 和 Ignored() 为参数，返回结果赋给 expand_default

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 aten.clone.default 函数，以 expand_default 和 memory_format=torch.contiguous_format 为参数，返回结果赋给 clone_default

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，以 clone_default 和 Ignored() 为参数，返回结果赋给 view_default

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，以 KeywordArg('key') 和 Ignored() 为参数，返回结果赋给 permute_default_1

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 再次调用 PyTorch 的 aten.permute.default 函数，以 permute_default_1 和 Ignored() 为参数，返回结果赋给 permute_default_2

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，以 permute_default_2 和 Ignored() 为参数，返回结果赋给 expand_default_1

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 aten.clone.default 函数，以 expand_default_1 和 memory_format=torch.contiguous_format 为参数，返回结果赋给 clone_default_1

view_default_1 = CallFunction(aten.view.default, clone_default_
# 创建一个名为 div_Tensor 的张量对象，使用 aten.div.Tensor 函数，传入 view_default_2 和关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个名为 add_Tensor 的张量对象，使用 aten.add.Tensor 函数，传入 div_Tensor 和关键字参数 'attn_mask'，同时指定 _users=2
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)

# 创建一个名为 amax_default 的对象，使用 aten.amax.default 函数，传入 add_Tensor 和一个未使用的参数对象，同时指定 True 参数
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 创建一个名为 sub_Tensor 的张量对象，使用 aten.sub.Tensor 函数，传入 add_Tensor 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 创建一个名为 exp_default 的对象，使用 aten.exp.default 函数，传入 sub_Tensor，并指定 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个名为 sum_dim_IntList 的对象，使用 aten.sum.dim_IntList 函数，传入 exp_default 和一个未使用的参数对象，同时指定 True 参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个名为 div_Tensor_1 的张量对象，使用 aten.div.Tensor 函数，传入 exp_default 和 sum_dim_IntList，同时指定 _users=3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建一个名为 convert_element_type_default 的对象，使用 prims.convert_element_type.default 函数，传入 div_Tensor_1 和一个未使用的参数对象
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个名为 expand_default_2 的对象，使用 aten.expand.default 函数，传入 convert_element_type_default 和一个未使用的参数对象
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 创建一个名为 view_default_3 的对象，使用 aten.view.default 函数，传入 expand_default_2 和一个未使用的参数对象
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个名为 permute_default_3 的对象，使用 aten.permute.default 函数，传入一个关键字参数 'value' 和一个未使用的参数对象
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个名为 expand_default_3 的对象，使用 aten.expand.default 函数，传入 permute_default_3 和一个未使用的参数对象
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个名为 clone_default_2 的对象，使用 aten.clone.default 函数，传入 expand_default_3 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个名为 view_default_4 的对象，使用 aten.view.default 函数，传入 clone_default_2 和一个未使用的参数对象
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个名为 bmm_default_1 的对象，使用 aten.bmm.default 函数，传入 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个名为 _sfdp_pattern_16_half_mask_fp32_inference 的对象，使用 aten.view.default 函数，传入 bmm_default_1 和一个未使用的参数对象，同时指定 _users=0
_sfdp_pattern_16_half_mask_fp32_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)

# 创建一个名为 rand_default 的对象，使用 aten.rand.default 函数，传入一个未使用的参数对象和设备、数据类型等信息
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个名为 gt_Scalar 的对象，使用 aten.gt.Scalar 函数，传入 rand_default 和关键字参数 'dropout_p'，同时指定 _users=2
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 创建一个名为 permute_default 的对象，使用 aten.permute.default 函数，传入一个关键字参数 'query' 和一个未使用的参数对象
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个名为 expand_default 的对象，使用 aten.expand.default 函数，传入 permute_default 和一个未使用的参数对象
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个名为 view_default 的对象，使用 aten.view.default 函数，传入 expand_default 和一个未使用的参数对象，同时指定 _users=2
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)

# 创建一个名为 permute_default_1 的对象，使用 aten.permute.default 函数，传入一个关键字参数 'key' 和一个未使用的参数对象
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个名为 permute_default_2 的对象，使用 aten.permute.default 函数，传入 permute_default_1 和一个未使用的参数对象
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个名为 expand_default_1 的对象，使用 aten.expand.default 函数，传入 permute_default_2 和一个未使用的参数对象
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个名为 view_default_1 的对象，使用 aten.view.default 函数，传入 expand_default_1 和一个未使用的参数对象，同时指定 _users=2
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)

# 创建一个名为 bmm_default 的对象，使用 aten.bmm.default 函数，传入 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个名为 view_default_2 的对象，使用 aten.view.default 函数，传入 bmm_default 和一个未使用的参数对象
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个名为 div_Tensor 的张量对象，使用 aten.div.Tensor 函数，传入 view_default_2 和关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个名为 add_Tensor 的张量对象，使用 aten.add.Tensor 函数，传入 div_Tensor 和关键字参数 'attn_mask'，同时指定 _users=2
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)

# 创建一个名为 amax_default 的对象，使用 aten.amax.default 函数，传入 add_Tensor 和一个未使用的参数对象，同时指定 True 参数
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 创建一个名为 sub_Tensor 的张量对象，使用 aten.sub.Tensor 函数，传入 add_Tensor 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 创建一个名为 exp_default 的对象，使用 aten.exp.default 函数，传入 sub_Tensor，并指定 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个名为 sum_dim_IntList 的对象，使用 aten.sum.dim_IntList 函数，传入 exp_default 和一个未使用的参数对象，同时指定 True 参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个名为 div_Tensor_1 的张量对象，使用 aten.div.Tensor 函数，传入 exp_default 和 sum_dim_IntList，并指定 _users=3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建一个名为 mul_Tensor 的张量对象，使用 aten.mul.Tensor 函数，传入 gt_S
# 定义一个调用函数对象，调用 prims.convert_element_type.default 函数
convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())

# 调用 aten.expand.default 函数，将 convert_element_type_default 参数进行扩展
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 调用 aten.view.default 函数，对 expand_default_2 进行视图变换，忽略其中的参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 调用 aten.permute.default 函数，对第一个参数使用默认排列顺序
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用 aten.expand.default 函数，将 permute_default_3 参数进行扩展
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用 aten.view.default 函数，对 expand_default_3 进行视图变换，忽略其中的参数
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)

# 调用 aten.bmm.default 函数，进行批次矩阵乘法操作
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用 aten.view.default 函数，对 bmm_default_1 进行视图变换，忽略其中的参数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 调用 aten.neg.default 函数，对 div_Tensor_1 进行取负操作
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 调用 aten.view.default 函数，对第一个参数进行视图变换，忽略其中的参数
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 调用 aten.permute.default 函数，对 view_default_4 进行排列操作
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 调用 aten.bmm.default 函数，进行批次矩阵乘法操作
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 调用 aten.view.default 函数，对 bmm_default_2 进行视图变换，忽略其中的参数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 调用 prims.convert_element_type.default 函数，将 view_default_7 参数转换为指定类型
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())

# 调用 prims.convert_element_type.default 函数，将 gt_Scalar 参数转换为指定类型
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 调用 aten.mul.Tensor 函数，进行张量相乘操作
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_2, Ignored())

# 调用 aten.mul.Tensor 函数，进行张量相乘操作
mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default_1, mul_Tensor_2)

# 调用 aten.mul.Tensor 函数，进行张量相乘操作
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)

# 调用 aten.sum.dim_IntList 函数，沿指定维度对张量进行求和操作
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 调用 prims.fma.default 函数，进行 Fused Multiply-Add 操作
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 调用 prims.convert_element_type.default 函数，将 fma_default 参数转换为指定类型
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 调用 aten.div.Tensor 函数，进行张量除法操作
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_3, KeywordArg('inv_scale'))

# 调用 aten.view.default 函数，对 div_Tensor_2 进行视图变换，忽略其中的参数
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 调用 aten.permute.default 函数，对 view_default_1 进行排列操作
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 调用 aten.bmm.default 函数，进行批次矩阵乘法操作
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 调用 aten.view.default 函数，对 bmm_default_3 进行视图变换，忽略其中的参数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 调用 aten.permute.default 函数，对 view_default_9 进行排列操作
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 调用 aten.permute.default 函数，对 view_default 进行排列操作
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 调用 aten.bmm.default 函数，进行批次矩阵乘法操作
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 调用 aten.view.default 函数，对 bmm_default_4 进行视图变换，忽略其中的参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 调用 aten.permute.default 函数，对 view_default_10 进行排列操作
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 调用 aten.permute.default 函数，对 permute_default_8 进行排列操作
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 调用 aten.permute.default 函数，对 view_default_3 进行排列操作
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 调用 aten.bmm.default 函数，进行批次矩阵乘法操作
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 调用函数 `aten.view.default`，传入参数 `bmm_default_5` 和 `Ignored()`，返回视图对象 `view_default_11`
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 调用函数 `aten.permute.default`，传入参数 `view_default_11` 和 `Ignored()`，返回排列后的对象 `permute_default_11`
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个多输出模式对象 `_sfdp_pattern_16_half_mask_fp32_bs1_training`，包含多个元素
# 这些元素依次为 `view_default_5`, `permute_default_6`, `permute_default_9`, `permute_default_11`，以及三个 `None` 值
_sfdp_pattern_16_half_mask_fp32_bs1_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])

# 调用函数 `aten.permute.default`，传入关键字参数 `query` 和 `Ignored()`，返回排列后的对象 `permute_default`
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 调用函数 `aten.expand.default`，传入参数 `permute_default` 和 `Ignored()`，返回扩展后的对象 `expand_default`
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 调用函数 `aten.view.default`，传入参数 `expand_default` 和 `Ignored()`，返回视图对象 `view_default`
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 调用函数 `aten.permute.default`，传入关键字参数 `key` 和 `Ignored()`，返回排列后的对象 `permute_default_1`
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 调用函数 `aten.permute.default`，传入参数 `permute_default_1` 和 `Ignored()`，返回排列后的对象 `permute_default_2`
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 调用函数 `aten.expand.default`，传入参数 `permute_default_2` 和 `Ignored()`，返回扩展后的对象 `expand_default_1`
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 调用函数 `aten.view.default`，传入参数 `expand_default_1` 和 `Ignored()`，返回视图对象 `view_default_1`
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 调用函数 `aten.bmm.default`，传入参数 `view_default` 和 `view_default_1`，返回矩阵乘积对象 `bmm_default`
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用函数 `aten.view.default`，传入参数 `bmm_default` 和 `Ignored()`，返回视图对象 `view_default_2`
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用函数 `aten.div.Tensor`，传入参数 `view_default_2` 和关键字参数 `inv_scale`，返回除法操作对象 `div_Tensor`
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 调用函数 `aten.add.Tensor`，传入参数 `div_Tensor`、关键字参数 `attn_mask`，以及 `_users=2`，返回加法操作对象 `add_Tensor`
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)

# 调用函数 `aten.amax.default`，传入参数 `add_Tensor` 和 `Ignored()`，以及参数 `True`，返回最大值对象 `amax_default`
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 调用函数 `aten.sub.Tensor`，传入参数 `add_Tensor` 和 `amax_default`，返回减法操作对象 `sub_Tensor`
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 调用函数 `aten.exp.default`，传入参数 `sub_Tensor`，以及 `_users=2`，返回指数操作对象 `exp_default`
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用函数 `aten.sum.dim_IntList`，传入参数 `exp_default` 和 `Ignored()`，以及参数 `True`，返回按维度求和对象 `sum_dim_IntList`
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用函数 `aten.div.Tensor`，传入参数 `exp_default` 和 `sum_dim_IntList`，返回除法操作对象 `div_Tensor_1`
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 调用函数 `prims.convert_element_type.default`，传入参数 `div_Tensor_1` 和 `Ignored()`，返回类型转换对象 `convert_element_type_default`
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 调用函数 `aten.expand.default`，传入参数 `convert_element_type_default` 和 `Ignored()`，返回扩展后的对象 `expand_default_2`
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 调用函数 `aten.view.default`，传入参数 `expand_default_2` 和 `Ignored()`，返回视图对象 `view_default_3`
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 调用函数 `aten.permute.default`，传入关键字参数 `value` 和 `Ignored()`，返回排列后的对象 `permute_default_3`
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用函数 `aten.expand.default`，传入参数 `permute_default_3` 和 `Ignored()`，返回扩展后的对象 `expand_default_3`
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用函数 `aten.view.default`，传入参数 `expand_default_3` 和 `Ignored()`，返回视图对象 `view_default_4`
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 调用函数 `aten.bmm.default`，传入参数 `view_default_3` 和 `view_default_4`，返回矩阵乘积对象 `bmm_default_1`
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用函数 `aten.view.default`，传入参数 `bmm_default_1` 和 `Ignored()`，以及 `_users=0`，返回视图对象 `_sfdp_pattern_16_half_mask_fp32_bs1_inference`
_sfdp_pattern_16_half_mask_fp32_bs1_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```