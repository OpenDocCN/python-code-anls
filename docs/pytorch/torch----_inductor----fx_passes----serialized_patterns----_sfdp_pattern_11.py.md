# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_11.py`

```py
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 库
import torch
# 导入 torch 的私有模块 torch._inductor
import torch._inductor

# 设置 aten 和 prims 变量，用于调用 torch.ops 下的 aten 和 prims 操作
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块导入一系列类
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

# 创建 permute_default 变量，调用 aten.permute.default 函数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 创建 expand_default 变量，调用 aten.expand.default 函数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建 clone_default 变量，调用 aten.clone.default 函数，并指定 memory_format 参数
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建 view_default 变量，调用 aten.view.default 函数，指定 _users 参数
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建一系列类似的变量，每个变量都是一个函数调用的模式匹配
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'), _users=2)

amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)

sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())

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
# 调用 aten.bmm.default 函数，执行矩阵乘法运算
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 调用 aten.mul.Tensor 函数，执行张量乘法操作
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_7, div_Tensor_1, _users=2)

# 调用 aten.sum.dim_IntList 函数，按指定维度对张量进行求和操作
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)

# 调用 prims.fma.default 函数，执行 fused multiply-add 操作
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)

# 调用 aten.div.Tensor 函数，执行张量除法操作
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, KeywordArg('inv_scale'))

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 调用 aten.bmm.default 函数，执行矩阵乘法运算
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 调用 aten.bmm.default 函数，执行矩阵乘法运算
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 调用 aten.bmm.default 函数，执行矩阵乘法运算
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建 MultiOutputPattern 对象，包含多个输出模式
_sfdp_pattern_11_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None
])

# 调用 aten.permute.default 函数，执行张量维度重新排列操作，使用关键字参数 'query'
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 调用 aten.expand.default 函数，执行张量扩展操作
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 调用 aten.clone.default 函数，执行张量克隆操作
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作，使用关键字参数 'key'
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 调用 aten.permute.default 函数，执行张量维度重新排列操作
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 调用 aten.expand.default 函数，执行张量扩展操作
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 调用 aten.clone.default 函数，执行张量克隆操作
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 调用 aten.bmm.default 函数，执行矩阵乘法运算
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用 aten.view.default 函数，执行张量视图变换操作
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用 aten.div.Tensor 函数，执行张量除法操作
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'), _users=2)

# 调用 aten.amax.default 函数，计算张量的最大值
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)

# 调用 aten.sub.Tensor 函数，执行张量减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 计算张量的指数函数，对 sub_Tensor 进行操作，_users 参数设置为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 对 exp_default 张量进行按维度求和操作，Ignored() 表示忽略其中一个参数，True 表示保持维度

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 对 exp_default 张量按元素进行除法操作，sum_dim_IntList 为除数

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 对 div_Tensor_1 张量进行维度扩展操作，Ignored() 表示忽略其中一个参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 对 expand_default_2 张量进行视图变换操作，Ignored() 表示忽略其中一个参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 对张量进行维度重新排列操作，使用关键字参数 'value'，Ignored() 表示忽略其中一个参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 对 permute_default_3 张量进行维度扩展操作，Ignored() 表示忽略其中一个参数
# 创建一个调用函数对象，调用 aten.clone.default 函数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.neg.default 函数
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)

# 创建一个调用函数对象，调用 prims.fma.default 函数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_4, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个调用函数对象，调用 aten.view.default 函数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，包含多个调用函数对象
_sfdp_pattern_11_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None
])

# 创建一个调用函数对象，调用 aten.permute.default 函数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，使用 expand_default 作为参数，并指定内存格式为连续格式
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 clone_default 作为参数，忽略第二个参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 KeywordArg('key') 作为参数，忽略第二个参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 permute_default_1 作为参数，忽略第二个参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 permute_default_2 作为参数，忽略第二个参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，使用 expand_default_1 作为参数，并指定内存格式为连续格式
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 clone_default_1 作为参数，忽略第二个参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default 和 view_default_1 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default 作为参数，忽略第二个参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，使用 view_default_2 作为参数，并指定关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，使用 div_Tensor 作为参数，并忽略第二个和第三个参数，_users=2
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.amax.default 函数，使用 convert_element_type_default 作为参数，并忽略第二个和第三个参数，同时使用 True 作为关键字参数
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.sub.Tensor 函数，使用 convert_element_type_default 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个 CallFunction 对象，调用 aten.exp.default 函数，使用 sub_Tensor 作为参数，并指定 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，使用 exp_default 作为参数，并忽略第二个和第三个参数，同时使用 True 作为关键字参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，使用 div_Tensor_1 作为参数，并忽略第二个参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 convert_element_type_default_1 作为参数，忽略第二个参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 expand_default_2 作为参数，忽略第二个参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 KeywordArg('value') 作为参数，忽略第二个参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 permute_default_3 作为参数，忽略第二个参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，使用 expand_default_3 作为参数，并指定内存格式为连续格式
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 clone_default_2 作为参数，忽略第二个参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_1 作为参数，并指定 _users=0，忽略第三个参数
_sfdp_pattern_11_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```