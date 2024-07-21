# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_12.py`

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

# 从 torch.ops 中导入 aten 和 prims 操作
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块中导入各种模式匹配类
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

# 定义模式匹配对象 rand_default
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 定义模式匹配对象 gt_Scalar
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 定义模式匹配对象 permute_default
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 定义模式匹配对象 expand_default
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 定义模式匹配对象 clone_default
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 定义模式匹配对象 view_default
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 定义模式匹配对象 permute_default_1
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 定义模式匹配对象 permute_default_2
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 定义模式匹配对象 expand_default_1
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 定义模式匹配对象 clone_default_1
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 定义模式匹配对象 view_default_1
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 定义模式匹配对象 bmm_default
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 定义模式匹配对象 view_default_2
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 定义模式匹配对象 div_Tensor
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'), _users=2)
# 定义模式匹配对象 amax_default
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
# 定义模式匹配对象 sub_Tensor
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
# 定义模式匹配对象 exp_default
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义模式匹配对象 sum_dim_IntList
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义模式匹配对象 div_Tensor_1
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 定义模式匹配对象 mul_Tensor
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
# 定义模式匹配对象 mul_Tensor_1
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 定义模式匹配对象 expand_default_2
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 定义模式匹配对象 view_default_3
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 定义模式匹配对象 permute_default_3
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 定义模式匹配对象 expand_default_3
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 定义模式匹配对象 clone_default_2
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 定义模式匹配对象 view_default_4
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 定义模式匹配对象 bmm_default_1
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 Torch 的 view 函数，使用 bmm_default_1 作为输入，生成一个新的视图

neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 调用 Torch 的 neg 函数，对 div_Tensor_1 进行取负运算，生成结果 neg_default

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 Torch 的 view 函数，使用 'tangents_1' 作为参数进行视图操作，标记有两个用户

permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_4 进行维度置换操作，生成 permute_default_4

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 调用 Torch 的 bmm 函数，对 view_default_6 和 permute_default_4 进行批量矩阵乘法，生成 bmm_default_2

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 Torch 的 view 函数，使用 bmm_default_2 作为输入，生成一个新的视图

convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用 Torch 的 convert_element_type 函数，将 gt_Scalar 转换为默认的元素类型，生成 convert_element_type_default

mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
# 调用 Torch 的 mul 函数，对 convert_element_type_default 进行张量乘法，生成 mul_Tensor_2

mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 调用 Torch 的 mul 函数，对 view_default_7 和 mul_Tensor_2 进行张量乘法，生成 mul_Tensor_3

mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
# 调用 Torch 的 mul 函数，对 mul_Tensor_3 和 div_Tensor_1 进行张量乘法，标记有两个用户

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 调用 Torch 的 sum 函数，对 mul_Tensor_4 按照 IntList 中指定的维度进行求和，保持维度

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用 Torch 的 fma 函数，执行 Fused Multiply-Add 操作，生成 fma_default

div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, KeywordArg('inv_scale_factor'))
# 调用 Torch 的 div 函数，对 fma_default 进行张量除法，使用 'inv_scale_factor' 作为参数，生成 div_Tensor_2

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 Torch 的 view 函数，使用 div_Tensor_2 作为输入，生成一个新的视图，标记有两个用户

permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_1 进行维度置换操作，生成 permute_default_5

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 调用 Torch 的 bmm 函数，对 view_default_8 和 permute_default_5 进行批量矩阵乘法，生成 bmm_default_3

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 Torch 的 view 函数，使用 bmm_default_3 作为输入，生成一个新的视图

permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_9 进行维度置换操作，生成 permute_default_6

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 Torch 的 permute 函数，对 view_default 进行维度置换操作，生成 permute_default_7

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用 Torch 的 bmm 函数，对 permute_default_7 和 view_default_8 进行批量矩阵乘法，生成 bmm_default_4

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 Torch 的 view 函数，使用 bmm_default_4 作为输入，生成一个新的视图

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_10 进行维度置换操作，生成 permute_default_8

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 调用 Torch 的 permute 函数，对 permute_default_8 进行维度置换操作，生成 permute_default_9

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_3 进行维度置换操作，生成 permute_default_10

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 调用 Torch 的 bmm 函数，对 permute_default_10 和 view_default_6 进行批量矩阵乘法，生成 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 Torch 的 view 函数，使用 bmm_default_5 作为输入，生成一个新的视图

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 调用 Torch 的 permute 函数，对 view_default_11 进行维度置换操作，生成 permute_default_11

_sfdp_pattern_12_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])
# 创建一个多输出模式对象 _sfdp_pattern_12_training，包含若干个函数调用作为其中的模式元素

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 Torch 的 permute 函数，使用 'query' 作为参数进行维度置换操作，生成 permute_default

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 Torch 的 expand 函数，对 permute_default 进行张量扩展操作，生成 expand_default

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用 Torch 的 clone 函数，对 expand_default 进行张量克隆操作，并指定内存格式为 torch.contiguous_format，生成 clone_default

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 调用 Torch 的 view 函数，使用 clone_default 作为输入，生成一个新的视图

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 Torch 的 permute 函数，使用 'key' 作为参数进行维度置换操作，生成 permute_default_1

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 Torch 的 permute 函数，对 permute_default_1 进行维度置换操作，生成 permute_default_2

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 Torch 的 expand 函数，对 permute_default_2 进行张量扩展操作，生成 expand_default_1
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用默认的aten.clone函数，使用expand_default_1作为参数，并指定内存格式为连续格式

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 调用默认的aten.view函数，使用clone_default_1作为参数，并忽略第二个参数

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用默认的aten.bmm函数，使用view_default和view_default_1作为参数，进行批量矩阵乘法计算

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用默认的aten.view函数，使用bmm_default作为参数，并忽略第二个参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'), _users=2)
# 调用Tensor类的aten.div方法，使用view_default_2作为第一个参数，指定inv_scale_factor作为关键字参数，并且用户数为2

amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
# 调用默认的aten.amax函数，使用div_Tensor作为参数，并忽略第二个参数，同时指定keepdim参数为True

sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
# 调用Tensor类的aten.sub方法，使用div_Tensor和amax_default作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用默认的aten.exp函数，使用sub_Tensor作为参数，并指定用户数为2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用aten.sum.dim_IntList方法，使用exp_default作为第一个参数，并忽略第二个参数，同时指定keepdim参数为True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用Tensor类的aten.div方法，使用exp_default作为第一个参数，sum_dim_IntList作为第二个参数

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 调用默认的aten.expand函数，使用div_Tensor_1作为参数，并忽略第二个参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用默认的aten.view函数，使用expand_default_2作为参数，并忽略第二个参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用默认的aten.permute函数，指定value作为关键字参数，并忽略第二个参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用默认的aten.expand函数，使用permute_default_3作为参数，并忽略第二个参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用默认的aten.clone函数，使用expand_default_3作为参数，并指定内存格式为连续格式

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 调用默认的aten.view函数，使用clone_default_2作为参数，并忽略第二个参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用默认的aten.bmm函数，使用view_default_3和view_default_4作为参数，进行批量矩阵乘法计算

_sfdp_pattern_12_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用默认的aten.view函数，使用bmm_default_1作为参数，并忽略第二个参数，同时指定用户数为0

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用默认的aten.rand函数，忽略第一个参数，同时忽略dtype、device，并指定pin_memory参数为False

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用Scalar类的aten.gt方法，使用rand_default作为第一个参数，指定dropout_p作为关键字参数，并且用户数为2

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用默认的aten.permute函数，指定query作为关键字参数，并忽略第二个参数

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用默认的aten.expand函数，使用permute_default作为参数，并忽略第二个参数

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用默认的aten.clone函数，使用expand_default作为参数，并指定内存格式为连续格式

view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 调用默认的aten.view函数，使用clone_default作为参数，并忽略第二个参数，同时指定用户数为2

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用默认的aten.permute函数，指定key作为关键字参数，并忽略第二个参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用默认的aten.permute函数，使用permute_default_1作为参数，并忽略第二个参数

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用默认的aten.expand函数，使用permute_default_2作为参数，并忽略第二个参数

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用默认的aten.clone函数，使用expand_default_1作为参数，并指定内存格式为连续格式

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 调用默认的aten.view函数，使用clone_default_1作为参数，并忽略第二个参数，同时指定用户数为2

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用默认的aten.bmm函数，使用view_default和view_default_1作为参数，进行批量矩阵乘法计算

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用默认的aten.view函数，使用bmm_default作为参数，并忽略第二个参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'))
# 调用Tensor类的aten.div方法，使用view_default_2作为第一个参数，并指定inv_scale_factor作为关键字参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 调用prims.convert_element_type.default函数，使用div_Tensor作为第一个参数，并忽略第二个参数，同时指定用户数为2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用默认的aten.amax函数，使用convert_element_type_default作为参数，并忽略第二个参数，同时指定keepdim参数为True
# 使用 CallFunction 调用 aten.sub.Tensor 函数，传入 convert_element_type_default 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 使用 CallFunction 调用 aten.exp.default 函数，传入 sub_Tensor 作为参数，_users=2 表示此结果被多个地方使用
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 使用 CallFunction 调用 aten.sum.dim_IntList 函数，传入 exp_default 和 Ignored()（未使用的参数），True 表示按维度求和
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 使用 CallFunction 调用 aten.div.Tensor 函数，传入 exp_default 和 sum_dim_IntList 作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 使用 CallFunction 调用 prims.convert_element_type.default 函数，传入 div_Tensor_1 和 Ignored()（未使用的参数），_users=2 表示此结果被多个地方使用
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)

# 使用 CallFunction 调用 aten.mul.Tensor 函数，传入 gt_Scalar 和 convert_element_type_default_1 作为参数
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)

# 使用 CallFunction 调用 aten.mul.Tensor 函数，传入 mul_Tensor 和 Ignored()（未使用的参数）
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 使用 CallFunction 调用 aten.expand.default 函数，传入 mul_Tensor_1 和 Ignored()（未使用的参数）
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())

# 使用 CallFunction 调用 aten.view.default 函数，传入 expand_default_2 和 Ignored()（未使用的参数），_users=2 表示此结果被多个地方使用
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 使用 CallFunction 调用 aten.permute.default 函数，传入 KeywordArg('value') 和 Ignored()（未使用的参数）
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 使用 CallFunction 调用 aten.expand.default 函数，传入 permute_default_3 和 Ignored()（未使用的参数）
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 使用 CallFunction 调用 aten.clone.default 函数，传入 expand_default_3 和 memory_format=torch.contiguous_format 作为参数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 使用 CallFunction 调用 aten.view.default 函数，传入 clone_default_2 和 Ignored()（未使用的参数），_users=2 表示此结果被多个地方使用
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 使用 CallFunction 调用 aten.bmm.default 函数，传入 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 使用 CallFunction 调用 aten.view.default 函数，传入 bmm_default_1 和 Ignored()（未使用的参数）
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 使用 CallFunction 调用 prims.convert_element_type.default 函数，传入 convert_element_type_default_1 和 Ignored()（未使用的参数），_users=2 表示此结果被多个地方使用
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 使用 CallFunction 调用 aten.neg.default 函数，传入 convert_element_type_default_2 作为参数
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)

# 使用 CallFunction 调用 aten.view.default 函数，传入 KeywordArg('tangents_1') 和 Ignored()（未使用的参数），_users=2 表示此结果被多个地方使用
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 使用 CallFunction 调用 aten.permute.default 函数，传入 view_default_4 和 Ignored()（未使用的参数）
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 使用 CallFunction 调用 aten.bmm.default 函数，传入 view_default_6 和 permute_default_4 作为参数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 使用 CallFunction 调用 aten.view.default 函数，传入 bmm_default_2 和 Ignored()（未使用的参数）
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 使用 CallFunction 调用 prims.convert_element_type.default 函数，传入 gt_Scalar 和 Ignored()（未使用的参数）
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 使用 CallFunction 调用 aten.mul.Tensor 函数，传入 convert_element_type_default_3 和 Ignored()（未使用的参数）
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())

# 使用 CallFunction 调用 aten.mul.Tensor 函数，传入 view_default_7 和 mul_Tensor_2 作为参数
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 使用 CallFunction 调用 prims.convert_element_type.default 函数，传入 mul_Tensor_3 和 Ignored()（未使用的参数）
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())

# 使用 CallFunction 调用 aten.mul.Tensor 函数，传入 convert_element_type_default_4、convert_element_type_default_2 和 _users=2（此结果被多个地方使用）
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)

# 使用 CallFunction 调用 aten.sum.dim_IntList 函数，传入 mul_Tensor_4 和 Ignored()（未使用的参数），True 表示按维度求和
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 使用 CallFunction 调用 prims.fma.default 函数，传入 neg_default、sum_dim_IntList_1 和 mul_Tensor_4 作为参数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 使用 CallFunction 调用 prims.convert_element_type.default 函数，传入 fma_default 和 Ignored()（未使用的参数）
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 使用 CallFunction 调用 aten.div.Tensor 函数，传入 convert_element_type_default_5 和 KeywordArg('inv_scale_factor') 作为参数
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_5, KeywordArg('inv_scale_factor'))

# 使用 CallFunction 调用 aten.view.default 函数，传入 div_Tensor_2 和 Ignored()（未使用的参数），_users=2 表示此结果被多个地方使用
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 使用 CallFunction 调用 aten.permute.default 函数，传入 view_default_1 和 Ignored()（未使用的参数）
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 使用 CallFunction 调用 aten.bmm.default 函数，传入 view_default_8 和 permute_default_5 作为参数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_3 进行视图变换，生成 view_default_9

permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_9 进行维度置换，生成 permute_default_6

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default 进行维度置换，生成 permute_default_7

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用 PyTorch 的 bmm 函数，对 permute_default_7 和 view_default_8 进行批量矩阵乘法，生成 bmm_default_4

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_4 进行视图变换，生成 view_default_10

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_10 进行维度置换，生成 permute_default_8

permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 调用 PyTorch 的 permute 函数，对 permute_default_8 进行维度置换，生成 permute_default_9

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_3 进行维度置换，生成 permute_default_10

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 调用 PyTorch 的 bmm 函数，对 permute_default_10 和 view_default_6 进行批量矩阵乘法，生成 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_5 进行视图变换，生成 view_default_11

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_11 进行维度置换，生成 permute_default_11

_sfdp_pattern_12_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])
# 创建一个 MultiOutputPattern 对象，包含了 view_default_5、permute_default_6、permute_default_9 和 permute_default_11 等作为输出模式的一部分

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的 permute 函数，使用 'query' 作为关键字参数，生成 permute_default

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的 expand 函数，对 permute_default 进行扩展操作，生成 expand_default

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 clone 函数，对 expand_default 进行克隆操作，使用连续内存格式，生成 clone_default

view_default = CallFunction(aten.view.default, clone_default, Ignored())
# 调用 PyTorch 的 view 函数，对 clone_default 进行视图变换，生成 view_default

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 PyTorch 的 permute 函数，使用 'key' 作为关键字参数，生成 permute_default_1

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 PyTorch 的 permute 函数，对 permute_default_1 进行维度置换，生成 permute_default_2

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 PyTorch 的 expand 函数，对 permute_default_2 进行扩展操作，生成 expand_default_1

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 clone 函数，对 expand_default_1 进行克隆操作，使用连续内存格式，生成 clone_default_1

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 调用 PyTorch 的 view 函数，对 clone_default_1 进行视图变换，生成 view_default_1

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的 bmm 函数，对 view_default 和 view_default_1 进行批量矩阵乘法，生成 bmm_default

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default 进行视图变换，生成 view_default_2

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'))
# 调用 PyTorch 的 div 函数，使用 view_default_2 作为输入张量，'inv_scale_factor' 作为关键字参数，生成 div_Tensor

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，对 div_Tensor 进行元素类型转换，生成 convert_element_type_default

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的 amax 函数，对 convert_element_type_default 进行最大值计算，生成 amax_default

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的 sub 函数，对 convert_element_type_default 和 amax_default 进行张量减法，生成 sub_Tensor

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的 exp 函数，对 sub_Tensor 进行指数运算，生成 exp_default

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的 sum 函数，对 exp_default 按指定维度列表进行求和，生成 sum_dim_IntList

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的 div 函数，对 exp_default 和 sum_dim_IntList 进行张量除法，生成 div_Tensor_1

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 调用 PyTorch 的 convert_element_type 函数，对 div_Tensor_1 进行元素类型转换，生成 convert_element_type_default_1

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的 expand 函数，对 convert_element_type_default_1 进行扩展操作，生成 expand_default_2

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 PyTorch 的 view 函数，对 expand_default_2 进行视图变换，生成 view_default_3

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的 permute 函数，使用 'value' 作为关键字参数，生成 permute_default_3
# 使用 aten.expand.default 函数对 permute_default_3 进行调用，返回扩展后的张量
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 使用 aten.clone.default 函数对 expand_default_3 进行调用，返回克隆后的张量，使用默认内存格式
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 使用 aten.view.default 函数对 clone_default_2 进行调用，返回视图张量
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 使用 aten.bmm.default 函数对 view_default_3 和 view_default_4 进行批量矩阵乘法运算
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 使用 aten.view.default 函数对 bmm_default_1 进行调用，返回视图张量，忽略其它参数
_sfdp_pattern_12_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```