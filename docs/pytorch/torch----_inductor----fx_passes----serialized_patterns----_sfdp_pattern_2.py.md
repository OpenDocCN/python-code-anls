# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_2.py`

```
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入torch模块
import torch
# 导入torch._inductor模块
import torch._inductor

# 导入aten操作符，用于直接操作底层的ATen库函数
aten = torch.ops.aten
# 导入prims操作符，用于直接操作底层的primitives库函数
prims = torch.ops.prims

# 导入模式匹配器相关的类和函数
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

# 定义扩展默认操作的模式
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 定义视图默认操作的模式
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 定义置换默认操作的模式
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 再次定义扩展默认操作的模式
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 再次定义视图默认操作的模式
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 定义批量矩阵乘法默认操作的模式
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 再次定义视图默认操作的模式
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 定义张量乘法操作的模式
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'), _users=2)
# 定义最大值操作的模式
amax_default = CallFunction(aten.amax.default, mul_Tensor, Ignored(), True)
# 定义张量减法操作的模式
sub_Tensor = CallFunction(aten.sub.Tensor, mul_Tensor, amax_default)
# 定义指数操作的模式
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义指定维度求和操作的模式
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义张量除法操作的模式
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 再次定义扩展默认操作的模式
expand_default_2 = CallFunction(aten.expand.default, div_Tensor, Ignored())
# 再次定义视图默认操作的模式
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 再次定义扩展默认操作的模式
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 再次定义视图默认操作的模式
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 定义批量矩阵乘法默认操作的模式
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 再次定义视图默认操作的模式
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 定义张量取负操作的模式
neg_default = CallFunction(aten.neg.default, div_Tensor)
# 再次定义视图默认操作的模式
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 定义置换默认操作的模式
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 定义批量矩阵乘法默认操作的模式
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 再次定义视图默认操作的模式
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 定义张量乘法操作的模式
mul_Tensor_1 = CallFunction(aten.mul.Tensor, view_default_7, div_Tensor, _users=2)
# 定义指定维度求和操作的模式
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_1, Ignored(), True)
# 定义Fused Multiply-Add操作的模式
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_1)
# 定义张量乘法操作的模式
mul_Tensor_2 = CallFunction(aten.mul.Tensor, fma_default, KeywordArg('scale_factor'))
# 再次定义视图默认操作的模式
view_default_8 = CallFunction(aten.view.default, mul_Tensor_2, Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 PyTorch 中 aten.permute.default 函数，对 view_default_1 进行排列操作，返回排列后的结果 permute_default_2

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用 PyTorch 中 aten.bmm.default 函数，对 view_default_8 和 permute_default_2 进行批量矩阵乘法操作，返回结果 bmm_default_3

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 中 aten.view.default 函数，对 bmm_default_3 进行视图操作，返回结果 view_default_9

permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 PyTorch 中 aten.permute.default 函数，对 view_default 进行排列操作，返回结果 permute_default_3

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 调用 PyTorch 中 aten.bmm.default 函数，对 permute_default_3 和 view_default_8 进行批量矩阵乘法操作，返回结果 bmm_default_4

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 PyTorch 中 aten.view.default 函数，对 bmm_default_4 进行视图操作，返回结果 view_default_10

permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 PyTorch 中 aten.permute.default 函数，对 view_default_10 进行排列操作，返回结果 permute_default_4

permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 PyTorch 中 aten.permute.default 函数，对 view_default_3 进行排列操作，返回结果 permute_default_5

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 调用 PyTorch 中 aten.bmm.default 函数，对 permute_default_5 和 view_default_6 进行批量矩阵乘法操作，返回结果 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 PyTorch 中 aten.view.default 函数，对 bmm_default_5 进行视图操作，返回结果 view_default_11

_sfdp_pattern_2_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None
])
# 创建一个多输出模式对象 _sfdp_pattern_2_training，包含 view_default_5、view_default_9、permute_default_4 和 view_default_11，最后一个元素为 None

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 中 aten.expand.default 函数，根据关键字参数 'query' 扩展数据，返回结果 expand_default

view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 调用 PyTorch 中 aten.view.default 函数，对 expand_default 进行视图操作，返回结果 view_default

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 PyTorch 中 aten.permute.default 函数，根据关键字参数 'key' 对数据进行排列操作，返回结果 permute_default

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 中 aten.expand.default 函数，对 permute_default 进行扩展操作，返回结果 expand_default_1

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 调用 PyTorch 中 aten.view.default 函数，对 expand_default_1 进行视图操作，返回结果 view_default_1

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 中 aten.bmm.default 函数，对 view_default 和 view_default_1 进行批量矩阵乘法操作，返回结果 bmm_default
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用默认视图函数，对 bmm_default 进行操作，忽略其他参数

mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'))
# 使用张量乘法函数，对 view_default_2 进行操作，并指定关键字参数 'scale_factor'

convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor, Ignored(), _users=2)
# 调用默认的元素类型转换函数，对 mul_Tensor 进行操作，忽略一个参数，并且此操作有两个用户引用了该函数

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用默认的最大值函数，对 convert_element_type_default 进行操作，忽略一个参数，并且使用了 True 作为第四个参数

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用张量减法函数，对 convert_element_type_default 和 amax_default 进行操作

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用默认的指数函数，对 sub_Tensor 进行操作，并且此操作有两个用户引用了该函数

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用带有整数列表维度参数的求和函数，对 exp_default 进行操作，忽略一个参数，并且使用了 True 作为第四个参数

div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用张量除法函数，对 exp_default 和 sum_dim_IntList 进行操作

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 调用默认的元素类型转换函数，对 div_Tensor 进行操作，忽略一个参数，并且此操作有两个用户引用了该函数

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用默认的扩展函数，对 convert_element_type_default_1 进行操作，忽略一个参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用默认的视图函数，对 expand_default_2 进行操作，并且此操作有两个用户引用了该函数

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用默认的扩展函数，指定关键字参数 'value'，并忽略一个参数

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用默认的视图函数，对 expand_default_3 进行操作，并且此操作有两个用户引用了该函数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用默认的批量矩阵乘法函数，对 view_default_3 和 view_default_4 进行操作

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用默认的视图函数，对 bmm_default_1 进行操作，并忽略一个参数

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 调用默认的元素类型转换函数，对 convert_element_type_default_1 进行操作，忽略一个参数，并且此操作有两个用户引用了该函数

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 调用默认的负数函数，对 convert_element_type_default_2 进行操作

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用默认的视图函数，指定关键字参数 'tangents_1'，并忽略一个参数，并且此操作有两个用户引用了该函数

permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用默认的置换函数，对 view_default_4 进行操作，并忽略一个参数

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 调用默认的批量矩阵乘法函数，对 view_default_6 和 permute_default_1 进行操作

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用默认的视图函数，对 bmm_default_2 进行操作，并忽略一个参数

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
# 调用默认的元素类型转换函数，对 view_default_7 进行操作，并忽略一个参数

mul_Tensor_1 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)
# 调用张量乘法函数，对 convert_element_type_default_3 和 convert_element_type_default_2 进行操作，并且此操作有两个用户引用了该函数

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_1, Ignored(), True)
# 调用带有整数列表维度参数的求和函数，对 mul_Tensor_1 进行操作，忽略一个参数，并且使用了 True 作为第四个参数

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_1)
# 调用默认的 FMA 函数，对 neg_default、sum_dim_IntList_1 和 mul_Tensor_1 进行操作

convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用默认的元素类型转换函数，对 fma_default 进行操作，并忽略一个参数

mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, KeywordArg('scale_factor'))
# 调用张量乘法函数，对 convert_element_type_default_4 进行操作，并指定关键字参数 'scale_factor'

view_default_8 = CallFunction(aten.view.default, mul_Tensor_2, Ignored(), _users=2)
# 调用默认的视图函数，对 mul_Tensor_2 进行操作，并且此操作有两个用户引用了该函数

permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用默认的置换函数，对 view_default_1 进行操作，并忽略一个参数

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用默认的批量矩阵乘法函数，对 view_default_8 和 permute_default_2 进行操作

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用默认的视图函数，对 bmm_default_3 进行操作，并忽略一个参数

permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用默认的置换函数，对 view_default 进行操作，并忽略一个参数

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 调用默认的批量矩阵乘法函数，对 permute_default_3 和 view_default_8 进行操作
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_4 进行张量视图操作，返回结果给 view_default_10

permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_10 进行张量置换操作，返回结果给 permute_default_4

permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_3 进行张量置换操作，返回结果给 permute_default_5

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 调用 PyTorch 的 bmm 函数，对 permute_default_5 和 view_default_6 进行批量矩阵乘法操作，返回结果给 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_5 进行张量视图操作，返回结果给 view_default_11

_sfdp_pattern_2_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None
])
# 创建一个多输出模式对象 _sfdp_pattern_2_half_training，包含 view_default_5、view_default_9、permute_default_4、view_default_11 和 None

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的 expand 函数，扩展 'query' 关键字参数，返回结果给 expand_default

view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 调用 PyTorch 的 view 函数，对 expand_default 进行张量视图操作，返回结果给 view_default

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 PyTorch 的 permute 函数，对 'key' 关键字参数进行张量置换操作，返回结果给 permute_default

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的 expand 函数，对 permute_default 进行扩展操作，返回结果给 expand_default_1

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 调用 PyTorch 的 view 函数，对 expand_default_1 进行张量视图操作，返回结果给 view_default_1

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的 bmm 函数，对 view_default 和 view_default_1 进行批量矩阵乘法操作，返回结果给 bmm_default

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default 进行张量视图操作，返回结果给 view_default_2

mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'))
# 调用 PyTorch 的 mul 函数，对 view_default_2 进行张量乘法操作，使用 'scale_factor' 关键字参数，返回结果给 mul_Tensor

convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，对 mul_Tensor 进行元素类型转换操作，返回结果给 convert_element_type_default

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的 amax 函数，对 convert_element_type_default 进行最大值计算操作，使用忽略参数，返回结果给 amax_default

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的 sub 函数，对 convert_element_type_default 和 amax_default 进行张量减法操作，返回结果给 sub_Tensor

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的 exp 函数，对 sub_Tensor 进行指数函数操作，返回结果给 exp_default

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的 sum 函数，对 exp_default 进行按维度求和操作，使用忽略参数，返回结果给 sum_dim_IntList

div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的 div 函数，对 exp_default 和 sum_dim_IntList 进行张量除法操作，返回结果给 div_Tensor

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
# 调用 PyTorch 的 convert_element_type 函数，对 div_Tensor 进行元素类型转换操作，返回结果给 convert_element_type_default_1

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的 expand 函数，对 convert_element_type_default_1 进行扩展操作，返回结果给 expand_default_2

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 PyTorch 的 view 函数，对 expand_default_2 进行张量视图操作，返回结果给 view_default_3

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的 expand 函数，扩展 'value' 关键字参数，返回结果给 expand_default_3

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 调用 PyTorch 的 view 函数，对 expand_default_3 进行张量视图操作，返回结果给 view_default_4

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的 bmm 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法操作，返回结果给 bmm_default_1

_sfdp_pattern_2_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 PyTorch 的 view 函数，对 bmm_default_1 进行张量视图操作，返回结果给 _sfdp_pattern_2_half_inference，用户数量为 0
```