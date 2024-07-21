# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_3.py`

```py
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入PyTorch库
import torch
# 导入torch._inductor模块
import torch._inductor

# 从torch.ops中导入相关的操作符
aten = torch.ops.aten
prims = torch.ops.prims

# 从torch._inductor.pattern_matcher模块中导入多个类和函数
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

# 创建CallFunction对象，并指定相关参数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 创建CallFunction对象，并指定相关参数
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)

# 创建CallFunction对象，并指定相关参数
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)

# 创建CallFunction对象，并指定相关参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建CallFunction对象，并指定相关参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建CallFunction对象，并指定相关参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'), _users=2)

# 创建CallFunction对象，并指定相关参数
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)

# 创建CallFunction对象，并指定相关参数
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)

# 创建CallFunction对象，并指定相关参数
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建CallFunction对象，并指定相关参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建CallFunction对象，并指定相关参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建CallFunction对象，并指定相关参数
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)

# 创建CallFunction对象，并指定相关参数
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 创建CallFunction对象，并指定相关参数
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 创建CallFunction对象，并指定相关参数
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)

# 创建CallFunction对象，并指定相关参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建CallFunction对象，并指定相关参数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建CallFunction对象，并指定相关参数
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建CallFunction对象，并指定相关参数
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建CallFunction对象，并指定相关参数
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)

# 创建CallFunction对象，并指定相关参数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建CallFunction对象，并指定相关参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用函数 `aten.mul.Tensor`，执行默认的元素类型转换
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
# 调用函数 `aten.mul.Tensor`，执行默认的视图操作
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 调用函数 `aten.mul.Tensor`，执行张量相乘操作，使用默认的用户数为2
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
# 调用函数 `aten.sum.dim_IntList`，沿指定维度对张量进行求和，忽略部分参数，保持输出张量的尺寸
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 调用函数 `prims.fma.default`，执行默认的 Fused-Multiply-Add 操作，对输入张量进行操作
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用函数 `aten.div.Tensor`，执行张量除法操作，指定逆比例因子
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, KeywordArg('inv_scale_factor'))
# 调用函数 `aten.view.default`，执行默认的视图操作，设置用户数为2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用函数 `aten.permute.default`，执行默认的维度置换操作
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用函数 `aten.bmm.default`，执行默认的批矩阵乘操作
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用函数 `aten.permute.default`，执行默认的维度置换操作
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用函数 `aten.bmm.default`，执行默认的批矩阵乘操作
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用函数 `aten.permute.default`，执行默认的维度置换操作
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用函数 `aten.permute.default`，执行默认的维度置换操作
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用函数 `aten.bmm.default`，执行默认的批矩阵乘操作
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 创建多输出模式 `_sfdp_pattern_3_training`，包含一系列的视图操作和置换操作
_sfdp_pattern_3_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None,
  None
])

# 调用函数 `aten.expand.default`，执行默认的张量扩展操作，指定关键字参数为 `query`
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 调用函数 `aten.permute.default`，执行默认的维度置换操作，指定关键字参数为 `key`
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用函数 `aten.expand.default`，执行默认的张量扩展操作
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 调用函数 `aten.bmm.default`，执行默认的批矩阵乘操作
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用函数 `aten.div.Tensor`，执行张量除法操作，指定关键字参数为 `inv_scale_factor`，用户数为2
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'), _users=2)
# 调用函数 `aten.amax.default`，执行默认的最大值计算操作
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
# 调用函数 `aten.sub.Tensor`，执行张量减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
# 调用函数 `aten.exp.default`，执行默认的指数运算操作，用户数为2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用函数 `aten.sum.dim_IntList`，沿指定维度对张量进行求和，忽略部分参数，保持输出张量的尺寸
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用函数 `aten.div.Tensor`，执行张量除法操作
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用函数 `aten.expand.default`，执行默认的张量扩展操作
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用函数 `aten.expand.default`，执行默认的张量扩展操作，指定关键字参数为 `value`
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用函数 `aten.view.default`，执行默认的视图操作
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 调用函数 `aten.bmm.default`，执行默认的批矩阵乘操作
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 创建一个 CallFunction 对象，用于推断模式 3 的操作
_sfdp_pattern_3_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)

# 创建一个随机张量，使用默认参数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个大于比较操作的张量，使用随机张量和关键字参数 'dropout_p'
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 创建一个张量扩展操作，使用关键字参数 'query'
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 创建一个张量视图操作，使用扩展后的张量
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)

# 创建一个张量置换操作，使用关键字参数 'key'
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个张量扩展操作，使用前一步置换后的张量
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个张量视图操作，使用扩展后的张量
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)

# 创建一个批量矩阵乘法操作，使用两个视图操作后的张量
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个张量视图操作，使用批量矩阵乘法后的张量
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个张量除法操作，使用前一步视图操作后的张量和关键字参数 'inv_scale_factor'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'))

# 创建一个元素类型转换操作，将除法操作后的张量转换为默认类型
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 创建一个张量最大值操作，使用元素类型转换后的张量和关键字参数 'True'
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个张量减法操作，使用元素类型转换后的张量和最大值操作后的张量
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个指数函数操作，使用减法操作后的张量
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个沿指定维度求和的操作，使用指数函数后的张量和关键字参数 'True'
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个张量除法操作，使用指数函数后的张量和上一步求和操作后的张量
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个元素类型转换操作，将除法操作后的张量转换为默认类型
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)

# 创建一个张量乘法操作，使用大于比较操作后的张量和元素类型转换操作后的张量
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)

# 创建一个张量乘法操作，使用前一步乘法操作后的张量和默认参数
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 创建一个张量扩展操作，使用前一步乘法操作后的张量
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())

# 创建一个张量视图操作，使用扩展操作后的张量
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 创建一个张量扩展操作，使用关键字参数 'value'
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 创建一个张量视图操作，使用扩展操作后的张量
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)

# 创建一个批量矩阵乘法操作，使用两个视图操作后的张量
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个张量视图操作，使用批量矩阵乘法后的张量
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个元素类型转换操作，将元素类型转换操作1后的张量转换为默认类型
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 创建一个负数操作，使用元素类型转换操作2后的张量
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)

# 创建一个张量视图操作，使用关键字参数 'tangents_1'
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个张量置换操作，使用视图操作4和默认参数
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个批量矩阵乘法操作，使用视图操作6和置换操作1后的张量
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)

# 创建一个张量视图操作，使用批量矩阵乘法操作后的张量
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个元素类型转换操作，使用大于比较操作后的张量和默认参数
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个张量乘法操作，使用元素类型转换操作3后的张量和默认参数
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_5, KeywordArg('inv_scale_factor'))
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
_sfdp_pattern_3_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None,
  None
])

# 对输入进行扩展操作
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 将结果视图化
view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 对输入进行排列操作
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 对扩展的结果进行再次扩展
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 将结果视图化
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 使用批量矩阵乘法操作
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 将结果视图化
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 对结果进行除法操作
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'))
# 将结果转换为指定类型
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 计算张量的最大值
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 对张量进行减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 对张量进行指数运算
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 按指定维度对张量进行求和
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 对张量进行除法操作
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 将结果转换为指定类型
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 ATen 扩展操作函数，默认模式，对 convert_element_type_default_1 进行扩展操作，并使用 Ignored() 作为第三个参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 ATen 视图操作函数，默认模式，对 expand_default_2 进行视图操作，并使用 Ignored() 作为第三个参数

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用 ATen 扩展操作函数，默认模式，使用关键字参数 'value'，并使用 Ignored() 作为第三个参数

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 调用 ATen 视图操作函数，默认模式，对 expand_default_3 进行视图操作，并使用 Ignored() 作为第三个参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 ATen 批量矩阵乘操作函数，默认模式，对 view_default_3 和 view_default_4 进行批量矩阵乘操作

_sfdp_pattern_3_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 ATen 视图操作函数，默认模式，对 bmm_default_1 进行视图操作，使用 Ignored() 作为第三个参数，_users 参数设为 0
```