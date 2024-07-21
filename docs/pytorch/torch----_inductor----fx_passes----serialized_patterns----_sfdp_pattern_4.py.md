# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_4.py`

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

# 从 torch.ops 中导入 aten 和 prims 对象
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

# 定义 rand_default 变量，调用 torch.ops.aten.rand.default 函数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 定义 gt_Scalar 变量，调用 torch.ops.aten.gt.Scalar 函数
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 定义 expand_default 变量，调用 torch.ops.aten.expand.default 函数
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 定义 view_default 变量，调用 torch.ops.aten.view.default 函数
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 定义 permute_default 变量，调用 torch.ops.aten.permute.default 函数
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 定义 expand_default_1 变量，调用 torch.ops.aten.expand.default 函数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 定义 view_default_1 变量，调用 torch.ops.aten.view.default 函数
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 定义 bmm_default 变量，调用 torch.ops.aten.bmm.default 函数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 定义 view_default_2 变量，调用 torch.ops.aten.view.default 函数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 定义 mul_Tensor 变量，调用 torch.ops.aten.mul.Tensor 函数
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'), _users=2)
# 定义 amax_default 变量，调用 torch.ops.aten.amax.default 函数
amax_default = CallFunction(aten.amax.default, mul_Tensor, Ignored(), True)
# 定义 sub_Tensor 变量，调用 torch.ops.aten.sub.Tensor 函数
sub_Tensor = CallFunction(aten.sub.Tensor, mul_Tensor, amax_default)
# 定义 exp_default 变量，调用 torch.ops.aten.exp.default 函数
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义 sum_dim_IntList 变量，调用 torch.ops.aten.sum.dim_IntList 函数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义 div_Tensor 变量，调用 torch.ops.aten.div.Tensor 函数
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 定义 mul_Tensor_1 变量，调用 torch.ops.aten.mul.Tensor 函数
mul_Tensor_1 = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor)
# 定义 mul_Tensor_2 变量，调用 torch.ops.aten.mul.Tensor 函数
mul_Tensor_2 = CallFunction(aten.mul.Tensor, mul_Tensor_1, Ignored())
# 定义 expand_default_2 变量，调用 torch.ops.aten.expand.default 函数
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_2, Ignored())
# 定义 view_default_3 变量，调用 torch.ops.aten.view.default 函数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 定义 expand_default_3 变量，调用 torch.ops.aten.expand.default 函数
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 定义 view_default_4 变量，调用 torch.ops.aten.view.default 函数
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 定义 bmm_default_1 变量，调用 torch.ops.aten.bmm.default 函数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 定义 view_default_5 变量，调用 torch.ops.aten.view.default 函数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 定义 neg_default 变量，调用 torch.ops.aten.neg.default 函数
neg_default = CallFunction(aten.neg.default, div_Tensor)
# 定义 view_default_6 变量，调用 torch.ops.aten.view.default 函数
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 定义 permute_default_1 变量，调用 torch.ops.aten.permute.default 函数
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 定义 bmm_default_2 变量，调用 torch.ops.aten.bmm.default 函数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 定义 view_default_7 变量，调用 torch.ops.aten.view.default 函数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 定义 convert_element_type_default 变量，调用 torch.ops.prims.convert_element_type.default 函数
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
mul_Tensor_4 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_3)
mul_Tensor_5 = CallFunction(aten.mul.Tensor, mul_Tensor_4, div_Tensor, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_5, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_5)
mul_Tensor_6 = CallFunction(aten.mul.Tensor, fma_default, KeywordArg('scale_factor'))
view_default_8 = CallFunction(aten.view.default, mul_Tensor_6, Ignored(), _users=2)
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

_sfdp_pattern_4_training = MultiOutputPattern([
    view_default_5,   # 第一个输出
    view_default_9,   # 第二个输出
    permute_default_4,   # 第三个输出
    view_default_11,  # 第四个输出
    None,   # 第五个输出
    None    # 第六个输出
])

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored())
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'), _users=2)
amax_default = CallFunction(aten.amax.default, mul_Tensor, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, mul_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_4_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 创建一个函数调用对象，调用 aten.view.default 函数，参数为 bmm_default_1 和一个未指定的值

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 创建一个函数调用对象，调用 aten.rand.default 函数，生成一个随机数张量

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 创建一个函数调用对象，调用 aten.gt.Scalar 函数，对随机数张量应用大于比较操作

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 创建一个函数调用对象，调用 aten.expand.default 函数，对参数 'query' 进行扩展操作

view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 创建一个函数调用对象，调用 aten.view.default 函数，对 expand_default 结果进行视图变换

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个函数调用对象，调用 aten.permute.default 函数，对参数 'key' 进行维度置换操作

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建一个函数调用对象，调用 aten.expand.default 函数，对 permute_default 结果进行扩展操作

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 创建一个函数调用对象，调用 aten.view.default 函数，对 expand_default_1 结果进行视图变换

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 创建一个函数调用对象，调用 aten.bmm.default 函数，对 view_default 和 view_default_1 进行批量矩阵乘法

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 创建一个函数调用对象，调用 aten.view.default 函数，对 bmm_default 结果进行视图变换

mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'))
# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，对 view_default_2 结果进行张量乘法，使用指定的 scale_factor 参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor, Ignored(), _users=2)
# 创建一个函数调用对象，调用 prims.convert_element_type.default 函数，对 mul_Tensor 结果进行元素类型转换

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 创建一个函数调用对象，调用 aten.amax.default 函数，对 convert_element_type_default 结果进行最大值计算

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 创建一个函数调用对象，调用 aten.sub.Tensor 函数，对 convert_element_type_default 和 amax_default 结果进行张量减法

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建一个函数调用对象，调用 aten.exp.default 函数，对 sub_Tensor 结果进行指数运算

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建一个函数调用对象，调用 aten.sum.dim_IntList 函数，对 exp_default 结果进行指定维度的求和

div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 创建一个函数调用对象，调用 aten.div.Tensor 函数，对 exp_default 和 sum_dim_IntList 结果进行张量除法

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 创建一个函数调用对象，调用 prims.convert_element_type.default 函数，对 div_Tensor 结果进行元素类型转换

mul_Tensor_1 = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，对 gt_Scalar 和 convert_element_type_default_1 结果进行张量乘法

mul_Tensor_2 = CallFunction(aten.mul.Tensor, mul_Tensor_1, Ignored())
# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，对 mul_Tensor_1 结果进行张量乘法

expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_2, Ignored())
# 创建一个函数调用对象，调用 aten.expand.default 函数，对 mul_Tensor_2 结果进行扩展操作

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 创建一个函数调用对象，调用 aten.view.default 函数，对 expand_default_2 结果进行视图变换

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 创建一个函数调用对象，调用 aten.expand.default 函数，对参数 'value' 进行扩展操作

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 创建一个函数调用对象，调用 aten.view.default 函数，对 expand_default_3 结果进行视图变换

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 创建一个函数调用对象，调用 aten.bmm.default 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 创建一个函数调用对象，调用 aten.view.default 函数，对 bmm_default_1 结果进行视图变换

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 创建一个函数调用对象，调用 prims.convert_element_type.default 函数，对 convert_element_type_default_1 结果进行元素类型转换

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 创建一个函数调用对象，调用 aten.neg.default 函数，对 convert_element_type_default_2 结果进行取负操作

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 创建一个函数调用对象，调用 aten.view.default 函数，对参数 'tangents_1' 进行视图变换

permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 创建一个函数调用对象，调用 aten.permute.default 函数，对 view_default_4 结果进行维度置换操作

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 创建一个函数调用对象，调用 aten.bmm.default 函数，对 view_default_6 和 permute_default_1 进行批量矩阵乘法

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 创建一个函数调用对象，调用 aten.view.default 函数，对 bmm_default_2 结果进行视图变换

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 创建一个函数调用对象，调用 prims.convert_element_type.default 函数，对 gt_Scalar 结果进行元素类型转换

mul_Tensor_3 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，对 convert_element_type_default_3 结果进行张量乘法
# 调用函数 `aten.mul.Tensor`，对 `view_default_7` 和 `mul_Tensor_3` 进行张量乘法操作
mul_Tensor_4 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_3)

# 调用函数 `prims.convert_element_type.default`，将 `mul_Tensor_4` 转换为默认的数据类型
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_4, Ignored())

# 调用函数 `aten.mul.Tensor`，对 `convert_element_type_default_4` 和 `convert_element_type_default_2` 进行张量乘法操作，_users 参数指定为 2
mul_Tensor_5 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)

# 调用函数 `aten.sum.dim_IntList`，对 `mul_Tensor_5` 指定维度求和
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_5, Ignored(), True)

# 调用函数 `prims.fma.default`，进行 Fused Multiply-Add (FMA) 操作，其中 `neg_default` 是第一个参数，`sum_dim_IntList_1` 是第二个参数，`mul_Tensor_5` 是第三个参数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_5)

# 调用函数 `prims.convert_element_type.default`，将 `fma_default` 转换为默认的数据类型
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 调用函数 `aten.mul.Tensor`，对 `convert_element_type_default_5` 进行张量乘法操作，其中使用了关键字参数 'scale_factor'
mul_Tensor_6 = CallFunction(aten.mul.Tensor, convert_element_type_default_5, KeywordArg('scale_factor'))

# 调用函数 `aten.view.default`，对 `mul_Tensor_6` 进行默认的张量视图操作，_users 参数指定为 2
view_default_8 = CallFunction(aten.view.default, mul_Tensor_6, Ignored(), _users=2)

# 调用函数 `aten.permute.default`，对 `view_default_1` 进行默认的张量置换操作
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 调用函数 `aten.bmm.default`，对 `view_default_8` 和 `permute_default_2` 进行批量矩阵乘法操作
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)

# 调用函数 `aten.view.default`，对 `bmm_default_3` 进行默认的张量视图操作
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 调用函数 `aten.permute.default`，对 `view_default` 进行默认的张量置换操作
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())

# 调用函数 `aten.bmm.default`，对 `permute_default_3` 和 `view_default_8` 进行批量矩阵乘法操作
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)

# 调用函数 `aten.view.default`，对 `bmm_default_4` 进行默认的张量视图操作
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 调用函数 `aten.permute.default`，对 `view_default_10` 进行默认的张量置换操作
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 调用函数 `aten.permute.default`，对 `view_default_3` 进行默认的张量置换操作
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 调用函数 `aten.bmm.default`，对 `permute_default_5` 和 `view_default_6` 进行批量矩阵乘法操作
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)

# 调用函数 `aten.view.default`，对 `bmm_default_5` 进行默认的张量视图操作
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个多输出模式 `MultiOutputPattern`，包含 `view_default_5`、`view_default_9`、`permute_default_4`、`view_default_11` 和两个空值
_sfdp_pattern_4_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None,
  None
])

# 调用函数 `aten.expand.default`，对关键字参数 'query' 进行默认的张量扩展操作
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 调用函数 `aten.view.default`，对 `expand_default` 进行默认的张量视图操作
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 调用函数 `aten.permute.default`，对关键字参数 'key' 进行默认的张量置换操作
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 调用函数 `aten.expand.default`，对 `permute_default` 进行默认的张量扩展操作
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 调用函数 `aten.view.default`，对 `expand_default_1` 进行默认的张量视图操作
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 调用函数 `aten.bmm.default`，对 `view_default` 和 `view_default_1` 进行批量矩阵乘法操作
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用函数 `aten.view.default`，对 `bmm_default` 进行默认的张量视图操作
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用函数 `aten.mul.Tensor`，对 `view_default_2` 进行张量乘法操作，使用关键字参数 'scale_factor'
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'))

# 调用函数 `prims.convert_element_type.default`，将 `mul_Tensor` 转换为默认的数据类型，_users 参数指定为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, mul_Tensor, Ignored(), _users=2)

# 调用函数 `aten.amax.default`，对 `convert_element_type_default` 进行默认的张量最大值操作，True 参数表示是否保持维度
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 调用函数 `aten.sub.Tensor`，对 `convert_element_type_default` 和 `amax_default` 进行张量减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 调用函数 `aten.exp.default`，对 `sub_Tensor` 进行默认的指数函数操作，_users 参数指定为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用函数 `aten.sum.dim_IntList`，对 `exp_default` 指定维度求和，True 参数表示是否保持维度
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用函数 `aten.div.Tensor`，对 `exp_default` 和 `sum_dim_IntList` 进行张量除法操作
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 调用函数 `prims.convert_element_type.default`，将 `div_Tensor` 转换为默认的数据类型
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
# 调用函数 `aten.expand.default`，传入参数 `convert_element_type_default_1`，并忽略返回值
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 调用函数 `aten.view.default`，对 `expand_default_2` 进行视图操作，并忽略返回值
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 调用函数 `aten.expand.default`，传入参数 `KeywordArg('value')`，并忽略返回值
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 调用函数 `aten.view.default`，对 `expand_default_3` 进行视图操作，并忽略返回值
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 调用函数 `aten.bmm.default`，传入 `view_default_3` 和 `view_default_4` 作为参数，并返回结果
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 调用函数 `aten.view.default`，对 `bmm_default_1` 进行视图操作，并忽略返回值，同时设置 `_users` 参数为 0
_sfdp_pattern_4_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```