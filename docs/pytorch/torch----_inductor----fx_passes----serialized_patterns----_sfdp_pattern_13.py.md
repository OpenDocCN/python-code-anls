# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_13.py`

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

# 设置 torch.ops.aten 和 torch.ops.prims 的简写变量
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块导入各种模式匹配类
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

# 定义 rand_default 模式，调用 torch.ops.aten.rand.default 函数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 定义 gt_Scalar 模式，调用 torch.ops.aten.gt.Scalar 函数，传入 rand_default 作为参数
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 定义 permute_default 模式，调用 torch.ops.aten.permute.default 函数，使用 'key' 作为关键字参数
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)

# 定义 bmm_default 模式，调用 torch.ops.aten.bmm.default 函数，使用 permute_default 作为关键字参数
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)

# 定义 amax_default 模式，调用 torch.ops.aten.amax.default 函数，使用 bmm_default 作为参数
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)

# 定义 sub_Tensor 模式，调用 torch.ops.aten.sub.Tensor 函数，使用 bmm_default 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)

# 定义 exp_default 模式，调用 torch.ops.aten.exp.default 函数，使用 sub_Tensor 作为参数
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 定义 sum_dim_IntList 模式，调用 torch.ops.aten.sum.dim_IntList 函数，使用 exp_default 作为参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 定义 div_Tensor 模式，调用 torch.ops.aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为参数
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 定义 mul_Tensor 模式，调用 torch.ops.aten.mul.Tensor 函数，使用 gt_Scalar 和 div_Tensor 作为参数
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor)

# 定义 mul_Tensor_1 模式，调用 torch.ops.aten.mul.Tensor 函数，使用 mul_Tensor 作为参数
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored(), _users=2)

# 定义 bmm_default_1 模式，调用 torch.ops.aten.bmm.default 函数，使用 mul_Tensor_1 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, mul_Tensor_1, KeywordArg('value'))

# 定义 neg_default 模式，调用 torch.ops.aten.neg.default 函数，使用 div_Tensor 作为参数
neg_default = CallFunction(aten.neg.default, div_Tensor)

# 定义 permute_default_1 模式，调用 torch.ops.aten.permute.default 函数，使用 'value' 作为关键字参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 定义 bmm_default_2 模式，调用 torch.ops.aten.bmm.default 函数，使用 'tangents_1' 作为关键字参数，
# permute_default_1 和 bmm_default_2 作为参数
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_1)

# 定义 convert_element_type_default 模式，调用 torch.ops.prims.convert_element_type.default 函数，
# 使用 gt_Scalar 和 Ignored() 作为参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 定义 mul_Tensor_2 模式，调用 torch.ops.aten.mul.Tensor 函数，使用 convert_element_type_default 和 Ignored() 作为参数
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())

# 定义 mul_Tensor_3 模式，调用 torch.ops.aten.mul.Tensor 函数，使用 bmm_default_2 和 mul_Tensor_2 作为参数
mul_Tensor_3 = CallFunction(aten.mul.Tensor, bmm_default_2, mul_Tensor_2)

# 定义 mul_Tensor_4 模式，调用 torch.ops.aten.mul.Tensor 函数，使用 mul_Tensor_3 和 div_Tensor 作为参数
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor, _users=2)

# 定义 sum_dim_IntList_1 模式，调用 torch.ops.aten.sum.dim_IntList 函数，使用 mul_Tensor_4 和 Ignored() 作为参数
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 定义 fma_default 模式，调用 torch.ops.prims.fma.default 函数，使用 neg_default、sum_dim_IntList_1、
# mul_Tensor_4 作为参数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4, _users=2)

# 定义 permute_default_2 模式，调用 torch.ops.aten.permute.default 函数，使用 permute_default 和 Ignored() 作为参数
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())

# 定义 bmm_default_3 模式，调用 torch.ops.aten.bmm.default 函数，使用 fma_default 和 permute_default_2 作为参数
bmm_default_3 = CallFunction(aten.bmm.default, fma_default, permute_default_2)

# 定义 permute_default_3 模式，调用 torch.ops.aten.permute.default 函数，使用 'query' 和 Ignored() 作为参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 定义 bmm_default_4 模式，调用 torch.ops.aten.bmm.default 函数，使用 permute_default_3 和 fma_default 作为参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, fma_default)

# 定义 permute_default_4 模式，调用 torch.ops.aten.permute.default 函数，使用 bmm_default_4 和 Ignored() 作为参数
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())

# 定义 permute_default_5 模式，调用 torch.ops.aten.permute.default 函数，使用 mul_Tensor_1 和 Ignored() 作为参数
permute_default_5 = CallFunction(aten.permute.default, mul_Tensor_1, Ignored())

# 定义 bmm_default_5 模式，调用 torch.ops.aten.bmm.default 函数，使用 permute_default_5 和 'tangents_1' 作为关键字参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, KeywordArg('tangents_1'))
# 定义多输出模式，包含多个函数或操作
_sfdp_pattern_13_training = MultiOutputPattern([
  bmm_default_1,  # 第一个默认矩阵乘法操作
  bmm_default_3,  # 第三个默认矩阵乘法操作
  permute_default_4,  # 第四个默认排列操作
  bmm_default_5,  # 第五个默认矩阵乘法操作
  None  # 空值作为最后一个操作
])

# 定义默认的排列操作
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 定义默认的矩阵乘法操作
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)

# 定义默认的最大值操作
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)

# 定义默认的张量减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)

# 定义默认的指数操作
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 定义在指定维度上求和的操作
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 定义默认的张量除法操作
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 定义默认的类型转换操作
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)

# 定义第二个默认的矩阵乘法操作
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)

# 定义第二个默认的最大值操作
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 定义第二个默认的张量减法操作
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 定义第二个默认的指数操作
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 定义在指定维度上求和的操作
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 定义第二个默认的张量除法操作
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 定义第三个默认的类型转换操作
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 定义第一个张量乘法操作，其中包含默认的大于标量操作
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)

# 定义第一个张量乘法操作
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored(), _users=2)

# 定义第一个默认的矩阵乘法操作
bmm_default_1 = CallFunction(aten.bmm.default, mul_Tensor_1, KeywordArg('value'))

# 定义第三个默认的类型转换操作
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 定义默认的取负操作
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)

# 定义第二个默认的排列操作
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 定义第二个默认的矩阵乘法操作
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_1)

# 定义第三个默认的类型转换操作
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 定义第一个张量乘法操作
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())

# 定义第二个张量乘法操作
mul_Tensor_3 = CallFunction(aten.mul.Tensor, bmm_default_2, mul_Tensor_2)

# 定义第四个默认的类型转换操作
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())

# 定义第二个张量乘法操作
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
# 调用名为 `sum.dim_IntList` 的函数，对 `mul_Tensor_4` 在维度列表上求和，返回结果
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 调用名为 `prims.fma.default` 的函数，执行 Fused Multiply-Add 操作，返回结果
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 调用名为 `prims.convert_element_type.default` 的函数，转换数据类型，返回结果
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored(), _users=2)

# 调用名为 `aten.permute.default` 的函数，对 `permute_default` 进行默认排列操作，返回结果
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())

# 调用名为 `aten.bmm.default` 的函数，执行 Batch Matrix Multiplication 操作，返回结果
bmm_default_3 = CallFunction(aten.bmm.default, convert_element_type_default_5, permute_default_2)

# 调用名为 `aten.permute.default` 的函数，对 `permute_default` 进行默认排列操作，返回结果
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 调用名为 `aten.bmm.default` 的函数，执行 Batch Matrix Multiplication 操作，返回结果
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, convert_element_type_default_5)

# 调用名为 `aten.permute.default` 的函数，对 `bmm_default_4` 进行默认排列操作，返回结果
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())

# 调用名为 `aten.permute.default` 的函数，对 `mul_Tensor_1` 进行默认排列操作，返回结果
permute_default_5 = CallFunction(aten.permute.default, mul_Tensor_1, Ignored())

# 调用名为 `aten.bmm.default` 的函数，执行 Batch Matrix Multiplication 操作，返回结果
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, KeywordArg('tangents_1'))

# 定义 `_sfdp_pattern_13_half_training` 为一个多输出模式，包含多个操作结果和一个空值
_sfdp_pattern_13_half_training = MultiOutputPattern([bmm_default_1,
  bmm_default_3,
  permute_default_4,
  bmm_default_5,
  None
])

# 调用名为 `aten.permute.default` 的函数，对 `permute_default` 进行默认排列操作，返回结果
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 调用名为 `aten.bmm.default` 的函数，执行 Batch Matrix Multiplication 操作，返回结果
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)

# 调用名为 `prims.convert_element_type.default` 的函数，转换数据类型，返回结果
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)

# 调用名为 `aten.amax.default` 的函数，沿着指定维度对 `convert_element_type_default` 进行最大值计算，返回结果
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 调用名为 `aten.sub.Tensor` 的函数，对 `convert_element_type_default` 和 `amax_default` 执行张量减法，返回结果
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 调用名为 `aten.exp.default` 的函数，对 `sub_Tensor` 执行指数运算，返回结果
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用名为 `aten.sum.dim_IntList` 的函数，对 `exp_default` 在维度列表上求和，返回结果
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用名为 `aten.div.Tensor` 的函数，对 `exp_default` 和 `sum_dim_IntList` 执行张量除法，返回结果
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 调用名为 `prims.convert_element_type.default` 的函数，转换数据类型，返回结果
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())

# 调用名为 `aten.bmm.default` 的函数，执行 Batch Matrix Multiplication 操作，返回结果
_sfdp_pattern_13_half_inference = CallFunction(aten.bmm.default, convert_element_type_default_1, KeywordArg('value'), _users=0)
```