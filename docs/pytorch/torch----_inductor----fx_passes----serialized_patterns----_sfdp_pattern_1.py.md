# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_1.py`

```
# mypy: ignore-errors
# 忽略类型检查错误，针对 mypy 工具

# noqa: F401, E501
# noqa 表示忽略指定的错误或警告类型
# F401 表示忽略未使用的导入，E501 表示忽略行超过规定长度的警告

# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py
# 自动生成的文件，请勿手动修改
# 若要重新生成，请执行以下命令：
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
# 导入 torch 模块及其私有模块 torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims
# 设置 aten 和 prims 变量为 torch 操作的 aten 和 prims 命名空间

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
# 从 torch._inductor.pattern_matcher 导入多个类和函数用于模式匹配和表达式构建

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 使用 torch.ops.aten.expand.default 函数创建一个 CallFunction 对象，并传递 KeywordArg 和 Ignored 参数

view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 expand_default、Ignored 和 _users 参数

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 使用 torch.ops.aten.permute.default 函数创建一个 CallFunction 对象，并传递 KeywordArg 和 Ignored 参数

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 使用 torch.ops.aten.expand.default 函数创建一个 CallFunction 对象，并传递 permute_default 和 Ignored 参数

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 expand_default_1、Ignored 和 _users 参数

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 使用 torch.ops.aten.bmm.default 函数创建一个 CallFunction 对象，并传递 view_default 和 view_default_1 参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 bmm_default 和 Ignored 参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'), _users=2)
# 使用 torch.ops.aten.div.Tensor 函数创建一个 CallFunction 对象，并传递 view_default_2、KeywordArg 和 _users 参数

amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
# 使用 torch.ops.aten.amax.default 函数创建一个 CallFunction 对象，并传递 div_Tensor、Ignored 和 True 参数

sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
# 使用 torch.ops.aten.sub.Tensor 函数创建一个 CallFunction 对象，并传递 div_Tensor 和 amax_default 参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 使用 torch.ops.aten.exp.default 函数创建一个 CallFunction 对象，并传递 sub_Tensor 和 _users 参数

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 使用 torch.ops.aten.sum.dim_IntList 函数创建一个 CallFunction 对象，并传递 exp_default、Ignored 和 True 参数

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 使用 torch.ops.aten.div.Tensor 函数创建一个 CallFunction 对象，并传递 exp_default、sum_dim_IntList 和 _users 参数

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 使用 torch.ops.aten.expand.default 函数创建一个 CallFunction 对象，并传递 div_Tensor_1 和 Ignored 参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 expand_default_2、Ignored 和 _users 参数

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 使用 torch.ops.aten.expand.default 函数创建一个 CallFunction 对象，并传递 KeywordArg 和 Ignored 参数

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 expand_default_3、Ignored 和 _users 参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 使用 torch.ops.aten.bmm.default 函数创建一个 CallFunction 对象，并传递 view_default_3 和 view_default_4 参数

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 bmm_default_1 和 Ignored 参数

neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 使用 torch.ops.aten.neg.default 函数创建一个 CallFunction 对象，并传递 div_Tensor_1 参数

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 KeywordArg、Ignored 和 _users 参数

permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 使用 torch.ops.aten.permute.default 函数创建一个 CallFunction 对象，并传递 view_default_4 和 Ignored 参数

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 使用 torch.ops.aten.bmm.default 函数创建一个 CallFunction 对象，并传递 view_default_6 和 permute_default_1 参数

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 bmm_default_2 和 Ignored 参数

mul_Tensor = CallFunction(aten.mul.Tensor, view_default_7, div_Tensor_1, _users=2)
# 使用 torch.ops.aten.mul.Tensor 函数创建一个 CallFunction 对象，并传递 view_default_7、div_Tensor_1 和 _users 参数

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
# 使用 torch.ops.aten.sum.dim_IntList 函数创建一个 CallFunction 对象，并传递 mul_Tensor、Ignored 和 True 参数

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
# 使用 torch.ops.prims.fma.default 函数创建一个 CallFunction 对象，并传递 neg_default、sum_dim_IntList_1 和 mul_Tensor 参数

div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, KeywordArg('inv_scale'))
# 使用 torch.ops.aten.div.Tensor 函数创建一个 CallFunction 对象，并传递 fma_default 和 KeywordArg 参数

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 使用 torch.ops.aten.view.default 函数创建一个 CallFunction 对象，并传递 div_Tensor_2、Ignored 和 _users 参数
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 ATen 库中的 permute 函数，对 view_default_1 进行默认的排列操作，结果赋给 permute_default_2

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用 ATen 库中的 bmm 函数，对 view_default_8 和 permute_default_2 进行默认的批量矩阵乘法操作，结果赋给 bmm_default_3

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 ATen 库中的 view 函数，对 bmm_default_3 进行默认的视图变换操作，结果赋给 view_default_9

permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 ATen 库中的 permute 函数，对 view_default 进行默认的排列操作，结果赋给 permute_default_3

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 调用 ATen 库中的 bmm 函数，对 permute_default_3 和 view_default_8 进行默认的批量矩阵乘法操作，结果赋给 bmm_default_4

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 ATen 库中的 view 函数，对 bmm_default_4 进行默认的视图变换操作，结果赋给 view_default_10

permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 ATen 库中的 permute 函数，对 view_default_10 进行默认的排列操作，结果赋给 permute_default_4

permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 ATen 库中的 permute 函数，对 view_default_3 进行默认的排列操作，结果赋给 permute_default_5

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 调用 ATen 库中的 bmm 函数，对 permute_default_5 和 view_default_6 进行默认的批量矩阵乘法操作，结果赋给 bmm_default_5

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 ATen 库中的 view 函数，对 bmm_default_5 进行默认的视图变换操作，结果赋给 view_default_11

_sfdp_pattern_1_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None
])
# 创建一个 MultiOutputPattern 对象，包含 view_default_5, view_default_9, permute_default_4, view_default_11 和 None

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用 ATen 库中的 expand 函数，对 'query' 进行默认的扩展操作，结果赋给 expand_default

view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 调用 ATen 库中的 view 函数，对 expand_default 进行默认的视图变换操作，结果赋给 view_default

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 ATen 库中的 permute 函数，对 'key' 进行默认的排列操作，结果赋给 permute_default

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 ATen 库中的 expand 函数，对 permute_default 进行默认的扩展操作，结果赋给 expand_default_1

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 调用 ATen 库中的 view 函数，对 expand_default_1 进行默认的视图变换操作，结果赋给 view_default_1

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 ATen 库中的 bmm 函数，对 view_default 和 view_default_1 进行默认的批量矩阵乘法操作，结果赋给 bmm_default
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的默认视图函数 aten.view.default，对 bmm_default 进行视图操作，忽略其他参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 PyTorch 的张量除法函数 aten.div.Tensor，用 view_default_2 除以 'inv_scale'

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
# 调用默认类型转换函数 prims.convert_element_type.default，将 div_Tensor 进行类型转换，忽略其他参数，有两个用户

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的最大值函数 aten.amax.default，计算 convert_element_type_default 的最大值，忽略其他参数，计算元素级最大值

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的张量减法函数 aten.sub.Tensor，用 amax_default 减去 convert_element_type_default

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的指数函数 aten.exp.default，对 sub_Tensor 进行指数运算，有两个用户

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的按指定维度求和函数 aten.sum.dim_IntList，对 exp_default 沿着指定维度进行求和，忽略其他参数，计算元素级求和

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的张量除法函数 aten.div.Tensor，用 sum_dim_IntList 除以 exp_default

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 调用默认类型转换函数 prims.convert_element_type.default，将 div_Tensor_1 进行类型转换，忽略其他参数，有两个用户

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的默认展开函数 aten.expand.default，对 convert_element_type_default_1 进行展开操作，忽略其他参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数 aten.view.default，对 expand_default_2 进行视图操作，忽略其他参数，有两个用户

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的默认展开函数 aten.expand.default，使用 'value' 进行展开操作，忽略其他参数

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数 aten.view.default，对 expand_default_3 进行视图操作，忽略其他参数，有两个用户

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的批量矩阵乘法函数 aten.bmm.default，对 view_default_3 和 view_default_4 进行批量矩阵乘法运算

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的默认视图函数 aten.view.default，对 bmm_default_1 进行视图操作，忽略其他参数

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 调用默认类型转换函数 prims.convert_element_type.default，将 convert_element_type_default_1 进行类型转换，忽略其他参数，有两个用户

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 调用 PyTorch 的取负数函数 aten.neg.default，对 convert_element_type_default_2 进行取负运算

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数 aten.view.default，使用 'tangents_1' 进行视图操作，忽略其他参数，有两个用户

permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 PyTorch 的默认排列函数 aten.permute.default，对 view_default_4 进行排列操作，忽略其他参数

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 调用 PyTorch 的批量矩阵乘法函数 aten.bmm.default，对 view_default_6 和 permute_default_1 进行批量矩阵乘法运算

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 PyTorch 的默认视图函数 aten.view.default，对 bmm_default_2 进行视图操作，忽略其他参数

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
# 调用默认类型转换函数 prims.convert_element_type.default，将 view_default_7 进行类型转换，忽略其他参数

mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)
# 调用 PyTorch 的张量乘法函数 aten.mul.Tensor，用 convert_element_type_default_3 和 convert_element_type_default_2 进行张量乘法运算，有两个用户

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
# 调用 PyTorch 的按指定维度求和函数 aten.sum.dim_IntList，对 mul_Tensor 沿着指定维度进行求和，忽略其他参数，计算元素级求和

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
# 调用默认 FMA 函数 prims.fma.default，对 neg_default、sum_dim_IntList_1 和 mul_Tensor 进行 FMA 运算

convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用默认类型转换函数 prims.convert_element_type.default，将 fma_default 进行类型转换，忽略其他参数

div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_4, KeywordArg('inv_scale'))
# 调用 PyTorch 的张量除法函数 aten.div.Tensor，用 convert_element_type_default_4 除以 'inv_scale'

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图函数 aten.view.default，对 div_Tensor_2 进行视图操作，忽略其他参数，有两个用户

permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 PyTorch 的默认排列函数 aten.permute.default，对 view_default_1 进行排列操作，忽略其他参数

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用 PyTorch 的批量矩阵乘法函数 aten.bmm.default，对 view_default_8 和 permute_default_2 进行批量矩阵乘法运算

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 的默认视图函数 aten.view.default，对 bmm_default_3 进行视图操作，忽略其他参数

permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 PyTorch 的默认排列函数 aten.permute
# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 bmm_default_4 和 Ignored()
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，参数为 view_default_10 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，参数为 view_default_3 和 Ignored()
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，参数为 permute_default_5 和 view_default_6
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 bmm_default_5 和 Ignored()
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个 MultiOutputPattern 对象，包含 view_default_5、view_default_9、permute_default_4、view_default_11 和 None
_sfdp_pattern_1_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None
])

# 创建一个调用函数对象，调用 aten.expand.default 函数，关键字参数为 'query' 和 Ignored()
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 expand_default 和 Ignored()
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，关键字参数为 'key' 和 Ignored()
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，参数为 permute_default 和 Ignored()
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 expand_default_1 和 Ignored()
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，参数为 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 bmm_default 和 Ignored()
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，参数为 view_default_2 和 关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，参数为 div_Tensor 和 Ignored()
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.amax.default 函数，参数为 convert_element_type_default 和 Ignored()，关键字参数为 True
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.sub.Tensor 函数，参数为 convert_element_type_default 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个调用函数对象，调用 aten.exp.default 函数，参数为 sub_Tensor 和 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，参数为 exp_default 和 Ignored()，关键字参数为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，参数为 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，参数为 div_Tensor_1 和 Ignored()
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，参数为 convert_element_type_default_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 expand_default_2 和 Ignored()
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，关键字参数为 'value' 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 expand_default_3 和 Ignored()
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，参数为 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 bmm_default_1 和 Ignored()，_users 参数为 0
_sfdp_pattern_1_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```