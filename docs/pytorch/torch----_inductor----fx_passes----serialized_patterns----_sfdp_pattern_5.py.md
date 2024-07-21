# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_5.py`

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

# 设置 torch.ops 中的两个命名空间别名
aten = torch.ops.aten
prims = torch.ops.prims

# 导入模式匹配相关的类和函数
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

# 定义 expand_default 模式匹配表达式
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 定义 view_default 模式匹配表达式
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 定义 permute_default 模式匹配表达式
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 定义 expand_default_1 模式匹配表达式
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 定义 view_default_1 模式匹配表达式
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 定义 bmm_default 模式匹配表达式
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 定义 view_default_2 模式匹配表达式
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 定义 div_Tensor 模式匹配表达式
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 定义 add_Tensor 模式匹配表达式
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
# 定义 amax_default 模式匹配表达式
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 定义 sub_Tensor 模式匹配表达式
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 定义 exp_default 模式匹配表达式
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义 sum_dim_IntList 模式匹配表达式
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义 div_Tensor_1 模式匹配表达式
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 定义 expand_default_2 模式匹配表达式
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 定义 view_default_3 模式匹配表达式
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 定义 expand_default_3 模式匹配表达式
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 定义 view_default_4 模式匹配表达式
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 定义 bmm_default_1 模式匹配表达式
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 定义 view_default_5 模式匹配表达式
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 定义 neg_default 模式匹配表达式
neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 定义 view_default_6 模式匹配表达式
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 定义 permute_default_1 模式匹配表达式
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 定义 bmm_default_2 模式匹配表达式
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 定义 view_default_7 模式匹配表达式
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 定义 mul_Tensor 模式匹配表达式
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_7, div_Tensor_1, _users=2)
# 定义 sum_dim_IntList_1 模式匹配表达式
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
# 定义 fma_default 模式匹配表达式
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
# 定义 div_Tensor_2 模式匹配表达式
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, Ignored())
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 Torch 的默认视图操作函数，对 div_Tensor_2 进行视图操作，并设定用户数为 2
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 Torch 的默认排列操作函数，对 view_default_1 进行排列操作
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用 Torch 的默认矩阵乘法操作函数，对 view_default_8 和 permute_default_2 进行矩阵乘法
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 Torch 的默认视图操作函数，对 bmm_default_3 进行视图操作
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 Torch 的默认排列操作函数，对 view_default 进行排列操作
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 调用 Torch 的默认矩阵乘法操作函数，对 permute_default_3 和 view_default_8 进行矩阵乘法
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 Torch 的默认视图操作函数，对 bmm_default_4 进行视图操作
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 Torch 的默认排列操作函数，对 view_default_10 进行排列操作
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 调用 Torch 的默认排列操作函数，对 view_default_3 进行排列操作
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 调用 Torch 的默认矩阵乘法操作函数，对 permute_default_5 和 view_default_6 进行矩阵乘法
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 调用 Torch 的默认视图操作函数，对 bmm_default_5 进行视图操作
_sfdp_pattern_5_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None
])
# 创建一个多输出模式对象，包含 view_default_5、view_default_9、permute_default_4、view_default_11 和 None

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用 Torch 的默认扩展操作函数，扩展关键字参数 'query'
view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 调用 Torch 的默认视图操作函数，对 expand_default 进行视图操作
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 Torch 的默认排列操作函数，对关键字参数 'key' 进行排列操作
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 Torch 的默认扩展操作函数，扩展 permute_default
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 调用 Torch 的默认视图操作函数，对 expand_default_1 进行视图操作
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 Torch 的默认矩阵乘法操作函数，对 view_default 和 view_default_1 进行矩阵乘法
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 Torch 的默认视图操作函数，对 bmm_default 进行视图操作
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 调用 Torch 的 Tensor 类的默认除法操作函数，对 view_default_2 进行除法操作
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
# 调用 Torch 的 Tensor 类的默认加法操作函数，对 div_Tensor 和关键字参数 'attn_mask' 进行加法操作，用户数为 2
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 调用 Torch 的默认最大值操作函数，对 add_Tensor 进行最大值操作，保持维度不变
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 调用 Torch 的 Tensor 类的默认减法操作函数，对 add_Tensor 和 amax_default 进行减法操作
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 Torch 的默认指数操作函数，对 sub_Tensor 进行指数操作，用户数为 2
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 Torch 的指定维度求和操作函数，对 exp_default 进行维度列表求和操作，保持维度不变
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 Torch 的 Tensor 类的默认除法操作函数，对 exp_default 和 sum_dim_IntList 进行除法操作
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 调用 Torch 的默认扩展操作函数，对 div_Tensor_1 进行扩展操作
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 Torch 的默认视图操作函数，对 expand_default_2 进行视图操作
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用 Torch 的默认扩展操作函数，扩展关键字参数 'value'
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 调用 Torch 的默认视图操作函数，对 expand_default_3 进行视图操作
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 Torch 的默认矩阵乘法操作函数，对 view_default_3 和 view_default_4 进行矩阵乘法
_sfdp_pattern_5_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 Torch 的默认视图操作函数，对 bmm_default_1 进行视图操作，并设定用户数为 0

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用 Torch 的默认扩展操作函数，扩展关键字参数 'query'
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 调用 Torch 的默认视图操作函数，对 expand_default 进行视图操作，并设定用户数为 2
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 Torch 的默认排列操作函数，对关键字参数 'key' 进行排列操作
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 Torch 的默认扩展操作函数，扩展 permute_default
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作，使用 expand_default_1 作为参数，忽略其他参数，用户数为 2

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的默认矩阵乘法操作，使用 view_default 和 view_default_1 作为参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的默认视图操作，使用 bmm_default 作为参数，忽略其他参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 调用 PyTorch 的张量除法操作，使用 view_default_2 作为参数，忽略其他参数

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))
# 调用 PyTorch 的张量加法操作，使用 div_Tensor 和 attn_mask 作为参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)
# 调用默认的元素类型转换操作，使用 add_Tensor 作为参数，忽略其他参数，用户数为 2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的默认最大值操作，使用 convert_element_type_default 作为参数，忽略其他参数，并指定 True 参数

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的张量减法操作，使用 convert_element_type_default 和 amax_default 作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的默认指数操作，使用 sub_Tensor 作为参数，用户数为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的指定维度求和操作，使用 exp_default 作为参数，忽略其他参数，并指定 True 参数

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的张量除法操作，使用 exp_default 和 sum_dim_IntList 作为参数

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 调用默认的元素类型转换操作，使用 div_Tensor_1 作为参数，忽略其他参数，用户数为 2

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的默认扩展操作，使用 convert_element_type_default_1 作为参数，忽略其他参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作，使用 expand_default_2 作为参数，忽略其他参数，用户数为 2

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的默认扩展操作，使用 value 参数作为关键字参数，忽略其他参数

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作，使用 expand_default_3 作为参数，忽略其他参数，用户数为 2

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的默认矩阵乘法操作，使用 view_default_3 和 view_default_4 作为参数

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的默认视图操作，使用 bmm_default_1 作为参数，忽略其他参数

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 调用默认的元素类型转换操作，使用 convert_element_type_default_1 作为参数，忽略其他参数，用户数为 2

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 调用 PyTorch 的默认负数操作，使用 convert_element_type_default_2 作为参数

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作，使用 tangents_1 参数作为关键字参数，忽略其他参数，用户数为 2

permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 PyTorch 的默认排列操作，使用 view_default_4 作为参数，忽略其他参数

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 调用 PyTorch 的默认矩阵乘法操作，使用 view_default_6 和 permute_default_1 作为参数

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 PyTorch 的默认视图操作，使用 bmm_default_2 作为参数，忽略其他参数

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
# 调用默认的元素类型转换操作，使用 view_default_7 作为参数，忽略其他参数

mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)
# 调用 PyTorch 的张量乘法操作，使用 convert_element_type_default_3 和 convert_element_type_default_2 作为参数，用户数为 2

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
# 调用 PyTorch 的指定维度求和操作，使用 mul_Tensor 作为参数，忽略其他参数，并指定 True 参数

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
# 调用默认的 FMA 操作，使用 neg_default、sum_dim_IntList_1 和 mul_Tensor 作为参数

convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用默认的元素类型转换操作，使用 fma_default 作为参数，忽略其他参数

div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_4, Ignored())
# 调用 PyTorch 的张量除法操作，使用 convert_element_type_default_4 作为参数，忽略其他参数

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作，使用 div_Tensor_2 作为参数，忽略其他参数，用户数为 2

permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 PyTorch 的默认排列操作，使用 view_default_1 作为参数，忽略其他参数

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 调用 PyTorch 的默认矩阵乘法操作，使用 view_default_8 和 permute_default_2 作为参数

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 的默认视图操作，使用 bmm_default_3 作为参数，忽略其他参数
# 创建 permute_default_3 变量，调用 aten.permute.default 函数，并传入 view_default 和 Ignored() 参数
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建 bmm_default_4 变量，调用 aten.bmm.default 函数，并传入 permute_default_3 和 view_default_8 参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)

# 创建 view_default_10 变量，调用 aten.view.default 函数，并传入 bmm_default_4 和 Ignored() 参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建 permute_default_4 变量，调用 aten.permute.default 函数，并传入 view_default_10 和 Ignored() 参数
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建 permute_default_5 变量，调用 aten.permute.default 函数，并传入 view_default_3 和 Ignored() 参数
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建 bmm_default_5 变量，调用 aten.bmm.default 函数，并传入 permute_default_5 和 view_default_6 参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)

# 创建 view_default_11 变量，调用 aten.view.default 函数，并传入 bmm_default_5 和 Ignored() 参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建 _sfdp_pattern_5_half_training 变量，调用 MultiOutputPattern 构造函数，并传入包含 view_default_5、view_default_9、permute_default_4、view_default_11 和 None 的列表
_sfdp_pattern_5_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None
])

# 创建 expand_default 变量，调用 aten.expand.default 函数，并传入 KeywordArg('query') 和 Ignored() 参数
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 创建 view_default 变量，调用 aten.view.default 函数，并传入 expand_default 和 Ignored() 参数
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 创建 permute_default 变量，调用 aten.permute.default 函数，并传入 KeywordArg('key') 和 Ignored() 参数
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建 expand_default_1 变量，调用 aten.expand.default 函数，并传入 permute_default 和 Ignored() 参数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建 view_default_1 变量，调用 aten.view.default 函数，并传入 expand_default_1 和 Ignored() 参数
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 创建 bmm_default 变量，调用 aten.bmm.default 函数，并传入 view_default 和 view_default_1 参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建 view_default_2 变量，调用 aten.view.default 函数，并传入 bmm_default 和 Ignored() 参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建 div_Tensor 变量，调用 aten.div.Tensor 函数，并传入 view_default_2 和 Ignored() 参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())

# 创建 add_Tensor 变量，调用 aten.add.Tensor 函数，并传入 div_Tensor 和 KeywordArg('attn_mask') 参数
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))

# 创建 convert_element_type_default 变量，调用 prims.convert_element_type.default 函数，并传入 add_Tensor 和 Ignored() 参数，_users=2
convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)

# 创建 amax_default 变量，调用 aten.amax.default 函数，并传入 convert_element_type_default、Ignored() 和 True 参数
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建 sub_Tensor 变量，调用 aten.sub.Tensor 函数，并传入 convert_element_type_default 和 amax_default 参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建 exp_default 变量，调用 aten.exp.default 函数，并传入 sub_Tensor 和 _users=2 参数
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建 sum_dim_IntList 变量，调用 aten.sum.dim_IntList 函数，并传入 exp_default、Ignored() 和 True 参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建 div_Tensor_1 变量，调用 aten.div.Tensor 函数，并传入 exp_default 和 sum_dim_IntList 参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建 convert_element_type_default_1 变量，调用 prims.convert_element_type.default 函数，并传入 div_Tensor_1 和 Ignored() 参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建 expand_default_2 变量，调用 aten.expand.default 函数，并传入 convert_element_type_default_1 和 Ignored() 参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建 view_default_3 变量，调用 aten.view.default 函数，并传入 expand_default_2 和 Ignored() 参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建 expand_default_3 变量，调用 aten.expand.default 函数，并传入 KeywordArg('value') 和 Ignored() 参数
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 创建 view_default_4 变量，调用 aten.view.default 函数，并传入 expand_default_3 和 Ignored() 参数
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建 bmm_default_1 变量，调用 aten.bmm.default 函数，并传入 view_default_3 和 view_default_4 参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建 _sfdp_pattern_5_half_inference 变量，调用 aten.view.default 函数，并传入 bmm_default_1 和 Ignored() 参数，_users=0
_sfdp_pattern_5_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```