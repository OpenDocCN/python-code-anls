# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_19.py`

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

# 从 torch.ops 中导入对应的操作符
aten = torch.ops.aten
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块导入多个模式匹配相关类
from torch._inductor.pattern_matcher import (
   Arg,  # 单个参数
   CallFunction,  # 调用函数的模式匹配
   CallFunctionVarArgs,  # 调用函数带可变参数的模式匹配
   CallMethod,  # 调用对象方法的模式匹配
   CallMethodVarArgs,  # 调用对象方法带可变参数的模式匹配
   CallModule,  # 调用模块的模式匹配
   CallModuleVarArgs,  # 调用模块带可变参数的模式匹配
   ExclusiveKeywordArg,  # 独占关键字参数的模式匹配
   Ignored,  # 表示忽略的模式匹配
   KeywordArg,  # 关键字参数的模式匹配
   ListOf,  # 列表类型的模式匹配
   MultiOutputPattern,  # 多输出模式匹配
   PatternExpr,  # 模式表达式
   RepeatedExpr,  # 重复表达式的模式匹配
   _TargetArgsExpr,  # 目标参数表达式的模式匹配
   _TargetExpr,  # 目标表达式的模式匹配
   _TargetExprVarArgs,  # 目标表达式带可变参数的模式匹配
)

# 定义模式匹配的具体调用和参数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False, _users=2)
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)
add_Tensor = CallFunction(aten.add.Tensor, where_self, KeywordArg('attn_mask'), _users=2)
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 tangents_1 参数，_users 参数为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_4 参数，忽略其他参数
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default_6 和 permute_default_1 参数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_2 参数，忽略其他参数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，使用 gt_Scalar 参数，忽略其他参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，使用 convert_element_type_default 参数，忽略其他参数
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，使用 view_default_7 和 mul_Tensor_2 参数
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，使用 mul_Tensor_3、div_Tensor_1 参数，_users 参数为 2
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，使用 mul_Tensor_4 参数，忽略其他参数，keepdim 参数为 True
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 创建一个 CallFunction 对象，调用 prims.fma.default 函数，使用 neg_default、sum_dim_IntList_1 和 mul_Tensor_4 参数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 创建一个 CallFunction 对象，调用 aten.scalar_tensor.default 函数，忽略第一个参数，dtype 参数为 Ignored()，layout 参数为 torch.strided，device 参数为 Ignored()
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())

# 创建一个 CallFunction 对象，调用 aten.where.self 函数，使用 'causal_mask' 关键字参数，fma_default 和 scalar_tensor_default 参数
where_self_1 = CallFunction(aten.where.self, KeywordArg('causal_mask'), fma_default, scalar_tensor_default)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，使用 where_self_1 和 full_default 参数
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, full_default)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 div_Tensor_2 参数，忽略其他参数，_users 参数为 2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_1 参数，忽略其他参数
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default_8 和 permute_default_2 参数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_3 参数，忽略其他参数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default 参数，忽略其他参数
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 permute_default_3 和 view_default_8 参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_4 参数，忽略其他参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_10 参数，忽略其他参数
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 view_default_3 参数，忽略其他参数
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 permute_default_5 和 view_default_6 参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default_5 参数，忽略其他参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个 MultiOutputPattern 对象，包含 view_default_5、view_default_9、permute_default_4、view_default_11 等对象
_sfdp_pattern_19_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None,
  None,
  None
])

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 'query' 关键字参数，忽略其他参数
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 expand_default 参数，忽略其他参数
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，使用 'key' 关键字参数，忽略其他参数
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，使用 permute_default 参数，忽略其他参数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 expand_default_1 参数，忽略其他参数
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，使用 view_default 和 view_default_1 参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，使用 bmm_default 参数，忽略其他参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.full.default 函数，使用空列表作为第一个参数，忽略其他参数
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，使用 view_default_2 和 full_default 参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 PyTorch 中的 full 函数，创建一个指定大小的张量，填充为默认值

where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)
# 调用 PyTorch 中的 where 函数，根据条件（causal_mask）选择性地返回张量的元素或者填充值，结果存储在 where_self 中

add_Tensor = CallFunction(aten.add.Tensor, where_self, KeywordArg('attn_mask'), _users=2)
# 调用 PyTorch 中的 add 函数，对 where_self 和 attn_mask 进行张量相加操作，结果存储在 add_Tensor 中

amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 调用 PyTorch 中的 amax 函数，计算 add_Tensor 中的最大值，忽略指定参数，返回结果作为单个张量

sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 调用 PyTorch 中的 sub 函数，对 add_Tensor 和 amax_default 进行张量相减操作，结果存储在 sub_Tensor 中

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 中的 exp 函数，计算 sub_Tensor 中每个元素的指数，结果存储在 exp_default 中

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 中的 sum 函数，沿着指定维度对 exp_default 进行求和操作，忽略指定参数，返回结果作为单个张量

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 调用 PyTorch 中的 div 函数，对 exp_default 中的张量元素进行除法操作，结果存储在 div_Tensor_1 中

convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 调用 PyTorch 中的 convert_element_type 函数（假设为 TorchScript 中的函数），将 div_Tensor_1 中的张量元素转换为指定的数据类型
# 使用 aten.mul.Tensor 函数计算两个张量的乘积
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default)
# 使用 aten.mul.Tensor 函数计算两个张量 mul_Tensor 和 Ignored() 的乘积
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 使用 aten.expand.default 函数对 mul_Tensor_1 进行扩展操作
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 使用 aten.view.default 函数对 expand_default_2 进行视图变换
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 使用 aten.expand.default 函数对 KeywordArg('value') 进行扩展操作
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 使用 aten.view.default 函数对 expand_default_3 进行视图变换
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 使用 aten.bmm.default 函数对 view_default_3 和 view_default_4 进行批量矩阵乘法
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 使用 aten.view.default 函数对 bmm_default_1 进行视图变换
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 使用 aten.neg.default 函数计算 div_Tensor_1 的负值
neg_default = CallFunction(aten.neg.default, div_Tensor_1)
# 使用 aten.view.default 函数对 KeywordArg('tangents_1') 进行视图变换
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 使用 aten.permute.default 函数对 view_default_4 进行维度重排
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 使用 aten.bmm.default 函数对 view_default_6 和 permute_default_1 进行批量矩阵乘法
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 使用 aten.view.default 函数对 bmm_default_2 进行视图变换
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 使用 prims.convert_element_type.default 函数进行数据类型转换
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 使用 aten.mul.Tensor 函数计算 convert_element_type_default_1 和 Ignored() 的乘积
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_1, Ignored())
# 使用 aten.mul.Tensor 函数计算 view_default_7 和 mul_Tensor_2 的乘积
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 使用 prims.convert_element_type.default 函数进行数据类型转换
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
# 使用 aten.mul.Tensor 函数计算 convert_element_type_default_2、div_Tensor_1 和 Ignored() 的乘积
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_2, div_Tensor_1, _users=2)
# 使用 aten.sum.dim_IntList 函数对 mul_Tensor_4 在指定维度上进行求和
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 使用 prims.fma.default 函数进行 Fused Multiply-Add 操作
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 使用 prims.convert_element_type.default 函数进行数据类型转换
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 使用 aten.scalar_tensor.default 函数创建一个标量张量
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())
# 使用 aten.where.self 函数根据条件创建一个新的张量
where_self_1 = CallFunction(aten.where.self, KeywordArg('causal_mask'), convert_element_type_default_3, scalar_tensor_default)
# 使用 aten.div.Tensor 函数计算 where_self_1 和 full_default 的除法
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, full_default)
# 使用 aten.view.default 函数对 div_Tensor_2 进行视图变换
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 使用 aten.permute.default 函数对 view_default_1 进行维度重排
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 使用 aten.bmm.default 函数对 view_default_8 和 permute_default_2 进行批量矩阵乘法
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 使用 aten.view.default 函数对 bmm_default_3 进行视图变换
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 使用 aten.permute.default 函数对 view_default 进行维度重排
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 使用 aten.bmm.default 函数对 permute_default_3 和 view_default_8 进行批量矩阵乘法
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 使用 aten.view.default 函数对 bmm_default_4 进行视图变换
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 使用 aten.permute.default 函数对 view_default_10 进行维度重排
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 使用 aten.permute.default 函数对 view_default_3 进行维度重排
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 使用 aten.bmm.default 函数对 permute_default_5 和 view_default_6 进行批量矩阵乘法
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 bmm_default_5 和一个忽略的参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个 MultiOutputPattern 对象，包含多个输出模式，其中有四个调用 aten.view.default 函数的调用，
# 以及三个值为 None 的空位
_sfdp_pattern_19_half_training = MultiOutputPattern([
  view_default_5,      # 第一个 aten.view.default 函数调用
  view_default_9,      # 第二个 aten.view.default 函数调用
  permute_default_4,   # 第三个 aten.view.default 函数调用
  view_default_11,     # 第四个 aten.view.default 函数调用
  None,                 # 空位
  None,                 # 空位
  None                  # 空位
])

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 'query' 作为关键字参数和一个忽略的参数
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 expand_default 和一个忽略的参数
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，并传入 'key' 作为关键字参数和一个忽略的参数
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 permute_default 和一个忽略的参数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 expand_default_1 和一个忽略的参数
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并传入 view_default 和 view_default_1 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 bmm_default 和一个忽略的参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.full.default 函数，并传入一些默认参数
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并传入 view_default_2 和 full_default 作为参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)

# 创建一个 CallFunction 对象，调用 aten.full.default 函数，并传入一些默认参数
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个 CallFunction 对象，调用 aten.where.self 函数，并传入 KeywordArg('causal_mask'), div_Tensor 和 full_default_1 作为参数
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)

# 创建一个 CallFunction 对象，调用 aten.add.Tensor 函数，并传入 where_self 和 KeywordArg('attn_mask') 作为参数，_users=2 表示该对象的用户数为 2
add_Tensor = CallFunction(aten.add.Tensor, where_self, KeywordArg('attn_mask'), _users=2)

# 创建一个 CallFunction 对象，调用 aten.amax.default 函数，并传入 add_Tensor 和一些默认参数
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.sub.Tensor 函数，并传入 add_Tensor 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 创建一个 CallFunction 对象，调用 aten.exp.default 函数，并传入 sub_Tensor 作为参数，_users=2 表示该对象的用户数为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，并传入 exp_default 和一些默认参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，并传入 exp_default 和 sum_dim_IntList 作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，并传入 div_Tensor_1 和一个忽略的参数
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 convert_element_type_default 和一个忽略的参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 expand_default_2 和一个忽略的参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，并传入 KeywordArg('value') 和一个忽略的参数
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 expand_default_3 和一个忽略的参数
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，并传入 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，并传入 bmm_default_1 和一个忽略的参数，_users=0 表示该对象的用户数为 0
_sfdp_pattern_19_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```