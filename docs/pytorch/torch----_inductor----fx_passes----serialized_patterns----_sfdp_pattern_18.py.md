# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_18.py`

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

# 定义 aten 和 prims 作为 torch 操作的别名
aten = torch.ops.aten
prims = torch.ops.prims

# 导入 torch._inductor.pattern_matcher 模块中的各种类
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

# 创建调用函数对象，设置各种默认参数和关键字参数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False, _users=2)
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1, _users=2)
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
# 创建一个函数调用对象，调用 aten.expand.default 函数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个函数调用对象，调用 aten.clone.default 函数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个函数调用对象，调用 aten.bmm.default 函数，表示批量矩阵乘法
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个函数调用对象，调用 aten.neg.default 函数，表示取负操作
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个函数调用对象，调用 aten.bmm.default 函数，表示批量矩阵乘法
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个函数调用对象，调用 prims.convert_element_type.default 函数，表示类型转换操作
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，表示张量的逐元素乘法
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())

# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，表示张量的逐元素乘法
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 创建一个函数调用对象，调用 aten.mul.Tensor 函数，表示张量的逐元素乘法
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)

# 创建一个函数调用对象，调用 aten.sum.dim_IntList 函数，表示按指定维度求和
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 创建一个函数调用对象，调用 prims.fma.default 函数，表示 Fused Multiply-Add 操作
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 创建一个函数调用对象，调用 aten.scalar_tensor.default 函数，表示创建一个标量张量
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())

# 创建一个函数调用对象，调用 aten.where.self 函数，表示条件选择操作
where_self_1 = CallFunction(aten.where.self, KeywordArg('causal_mask'), fma_default, scalar_tensor_default)

# 创建一个函数调用对象，调用 aten.div.Tensor 函数，表示张量的逐元素除法
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, full_default)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个函数调用对象，调用 aten.bmm.default 函数，表示批量矩阵乘法
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个函数调用对象，调用 aten.bmm.default 函数，表示批量矩阵乘法
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个函数调用对象，调用 aten.bmm.default 函数，表示批量矩阵乘法
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个函数调用对象，调用 aten.view.default 函数，用于对数据进行视图操作
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个函数调用对象，调用 aten.permute.default 函数，用于对数据进行维度置换操作
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个复合输出模式对象，表示多输出模式的操作
_sfdp_pattern_18_training = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])
# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，带有一个关键字参数 'query' 和一个被忽略的参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，带有 permute_default 作为参数和一个被忽略的参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，带有 expand_default 和一个 memory_format 参数
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 clone_default 和一个被忽略的参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，带有一个关键字参数 'key' 和一个被忽略的参数，_users 参数为 2
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，带有 permute_default_1 和一个被忽略的参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，带有 permute_default_2 和一个被忽略的参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，带有 expand_default_1 和一个 memory_format 参数
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 clone_default_1 和一个被忽略的参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.bmm.default 函数，带有 view_default 和 view_default_1 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 bmm_default 和一个被忽略的参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.full.default 函数，带有一个空列表作为参数，以及多个被忽略的参数
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，调用 torch.aten.div.Tensor 函数，带有 view_default_2 和 full_default 作为参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)

# 创建一个调用函数对象，调用 torch.aten.full.default 函数，带有一个空列表作为参数，以及多个被忽略的参数
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，调用 torch.aten.where.self 函数，带有多个参数，其中包括关键字参数 'causal_mask'
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1, _users=2)

# 创建一个调用函数对象，调用 torch.aten.amax.default 函数，带有 where_self 和多个被忽略的参数，以及一个 True 参数
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)

# 创建一个调用函数对象，调用 torch.aten.sub.Tensor 函数，带有 where_self 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)

# 创建一个调用函数对象，调用 torch.aten.exp.default 函数，带有 sub_Tensor 和 _users 参数为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 torch.aten.sum.dim_IntList 函数，带有 exp_default 和多个被忽略的参数，以及一个 True 参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 torch.aten.div.Tensor 函数，带有 exp_default 和 sum_dim_IntList 作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，带有 div_Tensor_1 和一个被忽略的参数
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 expand_default_2 和一个被忽略的参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，带有一个关键字参数 'value' 和一个被忽略的参数，_users 参数为 2
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，带有 permute_default_3 和一个被忽略的参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，带有 expand_default_3 和一个 memory_format 参数
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 clone_default_2 和一个被忽略的参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个调用函数对象，调用 torch.aten.bmm.default 函数，带有 view_default_3 和 view_default_4 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 bmm_default_1 和一个被忽略的参数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个 MultiOutputPattern 对象，包含 view_default_5, permute_default_1, permute_default_3 三个元素
_sfdp_pattern_18_inference = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3
])

# 创建一个调用函数对象，调用 torch.aten.rand.default 函数，带有多个被忽略的参数
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，调用 torch.aten.gt.Scalar 函数，带有 rand_default 和一个关键字参数 'dropout_p'，_users 参数为 2
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，带有一个关键字参数 'query' 和一个被忽略的参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，带有 permute_default 和一个被忽略的参数，_users 参数为 2
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，带有 expand_default 和一个被忽略的参数，_users 参数为 2
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 创建一个调用函数对象，调用 aten.permute.default 函数，其中包含一个关键字参数 'key'，一个忽略的参数，以及 2 个用户。
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传递上一个调用函数对象 permute_default_1 作为参数，还有一个忽略的参数。
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传递上一个调用函数对象 permute_default_2 作为参数，还有一个忽略的参数。
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，传递上一个调用函数对象 expand_default_1 作为参数，还有一个忽略的参数，以及 2 个用户。
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传递两个上一个调用函数对象 view_default_1 作为参数。
bmm_default = CallFunction(aten.bmm.default, view_default_1, view_default_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，传递上一个调用函数对象 bmm_default 作为参数，还有一个忽略的参数。
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 aten.full.default 函数，传递一个空列表作为参数，其他参数都是忽略的，设备内存不固定，有 2 个用户。
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False, _users=2)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传递两个上一个调用函数对象 view_default_2 和 full_default 作为参数。
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)

# 创建一个调用函数对象，调用 aten.full.default 函数，传递一个空列表作为参数，其他参数都是忽略的，设备内存不固定。
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，调用 aten.where.self 函数，传递四个参数，其中第一个是关键字参数 'causal_mask'，后面是前面定义的对象，有 2 个用户。
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1, _users=2)

# 创建一个调用函数对象，调用 aten.amax.default 函数，传递三个参数，其中第二个参数是忽略的，第三个参数是布尔值 True。
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)

# 创建一个调用函数对象，调用 aten.sub.Tensor 函数，传递两个上一个调用函数对象 where_self 和 amax_default 作为参数。
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)

# 创建一个调用函数对象，调用 aten.exp.default 函数，传递上一个调用函数对象 sub_Tensor 作为参数，有 2 个用户。
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，传递三个参数，其中第二个参数是忽略的，第三个参数是布尔值 True。
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传递三个上一个调用函数对象 exp_default 和 sum_dim_IntList，有 3 个用户。
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传递两个参数，其中第二个参数是忽略的。
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传递两个参数，其中第二个参数是忽略的。
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传递两个参数，其中第二个参数是忽略的。
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，传递三个参数，其中第二个参数是忽略的，有 2 个用户。
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传递三个参数，其中第一个是关键字参数 'value'，有 2 个用户。
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.expand.default 函数，传递两个参数，其中第二个参数是忽略的。
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，传递三个参数，其中第二个参数是忽略的，有 2 个用户。
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传递两个上一个调用函数对象 view_default_3 和 view_default_4 作为参数。
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传递两个参数，其中第二个参数是忽略的。
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.neg.default 函数，传递上一个调用函数对象 div_Tensor_1 作为参数。
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，传递三个参数，其中第一个是关键字参数 'tangents_1'，有 2 个用户。
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传递两个参数，其中第二个参数是忽略的。
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传递两个上一个调用函数对象 view_default_6 和 permute_default_4 作为参数。
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传递两个参数，其中第二个参数是忽略的。
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，传递两个参数，其中第二个参数是忽略的。
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传递两个参数，其中第二个参数是忽略的。
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传
# 创建一个包含默认参数的标量张量对象，调用aten.scalar_tensor.default函数
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())

# 调用aten.where.self函数，使用关键字参数'causal_mask'，传递fma_default和scalar_tensor_default作为参数
where_self_1 = CallFunction(aten.where.self, KeywordArg('causal_mask'), fma_default, scalar_tensor_default)

# 调用aten.div.Tensor函数，传递where_self_1和full_default作为参数
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, full_default)

# 调用aten.view.default函数，传递div_Tensor_2和Ignored()作为参数，并指定_users=2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 调用aten.permute.default函数，传递view_default_1和Ignored()作为参数
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 调用aten.bmm.default函数，传递view_default_8和permute_default_5作为参数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 调用aten.view.default函数，传递bmm_default_3和Ignored()作为参数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 调用aten.permute.default函数，传递view_default_9和Ignored()作为参数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 调用aten.permute.default函数，传递view_default和Ignored()作为参数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 调用aten.bmm.default函数，传递permute_default_7和view_default_8作为参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 调用aten.view.default函数，传递bmm_default_4和Ignored()作为参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 调用aten.permute.default函数，传递view_default_10和Ignored()作为参数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 调用aten.permute.default函数，传递permute_default_8和Ignored()作为参数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 调用aten.permute.default函数，传递view_default_3和Ignored()作为参数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 调用aten.bmm.default函数，传递permute_default_10和view_default_6作为参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 调用aten.view.default函数，传递bmm_default_5和Ignored()作为参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 调用aten.permute.default函数，传递view_default_11和Ignored()作为参数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个MultiOutputPattern对象，包含view_default_5等变量，并有两个None值
_sfdp_pattern_18_bs1_training = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])

# 调用aten.permute.default函数，使用关键字参数'query'，传递Ignored()作为参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 调用aten.expand.default函数，传递permute_default和Ignored()作为参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 调用aten.view.default函数，传递expand_default和Ignored()作为参数
view_default = CallFunction(aten.view.default, expand_default, Ignored())

# 调用aten.permute.default函数，使用关键字参数'key'，传递Ignored()作为参数，并指定_users=2
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)

# 调用aten.permute.default函数，传递permute_default_1和Ignored()作为参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 调用aten.expand.default函数，传递permute_default_2和Ignored()作为参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 调用aten.view.default函数，传递expand_default_1和Ignored()作为参数
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())

# 调用aten.bmm.default函数，传递view_default和view_default_1作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 调用aten.view.default函数，传递bmm_default和Ignored()作为参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用aten.full.default函数，传递[]和Ignored()作为参数，并指定dtype和device，pin_memory参数为False
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 调用aten.div.Tensor函数，传递view_default_2和full_default作为参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)

# 调用aten.full.default函数，传递[]和Ignored()作为参数，并指定dtype和device，pin_memory参数为False
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 调用aten.where.self函数，使用关键字参数'causal_mask'，传递div_Tensor、full_default_1，并指定_users=2
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1, _users=2)

# 调用aten.amax.default函数，传递where_self、Ignored()和True作为参数
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)

# 调用aten.sub.Tensor函数，传递where_self和amax_default作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)

# 调用aten.exp.default函数，传递sub_Tensor和指定_users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用函数对张量进行沿指定维度的求和操作，并将结果赋给sum_dim_IntList变量

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用函数对张量进行除法操作，将exp_default张量除以sum_dim_IntList张量的结果赋给div_Tensor_1变量

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 调用函数对张量进行扩展操作，扩展div_Tensor_1张量的形状，并将结果赋给expand_default_2变量

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用函数对张量进行视图变换操作，改变expand_default_2张量的视图，并将结果赋给view_default_3变量

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
# 调用函数对张量进行维度置换操作，根据给定的参数对张量进行维度的置换，并将结果赋给permute_default_3变量

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用函数对张量进行扩展操作，扩展permute_default_3张量的形状，并将结果赋给expand_default_3变量

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 调用函数对张量进行视图变换操作，改变expand_default_3张量的视图，并将结果赋给view_default_4变量

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用函数进行批量矩阵乘法操作，对view_default_3和view_default_4张量进行乘法操作，并将结果赋给bmm_default_1变量

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用函数对张量进行视图变换操作，改变bmm_default_1张量的视图，并将结果赋给view_default_5变量

_sfdp_pattern_18_bs1_inference = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3
])
# 创建一个多输出模式对象_sfdp_pattern_18_bs1_inference，包含view_default_5、permute_default_1和permute_default_3三个张量
# 用于某种推断过程，这些张量可能在模型的不同部分被使用

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用函数生成一个指定参数的随机张量，并将结果赋给rand_default变量

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用函数对张量进行大于比较操作，比较rand_default张量和给定的标量值，结果赋给gt_Scalar变量

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用函数对张量进行维度置换操作，根据指定参数对张量进行维度的置换，并将结果赋给permute_default变量

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用函数对张量进行扩展操作，扩展permute_default张量的形状，并将结果赋给expand_default变量

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用函数对张量进行克隆操作，复制expand_default张量的内容，并将结果赋给clone_default变量

view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 调用函数对张量进行视图变换操作，改变clone_default张量的视图，并将结果赋给view_default变量

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
# 调用函数对张量进行维度置换操作，根据指定参数对张量进行维度的置换，并将结果赋给permute_default_1变量

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用函数对张量进行维度置换操作，根据permute_default_1的置换顺序再次对张量进行维度的置换，并将结果赋给permute_default_2变量

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用函数对张量进行扩展操作，扩展permute_default_2张量的形状，并将结果赋给expand_default_1变量

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用函数对张量进行克隆操作，复制expand_default_1张量的内容，并将结果赋给clone_default_1变量

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 调用函数对张量进行视图变换操作，改变clone_default_1张量的视图，并将结果赋给view_default_1变量

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用函数进行批量矩阵乘法操作，对view_default和view_default_1张量进行乘法操作，并将结果赋给bmm_default变量

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用函数对张量进行视图变换操作，改变bmm_default张量的视图，并将结果赋给view_default_2变量

full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用函数生成一个指定参数的全零张量，并将结果赋给full_default变量

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)
# 调用函数对张量进行除法操作，将view_default_2张量除以full_default张量的结果赋给div_Tensor变量

full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用函数生成一个指定参数的全零张量，并将结果赋给full_default_1变量

where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)
# 调用函数根据条件生成张量，根据给定的条件causal_mask，生成一个条件张量，并将结果赋给where_self变量

convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
# 调用函数进行元素类型转换操作，将where_self张量的元素类型转换为指定类型，并将结果赋给convert_element_type_default变量

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用函数计算张量的最大值，并将结果赋给amax_default变量

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用函数对张量进行减法操作，将convert_element_type_default张量减去amax_default张量的结果赋给sub_Tensor变量

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用函数对张量进行指数运算操作，计算sub_Tensor张量的指数，并将结果赋给exp_default变量

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用函数对张量进行沿指定维度的求和操作，并将结果赋给sum_dim_IntList变量

div_Tensor_1 = CallFunction(aten.div.Tensor, exp
# 创建一个 CallFunction 实例，调用 prims.convert_element_type.default 函数，参数为 div_Tensor_1 和 Ignored()，_users=2 表示此函数被两处使用
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.mul.Tensor 函数，参数为 gt_Scalar 和 convert_element_type_default_1
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
# 创建一个 CallFunction 实例，调用 aten.mul.Tensor 函数，参数为 mul_Tensor 和 Ignored()
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 创建一个 CallFunction 实例，调用 aten.expand.default 函数，参数为 mul_Tensor_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 expand_default_2 和 Ignored()，_users=2 表示此函数被两处使用
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.permute.default 函数，参数为 KeywordArg('value') 和 Ignored()，_users=2 表示此函数被两处使用
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.expand.default 函数，参数为 permute_default_3 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 创建一个 CallFunction 实例，调用 aten.clone.default 函数，参数为 expand_default_3 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 clone_default_2 和 Ignored()，_users=2 表示此函数被两处使用
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.bmm.default 函数，参数为 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 bmm_default_1 和 Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 创建一个 CallFunction 实例，调用 prims.convert_element_type.default 函数，参数为 convert_element_type_default_1 和 Ignored()，_users=2 表示此函数被两处使用
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.neg.default 函数，参数为 convert_element_type_default_2
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 KeywordArg('tangents_1') 和 Ignored()，_users=2 表示此函数被两处使用
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.permute.default 函数，参数为 view_default_4 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 创建一个 CallFunction 实例，调用 aten.bmm.default 函数，参数为 view_default_6 和 permute_default_4
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 bmm_default_2 和 Ignored()
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 创建一个 CallFunction 实例，调用 prims.convert_element_type.default 函数，参数为 gt_Scalar 和 Ignored()
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 创建一个 CallFunction 实例，调用 aten.mul.Tensor 函数，参数为 convert_element_type_default_3 和 Ignored()
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
# 创建一个 CallFunction 实例，调用 aten.mul.Tensor 函数，参数为 view_default_7 和 mul_Tensor_2
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 创建一个 CallFunction 实例，调用 prims.convert_element_type.default 函数，参数为 mul_Tensor_3 和 Ignored()
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
# 创建一个 CallFunction 实例，调用 aten.mul.Tensor 函数，参数为 convert_element_type_default_4 和 convert_element_type_default_2，_users=2 表示此函数被两处使用
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
# 创建一个 CallFunction 实例，调用 aten.sum.dim_IntList 函数，参数为 mul_Tensor_4 和 Ignored()，True 表示保持维度
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 创建一个 CallFunction 实例，调用 prims.fma.default 函数，参数为 neg_default、sum_dim_IntList_1 和 mul_Tensor_4
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 创建一个 CallFunction 实例，调用 prims.convert_element_type.default 函数，参数为 fma_default 和 Ignored()
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 创建一个 CallFunction 实例，调用 aten.scalar_tensor.default 函数，参数为 Ignored()、dtype=Ignored()、layout=torch.strided 和 device=Ignored()
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())
# 创建一个 CallFunction 实例，调用 aten.where.self 函数，参数为 KeywordArg('causal_mask')、convert_element_type_default_5 和 scalar_tensor_default
where_self_1 = CallFunction(aten.where.self, KeywordArg('causal_mask'), convert_element_type_default_5, scalar_tensor_default)
# 创建一个 CallFunction 实例，调用 aten.div.Tensor 函数，参数为 where_self_1 和 full_default
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, full_default)
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 div_Tensor_2 和 Ignored()，_users=2 表示此函数被两处使用
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 创建一个 CallFunction 实例，调用 aten.permute.default 函数，参数为 view_default_1 和 Ignored()
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 创建一个 CallFunction 实例，调用 aten.bmm.default 函数，参数为 view_default_8 和 permute_default_5
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 创建一个 CallFunction 实例，调用 aten.view.default 函数，参数为 bmm_default_3 和 Ignored()
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 创建一个调用aten.permute.default函数的对象，并使用view_default_9作为参数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个调用aten.permute.default函数的对象，并使用view_default作为参数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个调用aten.bmm.default函数的对象，并使用permute_default_7和view_default_8作为参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个调用aten.view.default函数的对象，并使用bmm_default_4作为参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个调用aten.permute.default函数的对象，并使用view_default_10作为参数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个调用aten.permute.default函数的对象，并使用permute_default_8作为参数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个调用aten.permute.default函数的对象，并使用view_default_3作为参数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个调用aten.bmm.default函数的对象，并使用permute_default_10和view_default_6作为参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个调用aten.view.default函数的对象，并使用bmm_default_5作为参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个调用aten.permute.default函数的对象，并使用view_default_11作为参数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个MultiOutputPattern对象，包含view_default_5、permute_default_1、permute_default_3等作为输出
_sfdp_pattern_18_half_training = MultiOutputPattern([
  view_default_5,
  permute_default_1,
  permute_default_3,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])

# 创建一个调用aten.permute.default函数的对象，并使用KeywordArg('query')作为参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用aten.expand.default函数的对象，并使用permute_default作为参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用aten.clone.default函数的对象，并使用expand_default作为参数，使用torch.contiguous_format作为memory_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个调用aten.view.default函数的对象，并使用clone_default作为参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个调用aten.permute.default函数的对象，并使用KeywordArg('key')和_users=2作为参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)

# 创建一个调用aten.permute.default函数的对象，并使用permute_default_1作为参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用aten.expand.default函数的对象，并使用permute_default_2作为参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用aten.clone.default函数的对象，并使用expand_default_1作为参数，使用torch.contiguous_format作为memory_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用aten.view.default函数的对象，并使用clone_default_1作为参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个调用aten.bmm.default函数的对象，并使用view_default和view_default_1作为参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用aten.view.default函数的对象，并使用bmm_default作为参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用aten.full.default函数的对象，使用空列表作为参数，忽略Ignored()和其他参数
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用aten.div.Tensor函数的对象，并使用view_default_2和full_default作为参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)

# 创建一个调用aten.full.default函数的对象，使用空列表作为参数，忽略Ignored()和其他参数
full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用aten.where.self函数的对象，并使用KeywordArg('causal_mask')、div_Tensor和full_default_1作为参数
where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)

# 创建一个调用prims.convert_element_type.default函数的对象，并使用where_self作为参数，_users=2表示有两个用户
convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)

# 创建一个调用aten.amax.default函数的对象，并使用convert_element_type_default作为参数，True表示忽略其他参数
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个调用aten.sub.Tensor函数的对象，并使用convert_element_type_default和amax_default作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个调用aten.exp.default函数的对象，并使用sub_Tensor作为参数，_users=2表示有两个用户
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用aten.sum.dim_IntList函数的对象，并使用exp_default作为参数，True表示忽略其他参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用aten.div.Tensor函数的对象，并使用exp_default和sum_dim_IntList作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用prims.convert_element_type.default函数的对象，并使用div_Tensor_1作为参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的 expand 函数，使用 convert_element_type_default_1 作为参数进行张量扩展

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 PyTorch 的 view 函数，使用 expand_default_2 作为参数进行张量视图变换

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
# 调用 PyTorch 的 permute 函数，通过关键字参数 'value' 进行张量的维度置换

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 PyTorch 的 expand 函数，使用 permute_default_3 作为参数进行张量扩展

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 clone 函数，使用 expand_default_3 作为参数进行张量的克隆操作，并指定内存格式为连续格式

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 调用 PyTorch 的 view 函数，使用 clone_default_2 作为参数进行张量视图变换

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的 bmm 函数，使用 view_default_3 和 view_default_4 作为参数进行批量矩阵乘法计算

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的 view 函数，使用 bmm_default_1 作为参数进行张量视图变换

_sfdp_pattern_18_half_inference = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3
])
# 创建多输出模式对象 _sfdp_pattern_18_half_inference，包含 view_default_5、permute_default_1 和 permute_default_3

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 PyTorch 的 rand 函数，生成随机张量，可以指定数据类型、设备和是否锁定内存

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用 PyTorch 的 gt 函数，使用 rand_default 作为参数进行张量比较，通过关键字参数 'dropout_p' 指定 dropout 概率

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的 permute 函数，通过关键字参数 'query' 进行张量的维度置换

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的 expand 函数，使用 permute_default 作为参数进行张量扩展

view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，使用 expand_default 作为参数进行张量视图变换

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
# 调用 PyTorch 的 permute 函数，通过关键字参数 'key' 进行张量的维度置换

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 PyTorch 的 permute 函数，使用 permute_default_1 作为参数进行张量的维度置换

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 PyTorch 的 expand 函数，使用 permute_default_2 作为参数进行张量扩展

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，使用 expand_default_1 作为参数进行张量视图变换

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的 bmm 函数，使用 view_default 和 view_default_1 作为参数进行批量矩阵乘法计算

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的 view 函数，使用 bmm_default 作为参数进行张量视图变换

full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False, _users=2)
# 调用 PyTorch 的 full 函数，生成全零张量，可以指定数据类型、设备和是否锁定内存

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)
# 调用 PyTorch 的 div 函数，使用 view_default_2 和 full_default 作为参数进行张量除法运算

full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 PyTorch 的 full 函数，生成全零张量，可以指定数据类型、设备和是否锁定内存

where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)
# 调用 PyTorch 的 where 函数，通过关键字参数 'causal_mask' 进行条件选择操作，使用 div_Tensor 和 full_default_1 作为参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，使用 where_self 作为参数进行张量类型转换

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的 amax 函数，使用 convert_element_type_default 作为参数进行张量最大值计算，并指定返回一个元组

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的 sub 函数，使用 convert_element_type_default 和 amax_default 作为参数进行张量减法运算

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的 exp 函数，使用 sub_Tensor 作为参数进行张量指数运算

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的 sum 函数，通过维度列表进行张量求和操作，使用 exp_default 作为参数，并指定返回一个元组

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的 div 函数，使用 exp_default 和 sum_dim_IntList 作为参数进行张量除法运算

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，使用 div_Tensor_1 作为参数进行张量类型转换

mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
# 调用 PyTorch 的 mul 函数，使用 gt_Scalar 和 convert_element_type_default_1 作为参数进行张量乘法运算

mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 调用 PyTorch 的 mul 函数，使用 mul_Tensor 作为参数进行张量乘法运算
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 调用 PyTorch 的 expand 函数，默认模式，对 mul_Tensor_1 进行扩展操作

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，默认模式，对 expand_default_2 进行视图变换操作

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
# 调用 PyTorch 的 permute 函数，默认模式，对指定的维度进行排列操作

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 PyTorch 的 expand 函数，默认模式，对 permute_default_3 进行扩展操作

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，默认模式，对 expand_default_3 进行视图变换操作

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的 bmm 函数，默认模式，执行两个 batch 矩阵乘法操作

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的 view 函数，默认模式，对 bmm_default_1 进行视图变换操作

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 调用 convert_element_type 函数，将 convert_element_type_default_1 转换为指定类型

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 调用 PyTorch 的 neg 函数，默认模式，对 convert_element_type_default_2 进行取负操作

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，默认模式，对 tangents_1 进行视图变换操作

permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 PyTorch 的 permute 函数，默认模式，对 view_default_4 进行维度排列操作

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 调用 PyTorch 的 bmm 函数，默认模式，执行两个 batch 矩阵乘法操作

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 PyTorch 的 view 函数，默认模式，对 bmm_default_2 进行视图变换操作

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用 convert_element_type 函数，将 gt_Scalar 转换为指定类型

mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
# 调用 PyTorch 的 mul 函数，对 convert_element_type_default_3 进行张量乘法操作

mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 调用 PyTorch 的 mul 函数，对 view_default_7 和 mul_Tensor_2 进行张量乘法操作

convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
# 调用 convert_element_type 函数，将 mul_Tensor_3 转换为指定类型

mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
# 调用 PyTorch 的 mul 函数，对 convert_element_type_default_4 和 convert_element_type_default_2 进行张量乘法操作

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 调用 PyTorch 的 sum 函数，对 mul_Tensor_4 按指定维度进行求和操作

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用 fma 函数，进行 Fused Multiply-Add 操作，包括负数、求和、和乘法

convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用 convert_element_type 函数，将 fma_default 转换为指定类型

scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())
# 调用 PyTorch 的 scalar_tensor 函数，创建标量张量

where_self_1 = CallFunction(aten.where.self, KeywordArg('causal_mask'), convert_element_type_default_5, scalar_tensor_default)
# 调用 PyTorch 的 where 函数，应用于自身张量，包括 causal_mask 和 convert_element_type_default_5

div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, full_default)
# 调用 PyTorch 的 div 函数，对 where_self_1 和 full_default 进行张量除法操作

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，默认模式，对 div_Tensor_2 进行视图变换操作

permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 调用 PyTorch 的 permute 函数，默认模式，对 view_default_1 进行维度排列操作

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
# 调用 PyTorch 的 bmm 函数，默认模式，执行两个 batch 矩阵乘法操作

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 调用 PyTorch 的 view 函数，默认模式，对 bmm_default_3 进行视图变换操作

permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())
# 调用 PyTorch 的 permute 函数，默认模式，对 view_default_9 进行维度排列操作

permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
# 调用 PyTorch 的 permute 函数，默认模式，对 view_default 进行维度排列操作

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
# 调用 PyTorch 的 bmm 函数，默认模式，执行两个 batch 矩阵乘法操作

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 调用 PyTorch 的 view 函数，默认模式，对 bmm_default_4 进行视图变换操作

permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 调用 PyTorch 的 permute 函数，默认模式，对 view_default_10 进行维度排列操作
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
# 使用默认参数调用 torch.permute 函数，对 permute_default_8 进行排列操作

permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default_3 进行排列操作

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
# 使用默认参数调用 torch.bmm 函数，对 permute_default_10 和 view_default_6 进行批量矩阵乘法

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default_5 进行视图变换

permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
# 使用默认参数调用 torch.permute 函数，对 view_default_11 进行排列操作

_sfdp_pattern_18_half_bs1_training = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])
# 创建一个 MultiOutputPattern 对象，包含多个输出模式，其中 view_default_5, permute_default_1 等均为输入模式

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 使用默认参数调用 torch.permute 函数，使用关键字参数 'query'

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 使用默认参数调用 torch.expand 函数，对 permute_default 进行扩展操作

view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 使用默认参数调用 torch.view 函数，对 expand_default 进行视图变换

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
# 使用默认参数调用 torch.permute 函数，使用关键字参数 'key'，同时设定 _users 参数为 2

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 使用默认参数调用 torch.permute 函数，对 permute_default_1 进行排列操作

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 使用默认参数调用 torch.expand 函数，对 permute_default_2 进行扩展操作

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 使用默认参数调用 torch.view 函数，对 expand_default_1 进行视图变换

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 使用默认参数调用 torch.bmm 函数，对 view_default 和 view_default_1 进行批量矩阵乘法

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default 进行视图变换

full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 使用默认参数调用 torch.full 函数，创建一个全填充张量

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, full_default)
# 使用默认参数调用 torch.div 函数，对 view_default_2 和 full_default 进行张量除法操作

full_default_1 = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 使用默认参数调用 torch.full 函数，创建另一个全填充张量

where_self = CallFunction(aten.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_1)
# 使用默认参数调用 torch.where 函数，使用关键字参数 'causal_mask'，对 div_Tensor 和 full_default_1 进行条件操作

convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
# 使用默认参数调用 prims.convert_element_type 函数，对 where_self 进行元素类型转换

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 使用默认参数调用 torch.amax 函数，对 convert_element_type_default 进行求最大值操作，同时指定 keepdim=True

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 使用默认参数调用 torch.sub 函数，对 convert_element_type_default 和 amax_default 进行张量减法操作

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 使用默认参数调用 torch.exp 函数，对 sub_Tensor 进行指数运算

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 使用默认参数调用 torch.sum 函数，对 exp_default 按指定维度列表进行求和操作，同时指定 keepdim=True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 使用默认参数调用 torch.div 函数，对 exp_default 和 sum_dim_IntList 进行张量除法操作

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 使用默认参数调用 prims.convert_element_type 函数，对 div_Tensor_1 进行元素类型转换

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 使用默认参数调用 torch.expand 函数，对 convert_element_type_default_1 进行扩展操作

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 使用默认参数调用 torch.view 函数，对 expand_default_2 进行视图变换

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
# 使用默认参数调用 torch.permute 函数，使用关键字参数 'value'，同时设定 _users 参数为 2

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 使用默认参数调用 torch.expand 函数，对 permute_default_3 进行扩展操作

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 使用默认参数调用 torch.view 函数，对 expand_default_3 进行视图变换

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 使用默认参数调用 torch.bmm 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 使用默认参数调用 torch.view 函数，对 bmm_default_1 进行视图变换
# 定义一个名为 _sfdp_pattern_18_half_bs1_inference 的变量，其值为一个 MultiOutputPattern 对象
# MultiOutputPattern 对象由以下三个参数构成：view_default_5, permute_default_1, permute_default_3
_sfdp_pattern_18_half_bs1_inference = MultiOutputPattern([view_default_5,
  permute_default_1,
  permute_default_3
])
```