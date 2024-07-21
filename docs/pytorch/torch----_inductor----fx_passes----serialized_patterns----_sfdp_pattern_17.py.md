# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_17.py`

```py
# mypy: ignore-errors
# 忽略类型检查错误，mypy 指定的注释

# noqa: F401, E501
# 禁止 Flake8 检查中的 F401 未使用的导入和 E501 行过长错误
# 这是一个自动生成的文件，请勿手动修改。
# 要重新生成，请运行：
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py
# 注释内容：文件注释和重新生成命令的说明

import torch
import torch._inductor
# 导入 torch 和 torch._inductor 模块

aten = torch.ops.aten
prims = torch.ops.prims
# 设置简化的访问方式以避免重复输入 torch.ops.aten 和 torch.ops.prims

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
# 从 torch._inductor.pattern_matcher 模块导入多个类和对象

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 torch.ops.aten.rand.default 函数，忽略部分参数并设置 pin_memory=False

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用 torch.ops.aten.gt.Scalar 函数，指定部分参数和 _users=2

eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())
# 调用 torch.ops.aten.eq.Scalar 函数，指定部分参数和忽略部分参数

expand_default = CallFunction(aten.expand.default, eq_Scalar, Ignored(), _users=2)
# 调用 torch.ops.aten.expand.default 函数，指定部分参数和 _users=2

full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 torch.ops.aten.full.default 函数，指定部分参数和空列表作为参数

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 torch.ops.aten.permute.default 函数，指定部分参数和关键字参数 'query'

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 torch.ops.aten.expand.default 函数，指定部分参数和忽略部分参数

clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用 torch.ops.aten.clone.default 函数，指定部分参数和 memory_format=torch.contiguous_format

view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 调用 torch.ops.aten.view.default 函数，指定部分参数和 _users=2

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 torch.ops.aten.permute.default 函数，指定部分参数和关键字参数 'key'

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 torch.ops.aten.permute.default 函数，指定部分参数和忽略部分参数

expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 torch.ops.aten.expand.default 函数，指定部分参数和忽略部分参数

clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)
# 调用 torch.ops.aten.clone.default 函数，指定部分参数和 memory_format=torch.contiguous_format

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 调用 torch.ops.aten.view.default 函数，指定部分参数和 _users=2

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 torch.ops.aten.bmm.default 函数，指定部分参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 torch.ops.aten.view.default 函数，指定部分参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 torch.ops.aten.div.Tensor 函数，指定部分参数和关键字参数 'inv_scale'

where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor, _users=2)
# 调用 torch.ops.aten.where.self 函数，指定部分参数和 _users=2

amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)
# 调用 torch.ops.aten.amax.default 函数，指定部分参数和忽略部分参数以及布尔类型 True

sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)
# 调用 torch.ops.aten.sub.Tensor 函数，指定部分参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 torch.ops.aten.exp.default 函数，指定部分参数和 _users=2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 torch.ops.aten.sum.dim_IntList 函数，指定部分参数和忽略部分参数以及布尔类型 True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 调用 torch.ops.aten.div.Tensor 函数，指定部分参数和 _users=3

mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor_1)
# 调用 torch.ops.aten.mul.Tensor 函数，指定部分参数

mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 调用 torch.ops.aten.mul.Tensor 函数，指定部分参数和忽略部分参数

expand_default_3 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 调用 torch.ops.aten.expand.default 函数，指定部分参数和忽略部分参数

view_default_3 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用 torch.ops.aten.view.default 函数，指定部分参数和 _users=2

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 torch.ops.aten.permute.default 函数，指定部分参数和关键字参数 'value'
# 创建一个函数调用对象，调用 torch 中 aten.expand.default 函数
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.clone.default 函数
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个函数调用对象，调用 torch 中 aten.bmm.default 函数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.scalar_tensor.default 函数
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.neg.default 函数
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.bmm.default 函数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个函数调用对象，调用 torch 中 prims.convert_element_type.default 函数
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.mul.Tensor 函数
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.mul.Tensor 函数
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 创建一个函数调用对象，调用 torch 中 aten.mul.Tensor 函数
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)

# 创建一个函数调用对象，调用 torch 中 aten.sum.dim_IntList 函数
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 创建一个函数调用对象，调用 torch 中 prims.fma.default 函数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 创建一个函数调用对象，调用 torch 中 aten.where.self 函数
where_self_1 = CallFunction(aten.where.self, expand_default, scalar_tensor_default, fma_default)

# 创建一个函数调用对象，调用 torch 中 aten.div.Tensor 函数
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, KeywordArg('inv_scale'))

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.bmm.default 函数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.bmm.default 函数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.bmm.default 函数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个函数调用对象，调用 torch 中 aten.view.default 函数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个函数调用对象，调用 torch 中 aten.permute.default 函数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个函数调用对象，调用 torch 中 MultiOutputPattern 构造函数
_sfdp_pattern_17_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])
# 调用 CallFunction 函数创建 eq_Scalar 对象，用于执行 aten.eq.Scalar 操作
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())

# 调用 CallFunction 函数创建 view_default 对象，用于执行 aten.view.default 操作
view_default = CallFunction(aten.view.default, eq_Scalar, Ignored())

# 调用 CallFunction 函数创建 expand_default 对象，用于执行 aten.expand.default 操作
expand_default = CallFunction(aten.expand.default, view_default, Ignored())

# 调用 CallFunction 函数创建 full_default 对象，用于执行 aten.full.default 操作
# 生成一个空张量，用于存放计算结果，同时指定 dtype 和 device
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 调用 CallFunction 函数创建 permute_default 对象，用于执行 aten.permute.default 操作
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 调用 CallFunction 函数创建 expand_default_1 对象，用于执行 aten.expand.default 操作
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 调用 CallFunction 函数创建 clone_default 对象，用于执行 aten.clone.default 操作
# 克隆 expand_default_1 张量，并指定内存格式为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 调用 CallFunction 函数创建 view_default_1 对象，用于执行 aten.view.default 操作
view_default_1 = CallFunction(aten.view.default, clone_default, Ignored())

# 调用 CallFunction 函数创建 permute_default_1 对象，用于执行 aten.permute.default 操作
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 调用 CallFunction 函数创建 permute_default_2 对象，用于执行 aten.permute.default 操作
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 调用 CallFunction 函数创建 expand_default_2 对象，用于执行 aten.expand.default 操作
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 调用 CallFunction 函数创建 clone_default_1 对象，用于执行 aten.clone.default 操作
# 克隆 expand_default_2 张量，并指定内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)

# 调用 CallFunction 函数创建 view_default_2 对象，用于执行 aten.view.default 操作
view_default_2 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 调用 CallFunction 函数创建 bmm_default 对象，用于执行 aten.bmm.default 操作
bmm_default = CallFunction(aten.bmm.default, view_default_1, view_default_2)

# 调用 CallFunction 函数创建 view_default_3 对象，用于执行 aten.view.default 操作
view_default_3 = CallFunction(aten.view.default, bmm_default, Ignored())

# 调用 CallFunction 函数创建 div_Tensor 对象，用于执行 aten.div.Tensor 操作
# 执行张量之间的除法操作，其中第一个张量为 view_default_3，第二个张量由关键字参数 'inv_scale' 指定
div_Tensor = CallFunction(aten.div.Tensor, view_default_3, KeywordArg('inv_scale'))

# 调用 CallFunction 函数创建 where_self 对象，用于执行 aten.where.self 操作
# 使用 expand_default、full_default 和 div_Tensor 进行条件判断和元素选择
where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor, _users=2)

# 调用 CallFunction 函数创建 amax_default 对象，用于执行 aten.amax.default 操作
# 计算 where_self 张量中的最大值，同时记录计算梯度
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)

# 调用 CallFunction 函数创建 sub_Tensor 对象，用于执行 aten.sub.Tensor 操作
# 执行张量之间的减法操作，其中第一个张量为 where_self，第二个张量为 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)

# 调用 CallFunction 函数创建 exp_default 对象，用于执行 aten.exp.default 操作
# 计算 sub_Tensor 张量的指数
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 调用 CallFunction 函数创建 sum_dim_IntList 对象，用于执行 aten.sum.dim_IntList 操作
# 沿指定维度对 exp_default 张量进行求和，同时记录计算梯度
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 调用 CallFunction 函数创建 div_Tensor_1 对象，用于执行 aten.div.Tensor 操作
# 执行张量之间的除法操作，其中第一个张量为 exp_default，第二个张量为 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 调用 CallFunction 函数创建 expand_default_3 对象，用于执行 aten.expand.default 操作
expand_default_3 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())

# 调用 CallFunction 函数创建 view_default_4 对象，用于执行 aten.view.default 操作
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 调用 CallFunction 函数创建 permute_default_3 对象，用于执行 aten.permute.default 操作
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 调用 CallFunction 函数创建 expand_default_4 对象，用于执行 aten.expand.default 操作
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 调用 CallFunction 函数创建 clone_default_2 对象，用于执行 aten.clone.default 操作
# 克隆 expand_default_4 张量，并指定内存格式为 torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)

# 调用 CallFunction 函数创建 view_default_5 对象，用于执行 aten.view.default 操作
view_default_5 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 调用 CallFunction 函数创建 bmm_default_1 对象，用于执行 aten.bmm.default 操作
bmm_default_1 = CallFunction(aten.bmm.default, view_default_4, view_default_5)

# 调用 CallFunction 函数创建 _sfdp_pattern_17_inference 对象，用于执行 aten.view.default 操作
_sfdp_pattern_17_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)

# 调用 CallFunction 函数创建 rand_default 对象，用于执行 aten.rand.default 操作
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 调用 CallFunction 函数创建 gt_Scalar 对象，用于执行 aten.gt.Scalar 操作
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 调用 CallFunction 函数创建 eq_Scalar 对象，用于执行 aten.eq.Scalar 操作
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())

# 调用 CallFunction 函数创建 expand_default 对象，用于执行 aten.expand.default 操作
expand_default = CallFunction(aten.expand.default, eq_Scalar, Ignored(), _users=2)

# 调用 CallFunction 函数创建 full_default 对象，用于执行 aten.full.default 操作
# 生成一个空张量，用于存放计算结果，同时指定 dtype 和 device
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 创建一个调用函数对象，调用 aten.permute.default 函数，参数为 'query' 关键字参数和一个被忽略的参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，参数为 permute_default 和一个被忽略的参数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，参数为 expand_default_1 和 memory_format=torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 clone_default、一个被忽略的参数和 _users=2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，参数为 'key' 关键字参数和一个被忽略的参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，参数为 permute_default_1 和一个被忽略的参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，参数为 permute_default_2 和一个被忽略的参数
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，参数为 expand_default_2 和 memory_format=torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 clone_default_1、一个被忽略的参数和 _users=2
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，参数为 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 bmm_default 和一个被忽略的参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，参数为 view_default_2 和关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 aten.where.self 函数，参数为 expand_default、full_default 和 div_Tensor
where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，参数为 where_self 和一个被忽略的参数，_users=2
convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.amax.default 函数，参数为 convert_element_type_default 和一个被忽略的参数，True 表示 keepdim
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.sub.Tensor 函数，参数为 convert_element_type_default 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个调用函数对象，调用 aten.exp.default 函数，参数为 sub_Tensor 和 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，参数为 exp_default、一个被忽略的参数，True 表示 keepdim
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，参数为 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，参数为 div_Tensor_1 和一个被忽略的参数，_users=2
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，参数为 gt_Scalar 和 convert_element_type_default_1
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，参数为 mul_Tensor 和一个被忽略的参数
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，参数为 mul_Tensor_1 和一个被忽略的参数
expand_default_3 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 expand_default_3、一个被忽略的参数和 _users=2
view_default_3 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，参数为 'value' 关键字参数和一个被忽略的参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，参数为 permute_default_3 和一个被忽略的参数
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，参数为 expand_default_4 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 clone_default_2、一个被忽略的参数和 _users=2
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.bmm.default 函数，参数为 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，参数为 bmm_default_1 和一个被忽略的参数
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.scalar_tensor.default 函数，参数为一个被忽略的参数，dtype、layout 和 device 是固定的
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，参数为 convert_element_type_default_1 和一个被忽略的参数，_users=2
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.neg.default 函数，参数为 convert_element_type_default_2
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 tangents_1 参数，用户数为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_4 参数
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 view_default_6 和 permute_default_4 参数
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_2 参数
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，传入 gt_Scalar 参数
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，传入 convert_element_type_default_3 参数
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，传入 view_default_7 和 mul_Tensor_2 参数
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，传入 mul_Tensor_3 参数
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.mul.Tensor 函数，传入 convert_element_type_default_4 和 convert_element_type_default_2 参数，用户数为 2
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，传入 mul_Tensor_4 参数，忽略其他参数，keepdim 参数为 True
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)

# 创建一个 CallFunction 对象，调用 prims.fma.default 函数，传入 neg_default、sum_dim_IntList_1 和 mul_Tensor_4 参数
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，传入 fma_default 参数
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.where.self 函数，传入 expand_default、scalar_tensor_default 和 convert_element_type_default_5 参数
where_self_1 = CallFunction(aten.where.self, expand_default, scalar_tensor_default, convert_element_type_default_5)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，传入 where_self_1 和 KeywordArg('inv_scale') 参数
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, KeywordArg('inv_scale'))

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 div_Tensor_2 参数，用户数为 2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_1 参数
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 view_default_8 和 permute_default_5 参数
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_3 参数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_9 参数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default 参数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 permute_default_7 和 view_default_8 参数
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_4 参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_10 参数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 permute_default_8 参数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_3 参数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，传入 permute_default_10 和 view_default_6 参数
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 bmm_default_5 参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，传入 view_default_11 参数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，包含 view_default_5、permute_default_6、permute_default_9、permute_default_11 等参数
_sfdp_pattern_17_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None,
  None
])

# 创建一个 CallFunction 对象，调用 aten.eq.Scalar 函数，传入 KeywordArg('attn_mask') 参数，忽略其他参数
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，传入 eq_Scalar 参数
view_default = CallFunction(aten.view.default, eq_Scalar, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，传入 view_default 参数，忽略其他参数
expand_default = CallFunction(aten.expand.default, view_default, Ignored())

# 创建一个 CallFunction 对象，调用 aten.full.default 函数，传入 [] 参数，忽略其他参数，dtype、device 和 pin_memory 参数分别为 Ignored、Ignored 和 False
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 创建 CallFunction 对象，用于执行 aten.permute.default 函数，指定 'query' 参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 创建 CallFunction 对象，用于执行 aten.expand.default 函数，参数为 permute_default，另一参数被忽略
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建 CallFunction 对象，用于执行 aten.clone.default 函数，参数为 expand_default_1 和 memory_format=torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 创建 CallFunction 对象，用于执行 aten.view.default 函数，参数为 clone_default，另一参数被忽略
view_default_1 = CallFunction(aten.view.default, clone_default, Ignored())
# 创建 CallFunction 对象，用于执行 aten.permute.default 函数，指定 'key' 参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建 CallFunction 对象，用于执行 aten.permute.default 函数，参数为 permute_default_1，另一参数被忽略
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 创建 CallFunction 对象，用于执行 aten.expand.default 函数，参数为 permute_default_2，另一参数被忽略
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 创建 CallFunction 对象，用于执行 aten.clone.default 函数，参数为 expand_default_2 和 memory_format=torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)
# 创建 CallFunction 对象，用于执行 aten.view.default 函数，参数为 clone_default_1，另一参数被忽略
view_default_2 = CallFunction(aten.view.default, clone_default_1, Ignored())
# 创建 CallFunction 对象，用于执行 aten.bmm.default 函数，参数为 view_default_1 和 view_default_2
bmm_default = CallFunction(aten.bmm.default, view_default_1, view_default_2)
# 创建 CallFunction 对象，用于执行 aten.view.default 函数，参数为 bmm_default，另一参数被忽略
view_default_3 = CallFunction(aten.view.default, bmm_default, Ignored())
# 创建 CallFunction 对象，用于执行 aten.div.Tensor 函数，参数为 view_default_3 和 KeywordArg('inv_scale')
div_Tensor = CallFunction(aten.div.Tensor, view_default_3, KeywordArg('inv_scale'))
# 创建 CallFunction 对象，用于执行 aten.where.self 函数，参数为 expand_default, full_default, div_Tensor
where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor)
# 创建 CallFunction 对象，用于执行 prims.convert_element_type.default 函数，参数为 where_self 和 Ignored()，_users 设置为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
# 创建 CallFunction 对象，用于执行 aten.amax.default 函数，参数为 convert_element_type_default 和 Ignored()，指定 reduce_all 参数为 True
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 创建 CallFunction 对象，用于执行 aten.sub.Tensor 函数，参数为 convert_element_type_default 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 创建 CallFunction 对象，用于执行 aten.exp.default 函数，参数为 sub_Tensor，_users 设置为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建 CallFunction 对象，用于执行 aten.sum.dim_IntList 函数，参数为 exp_default 和 Ignored()，指定 keepdim 参数为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建 CallFunction 对象，用于执行 aten.div.Tensor 函数，参数为 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 创建 CallFunction 对象，用于执行 prims.convert_element_type.default 函数，参数为 div_Tensor_1 和 Ignored()
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
# 创建 CallFunction 对象，用于执行 aten.expand.default 函数，参数为 convert_element_type_default_1 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 创建 CallFunction 对象，用于执行 aten.view.default 函数，参数为 expand_default_3，另一参数被忽略
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 创建 CallFunction 对象，用于执行 aten.permute.default 函数，指定 'value' 参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 创建 CallFunction 对象，用于执行 aten.expand.default 函数，参数为 permute_default_3 和 Ignored()
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 创建 CallFunction 对象，用于执行 aten.clone.default 函数，参数为 expand_default_4 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)
# 创建 CallFunction 对象，用于执行 aten.view.default 函数，参数为 clone_default_2，另一参数被忽略
view_default_5 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 创建 CallFunction 对象，用于执行 aten.bmm.default 函数，参数为 view_default_4 和 view_default_5
bmm_default_1 = CallFunction(aten.bmm.default, view_default_4, view_default_5)
# 创建 CallFunction 对象，用于执行 aten.view.default 函数，参数为 bmm_default_1，另一参数被忽略，_users 设置为 0
_sfdp_pattern_17_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```