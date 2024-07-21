# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_15.py`

```
# 忽略类型检查错误
# noqa: F401, E501
# 这是一个自动生成的文件，请勿手动修改。
# 要重新生成，请运行:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

# 导入 torch 库
import torch
# 导入 torch._inductor 模块
import torch._inductor

# 使用 torch.ops.aten 别名为 aten
aten = torch.ops.aten
# 使用 torch.ops.prims 别名为 prims
prims = torch.ops.prims

# 从 torch._inductor.pattern_matcher 模块导入多个类
from torch._inductor.pattern_matcher import (
   Arg,  # 参数模式匹配
   CallFunction,  # 函数调用模式
   CallFunctionVarArgs,  # 可变参数函数调用模式
   CallMethod,  # 方法调用模式
   CallMethodVarArgs,  # 可变参数方法调用模式
   CallModule,  # 模块调用模式
   CallModuleVarArgs,  # 可变参数模块调用模式
   ExclusiveKeywordArg,  # 独占关键字参数模式
   Ignored,  # 忽略模式
   KeywordArg,  # 关键字参数模式
   ListOf,  # 列表模式
   MultiOutputPattern,  # 多输出模式
   PatternExpr,  # 模式表达式
   RepeatedExpr,  # 重复表达式
   _TargetArgsExpr,  # 目标参数表达式
   _TargetExpr,  # 目标表达式
   _TargetExprVarArgs,  # 可变参数目标表达式
)

# 定义 eq_Scalar 模式匹配对象
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())
# 定义 expand_default 模式匹配对象
expand_default = CallFunction(aten.expand.default, eq_Scalar, Ignored(), _users=2)
# 定义 full_default 模式匹配对象
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 定义 permute_default 模式匹配对象
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 定义 expand_default_1 模式匹配对象
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 定义 clone_default 模式匹配对象
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 定义 view_default 模式匹配对象
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 定义 permute_default_1 模式匹配对象
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 定义 permute_default_2 模式匹配对象
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 定义 expand_default_2 模式匹配对象
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 定义 clone_default_1 模式匹配对象
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)
# 定义 view_default_1 模式匹配对象
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 定义 bmm_default 模式匹配对象
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 定义 view_default_2 模式匹配对象
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 定义 div_Tensor 模式匹配对象
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 定义 where_self 模式匹配对象
where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor, _users=2)
# 定义 amax_default 模式匹配对象
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)
# 定义 sub_Tensor 模式匹配对象
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)
# 定义 exp_default 模式匹配对象
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 定义 sum_dim_IntList 模式匹配对象
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 定义 div_Tensor_1 模式匹配对象
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
# 定义 expand_default_3 模式匹配对象
expand_default_3 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 定义 view_default_3 模式匹配对象
view_default_3 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 定义 permute_default_3 模式匹配对象
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 定义 expand_default_4 模式匹配对象
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 定义 clone_default_2 模式匹配对象
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)
# 定义 view_default_4 模式匹配对象
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_1 和 Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.scalar_tensor.default 函数，传入参数 Ignored()，dtype=Ignored()，layout=torch.strided，device=Ignored()
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())

# 创建一个调用函数对象，调用 aten.neg.default 函数，传入参数 div_Tensor_1
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 KeywordArg('tangents_1') 和 Ignored()，_users=2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_4 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_6 和 permute_default_4
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_2 和 Ignored()
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.mul.Tensor 函数，传入参数 view_default_7 和 div_Tensor_1，_users=2
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_7, div_Tensor_1, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，传入参数 mul_Tensor 和 Ignored()，True
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)

# 创建一个调用函数对象，调用 prims.fma.default 函数，传入参数 neg_default 和 sum_dim_IntList_1，mul_Tensor
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)

# 创建一个调用函数对象，调用 aten.where.self 函数，传入参数 expand_default、scalar_tensor_default 和 fma_default
where_self_1 = CallFunction(aten.where.self, expand_default, scalar_tensor_default, fma_default)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，传入参数 where_self_1 和 KeywordArg('inv_scale')
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 div_Tensor_2 和 Ignored()，_users=2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_1 和 Ignored()
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 view_default_8 和 permute_default_5
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_3 和 Ignored()
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_9 和 Ignored()
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default 和 Ignored()
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 permute_default_7 和 view_default_8
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_4 和 Ignored()
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_10 和 Ignored()
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 permute_default_8 和 Ignored()
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_3 和 Ignored()
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，传入参数 permute_default_10 和 view_default_6
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 bmm_default_5 和 Ignored()
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 view_default_11 和 Ignored()
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个 MultiOutputPattern 对象，包含了 view_default_5、permute_default_6、permute_default_9、permute_default_11 等
_sfdp_pattern_15_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])

# 创建一个调用函数对象，调用 aten.eq.Scalar 函数，传入参数 KeywordArg('attn_mask') 和 Ignored()
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，传入参数 eq_Scalar 和 Ignored()
view_default = CallFunction(aten.view.default, eq_Scalar, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传入参数 view_default 和 Ignored()
expand_default = CallFunction(aten.expand.default, view_default, Ignored())

# 创建一个调用函数对象，调用 aten.full.default 函数，传入参数 []、Ignored()，dtype=Ignored()，device=Ignored()，pin_memory=False
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，调用 aten.permute.default 函数，传入参数 KeywordArg('query') 和 Ignored()
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，传入参数 permute_default 和 Ignored()
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，并传入 expand_default_1 和内存格式参数 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 clone_default 和一个未指定的参数
view_default_1 = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，并传入关键字参数 'key' 和一个未指定的参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，并传入 permute_default_1 和一个未指定的参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，并传入 permute_default_2 和一个未指定的参数
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，并传入 expand_default_2 和内存格式参数 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 clone_default_1 和一个未指定的参数
view_default_2 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.bmm.default 函数，并传入 view_default_1 和 view_default_2 作为参数
bmm_default = CallFunction(aten.bmm.default, view_default_1, view_default_2)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 bmm_default 和一个未指定的参数
view_default_3 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.div.Tensor 函数，并传入 view_default_3 和关键字参数 'inv_scale'
div_Tensor = CallFunction(aten.div.Tensor, view_default_3, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 torch.aten.where.self 函数，并传入 expand_default、full_default、div_Tensor 作为参数，_users 设置为 2
where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor, _users=2)

# 创建一个调用函数对象，调用 torch.aten.amax.default 函数，并传入 where_self、一个未指定的参数、True
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)

# 创建一个调用函数对象，调用 torch.aten.sub.Tensor 函数，并传入 where_self 和 amax_default 作为参数
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)

# 创建一个调用函数对象，调用 torch.aten.exp.default 函数，并传入 sub_Tensor 作为参数，_users 设置为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 torch.aten.sum.dim_IntList 函数，并传入 exp_default、一个未指定的参数、True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 torch.aten.div.Tensor 函数，并传入 exp_default 和 sum_dim_IntList 作为参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，并传入 div_Tensor_1 和一个未指定的参数
expand_default_3 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 expand_default_3 和一个未指定的参数
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，并传入关键字参数 'value' 和一个未指定的参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，并传入 permute_default_3 和一个未指定的参数
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，并传入 expand_default_4 和内存格式参数 torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 clone_default_2 和一个未指定的参数
view_default_5 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个调用函数对象，调用 torch.aten.bmm.default 函数，并传入 view_default_4 和 view_default_5 作为参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_4, view_default_5)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 bmm_default_1 和一个未指定的参数，_users 设置为 0
_sfdp_pattern_15_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)

# 创建一个调用函数对象，调用 torch.aten.eq.Scalar 函数，并传入关键字参数 'attn_mask' 和一个未指定的参数
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，并传入 eq_Scalar 和一个未指定的参数，_users 设置为 2
expand_default = CallFunction(aten.expand.default, eq_Scalar, Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch.aten.full.default 函数，并传入空列表、一个未指定的参数、dtype 和 device 等参数未指定，pin_memory 设置为 False
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，并传入关键字参数 'query' 和一个未指定的参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，并传入 permute_default 和一个未指定的参数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，并传入 expand_default_1 和内存格式参数 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 torch.aten.view.default 函数，并传入 clone_default 和一个未指定的参数，_users 设置为 2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，并传入关键字参数 'key' 和一个未指定的参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，调用 torch.aten.permute.default 函数，并传入 permute_default_1 和一个未指定的参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 torch.aten.expand.default 函数，并传入 permute_default_2 和一个未指定的参数
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 torch.aten.clone.default 函数，并传入 expand_default_2 和内存格式参数 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作函数，传入 clone_default_1 作为参数

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的默认批量矩阵乘法函数，传入 view_default 和 view_default_1 作为参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的默认视图操作函数，传入 bmm_default 作为参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 PyTorch 的张量除法函数，传入 view_default_2 和 'inv_scale' 作为参数

where_self = CallFunction(aten.where.self, expand_default, full_default, div_Tensor)
# 调用 PyTorch 的自身 where 函数，传入 expand_default、full_default 和 div_Tensor 作为参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
# 调用转换元素类型的默认函数，传入 where_self 作为参数

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的默认最大值函数，传入 convert_element_type_default 和 True 作为参数

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的张量减法函数，传入 convert_element_type_default 和 amax_default 作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的默认指数函数，传入 sub_Tensor 作为参数

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的按维度求和函数，传入 exp_default 和 True 作为参数

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的张量除法函数，传入 exp_default 和 sum_dim_IntList 作为参数

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 调用转换元素类型的默认函数，传入 div_Tensor_1 作为参数

expand_default_3 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的默认扩展函数，传入 convert_element_type_default_1 作为参数

view_default_3 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作函数，传入 expand_default_3 作为参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的默认排列函数，传入 'value' 作为参数

expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 PyTorch 的默认扩展函数，传入 permute_default_3 作为参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)
# 调用 PyTorch 的默认克隆函数，传入 expand_default_4 和内存格式为 torch.contiguous_format 作为参数

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作函数，传入 clone_default_2 作为参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的默认批量矩阵乘法函数，传入 view_default_3 和 view_default_4 作为参数

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的默认视图操作函数，传入 bmm_default_1 作为参数

scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())
# 调用 PyTorch 的默认标量张量函数，设置张量布局为 torch.strided

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 调用转换元素类型的默认函数，传入 convert_element_type_default_1 作为参数

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 调用 PyTorch 的默认负数函数，传入 convert_element_type_default_2 作为参数

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 PyTorch 的默认视图操作函数，传入 'tangents_1' 作为参数

permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 PyTorch 的默认排列函数，传入 view_default_4 作为参数

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
# 调用 PyTorch 的默认批量矩阵乘法函数，传入 view_default_6 和 permute_default_4 作为参数

view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 PyTorch 的默认视图操作函数，传入 bmm_default_2 作为参数

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
# 调用转换元素类型的默认函数，传入 view_default_7 作为参数

mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)
# 调用 PyTorch 的张量乘法函数，传入 convert_element_type_default_3 和 convert_element_type_default_2 作为参数

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
# 调用 PyTorch 的按维度求和函数，传入 mul_Tensor 和 True 作为参数

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
# 调用 fused multiply-add (FMA) 的默认函数，传入 neg_default、sum_dim_IntList_1 和 mul_Tensor 作为参数

convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用转换元素类型的默认函数，传入 fma_default 作为参数
# 创建一个名为 where_self_1 的对象，调用 CallFunction 函数，传入参数 aten.where.self, expand_default, scalar_tensor_default, convert_element_type_default_4
where_self_1 = CallFunction(aten.where.self, expand_default, scalar_tensor_default, convert_element_type_default_4)

# 创建一个名为 div_Tensor_2 的对象，调用 CallFunction 函数，传入参数 aten.div.Tensor, where_self_1，并设置关键字参数 'inv_scale'
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, KeywordArg('inv_scale'))

# 创建一个名为 view_default_8 的对象，调用 CallFunction 函数，传入参数 aten.view.default, div_Tensor_2，并忽略第三个参数
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个名为 permute_default_5 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, view_default_1，并忽略其他参数
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个名为 bmm_default_3 的对象，调用 CallFunction 函数，传入参数 aten.bmm.default, view_default_8, permute_default_5
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个名为 view_default_9 的对象，调用 CallFunction 函数，传入参数 aten.view.default, bmm_default_3，并忽略第三个参数
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个名为 permute_default_6 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, view_default_9，并忽略其他参数
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个名为 permute_default_7 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, view_default，并忽略其他参数
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个名为 bmm_default_4 的对象，调用 CallFunction 函数，传入参数 aten.bmm.default, permute_default_7, view_default_8
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个名为 view_default_10 的对象，调用 CallFunction 函数，传入参数 aten.view.default, bmm_default_4，并忽略第三个参数
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个名为 permute_default_8 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, view_default_10，并忽略其他参数
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个名为 permute_default_9 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, permute_default_8，并忽略其他参数
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个名为 permute_default_10 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, view_default_3，并忽略其他参数
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个名为 bmm_default_5 的对象，调用 CallFunction 函数，传入参数 aten.bmm.default, permute_default_10, view_default_6
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个名为 view_default_11 的对象，调用 CallFunction 函数，传入参数 aten.view.default, bmm_default_5，并忽略第三个参数
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个名为 permute_default_11 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, view_default_11，并忽略其他参数
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个名为 _sfdp_pattern_15_half_training 的对象，调用 MultiOutputPattern 函数，传入一个包含 view_default_5, permute_default_6, permute_default_9, permute_default_11 的列表，和两个 None 值
_sfdp_pattern_15_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])

# 创建一个名为 eq_Scalar 的对象，调用 CallFunction 函数，传入参数 aten.eq.Scalar，并设置关键字参数 'attn_mask'，并忽略其他参数
eq_Scalar = CallFunction(aten.eq.Scalar, KeywordArg('attn_mask'), Ignored())

# 创建一个名为 view_default 的对象，调用 CallFunction 函数，传入参数 aten.view.default, eq_Scalar，并忽略第三个参数
view_default = CallFunction(aten.view.default, eq_Scalar, Ignored())

# 创建一个名为 expand_default 的对象，调用 CallFunction 函数，传入参数 aten.expand.default, view_default，并忽略第三个参数
expand_default = CallFunction(aten.expand.default, view_default, Ignored())

# 创建一个名为 full_default 的对象，调用 CallFunction 函数，传入参数 aten.full.default, []，并设置 dtype, device, pin_memory=False 的值
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 创建一个名为 permute_default 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, 并设置关键字参数 'query'，并忽略其他参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个名为 expand_default_1 的对象，调用 CallFunction 函数，传入参数 aten.expand.default, permute_default，并忽略第三个参数
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个名为 clone_default 的对象，调用 CallFunction 函数，传入参数 aten.clone.default, expand_default_1，并设置 memory_format=torch.contiguous_format 的值
clone_default = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个名为 view_default_1 的对象，调用 CallFunction 函数，传入参数 aten.view.default, clone_default，并忽略第三个参数
view_default_1 = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个名为 permute_default_1 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, 并设置关键字参数 'key'，并忽略其他参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个名为 permute_default_2 的对象，调用 CallFunction 函数，传入参数 aten.permute.default, permute_default_1，并忽略其他参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个名为 expand_default_2 的对象，调用 CallFunction 函数，传入参数 aten.expand.default, permute_default_2，并忽略第三个参数
expand_default_2 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个名为 clone_default_1 的对象，调用 CallFunction 函数，传入参数 aten.clone.default, expand_default_2，并设置 memory_format=torch.contiguous_format 的值
clone_default_1 = CallFunction(aten.clone.default, expand_default_2, memory_format=torch.contiguous_format)

# 创建一个名为 view_default_2 的对象，调用 CallFunction 函数，传入参数 aten.view.default, clone_default_1，并忽略第三个参数
view_default_2 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个名为 bmm_default 的对象，调用 CallFunction 函数，传入参数 aten.bmm.default, view_default_1, view_default_2
bmm_default = CallFunction(aten.bmm.default, view_default_1, view_default_2)

# 创建一个名为 view_default_3 的对象，调用 CallFunction 函数，传入参数 aten.view.default, bmm_default，并忽略第三个参数
# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，初始参数为 where_self 和 Ignored()，设置 _users 为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)

# 创建一个 CallFunction 对象，调用 aten.amax.default 函数，初始参数为 convert_element_type_default 和 Ignored()，设置 True 参数
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.sub.Tensor 函数，初始参数为 convert_element_type_default 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个 CallFunction 对象，调用 aten.exp.default 函数，初始参数为 sub_Tensor，设置 _users 为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个 CallFunction 对象，调用 aten.sum.dim_IntList 函数，初始参数为 exp_default 和 Ignored()，设置 True 参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个 CallFunction 对象，调用 aten.div.Tensor 函数，初始参数为 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个 CallFunction 对象，调用 prims.convert_element_type.default 函数，初始参数为 div_Tensor_1 和 Ignored()
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，初始参数为 convert_element_type_default_1 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，初始参数为 expand_default_3 和 Ignored()
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.permute.default 函数，初始参数为 KeywordArg('value') 和 Ignored()
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个 CallFunction 对象，调用 aten.expand.default 函数，初始参数为 permute_default_3 和 Ignored()
expand_default_4 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个 CallFunction 对象，调用 aten.clone.default 函数，初始参数为 expand_default_4 和 memory_format=torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_4, memory_format=torch.contiguous_format)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，初始参数为 clone_default_2 和 Ignored()
view_default_5 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个 CallFunction 对象，调用 aten.bmm.default 函数，初始参数为 view_default_4 和 view_default_5
bmm_default_1 = CallFunction(aten.bmm.default, view_default_4, view_default_5)

# 创建一个 CallFunction 对象，调用 aten.view.default 函数，初始参数为 bmm_default_1 和 Ignored()，设置 _users 为 0
_sfdp_pattern_15_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```