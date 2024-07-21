# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_14.py`

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

# 设置 torch.ops.aten 别名为 aten
aten = torch.ops.aten
# 设置 torch.ops.prims 别名为 prims
prims = torch.ops.prims

# 导入 torch._inductor.pattern_matcher 中的多个类
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

# 创建 permute_default 变量，调用 torch.ops.aten.permute.default 函数，设置关键字参数 'query'，忽略其余参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 创建 expand_default 变量，调用 torch.ops.aten.expand.default 函数，使用 permute_default 结果，忽略其余参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建 clone_default 变量，调用 torch.ops.aten.clone.default 函数，使用 expand_default 结果，设置 memory_format 为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 创建 view_default 变量，调用 torch.ops.aten.view.default 函数，使用 clone_default 结果，忽略一个参数，并设置 _users 为 2
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)

# 类似地，创建多个变量，每个变量调用不同的 torch.ops.aten 函数，依次进行参数设置和变量赋值

# 最后一行的调用例子
# 创建 view_default_6 变量，调用 torch.ops.aten.view.default 函数，设置关键字参数 'tangents_1'，忽略其余参数，并设置 _users 为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 创建一个调用函数对象，用于执行 aten.permute.default 函数，输入参数是 KeywordArg('query') 和一个被忽略的参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，用于执行 aten.expand.default 函数，输入参数是上一个函数调用的结果和一个被忽略的参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，用于执行 aten.clone.default 函数，输入参数是上一个函数调用的结果和内存格式为 torch.contiguous_format
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，用于执行 aten.view.default 函数，输入参数是上一个函数调用的结果和一个被忽略的参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个调用函数对象，用于执行 aten.permute.default 函数，输入参数是 KeywordArg('key') 和一个被忽略的参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，用于执行 aten.permute.default 函数，输入参数是上一个函数调用的结果和一个被忽略的参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，用于执行 aten.expand.default 函数，输入参数是上一个函数调用的结果和一个被忽略的参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，用于执行 aten.clone.default 函数，输入参数是上一个函数调用的结果和内存格式为 torch.contiguous_format
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，用于执行 aten.view.default 函数，输入参数是上一个函数调用的结果和一个被忽略的参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个调用函数对象，用于执行 aten.bmm.default 函数，输入参数是两个上一个函数调用的结果
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，用于执行 aten.view.default 函数，输入参数是上一个函数调用的结果和一个被忽略的参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，用于执行 aten.div.Tensor 函数，输入参数是上一个函数调用的结果、KeywordArg('inv_scale') 和一个被忽略的参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，用于执行 aten.add.Tensor 函数，输入参数是上一个函数调用的结果、KeywordArg('attn_mask') 和一个被忽略的参数
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 调用 PyTorch 的 aten.amax.default 函数，使用 add_Tensor 作为参数，并忽略其他两个参数，设置最后一个参数为 True

sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 调用 PyTorch 的 aten.sub.Tensor 函数，使用 add_Tensor 和 amax_default 作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的 aten.exp.default 函数，使用 sub_Tensor 作为参数，并指定 _users 参数为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的 aten.sum.dim_IntList 函数，使用 exp_default 作为参数，并忽略第三个参数，设置最后一个参数为 True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的 aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为参数

expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，使用 div_Tensor_1 作为参数，并忽略第二个参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，使用 expand_default_2 作为参数，并忽略第二个参数

permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，使用 'value' 作为关键字参数，并忽略第二个参数

expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，使用 permute_default_3 作为参数，并忽略第二个参数

clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 aten.clone.default 函数，使用 expand_default_3 作为参数，并设置 memory_format 参数为 torch.contiguous_format

view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，使用 clone_default_2 作为参数，并忽略第二个参数

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的 aten.bmm.default 函数，使用 view_default_3 和 view_default_4 作为参数

_sfdp_pattern_14_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 PyTorch 的 aten.view.default 函数，使用 bmm_default_1 作为参数，并忽略第三个参数，设置 _users 参数为 0

permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，使用 'query' 作为关键字参数，并忽略第二个参数

expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，使用 permute_default 作为参数，并忽略第二个参数

clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 aten.clone.default 函数，使用 expand_default 作为参数，并设置 memory_format 参数为 torch.contiguous_format

view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
# 调用 PyTorch 的 aten.view.default 函数，使用 clone_default 作为参数，并忽略第三个参数，设置 _users 参数为 2

permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，使用 'key' 作为关键字参数，并忽略第二个参数

permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
# 调用 PyTorch 的 aten.permute.default 函数，使用 permute_default_1 作为参数，并忽略第二个参数

expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，使用 permute_default_2 作为参数，并忽略第二个参数

clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
# 调用 PyTorch 的 aten.clone.default 函数，使用 expand_default_1 作为参数，并设置 memory_format 参数为 torch.contiguous_format

view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
# 调用 PyTorch 的 aten.view.default 函数，使用 clone_default_1 作为参数，并忽略第三个参数，设置 _users 参数为 2

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的 aten.bmm.default 函数，使用 view_default 和 view_default_1 作为参数

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的 aten.view.default 函数，使用 bmm_default 作为参数，并忽略第二个参数

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
# 调用 PyTorch 的 aten.div.Tensor 函数，使用 view_default_2 作为参数，并使用 'inv_scale' 作为关键字参数

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))
# 调用 PyTorch 的 aten.add.Tensor 函数，使用 div_Tensor 作为参数，并使用 'attn_mask' 作为关键字参数

convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)
# 调用 PyTorch 的 prims.convert_element_type.default 函数，使用 add_Tensor 作为参数，并忽略第三个参数，设置 _users 参数为 2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的 aten.amax.default 函数，使用 convert_element_type_default 作为参数，并忽略第三个参数，设置最后一个参数为 True

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的 aten.sub.Tensor 函数，使用 convert_element_type_default 和 amax_default 作为参数

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的 aten.exp.default 函数，使用 sub_Tensor 作为参数，并指定 _users 参数为 2

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的 aten.sum.dim_IntList 函数，使用 exp_default 作为参数，并忽略第三个参数，设置最后一个参数为 True

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的 aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为参数

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 调用 PyTorch 的 prims.convert_element_type.default 函数，使用 div_Tensor_1 作为参数，并忽略第三个参数，设置 _users 参数为 2

expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
# 调用 PyTorch 的 aten.expand.default 函数，使用 convert_element_type_default_1 作为参数，并忽略第二个参数

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用 PyTorch 的 aten.view.default 函数，使用 expand_default_2 作为
# 创建一个调用aten.permute.default函数的对象，设置'value'关键字参数为Ignored()
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用aten.expand.default函数的对象，参数为上一步得到的permute_default_3对象，另一个参数为Ignored()
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用aten.clone.default函数的对象，参数为上一步得到的expand_default_3对象，另一个参数为内存格式torch.contiguous_format
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的clone_default_2对象和Ignored()，_users=2表示有两个用户使用这个对象
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)

# 创建一个调用aten.bmm.default函数的对象，参数为上一步得到的view_default_3和view_default_4对象
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的bmm_default_1对象和Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 创建一个调用prims.convert_element_type.default函数的对象，参数包括上一步得到的convert_element_type_default_1和Ignored()，_users=2表示有两个用户使用这个对象
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)

# 创建一个调用aten.neg.default函数的对象，参数为上一步得到的convert_element_type_default_2对象
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)

# 创建一个调用aten.view.default函数的对象，参数包括KeywordArg('tangents_1')和Ignored()，_users=2表示有两个用户使用这个对象
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 创建一个调用aten.permute.default函数的对象，参数为上一步得到的view_default_4和Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 创建一个调用aten.bmm.default函数的对象，参数为上一步得到的view_default_6和permute_default_4对象
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的bmm_default_2对象和Ignored()
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())

# 创建一个调用prims.convert_element_type.default函数的对象，参数包括上一步得到的view_default_7和Ignored()
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())

# 创建一个调用aten.mul.Tensor函数的对象，参数包括上一步得到的convert_element_type_default_3、convert_element_type_default_2和_users=2表示有两个用户使用这个对象
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)

# 创建一个调用aten.sum.dim_IntList函数的对象，参数包括上一步得到的mul_Tensor和Ignored()，True表示keepdim=True
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)

# 创建一个调用prims.fma.default函数的对象，参数包括上一步得到的neg_default、sum_dim_IntList_1和mul_Tensor
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)

# 创建一个调用prims.convert_element_type.default函数的对象，参数包括上一步得到的fma_default和Ignored()
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())

# 创建一个调用aten.div.Tensor函数的对象，参数包括上一步得到的convert_element_type_default_4和KeywordArg('inv_scale')
div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_4, KeywordArg('inv_scale'))

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的div_Tensor_2和Ignored()，_users=2表示有两个用户使用这个对象
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的view_default_1和Ignored()
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())

# 创建一个调用aten.bmm.default函数的对象，参数包括上一步得到的view_default_8和permute_default_5对象
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的bmm_default_3对象和Ignored()
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的view_default_9和Ignored()
permute_default_6 = CallFunction(aten.permute.default, view_default_9, Ignored())

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的view_default和Ignored()
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())

# 创建一个调用aten.bmm.default函数的对象，参数包括上一步得到的permute_default_7和view_default_8对象
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的bmm_default_4对象和Ignored()
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的view_default_10和Ignored()
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的permute_default_8和Ignored()
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的view_default_3和Ignored()
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())

# 创建一个调用aten.bmm.default函数的对象，参数包括上一步得到的permute_default_10和view_default_6对象
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)

# 创建一个调用aten.view.default函数的对象，参数包括上一步得到的bmm_default_5对象和Ignored()
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())

# 创建一个调用aten.permute.default函数的对象，参数包括上一步得到的view_default_11和Ignored()
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())

# 创建一个MultiOutputPattern对象，包括多个元素的列表，其中有5个有效元素，最后两个为None
_sfdp_pattern_14_half_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])
# 创建一个调用函数对象，调用 aten.permute.default 函数，并使用 'query' 作为关键字参数
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，并使用 permute_default 结果作为第一个参数，忽略第二个参数
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，使用 expand_default 结果作为第一个参数，并指定内存格式为连续格式
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，使用 clone_default 结果作为第一个参数，忽略第二个参数
view_default = CallFunction(aten.view.default, clone_default, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，并使用 'key' 作为关键字参数
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，使用 permute_default_1 结果作为第一个参数，忽略第二个参数
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，使用 permute_default_2 结果作为第一个参数，忽略第二个参数
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，使用 expand_default_1 结果作为第一个参数，并指定内存格式为连续格式
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，使用 clone_default_1 结果作为第一个参数，忽略第二个参数
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，使用 view_default 和 view_default_1 作为两个参数
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 创建一个调用函数对象，调用 aten.view.default 函数，使用 bmm_default 结果作为第一个参数，忽略第二个参数
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，使用 view_default_2 结果作为第一个参数，并指定 'inv_scale' 为关键字参数
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))

# 创建一个调用函数对象，调用 aten.add.Tensor 函数，使用 div_Tensor 结果作为第一个参数，并指定 'attn_mask' 为关键字参数
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，使用 add_Tensor 结果作为第一个参数，忽略第二个和第三个参数，标记用户数为 2
convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)

# 创建一个调用函数对象，调用 aten.amax.default 函数，使用 convert_element_type_default 结果作为第一个参数，忽略第二个和第三个参数，标记是否在原地修改为 True
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.sub.Tensor 函数，使用 convert_element_type_default 和 amax_default 作为两个参数
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)

# 创建一个调用函数对象，调用 aten.exp.default 函数，使用 sub_Tensor 结果作为第一个参数，标记用户数为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 aten.sum.dim_IntList 函数，使用 exp_default 结果作为第一个参数，忽略第二个和第三个参数，标记是否在原地修改为 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 aten.div.Tensor 函数，使用 exp_default 和 sum_dim_IntList 作为两个参数
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，使用 div_Tensor_1 结果作为第一个参数，忽略第二个参数
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，使用 convert_element_type_default_1 结果作为第一个参数，忽略第二个参数
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建一个调用函数对象，调用 aten.view.default 函数，使用 expand_default_2 结果作为第一个参数，忽略第二个参数
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.permute.default 函数，并使用 'value' 作为关键字参数
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 aten.expand.default 函数，使用 permute_default_3 结果作为第一个参数，忽略第二个参数
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())

# 创建一个调用函数对象，调用 aten.clone.default 函数，使用 expand_default_3 结果作为第一个参数，并指定内存格式为连续格式
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)

# 创建一个调用函数对象，调用 aten.view.default 函数，使用 clone_default_2 结果作为第一个参数，忽略第二个参数
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())

# 创建一个调用函数对象，调用 aten.bmm.default 函数，使用 view_default_3 和 view_default_4 作为两个参数
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 aten.view.default 函数，使用 bmm_default_1 结果作为第一个参数，忽略第二个参数，并标记用户数为 0
_sfdp_pattern_14_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```