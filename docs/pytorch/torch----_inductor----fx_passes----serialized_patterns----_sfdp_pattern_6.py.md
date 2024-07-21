# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\_sfdp_pattern_6.py`

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

# 定义 aten 和 prims 别名，分别引用 torch.ops.aten 和 torch.ops.prims
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

# 定义 rand_default 变量，调用 torch.ops.aten.rand.default 方法，指定参数 Ignored()，返回的 dtype 也是 Ignored()，设备是 Ignored()，pin_memory 设为 False
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)

# 定义 gt_Scalar 变量，调用 torch.ops.aten.gt.Scalar 方法，参数包括 rand_default，KeywordArg('dropout_p')，_users 设为 2
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)

# 定义 expand_default 变量，调用 torch.ops.aten.expand.default 方法，指定参数 KeywordArg('query') 和 Ignored()
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())

# 定义 view_default 变量，调用 torch.ops.aten.view.default 方法，指定参数 expand_default 和 Ignored()，_users 设为 2
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)

# 定义 permute_default 变量，调用 torch.ops.aten.permute.default 方法，指定参数 KeywordArg('key') 和 Ignored()
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())

# 定义 expand_default_1 变量，调用 torch.ops.aten.expand.default 方法，指定参数 permute_default 和 Ignored()
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())

# 定义 view_default_1 变量，调用 torch.ops.aten.view.default 方法，指定参数 expand_default_1 和 Ignored()，_users 设为 2
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)

# 定义 bmm_default 变量，调用 torch.ops.aten.bmm.default 方法，指定参数 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)

# 定义 view_default_2 变量，调用 torch.ops.aten.view.default 方法，指定参数 bmm_default 和 Ignored()
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())

# 定义 div_Tensor 变量，调用 torch.ops.aten.div.Tensor 方法，指定参数 view_default_2 和 Ignored()
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())

# 定义 add_Tensor 变量，调用 torch.ops.aten.add.Tensor 方法，指定参数 div_Tensor 和 KeywordArg('attn_mask')，_users 设为 2
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)

# 定义 amax_default 变量，调用 torch.ops.aten.amax.default 方法，指定参数 add_Tensor 和 Ignored()，True 表示使用 ignore_undefined
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)

# 定义 sub_Tensor 变量，调用 torch.ops.aten.sub.Tensor 方法，指定参数 add_Tensor 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)

# 定义 exp_default 变量，调用 torch.ops.aten.exp.default 方法，指定参数 sub_Tensor，_users 设为 2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 定义 sum_dim_IntList 变量，调用 torch.ops.aten.sum.dim_IntList 方法，指定参数 exp_default 和 Ignored()，True 表示使用 ignore_undefined
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 定义 div_Tensor_1 变量，调用 torch.ops.aten.div.Tensor 方法，指定参数 exp_default、sum_dim_IntList，_users 设为 3
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)

# 定义 mul_Tensor 变量，调用 torch.ops.aten.mul.Tensor 方法，指定参数 gt_Scalar 和 div_Tensor
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor)

# 定义 mul_Tensor_1 变量，调用 torch.ops.aten.mul.Tensor 方法，指定参数 mul_Tensor 和 Ignored()
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())

# 定义 expand_default_2 变量，调用 torch.ops.aten.expand.default 方法，指定参数 mul_Tensor_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())

# 定义 view_default_3 变量，调用 torch.ops.aten.view.default 方法，指定参数 expand_default_2 和 Ignored()，_users 设为 2
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)

# 定义 expand_default_3 变量，调用 torch.ops.aten.expand.default 方法，指定参数 KeywordArg('value') 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 定义 view_default_4 变量，调用 torch.ops.aten.view.default 方法，指定参数 expand_default_3 和 Ignored()，_users 设为 2
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)

# 定义 bmm_default_1 变量，调用 torch.ops.aten.bmm.default 方法，指定参数 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 定义 view_default_5 变量，调用 torch.ops.aten.view.default 方法，指定参数 bmm_default_1 和 Ignored()
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())

# 定义 neg_default 变量，调用 torch.ops.aten.neg.default 方法，指定参数 div_Tensor_1
neg_default = CallFunction(aten.neg.default, div_Tensor_1)

# 定义 view_default_6 变量，调用 torch.ops.aten.view.default 方法，指定参数 KeywordArg('tangents_1') 和 Ignored()，_users 设为 2
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)

# 定义 permute_default_1 变量，调用 torch.ops.aten.permute.default 方法，指定参数 view_default_4 和 Ignored()
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())

# 定义 bmm_default_2 变量，调用 torch.ops.aten.bmm.default 方法，指定参数 view_default_6 和 permute_default_1
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)

# 定义 view_default_7 变量，调用 torch.ops.aten.view.default 方法，指定参数 bmm_default_2 和 Ignored()
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 创建一个 CallFunction 对象，用于执行 prims.convert_element_type.default 函数
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.mul.Tensor 函数，参数是 convert_element_type_default 和 Ignored()
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.mul.Tensor 函数，参数是 view_default_7 和 mul_Tensor_2
mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 创建一个 CallFunction 对象，用于执行 aten.mul.Tensor 函数，参数是 mul_Tensor_3、div_Tensor_1 和 _users=2
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor_1, _users=2)
# 创建一个 CallFunction 对象，用于执行 aten.sum.dim_IntList 函数，参数是 mul_Tensor_4、Ignored() 和 True
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 创建一个 CallFunction 对象，用于执行 prims.fma.default 函数，参数是 neg_default、sum_dim_IntList_1 和 mul_Tensor_4
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 创建一个 CallFunction 对象，用于执行 aten.div.Tensor 函数，参数是 fma_default 和 Ignored()
div_Tensor_2 = CallFunction(aten.div.Tensor, fma_default, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 div_Tensor_2、Ignored() 和 _users=2
view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 创建一个 CallFunction 对象，用于执行 aten.permute.default 函数，参数是 view_default_1 和 Ignored()
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.bmm.default 函数，参数是 view_default_8 和 permute_default_2
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 bmm_default_3 和 Ignored()
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.permute.default 函数，参数是 view_default 和 Ignored()
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.bmm.default 函数，参数是 permute_default_3 和 view_default_8
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 bmm_default_4 和 Ignored()
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.permute.default 函数，参数是 view_default_10 和 Ignored()
permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.permute.default 函数，参数是 view_default_3 和 Ignored()
permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.bmm.default 函数，参数是 permute_default_5 和 view_default_6
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 bmm_default_5 和 Ignored()
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 创建一个 MultiOutputPattern 对象，包含 view_default_5、view_default_9、permute_default_4、view_default_11、None 和 None
_sfdp_pattern_6_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None,
  None
])

# 创建一个 CallFunction 对象，用于执行 aten.expand.default 函数，参数是 KeywordArg('query') 和 Ignored()
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 expand_default 和 Ignored()
view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.permute.default 函数，参数是 KeywordArg('key') 和 Ignored()
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.expand.default 函数，参数是 permute_default 和 Ignored()
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 expand_default_1 和 Ignored()
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.bmm.default 函数，参数是 view_default 和 view_default_1
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 bmm_default 和 Ignored()
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.div.Tensor 函数，参数是 view_default_2 和 Ignored()
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.add.Tensor 函数，参数是 div_Tensor、KeywordArg('attn_mask') 和 _users=2
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
# 创建一个 CallFunction 对象，用于执行 aten.amax.default 函数，参数是 add_Tensor、Ignored() 和 True
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
# 创建一个 CallFunction 对象，用于执行 aten.sub.Tensor 函数，参数是 add_Tensor 和 amax_default
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
# 创建一个 CallFunction 对象，用于执行 aten.exp.default 函数，参数是 sub_Tensor 和 _users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 创建一个 CallFunction 对象，用于执行 aten.sum.dim_IntList 函数，参数是 exp_default、Ignored() 和 True
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 创建一个 CallFunction 对象，用于执行 aten.div.Tensor 函数，参数是 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 创建一个 CallFunction 对象，用于执行 aten.expand.default 函数，参数是 div_Tensor_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.view.default 函数，参数是 expand_default_2 和 Ignored()
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
# 创建一个 CallFunction 对象，用于执行 aten.expand.default 函数，参数是 KeywordArg('value') 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
# 调用 PyTorch 的 view 函数，对 expand_default_3 进行默认视图变换

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的 bmm 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法

_sfdp_pattern_6_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
# 调用 PyTorch 的 view 函数，对 bmm_default_1 进行默认视图变换，此处 _users=0 表示未指定用户数

rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
# 调用 PyTorch 的 rand 函数，生成随机张量，默认忽略参数 dtype、device，并且 pin_memory=False

gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
# 调用 PyTorch 的 gt 函数，比较 rand_default 和 KeywordArg('dropout_p') 是否大于，_users=2 表示有两个用户

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 调用 PyTorch 的 expand 函数，扩展 KeywordArg('query') 的维度

view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，对 expand_default 进行默认视图变换，_users=2 表示有两个用户

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 调用 PyTorch 的 permute 函数，对 KeywordArg('key') 进行维度重排列

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 调用 PyTorch 的 expand 函数，扩展 permute_default 的维度

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，对 expand_default_1 进行默认视图变换，_users=2 表示有两个用户

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 调用 PyTorch 的 bmm 函数，对 view_default 和 view_default_1 进行批量矩阵乘法

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default 进行默认视图变换

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 调用 PyTorch 的 div 函数，对 view_default_2 进行张量除法运算

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))
# 调用 PyTorch 的 add 函数，对 div_Tensor 和 KeywordArg('attn_mask') 进行张量加法运算

convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，进行元素类型转换，默认忽略参数，并且有两个用户

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 调用 PyTorch 的 amax 函数，计算 convert_element_type_default 的最大值，最后一个参数 True 表示忽略未使用的参数

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 调用 PyTorch 的 sub 函数，对 convert_element_type_default 和 amax_default 进行张量减法运算

exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
# 调用 PyTorch 的 exp 函数，对 sub_Tensor 进行指数运算，_users=2 表示有两个用户

sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
# 调用 PyTorch 的 sum 函数，对 exp_default 进行按维度求和运算，True 表示保持输入张量的维度

div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
# 调用 PyTorch 的 div 函数，对 exp_default 和 sum_dim_IntList 进行张量除法运算

convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，进行元素类型转换，默认忽略参数，并且有两个用户

mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
# 调用 PyTorch 的 mul 函数，对 gt_Scalar 和 convert_element_type_default_1 进行张量乘法运算

mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored())
# 调用 PyTorch 的 mul 函数，对 mul_Tensor 和 Ignored() 进行张量乘法运算

expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
# 调用 PyTorch 的 expand 函数，扩展 mul_Tensor_1 的维度

view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，对 expand_default_2 进行默认视图变换，_users=2 表示有两个用户

expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
# 调用 PyTorch 的 expand 函数，扩展 KeywordArg('value') 的维度

view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，对 expand_default_3 进行默认视图变换，_users=2 表示有两个用户

bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
# 调用 PyTorch 的 bmm 函数，对 view_default_3 和 view_default_4 进行批量矩阵乘法

view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_1 进行默认视图变换

convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
# 调用 PyTorch 的 convert_element_type 函数，进行元素类型转换，默认忽略参数，并且有两个用户

neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
# 调用 PyTorch 的 neg 函数，对 convert_element_type_default_2 进行取负运算

view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
# 调用 PyTorch 的 view 函数，对 KeywordArg('tangents_1') 进行默认视图变换，_users=2 表示有两个用户

permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
# 调用 PyTorch 的 permute 函数，对 view_default_4 进行维度重排列

bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
# 调用 PyTorch 的 bmm 函数，对 view_default_6 和 permute_default_1 进行批量矩阵乘法
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
# 调用 PyTorch 的 view 函数，对 bmm_default_2 进行视图变换

convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
# 调用一个转换元素类型的函数，将 gt_Scalar 转换为默认类型

mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
# 使用 PyTorch 的乘法函数对 convert_element_type_default_3 进行张量相乘

mul_Tensor_3 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor_2)
# 使用 PyTorch 的乘法函数对 view_default_7 和 mul_Tensor_2 进行张量相乘

convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
# 再次调用转换元素类型的函数，将 mul_Tensor_3 转换为默认类型

mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
# 使用 PyTorch 的乘法函数对 convert_element_type_default_4 和 convert_element_type_default_2 进行张量相乘，并指定用户数为2

sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
# 使用 PyTorch 的求和函数对 mul_Tensor_4 沿着指定的维度列表进行求和，并保留维度

fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
# 调用 fused multiply-add (FMA) 函数，执行 neg_default 与 sum_dim_IntList_1 的 FMA 操作，并加上 mul_Tensor_4

convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
# 调用转换元素类型的函数，将 fma_default 转换为默认类型

div_Tensor_2 = CallFunction(aten.div.Tensor, convert_element_type_default_5, Ignored())
# 使用 PyTorch 的除法函数对 convert_element_type_default_5 进行张量除法

view_default_8 = CallFunction(aten.view.default, div_Tensor_2, Ignored(), _users=2)
# 使用 PyTorch 的 view 函数对 div_Tensor_2 进行视图变换，并指定用户数为2

permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
# 使用 PyTorch 的 permute 函数对 view_default_1 进行维度重排列

bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
# 使用 PyTorch 的批量矩阵乘法函数对 view_default_8 和 permute_default_2 进行矩阵乘法操作

view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
# 使用 PyTorch 的 view 函数对 bmm_default_3 进行视图变换

permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
# 使用 PyTorch 的 permute 函数对 view_default 进行维度重排列

bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
# 使用 PyTorch 的批量矩阵乘法函数对 permute_default_3 和 view_default_8 进行矩阵乘法操作

view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
# 使用 PyTorch 的 view 函数对 bmm_default_4 进行视图变换

permute_default_4 = CallFunction(aten.permute.default, view_default_10, Ignored())
# 使用 PyTorch 的 permute 函数对 view_default_10 进行维度重排列

permute_default_5 = CallFunction(aten.permute.default, view_default_3, Ignored())
# 使用 PyTorch 的 permute 函数对 view_default_3 进行维度重排列

bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
# 使用 PyTorch 的批量矩阵乘法函数对 permute_default_5 和 view_default_6 进行矩阵乘法操作

view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
# 使用 PyTorch 的 view 函数对 bmm_default_5 进行视图变换

_sfdp_pattern_6_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  permute_default_4,
  view_default_11,
  None,
  None
])
# 创建一个多输出模式，包含 view_default_5、view_default_9、permute_default_4、view_default_11 这四个元素，并包含两个空值

expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
# 使用 PyTorch 的 expand 函数，对 'query' 进行扩展操作

view_default = CallFunction(aten.view.default, expand_default, Ignored())
# 使用 PyTorch 的 view 函数对 expand_default 进行视图变换

permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
# 使用 PyTorch 的 permute 函数，对 'key' 进行维度重排列

expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
# 使用 PyTorch 的 expand 函数，对 permute_default 进行扩展操作

view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
# 使用 PyTorch 的 view 函数对 expand_default_1 进行视图变换

bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
# 使用 PyTorch 的批量矩阵乘法函数对 view_default 和 view_default_1 进行矩阵乘法操作

view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
# 使用 PyTorch 的 view 函数对 bmm_default 进行视图变换

div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
# 使用 PyTorch 的除法函数对 view_default_2 进行张量除法

add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'))
# 使用 PyTorch 的加法函数对 div_Tensor 和 'attn_mask' 进行张量加法

convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=2)
# 调用转换元素类型的函数，将 add_Tensor 转换为默认类型，并指定用户数为2

amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
# 使用 PyTorch 的最大值函数对 convert_element_type_default 进行求最大值，并保留维度

sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
# 使用 PyTorch 的减法函数对 convert_element_type_default 和 amax_default 进行张量减法
# 创建一个调用函数对象，调用 torch 的 exp.default 函数，并指定参数 sub_Tensor，_users=2
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)

# 创建一个调用函数对象，调用 torch 的 sum.dim_IntList 函数，参数为 exp_default 和 Ignored()，True 表示忽略第三个参数
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)

# 创建一个调用函数对象，调用 torch 的 div.Tensor 函数，参数为 exp_default 和 sum_dim_IntList
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)

# 创建一个调用函数对象，调用 prims.convert_element_type.default 函数，参数为 div_Tensor_1 和 Ignored()
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())

# 创建一个调用函数对象，调用 torch 的 expand.default 函数，参数为 convert_element_type_default_1 和 Ignored()
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())

# 创建一个调用函数对象，调用 torch 的 view.default 函数，参数为 expand_default_2 和 Ignored()
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())

# 创建一个调用函数对象，调用 torch 的 expand.default 函数，参数为 KeywordArg('value') 和 Ignored()
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())

# 创建一个调用函数对象，调用 torch 的 view.default 函数，参数为 expand_default_3 和 Ignored()
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())

# 创建一个调用函数对象，调用 torch 的 bmm.default 函数，参数为 view_default_3 和 view_default_4
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)

# 创建一个调用函数对象，调用 torch 的 view.default 函数，参数为 bmm_default_1 和 Ignored()，_users=0 表示忽略用户数目
_sfdp_pattern_6_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
```