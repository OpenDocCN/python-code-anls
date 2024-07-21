# `.\pytorch\tools\autograd\gen_variable_type.py`

```py
# 生成 VariableType.h/cpp 文件
#
# **如果对 VariableType 代码生成做了任何更改，请同时检查是否需要更新 torch/csrc/autograd/autograd_not_implemented_fallback.cpp
#
# VariableType 是 at::Type 的子类，提供绑定代码，用于提供 ATen 操作符的可微版本。这里有几种可能的情况：
#
#   - 对于非可微的前向实现，可以直接关联其反向实现以使其可微。这是常见情况。
#
#   - 有些函数不需要反向实现，因为反向传播永远不会传播到它们。这可能有几个原因：
#       - 函数没有可微的输入
#       - 函数的输出不可微
#       - 函数在其输入上没有数据依赖性
#
#   - 有些函数不需要反向实现，因为它们是由其他（可微的）ATen 函数组合而成的。这些直接分派到 Type 超类，后者将再次分派到 VariableType 进行其可微的子组件。
#

from __future__ import annotations

import re
from typing import Callable, Sequence

from torchgen.api import cpp  # 导入 cpp 模块
from torchgen.api.autograd import (  # 导入 autograd 相关模块和函数
    DifferentiableInput,
    dispatch_strategy,
    ForwardDerivative,
    gen_differentiable_outputs,
    is_differentiable,
    NativeFunctionWithDifferentiabilityInfo,
    SavedAttribute,
)
from torchgen.api.types import (  # 导入类型定义
    ArrayRefCType,
    BaseCppType,
    BaseCType,
    Binding,
    DispatcherSignature,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    MutRefCType,
    OptionalCType,
    scalarT,
    SpecialArgName,
    stringT,
    symIntArrayRefT,
    TENSOR_LIST_LIKE_CTYPES,
    tensorListT,
    tensorT,
    TupleCType,
    VectorCType,
)
from torchgen.code_template import CodeTemplate  # 导入代码模板
from torchgen.context import (  # 导入上下文管理器
    native_function_manager,
    with_native_function,
    with_native_function_and,
)
from torchgen.model import (  # 导入模型相关定义
    Argument,
    BaseType,
    ListType,
    NativeFunction,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
)
from torchgen.utils import FileManager, mapMaybe  # 导入文件管理器和 mapMaybe 函数

from .context import with_native_function_with_differentiability_info_and_key  # 导入特定上下文管理器
from .gen_inplace_or_view_type import (  # 导入处理就地操作或视图类型生成相关函数和常量
    ALL_VIEW_FUNCTIONS,
    ASSIGN_RETURN_VALUE,
    AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION,
    gen_formals,
    get_base_name,
    get_view_info,
    is_tensor_list_type,
    is_tensor_type,
    METHOD_DEFINITION,
    modifies_arguments,
    TMP_VAR,
    unpack_args,
    unpacked_name,
    use_derived,
    WRAPPER_REGISTRATION,
)
from .gen_trace_type import (  # 导入追踪类型生成相关函数和常量
    get_return_value,
    MANUAL_AUTOGRAD_AND_TRACER,
    MANUAL_BACKEND,
    tie_return_values,
    type_wrapper_name,
)
# 不要在这些方法上设置或修改 grad_fn。通常它们返回 requires_grad=False 的张量。
# 这些原地函数不会检查或修改 requires_grad 或 grad_fn。
# 注意：这些函数名称不包括重载的名称。

# 不需要导数的函数集合，这些函数不会记录梯度信息。
DONT_REQUIRE_DERIVATIVE = {
    # 这些函数仅依赖于输入张量的形状和设备，而不依赖于数据
    "empty_like",
    "ones_like",
    "full_like",
    "zeros_like",
    "rand_like",
    "randn_like",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_zeros",
    "new_ones",
    # 这些函数仅在整数类型上实现
    "__and__",
    "__iand__",
    "__ilshift__",
    "__ior__",
    "__irshift__",
    "__ixor__",
    "__lshift__",
    "__or__",
    "__rshift__",
    "__xor__",
    # 这些函数仅适用于整数数据类型，因此不需要梯度
    "_sobol_engine_draw",
    "_sobol_engine_ff",
    "_sobol_engine_scramble_",
    "_sobol_engine_initialize_state_",
    # 这是一个不安全的方法，意图是不受 autograd 影响
    "_coalesced_",
    # 量化函数不应记录梯度
    "quantize_per_tensor",
    "quantize_per_channel",
    # 返回整数的函数不应具有需要梯度的输出
    "argmax",
    "argmin",
    "argsort",
    "searchsorted",
    "bucketize",
    # 返回布尔值的函数不可微分
    "isnan",
    "isposinf",
    "isneginf",
    "isinf",
    "signbit",
    "isin",
    "allclose",
    # 返回 None 的函数不可微分
    "record_stream",
    # 这些函数不可微分
    "logical_and",
    "logical_xor",
    "logical_not",
    "logical_or",
    # 此函数将嵌套张量的形状作为张量返回，不可微分
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
}

# 在添加时，这些函数对复数到实数的梯度处理尚在审核和测试中，但不会出错。
# 对于 C -> C、R -> C 的函数，后向传播已经正确实现并测试通过。
GRADIENT_IMPLEMENTED_FOR_COMPLEX = {
    "fill",
    "t",
    "view",
    "reshape",
    "reshape_as",
    "view_as",
    "roll",
    "clone",
    "block_diag",
    "diag_embed",
    "repeat",
    "expand",
    "flip",
    "fliplr",
    "flipud",
    "rot90",
    "nanmean",
    "nansum",
    "transpose",
    "permute",
    "squeeze",
    "unsqueeze",
    "resize",
    "resize_as",
    "tril",
    "triu",
    "chunk",
    "zero_",
    "eq_",
    "ne_",
    "add",
    "__radd__",
    "sum",
    "_conj",
    "sin",
    "cos",
    "mul",
    "sinc",
    "sinh",
    "cosh",
    "__rmul__",
    "sgn",
    "asin",
    "acos",
    "sub",
    "div",
    "cat",
    "view_as_complex",
    "index_put",
    "neg",
    "complex",
    "select",
    "where",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "slice",
    "constant_pad_nd",
    "unbind",
    "split",
    "split_with_sizes",
    "unsafe_split",
}
    # 函数名："split_with_sizes_backward"
    # 功能：反向按大小拆分
    "split_with_sizes_backward",
    # 函数名："dot"
    # 功能：计算两个数组的点积
    "dot",
    # 函数名："vdot"
    # 功能：计算两个向量的点积
    "vdot",
    # 函数名："cholesky"
    # 功能：计算 Cholesky 分解
    "cholesky",
    # 函数名："triangular_solve"
    # 功能：解一个三角系统
    "triangular_solve",
    # 函数名："mm"
    # 功能：矩阵乘法
    "mm",
    # 函数名："_unsafe_view"
    # 功能：不安全的视图操作
    "_unsafe_view",
    # 函数名："mv"
    # 功能：矩阵与向量乘法
    "mv",
    # 函数名："outer"
    # 功能：计算外积
    "outer",
    # 函数名："bmm"
    # 功能：批量矩阵乘法
    "bmm",
    # 函数名："diagonal"
    # 功能：获取矩阵的对角线元素
    "diagonal",
    # 函数名："alias"
    # 功能：创建别名
    "alias",
    # 函数名："atan"
    # 功能：计算反正切
    "atan",
    # 函数名："log"
    # 功能：计算自然对数
    "log",
    # 函数名："log10"
    # 功能：计算以10为底的对数
    "log10",
    # 函数名："log1p"
    # 功能：计算 log(1 + x)
    "log1p",
    # 函数名："log2"
    # 功能：计算以2为底的对数
    "log2",
    # 函数名："logaddexp"
    # 功能：计算 log(exp(x) + exp(y))
    "logaddexp",
    # 函数名："logcumsumexp"
    # 功能：计算 log(cumsum(exp(x)))
    "logcumsumexp",
    # 函数名："reciprocal"
    # 功能：计算倒数
    "reciprocal",
    # 函数名："tan"
    # 功能：计算正切
    "tan",
    # 函数名："pow"
    # 功能：计算幂次方
    "pow",
    # 函数名："rsqrt"
    # 功能：计算平方根的倒数
    "rsqrt",
    # 函数名："tanh"
    # 功能：计算双曲正切
    "tanh",
    # 函数名："tanh_backward"
    # 功能：双曲正切的反向传播
    "tanh_backward",
    # 函数名："asinh"
    # 功能：计算反双曲正弦
    "asinh",
    # 函数名："acosh"
    # 功能：计算反双曲余弦
    "acosh",
    # 函数名："atanh"
    # 功能：计算反双曲正切
    "atanh",
    # 函数名："take"
    # 功能：按索引从数组中取值
    "take",
    # 函数名："fill_"
    # 功能：填充数组元素
    "fill_",
    # 函数名："exp"
    # 功能：计算指数
    "exp",
    # 函数名："exp2"
    # 功能：计算2的幂次方
    "exp2",
    # 函数名："expm1"
    # 功能：计算 exp(x) - 1
    "expm1",
    # 函数名："nonzero"
    # 功能：返回非零元素的索引
    "nonzero",
    # 函数名："mean"
    # 功能：计算平均值
    "mean",
    # 函数名："std_mean"
    # 功能：计算标准差的平均值
    "std_mean",
    # 函数名："var_mean"
    # 功能：计算方差的平均值
    "var_mean",
    # 函数名："inverse"
    # 功能：计算逆矩阵
    "inverse",
    # 函数名："solve"
    # 功能：求解线性系统
    "solve",
    # 函数名："linalg_cholesky"
    # 功能：线性代数中的 Cholesky 分解
    "linalg_cholesky",
    # 函数名："addcmul"
    # 功能：按元素相乘，然后加到第一个张量上
    "addcmul",
    # 函数名："addcdiv"
    # 功能：按元素相除，然后加到第一个张量上
    "addcdiv",
    # 函数名："matrix_exp"
    # 功能：矩阵指数运算
    "matrix_exp",
    # 函数名："linalg_matrix_exp"
    # 功能：线性代数中的矩阵指数运算
    "linalg_matrix_exp",
    # 函数名："_linalg_eigh"
    # 功能：不安全的特征值分解
    "_linalg_eigh",
    # 函数名："cholesky_solve"
    # 功能：Cholesky 解法
    "cholesky_solve",
    # 函数名："linalg_qr"
    # 功能：线性代数中的 QR 分解
    "linalg_qr",
    # 函数名："_linalg_svd"
    # 功能：不安全的奇异值分解
    "_linalg_svd",
    # 函数名："_fft_c2c"
    # 功能：复数到复数的快速傅里叶变换
    "_fft_c2c",
    # 函数名："_fft_r2c"
    # 功能：实数到复数的快速傅里叶变换
    "_fft_r2c",
    # 函数名："linalg_solve"
    # 功能：线性代数中的解法
    "linalg_solve",
    # 函数名："sqrt"
    # 功能：计算平方根
    "sqrt",
    # 函数名："stack"
    # 功能：堆叠张量
    "stack",
    # 函数名："gather"
    # 功能：根据索引收集张量
    "gather",
    # 函数名："index_select"
    # 功能：根据索引选择张量
    "index_select",
    # 函数名："index_add_"
    # 功能：按索引加到张量上
    "index_add_",
    # 函数名："linalg_inv"
    # 功能：线性代数中的逆运算
    "linalg_inv",
    # 函数名："linalg_inv_ex"
    # 功能：扩展的逆运算
    "linalg_inv_ex",
    # 函数名："baddbmm"
    # 功能：批量矩阵加权乘法
    "baddbmm",
    # 函数名："addbmm"
    # 功能：矩阵加权乘法
    "addbmm",
    # 函数名："addmm"
    # 功能：矩阵加权乘法
    "addmm",
    # 函数名："addmv"
    # 功能：向量加权乘法
    "addmv",
    # 函数名："addr"
    # 功能：向量外积加法
    "addr",
    # 函数名："linalg_householder_product"
    # 功能：Householder 变换乘积
    "linalg_householder_product",
    # 函数名："ormqr"
    # 功能：计算 Q*R
    "ormqr",
    # 函数名："reflection_pad1d"
    # 功能：一维反射填充
    "reflection_pad1d",
    # 函数名："reflection_pad2d"
    # 功能：二维反射填充
    "reflection_pad2d",
    # 函数名："reflection_pad3d"
    # 功能：三维反射填充
    "reflection_pad3d",
    # 函数名："linalg_cholesky_ex"
    # 功能：扩展
}

# 定义一个集合，包含了稀疏复杂对象的梯度实现
GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX = {
    "_to_dense",
    "_coalesce",
    "coalesce",
    "values",
    "_sparse_coo_tensor_with_dims_and_tensors",
    "_sparse_addmm",
}

# 将稀疏复杂对象的梯度实现更新到 GRADIENT_IMPLEMENTED_FOR_COMPLEX 中
GRADIENT_IMPLEMENTED_FOR_COMPLEX.update(GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX)

# 一些操作会使梯度累加器失效，因此需要重置它们
RESET_GRAD_ACCUMULATOR = {"set_", "resize_"}

# NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
#
# 我们检查以下属性：
#   1) 即使函数修改其输入张量（通过原地操作或者输出变体），也不应更改其底层 c10::TensorImpl 指针或 c10::Storage 指针
# 如果函数不修改其参数，还会检查以下关于其输出的属性：
#   2) 其 TensorImpl 的 use_count 为 1
#   3) 如果函数是视图函数，则其 StorageImpl 与其输入张量的 StorageImpl 相同。否则，其 StorageImpl 的 use_count 为 1
#
# 下面的代码模板实现了这些不变性的检查：
SAVE_TENSOR_STORAGE = CodeTemplate(
    """\
c10::optional<Storage> ${tensor_name}_storage_saved =
  ${tensor_name}.has_storage() ? c10::optional<Storage>(${tensor_name}.storage()) : c10::nullopt;
"""
)

# 如果 tensor_name == out_tensor_name，用于执行 (1)，否则用于执行 (2)
ENFORCE_SAME_TENSOR_STORAGE = CodeTemplate(
    """\
if (${tensor_name}_storage_saved.has_value() &&
    !at::impl::dispatch_mode_enabled() &&
    !at::impl::tensor_has_dispatch(${tensor_name}) &&
    !at::impl::tensor_has_dispatch(${out_tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}_storage_saved.value().is_alias_of(${out_tensor_name}.storage()));
"""
)

SAVE_TENSORLIST_STORAGE = CodeTemplate(
    """\
std::vector<c10::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const Tensor& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
"""
)

ENFORCE_SAME_TENSORLIST_STORAGE = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value() && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(${tensorlist_name}[i].storage()));
}
"""
)

SAVE_OPTIONALTENSORLIST_STORAGE = CodeTemplate(
    """\
std::vector<c10::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const c10::optional<Tensor>& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_value() && tensor->has_storage() ? c10::optional<Storage>(tensor->storage()) : c10::nullopt);
"""
)

ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE = CodeTemplate(
    """\
# 遍历 ${tensorlist_name} 中的张量，直到达到张量列表的大小或者未启用分发模式
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
    # 如果 ${tensorlist_name}_storage_saved[i] 中有值，并且 ${tensorlist_name} 中没有分发
    if (${tensorlist_name}_storage_saved[i].has_value() && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
        # 断言 ${tensorlist_name}_storage_saved[i] 的值是 static_cast<c10::optional<Tensor>>(${tensorlist_name}[i])->storage() 的别名
        TORCH_INTERNAL_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(
            static_cast<c10::optional<Tensor>>(${tensorlist_name}[i])->storage()));
}
"""
)

# 保存张量的实现
SAVE_TENSOR_IMPL = CodeTemplate(
    """\
# 如果 ${tensor_name} 已定义，则保存 ${tensor_name} 的内部指针
c10::intrusive_ptr<TensorImpl> ${tensor_name}_impl_saved;
if (${tensor_name}.defined()) ${tensor_name}_impl_saved = ${tensor_name}.getIntrusivePtr();
"""
)

# 强制要求相同的张量实现
ENFORCE_SAME_TENSOR_IMPL = CodeTemplate(
    """\
# 如果 ${tensor_name}_impl_saved 存在，并且未启用分发模式，并且 ${tensor_name} 没有分发
if (${tensor_name}_impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name}))
    # 断言 ${tensor_name}_impl_saved 等于 ${tensor_name} 的内部指针
    TORCH_INTERNAL_ASSERT(${tensor_name}_impl_saved == ${tensor_name}.getIntrusivePtr());
"""
)

# 强制要求张量实现的使用计数小于等于一
ENFORCE_TENSOR_IMPL_USE_COUNT_LT_OR_EQ_ONE = CodeTemplate(
    """\
# 如果未启用分发模式，并且 ${tensor_name} 没有分发
if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name}))
    # 断言 ${tensor_name} 的使用计数小于等于 1，附带函数名称信息
    TORCH_INTERNAL_ASSERT(${tensor_name}.use_count() <= 1, "function: ${fn_name}");
"""
)

# 强制要求张量存储的使用计数等于一
ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE = CodeTemplate(
    """\
# 如果 ${tensor_name} 有存储，并且未启用分发模式，并且 ${tensor_name} 没有分发
if (${tensor_name}.has_storage() && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name})) {
    # 断言 ${tensor_name} 的存储使用计数等于 1，附带函数名称信息
    TORCH_INTERNAL_ASSERT(${tensor_name}.storage().use_count() == 1, "function: ${fn_name}");
}
"""
)

# 保存张量列表的实现
SAVE_TENSORLIST_IMPL = CodeTemplate(
    """\
# 创建 ${tensorlist_name}_impl_saved 列表，大小为 ${tensorlist_name} 的大小
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
# 遍历 ${tensorlist_name} 中的张量
for (size_t i=0; i<${tensorlist_name}.size(); i++)
    # 如果 ${tensorlist_name}[i] 已定义，则保存 ${tensorlist_name}[i] 的内部指针
    if (${tensorlist_name}[i].defined()) ${tensorlist_name}_impl_saved[i] = ${tensorlist_name}[i].getIntrusivePtr();
"""
)

# 强制要求相同的张量列表实现
ENFORCE_SAME_TENSORLIST_IMPL = CodeTemplate(
    """\
# 遍历 ${tensorlist_name} 中的张量，直到达到张量列表的大小或者未启用分发模式
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
    # 如果 ${tensorlist_name}_impl_saved[i] 存在，并且 ${tensorlist_name} 没有分发
    if (${tensorlist_name}_impl_saved[i] && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
        # 断言 ${tensorlist_name}_impl_saved[i] 等于 ${tensorlist_name}[i] 的内部指针
        TORCH_INTERNAL_ASSERT(${tensorlist_name}_impl_saved[i] == ${tensorlist_name}[i].getIntrusivePtr());
}
"""
)

# 保存可选张量列表的实现
SAVE_OPTIONALTENSORLIST_IMPL = CodeTemplate(
    """\
# 创建 ${tensorlist_name}_impl_saved 列表，大小为 ${tensorlist_name} 的大小
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
# 遍历 ${tensorlist_name} 中的张量
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
    # 获取 ${tensorlist_name}[i] 的可选值
    c10::optional<Tensor> t = ${tensorlist_name}[i];
    # 如果 t 有值并且已定义，则保存 t 的内部指针
    if (t.has_value() && t->defined()) ${tensorlist_name}_impl_saved[i] = t->getIntrusivePtr();
}
"""
)

# 强制要求相同的可选张量列表实现
ENFORCE_SAME_OPTIONALTENSORLIST_IMPL = CodeTemplate(
    """\
# 遍历 ${tensorlist_name} 中的张量，直到达到张量列表的大小或者未启用分发模式
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
    # 如果 ${tensorlist_name}_impl_saved[i] 存在
    if (${tensorlist_name}_impl_saved[i])
        # 断言 ${tensorlist_name}_impl_saved[i] 等于 static_cast<c10::optional<Tensor>>(${tensorlist_name}[i])->getIntrusivePtr()
        TORCH_INTERNAL_ASSERT(
            ${tensorlist_name}_impl_saved[i] == static_cast<c10::optional<Tensor>>(${tensorlist_name}[i])->getIntrusivePtr());
}
"""
)

# 不强制执行不同张量实现或存储的不变性的函数列表
DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE = {
    # 这些函数预计会更改输入张量的实现或存储
    "set_",
    "_cudnn_rnn_flatten_weight",
}
DONT_ENFORCE_TENSOR_IMPL_USE_COUNT = {
    // 这些非原地且非输出的函数返回的张量具有 use_count > 1
    // 因此，它们可能（但不一定）原样返回其输入之一
    // 更多信息请参阅 https://github.com/pytorch/pytorch/issues/60426
    "_embedding_bag",
    "_embedding_bag_forward_only",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "lu_unpack",
    "_cudnn_rnn_backward",
    // 下面的函数在 StorageImpl use_count 检查失败，但我们跳过 tensor_impl 检查
    // 以防万一
    "_cudnn_rnn",
    "dequantize_self",
    // lift() 不应该用 requires_grad=True 的张量调用
    "lift",
    "lift_fresh",
    "lift_fresh_copy",
    // 关于嵌套张量的函数
    // _nested_tensor_size() 不应该用 requires_grad=True 的张量调用
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
};

DONT_ENFORCE_STORAGE_IMPL_USE_COUNT = {
    // 这些非视图函数返回的张量具有 storage use_count != 1
    "_slow_conv2d_forward",
    "slow_conv3d_forward",
    "channel_shuffle",
    // 如果输出中返回输入本身，则无法保证其 storage_impl 的 use count 为 1
    *_DONT_ENFORCE_TENSOR_IMPL_USE_COUNT,
};
// 结束 TensorImpl 和 Storage 指针健全性检查

DECLARE_GRAD_FN = CodeTemplate(
    """\
// 声明一个指向 ${op} 类型对象的共享指针 grad_fn
std::shared_ptr<${op}> grad_fn;
"""
);

DECLARE_VECTOR_OF_GRAD_FN = CodeTemplate(
    """\
// 声明一个 ${op} 类型对象的共享指针向量 grad_fns
std::vector<std::shared_ptr<${op}>> grad_fns;
"""
);

SETUP_ANY_REQUIRES_GRAD = CodeTemplate(
    """\
// 计算所有参数及其导数是否需要梯度
[[maybe_unused]] auto _any_requires_grad = compute_requires_grad( ${args_with_derivatives} );
// ${extra_differentiability_conditions}，其他可微条件设置
"""
);

SETUP_DERIVATIVE = CodeTemplate(
    """\
// 如果任何一个参数需要梯度，则执行以下设置
if (_any_requires_grad) {
  ${setup}
}
"""
);

SETUP_NONE_REQUIRES_GRAD = CodeTemplate(
    """\
// 如果任何一个参数需要梯度，则抛出错误
if (compute_requires_grad( ${args_to_check} )) {
  throw_error_out_requires_grad("${base_name}");
}
"""
);

ASSIGN_GRAD_FN = CodeTemplate(
    """\
// 分配一个 ${op} 类型对象的共享指针给 grad_fn
grad_fn = std::shared_ptr<${op}>(new ${op}(${op_ctor}), deleteNode);
grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
"""
);

// 注意：下面模板中的 `compute_requires_grad` 会被 `SETUP_ANY_REQUIRES_GRAD` 使用，其参数用 `i` 索引
ASSIGN_VECTOR_OF_GRAD_FN = CodeTemplate(
    """\
// 遍历范围为 ${irange} 的索引 i
for (const auto& i : c10::irange( ${irange} )) {
  // 计算第 i 个参数及其导数是否需要梯度
  const auto ith_requires_grad = compute_requires_grad(${args_with_derivatives});
  // 检查 inplace 操作是否可行
  check_inplace(self[i], ith_requires_grad);
  // 将一个 lambda 函数的返回值添加到 grad_fns 中
  grad_fns.push_back([&]() -> std::shared_ptr<${op}> {
      if (!ith_requires_grad) {
          return nullptr;
      } else {
          // 分配一个 ${op} 类型对象的共享指针给 grad_fn
          auto grad_fn = std::shared_ptr<${op}>(new ${op}(${op_ctor}), deleteNode);
          grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
          return grad_fn;
      }
  }());
}
"""
);

CALL_REDISPATCH = CodeTemplate(
    """\
// 创建一个代码模板，用于生成特定的操作代码，使用给定的API名称和参数
at::redispatch::${api_name}(${unpacked_args})"""
)

// 如果非变量操作具有返回值，则使用 `tmp` 变量暂时保存这些值，
// 并将这些值传递给 `at::AutoDispatchBelowAutograd` 保护块外的返回变量。
DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES_JVP_DECOMP = CodeTemplate(
    """\
auto ${tmp_var} = ([&]() {
  // 如果任何一个参数具有前向梯度，则执行下面的逻辑
  if (${any_has_forward_grad}) {
    // 创建完整的操作名
    static c10::OperatorName full_name("aten::${op_name}", "${op_overload}");
    // 查找操作的模式
    static c10::optional<c10::OperatorHandle> opt_op = c10::Dispatcher::singleton().findSchema(full_name);
    // 使用 JIT 分解和参数执行 JVP 运行
    return impl::run_jit_decomposition_with_args_for_jvp<${return_types}>("${op_name}", *opt_op, ks, ${arg_names});
  } else {
    // 否则执行下面的逻辑
    ${guard}
    return ${base_type_call};
  }
})();
"""
)

// 创建一个代码模板，用于生成特定的操作代码，不处理返回值
DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES = CodeTemplate(
    """\
auto ${tmp_var} = ([&]() {
  ${guard}
  return ${base_type_call};
})();
"""
)

// 创建一个代码模板，用于生成特定的操作代码，不处理返回值且没有临时变量
DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES = CodeTemplate(
    """\
{
  ${guard}
  ${base_type_call};
}
"""
)

// 创建一个代码模板，用于设置操作历史
SET_HISTORY = CodeTemplate(
    """\
if (grad_fn) {
    ${fn}_history(${differentiable_outputs}, grad_fn);
}
"""
)

// 创建一个代码模板，用于循环遍历梯度函数的向量
LOOP_OVER_VECTOR_OF_GRAD_FNS = CodeTemplate(
    """\
if (!grad_fns.empty()) {
    ${preamble}
    for (const auto& i : c10::irange(grad_fns.size())) {
        auto grad_fn = grad_fns[i];
        if (grad_fn != nullptr) {
            ${statements}
        }
    }
}
"""
)

// 创建一个代码模板，用于条件执行特定的代码块
CONDITIONAL = CodeTemplate(
    """\
if (${cond}) {
  ${statements}
}
"""
)

// 创建一个代码模板，用于在调试模式下执行特定的代码块
RUN_ONLY_IN_DEBUG_MODE = CodeTemplate(
    """\
#ifndef NDEBUG
${statements}
#endif
"""
)

// 创建一个代码模板，用于前向导数检查
FW_DERIVATIVE_CHECK_TEMPLATE = CodeTemplate(
    """\
isFwGradDefined(${req_inp})\
"""
)

// 创建一个代码模板，用于前向导数大小检查
FW_DERIVATIVE_SIZE_CHECK_TEMPLATE = CodeTemplate(
    """\
TORCH_CHECK(
    self.size() == ${inp_name}.size(),
      "Tensor lists must have the same number of tensors, got ",
    self.size(),
      " and ",
    ${inp_name}.size());
"""
)

// 创建一个代码模板，用于检查前向导数张量列表是否定义
FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE = CodeTemplate(
    """\
isFwGradDefinedTensorList(${req_inp})\
"""
)

// 创建一个代码模板，用于前向导数定义检查
FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE = CodeTemplate(
    """\
auto ${inp_name}_t_raw = toNonOptFwGrad(${inp});
auto ${inp_name}_tensor = toNonOptTensor(${inp});
auto ${inp_name}_t = (${inp_name}_t_raw.defined() || !${inp_name}_tensor.defined())
  ? ${inp_name}_t_raw : at::${zeros_fn}(${inp_name}_tensor.sym_sizes(), ${inp_name}_tensor.options());
"""
)

// 创建一个代码模板，用于原始值定义检查
FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE = CodeTemplate(
    """\
auto ${inp_name}_p = toNonOptPrimal(${inp});
"""
)

// 创建一个代码模板，用于设置前向导数张量
FW_DERIVATIVE_SETTER_TENSOR = CodeTemplate(
    """\
if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined() && ${out_arg}.defined()) {
  // 这里的硬编码 0 在支持多级别时需要更新
  ${out_arg}._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
}
"""
)

// 创建一个代码模板，用于设置前向导数张量列表
FW_DERIVATIVE_SETTER_TENSOR_FOREACH = CodeTemplate(
    """\
for (const auto& i : c10::irange(${out_arg}_new_fw_grad_opts.size())) {
  // 遍历 ${out_arg}_new_fw_grad_opts 的索引范围，i 为索引
  auto& ${out_arg}_new_fw_grad_opt = ${out_arg}_new_fw_grad_opts[i];
  // 获取 ${out_arg}_new_fw_grad_opts 中索引为 i 的元素的引用，命名为 ${out_arg}_new_fw_grad_opt
  if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined() && ${out_arg}[i].defined()) {
    // 检查 ${out_arg}_new_fw_grad_opt 是否有值，并且其值已定义，以及 ${out_arg}[i] 是否已定义
    // 固定的 0 需要在支持多级的情况下更新
    ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
    // 调用 ${out_arg}[i] 的 _set_fw_grad 方法，传入 ${out_arg}_new_fw_grad_opt 的值作为参数，指定 level 为 0，is_inplace_op 根据 ${is_inplace} 决定
  }
}



if (${all_res}_new_fw_grad_opt.has_value() && std::get<${idx}>(${all_res}_new_fw_grad_opt.value()).defined()
    && ${out_arg}.defined()) {
  // 检查 ${all_res}_new_fw_grad_opt 是否有值，并且获取其第 ${idx} 个元素是否已定义，同时检查 ${out_arg} 是否已定义
  ${out_arg}._set_fw_grad(std::get<${idx}>(${all_res}_new_fw_grad_opt.value()), /* level */ 0, /* is_inplace_op */ false);
  // 调用 ${out_arg} 的 _set_fw_grad 方法，传入 ${all_res}_new_fw_grad_opt 的第 ${idx} 个元素作为参数，指定 level 为 0，is_inplace_op 为 false
}



if (${out_arg}_new_fw_grad_opt.has_value()) {
  // 检查 ${out_arg}_new_fw_grad_opt 是否有值
  auto ${out_arg}_new_fw_grad = ${out_arg}_new_fw_grad_opt.value();
  // 获取 ${out_arg}_new_fw_grad_opt 的值，并命名为 ${out_arg}_new_fw_grad
  TORCH_INTERNAL_ASSERT(${out_arg}.size() == ${out_arg}_new_fw_grad.size());
  // 断言 ${out_arg} 的大小与 ${out_arg}_new_fw_grad 的大小相等
  for (const auto i : c10::irange(${out_arg}.size())) {
    // 遍历 ${out_arg} 的索引范围，i 为索引
    if (${out_arg}_new_fw_grad[i].defined() && ${out_arg}[i].defined()) {
      // 检查 ${out_arg}_new_fw_grad[i] 和 ${out_arg}[i] 是否已定义
      // 固定的 0 需要在支持多级的情况下更新
      ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad[i], /* level */ 0, /* is_inplace_op */ ${is_inplace});
      // 调用 ${out_arg}[i] 的 _set_fw_grad 方法，传入 ${out_arg}_new_fw_grad[i] 作为参数，指定 level 为 0，is_inplace_op 根据 ${is_inplace} 决定
    }
  }
}



${fw_grad_opt_definition}
// 替换为具体的 forward grad 选项定义

if (${requires_fw_grad}) {
    // 检查是否需要计算 forward gradient
    ${unpacked_arguments}
    // 展开参数列表
    ${out_arg}_new_fw_grad_opt = ${formula};
    // 计算 ${formula} 并将结果赋给 ${out_arg}_new_fw_grad_opt
}



${fw_grad_opt_definition}
// 替换为具体的 forward grad 选项定义

for (const auto& i : c10::irange(${vector_of_optional_tensor}.size())) {
  // 遍历 ${vector_of_optional_tensor} 的索引范围，i 为索引
  if (${any_has_forward_grad_for_current_index}) {
      // 检查是否有当前索引的 forward gradient
      ${unpacked_arguments}
      // 展开参数列表
      ${vector_of_optional_tensor}[i] = ${formula};
      // 计算 ${formula} 并将结果赋给 ${vector_of_optional_tensor}[i]
  }
}



TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${name} that does not support it ${msg}");
// 如果 ${cond} 为真，则抛出错误，说明 ${name} 不支持使用 forward AD



for (const auto& _t: ${arg}) {
    TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${name} that does not support it ${msg}");
}
// 遍历 ${arg}，如果 ${cond} 为真，则抛出错误，说明 ${name} 不支持使用 forward AD
    fm.write(
        "VariableType.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/VariableType.h"
        },
    )


# 使用 FileManager 对象 fm 写入文件 "VariableType.h"
# 文件内容由一个 lambda 函数生成，lambda 函数返回一个字典，包含生成的注释信息



    # helper that generates a TORCH_LIBRARY_IMPL macro for each
    # dispatch key that appears in derivatives.yaml
    def wrapper_registrations(used_keys: set[str]) -> str:
        library_impl_macro_list: list[str] = []
        for key in sorted(used_keys):
            dispatch_key = key
            if key == "Default":
                dispatch_key = "Autograd"
            # 生成 TORCH_LIBRARY_IMPL 宏的字符串
            library_impl_macro = (
                f"TORCH_LIBRARY_IMPL(aten, {dispatch_key}, m) "
                + "{\n"
                + "${"
                + f"wrapper_registrations_{key}"
                + "}\n}"
            )
            library_impl_macro_list += [library_impl_macro]
        return "\n\n".join(library_impl_macro_list)


# 定义一个函数 wrapper_registrations，用于生成 derivatives.yaml 中每个 dispatch key 对应的 TORCH_LIBRARY_IMPL 宏的字符串
# 返回一个包含所有生成宏字符串的文本块



    fm1 = FileManager(
        install_dir=out + "/templates", template_dir=template_path, dry_run=False
    )
    fm1.write(
        "VariableType.cpp",
        lambda: {
            "type_derived_method_definitions": "\n\n".join(
                [
                    "${" + f"type_derived_method_definitions_{key}" + "}"
                    for key in sorted(used_keys)
                ]
            ),
            "wrapper_registrations": wrapper_registrations(used_keys),
        },
    )


# 使用 FileManager 对象 fm1 初始化，设置安装目录和模板目录，并且非 dry_run 模式
# 使用 fm1 写入文件 "VariableType.cpp"
# 文件内容由一个 lambda 函数生成，lambda 函数返回一个字典，包含生成的 type_derived_method_definitions 和 wrapper_registrations



    fm2 = FileManager(install_dir=out, template_dir=out + "/templates", dry_run=False)
    
    sharded_keys = set(
        [f"type_derived_method_definitions_{key}" for key in sorted(used_keys)]
        + [f"wrapper_registrations_{key}" for key in sorted(used_keys)]
    )
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    fm2.write_sharded(
        "VariableType.cpp",
        [fn for fn in fns_with_diff_infos if use_derived(fn)],
        key_fn=lambda fn: cpp.name(fn.func.func),
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/VariableType.cpp",
        },
        env_callable=gen_variable_type_func,
        num_shards=5,
        sharded_keys=sharded_keys,
    )


# 使用 FileManager 对象 fm2 初始化，设置安装目录和模板目录，并且非 dry_run 模式
# 使用 fm2 的 write_sharded 方法写入文件 "VariableType.cpp" 的多个分片
# 分片的选择基于条件 [fn for fn in fns_with_diff_infos if use_derived(fn)]
# key_fn 用于生成每个分片的名称
# base_env 包含一个生成的注释信息
# env_callable 是一个函数，用于生成 VariableType.cpp 文件的内容
# num_shards 指定分片的数量为 5
# sharded_keys 是一个集合，包含需要在分片中使用的键
# 使用装饰器 `with_native_function_and` 注册函数 `gen_wrapper_registration`，接收一个 `NativeFunction` 类型的参数 `f`，并返回一个字符串
@with_native_function_and
def gen_wrapper_registration(f: NativeFunction, key: str = "Default") -> str:
    # 使用字符串模板 `WRAPPER_REGISTRATION` 替换相关参数，生成注册字符串
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f, key),
        class_type="VariableType",
    )


# 定义函数 `gen_variable_type_func`，接收一个 `NativeFunctionWithDifferentiabilityInfo` 类型的参数 `fn`，返回一个字典
def gen_variable_type_func(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> dict[str, list[str]]:
    # 提取函数对象 `f`
    f = fn.func
    result = {}

    # 断言：验证是否存在手动后端内核注册的情况
    # 参见注释 [Manual Backend kernels]
    assert (name in MANUAL_BACKEND) == f.manual_kernel_registration

    # 如果要注册一个自动求导的内核，必须将操作设置为抽象
    # 换句话说，此操作必须在 native_functions.yaml 中具有 dispatch 部分
    if name in MANUAL_AUTOGRAD_AND_TRACER or (
        fn.info and any(info.has_derivatives for info in fn.info.values())
    ):
        # 提示消息
        msg = (
            f"There's a formula for {name}(or its functional variant) in derivatives.yaml. "
            f"It's required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA "
            f"or CompositeExplicitAutograd in native_functions.yaml. Please see "
            f"https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword "
            f"for instructions to choose the right dispatch keyword."
        )
        # 断言：验证操作是否为抽象
        assert f.is_abstract, msg

    return result


# 针对没有不同iability信息的操作的字典
_foreach_ops_without_differentiability_info = {
    # 由于缺少 `{maximum, minimum}(tensor, scalar)`，因此没有参考的反向传播
    ("_foreach_maximum", "Scalar"),
    ("_foreach_maximum", "ScalarList"),
    ("_foreach_minimum", "Scalar"),
    ("_foreach_minimum", "ScalarList"),
    # 由于 addcdiv/addcmul 不支持 Tensor 作为缩放因子，因此没有参考的反向传播
    ("_foreach_addcdiv", "Tensor"),
    ("_foreach_addcmul", "Tensor"),
    ("_foreach_copy", ""),
}

# 针对具有不同参数数量的操作的字典
_foreach_ops_with_different_arity = {
    # 这些操作缺少要应用于右侧参数的缩放因子 `alpha`
    ("_foreach_add", "Scalar"),
    ("_foreach_add", "ScalarList"),
    ("_foreach_sub", "Scalar"),
    ("_foreach_sub", "ScalarList"),
}


# 使用装饰器 `with_native_function_with_differentiability_info_and_key` 注册函数 `emit_body`，接收一个 `NativeFunctionWithDifferentiabilityInfo` 类型的参数 `fn` 和一个字符串类型的参数 `key`，返回一个字符串列表
@with_native_function_with_differentiability_info_and_key
def emit_body(
    fn: NativeFunctionWithDifferentiabilityInfo, key: str = "Default"
) -> list[str]:
    # 断言：验证分发策略是否为 "use_derived"
    assert dispatch_strategy(fn) == "use_derived"
    # 提取函数对象 `f`
    f = fn.func
    # 提取信息对象 `info`
    info = fn.info[key] if fn.info else None
    # 提取正向导数列表 `fw_derivatives`
    fw_derivatives = fn.fw_derivatives.get(key, []) if fn.fw_derivatives else []

    # 获取函数名 `name`
    name = cpp.name(f.func)
    # 检查操作是否为原地操作
    inplace = f.func.kind() == SchemaKind.inplace
    # 检查操作是否为输出函数
    is_out_fn = f.func.kind() == SchemaKind.out
    # 检查函数是否返回 void
    returns_void = len(f.func.returns) == 0
    # 获取基本名称 `base_name`
    base_name = get_base_name(f)
    # 获取视图信息 `view_info`
    view_info = get_view_info(f)

    # 检查是否为 `_foreach` 开头的操作
    is_foreach = name.startswith("_foreach")
    # 检查是否为原地 `_foreach` 操作
    is_inplace_foreach = is_foreach and inplace
    # 如果是 inplace_foreach 模式，则执行以下逻辑
    if is_inplace_foreach:
        # 创建空字典，用于存储 foreacharg 到 refarg 的映射关系
        inplace_foreacharg2refarg: dict[Argument, Argument] = {}
        # 创建空字典，用于存储 refargname 到 inplace_foreacharg 的映射关系
        refargname2inplace_foreacharg: dict[str, Argument] = {}
        # 获取函数名称和重载名称的元组
        base_name_and_overload_name = (f.func.name.name.base, f.func.name.overload_name)
        # 如果 info 为 None，则验证该函数名是否在没有不同iability信息的操作列表中
        if info is None:
            assert (
                base_name_and_overload_name
                in _foreach_ops_without_differentiability_info
            ), f"{'.'.join(base_name_and_overload_name)} should have a differentiability info"
        else:
            # 否则，验证函数的非输出参数数量是否与引用的函数参数数量相同，或者验证函数名是否在具有不同数量的参数的操作列表中
            assert (
                len(f.func.arguments.flat_non_out)
                == len(info.func.func.arguments.flat_non_out)
            ) or (base_name_and_overload_name in _foreach_ops_with_different_arity), (
                f"{'.'.join(base_name_and_overload_name)} has {len(f.func.arguments.flat_non_out)} args "
                f"but the reference has {len(info.func.func.arguments.flat_non_out)}"
            )
            # 遍历函数参数和引用函数参数的对应关系，并进行类型验证
            for foreach_arg, ref_arg in zip(
                f.func.arguments.flat_non_out, info.func.func.arguments.flat_non_out
            ):
                foreach_arg_type = foreach_arg.type
                # 如果 foreach_arg 的类型是 ListType，则使用其元素类型
                if isinstance(foreach_arg_type, ListType):
                    foreach_arg_type = foreach_arg_type.elem
                # 验证 foreach_arg 的类型与 ref_arg 的类型是否一致
                assert foreach_arg_type == ref_arg.type
                # 将 foreach_arg 和 ref_arg 的映射关系存入 inplace_foreacharg2refarg 字典
                inplace_foreacharg2refarg[foreach_arg] = ref_arg
                # 将 ref_arg 的名称和 foreach_arg 的映射关系存入 refargname2inplace_foreacharg 字典
                refargname2inplace_foreacharg[ref_arg.name] = foreach_arg

    # 定义函数 gen_differentiable_input，用于生成可微输入
    def gen_differentiable_input(
        arg: Argument | SelfArgument | TensorOptionsArguments,
    ) -> DifferentiableInput | None:
        # 如果 arg 的类型是 TensorOptionsArguments，则返回 None
        if isinstance(arg, TensorOptionsArguments):
            return None
        # 如果 arg 是 SelfArgument 类型，则取其 argument 属性，否则直接使用 arg
        a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg

        # TODO: `cpp_type` 只是为了与旧代码兼容，实际上应该删除。
        # NB: 这不是 cpp.argument() 的克隆 - TensorOptionsArguments / faithful / binds
        # 没有正确处理，因为它们对于此代码生成无关紧要。
        # 使用 cpp.argument_type 方法获取参数 a 的 C++ 类型
        cpp_type = cpp.argument_type(a, binds=a.name, symint=True).cpp_type()

        # 如果参数 a 的名称、类型和信息 info 符合可微条件，则返回一个 DifferentiableInput 对象
        if not is_differentiable(a.name, a.type, info):
            return None
        return DifferentiableInput(
            name=a.name,
            type=a.type,
            cpp_type=cpp_type,
        )

    # 应用装饰器 with_native_function 到当前函数或方法上下文
    @with_native_function
    def gen_differentiable_inputs(f: NativeFunction) -> list[DifferentiableInput]:
        # 从函数 f 的非输出参数中获取参数列表
        arguments = list(f.func.arguments.non_out)
        # 如果是 inplace foreach 并且有信息 info
        if is_inplace_foreach and info is not None:
            # 遍历函数 f 的扁平化非输出参数
            for i, arg in enumerate(f.func.arguments.flat_non_out):
                # 如果参数 arg 存在于 inplace_foreacharg2refarg 中
                if arg in inplace_foreacharg2refarg:
                    # 注释(crcrpar): 根据我理解，重要的是参数名是否相同。
                    # 因此，我只有在名称不同时才替换参数。
                    # TODO(crcrpar): 简化这一过程。
                    # 获取映射后的参数
                    mapped_arg = inplace_foreacharg2refarg[arg]
                    # 使用映射后的参数创建 Argument 对象并替换原参数列表中的对应位置
                    arguments[i] = Argument(
                        mapped_arg.name,
                        mapped_arg.type,
                        mapped_arg.default,
                        mapped_arg.annotation,
                    )
        # 调用 gen_differentiable_input 处理每个参数，生成可微分输入的列表并返回
        return list(mapMaybe(gen_differentiable_input, arguments))

    def find_args_with_derivatives(
        differentiable_inputs: list[DifferentiableInput],
    ) -> list[DifferentiableInput]:
        """查找具有导数定义的参数"""
        # 如果 info 为 None 或没有 derivatives，则直接返回可微分输入列表
        if info is None or not info.has_derivatives:
            return differentiable_inputs
        # 获取所有 derivatives 中的变量名并放入集合 names 中
        names = {name for d in info.derivatives for name in d.var_names}
        # 从 differentiable_inputs 中筛选出具有导数定义的参数列表
        differentiable = [arg for arg in differentiable_inputs if arg.name in names]
        # 如果筛选出的参数数量与 names 集合中的数量不一致，抛出 RuntimeError
        if len(differentiable) != len(names):
            missing = names - {arg.name for arg in differentiable}
            raise RuntimeError(
                f"Missing arguments for derivatives: {missing} in {info.name}"
            )
        # 返回具有导数定义的参数列表
        return differentiable

    # 生成函数 f 的可微分输入列表
    differentiable_inputs = gen_differentiable_inputs(f)
    # 查找具有导数定义的参数列表
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    # 生成函数 fn 的可微分输出列表
    differentiable_outputs = gen_differentiable_outputs(fn, key)

    # 检查 base_name 或 name 是否在 DONT_REQUIRE_DERIVATIVE 中
    undifferentiable = (base_name in DONT_REQUIRE_DERIVATIVE) or (
        name in DONT_REQUIRE_DERIVATIVE
    )

    # 检查是否需要计算导数
    requires_derivative = (
        (not undifferentiable)
        and (len(differentiable_inputs) > 0)
        and (
            (len(differentiable_outputs) > 0)
            # 注释(crcrpar): inplace foreach 函数是无返回值函数。
            or is_inplace_foreach
        )
    )

    # 如果 info 存在且有 derivatives，但不需要计算导数且返回值数量大于零，则抛出 RuntimeError
    if (
        info is not None
        and info.has_derivatives
        and not requires_derivative
        # out= 操作允许零返回，这会导致 requires_derivative 为 False
        # 但我们不应该抛出错误（autograd 中的 out= 操作仅重新分发）
        and len(f.func.returns) > 0
    ):
        raise RuntimeError(
            f"ERROR: derivative ignored for {name} -- specified an autograd function without derivative"
        )

    # 注释(crcrpar): inplace foreach 函数不支持正向自动微分
    # 如果需要导数并且存在前向导数函数并且不是 inplace 操作
    if requires_derivative and len(fw_derivatives) > 0 and not is_inplace_foreach:
        # 断言前向导数函数变量名长度之和应该等于可微输出的数量
        assert sum(len(derivative.var_names) for derivative in fw_derivatives) == len(
            differentiable_outputs
        ), (
            "Expected the number of forward derivatives implemented to match the "
            "number of differentiable outputs. NB: This only applies when at least "
            "one forward derivative is implemented. Not implementing any forward "
            "derivatives is also okay, and we would require inputs to the op to "
            "not have associated tangents in that case."
        )
    
    # 尝试进行 JIT 分解
    try_jit_decomposition = (
        requires_derivative
        and len(fw_derivatives) == 0
        and (not modifies_arguments(f))
        and (not returns_void)
    )
    def setup_derivative(differentiable_inputs: list[DifferentiableInput]) -> list[str]:
        body: list[str] = []  # 初始化一个空列表，用于存储函数体内的代码行

        if is_out_fn:
            # 对于输出函数，确保输入和输出都不需要梯度
            body.append(DECLARE_GRAD_FN.substitute(op="Node"))  # 插入声明不需要梯度的函数代码
            body.append(
                SETUP_NONE_REQUIRES_GRAD.substitute(
                    base_name=base_name,
                    args_to_check=[arg.name for arg in differentiable_inputs],
                )
            )  # 插入设置不需要梯度的输入参数的代码
            body.append(
                SETUP_NONE_REQUIRES_GRAD.substitute(
                    base_name=base_name,
                    args_to_check=[arg.name for arg in differentiable_outputs],
                )
            )  # 插入设置不需要梯度的输出参数的代码
            return body  # 返回完整的函数体列表

        op = info.op if info is not None and info.has_derivatives else "NotImplemented"
        setup = []

        if not is_inplace_foreach:
            setup.extend(
                ASSIGN_GRAD_FN.substitute(
                    op=op,
                    op_ctor="" if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"',
                    args_with_derivatives=[arg.name for arg in args_with_derivatives],
                ).split("\n")
            )  # 如果不是原地操作的循环函数，插入分配梯度函数的代码
        else:
            # 注意(crcrpar)：假设原地循环函数的 self_arg 总是 TensorList。
            list_like_arg = "self"
            args = [arg.name for arg in args_with_derivatives]

            for i, arg in enumerate(args):
                if is_inplace_foreach and info is not None:
                    if arg in refargname2inplace_foreacharg:
                        foreach_arg = refargname2inplace_foreacharg[arg]
                        args[i] = foreach_arg.name + ("[i]" if isinstance(foreach_arg.type, ListType) else "")
                else:
                    if arg == list_like_arg:
                        args[i] = arg + "[i]"

            setup.extend(
                ASSIGN_VECTOR_OF_GRAD_FN.substitute(
                    op=op,
                    op_ctor="" if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"',
                    args_with_derivatives=args,
                    irange=f"{list_like_arg}.size()",
                ).split("\n")
            )  # 如果是原地操作的循环函数，插入分配梯度向量函数的代码

        setup.extend(emit_save_inputs())  # 插入保存输入的代码

        body.extend(
            emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives)
        )  # 插入检查是否有需要梯度的输入的代码

        declare_grad_fn_template = (
            DECLARE_GRAD_FN if not is_inplace_foreach else DECLARE_VECTOR_OF_GRAD_FN
        )
        body.append(declare_grad_fn_template.substitute(op=op))  # 根据是否是原地循环函数，插入声明梯度函数的代码
        body.append(SETUP_DERIVATIVE.substitute(setup=setup))  # 插入设置导数的代码
        return body  # 返回完整的函数体列表
    def emit_check_if_in_complex_autograd_allowlist() -> list[str]:
        # 初始化一个空列表来存储函数体的每一行代码
        body: list[str] = []
        # 如果 base_name 在 GRADIENT_IMPLEMENTED_FOR_COMPLEX 中，则直接返回空的函数体列表
        if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
            return body
        # 遍历 differentiable_outputs 中的每个参数对象
        for arg in differentiable_outputs:
            # 获取参数名
            name = arg.name
            # 检查参数的 C++ 类型是否为 "at::Tensor" 或者在 TENSOR_LIST_LIKE_CTYPES 中
            if arg.cpp_type == "at::Tensor" or arg.cpp_type in TENSOR_LIST_LIKE_CTYPES:
                # 如果满足条件，则向函数体列表中添加抛出错误的函数调用语句
                body.append(f'throw_error_for_complex_autograd({name}, "{base_name}");')
        # 返回函数体列表
        return body

    def emit_check_no_requires_grad(
        tensor_args: list[DifferentiableInput],
        args_with_derivatives: list[DifferentiableInput],
    ) -> list[str]:
        """Checks that arguments without derivatives don't require grad"""
        # 初始化一个空列表来存储函数体的每一行代码
        body: list[str] = []
        # 遍历 tensor_args 列表中的每个参数对象
        for arg in tensor_args:
            # 如果当前参数在 args_with_derivatives 中，则跳过本次循环
            if arg in args_with_derivatives:
                continue
            # 获取参数名
            arg_name = arg.name
            # 如果存在 info 并且 arg_name 在 info.non_differentiable_arg_names 中，则跳过本次循环
            if info and arg_name in info.non_differentiable_arg_names:
                continue
            # 如果参数名为 "output"，则跳过本次循环
            if arg_name == "output":
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            # 向函数体列表中添加检查参数是否需要 grad 的函数调用语句
            body.append(f'check_no_requires_grad({arg_name}, "{arg_name}", "{name}");')
        # 返回函数体列表
        return body

    def emit_original_self_definition() -> list[str]:
        # 初始化一个空列表来存储函数体的每一行代码
        body: list[str] = []
        # 如果 inplace 为 True，则执行以下逻辑
        if inplace:
            # 如果 is_inplace_foreach 为 True，则创建一个原始 self 引用列表
            if is_inplace_foreach:
                body.append(
                    "std::vector<c10::optional<at::Tensor>> original_selfs(self.size());"
                )
            else:
                body.append("c10::optional<at::Tensor> original_self;")
            
            # 初始化一个条件列表来存储是否需要原始 self 的条件
            all_forward_grad_cond = []
            # 遍历 fw_derivatives 中的每个导数对象
            for derivative in fw_derivatives:
                # 如果 derivative.required_original_self_value 为 True，则添加相关条件
                if derivative.required_original_self_value:
                    all_forward_grad_cond.append(
                        get_any_has_forward_grad_name(derivative.var_names)
                    )
            
            # 如果存在需要原始 self 的条件
            if all_forward_grad_cond:
                # 如果不是 inplace_foreach，则添加条件语句和克隆操作
                if not is_inplace_foreach:
                    body.append(f'if ({" || ".join(all_forward_grad_cond)}) {{')
                    body.append("  original_self = self.clone();")
                    body.append("}")
                else:
                    # 如果是 inplace_foreach，则添加循环语句和条件克隆操作
                    current_all_forward_grad_cond = [
                        f"{cond}[i]" for cond in all_forward_grad_cond
                    ]
                    body.append("for (const auto& i : c10::irange(self.size())) {")
                    body.append(
                        f"  if ({' || '.join(current_all_forward_grad_cond)}) {{"
                    )
                    body.append("    original_selfs[i] = self[i].clone();")
                    body.append("  }")
                    body.append("}")

        # 返回函数体列表
        return body

    def save_variables(
        saved_variables: Sequence[SavedAttribute],
        is_output: bool,
        guard_for: Callable[[SavedAttribute], str | None] = lambda name: None,
    # 生成一个 Dispatcher::redispatch() 调用到分发器中。主要出于性能考虑：
    #  - 预先计算完整的 DispatchKeySet。这样可以避免分发器从 TLS 中读取。
    #  - redispatch() 避免了对 RecordFunction 的多余调用，因为我们在进入此自动微分内核之前已经调用过了。
    def emit_dispatch_call(
        f: NativeFunction, input_base: str, unpacked_args: Sequence[str]
    ) -> str:
        """通过命名空间中的函数或张量上的方法进行分发调用。"""
        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        dispatcher_exprs = dispatcher_sig.exprs()

        # 代码生成的自动微分内核通过内核直接传递和重新计算分发键以提高性能。
        # 操作也总是具有 redispatch API 的函数变体。
        # 有关详细信息，请参阅注释 [Plumbing Keys Through The Dispatcher]。
        dispatch_key_set = "ks & c10::after_autograd_keyset"
        call = CALL_REDISPATCH.substitute(
            api_name=cpp.name(
                f.func,
                faithful_name_for_out_overloads=True,
                symint_overload=f.func.has_symint(),
            ),
            unpacked_args=[dispatch_key_set] + list(unpacked_args),
        )
        return call

    def wrap_output(
        f: NativeFunction, unpacked_bindings: list[Binding], var: str
    ) -> str:
        call = ""
        rhs_value: str | None = None
        if not any(r.type.is_tensor_like() for r in f.func.returns):
            rhs_value = var
        else:
            rhs_value = f"std::move({var})"
        assert rhs_value is not None
        call += ASSIGN_RETURN_VALUE.substitute(
            return_values=tie_return_values(f), rhs_value=rhs_value
        )
        return call

    def check_tensorimpl_and_storage(
        call: str, unpacked_bindings: list[Binding]
    def emit_call(
        f: NativeFunction, unpacked_bindings: list[Binding], try_jit_decomposition: bool
    ) -> str:
        # 我们只关心对非变量分发添加 `at::AutoDispatchBelowAutograd` 保护
        # （对应于'use_derived'策略）。这个保护的目的是确保基本类型的操作仍然分发到非变量类型，
        # 即使传入的参数现在是变量。
        # 详见注释 [ 在类型分发中将变量视为非变量 ]
        
        # 解包参数列表中的绑定变量的名称
        unpacked_args = [b.name for b in unpacked_bindings]
        
        # 生成调用类型分发的基本调用语句
        base_type_call = emit_dispatch_call(f, "self_", unpacked_args)

        # 根据函数是否获取视图信息或修改参数来选择保护类型
        if get_view_info(f) is not None or modifies_arguments(f):
            guard = "at::AutoDispatchBelowAutograd guard;"
        else:
            guard = "at::AutoDispatchBelowADInplaceOrView guard;"

        # 判断是否有任何正向梯度，根据需要导数来确定
        any_has_forward_grad = (
            get_any_has_fw_grad_cond(derivative=None)
            if requires_derivative
            else "false"
        )

        # 生成返回类型的字符串表示，如果有多个返回值则使用元组类型
        return_types = ", ".join(
            [cpp.return_type(a, symint=True).cpp_type() for a in f.func.returns]
        )
        if len(f.func.returns) > 1:
            return_types = f"std::tuple<{return_types}>"

        # 获取参数的名称列表
        arg_names = [
            a.name
            for a in cpp.arguments(
                f.func.arguments,
                faithful=True,
                symint=True,
                method=False,
                cpp_no_default_args=set(),
            )
        ]

        # 根据函数是否修改参数和是否返回void来选择调用语句模板
        if not modifies_arguments(f) and not returns_void:
            if try_jit_decomposition:
                # 使用带临时返回值的JVP分解版本的非变量类型分发调用语句模板
                call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES_JVP_DECOMP.substitute(
                    base_type_call=base_type_call,
                    tmp_var=TMP_VAR,
                    guard=guard,
                    any_has_forward_grad=any_has_forward_grad,
                    op_name=cpp.name(f.func),
                    op_overload=f.func.name.overload_name,
                    return_types=return_types,
                    arg_names=arg_names,
                )
            else:
                # 使用普通版本的非变量类型分发调用语句模板
                call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES.substitute(
                    base_type_call=base_type_call,
                    tmp_var=TMP_VAR,
                    guard=guard,
                )

            # 包装输出并添加到调用语句末尾
            call += wrap_output(f, unpacked_bindings, TMP_VAR)
        else:
            # 如果不尝试JIT分解，则使用不返回值的非变量类型分发调用语句模板
            assert not try_jit_decomposition
            call = DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES.substitute(
                base_type_call=base_type_call, guard=guard
            )

        # 检查TensorImpl和Storage，并返回最终的调用语句
        call = check_tensorimpl_and_storage(call, unpacked_bindings)
        return call
    # 返回一个字符串，根据条件选择生成不同的文件名前缀
    def emit_history() -> str:
        fn = "rebase" if modifies_arguments(f) and view_info is None else "set"
        # 从不同的可微输出中获取名称列表
        output_names = [r.name for r in differentiable_outputs]
        # 如果不是就地操作，将输出名称列表展平成参数字符串；否则将 "self" 作为参数
        outs = CodeTemplate("flatten_tensor_args( ${outs} )").substitute(
            outs=output_names if not is_inplace_foreach else "self"
        )
        if not is_inplace_foreach:
            # 返回设置历史记录的代码模板，用文件名前缀和可微输出替换占位符
            return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)
        else:
            # 返回循环遍历梯度函数向量的代码模板，设置不同的历史记录
            return LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(
                preamble=(
                    # 设置不同iable_outputs 为不同的outputs的variable
อเน่เป็น
    # 定义函数 emit_any_has_forward_grad，返回类型为 list[str]
    def emit_any_has_forward_grad() -> list[str]:
        # 初始化空列表 content，用于存储生成的代码片段
        content: list[str] = []
        
        # 如果不是 foreach 模式
        if not is_foreach:
            # 遍历 fw_derivatives 中的每个 derivative
            for derivative in fw_derivatives:
                # 获取需要前向梯度的条件 requires_fw_grad
                requires_fw_grad = get_any_has_fw_grad_cond(derivative=derivative)
                
                # 如果 info 存在且包含输出不同iability_conditions
                if info and info.output_differentiability_conditions:
                    # 断言只有一个输出的不同iability_conditions
                    assert len(info.output_differentiability_conditions) == 1
                    # 将 requires_fw_grad 与输出的不同iability_conditions 结合起来
                    requires_fw_grad = f"({info.output_differentiability_conditions[0]}) && {requires_fw_grad}"
                
                # 将生成的代码段添加到 content 中
                content.append(
                    f"[[maybe_unused]] auto {get_any_has_forward_grad_name(derivative.var_names)} = {requires_fw_grad};"
                )
        else:
            # 如果是 foreach 模式
            for derivative in fw_derivatives:
                # 获取 bool 向量的名称
                bool_vector_name = get_any_has_forward_grad_name(derivative.var_names)
                # 初始化当前导数条件列表
                cur_derivative_conditions = []
                
                # 遍历每个不同iable_input
                for inp in differentiable_inputs:
                    # 如果 derivative.required_inputs_fw_grad 为 None，则继续下一个循环
                    if derivative.required_inputs_fw_grad is None:
                        continue
                    # 如果 inp.name 不在 derivative.required_inputs_fw_grad 中，则继续下一个循环
                    if inp.name not in derivative.required_inputs_fw_grad:
                        continue
                    
                    # 获取输入名称 inp_name 和类型 inp_type
                    inp_name = (
                        inp.name
                        if not inplace
                        else refargname2inplace_foreacharg[inp.name].name
                    )
                    inp_type = (
                        inp.type
                        if not inplace
                        else refargname2inplace_foreacharg[inp.name].type
                    )
                    
                    # 检查是否为列表类型
                    is_list_type = is_tensor_list_type(inp_type)
                    
                    # 如果是列表类型
                    if is_list_type:
                        # 如果 inp_name 不是 "self"，则添加大小检查代码模板
                        if inp_name != "self":
                            content.append(
                                FW_DERIVATIVE_SIZE_CHECK_TEMPLATE.substitute(
                                    inp_name=inp_name
                                )
                            )
                        # 添加当前导数检查代码模板到 cur_derivative_conditions
                        cur_derivative_conditions.append(
                            FW_DERIVATIVE_CHECK_TEMPLATE.substitute(
                                req_inp=inp_name + "[i]"
                            )
                        )
                    else:
                        # 添加当前导数检查代码模板到 cur_derivative_conditions
                        cur_derivative_conditions.append(
                            FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp_name)
                        )
                
                # 生成 std::vector<bool> 类型的向量 bool_vector_name
                content.append(f"std::vector<bool> {bool_vector_name}(self.size());")
                # 添加循环开始的代码块
                content.append("for (const auto& i : c10::irange(self.size())) {")
                # 将当前导数条件应用到 bool_vector_name 的每个元素
                content.append(
                    f"  {bool_vector_name}[i] = {' || '.join(cur_derivative_conditions)};"
                )
                # 添加循环结束的代码块
                content.append("}")
        
        # 返回生成的代码片段列表 content
        return content
    
    # 定义函数 emit_check_inplace，返回类型为 list[str]
    def emit_check_inplace() -> list[str]:
        # 如果不是 inplace 模式，则返回空列表
        if not inplace:
            return []
        
        # 如果是 inplace 模式，则生成检查 inplace 的代码片段
        return [
            f"check_inplace({arg.name}, _any_requires_grad);"
            for arg in differentiable_outputs
        ]
    # 定义函数 emit_forbid_fw_derivatives，用于生成禁止前向导数的代码片段
    def emit_forbid_fw_derivatives(is_out_fn: bool = False) -> str:
        # 根据是否为输出函数设置不同的提示信息
        if is_out_fn:
            msg = "because it is an out= function"
        else:
            msg = (
                "because it has not been implemented yet.\\nPlease file an issue "
                "to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml "
                "so that we can prioritize its implementation."
            )
        # 获取任何具有前向梯度条件的字符串表示
        cond = get_any_has_fw_grad_cond(derivative=None)
        # 使用模板生成具有禁止前向导数信息的代码片段，如果条件不为空则生成代码，否则返回空字符串
        return (
            FW_DERIVATIVE_FORBID_TEMPLATE.substitute(cond=cond, name=name, msg=msg)
            if cond != ""
            else ""
        )

    # 初始化代码体列表
    body: list[str] = []
    # 解包参数统计和解包后的绑定
    unpack_args_stats, unpacked_bindings = unpack_args(f)

    # 将解包参数统计添加到代码体中
    body.extend(unpack_args_stats)
    # 如果需要导数
    if requires_derivative:
        # 添加任何要求梯度的检查代码
        body.extend(emit_any_requires_grad())
        # 添加任何具有前向梯度的检查代码
        body.extend(emit_any_has_forward_grad())
        # 添加检查是否为原地操作的代码
        body.extend(emit_check_inplace())
        # 添加原始 self 定义的代码
        body.extend(emit_original_self_definition())
        # 设置导数相关的代码
        body.extend(setup_derivative(differentiable_inputs))

    # 添加调用函数的代码，并传入解包后的绑定和尝试 JIT 分解的标志
    body.append(emit_call(f, unpacked_bindings, try_jit_decomposition))
    # 如果需要导数
    if requires_derivative:
        # 添加历史记录代码
        body.append(emit_history())
        # 添加检查是否在复杂自动求导白名单中的代码
        body.extend(emit_check_if_in_complex_autograd_allowlist())

    # 如果是输出函数
    if is_out_fn:
        # 添加生成禁止前向导数代码片段的调用
        body.append(emit_forbid_fw_derivatives(is_out_fn=True))
    else:
        # 如果需要导数且不尝试 JIT 分解
        if requires_derivative and not try_jit_decomposition:
            # 如果前向导数列表长度大于零，则添加生成前向导数代码的调用
            if len(fw_derivatives) > 0:
                body.extend(emit_fw_derivatives())
            else:
                # 否则添加生成禁止前向导数代码片段的调用
                body.append(emit_forbid_fw_derivatives())

    # 如果需要导数
    if requires_derivative:
        # 添加保存输出的代码
        body.append(emit_save_outputs())

    # 如果函数名字符串在 RESET_GRAD_ACCUMULATOR 中
    if str(f.func.name.name) in RESET_GRAD_ACCUMULATOR:
        # 对于 inplace 操作，断言确保只有一个名为 `self` 的输出
        assert inplace
        # 添加重置梯度累加器的代码
        body.append("reset_grad_accumulator(self);")
    # 如果不返回空值
    if not returns_void:
        # 添加返回值的代码
        body.append(f"return {get_return_value(f)};")
    # 返回最终的代码体列表
    return body
```