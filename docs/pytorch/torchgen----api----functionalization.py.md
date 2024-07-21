# `.\pytorch\torchgen\api\functionalization.py`

```
# 从未来模块导入注解，用于支持类型提示中的 `annotations`
from __future__ import annotations

# 导入 torchgen 库中的 dispatcher 和 types 模块
from torchgen.api import dispatcher
from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    longT,
    NamedCType,
    tensorT,
)

# 导入 torchgen 库中的 model 模块中的具体类和函数
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsViewGroup,
)

# 以下是文件的总体描述，描述了 JIT schema 到 API 转换的过程，
# 用于创建视图 lambda，这些 lambda 在 functionalization 阶段使用。
# 主要包括两种 lambda：前向 lambda 和反向 lambda。
# API 大部分遵循 dispatcher API 的约定，但有少许特殊之处：
# - lambda 捕获部分需将引用类型转换为值类型
# - 前向 lambda 直接调用 at::_ops API，反向 lambda 负责生成调用点和声明（手动实现在 at::functionalization::impl 命名空间中）。

# 在 functionalization 阶段为每个视图操作生成的 lambda 形式如下：
# [capture_arguments](outer_arguments) -> returns_type {
#     return name(inner_arguments);
# }

# 定义一些特定的 lambda 输入参数绑定。
base_binding = Binding(
    name="base",
    nctype=NamedCType(name="base", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
mutated_view_binding = Binding(
    name="mutated_view",
    nctype=NamedCType(name="mutated_view", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
mutated_view_idx_binding = Binding(
    name="mutated_view_idx",
    nctype=NamedCType(name="mutated_view_idx", type=BaseCType(longT)),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
reapply_views_binding = Binding(
    name="reapply_views",
    nctype=NamedCType(name="reapply_views", type=BaseCType(boolT)),
    argument=Argument(
        name="reapply_views", type=BaseType(BaseTy.bool), default=None, annotation=None
    ),
    default=None,
)

# 定义反向 lambda 的返回模式类型为 InverseReturnModeT
InverseReturnModeT = BaseCppType("at::functionalization", "InverseReturnMode")
inverse_return_mode_binding = Binding(
    name="inverse_return_mode",
    nctype=NamedCType(name="inverse_return_mode", type=BaseCType(InverseReturnModeT)),
    argument=Argument(
        name="inverse_return_mode",
        type=BaseType(BaseTy.bool),  # 注意：实际上不是 bool 类型，但这里的类型声明不影响程序逻辑
        default=None,
        annotation=None,
    ),
    default=None,
)

# lambda 捕获部分本身不具有名称。
# 这里返回的名称对应于 lambda 调用的内部函数的名称。
def name(
    g: NativeFunctionsViewGroup,
    *,
    is_reverse: bool,
    # 标志变量，指示是否要进行反向操作，类型为布尔值

    include_namespace: bool,
    # 标志变量，指示是否包含命名空间，类型为布尔值

    reapply_views: bool | None = None,
    # 标志变量，指示是否重新应用视图，可选布尔值或空值，默认为 None
def ) -> str:
    if reapply_views is None:
        # 如果 reapply_views 为 None，则只有在反向 lambda 函数中很重要，
        # 因为我们总是将运行时的 "reapply_views" 参数传递到反向函数中。
        assert is_reverse
    if is_reverse:
        # 如果是反向操作，则调用 reverse_name 函数获取反向操作的名称，包括命名空间
        return reverse_name(g.view, include_namespace)
    # 在正向操作中，直接调用 at::_ops API（因此总是需要命名空间）
    assert include_namespace
    assert g.view_copy is not None
    # 确保 g.view_copy 不为空，然后根据 reapply_views 决定使用 g.view 还是 g.view_copy 的函数名称
    api_name = (
        g.view.func.name.unambiguous_name()
        if reapply_views
        else g.view_copy.func.name.unambiguous_name()
    )
    # 返回组合好的 API 名称字符串
    return f"at::_ops::{api_name}::call"


def reverse_name(f: NativeFunction, include_namespace: bool) -> str:
    # 对于反向操作：将 "reapply_views" 标志传递给该函数，并支持复制和非复制变体。
    # （虽然可以避免这样做，但那将需要编写两倍数量的视图反函数）。
    api_name = f.func.name.unambiguous_name()
    # 在反向情况下，我们为调用点（需要完整命名空间）和声明（不需要）同时进行代码生成
    if include_namespace:
        return f"at::functionalization::FunctionalInverses::{api_name}_inverse"
    else:
        return f"{api_name}_inverse"


def capture_arguments(func: FunctionSchema, *, is_reverse: bool) -> list[Binding]:
    # 捕获参数包括除了 `self` 外的所有参数。
    # 需要注意的是，不包括任何 C++ 引用类型（否则会在捕获时出现悬空引用），
    # 因此任何引用类型（如 IntArrayRef）都需要转换为值类型（vector<int64_t>）。
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    # 获取除了 self 之外的参数列表
    non_self_args = args[1:]
    # 获取非 self 的值类型绑定列表
    non_self_value_bindings = [
        dispatcher.argument(a, remove_non_owning_ref_types=True) for a in non_self_args
    ]

    # 返回所有绑定，包括反向模式绑定或重新应用视图绑定（取决于 is_reverse 参数）
    all_bindings = [
        inverse_return_mode_binding if is_reverse else reapply_views_binding
    ]
    all_bindings.extend(non_self_value_bindings)
    return all_bindings


def returns_type(func: FunctionSchema) -> CType:
    # 断言：所有视图操作都返回类似张量的输出
    assert len(func.returns) >= 1
    for ret in func.returns:
        assert ret.type.is_tensor_like()
    # 然而，lambda 表达式的返回类型始终是单个张量。
    # 对于多张量输出，需要分别跟踪每个张量。
    return BaseCType(tensorT)


def outer_arguments(*, is_reverse: bool) -> list[Binding]:
    if is_reverse:
        # 如果是反向操作，返回基础绑定、变异视图绑定和变异视图索引绑定的列表
        return [base_binding, mutated_view_binding, mutated_view_idx_binding]
    else:
        # 如果是正向操作，只返回基础绑定和变异视图索引绑定的列表
        return [base_binding, mutated_view_idx_binding]


def inner_call_index(func: FunctionSchema) -> Binding | None:
    # 对于返回多个张量的视图操作（如 `split`），我们为每个输出生成单独的 lambda 表达式。
    # 当重新执行返回多个张量的视图操作时，需要适当地索引到输出。
    # （该函数暂未完整提供，没有返回值，可以根据需要补充）
    # 如果函数的返回类型数量大于1，或者仅有一个返回类型且其类型为类似列表的类型
    if len(func.returns) > 1 or (
        len(func.returns) == 1 and func.returns[0].type.is_list_like()
    ):
        # 返回已变异的视图索引绑定对象
        return mutated_view_idx_binding
    # 否则返回空值
    return None
# 定义函数 inner_arguments，接受一个 FunctionSchema 对象和一个布尔型参数 is_reverse，返回一个 Binding 对象列表
def inner_arguments(func: FunctionSchema, is_reverse: bool) -> list[Binding]:
    # 获取函数参数列表
    args = func.arguments.flat_all
    # 断言第一个参数是 BaseTy.Tensor 类型的基本类型
    assert args[0].type == BaseType(BaseTy.Tensor)
    # 去除第一个参数后的其余参数列表
    non_self_args = args[1:]
    
    # 如果不是反向操作
    if not is_reverse:
        # 创建非 self 参数对应的 Binding 对象列表，使用 dispatcher.argument 函数
        non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
        # 返回结果列表，将 base_binding 放在最前面
        return [base_binding] + non_self_bindings
    else:
        # 如果是反向操作
        # 创建非 self 参数对应的 Binding 对象列表，使用 dispatcher.argument 函数
        non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
        
        # 调用 inner_call_index 函数获取 index_binding
        index_binding = inner_call_index(func)
        
        # 如果存在 index_binding
        if index_binding is not None:
            # 返回结果列表，依次包括 base_binding, mutated_view_binding, inverse_return_mode_binding, index_binding 和 non_self_bindings
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
                index_binding,
            ] + non_self_bindings
        else:
            # 返回结果列表，依次包括 base_binding, mutated_view_binding, inverse_return_mode_binding 和 non_self_bindings
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
            ] + non_self_bindings
```