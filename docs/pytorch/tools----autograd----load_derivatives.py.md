# `.\pytorch\tools\autograd\load_derivatives.py`

```py
# 解析 derivatives.yaml 文件，生成自动微分函数
#
# 每个自动微分函数由包含一组 Derivative 的 DifferentiabilityInfo 表示。
# 详细数据模型可参考 torchgen.api.autograd。
from __future__ import annotations

import re  # 导入正则表达式模块
from collections import defaultdict  # 导入默认字典集合模块
from typing import Any, Counter, Dict, Sequence, Set, Tuple  # 导入类型提示模块

import yaml  # 导入 YAML 解析模块

from torchgen.api import cpp  # 导入 cpp 模块
from torchgen.api.autograd import (  # 导入自动微分相关模块
    Derivative,
    DifferentiabilityInfo,
    ForwardDerivative,
    SavedAttribute,
)
from torchgen.api.types import (  # 导入类型定义模块
    BaseCType,
    Binding,
    boolT,
    CppSignatureGroup,
    layoutT,
    longT,
    NamedCType,
    OptionalCType,
    scalarTypeT,
    SpecialArgName,
    stringT,
    symIntArrayRefT,
    SymIntT,
    tensorGeometryT,
    tensorOptionsT,
    typeAndSizeT,
    VectorCType,
)
from torchgen.context import with_native_function  # 导入原生函数上下文管理模块
from torchgen.gen import (  # 导入生成代码相关模块
    get_grouped_by_view_native_functions,
    parse_native_yaml,
)
from torchgen.model import (  # 导入模型定义模块
    AUTOGRAD_KEYS,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsViewGroup,
    OperatorName,
    SchemaKind,
    Type,
    Variant,
)
from torchgen.utils import (  # 导入实用工具模块
    concatMap,
    IDENT_REGEX,
    split_name_params,
)
from torchgen.yaml_utils import YamlLoader  # 导入 YAML 加载工具类


DerivativeRet = Tuple[Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], Set[str]]

# 全局缓存字典，用于存储加载的导数信息
_GLOBAL_LOAD_DERIVATIVE_CACHE: dict[tuple[str, str], DerivativeRet] = {}

# 有效的自动微分关键字集合
_VALID_AUTOGRAD_KEYS = set(AUTOGRAD_KEYS)


# 此函数直接为每个 dispatchkey 添加 {view}_copy 变体的导数条目。
# 由于每个 {view} 和 {view}_copy 操作共享相同的导数公式，
# 我们在此处生成它们，而不是在 YAML 中重复定义。
# 详见 Note [Codegen'd {view}_copy Operators]
def add_view_copy_derivatives(
    infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],  # 函数模式到导数信息字典的映射
    view_groups: list[NativeFunctionsViewGroup],  # 原生函数视图组列表
) -> None:
    # 获取每个视图操作名称到其对应视图组的映射
    view_name_to_group: dict[OperatorName, NativeFunctionsViewGroup] = {
        g.view.func.name: g for g in view_groups
    }

    # 初始化视图信息字典
    view_infos = {}
    for info_dispatch_dict in infos.values():
        # 遍历 infos 字典中的每一个值，这些值本身也是字典 info_dispatch_dict
        # maybe_view_group 变量用于保存视图组信息，初始值为 None，每个 info_dispatch_dict 只需计算一次
        maybe_view_group = None
        # view_copy_differentiability_infos 用于存储不同视图复制信息的字典
        view_copy_differentiability_infos = {}
        # 遍历 info_dispatch_dict 中的每一对键值对，dispatch_key 是键，info 是值
        for dispatch_key, info in info_dispatch_dict.items():
            # 获取可能的视图组，根据 info.func.func.name 查找 view_name_to_group 字典
            maybe_view_group = view_name_to_group.get(info.func.func.name, None)
            # 如果找到了 maybe_view_group，并且它的 view_copy 不为 None
            if maybe_view_group is not None and maybe_view_group.view_copy is not None:
                # 根据 maybe_view_group 创建一个视图的复制信息 view_copy_info
                view_copy_info = info.create_view_copy_from_view_derivative(
                    maybe_view_group
                )
                # 如果成功创建了 view_copy_info，则将其添加到 view_copy_differentiability_infos 字典中
                if view_copy_info is not None:
                    fn_schema = view_copy_info.func.func
                    view_copy_differentiability_infos[dispatch_key] = view_copy_info
            else:
                # 如果未找到 maybe_view_group 或其 view_copy 为 None，则跳出内层循环
                break
        # 如果 view_copy_differentiability_infos 中有信息，并且 fn_schema 不在 infos 中
        if len(view_copy_differentiability_infos) > 0 and fn_schema not in infos:
            # 断言确保 fn_schema 不为空
            assert fn_schema is not None
            # 将 view_copy_differentiability_infos 添加到 view_infos 字典中
            view_infos[fn_schema] = view_copy_differentiability_infos

    # 将 view_infos 字典中的信息更新到 infos 字典中
    infos.update(view_infos)
# 加载派生信息的函数，从缓存中返回结果以提高效率
def load_derivatives(
    derivatives_yaml_path: str, native_yaml_path: str, tags_yaml_path: str
) -> DerivativeRet:
    # 在这个确定性函数中进行缓存处理
    global _GLOBAL_LOAD_DERIVATIVE_CACHE
    key = (derivatives_yaml_path, native_yaml_path)
    return _GLOBAL_LOAD_DERIVATIVE_CACHE[key]


# TODO: Why is this going through CppSignatureGroup, that doesn't make sense...
# 使用装饰器将函数注册为本地函数，并获取其CppSignatureGroup
@with_native_function
def cpp_arguments(f: NativeFunction) -> Sequence[Binding]:
    sigs = CppSignatureGroup.from_native_function(f, method=False)
    if sigs.symint_signature is not None:
        return sigs.symint_signature.arguments()
    else:
        return sigs.signature.arguments()


# 创建一个派生对象，计算导数的公式和保存的变量
def create_derivative(
    f: NativeFunction,
    formula: str,
    var_names: tuple[str, ...],
    available_named_gradients: Sequence[str],
) -> Derivative:
    original_formula = formula
    # 获取本地函数的参数并移除const和引用
    arguments: list[NamedCType] = [
        a.nctype.remove_const_ref() for a in cpp_arguments(f)
    ]

    # 修改返回值名称，将self改为result
    return_names = tuple(n if n != "self" else "result" for n in cpp.return_names(f))
    # 获取返回类型并移除const和引用
    return_types = tuple(
        cpp.return_type(r, symint=True).remove_const_ref() for r in f.func.returns
    )

    # 创建以名称和类型为元素的NamedCType对象列表
    named_returns = [
        NamedCType(name, type) for name, type in zip(return_names, return_types)
    ]

    # 对公式应用saved_variables函数，提取保存的输入和输出变量
    formula, saved_inputs = saved_variables(formula, arguments, var_names)
    formula, saved_outputs = saved_variables(formula, named_returns, var_names)

    # 在公式中使用的命名梯度
    used_named_gradients = {
        name
        for name in available_named_gradients
        if re.search(IDENT_REGEX.format(name), formula)
    }

    # 检查公式中引用的梯度索引是否合法
    for i in used_gradient_indices(formula):
        if i >= len(f.func.returns):
            raise RuntimeError(
                f"Out of bounds grads access: derivative formula for {cpp.name(f.func)} "
                f"used grads[{i}], but the forward only returns {len(f.func.returns)} outputs."
            )

    # 返回Derivative对象，包含公式、原始公式、变量名、保存的输入输出和使用的命名梯度
    return Derivative(
        formula=formula,
        original_formula=original_formula,
        var_names=var_names,
        saved_inputs=saved_inputs,
        saved_outputs=saved_outputs,
        named_gradients=used_named_gradients,
    )


# 创建前向导数对象，计算前向导数公式中的变量类型
def create_forward_derivative(
    f: NativeFunction, formula: str, names: tuple[str, ...]
) -> ForwardDerivative:
    var_names = names
    var_types: tuple[Type, ...] | None = None
    for r in f.func.returns:
        if r.name in var_names:
            if var_types is None:
                var_types = tuple()
            var_types = var_types + (r.type,)

    # 处理默认的返回值名称
    # 如果未指定变量类型，则根据变量名推断
    if var_types is None:
        # 如果变量名仅包含("result",)单元素元组
        if var_names == ("result",):
            # 断言函数返回类型数量为1
            assert len(f.func.returns) == 1
            # 设置变量类型为返回类型的单元素元组
            var_types = (f.func.returns[0].type,)
        else:
            # 对每个变量名进行遍历
            for var_name in var_names:
                # 从变量名中匹配以"result"开头的数字部分
                res = re.findall(r"^result(\d+)$", var_name)
                # 如果匹配到一个结果
                if len(res) == 1:
                    # 如果变量类型未指定，则初始化为空元组
                    if var_types is None:
                        var_types = tuple()
                    # 提取匹配的索引号
                    arg_idx = int(res[0])
                    # 将对应返回类型添加到变量类型元组中
                    var_types = var_types + (f.func.returns[arg_idx].type,)

    # 断言确保变量类型不为None，否则抛出异常
    assert var_types is not None, "No matching output for forward derivative definition"
    # 返回ForwardDerivative对象，传入所需参数
    return ForwardDerivative(
        formula=formula,
        var_names=var_names,
        var_types=var_types,
        required_inputs_fw_grad=None,
        required_inputs_primal=None,
        required_original_self_value=False,
        is_reusing_outplace_formula=False,
    )
# 定义一个函数，用于后处理正向导数的派生结果
def postprocess_forward_derivatives(
    f: NativeFunction,
    defn_name: str,
    all_arg_names: list[str],
    derivatives: list[Derivative],
    forward_derivatives: list[ForwardDerivative],
    args_with_derivatives: Sequence[Binding],
) -> list[ForwardDerivative]:
    # 定义内部函数，用于查找正向公式中所需的输入参数
    def find_required_inputs(formula: str, postfix: str) -> tuple[str, ...]:
        # 检查函数名是否以 "_foreach_" 开头
        is_foreach = f.func.name.name.base.startswith("_foreach_")
        # 初始化一个空集合，用于存储必需的输入参数
        required_inputs = set()
        # 遍历所有带导数的参数
        for arg in args_with_derivatives:
            # 如果参数类型是 TensorList 并且不是 foreach 函数，则跳过
            if (
                arg.type in ("at::TensorList", "const at::ITensorListRef &")
                and not is_foreach
            ):
                # 处理 TensorList 的函数内部处理所有内容
                continue
            # 获取参数名
            arg_name = arg.name

            # 在公式中查找参数名的基本名称，如果找到则引发异常
            found = re.search(IDENT_REGEX.format(arg_name), formula)
            if found:
                raise RuntimeError(
                    f"The forward formula for {defn_name} is using the base name of the {arg_name} "
                    f"argument which is ambiguous. You should use {arg_name}_p to access the primal "
                    f"value and {arg_name}_t to access the tangent."
                )

            # 在公式中查找参数名加上后缀的形式，如果找到则将参数名添加到必需输入集合中
            found = re.search(IDENT_REGEX.format(arg_name + postfix), formula)
            if found:
                required_inputs.add(arg_name)

        # 返回必需的输入参数的元组形式
        return tuple(required_inputs)

    # 初始化一个空列表，用于存储更新后的正向导数信息
    updated_derivatives: list[ForwardDerivative] = []

    # 返回更新后的正向导数信息列表
    return updated_derivatives


# 定义一个函数，用于检查是否存在正向导数的定义
def is_forward_derivative_definition(
    all_arg_names: list[str], names: tuple[str, ...]
) -> bool:
    # 遍历给定的名称元组
    for name in names:
        # 如果某个名称不在所有参数名称列表中，则返回 True
        if name not in all_arg_names:
            return True
        else:
            return False
    # 如果循环结束时未返回任何结果，则引发运行时异常
    raise RuntimeError("Expected `names` to be non-empty")


# 定义一个函数，用于创建不同iability_info
def create_differentiability_info(
    defn_dict: dict[Any, Any],
    functions_by_signature: dict[FunctionSchema, list[NativeFunction]],
    functions_by_schema: dict[str, NativeFunction],
    op_counter: Counter[str],
    used_dispatch_keys: set[str],
) -> tuple[FunctionSchema, dict[str, DifferentiabilityInfo]]:
    """Processes a single entry `defn` in derivatives.yaml"""

    # 定义内部函数，用于查找规范函数
    def canonical_function(
        functions: Sequence[NativeFunction], name: str
    ) -> NativeFunction:
        # 遍历给定的函数序列
        for f in functions:
            # 如果函数既不是功能性函数也不是输出函数，并且函数名称与给定名称相同，则返回该函数
            if (
                not f.func.is_functional_fn()
                and not f.func.is_out_fn()
                and name == str(f.func.name.name)
            ):
                return f
        # 如果未找到符合条件的函数，则断言第一个函数名称加下划线的形式与给定名称相同
        assert name + "_" == cpp.name(functions[0].func)
        return functions[0]

    # 定义内部函数，用于将原始名称字符串拆分成名称元组
    def split_names(raw_names: str) -> tuple[str, ...]:
        """Given "foo, bar", return ["foo", "bar"]."""
        return tuple(x.strip() for x in raw_names.split(","))
    # 检查梯度使用情况的函数，用于分析导数定义中的常见错误
    def check_grad_usage(defn_name: str, derivatives: Sequence[Derivative]) -> None:
        """
        Check for some subtle mistakes one might make when writing derivatives.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        """

        uses_grad = False  # true if any derivative uses "grad"
        num_grads_uses = 0  # count of uses of "grads" or "grads[INDEX]"
        uses_named_grads = False  # true if any derivative uses "grad_{name}"
        used_grads_indices: list[int] = []  # which indices of grads are used
        
        # 遍历所有导数对象，分析导数公式中是否出现 "grad" 相关的用法
        for d in derivatives:
            formula = d.formula
            # 检查公式中是否包含 "grad" 关键字
            uses_grad = uses_grad or bool(
                re.findall(IDENT_REGEX.format("grad"), formula)
            )
            # 统计公式中 "grads" 的使用次数
            num_grads_uses += len(re.findall(IDENT_REGEX.format("grads"), formula))
            # 检查导数对象是否使用了命名的梯度
            uses_named_grads = uses_named_grads or bool(d.named_gradients)
            # 获取公式中使用的梯度索引
            used_grads_indices.extend(used_gradient_indices(formula))
        
        # 进行基本的逻辑检查："grads" 的使用次数应不少于出现在公式中的索引数量
        assert num_grads_uses >= len(used_grads_indices)
        
        # 如果 "grads" 的使用次数等于索引数量，则每个 "grads" 都应带有索引
        only_used_grads_indices = num_grads_uses == len(used_grads_indices)

        # 如果出现 "grad" 并且 "grads" 使用次数大于零，则抛出错误
        if uses_grad and num_grads_uses > 0:
            raise RuntimeError(
                f"Derivative definition of {defn_name} in derivatives.yaml illegally "
                "mixes use of 'grad' and 'grads'. Consider replacing "
                "occurrences of 'grad' with 'grads[0]'"
            )

        # 如果只有使用了 "grads[0]" 而且只使用了索引为 0 的情况，则抛出错误
        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError(
                f"Derivative definition of {defn_name} in derivatives.yaml solely "
                "refers to 'grads[0]'.  If the first output is indeed the "
                "only differentiable output, replace 'grads[0]' with 'grad'; "
                "otherwise, there is a likely error in your derivatives "
                "declaration."
            )

        # 如果使用了命名梯度并且同时使用了 "grad" 或者 "grads"，则抛出错误
        if uses_named_grads and (uses_grad or num_grads_uses > 0):
            raise RuntimeError(
                f"Derivative definition of {defn_name} in derivatives.yaml illegally "
                'mixes use of "grad_RETURN_NAME" and "grad" or "grads[x]". Use '
                "only one method for identifying gradients."
            )

    @with_native_function
    # 设置导数的函数装饰器
    def set_up_derivatives(
        f: NativeFunction,
    ) -> tuple[
        Sequence[Derivative],
        Sequence[ForwardDerivative],
        Sequence[Binding],
        Sequence[str],
        Sequence[str],
    # 从 defn 字典中移除 'name' 键
    specification = defn_dict.pop("name")
    # 从规范中获取函数名和参数列表
    defn_name, _ = split_name_params(specification)
    # 从定义字典中移除 'output_differentiability' 键，并获取其值
    # 如果该键不存在，则返回 None，表示所有的输出都是可微分的
    output_differentiability = defn_dict.pop("output_differentiability", None)
    
    # 初始化输出不同iability条件变量为 None
    output_differentiability_conditions = None
    
    # 如果 output_differentiability 存在，并且其中至少有一个元素是字符串类型
    if output_differentiability and any(
        isinstance(diff, str) for diff in output_differentiability
    ):
        # 如果 output_differentiability 中的元素个数不为 1，则抛出运行时错误
        raise RuntimeError(
            f"Not supported: for {specification},"
            f"output_differentiability must either be "
            f"List[bool] or a List[str] where each str is a "
            f"condition. In the case where it is a condition, "
            f"we only support single-output functions. "
            f"Please file us an issue. "
        )
        
        # 将 output_differentiability_conditions 设置为 output_differentiability
        output_differentiability_conditions = output_differentiability
        
        # 将 output_differentiability 设置为包含一个 True 值的列表
        output_differentiability = [True]

    # 根据规范名称获取函数定义
    schema_function = functions_by_schema.get(specification)
    
    # 如果找不到对应的函数定义，则抛出运行时错误
    if not schema_function:
        avail = "\n".join(
            k for k, v in functions_by_schema.items() if cpp.name(v.func) == defn_name
        )
        raise RuntimeError(
            f"could not find ATen function for schema: {specification} "
            f".  Available signatures:\n{avail}"
        )

    # 将函数定义的签名映射到传统的模式；虽然技术上不是必需的，但这里需要一些逻辑
    # 来将原地模式映射到非原地变体。
    # TODO: 或许不再需要处理传统模式的逻辑？
    
    # 获取函数定义的签名
    signature = schema_function.func.signature()
    
    # 根据签名获取对应的函数列表
    functions = functions_by_signature[signature]
    
    # 如果找不到对应的函数，则抛出运行时错误
    if len(functions) == 0:
        avail = "\n".join(
            str(k)
            for k, v in functions_by_signature.items()
            if cpp.name(k) == defn_name
        )
        raise RuntimeError(
            f"could not find ATen function for legacy signature: {signature} "
            f"corresponding to schema {specification}.  Please report a bug to PyTorch. "
            f"Available signatures:\n{avail}"
        )

    # 根据函数列表和定义名称找到规范的函数定义
    canonical = canonical_function(functions, defn_name)
    
    # 如果函数定义的参数中有名为 'grad_input_mask' 的参数，则抛出运行时错误
    if "grad_input_mask" in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(
            f"Schema for {defn_name} has an argument named grad_input_mask, "
            "but this name would be shadowed by our codegen. "
            "Please use a different name in native_functions.yaml."
        )

    # 如果函数定义的参数中有名为 'result' 的参数，则抛出运行时错误
    if "result" in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(
            f"Schema for {defn_name} has an argument named result, "
            "but this is only allowed for outputs."
            "Please use a different name in native_functions.yaml."
        )

    # 初始化 diffinfo_dict 为空字典
    diffinfo_dict = {}
    # 遍历 derivatives.yaml 文件中 "dispatch" 部分的每个键值对
    for key, defn in defn_dict["dispatch"].items():
        # 检查键名是否不是 "Default" 且不在 _VALID_AUTOGRAD_KEYS 中，如果是，则抛出运行时错误
        if key != "Default" and key not in _VALID_AUTOGRAD_KEYS:
            raise RuntimeError(
                f"Invalid dispatch key {key} in derivatives.yaml for {specification},"
                f" expected key to be one of {_VALID_AUTOGRAD_KEYS}"
            )
        
        # 将未使用的 dispatch 键添加到 used_dispatch_keys 集合中
        if key not in used_dispatch_keys:
            used_dispatch_keys.add(key)

        # 准备派生信息的设置，返回多个变量的元组
        (
            derivatives,
            forward_derivatives,
            args_with_derivatives,
            non_differentiable_arg_names,
            available_named_gradients,
        ) = set_up_derivatives(canonical)

        # 创建一个空集合，用于存储使用的命名梯度
        used_named_gradients: set[str] = set()
        # 遍历 derivatives 列表中的每个元素 d，将其命名梯度合并到 used_named_gradients 集合中
        for d in derivatives:
            used_named_gradients |= d.named_gradients

        # 只有在真正计算导数时才分配操作名
        op = None
        if args_with_derivatives:
            # 根据 defn_name 创建操作前缀
            op_prefix = _create_op_prefix(defn_name)
            # 如果键名不是 "Default"，则在操作前缀后添加键名
            if key != "Default":
                op_prefix = op_prefix + key
            # 构建操作名称，包括操作前缀和计数器后缀
            op = f"{op_prefix}{op_counter[op_prefix]}"
            op_counter[op_prefix] += 1

        # 将不同键名对应的 DifferentiabilityInfo 对象添加到 diffinfo_dict 中
        diffinfo_dict[key] = DifferentiabilityInfo(
            name=defn_name,
            func=canonical,
            op=op,
            derivatives=derivatives,
            forward_derivatives=forward_derivatives,
            all_saved_inputs=dedup_vars(
                [v for d in derivatives for v in d.saved_inputs]
            ),
            all_saved_outputs=dedup_vars(
                [v for d in derivatives for v in d.saved_outputs]
            ),
            available_named_gradients=available_named_gradients,
            used_named_gradients=used_named_gradients,
            args_with_derivatives=args_with_derivatives,
            non_differentiable_arg_names=non_differentiable_arg_names,
            output_differentiability=output_differentiability,
            output_differentiability_conditions=output_differentiability_conditions,
        )

    # 返回函数的原始功能和不同键名的导数信息字典
    return canonical.func, diffinfo_dict
# 正则表达式，用于匹配形如 grads[i] 的字符串，提取其中的索引 i
GRAD_INDEX_REGEX = r"(?:^|\W)grads\[(\d+)\]"

def used_gradient_indices(formula: str) -> list[int]:
    """根据公式确定使用的梯度索引列表（grads[i] 中的 i）。

    >>> used_gradient_indices("foo(grads[0], grads[1])")
    [0, 1]
    """
    return [int(i) for i in re.findall(GRAD_INDEX_REGEX, formula)]

def saved_variables(
    formula: str,
    nctypes: list[NamedCType],
    var_names: tuple[str, ...],
) -> tuple[str, tuple[SavedAttribute, ...]]:
    def stride_expr(name: str) -> str:
        assert var_names == (name,), (
            'Replacement for ".strides()" is currently only supported for single derivatives of the same tensor '
            'that ".strides()" is being called on.'
        )
        return f'strides_or_error({name}, "{name}")'

    # 需要保存的变量列表
    saved: list[SavedAttribute] = []

    # 检查公式中是否包含不支持的方法调用，抛出相应的运行时错误
    if ".sizes()" in formula or "->sizes()" in formula:
        raise RuntimeError(
            ".sizes() is not supported in derivative formulas. Instead, please use the SymInt version,"
            + f".sym_sizes(), which returned a c10::SymIntArrayRef. formula={formula}"
        )
    if re.search(r"\.size\([-]?\d+\)", formula) or re.search(
        r"->size\([-]?\d+\)", formula
    ):
        raise RuntimeError(
            ".size(int) is not supported in derivative formulas. Instead, please use the SymInt version,"
            + f".sym_size(int), which returned a c10::SymIntArrayRef. formula={formula}"
        )
    if ".strides()" in formula or "->strides()" in formula:
        raise RuntimeError(
            ".strides() is not supported in derivative formulas. Instead, please use the SymInt version,"
            + f".sym_strides(), which returned a c10::SymIntArrayRef. formula={formula}"
        )
    # 遍历 nctypes 列表中的每一个元素
    for nctype in nctypes:
        # 根据 nctype 的类型确定变量名
        name = (
            nctype.name.name if isinstance(nctype.name, SpecialArgName) else nctype.name
        )
        
        # 首先搜索公式中可以在创建 autograd Function 时计算的表达式，
        # 避免保存变量
        for regex, info in REPLACEMENTS:
            
            # 定义替换函数 repl，将匹配的表达式替换为指定的字符串
            def repl(m: re.Match[str]) -> str:
                # 如果 info["suffix"] 是可调用的，则计算后缀
                suffix: str = (
                    info["suffix"](m) if callable(info["suffix"]) else info["suffix"]
                )
                # 如果定义了 info["expr"]，则使用表达式函数生成表达式，否则使用匹配的原始表达式
                expr: str = info["expr"](name) if "expr" in info else m.group(0)
                # 将表达式及其类型信息保存到 saved 列表中
                saved.append(
                    SavedAttribute(
                        nctype=info["nctype"](name + suffix),
                        expr=expr,
                    )
                )
                # 如果定义了 info["res"]，则用其结果替换匹配的内容，否则返回 name + suffix
                if "res" in info:
                    replacement: str = info["res"](name)
                    return replacement
                return name + suffix
            
            # 使用正则表达式替换公式中匹配 regex.format(name) 的部分
            formula = re.sub(regex.format(name), repl, formula)
        
        # 对于类型为 OptionalCType(BaseCType(stringT)) 的 nctype，
        # 需要将其转换为 c10::optional<c10::string_view>
        if nctype.type == OptionalCType(BaseCType(stringT)):
            formula = re.sub(
                rf"\b{name}\b",
                f"{name}.has_value() ? c10::optional<c10::string_view>({name}.value()) : c10::nullopt",
                formula,
            )
        
        # 查找公式中仍然存在的变量，并将它们保存到 saved 列表中
        if re.search(IDENT_REGEX.format(name), formula):
            saved.append(
                SavedAttribute(
                    nctype=nctype,
                    expr=name,
                )
            )
    
    # 返回经过处理的公式 formula 和保存的变量信息 saved（以元组形式返回）
    return formula, tuple(saved)
# 将原生函数名转换为操作前缀名
def _create_op_prefix(name: str) -> str:
    """Takes a native function name converts to a op prefix name.

    Note that the "name" parameter must be the native function name
    without the optional variant suffix, so "add" instead of
    "add.out".

    OP names correspond to classes, hence the change to title case.

    Example::
    >>> _create_op_prefix('add')
    'AddBackward'
    """
    # 将函数名按下划线分割，每部分首字母大写并拼接起来，形成驼峰命名
    camel_case = "".join([p.title() for p in name.split("_")])
    # 在驼峰命名后添加 "Backward" 后缀，同时处理连续出现的 "ForwardBackward"
    return (camel_case + "Backward").replace("ForwardBackward", "Backward")


# 去除重复的变量，返回一个去重后的变量列表
def dedup_vars(vars: Sequence[SavedAttribute]) -> Sequence[SavedAttribute]:
    seen: set[str] = set()  # 用于存储已经见过的变量名集合
    saved: list[SavedAttribute] = []  # 存储去重后的变量列表
    for var in vars:
        # 获取变量名，处理特殊情况下的名称
        name = (
            var.nctype.name.name  # 如果变量名是特殊参数名，获取其名称
            if isinstance(var.nctype.name, SpecialArgName)
            else var.nctype.name  # 否则直接获取变量名
        )
        if name in seen:  # 如果变量名已存在于集合中，则跳过当前变量
            continue
        seen.add(name)  # 将当前变量名添加到集合中，表示已见过
        saved.append(var)  # 将当前变量添加到去重后的列表中
    return saved  # 返回去重后的变量列表
```