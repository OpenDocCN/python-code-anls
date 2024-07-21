# `.\pytorch\torch\fx\operator_schemas.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类型定义
import torch  # 导入PyTorch库
import inspect  # 导入inspect模块，用于获取对象信息
import numbers  # 导入numbers模块，用于数值类型检查
import types  # 导入types模块，用于类型检查和操作
import typing  # 导入typing模块，支持类型提示
import enum  # 导入enum模块，用于枚举类型的支持
import warnings  # 导入warnings模块，用于警告处理
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING  # 导入多种类型定义

from torch._jit_internal import boolean_dispatched  # 从torch._jit_internal导入boolean_dispatched
from ._compatibility import compatibility  # 导入兼容性处理模块
from torch._ops import OpOverloadPacket, OpOverload  # 从torch._ops导入操作重载相关内容

if TYPE_CHECKING:
    from .node import Argument  # 如果是类型检查阶段，从.node导入Argument类型

__all__ = ["ArgsKwargsPair", "check_for_mutable_operation", "get_signature_for_torch_op", "create_type_hint",
           "type_matches", "normalize_function", "normalize_module"]  # 模块公开的接口列表

@compatibility(is_backward_compatible=False)
class ArgsKwargsPair(NamedTuple):
    """
    Simple named tuple for wrapping args/kwargs pairs.
    """
    args: Tuple[Any, ...]  # 命名元组成员args，表示任意类型的元组
    kwargs: Dict[str, Any]  # 命名元组成员kwargs，表示键为字符串，值为任意类型的字典

_manual_overrides : Dict[Callable, List[inspect.Signature]] = {}  # 空字典，用于存储函数到签名列表的手动覆盖映射

def _nonzero_schemas():
    signatures = []

    def nonzero(self):
        pass
    signatures.append(inspect.signature(nonzero))  # 将nonzero函数的签名添加到signatures列表

    def nonzero(self, *, as_tuple : bool):  # type: ignore[no-redef]
        pass
    signatures.append(inspect.signature(nonzero))  # 将带有关键字参数as_tuple的nonzero函数签名添加到signatures列表

    return signatures  # 返回包含两个函数签名的列表

_manual_overrides[torch.nonzero] = _nonzero_schemas()  # 将torch.nonzero函数与其签名列表映射关联存储在_manual_overrides中

class _FakeGlobalNamespace:
    def __getattr__(self, name):
        if name == 'torch':
            return torch  # 返回torch模块对象
        raise RuntimeError('Expected a torch namespace lookup')  # 若请求的属性不是torch，抛出运行时错误

_type_eval_globals = {'Tensor' : torch.Tensor, 'Device' : torch.device, 'Layout' : torch.layout,
                      'number' : numbers.Number, 'Future' : torch.jit.Future,
                      'AnyEnumType' : enum.Enum, 'QScheme' : torch.qscheme,
                      '__torch__': _FakeGlobalNamespace(), 'NoneType': type(None),
                      'Storage': torch.UntypedStorage,
                      't': typing.TypeVar('t')}  # 包含预定义类型映射的全局字典

for k in dir(typing):
    _type_eval_globals[k] = getattr(typing, k)  # 将typing模块中所有属性及其对应的值添加到_type_eval_globals中

def _torchscript_type_to_python_type(ts_type : 'torch._C.JitType') -> Any:
    """
    Convert a TorchScript type to a Python type (including subtypes) via
    eval'ing the annotation_str. _type_eval_globals sets up expressions
    like "List" and "Future" to map to actual types (typing.List and jit.Future)
    """
    return eval(ts_type.annotation_str, _type_eval_globals)  # 通过评估注释字符串将TorchScript类型转换为Python类型，并利用_type_eval_globals设置类型映射关系

def _torchscript_schema_to_signature_impl(ts_schema : torch._C.FunctionSchema) -> inspect.Signature:
    from inspect import Parameter
    parameters : List[Parameter] = []  # 参数列表初始化为空列表
    # 遍历 TorchScript 模式中的每个参数定义
    for arg in ts_schema.arguments:
        # 将 TorchScript 类型转换为 Python 类型
        arg_type = _torchscript_type_to_python_type(arg.type)
        # 如果参数具有默认值，则获取默认值；否则使用 Parameter.empty 表示没有默认值
        default = arg.default_value if arg.has_default_value() else Parameter.empty
        # 对于参数名为 'self' 的情况，将其标准化为 'input'，以避免后续问题
        name = arg.name if arg.name != 'self' else 'input'
        # 如果参数仅限于关键字传参，则设置为 KEYWORD_ONLY；否则设置为 POSITIONAL_OR_KEYWORD
        kind = Parameter.KEYWORD_ONLY if arg.kwarg_only else Parameter.POSITIONAL_OR_KEYWORD
        
        # 如果参数名为 "from"，则强制其为 POSITIONAL_ONLY 参数
        if name == "from":
            assert kind == Parameter.POSITIONAL_OR_KEYWORD
            # 参数类型 ParameterKind 是 inspect 包的内部实现细节，难以进行类型注释
            kind = Parameter.POSITIONAL_ONLY  # type: ignore[assignment]
            # 将之前所有的参数修改为 POSITIONAL_ONLY 类型
            for idx, p in enumerate(parameters):
                assert p.kind == Parameter.POSITIONAL_OR_KEYWORD
                parameters[idx] = Parameter(name=p.name, kind=Parameter.POSITIONAL_ONLY, default=p.default, annotation=p.annotation)
        
        # 将处理好的参数信息添加到 parameters 列表中
        parameters.append(Parameter(name=name, kind=kind, default=default, annotation=arg_type))
    
    # 提取 TorchScript 模式中每个返回值的类型并转换为 Python 类型
    return_types = [_torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns]
    
    # 根据返回类型数量确定返回值类型
    if len(return_types) == 0:
        return_type = None
    elif len(return_types) == 1:
        return_type = return_types[0]
    else:
        return_type = tuple(return_types)
    
    # 构建并返回完整的函数签名对象
    return inspect.Signature(parameters, return_annotation=return_type)
# 字典，用于缓存从字符串元组到检查器签名的映射关系
_SCHEMA_TO_SIGNATURE_CACHE : Dict[Tuple[str, str], inspect.Signature] = {}

# 将 TorchScript 的函数模式转换为检查器的签名对象
def _torchscript_schema_to_signature(ts_schema : torch._C.FunctionSchema) -> inspect.Signature:
    # 在 FakeTensor 分发的热路径中被调用，因此进行缓存
    cache_key = ts_schema.name, ts_schema.overload_name
    cache_val = _SCHEMA_TO_SIGNATURE_CACHE.get(cache_key)
    if cache_val is not None:
        return cache_val

    # 实际的转换实现函数调用
    res = _torchscript_schema_to_signature_impl(ts_schema)
    _SCHEMA_TO_SIGNATURE_CACHE[cache_key] = res
    return res

# 标记此函数为不向后兼容的兼容性函数
@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(target : Callable, args : Tuple['Argument', ...], kwargs : Dict[str, 'Argument']):
    # 获取目标函数的签名列表和 TorchScript 的函数模式列表
    signatures, schemas = get_signature_for_torch_op(target, return_schemas=True)

    if signatures and schemas:
        matched_schemas = []

        # 遍历所有的模式，直到找到匹配的一个
        # 如果找到匹配的模式，则用新的 args/kwargs 值填充 `new_args_and_kwargs`
        # 如果没有匹配的模式，则 `new_args_and_kwargs` 将为 None
        for candidate_signature, schema in zip(signatures, schemas):
            try:
                candidate_signature.bind(*args, **kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError as e:
                continue

        # 检查是否存在可变操作，并抛出异常
        def throw_if_mutable(schema):
            if schema.is_mutable:
                raise RuntimeError(f'Tried to trace mutable operation {schema}. FX only supports functional '
                                   f'code, so operations that mutate operands in-place (e.g. via `out` arguments) '
                                   f'are not supported')

        if len(matched_schemas) == 0:
            # 没有匹配任何模式，无法检查变异
            pass
        elif len(matched_schemas) == 1:
            # 只匹配了一个模式，是唯一确定的
            _, schema_to_check = matched_schemas[0]
            throw_if_mutable(schema_to_check)
            pass
        else:
            # 匹配到多个模式，由于变异性检查是尽力而为，所以什么也不做
            pass

# 标记此函数为不向后兼容的兼容性函数
@compatibility(is_backward_compatible=False)
def get_signature_for_torch_op(op : Callable, return_schemas : bool = False):
    """
    给定 `torch` 命名空间中的运算符，返回与该运算符的重载对应的 `inspect.Signature` 对象列表。
    如果无法检索到签名，则可能返回 `None`。

    Args:
        op (Callable): 要查找签名的 `torch` 命名空间中的运算符

    Returns:
        Optional[List[inspect.Signature]]: 此运算符的重载的签名列表，如果无法检索到运算符签名，则返回 `None`。
        如果 `return_schemas=True`，则返回一个元组，其中包含可选的 Python 签名和可选的 TorchScript 函数签名
    """
    # 如果操作对象是 OpOverload 类的实例
    if isinstance(op, OpOverload):
        # 将操作对象的 _schema 属性放入列表中作为 schemas
        schemas = [op._schema]
    
    # 如果操作对象是 OpOverloadPacket 类的实例
    elif isinstance(op, OpOverloadPacket):
        # 遍历 op 的所有重载，并将每个重载对应的 _schema 属性放入列表中作为 schemas
        schemas = [getattr(op, overload)._schema for overload in op.overloads()]
    
    # 如果操作对象不属于以上两种类型
    else:
        # 查找 _manual_overrides 字典中是否有操作对象的手动覆盖
        override = _manual_overrides.get(op)
        
        # 如果找到手动覆盖
        if override:
            # 如果需要返回 schemas，则返回覆盖和 None 的元组；否则只返回 None
            return (override, None) if return_schemas else None
        
        # 使用 torch.jit._builtins._find_builtin(op) 查找操作对象对应的内置函数
        aten_fn = torch.jit._builtins._find_builtin(op)
        
        # 如果找不到内置函数
        if aten_fn is None:
            # 如果需要返回 schemas，则返回 None 和 None 的元组；否则返回 None
            return (None, None) if return_schemas else None
        
        # 获取操作对象的所有 schemas
        schemas = torch._C._jit_get_schemas_for_operator(aten_fn)

    # 将每个 schema 转换成 signature，组成列表
    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]
    
    # 如果需要返回 schemas，则返回 signatures 和 schemas 的元组；否则只返回 signatures
    return (signatures, schemas) if return_schemas else signatures
@compatibility(is_backward_compatible=False)
# 定义了一个装饰器 @compatibility，用于标记函数不向后兼容
def create_type_hint(x):
    """
    Produces a type hint for the given argument.

    The :func:`create_type_hint` looks for a type hint compatible with the input argument `x`.

    If `x` is a `list` or `tuple`, it looks for an object in the list whose type is a superclass
    of the rest, and uses that as `base_type` for the `List` or `Tuple` to be returned.
    If no such object is found, it defaults to `List[Any]`.

    If `x` is neither a `list` nor `tuple`, it returns `x`.
    """
    try:
        if isinstance(x, (list, tuple)):
            # 如果 x 是 list 类型
            if isinstance(x, list):
                # 定义一个返回类型函数，使用 List[x] 作为类型提示
                def ret_type(x):
                    return List[x]  # type: ignore[valid-type]
            else:
                # 如果 x 是 tuple 类型，使用 Tuple[x, ...] 作为类型提示
                def ret_type(x):
                    return Tuple[x, ...]
            
            # 如果 x 为空
            if len(x) == 0:
                return ret_type(Any)
            
            # 获取第一个元素作为 base_type
            base_type = x[0]
            
            # 遍历 x 中的每个元素
            for t in x:
                # 如果 t 是 base_type 的子类，继续下一个元素
                if issubclass(t, base_type):
                    continue
                # 如果 base_type 是 t 的子类，更新 base_type
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    # 如果没有找到合适的 base_type，返回默认的 Any 类型提示
                    return ret_type(Any)
            
            # 返回找到的 base_type 对应的类型提示
            return ret_type(base_type)
    except Exception as e:
        # 创建类型提示失败时，发出警告并继续执行
        warnings.warn(f"We were not able to successfully create type hint from the type {x}")
        pass
    
    # 如果 x 不是 list 或 tuple 类型，则直接返回 x
    return x

@compatibility(is_backward_compatible=False)
# 定义了一个装饰器 @compatibility，用于标记函数不向后兼容
def type_matches(signature_type : Any, argument_type : Any):
    # 获取 signature_type 的原始类型
    sig_origin_type = getattr(signature_type, '__origin__', signature_type)

    # 如果 signature_type 等于 argument_type，返回 True
    if signature_type is argument_type:
        return True

    # 如果 signature_type 是 Union 类型，并且不等于 argument_type
    if sig_origin_type is typing.Union and signature_type != argument_type:
        # 获取 Union 类型中包含的所有类型
        sig_contained = signature_type.__args__
        # 检查 argument_type 是否与 Union 类型中的任何一个类型匹配
        return any(type_matches(c, argument_type) for c in sig_contained)

    # 如果 signature_type 是 List[int] 类型，并且 argument_type 是 int 类型
    if signature_type is List[int] and argument_type is int:
        # int 类型可以转换为 List[int]，返回 True
        return True
    # 检查 signature_type 是否具有 __origin__ 属性，并且其值是 list 或 List 中的一种
    if getattr(signature_type, '__origin__', None) in {list, List}:
        # 获取列表元素的类型 sig_el_type
        sig_el_type = signature_type.__args__[0]
        # 如果 sig_el_type 不是类，则发出警告并返回 False
        if not inspect.isclass(sig_el_type):
            warnings.warn(
                f"Does not support nested parametric types, got {signature_type}. Please file a bug.")
            return False
        # 检查 argument_type 是否具有 __origin__ 属性，并且其值是 list 或 List 中的一种
        if getattr(argument_type, '__origin__', None) in {list, List}:
            # 检查 argument_type 中的元素类型是否是 sig_el_type 的子类
            return issubclass(argument_type.__args__[0], sig_el_type)

        # 定义函数检查是否是同质元组
        def is_homogeneous_tuple(t):
            # 如果 t 没有 __origin__ 属性，或者其值不是 tuple 或 Tuple，则返回 False
            if getattr(t, "__origin__", None) not in {tuple, Tuple}:
                return False
            # 获取元组中包含的类型
            contained = t.__args__
            # 特殊情况处理，Tuple[()].__args__ == ((),)
            if t.__args__ == ((),):
                return True
            # 检查元组中每个类型是否是 Ellipsis 或者是 sig_el_type 的子类
            return all((c is Ellipsis) or issubclass(c, sig_el_type) for c in contained)

        # 对于 List[T] 参数，接受 Tuple[T]
        return is_homogeneous_tuple(argument_type)

    # 在模式中，signature_type 是 int 类型
    if signature_type is int and argument_type is torch.dtype:
        return True

    # 如果 signature_type 是 numbers.Number，argument_type 是 int 或 float
    if signature_type is numbers.Number and argument_type in {int, float}:
        return True

    # 如果 argument_type 和 signature_type 都是类，则检查 argument_type 是否是 signature_type 的子类
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)

    # 默认情况，返回 False
    return False
# 定义一个装饰器，指定函数不兼容向后兼容
@compatibility(is_backward_compatible=False)
# 定义函数 `normalize_function`，用于规范化 PyTorch 函数的参数
def normalize_function(
        target: Callable, args: Tuple[Any], kwargs : Optional[Dict[str, Any]] = None, arg_types : Optional[Tuple[Any]] = None,
        kwarg_types : Optional[Dict[str, Any]] = None,
        normalize_to_only_use_kwargs : bool = False) -> Optional[ArgsKwargsPair]:
    """
    返回 PyTorch 函数的规范化参数。这意味着如果 `normalize_to_only_use_kwargs` 为 True，
    `args/kwargs` 将匹配函数的签名，并且仅以位置顺序返回 kwargs。
    同时填充默认值。不支持仅位置参数或可变参数 (*args, **kwargs)。不支持模块。

    可能需要 `arg_types` 和 `kwarg_types` 来消除重载歧义。

    Args:
        target (Callable): 要规范化的函数
        args (Tuple[Any]): 函数的参数元组
        kwargs (Optional[Dict[str, Any]]): 函数的关键字参数字典
        arg_types (Optional[Tuple[Any]]): 参数的类型元组
        kwarg_types (Optional[Dict[str, Any]]): 关键字参数的类型字典
        normalize_to_only_use_kwargs (bool): 是否规范化为仅使用 kwargs。

    Returns:
        返回规范化后的参数和关键字参数，如果不成功则返回 `None`。
    """
    # 如果 kwargs 为 None，则设为一个空字典
    if kwargs is None:
        kwargs = {}
    # 初始化 new_args_and_kwargs 为 None
    new_args_and_kwargs = None
    # 如果 target 不是内置函数类型并且不是 OpOverloadPacket 或 OpOverload 的实例
    if not isinstance(target, types.BuiltinFunctionType) and not (
        isinstance(target, (OpOverloadPacket, OpOverload))
    ):
        # 对于分析使用的目标函数，初始化为 target
        target_for_analysis = target
        # 如果 target 在 boolean_dispatched 中
        if target in boolean_dispatched:
            # HACK: `boolean_dispatch` 在 `torch.nn.functional` 中用于基于布尔值进行两路分发。
            # 在这里我们检查分发的“true”和“false”分支是否具有完全相同的签名。
            # 如果是，使用“true”分支的签名进行分析。否则，保持未规范化状态
            assert not isinstance(target, str)
            dispatched = boolean_dispatched[target]
            if_true, if_false = dispatched['if_true'], dispatched['if_false']
            if inspect.signature(if_true).parameters != inspect.signature(if_false).parameters:
                # 如果真假分支的签名不同，则返回 None
                return None
            # 使用 if_true 作为分析的目标函数
            target_for_analysis = if_true

        # 确保 target_for_analysis 是可调用的
        assert callable(target_for_analysis)
        # 获取 target_for_analysis 解包后的签名
        sig = inspect.signature(inspect.unwrap(target_for_analysis))
        # 调用辅助函数 `_args_kwargs_to_normalized_args_kwargs`，将 args 和 kwargs 规范化为新的参数和关键字参数
        new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(sig, args, kwargs, normalize_to_only_use_kwargs)
    else:
        assert callable(target)
        # 获取目标函数的 Torch 操作签名
        torch_op_schemas = get_signature_for_torch_op(target)
        matched_schemas = []
        if torch_op_schemas:
            # 遍历所有的签名直到找到匹配的签名
            # 如果找到匹配的签名，用新的参数和关键字参数填充 `new_args_and_kwargs`
            # 如果没有匹配的签名，`new_args_and_kwargs` 将为 None
            for candidate_signature in torch_op_schemas:
                try:
                    candidate_signature.bind(*args, **kwargs)
                    matched_schemas.append(candidate_signature)
                except TypeError as e:
                    continue

            if len(matched_schemas) == 0:
                # 没有匹配任何签名，无法规范化
                pass
            elif len(matched_schemas) == 1:
                # 只匹配到一个签名，是明确的
                new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(matched_schemas[0], args, kwargs,
                                                                             normalize_to_only_use_kwargs)
            else:
                if arg_types is not None or kwarg_types is not None:
                    arg_types = arg_types if arg_types else cast(Tuple[Any], ())
                    kwarg_types = kwarg_types if kwarg_types else {}
                    for candidate_signature in torch_op_schemas:
                        sig_matches = True
                        try:
                            bound_types = candidate_signature.bind(*arg_types, **kwarg_types)
                            for arg_name, arg_type in bound_types.arguments.items():
                                param = candidate_signature.parameters[arg_name]
                                sig_matches = sig_matches and type_matches(param.annotation, arg_type)
                        except TypeError as e:
                            sig_matches = False
                        if sig_matches:
                            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(candidate_signature, args, kwargs,
                                                                                         normalize_to_only_use_kwargs)
                            break
                else:
                    # 匹配到多个签名。在这种情况下，调用者必须提供期望的重载参数类型。
                    schema_printouts = '\n'.join(str(schema) for schema in matched_schemas)
                    raise RuntimeError(f'Tried to normalize arguments to {torch.typename(target)} but '
                                       f'the schema match was ambiguous! Please provide argument types to '
                                       f'the normalize_arguments() call. Available schemas:\n{schema_printouts}')

    return new_args_and_kwargs
# 标记函数为不向后兼容的装饰器
@compatibility(is_backward_compatible=False)
# 定义函数 normalize_module，用于规范化 PyTorch 模块的参数
def normalize_module(
        root: torch.nn.Module, target: str, args: Tuple[Any], kwargs : Optional[Dict[str, Any]] = None,
        normalize_to_only_use_kwargs : bool = False) -> Optional[ArgsKwargsPair]:
    """
    Returns normalized arguments to PyTorch modules. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).

    Args:
        root (nn.Module): root module upon which we query modules
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    try:
        # 获取目标函数的子模块
        submod = root.get_submodule(target)
    except AttributeError as e:
        # 抛出运行时错误，如果根模块没有目标子模块
        raise RuntimeError(f"Tried to normalize node with target {target} but root did not "
                           f"have that target!") from e
    # 如果子模块的类具有 '__name__' 属性
    if hasattr(submod.__class__, '__name__'):
        # 获取子模块类的名称
        classname = submod.__class__.__name__
        # 如果子模块类与 torch.nn 模块中的类匹配
        if getattr(torch.nn, classname, None) == submod.__class__:
            # 获取 submod.forward 方法的签名
            sig = inspect.signature(inspect.unwrap(submod.forward))
            # 如果 kwargs 为 None，则设为一个空字典
            if kwargs is None:
                kwargs = {}
            # 将参数和关键字参数规范化为标准化的参数和关键字参数
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(sig, args, kwargs,
                                                                         normalize_to_only_use_kwargs)
            return new_args_and_kwargs
    # 如果条件不满足，返回 None
    return None

def _args_kwargs_to_normalized_args_kwargs(sig : inspect.Signature, args : Tuple[Any, ...],
                                           kwargs : Dict[str, Any],
                                           normalize_to_only_use_kwargs : bool) -> Optional[ArgsKwargsPair]:
    """
    Given a call target, args, and kwargs, return the arguments normalized into
    an ArgsKwargsPair, or None if the type signature is not supported by
    this normalization.

    Args:

        sig (inspect.Signature): Signature object for the target
        args (Tuple): Arguments that appear at the callsite for `target`
        kwargs (Dict): Keyword arguments that appear at the callsite for `target`
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Optional[ArgsKwargsPair]: Normalized args and kwargs for `target`, or `None` if
            this target is not supported.
    """

    # 不支持目前的位置-仅限参数或可变参数 (*args, **kwargs) 签名
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    # 检查函数签名中是否有不支持的参数类型
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        # 如果有不支持的参数类型，针对某个特定签名添加异常处理
        # 该特定签名通常用于 random/uniform 函数，例如：
        # Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None
        # `from` 是 Python 关键字，因此具有这种签名的函数应该有仅限位置参数，但同时它们也可以作为关键字参数进行分派
        if list(sig.parameters.keys()) != ['input', 'from', 'to', 'generator']:
            return None

    # 使用传入的参数绑定到函数签名
    bound_args = sig.bind(*args, **kwargs)
    # 应用默认参数值
    bound_args.apply_defaults()

    # 初始化新的关键字参数和位置参数列表
    new_kwargs : Dict[str, Any] = {}
    new_args : List[Any] = []
    
    # 遍历函数签名中的参数
    for i, param in enumerate(sig.parameters):
        # 如果不要求只使用关键字参数，并且当前参数可以通过位置传递
        if not normalize_to_only_use_kwargs and i < len(args):
            # 将参数值添加到位置参数列表中
            new_args.append(bound_args.arguments[param])
        else:
            # 否则将参数值添加到关键字参数字典中
            new_kwargs[param] = bound_args.arguments[param]

    # 返回经过处理后的位置参数和关键字参数的元组
    return ArgsKwargsPair(tuple(new_args), new_kwargs)
```