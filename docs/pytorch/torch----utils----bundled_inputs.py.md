# `.\pytorch\torch\utils\bundled_inputs.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs
from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap  # 导入文本包装模块
import torch  # 导入PyTorch模块
from torch._C import TupleType, ListType  # 导入PyTorch内部类型
from torch.jit._recursive import wrap_cpp_module  # 导入PyTorch JIT模块中的函数


T = TypeVar("T")  # 定义泛型变量T

MAX_RAW_TENSOR_SIZE = 16  # 定义最大原始张量大小为16

class InflatableArg(NamedTuple):
    """Helper type for bundled inputs.

    'value' is the compressed/deflated input that is stored in the model. Value
    must be of the same type as the argument to the function that it is a deflated
    input for.

    'fmt' is a formatable code string that is executed to inflate the compressed data into
    the appropriate input. It can use 'value' as an input to the format str. It must result
    in a value of the same type as 'value'.

    'fmt_fn' is a formatable function code string that is executed to inflate the compressed
    data into the appropriate input. It must result in a value of the same type as 'value'.
    The function name should be the formatable part of the string.

    Note: Only top level InflatableArgs can be inflated. i.e. you cannot place
    an inflatable arg inside of some other structure. You should instead create
    an inflatable arg such that the fmt code string returns the full structure
    of your input.
    """
    value: Any  # 定义存储在模型中的压缩/紧缩输入数据
    fmt: str = "{}"  # 定义用于解压数据的可格式化代码字符串，默认为简单的格式化
    fmt_fn: str = ""  # 定义用于解压数据的可格式化函数代码字符串，默认为空字符串


def bundle_inputs(
        model: torch.jit.ScriptModule,
        inputs: Union[Optional[Sequence[Tuple[Any, ...]]], Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]]],
        info: Optional[Union[List[str], Dict[Callable, List[str]]]] = None,
        *,
        _receive_inflate_expr: Optional[List[str]] = None,
) -> torch.jit.ScriptModule:
    """Create and return a copy of the specified model with inputs attached.

    The original model is not mutated or changed in any way.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    If inputs is passed in as a list then the inputs will be bundled for 'forward'.
    If inputs is instead passed in as a map then all the methods specified in the map
    will have their corresponding inputs bundled. Info should match watchever type is
    chosen for the inputs.
    """
    pass  # 定义函数bundle_inputs，创建带有附加输入的模型副本，并返回该副本。不对原始模型进行任何更改。
    """
    The returned model will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions will also be defined on the returned module:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        Alternatively if only bundling inputs for forward the map can be omitted and a singular list of inputs
        can be provided instead.

        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. Alternatively if only bundling inputs for forward the map can be omitted and
    a singular list of information can be provided instead. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """

    # 检查模型是否为 torch.jit.ScriptModule 类型，如果不是则抛出异常
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")  # noqa: TRY002
    # 调用函数 _get_bundled_inputs_attributes_and_methods 获取模型的忽略方法和属性列表
    ignored_methods, ignored_attrs = _get_bundled_inputs_attributes_and_methods(model)
    # 使用 torch._C._hack_do_not_use_clone_module_with_class 函数进行模型的克隆操作
    # 注意：这是一个内部使用的函数，可能不建议直接使用，返回的对象是一个 torch._C.scriptmodule
    clone = torch._C._hack_do_not_use_clone_module_with_class(
        model._c,  # 克隆模型的 C++ 内部表示
        ignored_methods,  # 需要忽略的方法列表
        ignored_attrs,    # 需要忽略的属性列表
    )

    # 上面的克隆函数返回一个 torch._C.scriptmodule，但我们需要一个 torch.jit.scriptmodule
    # 幸运的是，_recursive 模块中有一个函数可以进行这种转换
    cloned_module = wrap_cpp_module(clone)

    # 如果输入 inputs 是一个字典，则调用 augment_many_model_functions_with_bundled_inputs 函数
    # 将 bundled inputs 与克隆的模块的多个模型函数进行增强
    if isinstance(inputs, dict):
        assert isinstance(info, dict) or info is None  # 确保 info 是字典类型或者 None
        augment_many_model_functions_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    else:
        # 如果 inputs 不是字典，则确保 info 是列表类型或者 None
        assert isinstance(info, list) or info is None
        # 调用 augment_model_with_bundled_inputs 函数将 bundled inputs 增强到克隆的模块中
        augment_model_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)

    # 返回增强后的克隆模块
    return cloned_module
# 为模型的 forward 函数添加捆绑的示例输入数据
def augment_model_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Optional[Sequence[Tuple[Any, ...]]] = None,
        _receive_inflate_expr: Optional[List[str]] = None,  # 用于调试的可选参数
        info: Optional[List[str]] = None,  # 可选参数，用于提供关于 forward 函数或其输入的信息
        skip_size_check=False,
) -> None:
    """Add bundled sample inputs to a model for the forward function.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_forward`.
        If the user chooses this method inputs should be None

      - `inputs` is a list of inputs of form List[Tuple[Any, ...]]. A list of tuples where the elements
        of each tuple are the args that make up one input.
    """
    # 检查模型是否是 torch.jit.ScriptModule 类型，否则抛出异常
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")  # noqa: TRY002

    # 获取模型的 forward 方法
    forward: Callable = model.forward

    # 如果 forward 方法没有 __name__ 属性，则为其赋值 'forward'，以防万一
    if not hasattr(forward, "__name__"):
        forward.__name__ = 'forward'

    # 调用函数，为模型的多个函数添加捆绑的输入数据
    augment_many_model_functions_with_bundled_inputs(
        model,
        inputs={forward : inputs},  # 传入 forward 函数及其对应的输入数据
        _receive_inflate_expr=_receive_inflate_expr,
        info={forward : info} if info else None,  # 如果提供了 info 参数，则传入 forward 函数及其信息
        skip_size_check=skip_size_check,
    )


# 为模型的多个公共函数添加捆绑的示例输入数据
def augment_many_model_functions_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]],
        _receive_inflate_expr: Optional[List[str]] = None,  # 用于调试的可选参数
        info: Optional[Dict[Callable, List[str]]] = None,  # 可选参数，用于提供关于函数或其输入的信息
        skip_size_check=False,
) -> None:
    """Add bundled sample inputs to a model for an arbitrary list of public functions.

    Models with bundled inputs can be invoked in a uniform manner by
    """
    # 如果传入的 model 不是 torch.jit.ScriptModule 类型，则抛出异常
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")  # noqa: TRY002

    # 如果未提供任何输入，则抛出异常，要求至少提供一个函数的输入
    if not inputs:
        raise Exception("Please provide inputs for at least 1 function")  # noqa: TRY002
    # 检查模型对象是否已经有了特定属性，如果有则抛出异常
    if hasattr(model, "get_all_bundled_inputs") or hasattr(model, "get_bundled_inputs_functions_and_info"):
        raise Exception(  # noqa: TRY002
            "Models can only be augmented with bundled inputs once. "
            "This Model seems to have already been augmented with "
            "bundled inputs. Please start afresh with one that "
            "doesn't have bundled inputs.",
        )

    # 初始化获取捆绑输入函数和信息的模板字符串
    get_bundled_inputs_functions_and_info_template = ""

    # 定义一些高级辅助方法，这些方法作用于所有捆绑输入
    model.define(textwrap.dedent(f"""
        def get_bundled_inputs_functions_and_info(self):
            # 创建一个空字典用于存储所有捆绑输入的函数和信息
            all_inputs : Dict[str, Dict[str,List[str]]] = {{}}
            # 将预定义的捆绑输入函数和信息模板添加到all_inputs中
            {get_bundled_inputs_functions_and_info_template}
            # 返回所有捆绑输入的函数和信息字典
            return all_inputs
        """))
def _inflate_expr(
    arg: T, ref: str, inflate_helper_fn_name: str, skip_size_check: bool = False
) -> Tuple[Union[T, torch.Tensor], str, Optional[str]]:
    # 允许定制膨胀表达式以处理任意对象。
    # 例如，调用自定义图像解码操作。
    # 或者仅使用 "{}" 作为格式字符串以忽略大小限制。
    
    # 如果参数是 InflatableArg 类型
    if isinstance(arg, InflatableArg):
        # 如果设置了格式化函数
        if arg.fmt_fn:
            # 如果格式不是 "{}" 或空字符串，抛出异常
            if arg.fmt not in ["{}", ""]:
                raise Exception(
                    f"Bundled input argument at position '{ref}' has "
                    f"both arg.fmt_fn => \n{arg.fmt_fn} "
                    f"\n and arg.fmt  => {arg.fmt}. "
                    "Please choose `arg.fmt` if the deflater is straightforward or "
                    "`arg.fmt_fn` if you need a function."
                )

            # 格式化帮助器定义
            helper_definition = arg.fmt_fn.format(inflate_helper_fn_name)
            # 构造表达式
            expr = f"self.{inflate_helper_fn_name}({ref})"

            return arg.value, expr, helper_definition
        else:
            # 直接使用格式化字符串格式化引用
            return arg.value, arg.fmt.format(ref), None

    # 如果参数是 torch.Tensor 类型
    if isinstance(arg, torch.Tensor):
        # 对于小存储张量，可以直接保存
        if arg._typed_storage().size() <= MAX_RAW_TENSOR_SIZE or skip_size_check:
            return arg, ref, None
        # 对于小连续张量，可以克隆为小存储张量
        if arg.is_contiguous() and arg.numel() <= MAX_RAW_TENSOR_SIZE:
            return arg.clone(), ref, None
        # 对于例子输入，通常来自 torch.zeros、torch.ones 或 torch.full
        # 这些可以被紧凑地表示
        for fmt in [torch.contiguous_format, torch.channels_last]:
            if arg.is_contiguous(memory_format=fmt) and (arg == arg.flatten()[0]).all().item():
                return (arg.flatten()[0].clone().expand(*arg.size()),
                        f"{ref}.contiguous(memory_format={fmt})", None)
        # 阻止默认情况下捆绑大张量
        # TODO: 提供更有用的诊断信息
        raise Exception(
            f"Bundled input argument at position '{ref}' is "
            f"a tensor with storage size {arg._typed_storage().size()}. "
            f"You probably don't want to bundle this as an input. "
        )
    else:
        # 对于非特定类型的参数，直接返回原始值和引用
        return arg, ref, None

def _get_bundled_inputs_attributes_and_methods(script_module: torch.jit.ScriptModule) -> Tuple[List[str], List[str]]:
    methods: List[str] = []
    attributes: List[str] = []

    # 如果 script_module 具有 'get_all_bundled_inputs' 属性
    if hasattr(script_module, 'get_all_bundled_inputs'):
        methods.append('get_all_bundled_inputs')
        methods.append('get_num_bundled_inputs')
        methods.append('run_on_bundled_input')
    # 检查脚本模块是否有名为 'get_bundled_inputs_functions_and_info' 的属性
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        # 将方法名 'get_bundled_inputs_functions_and_info' 添加到方法列表中
        methods.append('get_bundled_inputs_functions_and_info')

        # 获取所有捆绑输入函数和信息的字典
        all_info = script_module.get_bundled_inputs_functions_and_info()

        # 遍历所有捆绑输入函数的名称
        for function_name in all_info:
            # 构建方法名并添加到方法列表中
            methods.append("get_all_bundled_inputs_for_" + function_name)
            methods.append("_generate_bundled_inputs_for_" + function_name)

            # 构建属性名并添加到属性列表中
            attributes.append("_bundled_inputs_deflated_" + function_name)

            # 获取特定函数的捆绑输入函数对象
            bundled_inputs_fn = getattr(
                script_module,
                f"get_all_bundled_inputs_for_{function_name}"
            )

            # 获取特定函数的捆绑输入数量
            num_bundled_inputs: int = len(bundled_inputs_fn())

            # 获取特定函数对象
            func = getattr(script_module, function_name)

            # 遍历函数的参数列表（除最后一个参数外）
            for arg_idx in range(len(func.schema.arguments) - 1):
                # 遍历捆绑输入的索引范围
                for input_idx in range(num_bundled_inputs):
                    # 构建解压帮助函数的名称
                    helper_fn_name = _get_inflate_helper_fn_name(
                        arg_idx=arg_idx,
                        input_idx=input_idx,
                        function_name=function_name
                    )

                    # 如果脚本模块具有特定的解压帮助函数，则将其添加到方法列表中
                    if hasattr(script_module, helper_fn_name):
                        methods.append(helper_fn_name)

    # 返回包含方法和属性列表的元组
    return (methods, attributes)
# 根据输入参数的索引、输入数据的索引和函数名生成一个帮助函数名
def _get_inflate_helper_fn_name(
    arg_idx: int,
    input_idx: int,
    function_name: str,
) -> str:
    return f"_inflate_helper_for_{function_name}_input_{input_idx}_arg_{arg_idx}"

# 生成一个使用 torch.randn 填充的张量
def bundle_randn(*size, dtype=None):
    """Generate a tensor that will be inflated with torch.randn."""
    # 创建一个 dtype 类型的零张量，并将其扩展为指定的 size
    stub = torch.zeros(1, dtype=dtype).expand(*size)
    # 返回一个 InflatableArg 对象，其中 value 是生成的零张量，fmt 字段指定为 "torch.randn_like({})"
    return InflatableArg(value=stub, fmt="torch.randn_like({})")

# 封装一个张量，使其可以被打包，无论其大小如何
def bundle_large_tensor(t):
    """Wrap a tensor to allow bundling regardless of size."""
    # 返回一个 InflatableArg 对象，其中 value 是输入的张量 t，fmt 字段为 "{}"
    return InflatableArg(value=t, fmt="{}")
```