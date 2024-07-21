# `.\pytorch\torch\_dynamo\variables\functions.py`

```
# 忽略类型检查错误，这通常是为了兼容性或特定的工具设置
# 在导入所需的模块之前，声明忽略类型检查可能会影响这些模块的类型检查
# 这些模块可能会包含不符合类型注解规范的代码或特定的类型系统变通方法
# 但通常不建议随意忽略类型检查，除非确有必要

import collections  # 导入collections模块，用于特定的数据结构操作
import copy  # 导入copy模块，用于对象的浅复制和深复制操作
import functools  # 导入functools模块，提供了函数操作的高阶功能
import inspect  # 导入inspect模块，用于获取有关对象的信息，如源码和类型信息
import itertools  # 导入itertools模块，提供了高效的迭代工具
import types  # 导入types模块，用于操作Python类型和获取运行时类型信息
from typing import Dict, List, Optional, TYPE_CHECKING, Union  # 导入类型注解需要的模块

import torch  # 导入PyTorch模块，一个用于机器学习的开源深度学习库

from .. import polyfill, variables  # 导入当前目录下的polyfill和variables模块
from ..bytecode_transformation import create_call_function, create_rot_n  # 从bytecode_transformation子模块导入指定函数
from ..exc import unimplemented, Unsupported  # 导入异常模块中的unimplemented和Unsupported异常类
from ..guards import GuardBuilder, install_guard  # 导入guards模块中的GuardBuilder和install_guard
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource  # 从source模块导入多个类
from ..utils import check_constant_args, get_first_attr, identity, istype, make_cell  # 导入utils模块中的多个函数
from .base import MutableLocal, typestr, VariableTracker  # 从当前包中的base模块导入指定的类和函数
from .constant import ConstantVariable  # 从constant模块导入ConstantVariable类

if TYPE_CHECKING:
    from torch._guards import Source  # 在类型检查模式下，从torch._guards导入Source类

try:
    from torch.distributed._composable.fsdp import _fsdp_param_group  # 尝试导入torch.distributed._composable.fsdp模块中的_fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # 如果ModuleNotFoundError异常，则_fsdp_param_group设置为None


def wrap_bound_arg(tx, val, source=None):
    # 如果val是VariableTracker类型，则直接返回val
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        # 如果source为None，则使用SourcelessBuilder创建一个LazyVariableTracker
        from torch._dynamo.variables.builder import SourcelessBuilder
        return SourcelessBuilder.create(tx, val)
    else:
        # 否则，创建一个LazyVariableTracker以避免在__defaults__上进行守卫，除非真正需要
        return variables.LazyVariableTracker.create(val, source)


def wrap_args_kwargs(tx, result):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # 如果v是tuple或dict类型，则对其进行wrap_bound_arg处理
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v)


def init_cellvars(parent, result, code):
    closure_cells = dict()
    side_effects = parent.output.side_effects

    # 遍历code对象的cellvars属性，初始化闭包变量字典closure_cells
    for name in code.co_cellvars:
        closure_cells[name] = side_effects.track_cell_new()
        if name in result:
            # 如果name在result中，则将其存储到side_effects中对应的闭包变量中
            side_effects.store_cell(closure_cells[name], result.pop(name))

    return closure_cells  # 返回初始化后的闭包变量字典closure_cells


def _create_nested_fn(
    code, f_globals, name, defaults, closure, kwdefaults, annotations
):
    from types import FunctionType

    # 使用给定参数创建一个新的FunctionType对象func
    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults  # 设置func的关键字参数默认值
    # 如果annotations是tuple类型，则将其转换为dict类型
    if isinstance(annotations, tuple):
        from itertools import pairwise
        annotations = dict(pairwise(annotations))

    # 断言annotations必须是None或dict类型
    # 否则抛出TypeError异常：__annotations__必须设置为dict对象
    assert annotations is None or isinstance(annotations, dict)
    func.__annotations__ = annotations  # 设置func的注解信息

    return func  # 返回创建的函数对象func


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self):
        # 返回该变量的代码文件名
        return self.get_code().co_filename

    def get_name(self):
        # 返回该变量的函数名
        return self.get_code().co_name

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 调用函数，并返回调用结果
        return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    # 检查对象是否具有指定属性，返回一个变量追踪器对象
    def call_hasattr(self, tx, name: str) -> VariableTracker:
        # 初始化结果为 False
        result = False
    
        try:
            # 尝试检查对象是否具有指定名称的属性
            result = hasattr(self.get_function(), name)
        except NotImplementedError:
            # 如果出现 NotImplementedError 异常
            if name == "__name__" and isinstance(self, NestedUserFunctionVariable):
                # 当属性名为 "__name__" 并且对象是 NestedUserFunctionVariable 的实例时，将结果设置为 True
                result = True
    
        # 返回一个常量变量对象，其值为检查结果
        return variables.ConstantVariable.create(result)
    
    
    # 检查函数的参数名列表并返回
    def inspect_parameter_names(self):
        # 使用 inspect 模块获取函数签名，并返回其参数名列表
        return list(inspect.signature(self.get_function()).parameters)
    
    
    # 返回一个空的闭包变量字典
    def closure_vars(self, tx):
        return {}
class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    # 定义类变量 _nonvar_fields，包含不可变字段名集合
    _nonvar_fields = {
        "fn",
        "is_constant",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value, source):
        # 调用 source 对象的 make_guard 方法创建闭包匹配的守卫
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        # 使用 cls 类创建实例并返回
        return cls(
            value,
            source=source,
        )

    def __init__(self, fn, is_constant=False, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 根据 fn 对象的 _dynamo_marked_constant 属性设置 is_constant 标志
        if getattr(fn, "_dynamo_marked_constant", False):
            # 当 fn 被标记为常量时，设置 self.is_constant 为 True
            self.is_constant = True
        else:
            self.is_constant = False

        # 断言 fn 的类型为 FunctionType 或者 torch.jit.ScriptFunction
        assert isinstance(
            fn, (types.FunctionType, torch.jit.ScriptFunction)
        ), f"expected FunctionType found {typestr(fn)} {fn}"

        # 如果 fn 被 @torch._dynamo.optimize() 包装过，解包获取原始函数
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        # 将解包后的函数赋值给 self.fn
        self.fn: types.FunctionType = fn

    def as_python_constant(self):
        # 如果当前对象是 UserFunctionVariable 类型，则返回 self.fn
        if istype(self, UserFunctionVariable):
            return self.fn
        # 否则调用父类方法返回结果
        return super().as_python_constant()

    def self_args(self):
        # 返回一个空列表，表示没有 self 参数
        return []

    def get_function(self):
        # 返回 self.fn，即当前对象持有的函数对象
        return self.fn

    def get_code(self):
        # 返回 self.fn 的代码对象 __code__
        return self.fn.__code__

    def python_type(self):
        # 返回 types.FunctionType，表示当前对象持有的 Python 函数类型
        return types.FunctionType

    def has_self(self):
        # 检查 self.fn 是否具有 __self__ 属性，有则返回 True，否则返回 False
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self):
        # 返回 self.fn 的全局变量字典 __globals__
        return self.fn.__globals__

    def export_freevars(self, parent, child):
        # 空方法，无实际操作，用于导出自由变量

    def call_hasattr(self, tx, name: str) -> VariableTracker:
        # 检查 self.fn 是否具有指定名称的属性 name
        result = hasattr(self.fn, name)
        # 创建一个 ConstantVariable 对象来存储结果
        return variables.ConstantVariable.create(result)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 如果 self.is_constant 为 True，则调用 invoke_and_store_as_constant 方法执行函数
        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )

        # 否则调用父类的 call_function 方法执行函数
        return super().call_function(tx, args, kwargs)


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        # 调用父类构造函数，并传入 fn 参数
        super().__init__(fn=fn, **kwargs)
        # 设置 self.obj 为传入的 obj 参数
        self.obj = obj

    def __str__(self):
        # 返回对象的字符串表示，包括 fn 和 obj
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self):
        # 返回一个包含 self.obj 的列表，表示方法调用时的 self 参数
        return [self.obj]

    def python_type(self):
        # 返回 types.MethodType，表示当前对象持有的方法类型
        return types.MethodType

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 如果 self.is_constant 为 True，则调用 invoke_and_store_as_constant 方法执行函数
        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )

        # 否则调用父类的 call_function 方法执行函数
        return super().call_function(tx, args, kwargs)
        # 对于 nn.Module 方法，重定向到 NNModuleVariable.call_method 以优化解决方案，
        # 而不是简单地内联。例如，在 FX 图中为 `forward` 方法放置 `call_method` 操作，
        # 因为我们确保允许的模块的 `forward` 可以安全地被 AOT 追踪。
        # 注意，这不仅适用于允许的模块，用户自定义的模块可以从允许的模块扩展，
        # 但使用父类的 `forward` 方法，这也包含在此分支中。

        # 如果我们正在追踪高阶操作，我们希望 Dynamo 进入模块调用，
        # 以便 Dynamo 可以看到底层参数和缓冲区，并将它们作为图的输入引发。
        # is_root_tracer 检查跳过非根追踪器的 if 条件，并在最后直接调用 super().call_function，
        # 这基本上等同于内联方法。
        if tx.output.is_root_tracer() and isinstance(
            self.obj, variables.NNModuleVariable
        ):
            module_attr = getattr(self.fn, "__module__", "")
            # 内联 torch.nn.utils.parametrize
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                and module_attr != "torch.nn.utils.parametrize"
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, args, kwargs, constant=self.is_constant
                )
        
        # 否则，如果 _fsdp_param_group 不为空且 self.fn 是 _fsdp_param_group.FSDPParamGroup.use_training_state
        elif (
            _fsdp_param_group is not None
            and self.fn is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            return variables.TorchCtxManagerClassVariable(self.fn).call_function(
                tx, (self.obj, *args), kwargs
            )
        
        # 如果 self.is_constant 为真
        if self.is_constant:
            fn = getattr(self.obj.value, self.fn.__name__)
            return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
        
        # 否则调用父类的 call_function 方法
        return super().call_function(tx, args, kwargs)
    
    # 检查参数名，返回父类的 inspect_parameter_names 方法的切片结果，从第二个元素开始
    def inspect_parameter_names(self):
        return super().inspect_parameter_names()[1:]
class WrappedUserMethodVariable(UserMethodVariable):
    # 继承自 UserMethodVariable 的包装类，用于封装用户方法变量
    def __init__(self, wrapped, context, **kwargs):
        # 移除 kwargs 中的 "fn" 和 "obj" 键
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        # 调用父类的构造函数，传入 wrapped.fn 和 wrapped.obj，以及其余的 kwargs
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        # 保存被包装的方法对象和上下文对象
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 进入上下文
        self.context.enter(tx)
        # 调用父类的 call_function 方法，传入 tx, args, kwargs，并获取结果
        result = super().call_function(tx, args, kwargs)
        # 退出上下文
        self.context.exit(tx)
        # 返回调用结果
        return result


class WrappedUserFunctionVariable(UserFunctionVariable):
    # 继承自 UserFunctionVariable 的包装类，用于封装用户函数变量
    def __init__(self, wrapped, context, **kwargs):
        # 移除 kwargs 中的 "fn" 和 "obj" 键
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        # 调用父类的构造函数，传入 wrapped.fn 和其余的 kwargs
        super().__init__(wrapped.fn, **kwargs)
        # 保存被包装的函数对象和上下文对象
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 进入上下文
        self.context.enter(tx)
        # 调用父类的 call_function 方法，传入 tx, args, kwargs，并获取结果
        result = super().call_function(tx, args, kwargs)
        # 退出上下文
        self.context.exit(tx)
        # 返回调用结果
        return result


def invoke_and_store_as_constant(tx, fn, name, args, kwargs):
    # 定义一个内部函数 convert，用于将变量转换为 Python 常量
    def convert(x):
        if isinstance(x, variables.TensorVariable):
            return x.get_real_value()
        return x.as_python_constant()

    # 将 args 中的每个元素转换为 Python 常量
    args = [convert(x) for x in args]
    # 将 kwargs 中的每个值转换为 Python 常量
    kwargs = {k: convert(v) for k, v in kwargs.items()}
    # 调用 fn 函数，传入转换后的 args 和 kwargs，获取结果 res
    res = fn(*args, **kwargs)
    # 将结果 res 注册为常量或模块属性，使用给定的 name 和 ConstantSource
    return tx.output.register_attr_or_module(
        res,
        name,
        source=ConstantSource(name),
    )


class NestedUserFunctionVariable(BaseUserFunctionVariable):
    # 继承自 BaseUserFunctionVariable 的嵌套用户函数变量类
    _nonvar_fields = {
        "closure_scope",
        "f_globals",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn_name,
        code,
        f_globals,
        defaults,
        kwdefaults,
        annotations,
        closure,
        closure_scope,
        wrapped_reconstructible=None,
        **kwargs,
    ):
        # 调用父类的构造函数，传入其余的 kwargs
        super().__init__(**kwargs)
        # 断言 fn_name 是字符串类型的 Python 常量
        assert isinstance(fn_name.as_python_constant(), str)
        # 断言 code 是 types.CodeType 类型的 Python 常量
        assert isinstance(code.as_python_constant(), types.CodeType)
        # 初始化实例变量
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.closure = closure
        # 如果 closure 为 None，则将 closure_scope 设为 None
        if closure is None:
            closure_scope = None
        self.closure_scope = closure_scope
        # wrapped_reconstructible 可能是 Source 或 VariableTracker 类型的可重建对象
        self.wrapped_reconstructible: Optional[
            Union[Source, VariableTracker]
        ] = wrapped_reconstructible

    def self_args(self):
        # 返回空列表
        return []

    def get_code(self):
        # 返回 code 的 Python 常量值
        return self.code.as_python_constant()
    # 返回当前对象持有的函数对象
    def get_function(self):
        # 如果有闭包存在，则抛出未实现错误
        if self.closure:
            raise NotImplementedError
        # 使用代码对象创建函数类型对象
        func = types.FunctionType(
            self.code.as_python_constant(),   # 使用代码对象的 Python 表示形式作为函数体
            self.f_globals,                  # 函数的全局变量字典
            self.fn_name.as_python_constant(),  # 使用函数名称的 Python 表示形式作为函数名
        )
        # 如果有默认参数，则设置函数对象的默认参数
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        # 如果有关键字默认参数，则设置函数对象的关键字默认参数
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        # 如果有函数注解，则处理函数对象的注解
        if self.annotations:
            annotations = self.annotations.as_python_constant()
            # 如果注解是元组，则将其转换为字典形式
            if isinstance(annotations, tuple):
                from itertools import pairwise
                annotations = dict(pairwise(annotations))
            # 确保注解最终是字典类型，否则抛出类型错误
            assert isinstance(annotations, dict)
            func.__annotations__ = annotations
        # 返回创建的函数对象
        return func

    # 检查当前对象是否有闭包
    def has_closure(self):
        return self.closure is not None

    # 返回当前对象是否有 self 参数
    def has_self(self):
        return False   # 永远返回 False，表示没有 self 参数

    # 返回当前对象的全局变量字典
    def get_globals(self):
        return self.f_globals
    # 绑定函数参数到实际调用时的值，并处理闭包变量
    def bind_args(self, parent, args, kwargs):
        # 导入必要的模块
        from .misc import InlinedClosureVariable
        
        # 获取当前函数的字节码对象
        code = self.get_code()
        
        # 根据字节码创建函数对象，绑定全局变量和函数名称
        func = types.FunctionType(
            code,
            self.f_globals,
            self.fn_name.as_python_constant(),
            tuple(self.defaults.items) if self.defaults else None,
            tuple(make_cell(None) for _ in range(len(self.get_code().co_freevars))),
        )
        
        # 如果有关键字参数的默认值，则设置函数的 __kwdefaults__ 属性
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.keys_as_python_constant()
        
        # 使用函数签名绑定传入的参数和关键字参数
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        
        # 将绑定的参数转换为字典形式
        result = dict(bound.arguments.items())
        
        # 将参数和关键字参数包装并处理
        wrap_args_kwargs(parent.output.root_tx, result)
        
        # 初始化闭包变量
        closure_cells = init_cellvars(parent, result, code)

        # 处理自由变量的闭包项
        for idx, name in enumerate(code.co_freevars):
            # 获取闭包项的单元格
            cell = self.closure.items[idx]
            
            # 断言闭包单元格与变量名一致
            assert getattr(cell, name, name) == name
            
            # 断言结果中不包含该变量名
            assert name not in result
            
            # 如果是内联闭包变量
            if isinstance(cell, InlinedClosureVariable):
                # 从父级开始查找变量名是否在符号局部变量中
                cand = parent
                while cand and name not in cand.symbolic_locals:
                    cand = cand.parent
                # 如果找不到变量名，则抛出运行时错误
                if cand is None:
                    raise RuntimeError(
                        f"Couldn't find {name} in the symbolic_locals of the inline interpreter stack"
                    )
                # 将找到的符号局部变量值添加到结果中
                result[name] = cand.symbolic_locals[name]
            else:
                # 否则将闭包单元格添加到闭包单元格字典中
                closure_cells[name] = self.closure.items[idx]
        
        # 返回绑定的结果和闭包单元格
        return result, closure_cells

    # 导出自由变量到父级和子级的符号局部变量中
    def export_freevars(self, parent, child):
        # 获取当前函数的字节码对象
        code = self.get_code()
        
        # 遍历自由变量
        for var in code.co_freevars:
            # 如果子级的符号局部变量中包含该变量名，则添加到父级的符号局部变量中
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]
    # 定义一个方法，用于重建代码生成器的内容
    def reconstruct(self, codegen):
        # 添加一个推送空值的指令，加载 _create_nested_fn 函数作为导入
        codegen.add_push_null(
            lambda: codegen.load_import_from(__name__, "_create_nested_fn")
        )
        # 生成当前对象中存储的代码
        codegen(self.code)
        # 扩展输出，加载全局变量
        codegen.extend_output([codegen._create_load_const(self.f_globals)])
        # 加载代码对象的名称
        codegen(ConstantVariable.create(self.code.value.co_name))

        # 如果存在默认参数
        if self.defaults:
            codegen(self.defaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        # 如果存在闭包
        if self.closure:
            codegen(self.closure)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        # 如果存在关键字参数的默认值
        if self.kwdefaults:
            codegen(self.kwdefaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        # 如果存在注解
        if self.annotations:
            try:
                # 尝试将注解转换为 Python 常量，并加载
                annotations = self.annotations.as_python_constant()
                codegen.extend_output([codegen._create_load_const(annotations)])
            except NotImplementedError:
                # 如果无法转换，则直接生成注解
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        # 创建一个函数调用指令，参数为7个，不使用关键字参数
        codegen.extend_output(create_call_function(7, False))

        # 如果存在被包装的可重建对象
        if self.wrapped_reconstructible:
            # 添加一个推送空值的指令，加载 wraps 函数作为导入
            codegen.add_push_null(
                lambda: codegen.load_import_from("functools", "wraps")
            )
            # 生成被包装的可重建对象
            codegen(self.wrapped_reconstructible)
            # 执行函数调用，参数为1个，不使用关键字参数
            codegen.extend_output(create_call_function(1, False))
            # 执行栈顶旋转两次
            codegen.extend_output(create_rot_n(2))
            # 执行函数调用，参数为1个，使用关键字参数
            codegen.extend_output(create_call_function(1, True))
class SkipFunctionVariable(VariableTracker):
    # 定义非变量字段集合，包括"value"和"reason"，以及基类中的所有非变量字段
    _nonvar_fields = {
        "value",
        "reason",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value, reason=None, **kwargs):
        # 调用基类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)
        # 设置当前实例的"value"属性和"reason"属性
        self.value = value
        self.reason = reason

    def python_type(self):
        # 返回当前实例value属性的类型
        return type(self.value)

    def as_python_constant(self):
        # 返回当前实例value属性的值作为 Python 常量
        return self.value

    @classmethod
    def create_with_source(cls, value, source):
        # 调用source对象的make_guard方法创建一个守卫，安装该守卫
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        # 使用给定的value和source参数创建SkipFunctionVariable类的实例并返回
        return cls(
            value,
            source=source,
        )

    @staticmethod
    @functools.lru_cache(None)
    def fold_through_function_to_wrapper():
        # 返回一个字典，包含collections.namedtuple键对应的值为variables.UserDefinedClassVariable
        return {
            collections.namedtuple: variables.UserDefinedClassVariable,
        }

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ):
        # 在tx事务中调用UserFunctionVariable类的call_function方法，
        # 传入当前实例self，一个表示属性的常量变量和args参数列表，以及kwargs参数字典
        return variables.UserFunctionVariable(polyfill.getattr_and_trace).call_function(
            tx, [self, variables.ConstantVariable(self.attr_to_trace), *args], kwargs
        )


class WrapperUserFunctionVariable(VariableTracker):
    """
    Used to represent a wrapper object that contains the actual callable as an
    attribute. For example, torch.jit.script/trace have the original function at
    their _torchdynamo_inline attribute. Similarly, functions with
    __script_if_tracing_wrapper have the original attr at "__original_fn".
    """

    def __init__(self, wrapper_obj, attr_to_trace, **kwargs) -> None:
        # 调用基类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)
        # 设置当前实例的wrapper_obj和attr_to_trace属性
        self.wrapper_obj = wrapper_obj
        self.attr_to_trace = attr_to_trace

    def var_getattr(self, tx, name):
        # 如果name等于attr_to_trace，获取wrapper_obj对象的attr_to_trace属性的值，
        # 如果存在source属性，导入VariableBuilder类，并返回其调用结果
        if name == self.attr_to_trace:
            val = getattr(self.wrapper_obj, self.attr_to_trace)
            if self.source:
                from .builder import VariableBuilder

                return VariableBuilder(tx, AttrSource(self.source, name))(val)
            else:
                from .builder import SourcelessBuilder

                return SourcelessBuilder.create(tx, val)

        # 否则调用基类的var_getattr方法
        return super().var_getattr(tx, name)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 在tx事务中调用UserFunctionVariable类的call_function方法，
        # 传入当前实例self，一个表示属性的常量变量和args参数列表，以及kwargs参数字典
        return variables.UserFunctionVariable(polyfill.getattr_and_trace).call_function(
            tx, [self, variables.ConstantVariable(self.attr_to_trace), *args], kwargs
        )


def _traceable_collective_remaps():
    # 如果torch.distributed可用，从torch.distributed._functional_collectives导入traceable_collective_remaps函数
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    # 否则返回空字典
    return {}


def _traceable_collectives_source(tx, fn):
    # 断言torch.distributed可用，否则引发异常"Illegal invocation."
    assert torch.distributed.is_available(), "Illegal invocation."
    # 断言fn在_traceable_collective_remaps().values()中
    assert fn in _traceable_collective_remaps().values()

    # 获取fn的名称，导入torch.distributed._functional_collectives模块，
    # 并返回由path_source和inner_name构成的AttrSource对象
    inner_name = fn.__name__
    path_source = tx.import_source("torch.distributed._functional_collectives")
    return AttrSource(path_source, inner_name)
    """
    Some of the torch.distributed.* collective APIs are possible to rewrite to 'traceable' collectives.

    This class provides both a way to check if a function is remappable, and perform the remapping.

    In the case that a function is 'remappable' but only for some combinations of call-time arguments,
    we check the args at `call_function` time and fall back to graph-breaking if needed.  This is no worse
    than status-quo as we currently graph-break on all distributed.* collectives.
    """

    # 初始化函数，接受一个函数对象 `fn` 和替换变量 `replacement_var` 作为参数
    def __init__(self, fn, *, replacement_var, **kwargs):
        # 调用父类的初始化方法
        super().__init__(fn, **kwargs)
        # 断言确保 `replacement_var` 是 `UserFunctionVariable` 类的实例
        assert isinstance(replacement_var, UserFunctionVariable)
        # 将 `replacement_var` 赋值给实例变量 `self.replacement_var`
        self.replacement_var = replacement_var

    # 静态方法，用于创建一个 `CollectiveFunctionRewriteVariable` 实例
    @staticmethod
    def create(tx, old_fn, source, **options):
        # 调用 `rewrite` 方法来获取新函数和源代码
        new_fn, new_source = CollectiveFunctionRewriteVariable.rewrite(tx, old_fn)
        # 返回一个 `CollectiveFunctionRewriteVariable` 实例
        return CollectiveFunctionRewriteVariable(
            old_fn,
            replacement_var=UserFunctionVariable(new_fn, source=new_source, **options),
            source=source,
            **options,
        )

    # 静态方法，检查给定的变量是否可以重写
    @staticmethod
    def can_rewrite(variable):
        # 返回结果取决于给定变量是否是函数且在 `_traceable_collective_remaps()` 中
        return (
            inspect.isfunction(variable) and variable in _traceable_collective_remaps()
        )

    # 静态方法，重写函数，返回新的函数和对应的源代码
    @staticmethod
    def rewrite(tx, fn):
        # 使用 `_traceable_collective_remaps()` 获取 `fn` 对应的新函数
        new_fn = _traceable_collective_remaps()[fn]
        # 调用 `_traceable_collectives_source()` 获取新函数 `new_fn` 的源代码
        return new_fn, _traceable_collectives_source(tx, new_fn)

    # 调用函数方法，处理函数调用时的逻辑
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ):
        # 返回类型声明为 "VariableTracker"
        # call_function 必须检查任何不支持的参数并断开图连接。
        # 可以安全地假设 orig_fn 的 args/kwargs 与 remapped_fn 的 args/kwargs 一一对应，
        # 因为这是将映射放入 `traceable_collective_remaps` 的契约。
        import torch.distributed as dist
        from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

        # 将 args 合并到 kwargs 中，以便能够统一处理位置参数和关键字参数。
        signature = inspect.signature(self.fn)
        kwargs = dict(signature.bind(*args, **kwargs).arguments)
        args = ()

        # 如果 kwargs 中有 "async_op" 并且其值可以作为 Python 常量，
        # 则报告无法支持 async_op=True 的错误信息。
        if "async_op" in kwargs and kwargs["async_op"].as_python_constant():
            unimplemented(
                f"CollectiveFunctionRewriteVariable can't support async_op=True for {self.fn}"
            )

        # 如果 self.fn 是以下函数之一，则处理 reduce 操作的参数。
        if self.fn in (
            dist.all_reduce,
            dist.reduce_scatter_tensor,
            dist._reduce_scatter_base,
        ):
            reduce_op_var = kwargs.get("op")
            reduce_op = (
                reduce_op_var.value
                if reduce_op_var is not None
                else signature.parameters["op"].default
            )
            # 如果 reduce_op 不在 REDUCE_OP_TO_STR 中，则抛出错误。
            if reduce_op not in REDUCE_OP_TO_STR:
                raise ValueError(f"Unsupported all_reduce op: {reduce_op}")
            # 将 kwargs 中的 "op" 参数替换为 REDUCE_OP_TO_STR 中对应的常量变量。
            kwargs["op"] = variables.ConstantVariable.create(
                REDUCE_OP_TO_STR[reduce_op]
            )
        
        # 调用替代变量的 call_function 方法，传入 tx, args, kwargs 参数，
        # 返回结果作为函数的结果。
        return self.replacement_var.call_function(tx, args, kwargs)
class FunctoolsPartialVariable(VariableTracker):
    # 继承自VariableTracker的类，用于跟踪变量

    def __init__(self, func: VariableTracker, args, keywords, **kwargs):
        # 初始化方法，接受一个VariableTracker类型的func，一个列表args和一个字典keywords作为参数
        super().__init__(**kwargs)
        # 调用父类的初始化方法

        self.func = func
        # 将func参数赋值给实例变量func
        assert isinstance(args, list)
        # 断言args是一个列表
        self.args = args
        # 将args赋值给实例变量args
        assert isinstance(keywords, dict)
        # 断言keywords是一个字典
        self.keywords = keywords
        # 将keywords赋值给实例变量keywords

    def reconstruct(self, codegen):
        # 重建方法，接受一个codegen对象作为参数

        codegen.add_push_null(lambda: codegen.load_import_from("functools", "partial"))
        # 向codegen对象中添加推送空值的指令，并通过lambda加载"functools"中的"partial"模块

        codegen(self.func)
        # 调用codegen对象传入的func对象

        if self.args:
            codegen.foreach(self.args)
            # 如果args不为空，则对args中的每个元素执行遍历操作

        if not self.keywords:
            codegen.extend_output(create_call_function(len(self.args) + 1, False))
            # 如果keywords为空，则在codegen中扩展输出，创建一个函数调用指令，参数个数为len(self.args) + 1

            return

        codegen.foreach(self.keywords.values())
        # 对keywords字典中的每个值执行遍历操作

        keys = tuple(self.keywords.keys())
        # 将keywords的键转换为元组赋值给keys

        codegen.extend_output(
            codegen.create_call_function_kw(len(keys) + len(self.args) + 1, keys, False)
        )
        # 在codegen中扩展输出，创建一个带关键字参数的函数调用指令，参数个数为len(keys) + len(self.args) + 1

    def get_function(self):
        # 获取函数方法，返回self的Python常量表示
        return self.as_python_constant()

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 调用函数方法，接受一个tx对象，一个VariableTracker类型的args列表，一个字典类型的kwargs参数，返回一个VariableTracker对象

        merged_args = self.args + args
        # 将self.args和args合并为merged_args列表
        merged_kwargs = {**self.keywords, **kwargs}
        # 将self.keywords和kwargs合并为merged_kwargs字典

        return self.func.call_function(tx, merged_args, merged_kwargs)
        # 调用self.func的call_function方法，传入tx、merged_args和merged_kwargs参数，并返回结果

    def call_hasattr(self, tx, name: str) -> VariableTracker:
        # 调用hasattr方法，接受一个tx对象和一个字符串类型的name参数，返回一个VariableTracker对象

        # functools.partial使用槽，因此属性是常量
        return variables.ConstantVariable.create(
            hasattr(functools.partial(identity), name)
        )
        # 返回一个创建的常量VariableTracker对象，判断functools.partial(identity)是否具有name属性

    def as_python_constant(self):
        # 作为Python常量返回方法，返回self的Python常量表示
        return functools.partial(
            self.func.as_python_constant(),
            *[arg.as_python_constant() for arg in self.args],
            **{k: v.as_python_constant() for k, v in self.keywords.items()},
        )
        # 返回一个functools.partial对象，其中包含self.func、self.args和self.keywords的Python常量表示

    def guard_as_python_constant(self):
        # 作为Python常量的保护方法，类似于as_python_constant()，但添加了ID_MATCH guards以强制转换为常量

        return functools.partial(
            self.func.guard_as_python_constant(),
            *[v.guard_as_python_constant() for v in self.args],
            **{k: v.guard_as_python_constant() for k, v in self.keywords.items()},
        )
        # 返回一个functools.partial对象，其中包含self.func、self.args和self.keywords的保护的Python常量表示
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, kernel, kernel_idx, grid, **kwargs):
        # 从triton.runtime.autotuner模块导入Autotuner类
        from triton.runtime.autotuner import Autotuner

        # 从torch._higher_order_ops.triton_kernel_wrap模块导入kernel_side_table
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 断言kernel不为None
        assert kernel is not None

        # 将kernel保存到当前对象的属性中
        self.kernel = kernel
        # 将kernel添加到kernel_side_table中，并保存索引到kernel_idx属性中
        self.kernel_idx = kernel_side_table.add_kernel(kernel)

        # 断言如果kernel_idx不为None，则其值必须等于self.kernel_idx
        assert kernel_idx is None or self.kernel_idx == kernel_idx

        # 将grid参数保存到当前对象的grid属性中
        self.grid = grid

        # 如果kernel是Autotuner类的实例
        if isinstance(kernel, Autotuner):
            # 只支持triton.autotune的configs和keys参数
            # 确保其他参数都使用默认值
            defaults = inspect.signature(Autotuner.__init__).parameters

            # 新版本的triton将属性名从warmup改为num_warmup，rep改为num_rep
            # 使用get_first_attr函数来保持向后兼容性
            if (
                ("warmup" in defaults and defaults["warmup"].default != get_first_attr(kernel, "num_warmups", "warmup"))
                or ("rep" in defaults and defaults["rep"].default != get_first_attr(kernel, "num_reps", "rep"))
                or ("prune_configs_by" in defaults and defaults["prune_configs_by"].default != kernel.early_config_prune)
                or len(kernel.reset_idx) != 0
                or len(kernel.restore_idx) != 0
            ):
                # 抛出Unsupported异常，说明只支持triton.autotune的configs和keys参数
                raise Unsupported("Only configs and keys are supported for triton.autotune")
    ) -> "VariableTracker":
        # 定义方法的返回类型为 "VariableTracker"
        if name == "__getitem__":
            # 如果调用的方法名为 "__getitem__"
            # 只有在没有网格数据或参数不为1时才应调用 __getitem__
            if self.grid is not None or len(args) != 1:
                # 如果已经有了网格数据或参数个数不等于1，则抛出异常
                raise Unsupported(
                    "Triton kernels should be called with only a single grid"
                )

            # 返回一个 TritonKernelVariable 对象，表示要执行的操作是通过索引访问内核
            return TritonKernelVariable(
                kernel=self.kernel,
                kernel_idx=self.kernel_idx,
                grid=args[0],
            )
        elif name == "run":
            # 如果调用的方法名为 "run"
            if "grid" not in kwargs:
                # 如果关键字参数中没有 "grid"，则抛出异常
                raise Unsupported("Triton kernel requires to be called with a grid")
            # 从关键字参数中取出 "grid" 并移除
            grid = kwargs.pop("grid")
            # 移除关键字参数中的 "warmup"，如果存在的话
            kwargs.pop("warmup", None)
            # 重写调用方式 kernel.run(*args, grid=grid) 为 kernel[grid](*args)
            # 返回一个 TritonKernelVariable 对象，表示要执行的是通过 grid 执行内核
            return TritonKernelVariable(
                kernel=self.kernel, kernel_idx=self.kernel_idx, grid=grid
            ).call_function(tx, args, kwargs)

        # 如果以上条件都不满足，则执行父类的方法调用
        return super().call_method(tx, name, args, kwargs)
```