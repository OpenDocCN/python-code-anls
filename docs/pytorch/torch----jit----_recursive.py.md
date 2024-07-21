# `.\pytorch\torch\jit\_recursive.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import collections  # 导入collections模块
import functools  # 导入functools模块
import inspect  # 导入inspect模块
import sys  # 导入sys模块
import textwrap  # 导入textwrap模块
import types  # 导入types模块
import warnings  # 导入warnings模块
from typing import Dict, List, Set, Type  # 从typing模块导入类型注解相关内容

import torch  # 导入torch库

import torch._jit_internal as _jit_internal  # 导入torch._jit_internal模块
from torch._sources import fake_range  # 从torch._sources导入fake_range
from torch.jit._builtins import _find_builtin  # 从torch.jit._builtins导入_find_builtin函数
from torch.jit._check import AttributeTypeIsSupportedChecker  # 从torch.jit._check导入AttributeTypeIsSupportedChecker类
from torch.jit._state import _add_script_class, _get_script_class, _python_cu  # 从torch.jit._state导入相关函数
from torch.jit.frontend import (
    get_class_properties,  # 从torch.jit.frontend导入get_class_properties函数
    get_default_args,  # 从torch.jit.frontend导入get_default_args函数
    get_jit_class_def,  # 从torch.jit.frontend导入get_jit_class_def函数
    get_jit_def,  # 从torch.jit.frontend导入get_jit_def函数
)
from torch.nn import Module  # 从torch.nn导入Module类


ScriptMethodStub = collections.namedtuple(
    "ScriptMethodStub", ("resolution_callback", "def_", "original_method")
)  # 定义ScriptMethodStub命名元组，包含resolution_callback、def_、original_method字段
PropertyStub = collections.namedtuple("PropertyStub", ("resolution_callback", "def_"))  # 定义PropertyStub命名元组，包含resolution_callback、def_字段


# TODO: there should be a more principled way of doing this.
# 忽略的属性列表，这里应该有一个更加原则性的方式来处理
ignored_attributes = [
    "_version",  # 版本属性
    "_parameters",  # 参数属性
    "_buffers",  # 缓冲区属性
    "_non_persistent_buffers_set",  # 非持久性缓冲区集合属性
    "_backward_hooks",  # 反向钩子属性
    "_backward_pre_hooks",  # 反向预钩子属性
    "_forward_hooks",  # 前向钩子属性
    "_forward_hooks_with_kwargs",  # 带关键字参数的前向钩子属性
    "_forward_pre_hooks",  # 前向预钩子属性
    "_forward_pre_hooks_with_kwargs",  # 带关键字参数的前向预钩子属性
    "_forward_hooks_always_called",  # 总是调用的前向钩子属性
    "_state_dict_hooks",  # 状态字典钩子属性
    "_state_dict_pre_hooks",  # 状态字典预钩子属性
    "_load_state_dict_pre_hooks",  # 加载状态字典前钩子属性
    "_load_state_dict_post_hooks",  # 加载状态字典后钩子属性
    "_modules",  # 模块属性
    "_initializing",  # 初始化属性
    "dump_patches",  # 转储补丁属性
]


def _compile_and_register_class(obj, rcb, qualified_name):
    # 获取对象的脚本类表示
    script_class = _get_script_class(obj)

    if not script_class:
        # 如果不存在脚本类，则获取对象的JIT类定义AST和默认参数
        ast = get_jit_class_def(obj, obj.__name__)
        defaults = torch.jit.frontend.get_default_args_for_class(obj)
        # 编译脚本类，并注册到对象中
        script_class = torch._C._jit_script_class_compile(
            qualified_name, ast, defaults, rcb
        )
        _add_script_class(obj, script_class)

    return script_class  # 返回编译后的脚本类


def make_stub(func, name):
    # 创建函数存根，使用函数的闭包创建解析回调
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    # 获取函数的JIT定义AST
    ast = get_jit_def(func, name, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)  # 返回ScriptMethodStub对象


def make_stub_from_method(nn_module, method_name):
    # 获取模块中指定方法的函数对象
    func = getattr(nn_module, method_name)
    if isinstance(func, ScriptMethodStub):
        return func
    # 确保生成的AST中的名称与请求的名称匹配
    return make_stub(func, method_name)  # 返回函数的存根对象


def make_stubs_from_exported_methods(mod):
    # 从导出方法创建存根对象列表
    stubs = []
    for name in dir(mod):
        item = getattr(mod, name, None)
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.EXPORT
        ):
            stubs.append(make_stub_from_method(mod, name))  # 如果方法被导出，则创建存根对象并添加到列表中

    return stubs  # 返回存根对象列表


def jit_ignored_properties(module):
    # 返回模块中忽略的属性列表
    pass  # 空函数，仅作为占位符使用
    # 获取指定模块中被用户标记忽略的属性列表（如果有的话）
    user_annotated_ignored_attributes = getattr(
        module, "__jit_ignored_attributes__", list()
    )
    
    # 定义一个函数，用于获取给定模块中的所有属性名（包括 property 类型的属性）
    def get_properties_names(module):
        return {k for k, v in vars(module).items() if isinstance(v, property)}
    
    # 获取给定模块中所有的属性名（包括 property 类型的属性）
    properties = get_properties_names(type(module))
    
    # 存储用户标记忽略的属性名的集合
    user_annoted_ignored_properties = set()
    
    # 遍历用户标记忽略的属性列表，并将存在于模块属性中的属性名添加到集合中
    for ignored_attr in user_annotated_ignored_attributes:
        if ignored_attr in properties:
            user_annoted_ignored_properties.add(ignored_attr)
    
    # 返回最终的用户标记忽略的属性名集合
    return user_annoted_ignored_properties
# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
# 定义了一组基本类型，这些类型可以作为常量
# 此外，这些基本类型的元组和列表也被视为常量
# 如果你修改了这个列表，则还需要编辑 jit/script/init.cpp 中的 ConstantValue 处理程序
_constant_types = (
    bool,
    float,
    int,
    str,
    type(None),
    torch.device,
    torch.layout,
    torch.dtype,
)


def _get_valid_constant(attr, v, owner_type):
    # 检查 v 是否是 _constant_types 中的某种类型，如果是则返回 v 本身
    if isinstance(v, _constant_types):
        return v
    # 如果 v 是 tuple 或者 list，则递归检查其中的每个元素是否是有效常量类型，并返回元组类型
    elif isinstance(v, (tuple, list)):
        return tuple(_get_valid_constant(attr, x, owner_type) for x in v)
    # 如果 v 不是有效的常量类型，则抛出 TypeError 异常
    constants = ", ".join(torch.typename(typ) for typ in _constant_types)
    raise TypeError(
        textwrap.dedent(
            f"""
        '{torch.typename(type(v))}' object in attribute '{owner_type}.{attr}' is not a valid constant.
        Valid constants are:
        1. a nn.ModuleList
        2. a value of type {{{constants}}}
        3. a list or tuple of (2)
        """
        )
    )


class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):
    # 继承自 torch._C._jit_tree_views.SourceRangeFactory，表示源码的上下文信息
    def __init__(self, source, filename, file_lineno, leading_whitespace_len):
        # 初始化方法，设置源码的相关信息
        super().__init__(source, filename, file_lineno, leading_whitespace_len)


def get_annotations(obj):
    # 返回对象的类型注解字典（如果存在）
    if sys.version_info < (3, 10):
        return getattr(obj, "__annotations__", {})
    # 在 Python 3.10+ 中推荐使用 inspect.get_annotations 函数
    # 参考文档：https://docs.python.org/3.10/howto/annotations.html
    # 但在 Python 3.10 中，未注解的派生类不会继承基类的注解，因此需要手动提取
    annotations = inspect.get_annotations(obj)
    if annotations:
        return annotations

    def get_cls_annotations(cls):
        # 递归获取类及其基类的类型注解字典
        cls_annotations = inspect.get_annotations(cls)
        if cls_annotations:
            return cls_annotations
        for base in cls.__bases__:
            cls_annotations = get_cls_annotations(base)
            if cls_annotations:
                return cls_annotations
        return {}

    # 如果 obj 是类对象，则获取其类型注解；否则获取其类型的类型注解
    cls = obj if isinstance(obj, type) else type(obj)
    return get_cls_annotations(cls)


def infer_concrete_type_builder(nn_module, share_types=True):
    """
    Build a ConcreteModuleTypeBuilder from an nn.Module.

    This ConcreteModuleType doesn't have a JIT type associated with it yet, it
    must be filled in by the caller.
    """
    # 根据 nn.Module 构建一个 ConcreteModuleTypeBuilder 对象
    concrete_type_builder = torch._C.ConcreteModuleTypeBuilder(type(nn_module))
    if isinstance(nn_module, (torch.nn.ModuleDict)):
        concrete_type_builder.set_module_dict()
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential)):
        concrete_type_builder.set_module_list()
    if isinstance(nn_module, (torch.nn.ParameterList)):
        concrete_type_builder.set_parameter_list()
    if isinstance(nn_module, (torch.nn.ParameterDict)):
        concrete_type_builder.set_parameter_dict()

    # 获取 nn_module 的类注解信息
    class_annotations = get_annotations(nn_module)
    # 检查 nn_module 是否是 QuantWrapper 类型的实例
    if isinstance(nn_module, (torch.ao.quantization.QuantWrapper)):
        # 如果是，初始化一个空的类注解字典
        class_annotations = {}

    # 获取用户标注的忽略属性列表
    user_annotated_ignored_attributes = getattr(
        nn_module, "__jit_ignored_attributes__", list()
    )
    # 将用户标注的忽略属性添加到 concrete_type_builder 的忽略属性集合中
    concrete_type_builder.add_ignored_attributes(user_annotated_ignored_attributes)
    # 获取 nn_module 的 JIT 忽略属性
    ignored_properties = jit_ignored_properties(nn_module)

    # 尝试通过类型注解或对象本身推断类型
    def infer_type(name, item):
        # Module 的 forward 函数特殊处理，不使用该注解，直接使用 JIT 推断类型
        inferred = False
        try:
            if (
                name in class_annotations
                and class_annotations[name]
                != torch.nn.Module.__annotations__["forward"]
            ):
                # 将类注解转换为类型，并使用 fake_range() 进行推断
                ann_to_type = torch.jit.annotations.ann_to_type(
                    class_annotations[name], fake_range()
                )
                attr_type = torch._C.InferredType(ann_to_type)
            elif isinstance(item, torch.jit.Attribute):
                # 如果 item 是 torch.jit.Attribute 类型，则推断其类型
                ann_to_type = torch.jit.annotations.ann_to_type(item.type, fake_range())
                attr_type = torch._C.InferredType(ann_to_type)
            else:
                # 否则尝试使用 JIT 推断类型
                attr_type = torch._C._jit_try_infer_type(item)
                inferred = True
        except RuntimeError as re:
            # 捕获推断类型时可能发生的 RuntimeError
            raise RuntimeError(f"Error inferring type for {name}: {item}: {re}") from re

        return attr_type, inferred

    # 初始化一个空集合，用于记录已添加的属性名称
    added_names = set()

    # 遍历 nn_module 的参数 items
    for name, item in nn_module._parameters.items():
        # 如果属性名在用户标注的忽略属性列表中，则跳过本次循环
        if name in user_annotated_ignored_attributes:
            continue

        # 断言属性 item 要么为 None，要么为 torch.Tensor 类型
        assert item is None or isinstance(item, torch.Tensor)
        # 使用 infer_type 函数推断属性的类型，并将属性添加到 concrete_type_builder 中
        attr_type, _ = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), True, False)
        # 将属性名添加到已添加名称集合中
        added_names.add(name)

    # 遍历 nn_module 的缓冲区 items
    for name, item in nn_module._buffers.items():
        # 如果属性名在用户标注的忽略属性列表中，则跳过本次循环
        if name in user_annotated_ignored_attributes:
            continue

        # 断言属性 item 要么为 None，要么为 torch.Tensor 类型
        assert item is None or isinstance(item, torch.Tensor)
        # 使用 infer_type 函数推断属性的类型，并将属性添加到 concrete_type_builder 中
        attr_type, _ = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type.type(), False, True)
        # 将属性名添加到已添加名称集合中
        added_names.add(name)
    # 遍历神经网络模块(nn_module)中的所有子模块及其名称
    for name, item in nn_module._modules.items():
        # 如果名称在用户标记的被忽略属性列表(user_annotated_ignored_attributes)中，则跳过处理
        if name in user_annotated_ignored_attributes:
            continue

        # 推断当前子模块(item)的类型(attr_type)和是否可推断成功(_)
        attr_type, _ = infer_type(name, item)
        
        # 如果子模块(item)为None，则处理为NoneType属性
        if item is None:
            # 模块可以是None。我们没有直接支持可选模块，因此将其注册为NoneType属性。
            concrete_type_builder.add_attribute(name, attr_type.type(), False, False)
            continue
        
        # 如果能够推断出子模块的类型
        if attr_type.success():
            assert attr_type.type().is_interface_type()
            # 如果类型可以被推断，则应该是一个模块接口类型
            sub_concrete_type = torch._C.ConcreteModuleType.from_jit_type(
                attr_type.type()
            )
        else:
            # 否则，获取子模块的具体模块类型，并添加到concrete_type_builder中
            sub_concrete_type = get_module_concrete_type(item, share_types)
        
        # 将子模块添加到concrete_type_builder中
        concrete_type_builder.add_module(name, sub_concrete_type)

        # 记录已添加的名称
        added_names.add(name)

    # 填充常量集合(constants_set)
    constants_set = set(getattr(nn_module, "__constants__", ()))

    # 处理通过`Final[T]`注释的常量，而不是将其添加到`__constants__`中
    for name, ann in class_annotations.items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    # 遍历常量集合，处理每个常量
    for name in constants_set:
        # 如果常量名已经添加过，则发出警告，并继续下一个常量
        if name in added_names:
            # TODO: 在这种情况下，我们真的应该报错，但这会破坏向后兼容性，因此我们至少需要在一个版本中发出警告
            if name in nn_module._modules:
                hint = "submodule"
            elif name in nn_module._buffers:
                hint = "buffer"
            elif name in nn_module._parameters:
                hint = "parameter"
            else:
                raise AssertionError(
                    "added_names must be submodule, parameter, or buffer"
                )

            # 发出警告，指出常量在ScriptModule中被发现，但它不是常量类型。建议删除它。
            warnings.warn(
                f"'{name}' was found in ScriptModule constants, "
                f" but it is a non-constant {hint}. Consider removing it."
            )
            continue
        
        # 如果nn_module中没有这个常量名，则发出警告，并继续下一个常量
        if not hasattr(nn_module, name):
            # TODO: 在这种情况下，我们真的应该报错，但这会破坏向后兼容性，因此我们至少需要在一个版本中发出警告
            warnings.warn(
                f"'{name}' was found in ScriptModule constants, "
                "but was not actually set in __init__. "
                "Consider removing it."
            )
            continue
        
        # 获取常量的值
        value = getattr(nn_module, name)
        
        # 将常量添加到concrete_type_builder中
        concrete_type_builder.add_constant(
            name, _get_valid_constant(name, value, type(nn_module).__name__)
        )
        
        # 记录已添加的名称
        added_names.add(name)

    # 填充重载集合(overloads)
    overloads = getattr(nn_module, "__overloads__", {})
    # 更新重载集合，使用任何已注释的重载
    overloads.update(
        get_overload_name_mapping(
            get_overload_annotations(nn_module, ignored_properties)
        )
    )
    # 遍历重载映射字典中的每个项目，将每个重载的名称添加到具体类型构建器中
    for name, overloaded_names in overloads.items():
        concrete_type_builder.add_overload(name, overloaded_names)

    # 将神经网络模块中注册的每个 forward hook 添加到具体类型构建器中
    for hook in nn_module._forward_hooks.values():
        concrete_type_builder.add_forward_hook(hook)

    # 将神经网络模块中注册的每个 forward pre-hook 添加到具体类型构建器中
    for pre_hook in nn_module._forward_pre_hooks.values():
        concrete_type_builder.add_forward_pre_hook(pre_hook)

    # 返回构建完成的具体类型构建器实例
    return concrete_type_builder
class ConcreteTypeStore:
    # 存储不同 Python 模块类型到其对应的 JIT 类型列表的字典
    type_store: Dict[Type[Module], List[torch._C.ConcreteModuleType]]
    # 存储已经编译过方法的 ConcreteModuleType 的集合
    methods_compiled: Set[torch._C.ConcreteModuleType]

    def __init__(self):
        # 初始化 type_store 为空字典
        self.type_store = {}
        # 初始化 methods_compiled 为空集合
        self.methods_compiled = set()

    def get_or_create_concrete_type(self, nn_module):
        """从 nn.Module 实例推断出一个 ConcreteType。尽可能重用底层的 JIT 类型。"""
        # 推断出一个 ConcreteTypeBuilder
        concrete_type_builder = infer_concrete_type_builder(nn_module)

        # 获取 nn_module 的类型
        nn_module_type = type(nn_module)
        # 如果 type_store 中没有这个类型，创建一个空列表
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []

        # 在 type_store 中查找已有的 JIT 类型
        known_types = self.type_store[nn_module_type]
        for known_type in known_types:
            # 如果找到与 concrete_type_builder 相等的已知类型，返回它
            if known_type.equals(concrete_type_builder):
                return known_type

        # 如果没有找到，从 concrete_type_builder 构建一个新的 JIT 类型
        concrete_type = concrete_type_builder.build()
        # 将新生成的 JIT 类型添加到 type_store 中
        self.type_store[nn_module_type].append(concrete_type)
        return concrete_type


concrete_type_store = ConcreteTypeStore()


def create_methods_and_properties_from_stubs(
    concrete_type, method_stubs, property_stubs
):
    # 获取方法的定义、解析回调和默认参数
    method_defs = [m.def_ for m in method_stubs]
    method_rcbs = [m.resolution_callback for m in method_stubs]
    method_defaults = [get_default_args(m.original_method) for m in method_stubs]

    # 获取属性的定义和解析回调
    property_defs = [p.def_ for p in property_stubs]
    property_rcbs = [p.resolution_callback for p in property_stubs]

    # 调用 concrete_type 对象的方法来创建方法和属性
    concrete_type._create_methods_and_properties(
        property_defs, property_rcbs, method_defs, method_rcbs, method_defaults
    )


def create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs):
    # 获取钩子和预钩子的定义和解析回调
    hook_defs = [h.def_ for h in hook_stubs]
    hook_rcbs = [h.resolution_callback for h in hook_stubs]

    pre_hook_defs = [h.def_ for h in pre_hook_stubs]
    pre_hook_rcbs = [h.resolution_callback for h in pre_hook_stubs]

    # 调用 concrete_type 对象的方法来创建钩子和预钩子
    concrete_type._create_hooks(hook_defs, hook_rcbs, pre_hook_defs, pre_hook_rcbs)


def get_module_concrete_type(nn_module, share_types=True):
    """
    获取 nn_modules 的具体类型。

    如果 share_types 是 True，则从 concrete_type_store 中获取具体类型。
    如果是 False，则直接创建一个新的具体类型，不首先搜索 concrete_type_store。

    Args:
        nn_module: 要为其创建 ScriptModule 的原始 Python nn.Module。
        share_types: 是否在模块之间共享底层 JIT 类型（如果可能）。

    Returns:
        nn_module 的具体类型。
    """
    # 断言 nn_module 是 Module 类的实例
    assert isinstance(nn_module, Module)
    # 如果 nn_module 是 ScriptModule 并且有 _concrete_type 属性，返回它的具体类型
    if isinstance(nn_module, torch.jit.ScriptModule) and hasattr(
        nn_module, "_concrete_type"
    ):
        return nn_module._concrete_type
    # 如果存在分享类型（share_types 列表非空），执行以下操作：
    # 从缓存的 JIT 类型存储中获取或创建给定 nn_module 对应的具体类型。
    if share_types:
        concrete_type = concrete_type_store.get_or_create_concrete_type(nn_module)
    # 如果 share_types 列表为空，则执行以下操作：
    else:
        # 推断给定 nn_module 的具体类型构建器，并设置其为 poisoned 状态。
        concrete_type_builder = infer_concrete_type_builder(nn_module, share_types)
        concrete_type_builder.set_poisoned()
        # 构建具体类型并将其赋值给 concrete_type。
        concrete_type = concrete_type_builder.build()
    
    # 返回最终确定的具体类型 concrete_type。
    return concrete_type
def create_script_class(obj):
    """
    Create and return a RecursiveScriptClass instance from a Python object.

    Arguments:
        obj: A Python object.
    """
    # 获取对象类型的限定类名
    qualified_class_name = _jit_internal._qualified_name(type(obj))
    # 为对象类型创建类方法的解析回调
    rcb = _jit_internal.createResolutionCallbackForClassMethods(type(obj))
    # 如果对象类型尚未进行脚本化，则进行编译和注册
    _compile_and_register_class(type(obj), rcb, qualified_class_name)
    # 获取对象类型对应的脚本化类型
    class_ty = _python_cu.get_class(qualified_class_name)
    # 使用脚本化类型创建一个空的 torch._C.ScriptObject 对象
    cpp_object = torch._C._create_object_with_type(class_ty)
    # 将对象的所有属性复制到 torch._C.ScriptObject 中
    for name, value in obj.__dict__.items():
        cpp_object.setattr(name, value)

    # 将 torch._C.ScriptObject 包装在 RecursiveScriptClass 实例中并返回
    return wrap_cpp_class(cpp_object)


def create_script_module(nn_module, stubs_fn, share_types=True, is_tracing=False):
    """
    Create a new ScriptModule from an nn.Module.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
        share_types:  Whether to share underlying JIT types between modules (if possible).
            NOTE: Only set to False this when we cannot guarantee type sharing will work
                correctly. This only happens today for traced modules, where the same
                module can produce different traced methods depending on the inputs.
        is_tracing: Whether this function is called during tracing or scripting. If tracing,
                we don't need to do AttributeTypeIsSupportedChecker because all the unsupported
                attributes will be baked as constant in the tracing graph. In addition,
                this check significantly slows down the traced modules when the module size is big.
    """
    # 断言输入的 nn.Module 不是 torch.jit.RecursiveScriptModule 的实例
    assert not isinstance(nn_module, torch.jit.RecursiveScriptModule)
    # 检查 nn.Module 是否已初始化
    check_module_initialized(nn_module)
    # 获取模块的具体类型，用于创建脚本模块
    concrete_type = get_module_concrete_type(nn_module, share_types)
    # 如果不是在追踪状态下，执行属性类型支持检查
    if not is_tracing:
        AttributeTypeIsSupportedChecker().check(nn_module)
    # 调用具体的创建脚本模块的实现函数
    return create_script_module_impl(nn_module, concrete_type, stubs_fn)


def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
    # 使用具体类型创建一个 torch._C.ScriptModule 对象
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)
    # 生成模块的方法存根
    method_stubs = stubs_fn(nn_module)
    # 获取模块的属性存根
    property_stubs = get_property_stubs(nn_module)
    # 获取模块的钩子存根和预钩子存根
    hook_stubs, pre_hook_stubs = get_hook_stubs(nn_module)
    # 获取 nn_module 中的 "__jit_ignored_attributes__" 属性，如果不存在则默认为空列表
    user_annotated_ignored_attributes = getattr(
        nn_module, "__jit_ignored_attributes__", list()
    )
    
    # 调用 jit_ignored_properties 函数获取 nn_module 中忽略的属性列表
    ignored_properties = jit_ignored_properties(nn_module)

    # 定义初始化函数 init_fn，用于初始化 ScriptModule：
    def init_fn(script_module):
        # 1. 将原始 nn_module 中的属性/参数/缓冲复制到新的 ScriptModule 中。
        for name in concrete_type.get_attributes().keys():
            orig_value = getattr(nn_module, name)
            # 如果 orig_value 是 torch.jit.Attribute 类型，则取其 value 属性作为原始值
            orig_value = (
                orig_value.value
                if isinstance(orig_value, torch.jit.Attribute)
                else orig_value
            )
            # 将原始值设置到 cpp_module 对象的对应属性名下
            cpp_module.setattr(name, orig_value)

        # 2. 将原始 nn_module 中的子模块复制到新的 ScriptModule 中，并递归地对其进行脚本化。
        for name, sub_concrete_type in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            # 检查原始值是否为 Module 类型
            assert isinstance(
                orig_value, Module
            ), f"Expected Module but got {type(orig_value)}"
            # 获取子模块的类型
            module_type = sub_concrete_type.jit_type
            if isinstance(module_type, torch._C.InterfaceType):
                # 使用接口推断规则编译模块
                scripted = interface_script(module_type, orig_value)
            elif isinstance(orig_value, torch.jit.ScriptModule):
                # 如果原始值已经是 ScriptModule，则直接使用
                scripted = orig_value
            else:
                # 使用提供的 stubs_fn 推断要编译的方法
                scripted = create_script_module_impl(
                    orig_value, sub_concrete_type, stubs_fn
                )

            # 将脚本化的子模块设置到 cpp_module 对象的对应属性名下
            cpp_module.setattr(name, scripted)
            # 将脚本化的子模块保存到 script_module 的 _modules 字典中
            script_module._modules[name] = scripted

        # 3. 将原始 nn_module 中的 @ignored/@unused 方法和属性复制到新的 ScriptModule 中，
        #    以便在 ScriptModule 上访问这些 Python 方法。
        for name in dir(nn_module):
            if name in ignored_properties:
                continue
            item = getattr(nn_module, name, None)
            if inspect.ismethod(item) and _jit_internal.is_ignored_fn(item):
                # 如果 item 是被忽略的函数，则获取其未绑定版本并绑定到 script_module 上
                unbound_function = getattr(nn_module, name).__func__
                bound_method = unbound_function.__get__(script_module)
                setattr(script_module, name, bound_method)
            elif concrete_type.is_ignored_attribute(name):
                # 如果属性名在被忽略的属性列表中，则直接将其设置到 script_module 上
                setattr(script_module, name, item)

        # 将具体类型 concrete_type 附加到新的 ScriptModule 中以便访问
        script_module._concrete_type = concrete_type

    # 实际创建 ScriptModule，使用刚刚定义的初始化函数
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

    # 如果需要，编译方法
    # 如果具体类型不在已编译方法存储中
    if concrete_type not in concrete_type_store.methods_compiled:
        # 从存根中创建方法和属性到具体类型
        create_methods_and_properties_from_stubs(
            concrete_type, method_stubs, property_stubs
        )
        # 创建钩子，确保钩子和方法之间没有名称冲突
        # 如果在方法之前完成，钩子可能会遮蔽未导出的方法
        create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs)
        # 运行 Torch 的模块发射钩子
        torch._C._run_emit_module_hook(cpp_module)
        # 将具体类型添加到已编译方法集合中
        concrete_type_store.methods_compiled.add(concrete_type)

    # 复制前向钩子和前钩子到新的 ScriptModule
    # 允许从 eager 模式下作为 ScriptFunction 运行钩子
    for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
        script_module._forward_pre_hooks[idx] = fn
    for idx, fn in enumerate(script_module._c._get_forward_hooks()):
        script_module._forward_hooks[idx] = fn

    # 特殊处理，使像 __len__ 这样的方法能在派生自容器类的脚本方法中正常工作
    if (
        isinstance(
            nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)
        )
        and "__len__" not in cpp_module._method_names()
    ):
        # 在脚本模块中定义 __len__ 方法
        script_module.define(f"def __len__(self):\n   return {len(nn_module)}\n")
    
    # 如果是 ModuleDict 类型且 __contains__ 方法不在 C++ 模块的方法名中
    if (
        isinstance(nn_module, torch.nn.ModuleDict)
        and "__contains__" not in cpp_module._method_names()
    ):
        # 如果 nn_module 有键
        if len(nn_module.keys()):
            keys = repr(list(nn_module.keys()))
            # 在脚本模块中定义 __contains__ 方法
            script_module.define(
                f"def __contains__(self, key: str):\n   return key in {keys}\n"
            )
        else:
            # 如果 nn_module 没有键，定义返回 False 的 __contains__ 方法
            script_module.define("def __contains__(self, key: str):\n   return False\n")

    # 将编译的方法添加到 Python ScriptModule 类中
    for method_stub in method_stubs:
        if method_stub.original_method is None:
            # 如果是使用 define() 定义的方法，不需要进行 Python 包装处理
            continue

        name = method_stub.original_method.__name__
        if name != method_stub.def_.name().name:
            # 跳过名称不匹配的方法，因为 @torch.jit._overload_method 会修改函数名
            continue
        script_method = cpp_module._get_method(name)

        # 包装原始方法以传播文档字符串等信息
        wrapped_script_method = functools.wraps(method_stub.original_method)(
            script_method
        )

        # 将方法直接添加到 script_module 中，确保在查找 `name` 时优先找到
        script_module.__dict__[name] = wrapped_script_method

    # 将模块属性添加到 Python ScriptModule 类中
    # 对每个属性存根进行迭代处理
    for property_stub in property_stubs:
        # 获取属性名称
        property_name = property_stub.def_.name().name
        # 获取属性的 getter 方法
        fget = cpp_module._get_method(property_stub.def_.getter_name().name)
        # 检查是否存在 setter 方法（setter 是可选的，可能不存在）
        setter_name = property_stub.def_.setter_name()
        fset = cpp_module._get_method(setter_name.name) if setter_name else None
        # 将属性名及其对应的 getter 和 setter 方法（如果存在）添加到 script_module 的字典中作为属性
        script_module.__dict__[property_name] = property(property_name, fget, fset)  # type: ignore[arg-type]

    # 如果在 nn_module 中定义了但在 script_module 中未定义的 Python 方法，将其复制到 script_module 中
    # 这是当前仅在模块容器上使用的内部 API
    for name in dir(nn_module):
        # 如果属性名在忽略列表中，则跳过
        if name in ignored_properties:
            continue
        # 获取属性值
        item = getattr(nn_module, name, None)
        # 如果属性被标记为需要复制到脚本化模型的 Python 方法
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
        ):
            # 将 Python 方法添加到脚本化模型中
            add_python_attr_to_scripted_model(script_module, nn_module, name)

    # 返回脚本化的模块 script_module
    return script_module
# 定义一个函数，用于检查脚本模型是否定义了特定的属性，并且该属性不是其默认的属性
def script_model_defines_attr(script_model, attr):
    # 获取脚本模型中指定属性的当前值
    script_attr = getattr(script_model, attr, None)
    # 如果属性不存在，则返回 False
    if script_attr is None:
        return False
    # 获取默认的属性值，即递归脚本模块中指定属性的默认值
    default_attr = getattr(torch.jit.RecursiveScriptModule, attr, None)
    # 如果默认属性不存在，则返回 False
    if default_attr is None:
        return False
    # 返回判断脚本模型中属性是否不等于其默认属性的布尔值
    return script_attr != default_attr


# 将原始模型（orig）的特定属性添加到脚本化模型（script_model）中
def add_python_attr_to_scripted_model(script_model, orig, attr):
    # 如果原始模型具有指定的属性，并且脚本模型确实定义了该属性，则将原始属性添加到脚本模型中
    if hasattr(orig, attr) and script_model_defines_attr(script_model, attr):
        setattr(script_model, attr, getattr(orig, attr))


# 获取模块中的重载函数的注解信息
def get_overload_annotations(mod, jit_ignored_properties):
    # 初始化一个空字典，用于存储重载函数及其相关信息
    overloads = {}

    # 遍历模块类型中的所有属性名
    for name in dir(type(mod)):
        # 如果属性名在被忽略的属性列表中，则跳过
        if name in jit_ignored_properties:
            continue
        # 获取属性对应的对象
        item = getattr(mod, name, None)
        # 如果对象不是可调用的，则跳过
        if not callable(item):
            continue

        # 检查对象是否具有 __module__ 属性，并且该属性不为 None
        if hasattr(item, "__module__") and item.__module__ is not None:
            # 获取对象的重载方法信息
            method_overloads = _jit_internal._get_overloaded_methods(
                item, mod.__class__
            )
            # 如果未找到重载方法信息，则继续下一轮循环
            if method_overloads is None:
                continue

            # 检查是否存在未实现的重载方法
            if item.__func__ in method_overloads:
                raise RuntimeError(
                    _jit_internal.get_overload_no_implementation_error_message(
                        "method", item.__func__
                    )
                )

            # 为每个重载方法生成一个唯一的名称，并将其与方法函数对应起来存储
            names = [name + "__" + str(i) for i in range(len(method_overloads))]
            overloads[item] = list(zip(names, method_overloads))

    # 返回所有重载函数及其相关信息的字典
    return overloads


# 获取重载函数的名称映射关系
def get_overload_name_mapping(overload_info):
    # 初始化一个空字典，用于存储重载函数及其名称映射关系
    overload_name_mappings: Dict[str, List[str]] = {}
    
    # 遍历重载函数信息字典中的每个原始函数及其对应的重载函数列表
    for orig_fn, overloads in overload_info.items():
        # 获取原始函数的名称
        original_name = orig_fn.__name__
        # 如果映射中尚未包含该函数名，则添加该函数名并初始化一个空列表
        if original_name not in overload_name_mappings:
            overload_name_mappings[original_name] = []

        # 遍历该函数的每个重载函数，并将其名称添加到映射关系中
        for overload_name, _ in overloads:
            overload_name_mappings[original_name].append(overload_name)
    
    # 返回所有原始函数及其重载函数名称映射关系的字典
    return overload_name_mappings


# 检查函数是否缺少类型签名注解
def _check_no_signature(func):
    # 获取函数的类型签名
    signature = torch.jit.annotations.get_signature(
        func, None, fake_range(), inspect.ismethod(func)
    )
    # 如果未找到类型签名，则抛出运行时错误
    if signature is None:
        qual_name = _jit_internal._qualified_name(func)
        raise RuntimeError(
            f"Must explicitly add type annotations to overloaded functions: {qual_name}"
        )


# 生成重载函数的存根（stub）
def make_stubs_for_overloads(overload_info):
    # 初始化一个空列表，用于存储所有重载函数的存根
    overload_stubs = []
    # 遍历 overload_info 字典中的每个原始函数及其重载信息
    for orig_fn, overloads in overload_info.items():
        # 获取原始函数的 JIT 定义（抽象语法树）
        orig_ast = get_jit_def(
            orig_fn, orig_fn.__name__, self_name="RecursiveScriptModule"
        )
        # 遍历当前原始函数的所有重载函数及其名称
        for overload_name, overload_fn in overloads:
            # 检查重载函数是否没有签名
            _check_no_signature(overload_fn)
            # 获取重载函数的 JIT 定义（抽象语法树）
            over_ast = get_jit_def(
                overload_fn, overload_fn.__name__, self_name="RecursiveScriptModule"
            )
            # 替换重载方法的声明，并创建新的抽象语法树
            new_ast = torch._C._replace_overloaded_method_decl(
                over_ast.decl(), orig_ast, overload_name
            )
            # 从闭包中创建解析回调函数
            _rcb = _jit_internal.createResolutionCallbackFromClosure(orig_fn)
            # 将重载方法的存根（包含解析回调、新的抽象语法树和重载函数）添加到列表中
            overload_stubs.append(ScriptMethodStub(_rcb, new_ast, overload_fn))
    # 返回所有重载方法的存根列表
    return overload_stubs
def check_module_initialized(mod):
    # 断言模块是 torch.nn.Module 的实例
    assert isinstance(mod, torch.nn.Module)
    # 检查模块是否具有 "_parameters" 属性
    if not hasattr(mod, "_parameters"):
        # 如果没有初始化，则抛出运行时错误
        raise RuntimeError(
            f"'{torch.typename(type(mod))}' has not been initialized, did you forget to call 'super()'?"
        )

    # 避免导入 torch.distributed.nn
    # 检查模块是否具有 "remote_parameters" 属性
    if not hasattr(mod, "remote_parameters"):
        # 遍历模块的参数，检查是否有延迟加载的参数
        for name, param in mod._parameters.items():
            if param is not None and torch.nn.parameter.is_lazy(param):
                # 如果有延迟加载的参数，则抛出运行时错误
                raise RuntimeError(
                    f"'{torch.typename(type(mod))}' has uninitialized parameters {name}. Did you forget to run a forward pass?"
                )
        # 遍历模块的缓冲区，检查是否有延迟加载的缓冲区
        for name, buf in mod._buffers.items():
            if buf is not None and torch.nn.parameter.is_lazy(buf):
                # 如果有延迟加载的缓冲区，则抛出运行时错误
                raise RuntimeError(
                    f"'{torch.typename(type(mod))}' has uninitialized buffers {name}. Did you forget to run a forward pass?"
                )


def infer_methods_to_compile(nn_module):
    """Implement the default rules for which methods should act as starting points for compilation.

    (TODO add a link when the rules are published).
    """
    # 检查模块是否已经初始化
    check_module_initialized(nn_module)
    # 获取用户注释的忽略属性列表
    user_annotated_ignored_attributes = getattr(
        nn_module, "__jit_ignored_attributes__", list()
    )
    # 获取需要忽略的属性列表
    ignored_properties = jit_ignored_properties(nn_module)

    # 初始化方法列表
    methods: List[str] = []
    # 检查模块是否具有 "forward" 方法且未被忽略
    if hasattr(nn_module, "forward") and not _jit_internal.is_ignored_fn(
        nn_module.forward
    ):
        forward_func = getattr(nn_module.forward, "__func__", None)
        module_forward = getattr(torch.nn.Module, "forward", None)
        # 如果模块的 forward 方法不是继承自 torch.nn.Module 的 forward 方法，则加入方法列表
        if forward_func != module_forward:
            methods = ["forward"]

    # 导出的方法列表
    exported = []
    # 遍历模块的所有属性
    for name in dir(nn_module):
        # 如果属性在忽略列表中，则跳过
        if name in ignored_properties:
            continue
        # 获取属性值
        item = getattr(nn_module, name, None)
        # 检查属性是否标记为导出的 TorchScript 修饰符
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.EXPORT
        ):
            exported.append(name)

    # 将导出的方法加入到方法列表中
    methods = methods + exported

    # 获取方法重载的名称映射
    overload_name_mappings = dict(getattr(nn_module, "__overloads__", {}))
    # 获取方法重载的信息
    overload_info = get_overload_annotations(nn_module, ignored_properties)
    # 更新方法重载的名称映射
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    # 为方法重载生成存根
    overload_stubs = make_stubs_for_overloads(overload_info)

    # 将方法重载名称映射存储到模块的 "__overloads__" 属性中
    nn_module.__overloads__ = overload_name_mappings

    # 过滤掉已重载的方法，只保留未重载的方法
    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings

    filtered_methods = filter(ignore_overloaded, methods)

    # 去重方法列表，保证方法的唯一性
    # 我们不使用集合来存储方法，以保证编译顺序的确定性
    uniquer: Set[str] = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)
    # 创建一个空列表，用于存储生成的 stubs（存根）对象
    stubs = []
    # 遍历去重后的方法列表 uniqued_methods
    for method in uniqued_methods:
        # 调用 make_stub_from_method 函数生成当前方法 method 的存根，并添加到 stubs 列表中
        stubs.append(make_stub_from_method(nn_module, method))
    # 返回 overload_stubs（过载存根）与生成的 stubs（存根）列表的组合结果
    return overload_stubs + stubs
def get_hook_stubs(nn_module):
    """Return forward hook and pre_hook ScriptModuleStubs."""
    # 检查模块是否已初始化
    check_module_initialized(nn_module)
    # 创建一个空字典来存储钩子映射关系
    hook_map: Dict = {}

    # 存储所有的 forward hook stubs
    hook_stubs = []
    for hook in nn_module._forward_hooks.values():
        # 检查钩子是否已在映射中，如果是，则检查其唯一性
        if hook.__name__ in hook_map:
            if id(hook) != id(hook_map[hook.__name__]):
                raise RuntimeError(
                    f"Hook '{hook.__name__}' on {type(nn_module).__name__} "
                    "has at least two different python definitions."
                    " Please use unique names for all hooks."
                )
        else:
            hook_map[hook.__name__] = hook
        # 创建钩子的 stub 并添加到 hook_stubs 列表中
        hook_stubs.append(make_stub(hook, hook.__name__))

    # 存储所有的 pre_hook stubs
    pre_hook_stubs = []
    for pre_hook in nn_module._forward_pre_hooks.values():
        # 检查 pre_hook 是否已在钩子映射中，如果是，则检查其唯一性
        if pre_hook.__name__ in hook_map:
            if id(pre_hook) != id(hook_map[pre_hook.__name__]):
                raise RuntimeError(
                    f"Pre-hook '{pre_hook.__name__}' on {type(nn_module).__name__} "
                    "has at least two different python definitions."
                    " Please use unique names for all hooks."
                )
        else:
            hook_map[pre_hook.__name__] = pre_hook
        # 创建 pre_hook 的 stub 并添加到 pre_hook_stubs 列表中
        pre_hook_stubs.append(make_stub(pre_hook, pre_hook.__name__))

    # 返回所有的 hook_stubs 和 pre_hook_stubs
    return hook_stubs, pre_hook_stubs


def get_property_stubs(nn_module):
    """Create property stubs for the properties of the module by creating method stubs for the getter and setter."""
    # 获取 nn.Module 的类型
    module_ty = type(nn_module)
    # 获取模块的所有属性的 AST（抽象语法树）
    properties_asts = get_class_properties(module_ty, self_name="RecursiveScriptModule")
    rcbs = {}

    # 遍历模块类型中的所有属性
    for name in dir(module_ty):
        item = getattr(module_ty, name, None)
        # 如果属性是一个 property
        if isinstance(item, property):
            # 如果属性没有 getter，则抛出 RuntimeError
            if not item.fget:
                raise RuntimeError(
                    f"Property {name} of {nn_module.__name__} must have a getter"
                )
            # 创建用于属性 getter 的解析回调
            rcbs[name] = _jit_internal.createResolutionCallbackFromClosure(item.fget)

    # 创建并返回属性的 stubs
    stubs = [PropertyStub(rcbs[ast.name().name], ast) for ast in properties_asts]
    return stubs


def interface_script(mod_interface, nn_module):
    """
    Make a ScriptModule from an nn.Module, using the interface methods rule for determining which methods to compile.

    Args:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    # 如果 nn_module 已经是一个 ScriptModule，则直接返回
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module

    # 检查模块是否已初始化
    check_module_initialized(nn_module)

    def infer_interface_methods_to_compile(nn_module):
        """Rule to infer the methods from the interface type.

        It is used to know which methods need to act as starting points for compilation.
        """
        stubs = []
        # 遍历接口类型中的所有方法名，并为每个方法创建 stub
        for method in mod_interface.getMethodNames():
            stubs.append(make_stub_from_method(nn_module, method))
        return stubs
    # 调用函数 create_script_module，传入 nn_module 和 infer_interface_methods_to_compile 作为参数，并返回其结果
    return create_script_module(nn_module, infer_interface_methods_to_compile)
def try_compile_fn(fn, loc):
    # 检查函数是否被标记为 @ignore，如果是则不进行任何操作
    if _jit_internal.is_ignored_fn(fn):
        return None

    # 如果函数是 torch.nn.Module 类型，则也不进行任何操作
    if isinstance(fn, torch.nn.Module):
        return None

    # 如果 fn 不是 Python 函数或方法，则抛出运行时错误
    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        raise RuntimeError(
            f"`{fn}` is not a function. Recursive scripting only supports "
            "Python functions or methods currently.\n"
            f"Consider manually annotating `{fn}` with @torch.jit.script."
        )

    # 如果 fn 有 __prepare_scriptable__ 方法，则调用它，否则保持原样
    fn = fn.__prepare_scriptable__() if hasattr(fn, "__prepare_scriptable__") else fn  # type: ignore[operator]

    # 从闭包中创建解析回调函数
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    # 使用 torch.jit.script 将函数编译为 TorchScript 函数，同时传入解析回调函数
    return torch.jit.script(fn, _rcb=rcb)


def wrap_cpp_class(cpp_class):
    """将 torch._C.Object 包装为 Python 递归脚本类 RecursiveScriptClass。"""
    return torch.jit.RecursiveScriptClass(cpp_class)


def wrap_cpp_module(cpp_module):
    """
    将 torch._C.ScriptModule 包装为 Python 脚本模块 ScriptModule，并递归处理所有子模块。

    初始化函数 init_fn 用于为 script_module 中的每个子模块添加包装后的模块，
    并设置 _concrete_type 属性为从 jit_type 创建的 ConcreteModuleType。
    """
    def init_fn(script_module):
        for name, cpp_module in torch._C.ModuleDict(script_module._c).items():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
        script_module._concrete_type = torch._C.ConcreteModuleType.from_jit_type(
            script_module._c._type()
        )

        for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
            script_module._forward_pre_hooks[idx] = fn
        for idx, fn in enumerate(script_module._c._get_forward_hooks()):
            script_module._forward_hooks[idx] = fn

    return torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)


def compile_unbound_method(concrete_type, fn):
    # 如果函数被标记为 @ignore，则返回 None
    if _jit_internal.is_ignored_fn(fn):
        return None
    # 使用 make_stub 创建函数的存根
    stub = make_stub(fn, fn.__name__)
    # 禁用代码生成钩子，以防止在尚未完成的图中调用钩子
    with torch._jit_internal._disable_emit_hooks():
        create_methods_and_properties_from_stubs(concrete_type, (stub,), ())
    return stub


def lazy_bind(concrete_type, unbound_method):
    """
    返回一个函数，该函数将 unbound_method 惰性绑定到提供的 Module IValue，然后调用该方法。

    这样做是为了在编译时避免任何可能污染类型共享的 Python 操作。
    """
    def lazy_binding_method(cpp_module, *args):
        def init_fn(script_module):
            orig_class = concrete_type.py_class
    
            # 从原始类中复制 @ignored/@unused 方法到新模块中
            # 这确保它们在执行过程中可用。
            for name in dir(orig_class):
                item = getattr(orig_class, name, None)
                if _jit_internal.is_ignored_fn(item):
                    setattr(script_module, name, item)
    
            # 将常量复制到新模块中，以便在执行过程中可用。
            for name, value in concrete_type.get_constants().items():
                setattr(script_module, name, value)
    
        # 使用 init_fn 函数构造 RecursiveScriptModule 对象
        script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
        
        # 将 unbound_method 绑定到 script_module 上
        method = types.MethodType(unbound_method, script_module)
        
        # 返回绑定后的方法
        return method(*args)
    
    # 将 lazy_binding_method 看作原始方法的替代品
    lazy_binding_method.original_fn = unbound_method  # type: ignore[attr-defined]
    
    # 设置 lazy_binding_method 的名称与 unbound_method 相同
    lazy_binding_method.__name__ = unbound_method.__name__
    
    # 复制 TorchScript 修饰器到 lazy_binding_method 上
    torch._jit_internal.copy_torchscript_modifier(unbound_method, lazy_binding_method)
    
    # 返回 lazy_binding_method 方法
    return lazy_binding_method
```