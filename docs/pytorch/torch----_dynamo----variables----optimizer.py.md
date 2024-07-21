# `.\pytorch\torch\_dynamo\variables\optimizer.py`

```py
# mypy: ignore-errors

# 引入弱引用模块，用于管理对象的弱引用
import weakref
# 引入类型提示相关的模块
from typing import Dict, List, TYPE_CHECKING

# 引入 PyTorch 库
import torch
# 从 torch.utils._pytree 模块中仅引入 tree_map_only 函数
from torch.utils._pytree import tree_map_only

# 引入自定义模块和类
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    GetItemSource,
    GlobalWeakRefSource,
    GradSource,
)
# 引入工具函数
from ..utils import GLOBAL_KEY_PREFIX

# 引入自定义变量类和常量
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable

# 如果是类型检查模式，引入 VariableTracker 类
if TYPE_CHECKING:
    from .base import VariableTracker

# 定义异常类，用于参数映射异常
class ArgMappingException(Exception):
    pass

# 定义异常类，用于守卫安装异常
class GuardInstallException(Exception):
    pass

# 定义优化器变量类，继承自用户自定义对象变量类
class OptimizerVariable(UserDefinedObjectVariable):
    # 定义不可变字段集合
    _nonvar_fields = {
        "grad_to_source",
        "tensor_to_source",
        "static_tensor_names",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    # 初始化方法
    def __init__(
        self,
        value,
        grad_to_source=None,
        static_tensor_names=None,
        tensor_to_source=None,
        **kwargs,
    ):
        super().__init__(value, **kwargs)
        # 初始化梯度到来源的映射字典，默认为空字典
        self.grad_to_source = grad_to_source or {}
        # 初始化张量到来源的映射字典，默认为空字典
        self.tensor_to_source = tensor_to_source or {}
        # 初始化静态张量名称集合，默认为空集合
        self.static_tensor_names = static_tensor_names or set()

    # 方法调用函数，用于优化避免追踪优化器初始化过程的性能问题
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow initialization of the optimizer"""
        # 如果方法名为 '_init_group'
        if name == "_init_group":
            try:
                # 执行图形中断，以避免潜在的变异问题
                self.graph_break_if_pending_mutation(tx)
                # 如果运行在 CPU 上，则移动步骤
                self.move_step_if_cpu()
                # 获取 Python 参数和关键字参数
                py_args, py_kwargs = self.get_python_args(*args, **kwargs)
                # 调用优化器对象的 '_init_group' 方法
                ret_val = self.value._init_group(*py_args, **py_kwargs)
                # 映射来源并安装守卫
                self.map_sources_and_install_guards(tx)
                # 更新列表参数
                self.update_list_args(tx, args, kwargs, py_args, py_kwargs)
                # 将优化器对象的弱引用存储为全局对象
                mangled_name = f"__optimizer_{id(self.value)}"
                tx.store_global_weakref_by_id(mangled_name, self.value)
                # 创建最终器
                self.create_finalizer(tx)

                # 返回初始化结果的常量变量
                return ConstantVariable.create(ret_val)
            except (ArgMappingException, GuardInstallException) as _:
                # 如果参数映射异常或守卫安装异常，正常进行跟踪
                pass

        # 调用父类的方法调用函数
        return super().call_method(tx, name, args, kwargs)
    # 定义一个方法 var_getattr，用于获取属性值
    def var_getattr(self, tx, name):
        # 注意：这允许我们在 call_method 中拦截调用
        # 在典型情况下，返回一个 UserMethodVariable
        # 这将直接内联
        if name in ("_init_group", "step"):
            # 如果属性名为 "_init_group" 或 "step"，返回相应的 GetAttrVariable
            return GetAttrVariable(self, name, source=AttrSource(self.source, name))

        if name == "param_groups":
            # 如果属性名为 "param_groups"
            from ..decorators import mark_static_address

            # 遍历 self.value.param_groups 中的每个参数组
            for group in self.value.param_groups:
                # 对每个参数组中的参数调用 mark_static_address 函数
                for p in group["params"]:
                    mark_static_address(p)

            # 设置当前对象为可捕获状态
            self._set_capturable(tx)

        # 调用父类的 var_getattr 方法处理其他情况
        return super().var_getattr(tx, name)

    # 定义一个方法 graph_break_if_pending_mutation，用于处理待处理变异时的图破坏
    def graph_break_if_pending_mutation(self, tx):
        # 如果参数中有待处理变异，则需要进行图破坏
        for g in self.value.param_groups:
            for p in g["params"]:
                # 获取参数 p 对应的变量
                side_effects = tx.output.side_effects
                variable = side_effects.id_to_variable.get(id(p), None)
                # 如果变量存在并且有待处理变异，则抛出异常
                if variable and side_effects.has_pending_mutation(variable):
                    from ..exc import Unsupported

                    raise Unsupported("Pending mutation on parameter")

    # 定义一个方法 _set_capturable，用于设置可捕获状态
    def _set_capturable(self, tx):
        from . import LazyVariableTracker
        from .builder import VariableBuilder

        # 只有在参数在 CUDA 上并且状态未初始化时才设置可捕获
        def safe_to_set_capturable(group):
            all_uninitialized = True
            all_cuda = True

            for p in group.get("params", list()):
                all_cuda &= p.is_cuda
                all_uninitialized &= p not in self.value.state

            return "capturable" in group and all_uninitialized and all_cuda

        # 遍历 self.value.param_groups 中的参数组，设置可捕获状态
        for ind, group in enumerate(self.value.param_groups):
            if safe_to_set_capturable(group):
                group["capturable"] = True

        # 实现 LazyVariableTracker 中的所有变量
        param_groups_vt = LazyVariableTracker.realize_all(
            VariableBuilder(tx, AttrSource(self.source, "param_groups"))(
                self.value.param_groups
            )
        )

        # 设置 param_groups_vt 中每个参数组的 capturable 属性为 True
        for ind, param_group_vt in enumerate(param_groups_vt.items):
            key = ConstDictVariable._HashableTracker(
                ConstantVariable.create("capturable")
            )
            param_group_vt.items[key] = ConstantVariable.create(True)
    # 获取与变量追踪器参数等效的 Python 值

    def get_python_args(self, *args, **kwargs):
        """Get python values equivalent to the variable tracker args"""
        
        def map_arg(arg):
            # 如果参数是 ConstantVariable 类型，则返回其 Python 常量值
            if isinstance(arg, ConstantVariable):
                return arg.as_python_constant()
            # 如果参数是 ListVariable 类型且为空列表，则返回空列表
            elif isinstance(arg, ListVariable) and not arg.items:
                return []
            # 如果参数是 ConstDictVariable 类型，并且其源是 GetItemSource，
            # 其基础是 AttrSource，并且成员是 "param_groups"，
            # 则返回对应的 param_groups 中的值
            elif (
                isinstance(arg, ConstDictVariable)
                and isinstance(arg.source, GetItemSource)
                and isinstance(arg.source.base, AttrSource)
                and arg.source.base.member == "param_groups"
            ):
                return self.value.param_groups[arg.source.index]
            
            # 如果以上条件都不满足，则抛出异常
            raise ArgMappingException

        # 对 args 中的每个参数应用 map_arg 函数，得到新的参数列表 new_args
        new_args = [map_arg(arg) for arg in args]
        # 对 kwargs 中的每个参数应用 map_arg 函数，得到新的参数字典 new_kwargs
        new_kwargs = {k: map_arg(v) for k, v in kwargs.items()}

        # 返回新的参数列表和参数字典
        return new_args, new_kwargs

    # 如果用户加载的是旧的状态字典，
    # 可能 step 在 CPU 上，如果是这种情况，将其移动到对应的 GPU 上
    # 大多数情况下这是一个空操作，因为状态是空的
    def move_step_if_cpu(self):
        # 遍历 self.value.state 中的每个元素
        for p, state in self.value.state.items():
            # 如果状态中包含 "step" 并且其 is_cpu 属性为 True
            if "step" in state and state["step"].is_cpu:
                # 将 state["step"] 移动到 p.device 对应的 GPU 上
                state["step"] = state["step"].to(p.device)

    # 将状态张量包装在 TensorVariable 中
    def wrap_tensor(self, tx, tensor_value):
        """Wrap state tensor in a TensorVariable"""
        # 导入必要的模块和类
        from ..decorators import mark_static_address
        from .builder import VariableBuilder

        # 如果 tensor_value 已经在 self.tensor_to_source 中
        if tensor_value in self.tensor_to_source:
            # 标记这些张量为 cudagraphs 的静态地址
            mark_static_address(tensor_value)
            # 使用 self.tensor_to_source[tensor_value] 创建 VariableBuilder 对象
            builder = VariableBuilder(tx, self.tensor_to_source[tensor_value])
            # 将 builder.name 添加到 self.static_tensor_names 中
            self.static_tensor_names.add(tx.output.module_key_name(builder.name))
        # 如果 tensor_value 在 self.grad_to_source 中
        elif tensor_value in self.grad_to_source:
            # 使用 self.grad_to_source[tensor_value] 创建 VariableBuilder 对象
            builder = VariableBuilder(tx, self.grad_to_source[tensor_value])
        else:
            # 标记这些张量为 cudagraphs 的静态地址
            mark_static_address(tensor_value)

            # 在全局创建 tensor_value 的全局弱引用，并使用 GlobalWeakRefSource 创建 VariableBuilder 对象
            global_name = tx.store_global_weakref_by_id(GLOBAL_KEY_PREFIX, tensor_value)
            builder = VariableBuilder(tx, GlobalWeakRefSource(global_name))
            # 将 builder.name 添加到 self.static_tensor_names 中
            self.static_tensor_names.add(tx.output.module_key_name(builder.name))

        # 使用 builder(tensor_value) 创建 result
        result = builder(tensor_value)
        # 返回结果
        return result
    def update_list_args(self, tx, args, kwargs, py_args, py_kwargs):
        """Update the args and kwargs to the traced optimizer call"""
        # 遍历原始参数和Python参数的对应关系
        for arg, py_arg in zip(args, py_args):
            # 如果参数是 ListVariable 类型
            if isinstance(arg, ListVariable):
                # 断言 Python 参数是一个列表
                assert isinstance(
                    py_arg, list
                ), "py_arg should be a list in optimizer variable"
                # 遍历 Python 参数列表
                for i, val in enumerate(py_arg):
                    # 在 tx.output.side_effects 中记录参数变异
                    tx.output.side_effects.mutation(arg)
                    # 如果值是 torch.Tensor 类型，封装成 tensor 对象并添加到参数的 items 列表中
                    if isinstance(val, torch.Tensor):
                        arg.items.append(self.wrap_tensor(tx, val))
                    else:
                        # 如果值不是 tensor，根据源和索引构建变量
                        from .builder import SourcelessBuilder, VariableBuilder

                        if arg.source:
                            arg.items.append(
                                VariableBuilder(tx, GetItemSource(arg.source, i))(val)
                            )
                        else:
                            arg.items.append(SourcelessBuilder.create(tx, val))

    def create_finalizer(self, tx):
        # 获取静态张量名列表和值
        names_to_delete = self.static_tensor_names
        value = self.value
        tc = tx.output.tracing_context

        # 初始化 finalizer 函数
        def init_finalizer(gm):
            # 清理静态张量引用的函数
            def clear_static_tensor_refs():
                for name in names_to_delete:
                    gm._buffers.pop(name, None)
                    gm._parameters.pop(name, None)
                    if tc.params_flat:
                        tc.params_flat.clear()

            # 使用 weakref.finalize() 来绑定清理函数
            weakref.finalize(value, clear_static_tensor_refs)

        # 将 finalizer 函数添加到输出的图最终化程序中
        tx.output.add_graph_finalizer(init_finalizer)
```