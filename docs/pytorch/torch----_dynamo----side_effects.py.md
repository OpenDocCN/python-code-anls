# `.\pytorch\torch\_dynamo\side_effects.py`

```
# mypy: allow-untyped-defs
# 引入用于类型检查的标记
import inspect
# 引入警告模块，用于处理警告信息
import warnings
# 引入类型提示所需的类型
from typing import Any, Dict, List, Optional, Union

# 引入PyTorch的神经网络模块
import torch.nn

# 引入本地的工具和变量
from . import utils, variables
# 引入字节码转换相关模块
from .bytecode_transformation import (
    bytecode_from_template,
    create_call_function,
    create_call_method,
    create_instruction,
)
# 引入代码生成器
from .codegen import PyCodegen
# 引入未实现异常
from .exc import unimplemented
# 引入源码相关类
from .source import GlobalSource, LocalSource, Source
# 引入工具函数
from .utils import nn_module_new, object_new
# 引入变量基类和相关方法
from .variables.base import (
    is_side_effect_safe,
    MutableLocalBase,
    MutableLocalSource,
    VariableTracker,
)

# 可变局部变量基类，用于跟踪局部变量的修改
class MutableSideEffects(MutableLocalBase):
    """
    VariableTracker.mutable_local marker to indicate a list passed as
    an input that if we mutate we need to re-apply those mutations after
    the graph runs.
    """

    def __init__(self, source: Source, is_modified: bool = False):
        super().__init__(MutableLocalSource.Existing)
        self.source = source
        self.is_modified = is_modified

# 属性变更，用于跟踪属性变更的标记
class AttributeMutation(MutableLocalBase):
    """
    VariableTracker.mutable_local marker to track changes to attributes
    """

    def __init__(self, typ: MutableLocalSource, source: Optional[Source]):
        super().__init__(typ)
        self.source = source

# 已存在的属性变更
class AttributeMutationExisting(AttributeMutation):
    def __init__(self, source: Source):
        super().__init__(MutableLocalSource.Existing, source)
        self.source = source

# 新建的属性变更
class AttributeMutationNew(AttributeMutation):
    def __init__(self, source: Optional[Source], cls_source: Optional[Source]):
        super().__init__(MutableLocalSource.Local, source)
        self.cls_source = cls_source

# 手动更新字典的函数，将一个字典中的内容更新到另一个字典中
def _manual_update_dict(dict_from, dict_to):
    for k, v in dict_from.items():
        dict_to[k] = v

# 用于跟踪副作用的类，记录需要在FX图运行后应用的副作用（如列表变异、setattr等）
class SideEffects:
    """
    Track side effects (list mutation, setattr, etc) that need to be
    applied after an FX graph is run.
    """

    id_to_variable: Dict[int, VariableTracker]
    store_attr_mutations: Dict[MutableLocalBase, Dict[str, VariableTracker]]
    keepalive: List[Any]

    def __init__(
        self,
        id_to_variable=None,
        store_attr_mutations=None,
        keepalive=None,
        save_for_backward=None,
        tensor_hooks=None,
    ):
        super().__init__()
        # 初始化变量标识到变量跟踪器的字典
        self.id_to_variable = id_to_variable or {}
        # 初始化存储属性变更的字典
        self.store_attr_mutations = store_attr_mutations or {}
        # 初始化保持存活状态的对象列表
        self.keepalive = keepalive or []
        # 初始化用于反向传播保存的对象列表
        self.save_for_backward = save_for_backward or []
        # 初始化张量钩子字典
        self.tensor_hooks = tensor_hooks or {}
        # 跟踪Compiled Autograd的最终回调函数，这些函数需要在Compiled Autograd反向图结束时调用
        # 只有在从Dynamo追踪创建的图中才适用于此图
        self.ca_final_callbacks_var = None
    # 定义一个特殊方法 __eq__，用于比较对象是否相等
    def __eq__(self, other: object) -> bool:
        # 断言确保参数 other 是 SideEffects 类的实例
        assert isinstance(other, SideEffects)
        # 注意：不要测试 keepalive 属性
        # 比较各个属性是否相等并返回结果
        return (
            self.id_to_variable == other.id_to_variable
            and self.store_attr_mutations == other.store_attr_mutations
            and self.save_for_backward == other.save_for_backward
            and self.tensor_hooks == other.tensor_hooks
        )

    # 定义方法 diff，用于比较两个 SideEffects 对象的差异
    def diff(self, other: "SideEffects") -> Optional[str]:
        # 检查 id_to_variable 属性是否不相等
        if self.id_to_variable != other.id_to_variable:
            # 获取各自 id_to_variable 的键集合
            sk_itv = self.id_to_variable.keys()
            ok_itv = other.id_to_variable.keys()
            # 如果键集合不相等，返回差异信息
            if sk_itv != ok_itv:
                return f"id_to_variable keys: {sk_itv} != {ok_itv}"
            # 如果键集合相等，返回通用差异信息
            return "id_to_variable: unknown diff"
        # 检查 store_attr_mutations 属性是否不相等
        elif self.store_attr_mutations != other.store_attr_mutations:
            # 获取各自 store_attr_mutations 的键集合
            sk_sam = self.store_attr_mutations.keys()
            ok_sam = other.store_attr_mutations.keys()
            # 如果键集合不相等，返回差异信息
            if sk_sam != ok_sam:
                return f"store_attr_mutations keys: {sk_sam} != {ok_sam}"
            # 如果键集合相等，返回通用差异信息
            return "store_attr_mutations: unknown diff"
        # 检查 save_for_backward 属性是否不相等
        elif self.save_for_backward != other.save_for_backward:
            return "save_for_backward"
        # 检查 tensor_hooks 属性是否不相等
        elif self.tensor_hooks != other.tensor_hooks:
            return "tensor_hooks"
        else:
            # 如果所有属性都相等，返回 None 表示无差异
            return None

    # 定义方法 clone，用于创建对象的浅复制
    def clone(self):
        """Create a shallow copy"""
        # 返回当前类的新实例，复制对象的属性值
        return self.__class__(
            id_to_variable=dict(self.id_to_variable),  # 复制 id_to_variable 属性
            store_attr_mutations={  # 复制 store_attr_mutations 属性
                k: dict(v) for k, v in self.store_attr_mutations.items()
            },
            keepalive=list(self.keepalive),  # 复制 keepalive 属性
            save_for_backward=self.save_for_backward,  # 复制 save_for_backward 属性
            tensor_hooks=self.tensor_hooks,  # 复制 tensor_hooks 属性
        )

    # 定义特殊方法 __contains__，用于判断对象是否包含指定项
    def __contains__(self, item):
        # 判断 item 对象的标识是否在 id_to_variable 字典中
        return id(item) in self.id_to_variable

    # 定义特殊方法 __getitem__，用于获取对象中指定项的值
    def __getitem__(self, item):
        # 返回 id_to_variable 字典中对应 item 对象标识的值
        return self.id_to_variable[id(item)]

    # 定义方法 check_allowed_side_effect，用于检查是否允许执行副作用
    def check_allowed_side_effect(self, item):
        # 导入 AutogradFunctionContextVariable 类
        from torch._dynamo.variables.misc import AutogradFunctionContextVariable

        # 检查 item 是否为 AutogradFunctionContextVariable 类的实例
        # 如果是，则返回 True 表示允许
        if isinstance(item, AutogradFunctionContextVariable):
            return True
        # 如果 item 不是 AutogradFunctionContextVariable 类的实例
        # 检查 item.mutable_local 是否安全进行副作用操作
        if not is_side_effect_safe(item.mutable_local):
            # 如果不安全，抛出未实现异常，指示副作用不在当前作用域中
            unimplemented(
                "HigherOrderOperator: Mutating a variable not in the current scope (SideEffects)"
            )

    # 定义方法 store_attr，用于存储属性的变化
    def store_attr(self, item: VariableTracker, name: str, value: VariableTracker):
        # 断言确认 item 是属性变化追踪器对象
        assert self.is_attribute_mutation(item)
        # 检查是否允许执行 item 对象的副作用
        self.check_allowed_side_effect(item)
        # 如果 item.mutable_local 不在 store_attr_mutations 中
        if item.mutable_local not in self.store_attr_mutations:
            self.store_attr_mutations[item.mutable_local] = {}
        # 存储属性变化信息到 store_attr_mutations 中
        self.store_attr_mutations[item.mutable_local][name] = value
    def load_attr(self, item, name, deleted_ok=False):
        # 断言检查是否为属性变异
        assert self.is_attribute_mutation(item)
        # 获取存储的属性变异中的指定名称的值
        result = self.store_attr_mutations[item.mutable_local][name]
        # 如果 deleted_ok 为 False，并且结果是 DeletedVariable 类型，则报未实现的错误
        if not deleted_ok and isinstance(result, variables.DeletedVariable):
            unimplemented("read deleted attribute")
        # 返回获取到的结果
        return result

    def store_cell(self, cellvar, value):
        # 断言检查 cellvar 是否为 NewCellVariable 类型
        assert isinstance(cellvar, variables.NewCellVariable)
        # 断言检查 value 是否为 VariableTracker 类型
        assert isinstance(value, variables.VariableTracker)
        # 存储 cellvar 的 cell_contents 属性为给定的 value
        self.store_attr(cellvar, "cell_contents", value)

    def load_cell(self, cellvar):
        # 断言检查 cellvar 是否为 NewCellVariable 类型
        assert isinstance(cellvar, variables.NewCellVariable)
        # 加载 cellvar 的 cell_contents 属性的值
        return self.load_attr(cellvar, "cell_contents")

    def load_global(self, gvar: VariableTracker, name: str):
        # 断言检查 gvar 是否为 VariableTracker 类型
        assert isinstance(gvar, variables.VariableTracker)
        # 加载 gvar 的指定名称的全局属性值
        return self.load_attr(gvar, name)

    def store_global(self, gvar: VariableTracker, name: str, value: VariableTracker):
        # 断言检查 gvar 是否为 VariableTracker 类型，value 是否也是 VariableTracker 类型
        assert isinstance(gvar, variables.VariableTracker)
        assert isinstance(value, variables.VariableTracker)
        # 存储 gvar 的指定名称的全局属性为给定的 value
        self.store_attr(gvar, name, value)

    @staticmethod
    def cls_supports_mutation_side_effects(cls):
        # 检查给定类是否支持属性变异副作用
        return (
            inspect.getattr_static(cls, "__getattribute__", None)
            is object.__getattribute__
        )

    def is_attribute_mutation(self, item):
        # 检查 item 是否为 AttributeMutation 类型
        return isinstance(item.mutable_local, AttributeMutation)

    def has_pending_mutation(self, item):
        # 检查 item 是否为属性变异，并且是否有待处理的变异
        return self.is_attribute_mutation(item) and bool(
            self.store_attr_mutations.get(item.mutable_local)
        )

    def has_pending_mutation_of_attr(self, item, name):
        # 检查 item 是否为属性变异，并且特定名称是否在其待处理变异中
        return self.is_attribute_mutation(
            item
        ) and name in self.store_attr_mutations.get(item.mutable_local, ())

    def is_modified(self, item):
        # 如果 item 是 AttributeMutationNew 类型，则返回 True
        if isinstance(item.mutable_local, AttributeMutationNew):
            return True
        # 如果 item 是属性变异类型，则检查其是否在存储的属性变异中
        if self.is_attribute_mutation(item):
            return item.mutable_local in self.store_attr_mutations
        # 否则，返回 item.mutable_local.is_modified 的结果
        return item.mutable_local.is_modified

    def _track_obj(
        self,
        item: Any,
        variable: VariableTracker,
        mutable_cls=MutableSideEffects,
    ):
        """Start tracking a new variable for mutation"""
        # 断言检查 variable 的源不为 None
        assert variable.source is not None

        # 如果 item 的 id 已经在 id_to_variable 中存在，则抛出断言错误
        if id(item) in self.id_to_variable:
            raise AssertionError(
                "Variable is already tracked for mutation. This could be "
                "because you are not using VariableBuilder to construct "
                "the variable tracker."
            )

        # 将 variable 的 mutable_local 设置为 mutable_cls 类型，并将其映射到 id(item)
        variable.mutable_local = mutable_cls(variable.source)
        # 将 id(item) 映射到 variable
        self.id_to_variable[id(item)] = variable
        # 将 item 添加到 keepalive 列表中
        self.keepalive.append(item)
        # 返回 variable
        return variable

    track_mutable = _track_obj

    def track_object_existing(
        self,
        item: Any,
        variable: VariableTracker,
    ):
        # 调用 _track_obj 方法来追踪一个已存在的对象的变异
        return self._track_obj(item, variable, mutable_cls=AttributeMutationExisting)
    # 定义一个方法来追踪新对象，根据不同的类别和选项进行处理
    def track_object_new(
        self,
        cls_source: Source,
        user_cls: Any,
        variable_cls: Any,
        options,
    ):
        # 如果用户类是 torch.autograd.function.FunctionCtx 类型
        if user_cls is torch.autograd.function.FunctionCtx:
            # 使用警告记录的方式，创建一个 torch 的自动求导函数对象
            with warnings.catch_warnings(record=True):
                obj = torch.autograd.Function()
        # 如果用户类是 torch.nn.Module 的子类
        elif issubclass(user_cls, torch.nn.Module):
            # 使用 nn_module_new 方法创建一个新的 torch 的神经网络模块对象
            obj = nn_module_new(user_cls)
        # 其他情况下
        else:
            # 使用 object_new 方法创建一个新的普通对象
            obj = object_new(user_cls)
        # 使用 variable_cls 创建一个变量，包括对象本地的可变属性变化和给定的选项
        variable = variable_cls(
            obj,
            mutable_local=AttributeMutationNew(None, cls_source),
            **options,
        )
        # 将对象 ID 映射到变量，并将对象添加到 keepalive 列表中
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        # 返回创建的变量
        return variable

    # 定义一个方法来追踪新的 cell 对象
    def track_cell_new(
        self,
    ):
        # 创建一个新的普通对象作为 cell 对象
        obj = object()
        # 使用 NewCellVariable 创建一个变量，设置本地的可变属性变化为 None
        variable = variables.NewCellVariable(
            mutable_local=AttributeMutationNew(None, None),
        )
        # 将对象 ID 映射到变量，并将对象添加到 keepalive 列表中
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        # 返回创建的变量
        return variable

    # 定义一个方法来追踪现有的 cell 对象
    def track_cell_existing(self, source: Source, item: Any):
        # 使用 NewCellVariable 创建一个变量，设置本地的可变属性变化为给定的 source
        variable = variables.NewCellVariable(
            mutable_local=AttributeMutationExisting(source),
        )
        # 将对象 ID 映射到变量，并将对象添加到 keepalive 列表中
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        # 返回创建的变量
        return variable

    # 定义一个方法来追踪现有的全局对象
    def track_global_existing(self, source: Source, item: Any):
        # 使用 NewGlobalVariable 创建一个变量，设置本地的可变属性变化为给定的 source
        variable = variables.NewGlobalVariable(
            mutable_local=AttributeMutationExisting(source),
        )
        # 将对象 ID 映射到变量，并将对象添加到 keepalive 列表中
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        # 返回创建的变量
        return variable

    # 定义一个方法来追踪保存用于反向传播的上下文和参数
    def track_save_for_backward(self, ctx, args):
        # 断言上下文是 AutogradFunctionContextVariable 类型的实例
        assert isinstance(ctx, variables.AutogradFunctionContextVariable)
        # 将上下文和参数作为元组添加到 save_for_backward 列表中
        self.save_for_backward.append((ctx, args))

    # 定义一个方法来追踪从其他对象的副作用中看到的张量变量
    def track_tensor_variables_from_runahead_side_effects(self, other):
        # 在高阶操作中，我们希望跟踪在 speculate_subgraph 中看到的张量，以便在其他 speculate_subgraph 或根跟踪器中不再将其提升为新的输入
        for other_item in other.keepalive:
            other_id = id(other_item)
            other_variable = other.id_to_variable[other_id]
            # 如果其他对象 ID 不在当前对象的 id_to_variable 映射中，并且其对应的变量是 TensorVariable 类型
            if other_id not in self.id_to_variable and isinstance(
                other_variable, variables.TensorVariable
            ):
                # 调用 track_object_existing 方法来追踪现有的其他对象和其对应的变量
                self.track_object_existing(other_item, other_variable)
    def prune_dead_object_new(self, tx):
        live_new_objects = set()

        # use this to avoid cycles in mutable_local (though I'm not sure if that
        # can actually happen).
        visited: Any = set({})

        def visit(var: VariableTracker):
            mutable_local = var.mutable_local
            if mutable_local is None:
                return
            if mutable_local in visited:
                return
            visited.add(mutable_local)
            # Object may have been mutated, store this mutation.
            if isinstance(mutable_local, AttributeMutationNew):
                live_new_objects.add(mutable_local)
            # It's possible that we have mutated the value of this variable
            # to be another one. The new value is in store_attr_mutations.
            # Also recurse through the new value to detect alive AttributeMutationNew.
            if var.mutable_local in self.store_attr_mutations:
                VariableTracker.visit(
                    visit, self.store_attr_mutations[var.mutable_local]
                )

        def is_live(var: Union[MutableLocalBase, VariableTracker]):
            if isinstance(var, AttributeMutationNew):
                return var in live_new_objects
            if isinstance(var, VariableTracker):
                return is_live(var.mutable_local)
            return True

        pre_existing_vars = [
            var
            for var in self.id_to_variable.values()
            if not isinstance(var.mutable_local, AttributeMutationNew)
        ]

        # The only live side effects come from returns (tx.stack), any intermediates
        # during a graph break (tx.symbolic_locals), and mutation on pre-existing variables.
        # Recursively visit Variables and see if any of them have been mutated.
        VariableTracker.visit(visit, (tx.stack, tx.symbolic_locals, pre_existing_vars))

        # NB: cell variable handling.is tricky.
        # cell variables must stay alive if any NestedUserFunctionVariable
        # are live. "visit"-ing the NestedUserFunctionVariable visits
        # the .closures field, from which we will see if we need to keep
        # any mutations to cell variables alive.

        # Filter out variables that are not live based on the `is_live` function.
        self.id_to_variable = {
            k: v for k, v in self.id_to_variable.items() if is_live(v)
        }
        # Filter out attribute mutations that are not live based on the `is_live` function.
        self.store_attr_mutations = {
            k: v for k, v in self.store_attr_mutations.items() if is_live(k)
        }

    def mutation(self, var):
        # Check if mutating `var` is allowed.
        self.check_allowed_side_effect(var)
        # If `var.mutable_local` is an instance of `MutableSideEffects`,
        # replace it with a new `MutableSideEffects` instance.
        if isinstance(var.mutable_local, MutableSideEffects):
            var.mutable_local = MutableSideEffects(var.mutable_local.source, True)

    def _get_modified_vars(self):
        # Return a list of variables that have been modified.
        return [var for var in self.id_to_variable.values() if self.is_modified(var)]
    # 生成并保存临时变量相关的代码，使用指定的代码生成器 `cg`
    def codegen_save_tempvars(self, cg: PyCodegen):
        # 遍历所有被修改的变量
        for var in self._get_modified_vars():
            # 检查变量是否为新创建的 Cell 变量，并且需要导入 make_cell 函数
            if isinstance(
                var.mutable_local, (AttributeMutationExisting, AttributeMutationNew)
            ) and isinstance(var, variables.NewCellVariable):
                # 添加一个将 make_cell 函数加载到栈顶的指令
                cg.add_push_null(
                    lambda: cg.load_import_from(utils.__name__, "make_cell")
                )
                # 扩展输出，创建调用函数的指令
                cg.extend_output(create_call_function(0, False))
                # 缓存变量
                cg.add_cache(var)
                # 如果是新创建的属性变量，更新其源为当前临时变量的本地来源
                if isinstance(var.mutable_local, AttributeMutationNew):
                    var.mutable_local.source = LocalSource(cg.tempvars[var])  # type: ignore[attr-defined]
            # 对于新创建的属性变量
            elif isinstance(var.mutable_local, AttributeMutationNew):
                # 如果是 AutogradFunctionContextVariable 类型的变量，报告未实现的功能
                if isinstance(var, variables.AutogradFunctionContextVariable):
                    unimplemented("AutogradFunctionContextVariable escaped")
                # 添加一个将 object_new 函数加载到栈顶的指令
                cg.add_push_null(
                    lambda: cg.load_import_from(utils.__name__, "object_new")
                )
                # 生成代码来创建变量并调用相应的函数
                cg(var.mutable_local.cls_source)
                cg.extend_output(create_call_function(1, False))
                # 缓存变量
                cg.add_cache(var)
                # 更新变量的源为当前临时变量的本地来源
                var.mutable_local.source = LocalSource(cg.tempvars[var])
            # 对于已存在的临时变量
            elif var in cg.tempvars:
                # 确保临时变量在缓存中为 None
                assert cg.tempvars.get(var) is None
                # 后续使用应指向原始变量
                cg(var.mutable_local.source)
                # 缓存变量
                cg.add_cache(var)

        # 遍历 self.save_for_backward 中的上下文和参数
        for ctx, args in self.save_for_backward:
            # 生成代码来加载上下文的源
            cg(ctx.source)
            # 加载 save_for_backward 方法
            cg.load_method("save_for_backward")
            # 遍历参数并生成相应的指令
            for arg in args:
                cg(arg)
            # 扩展输出，包括创建方法调用的指令和一个 POP_TOP 指令
            cg.extend_output(
                [
                    *create_call_method(len(args)),
                    create_instruction("POP_TOP"),
                ]
            )

    # 注册钩子函数到指定的张量变量
    def register_hook(self, tensor, hook, handle, name):
        assert isinstance(tensor, variables.TensorVariable)
        assert isinstance(hook, variables.VariableTracker)
        assert (
            isinstance(handle, variables.RemovableHandleVariable)
            and handle.mutable_local
        )
        # 确保 torch.Tensor 中有指定的属性名
        assert hasattr(torch.Tensor, name)
        # 计算当前钩子函数的索引
        idx = len(self.tensor_hooks.keys())
        # 处理可能的索引重复情况，因为可能被 self.remove_hook() 导致
        while idx in self.tensor_hooks:
            idx += 1
        # 将钩子函数相关信息存储到 self.tensor_hooks 中
        self.tensor_hooks[idx] = (tensor, hook, handle, name)
        # 确保 handle.idx 为 None
        assert not handle.idx
        # 设置 handle.idx 为当前索引值
        handle.idx = idx

    # 移除指定索引的钩子函数
    def remove_hook(self, idx):
        # 删除 self.tensor_hooks 中指定索引的钩子函数信息
        del self.tensor_hooks[idx]

    # 获取 CA 最终回调变量
    def get_ca_final_callbacks_var(self):
        from .variables.base import MutableLocal

        # 如果 self.ca_final_callbacks_var 为 None，则创建一个空列表变量，并设置其为可变本地变量
        if self.ca_final_callbacks_var is None:
            self.ca_final_callbacks_var = variables.ListVariable(
                [], mutable_local=MutableLocal()
            )
        # 返回 CA 最终回调变量
        return self.ca_final_callbacks_var
    # 检查对象是否为空的方法，返回布尔值
    def is_empty(self):
        # 返回一个布尔值，指示是否有任何变量被修改
        return not (
            any(map(self.is_modified, self.id_to_variable.values()))  # 检查所有变量是否被修改过
            or self.tensor_hooks  # 检查是否有张量钩子
            or self.save_for_backward  # 检查是否有用于反向传播的张量保存
            or self.tensor_hooks  # 再次检查是否有张量钩子（此处可能是重复）
        )

    # 清空对象的方法
    def clear(self):
        # 清空存活对象的集合
        self.keepalive.clear()
        # 清空 ID 到变量的映射字典
        self.id_to_variable.clear()
```