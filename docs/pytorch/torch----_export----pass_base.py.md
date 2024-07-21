# `.\pytorch\torch\_export\pass_base.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import operator
import traceback
import typing
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# 导入 PyTorch 相关模块和函数
import torch
from functorch.experimental.control_flow import _unstack_pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree
from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts, compute_unbacked_bindings

# 定义导出模块中公开的符号
__all__ = ["_ExportPassBaseDeprecatedDoNotUse"]

# 定义类型别名
Argument = Any
Value = Any
Fn = Callable[..., Any]
PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]

# 定义一组 Torch 符号操作的集合
_TORCH_SYM_OPS: Set[Callable] = {
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_not,
    torch.sym_sqrt,
}

# 定义自定义的异常类，继承自 RuntimeError
class ExportPassBaseError(RuntimeError):
    pass

# 定义私有类 _ExportPassBaseDeprecatedDoNotUse，继承自 PassBase 类
class _ExportPassBaseDeprecatedDoNotUse(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    # 静态方法：创建一个虚拟节点的元数据对象，包括当前调用栈信息
    @staticmethod
    def _create_dummy_node_metadata():
        return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})

    # 初始化方法，创建实例时调用
    def __init__(self) -> None:
        # 创建传播未支持符号整数的对象，用于处理图模块
        self.interpreter = PropagateUnbackedSymInts(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        # 创建导出追踪器对象，使用代码生成器
        self.tracer = self.ExportTracer(self, CodeGen())
        # 可选的虚拟张量模式
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        # 已初始化标志
        self._initialized = True
        # 节点调试信息字符串
        self.node_debug_str: typing.Optional[str] = None

    # 内部方法 _fx，用于处理转换相关操作
    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # 使用pytree.tree_map_only方法，将args和kwargs中的数据提取为ProxyValue对象
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        # 调用self.interpreter对象的kind方法，处理目标target和提取的args_data、kwargs_data数据，返回处理结果
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        # 使用pytree.tree_map_only方法，将args和kwargs中的ProxyValue对象提取为其proxy属性
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        # 如果target是torch._ops.OpOverload类型，则获取其overloadpacket的名称，并转换为字符串
        if isinstance(target, torch._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        # 使用tracer对象的create_proxy方法创建一个代理值res_proxy，传入kind、target、args_proxy、kwargs_proxy和可选的name参数
        res_proxy = self.tracer.create_proxy(kind, target, args_proxy, kwargs_proxy, name=name)
        # 更新res_proxy的节点元数据信息
        res_proxy.node.meta.update(meta.data)
        # 如果self.fake_tensor_mode为真，并且存在shape_env的非空形式，则计算未支持的绑定信息并添加到res_proxy的节点元数据中
        if self.fake_tensor_mode and (shape_env := self.fake_tensor_mode.shape_env):
            if symbol_to_path := compute_unbacked_bindings(shape_env, res_data):
                res_proxy.node.meta["unbacked_bindings"] = symbol_to_path
        # 使用tracer对象的set_metadata方法，将res_data设置为res_proxy的节点元数据
        self.tracer.set_metadata(res_proxy.node, res_data)
        # 返回一个ProxyValue对象，包含res_data和res_proxy
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: torch.fx.GraphModule) -> List[Argument]:
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        # 如果graph_module的meta中包含"args"键，则返回其值args的列表形式
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
            # 如果node.meta中包含"val"键，则提取其值为fake，并检查其是否包含constant属性，如果有则返回其constant值，否则返回fake本身
            if "val" in node.meta:
                fake = node.meta["val"]
                if hasattr(fake, "constant") and fake.constant is not None:
                    return fake.constant
                return fake
            # 如果node.meta中包含"tensor_meta"键，则根据tensor_meta创建一个FakeTensor对象并返回
            elif tensor_meta := node.meta.get("tensor_meta"):
                assert self.fake_tensor_mode is not None
                return FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    torch.device("cpu"),
                )
            # 如果node没有用户使用，则返回None
            elif len(node.users) == 0:
                return None
            # 如果以上条件均不满足，则抛出ExportPassBaseError异常，提示无法为图模块构造输入
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        # 返回graph_module中所有操作为"placeholder"的节点的输入列表，使用extract_input函数提取
        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        # 空方法，用于处理ProxyValue对象的属性，无实际操作
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        # 使用tracer对象的create_proxy方法创建一个占位符代理arg_proxy，传入"placeholder"、name、空元组()和空字典{}
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        # 将meta.data更新到arg_proxy的节点元数据中
        arg_proxy.node.meta = meta.data
        # 使用tracer对象的set_metadata方法，将arg设置为arg_proxy的节点元数据
        self.tracer.set_metadata(arg_proxy.node, arg)
        # 返回一个ProxyValue对象，包含arg和arg_proxy
        return ProxyValue(arg, arg_proxy)
    # 调用运算符，执行函数调用操作
    def call_operator(
        self,
        op,  # 要调用的操作符或函数
        args: Tuple[Argument, ...],  # 函数的位置参数列表
        kwargs: Dict[str, Argument],  # 函数的关键字参数字典
        meta: NodeMetadata,  # 节点的元数据
    ) -> ProxyValue:  # 返回值类型为 ProxyValue
        return self._fx("call_function", op, args, kwargs, meta)  # 调用内部函数 _fx 进行函数调用操作

    # 调用符号函数，执行函数调用操作
    def call_sym(
        self,
        target: Fn,  # 要调用的函数对象
        args: Tuple[Argument, ...],  # 函数的位置参数列表
        meta: NodeMetadata,  # 节点的元数据
    ) -> ProxyValue:  # 返回值类型为 ProxyValue
        return self._fx("call_function", target, args, {}, meta)  # 调用内部函数 _fx 进行函数调用操作

    # 调用条件函数，执行条件分支判断操作
    def call_cond(
        self,
        pred: ProxyValue,  # 条件断言的预测值
        true_fn: torch.fx.GraphModule,  # 真实分支执行的函数模块
        false_fn: torch.fx.GraphModule,  # 虚假分支执行的函数模块
        inputs: List[Argument],  # 函数的输入参数列表
        meta: NodeMetadata,  # 节点的元数据
    ) -> ProxyValue:  # 返回值类型为 ProxyValue
        true_branch = self.call_submodule(true_fn, tuple(inputs))  # 调用子模块执行真实分支函数
        false_branch = self.call_submodule(false_fn, tuple(inputs))  # 调用子模块执行虚假分支函数
        assert true_branch is not None
        assert false_branch is not None
        return self._fx(
            "call_function",  # 执行函数调用操作
            torch.ops.higher_order.cond,  # 使用 Torch 的 higher_order.cond 运算
            (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)),  # 调用参数列表
            {},  # 空的关键字参数字典
            meta,  # 节点的元数据
        )

    # 调用映射函数，执行函数映射操作
    def call_map(
        self,
        f: torch.fx.GraphModule,  # 要映射执行的函数模块
        mapped_args: List[ProxyValue],  # 映射参数的代理值列表
        operands: List[ProxyValue],  # 操作数的代理值列表
        meta: NodeMetadata,  # 节点的元数据
    ) -> ProxyValue:  # 返回值类型为 ProxyValue
        xs = _unstack_pytree([arg.data for arg in mapped_args])[0]  # 解包映射参数并获取其数据
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in operands]))  # 调用子模块执行映射函数
        assert f_branch is not None
        return self._fx(
            "call_function",  # 执行函数调用操作
            torch.ops.higher_order.map_impl,  # 使用 Torch 的 higher_order.map_impl 运算
            (f_branch.graph_module, mapped_args, operands),  # 调用参数列表
            {},  # 空的关键字参数字典
            meta,  # 节点的元数据
        )

    # 调用获取项函数，执行获取特定项操作
    def call_getitem(
        self, value: ProxyValue,  # 要获取项的代理值
        key: int,  # 要获取的项的键
        meta: NodeMetadata,  # 节点的元数据
    ) -> ProxyValue:  # 返回值类型为 ProxyValue
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)  # 调用内部函数 _fx 进行函数调用操作

    # 输出结果函数，执行结果输出操作
    def output(self, results: List[Argument], meta: NodeMetadata) -> ProxyValue:  # 结果列表和节点的元数据作为参数
        return self._fx("output", "output", (results,), {}, meta)  # 调用内部函数 _fx 进行输出操作

    # 调用子模块函数，执行子模块的导出操作
    def call_submodule(
        self, graph_module: fx.GraphModule,  # 要调用的子模块对象
        inputs: Tuple[Argument, ...],  # 输入参数的元组
    ) -> PassResult:  # 返回类型为 PassResult
        prev_tracer, self.tracer = self.tracer, self.ExportTracer(
            self, graph_module.graph._codegen
        )  # 备份和设置追踪器对象
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode  # 设置假张量模式
        interpreter = self.ExportInterpreter(self, graph_module)  # 创建导出解释器对象
        prev_interpreter, self.interpreter = self.interpreter, torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )  # 备份和设置解释器对象
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)  # 映射输入数据
        with fx_traceback.preserve_node_meta():  # 保留节点元数据的上下文
            interpreter.run(*inputs_data)  # 运行解释器并传入映射的输入数据

        new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)  # 创建新的图模块对象

        self.tracer = prev_tracer  # 恢复追踪器对象
        self.interpreter = prev_interpreter  # 恢复解释器对象
        return PassResult(
            new_graph_module,  # 返回新的图模块对象
            True,  # 返回成功标志
        )
    # 调用方法，接受一个图模块作为参数，并返回处理结果
    def call(self, graph_module: fx.GraphModule) -> PassResult:
        # 检查是否已经初始化，如果没有则抛出错误
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        # 获取输入参数列表
        inputs = self.inputs(graph_module)

        # 初始化虚拟张量模式为 None
        fake_tensor_mode = None
        for i in inputs:
            # 检查输入参数是否为虚拟张量
            if isinstance(i, FakeTensor):
                # 确保仅有一个虚拟张量模式，否则抛出异常
                assert (
                    fake_tensor_mode is None or fake_tensor_mode is i.fake_mode
                ), "Multiple fake tensor mode detected."
                fake_tensor_mode = i.fake_mode

        # 如果没有检测到虚拟张量模式，则设置默认模式并创建 nullcontext 上下文管理器
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            # 如果检测到虚拟张量模式，则允许非虚拟输入，并设置追踪器的虚拟张量模式
            fake_tensor_mode.allow_non_fake_inputs = True
            self.tracer.fake_tensor_mode = fake_tensor_mode
            # 启用 Python 调度器并创建相应的上下文管理器
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]

        # 将追踪器的虚拟张量模式赋给对象的 fake_tensor_mode 属性
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        # 使用虚拟张量模式和调度器模式执行子模块调用
        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        # 返回调用子模块的结果
        return result
```