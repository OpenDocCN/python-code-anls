# `.\pytorch\torch\fx\interpreter.py`

```
# 导入需要的模块和类
# mypy: allow-untyped-defs
from .graph_module import GraphModule                     # 导入 GraphModule 类
from ._lazy_graph_module import _make_graph_module        # 导入 _make_graph_module 函数
from .graph import Graph                                  # 导入 Graph 类
from .node import Argument, Node, Target, map_arg, map_aggregate  # 导入 Argument, Node, Target, map_arg, map_aggregate 类和函数
from .proxy import Proxy                                  # 导入 Proxy 类
from ._symbolic_trace import Tracer                       # 导入 Tracer 类
from ._compatibility import compatibility                 # 导入 compatibility 函数
from . import config                                      # 导入 config 模块
import torch.fx.traceback as fx_traceback                 # 导入 torch.fx.traceback 模块
import torch                                              # 导入 torch 库
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union  # 导入需要的类型声明
import inspect                                            # 导入 inspect 库
from contextlib import contextmanager                     # 导入 contextmanager 类
from torch.hub import tqdm                                # 导入 tqdm 函数

__all__ = ['Interpreter', 'Transformer']                 # 模块导出的公共接口列表

@compatibility(is_backward_compatible=True)
class Interpreter:
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overrideable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(self, target : Target,
                                  args : Tuple, kwargs : Dict) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(n)

                def call_method(self, target : Target,
                                args : Tuple, kwargs : Dict) -> Any:
                    if target == 'neg':
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_close(result, torch.neg(input).sigmoid())
    """
    # 解释器类，执行 FX 图节点。可用于代码转换和分析过程。
    
    def __init__(self, graph_module: GraphModule):
        # 初始化方法，接收一个 GraphModule 实例作为参数
        pass

    def run(self, *args, **kwargs) -> Any:
        # 执行方法，运行解释器
        pass

    def run_node(self, n: Node) -> Any:
        # 执行单个节点方法
        pass

    def placeholder(self, target: Target) -> Any:
        # 占位符方法
        pass

    def get_attr(self, target: Target, name: str) -> Any:
        # 获取属性方法
        pass

    def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        # 调用函数方法
        pass

    def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        # 调用方法方法
        pass

    def call_module(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        # 调用模块方法
        pass

    def output(self, n: Node, value: Any) -> None:
        # 输出方法
        pass
    # 定义一个名为 Interpreter 的类，用于执行给定的 torch.nn.Module 对象
    @compatibility(is_backward_compatible=True)
    def __init__(self, module: torch.nn.Module, garbage_collect_values: bool = True, graph: Optional[Graph] = None):
        # 初始化 Interpreter 对象时，传入的模块对象
        self.module = module
        # 创建当前模块及其所有子模块的字典，以便进行查找和使用
        self.submodules = dict(self.module.named_modules())
        # 如果传入了 graph 参数，则使用传入的图来执行，否则使用模块自带的图
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self.module.graph
        # 环境变量字典，用于存储节点与其对应的值
        self.env : Dict[Node, Any] = {}
        # Interpreter 对象的名称
        self.name = "Interpreter"
        # 是否启用垃圾回收值功能，以在执行过程中优化内存使用
        self.garbage_collect_values = garbage_collect_values
        # 是否额外显示回溯信息
        self.extra_traceback = True
    
        # 如果启用了垃圾回收值功能
        if self.garbage_collect_values:
            # 通过反向遍历图的节点，记录每个节点的最后使用情况
            node_to_last_use : Dict[Node, Node] = {}
            # 记录每个节点的最后使用节点列表
            self.user_to_last_uses : Dict[Node, List[Node]] = {}
    
            # 函数用于注册节点的最后使用情况
            def register_last_uses(n : Node, user : Node):
                # 如果节点 n 尚未记录最后使用节点，则记录下来
                if n not in node_to_last_use:
                    node_to_last_use[n] = user
                    # 将使用节点 user 加入到节点 n 的最后使用节点列表中
                    self.user_to_last_uses.setdefault(user, []).append(n)
    
            # 反向遍历图中的节点
            for node in reversed(self.graph.nodes):
                # 对节点的参数列表应用 register_last_uses 函数
                map_arg(node.args, lambda n: register_last_uses(n, node))
                # 对节点的关键字参数列表应用 register_last_uses 函数
                map_arg(node.kwargs, lambda n: register_last_uses(n, node))
    
    @compatibility(is_backward_compatible=True)
    def run(self, *args, initial_env : Optional[Dict[Node, Any]] = None, enable_io_processing : bool = True) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        # 初始化环境变量，如果提供了初始环境，则使用提供的；否则使用空字典
        self.env = initial_env if initial_env is not None else {}

        # 如果启用了输入输出处理，则对传入的参数进行处理
        if enable_io_processing:
            args = self.graph.process_inputs(*args)

        # 使用迭代器来逐个获取位置参数，并设置为类属性
        self.args_iter : Iterator[Any] = iter(args)

        # 初始化进度条，显示节点数量和描述信息
        pbar = tqdm(total=len(self.graph.nodes),
                    desc=f"{self.name}: {str(list(self.graph.nodes)) if config.verbose_progress else ''}",
                    initial=0, position=0, leave=True, disable=config.disable_progress, delay=0)

        # 遍历图中的每个节点
        for node in self.graph.nodes:
            pbar.update(1)

            # 如果节点在环境变量中已存在，跳过执行
            if node in self.env:
                # 如果已经有这个值，可以用来进行部分计算的情况
                continue

            try:
                # 执行节点对应的运行操作，并将结果存入环境变量
                self.env[node] = self.run_node(node)
            except Exception as e:
                # 处理异常情况
                if self.extra_traceback:
                    msg = f"While executing {node.format_node()}"
                    msg = f'{e.args[0]}\n\n{msg}' if e.args else str(msg)
                    msg += f"\nOriginal traceback:\n{node.stack_trace}"
                    e.args = (msg,) + e.args[1:]
                    if isinstance(e, KeyError):
                        raise RuntimeError(*e.args) from e
                raise

            # 如果设置了垃圾回收值，删除用户最后使用的节点的环境变量值
            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            # 如果节点是输出节点，则返回处理后的输出值（如果启用了输入输出处理），否则返回原始输出值
            if node.op == 'output':
                output_val = self.env[node]
                return self.graph.process_outputs(output_val) if enable_io_processing else output_val

    @compatibility(is_backward_compatible=True)
    # 使用“boxed”调用约定运行模块，并返回结果。这种约定要求传入一个参数列表，解释器会清除它们，确保输入张量及时释放。
    def boxed_run(self, args_list):
        args_iter = iter(args_list)  # 创建参数列表的迭代器
        env = {}
        for n in self.graph.nodes:  # 遍历图中的节点
            if n.op == "placeholder":  # 如果节点是占位符
                env[n] = next(args_iter)  # 将下一个参数与占位符关联起来
        args_list.clear()  # 清空参数列表
        return self.run(initial_env=env)  # 运行模块，使用创建的环境

    # 使用上下文管理器设置当前节点
    @contextmanager
    def _set_current_node(self, node):
        with fx_traceback.set_current_meta(node):  # 设置当前节点的元信息
            yield  # 返回控制权

    # 运行特定节点并返回结果
    @compatibility(is_backward_compatible=True)
    def run_node(self, n : Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        with self._set_current_node(n):  # 使用当前节点设置上下文管理器
            args, kwargs = self.fetch_args_kwargs_from_env(n)  # 从环境中获取参数和关键字参数
            assert isinstance(args, tuple)  # 断言args是元组类型
            assert isinstance(kwargs, dict)  # 断言kwargs是字典类型
            return getattr(self, n.op)(n.target, args, kwargs)  # 调用相应的操作（如placeholder、get_attr等）

    # 主要的节点运行API

    # 执行placeholder节点的方法
    @compatibility(is_backward_compatible=True)
    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)  # 断言target是字符串类型
        if target.startswith('*'):  # 如果target以'*'开头，例如`*args`
            # 对于星号参数，例如`*args`，从参数列表中获取所有剩余的值。
            return list(self.args_iter)  # 返回参数迭代器的列表形式
        else:
            try:
                return next(self.args_iter)  # 尝试从参数迭代器中获取下一个值
            except StopIteration as si:  # 如果迭代器已经停止
                if len(args) > 0:
                    return args[0]  # 返回第一个传入的位置参数
                else:
                    raise RuntimeError(f'Expected positional argument for parameter {target}, but one was not passed in!') from si
    @compatibility(is_backward_compatible=True)
    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        # 确保 `target` 是一个字符串，表明要获取的属性名
        assert isinstance(target, str)
        # 调用 fetch_attr 方法来获取属性值
        return self.fetch_attr(target)

    @compatibility(is_backward_compatible=True)
    def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        # 确保 `target` 不是字符串，表明要调用的是一个函数
        assert not isinstance(target, str)

        # 执行函数调用，并返回结果
        return target(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] 是此方法调用的 `self` 对象
        self_obj, *args_tail = args

        # 确保 `target` 是一个字符串，表明要调用的是一个方法
        assert isinstance(target, str)
        # 执行方法调用，并返回结果
        return getattr(self_obj, target)(*args_tail, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # 从环境中检索执行的 args 和 kwargs 值

        # 执行方法并返回结果
        assert isinstance(target, str)  # 确保 target 是字符串类型
        submod = self.fetch_attr(target)  # 获取目标字符串表示的属性值

        return submod(*args, **kwargs)  # 调用 submod 对象的方法，传入 args 和 kwargs

    @compatibility(is_backward_compatible=True)
    def output(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
        return args[0]  # 返回 args 元组的第一个元素作为输出值

    # Helper methods
    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target : str):
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualified name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split('.')  # 使用点号分割目标字符串
        attr_itr = self.module  # 设置属性迭代器为 self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):  # 检查当前属性迭代器是否有指定的属性名
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)  # 获取当前属性迭代器的属性值
        return attr_itr  # 返回最终获取的属性值

    @compatibility(is_backward_compatible=True)
    # 从当前执行环境中获取节点 `n` 的 `args` 和 `kwargs` 的具体数值。
    # `args` 是一个元组，存储了节点 `n` 的参数值。
    # `kwargs` 是一个字典，存储了节点 `n` 的关键字参数值。
    def fetch_args_kwargs_from_env(self, n : Node) -> Tuple[Tuple, Dict]:
        """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
        # 使用 `map_nodes_to_values` 方法获取节点 `n` 的 `args` 的具体值，并断言其为元组类型
        args = self.map_nodes_to_values(n.args, n)
        assert isinstance(args, tuple)
        # 使用 `map_nodes_to_values` 方法获取节点 `n` 的 `kwargs` 的具体值，并断言其为字典类型
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        assert isinstance(kwargs, dict)
        # 返回 `args` 和 `kwargs`，这两者分别存储了节点 `n` 的参数和关键字参数的具体值
        return args, kwargs

    # 声明一个兼容性装饰器，用于标记方法为向后兼容的
    @compatibility(is_backward_compatible=True)
    # 将节点 `n` 的 `args` 映射到其具体值，并返回结果
    def map_nodes_to_values(self, args : Argument, n : Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """
        # 定义一个内部函数 `load_arg`，用于加载节点 `n_arg` 的具体值
        def load_arg(n_arg : Node) -> Any:
            # 如果环境中不存在节点 `n_arg`，则抛出运行时异常，并提供详细信息
            if n_arg not in self.env:
                raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() '
                                   f'to diagnose such issues')
            # 返回节点 `n_arg` 在环境中的具体值
            return self.env[n_arg]
        # 使用 `map_arg` 函数将 `args` 中的每个元素映射到其具体值，并返回映射结果
        return map_arg(args, load_arg)
# 使用装饰器指定此类在向后兼容性方面是兼容的
@compatibility(is_backward_compatible=True)
# 定义一个名为 Transformer 的类，它是 Interpreter 类的子类
class Transformer(Interpreter):
    """
    ``Transformer`` 是一种特殊类型的解释器，用于生成新的 ``Module``。
    它公开了一个 ``transform()`` 方法，用于返回转换后的 ``Module``。
    ``Transformer`` 不需要参数来运行，与 ``Interpreter`` 不同。
    ``Transformer`` 完全通过符号方式工作。

    Example:

        假设我们想要将所有 ``torch.neg`` 的实例与 ``torch.sigmoid`` 互换，并反之亦然
        （包括它们的 ``Tensor`` 方法等效项）。我们可以这样子类化 ``Transformer``::

            class NegSigmSwapXformer(Transformer):
                def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(n)

                def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
                    if target == 'neg':
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)

            transformed : torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): 要进行转换的 ``Module``。
    """

    @compatibility(is_backward_compatible=True)
    # 初始化方法，接受一个 module 参数作为输入
    def __init__(self, module):
        # 调用父类 Interpreter 的初始化方法
        super().__init__(module)
        # 创建一个新的 Graph 对象作为新图
        self.new_graph = Graph()
        # 设置新图的代码生成器与原始模块图的代码生成器相同
        self.new_graph.set_codegen(module.graph._codegen)

        # 定义一个内部类 TransformerTracer，继承自 Tracer
        class TransformerTracer(Tracer):
            # 初始化方法接受一个 graph 参数作为输入
            def __init__(self, graph: Graph):
                super().__init__()
                self.graph = graph
                # 初始化一个字典，用于存储张量对象及其对应的属性名称
                self.tensor_attrs: Dict[torch.Tensor, str] = {}  # type: ignore[assignment]

            # 判断是否为叶子模块的方法，暂时未使用参数 _ 和 __
            def is_leaf_module(self, _, __) -> bool:
                return True

        # 创建 TransformerTracer 的实例，并将新图作为参数传入
        self.tracer = TransformerTracer(self.new_graph)
        # 设置 tracer 的根节点为传入的模块对象
        self.tracer.root = module

    @compatibility(is_backward_compatible=True)
    # 用装饰器指定此方法在向后兼容性方面是兼容的
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Proxy:
        """
        Execute a ``placeholder`` node. In ``Transformer``, this is
        overridden to insert a new ``placeholder`` into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        # 确保 `target` 是字符串类型
        assert isinstance(target, str)
        # 如果有参数，将第一个参数作为默认值，否则使用 `inspect.Signature.empty`
        default_value = next(iter(args)) if args else inspect.Signature.empty
        # 创建并返回一个 `Proxy` 对象，代表新的 `placeholder` 节点
        return Proxy(self.new_graph.placeholder(target, default_value=default_value), self.tracer)

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Proxy:
        """
        Execute a ``get_attr`` node. In ``Transformer``, this is
        overridden to insert a new ``get_attr`` node into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        # 确保 `target` 是字符串类型
        assert isinstance(target, str)
        # 创建并返回一个 `Proxy` 对象，代表新的 `get_attr` 节点
        return self.tracer.create_proxy("get_attr", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        # Override so that the leaf module policy from `self.tracer` is respected.
        # 确保 `target` 是字符串类型
        assert isinstance(target, str)
        # 获取目标属性并调用，返回结果
        submod = self.fetch_attr(target)
        return self.tracer.call_module(submod, submod.forward, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        # Override so that functions that were wrapped are still wrapped.
        # 调用函数并返回一个 `Proxy` 对象，代表新的 `call_function` 节点
        return self.tracer.create_proxy('call_function', target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        # 在转换过程中保留节点的元数据
        with fx_traceback.preserve_node_meta():
            # 运行超类的 `run` 方法，禁用 IO 处理
            result = super().run(enable_io_processing=False)
        # 如果结果不为 None，则对结果进行处理并输出
        if result is not None:
            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a
            # 将结果映射聚合后输出到新图的输出中
            self.new_graph.output(map_aggregate(result, strip_proxy))
        # 使用 `_make_graph_module` 创建并返回经过转换后的 `GraphModule` 对象
        return _make_graph_module(self.module, self.new_graph)
```