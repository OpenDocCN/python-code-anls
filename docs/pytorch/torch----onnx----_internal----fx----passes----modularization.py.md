# `.\pytorch\torch\onnx\_internal\fx\passes\modularization.py`

```py
# mypy: allow-untyped-defs
from __future__ import annotations  # 导入未来的注解语法支持

import abc  # 导入抽象基类模块

import collections  # 导入集合模块
import copy  # 导入复制模块
import operator  # 导入操作符模块

from typing import (  # 导入类型提示相关模块
    Any,
    Dict,
    Final,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch  # 导入PyTorch模块
import torch.fx  # 导入PyTorch FX模块
from torch.onnx._internal import _beartype  # 导入PyTorch ONNX内部的_beartype

from torch.onnx._internal.fx import _pass, diagnostics  # 导入PyTorch ONNX内部的_pass和diagnostics
from torch.utils import _pytree as pytree  # 导入PyTorch工具模块中的_pytree

_FX_TRACER_NN_MODULE_META_TYPE = Tuple[str, type]  # FX符号跟踪器生成的`node.meta["nn_module_stack"].items()`中的项的传统类型
_FX_TRACER_NN_MODULE_STACK_META_TYPE = collections.OrderedDict  # FX符号跟踪器生成的`node.meta["nn_module_stack"]`的传统类型

_DYNAMO_NN_MODULE_META_TYPE = Tuple[str, Tuple[str, type]]  # FX Dynamo跟踪器生成的`node.meta["nn_module_stack"].items()`中的项的类型
_DYNAMO_NN_MODULE_STACK_META_TYPE = Dict[str, _DYNAMO_NN_MODULE_META_TYPE]  # FX Dynamo跟踪器生成的`node.meta["nn_module_stack"]`的类型


class _ModuleMeta:
    """模块的元信息类。

    用于以更结构化的方式表示模块信息。从`node.meta["nn_module_stack"].items()`中解析原始模块信息。

    查看`from_raw_meta`、`from_fx_tracer_produced_raw_meta`和`from_dynamo_produced_raw_meta`的用法以了解如何创建实例。

    Attributes:
        _module_class: 模块的类。例如 `torch.nn.module.sparse.Embedding`。
        _module_name: 模块的名称。例如 `L__self___h_1_mlp_c_proj`。
        _raw_meta: 原始元信息 '(module_name, node.meta["nn_module_stack"][module_name])'。
    """

    _module_class: Final[Optional[Union[type, str]]]  # 不可变的最终属性，模块的类
    _module_name: Final[str]  # 不可变的最终属性，模块的名称
    _raw_meta: Final[Tuple[Any, Any]]  # 不可变的最终属性，原始元信息

    @_beartype.beartype
    def __init__(
        self,
        module_name: str,
        module_class: Optional[Union[type, str]],
        raw_meta: Tuple[Any, Any],
    ):
        """初始化方法。

        Args:
            module_name: 模块的名称。
            module_class: 模块的类。
            raw_meta: 原始元信息。
        """
        self._module_name = module_name
        self._module_class = module_class
        self._raw_meta = raw_meta

    @property
    def module_display_name(self) -> str:
        """模块的显示名称。

        例如 `h_1_mlp_c_proj`。
        """
        # 例如，从 'L__self___h_1_mlp_c_proj' 到 'h_1_mlp_c_proj'。
        name = self.module_name
        if name.startswith("L__self___"):
            name = name[len("L__self___") :]
        return name

    @property
    def qualified_module_class_name(self) -> str:
        """模块类的限定名称。

        例如 `torch_nn_module_sparse_Embedding`。
        """
        if self._module_class is None:
            return ""
        mod_cls = self._module_class
        if isinstance(mod_cls, type):
            mod_cls = mod_cls.__module__ + "." + mod_cls.__qualname__
        return mod_cls.replace(".", "_")

    @property
    def module_class_name(self) -> str:
        """Name of the module class.

        E.g. `Embedding`.
        """
        # 如果模块类别为空，则返回空字符串
        if self._module_class is None:
            return ""
        # 如果模块类别是一个类对象，则返回其类名
        if isinstance(self._module_class, type):
            return self._module_class.__name__
        # 否则返回模块类别本身（通常是字符串形式的类名）
        return self._module_class

    @property
    def module_name(self) -> str:
        """Name of the module.

        E.g. `L__self___h_1_mlp_c_proj`.
        """
        # 返回模块的名称
        return self._module_name

    @property
    def raw_meta(self) -> Tuple[Any, Any]:
        """Returns the raw module meta data.

        I.e. (module_name, node.meta['nn_module_stack'][module_name]).
        """
        # 返回原始模块元数据
        return self._raw_meta

    def __eq__(self, __value: object) -> bool:
        # 判断两个 _ModuleMeta 对象是否相等
        if not isinstance(__value, _ModuleMeta):
            return False
        return (
            self._module_name == __value._module_name
            and self._module_class == __value._module_class
        )

    def __hash__(self) -> int:
        # 返回 _ModuleMeta 对象的哈希值
        return hash((self._module_name, self._module_class))

    def __repr__(self) -> str:
        # 返回 _ModuleMeta 对象的字符串表示
        return f"ModuleMeta(name={self._module_name}, class={self._module_class})"

    @classmethod
    def create_root(cls) -> _ModuleMeta:
        """Create an empty module meta representing root module."""
        # 创建一个表示根模块的空模块元数据对象
        return _ModuleMeta("", None, ("", None))

    @classmethod
    def from_fx_tracer_produced_raw_meta(
        cls, raw_meta: _FX_TRACER_NN_MODULE_META_TYPE
    ) -> _ModuleMeta:
        """Create a module meta from raw meta produced by FX symbolic tracer."""
        # 从 FX 符号跟踪器生成的原始元数据创建模块元数据对象
        module_name, module_class = raw_meta
        return _ModuleMeta(module_name, module_class, raw_meta)

    @classmethod
    def from_dynamo_produced_raw_meta(
        cls, raw_meta: _DYNAMO_NN_MODULE_META_TYPE
    ) -> _ModuleMeta:
        """Create a module meta from raw meta produced by FX dynamo tracer."""
        # 从 FX dynamo 跟踪器生成的原始元数据创建模块元数据对象
        module_name, (qualified_name, module_class) = raw_meta
        return _ModuleMeta(module_name, module_class, raw_meta)

    @classmethod
    def from_raw_meta(
        cls,
        raw_meta: Union[_FX_TRACER_NN_MODULE_META_TYPE, _DYNAMO_NN_MODULE_META_TYPE],
    ) -> _ModuleMeta:
        # 根据不同类型的原始元数据创建模块元数据对象
        if (
            isinstance(raw_meta, tuple)
            and len(raw_meta) == 2
            and isinstance(raw_meta[1], type)
        ):
            # 如果是 FX 符号跟踪器生成的元数据类型，则调用相应的方法创建对象
            return _ModuleMeta.from_fx_tracer_produced_raw_meta(raw_meta)
        if (
            isinstance(raw_meta, tuple)
            and len(raw_meta) == 2
            and isinstance(raw_meta[1], tuple)
        ):
            # 如果是 FX dynamo 跟踪器生成的元数据类型，则调用相应的方法创建对象
            return _ModuleMeta.from_dynamo_produced_raw_meta(raw_meta)
        # 如果类型未知，则抛出类型错误异常
        raise TypeError(
            f"Unknown type of raw meta item from node.meta['nn_module_stack'].items(): {type(raw_meta)}"
        )
class _ModuleStackMeta:
    """Meta information about the module call stack.

    This class is used to represent the module call stack information in a more
    structured way. It parses raw module stack information from `node.meta["nn_module_stack"]`.

    Example of raw module stack information:

        If produced by dynamo:

            {
                'L__self___h_1': (
                    "L['self'].h[1]",
                    <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>
                ),
                'L__self___h_1_attn': (
                    "L['self'].h[1].attn",
                    <class 'transformers.models.gpt2.modeling_gpt2.GPT2Attention'>
                )
            }

        If produced by fx.symbolic_trace:

            {
                'h.1': <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>,
                'h.1.attn': <class 'transformers.models.gpt2.modeling_gpt2.GPT2Attention'>
            }
    """

    _module_stack: Final[List[_ModuleMeta]]  # type: ignore[misc]

    @_beartype.beartype
    def __init__(
        self,
        nn_module_stack_meta: Optional[
            Union[
                _FX_TRACER_NN_MODULE_STACK_META_TYPE, _DYNAMO_NN_MODULE_STACK_META_TYPE
            ]
        ],
        is_exported_program: bool = True,
    ):
        self._module_stack = []  # 初始化模块堆栈列表

        # 如果没有提供模块堆栈元信息，直接返回
        if nn_module_stack_meta is None:
            return

        # 复制原始元信息以防修改
        raw_meta = copy.copy(nn_module_stack_meta)

        # 遍历原始元信息中的每个项
        for item in raw_meta.items():
            # 如果是通过 torch.export.export 生成的堆栈，需要跳过另一个调用堆栈层
            if is_exported_program:
                is_exported_program = False
                continue

            # 将从原始元信息中创建的模块元信息压入堆栈
            self.push(_ModuleMeta.from_raw_meta(item))

    def __len__(self) -> int:
        return len(self._module_stack)  # 返回模块堆栈的长度

    def __getitem__(self, index: int) -> _ModuleMeta:
        return self._module_stack[index]  # 获取指定索引位置的模块元信息对象

    def __iter__(self) -> Iterator[_ModuleMeta]:
        return iter(self._module_stack)  # 返回模块堆栈的迭代器

    def is_empty_or_root(self) -> bool:
        return len(self._module_stack) == 0  # 判断模块堆栈是否为空或者仅有根节点

    def top(self) -> _ModuleMeta:
        """Returns the top module meta in the stack. I.e., the meta for leaf module.

        Example:

            Consider the following module stack:

            stack = [GPT, block1, Attention_1, MLP]

            stack.top() == MLP
        """
        if self.is_empty_or_root():
            return _ModuleMeta.create_root()  # 如果堆栈为空或只有根节点，返回根节点元信息

        return self._module_stack[-1]  # 返回堆栈顶部（最后一个元素）的模块元信息对象

    @_beartype.beartype
    def is_superset_of(
        self,
        module_stack: _ModuleStackMeta,
        ...
    ) -> bool:
        """Determines if self is a superset of the provided module stack.

        I.e., If self includes all elements from the provided module stack, plus additional
        elements on top. If self is empty or root, this method always return False.

        Example:

            Consider the following module stack:

            stack_1 = [GPT, block1, Attention_1, MLP]
            stack_2 = [GPT, block1]

            stack_1.is_superset_of(stack_2) == True
            stack_2.is_superset_of(stack_1) == False

            stack_3 = [GPT, block2, Attention_1]

            stack_1.is_superset_of(stack_3) == False
            stack_3.is_superset_of(stack_1) == False
        """
        # 如果 self 是空或根节点，则返回 False
        if self.is_empty_or_root():
            return False

        # 如果 module_stack 是空或根节点，返回 True
        if module_stack.is_empty_or_root() is None:
            return True

        # 如果 self 的长度小于等于 module_stack 的长度，则返回 False
        if len(self) <= len(module_stack):
            return False

        # 遍历 module_stack，比较 self 和 module_stack 对应位置的元素
        for i, parent_key in enumerate(module_stack):
            if self[i] != parent_key:
                return False

        # 如果没有发现不匹配的元素，则 self 是 module_stack 的超集，返回 True
        return True

    def push(self, module_meta: _ModuleMeta) -> None:
        """Pushes a module meta to the stack."""
        # 将 module_meta 添加到 _module_stack 中
        self._module_stack.append(module_meta)

    @_beartype.beartype
    def __eq__(self, __value: object) -> bool:
        # 检查是否 __value 是 _ModuleStackMeta 类型的对象
        if not isinstance(__value, _ModuleStackMeta):
            return False
        # 比较 self 和 __value 的 _module_stack 是否相等
        return self._module_stack == __value._module_stack

    @property
    def raw_meta(self) -> Optional[Dict[str, Tuple[str, type]]]:
        """Returns the raw module stack meta data, i.e. node.meta['nn_module_stack']."""
        # 返回 _module_stack 中每个 module_meta 的 raw_meta 构成的字典
        return {
            module_meta.raw_meta[0]: module_meta.raw_meta[1]
            for module_meta in self._module_stack
        }

    def __repr__(self) -> str:
        # 返回对象的字符串表示，包含 _module_stack 的内容
        return f"ModuleStackMeta({self._module_stack})"

    @property
    def module_display_name(self) -> str:
        """Returns the module display name of the top module."""
        # 返回栈顶模块的 module_display_name
        return self.top().module_display_name

    @property
    def qualified_module_class_name(self) -> str:
        """Returns the qualified module class name of the top module."""
        # 返回栈顶模块的 qualified_module_class_name
        return self.top().qualified_module_class_name

    @property
    def module_class(self) -> Optional[Union[type, str]]:
        """Returns the module class of the top module."""
        # 返回栈顶模块的 module_class
        return self.top()._module_class
# 定义函数 _module_stack_meta_from_node，根据给定的 torch.fx.Node 节点创建 _ModuleStackMeta 对象
# 如果 is_exported_program 为 True，则标记为导出程序
def _module_stack_meta_from_node(
    node: torch.fx.Node, is_exported_program: bool = False
) -> _ModuleStackMeta:
    # 从节点的元数据中获取 nn_module_stack 属性，作为 _ModuleStackMeta 的参数
    return _ModuleStackMeta(
        node.meta.get("nn_module_stack"), is_exported_program=is_exported_program
    )


# 定义函数 _get_unique_module_name，为给定的模块名称创建唯一的模块名称
def _get_unique_module_name(module_names: Dict[str, int], module_name: str) -> str:
    # 如果 module_name 在 module_names 中不存在，则将其初始化为 0
    module_names.setdefault(module_name, 0)
    # 递增 module_name 的计数器
    module_names[module_name] += 1
    # 返回格式化后的唯一模块名称，附加当前计数器值
    return f"{module_name}_{module_names[module_name]}"


# 定义类 _IRNode，继承自 abc.ABC，用作 IR 节点的基类
class _IRNode(abc.ABC):
    """Base class for IR nodes.

    IR nodes are used for Modularize pass only. They add a layer of abstraction on top of
    torch.fx.Node.

    [NOTE: Modularize Pass Implementation]
    The main job of the pass is to group `fx.Node`s that belong to the same `nn.Module`
    forward call, and then create `call_module` node and sub `fx.GraphModule` from them.
    Each `fx.Node` possesses an `nn_module_stack` meta data that contains information
    about the module call stack. See `_ModuleStackMeta` for examples.

    Analysis step
    -------------

    Each module call is identified by a set of base stack layers. For each module call,
    the pass creates a `_ModuleNode` and groups the sequence of nodes that shares the
    same base stack layers.

    For example,

        stack_of_node_0 = [GPT, block0]
        stack_of_node_1 = [GPT, block1]
        stack_of_node_2 = [GPT, block1, Attention1, MLP]
        stack_of_node_3 = [GPT, block1, Attention1]
        stack_of_node_4 = [GPT, block2]

    All nodes belong to the `GPT` module call, since they share the base stack layers [GPT].
    [node_1, node_2, node_3] are grouped for `GPT.block1`, because they share the base
    stack layers [GPT, block1]. And [node_2, node_3] for `GPT.block1.Attention1`, [node_0]
    for `GPT.block0`, and [node_4] for `GPT.block2` respectfully.

    After the analysis step, a hierarchical representation is generated.

    For above example, the representation is:

        _ModuleNode(GPT)
            _ModuleNode(block0)
                _LeafNode(node_0)
            _ModuleNode(block1)
                _LeafNode(node_1)
                _ModuleNode(Attention1)
                    _ModuleNode(MLP)
                        _LeafNode(node_2)
                _LeafNode(node_3)
            _ModuleNode(block2)
                _LeafNode(node_4)

    Construction step
    -----------------

    The second step is to build the actual `call_module` node and the sub `fx.GraphModule`.
    This is done recursively from the leaf `_ModuleNode` to the root.

    For example, the first submodule to be built is `GPT.block1.Attention1.MLP`. Below pair
    is generated from `_ModuleNode(MLP)`.

        fx.GraphModule(GPT.block1.Attention1.MLP)
            graph:
                node_2

        new_mlp_node = `call_module[GPT.block1.Attention1.MLP](...)`

    Next, the `GPT.block1.Attention1` submodule is built. Below is generated from
    _ModuleNode(MLP).
    """
    pass
    # 定义一个抽象基类属性，表示模块节点的堆栈元数据
    @property
    @abc.abstractmethod
    def stack_meta(self) -> _ModuleStackMeta:
        """返回与此节点关联的模块堆栈元数据。"""
        ...
    
    # 定义一个抽象基类属性，表示模块节点的堆栈跟踪信息
    @property
    @abc.abstractmethod
    def stack_trace(self) -> Optional[str]:
        """返回与此节点关联的堆栈跟踪信息，如果没有则返回 None。"""
        ...
class _ModuleNode(_IRNode):
    """Representing a sequence of fx.Nodes to be formed into a fx.GraphModule.

    This class encapsulates metadata and provides building block methods to construct this
    layered abstraction from a sequence of flat fx.Nodes.

    Attributes:
    - _stack_meta: Metadata of the module stack.
    - _nodes: List of IR nodes in the module.
    - _reference_root_module: Reference to the root flat fx.GraphModule instance.
    """

    def __init__(
        self, reference_root_module: torch.fx.GraphModule, stack_meta: _ModuleStackMeta
    ):
        # 初始化 ModuleNode 实例
        self._stack_meta = stack_meta  # 设置模块堆栈的元数据
        self._nodes: List[_IRNode] = []  # 初始化空列表以存储模块中的 IR 节点
        self._reference_module = reference_root_module  # 设置对根 flat fx.GraphModule 实例的引用

    @property
    def stack_meta(self) -> _ModuleStackMeta:
        # 返回模块堆栈的元数据
        return self._stack_meta

    @property
    def stack_trace(self) -> Optional[str]:
        assert self._nodes  # 确保节点列表不为空
        return self._nodes[0].stack_trace  # 返回第一个节点的堆栈跟踪信息

    def __str__(self) -> str:
        # 返回 ModuleNode 的字符串表示
        return f"ModuleNode({self._stack_meta})"

    def is_same_module_as(self, node: _IRNode) -> bool:
        """Determines if the provided node pertains to the same module as this node."""
        # 判断给定的节点是否属于与当前节点相同的模块
        return self.stack_meta == node.stack_meta

    def is_parent_module_of(self, node: _IRNode) -> bool:
        """Determines if this node represents a parent module of the provided node."""
        # 判断当前节点是否代表提供的节点的父模块
        return node.stack_meta.is_superset_of(self.stack_meta)
    def add_leaf_node(self, leaf_node: _LeafNode) -> None:
        """Adds a leaf node to the module.

        The leaf node must belong to the same or a child module. This method will recursively
        construct _ModuleNode instance based on the stack_meta information of the leaf node.
        """
        # 如果 leaf_node 属于当前模块或其子模块，将其添加到节点列表中
        if self.is_same_module_as(leaf_node) or leaf_node.fx_op == "call_module":
            self._nodes.append(leaf_node)
        elif leaf_node.fx_op == "placeholder":
            # 对于 placeholder，虽然原始的 placeholder 没有 nn_module_stack，但是通过导出的程序
            # 中提取的 placeholder 会保留其原始的 nn_module_stack。在这里我们需要避免构建子模块。
            self._nodes.append(leaf_node)
        elif self.is_parent_module_of(leaf_node):
            # 当前节点属于一个子模块。
            # 检查最后一个节点是否为子模块，并且是否是当前节点的父模块。
            last_node = self._nodes[-1] if self._nodes else None
            if isinstance(last_node, _ModuleNode) and (
                last_node.is_parent_module_of(leaf_node)
                or last_node.is_same_module_as(leaf_node)
            ):
                # 当前节点属于最后一个节点。
                last_node.add_leaf_node(leaf_node)
            else:
                # 为当前模块的直接子模块创建一个新的 SubmoduleNode。
                # leaf_node 可能是当前模块的孙子节点。
                # 例如：
                #   self.stack_meta = [A, B, C]
                #   leaf_node.stack_meta = [A, B, C, D, E, F]
                # 创建一个新的 ModuleNode，stack_meta = [A, B, C, D]，并将 leaf_node 添加到其中。
                stack_meta = copy.deepcopy(self.stack_meta)
                stack_meta.push(leaf_node.stack_meta[len(self.stack_meta)])
                last_node = _ModuleNode(
                    self._reference_module,
                    stack_meta,
                )
                self._nodes.append(last_node)
                last_node.add_leaf_node(leaf_node)
        else:
            # 如果 leaf_node 不属于当前模块或其子模块，抛出 AssertionError。
            raise AssertionError(
                f"Node {leaf_node} ({leaf_node.stack_meta}) does not belong to module "
                f"{self._stack_meta}."
            )

    def fx_nodes(self) -> Generator[torch.fx.Node, None, None]:
        """Returns an iterator for the sequence of fx nodes this instance holds."""
        # 返回一个迭代器，遍历当前实例持有的 fx 节点序列。
        for node in self._nodes:
            if isinstance(node, _ModuleNode):
                # 如果节点是 _ModuleNode 类型，则递归地生成其 fx 节点。
                yield from node.fx_nodes()
            else:
                # 否则，断言节点为 _LeafNode 类型，然后返回其 fx_node。
                assert isinstance(node, _LeafNode)
                yield node.fx_node
    def module_inputs(self) -> Sequence[torch.fx.Node]:
        """Extract module inputs from the sequence of fx nodes this instance holds.
        
        All node args that are produced by nodes outside of the module are considered module
        inputs. The order of returned module inputs is the same as their use order.

        ### Known limitations

        The original ordering of module inputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module inputs.
        """
        nodes = list(self.fx_nodes())  # 获取当前实例持有的所有 fx 节点列表
        assert len(nodes) > 0, "Cannot extract module inputs from empty nodes."  # 断言节点列表不为空

        module_inputs: Dict[torch.fx.Node, None] = {}  # 用于存储模块输入的字典，键为节点，值为 None
        node_set: Set[torch.fx.Node] = set(nodes)  # 将节点列表转换为集合，方便后续判断节点是否在模块内部

        def _extract_arg_if_node_outside_module(arg: Any):
            """Helper function to check if an argument is a node outside the module."""
            if isinstance(arg, torch.fx.Node) and arg not in node_set:
                module_inputs[arg] = None  # 如果节点是模块外部的节点，则将其添加到模块输入字典中

        for node in nodes:
            pytree.tree_map(_extract_arg_if_node_outside_module, node.args)  # 遍历节点的参数并检查是否是模块外部的节点
            pytree.tree_map(_extract_arg_if_node_outside_module, node.kwargs)  # 遍历节点的关键字参数并检查是否是模块外部的节点

        return list(module_inputs.keys())  # 返回模块输入节点的列表

    def module_outputs(self) -> Sequence[torch.fx.Node]:
        """Extract module outputs from the sequence of fx nodes this instance holds.
        
        All nodes that are used by nodes outside of the module are considered module
        outputs. The order of returned module outputs is the same as their creation order.

        ### Known limitations

        The original ordering of module outputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module outputs.
        """
        nodes = list(self.fx_nodes())  # 获取当前实例持有的所有 fx 节点列表
        assert len(nodes) > 0, "Cannot extract module inputs from empty nodes."  # 断言节点列表不为空

        # Need ordered set. Emulate with dict.
        module_outputs: Dict[torch.fx.Node, None] = {}  # 用于存储模块输出的字典，键为节点，值为 None
        node_set: Set[torch.fx.Node] = set(nodes)  # 将节点列表转换为集合，方便后续判断节点是否在模块内部

        for node in nodes:
            if any(user not in node_set for user in node.users):
                module_outputs[node] = None  # 如果节点被模块外部的节点使用，则将其添加到模块输出字典中

        return list(module_outputs.keys())  # 返回模块输出节点的列表
class _LeafNode(_IRNode):
    """Representing a single fx.Node."""

    def __init__(self, node: torch.fx.Node, is_exported_program: bool = False):
        # 初始化 LeafNode 对象，存储传入的 torch.fx.Node 对象
        self._node = node
        # 从给定的 fx.Node 中获取模块堆栈元数据，存储在 _stack_meta 属性中
        self._stack_meta = _module_stack_meta_from_node(
            node, is_exported_program=is_exported_program
        )

    @property
    def fx_op(self) -> str:
        """Syntax sugar for self.fx_node.op."""
        # 返回当前 LeafNode 对象所包装的 fx.Node 的操作类型
        return self._node.op

    @property
    def fx_node(self) -> torch.fx.Node:
        """Returns the fx.Node this instance represents."""
        # 返回当前 LeafNode 对象所包装的 fx.Node 对象
        return self._node

    @property
    def stack_meta(self) -> _ModuleStackMeta:
        """Returns the module stack meta data associated with this node."""
        # 返回与当前节点关联的模块堆栈元数据对象
        return self._stack_meta

    @property
    def stack_trace(self) -> Optional[str]:
        """Returns the stack trace associated with this node."""
        # 返回与当前节点关联的堆栈跟踪信息，如果没有则返回 None
        return self.fx_node.meta.get("stack_trace")

    def __str__(self) -> str:
        # 返回描述当前 LeafNode 对象的字符串表示
        return f"LeafNode({self._node})"


class Modularize(_pass.Transform):
    """Transforms a flattened `fx.GraphModule` into a modular structure.

    In the flattened `fx.GraphModule`, each `nn.Module` forward call has been traced as
    a sequence of `fx.Node`s. All these `fx.Node`s are flattened and reside in the same
    `fx.GraphModule`. `fx.GraphModule` could be from `torch.export.ExportedProgram` or
    directly generated by `torch._dynamo.export` with torch.nn.Module.

    This pass generates a new `fx.GraphModule`. It groups the flattened `fx.Node`s that belong
    to the same `nn.Module` forward call into a sub `fx.GraphModule`. It then replaces the
    sequence of flattened `fx.Node`s with a single `call_module` node, which is linked with
    the sub `fx.GraphModule` by `node.target`. The sub `fx.GraphModule` is registered as a
    submodule of the new `fx.GraphModule`.

    The process is done based on information from the `nn_module_stack` metadata of each node, i.e.
    `node.meta["nn_module_stack"]`. For more implementation details, see [NOTE: Modularize Pass Implementation].

    An fx submodule under this context can typically be interpreted in three different ways:

        1. As an embodiment of an nn.Module class, which is considered stateless.
        Its execution path can vary depending on the configuration of module initialization,
        which should also be part of the inputs.

        2. As a representation of an nn.Module instance. It maintains the state initialized in the module.
        The execution path can vary based on actual input data.

        3. As a captured call of an nn.Module instance, where the execution path
        is set.

    The generality decreases along this list. Within the scope of this function, the pass
    creates fx submodules according to the third interpretation.

    The first interpretation is the most general case. It requires complex analysis and additional
    metadata and code information to construct its general form. Consider an example nn.Module
    """

    # 构造函数，初始化 Modularize 类
    def __init__(self):
        # 调用父类 _pass.Transform 的构造函数
        super().__init__()
    # 装饰器函数声明，使用beartype库提供的类型检查功能
    @_beartype.beartype
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        is_exported_program: bool = False,
    ):
        # 调用父类的构造函数初始化对象
        super().__init__(diagnostic_context, module)
        # 将传入的模块对象保存到实例变量中
        self.module = module
        # 将是否导出程序的标志保存到实例变量中
        self.is_exported_program = is_exported_program

    @_beartype.beartype
    def _run(self) -> torch.fx.GraphModule:
        # 执行 DCE（死代码消除）以移除未使用的节点
        # 如果子模块未被使用，分析子模块的节点构成子模块输出是困难的
        # 但是既然未被使用，我们可以直接移除它
        self.module.graph.eliminate_dead_code()

        # 创建一个引用模块，与原模块共享同一个图结构
        reference_module = torch.fx.GraphModule(self.module, self.module.graph)

        # 创建根模块节点
        root_module_node = _ModuleNode(
            reference_module,
            _ModuleStackMeta(
                nn_module_stack_meta=None, is_exported_program=self.is_exported_program
            ),
        )

        # 遍历原模块图中的节点
        for fx_node in self.module.graph.nodes:
            # 将每个节点作为叶子节点添加到根模块节点中
            root_module_node.add_leaf_node(
                _LeafNode(fx_node, is_exported_program=self.is_exported_program)
            )

        # 构建模块并返回
        return root_module_node.build_module({})
```