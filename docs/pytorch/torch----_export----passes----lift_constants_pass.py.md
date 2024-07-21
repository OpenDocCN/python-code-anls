# `.\pytorch\torch\_export\passes\lift_constants_pass.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
import collections
from typing import Any, Dict, List, Union

import torch
from torch._export.verifier import SpecViolationError
from torch._guards import detect_fake_mode

from torch._library.fake_class_registry import FakeScriptObject
from torch.export.exported_program import (
    ArgumentSpec,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)

# 定义 ConstantAttrMap 类，实现 MutableMapping 接口
class ConstantAttrMap(collections.abc.MutableMapping):
    """A mapping class that understands how to use module constants (tensors,
    ScriptObjects, FakeScriptObjects) as keys. We store tensors and FakeScriptObjects normally,
    but ScriptObjects are stored by hash, because different torch.ScriptObjects can point to
    the same underlying value (but we guarantee that they will `hash()` to the same value
    if that's the case).
    """

    def __init__(self):
        # Underlying dict that we use to implement this mapping.
        # 存储常量属性的字典，键可以是 int、torch.Tensor 或 FakeScriptObject
        self._constant_attrs: Dict[
            Union[int, torch.Tensor, FakeScriptObject], List[Any]
        ] = {}
        # Map from the hash(ScriptObject) to the ScriptObject itself. Used for
        # APIs like `__iter__` that should look like they're returning the
        # original ScriptObjects.
        # 将 hash(ScriptObject) 映射到 ScriptObject 本身的字典，用于保证返回原始 ScriptObject 的情况
        self._script_object_map: Dict[int, torch.ScriptObject] = {}

    def __getitem__(
        self, key: Union[torch.Tensor, torch.ScriptObject, FakeScriptObject]
    ) -> Any:
        # 根据键获取值，如果键是 ScriptObject，则使用其哈希值作为真实键
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        assert isinstance(real_key, (int, torch.Tensor, FakeScriptObject))
        return self._constant_attrs[real_key]

    def __setitem__(self, key: Union[torch.Tensor, torch.ScriptObject], value):
        # we shouldn't actually call this, should go to add() instead to handle aliasing
        # 不应直接调用此方法，应使用 add() 处理常量别名
        raise NotImplementedError(
            """Directly setting values for ConstantAttrMap is not supported, please use add(key, value) instead.
The same key can be mapped to multiple values, for handling constant aliasing."""
        )

    def add(
        self, key: Union[torch.Tensor, torch.ScriptObject, FakeScriptObject], value: Any
    ) -> None:
        # 添加键值对到映射中，处理常量别名问题
        if isinstance(key, torch.ScriptObject):
            if hash(key) not in self._constant_attrs:
                self._constant_attrs[hash(key)] = []
            self._constant_attrs[hash(key)].append(value)
            self._script_object_map[hash(key)] = key
        elif isinstance(key, (torch.Tensor, FakeScriptObject)):
            if key not in self._constant_attrs:
                self._constant_attrs[key] = []
            self._constant_attrs[key].append(value)
        else:
            # 如果键类型不符合预期，抛出类型错误
            raise TypeError(
                f"Expected key to be a tensor or ScriptObject, got {type(key)}"
            )

    def __delitem__(self, key):
        # 根据键删除映射中的条目，如果键是 ScriptObject，则使用其哈希值作为真实键
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        del self._constant_attrs[real_key]
    # 定义迭代器方法，使对象可迭代
    def __iter__(self):
        # 遍历存储在 self._constant_attrs 中的每个元素
        for key in self._constant_attrs:
            # 如果 key 是整数，则返回对应的 self._script_object_map[key]
            if isinstance(key, int):
                yield self._script_object_map[key]
            # 否则直接返回 key
            else:
                yield key

    # 定义返回对象长度的方法
    def __len__(self):
        # 返回 self._constant_attrs 的长度
        return len(self._constant_attrs)

    # 定义成员关系判断方法
    def __contains__(self, key: object) -> bool:
        # 如果 key 是 torch.ScriptObject 类型，则使用其哈希值作为真实 key
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        # 判断 real_key 是否存在于 self._constant_attrs 中
        return real_key in self._constant_attrs
def get_constant_fqn(node: torch.fx.Node, constant_name: str) -> str:
    # 获取常量张量在状态字典中的完全限定名称（FQN），应与常量张量最初使用的模块对应。
    parent_fqn = list(node.meta["nn_module_stack"].values())[-1][0]
    if len(parent_fqn) > 0:
        return f"{parent_fqn}.{constant_name}"
    else:
        return constant_name


def _get_first_fqn(
    const_attrs: ConstantAttrMap,
    key: Union[torch.Tensor, torch.ScriptObject, FakeScriptObject],
) -> Any:
    # 获取与给定键相关联的第一个完全限定名称（FQN），用于常量属性映射。
    fqns = const_attrs.get(key)
    return fqns[0] if fqns else None


def lift_constants_pass(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    constant_attrs: ConstantAttrMap,
) -> Dict[str, Union[torch.Tensor, torch.ScriptObject, FakeScriptObject]]:
    """
    将常量（张量或自定义类）作为图的输入，从而修改图模块和常量属性。返回一个
    名称到常量的字典。

    参数:
        gm (torch.fx.GraphModule): 包含要提取常量的图和常量的图模块。
        graph_signature (ExportGraphSignature): 将被修改以添加额外的CONSTANT_TENSOR和CUSTOM_OBJ输入的图签名。
        constant_attrs (ConstantAttr): 常量值到其在`gm`中完全限定路径的映射，用于保持原始模块与导出版本之间常量位置的一致性。

    返回:
        名称到常量值的字典。
    """
    all_constants: Dict[
        str, Union[torch.Tensor, torch.ScriptObject, FakeScriptObject]
    ] = {}

    inputs = graph_signature.input_specs
    num_custom_obj = sum(
        input_specs.kind == InputKind.CUSTOM_OBJ for input_specs in inputs
    )
    num_tensor_constants = sum(
        input_specs.kind == InputKind.CONSTANT_TENSOR for input_specs in inputs
    )

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder")
    )

    first_user_input_loc, first_user_input = 0, None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break
        first_user_input_loc += 1

    lifted_objs = ConstantAttrMap()
    return all_constants


def rewrite_script_object_meta(
    gm: torch.fx.GraphModule,
) -> Dict[str, Union[torch.Tensor, torch.ScriptObject, FakeScriptObject],]:
    """
    在追踪时，我们会生成一个带有FakeScriptObject的图，存储在meta["val"]中。

    目前，我们将meta["val"]重写为一个占位符CustomObjArgument。
    """
    constants: Dict[
        str,
        Union[
            torch.Tensor,
            torch.ScriptObject,
            FakeScriptObject,
        ],
    ] = {}
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 检查节点的元数据中是否存在 "val" 键
        if "val" not in node.meta:
            # 如果不存在，则跳过当前节点，继续下一个节点的处理
            continue
        
        # 获取节点中原始的 "val" 元数据
        old_meta = node.meta["val"]

        # 如果原始元数据是 torch 的 ScriptObject 类型
        if isinstance(old_meta, torch.ScriptObject):
            # 获取 ScriptObject 的类型的完全限定名
            class_fqn = old_meta._type().qualified_name()  # type: ignore[attr-defined]
            # 创建一个新的自定义对象参数，用节点的名称和类的完全限定名
            new_meta = CustomObjArgument(node.name, class_fqn)
            # 将原始对象保存到常量字典中，键为节点的名称
            constants[node.name] = old_meta
            # 更新节点的 "val" 元数据为新的自定义对象参数
            node.meta["val"] = new_meta

        # 如果原始元数据是 FakeScriptObject 类型
        elif isinstance(old_meta, FakeScriptObject):
            # 获取 FakeScriptObject 的脚本类名称
            class_fqn = old_meta.script_class_name  # type: ignore[attr-defined]
            # 创建一个新的自定义对象参数，用节点的名称、类的完全限定名和原始对象
            new_meta = CustomObjArgument(node.name, class_fqn, old_meta)
            # 将原始对象保存到常量字典中，键为节点的名称
            constants[node.name] = old_meta
            # 更新节点的 "val" 元数据为新的自定义对象参数
            node.meta["val"] = new_meta

    # 返回包含转换后常量的字典
    return constants
```