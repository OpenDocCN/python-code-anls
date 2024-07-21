# `.\pytorch\torch\_functorch\_aot_autograd\utils.py`

```
"""
Contains various utils for AOTAutograd, including those for handling collections.
"""

# 导入必要的模块和库
import dataclasses
import operator
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Union

# 导入 PyTorch 相关模块
import torch
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import py_sym_types

# 定义已知类型列表
KNOWN_TYPES = [
    torch.Tensor,
    BackwardState,
    int,
    str,
    float,
    bool,
    type(None),
    *py_sym_types,
    FakeScriptObject,
    torch.ScriptObject,
]

# 保存原始的 zip 函数
original_zip = zip


# 定义严格模式的 zip 函数
def strict_zip(*iterables, strict=True, **kwargs):
    if not strict:
        return original_zip(*iterables, **kwargs)

    # 检查所有迭代器的长度是否一致
    shortest_length = min(len(it) for it in iterables)
    for iterable in iterables:
        if len(iterable) != shortest_length:
            raise ValueError(
                "The iterables have different lengths and strict mode is enabled."
            )

    # 使用原始的 zip 函数进行打包
    return original_zip(*iterables, **kwargs)


# 获取表达式列表/元组中 int/SymInt 的提示信息
def _get_symint_hints(exprs):
    """
    Get the hints of a list/tuple of int/SymInt.
    """
    if isinstance(exprs, (list, tuple)):
        return type(exprs)(_get_symint_hints(e) for e in exprs)
    elif isinstance(exprs, torch.SymInt):
        return exprs.node.shape_env.size_hint(exprs.node.expr)
    else:
        return exprs


# 将数据类对象部分展开为字典
def partial_flatten_asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)
        }
    elif isinstance(obj, (list, tuple)):
        return obj.__class__([partial_flatten_asdict(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: partial_flatten_asdict(v) for k, v in obj.items()}
    else:
        return obj


# 将输入参数规范化为列表
def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


# 获取自动转换（autocast）状态
def _get_autocast_states():
    return [
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cuda"),
        torch.get_autocast_dtype("cpu"),
        torch.is_autocast_cache_enabled(),
    ]


# 创建包装后的函数
def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True  # type: ignore[attr-defined]
    return g


# 创建包装后的编译器
def make_boxed_compiler(compiler):
    @wraps(compiler)
    def f(fx_g, inps):
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


# 在运行时使用参数调用函数
def call_func_at_runtime_with_args(
    f, args: Union[Tuple[Any], List[Any]], steal_args=False, disable_amp=False
):
    if not steal_args:
        args = list(args)
    assert isinstance(args, list)

    # 根据是否禁用 amp（自动混合精度）选择上下文
    context = torch._C._DisableAutocast if disable_amp else nullcontext
    # 使用 context() 上下文管理器，确保执行过程中的一致性和资源管理
    with context():
        # 检查函数对象 f 是否具有 "_boxed_call" 属性
        if hasattr(f, "_boxed_call"):
            # 如果有 "_boxed_call" 属性，调用 f(args)，并将结果标准化为列表形式
            out = normalize_as_list(f(args))
        else:
            # 如果没有 "_boxed_call" 属性，发出警告信息，提示用户修改代码
            warnings.warn(
                "Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. "
                "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
                "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale."
            )
            # 调用 f(*args)，并将结果标准化为列表形式
            out = normalize_as_list(f(*args))
    # 返回处理结果 out
    return out
# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec: Optional[pytree.TreeSpec] = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple: Optional[bool] = None  # 如果输出规范是元组/列表，则不会解开它们。
    is_really_simple: Optional[bool] = None  # 如果输出规范是 LeafSpec

    def set(self, spec: pytree.TreeSpec) -> None:
        assert self.spec is None or self.spec == spec
        assert spec is not None
        self.spec: pytree.TreeSpec = spec
        if self.spec.type in {tuple, list} and all(
            child.is_leaf() for child in spec.children_specs
        ):
            self.is_simple = True
        if self.spec.is_leaf():
            self.is_really_simple = True

    def unflatten(self, x: List[Any]) -> Any:
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        assert self.spec is not None
        return pytree.tree_unflatten(x, self.spec)


# Creates a function that returns flattened inputs and outputs
# Also returns the output tree spec, which is needed to recover the "unflattened"
# output tree structure later.
def create_tree_flattened_fn(fn, args, kwargs=None) -> Tuple[Callable, PytreeThunk]:
    if kwargs is None:
        kwargs = {}
    # Save the args_spec for flat_tensor_args to unflatten while tracing
    _, tensor_args_spec = pytree.tree_flatten((args, kwargs))
    out_spec = PytreeThunk()

    def flat_fn(*flat_args):
        # The input are flattened tensor args. Prepare the args in the
        # order that original function expects. Add static args as well.
        # They will appear as tensor constants in the traced graph.
        nonlocal out_spec
        args, kwargs = pytree.tree_unflatten(flat_args, tensor_args_spec)
        tree_out = fn(*args, **kwargs)
        flat_out, spec = pytree.tree_flatten(tree_out)
        for i in flat_out:
            is_known_type = False
            for j in KNOWN_TYPES:
                if isinstance(i, j):
                    is_known_type = True
                    break
            if not is_known_type:
                raise RuntimeError(
                    f"Found {type(i)} in output, which is not a known type. "
                    "If this type holds tensors, you need to register a pytree for it. "
                    "See https://github.com/pytorch/functorch/issues/475 for a brief "
                    "explanation why. If you don't need to register a pytree, please "
                    "leave a comment explaining your use case and we'll make this more "
                    "ergonomic to deal with"
                )
        out_spec.set(spec)
        return flat_out

    # Can't use functools.wraps here because the wrapper has different
    # calling convention
    if hasattr(fn, "_orig_mod"):
        flat_fn._orig_mod = fn._orig_mod  # type: ignore[attr-defined]

    return flat_fn, out_spec
# This function adjusts a given tensor `t` based on metadata `meta` and an index `idx`.
# If `t` is not a torch.Tensor, it returns `t` unchanged.
def maybe_to_fresh_input(idx, t, meta):
    # Check if `t` is not a torch.Tensor; if so, return `t` unchanged
    if not isinstance(t, torch.Tensor):
        return t
    # Check if `idx` is in the list of indices where inputs have been mutated at runtime
    if idx in meta.mutated_inp_runtime_indices:
        # Find the index of `idx` in the list of mutated input indices
        mutated_inp_idx = meta.mutated_inp_runtime_indices.index(idx)
        # Check if the input tensor requires gradient and mutates data
        if meta.input_info[idx].requires_grad and meta.input_info[idx].mutates_data:
            # Return a cloned copy of `t` to ensure the original tensor before mutation is used in autograd.grad()
            return t.clone()
        # Check if the input tensor mutates metadata
        if meta.input_info[idx] and meta.input_info[idx].mutates_metadata:
            # Return a view of `t` to ensure the original tensor before metadata mutation is used in autograd.grad()
            return t.view(t.shape)
    # Return `t` unchanged if it does not need adjustments based on the metadata
    return t


# This function modifies the inputs/outputs of a forward module `fw_module` and its metadata `fw_metadata`
# by removing tokens and replacing them with specific functions (_make_token and _sink_tokens).
# It addresses the management of tokens in the context of autograd in AOTAutograd.
def unlift_tokens(fw_module, fw_metadata):
    # Determine the number of tokens present in `fw_metadata`
    num_tokens = len(fw_metadata.tokens)
    
    # Initialize an empty list to store input token nodes
    input_token_nodes = []
    # 遍历fw_module.graph中的节点，同时获取索引i和节点node
    for i, node in enumerate(fw_module.graph.nodes):
        # 如果索引i小于num_tokens，则断言节点的操作为"placeholder"
        if i < num_tokens:
            assert node.op == "placeholder"
            # 将满足条件的节点添加到input_token_nodes列表中
            input_token_nodes.append(node)

        # 如果节点的操作为"call_function"且目标函数的名称为"with_effects"
        elif node.op == "call_function" and node.target.__name__ == "with_effects":
            # 如果node.args[0]在input_token_nodes中
            if node.args[0] in input_token_nodes:
                # 在节点node之前插入新节点
                with fw_module.graph.inserting_before(node):
                    # 创建一个新的调用函数节点，调用torch.ops.prims._make_token.default函数
                    new_token_node = fw_module.graph.call_function(
                        torch.ops.prims._make_token.default, ()
                    )
                    # 设置新节点的元数据
                    new_token_node.meta["val"] = torch.tensor([])
                    new_token_node.meta["tensor_meta"] = torch.tensor([])

                    # 更新node.args中的参数列表
                    args = list(node.args)
                    args[0] = new_token_node
                    node.args = tuple(args)

        # 如果节点的操作为"output"
        elif node.op == "output":
            # 获取输出节点中的前num_tokens个token节点和其余的参数
            output_token_nodes = node.args[0][:num_tokens]
            other_output_args = node.args[0][num_tokens:]

            # 对每个输出token节点进行断言，确保其操作为"call_function"，目标函数为operator.getitem，第二个参数为0
            for output_token_node in output_token_nodes:
                assert (
                    output_token_node.op == "call_function"
                    and output_token_node.target == operator.getitem
                    and output_token_node.args[1] == 0
                )
            # 在节点node之前插入新节点
            with fw_module.graph.inserting_before(node):
                # 创建一个新的调用函数节点，调用torch.ops.prims._sink_tokens.default函数
                sink_token_node = fw_module.graph.call_function(
                    torch.ops.prims._sink_tokens.default,
                    (output_token_nodes,),
                )
                # 更新node.args，保留其余的参数
                node.args = (other_output_args,)

    # 遍历input_token_nodes中的每个节点，从fw_module.graph中删除这些节点
    for input_token_node in input_token_nodes:
        fw_module.graph.erase_node(input_token_node)

    # 重新编译fw_module，以应用节点的修改
    fw_module.recompile()

    # 更新fw_metadata中与tokens相关的元数据信息，减去num_tokens数量
    # 以清除tokens
    fw_metadata.num_forward_returns -= num_tokens
    fw_metadata.num_forward -= num_tokens
    fw_metadata.tokens = {}
def root_module_when_exporting_non_strict(flat_fn):
    # 当以非严格模式导出时，我们以特定的模式包装根模块。
    # 参见 torch.export._trace.py 中的 `_aot_export_non_strict`。
    # 这里检查是否存在该包装模式。

    # 检查 flat_fn 是否具有 "_orig_mod" 属性，并且其 "_orig_mod" 属性具有 "_export_root" 属性
    if hasattr(flat_fn, "_orig_mod") and hasattr(flat_fn._orig_mod, "_export_root"):
        # 如果满足条件，则返回 flat_fn._orig_mod._export_root
        return flat_fn._orig_mod._export_root
    else:
        # 如果条件不满足，则返回 None
        return None
```