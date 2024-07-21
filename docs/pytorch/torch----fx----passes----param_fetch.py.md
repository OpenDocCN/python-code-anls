# `.\pytorch\torch\fx\passes\param_fetch.py`

```py
from torch.fx.graph_module import GraphModule
from typing import Any, Callable, Dict, List, Tuple, Type
import torch
import torch.nn as nn

from torch.fx._compatibility import compatibility

__all__ = ['default_matching', 'extract_attrs_for_lowering', 'lift_lowering_attrs_to_nodes']

# Matching method matches the attribute name of current version to the attribute name of `target_version`
@compatibility(is_backward_compatible=False)
def default_matching(name: str, target_version: int) -> str:
    """Default matching method
    """
    return name

# This dict maps the nn.Module class name to the attribute name list that we want to fetch for lowering.
# The first integer in the tuple is the version number of the nn.Module class when we create the parameter list.
# If there's a version mismatch then it means the parameter names in the book might be mismatched with nn.Module.
module_fetch_book: Dict[Type, Tuple[int, List[str], Callable[[str, int], str]]] = {
    torch.nn.modules.linear.Linear: (1, ["weight", "bias"], default_matching),
    torch.nn.modules.conv.Conv2d: (
        1, ["weight", "bias", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"], default_matching
    ),
    torch.nn.modules.batchnorm.BatchNorm2d: (2, ["weight", "bias", "running_mean", "running_var", "eps"], default_matching),
    torch.nn.modules.pooling.AdaptiveAvgPool2d: (1, [], default_matching),
    torch.nn.modules.pooling.MaxPool2d: (
        1, ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"], default_matching
    ),
    torch.nn.modules.activation.ReLU: (1, ["inplace"], default_matching),
}

@compatibility(is_backward_compatible=False)
def extract_attrs_for_lowering(mod: nn.Module) -> Dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
    attrs_for_lowering: Dict[str, Any] = {}
    attrs_for_lowering["name"] = torch.typename(mod)

    if type(mod) in module_fetch_book:
        version, param_to_fetch, matching_method = module_fetch_book[type(mod)]
        if version < mod._version:
            raise RuntimeError(f"Fetcher version {version} try to fetch {torch.typename(mod)} version {mod._version}, "
                               "please upgrade the module_fetch_book, open an issue and @842974287 "
                               "or report a bug to AIACC team directly.")
        for attr in param_to_fetch:
            # Fetch the attribute from `mod` using the matching method based on the module's version
            attrs_for_lowering[attr] = getattr(mod, matching_method(attr, mod._version))
    else:
        # Raise error if `mod` class type is not found in `module_fetch_book`
        raise RuntimeError(f"{torch.typename(mod)} is not in the module_fetch_book yet, "
                           "please add it to the module_fetch_book, open an issue and @842974287 "
                           "or report a bug to AIACC team directly.")
    return attrs_for_lowering

@compatibility(is_backward_compatible=False)
# 递归地遍历所有 `fx_module` 节点，并且如果节点是叶子模块，则获取模块的属性。
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module.
    """
    # 将 `fx_module` 的所有子模块作为字典存储起来
    submodules = dict(fx_module.named_modules())

    # 遍历 `fx_module` 的图中的每个节点
    for node in fx_module.graph.nodes:
        # 检查节点操作是否为 "call_module"
        if node.op == "call_module":
            # 如果该节点的目标模块是 GraphModule 类型，则递归调用 lift_lowering_attrs_to_nodes 函数
            if isinstance(submodules[node.target], GraphModule):
                lift_lowering_attrs_to_nodes(submodules[node.target])
            else:
                # 如果目标模块不是 GraphModule 类型，则调用 extract_attrs_for_lowering 函数获取其属性
                node.attrs_for_lowering = extract_attrs_for_lowering(submodules[node.target])
```