# `.\pytorch\torch\_namedtensor_internals.py`

```
# mypy: allow-untyped-defs
# 导入有序字典类型，用于维护命名张量的维度顺序
from collections import OrderedDict

"""
This file contains helper functions that implement experimental functionality
for named tensors in python. All of these are experimental, unstable, and
subject to change or deletion.
"""


# 检查命名张量是否支持序列化，如果支持则抛出运行时异常
def check_serializing_named_tensor(tensor):
    if tensor.has_names():
        raise RuntimeError(
            "NYI: Named tensors don't support serialization. Please drop "
            "names via `tensor = tensor.rename(None)` before serialization."
        )


# 构建一个维度映射字典 { dim: dim_name }，其中命名维度的值为维度名称，非命名维度的值为维度索引
def build_dim_map(tensor):
    """Returns a map of { dim: dim_name } where dim is a name if the dim is named
    and the dim index otherwise."""
    return OrderedDict(
        [(idx if name is None else name, name) for idx, name in enumerate(tensor.names)]
    )


# 解压命名形状(namedshape)，如果是有序字典则返回其条目，否则抛出运行时异常
def unzip_namedshape(namedshape):
    if isinstance(namedshape, OrderedDict):
        namedshape = namedshape.items()
    if not hasattr(namedshape, "__iter__") and not isinstance(namedshape, tuple):
        raise RuntimeError(
            f"Expected namedshape to be OrderedDict or iterable of tuples, got: {type(namedshape)}"
        )
    if len(namedshape) == 0:
        raise RuntimeError("Expected namedshape to non-empty.")
    return zip(*namedshape)


# 根据是否原地操作返回适当的重命名方法的名称
def namer_api_name(inplace):
    if inplace:
        return "rename_"
    else:
        return "rename"


# 检查项目是否为省略号
def is_ellipsis(item):
    return item == Ellipsis or item == "..."


# 查找命名列表中单个省略号的索引位置
def single_ellipsis_index(names, fn_name):
    ellipsis_indices = [i for i, name in enumerate(names) if is_ellipsis(name)]
    if len(ellipsis_indices) >= 2:
        raise RuntimeError(
            f"{fn_name}: More than one Ellipsis ('...') found in names ("
            f"{names}). This function supports up to one Ellipsis."
        )
    if len(ellipsis_indices) == 1:
        return ellipsis_indices[0]
    return None


# 根据省略号的位置扩展命名列表
def expand_single_ellipsis(numel_pre_glob, numel_post_glob, names):
    return names[numel_pre_glob : len(names) - numel_post_glob]


# 将命名列表中的省略号替换为张量名称中的相应维度名称列表
def replace_ellipsis_by_position(ellipsis_idx, names, tensor_names):
    globbed_names = expand_single_ellipsis(
        ellipsis_idx, len(names) - ellipsis_idx - 1, tensor_names
    )
    return names[:ellipsis_idx] + globbed_names + names[ellipsis_idx + 1 :]


# 解析命名列表中的省略号，将其扩展为张量名称中的对应维度名称列表
def resolve_ellipsis(names, tensor_names, fn_name):
    """
    Expands ... inside `names` to be equal to a list of names from `tensor_names`.
    """
    ellipsis_idx = single_ellipsis_index(names, fn_name)
    if ellipsis_idx is None:
        return names
    return replace_ellipsis_by_position(ellipsis_idx, names, tensor_names)


# 使用给定的名称列表更新张量的名称，可以选择是否原地操作
def update_names_with_list(tensor, names, inplace):
    # 特殊情况处理：tensor.rename(None)
    if len(names) == 1 and names[0] is None:
        return tensor._update_names(None, inplace)

    # 更新张量的名称，根据解析后的名称列表和张量当前的名称
    return tensor._update_names(
        resolve_ellipsis(names, tensor.names, namer_api_name(inplace)), inplace
    )


# 使用给定的映射更新张量的名称
def update_names_with_mapping(tensor, rename_map, inplace):
    dim_map = build_dim_map(tensor)
    # 遍历重命名映射字典中的旧维度名
    for old_dim in rename_map.keys():
        # 获取旧维度名对应的新维度名
        new_dim = rename_map[old_dim]
        # 如果旧维度名在维度映射字典中存在
        if old_dim in dim_map.keys():
            # 更新维度映射字典中的旧维度名为新维度名
            dim_map[old_dim] = new_dim
        else:
            # 抛出运行时异常，说明在尝试将旧维度名重命名为新维度名时，旧维度名在张量的维度中不存在
            raise RuntimeError(
                f"{namer_api_name(inplace)}: Tried to rename dim '{old_dim}' to dim "
                f"{new_dim} in Tensor[{tensor.names}] but dim '{old_dim}' does not exist"
            )
    # 返回更新后的张量，并更新其名称
    return tensor._update_names(tuple(dim_map.values()), inplace)
# 定义函数 update_names，用于更新张量的维度名称
def update_names(tensor, names, rename_map, inplace):
    """There are two usages:

    tensor.rename(*names) returns a view on tensor with named dims `names`.
    `names` must be of length `tensor.dim()`; otherwise, if '...' is in `names`,
    then it is expanded greedily to be equal to the corresponding names from
    `tensor.names`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename('...', 'height', 'width').names
    ('N', 'C', 'height', 'width')

    >>> # xdoctest: +SKIP
    >>> x.rename('batch', '...', 'width').names
    ('batch', 'C', 'H', 'width')

    ```

    tensor.rename(**rename_map) returns a view on tensor that has rename dims
        as specified in the mapping `rename_map`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename(W='width', H='height').names
    ('N', 'C', 'height', 'width')

    ```

    Finally, tensor.rename has an in-place version called tensor.rename_.
    """

    # 检查是否提供了维度名称列表
    has_names = len(names) > 0
    # 检查是否提供了重命名映射字典
    has_rename_pairs = bool(rename_map)

    # 如果同时提供了位置参数和关键字参数，抛出运行时错误
    if has_names and has_rename_pairs:
        raise RuntimeError(
            f"{namer_api_name(inplace)}: This function takes either positional "
            f"args or keyword args, but not both. Use tensor.{namer_api_name(inplace)}(*names) "
            f"to name dims and tensor.{namer_api_name(inplace)}(**rename_map) to rename "
            "dims."
        )

    # 处理特殊情况：当未提供维度名称时，根据列表更新维度名称
    if not has_names and not has_rename_pairs:
        return update_names_with_list(tensor, names, inplace)

    # 如果提供了维度名称列表，则根据列表更新维度名称
    if has_names:
        return update_names_with_list(tensor, names, inplace)
    # 否则，根据重命名映射字典更新维度名称
    return update_names_with_mapping(tensor, rename_map, inplace)
```