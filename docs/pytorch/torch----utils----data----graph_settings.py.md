# `.\pytorch\torch\utils\data\graph_settings.py`

```
# mypy: allow-untyped-defs
# 引入模块 inspect 和 warnings，以及类型提示 Any、List、Optional、Set
import inspect
import warnings
from typing import Any, List, Optional, Set
# 从 typing_extensions 中引入 deprecated
from typing_extensions import deprecated

# 引入 torch 库
import torch
# 从 torch.utils.data.datapipes.iter.sharding 模块中导入 _ShardingIterDataPipe 和 SHARDING_PRIORITIES
from torch.utils.data.datapipes.iter.sharding import (
    _ShardingIterDataPipe,
    SHARDING_PRIORITIES,
)
# 从 torch.utils.data.graph 中导入 DataPipe, DataPipeGraph 和 traverse_dps 函数
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps

# 定义 __all__ 列表，包含可以导出的公共接口
__all__ = [
    "apply_random_seed",
    "apply_sharding",
    "apply_shuffle_seed",
    "apply_shuffle_settings",
    "get_all_graph_pipes",
]

# 定义函数 get_all_graph_pipes，接收一个 DataPipeGraph 对象，返回 DataPipe 对象列表
def get_all_graph_pipes(graph: DataPipeGraph) -> List[DataPipe]:
    return _get_all_graph_pipes_helper(graph, set())

# 定义内部函数 _get_all_graph_pipes_helper，递归获取所有子图中的 DataPipe 对象
def _get_all_graph_pipes_helper(
    graph: DataPipeGraph, id_cache: Set[int]
) -> List[DataPipe]:
    results: List[DataPipe] = []
    for dp_id, (datapipe, sub_graph) in graph.items():
        if dp_id in id_cache:
            continue
        id_cache.add(dp_id)
        results.append(datapipe)
        results.extend(_get_all_graph_pipes_helper(sub_graph, id_cache))
    return results

# 定义内部函数 _is_sharding_datapipe，判断给定的 DataPipe 是否支持分片操作
def _is_sharding_datapipe(datapipe: DataPipe) -> bool:
    if isinstance(datapipe, _ShardingIterDataPipe):
        return True
    if hasattr(datapipe, "apply_sharding") and inspect.ismethod(
        datapipe.apply_sharding
    ):
        return True
    return False

# 定义函数 apply_sharding，对输入的 DataPipe 对象应用动态分片
def apply_sharding(
    datapipe: DataPipe,
    num_of_instances: int,
    instance_id: int,
    sharding_group=SHARDING_PRIORITIES.DEFAULT,
) -> DataPipe:
    r"""
    对具有 ``apply_sharding`` 方法的 ``sharding_filter`` DataPipe 应用动态分片。

    当同一分支中存在多个 ``sharding_filter`` 时，将引发 RuntimeError。
    """
    # 使用 traverse_dps 函数获取数据管道的图结构
    graph = traverse_dps(datapipe)

    # 定义内部辅助函数 _helper，递归遍历并应用分片操作
    def _helper(graph, prev_applied=None):
        for dp, sub_graph in graph.values():
            applied = None
            # 如果当前数据管道支持分片操作，则进行分片
            if _is_sharding_datapipe(dp):
                if prev_applied is not None:
                    raise RuntimeError(
                        "Sharding twice on a single pipeline is likely unintended and will cause data loss. "
                        f"Sharding already applied to {prev_applied} while trying to apply to {dp}"
                    )
                # 对数据管道应用分片操作，根据参数个数决定是否传递 sharding_group
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(
                        num_of_instances, instance_id, sharding_group=sharding_group
                    )
                applied = dp
            if applied is None:
                applied = prev_applied
            # 递归处理子图
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe

# 定义内部函数 _is_shuffle_datapipe，判断给定的 DataPipe 是否支持数据混洗操作
def _is_shuffle_datapipe(datapipe: DataPipe) -> bool:
    if not hasattr(datapipe, "set_shuffle") or not hasattr(datapipe, "set_seed"):
        return False
    # 检查datapipe.set_shuffle是否不是一个方法或者datapipe.set_seed是否不是一个方法，
    # 如果其中任意一个条件成立，则返回False
    if not inspect.ismethod(datapipe.set_shuffle) or not inspect.ismethod(
        datapipe.set_seed
    ):
        # 如果条件不成立（即datapipe.set_shuffle和datapipe.set_seed都是方法），返回True
        return False
    # 如果两个条件都成立，返回True
    return True
def apply_shuffle_settings(
    datapipe: DataPipe, shuffle: Optional[bool] = None
) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute.

    Apply the method to each `DataPipe` that has APIs of ``set_shuffle``
    and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
    # 如果 shuffle 参数为 None，则直接返回原始的 datapipe
    if shuffle is None:
        return datapipe

    # 获取 datapipe 对应的图结构
    graph = traverse_dps(datapipe)
    # 获取图中所有的 DataPipe 对象
    all_pipes = get_all_graph_pipes(graph)
    # 找到所有具有设置 shuffle 的 DataPipe 对象
    shufflers = [pipe for pipe in all_pipes if _is_shuffle_datapipe(pipe)]
    # 如果未找到任何具有 shuffle 能力的 DataPipe，并且 shuffle 参数为 True，则发出警告并添加一个 Shuffler
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task."
        )
        # 对 datapipe 进行 shuffle 操作，并将其添加到 shufflers 列表中
        datapipe = datapipe.shuffle()
        shufflers = [
            datapipe,
        ]  # type: ignore[list-item]

    # 针对所有找到的具有 shuffle 能力的 DataPipe，设置它们的 shuffle 属性
    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    # 返回设置后的 datapipe
    return datapipe


@deprecated(
    "`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases. "
    "Please use `apply_random_seed` instead.",
    category=FutureWarning,
)
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    # 调用 apply_random_seed 方法来代替已废弃的 apply_shuffle_seed 方法
    return apply_random_seed(datapipe, rng)


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    # 检查一个 DataPipe 是否具有设置随机种子的能力
    if hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed):
        return True
    return False


def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of ``set_seed``.

    Then set the random seed based on the provided RNG to those ``DataPipe``.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    """
    # 获取 datapipe 对应的图结构
    graph = traverse_dps(datapipe)
    # 获取图中所有的 DataPipe 对象
    all_pipes = get_all_graph_pipes(graph)
    # 使用集合来缓存 DataPipe 的 id，以防止重复设置随机性
    cache = set()
    # 存储所有具有设置随机种子能力的 DataPipe 对象
    random_datapipes = []
    for pipe in all_pipes:
        # 如果 DataPipe 已经在缓存中，跳过
        if id(pipe) in cache:
            continue
        # 如果 DataPipe 具有设置随机种子的能力，则将其添加到 random_datapipes 列表中
        if _is_random_datapipe(pipe):
            random_datapipes.append(pipe)
            cache.add(id(pipe))

    # 遍历所有具有设置随机种子能力的 DataPipe，并设置它们的随机种子
    for pipe in random_datapipes:
        random_seed = int(
            torch.empty((), dtype=torch.int64).random_(generator=rng).item()
        )
        pipe.set_seed(random_seed)

    # 返回设置后的 datapipe
    return datapipe
```