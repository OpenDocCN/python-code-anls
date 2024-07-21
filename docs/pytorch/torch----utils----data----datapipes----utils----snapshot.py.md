# `.\pytorch\torch\utils\data\datapipes\utils\snapshot.py`

```
# mypy: allow-untyped-defs
# 导入需要的模块和类
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.graph_settings import apply_random_seed


# TODO: Caveats
#   1. Caller (either the ReadingService or DataLoader) must pass in the initial RNG
#   2. `in_batch_shuffle` and `bucketbatch` are not compatible with this because they currently
#      lack the option to `set_seed`.
# 定义一个函数 `_simple_graph_snapshot_restoration`，接收一个 `IterDataPipe` 对象和整数 `n_iterations`，可选参数 `rng`
def _simple_graph_snapshot_restoration(
    datapipe: IterDataPipe, n_iterations: int, rng=None
) -> None:
    r"""
    Fast-forward the given DataPipe and its parents by ``n_iterations``, re-doing computations to restore a snapshot.

    For instance, applying this function to the final DataPipe of a graph will restore the snapshot
    (via fast-forward) every DataPipe within the graph.

    After you deserialize a DataPipe, you can use its `_number_of_samples_yielded` attribute as the input
    to this function to forward the DataPipe.

    A DataPipe cannot be restored twice in a row unless there is an iteration started between the restoration
    attempts.

    Note:
        This is the simplest but least efficient way to fast-forward a DataPipe. Usage of other fast-forwarding
        methods (custom ones if necessary) are recommended.

    Args:
        datapipe: IterDataPipe to be fast-forwarded
        n_iterations: number of iterations to fast-forward
        rng: ``Optional[torch.Generator]``. If not ``None``, this RNG will be used for shuffling. The generator
            should be in its `initial` state as it was first passed into ``DataLoader`` or ``ReadingService``.
    """
    # 如果 `datapipe` 的 `_snapshot_state` 已经是 `Restored` 状态，则抛出异常
    if datapipe._snapshot_state == _SnapshotState.Restored:
        raise RuntimeError(
            "Snapshot restoration cannot be applied. You can only restore simple snapshot to the graph "
            "if your graph has not been restored."
        )

    # 确保 `datapipe` 处于初始状态，以便进行快速前进
    datapipe.reset()  # This ensures `SnapshotState` is `Iterating` by this point, even if it was `Restored`.
    # 应用随机种子 `rng` 到 `datapipe`
    apply_random_seed(datapipe, rng)

    # 计算剩余迭代次数
    remainder = n_iterations
    # 获取 `datapipe` 的迭代器
    it = iter(datapipe)  # This always reset the DataPipe if it hasn't already.
    # 循环进行迭代，直到达到指定的迭代次数 `n_iterations`
    while remainder > 0:
        try:
            next(it)
            remainder -= 1
        except StopIteration as e:
            raise RuntimeError(
                f"Fast-forward {datapipe} by {n_iterations} iterations "
                "exceeds the number of samples available."
            ) from e
    # 将 `_fast_forward_iterator` 设置为当前迭代器 `it`
    datapipe._fast_forward_iterator = it
    # 当 `datapipe` 具有 `_fast_forward_iterator` 时，`next()` 将从该迭代器获取结果而不是其他地方。

    # 防止在 `iter()` 调用中 `datapipe` 重置
    # 设置数据管道的快照状态为已恢复，以便如果另一个数据管道正在使用它，它不需要重新开始
    datapipe._snapshot_state = _SnapshotState.Restored
```