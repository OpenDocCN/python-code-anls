# `.\pytorch\torch\utils\data\graph.py`

```py
# mypy: allow-untyped-defs
# 引入需要使用的模块和函数
import io  # 提供了用于处理字节流的工具
import pickle  # 提供了对象序列化和反序列化功能
import warnings  # 用于管理警告信息
from collections.abc import Collection  # 提供了抽象基类，用于集合类的统一操作
from typing import Dict, List, Optional, Set, Tuple, Type, Union  # 提供了静态类型检查支持

# 从 torch.utils 中导入所需的模块和函数
from torch.utils._import_utils import dill_available  # 检查是否存在 dill 库
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe  # 导入数据管道相关类

# 模块导出的变量列表
__all__ = ["traverse", "traverse_dps"]

# 定义数据管道类型
DataPipe = Union[IterDataPipe, MapDataPipe]
# 数据管道图，字典类型，键为整数，值为数据管道和其子图的元组
DataPipeGraph = Dict[int, Tuple[DataPipe, "DataPipeGraph"]]  # type: ignore[misc]


# 返回一个占位字符串 "STUB" 的函数
def _stub_unpickler():
    return "STUB"


# TODO(VitalyFedyunin): Make sure it works without dill module installed
# 列出与给定数据管道相连接的所有数据管道对象
def _list_connected_datapipes(
    scan_obj: DataPipe, only_datapipe: bool, cache: Set[int]
) -> List[DataPipe]:
    # 创建一个字节流对象
    f = io.BytesIO()
    # 创建一个 Pickler 对象，用于序列化对象到字节流
    p = pickle.Pickler(f)
    
    # 如果 dill 可用，创建 dill Pickler 对象
    if dill_available():
        from dill import Pickler as dill_Pickler
        d = dill_Pickler(f)
    else:
        d = None

    # 用于存储捕获连接的数据管道对象列表
    captured_connections = []

    # 定义 getstate_hook 函数，用于获取对象状态的钩子函数
    def getstate_hook(ori_state):
        state = None
        # 如果原始状态是字典类型
        if isinstance(ori_state, dict):
            state = {}  # type: ignore[assignment]
            # 遍历字典，捕获数据管道类型的值
            for k, v in ori_state.items():
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state[k] = v  # type: ignore[attr-defined]
        # 如果原始状态是元组或列表类型
        elif isinstance(ori_state, (tuple, list)):
            state = []  # type: ignore[assignment]
            # 遍历元组或列表，捕获数据管道类型的元素
            for v in ori_state:
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state.append(v)  # type: ignore[attr-defined]
        # 如果原始状态是数据管道类型
        elif isinstance(ori_state, (IterDataPipe, MapDataPipe, Collection)):
            state = ori_state  # type: ignore[assignment]
        return state

    # 定义 reduce_hook 函数，用于序列化对象时的钩子函数
    def reduce_hook(obj):
        # 如果对象与扫描对象相同或者其 id 已存在于缓存中
        if obj == scan_obj or id(obj) in cache:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            # 添加对象的 id 到缓存，以移除在同一级别上序列化的重复数据管道
            cache.add(id(obj))
            return _stub_unpickler, ()

    # 定义数据管道类型元组
    datapipe_classes: Tuple[Type[DataPipe]] = (IterDataPipe, MapDataPipe)  # type: ignore[assignment]

    try:
        # 遍历数据管道类型元组
        for cls in datapipe_classes:
            # 设置 reduce_hook 函数到数据管道类中
            cls.set_reduce_ex_hook(reduce_hook)
            # 如果只处理数据管道，设置 getstate_hook 函数到数据管道类中
            if only_datapipe:
                cls.set_getstate_hook(getstate_hook)
        try:
            # 尝试对扫描对象进行序列化并写入字节流
            p.dump(scan_obj)
        except (pickle.PickleError, AttributeError, TypeError):
            # 如果失败，尝试使用 dill 序列化对象
            if dill_available():
                d.dump(scan_obj)
            else:
                raise
    finally:
        # 清理操作：移除 reduce_hook 函数，如果 dill 可用，则撤销对 dispatch 表的更改
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(None)
            if only_datapipe:
                cls.set_getstate_hook(None)
        if dill_available():
            from dill import extend as dill_extend

            dill_extend(False)  # Undo change to dispatch table
    
    # 返回捕获到的连接数据管道对象列表
    return captured_connections


# 遍历给定数据管道对象，返回其及其子图的字典表示
def traverse_dps(datapipe: DataPipe) -> DataPipeGraph:
    r"""
    # 创建一个空的集合用于缓存已访问的 DataPipe 实例的 id
    cache: Set[int] = set()
    # 调用 _traverse_helper 函数来遍历 DataPipe 的图结构，仅考虑 DataPipe 类型的属性
    # 并将 cache 参数传递给函数，确保在遍历过程中不重复访问同一 DataPipe 实例
    return _traverse_helper(datapipe, only_datapipe=True, cache=cache)
# 定义函数 traverse，用于遍历 DataPipe 对象及其属性，提取数据管道图
def traverse(datapipe: DataPipe, only_datapipe: Optional[bool] = None) -> DataPipeGraph:
    r"""
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    [Deprecated]
    When ``only_dataPipe`` is specified as ``True``, it would only look into the
    attribute from each DataPipe that is either a DataPipe and a Python collection object
    such as ``list``, ``tuple``, ``set`` and ``dict``.

    Note:
        This function is deprecated. Please use `traverse_dps` instead.

    Args:
        datapipe: the end DataPipe of the graph
        only_datapipe: If ``False`` (default), all attributes of each DataPipe are traversed.
          This argument is deprecating and will be removed after the next release.
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    # 构建警告信息，提示函数已弃用，建议使用 traverse_dps 替代
    msg = (
        "`traverse` function and will be removed after 1.13. "
        "Please use `traverse_dps` instead."
    )
    # 如果 only_datapipe 为 False 或 None，则追加行为变更说明
    if not only_datapipe:
        msg += " And, the behavior will be changed to the equivalent of `only_datapipe=True`."
    # 发出 FutureWarning，提醒用户该函数即将被移除
    warnings.warn(msg, FutureWarning)
    # 如果 only_datapipe 为 None，则将其设为 False
    if only_datapipe is None:
        only_datapipe = False
    # 初始化缓存集合，用于避免递归过程中的无限循环
    cache: Set[int] = set()
    # 调用 _traverse_helper 函数进行实际的遍历操作
    return _traverse_helper(datapipe, only_datapipe, cache)


# 辅助函数 _traverse_helper，用于递归遍历 DataPipe 对象及其连接的子对象
def _traverse_helper(
    datapipe: DataPipe, only_datapipe: bool, cache: Set[int]
) -> DataPipeGraph:
    # 检查 datapipe 是否为 IterDataPipe 或 MapDataPipe 类型，否则抛出异常
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise RuntimeError(
            f"Expected `IterDataPipe` or `MapDataPipe`, but {type(datapipe)} is found"
        )

    # 获取当前 datapipe 对象的唯一标识符
    dp_id = id(datapipe)
    # 如果该标识符已经存在于缓存中，则直接返回空字典
    if dp_id in cache:
        return {}
    # 将当前 datapipe 对象的标识符添加到缓存中，避免重复处理
    cache.add(dp_id)
    
    # 调用 _list_connected_datapipes 函数获取当前 datapipe 相关联的子对象列表
    items = _list_connected_datapipes(datapipe, only_datapipe, cache.copy())
    
    # 初始化字典 d，存储当前 datapipe 对象及其子图的信息
    d: DataPipeGraph = {dp_id: (datapipe, {})}
    
    # 遍历 items 列表中的每个子对象，递归调用 _traverse_helper 函数获取其子图信息并更新到 d 中
    for item in items:
        d[dp_id][1].update(_traverse_helper(item, only_datapipe, cache.copy()))
    
    # 返回构建好的字典 d，表示当前 datapipe 对象及其完整的子图结构
    return d
```