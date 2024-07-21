# `.\pytorch\torch\distributed\_tools\mem_tracker.py`

```py
import math  # 导入数学模块
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from copy import deepcopy  # 导入深拷贝函数
from enum import auto, Enum  # 导入枚举相关函数
from functools import partial, wraps  # 导入函数工具
from typing import (  # 导入类型提示相关
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

from typing_extensions import Self  # 导入类型提示扩展

import torch  # 导入PyTorch模块
from torch import nn, optim  # 导入神经网络与优化器模块
from torch.distributed._tools.mod_tracker import ModTracker  # 导入模块跟踪工具
from torch.optim.optimizer import (  # 导入优化器相关函数
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.utils._python_dispatch import (  # 导入Python调度工具
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten, tree_map_only  # 导入树结构相关函数

from torch.utils.weak import WeakIdKeyDictionary, weakref  # 导入弱引用相关模块

if TYPE_CHECKING:  # 如果是类型检查模式
    from torch.utils.hooks import RemovableHandle  # 导入可移除处理函数

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (  # PyTorch最小分配内存值，依据环境变量设置
    2 ** 9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)
_TOTAL_KEY = "Total"  # 总关键字

__all__ = ["MemTracker"]  # 导出模块的公开接口列表


class _RefType(str, Enum):
    """Base Class for defining memory reference types, categorizing tensors based on their usage within a model."""
    # 定义内存引用类型基类，用于根据模型内部使用情况对张量进行分类
    pass


class _State(str, Enum):
    """Base Class for defining module state to capture snapshots ."""
    # 定义模块状态基类，用于捕获快照
    pass


class _MemRefType(_RefType):
    """
    An enum to define memory reference types, categorizing tensors based on their usage within a model.

        - PARAM: Tensors registered as nn.Parameter within modules.
        - BUFFER: Tensors registered as nn.Buffer within modules.
        - GRAD: Gradients associated with parameters.
        - ACT: Tensors produced during the forward pass and recomputation in activation checkpointing.
        - TMP: Temporary memory used during the backward pass, including gradients of activations.
        - OPT: Tensors holding optimizer states.
        - OTH: Tensors registered via `track_external` that do not fit the above categories.
    """
    # 用于定义内存引用类型的枚举，根据模型内使用情况对张量进行分类

    PARAM = "Parameter"  # 参数类型：注册为nn.Parameter的张量
    BUFFER = "Buffer"  # 缓冲类型：注册为nn.Buffer的张量
    GRAD = "Gradient"  # 梯度类型：与参数相关的梯度
    ACT = "Activation"  # 激活类型：在前向传播和激活检查点中产生的张量
    TEMP = "Temp"  # 临时类型：在反向传播中使用的临时内存，包括激活的梯度
    OPT = "Optstate"  # 优化状态类型：保存优化器状态的张量
    OTH = "Other"  # 其他类型：通过track_external注册的不属于上述类别的张量


class _ModState(_State):
    """
    An enum to define the state of a module.

        - PRE_FW: The module is about to run the forward pass.
        - POST_FW: The module has finished running the forward pass.
        - PEAK_FW: The module has reached the peak memory usage during the forward pass.
        - PRE_BW: The module is about to run the backward pass.
        - PRE_FW_AC: The module is about to run the forward pass with activation checkpointing.
        - POST_FW_AC: The module has finished running the forward pass with activation checkpointing.
        - POST_BW: The module has finished running the backward pass.
        - PEAK_BW: The module has reached the peak memory usage during the backward pass.
    """
    # 用于定义模块状态的枚举

    PRE_FW = "Pre-Forward"  # 准备前向传播状态
    POST_FW = "Post-Forward"  # 完成前向传播状态
    PEAK_FW = "Peak-Forward"  # 达到前向传播的内存使用峰值状态
    PRE_BW = "Pre-Backward"  # 准备反向传播状态
    PRE_FW_AC = "Pre-Forward-AC"  # 准备使用激活检查点的前向传播状态
    POST_FW_AC = "Post-Forward-AC"  # 完成使用激活检查点的前向传播状态
    POST_BW = "Post-Backward"  # 完成反向传播状态
    PEAK_BW = "Peak-Backward"  # 达到反向传播的内存使用峰值状态
    # 定义常量 POST_FW，表示后向传播
    POST_FW = "Post-Forward"
    # 定义常量 PEAK_FW，表示峰值前向传播
    PEAK_FW = "Peak-Forward"
    # 定义常量 PRE_BW，表示前向传播
    PRE_BW = "Pre-Backward"
    # 定义常量 PRE_FW_AC，表示预前向AC传播
    PRE_FW_AC = "Pre-Forward-AC"
    # 定义常量 POST_FW_AC，表示后前向AC传播
    POST_FW_AC = "Post-Forward-AC"
    # 定义常量 POST_BW，表示后向传播
    POST_BW = "Post-Backward"
    # 定义常量 PEAK_BW，表示峰值后向传播
    PEAK_BW = "Peak-Backward"
class _ModMemStats:
    """
    A class to store the memory statistics of a module.

    Args:
        mod_fqn (str): The fully qualified name of the module.
    Attributes:
        mod_fqn (str): The fully qualified name of the module.
        parameter_mem (int): The memory usage of the parameters of the module.
        buffer_mem (int): The memory usage of the buffers of the module.
        input_mem (int): The memory usage of the inputs to the module.
        output_mem (int): The memory usage of the outputs from the module.
        snapshots (Dict[_ModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states defined by ``_ModState``.
    Note:
        The memory snapshot is stored as a dictionary - Dict[torch.device, Dict[str, int]], where each key is a device,
        and each value is another dictionary with keys as memory reference types defined by `_MemRefType` and
        values as the memory consumed in bytes.
    """

    def __init__(self, mod_fqn: str):
        # Initialize _ModMemStats object with the fully qualified module name
        self.mod_fqn = mod_fqn
        # Declare attributes for memory usage of parameters, buffers, inputs, and outputs
        self.parameter_mem: int
        self.buffer_mem: int
        self.input_mem: int
        self.output_mem: int
        # Initialize local_peak as an empty dictionary to track peak memory usage per device
        self.local_peak: Dict[torch.device, int] = {}
        # Initialize snapshots as an empty dictionary to store memory snapshots at different module states
        self.snapshots: Dict[_ModState, List[Dict[torch.device, Dict[str, int]]]] = {}


class _WeakRefInfo:
    """
    Manages memory statistics and device attributes for tensor storages.
    """

    def __init__(
        self, size: int, element_size: int, device: torch.device, reftype: _RefType
    ) -> None:
        """
        Initializes the ``_WeakRefInfo`` object with tensor storage properties.

        Args:
            size (int): The number of elements in the tensor storage.
            element_size (int): The size of each element in the tensor storage.
            device (torch.device): The device on which the tensor is allocated.
            reftype (_RefType): The reference type of the tensor.
        """
        # Initialize _WeakRefInfo object with size, element_size, device, and reference type
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.device = device
        # Calculate and store memory consumed by tensor storage using private method _calculate_mem_consumed()
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        """
        Calculates the memory consumed by the tensor storage, considering device-specific allocation rules.

        Returns:
            int: The memory consumed in bytes.
        """
        # Calculate memory consumption based on size and element size of tensor storage
        mem = self.size * self.element_size
        # Adjust memory consumption if tensor is allocated on CUDA device
        if self.device.type == "cuda":
            return math.ceil((mem) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
        return mem
    # 更新并返回如果存储大小发生变化则更新的内存消耗量

    def update_mem_consumed(self, st: torch.UntypedStorage) -> int:
        """
        更新并返回内存消耗量，如果存储大小发生变化的话。

        Args:
            st (torch.UntypedStorage): 要检查大小更新的张量存储。

        Returns:
            int: 更新后的内存消耗量，单位为字节。
        """
        if st.size() != self.size:
            self.size = st.size()  # 更新对象的存储大小
            self.mem_consumed = self._calculate_mem_consumed()  # 调用计算内存消耗量的方法
        return self.mem_consumed  # 返回更新后的内存消耗量

    @staticmethod
    def get_untyped_storages(t: torch.Tensor) -> Set[torch.UntypedStorage]:
        """
        从张量或其子类递归提取未类型化的存储。

        Args:
            t (torch.Tensor): 要从中提取存储的张量。

        Returns:
            Set[torch.UntypedStorage]: 一组未类型化的存储。
        """
        unflattened_tensors = [t]  # 初始化待展开的张量列表
        flattened_tensor_storages = set()  # 初始化存储结果的集合
        while len(unflattened_tensors) > 0:
            obj = unflattened_tensors.pop()  # 弹出列表中的对象
            if is_traceable_wrapper_subclass(obj):
                attrs, _ = obj.__tensor_flatten__()  # 展开可追踪包装子类的张量
                unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])  # 将展开的属性添加到待展开的列表
            else:
                if not hasattr(obj, "untyped_storage"):
                    warnings.warn(
                        f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                        category=UserWarning,
                        stacklevel=2,
                    )
                else:
                    flattened_tensor_storages.add(obj.untyped_storage())  # 将未类型化存储添加到结果集合中
        return flattened_tensor_storages  # 返回所有未类型化的存储的集合

    @classmethod
    def create_winfo(
        cls,
        st: torch.UntypedStorage,
        device: torch.device,
        reftype: _RefType,
        callback: Optional[Callable[[Self, weakref.ref], Any]] = None,
    ) -> Tuple[Self, weakref.ref]:
        """
        创建一个新的 `_WeakRefInfo` 实例和一个指向 `torch.UntypedStorage` 对象的弱引用，
        可选择将回调函数附加到弱引用上。

        Args:
            st (torch.UntypedStorage): 要创建弱引用信息的存储对象。
            device (torch.device): 与存储对象关联的设备。
            reftype (_RefType): 引用类型，用于对存储进行分类。
            callback (Optional[Callable[[Self, weakref.ref]]]): 当存储对象即将被终结（垃圾回收）时调用的回调函数。
                回调函数应接受两个参数：`_WeakRefInfo` 实例和存储的弱引用。
        Returns:
            Tuple[Self, weakref.ref]: 包含新创建的 `_WeakRefInfo` 实例和存储对象的弱引用的元组。
            如果提供了回调函数，则弱引用可能附带回调函数。
        """
        
        # 创建一个 `_WeakRefInfo` 实例，传入存储对象的大小、元素大小、设备和引用类型
        winfo = cls(st.size(), st.element_size(), device, reftype)
        
        # 创建存储对象 `st` 的弱引用 `w_st`，如果提供了回调函数 `callback`，则使用偏函数 `partial(callback, winfo)`
        # 来将 `_WeakRefInfo` 实例 `winfo` 作为参数传递给回调函数
        w_st = weakref.ref(st, partial(callback, winfo) if callback else None)
        
        # 返回包含 `_WeakRefInfo` 实例和存储对象弱引用的元组
        return winfo, w_st
# 定义一个函数，根据给定单位返回相应的内存大小划分因子
def _get_mem_divisor(units: str) -> int:
    # 创建单位到大小划分因子的映射字典
    unit_dict = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
    # 如果给定单位在映射字典中，则返回对应的大小划分因子
    if units in unit_dict:
        return unit_dict[units]
    else:
        # 如果单位不在映射字典中，则抛出错误，列出支持的单位
        raise ValueError(
            f"Unsupported unit: {units}. Supported units are: {', '.join(unit_dict.keys())}"
        )


# 定义一个函数，根据划分因子和精度对数值进行舍入，返回浮点数或整数
def _rounding_fn(value: int, divisor: int, precision: int) -> Union[float, int]:
    return value if divisor == 1 else round(value / divisor, precision)


# 定义一个函数，打印给定设备的内存快照信息，以特定单位显示
def _print_snapshot(snapshot: Dict[torch.device, Dict[str, int]], units: str) -> None:
    # 如果快照为空，则打印无内存跟踪信息并返回
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    # 获取内存大小划分因子
    divisor = _get_mem_divisor(units)
    # 遍历每个设备的快照信息
    for dev, dev_snap in snapshot.items():
        # 如果总内存小于等于0，则跳过当前设备
        if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
            continue
        # 打印设备名称及其各项内存数据
        print(
            f"Device: {dev}",
            *(
                f"\t{k}: {_rounding_fn(v, divisor, 2)} {units}"
                for k, v in dev_snap.items()
            ),
            sep="\n",
        )


# 定义一个函数，以表格形式打印给定设备的内存快照信息，以特定单位显示
def _print_snapshot_tabular(
    snapshot: Dict[torch.device, Dict[str, int]], units: str
) -> None:
    # 如果快照为空，则打印无内存跟踪信息并返回
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    try:
        # 尝试导入 tabulate 库，用于生成表格
        from tabulate import tabulate
    except ImportError as err:
        # 如果导入失败，提示用户安装 tabulate 库
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err
    # 获取内存大小划分因子
    divisor = _get_mem_divisor(units)
    table_data = []
    # 获取快照中第一个设备的键列表作为表头
    key_list = list(next(iter(snapshot.values())).keys())
    headers = ["Device"] + [f"{key}" for key in key_list]

    # 遍历每个设备的快照信息，构建表格数据
    for dev, dev_snap in snapshot.items():
        # 如果总内存小于等于0，则跳过当前设备
        if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
            continue
        # 构建当前设备的一行数据，并添加到表格数据中
        row = [str(dev)]
        row.extend(f"{_rounding_fn(v, divisor, 2)} {units}" for v in dev_snap.values())
        table_data.append(row)
    # 使用 tabulate 库生成并打印表格
    print(tabulate(table_data, headers=headers, tablefmt="rst"))


# 定义一个函数，打印状态快照列表中每个状态及其相关的设备内存快照信息
def _print_state_snapshots(
    snapshots: Dict[_State, List[Dict[torch.device, Dict[str, int]]]], units: str
) -> None:
    # 遍历状态快照字典中的每个状态及其对应的快照列表
    for state, snapshot_list in snapshots.items():
        # 打印当前状态信息
        print(f"{state}")
        # 遍历当前状态的每个快照，依次打印编号和设备内存快照信息
        for i, snapshot in enumerate(snapshot_list):
            print(f"# {i + 1}:")
            _print_snapshot(snapshot, units)
    # 打印空行
    print()


# 定义一个函数，以表格形式打印状态快照列表中每个状态及其相关的设备内存快照信息
def _print_state_snapshots_tabular(
    snapshots: Dict[_State, List[Dict[torch.device, Dict[str, int]]]], units: str
) -> None:
    try:
        # 尝试导入 tabulate 库，用于生成表格
        from tabulate import tabulate
    except ImportError as err:
        # 如果导入失败，提示用户安装 tabulate 库
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err

    table_data = []
    last_state_call = None
    # 获取内存大小划分因子
    divisor = _get_mem_divisor(units)
    # 遍历字典 `snapshots`，获取每个州（state）及其对应的快照列表（snapshot_list）
    for state, snapshot_list in snapshots.items():
        # 遍历每个州的快照列表中的快照数据
        for i, snapshot in enumerate(snapshot_list):
            # 构建用于显示的状态调用字符串，格式为 "{state} # {i + 1}"
            state_call = f"{state} # {i + 1}"
            # 遍历每个快照中的设备及其快照数据
            for dev, dev_snap in snapshot.items():
                # 如果设备快照中的总值小于等于 0，跳过当前循环
                if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
                    continue
                # 准备要添加到表格数据中的行，包含状态调用和设备信息
                row = {
                    "State & Call": (
                        state_call if state_call != last_state_call else ""
                    ),
                    "Device": str(dev),
                }
                # 更新上一个状态调用的值为当前状态调用
                last_state_call = state_call
                # 将设备快照中的每个键值对添加到行中，保留两位小数并添加单位
                for k, v in dev_snap.items():
                    row[f"{k}"] = f"{_rounding_fn(v, divisor, 2)} {units}"
                # 将当前行添加到表格数据列表中
                table_data.append(row)
    # 使用rst格式打印整理好的表格数据
    print(tabulate(table_data, headers="keys", tablefmt="rst"))
class _UpdateType(Enum):
    # 定义一个枚举类型，用于跟踪持续维护的内存快照的更新类型。

    # ADD - 当跟踪新的张量存储时使用
    # DEL - 当张量存储即将被销毁（垃圾回收时）使用
    # REF - 当张量引用被更新时使用，例如，梯度被标记为通用的后向引用类型，直到梯度钩子将它们分类为梯度。
    # SIZE - 当张量的存储大小被调整时使用
    ADD = auto()
    DEL = auto()
    REF = auto()
    SIZE = auto()


class MemTracker(TorchDispatchMode):
    """
    一个 TorchDispatchMode 类，用于在其上下文中跟踪、分类和归因于张量内存的创建或访问。

    根据其上下文中定义的 _MemRefType，将跟踪的张量分类为参数、缓冲区、激活、梯度、临时内存和优化器状态。它在其上下文中调用的模块的各种状态（由 _ModState 定义）下捕获内存快照。

    Attributes:
        memory_tracking: 一个弱引用键字典，用于存储每个模块的内存统计。每个键是模块的引用，每个值是一个 _ModMemStats 对象，用于存储模块的内存统计信息。

    Note:
        MemTracker 应该作为上下文管理器使用。默认情况下，MemTracker 上下文中创建或访问的模块、优化器和任何其他张量将被跟踪。任何在 MemTracker 外部创建的张量或有状态对象（如模块、优化器等），如果需要跟踪，应该使用 `track_external` 方法进行注册。
        在使用 MemTracker 前应调用 `track_external` 方法。未在 MemTracker 中创建且未通过 `track_external` 方法提供的张量将不会被 MemTracker 跟踪。

    Example usage:

        .. code-block:: python

            module = ...
            optimizer = ...
            inp = ...
            mem_tracker = MemTracker()
            mem_tracker.track_external(module, optimizer, inp)
            with mem_tracker as mt:
                loss = module(inp)
                print("After Forward:")
                mt.display_snapshot("current")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mt.display_snapshot("peak")
            mt.display_modulewise_snapshots(depth = 3, units = "MiB")
    # 这部分代码定义了一个名为 ``MemTracker`` 的类，用于跟踪和管理内存使用情况。
    
    def __init__(self) -> None:
        # 初始化一个弱引用字典，用于跟踪内存
        self.memory_tracking = WeakIdKeyDictionary()
        # 当前内存快照字典，按设备和键值对应存储当前内存使用量
        self._curr_mem_snap: Dict[torch.device, Dict[str, int]] = {}
        # 每个设备上的峰值内存使用量
        self._peak_mem: Dict[torch.device, int] = {}
        # 峰值内存快照字典，按设备和键值对应存储记录的峰值内存使用量
        self._peak_mem_snap: Dict[torch.device, Dict[str, int]] = {}
        # 弱引用字典，用于存储与每个张量存储相关的 ``_WeakRefInfo`` 实例
        self._WINFO = WeakIdKeyDictionary()
        # 创建一个 ModTracker 实例，用于跟踪模块的修改
        self._mod_tracker = ModTracker()
        # 通用的内存跟踪器，可用于任何 ``_RefType`` 子类
        self._ref_class: Type[_RefType] = _MemRefType
        # 用于标记是否在 AC 区域或优化器步骤区域的标志
        self._in_opt: bool = False
        self._in_ac: bool = False
        # 弱引用，指向当前活动的顶层 AC 模块
        self._ac_mod: Optional[weakref.ref] = None
        # 保存原始的 torch.UntypedStorage.resize_ 方法的引用
        self._orig_resize = torch.UntypedStorage.resize_
    
    def _update_snap(
        self,
        u_type: _UpdateType,
        winfo: _WeakRefInfo,
        old_mem_consumed: Optional[int] = None,
        old_reftype: Optional[_RefType] = None,
    ) -> None:
        # 初始化一个标志，用于跟踪在更新后总内存可能会降为零的情况。
        maybe_zero = False
        # 确保设备条目存在于当前内存快照中，必要时进行初始化。
        dev_snap = self._curr_mem_snap.setdefault(
            winfo.device, {reftype: 0 for reftype in self._ref_class}
        )
        dev_snap.setdefault(_TOTAL_KEY, 0)
        # 根据更新类型（`u_type`）处理不同类型的更新。
        if u_type == _UpdateType.ADD:
            # 增加特定引用类型的内存消耗，并更新总计。
            dev_snap[winfo.reftype] += winfo.mem_consumed
            dev_snap[_TOTAL_KEY] += winfo.mem_consumed
        elif u_type == _UpdateType.DEL:
            # 减少特定引用类型的内存消耗，并减少总计。
            dev_snap[winfo.reftype] -= winfo.mem_consumed
            dev_snap[_TOTAL_KEY] -= winfo.mem_consumed
            maybe_zero = True
        elif u_type == _UpdateType.REF:
            assert old_reftype is not None
            # 在同一设备内调整两种引用类型之间的内存消耗。
            dev_snap[old_reftype] -= winfo.mem_consumed
            dev_snap[winfo.reftype] += winfo.mem_consumed
        elif u_type == _UpdateType.SIZE:
            assert old_mem_consumed is not None
            # 根据大小变化调整引用类型的内存消耗。
            change = winfo.mem_consumed - old_mem_consumed
            dev_snap[winfo.reftype] += change
            dev_snap[_TOTAL_KEY] += change
            maybe_zero = True
        else:
            raise ValueError(f"Invalid update type: {u_type}")
        # 检查设备的总内存是否降为零。
        if maybe_zero:
            if self._curr_mem_snap[winfo.device][_TOTAL_KEY] == 0:
                # 如果总内存为零，则从内存快照中删除设备条目。
                del self._curr_mem_snap[winfo.device]

    def _update_and_maybe_create_winfos(
        self,
        t: torch.Tensor,
        reftype: _RefType,
        update_existing: bool = False,
    ) -> Set[_WeakRefInfo]:
        # 获取与给定类型 `t` 相关的所有 `_WeakRefInfo` 实例
        sts = _WeakRefInfo.get_untyped_storages(t)
        # 用于存储最终结果的空集合
        winfos = set()
        # 遍历所有的 `_WeakRefInfo` 实例
        for st in sts:
            # 尝试从跟踪字典中获取现有的 `_WeakRefInfo` 和其弱引用
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None:
                # 如果存在 `_WeakRefInfo`，检查其引用类型是否需要更新
                old_reftype = winfo.reftype
                if old_reftype != reftype:
                    # 更新引用类型，并通过 `_update_snap` 应用更改
                    winfo.reftype = reftype
                    self._update_snap(_UpdateType.REF, winfo, old_reftype=old_reftype)
                # 将更新后的 `_WeakRefInfo` 添加到结果集合中
                winfos.add(winfo)
            elif update_existing:
                # 如果未找到现有的 `_WeakRefInfo` 并且 `update_existing` 为 True，则引发错误
                raise KeyError("No existing winfo found")
            else:
                # 如果未找到现有的 `_WeakRefInfo` 并且 `update_existing` 为 False，则创建一个新的 `_WeakRefInfo`
                winfo, w_st = _WeakRefInfo.create_winfo(
                    st, t.device, reftype, self._delete_callback
                )
                # 将新创建的 `_WeakRefInfo` 和其弱引用存储在跟踪字典中
                self._WINFO[st] = (winfo, w_st)
                # 更新新添加的 `_WeakRefInfo` 的快照
                if winfo.mem_consumed > 0:
                    self._update_snap(_UpdateType.ADD, winfo)
                # 将新创建的 `_WeakRefInfo` 添加到结果集合中
                winfos.add(winfo)
        # 返回结果集合
        return winfos

    def _delete_callback(self, winfo: _WeakRefInfo, w_st: weakref.ref) -> None:
        # 当与 `_WeakRefInfo` 实例对应的存储对象即将被销毁时调用的回调函数
        if winfo.mem_consumed > 0:
            # 如果 `_WeakRefInfo` 实例的内存消耗大于零，则更新快照以标记其删除
            self._update_snap(_UpdateType.DEL, winfo)

    def _track_resize(self) -> None:
        # 需要进行 monkey-patch 的原因是 `torch.UntypedStorage.resize_` 没有被 `TorchDispatchMode` 捕获
        @wraps(self._orig_resize)
        def resize_(st: torch.UntypedStorage, size: int) -> None:
            # 调整 `torch.UntypedStorage` 的大小，并在必要时更新相关 `_WeakRefInfo` 的快照
            self._orig_resize(st, size)
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None and winfo.size != st.size():
                old_mem_consumed = winfo.mem_consumed
                winfo.update_mem_consumed(st)
                self._update_snap(
                    _UpdateType.SIZE, winfo, old_mem_consumed=old_mem_consumed
                )

        # 将定义好的 `resize_` 函数赋值给 `torch.UntypedStorage.resize_`
        torch.UntypedStorage.resize_ = resize_  # type: ignore[method-assign, assignment]

    def _restore_resize(self) -> None:
        # 恢复原始的 `torch.UntypedStorage.resize_` 函数定义
        torch.UntypedStorage.resize_ = self._orig_resize  # type: ignore[method-assign]
    def _update_peak_stats(self, peak_state: _State) -> None:
        # 首先捕获当前跟踪器状态的内存快照
        curr_snap = self._curr_mem_snap

        # 遍历 memory_tracking 中的每个模块统计信息
        for mod_stats in self.memory_tracking.values():
            # 检查当前模块是否在 _mod_tracker.parents 中，即是否活动
            if mod_stats.mod_fqn in self._mod_tracker.parents:
                # 如果模块在指定状态（peak_state）的快照中
                if peak_state in mod_stats.snapshots:
                    # 遍历当前内存快照的每个设备及其总内存使用量
                    for dev, dev_snap in curr_snap.items():
                        # 更新模块在每个设备上的峰值内存使用量
                        if mod_stats.local_peak.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                            mod_stats.local_peak[dev] = dev_snap[_TOTAL_KEY]
                            mod_stats.snapshots[peak_state][-1][dev] = deepcopy(
                                dev_snap
                            )

        # 再次遍历当前内存快照的每个设备及其总内存使用量
        for dev, dev_snap in curr_snap.items():
            # 更新整体跟踪器在每个设备上的峰值内存使用量
            if self._peak_mem.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                self._peak_mem[dev] = dev_snap[_TOTAL_KEY]
                self._peak_mem_snap[dev] = deepcopy(dev_snap)

    def _track(self, reftype: _RefType, t: torch.Tensor) -> None:
        # 获取张量的存储并检查是否已经跟踪过
        sts = _WeakRefInfo.get_untyped_storages(t)
        for st in sts:
            # 检查是否已经有 _WeakRefInfo 实例，如果有则更新内存消耗快照
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None:
                if winfo.size != st.size():
                    old_mem_consumed = winfo.mem_consumed
                    winfo.update_mem_consumed(st)
                    self._update_snap(
                        _UpdateType.SIZE, winfo, old_mem_consumed=old_mem_consumed
                    )
                return
            else:
                # 创建一个新的 _WeakRefInfo 实例并添加到字典中
                winfo, w_st = _WeakRefInfo.create_winfo(
                    st, t.device, reftype, self._delete_callback
                )
                self._WINFO[st] = (winfo, w_st)
                # 更新当前快照以反映新添加的 _WeakRefInfo
                if winfo.mem_consumed > 0:
                    self._update_snap(_UpdateType.ADD, winfo)

    def get_tracker_snapshot(
        self, type: str = "current"
    ) -> Dict[torch.device, Dict[str, int]]:
        """
        Capture a snapshot of the memory usage breakdown per device, based on the specified type.

        Args:
            type (str): The type of snapshot to capture. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
        Returns:
            Dict[torch.device, Dict[str, int]]: A dictionary where each key is a torch.device, and each value is another
                                                dictionary. This inner dictionary has keys representing memory reference
                                                types as defined in ``_MemRefType`` and values representing the amount of
                                                memory consumed in bytes.
        Raises:
            ValueError: If an invalid type is specified.
        """
        # 根据指定的类型，返回当前或峰值内存使用情况的深拷贝快照
        if type == "current":
            # 返回当前内存快照的深拷贝
            return deepcopy(self._curr_mem_snap)
        elif type == "peak":
            # 返回峰值内存快照的深拷贝
            return deepcopy(self._peak_mem_snap)
        else:
            # 如果指定的类型无效，则抛出 ValueError 异常
            raise ValueError(f"Invalid type {type}")

    def _track_module_params_and_buffers(
        self, module: nn.Module, install_grad_hooks: bool = True
    ) -> Tuple[int, int]:
        # 跟踪模块的参数和缓冲区，如果尚未跟踪。
        # 如果参数有梯度，则也跟踪梯度。
        # 如果 install_grad_hooks 是 True，且尚未安装梯度钩子，则在参数上安装梯度钩子来跟踪梯度。
        # 返回参数和缓冲区总共消耗的内存。

        def _grad_hook(grad: torch.Tensor) -> None:
            # 梯度钩子函数，更新并可能创建与其相关的内存信息。
            self._update_and_maybe_create_winfos(
                grad,
                _MemRefType.GRAD,
            )

        param_memory = 0
        for param in module.parameters():
            # 更新并可能创建与参数相关的内存信息。
            winfos = self._update_and_maybe_create_winfos(
                param,
                _MemRefType.PARAM,
            )
            param_memory += sum(winfo.mem_consumed for winfo in winfos)
            if param.grad is not None:
                # 如果参数有梯度，则更新并可能创建与梯度相关的内存信息。
                self._update_and_maybe_create_winfos(
                    param.grad,
                    _MemRefType.GRAD,
                )
            if (
                # 如果参数没有梯度钩子并且 install_grad_hooks 是 True，则注册梯度钩子。
                self._param_to_grad_hook_handles.get(param, None) is None
                and install_grad_hooks
            ):
                grad_hook_handle = param.register_hook(_grad_hook)
                post_acc_grad_hook_handle = param.register_post_accumulate_grad_hook(
                    lambda p: (_grad_hook(p.grad))
                )
                self._param_to_grad_hook_handles[param] = (
                    grad_hook_handle,
                    post_acc_grad_hook_handle,
                )
        
        buffer_memory = 0
        for buffer in module.buffers():
            # 更新并可能创建与缓冲区相关的内存信息。
            winfos = self._update_and_maybe_create_winfos(
                buffer,
                _MemRefType.BUFFER,
            )
            buffer_memory += sum(winfo.mem_consumed for winfo in winfos)
        
        return (param_memory, buffer_memory)

    def _track_inputs_or_outputs(self, args: Any) -> int:
        # 计算模块输入或输出消耗的内存。
        input_or_output_memory = 0

        def add_inps_or_outs(t: torch.Tensor) -> None:
            nonlocal input_or_output_memory
            # 获取张量 t 的未类型化存储器，并对其进行迭代。
            sts = _WeakRefInfo.get_untyped_storages(t)
            for st in sts:
                # 获取与存储器相关的弱引用信息。
                winfo, _ = self._WINFO.get(st, (None, None))
                if winfo is not None:
                    # 如果有相关的内存信息，则将其内存消耗加到输入或输出内存中。
                    input_or_output_memory += winfo.mem_consumed

        # 仅对 torch.Tensor 类型的参数 args 进行树形映射操作。
        tree_map_only(torch.Tensor, add_inps_or_outs, args)
        
        return input_or_output_memory
    def _post_fw_hook(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        # 这个函数作为一个后向传播的用户钩子函数被安装在 ``ModTracker`` 上。根据以下情况设置状态并捕获模块的内存快照。
        # Case 1: 如果在反向传播中调用，表示我们在 AC 区域。如果这是 AC 区域中的顶级模块，
        #         我们将标志 ``_in_ac`` 设置为 False。
        # Case 2: 如果在前向传播中调用，我们计算模块的输出内存，并更新其 mod_stats。
        mod_stats = self.memory_tracking[module]
        if self._mod_tracker.is_bw:
            state = _ModState.POST_FW_AC
            if self._ac_mod is not None and self._ac_mod() is module:
                self._ac_mod = None
                self._in_ac = False
        else:
            state = _ModState.POST_FW
            output_mem = self._track_inputs_or_outputs(outputs)
            mod_stats.output_mem = output_mem
        mod_stats.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())

    def _pre_bw_hook(self, module: nn.Module, args: Any) -> None:
        # 这个函数作为一个前向传播的用户钩子函数被安装在 ``ModTracker`` 上。设置状态并捕获模块的快照。
        # 同时初始化 ``local_peak`` 和 ``PEAK_BW`` 的快照。
        # 如果模块为 None，我们跳过钩子。
        # 这可能发生因为它被安装在模块输出张量的多梯度钩子中，而在反向传播期间模块本身可能不存活。
        if module is None:
            warnings.warn("Module is None. Skipping PRE_BW hook.", stacklevel=2)
            return
        mod_stats = self.memory_tracking[module]
        mem_snapshot = self.get_tracker_snapshot()
        mod_stats.local_peak = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
        }
        mod_stats.snapshots.setdefault(_ModState.PEAK_BW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(_ModState.PRE_BW, []).append(
            deepcopy(mem_snapshot)
        )

    def _post_bw_hook(self, module: nn.Module, args: Any) -> None:
        # 这个函数作为一个后向传播的用户钩子函数被安装在 ``ModTracker`` 上。设置状态并捕获模块的快照。
        # 如果模块为 None，我们跳过钩子。
        # 这可能发生因为它被安装在模块输入张量的多梯度钩子中，而在反向传播期间模块本身可能不存活。
        if module is None:
            warnings.warn("Module is None. Skipping POST_BW hook.", stacklevel=2)
            return
        mod_stats = self.memory_tracking[module]
        mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
            self.get_tracker_snapshot()
        )

    def _track_optimizer_states(
        self, reftype: _RefType, optimizer: optim.Optimizer
    ) -> None:
        # Iterate over each optimizer state dictionary
        for states in optimizer.state.values():
            # Iterate over values in each optimizer state dictionary
            for val in states.values():
                # Check if the value is a torch.Tensor instance
                if isinstance(val, torch.Tensor):
                    # Update and possibly create weight infos for the tensor
                    self._update_and_maybe_create_winfos(
                        val,
                        reftype,
                    )

    def _register_global_optimizer_hook(self) -> None:
        # Register hooks to track optimizer states during optimizer steps.
        # The pre-hook sets `_in_opt` flag to True before optimization step.
        # The post-hook resets `_in_opt` flag to False and tracks optimizer states.
        def _opt_step_pre_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            self._in_opt = True

        def _opt_step_post_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            # Track optimizer states after optimizer step
            self._track_optimizer_states(_MemRefType.OPT, optimizer)
            self._in_opt = False

        # Register pre and post optimizer step hooks and store handles
        self._optimizer_hook_handles = (
            register_optimizer_step_pre_hook(_opt_step_pre_hook),
            register_optimizer_step_post_hook(_opt_step_post_hook),
        )

    def _deregister_param_and_optimizer_hooks(self) -> None:
        # Remove hooks associated with parameters and optimizer
        for (
            grad_hook_handle,
            post_acc_grad_hook_handle,
        ) in self._param_to_grad_hook_handles.values():
            grad_hook_handle.remove()
            post_acc_grad_hook_handle.remove()
        self._param_to_grad_hook_handles.clear()

        # Remove optimizer hooks if they exist
        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def track_external(
        self, *external: Union[nn.Module, optim.Optimizer, torch.Tensor]
        # Method to track external components like nn.Module, optim.Optimizer, torch.Tensor

    ) -> None:
        # Iterate over each optimizer state dictionary
        for states in optimizer.state.values():
            # Iterate over values in each optimizer state dictionary
            for val in states.values():
                # Check if the value is a torch.Tensor instance
                if isinstance(val, torch.Tensor):
                    # Update and possibly create weight infos for the tensor
                    self._update_and_maybe_create_winfos(
                        val,
                        reftype,
                    )

    def _register_global_optimizer_hook(self) -> None:
        # Register hooks to track optimizer states during optimizer steps.
        # The pre-hook sets `_in_opt` flag to True before optimization step.
        # The post-hook resets `_in_opt` flag to False and tracks optimizer states.
        def _opt_step_pre_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            self._in_opt = True

        def _opt_step_post_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            # Track optimizer states after optimizer step
            self._track_optimizer_states(_MemRefType.OPT, optimizer)
            self._in_opt = False

        # Register pre and post optimizer step hooks and store handles
        self._optimizer_hook_handles = (
            register_optimizer_step_pre_hook(_opt_step_pre_hook),
            register_optimizer_step_post_hook(_opt_step_post_hook),
        )

    def _deregister_param_and_optimizer_hooks(self) -> None:
        # Remove hooks associated with parameters and optimizer
        for (
            grad_hook_handle,
            post_acc_grad_hook_handle,
        ) in self._param_to_grad_hook_handles.values():
            grad_hook_handle.remove()
            post_acc_grad_hook_handle.remove()
        self._param_to_grad_hook_handles.clear()

        # Remove optimizer hooks if they exist
        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def track_external(
        self, *external: Union[nn.Module, optim.Optimizer, torch.Tensor]
        # Method to track external components like nn.Module, optim.Optimizer, torch.Tensor
    ) -> None:
        """
        Track tensors and stateful objects like modules, optimizers etc. that are created outside the MemTracker.

        This method should be called before the ``MemTracker`` is used. Any tensors that are not module parameters, buffers,
        gradients activations, or optimizer states will be categorized as ``Other``. If you want them categorized with a
        custom name, please file a GitHub issue. Any tensors created outside the MemTracker and not supplied to this
        method will not be tracked by ``MemTracker``.

        Args:
            *external (Union[nn.Module, optim.Optimizer, torch.Tensor]): The external modules, optimizers, and
                                                                         tensors to be tracked.
        """
        # Flatten the list of external objects to ensure each item is individually accessible
        flat_external, _ = tree_flatten(external)
        # Iterate over each external object
        for obj in flat_external:
            # Check if the object is a torch.Tensor
            if isinstance(obj, torch.Tensor):
                # Update the memory tracker for the tensor, categorizing it as 'Other'
                self._update_and_maybe_create_winfos(
                    obj,
                    _MemRefType.OTH,
                )
            # Check if the object is a torch.nn.Module
            elif isinstance(obj, torch.nn.Module):
                # Track parameters and buffers of the module (without installing gradient hooks)
                self._track_module_params_and_buffers(obj, install_grad_hooks=False)
            # Check if the object is an optim.Optimizer
            elif isinstance(obj, optim.Optimizer):
                # Track states of the optimizer
                self._track_optimizer_states(_MemRefType.OPT, obj)
            # Raise an error if the object type is not supported
            else:
                raise TypeError(
                    f"Object of type {type(obj)} is not supported for tracking. "
                    f"Only stateful objects like modules, optimizers, and tensors are supported."
                )

    def display_snapshot(
        self, type: str = "current", units: str = "B", tabulate: bool = False
    ) -> None:
        """
        Display the memory usage breakdown snapshot of the tracker based on the specified type and units.

        Keyword args:
            type (str): The type of snapshot to display. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
            units (str): The units to use for displaying memory usage. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool): Whether to display the snapshot in a tabular format. Defaults to False.
        """
        # Get the memory snapshot based on the specified type
        snapshot = self.get_tracker_snapshot(type)
        # Print the snapshot in a tabular format if tabulate is True, otherwise print in a non-tabular format
        if tabulate:
            _print_snapshot_tabular(snapshot, units)
        else:
            _print_snapshot(snapshot, units)

    def display_modulewise_snapshots(
        self, depth: int = 2, units: str = "B", tabulate: bool = False
    ) -> None:
        """
        Display memory usage breakdown snapshots for modules based on the specified depth and units.

        Keyword args:
            depth (int): The depth of module hierarchy to display snapshots for. Defaults to 2.
            units (str): The units to use for displaying memory usage. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool): Whether to display the snapshots in a tabular format. Defaults to False.
        """
    def print_memory_breakdown_snapshot(
        self,
        depth: int = 2,
        units: str = "B",
        tabulate: bool = False
    ) -> None:
        """
        Print per device memory breakdown snapshot for each module called within MemTracker.

        Snapshots are displayed for the states defined by ``_ModState``.
        The module hierarchy is displayed up to the specified depth.

        Keyword Args:
            depth (int, optional): The depth of the module hierarchy to display. Defaults to 2.
            units (str, optional): The units to use for memory tracking. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool, optional): Whether to display the snapshot in a tabular format. Defaults to False.
        """

        def natural_sort_key(s: str) -> List[Union[int, str]]:
            """
            Generate a natural sorting key for strings, treating numbers numerically.

            Args:
                s (str): String to generate natural sort key for.

            Returns:
                List[Union[int, str]]: List suitable for sorting strings naturally.
            """
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("([0-9]+)", s)
            ]

        # Iterate over memory tracking values, sorted by module fully qualified name
        for mod_stats in sorted(
            self.memory_tracking.values(),
            key=lambda m_stats: natural_sort_key(m_stats.mod_fqn),
        ):
            mod_fqn = mod_stats.mod_fqn
            mod_depth = mod_fqn.count(".") + 1
            # Skip modules deeper than specified depth
            if mod_depth > depth:
                continue
            # Print module fully qualified name
            print(f"Module:  {mod_fqn}")
            # Print memory state snapshots in either tabular or non-tabular format based on `tabulate` flag
            if tabulate:
                _print_state_snapshots_tabular(mod_stats.snapshots, units)
            else:
                _print_state_snapshots(mod_stats.snapshots, units)

    def reset_mod_stats(self) -> None:
        """
        Reset all the module memory stats. Clears ``memory_tracking`` dictionary.
        """
        self.memory_tracking.clear()

    def __enter__(self) -> "MemTracker":
        """
        Enter method for context management.

        Registers hooks and initializes tracking variables.
        """
        self._register_global_optimizer_hook()
        self._mod_tracker.register_user_hooks(
            self._pre_fw_hook,
            self._post_fw_hook,
            self._pre_bw_hook,
            self._post_bw_hook,
        )
        self._track_resize()
        self._peak_mem_snap = self.get_tracker_snapshot()
        self._peak_mem = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in self._peak_mem_snap.items()
        }
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """
        Exit method for context management.

        Deregisters hooks and cleans up tracking state.
        """
        self._deregister_param_and_optimizer_hooks()
        self._mod_tracker.clear_user_hooks()
        self._restore_resize()
        super().__exit__(*args)
        self._mod_tracker.__exit__(*args)
    # 定义特殊方法 __torch_dispatch__，处理函数调度逻辑，并且允许未注明类型的定义
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        # 调用传入的函数 func，并传递 args 和 kwargs 参数
        res = func(*args, **kwargs or {})
        
        # 根据当前状态选择引用类型
        if self._in_opt:
            # 如果当前正在跟踪优化器状态，则使用优化器引用类型
            reftype = _MemRefType.OPT
        elif self._mod_tracker.is_bw and not self._in_ac:
            # 如果当前在反向传播区域且不在自动编码区域，则使用反向传播引用类型
            reftype = _MemRefType.TEMP
        else:
            # 否则使用正向传播引用类型
            reftype = _MemRefType.ACT
        
        # 通过 tree_map_only 函数，对 res 中的 torch.Tensor 类型对象应用部分函数 _track(reftype)
        tree_map_only(torch.Tensor, partial(self._track, reftype), res)
        
        # 根据模块状态选择更新峰值统计信息
        peak_state = _ModState.PEAK_BW if self._mod_tracker.is_bw else _ModState.PEAK_FW
        self._update_peak_stats(peak_state)
        
        # 返回调用 func 后的结果 res
        return res
```