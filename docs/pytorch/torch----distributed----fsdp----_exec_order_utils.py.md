# `.\pytorch\torch\distributed\fsdp\_exec_order_utils.py`

```py
# 设置类型检查的全局选项，允许未声明类型的函数
# mypy: allow-untyped-defs
# 导入必要的库和模块
import itertools  # 提供了创建迭代器的函数
import warnings  # 提供了警告处理功能
from enum import auto, Enum  # 导入枚举相关的类和函数
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入FSDP中的遍历工具模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns  # 导入FSDP中的公共工具函数
from torch.distributed.fsdp._flat_param import FlatParamHandle  # 导入FSDP中的扁平化参数处理类


class _ExecOrderWarnStatus(Enum):
    """用于内部执行顺序验证的枚举类。"""

    NONE = auto()  # 没有偏差
    WARNING = auto()  # 当前迭代有偏差，正在发出警告
    WARNED = auto()  # 先前迭代有偏差


class _ExecOrderData:
    """
    包含用于跟踪执行顺序的数据结构。我们在第一个迭代中跟踪前向预取的前向顺序
    （假设静态图），并在每个迭代中跟踪后向预取的后向顺序（不假设静态图，但可能提供不正确的顺序）。
    """

    def __init__(
        self,
        debug_level: dist.DebugLevel,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None:
        # 用于执行顺序验证和前向预取的（静态）前向顺序的跟踪
        self.handles_pre_forward_order: List[FlatParamHandle] = []
        # 用于前向预取的后向顺序的跟踪
        self.handles_post_forward_order: List[Optional[FlatParamHandle]] = []
        self._iter = 0

        # 每个模块可以进行的最大后向/前向预取的全收集次数
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        # 用于执行顺序验证的数据结构
        self._checking_order: bool = debug_level == dist.DebugLevel.DETAIL
        self.process_group: Optional[dist.ProcessGroup] = None
        self.world_size: Optional[int] = None
        self.all_handles: List[FlatParamHandle] = []
        # 参数名从根模块开始前缀化
        self.param_to_fqn: Dict[nn.Parameter, List[str]] = {}
        # 当前前向执行顺序中的索引
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(
        self,
        state: _FSDPState,
        root_module: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> None:
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        # Fix an order over the handles, which should be the same across ranks
        for handle in traversal_utils._get_fsdp_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index
        self.param_to_fqn = _get_param_to_fqns(root_module)
        # TODO (awgu): We can broadcast the metadata of rank 0's `all_handles`
        # to check that all ranks have the same handles in the same order.
        # https://github.com/pytorch/pytorch/issues/79620

    @property
    def is_first_iter(self) -> bool:
        """
        Property method to check if it's the first iteration.
        """
        return self._iter == 0

    def get_handle_to_backward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> Optional[FlatParamHandle]:
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> Optional[FlatParamHandle]:
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle
    def record_post_forward(self, handle: Optional[FlatParamHandle]) -> None:
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
        # 如果 handle 为空，直接返回，不进行记录
        if not handle:
            return
        # 只记录首次使用 handles 键的情况
        if handle._post_forward_index:
            self.handles_post_forward_order.append(handle)
            return
        # 获取当前 handles_post_forward_order 的长度作为索引，设置 handle 的 _post_forward_index
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index
        # 将 handle 记录到 handles_post_forward_order 中
        self.handles_post_forward_order.append(handle)

    def record_pre_forward(
        self, handle: Optional[FlatParamHandle], is_training: bool
    ) -> None:
        """
        Records ``handles`` in the pre-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
        # 如果 handle 为空，直接返回，不进行记录
        if not handle:
            return
        # 在第一次迭代时，检查跨 ranks 的执行顺序
        self._check_order(handle, is_training)
        # 在第一次迭代后固定顺序，并且只记录首次使用 handles 键的情况
        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        # 获取当前 handles_pre_forward_order 的长度作为索引，设置 handle 的 _pre_forward_order_index
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        # 将 handle 记录到 handles_pre_forward_order 中
        self.handles_pre_forward_order.append(handle)

    def _get_handle_indices(
        self,
        handle: FlatParamHandle,
    ) -> Tuple[Optional[int], ...]:
        """
        Returns the handle indices (i.e. indices into ``self.all_handles``)
        corresponding to the handles in ``handle``. An entry in the
        returned tuple is ``None`` if the handle is invalid.
        """
        # 初始化空列表 indices，用于存储 handle 的索引
        indices: List[Optional[int]] = []
        # 如果 handle 不为空，将 handle 的 _handle_index 加入 indices 列表中
        if handle:
            indices.append(handle._handle_index)
        # 将 indices 转换为元组并返回
        return tuple(indices)

    def _get_names_from_handle_indices(
        self,
        handle_indices: Tuple[int, ...],
    ) -> List[List[str]]:
        """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
        # 初始化空列表 fqns，用于存储每个 handle 的 FQN
        fqns: List[List[str]] = []
        # 遍历 handle_indices 中的每个索引
        for index in handle_indices:
            # 如果索引为空或者索引无效，则跳过
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            # 获取索引对应的 handle，并获取其完全限定名（FQN），加入 fqns 列表中
            handle = self.all_handles[index]
            flat_param = handle.flat_param
            fqns.append(self.param_to_fqn[flat_param])
        # 返回包含 FQN 列表的 fqns
        return fqns

    def _get_names_from_handles(
        self, handle: FlatParamHandle,
        handle_indices: Tuple[int, ...]
    ) -> List[List[str]]:
        """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
        # 略，未完待续...
    ) -> List[List[str]]:
        """
        返回一个包含每个句柄在 ``handles_key`` 中的全限定名（FQN）列表。如果句柄无效，则其全限定名将从返回列表中省略。
        """
        fqns: List[List[str]] = []  # 初始化一个空列表，用于存储全限定名列表的列表
        if handle:
            flat_param = handle.flat_param  # 获取句柄的 flat_param 属性
            if flat_param in self.param_to_fqn:  # 检查 flat_param 是否存在于 param_to_fqn 字典中
                fqns.append(self.param_to_fqn[flat_param])  # 将对应的全限定名列表添加到 fqns 中
        return fqns  # 返回全限定名列表的列表

    def next_iter(self):
        """
        在每次迭代后更新内部数据结构。这应当在后向回调函数中调用，因为这标志着迭代的真正结束。
        """
        self._iter += 1  # 迭代次数加一
        self.handles_post_forward_order.clear()  # 清空 handles_post_forward_order 列表
        if self._checking_order:
            self.current_order_index = 0  # 将 current_order_index 设置为 0
            if self.warn_status == _ExecOrderWarnStatus.WARNING:
                self.warn_status = _ExecOrderWarnStatus.WARNED  # 如果 warn_status 是 WARNING，则将其设置为 WARNED
```