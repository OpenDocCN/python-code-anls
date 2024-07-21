# `.\pytorch\torch\distributed\checkpoint\format_utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的库
import argparse  # 用于命令行参数解析
import os  # 提供与操作系统交互的功能
from enum import Enum  # 提供枚举类型的支持
from typing import cast, Dict, List, Optional, Union  # 提供类型提示功能

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式支持
from torch.distributed._shard._utils import narrow_tensor_by_index  # 分布式相关工具函数
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter  # 分布式存储系统读写支持
from torch.distributed.checkpoint._nested_dict import flatten_state_dict  # 处理嵌套字典的状态字典扁平化
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
    DefaultLoadPlanner,
)  # 默认的状态字典加载策划器
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)  # 分布式存储元数据相关
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner  # 状态字典加载计划相关
from torch.distributed.checkpoint.planner_helpers import _create_chunk_list  # 辅助函数，创建数据块列表
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict  # 加载状态字典
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict  # 保存状态字典
from torch.distributed.checkpoint.storage import StorageReader  # 分布式存储读取支持
from torch.futures import Future  # 用于处理异步操作的 Future 类

__all__ = [
    "dcp_to_torch_save",
    "torch_save_to_dcp",
    "BroadcastingTorchSaveReader",
    "DynamicMetaLoadPlanner",
]

class BroadcastingTorchSaveReader(StorageReader):
    """
    StorageReader for reading a Torch Save file. This reader will read the entire checkpoint
    on the coordinator rank, and then broadcast and shard each tensor to all ranks.

    . N.B. Intended to be used with DynamicMetaLoadPlanner

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def __init__(
        self,
        checkpoint_id: Optional[Union[str, os.PathLike]] = None,
        coordinator_rank: int = 0,
    ) -> None:
        self.checkpoint_id = checkpoint_id  # 初始化检查点 ID
        self.coordinator_rank = coordinator_rank  # 初始化协调器排名

    def read_metadata(self) -> Metadata:
        """Extends the default StorageReader to support building the metadata file"""
        # Metadata is built in planner.set_up_planner, since we are not actually reading metadata from
        # the disk
        return Metadata(state_dict_metadata={})  # 返回空的元数据对象
    # 读取数据方法，异步返回空结果Future[None]
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Reads torch save data on the coordinator rank, and broadcast afterwards
        this incurrs a communication cost, but avoids having to load
        the entire checkpoint on each rank, hopefully preventing OOM issues
        """
        # 将planner强制转换为DefaultLoadPlanner类型
        planner = cast(DefaultLoadPlanner, planner)

        # 数据在协调器等级读取后广播，虽然有通信成本，但避免了在每个等级上加载整个检查点，有望避免OOM问题
        # TODO: 在每个主机上读取，而不仅限于协调器
        if self.is_coordinator:
            # 断言是否为协调器，并且检查点ID不为None
            assert self.checkpoint_id is not None
            # 使用"cpu"位置加载torch保存的状态字典
            torch_state_dict = torch.load(self.checkpoint_id, map_location="cpu")
            # 如果planner要求扁平化状态字典，则执行扁平化操作
            if planner.flatten_state_dict:
                torch_state_dict, _ = flatten_state_dict(torch_state_dict)
        else:
            torch_state_dict = None

        # 遍历加载计划中的每一个请求
        for req in plan.items:
            if req.type == LoadItemType.BYTE_IO:
                # 如果请求类型是BYTE_IO，则引发运行时错误
                raise RuntimeError(
                    f"Non-tensor value identified at {req.storage_index.fqn}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            # 如果当前实例是协调器
            if self.is_coordinator:
                # 获取分布式默认设备
                pg_device = dist.distributed_c10d._get_pg_default_device()
                # 将tensor移动到pg_device设备上
                tensor = torch_state_dict[req.storage_index.fqn].to(pg_device)
            else:
                # 否则创建一个与planner状态字典中req索引相同形状的空tensor
                tensor = torch.empty_like(planner.state_dict[req.storage_index.fqn])

            # 广播tensor，从协调器等级发送
            dist.broadcast(tensor, src=self.coordinator_rank, async_op=False)

            # 根据请求的偏移量和长度调整tensor大小
            tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
            # 解析目标tensor并分离
            target_tensor = planner.resolve_tensor(req).detach()
            # 断言目标tensor大小与tensor大小匹配
            assert target_tensor.size() == tensor.size(), (
                f"req {req.storage_index} mismatch sizes, "
                f"{target_tensor.size()} vs {tensor.size()}"
            )
            # 将tensor的数据复制到目标tensor中
            target_tensor.copy_(tensor)
            # 提交目标tensor到planner中
            planner.commit_tensor(req, target_tensor)

        # 创建一个空的Future对象，并设置结果为None，返回
        fut: Future = Future()
        fut.set_result(None)
        return fut

    # 设置存储读取器，接受元数据和是否为协调器的布尔值
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Implementation of the StorageReader method"""
        # 设置实例是否为协调器
        self.is_coordinator = is_coordinator
        # 如果实例是协调器，则断言当前进程的等级必须与协调器等级相同
        if self.is_coordinator:
            assert dist.get_rank() == self.coordinator_rank

        # 断言检查点ID不为None
        assert self.checkpoint_id is not None

    # 准备本地加载计划，返回加载计划本身
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Implementation of the StorageReader method"""
        return plan

    # 准备全局加载计划，返回加载计划列表本身
    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        """Implementation of the StorageReader method"""
        return global_plan
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """
        重置对象的状态，设置新的检查点 ID
        
        Parameters:
        - checkpoint_id: 检查点 ID，可以是字符串、路径或者 None
        
        Returns:
        - None
        
        Notes:
        这是 StorageReader 类的方法实现，用于重置对象的状态。
        """
        self.checkpoint_id = checkpoint_id



    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        验证检查点 ID 是否有效
        
        Parameters:
        - checkpoint_id: 检查点 ID，可以是字符串或者路径
        
        Returns:
        - bool: 如果检查点 ID 是一个文件则返回 True，否则返回 False
        
        Notes:
        这是 StorageReader 类的类方法实现，用于验证给定的检查点 ID 是否对应一个存在的文件。
        """
        return os.path.isfile(checkpoint_id)
class DynamicMetaLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which creates a new Metadata object based on the passed in state dict,
    avoiding the need to read metadata from disk. This is useful when reading formats which don't have a
    metadata file, like Torch Save files.

    . N.B. Intended to be used with BroadcastingTorchSaveReader

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        """Setups of the planner, extnding default behavior by creating the Metadata object from the state dict"""
        # 调用父类的设置计划方法，扩展默认行为以从状态字典创建 Metadata 对象
        super().set_up_planner(state_dict, metadata, is_coordinator)

        # 初始化状态字典的元数据为一个空字典
        state_dict_metadata: Dict[str, STORAGE_TYPES] = {}
        # 遍历状态字典中的每个键值对
        for key, tensor in self.state_dict.items():
            # 如果值不是张量，则抛出运行时错误
            if not torch.is_tensor(tensor):
                raise RuntimeError(
                    f"Non-tensor value identified at {key}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            # 为当前键创建张量存储的元数据
            state_dict_metadata[key] = TensorStorageMetadata(
                TensorProperties(dtype=tensor.dtype),
                tensor.size(),
                _create_chunk_list(tensor),
            )
        # 创建 Metadata 对象，使用状态字典的元数据
        self.metadata = Metadata(state_dict_metadata=state_dict_metadata)


def dcp_to_torch_save(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    torch_save_path: Union[str, os.PathLike],
):
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    # 初始化一个空的状态字典
    sd: STATE_DICT_TYPE = {}
    # 加载状态字典，使用文件系统阅读器读取 DCP 检查点目录下的内容
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    # 将状态字典保存为 Torch 保存文件
    torch.save(sd, torch_save_path)


def torch_save_to_dcp(
    torch_save_path: Union[str, os.PathLike],
    dcp_checkpoint_dir: Union[str, os.PathLike],
):
    """
    Given the location of a torch save file, converts it into a DCP checkpoint.

    Args:
        torch_save_path: Filename of the Torch save file.
        dcp_checkpoint_dir: Directory to store the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    # 加载 Torch 保存文件的状态字典
    state_dict = torch.load(torch_save_path)
    # 在这里我们不需要状态行为，因为预期由torch.load加载的内容不会包含有状态的对象。
    # 调用_save_state_dict函数，将state_dict保存到文件系统中的指定目录dcp_checkpoint_dir。
    # 参数storage_writer指定为FileSystemWriter，参数no_dist设为True，表示不进行分布式存储。
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )
if __name__ == "__main__":
    # 确保该脚本作为主程序运行而不是被导入
    # 定义枚举类型FormatMode，包含两个转换模式：TORCH_TO_DCP和DCP_TO_TORCH
    class FormatMode(Enum):
        TORCH_TO_DCP = "torch_to_dcp"
        DCP_TO_TORCH = "dcp_to_torch"

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加mode参数，指定转换模式，可选项为FormatMode枚举类型的值
    parser.add_argument(
        "mode",
        type=str,
        help="Conversion mode",
        choices=[m.value for m in FormatMode],
        default=FormatMode.TORCH_TO_DCP,
    )
    # 添加src参数，指定源模型的路径
    parser.add_argument("src", type=str, help="Path to the source model")
    # 添加dst参数，指定目标模型的路径
    parser.add_argument("dst", type=str, help="Path to the destination model")
    # 解析命令行参数
    args = parser.parse_args()

    # 打印转换的信息，包括源模型路径、目标模型路径和转换方法
    print(
        f"Converting checkpoint from {args.src} to {args.dst} using method: '{args.mode}'"
    )
    # 当源模型不存在时输出的警告信息
    checkpoint_missing_warning = (
        f"No checkpoint found at {args.src}. Skipping conversion."
    )
    
    # 根据转换模式执行相应的操作
    if args.mode == FormatMode.TORCH_TO_DCP.value:
        # 如果源模型文件存在，则调用torch_save_to_dcp函数进行转换
        if os.path.isfile(args.src):
            torch_save_to_dcp(args.src, args.dst)
        else:
            # 否则输出源模型不存在的警告信息
            print(checkpoint_missing_warning)
    elif args.mode == FormatMode.DCP_TO_TORCH.value:
        # 如果源模型是目录存在，则调用dcp_to_torch_save函数进行转换
        if os.path.isdir(args.src):
            dcp_to_torch_save(args.src, args.dst)
        else:
            # 否则输出源模型不存在的警告信息
            print(checkpoint_missing_warning)
    else:
        # 如果mode不是预定义的转换模式，则抛出ValueError异常
        raise ValueError(f"Unknown conversion mode: {args.mode}")
```