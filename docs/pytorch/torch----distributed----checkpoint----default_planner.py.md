# `.\pytorch\torch\distributed\checkpoint\default_planner.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的模块和库
import dataclasses                      # 用于数据类的模块
import io                               # 提供核心的 Python I/O 功能的模块
import logging                          # 提供日志记录功能的模块
import operator                         # 提供了一组内置的操作符函数的模块
from collections import ChainMap        # 提供链式映射类的模块
from functools import reduce            # 提供高阶函数：部分函数应用、链式调用等的模块
from typing import Any, cast, Dict, List, Optional, Tuple, Union  # 提供类型提示相关的模块

import torch                            # PyTorch 深度学习框架
from torch.distributed._shard._utils import narrow_tensor_by_index  # 从分布式相关模块导入的函数
from torch.distributed._tensor import DTensor  # 分布式张量相关模块
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans  # 导入去重保存计划相关模块
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,                     # 导入用于扁平化字典的映射
    flatten_state_dict,                  # 导入用于扁平化状态字典的函数
)
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors  # 导入用于扁平化分片张量的函数
from torch.distributed.checkpoint._traverse import set_element  # 导入用于设置元素的函数
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,                # 导入字节存储元数据类
    ChunkStorageMetadata,                # 导入块存储元数据类
    Metadata,                            # 导入元数据类
    MetadataIndex,                       # 导入元数据索引类
    STATE_DICT_TYPE,                     # 导入状态字典类型
    STORAGE_TYPES,                       # 导入存储类型
    StorageMeta,                         # 导入存储元数据类
    TensorStorageMetadata,               # 导入张量存储元数据类
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,                            # 导入加载计划类
    LoadPlanner,                         # 导入加载规划器类
    ReadItem,                            # 导入读取项类
    SavePlan,                            # 导入保存计划类
    SavePlanner,                         # 导入保存规划器类
    WriteItem,                           # 导入写入项类
    WriteItemType,                       # 导入写入项类型枚举
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_default_metadata_only_plan,  # 导入创建默认仅元数据计划的函数
    _create_read_items,                  # 导入创建读取项的函数
    _create_write_items,                 # 导入创建写入项的函数
    _init_state_dict,                    # 导入初始化状态字典的函数
)
from torch.distributed.checkpoint.utils import find_state_dict_object  # 导入查找状态字典对象的函数

logger: logging.Logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


__all__ = [
    "DefaultSavePlanner",                 # 将以下类和函数加入模块的公开接口中
    "DefaultLoadPlanner",
    "create_default_local_load_plan",
    "create_default_global_load_plan",
    "create_default_local_save_plan",
    "create_default_global_save_plan",
]


# TODO: Update docstrings for default_planner.py
class DefaultSavePlanner(SavePlanner):
    mappings: FLATTEN_MAPPING              # 定义映射关系的类型为 FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,    # 是否扁平化状态字典的标志
        flatten_sharded_tensors: bool = True,  # 是否扁平化分片张量的标志
        dedup_replicated_tensors: Optional[bool] = None,  # 是否去重复制张量的标志（已废弃）
        dedup_save_to_lowest_rank: bool = False,  # 是否保存到最低等级的节点的标志
    ) -> None:
        self.flatten_state_dict = flatten_state_dict  # 初始化是否扁平化状态字典的属性
        self.flatten_sharded_tensors = flatten_sharded_tensors  # 初始化是否扁平化分片张量的属性
        self.mappings = {}                  # 初始化映射关系为空字典
        self.dedup_save_to_lowest_rank = dedup_save_to_lowest_rank  # 初始化保存到最低等级节点的属性
        if dedup_replicated_tensors is not None:
            logger.warning(
                "DefaultSavePlanner's `dedup_replicated_tensors` argument is being "
                "deprecated, and no longer has any effect. Please remove this argument "
                "from your call."
            )                               # 若 dedup_replicated_tensors 参数不为空则发出警告

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,        # 状态字典的类型
        storage_meta: Optional[StorageMeta] = None,  # 存储元数据的可选类型
        is_coordinator: bool = False,       # 是否为协调节点的标志
    ) -> None:
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)  # 若需要扁平化状态字典，则进行扁平化并获取映射关系
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)  # 若需要扁平化分片张量，则进行扁平化
        self.state_dict = state_dict         # 将处理后的状态字典保存到对象属性中
        self.is_coordinator = is_coordinator  # 设置是否为协调节点的属性
    # 创建本地保存计划并返回
    def create_local_plan(self) -> SavePlan:
        # 调用函数创建默认的本地保存计划，使用当前状态和协调员标志
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        # 如果需要展开状态字典，则替换计划中的 planner_data 属性为 mappings
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        # 将创建的计划赋值给对象的 plan 属性
        self.plan = plan
        # 返回创建的计划
        return self.plan

    # 创建全局保存计划并返回计划列表和元数据
    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
        # 去除重复的保存计划，根据指定的 dedup_save_to_lowest_rank 策略
        all_plans = dedup_save_plans(all_plans, self.dedup_save_to_lowest_rank)
        # 创建全局保存计划和元数据
        global_plan, metadata = create_default_global_save_plan(all_plans)

        # 如果需要展开状态字典
        if self.flatten_state_dict:
            # 从全局计划列表中获取每个计划的 planner_data 属性并合并成字典
            planner_data_dict = [p.planner_data for p in global_plan]
            merged_mappings = dict(ChainMap(*planner_data_dict))
            # 替换元数据中的 planner_data 属性为合并后的字典
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

        # 验证全局计划和元数据是否有效
        if not _validate_global_plan(global_plan, metadata):
            raise ValueError("Failed to validate global plan")

        # 将全局计划和元数据赋值给对象的相应属性
        self.global_plan = global_plan
        self.metadata = metadata

        # 返回全局计划列表和元数据
        return self.global_plan, self.metadata

    # 完成计划并返回新的计划
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        # 将传入的新计划赋值给对象的 plan 属性
        self.plan = new_plan
        # 返回新计划
        return new_plan

    # 根据写入项查找对象并返回对应对象
    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        # 查找写入项对应的对象
        object = self.lookup_object(write_item.index)
        # 转换对象并返回结果
        return self.transform_object(write_item, object)

    # 根据元数据索引查找对象并返回
    def lookup_object(self, index: MetadataIndex) -> Any:
        """从规划接口扩展以便于扩展默认规划器。"""
        # 调用函数查找状态字典中对应索引的对象并返回
        return find_state_dict_object(self.state_dict, index)

    # 转换对象，根据写入项类型进行不同的处理
    def transform_object(self, write_item: WriteItem, object: Any):
        """从规划接口扩展以便于扩展默认规划器。"""
        # 如果写入项类型为 BYTE_IO
        if write_item.type == WriteItemType.BYTE_IO:
            # 创建字节流对象
            bytes = io.BytesIO()
            # 使用 Torch 将对象保存到字节流中
            torch.save(object, bytes)
            # 将字节流对象赋值给 object
            object = bytes
        # 返回转换后的对象
        return object
class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint.
    """

    original_state_dict: STATE_DICT_TYPE  # 原始状态字典的类型声明
    mappings: FLATTEN_MAPPING  # 扁平化映射的类型声明

    def __init__(
        self,
        flatten_state_dict: bool = True,  # 是否扁平化处理状态字典，默认为True
        flatten_sharded_tensors: bool = True,  # 是否处理分片张量，默认为True
        allow_partial_load: bool = False,  # 是否允许部分加载，默认为False
    ) -> None:
        self.flatten_state_dict = flatten_state_dict  # 初始化是否扁平化状态字典的选项
        self.flatten_sharded_tensors = flatten_sharded_tensors  # 初始化是否处理分片张量的选项
        self.original_state_dict = {}  # 初始化原始状态字典为空字典
        self.mappings = {}  # 初始化映射为空字典
        self.allow_partial_load = allow_partial_load  # 初始化是否允许部分加载的选项

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,  # 状态字典的类型声明
        metadata: Optional[Metadata] = None,  # 元数据的可选类型声明，默认为None
        is_coordinator: bool = False,  # 是否为协调器，默认为False
    ) -> None:
        _init_state_dict(state_dict)  # 初始化状态字典的辅助函数调用
        self.original_state_dict = state_dict  # 将状态字典赋值给原始状态字典

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)  # 如果需要处理分片张量，则调用处理函数

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)  # 如果需要扁平化处理状态字典，则调用处理函数并获取映射关系

        self.state_dict = state_dict  # 将处理后的状态字典赋值给类属性
        self.metadata = metadata  # 将元数据赋值给类属性
        self.is_coordinator = is_coordinator  # 将是否为协调器赋值给类属性

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None  # 断言元数据不为None
        return create_default_local_load_plan(
            self.state_dict, self.metadata, not self.allow_partial_load
        )  # 创建本地加载计划

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_load_plan(global_plan)  # 创建全局加载计划

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan  # 完成加载计划并返回新计划

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if self.flatten_state_dict:
            set_element(
                self.original_state_dict,
                self.mappings[read_item.dest_index.fqn],
                torch.load(value),
            )  # 如果需要扁平化处理状态字典，则设置元素到原始状态字典
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(value)  # 否则直接加载字节流到状态字典

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)  # 查找张量
        return self.transform_tensor(read_item, tensor)  # 转换张量并返回结果

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass  # 提交张量，此处暂未实现具体功能

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)  # 查找状态字典对象并返回张量
    # 定义一个方法transform_tensor，`
# 定义一个方法来转换张量数据
def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
    """Extension from the planner interface to make it easy to extend the default planner."""
    # 调用一个函数来根据给定的索引信息对张量进行裁e default planner."""
        # 调用narrow_tensor_by_index函数，根据read_item中的偏移量和长度对tensor进行裁剪
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)
class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

    """

    def __init__(self, keys=None, *args, **kwargs):
        self.keys = keys  # 初始化时接收的键列表，用于选择性地加载状态字典的部分内容
        super().__init__(*args, **kwargs)

    def _should_include_key(self, key: str, metadata: Metadata) -> bool:
        if self.keys is None:
            return True

        if key in self.keys:
            True  # 如果键在指定的 keys 列表中，则返回 True

        unflattened_keys: List[str] = []
        planner_data = metadata.planner_data.get(key)
        for unflattened_key in planner_data:
            if unflattened_keys:
                unflattened_keys.append(
                    ".".join([unflattened_keys[-1], str(unflattened_key)])
                )
            else:
                unflattened_keys.append(unflattened_key)

        if any(unflattened_key in self.keys for unflattened_key in unflattened_keys):
            return True  # 如果任何一个展开后的键在 keys 列表中，则返回 True

        return False  # 否则返回 False

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        assert not state_dict  # 确保传入的 state_dict 是空字典，否则会抛出 AssertionError
        assert metadata is not None  # 确保传入的 metadata 不为 None，否则会抛出 AssertionError

        # 从 metadata 中重建 state_dict
        for k, v in metadata.state_dict_metadata.items():
            if not self._should_include_key(k, metadata):
                continue  # 如果不应该包含该键，则跳过

            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # 创建一个空的 Tensor 以替代原有的 v
            if k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)  # 根据 planner_data 设置元素
            else:
                state_dict[k] = v  # 直接设置元素到 state_dict 中

        super().set_up_planner(state_dict, metadata, is_coordinator)


def create_default_local_load_plan(
    state_dict: Dict[str, Any], metadata: Metadata, strict: bool = True
) -> LoadPlan:
    requests = []
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

    The default behavior is to match key exactly between state_dict and metadata.
    It handles resharding by issuing multiple read requests against storage in order to match
    load requirements.
    """
    # 对于状态字典中的每个键值对进行迭代
    for fqn, obj in state_dict.items():
        # 如果 strict=True 并且 fqn 不在 metadata.state_dict_metadata 中，则抛出错误
        # 如果 strict=False 并且 fqn 不在 metadata.state_dict_metadata 中，则跳过当前循环
        if fqn not in metadata.state_dict_metadata:
            if strict:
                raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
            else:
                continue
        
        # 获取当前键值对对应的元数据
        md = metadata.state_dict_metadata[fqn]
        
        # 检查 obj 是否为 DTensor 类型
        if isinstance(obj, DTensor):
            # 如果 obj 的设备网格坐标不为 None，则调用 _create_read_items() 来生成请求
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            # 对于非 DTensor 类型的对象，也调用 _create_read_items() 生成请求
            requests += _create_read_items(fqn, md, obj)

    # 返回一个包含所有请求的 LoadPlan 对象
    return LoadPlan(requests)
# 创建默认的全局加载计划，由 DefaultLoadPlanner 使用
def create_default_global_load_plan(
    all_plans: List[LoadPlan],
) -> List[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.
    """
    # 直接返回传入的所有加载计划列表
    return all_plans


# 创建 DefaultSavePlanner 使用的默认保存计划 SavePlan
def create_default_local_save_plan(
    state_dict: Dict[str, Any], is_coordinator: bool
) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.
    """
    # 初始化空的请求列表
    requests = []
    # 遍历状态字典中的每个键值对
    for fqn, obj in state_dict.items():
        # 如果对象是 DTensor 类型
        if isinstance(obj, DTensor):
            # 检查当前设备网格坐标是否存在，如果存在则添加写入请求
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        else:
            # 对于普通张量和非张量值，添加请求给所有的秩
            requests += _create_write_items(fqn, obj)

    # 返回保存计划 SavePlan 对象，包含所有的写入请求
    return SavePlan(requests)


# 创建 DefaultSavePlanner 使用的默认全局保存计划 SavePlan 和元数据 Metadata
def create_default_global_save_plan(
    all_plans: List[SavePlan],
    rewrite_index_hints: bool = True,
) -> Tuple[List[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects if
    ``rewrite_index_hints`` is True.
    """
    # 初始化元数据字典
    md: Dict[str, STORAGE_TYPES] = {}
    # 初始化新的计划列表
    new_plans = []
    # 返回更新后的计划列表和元数据对象元组
    return new_plans, Metadata(md)
    # 遍历所有计划列表中的每个计划
    for plan in all_plans:
        # 初始化一个空列表，用于存储更新后的项目
        new_items = []
        # 遍历当前计划中的每个项目
        for item in plan.items:
            # 如果项目类型不是 SHARD，则断言其索引在元数据字典中不存在
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md

            # 如果项目类型是 BYTE_IO
            if item.type == WriteItemType.BYTE_IO:
                # 将项目的完全限定名称作为键，关联一个新的 BytesStorageMetadata 对象
                md[item.index.fqn] = BytesStorageMetadata()
                # 将该项目添加到新项目列表中
                new_items.append(item)
            else:
                # 断言张量数据不为空
                assert item.tensor_data is not None
                # 如果元数据字典中不存在该项目的完全限定名称，则设置默认值为一个新的 TensorStorageMetadata 对象
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(
                        item.index.fqn,
                        TensorStorageMetadata(
                            properties=item.tensor_data.properties,
                            size=item.tensor_data.size,
                            chunks=[],
                        ),
                    ),
                )
                # 创建一个新项目，初步等于当前项目
                new_item = item
                # 如果需要重写索引提示
                if rewrite_index_hints:
                    # 创建一个新索引，用于更新当前项目的索引长度
                    new_index = dataclasses.replace(
                        item.index, index=len(tensor_md.chunks)
                    )
                    # 使用新索引替换当前项目的索引
                    new_item = dataclasses.replace(item, index=new_index)
                # 将更新后的项目添加到新项目列表中
                new_items.append(new_item)

                # 断言张量数据的块不为空
                assert (
                    item.tensor_data.chunk is not None
                ), f"""
                    Cannot create MD for tensor without bounds.
                    FQN: {item.index.fqn}
                """
                # 将张量数据的块添加到张量元数据对象的块列表中
                tensor_md.chunks.append(item.tensor_data.chunk)
        # 使用更新后的项目列表替换当前计划中的项目列表，并将更新后的计划添加到新计划列表中
        new_plans.append(dataclasses.replace(plan, items=new_items))
    # 返回包含新计划列表和元数据对象的元组
    return (new_plans, Metadata(md))
# 返回``Metadata``对象，如果``DefaultSavePlanner``用于检查点``state_dict``
def _create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    # 创建仅包含元数据的默认计划
    plan = _create_default_metadata_only_plan(state_dict)
    # 创建全局默认保存计划，并获取元数据
    _, md = create_default_global_save_plan([plan])
    # 返回元数据对象
    return md


# 检查两个框是否重叠。元组是（偏移量，长度）
def _check_box_overlap(box0: ChunkStorageMetadata, box1: ChunkStorageMetadata) -> bool:
    # 对于每个维度的每个分片，检查一个分片是否在另一个分片的另一端
    # 例如，对于一个2D分片，我们会检查一个分片是否在另一个分片的上方或左侧
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False

    return True


# 检查内部框是否在外部框的边界内
def _check_box_bounds(
    outer_box_size: torch.Size, inner_box: ChunkStorageMetadata
) -> bool:
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True


# 验证全局计划是否有效
def _validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    all_good = True
    # 遍历每个状态字典的元数据
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        # 遍历每个块
        for chunk_idx, chunk0 in enumerate(value.chunks):
            # 计算体积
            if not _check_box_bounds(value.size, chunk0):
                # 记录警告日志，指出边界框超出范围
                logger.warning(
                    """
                        key:%s has out of bounds chunk:
                        tensor-size:%s chunk: %s
                    """,
                    key,
                    value.size,
                    chunk0,
                )
                all_good = False
            chunks_volume += reduce(operator.mul, chunk0.sizes, 1)

            # 检查重叠
            for chunk1 in value.chunks[chunk_idx + 1:]:
                if _check_box_overlap(chunk0, chunk1):
                    # 记录警告日志，指出重叠的块
                    logger.warning(
                        "key:%s has overlapping chunks: %s %s", key, chunk0, chunk1
                    )
                    all_good = False

        # 检查组合块是否覆盖整个张量
        tensor_volume = reduce(operator.mul, value.size, 1)
        if chunks_volume != tensor_volume:
            # 记录警告日志，指出填充不正确的张量体积
            logger.warning(
                """
                    key:%s invalid fill tensor-volume:
                    %s chunks-volume: %s
                """,
                key,
                tensor_volume,
                chunks_volume,
            )
            all_good = False

    return all_good
```