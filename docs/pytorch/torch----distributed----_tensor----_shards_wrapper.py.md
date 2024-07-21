# `.\pytorch\torch\distributed\_tensor\_shards_wrapper.py`

```py
    # 必要的导入声明
    from typing import Any, List, Tuple

    # 导入 PyTorch 库
    import torch
    # 导入分布式检查点相关的元数据类
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        MetadataIndex,
        TensorProperties,
        TensorStorageMetadata,
    )
    # 导入分布式检查点的规划器类
    from torch.distributed.checkpoint.planner import (
        TensorWriteData,
        WriteItem,
        WriteItemType,
    )

    # 定义全局变量 aten，用于访问 PyTorch 的 aten 操作，忽略 Pyre 类型检查
    aten = (
        torch.ops.aten
    )  # pyre-ignore[5]: Globally accessible variable `aten` has no type specified.

    # 定义一个继承自 torch.Tensor 的本地分片包装类 LocalShardsWrapper
    class LocalShardsWrapper(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
        """
        A wrapper class to hold local shards of a DTensor.
        This class is used largely for checkpointing purposes and implicity subtypes
        the _Checkpointable protocol.
        """

        # 限定类属性，包括本地分片列表和张量存储元数据
        __slots__ = ["_local_shards", "_storage_meta"]
        _local_shards: List[torch.Tensor]
        _storage_meta: TensorStorageMetadata

        # 静态方法，用于创建 LocalShardsWrapper 实例
        @staticmethod
        def __new__(
            cls, local_shards: List[torch.Tensor], local_offsets: List[Tuple[int, ...]]
        ) -> "LocalShardsWrapper":
            assert len(local_shards) > 0
            assert len(local_shards) == len(local_offsets)
            assert all(
                tensor.device == local_shards[0].device for tensor in local_shards[1:]
            )

            # 计算总张量大小，针对第二个张量维度进行拼接
            cat_tensor_shape = list(local_shards[0].size())
            if len(local_shards) > 1:  # 如果有多个分片，按列进行分片
                for shard in local_shards[1:]:
                    cat_tensor_shape[1] += shard.size()[1]

            # 从第一个本地分片创建张量属性
            wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
            wrapper_shape = torch.Size(cat_tensor_shape)

            # 创建每个分片的存储元数据
            chunks_meta = [
                ChunkStorageMetadata(
                    offsets=torch.Size(offset),
                    sizes=shard.size(),
                )
                for shard, offset in zip(local_shards, local_offsets)
            ]

            # 使用 torch.Tensor._make_wrapper_subclass 方法创建子类实例
            r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                cls,
                torch.Size(cat_tensor_shape),
            )
            r._local_shards = local_shards
            r._storage_meta = TensorStorageMetadata(
                properties=wrapper_properties,
                size=wrapper_shape,
                chunks=chunks_meta,
            )

            return r

        # 必要的类方法，用于将操作分发给本地分片
        @classmethod
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
    # 定义一个静态方法，用于处理 Torch 分发时的特定函数调用
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        kwargs = kwargs or {}

        # 分发器字典，将函数映射到相应的处理方法
        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather_into_tensor,
            torch.ops._c10d_functional.wait_tensor.default: cls.handle_wait_tensor,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
            aten.detach.default: cls.handle_detach,
            aten.clone.default: cls.handle_clone,
        }

        # 如果 func 在分发器中，则调用相应的处理方法并返回结果
        if func in dispatcher:
            return dispatcher[func](
                args, kwargs
            )  # pyre-ignore [29] - `Variable[_VT]` is not a function.
        else:
            # 如果 func 不在分发器中，则抛出未实现错误
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    # 处理 all_gather_into_tensor 函数调用的静态方法
    def handle_all_gather_into_tensor(args, kwargs):
        # 获取维度信息
        dim = args[0].local_sizes()[0][1]
        # 将所有本地分片张量进行连接，并重新调整形状
        cat_tensor = torch.cat(
            [t.view(-1) for t in args[0].local_shards()], dim=0
        ).view(-1, dim)
        # 调用 all_gather_into_tensor 的默认实现
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            cat_tensor, *args[1:], **kwargs
        )

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    # 处理 wait_tensor 函数调用的静态方法
    def handle_wait_tensor(args, kwargs):
        # 调用 wait_tensor 的默认实现
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    # 处理 to_copy 函数调用的静态方法
    def handle_to_copy(args, kwargs):
        # 对每个本地分片调用 to_copy 函数，并获取结果列表
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        # 返回使用结果列表和本地偏移创建的 LocalShardsWrapper 对象
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    # 处理 view 函数调用的静态方法
    def handle_view(args, kwargs):
        # TODO, do we need to change the shape of associated offsets?
        # 对每个本地分片调用 view 函数，并获取结果列表
        res_shards_list = [
            aten.view.default(shard, args[1], **kwargs)
            for shard in args[0].local_shards()
        ]
        # 返回使用结果列表和本地偏移创建的 LocalShardsWrapper 对象
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_equal(args, kwargs):
        """
        判断两个LocalShardsWrapper对象是否相等，包括存储元数据和分片顺序
        """
        a, b = args[0], args[1]
        # 检查本地分片数量是否相同
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        # 检查所有对应分片是否相等
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        # 检查存储元数据是否相等
        if not a.storage_metadata() == b.storage_metadata():
            return False
        return True

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_detach(args, kwargs):
        """
        分离本地分片并禁用梯度追踪
        """
        self_ls = args[0]
        # 分离每个本地分片
        detached_local_shards = [
            aten.detach.default(shard) for shard in self_ls.local_shards()
        ]
        # 更新本地分片列表
        self_ls._local_shards = detached_local_shards
        # 禁用存储元数据的梯度追踪
        self_ls._storage_meta.properties.requires_grad = False
        return self_ls

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_clone(args, kwargs):
        """
        克隆LocalShardsWrapper对象，支持指定内存格式
        """
        self_ls = args[0]
        desired_memory_format = kwargs.get("memory_format", None)
        # 如果指定了内存格式但不是torch.preserve_format，则抛出异常
        if desired_memory_format and desired_memory_format != torch.preserve_format:
            raise NotImplementedError(
                f"{desired_memory_format} is not supported for LocalShardsWrapper!"
            )
        # 克隆每个本地分片并创建新的LocalShardsWrapper对象
        cloned_local_shards = [
            shard.clone(memory_format=desired_memory_format)
            for shard in self_ls._local_shards
        ]
        return LocalShardsWrapper(cloned_local_shards, self_ls.local_offsets())

    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        """
        返回第一个本地分片的设备
        """
        return self._local_shards[0].device

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        """
        返回第一个本地分片的是否为元数据属性
        """
        return self._local_shards[0].is_meta

    # pyre-ignore[14]
    def is_pinned(self) -> bool:  # type: ignore[override]
        """
        返回存储元数据的内存是否被固定
        """
        return self._storage_meta.properties.pin_memory

    # pyre-ignore[14]
    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        """
        设置是否需要梯度追踪，并更新所有本地分片的梯度追踪状态
        """
        self._storage_meta.properties.requires_grad = requires_grad
        [shard.requires_grad_(requires_grad) for shard in self._local_shards]
        return self

    def local_shards(self) -> List[torch.Tensor]:
        """
        返回本地分片列表
        """
        return self._local_shards

    def local_sizes(self) -> List[torch.Size]:
        """
        返回本地分片大小列表
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]
    def local_offsets(self) -> List[torch.Size]:
        """
        返回一个由 `torch.Size` 对象组成的列表，表示当前进程中各分片的本地偏移量。
        如果当前进程没有托管任何该张量的分片，则返回空列表。
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> List[ChunkStorageMetadata]:
        """
        返回一个 `List[ChunkStorageMetadata]` 对象，包含每个张量分片的元数据。
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        """
        返回一个 `TensorStorageMetadata` 对象，包含当前进程中本地张量的元数据。
        """
        return self._storage_meta

    def __create_write_items__(
        self, fqn: str, object: Any
    ) -> List[WriteItem]:  # pyre-ignore[2]
        """
        为了兼容 DCP，支持创建 WriteItem 对象以便正确保存数据。
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=object.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __create_chunk_list__(self) -> List[ChunkStorageMetadata]:
        """
        为了兼容 DCP，支持创建 chunk 列表以便正确保存数据。
        """
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """
        为了兼容 DCP，支持根据索引查找分片。
        返回一个基于 MetadataIndex 的 `torch.Tensor` 分片。
        """
        # 快速查找路径
        if index.index is not None:
            if (
                len(self._local_shards) > index.index
                and self._storage_meta.chunks[index.index].offsets == index.offset
            ):
                return self._local_shards[index.index]

        # 若指定了偏移量，则遍历查找对应的分片
        if index.offset is not None:
            for shard, chunk in zip(self._local_shards, self._storage_meta.chunks):
                if chunk.offsets == index.offset:
                    return shard

        raise ValueError(
            f"无法找到 FQN: '{index.fqn}' 的偏移量为 '{index.offset}' 的分片。"
        )

    def _get_tensor_size_bytes(self) -> int:
        """
        计算本地分片所占用的总字节数。
        """
        object_size = 0
        for shard in self.local_shards():
            object_size += shard.nelement() * shard.element_size()
        return object_size
    # pyre-fixme[3]: Return type must be annotated.
    # 返回当前对象的哈希值，使用内置函数 `id()` 返回对象的唯一标识符作为哈希值
    def __hash__(self):
        return id(self)

    # pyre-fixme[14]: `__repr__` overrides method defined in `torch._tensor.Tensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # 返回对象的字符串表示形式，包含 `_local_shards` 和 `_storage_meta` 属性的信息
    def __repr__(self):
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    # 返回对象的字符串表示形式，应注明返回类型为 `str`
    def __str__(self) -> str:
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"
```