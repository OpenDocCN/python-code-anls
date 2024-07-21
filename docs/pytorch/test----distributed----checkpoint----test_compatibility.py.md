# `.\pytorch\test\distributed\checkpoint\test_compatibility.py`

```py
# 导入必要的模块和类
from unittest.mock import patch

# 导入PyTorch相关模块
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

# 定义测试类 TestDCPCompatbility，继承自 TestCase
class TestDCPCompatbility(TestCase):

    # 测试函数 test_metadata，用于测试元数据的行为
    def test_metadata(self) -> None:
        try:
            # 创建一个全零张量 tensor
            tensor = torch.zeros(4, 4)
            # 创建 ChunkStorageMetadata 对象 chunk_meta
            chunk_meta = ChunkStorageMetadata(
                torch.Size((1, 1)),
                torch.Size((1, 1)),
            )
            # 创建 TensorStorageMetadata 对象 tensor_meta
            tensor_meta = TensorStorageMetadata(
                properties=TensorProperties.create_from_tensor(tensor),
                size=tensor.size(),
                chunks=[chunk_meta],
            )
            # 创建 BytesStorageMetadata 对象 b_meta
            b_meta = BytesStorageMetadata()
            # 创建 Metadata 对象，使用 state_dict_metadata 参数初始化
            _ = Metadata(state_dict_metadata={"a": tensor_meta, "b": b_meta})

            # 创建 MetadataIndex 对象，指定 fqn 参数
            _ = MetadataIndex(fqn="a.b.c")
        except Exception as e:
            # 如果发生异常，抛出 RuntimeError 异常
            raise RuntimeError(
                "The change may break the BC of distributed checkpoint."
            ) from e

    # 测试函数 test_sharded_tensor_dependency，测试分片张量的依赖
    def test_sharded_tensor_dependency(self) -> None:
        # 动态导入 sharded_tensor.metadata 模块中的 TensorProperties 类
        from torch.distributed._shard.sharded_tensor.metadata import (
            TensorProperties as stp,
        )

        # 使用 patch 替换 TensorProperties，确保兼容性
        with patch("torch.distributed.checkpoint.metadata.TensorProperties", stp):
            # 使用 dcp.save 保存张量到文件系统
            dcp.save(
                {"a": torch.zeros(4, 4)},
                dcp.FileSystemWriter("/tmp/dcp_testing"),
            )

        # 使用 dcp.load 加载保存的张量数据
        dcp.load(
            {"a": torch.zeros(4, 4)},
            dcp.FileSystemReader("/tmp/dcp_testing"),
        )

    # 测试函数 test_storage_meta，测试存储元数据的行为
    @with_temp_dir  # 使用装饰器指定临时目录
    def test_storage_meta(self) -> None:
        # 创建 FileSystemWriter 对象 writer，指定临时目录
        writer = dcp.FileSystemWriter(self.temp_dir)
        # 使用 dcp.save 保存张量到文件系统
        dcp.save({"a": torch.zeros(4, 4)}, storage_writer=writer)

        # 创建 FileSystemReader 对象 reader，指定临时目录
        reader = dcp.FileSystemReader(self.temp_dir)
        # 读取元数据并获取 storage_meta 属性
        storage_meta = reader.read_metadata().storage_meta
        # 断言 storage_meta 不为 None
        self.assertNotEqual(storage_meta, None)
        # 断言 storage_meta.checkpoint_id 的字符串表示与临时目录一致
        self.assertEqual(str(storage_meta.checkpoint_id), self.temp_dir)
        # 断言 storage_meta.save_id 与 writer.save_id 一致
        self.assertEqual(storage_meta.save_id, writer.save_id)
        # 断言 storage_meta.load_id 与 reader.load_id 一致
        self.assertEqual(storage_meta.load_id, reader.load_id)

# 如果该脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```