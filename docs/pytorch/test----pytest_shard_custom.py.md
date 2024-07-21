# `.\pytorch\test\pytest_shard_custom.py`

```py
"""
Custom pytest shard plugin
https://github.com/AdamGleave/pytest-shard/blob/64610a08dac6b0511b6d51cf895d0e1040d162ad/pytest_shard/pytest_shard.py#L1
Modifications:
* shards are now 1 indexed instead of 0 indexed
* option for printing items in shard
"""
# 导入 hashlib 库，用于生成哈希值
import hashlib

# 导入 _pytest.config.argparsing 模块中的 Parser 类
from _pytest.config.argparsing import Parser


def pytest_addoptions(parser: Parser):
    """Add options to control sharding."""
    # 添加一个名为 "shard" 的选项组
    group = parser.getgroup("shard")
    # 在选项组中添加一个参数选项，用于指定此分片的编号，默认为 1
    group.addoption(
        "--shard-id", dest="shard_id", type=int, default=1, help="Number of this shard."
    )
    # 在选项组中添加一个参数选项，用于指定分片的总数，默认为 1
    group.addoption(
        "--num-shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    # 在选项组中添加一个参数选项，用于控制是否打印此分片中正在测试的项目
    group.addoption(
        "--print-items",
        dest="print_items",
        action="store_true",
        default=False,
        help="Print out the items being tested in this shard.",
    )


class PytestShardPlugin:
    def __init__(self, config):
        self.config = config

    def pytest_report_collectionfinish(self, config, items) -> str:
        """Log how many and which items are tested in this shard."""
        # 构造日志消息，显示在此分片中测试的项目数目
        msg = f"Running {len(items)} items in this shard"
        # 如果设置了打印选项，则追加显示正在测试的项目的节点标识符
        if config.getoption("print_items"):
            msg += ": " + ", ".join([item.nodeid for item in items])
        return msg

    def sha256hash(self, x: str) -> int:
        """Calculate SHA-256 hash of input string `x`."""
        # 计算输入字符串 `x` 的 SHA-256 哈希值，并转换为整数返回
        return int.from_bytes(hashlib.sha256(x.encode()).digest(), "little")

    def filter_items_by_shard(self, items, shard_id: int, num_shards: int):
        """Computes `items` that should be tested in `shard_id` out of `num_shards` total shards."""
        # 根据分片编号 `shard_id` 和总分片数 `num_shards`，筛选出应该在当前分片中测试的项目列表
        new_items = [
            item
            for item in items
            if self.sha256hash(item.nodeid) % num_shards == shard_id - 1
        ]
        return new_items

    def pytest_collection_modifyitems(self, config, items):
        """Mutate the collection to consist of just items to be tested in this shard."""
        # 获取当前分片的 ID 和总分片数
        shard_id = config.getoption("shard_id")
        shard_total = config.getoption("num_shards")
        # 如果分片 ID 小于 1 或大于总分片数，抛出值错误异常
        if shard_id < 1 or shard_id > shard_total:
            raise ValueError(
                f"{shard_id} is not a valid shard ID out of {shard_total} total shards"
            )

        # 修改 `items` 列表，仅保留应该在当前分片中测试的项目
        items[:] = self.filter_items_by_shard(items, shard_id, shard_total)
```