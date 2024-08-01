# `.\DB-GPT-src\dbgpt\util\id_generator.py`

```py
from typing import Optional  # 导入类型提示模块 Optional，用于指定可选参数类型

from snowflake import Snowflake, SnowflakeGenerator  # 导入 Snowflake 和 SnowflakeGenerator 类

_GLOBAL_GENERATOR = SnowflakeGenerator(42)  # 创建全局的 SnowflakeGenerator 对象，初始实例化参数为 42


def initialize_id_generator(
    instance: int, *, seq: int = 0, epoch: int = 0, timestamp: Optional[int] = None
):
    """Initialize the global ID generator.
    
    Args:
        instance (int): 用于传统 Snowflake 算法的标识符，结合数据中心和机器 ID。这个单一的值用于在分布式环境中唯一标识 ID 生成请求的来源。
            在标准 Snowflake 中，这将分为 datacenter_id 和 worker_id，但在这里为简单起见合并为一个值。
        seq (int, optional): 生成器的初始序列号，默认为 0。序列号在同一毫秒内递增，允许快速连续生成多个 ID。当时间戳增加时会重置。
        epoch (int, optional): 生成器的起始时间（毫秒级 epoch 时间）。这个值通过设置自定义的“起始时间”来减少生成数字的长度。默认为 0。
        timestamp (int, optional): 生成器的初始时间戳（自 epoch 以来的毫秒数）。如果未提供，生成器将使用当前系统时间。可用于测试或需要固定开始时间的场景。
    """
    global _GLOBAL_GENERATOR
    _GLOBAL_GENERATOR = SnowflakeGenerator(
        instance, seq=seq, epoch=epoch, timestamp=timestamp
    )


def new_id() -> int:
    """Generate a new Snowflake ID.
    
    Returns:
        int: 一个新的 Snowflake ID。
    """
    return next(_GLOBAL_GENERATOR)


def parse(snowflake_id: int, epoch: int = 0) -> Snowflake:
    """Parse a Snowflake ID into its components.
    
    Example:
        .. code-block:: python

            from dbgpt.util.id_generator import parse, new_id

            snowflake_id = new_id()
            snowflake = parse(snowflake_id)
            print(snowflake.timestamp)
            print(snowflake.instance)
            print(snowflake.seq)
            print(snowflake.datetime)

    Args:
        snowflake_id (int): 要解析的 Snowflake ID。
        epoch (int, optional): 生成器的起始时间（毫秒级 epoch 时间）。

    Returns:
        Snowflake: 解析后的 Snowflake 对象。
    """
    return Snowflake.parse(snowflake_id, epoch=epoch)
```