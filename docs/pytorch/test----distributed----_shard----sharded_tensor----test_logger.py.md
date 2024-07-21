# `.\pytorch\test\distributed\_shard\sharded_tensor\test_logger.py`

```
# Owner(s): ["oncall: distributed"]

# 导入日志模块
import logging

# 导入获取或创建日志记录器的函数
from torch.distributed._shard.sharded_tensor.logger import _get_or_create_logger
# 导入测试相关的工具函数和类
from torch.testing._internal.common_utils import run_tests, TestCase


class ShardingSpecLoggerTest(TestCase):
    def test_get_or_create_logger(self):
        # 调用获取或创建日志记录器的函数
        logger = _get_or_create_logger()
        # 断言日志记录器不为None
        self.assertIsNotNone(logger)
        # 断言日志记录器的处理器数量为1
        self.assertEqual(1, len(logger.handlers))
        # 断言日志记录器的第一个处理器是 NullHandler 类型的实例
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)


if __name__ == "__main__":
    # 运行测试用例
    run_tests()
```