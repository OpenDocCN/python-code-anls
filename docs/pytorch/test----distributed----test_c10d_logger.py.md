# `.\pytorch\test\distributed\test_c10d_logger.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块
import json
import logging
import os
import re
import sys
import time
from functools import partial, wraps

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist

# 导入分布式日志记录器
from torch.distributed.c10d_logger import _c10d_logger, _exception_logger, _time_logger

# 如果分布式环境不可用，则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试所需的其他模块和变量
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 如果开启了开发者调试的地址安全（asan），则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 使用 NCCL 作为后端
BACKEND = dist.Backend.NCCL
# 计算世界大小，最小为2，最大为4与当前 CUDA 设备数量之间的较小值
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))


# 装饰器函数，用于设置通信环境
def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 如果使用 NCCL 后端且 CUDA 设备数量小于世界大小，则退出测试
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        
        # 初始化分布式环境
        self.dist_init()
        # 执行测试函数
        func(self)
        # 销毁通信环境
        self.destroy_comms()

    return wrapper


# 测试类，继承自 MultiProcessTestCase
class C10dErrorLoggerTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # 设置环境变量 WORLD_SIZE 和 BACKEND
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        # 启动多进程测试
        self._spawn_processes()

    @property
    def device(self):
        # 根据不同的后端返回设备类型
        return (
            torch.device(self.rank)
            if BACKEND == dist.Backend.NCCL
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        # 返回世界大小
        return WORLD_SIZE

    @property
    def process_group(self):
        # 返回分布式组
        return dist.group.WORLD

    def destroy_comms(self):
        # 等待所有进程到达此处后再开始销毁进程组
        dist.barrier()
        dist.destroy_process_group()

    def dist_init(self):
        # 初始化进程组
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # 如果使用的是 NCCL 后端，设置当前设备用于集体通信
        if BACKEND == "nccl":
            torch.cuda.set_device(self.rank)

    def test_get_or_create_logger(self):
        # 测试获取或创建日志记录器
        self.assertIsNotNone(_c10d_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))
        self.assertIsInstance(_c10d_logger.handlers[0], logging.NullHandler)

    @_exception_logger
    def _failed_broadcast_raise_exception(self):
        # 测试广播时抛出异常的情况
        tensor = torch.arange(2, dtype=torch.int64)
        dist.broadcast(tensor, self.world_size + 1)

    @_exception_logger
    def _failed_broadcast_not_raise_exception(self):
        # 测试广播时不抛出异常的情况
        try:
            tensor = torch.arange(2, dtype=torch.int64)
            dist.broadcast(tensor, self.world_size + 1)
        except Exception:
            pass

    @with_comms
    # 定义一个测试方法，用于测试异常日志记录
    def test_exception_logger(self) -> None:
        # 断言捕获异常
        with self.assertRaises(Exception):
            self._failed_broadcast_raise_exception()

        # 使用 assertLogs 捕获 _c10d_logger 的 DEBUG 级别日志
        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            # 调用方法，不应抛出异常
            self._failed_broadcast_not_raise_exception()
            # 从捕获的日志中提取错误消息 JSON 字典部分并转换为字典对象
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )

            # 断言错误消息字典长度为 10
            self.assertEqual(len(error_msg_dict), 10)

            # 断言错误消息字典中包含特定的键值对
            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("args", error_msg_dict.keys())

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual("nccl", error_msg_dict["backend"])

            # 获取当前环境的 NCCL 版本并与错误消息中的版本进行比较
            self.assertIn("nccl_version", error_msg_dict.keys())
            nccl_ver = torch.cuda.nccl.version()
            self.assertEqual(
                ".".join(str(v) for v in nccl_ver), error_msg_dict["nccl_version"]
            )

            # 在这个测试案例中，group_size = world_size，因为在一个节点上没有多个进程
            self.assertIn("group_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["group_size"])

            self.assertIn("world_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["world_size"])

            # 断言全局排名在错误消息字典中
            self.assertIn("global_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["global_rank"])

            # 在这个测试案例中，local_rank = global_rank，因为在一个节点上没有多个进程
            self.assertIn("local_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["local_rank"])

    # 使用装饰器 @_time_logger 记录方法执行时间
    @_time_logger
    def _dummy_sleep(self):
        # 睡眠 5 秒钟
        time.sleep(5)

    # 使用装饰器 @with_comms
    @with_comms
    # 定义一个测试方法，用于测试时间记录器功能，不返回任何内容
    def test_time_logger(self) -> None:
        # 使用 assertLogs 上下文管理器捕获 _c10d_logger 的 DEBUG 级别日志
        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            # 调用一个虚拟的睡眠方法
            self._dummy_sleep()
            # 从捕获的日志输出中找到包含 JSON 数据的字符串，并将其解析为字典
            msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )
            # 断言消息字典的长度为 10
            self.assertEqual(len(msg_dict), 10)

            # 断言消息字典中包含键 'pg_name'，并且其值为 'None'
            self.assertIn("pg_name", msg_dict.keys())
            self.assertEqual("None", msg_dict["pg_name"])

            # 断言消息字典中包含键 'func_name'，并且其值为 '_dummy_sleep'
            self.assertIn("func_name", msg_dict.keys())
            self.assertEqual("_dummy_sleep", msg_dict["func_name"])

            # 断言消息字典中包含键 'args'
            self.assertIn("args", msg_dict.keys())

            # 断言消息字典中包含键 'backend'，并且其值为 'nccl'
            self.assertIn("backend", msg_dict.keys())
            self.assertEqual("nccl", msg_dict["backend"])

            # 断言消息字典中包含键 'nccl_version'，并且其值与当前 CUDA NCCL 版本匹配
            self.assertIn("nccl_version", msg_dict.keys())
            nccl_ver = torch.cuda.nccl.version()
            self.assertEqual(
                ".".join(str(v) for v in nccl_ver), msg_dict["nccl_version"]
            )

            # 在这个测试用例中，group_size = world_size，因为在一个节点上没有多个进程
            # 断言消息字典中包含键 'group_size'，并且其值为当前的 world_size 字符串表示
            self.assertIn("group_size", msg_dict.keys())
            self.assertEqual(str(self.world_size), msg_dict["group_size"])

            # 断言消息字典中包含键 'world_size'，并且其值为当前的 world_size 字符串表示
            self.assertIn("world_size", msg_dict.keys())
            self.assertEqual(str(self.world_size), msg_dict["world_size"])

            # 断言消息字典中包含键 'global_rank'，并且其值为当前进程的分布式全局排名字符串表示
            self.assertIn("global_rank", msg_dict.keys())
            self.assertIn(str(dist.get_rank()), msg_dict["global_rank"])

            # 在这个测试用例中，local_rank = global_rank，因为在一个节点上没有多个进程
            # 断言消息字典中包含键 'local_rank'，并且其值为当前进程的分布式全局排名字符串表示
            self.assertIn("local_rank", msg_dict.keys())
            self.assertIn(str(dist.get_rank()), msg_dict["local_rank"])

            # 断言消息字典中包含键 'time_spent'，并且其值表示的时间花费应为 5 秒
            self.assertIn("time_spent", msg_dict.keys())
            time_ns = re.findall(r"\d+", msg_dict["time_spent"])[0]
            self.assertEqual(5, int(float(time_ns) / pow(10, 9)))
# 如果当前脚本作为主程序运行，执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```