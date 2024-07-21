# `.\pytorch\test\distributed\elastic\rendezvous\static_rendezvous_test.py`

```
# 导入单元测试模块
import unittest
# 导入上下文管理工具 closing
from contextlib import closing

# 从torch.distributed.elastic.rendezvous模块中导入RendezvousParameters类
# 以及static_tcp_rendezvous模块中的create_rdzv_handler函数
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.static_tcp_rendezvous import (
    create_rdzv_handler,
)
# 从torch.distributed.elastic.utils模块导入get_socket_with_port函数
from torch.distributed.elastic.utils import get_socket_with_port

# 定义StaticTCPRendezvousTest类，继承自unittest.TestCase
class StaticTCPRendezvousTest(unittest.TestCase):

    # 定义测试方法test_missing_port，验证缺少端口时的行为
    def test_missing_port(self):
        # 创建RendezvousParameters对象，指定后端为static，端点为localhost，运行ID为test_id
        # 最小节点数和最大节点数均为1
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="localhost",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        # 断言调用create_rdzv_handler(rdzv_params)时会引发ValueError异常
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    # 定义测试方法test_empty_endpoint，验证空端点时的行为
    def test_empty_endpoint(self):
        # 创建RendezvousParameters对象，指定后端为static，端点为空字符串，运行ID为test_id
        # 最小节点数和最大节点数均为1
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        # 断言调用create_rdzv_handler(rdzv_params)时会引发ValueError异常
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    # 定义测试方法test_ipv6_addr，验证IPv6地址时的行为
    def test_ipv6_addr(self):
        # 创建RendezvousParameters对象，指定后端为static，端点为IPv6地址和端口号，运行ID为test_id
        # 最小节点数和最大节点数均为1
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:90",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        # 断言调用create_rdzv_handler(rdzv_params)时会引发ValueError异常
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    # 定义测试方法test_ipv6_addr_localhost，验证IPv6本地主机地址时的行为
    def test_ipv6_addr_localhost(self):
        # 创建RendezvousParameters对象，指定后端为static，端点为IPv6本地主机地址和端口号，运行ID为test_id
        # 最小节点数和最大节点数均为1
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="[::1]:90",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        # 断言调用create_rdzv_handler(rdzv_params)时会引发ValueError异常
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    # 定义测试方法test_get_backend，验证获取后端类型时的行为
    def test_get_backend(self):
        # 创建RendezvousParameters对象，指定后端为static，端点为localhost:123，运行ID为test
        # 最小节点数和最大节点数均为1，超时时间为60秒，rank为0
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="localhost:123",
            run_id="test",
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            rank=0,
        )

        # 调用create_rdzv_handler(rdzv_params)创建静态Rendezvous对象
        static_rdzv = create_rdzv_handler(rdzv_params)
        # 断言static_rdzv的后端类型为"static"
        self.assertEqual("static", static_rdzv.get_backend())
    # 定义一个测试方法，用于测试静态分布式调度的多次调用情况
    def test_static_rdzv_multiple_calls(self):
        # 获取一个带有端口的套接字对象
        sock = get_socket_with_port()
        # 使用 'with' 语句确保在退出代码块时关闭套接字
        with closing(sock):
            # 获取套接字的本地端口号
            master_port = sock.getsockname()[1]
        # 设置主机地址为本地主机
        master_addr = "localhost"

        # 创建静态分布式调度参数对象
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint=f"{master_addr}:{master_port}",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
            rank=0,
        )
        # 根据参数创建分布式调度处理器对象
        rdzv_handler = create_rdzv_handler(rdzv_params)

        # 第一次调用分布式调度处理器的下一步方法
        rdzv_info = rdzv_handler.next_rendezvous()
        # 断言确保返回的调度信息存储不为空
        self.assertIsNotNone(rdzv_info.store)
        # 断言确保返回的调度信息中的进程编号为0
        self.assertEqual(0, rdzv_info.rank)
        # 断言确保返回的调度信息中的世界大小为1
        self.assertEqual(1, rdzv_info.world_size)

        # 第二次调用分布式调度处理器的下一步方法
        rdzv_info = rdzv_handler.next_rendezvous()
        # 断言确保返回的调度信息存储不为空
        self.assertIsNotNone(rdzv_info.store)
        # 断言确保返回的调度信息中的进程编号为0
        self.assertEqual(0, rdzv_info.rank)
        # 断言确保返回的调度信息中的世界大小为1
        self.assertEqual(1, rdzv_info.world_size)
```