# `.\pytorch\test\distributed\elastic\rendezvous\etcd_server_test.py`

```py
# 导入所需的标准库和第三方库
import os
import sys
import unittest

# 导入 etcd 客户端库
import etcd

# 从 torch 分布式模块导入 etcd 相关的 rendezvous 类和处理器类
from torch.distributed.elastic.rendezvous.etcd_rendezvous import (
    EtcdRendezvous,
    EtcdRendezvousHandler,
)
# 从 torch 分布式模块导入 etcd 服务器类
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer

# 如果在环境变量 CIRCLECI 中设置了值，输出一条信息并退出程序
if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)

# 定义测试类 EtcdServerTest，继承自 unittest.TestCase
class EtcdServerTest(unittest.TestCase):

    # 测试 etcd 服务器的启动和停止
    def test_etcd_server_start_stop(self):
        # 创建 EtcdServer 实例
        server = EtcdServer()
        # 启动 etcd 服务器
        server.start()

        try:
            # 获取服务器的端口号和主机名
            port = server.get_port()
            host = server.get_host()

            # 断言端口号大于 0
            self.assertGreater(port, 0)
            # 断言主机名为 "localhost"
            self.assertEqual("localhost", host)
            # 断言服务器的完整终端点地址
            self.assertEqual(f"{host}:{port}", server.get_endpoint())
            # 断言获取到的 etcd 客户端的版本信息不为空
            self.assertIsNotNone(server.get_client().version)
        finally:
            # 无论如何都要停止 etcd 服务器
            server.stop()

    # 测试带有 rendezvous 的 etcd 服务器
    def test_etcd_server_with_rendezvous(self):
        # 创建 EtcdServer 实例
        server = EtcdServer()
        # 启动 etcd 服务器
        server.start()

        try:
            # 创建 etcd 客户端实例
            client = etcd.Client(server.get_host(), server.get_port())

            # 创建 EtcdRendezvous 实例
            rdzv = EtcdRendezvous(
                client=client,
                prefix="test",
                run_id=1,
                num_min_workers=1,
                num_max_workers=1,
                timeout=60,
                last_call_timeout=30,
            )
            # 创建 EtcdRendezvousHandler 实例
            rdzv_handler = EtcdRendezvousHandler(rdzv)
            # 获取下一个 rendezvous 的信息
            rdzv_info = rdzv_handler.next_rendezvous()

            # 断言 rendezvous 信息中的存储对象不为空
            self.assertIsNotNone(rdzv_info.store)
            # 断言 rank 应为 0
            self.assertEqual(0, rdzv_info.rank)
            # 断言 world size 应为 1
            self.assertEqual(1, rdzv_info.world_size)
        finally:
            # 无论如何都要停止 etcd 服务器
            server.stop()
```