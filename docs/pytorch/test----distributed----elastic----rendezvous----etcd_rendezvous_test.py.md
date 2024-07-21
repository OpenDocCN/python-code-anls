# `.\pytorch\test\distributed\elastic\rendezvous\etcd_rendezvous_test.py`

```
`
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# 导入所需的标准库和第三方模块
import os  # 导入操作系统功能模块
import sys  # 导入系统相关的模块
import unittest  # 导入单元测试框架
import uuid  # 导入用于生成唯一标识符的模块

# 导入与分布式任务调度和协调相关的模块
from torch.distributed.elastic.rendezvous import RendezvousParameters  # 导入任务协调参数类
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler  # 导入创建协调处理程序函数
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer  # 导入Etcd服务器类

# 如果环境变量CIRCLECI被设置，则输出相应信息并退出程序
if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)

# 定义一个用于测试EtcdRendezvous类的单元测试类
class EtcdRendezvousTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 在所有测试开始前，启动一个独立的Etcd服务器进程
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # 在所有测试结束后，停止Etcd服务器进程
        cls._etcd_server.stop()

    def test_etcd_rdzv_basic_params(self):
        """
        Check that we can create the handler with a minimum set of
        params
        """
        # 创建RendezvousParameters对象，设置基本的参数
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=f"{uuid.uuid4()}",
            min_nodes=1,
            max_nodes=1,
        )
        # 使用参数创建Etcd协调处理程序
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        # 断言确保创建的处理程序不为None
        self.assertIsNotNone(etcd_rdzv)

    def test_etcd_rdzv_additional_params(self):
        # 创建一个随机的run_id
        run_id = str(uuid.uuid4())
        # 创建RendezvousParameters对象，设置额外的参数
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )
        # 使用参数创建Etcd协调处理程序
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        # 断言确保创建的处理程序不为None
        self.assertIsNotNone(etcd_rdzv)
        # 断言确保获取的run_id与预期值一致
        self.assertEqual(run_id, etcd_rdzv.get_run_id())

    def test_get_backend(self):
        # 创建一个随机的run_id
        run_id = str(uuid.uuid4())
        # 创建RendezvousParameters对象，设置用于测试的参数
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )
        # 使用参数创建Etcd协调处理程序
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        # 断言确保获取的后端类型与预期值一致
        self.assertEqual("etcd", etcd_rdzv.get_backend())
```