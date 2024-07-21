# `.\pytorch\test\distributed\launcher\launch_test.py`

```
    #!/usr/bin/env python3
    # Owner(s): ["oncall: r2p"]

    # Copyright (c) Facebook, Inc. and its affiliates.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in the root directory of this source tree.
    import os  # 导入操作系统相关功能
    import shutil  # 导入文件和目录操作工具
    import tempfile  # 导入临时文件和目录创建工具
    import unittest  # 导入单元测试框架
    from contextlib import closing  # 导入上下文管理工具

    import torch.distributed.launch as launch  # 导入 PyTorch 分布式训练启动工具
    from torch.distributed.elastic.utils import get_socket_with_port  # 导入获取端口的工具函数
    from torch.testing._internal.common_utils import (
        skip_but_pass_in_sandcastle_if,  # 导入条件跳过测试的装饰器函数
        TEST_WITH_DEV_DBG_ASAN,  # 导入用于开发和调试的 ASAN 测试标志
    )


    def path(script):
        return os.path.join(os.path.dirname(__file__), script)  # 返回指定脚本在当前文件所在目录的路径


    class LaunchTest(unittest.TestCase):
        def setUp(self):
            self.test_dir = tempfile.mkdtemp()  # 创建临时目录用于测试
            # set a sentinel env var on the parent proc
            # this should be present on the child and gets
            # asserted in ``bin/test_script.py``
            os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"  # 设置环境变量，用于父进程到子进程的信息传递

        def tearDown(self):
            shutil.rmtree(self.test_dir)  # 清理临时目录及其内容

        @skip_but_pass_in_sandcastle_if(
            TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
        )
        def test_launch_without_env(self):
            nnodes = 1  # 分布式训练节点数
            nproc_per_node = 4  # 每个节点的进程数
            world_size = nnodes * nproc_per_node  # 总的进程数
            sock = get_socket_with_port()  # 获取一个空闲端口的套接字对象
            with closing(sock):
                master_port = sock.getsockname()[1]  # 获取套接字的端口号
            args = [
                f"--nnodes={nnodes}",  # 设置分布式训练节点数的命令行参数
                f"--nproc-per-node={nproc_per_node}",  # 设置每个节点的进程数的命令行参数
                "--monitor-interval=1",  # 设置监控间隔的命令行参数
                "--start-method=spawn",  # 设置启动方法的命令行参数
                "--master-addr=localhost",  # 设置主节点地址的命令行参数
                f"--master-port={master_port}",  # 设置主节点端口的命令行参数
                "--node-rank=0",  # 设置节点排名的命令行参数
                path("bin/test_script_local_rank.py"),  # 设置测试脚本路径的命令行参数
            ]
            launch.main(args)  # 调用分布式训练启动函数

        @skip_but_pass_in_sandcastle_if(
            TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
        )
        def test_launch_with_env(self):
            nnodes = 1  # 分布式训练节点数
            nproc_per_node = 4  # 每个节点的进程数
            world_size = nnodes * nproc_per_node  # 总的进程数
            sock = get_socket_with_port()  # 获取一个空闲端口的套接字对象
            with closing(sock):
                master_port = sock.getsockname()[1]  # 获取套接字的端口号
            args = [
                f"--nnodes={nnodes}",  # 设置分布式训练节点数的命令行参数
                f"--nproc-per-node={nproc_per_node}",  # 设置每个节点的进程数的命令行参数
                "--monitor-interval=1",  # 设置监控间隔的命令行参数
                "--start-method=spawn",  # 设置启动方法的命令行参数
                "--master-addr=localhost",  # 设置主节点地址的命令行参数
                f"--master-port={master_port}",  # 设置主节点端口的命令行参数
                "--node-rank=0",  # 设置节点排名的命令行参数
                "--use-env",  # 设置使用环境变量的命令行参数
                path("bin/test_script.py"),  # 设置测试脚本路径的命令行参数
                f"--touch-file-dir={self.test_dir}",  # 设置触摸文件目录的命令行参数
            ]
            launch.main(args)  # 调用分布式训练启动函数
            # make sure all the workers ran
            # each worker touches a file with its global rank as the name
            self.assertSetEqual(
                {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
            )  # 断言确保所有的工作进程都运行，并且每个工作进程在临时目录下创建了相应的文件
```