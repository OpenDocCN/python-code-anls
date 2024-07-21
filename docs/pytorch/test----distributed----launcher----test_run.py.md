# `.\pytorch\test\distributed\launcher\test_run.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io  # 导入io模块，提供了在Python中进行I/O操作的核心工具
import multiprocessing as mp  # 导入multiprocessing模块，用于实现多进程支持
import os  # 导入os模块，提供与操作系统交互的功能
import runpy  # 导入runpy模块，用于运行Python脚本文件
import shutil  # 导入shutil模块，提供了一些高级的文件操作功能
import subprocess  # 导入subprocess模块，用于创建新的进程，与它们进行交互
import sys  # 导入sys模块，提供了对Python解释器的访问和操作
import tempfile  # 导入tempfile模块，用于创建临时文件和目录
import uuid  # 导入uuid模块，用于生成UUID（通用唯一标识符）
from contextlib import closing, redirect_stderr, redirect_stdout  # 从contextlib模块导入关闭资源、重定向标准错误和标准输出的功能
from unittest import mock  # 导入mock模块，用于模拟测试中的对象和行为
from unittest.mock import MagicMock, Mock, patch  # 从unittest.mock导入用于模拟对象和修补功能的相关工具

import torch.distributed.run as launch  # 导入torch.distributed.run模块中的launch对象，用于分布式运行
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState  # 导入torch.distributed.elastic.agent.server.api模块中的RunResult和WorkerState对象
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs  # 导入torch.distributed.elastic.multiprocessing模块中的DefaultLogsSpecs对象
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError  # 导入torch.distributed.elastic.multiprocessing.errors模块中的ChildFailedError异常类
from torch.distributed.elastic.utils import get_socket_with_port  # 导入torch.distributed.elastic.utils模块中的get_socket_with_port函数
from torch.distributed.elastic.utils.distributed import get_free_port  # 导入torch.distributed.elastic.utils.distributed模块中的get_free_port函数
from torch.testing._internal.common_utils import (
    run_tests,  # 导入torch.testing._internal.common_utils模块中的run_tests函数，用于运行测试
    skip_but_pass_in_sandcastle_if,  # 导入torch.testing._internal.common_utils模块中的skip_but_pass_in_sandcastle_if函数，用于在Sandcastle环境下跳过测试但标记为通过
    TEST_WITH_DEV_DBG_ASAN,  # 导入torch.testing._internal.common_utils模块中的TEST_WITH_DEV_DBG_ASAN常量，用于调试
    TestCase,  # 导入unittest模块中的TestCase类，用于编写测试用例
)

def launch_in_proc(args):
    launch.main(args)  # 调用torch.distributed.run模块中的launch.main函数来执行传入的参数中指定的任务

def path(script):
    return os.path.join(os.path.dirname(__file__), script)  # 返回当前脚本文件所在目录与指定脚本文件名的路径

def get_child_pids(pid):
    pgrep = subprocess.Popen(args=f"pgrep -P {pid}", shell=True, stdout=subprocess.PIPE)  # 使用subprocess模块创建新进程来执行pgrep命令，查找指定父进程PID的子进程
    pgrep.wait()  # 等待pgrep命令执行完成
    out = pgrep.stdout.read().decode("utf-8").rstrip().split("\n")  # 读取pgrep命令的标准输出，解码为UTF-8字符串，去除末尾的换行符并按行分割
    pids = []
    for pid in out:
        if pid:
            pids.append(int(pid))  # 将找到的子进程PID转换为整数并添加到列表中
    return pids  # 返回找到的子进程PID列表

def pid_exists(pid):
    try:
        os.kill(pid, 0)  # 使用os.kill检查指定PID的进程是否存在
        return True  # 如果进程存在则返回True
    except OSError:
        return False  # 如果进程不存在则返回False

class MockException(Exception):
    pass  # 定义一个自定义异常类MockException，继承自Python内置的Exception类

class ElasticLaunchTest(TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()  # 创建一个临时目录，并将其路径存储在self.test_dir属性中

        # 移除任何悬挂的环境变量
        for env in os.environ.keys():
            if env.startswith("PET_"):
                del os.environ[env]  # 遍历并删除所有以"PET_"开头的环境变量

        # 在父进程上设置一个标志性环境变量
        # 该环境变量应该存在于子进程中，并在``bin/test_script.py``中进行断言
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"  # 设置环境变量"TEST_SENTINEL_PARENT"的值为"FOOBAR"

    def tearDown(self):
        shutil.rmtree(self.test_dir)  # 在测试结束后删除临时目录及其内容

    def test_launch_user_script_python(self):
        self._test_launch_user_script_python()  # 调用私有方法_test_launch_user_script_python()来执行Python用户脚本的启动测试
    # 定义一个用于测试启动用户脚本的方法（Python版本）
    def _test_launch_user_script_python(self):
        # 生成一个随机的运行ID
        run_id = str(uuid.uuid4().int)
        # 每个节点的数量
        nnodes = 1
        # 每个节点的处理器数量
        nproc_per_node = 4
        # 计算集群中的总进程数
        world_size = nnodes * nproc_per_node
        # 构建命令行参数列表
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            # 要运行的脚本路径
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        # 调用启动函数，传入参数列表
        launch.main(args)

        # 确保所有的工作节点都已运行
        # 每个工作节点会触碰一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    # 定义一个测试启动用户脚本的方法（Python Caffe2 BC版本）
    def test_launch_user_script_python_caffe2_bc(self):
        # 每个节点的数量
        nnodes = 1
        # 每个节点的处理器数量
        nproc_per_node = 4
        # 计算集群中的总进程数
        world_size = nnodes * nproc_per_node
        # 获取一个可用端口的socket对象
        sock = get_socket_with_port()
        with closing(sock):
            # 获取主端口号
            master_port = sock.getsockname()[1]
        # 构建命令行参数列表
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            # 要运行的脚本路径
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        # 调用启动函数，传入参数列表
        launch.main(args)

        # 确保所有的工作节点都已运行
        # 每个工作节点会触碰一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 定义一个测试启动用户脚本的方法（Bash版本）
    def test_launch_user_script_bash(self):
        # 生成一个随机的运行ID
        run_id = str(uuid.uuid4().int)
        # 每个节点的数量
        nnodes = 1
        # 每个节点的处理器数量
        nproc_per_node = 4
        # 计算集群中的总进程数
        world_size = nnodes * nproc_per_node
        # 构建命令行参数列表
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        # 要运行的Shell脚本路径及其参数
        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        # 确保在试图使用 --no-python 和 --module 时会抛出异常
        with self.assertRaises(ValueError):
            launch.main(args + ["--module"] + script_args)

        # 调用启动函数，传入参数列表
        launch.main(args + script_args)

        # 确保所有的工作节点都已运行
        # 每个工作节点会触碰一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )
    def test_launch_user_script_default_nproc(self):
        # 生成一个随机的运行 ID
        run_id = str(uuid.uuid4().int)
        # 设置节点数为 1
        nnodes = 1
        # 设置全局大小为 1
        world_size = 1
        # 构建命令行参数列表
        args = [
            f"--nnodes={nnodes}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]
        
        # 准备测试脚本的参数列表
        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]
        
        # 使用断言验证下面的操作会引发 ValueError 异常
        with self.assertRaises(ValueError):
            # --no-python 不能与 --module 同时使用
            launch.main(args + ["--module"] + script_args)

        # 执行命令，启动主程序
        launch.main(args + script_args)

        # 确保所有的工作进程都运行了
        # 每个工作进程会触碰一个文件，文件名为其全局等级
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_with_env_vars(self):
        # 生成一个随机的运行 ID
        run_id = str(uuid.uuid4().int)
        # 设置节点数为 1
        nnodes = 1
        # 每个节点上的进程数为 4
        nproc_per_node = 4
        # 计算全局大小
        world_size = nnodes * nproc_per_node

        # 设置环境变量
        os.environ["PET_NNODES"] = str(nnodes)
        os.environ["PET_NPROC_PER_NODE"] = str(nproc_per_node)
        os.environ["PET_RDZV_ID"] = run_id
        os.environ["PET_MONITOR_INTERVAL"] = "1"
        os.environ["PET_START_METHOD"] = "spawn"
        os.environ["PET_NO_PYTHON"] = "1"

        # 准备测试脚本的参数列表
        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        # 使用断言验证下面的操作会引发 ValueError 异常
        with self.assertRaises(ValueError):
            # --no-python 不能与 --module 同时使用
            os.environ["PET_MODULE"] = "1"
            launch.main(script_args)

        # 关闭 --module 选项
        os.environ["PET_MODULE"] = "0"
        # 执行命令，启动主程序
        launch.main(script_args)

        # 确保所有的工作进程都运行了
        # 每个工作进程会触碰一个文件，文件名为其全局等级
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def _test_nproc_launch_configuration(self, nproc_type, expected_number):
        # 生成一个随机的运行 ID
        run_id = str(uuid.uuid4().int)
        # 设置节点数为 1
        nnodes = 1

        # 构建命令行参数列表
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_type}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        # 准备测试脚本的参数列表
        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        # 执行命令，启动主程序
        launch.main(args + script_args)

        # 计算全局大小
        world_size = nnodes * expected_number
        # 确保所有的工作进程都运行了
        # 每个工作进程会触碰一个文件，文件名为其全局等级
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.cuda.is_available", return_value=False)
    def test_nproc_launch_auto_configurations(self, _mock1):
        # 测试以自动配置模式启动的处理器数量配置
        self._test_nproc_launch_configuration("auto", os.cpu_count())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_nproc_launch_number_configurations(self):
        # 测试以指定处理器数量启动的处理器配置
        self._test_nproc_launch_configuration("4", 4)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_nproc_launch_unknown_configurations(self):
        # 测试以未知配置启动时的异常处理
        with self.assertRaises(ValueError):
            self._test_nproc_launch_configuration("unknown", 4)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=3)
    def test_nproc_gpu_launch_configurations(self, _mock1, _mock2):
        # 测试以 GPU 加速配置启动的处理器数量配置
        self._test_nproc_launch_configuration("auto", 3)
        self._test_nproc_launch_configuration("gpu", 3)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic(self):
        # 弹性启动测试
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        # 我们只启动了1个节点（尽管最大为2）
        world_size = nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv-conf='join_timeout=5,last_call_timeout=1,timeout=5'",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # 确保所有的 worker 都运行了
        # 每个 worker 都会触发一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @mock.patch("torch.distributed.elastic.events.record")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        """
        Asserts that when the worker program fails and launcher raises exception
        to indicate that worker process failed.
        """
        # 生成一个随机的运行ID作为标识符
        run_id = str(uuid.uuid4().int)
        # 定义最小和最大节点数以及每个节点的处理器数量
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        # 构建命令行参数列表
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv-conf='join_timeout=5,last_call_timeout=1,timeout=5'",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--max-restarts=0",
            "--start-method=spawn",
            path("bin/test_script.py"),
            "--fail",
        ]
        # 断言在运行期间会抛出ChildFailedError异常
        with self.assertRaises(ChildFailedError):
            # 调用启动函数，传入参数列表
            launch.main(args)

        # 确认记录函数被调用了一次
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @mock.patch(
        "torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run"
    )
    @mock.patch("torch.distributed.elastic.events.record")
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception.
        """
        # 生成一个随机的运行ID作为标识符
        run_id = str(uuid.uuid4().int)
        # 定义最小和最大节点数以及每个节点的处理器数量
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        # 构建命令行参数列表
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv_conf=timeout=5",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--max-restarts=0",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]

        # 模拟当agent运行时抛出MockException异常
        mock_agent_run.side_effect = MockException
        # 断言在运行期间会抛出MockException异常
        with self.assertRaises(MockException):
            # 调用启动函数，传入参数列表
            launch.main(args)
        # 确认记录函数被调用了一次
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_standalone(self):
        nnodes = 1  # 定义节点数为1
        nproc_per_node = 4  # 每个节点的进程数为4
        world_size = nnodes * nproc_per_node  # 计算总的进程数
        args = [  # 定义命令行参数列表
            f"--nnodes={nnodes}",  # 设置节点数参数
            f"--nproc-per-node={nproc_per_node}",  # 设置每个节点的进程数参数
            "--standalone",  # 使用独立模式运行
            "--monitor-interval=1",  # 设置监控间隔为1秒
            "--start-method=spawn",  # 使用spawn方法启动
            path("bin/test_script.py"),  # 指定测试脚本的路径
            f"--touch-file-dir={self.test_dir}",  # 设置触摸文件的目录参数
        ]
        launch.main(args)  # 调用launch模块的主函数执行测试启动

        # 确保所有的工作进程都运行了
        # 每个工作进程都会触摸一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_run_path(self):
        nnodes = 1  # 定义节点数为1
        nproc_per_node = 4  # 每个节点的进程数为4
        world_size = nnodes * nproc_per_node  # 计算总的进程数
        args = [
            "--run-path",  # 指定运行路径
            f"--nnodes={nnodes}",  # 设置节点数参数
            f"--nproc-per-node={nproc_per_node}",  # 设置每个节点的进程数参数
            "--monitor-interval=1",  # 设置监控间隔为1秒
            "--start-method=spawn",  # 使用spawn方法启动
            path("bin/test_script.py"),  # 指定测试脚本的路径
            f"--touch-file-dir={self.test_dir}",  # 设置触摸文件的目录参数
        ]
        launch.main(args)  # 调用launch模块的主函数执行测试启动

        # 确保所有的工作进程都运行了
        # 每个工作进程都会触摸一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_multiple_agents(self):
        run_id = str(uuid.uuid4().int)  # 生成一个唯一的运行ID
        min_nodes = 1  # 定义最小节点数为1
        max_nodes = 2  # 定义最大节点数为2
        nproc_per_node = 4  # 每个节点的进程数为4
        nnodes = 2  # 定义节点数为2
        world_size = nnodes * nproc_per_node  # 计算总的进程数
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",  # 设置节点数范围参数
            f"--nproc-per-node={nproc_per_node}",  # 设置每个节点的进程数参数
            "--rdzv-backend=c10d",  # 指定后端通信方式为c10d
            f"--rdzv-endpoint=localhost:{get_free_port()}",  # 设置通信端点
            "--rdzv_conf=timeout=5",  # 设置通信配置，超时时间为5秒
            f"--rdzv-id={run_id}",  # 设置通信ID
            "--monitor-interval=1",  # 设置监控间隔为1秒
            "--start-method=spawn",  # 使用spawn方法启动
            path("bin/test_script.py"),  # 指定测试脚本的路径
            f"--touch-file-dir={self.test_dir}",  # 设置触摸文件的目录参数
        ]
        procs = []  # 创建进程列表
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[args])  # 创建进程对象
            procs.append(p)  # 将进程对象添加到列表
            p.start()  # 启动进程
        launch.main(args)  # 调用launch模块的主函数执行测试启动
        for i in range(nnodes - 1):
            p = procs[i]  # 获取已启动的进程对象
            p.join()  # 等待进程结束
            self.assertEqual(0, p.exitcode)  # 断言进程退出码为0

        # 确保所有的工作进程都运行了
        # 每个工作进程都会触摸一个以其全局排名命名的文件
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )
    # 定义测试函数，用于测试解析最小和最大节点数的功能
    def test_min_max_nodes_parse(self):
        # 解析输入字符串 "1"，返回最小节点数和最大节点数，两者应该相等
        min_nodes, max_nodes = launch.parse_min_max_nnodes("1")
        self.assertEqual(min_nodes, max_nodes)
        self.assertEqual(1, min_nodes)
        # 解析输入字符串 "2:20"，返回最小节点数和最大节点数
        min_nodes, max_nodes = launch.parse_min_max_nnodes("2:20")
        self.assertEqual(2, min_nodes)
        self.assertEqual(20, max_nodes)
        # 使用 assertRaises 检测解析非法输入 "2:20:30" 是否会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            launch.parse_min_max_nnodes("2:20:30")

    # 使用装饰器 patch 修饰，替换 torch.distributed.launcher.api.LocalElasticAgent 类
    def test_launch_shutdown(self, agent_mock_cls):
        # 定义测试参数
        nnodes = 1
        nproc_per_node = 4
        # 构建命令行参数列表
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        # 创建 Mock 对象并配置其行为
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        # 使用 patch 替换 rendezvous handler
        with patch(
            "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            # 调用 launch.main 函数执行测试
            launch.main(args)
            # 验证 rdzv_handler_mock.shutdown 被调用了一次
            rdzv_handler_mock.shutdown.assert_called_once()

    # 使用装饰器 skip_but_pass_in_sandcastle_if 过滤特定情况下的测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_is_torchelastic_launched(self):
        # 使用 torchelastic 启动测试脚本，并验证 torch.distributed.is_torchelastic_launched() 返回 True
        out_file = f"{os.path.join(self.test_dir, 'out')}"
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--monitor-interval=1",
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ]
        )
        # 打开输出文件，验证内容是否为 "True"
        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()
            self.assertEqual("True", is_torchelastic_launched)

    # 使用装饰器 patch 修饰，替换 torch.distributed.run.metadata
    @patch("torch.distributed.run.metadata")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_is_torchelastic_launched_with_logs_spec_defined(self, metadata_mock):
        # mock the entrypoint API to avoid version issues.
        entrypoints = MagicMock()
        metadata_mock.entry_points.return_value = entrypoints  # 设置 metadata_mock 的 entry_points 方法返回 MagicMock 对象

        group = MagicMock()
        entrypoints.select.return_value = group  # 设置 entrypoints 对象的 select 方法返回 MagicMock 对象

        ep = MagicMock()
        ep.load.return_value = DefaultLogsSpecs  # 设置 ep 对象的 load 方法返回 DefaultLogsSpecs

        group.select.return_value = ep  # 设置 group 对象的 select 方法返回 ep 对象
        group.__getitem__.return_value = ep  # 设置 group 对象的 __getitem__ 方法返回 ep 对象

        out_file = f"{os.path.join(self.test_dir, 'out')}"
        if os.path.exists(out_file):
            os.remove(out_file)  # 如果 out_file 存在，则删除它

        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--monitor-interval=1",
                "--logs_specs=default",
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ]
        )
        # 调用 launch.main 运行测试脚本，生成一个输出文件 out_file

        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()  # 读取 out_file 的第一行内容到 is_torchelastic_launched
            self.assertEqual("True", is_torchelastic_launched)  # 断言 is_torchelastic_launched 是否为 "True"

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_logs_logs_spec_entrypoint_must_be_defined(self):
        # 断言 logs_specs 为 DOESNOT_EXIST 时会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            launch.main(
                [
                    "--run-path",
                    "--nnodes=1",
                    "--nproc-per-node=1",
                    "--monitor-interval=1",
                    "--logs_specs=DOESNOT_EXIST",
                    path("bin/test_script_is_torchelastic_launched.py"),
                ]
            )

    def test_is_not_torchelastic_launched(self):
        # 启动测试脚本，验证 torch.distributed.is_torchelastic_launched() 返回 False

        out_file = f"{os.path.join(self.test_dir, 'out')}"
        # 需要在相同的解释器中使用 runpy 运行脚本，因为否则（根据环境）将无法找到 torch 作为依赖项
        with patch.object(
            sys,
            "argv",
            [
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ],
        ):
            runpy.run_path(sys.argv[0], run_name="__main__")  # 在当前解释器中运行指定的脚本文件
            with open(out_file) as fp:
                is_torchelastic_launched = fp.readline()  # 读取 out_file 的第一行内容到 is_torchelastic_launched
                self.assertEqual("False", is_torchelastic_launched)  # 断言 is_torchelastic_launched 是否为 "False"

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 定义测试方法，用于测试使用 torchelastic 的 TCP 初始化方法
    def test_init_method_tcp_with_torchelastic(self):
        # 获取一个空闲端口号
        port = get_free_port()
        # 调用 launch.main() 函数，启动测试脚本 init_method.py，并传入参数
        launch.main(
            [
                "--run-path",  # 指定运行路径
                "--nnodes=1",  # 设置节点数为1
                "--nproc-per-node=4",  # 每个节点的进程数为4
                "--master-addr=localhost",  # 指定主地址为本地主机
                f"--master-port={port}",  # 使用动态分配的端口号
                "--monitor-interval=1",  # 设置监控间隔为1秒
                path("bin/test_script_init_method.py"),  # 测试脚本的路径
                f"--init-method=tcp://localhost:{port}",  # 使用 TCP 协议进行初始化
            ]
        )
        # 没有特定的验证目的，仅确保脚本能够正常运行

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    # 定义测试方法，用于测试使用 torchelastic 的环境变量初始化方法
    def test_init_method_env_with_torchelastic(self):
        # 获取一个空闲端口号
        port = get_free_port()
        # 调用 launch.main() 函数，启动测试脚本 init_method.py，并传入参数
        launch.main(
            [
                "--run-path",  # 指定运行路径
                "--nnodes=1",  # 设置节点数为1
                "--nproc-per-node=4",  # 每个节点的进程数为4
                "--master-addr=localhost",  # 指定主地址为本地主机
                f"--master-port={port}",  # 使用动态分配的端口号
                "--monitor-interval=1",  # 设置监控间隔为1秒
                path("bin/test_script_init_method.py"),  # 测试脚本的路径
                "--init-method=env://",  # 使用环境变量进行初始化
            ]
        )
        # 没有特定的验证目的，仅确保脚本能够正常运行

    # 定义测试方法，用于测试使用默认日志规范捕获日志
    def test_capture_logs_using_default_logs_specs(self):
        # 生成一个运行 ID
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        # 设置命令行参数列表
        args = [
            f"--nnodes={nnodes}",  # 指定节点数
            f"--nproc-per-node={nproc_per_node}",  # 每个节点的进程数
            f"--rdzv-id={run_id}",  # 设置 rendezvous ID
            "--redirect=3",  # 重定向标准输出
            "--tee=3",  # 重定向标准错误和标准输出
            "--monitor-interval=1",  # 设置监控间隔为1秒
            "--start-method=spawn",  # 使用 spawn 启动方法
            "--no-python",  # 禁用 Python
        ]

        # 设置测试脚本的参数
        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        # 创建用于捕获标准输出和标准错误的流
        captured_out = io.StringIO()
        captured_err = io.StringIO()

        # 使用 patch.dict() 修改环境变量，指定 TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            with patch.dict(
                os.environ, {"TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": "[rank${rank}]: "}
            ):
                # 调用 launch.main() 函数，启动测试脚本，并传入参数
                launch.main(args + script_args)

        # 验证每个进程是否成功创建了日志行前缀
        for i in range(nproc_per_node):
            self.assertTrue(f"[rank{i}]: creating " in captured_out.getvalue())
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```