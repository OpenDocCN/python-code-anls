# `.\pytorch\test\backends\xeon\test_launch.py`

```
# Owner(s): ["module: intel"]

# 导入所需的模块
import shutil  # 提供高级文件操作功能
import subprocess  # 允许创建和管理子进程
import tempfile  # 提供临时文件和目录的支持
import unittest  # 提供单元测试框架

# 导入自定义模块和函数
from torch.testing._internal.common_utils import IS_LINUX, run_tests, TestCase

# 如果不在Linux环境下，跳过测试
@unittest.skipIf(not IS_LINUX, "Only works on linux")
class TestTorchrun(TestCase):
    # 设置测试前的准备工作，创建临时目录
    def setUp(self):
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)

    # 清理测试后的工作，删除临时目录
    def tearDown(self):
        shutil.rmtree(self._test_dir)

    # 测试CPU信息获取函数
    def test_cpu_info(self):
        # 模拟的lscpu信息字符串
        lscpu_info = """# The following is the parsable format, which can be fed to other
# programs. Each different item in every column has an unique ID
# starting from zero.
# CPU,Core,Socket,Node
0,0,0,0
1,1,0,0
2,2,0,0
3,3,0,0
4,4,1,1
5,5,1,1
6,6,1,1
7,7,1,1
8,0,0,0
9,1,0,0
10,2,0,0
11,3,0,0
12,4,1,1
13,5,1,1
14,6,1,1
15,7,1,1
"""

        # 导入CPU信息处理函数
        from torch.backends.xeon.run_cpu import _CPUinfo

        # 创建CPU信息对象
        cpuinfo = _CPUinfo(lscpu_info)

        # 断言检查各个方法的返回结果是否符合预期
        assert cpuinfo._physical_core_nums() == 8
        assert cpuinfo._logical_core_nums() == 16
        assert cpuinfo.get_node_physical_cores(0) == [0, 1, 2, 3]
        assert cpuinfo.get_node_physical_cores(1) == [4, 5, 6, 7]
        assert cpuinfo.get_node_logical_cores(0) == [0, 1, 2, 3, 8, 9, 10, 11]
        assert cpuinfo.get_node_logical_cores(1) == [4, 5, 6, 7, 12, 13, 14, 15]
        assert cpuinfo.get_all_physical_cores() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert cpuinfo.get_all_logical_cores() == [
            0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
        ]
        assert cpuinfo.numa_aware_check([0, 1, 2, 3]) == [0]
        assert cpuinfo.numa_aware_check([4, 5, 6, 7]) == [1]
        assert cpuinfo.numa_aware_check([2, 3, 4, 5]) == [0, 1]

    # 测试多线程功能
    def test_multi_threads(self):
        num = 0
        # 使用subprocess启动多个实例，测试CPU推理的并行性
        with subprocess.Popen(
            f"python -m torch.backends.xeon.run_cpu --ninstances 4 --use-default-allocator \
            --disable-iomp --disable-numactl --disable-taskset --log-path {self._test_dir} --no-python pwd",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            # 读取子进程的输出，并进行处理
            for line in p.stdout.readlines():
                # 将输出行按照"-"分割，提取关键信息
                segs = str(line, "utf-8").strip().split("-")
                # 如果最后一段是"pwd"，则计数器加一
                if segs[-1].strip() == "pwd":
                    num += 1
        # 断言检查实际启动的实例数量是否为4
        assert num == 4, "Failed to launch multiple instances for inference"

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```