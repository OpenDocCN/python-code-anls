# `.\pytorch\test\test_mkldnn_verbose.py`

```
# Owner(s): ["module: unknown"]

# 从 torch.testing._internal.common_utils 导入 TestCase 和 run_tests
from torch.testing._internal.common_utils import TestCase, run_tests
# 导入操作系统相关模块
import os
# 导入子进程管理模块
import subprocess
# 导入系统相关模块
import sys

# 定义 TestMKLDNNVerbose 类，继承自 TestCase
class TestMKLDNNVerbose(TestCase):
    
    # 测试启用详细输出
    def test_verbose_on(self):
        # 初始化计数器
        num = 0
        # 获取当前文件所在目录的绝对路径
        loc = os.path.dirname(os.path.abspath(__file__))
        # 执行带有详细输出参数的子进程
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkldnn_verbose.py --verbose-level=1', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            # 逐行读取子进程标准输出
            for line in p.stdout.readlines():
                # 将字节转换为 UTF-8 字符串并去除首尾空白符
                line = str(line, 'utf-8').strip()
                # 如果行以 "onednn_verbose" 开头，则增加计数器
                if line.startswith("onednn_verbose"):
                    num = num + 1
                # 如果行是指定的禁用详细模式消息，则返回退出函数
                elif line == 'Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope.':
                    return
        # 断言确保至少找到一个 oneDNN 详细输出消息
        self.assertTrue(num > 0, 'oneDNN verbose messages not found.')

    # 测试禁用详细输出
    def test_verbose_off(self):
        # 初始化计数器
        num = 0
        # 获取当前文件所在目录的绝对路径
        loc = os.path.dirname(os.path.abspath(__file__))
        # 执行带有禁用详细输出参数的子进程
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkldnn_verbose.py --verbose-level=0', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            # 逐行读取子进程标准输出
            for line in p.stdout.readlines():
                # 将字节转换为 UTF-8 字符串并去除首尾空白符
                line = str(line, 'utf-8').strip()
                # 如果行以 "onednn_verbose" 开头，则增加计数器
                if line.startswith("onednn_verbose"):
                    num = num + 1
        # 断言确保没有找到意外的 oneDNN 详细输出消息
        self.assertEqual(num, 0, 'unexpected oneDNN verbose messages found.')

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    run_tests()
```