# `.\pytorch\test\test_mkl_verbose.py`

```py
# 导入必要的模块和库
from torch.testing._internal.common_utils import TestCase, run_tests
import os  # 导入操作系统相关功能的模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关功能的模块

# 定义测试类 TestMKLVerbose，继承自 TestCase 类
class TestMKLVerbose(TestCase):

    # 测试 MKL verbose 开启的方法
    def test_verbose_on(self):
        num = 0  # 初始化计数器为 0
        loc = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
        # 执行带有 verbose 参数的 Python 脚本子进程
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkl_verbose.py --verbose-level=1', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            # 逐行读取子进程的标准输出
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()  # 将字节转换为 UTF-8 字符串，并去除首尾空白字符
                if line.startswith("MKL_VERBOSE"):
                    num = num + 1  # 统计以 "MKL_VERBOSE" 开头的行数
                elif line == 'Failed to set MKL into verbose mode. Please consider to disable this verbose scope.':
                    return  # 如果出现特定消息，则提前返回，不再继续统计
        self.assertTrue(num > 0, 'oneMKL verbose messages not found.')  # 断言确保至少找到一条 MKL verbose 消息

    # 测试 MKL verbose 关闭的方法
    def test_verbose_off(self):
        num = 0  # 初始化计数器为 0
        loc = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
        # 执行带有 verbose 参数为 0 的 Python 脚本子进程
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkl_verbose.py --verbose-level=0', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            # 逐行读取子进程的标准输出
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()  # 将字节转换为 UTF-8 字符串，并去除首尾空白字符
                if line.startswith("MKL_VERBOSE"):
                    num = num + 1  # 统计以 "MKL_VERBOSE" 开头的行数
        self.assertEqual(num, 0, 'unexpected oneMKL verbose messages found.')  # 断言确保没有找到任何 MKL verbose 消息

# 如果该脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    run_tests()
```