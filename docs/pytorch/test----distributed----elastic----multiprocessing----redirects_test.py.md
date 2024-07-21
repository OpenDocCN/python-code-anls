# `.\pytorch\test\distributed\elastic\multiprocessing\redirects_test.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# 导入必要的库和模块
import ctypes  # 用于访问 C 语言函数库
import os  # 与操作系统交互的标准库
import shutil  # 提供高级的文件和文件夹操作功能
import sys  # 提供对 Python 解释器的访问和控制
import tempfile  # 创建临时文件和目录的库
import unittest  # Python 的单元测试框架

# 导入需要测试的模块和函数
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect,
    redirect_stderr,
    redirect_stdout,
)

# 访问 libc.so.6 动态链接库，获取其 stderr 的文件描述符
libc = ctypes.CDLL("libc.so.6")
c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")

# 定义单元测试类 RedirectsTest，继承自 unittest.TestCase
class RedirectsTest(unittest.TestCase):

    # 在每个测试方法执行之前调用，设置测试环境
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    # 在每个测试方法执行之后调用，清理测试环境
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # 测试 redirect 函数处理无效标准流时是否抛出 ValueError 异常
    def test_redirect_invalid_std(self):
        with self.assertRaises(ValueError):
            with redirect("stdfoo", os.path.join(self.test_dir, "stdfoo.log")):
                pass

    # 测试 redirect_stdout 函数是否能正确重定向标准输出流
    def test_redirect_stdout(self):
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        # 在重定向之前，输出一些信息到标准输出，标准错误和命令行
        print("foo first from python")
        libc.printf(b"foo first from c\n")
        os.system("echo foo first from cmd")

        # 使用 redirect_stdout 函数重定向标准输出到 stdout_log 文件
        with redirect_stdout(stdout_log):
            print("foo from python")
            libc.printf(b"foo from c\n")
            os.system("echo foo from cmd")

        # 确保标准输出已经恢复正常
        print("foo again from python")
        libc.printf(b"foo again from c\n")
        os.system("echo foo again from cmd")

        # 检查 stdout_log 文件中的输出是否符合预期
        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"foo from python\n", "foo from c\n", "foo from cmd\n"}, lines
            )

    # 测试 redirect_stderr 函数是否能正确重定向标准错误流
    def test_redirect_stderr(self):
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        # 在重定向之前，输出一些信息到标准输出，标准错误和命令行
        print("bar first from python")
        libc.fprintf(c_stderr, b"bar first from c\n")
        os.system("echo bar first from cmd 1>&2")

        # 使用 redirect_stderr 函数重定向标准错误到 stderr_log 文件
        with redirect_stderr(stderr_log):
            print("bar from python", file=sys.stderr)
            libc.fprintf(c_stderr, b"bar from c\n")
            os.system("echo bar from cmd 1>&2")

        print("bar again from python")
        libc.fprintf(c_stderr, b"bar again from c\n")
        os.system("echo bar again from cmd 1>&2")

        # 检查 stderr_log 文件中的输出是否符合预期
        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"bar from python\n", "bar from c\n", "bar from cmd\n"}, lines
            )
    # 定义一个测试函数，测试重定向标准输出和标准错误流
    def test_redirect_both(self):
        # 设置标准输出和标准错误的日志文件路径
        stdout_log = os.path.join(self.test_dir, "stdout.log")
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        # 输出到标准输出流
        print("first stdout from python")
        # 使用 C 函数输出到标准输出流
        libc.printf(b"first stdout from c\n")

        # 输出到标准错误流
        print("first stderr from python", file=sys.stderr)
        # 使用 C 函数输出到标准错误流
        libc.fprintf(c_stderr, b"first stderr from c\n")

        # 使用上下文管理器重定向标准输出和标准错误流到指定文件
        with redirect_stdout(stdout_log), redirect_stderr(stderr_log):
            # 输出重定向后的标准输出内容
            print("redir stdout from python")
            # 输出重定向后的标准错误内容
            print("redir stderr from python", file=sys.stderr)
            # 使用 C 函数输出重定向后的标准输出内容
            libc.printf(b"redir stdout from c\n")
            # 使用 C 函数输出重定向后的标准错误内容
            libc.fprintf(c_stderr, b"redir stderr from c\n")

        # 恢复到原始的标准输出流
        print("again stdout from python")
        # 使用 C 函数输出到标准错误流
        libc.fprintf(c_stderr, b"again stderr from c\n")

        # 打开标准输出日志文件，读取内容并进行断言
        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stdout from python\n", "redir stdout from c\n"}, lines
            )

        # 打开标准错误日志文件，读取内容并进行断言
        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stderr from python\n", "redir stderr from c\n"}, lines
            )

    # 定义一个测试函数，测试重定向大量输出到标准输出流
    def _redirect_large_buffer(self, print_fn, num_lines=500_000):
        # 设置标准输出日志文件路径
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        # 使用上下文管理器重定向标准输出流到指定文件
        with redirect_stdout(stdout_log):
            # 循环输出大量内容到标准输出流
            for i in range(num_lines):
                print_fn(i)

        # 打开标准输出日志文件，读取内容并进行断言
        with open(stdout_log) as fp:
            actual = {int(line.split(":")[1]) for line in fp}
            expected = set(range(num_lines))
            self.assertSetEqual(expected, actual)

    # 定义一个测试函数，测试 Python 函数输出大量内容到标准输出流
    def test_redirect_large_buffer_py(self):
        # 定义一个 Python 函数用于输出内容
        def py_print(i):
            print(f"py:{i}")

        # 调用重定向大量输出函数，传入 Python 输出函数
        self._redirect_large_buffer(py_print)

    # 定义一个测试函数，测试 C 函数输出大量内容到标准输出流
    def test_redirect_large_buffer_c(self):
        # 定义一个 C 函数用于输出内容
        def c_print(i):
            libc.printf(bytes(f"c:{i}\n", "utf-8"))

        # 调用重定向大量输出函数，传入 C 输出函数
        self._redirect_large_buffer(c_print)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行单元测试的主函数
if __name__ == "__main__":
    # 调用 unittest 模块的主函数，启动测试运行
    unittest.main()
```