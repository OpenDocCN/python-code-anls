# `.\pytorch\torch\_dynamo\test_minifier_common.py`

```
# mypy: allow-untyped-defs
# 引入所需的模块和库
import dataclasses  # 用于数据类的装饰器
import io  # 提供了核心的Python I/O功能
import logging  # 用于记录日志消息
import os  # 提供了与操作系统交互的功能
import re  # 提供了正则表达式的支持
import shutil  # 提供了高级的文件操作功能
import subprocess  # 允许在新进程中运行命令
import sys  # 提供了对Python解释器的访问和控制
import tempfile  # 提供了创建临时文件和目录的功能
import traceback  # 提供了获取和打印异常信息的功能
from typing import Optional  # 提供了类型提示支持，用于声明变量的类型
from unittest.mock import patch  # 提供了用于测试和模拟的装饰器

import torch  # PyTorch深度学习库
import torch._dynamo  # PyTorch内部动态计算图相关的模块
import torch._dynamo.test_case  # PyTorch内部动态计算图的测试用例
from torch.utils._traceback import report_compile_source_on_error  # PyTorch工具函数，用于编译错误时的报告


@dataclasses.dataclass
# 数据类，用于定义只包含数据的类
class MinifierTestResult:
    minifier_code: str  # 缩小器的代码字符串
    repro_code: str  # 可复现代码的字符串

    def _get_module(self, t):
        # 从给定的字符串中查找和返回模块定义
        match = re.search(r"class Repro\(torch\.nn\.Module\):\s+([ ].*\n| *\n)+", t)
        assert match is not None, "failed to find module"
        r = match.group(0)
        r = re.sub(r"\s+$", "\n", r, flags=re.MULTILINE)
        r = re.sub(r"\n{3,}", "\n\n", r)
        return r.strip()

    def minifier_module(self):
        # 返回缩小器代码中的模块定义
        return self._get_module(self.minifier_code)

    def repro_module(self):
        # 返回可复现代码中的模块定义
        return self._get_module(self.repro_code)


class MinifierTestBase(torch._dynamo.test_case.TestCase):
    DEBUG_DIR = tempfile.mkdtemp()  # 创建一个临时目录来存放调试信息

    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # 调用父类的setUpClass方法进行初始化
        cls._exit_stack.enter_context(  # 进入一个上下文管理器
            torch._dynamo.config.patch(debug_dir_root=cls.DEBUG_DIR)
        )
        # 为了加快缩小测试的速度，禁用以下配置
        cls._exit_stack.enter_context(  # 进入一个上下文管理器
            torch._inductor.config.patch(
                {
                    # https://github.com/pytorch/pytorch/issues/100376
                    "pattern_matcher": False,
                    # 多进程编译需要很长时间进行预热
                    "compile_threads": 1,
                    # https://github.com/pytorch/pytorch/issues/100378
                    "cpp.vec_isa_ok": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        if os.getenv("PYTORCH_KEEP_TMPDIR", "0") != "1":
            shutil.rmtree(cls.DEBUG_DIR)  # 删除临时目录
        else:
            print(f"test_minifier_common tmpdir kept at: {cls.DEBUG_DIR}")  # 如果环境变量设置为保留临时目录，则打印消息
        cls._exit_stack.close()  # 关闭上下文管理器

    def _gen_codegen_fn_patch_code(self, device, bug_type):
        assert bug_type in ("compile_error", "runtime_error", "accuracy")
        # 生成用于修补代码生成函数的代码片段，根据设备和bug类型生成不同的配置
        return f"""\
{torch._dynamo.config.codegen_config()}  # 调用PyTorch动态图配置函数
{torch._inductor.config.codegen_config()}  # 调用PyTorch感应器配置函数
torch._inductor.config.{"cpp" if device == "cpu" else "triton"}.inject_relu_bug_TESTING_ONLY = {bug_type!r}
"""
    # 如果不是隔离运行，则执行以下操作
    def _maybe_subprocess_run(self, args, *, isolate, cwd=None):
        if not isolate:
            # 断言确保参数 args 至少有两个元素，并且第一个元素是 "python3"
            assert len(args) >= 2, args
            assert args[0] == "python3", args
            # 如果第二个参数是 "-c"
            if args[1] == "-c":
                # 断言确保参数长度为 3
                assert len(args) == 3, args
                # 将第三个参数（即代码段）赋给变量 code，重置 args 为 ["-c"]
                code = args[2]
                args = ["-c"]
            else:
                # 否则，确保参数长度至少为 2，打开第二个参数指定的文件并读取内容给 code
                assert len(args) >= 2, args
                with open(args[1]) as f:
                    code = f.read()
                # 重置 args 为第二个参数之后的内容
                args = args[1:]

            # 警告：这并不是完美模拟在树外运行程序的行为。
            # 我们只拦截我们知道需要处理的测试相关内容。如果需要更多功能，
            # 需要相应地增强此功能。

            # 注意：不能使用 save_config，因为它会省略某些字段，但我们必须保存和重置所有字段
            # 浅复制 torch._dynamo.config 和 torch._inductor.config 的配置到局部变量
            dynamo_config = torch._dynamo.config.shallow_copy_dict()
            inductor_config = torch._inductor.config.shallow_copy_dict()
            try:
                # 捕获标准错误输出
                stderr = io.StringIO()
                # 设置一个日志处理器来捕获日志到 stderr
                log_handler = logging.StreamHandler(stderr)
                log = logging.getLogger("torch._dynamo")
                log.addHandler(log_handler)
                try:
                    # 保存当前工作目录
                    prev_cwd = os.getcwd()
                    # 如果指定了 cwd，则切换工作目录到指定目录
                    if cwd is not None:
                        os.chdir(cwd)
                    # 使用 patch 修改 sys.argv，然后执行代码块，传入 "__name__": "__main__" 和 "__compile_source__": code
                    with patch("sys.argv", args), report_compile_source_on_error():
                        exec(code, {"__name__": "__main__", "__compile_source__": code})
                    # 执行成功返回状态码 0
                    rc = 0
                except Exception:
                    # 发生异常返回状态码 1，并打印异常信息到 stderr
                    rc = 1
                    traceback.print_exc(file=stderr)
                finally:
                    # 移除日志处理器
                    log.removeHandler(log_handler)
                    # 如果指定了 cwd，则恢复到之前的工作目录
                    if cwd is not None:
                        os.chdir(prev_cwd)  # type: ignore[possibly-undefined]
                    # 确保不留下有问题的编译帧
                    torch._dynamo.reset()
            finally:
                # 恢复 torch._dynamo 和 torch._inductor 的配置
                torch._dynamo.config.load_config(dynamo_config)
                torch._inductor.config.load_config(inductor_config)

            # TODO: 在此处返回一个更适合的数据结构
            # 返回 subprocess.CompletedProcess 对象，包括 args、rc、空字节串和 stderr 内容的编码版本
            return subprocess.CompletedProcess(
                args,
                rc,
                b"",
                stderr.getvalue().encode("utf-8"),
            )
        else:
            # 如果隔离为真，则直接运行子进程，并捕获输出，指定工作目录为 cwd，不检查返回值
            return subprocess.run(args, capture_output=True, cwd=cwd, check=False)

    # 在单独的 Python 进程中运行代码块。
    # 返回完成的进程状态和包含缩小启动脚本的目录，如果代码输出了它的话。
    # 定义一个方法来运行测试代码，传入代码字符串和是否隔离的参数
    def _run_test_code(self, code, *, isolate):
        # 可能会以子进程方式运行给定的 Python 代码字符串
        proc = self._maybe_subprocess_run(
            ["python3", "-c", code], isolate=isolate, cwd=self.DEBUG_DIR
        )

        # 打印测试标准输出的内容
        print("test stdout:", proc.stdout.decode("utf-8"))
        # 打印测试标准错误的内容
        print("test stderr:", proc.stderr.decode("utf-8"))

        # 从标准错误中查找符合特定模式的内容，通常用于定位 repro 目录
        repro_dir_match = re.search(
            r"(\S+)minifier_launcher.py", proc.stderr.decode("utf-8")
        )
        # 如果找到了匹配的内容，则返回进程对象和匹配到的 repro 目录
        if repro_dir_match is not None:
            return proc, repro_dir_match.group(1)
        # 否则只返回进程对象和 None
        return proc, None

    # 运行指定 repro 目录下的 minifier_launcher.py 脚本
    def _run_minifier_launcher(self, repro_dir, isolate, *, minifier_args=()):
        # 断言 repro 目录不为空
        self.assertIsNotNone(repro_dir)
        # 构建 minifier_launcher.py 的完整路径
        launch_file = os.path.join(repro_dir, "minifier_launcher.py")
        # 读取 minifier_launcher.py 的内容
        with open(launch_file) as f:
            launch_code = f.read()
        # 断言 minifier_launcher.py 文件存在
        self.assertTrue(os.path.exists(launch_file))

        # 准备运行 minifier_launcher.py 的命令行参数列表
        args = ["python3", launch_file, "minify", *minifier_args]
        # 如果不隔离执行，则添加 --no-isolate 参数
        if not isolate:
            args.append("--no-isolate")
        # 以子进程方式运行 minifier_launcher.py 脚本
        launch_proc = self._maybe_subprocess_run(args, isolate=isolate, cwd=repro_dir)
        # 打印 minifier 的标准输出内容
        print("minifier stdout:", launch_proc.stdout.decode("utf-8"))
        # 获取 minifier 的标准错误内容
        stderr = launch_proc.stderr.decode("utf-8")
        # 打印 minifier 的标准错误内容
        print("minifier stderr:", stderr)
        # 断言标准错误中不包含特定的错误信息，用于确认测试通过
        self.assertNotIn("Input graph did not fail the tester", stderr)

        # 返回 minifier 进程对象和 minifier_launcher.py 的代码内容
        return launch_proc, launch_code

    # 运行指定 repro 目录下的 repro.py 脚本
    def _run_repro(self, repro_dir, *, isolate=True):
        # 断言 repro 目录不为空
        self.assertIsNotNone(repro_dir)
        # 构建 repro.py 的完整路径
        repro_file = os.path.join(repro_dir, "repro.py")
        # 读取 repro.py 的内容
        with open(repro_file) as f:
            repro_code = f.read()
        # 断言 repro.py 文件存在
        self.assertTrue(os.path.exists(repro_file))

        # 以子进程方式运行 repro.py 脚本
        repro_proc = self._maybe_subprocess_run(
            ["python3", repro_file], isolate=isolate, cwd=repro_dir
        )
        # 打印 repro 的标准输出内容
        print("repro stdout:", repro_proc.stdout.decode("utf-8"))
        # 打印 repro 的标准错误内容
        print("repro stderr:", repro_proc.stderr.decode("utf-8"))

        # 返回 repro 进程对象和 repro.py 的代码内容
        return repro_proc, repro_code

    # 生成测试代码的模板。
    # `run_code` 是要运行的测试用例代码。
    # `repro_after` 是在每个生成的文件中要修复的代码；通常通过配置来打开错误。
    # `repro_level` 是修复错误的级别。
    def _gen_test_code(self, run_code, repro_after, repro_level):
        # 返回格式化后的测试代码字符串
        return f"""\
import torch
import torch._dynamo
{torch._dynamo.config.codegen_config()}  # 调用torch._dynamo.config.codegen_config()函数并插入结果
{torch._inductor.config.codegen_config()}  # 调用torch._inductor.config.codegen_config()函数并插入结果
torch._dynamo.config.repro_after = "{repro_after}"  # 设置torch._dynamo.config.repro_after为指定的值
torch._dynamo.config.repro_level = {repro_level}  # 设置torch._dynamo.config.repro_level为指定的值
torch._dynamo.config.debug_dir_root = "{self.DEBUG_DIR}"  # 设置torch._dynamo.config.debug_dir_root为指定的值
{run_code}  # 插入指定的run_code内容
"""

# Runs a full minifier test.
# Minifier tests generally consist of 3 stages:
# 1. Run the problematic code
# 2. Run the generated minifier launcher script
# 3. Run the generated repro script
#
# If possible, you should run the test with isolate=False; use
# isolate=True only if the bug you're testing would otherwise
# crash the process
def _run_full_test(
    self, run_code, repro_after, expected_error, *, isolate, minifier_args=()
) -> Optional[MinifierTestResult]:
    if isolate:
        repro_level = 3  # 如果隔离，设置repro_level为3
    elif expected_error is None or expected_error == "AccuracyError":
        repro_level = 4  # 如果没有预期错误或者预期错误为"AccuracyError"，设置repro_level为4
    else:
        repro_level = 2  # 其他情况下，设置repro_level为2
    test_code = self._gen_test_code(run_code, repro_after, repro_level)  # 生成测试用的代码
    print("running test", file=sys.stderr)  # 输出正在运行测试
    test_proc, repro_dir = self._run_test_code(test_code, isolate=isolate)  # 运行测试代码并获取测试进程和复现目录
    if expected_error is None:
        # Just check that there was no error
        self.assertEqual(test_proc.returncode, 0)  # 断言测试进程的返回码为0，即无错误
        self.assertIsNone(repro_dir)  # 断言复现目录为None
        return None
    # NB: Intentionally do not test return code; we only care about
    # actually generating the repro, we don't have to crash
    self.assertIn(expected_error, test_proc.stderr.decode("utf-8"))  # 断言预期错误在测试进程的标准错误输出中
    self.assertIsNotNone(repro_dir)  # 断言复现目录不为None
    print("running minifier", file=sys.stderr)  # 输出正在运行缩小器
    minifier_proc, minifier_code = self._run_minifier_launcher(
        repro_dir, isolate=isolate, minifier_args=minifier_args
    )  # 运行缩小器启动器并获取缩小器进程和代码
    print("running repro", file=sys.stderr)  # 输出正在运行复现
    repro_proc, repro_code = self._run_repro(repro_dir, isolate=isolate)  # 运行复现并获取复现进程和代码
    self.assertIn(expected_error, repro_proc.stderr.decode("utf-8"))  # 断言预期错误在复现进程的标准错误输出中
    self.assertNotEqual(repro_proc.returncode, 0)  # 断言复现进程的返回码不为0
    return MinifierTestResult(minifier_code=minifier_code, repro_code=repro_code)  # 返回缩小器测试结果对象
```