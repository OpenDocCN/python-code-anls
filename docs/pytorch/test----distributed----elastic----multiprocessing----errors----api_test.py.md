# `.\pytorch\test\distributed\elastic\multiprocessing\errors\api_test.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

import json  # 导入用于 JSON 操作的模块
import os  # 导入操作系统功能的模块
import shutil  # 导入用于文件和目录操作的模块
import signal  # 导入信号处理相关的模块
import tempfile  # 导入临时文件和目录创建的模块
import unittest  # 导入单元测试框架模块
from unittest import mock  # 导入用于模拟对象的模块

from torch.distributed.elastic.multiprocessing.errors import (  # 导入 Torch 分布式处理错误相关的模块
    ChildFailedError,
    ProcessFailure,
    record,
)
from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler  # 导入错误处理器类


class SentinelError(Exception):
    # 自定义异常类，用于验证正确的异常是否被引发和传播
    pass


@record
def raise_exception_fn():
    raise SentinelError("foobar")  # 引发自定义异常 SentinalError


@record
def raise_system_exit_exception_fn(exit_code: int = 1):
    exp = SystemExit()  # 创建系统退出异常对象
    exp.code = exit_code  # 设置退出码
    raise exp  # 抛出系统退出异常


@record
def good_fn():
    print("hello world")  # 打印 "hello world"


@record
def raise_child_failure_error_fn(name, child_error_file=""):
    if child_error_file:
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": child_error_file}):
            ErrorHandler().record_exception(SentinelError("foobar"))  # 记录异常到错误处理器
    pf = ProcessFailure(local_rank=0, pid=997, exitcode=1, error_file=child_error_file)
    raise ChildFailedError(name, {0: pf})  # 引发子进程失败错误


def read_resource_file(resource_file: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), resource_file)) as fp:
        return "".join(fp.readlines())  # 读取资源文件内容并返回字符串形式


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)  # 创建临时测试目录
        self.test_error_file = os.path.join(self.test_dir, "error.json")  # 设置测试错误文件路径

    def tearDown(self):
        shutil.rmtree(self.test_dir)  # 清理临时测试目录及其内容

    def test_failure_incorrect_reply_file(self):
        content = {"unknown_key": "unknown_value"}
        with open(self.test_error_file, "w") as fp:
            json.dump(content, fp)  # 写入错误内容到 JSON 文件
        with self.assertRaises(Exception):
            ProcessFailure(
                local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
            )  # 测试用例：断言在处理失败时引发异常

    def failure_with_error_file(self, exception):
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            ErrorHandler().record_exception(exception)  # 记录异常到错误处理器
        return ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )  # 返回处理失败对象

    def failure_without_error_file(self, exitcode):
        return ProcessFailure(
            local_rank=0, pid=997, exitcode=exitcode, error_file="ignored.json"
        )  # 返回处理失败对象，指定错误文件为 "ignored.json"

    def test_process_failure_new_format(self):
        error_data = {"message": "test error message", "timestamp": 10}
        with open(self.test_error_file, "w") as fp:
            json.dump(error_data, fp)  # 写入错误数据到 JSON 文件
        pf = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )  # 创建处理失败对象
        self.assertEqual("test error message", pf.message)  # 断言处理失败消息内容
        self.assertEqual(10, pf.timestamp)  # 断言处理失败的时间戳
    # 测试处理错误格式的情况
    def test_process_mast_error_format(self):
        # 准备一个测试用的错误数据字典
        error_data = {"message": "test error message", "timestamp": "10"}
        # 将错误数据字典写入到测试错误文件中
        with open(self.test_error_file, "w") as fp:
            json.dump(error_data, fp)
        # 创建一个 ProcessFailure 对象，并指定相关属性
        pf = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )
        # 断言错误消息与时间戳与预期相符
        self.assertEqual("test error message", pf.message)
        self.assertEqual(10, pf.timestamp)

    # 测试处理进程失败情况
    def test_process_failure(self):
        # 创建一个具有错误文件的 ProcessFailure 对象
        pf = self.failure_with_error_file(exception=SentinelError("foobar"))
        # 断言属性值与预期相符
        self.assertEqual(0, pf.local_rank)
        self.assertEqual(997, pf.pid)
        self.assertEqual(1, pf.exitcode)
        self.assertEqual(self.test_error_file, pf.error_file)
        # 断言错误文件数据中的时间戳与 ProcessFailure 对象中的时间戳字符串相符
        self.assertEqual(
            pf.error_file_data["message"]["extraInfo"]["timestamp"], str(pf.timestamp)
        )
        # 断言消息不为空字符串或 None
        self.assertTrue(pf.message)
        # 检查信号名称是否为 "<N/A>"
        self.assertEqual("<N/A>", pf.signal_name())

    # 测试处理进程失败信号的情况
    def test_process_failure_signal(self):
        # 创建一个没有错误文件的 ProcessFailure 对象，并指定退出码为 SIGSEGV
        pf = self.failure_without_error_file(exitcode=-signal.SIGSEGV)
        # 断言信号名称为 "SIGSEGV"
        self.assertEqual("SIGSEGV", pf.signal_name())
        # 断言消息内容与预期相符
        self.assertEqual(
            f"Signal {signal.SIGSEGV} (SIGSEGV) received by PID {pf.pid}", pf.message
        )

    # 测试处理进程失败没有错误文件的情况
    def test_process_failure_no_error_file(self):
        # 创建一个没有错误文件的 ProcessFailure 对象，并指定退出码为 138
        pf = self.failure_without_error_file(exitcode=138)
        # 断言信号名称为 "<N/A>"
        self.assertEqual("<N/A>", pf.signal_name())
        # 断言错误文件为 "<N/A>"
        self.assertEqual("<N/A>", pf.error_file)
        # 断言消息内容与预期相符
        self.assertEqual(
            "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
            pf.message,
        )
    def test_child_failed_error(self):
        # 创建三个模拟的失败文件（pf0, pf1, pf2），每个文件包含特定异常或退出码信息
        pf0 = self.failure_with_error_file(exception=SentinelError("rank 0"))
        pf1 = self.failure_with_error_file(exception=SentinelError("rank 1"))
        pf2 = self.failure_without_error_file(exitcode=138)
        
        # 创建 ChildFailedError 对象，包含失败信息和相关的失败文件信息
        ex = ChildFailedError("trainer.par", {0: pf0, 1: pf1, 2: pf2})
        
        # 断言：验证第一个失败文件（pf0）是否与 ex.get_first_failure()[1] 相等
        self.assertEqual(pf0, ex.get_first_failure()[1])
        
        # 打印 ex 对象，输出详细的失败信息（用于调试目的）
        # 预期输出类似如下格式的信息
        """
        *********************************************
              trainer.par FAILED
        =============================================
        Root Cause:
        [0]:
          time: 2020-11-25_21:22:31
          rank: 0 (local_rank: 0)
          exitcode: 1 (pid: 997)
          error_file: /tmp/ApiTesttbb37ier/error.json
          traceback: "SentinelError: rank 0"
        =============================================
        Other Failures:
        [1]:
          time: 2020-11-25_21:22:31
          rank: 1 (local_rank: 0)
          exitcode: 1 (pid: 997)
          error_file: /tmp/ApiTesttbb37ier/error.json
          msg: "SentinelError: rank 1"
        [2]:
          time: 2020-11-25_21:22:31
          rank: 2 (local_rank: 0)
          exitcode: 138 (pid: 997)
          error_file: <N/A>
          traceback: To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
        *********************************************
        """
        print(ex)

    def test_record(self):
        # 使用 mock.patch.dict 临时修改环境变量 TORCHELASTIC_ERROR_FILE 指向测试错误文件
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            # 断言：当引发 SentinelError 异常时，确保会被捕获
            with self.assertRaises(SentinelError):
                raise_exception_fn()

        # 打开测试错误文件，加载其中的 JSON 数据并进行断言检查
        with open(self.test_error_file) as fp:
            err = json.load(fp)
            # 断言：确保 JSON 数据中的必要字段不为空
            self.assertIsNotNone(err["message"]["message"])
            self.assertIsNotNone(err["message"]["extraInfo"]["py_callstack"])
            self.assertIsNotNone(err["message"]["extraInfo"]["timestamp"])

    def test_record_system_exit(self):
        # 使用 mock.patch.dict 临时修改环境变量，确保 TORCHELASTIC_ERROR_FILE 为空
        with mock.patch.dict(os.environ, {}):
            # 引发系统退出异常，指定退出码为 0
            raise_system_exit_exception_fn(exit_code=0)

        # 断言：未生成错误文件
        # 期望结果：self.test_error_file 文件不存在
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_system_exit_erronr(self):
        # 使用 mock.patch.dict 临时修改环境变量，确保 TORCHELASTIC_ERROR_FILE 为空
        with mock.patch.dict(os.environ, {}):
            # 引发 SystemExit 异常，预期会捕获该异常
            with self.assertRaises(SystemExit):
                raise_system_exit_exception_fn()

        # 断言：未生成错误文件
        # 期望结果：self.test_error_file 文件不存在
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_no_error_file(self):
        # 使用 mock.patch.dict 临时修改环境变量，确保 TORCHELASTIC_ERROR_FILE 为空
        with mock.patch.dict(os.environ, {}):
            # 断言：当引发 SentinelError 异常时，确保会被捕获
            with self.assertRaises(SentinelError):
                raise_exception_fn()

        # 断言：未生成错误文件
        # 期望结果：self.test_error_file 文件不存在
        self.assertFalse(os.path.isfile(self.test_error_file))
    # 定义测试函数，用于测试正常情况下的函数执行结果
    def test_record_good_fn(self):
        # 使用 mock.patch.dict() 上下文管理器修改环境变量，设置错误文件路径
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            # 调用 good_fn() 函数，预期不会产生错误，因此不应生成错误文件
            good_fn()
            # 断言：验证错误文件不存在
            self.assertFalse(os.path.isfile(self.test_error_file))

    # 定义测试函数，测试在子进程失败时的记录行为
    def test_record_child_failure(self):
        # 设置训练日志目录
        trainer_log_dir = os.path.join(self.test_dir, "trainer", "0")
        # 递归创建目录
        os.makedirs(trainer_log_dir)
        # 设置训练错误文件路径
        trainer_error_file = os.path.join(trainer_log_dir, "error.json")

        # 使用 mock.patch.dict() 上下文管理器修改环境变量，设置错误文件路径
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            # 使用上下文管理器断言预期会抛出 ChildFailedError 异常
            with self.assertRaises(ChildFailedError) as cm:
                # 调用 raise_child_failure_error_fn() 抛出子进程失败异常
                raise_child_failure_error_fn("trainer", trainer_error_file)
            # 获取第一个失败的信息
            pf = cm.exception.get_first_failure()[1]
            # 比较 worker 错误文件与回复文件，并覆盖的错误代码
            expect = json.load(open(pf.error_file))
            expect["message"]["errorCode"] = pf.exitcode
            actual = json.load(open(self.test_error_file))
            # 断言：验证预期的 JSON 序列化数据与实际的 JSON 序列化数据相等
            self.assertTrue(
                json.dumps(expect, sort_keys=True),
                json.dumps(actual, sort_keys=True),
            )

    # 定义测试函数，测试在子进程失败时没有子进程错误文件的记录行为
    def test_record_child_failure_no_child_error_file(self):
        # 使用 mock.patch.dict() 上下文管理器修改环境变量，设置错误文件路径
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            # 使用上下文管理器断言预期会抛出 ChildFailedError 异常
            with self.assertRaises(ChildFailedError):
                # 调用 raise_child_failure_error_fn() 抛出子进程失败异常
                raise_child_failure_error_fn("trainer")

            # 断言：验证错误文件不存在
            # @record 在抛出 ChildFailedError 异常时仅复制子进程错误文件，
            # 不应记录 ChildFailedError 本身，应重新抛出 ChildFailedError
            # 供上游系统处理。
            self.assertFalse(os.path.isfile(self.test_error_file))
```