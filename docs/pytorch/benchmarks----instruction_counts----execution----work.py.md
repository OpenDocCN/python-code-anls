# `.\pytorch\benchmarks\instruction_counts\execution\work.py`

```
# 导入需要的模块和库
import dataclasses  # 用于定义数据类
import json  # 用于处理 JSON 格式数据
import os  # 提供与操作系统交互的功能
import pickle  # 用于序列化和反序列化 Python 对象
import signal  # 用于处理信号
import subprocess  # 用于执行外部命令或程序
import time  # 提供时间相关的功能
import uuid  # 用于生成唯一标识符
from typing import List, Optional, TYPE_CHECKING, Union  # 提供类型提示支持

from core.api import AutoLabels  # 导入自定义模块
from core.types import Label  # 导入自定义模块
from core.utils import get_temp_dir  # 导入自定义模块
from worker.main import (
    WORKER_PATH,  # 导入自定义模块中的常量
    WorkerFailure,  # 导入自定义模块中的异常类
    WorkerOutput,  # 导入自定义模块中的类
    WorkerTimerArgs,  # 导入自定义模块中的类
    WorkerUnpickler,  # 导入自定义模块中的类
)

if TYPE_CHECKING:
    PopenType = subprocess.Popen[bytes]  # 类型声明，用于类型检查
else:
    PopenType = subprocess.Popen  # 类型声明，用于类型检查


# 解决特定问题的环境变量设置
_ENV = "MKL_THREADING_LAYER=GNU"
_PYTHON = "python"
PYTHON_CMD = f"{_ENV} {_PYTHON}"  # 构建 Python 命令字符串


# 必须指定 `bash`，以便 `source activate ...` 始终有效
SHELL = "/bin/bash"


@dataclasses.dataclass(frozen=True)
class WorkOrder:
    """用于调度基准测试运行的规范。"""

    label: Label  # 标签对象，描述工作类型
    autolabels: AutoLabels  # 自动标签对象，提供额外的标签信息
    timer_args: WorkerTimerArgs  # 工作器计时器参数对象
    source_cmd: Optional[str] = None  # 可选的源命令字符串
    timeout: Optional[float] = None  # 可选的超时时间
    retries: int = 0  # 重试次数，默认为 0

    def __hash__(self) -> int:
        return id(self)  # 返回对象的唯一标识符

    def __str__(self) -> str:
        return json.dumps(
            {
                "label": self.label,  # 转换为 JSON 字符串的标签
                "autolabels": self.autolabels.as_dict,  # 自动标签转换为字典形式
                "num_threads": self.timer_args.num_threads,  # 计时器参数中的线程数
            }
        )


class _BenchmarkProcess:
    """封装了给定 WorkOrder 的 subprocess.Popen 对象。"""

    _work_order: WorkOrder  # 关联的工作订单对象
    _cpu_list: Optional[str]  # 可选的 CPU 列表字符串
    _proc: PopenType  # 子进程对象的类型

    # 内部状态管理
    _communication_file: str  # 通信文件的路径
    _start_time: float  # 进程启动时间
    _end_time: Optional[float] = None  # 进程结束时间（可选）
    _retcode: Optional[int]  # 进程返回码（可选）
    _result: Optional[Union[WorkerOutput, WorkerFailure]] = None  # 进程结果（可选）

    def __init__(self, work_order: WorkOrder, cpu_list: Optional[str]) -> None:
        self._work_order = work_order  # 初始化关联的工作订单对象
        self._cpu_list = cpu_list  # 初始化可选的 CPU 列表字符串
        self._start_time = time.time()  # 记录进程启动时间
        self._communication_file = os.path.join(get_temp_dir(), f"{uuid.uuid4()}.pkl")  # 创建唯一的通信文件名
        with open(self._communication_file, "wb") as f:
            pickle.dump(self._work_order.timer_args, f)  # 序列化工作器计时器参数对象到通信文件中

        self._proc = subprocess.Popen(
            self.cmd,  # 执行的命令
            stdout=subprocess.PIPE,  # 标准输出重定向到管道
            stderr=subprocess.STDOUT,  # 标准错误输出重定向到标准输出
            shell=True,  # 使用 shell 解析命令
            executable=SHELL,  # 指定的 shell 可执行程序
        )

    def clone(self) -> "_BenchmarkProcess":
        return _BenchmarkProcess(self._work_order, self._cpu_list)  # 克隆当前对象的新实例
    # 返回一个字符串，包含构建的命令行命令
    def cmd(self) -> str:
        cmd: List[str] = []
        # 如果存在源命令，则将其添加到命令列表中
        if self._work_order.source_cmd is not None:
            cmd.extend([self._work_order.source_cmd, "&&"])

        # 添加环境变量至命令列表中
        cmd.append(_ENV)

        # 如果存在 CPU 列表，则设置 GOMP_CPU_AFFINITY 环境变量，并使用 taskset 指定 CPU 列表
        if self._cpu_list is not None:
            cmd.extend(
                [
                    f"GOMP_CPU_AFFINITY={self._cpu_list}",
                    "taskset",
                    "--cpu-list",
                    self._cpu_list,
                ]
            )

        # 添加 Python 解释器路径、工作路径、通信文件路径至命令列表中
        cmd.extend(
            [
                _PYTHON,
                WORKER_PATH,
                "--communication-file",
                self._communication_file,
            ]
        )
        # 返回用空格连接的命令列表字符串
        return " ".join(cmd)

    # 返回任务执行时长，单位为秒
    @property
    def duration(self) -> float:
        return (self._end_time or time.time()) - self._start_time

    # 返回任务执行结果，可能为 WorkerOutput 或 WorkerFailure 对象
    @property
    def result(self) -> Union[WorkerOutput, WorkerFailure]:
        # 确保已经收集了结果
        self._maybe_collect()
        assert self._result is not None
        return self._result

    # 返回进程的退出码，若进程尚未结束则返回 None
    def poll(self) -> Optional[int]:
        # 确保已经收集了结果
        self._maybe_collect()
        return self._retcode

    # 发送软中断信号给子进程，允许其进行清理操作
    def interrupt(self) -> None:
        """Soft interrupt. Allows subprocess to cleanup."""
        self._proc.send_signal(signal.SIGINT)

    # 发送硬中断信号给子进程，立即终止其运行
    def terminate(self) -> None:
        """Hard interrupt. Immediately SIGTERM subprocess."""
        self._proc.terminate()

    # 可能进行结果收集的内部方法
    def _maybe_collect(self) -> None:
        # 如果已经收集了结果，则直接返回
        if self._result is not None:
            return

        # 查询子进程的退出码
        self._retcode = self._proc.poll()
        # 如果子进程仍在运行，则返回
        if self._retcode is None:
            return

        # 从通信文件中读取结果
        with open(self._communication_file, "rb") as f:
            result = WorkerUnpickler(f).load_output()

        # 如果结果为 WorkerOutput 类型且进程返回码非零，表示任务完成但进程未干净退出
        if isinstance(result, WorkerOutput) and self._retcode:
            result = WorkerFailure("Worker failed silently.")

        # 如果结果为 WorkerTimerArgs 类型，则表示任务执行失败，读取标准输出作为失败追踪信息
        if isinstance(result, WorkerTimerArgs):
            proc_stdout = self._proc.stdout
            assert proc_stdout is not None
            result = WorkerFailure(failure_trace=proc_stdout.read().decode("utf-8"))

        # 记录结果并记录结束时间
        self._result = result
        self._end_time = time.time()

        # 删除通信文件
        os.remove(self._communication_file)
class InProgress:
    """Used by the benchmark runner to track outstanding jobs.
    This class handles bookkeeping and timeout + retry logic.
    """

    _proc: _BenchmarkProcess  # 声明私有属性 _proc，类型为 _BenchmarkProcess
    _timeouts: int = 0  # 声明类级别的超时次数计数器，默认为0

    def __init__(self, work_order: WorkOrder, cpu_list: Optional[str]):
        self._work_order = work_order  # 初始化实例属性 _work_order，传入的工作订单
        self._proc = _BenchmarkProcess(work_order, cpu_list)  # 初始化 _proc 属性为 _BenchmarkProcess 实例

    @property
    def work_order(self) -> WorkOrder:
        return self._proc._work_order  # 返回 _proc 对象的工作订单属性

    @property
    def cpu_list(self) -> Optional[str]:
        return self._proc._cpu_list  # 返回 _proc 对象的 CPU 列表属性

    @property
    def proc(self) -> _BenchmarkProcess:
        # NB: For cleanup only.
        return self._proc  # 返回 _proc 对象本身，用于清理操作

    @property
    def duration(self) -> float:
        return self._proc.duration  # 返回 _proc 对象的持续时间属性

    def check_finished(self) -> bool:
        if self._proc.poll() is not None:  # 如果 _proc 对象的 poll 方法不为 None，则任务已完成
            return True

        timeout = self.work_order.timeout  # 获取工作订单的超时时间
        if timeout is None or self._proc.duration < timeout:  # 如果没有超时时间或者实际运行时间小于超时时间
            return False

        self._timeouts += 1  # 增加超时次数计数器
        max_attempts = (self._work_order.retries or 0) + 1  # 最大重试次数
        if self._timeouts < max_attempts:  # 如果未超过最大重试次数
            print(
                f"\nTimeout: {self._work_order.label}, {self._work_order.autolabels} "
                f"(Attempt {self._timeouts} / {max_attempts})"
            )
            self._proc.interrupt()  # 中断当前进程
            self._proc = self._proc.clone()  # 克隆当前进程对象
            return False

        raise subprocess.TimeoutExpired(cmd=self._proc.cmd, timeout=timeout)  # 超过最大重试次数，抛出超时异常

    @property
    def result(self) -> Union[WorkerOutput, WorkerFailure]:
        return self._proc.result  # 返回 _proc 对象的结果属性

    def __hash__(self) -> int:
        return id(self)  # 返回当前对象的哈希值
```