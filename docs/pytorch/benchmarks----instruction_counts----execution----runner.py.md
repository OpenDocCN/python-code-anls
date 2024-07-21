# `.\pytorch\benchmarks\instruction_counts\execution\runner.py`

```
"""Run benchmarks while handling parallelism, isolation, and fault tolerance."""
# 导入必要的模块
import math  # 数学计算模块
import multiprocessing  # 多进程处理模块
import subprocess  # 子进程管理模块
import textwrap  # 文本包装模块
import threading  # 线程管理模块
import time  # 时间操作模块
from typing import Dict, List, Optional, Set, Tuple, Union  # 类型提示模块

from worker.main import WorkerFailure, WorkerOutput  # 导入工作线程异常和输出类

from execution.work import InProgress, PYTHON_CMD, SHELL, WorkOrder  # 导入执行工作模块

# 获取CPU核心数量
CPU_COUNT: int = multiprocessing.cpu_count()

class WorkerFailed(Exception):
    """Raised in the main process when a worker failure is detected."""

    def __init__(self, cmd: str, wrapped_trace: Optional[str] = None) -> None:
        # 初始化异常类，记录失败命令和可能的堆栈信息
        self.cmd: str = cmd
        self.wrapped_trace: Optional[str] = wrapped_trace
        super().__init__()

class CorePool:
    """Allocator style helper class to assign individual tasks to a core range.

    Pinning tasks to separate cores (or core ranges if `num_threads` > 1)
    serves two purposes. First, it prevents the machine from being overloaded,
    which can result in OOMs or Callgrind crashes. Second, it helps reduce
    noise in the wall times, which are collected as a secondary metric. For
    multi-threaded workloads, adjacency is important. Often pairs of cores
    share silicon (e.g. cache), while far away cores may lie on separate NUMA
    nodes. For this reason, CorePool will only allocate contiguous core ranges.
    This falls short of full architecture awareness, and instead tries to find
    a balance between rigor and engineering complexity.
    """

    def __init__(self, min_core_id: int, max_core_id: int) -> None:
        # 初始化函数，指定核心ID范围，并进行参数合法性检查
        assert min_core_id >= 0
        assert max_core_id >= min_core_id
        assert max_core_id < CPU_COUNT

        self._min_core_id: int = min_core_id
        self._max_core_id: int = max_core_id
        self._num_cores = max_core_id - min_core_id + 1
        print(f"Core pool created: cores {self._min_core_id}-{self._max_core_id}")

        # 初始化可用核心列表，默认所有核心可用
        self._available: List[bool] = [
            True for _ in range(min_core_id, min_core_id + self._num_cores)
        ]

        # 初始化预约字典和线程锁
        self._reservations: Dict[str, Tuple[int, ...]] = {}
        self._lock = threading.Lock()

    def reserve(self, n: int) -> Optional[str]:
        """Simple first-fit policy.

        If successful, return a string for `taskset`. Otherwise, return None.
        """
        with self._lock:
            # 使用简单的首次适配策略分配核心
            for lower_index in range(self._num_cores - n + 1):
                indices = tuple(range(lower_index, lower_index + n))
                if all(self._available[i] for i in indices):
                    # 如果找到合适的核心范围，标记为不可用
                    for i in indices:
                        self._available[i] = False

                    lower_core = indices[0] + self._min_core_id
                    upper_core = indices[-1] + self._min_core_id
                    key = f"{lower_core}-{upper_core}" if n > 1 else f"{lower_core}"
                    self._reservations[key] = indices
                    return key
        return None
    # 定义一个方法用于释放资源，参数是一个字符串类型的键，返回值为None
    def release(self, key: str) -> None:
        # 使用self._lock上下文管理器确保线程安全操作
        with self._lock:
            # 遍历self._reservations[key]中的每个索引 i
            for i in self._reservations[key]:
                # 将self._available[i]设置为True，表示资源可用
                self._available[i] = True
            # 删除self._reservations中的键key及其对应的值，释放资源
            self._reservations.pop(key)
# 定义一个名为 Runner 的类，用于管理和执行工作项的调度
class Runner:
    # 初始化方法，接受工作项列表、核心线程池和执行频率参数
    def __init__(
        self,
        work_items: Tuple[WorkOrder, ...],  # 工作项元组，每个元素是 WorkOrder 类型
        core_pool: Optional[CorePool] = None,  # 可选的核心线程池，默认为 None
        cadence: float = 1.0,  # 执行频率，默认为 1.0 秒
    ) -> None:
        # 初始化实例变量
        self._work_items: Tuple[WorkOrder, ...] = work_items  # 存储工作项的元组
        self._core_pool: CorePool = core_pool or CorePool(0, CPU_COUNT - 4)  # 存储核心线程池对象
        self._cadence: float = cadence  # 存储执行频率

        # 工作状态相关的实例变量
        self._work_queue: List[WorkOrder] = list(work_items)  # 工作队列，初始为工作项列表的副本
        self._active_jobs: List[InProgress] = []  # 当前活跃的工作列表
        self._results: Dict[WorkOrder, WorkerOutput] = {}  # 存储工作项和工作者输出结果的字典

        # 用于调试的信息，如预估完成时间（ETA）和错误消息
        self._start_time: float = -1  # 记录运行开始时间，初始值为 -1
        self._durations: Dict[WorkOrder, float] = {}  # 记录每个工作项的执行时间的字典
        self._currently_processed: Optional[WorkOrder] = None  # 当前正在处理的工作项，初始为 None

        # 检查工作项是否有重复，若有则抛出 ValueError 异常
        if len(work_items) != len(set(work_items)):
            raise ValueError("Duplicate work items.")

    # 公共方法，启动 Runner 对象的运行
    def run(self) -> Dict[WorkOrder, WorkerOutput]:
        try:
            return self._run()  # 调用内部的 _run 方法执行具体逻辑

        except KeyboardInterrupt:
            # 处理 KeyboardInterrupt 异常，打印相关信息并强制关闭子任务
            print("\n\nKeyboardInterrupt (ctrl-c) detected. Shutting down children.")
            self._force_shutdown(verbose=False)
            raise  # 继续向上层抛出异常

        except subprocess.TimeoutExpired:
            # 处理 TimeoutExpired 异常，打印相关信息并强制关闭子任务
            print("\n\nJob timed out. Shutting down children.")
            self._force_shutdown(verbose=True)
            raise  # 继续向上层抛出异常

        except WorkerFailed as e:
            # 处理 WorkerFailed 异常，打印相关信息并强制关闭子任务
            print("Shutting down all outstanding jobs before re-raising.")
            self._force_shutdown(verbose=True)
            print(f"Cmd: {e.cmd}")  # 打印导致异常的命令信息
            if e.wrapped_trace:
                print(e.wrapped_trace)  # 若异常有详细的追踪信息，则打印出来
            else:
                print("Unknown failure. (Worker did not report exception contents.)")
            raise  # 继续向上层抛出异常

        except BaseException:
            # 处理其他任何异常，打印相关信息并强制关闭子任务
            print("\n\nUnknown exception. Shutting down jobs before re-raising.")
            self._force_shutdown(verbose=True)
            raise  # 继续向上层抛出异常

    # 内部方法，执行具体的工作调度逻辑
    def _run(self) -> Dict[WorkOrder, WorkerOutput]:
        self._start_time = time.time()  # 记录开始运行的时间戳
        self._canary_import()  # 执行一个与运行环境相关的初始化操作
        while self._work_queue or self._active_jobs:
            t0 = time.time()  # 记录每轮循环开始的时间戳
            self._update_active_jobs()  # 更新当前活跃的工作列表
            self._enqueue_new_jobs()  # 将新的工作项加入到工作队列中
            self._print_progress()  # 打印当前的执行进度信息
            time.sleep(max(self._cadence - (time.time() - t0), 0.0))  # 根据执行频率进行等待
        print(f"\nTotal time: {time.time() - self._start_time:.0f} seconds")  # 打印总运行时间
        return self._results.copy()  # 返回执行结果的副本字典
    # 更新活跃的作业列表，处理每个作业的状态更新和结果处理
    def _update_active_jobs(self) -> None:
        # 初始化活跃作业列表为空
        active_jobs: List[InProgress] = []
        
        # 遍历当前活跃作业列表中的每个作业
        for job in self._active_jobs:
            # 将当前正在处理的作业设置为当前作业的工作顺序
            self._currently_processed = job.work_order
            
            # 检查作业是否已完成
            if not job.check_finished():
                # 若作业未完成，将其加入活跃作业列表并继续下一个作业的处理
                active_jobs.append(job)
                continue
            
            # 获取作业的处理结果
            result: Union[WorkerOutput, WorkerFailure] = job.result
            
            # 如果作业成功完成，处理成功的情况
            if isinstance(result, WorkerOutput):
                # 将作业的工作顺序和处理结果存入结果字典中
                self._results[job.work_order] = result
                # 断言作业的 CPU 列表不为空，并释放该 CPU 列表到核心池
                assert job.cpu_list is not None
                self._core_pool.release(job.cpu_list)
                # 记录作业的持续时间
                self._durations[job.work_order] = job.duration

            # 如果作业失败，抛出作业失败异常，传递作业的命令和失败的详细信息
            else:
                assert isinstance(result, WorkerFailure)
                raise WorkerFailed(cmd=job.proc.cmd, wrapped_trace=result.failure_trace)
        
        # 清空当前正在处理的作业标记，更新活跃作业列表
        self._currently_processed = None
        self._active_jobs.clear()
        self._active_jobs.extend(active_jobs)

    # 将新的工作顺序加入工作队列
    def _enqueue_new_jobs(self) -> None:
        # 初始化工作队列为空
        work_queue: List[WorkOrder] = []
        
        # 遍历当前工作队列中的每个工作顺序
        for i, work_order in enumerate(self._work_queue):
            # 将当前正在处理的工作顺序设置为当前工作顺序
            self._currently_processed = work_order
            
            # 从核心池中预留该工作顺序所需的 CPU 列表
            cpu_list = self._core_pool.reserve(work_order.timer_args.num_threads)

            # 如果未能获取到 CPU 列表，将工作顺序加入工作队列
            if cpu_list is None:
                work_queue.append(work_order)
            else:
                # 如果成功获取到 CPU 列表，将该工作顺序作为新的活跃作业加入活跃作业列表
                self._active_jobs.append(InProgress(work_order, cpu_list))

                # 等待一段时间以减少同时创建的竞争
                time.sleep(0.5)
        
        # 清空当前正在处理的工作顺序标记，更新工作队列
        self._currently_processed = None
        self._work_queue.clear()
        self._work_queue.extend(work_queue)

    # 打印处理进度信息
    def _print_progress(self) -> None:
        # 计算处理完成的作业数量与总工作项数量的比例
        fraction = f"{len(self._results)} / {len(self._work_items)}"
        # 计算已经经过的时间
        elapsed = f"{time.time() - self._start_time:.0f} seconds"
        
        # 如果完成作业数量少于5个，估计剩余时间为未知
        if len(self._results) < 5:
            eta = "Unknown"
        else:
            # 否则计算剩余的工作项数量和剩余迭代次数
            remaining = len(self._work_items) - len(self._results)
            iters_remaining = math.ceil(remaining / self._core_pool._num_cores)
            # 计算平均处理时间
            mean_time = sum(self._durations.values()) / len(self._durations)
            # 计算预计剩余时间
            eta_minutes = math.ceil(iters_remaining * mean_time / 60)
            eta = f"~{eta_minutes:.0f} minute{'s' if eta_minutes > 1 else ''}"
        
        # 打印进度信息到标准输出，不换行
        print(f"\r{fraction} ({elapsed}), ETA: {eta}", end="")
    def _force_shutdown(self, verbose: bool = False) -> None:
        """强制关闭程序，尝试中断和必要时强制终止任务。
        我们倾向于软关闭任务，以便它们有机会在关闭前进行清理。
        """
        # 遍历所有活跃的任务，并尝试中断其进程
        for job in self._active_jobs:
            job.proc.interrupt()

        # 如果设置了 verbose 并且当前有正在处理的任务，则输出详细信息
        if verbose and self._currently_processed is not None:
            print(
                textwrap.dedent(
                    f"""
                处理以下作业失败：
                  标签：      {self._currently_processed.label}
                  自动标签：  {self._currently_processed.autolabels}
                  源命令：    {self._currently_processed.source_cmd}
            """
                ).strip()
                + "\n"
            )

        # 如果仍有活跃的任务，则休眠 0.5 秒
        if self._active_jobs:
            time.sleep(0.5)

        # 筛选出仍未退出的任务
        remaining_jobs = [j for j in self._active_jobs if j.proc.poll() is None]
        if remaining_jobs:
            print(
                f"已发送 SIGINT 信号给 {len(self._active_jobs)} 个任务，"
                f"{len(remaining_jobs)} 个任务尚未退出。\n"
                "进入短期清理循环，之后将强制终止未退出的任务。"
            )

            # 等待 2 秒钟，尝试清理未退出的任务
            for _ in range(5):
                time.sleep(2.0)
                remaining_jobs = [j for j in remaining_jobs if j.proc.poll() is None]
                if remaining_jobs:
                    print(f"仍有 {len(remaining_jobs)} 个任务未退出。")
                else:
                    print("所有剩余任务已优雅地终止。")
                    return

            # 输出仍未退出的任务数量，并强制终止它们
            print(f"{len(remaining_jobs)} 个任务拒绝退出。正在强制终止。")
            for j in remaining_jobs:
                j.proc.terminate()

    def _canary_import(self) -> None:
        """确保在启动大量工作进程之前能够导入 torch 库。"""
        # 收集所有工作项中的源命令
        source_cmds: Set[str] = set()
        for w in self._work_items:
            if w.source_cmd is not None:
                source_cmds.add(f"{w.source_cmd} && ")

        # 对每个源命令或空命令执行 torch 导入检查
        for source_cmd in source_cmds or {""}:
            cmd = f'{source_cmd}{PYTHON_CMD} -c "import torch"'
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                executable=SHELL,
            )

            # 如果导入失败，则抛出 ImportError 异常
            if proc.returncode:
                raise ImportError(
                    f"在子进程中导入 torch 失败：{cmd}\n{proc.stdout}"
                )
```