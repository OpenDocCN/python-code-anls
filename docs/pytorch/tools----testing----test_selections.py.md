# `.\pytorch\tools\testing\test_selections.py`

```py
from __future__ import annotations

import math  # 导入数学库，用于数学计算
import os  # 导入操作系统库，用于操作系统相关功能
import subprocess  # 导入子进程管理库，用于执行外部命令
from pathlib import Path  # 导入路径操作库，用于处理文件路径
from typing import Callable, Sequence  # 导入类型提示库，用于类型标注

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests  # 导入自定义函数，用于测试统计
from tools.testing.test_run import ShardedTest, TestRun  # 导入自定义类，用于测试运行管理

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # 获取当前文件所在目录的上级目录的上级目录作为项目根目录路径

IS_MEM_LEAK_CHECK = os.getenv("PYTORCH_TEST_CUDA_MEM_LEAK_CHECK", "0") == "1"  # 检查环境变量，判断是否进行 CUDA 内存泄漏检查
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")  # 获取环境变量 BUILD_ENVIRONMENT，用于构建环境信息
USE_3_PROCS = "sm86" in BUILD_ENVIRONMENT or "cuda" not in BUILD_ENVIRONMENT  # 检查 BUILD_ENVIRONMENT 是否包含 sm86 或不包含 cuda，判断是否使用 3 个处理器

# NUM_PROCS_FOR_SHARDING_CALC 必须在作业的所有分片中保持一致，以确保分片的一致性。
# NUM_PROCS 是用于运行测试的实际处理器数量。如果它们不相等，则唯一的后果应该是不同的分片。
IS_ROCM = os.path.exists("/opt/rocm")  # 检查系统中是否存在 /opt/rocm 目录，判断是否为 ROCm 环境
NUM_PROCS = 1 if IS_MEM_LEAK_CHECK else 3 if USE_3_PROCS else 2  # 根据条件设置 NUM_PROCS 的值
NUM_PROCS_FOR_SHARDING_CALC = NUM_PROCS if not IS_ROCM or IS_MEM_LEAK_CHECK else 2  # 根据条件设置 NUM_PROCS_FOR_SHARDING_CALC 的值
THRESHOLD = 60 * 10  # 设置阈值为 10 分钟，以秒为单位

# See Note [ROCm parallel CI testing]
# ROCm GHA 运行器的特殊逻辑，用于查询可用的 GPU 数量。
# 无法使用 torch.version.hip 检查是否为 ROCm 自托管运行器。
# 必须用另一种方式检查 ROCm 运行器。我们查找 /opt/rocm 目录。
if IS_ROCM and not IS_MEM_LEAK_CHECK:
    try:
        # 这与 GHA 健康检查中使用的逻辑相同，请参阅 .github/templates/common.yml.j2
        lines = (
            subprocess.check_output(["rocminfo"], encoding="ascii").strip().split("\n")
        )
        count = 0
        for line in lines:
            if " gfx" in line:
                count += 1
        assert count > 0  # 至少必须有 1 个 GPU
        # 限制到 8 个 GPU（处理器）
        NUM_PROCS = min(count, 8)
    except subprocess.CalledProcessError as e:
        # 对于 ROCm GHA 运行器，安全默认值是串行运行测试。
        NUM_PROCS = 1


class ShardJob:
    def __init__(self) -> None:
        self.serial: list[ShardedTest] = []  # 初始化串行测试列表
        self.parallel: list[ShardedTest] = []  # 初始化并行测试列表

    def get_total_time(self) -> float:
        """Default is the value for which to substitute if a test has no time"""
        procs = [0.0 for _ in range(NUM_PROCS_FOR_SHARDING_CALC)]  # 创建长度为 NUM_PROCS_FOR_SHARDING_CALC 的零列表
        for test in self.parallel:  # 遍历并行测试列表
            min_index = procs.index(min(procs))  # 找到 procs 中值最小的索引
            procs[min_index] += test.get_time()  # 将测试的时间添加到最小值所在位置
        time = max(procs) + sum(test.get_time() for test in self.serial)  # 计算总时间，最大的 procs 值加上所有串行测试的时间之和
        return time

    def convert_to_tuple(self) -> tuple[float, list[ShardedTest]]:
        return (self.get_total_time(), self.serial + self.parallel)  # 返回总时间和串行、并行测试的组合列表


def get_with_pytest_shard(
    tests: Sequence[TestRun],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]] | None,
) -> list[ShardedTest]:
    sharded_tests: list[ShardedTest] = []  # 初始化分片测试列表
    # 对于测试集中的每一个测试进行处理
    for test in tests:
        # 调用函数计算测试的持续时间，使用给定的文件时间和类别时间，如果类别时间未提供则使用空字典
        duration = get_duration(test, test_file_times, test_class_times or {})

        # 如果成功计算出了持续时间，并且持续时间超过阈值
        if duration and duration > THRESHOLD:
            # 计算需要分割的片段数，向上取整
            num_shards = math.ceil(duration / THRESHOLD)
            
            # 对于每一个片段，创建一个分割后的测试对象，并加入到 sharded_tests 列表中
            for i in range(num_shards):
                sharded_tests.append(
                    ShardedTest(test, i + 1, num_shards, duration / num_shards)
                )
        else:
            # 如果持续时间未计算成功或者小于等于阈值，则创建一个未分割的测试对象并加入到 sharded_tests 列表中
            sharded_tests.append(ShardedTest(test, 1, 1, duration))
    
    # 返回所有处理后的分割测试对象列表
    return sharded_tests
def get_duration(
    test: TestRun,
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
) -> float | None:
    """Calculate the time for a TestRun based on the given test_file_times and
    test_class_times.  Returns None if the time is unknown."""
    # 获取指定测试文件的持续时间，如果未知则返回 None
    file_duration = test_file_times.get(test.test_file, None)
    # 如果是完整文件测试，则直接返回文件持续时间
    if test.is_full_file():
        return file_duration

    def get_duration_for_classes(
        test_file: str, test_classes: frozenset[str]
    ) -> float | None:
        """Calculate the total duration for given test classes within a test file."""
        duration: float = 0

        # 遍历测试类集合，累加各个类的持续时间
        for test_class in test_classes:
            class_duration = test_class_times.get(test_file, {}).get(test_class, None)
            if class_duration is None:
                return None
            duration += class_duration
        return duration

    # 获取包含的测试类和排除的测试类
    included = test.included()
    excluded = test.excluded()
    # 获取包含测试类的总持续时间和排除测试类的总持续时间
    included_classes_duration = get_duration_for_classes(test.test_file, included)
    excluded_classes_duration = get_duration_for_classes(test.test_file, excluded)

    # 如果任一类的持续时间为 None，则返回 None 表示时间未知
    if included_classes_duration is None or excluded_classes_duration is None:
        return None

    # 如果有包含的测试类，则返回包含测试类的总持续时间
    if included:
        return included_classes_duration
    # 否则，如果有排除的测试类，则返回文件持续时间减去排除测试类的总持续时间
    assert (
        excluded
    ), f"TestRun {test} is not full file but doesn't have included or excluded classes"
    if file_duration is None:
        return None
    return file_duration - excluded_classes_duration


def shard(
    sharded_jobs: list[ShardJob],
    pytest_sharded_tests: Sequence[ShardedTest],
    estimated_time_limit: float | None = None,
    serial: bool = False,
) -> None:
    """Divide sharded_jobs among available pytest_sharded_tests based on time limits."""
    # 修改 sharded_jobs 列表，实现就地修改
    if len(sharded_jobs) == 0:
        # 如果没有分片作业，但有需要分片的测试，则断言失败
        assert (
            len(pytest_sharded_tests) == 0
        ), "No shards provided but there are tests to shard"
        return

    round_robin_index = 0

    def _get_min_sharded_job(
        sharded_jobs: list[ShardJob], test: ShardedTest
    ) -> ShardJob:
        """Return the ShardJob with the least total time among sharded_jobs."""
        if test.time is None:
            nonlocal round_robin_index
            # 如果测试时间未知，则按照轮询顺序选择分片作业
            job = sharded_jobs[round_robin_index % len(sharded_jobs)]
            round_robin_index += 1
            return job
        # 否则，选择总时间最少的分片作业
        return min(sharded_jobs, key=lambda j: j.get_total_time())

    def _shard_serial(
        tests: Sequence[ShardedTest], sharded_jobs: list[ShardJob]
    ) -> None:
        """Assign tests to sharded_jobs in a serial manner."""
        # 断言需要提供预估的时间限制
        assert estimated_time_limit is not None, "Estimated time limit must be provided"
        new_sharded_jobs = sharded_jobs
        # 遍历测试列表，分配测试到分片作业中
        for test in tests:
            if (
                len(sharded_jobs) > 1
                and sharded_jobs[-1].get_total_time() > estimated_time_limit
            ):
                new_sharded_jobs = sharded_jobs[:-1]
            min_sharded_job = _get_min_sharded_job(new_sharded_jobs, test)
            min_sharded_job.serial.append(test)

    def _shard_parallel(
        tests: Sequence[ShardedTest], sharded_jobs: list[ShardJob]
    ) -> None:
        # 函数定义：将测试用例按照最小分片作业分配到对应的并行作业中
        for test in tests:
            # 获取当前测试用例应分配的最小分片作业
            min_sharded_job = _get_min_sharded_job(sharded_jobs, test)
            # 将当前测试用例添加到最小分片作业的并行执行列表中
            min_sharded_job.parallel.append(test)

    # 如果选择串行执行
    if serial:
        # 调用串行分片函数，将测试用例分片到作业中
        _shard_serial(pytest_sharded_tests, sharded_jobs)
    else:
        # 否则，调用并行分片函数，将测试用例分片到作业中
        _shard_parallel(pytest_sharded_tests, sharded_jobs)

    # 函数执行完毕，返回到调用者处
    return
def calculate_shards(
    num_shards: int,
    tests: Sequence[TestRun],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]] | None,
    must_serial: Callable[[str], bool] | None = None,
    sort_by_time: bool = True,
) -> list[tuple[float, list[ShardedTest]]]:
    # 如果未提供 must_serial 函数，则使用默认函数，始终返回 True
    must_serial = must_serial or (lambda x: True)
    # 如果 test_class_times 为 None，则设为空字典
    test_class_times = test_class_times or {}

    # 将测试用例分割成 pytest 的分片
    if sort_by_time:
        # 获取已知耗时的测试用例列表
        known_tests = [
            x
            for x in tests
            if get_duration(x, test_file_times, test_class_times) is not None
        ]
        # 获取未知耗时的测试用例列表
        unknown_tests = [x for x in tests if x not in known_tests]

        # 将已知耗时的测试用例按照时间排序，降序排列
        pytest_sharded_tests = sorted(
            get_with_pytest_shard(known_tests, test_file_times, test_class_times),
            key=lambda j: j.get_time(),
            reverse=True,
        ) + get_with_pytest_shard(unknown_tests, test_file_times, test_class_times)
    else:
        # 不按时间排序，直接使用 pytest 的分片函数获取结果
        pytest_sharded_tests = get_with_pytest_shard(
            tests, test_file_times, test_class_times
        )
    # 删除 tests 变量，释放内存
    del tests

    # 将需要串行执行的测试用例放入 serial_tests 列表
    serial_tests = [test for test in pytest_sharded_tests if must_serial(test.name)]
    # 将可以并行执行的测试用例放入 parallel_tests 列表
    parallel_tests = [test for test in pytest_sharded_tests if test not in serial_tests]

    # 计算串行执行测试用例的总耗时
    serial_time = sum(test.get_time() for test in serial_tests)
    # 计算并行执行测试用例的总耗时
    parallel_time = sum(test.get_time() for test in parallel_tests)
    # 计算总耗时，除以可用的处理器数量，得到每个分片的预估时间
    total_time = serial_time + parallel_time / NUM_PROCS_FOR_SHARDING_CALC
    estimated_time_per_shard = total_time / num_shards

    # 计算估计的最小时间限制，以便尽可能分离串行和并行测试用例
    estimated_time_limit = 0.0
    if estimated_time_per_shard != 0:
        estimated_time_limit = serial_time % estimated_time_per_shard
    if estimated_time_limit <= 0.01:
        estimated_time_limit = estimated_time_per_shard
    if total_time == 0:
        num_serial_shards = num_shards
    else:
        # 计算需要的串行分片数量，确保每个分片尽可能填充串行测试用例
        num_serial_shards = max(math.ceil(serial_time / total_time * num_shards), 1)

    # 创建 num_shards 个 ShardJob 对象，用于分片执行测试用例
    sharded_jobs = [ShardJob() for _ in range(num_shards)]
    # 将串行执行的测试用例分配到前 num_serial_shards 个分片中
    shard(
        sharded_jobs=sharded_jobs[:num_serial_shards],
        pytest_sharded_tests=serial_tests,
        estimated_time_limit=estimated_time_limit,
        serial=True,
    )
    # 使用shard函数执行分片作业，传入参数包括分片的作业列表(sharded_jobs)、并行测试的pytest_sharded_tests列表、以及设置serial参数为False。
    shard(
        sharded_jobs=sharded_jobs,
        pytest_sharded_tests=parallel_tests,
        serial=False,
    )
    
    # 将sharded_jobs列表中的每个作业(job)转换为元组(tuple)的形式，并将所有转换后的元组组成一个列表返回。
    return [job.convert_to_tuple() for job in sharded_jobs]
# 获取测试用例配置信息的函数，参数为测试用例文件夹路径，无返回值
def get_test_case_configs(dirpath: str) -> None:
    # 调用函数获取慢速测试用例配置信息，传入文件夹路径作为参数
    get_slow_tests(dirpath=dirpath)
    # 调用函数获取禁用的测试用例配置信息，传入文件夹路径作为参数
    get_disabled_tests(dirpath=dirpath)
```