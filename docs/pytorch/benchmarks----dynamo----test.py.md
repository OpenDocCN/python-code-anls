# `.\pytorch\benchmarks\dynamo\test.py`

```
import os  # 导入操作系统模块
import unittest  # 导入单元测试模块

from .common import parse_args, run  # 导入自定义模块中的函数
from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner  # 导入自定义模块中的函数和类

try:
    # 仅在 fbcode 中存在时导入，用于检测是否为 ASAN 或 TSAN
    from aiplatform.utils.sanitizer_status import is_asan_or_tsan
except ImportError:

    def is_asan_or_tsan():
        return False  # 如果导入失败，则定义一个始终返回 False 的函数


class TestDynamoBenchmark(unittest.TestCase):  # 定义单元测试类 TestDynamoBenchmark，继承自 unittest.TestCase
    @unittest.skipIf(is_asan_or_tsan(), "ASAN/TSAN not supported")  # 根据 is_asan_or_tsan 函数的返回值决定是否跳过测试
    def test_benchmark_infra_runs(self) -> None:
        """
        Basic smoke test that TorchBench runs.

        This test is mainly meant to check that our setup in fbcode
        doesn't break.

        If you see a failure here related to missing CPP headers, then
        you likely need to update the resources list in:
            //caffe2:inductor
        """
        original_dir = setup_torchbench_cwd()  # 设置 TorchBench 的当前工作目录，并保存原始目录
        try:
            args = parse_args(
                [
                    "-dcpu",  # 参数：使用 CPU
                    "--inductor",  # 参数：使用 Inductor
                    "--training",  # 参数：执行训练
                    "--performance",  # 参数：性能测试
                    "--only=BERT_pytorch",  # 参数：仅运行 BERT_pytorch 测试
                    "-n1",  # 参数：单次运行
                    "--batch-size=1",  # 参数：批大小为 1
                ]
            )
            run(TorchBenchmarkRunner(), args, original_dir)  # 运行 TorchBench，传入参数和原始目录
        finally:
            os.chdir(original_dir)  # 在最终步骤中切换回原始目录
```