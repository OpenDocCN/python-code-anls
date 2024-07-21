# `.\pytorch\test\test_functional_autograd_benchmark.py`

```
# Owner(s): ["module: autograd"]

# 导入必要的库
import os
import subprocess
import tempfile
import unittest

# 从 torch.testing._internal.common_utils 导入所需的函数和变量
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    slowTest,
    TestCase,
)

# 环境变量 PYTORCH_COLLECT_COVERAGE 是否设置为真
PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))


# 这是一个对 functional autograd benchmark 脚本的简单 smoke test。
class TestFunctionalAutogradBenchmark(TestCase):

    # 内部方法，运行指定模型的测试
    def _test_runner(self, model, disable_gpu=False):
        # 注意：关于 Windows 平台的说明：
        # 临时文件由当前进程独占打开，子进程不得再次打开。由于这是一个简单的 smoke test，
        # 目前选择不在 Windows 上运行此测试，以保持代码简单。
        with tempfile.NamedTemporaryFile() as out_file:
            # 构建命令行参数
            cmd = [
                "python3",
                "../benchmarks/functional_autograd_benchmark/functional_autograd_benchmark.py",
            ]
            # 只运行预热
            cmd += ["--num-iters", "0"]
            # 只运行 vjp 任务（最快的任务）
            cmd += ["--task-filter", "vjp"]
            # 只运行指定的模型
            cmd += ["--model-filter", model]
            # 输出文件
            cmd += ["--output", out_file.name]
            if disable_gpu:
                # 禁用 GPU
                cmd += ["--gpu", "-1"]

            # 执行命令
            res = subprocess.run(cmd)

            # 断言命令返回码为 0
            self.assertTrue(res.returncode == 0)
            # 检查文件是否有写入内容
            out_file.seek(0, os.SEEK_END)
            self.assertTrue(out_file.tell() > 0)

    # 跳过 Windows 平台下 NamedTemporaryFile 不支持的特性
    @unittest.skipIf(
        IS_WINDOWS,
        "NamedTemporaryFile on windows does not have all the features we need.",
    )
    # 如果设置了 PYTORCH_COLLECT_COVERAGE，跳过测试以避免 gcov 死锁问题
    @unittest.skipIf(
        PYTORCH_COLLECT_COVERAGE,
        "Can deadlocks with gcov, see https://github.com/pytorch/pytorch/issues/49656",
    )
    # 测试快速任务
    def test_fast_tasks(self):
        fast_tasks = [
            "resnet18",
            "ppl_simple_reg",
            "ppl_robust_reg",
            "wav2letter",
            "transformer",
            "multiheadattn",
        ]

        for task in fast_tasks:
            self._test_runner(task)

    # 标记为慢速测试
    @slowTest
    # 跳过 Windows 平台下 NamedTemporaryFile 不支持的特性
    @unittest.skipIf(
        IS_WINDOWS,
        "NamedTemporaryFile on windows does not have all the features we need.",
    )
    # 测试慢速任务
    def test_slow_tasks(self):
        slow_tasks = ["fcn_resnet", "detr"]
        # deepspeech 被自愿排除，因为在没有正确调整线程数量的情况下运行时间过长。

        for task in slow_tasks:
            # 禁用 GPU 以便在 CI GPU 上内存不足时能够运行慢速测试
            self._test_runner(task, disable_gpu=True)


# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```