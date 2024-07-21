# `.\pytorch\test\inductor\test_inductor_utils.py`

```
# Owner(s): ["module: inductor"]

# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于创建偏函数
import logging    # 导入 logging 模块，用于记录日志信息

import torch  # 导入 PyTorch 模块
from torch._inductor.runtime.runtime_utils import do_bench  # 从 torch._inductor.runtime.runtime_utils 模块导入 do_bench 函数

from torch._inductor.test_case import run_tests, TestCase  # 从 torch._inductor.test_case 模块导入 run_tests 函数和 TestCase 类

from torch._inductor.utils import do_bench_using_profiling  # 从 torch._inductor.utils 模块导入 do_bench_using_profiling 函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class TestBench(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # 调用父类的 setUpClass 方法
        x = torch.rand(1024, 10).cuda().half()  # 在 CUDA 上生成一个随机的半精度张量 x
        w = torch.rand(512, 10).cuda().half()   # 在 CUDA 上生成一个随机的半精度张量 w
        cls._bench_fn = functools.partial(torch.nn.functional.linear, x, w)  # 使用 functools.partial 创建一个偏函数 _bench_fn，对应 torch.nn.functional.linear 函数

    def test_do_bench(self):
        res = do_bench(self._bench_fn)  # 调用 do_bench 函数执行基准测试，并得到结果 res
        log.warning("do_bench result: %s", res)  # 记录基准测试的结果到日志中
        self.assertGreater(res, 0)  # 使用 self.assertGreater 方法断言 res 大于 0

    def test_do_bench_using_profiling(self):
        res = do_bench_using_profiling(self._bench_fn)  # 调用 do_bench_using_profiling 函数执行基准测试，并得到结果 res
        log.warning("do_bench_using_profiling result: %s", res)  # 记录基准测试的结果到日志中
        self.assertGreater(res, 0)  # 使用 self.assertGreater 方法断言 res 大于 0


if __name__ == "__main__":
    run_tests("cuda")  # 如果脚本被直接执行，则运行 "cuda" 环境下的测试用例
```