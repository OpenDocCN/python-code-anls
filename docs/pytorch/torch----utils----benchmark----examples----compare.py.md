# `.\pytorch\torch\utils\benchmark\examples\compare.py`

```
# mypy: allow-untyped-defs
"""Example of Timer and Compare APIs:

$ python -m examples.compare
"""

import pickle
import sys
import time

import torch

import torch.utils.benchmark as benchmark_utils


class FauxTorch:
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element):
        # 初始化 FauxTorch 类，接收真实的 torch 对象和额外的每个元素的时间开销（以纳秒为单位）
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    def extra_overhead(self, result):
        # 如果 numel（张量元素数）大于5000，则休眠时间为每个元素的额外开销乘以元素数（以秒为单位）
        numel = int(result.numel())
        if numel > 5000:
            time.sleep(numel * self._extra_ns_per_element * 1e-9)
        return result

    def add(self, *args, **kwargs):
        # 执行真实的 torch.add，并加上额外开销
        return self.extra_overhead(self._real_torch.add(*args, **kwargs))

    def mul(self, *args, **kwargs):
        # 执行真实的 torch.mul，并加上额外开销
        return self.extra_overhead(self._real_torch.mul(*args, **kwargs))

    def cat(self, *args, **kwargs):
        # 执行真实的 torch.cat，并加上额外开销
        return self.extra_overhead(self._real_torch.cat(*args, **kwargs))

    def matmul(self, *args, **kwargs):
        # 执行真实的 torch.matmul，并加上额外开销
        return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))


def main():
    tasks = [
        ("add", "add", "torch.add(x, y)"),
        ("add", "add (extra +0)", "torch.add(x, y + zero)"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": torch.ones((size, 4)),
                "y": torch.ones((1, 4)),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 5)]
        for label, sub_label, stmt in tasks
        for size in [1, 10, 100, 1000, 10000, 50000]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        # 使用 blocked_autorange 方法运行定时器，并将结果序列化后存入列表
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    # 使用 pickle 加载所有序列化结果，并进行比较
    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    # 打印未格式化的比较结果
    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    # 打印格式化后的比较结果
    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    main()
```