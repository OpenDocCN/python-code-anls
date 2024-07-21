# `.\pytorch\torch\utils\benchmark\utils\compile.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型定义，允许未定义的函数
from typing import Any, Callable, cast, List, Optional, Union

# 导入 PyTorch 库
import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer

# 定义可以被导出的模块成员列表
__all__ = ["bench_all", "benchmark_compile"]

# 初始设定，用于记录是否已警告过 Tensor Cores 功能
_warned_tensor_cores = False
# 获取默认的 float32 精度设置
_default_float_32_precision = torch.get_float32_matmul_precision()

# 尝试导入 tabulate 库
try:
    from tabulate import tabulate
    # 标记 tabulate 库已成功导入
    HAS_TABULATE = True
except ModuleNotFoundError:
    # 若未能导入 tabulate 库，标记未安装
    HAS_TABULATE = False
    # 设置 tabulate 为 None，类型标注忽略赋值错误
    tabulate = None  # type: ignore[assignment]
    # 打印错误消息，提示安装 tabulate 库以使用此工具
    print("tabulate is not installed, please pip install tabulate to use this utility")

# 若成功导入 tabulate 库，则定义以下函数
if HAS_TABULATE:
    def _enable_tensor_cores():
        # 全局变量，记录是否已警告过 Tensor Cores 功能
        global _warned_tensor_cores

        # 若 CUDA 可用
        if torch.cuda.is_available():
            # 若禁用了 TF32 并且 GPU 设备支持 Tensor Cores
            if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
                # 设置 float32 矩阵乘法的精度为高精度模式
                torch.set_float32_matmul_precision("high")
                # 若未曾警告过，打印提示信息
                if not _warned_tensor_cores:
                    print("Your GPU supports tensor cores")
                    print("we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
                    # 标记已警告过
                    _warned_tensor_cores = True

    def _disable_tensor_cores():
        # 恢复默认的 float32 矩阵乘法精度设置
        torch.set_float32_matmul_precision(_default_float_32_precision)

    def bench_loop(
        model: Union[torch.nn.Module, Callable],
        sample_input: Union[torch.Tensor, Any],
        num_iters: int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
    ):
        # 定义基准测试的语句和设置
        if optimizer and loss_fn:
            # 训练模式下的语句
            stmt = """
    output = model(sample_input)
    loss = loss_fn(output) if loss_fn else output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            """
        else:
            # 推断模式下的语句
            stmt = "model(sample_input)"

        # 创建 Timer 对象，用于性能测量
        timer = Timer(
            stmt=stmt,
            globals={"model": model, "sample_input": sample_input, "optimizer": optimizer, "loss_fn": loss_fn},
        )

        # 进行 num_iters 次测试，并获取平均每次迭代的时间
        result = timer.timeit(number=num_iters)

        # 将平均时间转换为毫秒并保留两位小数
        avg_time = result.mean * 1000
        return round(avg_time, 2)

    def benchmark_compile(
        model: Union[torch.nn.Module, Callable],
        sample_input: Union[torch.Tensor, Any],
        num_iters: int = 5,
        backend: Optional[str] = None,
        mode: Optional[str] = "default",
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn : Union[torch.nn.Module, Callable, None] = None,
    ):
        """
        Use this utility to benchmark torch.compile
        """
        # 如果指定了后端，执行以下操作
        if backend:
            try:
                # 重置 torch._dynamo
                torch._dynamo.reset()
                # 创建 CompileCounterWithBackend 对象，用指定后端编译模型
                compile_counter_with_backend = CompileCounterWithBackend(backend)
                # 使用指定后端编译模型，返回优化后的模型
                opt_model = torch.compile(model, backend=compile_counter_with_backend, mode=mode)

                # 编译只会在第一次推断之后发生
                # 进行编译时间的基准测试
                compilation_time = bench_loop(opt_model, sample_input, 1, optimizer, loss_fn)

                # 运行时间的基准测试
                running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

                # 如果在基准测试期间没有发生编译，则抛出异常
                if compile_counter_with_backend.frame_count == 0:
                    raise RuntimeError("No compilation occurred during benchmarking.")

                # 如果基准测试期间发生重新编译，则抛出异常
                if compile_counter_with_backend.frame_count > 1:
                    raise RuntimeError("Recompilation occurred during benchmarking.")

            except Exception as e:
                # 捕获并打印异常信息
                print(e)
                print(f"Failed to compile {backend} with mode {mode}")
                return None, None
        else:
            # 如果未指定后端，则使用原始模型
            opt_model = model
            compilation_time = None
            # 运行时间的基准测试
            running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

        # 对编译时间和运行时间进行四舍五入处理
        compilation_time = round(compilation_time, 2) if compilation_time else None
        running_time = round(running_time, 2) if running_time else None

        # 返回编译时间和运行时间
        return compilation_time, running_time


    def bench_all(
        model : Union[torch.nn.Module, Callable],
        sample_input: Union[torch.Tensor, Any],
        num_iters : int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn : Union[torch.nn.Module, Callable, None] = None,
```