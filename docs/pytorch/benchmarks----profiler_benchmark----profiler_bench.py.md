# `.\pytorch\benchmarks\profiler_benchmark\profiler_bench.py`

```py
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
import sys  # 导入系统相关的模块
import timeit  # 导入用于测量代码执行时间的模块

import torch  # 导入PyTorch库

from torch.utils.benchmark import Timer  # 从PyTorch中导入计时器模块

# 并行任务数
PARALLEL_TASKS_NUM = 4
# 内部迭代次数，初始设为None
INTERNAL_ITER = None


# 循环负载函数定义，接收一个张量x作为参数
def loop_workload(x):
    # 使用内部迭代次数进行循环
    for i in range(INTERNAL_ITER):
        x = torch.mm(x, x)  # 执行张量的矩阵乘法
    return x


# 并行负载函数定义，接收一个张量x作为参数
def parallel_workload(x):
    # 并行任务函数定义，接收一个张量x作为参数
    def parallel_task(x):
        # 根据内部迭代次数的四分之一进行循环
        for i in range(int(INTERNAL_ITER / PARALLEL_TASKS_NUM)):
            x = torch.mm(x, x)  # 执行张量的矩阵乘法
        return x

    futs = []
    # 创建并行任务数目的子任务
    for i in range(PARALLEL_TASKS_NUM):
        futs.append(torch.jit._fork(parallel_task, x))  # 并行执行子任务
    # 等待所有子任务完成
    for i in range(PARALLEL_TASKS_NUM):
        torch.jit._wait(futs[i])
    return x


if __name__ == "__main__":
    # 禁用图执行优化
    torch._C._set_graph_executor_optimize(False)
    # 创建参数解析器对象，描述为Profiler benchmark
    parser = argparse.ArgumentParser(description="Profiler benchmark")

    # 添加命令行参数
    parser.add_argument("--with-cuda", "--with_cuda", action="store_true")  # 使用CUDA加速
    parser.add_argument("--with-stack", "--with_stack", action="store_true")  # 使用堆栈信息
    parser.add_argument("--use-script", "--use_script", action="store_true")  # 使用脚本模式
    parser.add_argument("--use-kineto", "--use_kineto", action="store_true")  # 使用Kineto性能分析
    parser.add_argument(
        "--profiling-tensor-size", "--profiling_tensor_size", default=1, type=int
    )  # 性能分析张量大小
    parser.add_argument("--workload", "--workload", default="loop", type=str)  # 执行负载类型
    parser.add_argument("--internal-iter", "--internal_iter", default=256, type=int)  # 内部迭代次数
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=10, type=int
    )  # 计时器最小运行时间
    parser.add_argument("--cuda-only", "--cuda_only", action="store_true")  # 仅使用CUDA加速

    # 解析命令行参数
    args = parser.parse_args()

    # 如果使用CUDA加速但CUDA不可用，则打印提示信息并退出程序
    if args.with_cuda and not torch.cuda.is_available():
        print("No CUDA available")
        sys.exit()

    # 打印负载类型、内部迭代次数和计时器最小运行时间的信息
    print(
        f"Payload: {args.workload}, {args.internal_iter} iterations; timer min. runtime = {args.timer_min_run_time}\n"
    )
    # 将全局变量INTERNAL_ITER设为命令行参数指定的内部迭代次数
    INTERNAL_ITER = args.internal_iter
    # 遍历两种性能分析情况：禁用和启用
    for profiling_enabled in [False, True]:
        # 打印性能分析状态及相关配置信息
        print(
            "Profiling {}, tensor size {}x{}, use cuda: {}, use kineto: {}, with stacks: {}, use script: {}".format(
                "enabled" if profiling_enabled else "disabled",
                args.profiling_tensor_size,
                args.profiling_tensor_size,
                args.with_cuda,
                args.use_kineto,
                args.with_stack,
                args.use_script,
            )
        )

        # 创建一个指定大小的随机张量
        input_x = torch.rand(args.profiling_tensor_size, args.profiling_tensor_size)

        # 如果使用 CUDA，将输入张量移动到 GPU 上
        if args.with_cuda:
            input_x = input_x.cuda()

        # 根据参数选择工作负载类型
        workload = None
        assert args.workload in ["loop", "parallel"]
        if args.workload == "loop":
            workload = loop_workload
        else:
            workload = parallel_workload

        # 如果使用脚本模式，对工作负载进行追踪
        if args.use_script:
            traced_workload = torch.jit.trace(workload, (input_x,))
            workload = traced_workload

        # 如果启用性能分析
        if profiling_enabled:

            # 定义一个执行负载的函数，使用 PyTorch 自带的性能分析器
            def payload():
                x = None
                with torch.autograd.profiler.profile(
                    use_cuda=args.with_cuda,
                    with_stack=args.with_stack,
                    use_kineto=args.use_kineto,
                    use_cpu=not args.cuda_only,
                ) as prof:
                    x = workload(input_x)
                return x

        else:

            # 定义一个执行负载的函数，不进行性能分析
            def payload():
                return workload(input_x)

        # 使用 Timer 来执行并测量 payload 函数的运行时间
        t = Timer(
            "payload()",
            globals={"payload": payload},
            timer=timeit.default_timer,
        ).blocked_autorange(min_run_time=args.timer_min_run_time)
        # 打印计时结果
        print(t)
```