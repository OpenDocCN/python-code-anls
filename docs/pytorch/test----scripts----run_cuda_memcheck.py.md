# `.\pytorch\test\scripts\run_cuda_memcheck.py`

```
# 设置脚本使用 Python3 解释器运行
#!/usr/bin/env python3

# 本脚本运行 cuda-memcheck 在指定的单元测试上。每个测试用例在其独立的进程中运行，并带有超时设置，
# 以确保：
# 1) 不同的测试用例互不影响，
# 2) 在出现挂起情况时，脚本仍能在有限时间内完成执行。
# 输出将写入日志文件 result.log

"""This script runs cuda-memcheck on the specified unit test. Each test case
is run in its isolated process with a timeout so that:
1) different test cases won't influence each other, and
2) in case of hang, the script would still finish in a finite amount of time.
The output will be written to a log file result.log

Example usage:
    python run_cuda_memcheck.py ../test_torch.py 600

Note that running cuda-memcheck could be very slow.
"""

# 导入必要的库和模块
import argparse  # 导入用于解析命令行参数的模块
import asyncio   # 异步 IO 支持库
import multiprocessing  # 多进程处理库
import os        # 操作系统功能模块
import subprocess  # 启动和管理子进程
import sys       # 系统相关的参数和功能

# 导入额外的自定义模块和第三方库
import cuda_memcheck_common as cmc  # 导入 cuda-memcheck 公共函数和常量
import tqdm      # 导入进度条模块

import torch     # 导入 PyTorch 深度学习框架

# 初始化测试用例列表为空列表
ALL_TESTS = []

# 获取系统中可用的 GPU 数量
GPUS = torch.cuda.device_count()

# 解析命令行参数
# 创建参数解析器对象，描述为“运行独立的 cuda-memcheck 在单元测试上”
parser = argparse.ArgumentParser(description="Run isolated cuda-memcheck on unit tests")
# 添加位置参数，用于指定测试用例的 Python 文件名
parser.add_argument(
    "filename", help="the python file for a test, such as test_torch.py"
)
# 添加超时时间参数，以秒为单位，如果测试用例未在指定时间内终止，则强制终止
parser.add_argument(
    "timeout",
    type=int,
    help="kill the test if it does not terminate in a certain amount of seconds",
)
# 添加可选参数 --strict，用于控制是否显示 cublas/cudnn 错误
parser.add_argument(
    "--strict",
    action="store_true",
    help="Whether to show cublas/cudnn errors. These errors are ignored by default because"
    "cublas/cudnn does not run error-free under cuda-memcheck, and ignoring these errors",
)
# 添加可选参数 --nproc，用于指定运行测试的进程数，默认为系统中的 CPU 核心数
parser.add_argument(
    "--nproc",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of processes running tests, default to number of cores in the system",
)
# 添加可选参数 --gpus，用于指定每个进程分配的 GPU，可以是 "all"，或以冒号分隔的 GPU 列表字符串
parser.add_argument(
    "--gpus",
    default="all",
    help='GPU assignments for each process, it could be "all", or : separated list like "1,2:3,4:5,6"',
)
# 添加可选参数 --ci，用于指示脚本是否在 CI 环境中运行
parser.add_argument(
    "--ci",
    action="store_true",
    help="Whether this script is executed in CI. When executed inside a CI, this script fails when "
    "an error is detected. Also, it will not show tqdm progress bar, but directly print the error"
    "to stdout instead.",
)
# 添加可选参数 --nohang，用于将超时视为成功的标志
parser.add_argument("--nohang", action="store_true", help="Treat timeout as success")
# 添加可选参数 --split，用于将作业拆分为多个部分
parser.add_argument("--split", type=int, default=1, help="Split the job into pieces")
# 添加可选参数 --rank，用于指定当前进程应该选择的作业部分
parser.add_argument(
    "--rank", type=int, default=0, help="Which piece this process should pick"
)
# 解析所有的命令行参数，并存储到 args 对象中
args = parser.parse_args()

# 过滤器函数，用于忽略 cublas/cudnn 错误
# TODO (@zasdfgbnm): When can we remove this? Will cublas/cudnn run error-free under cuda-memcheck?
def is_ignored_only(output):
    try:
        # 尝试解析 cuda-memcheck 输出报告
        report = cmc.parse(output)
    except cmc.ParseError:
        # 如果简单解析器无法解析 cuda memcheck 的输出，则不忽略错误
        return False
    count_ignored_errors = 0
    # 遍历报告中的每个错误
    for e in report.errors:
        # 如果错误堆栈中包含 cublas、cudnn 或 cufft 相关信息，则将其计数为被忽略的错误
        if (
            "libcublas" in "".join(e.stack)
            or "libcudnn" in "".join(e.stack)
            or "libcufft" in "".join(e.stack)
        ):
            count_ignored_errors += 1
    # 判断是否所有错误都被忽略
    return count_ignored_errors == report.num_errors

# 设置环境变量 PYTORCH_CUDA_MEMCHECK=1，以便跳过某些测试
os.environ["PYTORCH_CUDA_MEMCHECK"] = "1"
# Discover tests:
# To get a list of tests, run:
# pytest --setup-only test/test_torch.py
# and then parse the output
proc = subprocess.Popen(
    ["pytest", "--setup-only", args.filename],  # 启动一个子进程来运行 pytest 命令，只执行设置阶段，并指定文件名参数
    stdout=subprocess.PIPE,  # 捕获子进程的标准输出
    stderr=subprocess.PIPE,  # 捕获子进程的标准错误输出
)
stdout, stderr = proc.communicate()  # 获取子进程的输出内容
lines = stdout.decode().strip().splitlines()  # 将标准输出解码为字符串，并按行拆分保存到列表 lines 中
for line in lines:
    if "(fixtures used:" in line:  # 检查每行输出，定位包含 "(fixtures used:" 的行
        line = line.strip().split()[0]  # 去除首尾空格并按空格拆分，取第一个元素
        line = line[line.find("::") + 2 :]  # 找到 "::" 的位置，取其后的子字符串
        line = line.replace("::", ".")  # 将 "::" 替换为 "."
        ALL_TESTS.append(line)  # 将处理后的测试名称添加到 ALL_TESTS 列表中


# Do a simple filtering:
# if 'cpu' or 'CPU' is in the name and 'cuda' or 'CUDA' is not in the name, then skip it
def is_cpu_only(name):
    name = name.lower()  # 将测试名称转换为小写
    return ("cpu" in name) and "cuda" not in name  # 判断测试名称中是否包含 "cpu" 且不包含 "cuda"


ALL_TESTS = [x for x in ALL_TESTS if not is_cpu_only(x)]  # 根据 is_cpu_only 函数的结果过滤 ALL_TESTS 列表


# Split all tests into chunks, and only on the selected chunk
ALL_TESTS.sort()  # 对 ALL_TESTS 列表进行排序
chunk_size = (len(ALL_TESTS) + args.split - 1) // args.split  # 计算每个分块的大小
start = chunk_size * args.rank  # 计算当前进程开始处理的索引位置
end = chunk_size * (args.rank + 1)  # 计算当前进程结束处理的索引位置
ALL_TESTS = ALL_TESTS[start:end]  # 根据进程的排名选择处理的测试名称范围


# Run tests:
# Since running cuda-memcheck on PyTorch unit tests is very slow, these tests must be run in parallel.
# This is done by using the coroutine feature in new Python versions.  A number of coroutines are created;
# they create subprocesses and awaiting them to finish. The number of running subprocesses could be
# specified by the user and by default is the same as the number of CPUs in the machine.
# These subprocesses are balanced across different GPUs on the system by assigning one devices per process,
# or as specified by the user
progress = 0  # 初始化进度计数器为 0
if not args.ci:  # 如果不是在 CI 环境下
    logfile = open("result.log", "w")  # 打开一个用于写入结果的日志文件
    progressbar = tqdm.tqdm(total=len(ALL_TESTS))  # 使用 tqdm 创建一个进度条，总数为 ALL_TESTS 的长度
else:  # 如果在 CI 环境下
    logfile = sys.stdout  # 将日志输出到标准输出
    # create a fake progress bar that does not display anything
    class ProgressbarStub:
        def update(self, *args):
            return
    progressbar = ProgressbarStub()  # 创建一个虚拟的进度条对象，不显示任何内容


async def run1(coroutine_id):
    global progress  # 声明引用全局变量 progress

    if args.gpus == "all":  # 如果指定了使用所有可用 GPU
        gpuid = coroutine_id % GPUS  # 根据协程 ID 计算使用的 GPU ID
    else:  # 如果指定了具体的 GPU 分配方式
        gpu_assignments = args.gpus.split(":")  # 按 ":" 拆分 GPU 分配参数
        assert args.nproc == len(
            gpu_assignments
        ), "Please specify GPU assignment for each process, separated by :"  # 断言检查 GPU 分配参数的数量是否与进程数相匹配
        gpuid = gpu_assignments[coroutine_id]  # 根据协程 ID 选择对应的 GPU ID
    # 当进度小于所有测试的数量时，执行循环
    while progress < len(ALL_TESTS):
        # 获取当前进度位置的测试名称
        test = ALL_TESTS[progress]
        # 进度加一，准备下一个测试
        progress += 1
        # 构建运行 CUDA 内存检查的命令，指定使用的 GPU 设备和测试文件
        cmd = f"CUDA_VISIBLE_DEVICES={gpuid} cuda-memcheck --error-exitcode 1 python {args.filename} {test}"
        # 创建异步子进程来执行命令，捕获其标准输出和标准错误
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            # 等待子进程执行完成或超时，获取其输出和错误信息
            stdout, stderr = await asyncio.wait_for(proc.communicate(), args.timeout)
        except asyncio.TimeoutError:
            # 如果超时，记录超时信息到日志文件，杀死子进程并根据参数决定是否退出程序
            print("Timeout:", test, file=logfile)
            proc.kill()
            if args.ci and not args.nohang:
                sys.exit("Hang detected on cuda-memcheck")
        else:
            # 如果子进程返回码为0，表示测试成功，记录成功信息到日志文件
            if proc.returncode == 0:
                print("Success:", test, file=logfile)
            else:
                # 如果返回码不为0，解码标准输出和标准错误，并根据严格模式或输出内容判断是否显示失败信息
                stdout = stdout.decode()
                stderr = stderr.decode()
                should_display = args.strict or not is_ignored_only(stdout)
                if should_display:
                    # 记录测试失败信息及其输出到日志文件，并根据参数决定是否退出程序
                    print("Fail:", test, file=logfile)
                    print(stdout, file=logfile)
                    print(stderr, file=logfile)
                    if args.ci:
                        sys.exit("Failure detected on cuda-memcheck")
                else:
                    # 如果测试被忽略，则记录忽略信息到日志文件
                    print("Ignored:", test, file=logfile)
        # 删除子进程对象，准备处理下一个测试
        del proc
        # 更新进度条，表示处理了一个测试
        progressbar.update(1)
# 定义一个异步函数 `main()`，用于管理并发任务
async def main():
    # 创建一个任务列表，每个任务使用 `run1(i)` 函数并发执行，范围是 `args.nproc` 的取值
    tasks = [asyncio.ensure_future(run1(i)) for i in range(args.nproc)]
    # 遍历所有任务
    for t in tasks:
        # 等待每个任务完成
        await t

# 如果脚本作为主程序执行
if __name__ == "__main__":
    # 获取当前事件循环
    loop = asyncio.get_event_loop()
    # 运行异步函数 `main()` 直到其完成所有任务
    loop.run_until_complete(main())
```