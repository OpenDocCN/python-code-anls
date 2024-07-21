# `.\pytorch\torch\utils\bottleneck\__main__.py`

```py
# mypy: allow-untyped-defs
# 引入必要的库和模块
import argparse  # 解析命令行参数的库
import cProfile  # Python 的性能分析工具
import pstats    # 用于处理和统计 cProfile 生成的统计数据
import sys       # 提供与 Python 解释器交互的函数
import os        # 提供与操作系统交互的功能
from typing import Dict  # 引入类型提示

import torch  # PyTorch 深度学习框架
from torch.autograd import profiler  # PyTorch 自动求导模块中的性能分析器
from torch.utils.collect_env import get_env_info  # PyTorch 工具包，收集环境信息

def redirect_argv(new_argv):
    # 重定向 sys.argv 到新的命令行参数列表
    sys.argv[:] = new_argv[:]

def compiled_with_cuda(sysinfo):
    # 检查是否使用 CUDA 编译
    if sysinfo.cuda_compiled_version:
        return f'compiled w/ CUDA {sysinfo.cuda_compiled_version}'
    return 'not compiled w/ CUDA'

env_summary = """
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch {pytorch_version}{debug_str} {cuda_compiled}
Running with Python {py_version} and {cuda_runtime}

`{pip_version} list` truncated output:
{pip_list_output}
""".strip()

def run_env_analysis():
    # 运行环境分析
    print('Running environment analysis...')
    info = get_env_info()

    result: Dict[str, str] = {}

    debug_str = ''
    if info.is_debug_build:
        debug_str = ' DEBUG'

    cuda_avail = ''
    if info.is_cuda_available:
        cuda = info.cuda_runtime_version
        if cuda is not None:
            cuda_avail = 'CUDA ' + cuda
    else:
        cuda = 'CUDA unavailable'

    pip_version = info.pip_version
    pip_list_output = info.pip_packages
    if pip_list_output is None:
        pip_list_output = 'Unable to fetch'

    # 构建环境信息字典
    result = {
        'debug_str': debug_str,
        'pytorch_version': info.torch_version,
        'cuda_compiled': compiled_with_cuda(info),
        'py_version': f'{sys.version_info[0]}.{sys.version_info[1]}',
        'cuda_runtime': cuda_avail,
        'pip_version': pip_version,
        'pip_list_output': pip_list_output,
    }

    # 返回格式化后的环境摘要信息
    return env_summary.format(**result)

def run_cprofile(code, globs, launch_blocking=False):
    # 运行 cProfile 对代码进行性能分析
    print('Running your script with cProfile')
    prof = cProfile.Profile()
    prof.enable()
    exec(code, globs, None)
    prof.disable()
    return prof

cprof_summary = """
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
""".strip()

def print_cprofile_summary(prof, sortby='tottime', topk=15):
    # 打印 cProfile 分析的摘要信息
    print(cprof_summary)
    cprofile_stats = pstats.Stats(prof).sort_stats(sortby)
    cprofile_stats.print_stats(topk)

def run_autograd_prof(code, globs):
    # 运行自动求导模块的性能分析
    def run_prof(use_cuda=False):
        with profiler.profile(use_cuda=use_cuda) as prof:
            exec(code, globs, None)
        return prof

    print('Running your script with the autograd profiler...')
    result = [run_prof(use_cuda=False)]
    if torch.cuda.is_available():
        result.append(run_prof(use_cuda=True))
    else:
        result.append(None)

    return result

autograd_prof_summary = """
--------------------------------------------------------------------------------
  autograd profiler output ({mode} mode)

```  
# 导入所需的模块和库
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统接口模块
import sys  # 导入系统特定参数和函数模块
import torch  # 导入PyTorch库

# 描述信息，说明脚本功能和使用autograd profiler的警告
descript = """
`bottleneck` is a tool that can be used as an initial step for debugging
bottlenecks in your program.

It summarizes runs of your script with the Python profiler and PyTorch's
autograd profiler. Because your script will be profiled, please ensure that it
exits in a finite amount of time.

For more complicated uses of the profilers, please see
https://docs.python.org/3/library/profile.html and
https://pytorch.org/docs/main/autograd.html#profiler for more information.
""".strip()

# 解析命令行参数函数
def parse_args():
    parser = argparse.ArgumentParser(description=descript)
    parser.add_argument('scriptfile', type=str,
                        help='Path to the script to be run. '
                        'Usually run with `python path/to/script`.')
    parser.add_argument('args', type=str, nargs=argparse.REMAINDER,
                        help='Command-line arguments to be passed to the script.')
    return parser.parse_args()

# 计算autograd profiler中总CPU时间的函数
def cpu_time_total(autograd_prof):
    return sum(event.cpu_time_total for event in autograd_prof.function_events)

# 主函数
def main():
    args = parse_args()  # 解析命令行参数

    # 自定义常量
    scriptfile = args.scriptfile  # 脚本文件路径
    scriptargs = [] if args.args is None else args.args  # 脚本参数列表
    scriptargs.insert(0, scriptfile)  # 将脚本文件路径作为第一个参数
    cprofile_sortby = 'tottime'  # cProfile排序方式
    cprofile_topk = 15  # cProfile显示前几个函数
    autograd_prof_sortby = 'cpu_time_total'  # autograd profiler排序方式
    autograd_prof_topk = 15  # autograd profiler显示前几个函数

    redirect_argv(scriptargs)  # 重定向命令行参数

    sys.path.insert(0, os.path.dirname(scriptfile))  # 将脚本文件所在目录添加到系统路径中
    with open(scriptfile, 'rb') as stream:
        code = compile(stream.read(), scriptfile, 'exec')  # 编译脚本文件内容
    # 定义一个全局变量字典，包含脚本文件名、模块名、包名、缓存信息等
    globs = {
        '__file__': scriptfile,  # 脚本文件名
        '__name__': '__main__',  # 模块名为主程序
        '__package__': None,     # 没有包名
        '__cached__': None,      # 没有缓存信息
    }

    # 打印描述信息
    print(descript)

    # 运行环境分析，并返回摘要信息
    env_summary = run_env_analysis()

    # 如果 CUDA 可用，则初始化 CUDA
    if torch.cuda.is_available():
        torch.cuda.init()

    # 运行 cProfile 分析代码运行情况，并返回分析结果
    cprofile_prof = run_cprofile(code, globs)

    # 运行 autograd 分析 CPU 模式和 CUDA 模式，分别返回分析结果
    autograd_prof_cpu, autograd_prof_cuda = run_autograd_prof(code, globs)

    # 打印环境摘要信息
    print(env_summary)

    # 打印 cProfile 分析摘要信息，按指定的排序方式和 top k 值
    print_cprofile_summary(cprofile_prof, cprofile_sortby, cprofile_topk)

    # 如果 CUDA 不可用，打印 CPU 模式的 autograd 分析摘要并返回
    if not torch.cuda.is_available():
        print_autograd_prof_summary(autograd_prof_cpu, 'CPU', autograd_prof_sortby, autograd_prof_topk)
        return

    # 计算 CUDA 模式下 autograd 分析的执行时间
    cuda_prof_exec_time = cpu_time_total(autograd_prof_cuda)

    # 如果 CPU 模式的 autograd 分析有函数事件记录
    if len(autograd_prof_cpu.function_events) > 0:
        # 计算 CPU 模式下 autograd 分析的执行时间
        cpu_prof_exec_time = cpu_time_total(autograd_prof_cpu)
        # 计算 CUDA 模式执行时间与 CPU 模式执行时间的百分比差异
        pct_diff = (cuda_prof_exec_time - cpu_prof_exec_time) / cuda_prof_exec_time
        # 如果差异超过 5%，则打印 CPU 模式的 autograd 分析摘要
        if abs(pct_diff) > 0.05:
            print_autograd_prof_summary(autograd_prof_cpu, 'CPU', autograd_prof_sortby, autograd_prof_topk)

    # 打印 CUDA 模式的 autograd 分析摘要
    print_autograd_prof_summary(autograd_prof_cuda, 'CUDA', autograd_prof_sortby, autograd_prof_topk)
# 如果当前脚本作为主程序执行（而不是作为模块被导入执行），则执行 main() 函数
if __name__ == '__main__':
    main()
```