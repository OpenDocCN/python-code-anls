# `.\pytorch\torch\utils\benchmark\examples\spectral_ops_fuzz_test.py`

```
"""Microbenchmarks for the torch.fft module"""
# 导入必要的模块和库
from argparse import ArgumentParser  # 用于解析命令行参数的模块
from collections import namedtuple  # 命名元组，用于创建具名字段的类
from collections.abc import Iterable  # 抽象基类，用于判断对象是否可迭代

import torch  # PyTorch库
import torch.fft  # PyTorch中的FFT模块
from torch.utils import benchmark  # PyTorch中的基准测试工具
from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer  # 光谱操作模糊器

def _dim_options(ndim):
    # 根据给定的维度数返回适当的维度选项列表
    if ndim == 1:
        return [None]
    elif ndim == 2:
        return [0, 1, None]
    elif ndim == 3:
        return [0, 1, 2, (0, 1), (0, 2), None]
    # 如果维度超出范围，则抛出异常
    raise ValueError(f"Expected ndim in range 1-3, got {ndim}")

def run_benchmark(name: str, function: object, dtype: torch.dtype, seed: int, device: str, samples: int,
                  probability_regular: float):
    # 根据输入的参数运行基准测试，并返回结果列表
    cuda = device == 'cuda'
    # 创建光谱操作模糊器实例
    spectral_fuzzer = SpectralOpFuzzer(seed=seed, dtype=dtype, cuda=cuda,
                                       probability_regular=probability_regular)
    # 存储测试结果的列表
    results = []
    # 迭代光谱操作模糊器生成的样本数据
    for tensors, tensor_params, params in spectral_fuzzer.take(samples):
        # 根据参数中的形状信息创建形状列表
        shape = [params['k0'], params['k1'], params['k2']][:params['ndim']]
        # 将形状信息格式化为字符串
        str_shape = ' x '.join([f"{s:<4}" for s in shape])
        # 创建子标签，描述张量是否连续的信息
        sub_label = f"{str_shape} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
        # 遍历维度选项
        for dim in _dim_options(params['ndim']):
            # 根据是否使用CUDA选择线程数选项
            for nthreads in (1, 4, 16) if not cuda else (1,):
                # 创建基准计时器对象
                measurement = benchmark.Timer(
                    stmt='func(x, dim=dim)',
                    globals={'func': function, 'x': tensors['x'], 'dim': dim},
                    label=f"{name}_{device}",
                    sub_label=sub_label,
                    description=f"dim={dim}",
                    num_threads=nthreads,
                ).blocked_autorange(min_run_time=1)
                # 设置基准计时器的元数据
                measurement.metadata = {
                    'name': name,
                    'device': device,
                    'dim': dim,
                    'shape': shape,
                }
                # 更新张量参数的元数据
                measurement.metadata.update(tensor_params['x'])
                # 将当前测量结果添加到结果列表中
                results.append(measurement)
    # 返回所有测量结果的列表
    return results

# 定义命名元组Benchmark，用于存储基准测试的名称、函数和数据类型
Benchmark = namedtuple('Benchmark', ['name', 'function', 'dtype'])
# 预定义多个基准测试的名称、函数和数据类型的列表
BENCHMARKS = [
    Benchmark('fft_real', torch.fft.fftn, torch.float32),
    Benchmark('fft_complex', torch.fft.fftn, torch.complex64),
    Benchmark('ifft', torch.fft.ifftn, torch.complex64),
    Benchmark('rfft', torch.fft.rfftn, torch.float32),
    Benchmark('irfft', torch.fft.irfftn, torch.complex64),
]
# 创建基准测试名称到基准测试命名元组的映射
BENCHMARK_MAP = {b.name: b for b in BENCHMARKS}
# 存储所有基准测试名称的列表
BENCHMARK_NAMES = [b.name for b in BENCHMARKS]
# 存储所有设备名称的列表
DEVICE_NAMES = ['cpu', 'cuda']

def _output_csv(file, results):
    # 向文件中写入CSV格式的基准测试结果
    file.write('benchmark,device,num_threads,numel,shape,contiguous,dim,mean (us),median (us),iqr (us)\n')
    # 遍历results列表中的每个measurement对象
    for measurement in results:
        # 从measurement对象中获取metadata字典
        metadata = measurement.metadata
        # 从metadata字典中提取设备信息、维度信息、形状信息、名称、元素数量和是否连续的布尔值
        device, dim, shape, name, numel, contiguous = (
            metadata['device'], metadata['dim'], metadata['shape'],
            metadata['name'], metadata['numel'], metadata['is_contiguous'])

        # 如果dim是可迭代对象，则将其转换为用'-'分隔的字符串
        if isinstance(dim, Iterable):
            dim_str = '-'.join(str(d) for d in dim)
        else:
            # 否则，将dim转换为字符串形式
            dim_str = str(dim)
        
        # 将形状信息转换为用'x'分隔的字符串
        shape_str = 'x'.join(str(s) for s in shape)

        # 打印measurement的名称、设备、任务特定的线程数、元素数量、形状字符串、是否连续、维度字符串以及三个统计量乘以1e6后的值
        # 输出结果以逗号分隔，并写入指定文件对象中
        print(name, device, measurement.task_spec.num_threads, numel, shape_str, contiguous, dim_str,
              measurement.mean * 1e6, measurement.median * 1e6, measurement.iqr * 1e6,
              sep=',', file=file)
if __name__ == '__main__':
    # 如果脚本作为主程序运行，则执行以下代码块

    parser = ArgumentParser(description=__doc__)
    # 创建参数解析器，用于处理命令行参数，并设置描述信息为文档字符串

    parser.add_argument('--device', type=str, choices=DEVICE_NAMES, nargs='+', default=DEVICE_NAMES)
    # 添加命令行参数 --device，参数类型为字符串，可选值为 DEVICE_NAMES 中的值，支持多个参数，如果未提供则使用 DEVICE_NAMES 的默认值

    parser.add_argument('--bench', type=str, choices=BENCHMARK_NAMES, nargs='+', default=BENCHMARK_NAMES)
    # 添加命令行参数 --bench，参数类型为字符串，可选值为 BENCHMARK_NAMES 中的值，支持多个参数，如果未提供则使用 BENCHMARK_NAMES 的默认值

    parser.add_argument('--seed', type=int, default=0)
    # 添加命令行参数 --seed，参数类型为整数，默认值为 0

    parser.add_argument('--samples', type=int, default=10)
    # 添加命令行参数 --samples，参数类型为整数，默认值为 10

    parser.add_argument('--probability-regular', '--probability_regular', type=float, default=1.0)
    # 添加命令行参数 --probability-regular 或 --probability_regular，参数类型为浮点数，默认值为 1.0

    parser.add_argument('-o', '--output', type=str)
    # 添加命令行参数 -o 或 --output，参数类型为字符串，用于指定输出文件路径

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 变量中

    num_benchmarks = len(args.device) * len(args.bench)
    # 计算要执行的基准测试数量，等于设备数乘以基准名称数

    i = 0
    results = []
    # 初始化计数器 i 和结果列表 results

    for device in args.device:
        for bench in (BENCHMARK_MAP[b] for b in args.bench):
            # 遍历每个设备和每个基准测试的组合，通过生成器表达式生成 bench 对象

            results += run_benchmark(
                name=bench.name, function=bench.function, dtype=bench.dtype,
                seed=args.seed, device=device, samples=args.samples,
                probability_regular=args.probability_regular)
            # 调用 run_benchmark 函数执行基准测试，并将结果添加到 results 列表中

            i += 1
            # 计数器加一，表示完成了一个基准测试

            print(f'Completed {bench.name} benchmark on {device} ({i} of {num_benchmarks})')
            # 打印完成的基准测试信息，包括基准名称、设备和完成数量

    if args.output is not None:
        with open(args.output, 'w') as f:
            _output_csv(f, results)
        # 如果指定了输出文件路径 args.output，则将 results 列表中的结果写入到文件中

    compare = benchmark.Compare(results)
    # 创建比较器对象，用于比较基准测试结果

    compare.trim_significant_figures()
    # 调用比较器对象的 trim_significant_figures 方法，修剪结果的有效数字

    compare.colorize()
    # 调用比较器对象的 colorize 方法，给结果着色显示

    compare.print()
    # 调用比较器对象的 print 方法，打印比较结果
```