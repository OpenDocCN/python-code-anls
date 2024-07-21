# `.\pytorch\benchmarks\serialization\simple_measurement.py`

```py
# 导入必要的模块
from pyarkbench import Benchmark, default_args, Timer
import torch

# 是否使用新的 ZIP 文件序列化方式
use_new = True

# 定义一个继承自Benchmark的类Basic，用于性能基准测试
class Basic(Benchmark):
    def benchmark(self):
        # 创建一个包含30个200x200的全1张量的列表
        x = [torch.ones(200, 200) for i in range(30)]
        
        # 计时器：记录保存大张量到 ZIP 文件的时间
        with Timer() as big1:
            torch.save(x, "big_tensor.zip", _use_new_zipfile_serialization=use_new)
        
        # 计时器：记录从 ZIP 文件加载大张量的时间
        with Timer() as big2:
            v = torch.load("big_tensor.zip")
        
        # 重新分配x变量，创建一个包含200个10x10的全1张量的列表
        x = [torch.ones(10, 10) for i in range(200)]
        
        # 计时器：记录保存小张量到 ZIP 文件的时间
        with Timer() as small1:
            torch.save(x, "small_tensor.zip", _use_new_zipfile_serialization=use_new)
        
        # 计时器：记录从 ZIP 文件加载小张量的时间
        with Timer() as small2:
            v = torch.load("small_tensor.zip")
        
        # 返回性能测试结果，包括保存和加载大小张量的时间
        return {
            "Big Tensors Save": big1.ms_duration,
            "Big Tensors Load": big2.ms_duration,
            "Small Tensors Save": small1.ms_duration,
            "Small Tensors Load": small2.ms_duration,
        }

# 当脚本被直接执行时
if __name__ == "__main__":
    # 创建一个Basic类的实例bench，使用默认的基准测试参数
    bench = Basic(*default_args.bench())
    # 打印是否使用新的 ZIP 文件序列化方式
    print("Use zipfile serialization:", use_new)
    # 运行基准测试，获取结果
    results = bench.run()
    # 打印基准测试的统计结果，包括均值和中位数
    bench.print_stats(results, stats=["mean", "median"])
```