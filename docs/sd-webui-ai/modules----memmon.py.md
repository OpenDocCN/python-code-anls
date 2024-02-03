# `stable-diffusion-webui\modules\memmon.py`

```
# 导入所需的库
import threading
import time
from collections import defaultdict

import torch

# 定义一个继承自 threading.Thread 的类 MemUsageMonitor
class MemUsageMonitor(threading.Thread):
    # 类变量
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    # 初始化方法
    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        # 设置为守护线程
        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        # 尝试获取 CUDA 内存信息
        try:
            self.cuda_mem_get_info()
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # 捕获异常，例如 AMD 显卡
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    # 获取 CUDA 内存信息
    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    # 线程运行方法
    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            # 重置 CUDA 内存峰值统计，清空数据
            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    # 打印调试信息
    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(torch.cuda.memory_summary())
    # 启动监控标志位
    def monitor(self):
        self.run_flag.set()

    # 读取 GPU 内存信息
    def read(self):
        # 如果未禁用，则获取 CUDA 内存信息
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            # 更新数据字典中的 free 和 total 字段
            self.data["free"] = free
            self.data["total"] = total

            # 获取 torch 库中的 GPU 内存统计信息
            torch_stats = torch.cuda.memory_stats(self.device)
            # 更新数据字典中的 active, active_peak, reserved, reserved_peak 和 system_peak 字段
            self.data["active"] = torch_stats["active.all.current"]
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        # 返回数据字典
        return self.data

    # 停止监控
    def stop(self):
        # 清除监控标志位
        self.run_flag.clear()
        # 返回读取的数据
        return self.read()
```