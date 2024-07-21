# `.\pytorch\test\test_cuda_expandable_segments.py`

```
# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments

# 导入操作系统相关的模块
import os

# 导入 PyTorch 库
import torch

# 从 PyTorch 的内部测试模块中导入 IS_JETSON 变量
from torch.testing._internal.common_cuda import IS_JETSON

# 检查当前环境是否支持 CUDA 并且不是 Jetson 平台
if torch.cuda.is_available() and not IS_JETSON:
    # 设置 CUDA 内存分配器的配置参数，使用可扩展段
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    # 获取当前文件的绝对路径所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建测试文件的完整路径
    filepath = os.path.join(current_dir, "test_cuda.py")
    # 执行指定路径下的 Python 文件作为脚本
    exec(compile(open(filepath).read(), filepath, mode="exec"))
```