# `.\pytorch\test\typing\pass\cuda_steam.py`

```py
# 导入PyTorch库
import torch

# 定义函数foo，接受一个torch.Tensor类型的参数x，无返回值
def foo(x: torch.Tensor) -> None:
    # 获取当前CUDA流
    stream = torch.cuda.current_stream()
    # 让张量x记录当前CUDA流以便异步执行
    x.record_stream(stream)
```