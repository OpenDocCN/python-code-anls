# `.\pytorch\benchmarks\overrides_benchmark\common.py`

```py
```python`
# 导入 PyTorch 库
import torch

# 定义常量，表示重复次数
NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000

# 定义一个名为 SubTensor 的类，继承自 torch.Tensor
class SubTensor(torch.Tensor):
    pass

# 定义一个类 WithTorchFunction，包含一个初始化方法和一个类方法 __torch_function__
class WithTorchFunction:
    # 初始化方法，接收数据和一个可选的 requires_grad 参数
    def __init__(self, data, requires_grad=False):
        # 如果数据已经是一个 torch.Tensor 对象，直接赋值
        if isinstance(data, torch.Tensor):
            self._tensor = data
            return

        # 否则，将数据转换为 torch.Tensor 对象，并设置 requires_grad
        self._tensor = torch.tensor(data, requires_grad=requires_grad)

    # 类方法 __torch_function__，用于处理自定义的运算符重载
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 返回一个新的 WithTorchFunction 对象，其内容为 args[0]._tensor 加上 args[1]._tensor
        return WithTorchFunction(args[0]._tensor + args[1]._tensor)

# 定义一个名为 SubWithTorchFunction 的类，继承自 torch.Tensor
class SubWithTorchFunction(torch.Tensor):
    # 类方法 __torch_function__，在这里调用父类的 __torch_function__ 方法
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 调用父类的 __torch_function__ 方法，并返回结果
        return super().__torch_function__(func, types, args, kwargs)
```