# `.\lucidrains\self-rewarding-lm-pytorch\self_rewarding_lm_pytorch\mocks.py`

```py
# 导入 functools 模块中的 wraps 函数
# 导入 typing 模块中的 Type 和 Any 类型
# 导入 torch.utils.data 模块中的 Dataset 类

from functools import wraps
from typing import Type, Any
from torch.utils.data import Dataset

# 创建一个装饰器函数，根据传入的值返回一个装饰器
def always(val):
    # 装饰器函数，接受一个函数作为参数
    def decorator(fn):
        # 内部函数，使用 functools 模块中的 wraps 函数装饰传入的函数
        @wraps(fn)
        # 接受任意参数并根据传入的值返回结果
        def inner(*args, **kwargs):
            # 如果传入的值是可调用的函数，则调用该函数并返回结果
            if callable(val):
                return val()

            # 否则直接返回传入的值
            return val
        return inner
    return decorator

# 创建一个模拟数据集的函数，根据传入的长度和输出值创建一个 Dataset 对象
def create_mock_dataset(
    length: int,
    output: Any
) -> Dataset:

    # 定义一个内部类 MockDataset，继承自 Dataset 类
    class MockDataset(Dataset):
        # 重写 __len__ 方法，返回传入的长度
        def __len__(self):
            return length

        # 重写 __getitem__ 方法，根据索引返回传入的输出值
        def __getitem__(self, idx):
            # 如果传入的输出值是可调用的函数，则调用该函数并返回结果
            if callable(output):
                return output()

            # 否则直接返回传入的输出值
            return output

    # 返回创建的 MockDataset 对象
    return MockDataset()
```