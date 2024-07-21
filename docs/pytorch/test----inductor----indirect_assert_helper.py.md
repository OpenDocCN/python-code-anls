# `.\pytorch\test\inductor\indirect_assert_helper.py`

```py
# 导入系统模块
import sys

# 导入PyTorch库
import torch

# 导入GPU类型信息
from torch.testing._internal.inductor_utils import GPU_TYPE

# 定义函数，返回列表或张量的第y个元素
def first_arg(x, y):
    return x[y]

# 定义函数，返回二维张量的第y列
def second_arg(x, y):
    return x[:, y]

# 定义函数，返回张量中索引为(y+1, y-1)位置的元素
def same_pm_one(x, y):
    return x[y + 1, y - 1]

# 定义函数，返回张量中索引为(y+1, y+1)位置的元素
def same_pp_one(x, y):
    return x[y + 1, y + 1]

# 定义函数，将张量x中索引为(y+1, y+1)位置的元素赋值为z
def store(x, y, z):
    x[y + 1, y + 1] = z

# 定义函数，返回张量x中索引为[0, 1, 2, 3]的元素
def upper1(x):
    b = torch.arange(4, device=x.device)
    return x[b]

# 定义函数，返回张量x中索引为[-4]的元素
def lower1(x):
    b = x.new_full((), -4, dtype=torch.int64)
    return x[b]

# 定义函数，返回张量x中索引为[4]的元素
def upper2(x):
    b = x.new_full((), 4, dtype=torch.int64)
    return x[b]

# 定义函数，返回张量x中索引为[-4]的元素
def lower2(x):
    b = x.new_zeros((), dtype=torch.int64)
    return x[b - 4]

# 程序主入口
if __name__ == "__main__":
    # 获取当前模块中所有可调用对象的名称列表
    fns = [
        name
        for name, obj in locals().items()
        if callable(obj) and obj.__module__ == __name__
    ]

    # 从命令行参数获取函数名、维度数、动态形状标志、大小标志
    _, fn_name, dims, dyn_shape, one_size = sys.argv

    # 确保函数名存在于可调用对象列表中
    assert fn_name in fns

    # 确保大小标志为"True"或"False"之一，并转换为布尔值
    assert one_size in ("True", "False")
    one_size = one_size == "True"

    # 确保维度数为"2"或"3"之一，并根据维度数设定张量形状
    assert dims in ("2", "3")
    shape_x = [3, 2, 4] if dims == "3" else [3, 2]

    # 如果大小标志为True，则只有first_arg函数可以用于测试特殊情况下的1大小张量
    if one_size:
        assert fn_name == "first_arg", "only first_arg can be tested for a special case of 1-size tensor"
        shape_x[0] = 1

    # 确保动态形状标志为"True"或"False"之一，并转换为布尔值
    assert dyn_shape in ("True", "False")
    dynamic_shapes = dyn_shape == "True"

    # 在指定的GPU上生成随机张量x和y
    x = torch.randn(shape_x, device=GPU_TYPE)
    y = torch.arange(4, device=GPU_TYPE)

    # 根据函数名获取对应的函数对象，并进行编译
    fn = vars()[fn_name]
    fn = torch.compile(dynamic=dynamic_shapes)(fn)

    # 如果函数名为"store"，生成形状与y长度相同的随机张量z，然后调用fn函数
    if fn_name == "store":
        shape = (y.numel(),) + x.shape[2:]
        z = torch.randn(shape, device=GPU_TYPE)
        fn(x, y, z)

    # 如果函数名为"upper1", "upper2", "lower1", "lower2"中的一个，直接调用fn函数
    elif fn_name in ("upper1", "upper2", "lower1", "lower2"):
        fn(x)

    # 否则，调用fn函数并传入参数x和y
    else:
        fn(x, y)
```