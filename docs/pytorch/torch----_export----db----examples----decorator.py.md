# `.\pytorch\torch\_export\db\examples\decorator.py`

```
# 导入mypy的配置选项，允许未标记类型的函数定义
# 该注释指定了在类型检查时允许未经类型标注的函数定义
# 这对于部分动态语言（如Python）中的类型检查工具很有用
# 详情可参考：https://mypy.readthedocs.io/en/stable/command_line.html#mypy-allow-untyped-defs
# 
import functools

# 导入PyTorch库
# PyTorch是一个开源的机器学习库，提供了广泛的功能和工具，特别是在深度学习领域
import torch

# 定义一个装饰器函数test_decorator，用于对函数进行装饰
def test_decorator(func):
    # 使用functools.wraps装饰器，用来保留原始函数的元数据（如函数名、文档字符串等）
    @functools.wraps(func)
    # 定义装饰器函数的内部函数wrapper，用于实际执行装饰逻辑
    def wrapper(*args, **kwargs):
        # 调用原始函数并返回结果加1
        return func(*args, **kwargs) + 1

    # 返回装饰后的函数wrapper
    return wrapper

# 定义一个继承自torch.nn.Module的类Decorator
class Decorator(torch.nn.Module):
    """
    Decorators calls are inlined into the exported function during tracing.
    """
    
    # 使用test_decorator装饰器装饰类方法forward
    @test_decorator
    def forward(self, x, y):
        # 返回输入张量x和y的和
        return x + y

# 创建一个示例输入example_inputs，包含两个形状为(3, 2)的随机张量
example_inputs = (torch.randn(3, 2), torch.randn(3, 2))

# 创建Decorator类的实例model，即一个继承自torch.nn.Module的对象
model = Decorator()
```