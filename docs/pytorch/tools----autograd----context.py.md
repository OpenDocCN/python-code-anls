# `.\pytorch\tools\autograd\context.py`

```py
# 导入 functools 模块，用于高阶函数和函数工具
# 从 typing 模块导入 Callable 类型提示，用于指定函数的类型
from typing import Callable

# 从 torchgen.api.autograd 模块导入 NativeFunctionWithDifferentiabilityInfo 类，并重命名为 NFWDI
from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo as NFWDI
# 从 torchgen.context 模块导入 native_function_manager 函数
from torchgen.context import native_function_manager
# 从 torchgen.utils 模块导入 T 类型
from torchgen.utils import T


# 类似于 tools.api.context.with_native_function，但适用于 NativeFunctionWithDifferentiabilityInfo 类型的函数
def with_native_function_with_differentiability_info(
    func: Callable[[NFWDI], T]  # 接受一个以 NFWDI 类型对象为参数且返回类型为 T 的函数
) -> Callable[[NFWDI], T]:  # 返回一个接受 NFWDI 类型对象为参数且返回类型为 T 的函数
    @functools.wraps(func)  # 使用 functools 模块的 wraps 装饰器，保留原函数的元数据
    def wrapper(f: NFWDI) -> T:  # 定义一个接受 NFWDI 类型对象为参数且返回类型为 T 的内部函数 wrapper
        with native_function_manager(f.func):  # 使用 native_function_manager 管理 f.func
            return func(f)  # 调用原始函数 func，传入参数 f，并返回结果

    return wrapper  # 返回内部函数 wrapper


# 类似于上面的函数，但增加了一个字符串类型的 dispatch key 参数
def with_native_function_with_differentiability_info_and_key(
    func: Callable[[NFWDI, str], T]  # 接受一个以 NFWDI 类型对象和 str 类型对象为参数且返回类型为 T 的函数
) -> Callable[[NFWDI, str], T]:  # 返回一个接受 NFWDI 类型对象和 str 类型对象为参数且返回类型为 T 的函数
    @functools.wraps(func)  # 使用 functools 模块的 wraps 装饰器，保留原函数的元数据
    def wrapper(f: NFWDI, key: str) -> T:  # 定义一个接受 NFWDI 类型对象和 str 类型对象为参数且返回类型为 T 的内部函数 wrapper
        with native_function_manager(f.func):  # 使用 native_function_manager 管理 f.func
            return func(f, key)  # 调用原始函数 func，传入参数 f 和 key，并返回结果

    return wrapper  # 返回内部函数 wrapper
```