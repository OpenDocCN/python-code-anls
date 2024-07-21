# `.\pytorch\torch\_dynamo\backends\torchxla.py`

```
# 忽略类型检查错误，可能是由于类型系统不完全匹配而引起的问题
# mypy: ignore-errors

# 导入日志模块
import logging

# 从functorch.compile模块中导入make_boxed_func函数
from functorch.compile import make_boxed_func

# 从..backends.common模块中导入aot_autograd函数
from ..backends.common import aot_autograd

# 从.registry模块中导入register_backend和register_experimental_backend函数
from .registry import register_backend, register_experimental_backend

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 将openxla_eval函数注册为实验性后端
@register_experimental_backend
def openxla_eval(model, fake_tensor_inputs):
    # 调用xla_backend_helper函数，传递boxed参数为False
    return xla_backend_helper(model, fake_tensor_inputs, boxed=False)


# 定义openxla_eval_boxed函数，调用xla_backend_helper函数，传递boxed参数为True
def openxla_eval_boxed(model, fake_tensor_inputs):
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)


# 定义xla_backend_helper函数，接收model、fake_tensor_inputs和boxed参数
def xla_backend_helper(model, fake_tensor_inputs, boxed=False):
    try:
        # 尝试导入torch_xla.core.dynamo_bridge模块，并将其命名为bridge
        import torch_xla.core.dynamo_bridge as bridge
    except ImportError as e:
        # 如果导入失败，抛出带有安装指南链接的ImportError
        raise ImportError(
            "Please follow the instruction in https://github.com/pytorch/xla#pytorchxla to install torch_xla"
        ) from e

    # 初始化compiled_graph为None
    compiled_graph = None

    # 定义内部函数fwd，接收任意数量的参数args
    def fwd(*args):
        nonlocal model  # 使用外部函数的model变量
        nonlocal compiled_graph  # 使用外部函数的compiled_graph变量
        # 如果compiled_graph为None，则使用bridge.extract_compiled_graph函数提取编译后的图形
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model  # 删除外部函数的model变量引用，释放内存
        return compiled_graph(*args)  # 调用编译后的图形函数并返回结果

    # 如果boxed为True，返回make_boxed_func(fwd)；否则返回fwd函数
    return make_boxed_func(fwd) if boxed else fwd


# 使用aot_autograd装饰器，传递fw_compiler参数为openxla_eval_boxed函数
openxla = aot_autograd(
    fw_compiler=openxla_eval_boxed,
)

# 注册名为"openxla"的后端，使用register_backend函数，传递compiler_fn参数为openxla
register_backend(name="openxla", compiler_fn=openxla)
```