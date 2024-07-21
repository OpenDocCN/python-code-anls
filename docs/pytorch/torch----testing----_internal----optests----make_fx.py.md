# `.\pytorch\torch\testing\_internal\optests\make_fx.py`

```
# 忽略 mypy 类型检查错误

# 导入 torch 库
import torch
# 导入 make_fx 函数，用于创建函数的 FX 代理
from torch.fx.experimental.proxy_tensor import make_fx
# 导入 wrapper_set_seed 函数，用于设置随机数种子
from torch.testing._utils import wrapper_set_seed
# 导入 pytree 模块
import torch.utils._pytree as pytree


# 定义 make_fx_check 函数，用于检查 make_fx 的运行结果
def make_fx_check(
    func,
    args,
    kwargs,
    tracing_mode,
    assert_close=torch.testing.assert_close,
    randomize_data=False,
):
    # 调用 handle_sizes_for_dynamic_shapes 处理函数和参数
    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    # 定义 run 函数，用 wrapper_set_seed 包装 f 函数
    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    # 使用 make_fx 创建 traced_f 对象，对 f 进行 FX 代理
    traced_f = make_fx(f, tracing_mode=tracing_mode)(*new_args)

    # 错误消息，描述不同情况下 op 和 make_fx(op) 结果不一致的可能原因
    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) produced different "
        "values. This could mean that your abstract impls (meta/FakeTensor impls) "
        "are incorrect, that your operator is not completely traceable (e.g., "
        "it relies on some global state), or that there is a bug in make_fx. "
        "Note that if you passed a python function (and not an operator) to "
        "make_fx_check, it is still possible that the python function will still "
        "work with torch.compile because it handles capturing pieces of "
        "your python code to compile."
    )

    # 如果 randomize_data 为 True，随机化 new_args 的数据并运行追踪图
    # 以捕获可能在跟踪中固化 Tensor 数据的错误
    if randomize_data:
        new_args = randomize(new_args)
    
    # 尝试运行 f(*new_args) 获取期望结果
    try:
        expected = run(f, *new_args)
    except Exception:
        # 如果使用了 randomize_data 并且失败，则忽略异常
        if randomize_data:
            return
        raise
    
    # 运行 traced_f(*new_args) 获取结果
    result = run(traced_f, *new_args)
    # 使用 assert_close 检查 result 和 expected 的近似性，否则抛出 msg 中定义的错误信息
    assert_close(result, expected, msg=msg)


# 处理 torch.Size 对象以获得动态形状的策略说明
#
# 如果任何参数是 torch.Size 对象，则可能通过以下方式获取其动态形状：
# - 创建一个临时的 Tensor，其大小是我们想要的 torch.Size。注意我们使用扩展的 Tensor，
#   因为我们不能将 "meta" Tensor 传递给 make_fx。
# - 将其传递给 make_fx，使其转换为代理 Tensor
# - 在包装器中解压大小，以获取带有动态形状的 torch.Size（在符号模式下为无操作，否则为普通模式）
def handle_sizes_for_dynamic_shapes(func, args, kwargs):
    # 定义内部函数 f，处理额外的参数和关键字参数
    def f(args, kwargs, extra_args, extra_kwargs):
        # 处理额外的位置参数
        if extra_args:
            for i, t in extra_args:
                args[i] = t.size()
        # 处理额外的关键字参数
        if extra_kwargs:
            for k, t in extra_kwargs.items():
                kwargs[k] = t.size()

        # 调用原始函数 func，并返回其结果
        return func(*args, **kwargs)

    # 初始化额外的参数列表和关键字参数字典
    extra_args = []
    extra_kwargs = {}
    
    # 遍历位置参数 args，如果是 torch.Size 对象，则创建一个空的 Tensor，并加入额外参数列表
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Size):
            extra_args.append((i, torch.empty(arg, device="cpu")))
    
    # 遍历关键字参数 kwargs，如果是 torch.Size 对象，则创建一个空的 Tensor，并加入额外关键字参数字典
    for key, value in kwargs.items():
        if isinstance(value, torch.Size):
            extra_kwargs[key] = torch.empty(value, device="cpu")
    # 返回函数对象 f，位置参数 args，关键字参数 kwargs，额外的位置参数 extra_args，额外的关键字参数 extra_kwargs
    return f, args, kwargs, extra_args, extra_kwargs
# 定义函数 randomize，接受一个参数 args
def randomize(args):
    # 定义内部函数 transform，对输入的 x 进行转换操作
    def transform(x):
        # 检查 x 是否不是浮点数类型，如果不是则直接返回 x
        if not x.dtype.is_floating_point:
            return x
        # 如果 x 是浮点数类型，进行以下操作：
        # 分离 x 的计算图并复制其数据，然后在副本上生成均匀分布的随机数
        randomized = x.detach().clone().uniform_(0, 1).requires_grad_(x.requires_grad)
        return randomized
    # 使用 pytree 库对 torch.Tensor 类型的数据结构 args 进行转换，应用 transform 函数
    return pytree.tree_map_only(torch.Tensor, transform, args)
```