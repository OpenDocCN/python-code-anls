# `.\pytorch\torch\testing\_internal\optests\autograd_registration.py`

```
# 忽略类型检查错误，通常用于标记代码中已知的类型问题
# --------------------------------------------------------
# 导入 contextlib 模块，用于上下文管理器
import contextlib

# 导入 torch 库
import torch
# 导入 torch.utils._pytree 模块，并重命名为 pytree
import torch.utils._pytree as pytree

# 定义上下文管理器函数 set_autograd_fallback_mode，用于设置自动求导回退模式
@contextlib.contextmanager
def set_autograd_fallback_mode(mode):
    # 获取当前自动求导回退模式并保存
    prev = torch._C._get_autograd_fallback_mode()
    try:
        # 设置新的自动求导回退模式
        torch._C._set_autograd_fallback_mode(mode)
        # 执行代码块
        yield
    finally:
        # 恢复之前保存的自动求导回退模式
        torch._C._set_autograd_fallback_mode(prev)

# 定义函数 autograd_registration_check，用于检查操作符的自动求导注册情况
def autograd_registration_check(op, args, kwargs):
    """Check if autograd was registered correctly (for the operator).

    Operators should have "autograd support" registered directly to an
    autograd dispatch key.
    An incorrect registration may lead to unexpected silent incorrectness.
    Note that this check won't catch all problems but will catch
    the most common ones.

    Example usage:
        >>> x = torch.randn(3, requires_grad=True)
        >>> autograd_registration_check(torch.ops.aten.sin.default, (x,), {})

    Here are some best practices if you do find your autograd is
    registered incorrectly:
    - If the operator is composite (i.e. consists of other PyTorch ops)
      and you wish the operator to decompose and get autograd support
      that way, then please register the implementation to
      DispatchKey::CompositeImplicitAutograd
    - If you're adding an autograd formula for the operator, the correct
      thing to do is to register an autograd.Function to
      DispatchKey::Autograd (preferred) or one of the
      DispatchKey::Autograd<BACKEND> keys. It is NOT OK to register
      an autograd.Function to a backend (e.g. CPU/CUDA) key.
    - If your operator is non-differentiable, then you should register
      an implementation to the Autograd key that uses
      AutoDispatchBelowAutograd and re-invokes the operator.

    """
    # 断言操作符 op 是 torch._ops.OpOverload 类型
    assert isinstance(op, torch._ops.OpOverload)
    
    # Implementation details
    # -----------------------------------------------
    # If an operator doesn't have an autograd kernel at an autograd key,
    # and the operator does not return inputs as-is, then all of
    # the outputs should have requires_grad=False before we apply
    # special behaviors of our default autograd fallback.
    # (The default autograd fallback may set requires_grad=True on output
    # tensors in certain modes so that when they are backpropped through,
    # they raise an error).
    #
    # Our strategy for detecting if an operator doesn't have an autograd
    # kernel at the autograd key is:
    # - set the autograd fallback mode to "nothing" (so it does not change
    #   the required-gradness of outputs)
    # - run the operator
    # - Check if any outputs of the operator (that are not inputs) require
    #   grad. This would only happen if the user calls regular PyTorch
    #   operations in their backend key (this op should instead be
    #   CompositeImplicitAutograd or not an op) or if the user invokes
    #   an autograd.Function in the backend key.
    #
    # Note that it's already likely a bug if the operator directly returns
    # 使用 pytree.arg_tree_leaves() 函数展开参数列表，获取所有参数的叶子节点
    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    
    # 筛选出所有是 torch.Tensor 类型的参数
    all_tensors = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]
    
    # 检查是否有任何一个张量需要梯度计算
    if not any(t.requires_grad for t in all_tensors):
        # 如果没有张量需要梯度计算，则抛出 RuntimeError 异常
        raise RuntimeError(
            "autograd_registration_check: no inputs have requires_grad=True so "
            "we are unable to actually perform this test. Please pass inputs "
            "that do require grad."
        )

    # 获取所有张量的设备类型，并存储在集合中
    all_device_types = {arg.device.type for arg in all_tensors}
    
    # 检查所有设备类型是否仅为 "cpu" 或 "cuda"
    if not all_device_types.issubset(["cpu", "cuda"]):
        # 如果设备类型包含其他值，则抛出 NotImplementedError 异常
        raise NotImplementedError(
            f"autograd_registration_check: NYI devices other than CPU/CUDA, got {all_device_types}"
        )
    
    # 根据设备类型确定要检查的 AutogradBACKEND 键
    if "cuda" in all_device_types:
        key = "AutogradCUDA"
    elif "cpu" in all_device_types:
        key = "AutogradCPU"

    # 检查 op.name() 对应的操作是否已注册到指定的 AutogradBACKEND 键
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key):
        return
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Autograd"):
        return
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        op.name(), "CompositeImplicitAutograd"
    ):
        return

    # 若操作没有注册到任何 AutogradBACKEND 键，则进行下一步测试
    # 使用 set_autograd_fallback_mode("nothing") 设置自动求导回退模式
    with set_autograd_fallback_mode("nothing"):
        # 执行操作 op，并获取所有输出
        all_outs = op(*args, **kwargs)

    # 获取所有输入参数的唯一标识集合
    inp_ids = {id(arg) for arg in flat_args}

    # 定义一个函数，用于检查是否为非输入且需要梯度计算的张量
    def not_an_input_and_requires_grad(tensor):
        if not tensor.requires_grad:
            return False
        if id(tensor) in inp_ids:
            return False
        return True

    # 检查所有输出是否至少有一个张量满足条件 not_an_input_and_requires_grad
    if not pytree.tree_any_only(torch.Tensor, not_an_input_and_requires_grad, all_outs):
        return

    # 若以上条件均不满足，则抛出 AssertionError 异常，提示操作没有定义对应的 Autograd 内核
    raise AssertionError(
        f"{op.name()}: at least one output of this operator has requires_grad=True "
        f"but the operator does not have an autograd kernel defined at an autograd "
        f"key (e.g. DispatchKey::Autograd). This could mean that you have "
        f"incorrectly registered an autograd kernel to a non-Autograd DispatchKey, "
        f"which may lead to silently incorrect results. If your operator consists "
        f"of regular PyTorch operations, consider not using an operator at all "
        f"or registering your operator as CompositeImplicitAutograd. If you have "
        f"an autograd.Function registered to a backend (CPU/CUDA) key, the correct "
        f"location for it is the Autograd key."
    )
```