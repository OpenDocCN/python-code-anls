# `.\pytorch\torch\testing\_internal\optests\aot_autograd.py`

```
# mypy: ignore-errors

# 导入PyTorch库
import torch
# 导入PyTorch的_pytree模块
import torch.utils._pytree as pytree
# 从torch.testing._utils中导入设置随机种子的函数
from torch.testing._utils import wrapper_set_seed
# 从functorch.compile中导入编译函数相关模块
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
# 从当前目录中的make_fx模块中导入randomize函数
from .make_fx import randomize
# 导入re模块用于正则表达式操作
import re

# 定义一个上下文管理器类assert_raises_regex，用于验证抛出的异常是否符合正则表达式
class assert_raises_regex:
    def __init__(self, exception_cls, regex):
        self.exception_cls = exception_cls
        self.regex = regex

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        # 如果捕获到的异常类型与预期的异常类型匹配
        if exc_type == self.exception_cls:
            msg = str(exc_val)
            # 如果异常信息不符合预期的正则表达式，则抛出AssertionError
            if not re.search(self.regex, msg):
                raise AssertionError(
                    f"Expected exception to match regex. regex: {self.regex}, exception: {msg}")
            return True  # Squashes the exception
        # 如果捕获到的异常类型不是预期的异常类型，则抛出AssertionError
        if exc_type is not None:
            raise AssertionError(
                f"Expected {self.exception_cls} to be raised, instead got exception {exc_type}")
        # 如果没有捕获到异常，则抛出AssertionError
        raise AssertionError("Expected exception to be raised but none was")

# 定义函数aot_autograd_check，用于比较在eager模式下和AOTAutograd下的函数执行结果
def aot_autograd_check(
        func,
        args,
        kwargs,
        dynamic,
        assert_raises_regex_fn=assert_raises_regex,
        assert_equals_fn=torch.testing._comparison.assert_close,
        check_gradients=True,
        try_check_data_specialization=False):
    """Compares func(*args, **kwargs) in eager-mode to under AOTAutograd.

    Compares outputs and (if check_gradients=True) gradients produced by
    AOTAutograd against eager-mode PyTorch.

    We assume that func(*args, **kwargs) succeeds in eager-mode PyTorch.

    """
    # 将参数args和kwargs展平并获取其规范化表示
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    # 检查flat_args中哪些参数是Tensor类型
    args_is_tensor = [isinstance(arg, torch.Tensor) for arg in flat_args]
    # 将flat_args中的Tensor参数提取出来
    args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

    # 定义一个新的函数func_no_tensors，该函数接受Tensor作为输入
    def func_no_tensors(args):
        reconstructed_flat_args = []
        args = iter(args)
        # 重新构建flat_args，将Tensor参数替换回原来的位置
        for v in flat_args:
            if isinstance(v, torch.Tensor):
                reconstructed_flat_args.append(next(args))
            else:
                reconstructed_flat_args.append(v)

        # 使用重构后的参数调用原始函数func
        c_args, c_kwargs = pytree.tree_unflatten(reconstructed_flat_args, args_spec)
        return func(*c_args, **c_kwargs)

    # 编译func_no_tensors函数，以便在AOTAutograd中使用
    compiled_f = compiled_function(
        func_no_tensors, nop, nop, dynamic=dynamic, partition_fn=min_cut_rematerialization_partition)

    # 在eager模式下执行func_no_tensors函数，并获取其输出
    out = wrapper_set_seed(func_no_tensors, args)
    # 如果需要检查梯度，则根据条件判断是否进行梯度检查
    if check_gradients == "auto":
        any_tensor_requires_grad = pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, args)
        any_output_requires_grad = pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, out)
        check_gradients = any_tensor_requires_grad and any_output_requires_grad
    # 如果不需要检查梯度，则直接比较编译后的输出和原始输出
    if not check_gradients:
        compiled_out = wrapper_set_seed(compiled_f, args)
        assert_equals_fn(compiled_out, out, msg=outputs_msg)
        return
    # 调用测试函数 _test_aot_autograd_forwards_backwards_helper，并传入多个参数
    # func_no_tensors: 不包含张量的函数
    # compiled_f: 编译后的函数
    # args: 参数列表
    # assert_raises_regex_fn: 断言异常信息的函数
    # assert_equals_fn: 断言相等的函数
    # try_check_data_specialization: 尝试检查数据特化的函数
    _test_aot_autograd_forwards_backwards_helper(
        func_no_tensors, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization)
outputs_msg = (
    "Outputs of the operator are different in eager-mode PyTorch vs "
    "AOTAutograd. This means the operator will have incorrect output "
    "underneath torch.compile. This could be because the operator's "
    "implementation not traceable or that there is a bug in AOTAutograd."
)
# 定义一个消息，用于描述在 eager 模式 PyTorch 和 AOTAutograd 中操作符的输出差异问题

def _test_aot_autograd_forwards_backwards_helper(
        f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization):
    # 辅助函数，验证编译和非编译版本的 f 的梯度是否相等

    def call_forwards_backwards(f, args):
        # 执行 f 的前向传播和反向传播，并验证梯度是否相等
        flat_args = pytree.arg_tree_leaves(*args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and
                     arg.requires_grad]
        out = wrapper_set_seed(f, args)
        flat_out = pytree.tree_leaves(out)

        sm = 0
        for i in flat_out:
            if isinstance(i, torch.Tensor):
                # 由于操作符的输出可能是复杂张量，需要调用 .abs()，
                # 因为 autograd 会在复杂张量上抛出错误，除非手动提供 grad_output 标志。
                sm += i.sum().abs()
        assert isinstance(sm, torch.Tensor)
        return out, torch.autograd.grad(sm, diff_args, allow_unused=True)

    def check(args, ignore_failure=False):
        # 检查编译和非编译版本的 f 是否输出相等的梯度
        try:
            orig_out, orig_grad = call_forwards_backwards(f, args)
        except Exception:
            if ignore_failure:
                return
            raise

        # 如果所有的梯度都为 None，则断言应该抛出 RuntimeError，并给出特定消息
        if all(x is None for x in orig_grad):
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(compiled_f, args)
            return

        msg = (
            "Gradients of the operator are different in eager-mode PyTorch vs "
            "AOTAutograd. This means the operator will have incorrect gradients "
            "underneath torch.compile. This could be because the operator's "
            "backward is incorrectly registered or not traceable or that there "
            "is a bug in AOTAutograd."
        )
        # 验证编译版本的 f 的输出和梯度与原始版本的 f 相等
        compiled_out, compiled_grad = call_forwards_backwards(compiled_f, args)
        assert_equals_fn(compiled_out, orig_out, msg=outputs_msg)
        assert_equals_fn(compiled_grad, orig_grad, msg=msg)

    check(args, ignore_failure=False)

    # 随机化数据并使用跟踪图运行，以捕获可能已经嵌入张量数据的跟踪中的 bug
    # 这不保证成功，因为 `f` 可能对输入值有前提条件，因此如果此测试失败，我们将忽略它。
    if try_check_data_specialization:
        args = randomize(args)
        check(args, ignore_failure=True)
```