# `.\pytorch\torch\testing\_utils.py`

```py
# mypy: allow-untyped-defs
# 导入上下文管理工具
import contextlib

# 导入 PyTorch 库
import torch

# 提供公共测试实用工具的函数集合
# 注意：这些函数应该在没有可选依赖项（如 numpy 和 expecttest）的情况下可导入。

def wrapper_set_seed(op, *args, **kwargs):
    """包装函数，手动设置某些函数（如 dropout）的随机种子
    参考：https://github.com/pytorch/pytorch/pull/62315#issuecomment-896143189 获取更多细节。
    """
    # 冻结随机数生成状态
    with freeze_rng_state():
        # 设置随机种子为 42
        torch.manual_seed(42)
        # 执行操作
        output = op(*args, **kwargs)

        # 如果输出是 torch.Tensor 并且设备类型是 "lazy"
        if isinstance(output, torch.Tensor) and output.device.type == "lazy":
            # 在 freeze_rng_state 内部调用 mark_step 以确保数值与 eager 执行匹配
            torch._lazy.mark_step()  # type: ignore[attr-defined]

        return output


@contextlib.contextmanager
def freeze_rng_state():
    """上下文管理器，用于冻结随机数生成器的状态"""
    # 对于 test_composite_compliance 需要禁用分发
    # 一些 OpInfos 使用 freeze_rng_state 来保证随机数的确定性，
    # 但 test_composite_compliance 覆盖了所有 torch 函数的分发，我们需要禁用它来获取和设置随机数状态
    with torch.utils._mode_utils.no_dispatch(), torch._C._DisableFuncTorch():
        # 获取当前的 RNG 状态
        rng_state = torch.get_rng_state()
        # 如果 CUDA 可用，获取 CUDA 的 RNG 状态
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
    try:
        yield
    finally:
        # 由于模式不支持 torch.cuda.set_rng_state
        # 因为它会克隆状态（可能会生成一个 Tensor 子类），然后在 generator.set_state 中获取新张量的数据指针。
        #
        # 长期来看，torch.cuda.set_rng_state 应该可能成为一个操作符。
        #
        # 注意：模式禁用是为了避免在这些种子设置上运行交叉引用测试
        with torch.utils._mode_utils.no_dispatch(), torch._C._DisableFuncTorch():
            # 如果 CUDA 可用，设置 CUDA 的 RNG 状态
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]
            # 恢复 RNG 状态
            torch.set_rng_state(rng_state)
```