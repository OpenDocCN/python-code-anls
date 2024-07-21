# `.\pytorch\test\_test_bazel.py`

```
"""
This test module contains a minimalistic "smoke tests" for the bazel build.

Currently it doesn't use any testing framework (i.e. pytest)
TODO: integrate this into the existing pytorch testing framework.

The name uses underscore `_test_bazel.py` to avoid globbing into other non-bazel configurations.
"""

# 导入 torch 模块
import torch


# 定义一个测试函数，验证张量加法的正确性
def test_sum() -> None:
    assert torch.eq(
        torch.tensor([[1, 2, 3]]) + torch.tensor([[4, 5, 6]]), torch.tensor([[5, 7, 9]])
    ).all()


# 定义一个测试函数，验证编译和执行具有特定后端的简单函数
def test_simple_compile_eager() -> None:
    # 定义一个简单的函数 foo，接受两个张量参数，返回它们的正弦和余弦之和
    def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)  # 计算 x 的正弦
        b = torch.cos(y)  # 计算 y 的余弦
        return a + b  # 返回正弦和余弦之和

    # 使用 eager 模式编译函数 foo，得到一个优化后的函数 opt_foo1
    opt_foo1 = torch.compile(foo, backend="eager")
    # 确保调用优化后的函数不会引发异常
    assert opt_foo1(torch.randn(10, 10), torch.randn(10, 10)) is not None


# 执行定义的两个测试函数
test_sum()
test_simple_compile_eager()
```