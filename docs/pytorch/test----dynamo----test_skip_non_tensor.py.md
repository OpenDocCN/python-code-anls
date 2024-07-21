# `.\pytorch\test\dynamo\test_skip_non_tensor.py`

```py
# Owner(s): ["module: dynamo"]
from unittest.mock import patch  # 导入 patch 函数，用于模拟对象的方法调用

import torch  # 导入 PyTorch 库

import torch._dynamo  # 导入 PyTorch 私有模块
import torch._dynamo.test_case  # 导入 PyTorch 私有测试用例模块
from torch._dynamo.testing import CompileCounter  # 从 PyTorch 私有测试工具中导入编译计数器类

_variable = 0  # 全局变量 _variable，初始化为 0
_variable_2 = 0  # 全局变量 _variable_2，初始化为 0


def user_function():
    return torch._utils.is_compiling()  # 返回当前是否处于编译状态的布尔值


def user_generator():
    for _ in range(1):
        yield torch._utils.is_compiling()  # 生成器函数，每次生成当前是否处于编译状态的布尔值
    return  # 返回空值


class MyModule(torch.nn.Module):
    def __init__(self, mode: int):
        super().__init__()
        self.mode = mode  # 初始化模式
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)  # 注册前向传播前钩子函数

    def pre_forward(self, module, args, kwargs):
        if self.mode == 5:
            if user_function():
                global _variable
                _variable += 1  # 如果模式为 5 并且处于编译状态，则增加全局变量 _variable
        return args, kwargs

    def forward(self, x):
        global _variable, _variable_2

        if self.mode == 1:
            if torch._utils.is_compiling():
                _variable += 1  # 如果模式为 1 并且处于编译状态，则增加全局变量 _variable
            else:
                _variable_2 += 1  # 如果模式为 1 并且不处于编译状态，则增加全局变量 _variable_2
        elif self.mode == 2:
            if user_function():
                _variable += 1  # 如果模式为 2 并且处于编译状态，则增加全局变量 _variable
        elif self.mode == 3:
            lambda_f = lambda: torch._utils.is_compiling()  # 创建匿名函数 lambda_f，判断是否处于编译状态
            if lambda_f():
                _variable += 1  # 如果模式为 3 并且处于编译状态，则增加全局变量 _variable
        elif self.mode == 4:
            for cond in user_generator():
                if cond:
                    _variable += 1  # 如果模式为 4 并且处于编译状态，则增加全局变量 _variable
        elif self.mode == 5:
            x += 1  # 如果模式为 5，则将输入 x 增加 1
        elif self.mode == 6:
            if user_function():
                torch._dynamo.graph_break()  # 如果模式为 6 并且处于编译状态，则调用 PyTorch 私有函数 graph_break()
                _variable += 1  # 增加全局变量 _variable
        return x  # 返回处理后的输入 x


class SkipNonTensorTests(torch._dynamo.test_case.TestCase):
    def test_add_tensor1(self):
        def fn(a, b):
            return a + b  # 返回 a 和 b 的和

        counter = CompileCounter()  # 创建编译计数器对象
        x = torch.randn(4)  # 创建一个形状为 (4,) 的随机张量 x
        y = 5  # 创建一个整数 y
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)  # 对函数 fn 进行优化并断言
        opt_fn(x, y)  # 调用优化后的函数 fn

        assert counter.op_count == 1  # 断言编译计数器的操作计数为 1

    def test_add_tensor2(self):
        def fn(a, b):
            return torch.add(a, b)  # 返回 a 和 b 的加法运算结果

        counter = CompileCounter()  # 创建编译计数器对象

        x = torch.randn(4)  # 创建一个形状为 (4,) 的随机张量 x
        y = 5  # 创建一个整数 y
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)  # 对函数 fn 进行优化并断言
        opt_fn(x, y)  # 调用优化后的函数 fn

        assert counter.op_count == 1  # 断言编译计数器的操作计数为 1

    def test_add_tensor_list(self):
        def fn(lst):
            return lst[0] + lst[1]  # 返回列表 lst 中第一个元素和第二个元素的和

        counter = CompileCounter()  # 创建编译计数器对象
        x = torch.randn(4)  # 创建一个形状为 (4,) 的随机张量 x
        y = 5  # 创建一个整数 y
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)  # 对函数 fn 进行优化并断言
        opt_fn([x, y])  # 调用优化后的函数 fn，传入列表作为参数

        assert counter.op_count == 1  # 断言编译计数器的操作计数为 1

    def test_add_tensor_dict(self):
        def fn(dt):
            return dt["a"] + dt["b"]  # 返回字典 dt 中键为 "a" 和 "b" 的值的和

        counter = CompileCounter()  # 创建编译计数器对象
        x = torch.randn(4)  # 创建一个形状为 (4,) 的随机张量 x
        y = 5  # 创建一个整数 y
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)  # 对函数 fn 进行优化并断言
        opt_fn({"a": x, "b": y})  # 调用优化后的函数 fn，传入字典作为参数

        assert counter.op_count == 1  # 断言编译计数器的操作计数为 1
    def test_do_not_skip_side_effects(self):
        # https://github.com/pytorch/pytorch/issues/110765

        # 通过调用 torch._utils.is_compiling()，
        # 可能会引发与 eager 模式不一致的副作用，
        # 因此我们强制 dynamo 提交计算图，
        # 即使它没有执行任何张量操作也要这样做
        global _variable, _variable_2

        # 对于模式在 1 到 6 之间的循环
        for mode in range(1, 7):
            # 重置 dynamo 状态
            torch._dynamo.reset()

            # 初始化全局变量
            _variable = 0
            _variable_2 = 0

            # 创建 MyModule 实例，使用特定的模式
            mod = MyModule(mode=mode)
            # 对模型应用优化，根据不同的模式选择是否启用 nopython
            model = torch._dynamo.optimize(backend="eager", nopython=mode != 6)(mod)
            # 断言 _variable 和 _variable_2 均为 0
            assert _variable == 0
            assert _variable_2 == 0

            # 调用模型，传入张量 [1]
            model(torch.tensor([1]))
            # 断言 _variable 变为 1，_variable_2 仍为 0
            assert _variable == 1
            assert _variable_2 == 0

            # 再次调用模型，传入张量 [1]
            model(torch.tensor([1]))
            # 断言 _variable 变为 2，_variable_2 仍为 0
            assert _variable == 2
            assert _variable_2 == 0
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中执行）
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的 run_tests 函数，用于执行测试用例
    run_tests()
```