# `.\pytorch\test\dynamo\test_compile.py`

```
# Owner(s): ["module: dynamo"]

# 导入所需模块和库
import inspect  # 用于检查对象的属性和方法
import io  # 用于处理文件流
import os  # 提供操作系统相关的功能
import tempfile  # 用于创建临时文件和目录
from unittest.mock import patch  # 用于模拟对象的库

import torch  # PyTorch 深度学习框架
from torch._dynamo.test_case import run_tests, TestCase  # 导入测试相关的类和函数
from torch._dynamo.testing import CompileCounter  # 导入用于计数编译次数的类

# 定义一个简单的神经网络模型
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)  # 添加一个线性层
        self.relu = torch.nn.ReLU()  # 添加一个ReLU激活函数层

    def forward(self, x):
        return self.relu(self.linear(x))  # 前向传播函数

# 测试类，用于测试编译和保存功能
class InPlaceCompilationTests(TestCase):
    def test_compilation(self):
        torch._dynamo.reset()  # 重置Dynamo的状态
        model = ToyModel()  # 创建ToyModel实例
        cnt = CompileCounter()  # 创建编译计数器实例
        model.compile(backend=cnt)  # 编译模型并指定编译器
        x = torch.randn(10, 10)  # 创建输入张量
        model(x)  # 进行一次前向传播
        self.assertEqual(cnt.frame_count, 1)  # 断言编译计数为1次

    def test_overwrite_call_impl(self):
        torch._dynamo.reset()  # 重置Dynamo的状态
        model = ToyModel()  # 创建ToyModel实例
        self.assertTrue(model._compiled_call_impl is None)  # 断言未编译的调用实现为空
        model.compile()  # 编译模型
        self.assertTrue(model._compiled_call_impl is not None)  # 断言编译后的调用实现不为空

    def test_save(self):
        torch._dynamo.reset()  # 重置Dynamo的状态
        model = ToyModel()  # 创建ToyModel实例
        model.compile()  # 编译模型
        model(torch.randn(1, 10))  # 进行一次前向传播

        # 使用临时目录保存和加载模型
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model, os.path.join(tmpdirname, "model.pt"))  # 保存模型
            loaded_model = torch.load(os.path.join(tmpdirname, "model.pt"))  # 加载模型
            loaded_model(torch.randn(1, 10))  # 进行一次前向传播

    def test_state_dict_save(self):
        torch._dynamo.reset()  # 重置Dynamo的状态
        model = ToyModel()  # 创建ToyModel实例
        model.compile()  # 编译模型
        model(torch.randn(1, 10))  # 进行一次前向传播

        # 使用临时目录保存和加载模型的状态字典
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model.state_dict(), os.path.join(tmpdirname, "model.pt"))  # 保存模型的状态字典
            loaded_model = ToyModel()  # 创建新的ToyModel实例
            loaded_model.load_state_dict(
                torch.load(os.path.join(tmpdirname, "model.pt"))  # 加载保存的状态字典
            )
            loaded_model(torch.randn(1, 10))  # 进行一次前向传播

    def test_jit_save(self):
        torch._dynamo.reset()  # 重置Dynamo的状态
        model = ToyModel()  # 创建ToyModel实例
        model.compile()  # 编译模型
        model(torch.randn(1, 10))  # 进行一次前向传播
        scripted_model = torch.jit.script(model)  # 对模型进行脚本化

        # 使用临时目录保存和加载脚本化模型
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.jit.save(scripted_model, os.path.join(tmpdirname, "model.pt"))  # 保存脚本化模型
            loaded_model = torch.jit.load(os.path.join(tmpdirname, "model.pt"))  # 加载脚本化模型
            loaded_model(torch.randn(1, 10))  # 进行一次前向传播
    # 定义一个测试方法，用于测试编译回调函数的功能
    def test_compilation_callback(self):
        # 重置 torch._dynamo 模块状态，确保测试环境干净
        torch._dynamo.reset()

        # 注册编译开始时的回调函数
        @torch._dynamo.on_compile_start
        def start_callback():
            print("Compilation started.")

        # 注册编译结束时的回调函数
        @torch._dynamo.on_compile_end
        def end_callback():
            print("Compilation ended.")

        # 创建一个 ToyModel 实例
        mod = ToyModel()
        # 生成一个大小为 10x10 的随机张量
        x = torch.randn(10, 10)

        # 使用 patch 将标准输出重定向到 mock_stdout 对象
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            # 对模型进行编译，使用 eager 模式，并启用完整图优化
            opt_mod = torch.compile(backend="eager", fullgraph=True)(mod)
            # 对优化后的模型进行计算
            opt_mod(x)
            # 获取 mock_stdout 中的输出内容，并移除首尾空白字符
            printed_output = mock_stdout.getvalue().strip()

        # 断言输出内容与预期输出相符
        self.assertEqual(printed_output, "Compilation started.\nCompilation ended.")

    # 定义另一个测试方法，测试带有图断点的编译回调函数的功能
    def test_compilation_callback_with_graph_break(self):
        # 重置 torch._dynamo 模块状态，确保测试环境干净
        torch._dynamo.reset()
        # 初始化计数器为 0
        counter = 0

        # 注册编译开始时的回调函数
        @torch._dynamo.on_compile_start
        def start_callback():
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        # 注册编译结束时的回调函数
        @torch._dynamo.on_compile_end
        def end_callback():
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        # 使用 eager 模式编译以下函数
        @torch.compile(backend="eager")
        def fn(x):
            # 对输入张量 x 加 1
            x = x + 1
            # 创建图断点，中断图的构建
            torch._dynamo.graph_break()
            # 返回对 x 中每个元素执行正弦函数的结果
            return torch.sin(x)

        # 生成一个大小为 10x10 的随机张量
        x = torch.randn(10, 10)

        # 使用 patch 将标准输出重定向到 mock_stdout 对象
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            # 调用函数 fn，并获取输出到 mock_stdout 的内容
            fn(x)
            printed_output = mock_stdout.getvalue().strip()

        # 断言输出内容与预期输出相符
        self.assertEqual(
            printed_output, "Counter = 1\nCounter = 2\nCounter = 3\nCounter = 4"
        )
# The private variants of the below functions are extensively tested
# So as long as the signatures match we're good
class PublicTorchCompilerTests(TestCase):
    # 定义一个方法，用于比较公共函数和私有函数的签名是否匹配
    def check_signature(self, public_fn_name, private_fn_name, private_namespace):
        # 获取torch.compiler中的公共函数对象
        public_fn = getattr(torch.compiler, public_fn_name)
        # 获取私有命名空间中的对应私有函数对象
        private_fn = getattr(private_namespace, private_fn_name)

        # 获取公共函数和私有函数的签名对象
        public_sig = inspect.signature(public_fn)
        private_sig = inspect.signature(private_fn)

        # 断言检查公共函数和私有函数的签名是否一致，若不一致则输出详细信息
        self.assertEqual(
            public_sig,
            private_sig,
            f"Signatures do not match for function {public_fn_name}() \n Public: {public_sig} \n Private: {private_sig}",
        )

    # 测试动态签名的方法
    def test_dynamo_signatures(self):
        # 定义需要检查的函数名列表
        function_names = [
            "reset",
            "allow_in_graph",
            "list_backends",
            "assume_constant_result",
            "disable",
        ]

        # 遍历每个函数名，调用check_signature方法检查公共函数和私有函数的签名是否一致
        for fn_name in function_names:
            self.check_signature(fn_name, fn_name, torch._dynamo)


if __name__ == "__main__":
    # 运行测试
    run_tests()
```