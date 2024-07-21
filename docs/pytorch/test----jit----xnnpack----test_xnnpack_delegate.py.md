# `.\pytorch\test\jit\xnnpack\test_xnnpack_delegate.py`

```
# Owner(s): ["oncall: jit"]

# 导入unittest模块，用于编写和运行测试用例
import unittest

# 导入PyTorch库及其C扩展
import torch
import torch._C

# 加载xnnpack后端库
torch.ops.load_library("//caffe2:xnnpack_backend")

# 定义测试类TestXNNPackBackend，继承自unittest.TestCase
class TestXNNPackBackend(unittest.TestCase):

    # 测试xnnpack常量数据
    def test_xnnpack_constant_data(self):
        # 定义Module类，继承自torch.nn.Module
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._constant = torch.ones(4, 4, 4)

            def forward(self, x):
                return x + self._constant

        # 使用torch.jit.script对Module进行脚本化
        scripted_module = torch.jit.script(Module())

        # 将脚本化的Module转换为xnnpack后端支持的形式
        lowered_module = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [torch.randn(4, 4, 4)],
                    "outputs": [torch.randn(4, 4, 4)],
                }
            },
        )

        # 遍历多次测试模型输出与预期输出的接近程度
        for i in range(0, 20):
            sample_input = torch.randn(4, 4, 4)
            actual_output = scripted_module(sample_input)
            expected_output = lowered_module(sample_input)
            # 断言实际输出与预期输出在一定误差范围内相等
            self.assertTrue(
                torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
            )

    # 测试xnnpack的转换（lowering）过程
    def test_xnnpack_lowering(self):
        # 定义Module类，仅执行输入与自身的加法操作
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        # 使用torch.jit.script对Module进行脚本化
        scripted_module = torch.jit.script(Module())

        # 测试错误的编译规范是否引发特定错误消息
        faulty_compile_spec = {
            "backward": {
                "inputs": [torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = 'method_compile_spec does not contain the "forward" key.'
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                scripted_module,
                faulty_compile_spec,
            )

        # 测试不匹配的编译规范是否引发特定错误消息
        mismatch_compile_spec = {
            "forward": {
                "inputs": [torch.zeros(1), torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = (
            "method_compile_spec inputs do not match expected number of forward inputs"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack", scripted_module, mismatch_compile_spec
            )

        # 将Module转换为xnnpack后端支持的形式
        lowered = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [torch.zeros(1)],
                    "outputs": [torch.zeros(1)],
                }
            },
        )
        lowered(torch.zeros(1))
    def test_xnnpack_backend_add(self):
        # 定义一个简单的 Torch Module，实现两个输入张量的加法
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                return z
        
        # 创建 AddModule 实例
        add_module = AddModule()
        # 准备样本输入张量和输出张量
        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        # 对 AddModule 进行 Torch 脚本化
        add_module = torch.jit.script(add_module)
        # 获取预期输出
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        # 将 AddModule 降级到 xnnpack 后端
        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0].clone(), sample_inputs[1].clone()],
                    "outputs": [sample_output],
                }
            },
        )

        # 获取实际输出
        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        # 断言实际输出与预期输出在一定误差范围内相等
        self.assertTrue(
            torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_broadcasting(self):
        # 定义一个 Torch Module，实现张量的广播加法
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # 创建 AddModule 实例
        add_module = AddModule()
        # 准备样本输入张量和输出张量
        sample_inputs = (torch.rand(5, 1, 4, 1), torch.rand(3, 1, 1))
        sample_output = torch.zeros(5, 3, 4, 1)

        # 对 AddModule 进行 Torch 脚本化
        add_module = torch.jit.script(add_module)
        # 获取预期输出
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        # 将 AddModule 降级到 xnnpack 后端
        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0], sample_inputs[1]],
                    "outputs": [sample_output],
                }
            },
        )

        # 获取实际输出
        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        # 断言实际输出与预期输出在一定误差范围内相等
        self.assertTrue(
            torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_unsupported(self):
        # 定义一个包含不支持操作的 Torch Module
        class AddSpliceModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y[:, :, 1, :]
                return z

        # 准备样本输入张量和输出张量
        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        # 准备错误信息
        error_msg = (
            "the module contains the following unsupported ops:\n"
            "aten::select\n"
            "aten::slice\n"
        )

        # 对 AddSpliceModule 进行 Torch 脚本化
        add_module = torch.jit.script(AddSpliceModule())
        # 使用断言检查是否会抛出预期的 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                add_module,
                {
                    "forward": {
                        "inputs": [sample_inputs[0], sample_inputs[1]],
                        "outputs": [sample_output],
                    }
                },
            )
```