# `.\pytorch\test\quantization\jit\test_deprecated_jit_quant.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入 PyTorch 模块
from torch.testing._internal.common_quantization import skipIfNoFBGEMM  # 导入测试中的条件跳过装饰器
from torch.testing._internal.jit_utils import JitTestCase  # 导入 JIT 测试用例基类


class TestDeprecatedJitQuantized(JitTestCase):
    @skipIfNoFBGEMM  # 如果没有 FBGEMM 支持，则跳过此测试函数
    def test_rnn_cell_quantized(self):
        d_in, d_hid = 2, 2  # 设置输入和隐藏状态的维度

        # 对于每种 RNN 单元类型，创建不同的测试单元并进行测试
        for cell in [
            torch.nn.LSTMCell(d_in, d_hid).float(),
            torch.nn.GRUCell(d_in, d_hid).float(),
            torch.nn.RNNCell(d_in, d_hid).float(),
        ]:
            if isinstance(cell, torch.nn.LSTMCell):
                num_chunks = 4  # LSTM 单元的块数
            elif isinstance(cell, torch.nn.GRUCell):
                num_chunks = 3  # GRU 单元的块数
            elif isinstance(cell, torch.nn.RNNCell):
                num_chunks = 1  # RNN 单元的块数

            # 替换参数值，使得数值范围正好为 255，确保在量化 GEMM 调用中没有量化误差
            #
            # 注意，当前的实现不支持累积值超出 16 位整数可表示的范围，
            # 而是会产生饱和值。因此，在测试中必须小心，确保我们的点积不会
            # 超出 int16 的范围，例如 (255*127+255*127) = 64770。
            # 因此，我们在这里硬编码测试值，并确保包含有符号数。
            vals = [
                [100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
            ]
            vals = vals[: d_hid * num_chunks]  # 根据隐藏状态和块数修剪值列表
            cell.weight_ih = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )
            cell.weight_hh = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )

            # 使用断言检查是否捕获到预期的 RuntimeError 异常，并且异常信息包含特定文本
            with self.assertRaisesRegex(
                RuntimeError,
                "quantize_rnn_cell_modules function is no longer supported",
            ):
                cell = torch.jit.quantized.quantize_rnn_cell_modules(cell)

    @skipIfNoFBGEMM  # 如果没有 FBGEMM 支持，则跳过此测试函数
    # 定义一个测试方法，用于测试量化循环神经网络（RNN）
    def test_rnn_quantized(self):
        # 设置输入和隐藏层维度
        d_in, d_hid = 2, 2

        # 遍历不同类型的RNN单元：LSTM和GRU
        for cell in [
            torch.nn.LSTM(d_in, d_hid).float(),
            torch.nn.GRU(d_in, d_hid).float(),
        ]:
            # 替换参数值，确保数值范围精确为255，以保证在量化GEMM调用中没有量化误差。这是为了测试目的。
            #
            # 注意，当前的实现不支持超出16位整数表示范围的累积值，而是导致饱和值。因此，在测试中，我们必须确保不会产生超出int16范围的点积，例如
            # (255*127+255*127) = 64770。因此，在这里我们硬编码测试值，并确保混合符号。
            vals = [
                [100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
            ]
            # 根据RNN单元类型确定num_chunks（块数）
            if isinstance(cell, torch.nn.LSTM):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRU):
                num_chunks = 3
            # 截取vals列表，以匹配隐藏层大小乘以num_chunks
            vals = vals[: d_hid * num_chunks]
            
            # 设置权重矩阵的输入到隐藏层和隐藏到隐藏层，使用torch.nn.Parameter表示，不需要梯度计算
            cell.weight_ih_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )
            cell.weight_hh_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )

            # 断言异常，验证quantize_rnn_modules函数不再支持
            with self.assertRaisesRegex(
                RuntimeError, "quantize_rnn_modules function is no longer supported"
            ):
                cell_int8 = torch.jit.quantized.quantize_rnn_modules(
                    cell, dtype=torch.int8
                )

            # 断言异常，验证quantize_rnn_modules函数不再支持
            with self.assertRaisesRegex(
                RuntimeError, "quantize_rnn_modules function is no longer supported"
            ):
                cell_fp16 = torch.jit.quantized.quantize_rnn_modules(
                    cell, dtype=torch.float16
                )
    # 检查是否支持量化的后端引擎中包含 "fbgemm"
    if "fbgemm" in torch.backends.quantized.supported_engines:

        # 定义测试量化模块的方法
        def test_quantization_modules(self):
            # 定义线性层的输入和输出维度
            K1, N1 = 2, 2

            # 定义一个继承自 torch.nn.Module 的类 FooBar
            class FooBar(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个包含输入输出维度的线性层，使用单精度浮点数
                    self.linear1 = torch.nn.Linear(K1, N1).float()

                # 定义前向传播方法
                def forward(self, x):
                    # 应用线性层到输入 x
                    x = self.linear1(x)
                    return x

            # 创建 FooBar 类的实例 fb
            fb = FooBar()

            # 设置线性层的权重，不可训练
            fb.linear1.weight = torch.nn.Parameter(
                torch.tensor([[-150, 100], [100, -150]], dtype=torch.float),
                requires_grad=False,
            )

            # 设置线性层的偏置，不可训练
            fb.linear1.bias = torch.nn.Parameter(
                torch.zeros_like(fb.linear1.bias), requires_grad=False
            )

            # 创建一个随机输入张量 x，形状为 (1, K1)，数值在 -0.05 到 0.05 之间
            x = (torch.rand(1, K1).float() - 0.5) / 10.0

            # 创建一个参考输出张量 value，形状为 (1, N1)
            value = torch.tensor([[100, -150]], dtype=torch.float)

            # 计算 fb 对 value 的前向传播结果
            y_ref = fb(value)

            # 使用断言检查调用 quantize_linear_modules 函数是否引发 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, "quantize_linear_modules function is no longer supported"
            ):
                fb_int8 = torch.jit.quantized.quantize_linear_modules(fb)

            # 使用断言检查调用 quantize_linear_modules 函数是否引发 RuntimeError 异常，并指定 torch.float16 类型
            with self.assertRaisesRegex(
                RuntimeError, "quantize_linear_modules function is no longer supported"
            ):
                fb_fp16 = torch.jit.quantized.quantize_linear_modules(fb, torch.float16)

    # 使用装饰器 @skipIfNoFBGEMM，跳过没有 FBGEMM 支持的情况下执行的测试方法
    @skipIfNoFBGEMM
    def test_erase_class_tensor_shapes(self):
        # 定义一个继承自 torch.nn.Module 的线性层类 Linear
        class Linear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                # 创建一个空的量化权重张量 qweight，形状为 [out_features, in_features]，类型为 torch.qint8
                qweight = torch._empty_affine_quantized(
                    [out_features, in_features],
                    scale=1,
                    zero_point=0,
                    dtype=torch.qint8,
                )
                # 使用量化操作函数 linear_prepack 对权重进行打包
                self._packed_weight = torch.ops.quantized.linear_prepack(qweight)

            # 导出 __getstate__ 方法
            @torch.jit.export
            def __getstate__(self):
                # 返回解包后的权重和训练状态
                return (
                    torch.ops.quantized.linear_unpack(self._packed_weight)[0],
                    self.training,
                )

            # 定义前向传播方法
            def forward(self):
                # 返回打包后的权重
                return self._packed_weight

            # 导出 __setstate__ 方法
            @torch.jit.export
            def __setstate__(self, state):
                # 使用量化操作函数 linear_prepack 对状态的第一个元素进行打包
                self._packed_weight = torch.ops.quantized.linear_prepack(state[0])
                # 设置训练状态
                self.training = state[1]

            # 定义权重属性
            @property
            def weight(self):
                # 返回解包后的权重
                return torch.ops.quantized.linear_unpack(self._packed_weight)[0]

            # 定义权重的设置方法
            @weight.setter
            def weight(self, w):
                # 使用量化操作函数 linear_prepack 对权重 w 进行打包
                self._packed_weight = torch.ops.quantized.linear_prepack(w)

        # 使用 torch._jit_internal._disable_emit_hooks() 上下文管理器，禁用图形传输钩子
        with torch._jit_internal._disable_emit_hooks():
            # 对 Linear 类进行脚本化处理，输入维度为 10，输出维度为 10
            x = torch.jit.script(Linear(10, 10))
            # 使用 _jit_pass_erase_shape_information 函数擦除张量形状信息
            torch._C._jit_pass_erase_shape_information(x.graph)
# 如果当前脚本被直接运行而不是被导入作为模块，则抛出运行时错误
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )
```