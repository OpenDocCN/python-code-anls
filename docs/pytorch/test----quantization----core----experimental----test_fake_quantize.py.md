# `.\pytorch\test\quantization\core\experimental\test_fake_quantize.py`

```py
# Owner(s): ["oncall: quantization"]

# 导入必要的库和模块
import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.experimental.fake_quantize_function import fake_quantize_function

# 导入模块中的函数并赋予别名
forward_helper = fake_quantize_function.forward
backward = fake_quantize_function.backward
from torch.autograd import gradcheck

class TestFakeQuantize(unittest.TestCase):
    r""" Tests fake quantize calculate_qparams() method
         by comparing with result from observer calculate_qparams.
         Uses hard-coded values: alpha=1.0, b=4, k=2.
    """
    def test_fake_calc_qparams(self):
        # 创建一个 APoTFakeQuantize 对象并设置属性
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.activation_post_process.min_val = torch.tensor([0.0])
        apot_fake.activation_post_process.max_val = torch.tensor([1.0])

        # 调用 calculate_qparams 方法计算量化参数
        alpha, gamma, quantization_levels, level_indices = apot_fake.calculate_qparams(signed=False)

        # 创建一个 APoTObserver 对象并设置属性
        observer = APoTObserver(b=4, k=2)
        observer.min_val = torch.tensor([0.0])
        observer.max_val = torch.tensor([1.0])

        # 调用 observer 的 calculate_qparams 方法计算期望的量化参数
        qparams_expected = observer.calculate_qparams(signed=False)

        # 断言计算得到的量化参数与期望的量化参数一致
        self.assertEqual(alpha, qparams_expected[0])
        self.assertTrue(torch.equal(gamma, qparams_expected[1]))
        self.assertTrue(torch.equal(quantization_levels, qparams_expected[2]))
        self.assertTrue(torch.equal(level_indices, qparams_expected[3]))

    r""" Tests fake quantize forward() method
         by comparing result with expected
         quant_dequant_APoT mapping of input tensor.
         Uses input tensor with random values from 0 -> 1000
         and APoT observer with hard-coded values b=4, k=2
    """
    def test_forward(self):
        # 生成大小为 20 的随机张量，值在 0 到 1000 之间
        X = 1000 * torch.rand(20)

        # 创建一个 APoTObserver 对象并进行前向传播
        observer = APoTObserver(b=4, k=2)
        observer.forward(X)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # 创建一个 APoTFakeQuantize 对象并启用观察器和伪量化
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.enable_observer()
        apot_fake.enable_fake_quant()

        # 对输入张量 X 进行降低精度的前向传播
        X_reduced_precision_fp = apot_fake.forward(torch.clone(X), False)

        # 获取期望的 X_expected，模拟量化 -> 反量化的过程
        X_to_apot = quantize_APoT(X, alpha, gamma, quantization_levels, level_indices)
        X_expected = dequantize_APoT(X_to_apot)

        # 断言降低精度后的张量与期望的张量 X_expected 相等
        self.assertTrue(torch.equal(X_reduced_precision_fp, X_expected))

    r""" Tests fake quantize forward() method
         throws error when qparams are None
    """
    def test_forward_exception(self):
        # 生成一个大小为20的张量，其中包含0到1000之间的随机值，用于量化和反量化操作
        X = 1000 * torch.rand(20)

        # 创建一个带有指定参数的 APoTFakeQuantize 实例，b=4, k=2
        apot_fake = APoTFakeQuantize(b=4, k=2)
        # 禁用观察器以确保 qparams 未设置，所有 qparams 均为 None
        apot_fake.disable_observer()
        # 启用伪量化功能
        apot_fake.enable_fake_quant()

        # 使用断言检查是否会抛出异常
        with self.assertRaises(Exception):
            # 调用 forward 方法，传入 X 的克隆和 False 作为参数
            apot_fake.forward(torch.clone(X), False)

    r""" Tests fake quantize helper backward() method
         using torch.autograd.gradcheck function.
    """
    def test_backward(self):
        # 创建一个包含20个元素的双精度浮点类型张量，并且要求计算梯度
        input = torch.randn(20, dtype=torch.double, requires_grad=True)

        # 创建 APoTObserver 实例，b=4, k=2，并将 input 传入观察器
        observer = APoTObserver(b=4, k=2)
        observer(input)
        # 计算输入张量的量化参数 alpha, gamma, quantization_levels 和 level_indices，其中 signed=False
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # 使用 gradcheck 函数测试 fake_quantize_function 的 backward 方法
        test = gradcheck(fake_quantize_function.apply, (input, alpha, gamma, quantization_levels, level_indices), atol=1e-4)
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 运行单元测试的主函数
    unittest.main()
```