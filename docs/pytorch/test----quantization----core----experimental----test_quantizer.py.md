# `.\pytorch\test\quantization\core\experimental\test_quantizer.py`

```
# 导入PyTorch库
import torch
# 导入量化函数
from torch import quantize_per_tensor
# 导入MinMaxObserver观察器
from torch.ao.quantization.observer import MinMaxObserver
# 导入APoTObserver观察器
from torch.ao.quantization.experimental.observer import APoTObserver
# 导入APoTQuantizer量化器
from torch.ao.quantization.experimental.quantizer import APoTQuantizer, quantize_APoT, dequantize_APoT
# 导入单元测试模块
import unittest
# 导入随机数生成模块
import random

# 定义测试类TestQuantizer，继承自unittest.TestCase
class TestQuantizer(unittest.TestCase):
    r""" Tests quantize_APoT result on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform quantization
        (non-uniform quantization reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 8
        * k: 1
    """
    
    # 定义测试函数test_quantize_APoT_rand_k1
    def test_quantize_APoT_rand_k1(self):
        # 生成随机大小的tensor2quantize张量，大小在1到20之间
        size = random.randint(1, 20)
        # 生成随机的浮点数值在0到1000之间的张量
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        # 创建APoTObserver观察器对象，设置b=8, k=1，并对tensor2quantize进行观察
        apot_observer = APoTObserver(b=8, k=1)
        apot_observer(tensor2quantize)
        # 计算alpha, gamma, quantization_levels和level_indices参数
        alpha, gamma, quantization_levels, level_indices = apot_observer.calculate_qparams(signed=False)

        # 使用quantize_APoT函数进行APoT量化，传入相关参数
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                alpha=alpha,
                                gamma=gamma,
                                quantization_levels=quantization_levels,
                                level_indices=level_indices)

        # 创建MinMaxObserver观察器对象，并对tensor2quantize进行观察
        uniform_observer = MinMaxObserver()
        uniform_observer(tensor2quantize)
        # 计算scale和zero_point参数
        scale, zero_point = uniform_observer.calculate_qparams()

        # 使用torch.quantize_per_tensor进行均匀量化，传入相关参数，并转换为整数表示
        uniform_quantized = quantize_per_tensor(input=tensor2quantize,
                                                scale=scale,
                                                zero_point=zero_point,
                                                dtype=torch.quint8).int_repr()

        # 将qtensor的数据转换为整数表示
        qtensor_data = qtensor.data.int()
        # 将uniform_quantized的数据转换为整数表示
        uniform_quantized_tensor = uniform_quantized.data.int()

        # 使用assertTrue断言，验证qtensor_data和uniform_quantized_tensor是否相等
        self.assertTrue(torch.equal(qtensor_data, uniform_quantized_tensor))

    r""" Tests quantize_APoT for k != 1.
        Tests quantize_APoT result on random 1-dim tensor and hardcoded values for
        b=4, k=2 by comparing results to hand-calculated results from APoT paper
        https://arxiv.org/pdf/1909.13144.pdf
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_quantize_APoT_k2(self):
        r"""
        given b = 4, k = 2, alpha = 1.0, we know:
        (from APoT paper example: https://arxiv.org/pdf/1909.13144.pdf)

        quantization_levels = tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667,
        0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])

        level_indices = tensor([ 0, 3, 12, 15,  2, 14,  8, 11, 10, 1, 13,  9,  4,  7,  6,  5]))
        """

        # generate tensor with random fp values
        tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])

        # Initialize APoTObserver with parameters b=4, k=2
        observer = APoTObserver(b=4, k=2)
        # Perform forward pass of the tensor through the observer
        observer.forward(tensor2quantize)
        # Calculate alpha, gamma, quantization_levels, level_indices based on observations
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # Quantize tensor2quantize using APoT quantization function
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                alpha=alpha,
                                gamma=gamma,
                                quantization_levels=quantization_levels,
                                level_indices=level_indices)

        # Convert qtensor data to integers
        qtensor_data = qtensor.data.int()

        # Define expected quantized tensor values based on level_indices
        expected_qtensor = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.int32)

        # Assert that qtensor_data matches expected_qtensor
        self.assertTrue(torch.equal(qtensor_data, expected_qtensor))

    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    # 定义一个测试方法，用于测试 dequantize_APoT 函数的结果
    def test_dequantize_quantize_rand_b4(self):
        # 创建一个 APoTObserver 对象，使用参数 4 和 2
        observer = APoTObserver(4, 2)

        # 生成一个随机大小的 tensor2quantize，大小在 1 到 20 之间
        size = random.randint(1, 20)

        # 创建 tensor2quantize，包含随机生成的浮点数值，范围在 0 到 1000 之间
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        # 将 tensor2quantize 输入到 observer 的 forward 方法中
        observer.forward(tensor2quantize)

        # 调用 observer 的 calculate_qparams 方法，获取 alpha、gamma、quantization_levels 和 level_indices
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # 使用给定参数创建一个 mock APoT 对象 original_apot
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize,
                                      alpha=alpha,
                                      gamma=gamma,
                                      quantization_levels=quantization_levels,
                                      level_indices=level_indices)

        # 对 original_apot 中的数据进行克隆，并转换为整数类型
        original_input = torch.clone(original_apot.data).int()

        # 对 original_apot 进行反量化操作，得到反量化结果 dequantize_result
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)

        # 再次对 dequantize_result 进行量化操作，得到最终的 APoT 对象 final_apot
        final_apot = quantize_APoT(tensor2quantize=dequantize_result,
                                   alpha=alpha,
                                   gamma=gamma,
                                   quantization_levels=quantization_levels,
                                   level_indices=level_indices)

        # 获取 final_apot 中的数据，并转换为整数类型
        result = final_apot.data.int()

        # 使用 assertTrue 方法断言 original_input 与 result 是否完全相等
        self.assertTrue(torch.equal(original_input, result))

    # 文档字符串，描述了该测试方法的功能和输入参数
    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 12
        * k: 4
    """
    # 定义测试方法，用于测试 dequantize 和 quantize 的正确性
    def test_dequantize_quantize_rand_b6(self):
        # 创建一个 APoTObserver 对象，设置参数为 (12, 4)
        observer = APoTObserver(12, 4)

        # 生成一个随机大小的 tensor2quantize，大小在 1 到 20 之间
        size = random.randint(1, 20)

        # 创建一个 tensor2quantize，其中元素是在 [0, 1000] 范围内的随机浮点数
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        # 将 tensor2quantize 传入 observer 的 forward 方法进行处理
        observer.forward(tensor2quantize)

        # 调用 observer 的 calculate_qparams 方法计算 alpha、gamma、quantization_levels 和 level_indices
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # 创建一个模拟的 quantize_APoT 对象 original_apot，使用 tensor2quantize 和计算出的参数
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize,
                                      alpha=alpha,
                                      gamma=gamma,
                                      quantization_levels=quantization_levels,
                                      level_indices=level_indices)

        # 克隆 original_apot 的数据部分并转换为整数类型作为 original_input
        original_input = torch.clone(original_apot.data).int()

        # 对 original_apot 进行反量化操作
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)

        # 对反量化后的结果进行量化
        final_apot = quantize_APoT(tensor2quantize=dequantize_result,
                                   alpha=alpha,
                                   gamma=gamma,
                                   quantization_levels=quantization_levels,
                                   level_indices=level_indices)

        # 提取 final_apot 的数据部分并转换为整数类型作为 result
        result = final_apot.data.int()

        # 使用 assertTrue 方法断言 original_input 和 result 是否相等
        self.assertTrue(torch.equal(original_input, result))

    # 定义测试方法，用于测试 dequantize_APoT 返回结果的维度是否与输入相同
    def test_dequantize_dim(self):
        # 创建一个 APoTObserver 对象，设置参数为 (4, 2)
        observer = APoTObserver(4, 2)

        # 分别生成三个随机大小的维度 size1, size2, size3
        size1 = random.randint(1, 20)
        size2 = random.randint(1, 20)
        size3 = random.randint(1, 20)

        # 创建一个三维 tensor2quantize，元素为在 [0, 1000] 范围内的随机浮点数
        tensor2quantize = 1000 * torch.rand(size1, size2, size3, dtype=torch.float)

        # 将 tensor2quantize 传入 observer 的 forward 方法进行处理
        observer.forward(tensor2quantize)

        # 调用 observer 的 calculate_qparams 方法计算 alpha、gamma、quantization_levels 和 level_indices
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # 创建一个 quantize_APoT 对象 original_apot，使用 tensor2quantize 和计算出的参数
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize,
                                      alpha=alpha,
                                      gamma=gamma,
                                      quantization_levels=quantization_levels,
                                      level_indices=level_indices)

        # 对 original_apot 进行反量化操作
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)

        # 使用 assertEqual 方法断言 original_apot 的数据维度与 dequantize_result 的数据维度相同
        self.assertEqual(original_apot.data.size(), dequantize_result.size())
    # 定义一个测试方法 test_q_apot_alpha，用于测试 APoTQuantizer 类中的 q_apot_alpha 方法是否会抛出 NotImplementedError 异常
    def test_q_apot_alpha(self):
        # 使用断言语句，期望调用 APoTQuantizer 类的 q_apot_alpha 方法会抛出 NotImplementedError 异常
        with self.assertRaises(NotImplementedError):
            # 调用 APoTQuantizer 类的 q_apot_alpha 方法，预期会抛出 NotImplementedError 异常
            APoTQuantizer.q_apot_alpha(self)
# 如果当前脚本作为主程序执行（而不是被导入其他模块），则执行单元测试框架的主函数
if __name__ == '__main__':
    # 运行 Python 的单元测试框架的主函数，用于执行测试用例
    unittest.main()
```