# `.\pytorch\test\quantization\core\experimental\test_quantized_tensor.py`

```
# Owner(s): ["oncall: quantization"]

# 导入必要的库
import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT

# 定义测试类 TestQuantizedTensor，继承自 unittest.TestCase
class TestQuantizedTensor(unittest.TestCase):
    r""" Tests int_repr on APoTQuantizer with random tensor2quantize
    and hard-coded values
    """

    # 定义测试方法 test_int_repr
    def test_int_repr(self):
        # 生成包含随机浮点值的张量
        tensor2quantize = tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])

        # 创建 APoTObserver 实例，设置 b=4, k=2
        observer = APoTObserver(b=4, k=2)

        # 对 tensor2quantize 进行前向观察
        observer.forward(tensor2quantize)

        # 计算量化参数 qparams
        qparams = observer.calculate_qparams(signed=False)

        # 使用 APoT 方法对 tensor2quantize 进行量化
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                alpha=qparams[0],
                                gamma=qparams[1],
                                quantization_levels=qparams[2],
                                level_indices=qparams[3])

        # 获取量化后的整数表示的张量数据
        qtensor_data = qtensor.int_repr().int()

        # 根据预期结果计算 qtensor 的预期整数表示数据
        # 使用 level_indices 将每个浮点值量化到最接近的量化级别
        expected_qtensor_data = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.int32)

        # 断言 qtensor_data 和 expected_qtensor_data 是否相等
        self.assertTrue(torch.equal(qtensor_data, expected_qtensor_data))

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    unittest.main()
```