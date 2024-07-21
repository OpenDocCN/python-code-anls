# `.\pytorch\test\quantization\core\experimental\test_linear.py`

```py
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库
from torch.ao.quantization.experimental.linear import LinearAPoT  # 导入LinearAPoT量化模块
from torch.nn.modules.linear import Linear  # 导入PyTorch的线性层模块
import unittest  # 导入unittest模块，用于编写和运行测试

class TestNonUniformObserver(unittest.TestCase):
    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (4,4) and k=1
    """
    def test_linear_APoT_k1(self):
        # weight: fp tensor
        weight = 1000 * torch.rand(4, 4)  # 创建一个形状为(4, 4)的随机浮点权重张量

        # activtion: fp32 tensor with ~ integer values
        activation = torch.randint(low=0, high=255, size=(4, 4), dtype=torch.float)  # 创建一个形状为(4, 4)的随机整数值浮点张量作为激活值

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 1)  # 创建LinearAPoT对象，使用权重、量化位宽8和k=1
        apot_linear_result = apot_linear(activation)  # 调用apot_linear对象进行前向传播计算结果

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)  # 创建一个标准的PyTorch线性层，与apot_linear对比用

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)  # 将apot_linear的权重转换为浮点数张量
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)  # 将apot_linear的权重作为参数设置给fp_linear
        fp_linear.weight = fp_linear_weight  # 将fp_linear的权重设置为apot_linear的权重

        fp_linear_result = fp_linear(activation).data  # 使用fp_linear进行前向传播计算结果

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))  # 断言apot_linear计算结果与fp_linear计算结果是否相等

    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (5,3), (3, 5) and k=2
    """
    def test_linear_APoT_k2(self):
        # weight: fp tensor
        weight = 1000 * torch.rand(5, 3)  # 创建一个形状为(5, 3)的随机浮点权重张量

        # activtion: fp32 tensor with ~ integer values
        # note: transpose of activation matrix will have dimension (3, 5)
        activation = torch.randint(low=0, high=255, size=(5, 3), dtype=torch.float)  # 创建一个形状为(5, 3)的随机整数值浮点张量作为激活值

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 2)  # 创建LinearAPoT对象，使用权重、量化位宽8和k=2
        apot_linear_result = apot_linear(activation)  # 调用apot_linear对象进行前向传播计算结果

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)  # 创建一个标准的PyTorch线性层，与apot_linear对比用

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)  # 将apot_linear的权重转换为浮点数张量
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)  # 将apot_linear的权重作为参数设置给fp_linear
        fp_linear.weight = fp_linear_weight  # 将fp_linear的权重设置为apot_linear的权重

        fp_linear_result = fp_linear(activation).data  # 使用fp_linear进行前向传播计算结果

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))  # 断言apot_linear计算结果与fp_linear计算结果是否相等

if __name__ == '__main__':
    unittest.main()  # 运行unittest框架的主函数，执行测试用例
```