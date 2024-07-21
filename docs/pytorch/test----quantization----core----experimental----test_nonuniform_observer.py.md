# `.\pytorch\test\quantization\core\experimental\test_nonuniform_observer.py`

```py
# 所有权者：["oncall: quantization"]

# 从 torch 库中导入 APoTObserver 类
from torch.ao.quantization.experimental.observer import APoTObserver
# 导入 unittest 模块用于编写和运行测试
import unittest
# 导入 torch 库
import torch

# 定义 TestNonUniformObserver 类，继承自 unittest.TestCase
class TestNonUniformObserver(unittest.TestCase):
    """
    Test case 1: calculate_qparams
    Test that error is thrown when k == 0
    """
    # 定义 test_calculate_qparams_invalid 方法
    def test_calculate_qparams_invalid(self):
        # 创建 APoTObserver 对象，其中 b=0, k=0
        obs = APoTObserver(b=0, k=0)
        # 设置 obs 对象的 min_val 和 max_val 属性为单元素张量 [0.0]
        obs.min_val = torch.tensor([0.0])
        obs.max_val = torch.tensor([0.0])

        # 使用 assertRaises 断言检查是否抛出 AssertionError 异常
        with self.assertRaises(AssertionError):
            # 调用 obs 对象的 calculate_qparams 方法，期望抛出异常
            alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=False)

    """
    Test case 2: calculate_qparams
    APoT paper example: https://arxiv.org/pdf/1909.13144.pdf
    Assume hardcoded parameters:
    * b = 4 (total number of bits across all terms)
    * k = 2 (base bitwidth, i.e. bitwidth of every term)
    * n = 2 (number of additive terms)
    * note: b = k * n
    """
    # 定义 test_calculate_qparams_2terms 方法
    def test_calculate_qparams_2terms(self):
        # 创建 APoTObserver 对象，其中 b=4, k=2
        obs = APoTObserver(b=4, k=2)

        # 设置 obs 对象的 min_val 和 max_val 属性为单元素张量 [0.0], [1.0]
        obs.min_val = torch.tensor([0.0])
        obs.max_val = torch.tensor([1.0])
        
        # 调用 obs 对象的 calculate_qparams 方法，返回 alpha, gamma, quantization_levels, level_indices
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=False)

        # 计算 alpha 的预期值
        alpha_test = torch.max(-obs.min_val, obs.max_val)

        # 使用 self.assertEqual 断言检查 alpha 是否等于 alpha_test
        self.assertEqual(alpha, alpha_test)

        # 计算预期的 gamma 值
        gamma_test = 0
        for i in range(2):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # 使用 self.assertEqual 断言检查 gamma 是否等于 gamma_test
        self.assertEqual(gamma, gamma_test)

        # 使用 self.assertEqual 断言检查 quantization_levels 的长度是否等于 2**4
        quantlevels_size_test = int(len(quantization_levels))
        quantlevels_size = 2**4
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # 使用 self.assertEqual 断言检查 level_indices 的长度是否等于 16
        levelindices_size_test = int(len(level_indices))
        self.assertEqual(levelindices_size_test, 16)

        # 将 level_indices 转换为列表并检查其中的唯一值数量
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
    Test case 3: calculate_qparams
    Assume hardcoded parameters:
    * b = 6 (total number of bits across all terms)
    * k = 2 (base bitwidth, i.e. bitwidth of every term)
    * n = 3 (number of additive terms)
    """
    # 定义一个测试方法，用于测试带有三个参数的APoTObserver类的calculate_qparams方法
    def test_calculate_qparams_3terms(self):
        # 创建APoTObserver对象，设置参数b=6和k=2
        obs = APoTObserver(b=6, k=2)

        # 设置obs对象的最小值为0.0和最大值为1.0的张量
        obs.min_val = torch.tensor([0.0])
        obs.max_val = torch.tensor([1.0])

        # 调用calculate_qparams方法，获取返回的alpha、gamma、quantization_levels和level_indices
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=False)

        # 计算alpha的预期值，为obs.min_val和obs.max_val中的较大值
        alpha_test = torch.max(-obs.min_val, obs.max_val)

        # 检查计算得到的alpha是否等于预期值
        self.assertEqual(alpha, alpha_test)

        # 计算预期的gamma值
        gamma_test = 0
        for i in range(3):
            gamma_test += 2**(-i)
        gamma_test = 1 / gamma_test

        # 检查计算得到的gamma是否等于预期值
        self.assertEqual(gamma, gamma_test)

        # 检查quantization_levels的大小是否等于2的6次方
        quantlevels_size_test = int(len(quantization_levels))
        quantlevels_size = 2**6
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # 检查level_indices的大小是否等于64
        levelindices_size_test = int(len(level_indices))
        self.assertEqual(levelindices_size_test, 64)

        # 检查level_indices中的元素是否唯一
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
        Test case 4: calculate_qparams
        Same as test case 2 but with signed = True
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * signed = True
    """
    # 定义一个测试方法，用于测试带有四个参数的APoTObserver类的calculate_qparams方法，signed=True
    def test_calculate_qparams_signed(self):
        # 创建APoTObserver对象，设置参数b=4和k=2
        obs = APoTObserver(b=4, k=2)

        # 设置obs对象的最小值为0.0和最大值为1.0的张量
        obs.min_val = torch.tensor([0.0])
        obs.max_val = torch.tensor([1.0])

        # 调用calculate_qparams方法，获取返回的alpha、gamma、quantization_levels和level_indices
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=True)

        # 计算alpha的预期值，为obs.min_val和obs.max_val中的较大值
        alpha_test = torch.max(-obs.min_val, obs.max_val)

        # 检查计算得到的alpha是否等于预期值
        self.assertEqual(alpha, alpha_test)

        # 计算预期的gamma值
        gamma_test = 0
        for i in range(2):
            gamma_test += 2**(-i)
        gamma_test = 1 / gamma_test

        # 检查计算得到的gamma是否等于预期值
        self.assertEqual(gamma, gamma_test)

        # 检查quantization_levels的大小是否等于49
        quantlevels_size_test = int(len(quantization_levels))
        self.assertEqual(quantlevels_size_test, 49)

        # 检查quantization_levels中每个元素的负值是否也包含在quantization_levels中
        quantlevels_test_list = quantization_levels.tolist()
        negatives_contained = True
        for ele in quantlevels_test_list:
            if -ele not in quantlevels_test_list:
                negatives_contained = False
        self.assertTrue(negatives_contained)

        # 检查level_indices的大小是否等于49
        levelindices_size_test = int(len(level_indices))
        self.assertEqual(levelindices_size_test, 49)

        # 检查level_indices中的元素是否唯一
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))
    """
    Test case 5: calculate_qparams
        Assume hardcoded parameters:
        * b = 6 (total number of bits across all terms)
        * k = 1 (base bitwidth, i.e. bitwidth of every term)
        * n = 6 (number of additive terms)
    """
    # 定义一个测试类中的测试方法，测试 calculate_qparams 方法
    def test_calculate_qparams_k1(self):
        # 创建一个 APoTObserver 对象，设置参数 b=6, k=1
        obs = APoTObserver(b=6, k=1)

        # 设置观察器的最小值为 0.0
        obs.min_val = torch.tensor([0.0])
        # 设置观察器的最大值为 1.0
        obs.max_val = torch.tensor([1.0])

        # 调用 calculate_qparams 方法，获取返回的 alpha, gamma, quantization_levels, level_indices
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=False)

        # 计算预期的 gamma 值
        gamma_test = 0
        for i in range(6):
            gamma_test += 2**(-i)

        gamma_test = 1 / gamma_test

        # 检查计算得到的 gamma 值是否与预期值相等
        self.assertEqual(gamma, gamma_test)

        # 检查 quantization_levels 的大小是否符合预期
        quantlevels_size_test = int(len(quantization_levels))
        quantlevels_size = 2**6
        self.assertEqual(quantlevels_size_test, quantlevels_size)

        # 检查 level_indices 的大小是否符合预期
        levelindices_size_test = int(len(level_indices))
        level_indices_size = 2**6
        self.assertEqual(levelindices_size_test, level_indices_size)

        # 检查 level_indices 的唯一值是否符合预期
        level_indices_test_list = level_indices.tolist()
        self.assertEqual(len(level_indices_test_list), len(set(level_indices_test_list)))

    """
    Test forward method on hard-coded tensor with arbitrary values.
    Checks that alpha is max of abs value of max and min values in tensor.
    """
    # 定义测试类中的另一个测试方法，测试 forward 方法
    def test_forward(self):
        # 创建一个 APoTObserver 对象，设置参数 b=4, k=2
        obs = APoTObserver(b=4, k=2)

        # 创建一个包含固定值的张量 X
        X = torch.tensor([0.0, -100.23, -37.18, 3.42, 8.93, 9.21, 87.92])

        # 调用 forward 方法处理张量 X
        X = obs.forward(X)

        # 调用 calculate_qparams 方法，获取返回的 alpha, gamma, quantization_levels, level_indices
        alpha, gamma, quantization_levels, level_indices = obs.calculate_qparams(signed=True)

        # 计算张量 X 的最小值和最大值
        min_val = torch.min(X)
        max_val = torch.max(X)

        # 计算预期的 alpha 值
        expected_alpha = torch.max(-min_val, max_val)

        # 检查计算得到的 alpha 值是否与预期值相等
        self.assertEqual(alpha, expected_alpha)
# 如果当前脚本被直接执行（而非被导入作为模块），则执行以下代码
if __name__ == '__main__':
    # 运行单元测试主程序，用于执行测试用例
    unittest.main()
```