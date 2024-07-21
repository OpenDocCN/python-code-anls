# `.\pytorch\test\nn\test_pruning.py`

```
# Owner(s): ["module: nn"]
import pickle
import unittest
import unittest.mock as mock

import torch

import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TemporaryFileName,
    TEST_NUMPY,
)

class TestPruningNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    # torch/nn/utils/prune.py
    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_validate_pruning_amount_init(self):
        r"""Test the first util function that validates the pruning
        amount requested by the user the moment the pruning method
        is initialized. This test checks that the expected errors are
        raised whenever the amount is invalid.
        The original function runs basic type checking + value range checks.
        It doesn't check the validity of the pruning amount with
        respect to the size of the tensor to prune. That's left to
        `_validate_pruning_amount`, tested below.
        """
        # neither float not int should raise TypeError
        with self.assertRaises(TypeError):
            prune._validate_pruning_amount_init(amount="I'm a string")

        # float not in [0, 1] should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=1.1)
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=20.0)

        # negative int should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=-10)

        # all these should pass without errors because they're valid amounts
        prune._validate_pruning_amount_init(amount=0.34)
        prune._validate_pruning_amount_init(amount=1500)
        prune._validate_pruning_amount_init(amount=0)
        prune._validate_pruning_amount_init(amount=0.0)
        prune._validate_pruning_amount_init(amount=1)
        prune._validate_pruning_amount_init(amount=1.0)
        self.assertTrue(True)
    def test_validate_pruning_amount(self):
        r"""Tests the second util function that validates the pruning
        amount requested by the user, this time with respect to the size
        of the tensor to prune. The rationale is that if the pruning amount,
        converted to absolute value of units to prune, is larger than
        the number of units in the tensor, then we expect the util function
        to raise a value error.
        """
        # 如果 amount 是整数并且大于 tensor_size，则抛出 ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount(amount=20, tensor_size=19)

        # 如果 amount 是浮点数，则不应该引发错误
        prune._validate_pruning_amount(amount=0.3, tensor_size=0)

        # 这是正常情况
        prune._validate_pruning_amount(amount=19, tensor_size=20)
        prune._validate_pruning_amount(amount=0, tensor_size=0)
        prune._validate_pruning_amount(amount=1, tensor_size=1)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_compute_nparams_to_prune(self):
        r"""Test that requested pruning `amount` gets translated into the
        correct absolute number of units to prune.
        """
        # 测试请求的 pruning amount 转换为正确的要修剪的单元数
        self.assertEqual(prune._compute_nparams_toprune(amount=0, tensor_size=15), 0)
        self.assertEqual(prune._compute_nparams_toprune(amount=10, tensor_size=15), 10)
        # 如果 1 是整数，表示修剪一个单元
        self.assertEqual(prune._compute_nparams_toprune(amount=1, tensor_size=15), 1)
        # 如果 1. 是浮点数，表示修剪全部单元的百分之百
        self.assertEqual(prune._compute_nparams_toprune(amount=1.0, tensor_size=15), 15)
        self.assertEqual(prune._compute_nparams_toprune(amount=0.4, tensor_size=17), 7)
    def test_random_pruning_sizes(self):
        r"""Test that the new parameters and buffers created by the pruning
        method have the same size as the input tensor to prune. These, in
        fact, correspond to the pruned version of the tensor itself, its
        mask, and its original copy, so the size must match.
        """
        # 获取测试固定模块和参数
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 需要进行修剪的参数名称列表
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # 获取原始参数张量
                    original_tensor = getattr(m, name)

                    # 随机非结构化修剪方法应用于参数
                    prune.random_unstructured(m, name=name, amount=0.1)
                    
                    # 断言：修剪后的 mask 张量应与原始张量大小一致
                    self.assertEqual(
                        original_tensor.size(), getattr(m, name + "_mask").size()
                    )
                    # 断言：'orig' 张量应与原始张量大小一致
                    self.assertEqual(
                        original_tensor.size(), getattr(m, name + "_orig").size()
                    )
                    # 断言：新张量应与原始张量大小一致
                    self.assertEqual(original_tensor.size(), getattr(m, name).size())

    def test_random_pruning_orig(self):
        r"""Test that original tensor is correctly stored in 'orig'
        after pruning is applied. Important to make sure we don't
        lose info about the original unpruned parameter.
        """
        # 获取测试固定模块和参数
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 需要进行修剪的参数名称列表
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # 获取修剪前的原始参数张量
                    original_tensor = getattr(m, name)
                    
                    # 随机非结构化修剪方法应用于参数
                    prune.random_unstructured(m, name=name, amount=0.1)
                    
                    # 断言：'orig' 张量应与修剪前的原始参数张量相等
                    self.assertEqual(original_tensor, getattr(m, name + "_orig"))

    def test_random_pruning_new_weight(self):
        r"""Test that module.name now contains a pruned version of
        the original tensor obtained from multiplying it by the mask.
        """
        # 获取测试固定模块和参数
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 需要进行修剪的参数名称列表
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # 获取修剪前的原始参数张量
                    original_tensor = getattr(m, name)
                    
                    # 随机非结构化修剪方法应用于参数
                    prune.random_unstructured(m, name=name, amount=0.1)
                    
                    # 断言：修剪后的张量应为原始张量乘以 mask 张量
                    self.assertEqual(
                        getattr(m, name),
                        getattr(m, name + "_orig")
                        * getattr(m, name + "_mask").to(dtype=original_tensor.dtype),
                    )
    def test_identity_pruning(self):
        r"""Test that a mask of 1s does not change forward or backward."""
        # 创建一个输入张量，全为1，形状为(1, 5)
        input_ = torch.ones(1, 5)
        # 创建一个线性层模块，输入维度为5，输出维度为2
        m = nn.Linear(5, 2)
        # 记录修剪前的输出
        y_prepruning = m(input_)  # output prior to pruning

        # 计算修剪前的梯度并验证其与全1张量相等
        y_prepruning.sum().backward()
        # 克隆权重的梯度，避免使用指针引用
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        # 克隆偏置的梯度
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))

        # 清除所有梯度
        m.zero_grad()

        # 强制权重的修剪掩码全部为1
        prune.identity(m, name="weight")

        # 使用全1的修剪掩码，输出应与未修剪时相同
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)

        # 使用全1的修剪掩码，梯度应与未修剪时相同
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)

        # 连续调用两次前向传播应该得到相同的输出
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    def test_random_pruning_0perc(self):
        r"""Test that a mask of 1s does not change forward or backward."""
        # 创建一个输入张量，全为1，形状为(1, 5)
        input_ = torch.ones(1, 5)
        # 创建一个线性层模块，输入维度为5，输出维度为2
        m = nn.Linear(5, 2)
        # 记录修剪前的输出
        y_prepruning = m(input_)  # output prior to pruning

        # 计算修剪前的梯度并验证其与全1张量相等
        y_prepruning.sum().backward()
        # 克隆权重的梯度，避免使用指针引用
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        # 克隆偏置的梯度
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))

        # 清除所有梯度
        m.zero_grad()

        # 使用模拟的随机修剪方法，强制权重的修剪掩码全部为1
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = torch.ones_like(m.weight)
            # 执行随机结构化修剪，amount参数不起作用
            prune.random_unstructured(
                m, name="weight", amount=0.9
            )  # amount won't count

        # 使用全1的修剪掩码，输出应与未修剪时相同
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)

        # 使用全1的修剪掩码，梯度应与未修剪时相同
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)

        # 连续调用两次前向传播应该得到相同的输出
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)
    def test_random_pruning(self):
        # 创建输入张量，形状为 (1, 5)，所有元素初始化为1
        input_ = torch.ones(1, 5)
        # 创建线性层，输入维度为5，输出维度为2
        m = nn.Linear(5, 2)

        # 定义自定义掩码以用于模拟
        mask = torch.ones_like(m.weight)
        # 将掩码中指定位置的元素设为0
        mask[1, 0] = 0
        mask[0, 3] = 0

        # 检查掩码后的权重的梯度是否为零
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            # 使用随机非结构化剪枝方法对权重进行剪枝
            prune.random_unstructured(m, name="weight", amount=0.9)

        # 计算经过剪枝后的输出
        y_postpruning = m(input_)
        # 计算梯度
        y_postpruning.sum().backward()
        # 断言权重参数的梯度与掩码相同，除了被掩盖的单元
        self.assertEqual(m.weight_orig.grad, mask)  # 所有元素为1，除了被掩盖的单元
        self.assertEqual(m.bias.grad, torch.ones_like(m.bias))

        # 确保 weight_orig 的更新不会修改 [1, 0] 和 [0, 3] 的值
        old_weight_orig = m.weight_orig.clone()
        # 更新权重
        learning_rate = 1.0
        for p in m.parameters():
            p.data.sub_(p.grad.data * learning_rate)
        # 由于这些被剪枝了，它们不应该被更新
        self.assertEqual(old_weight_orig[1, 0], m.weight_orig[1, 0])
        self.assertEqual(old_weight_orig[0, 3], m.weight_orig[0, 3])

    def test_random_pruning_forward(self):
        r"""check forward with mask (by hand)."""
        # 创建输入张量，形状为 (1, 5)，所有元素初始化为1
        input_ = torch.ones(1, 5)
        # 创建线性层，输入维度为5，输出维度为2
        m = nn.Linear(5, 2)

        # 定义自定义掩码以用于模拟
        mask = torch.zeros_like(m.weight)
        # 将掩码中指定位置的元素设为1
        mask[1, 0] = 1
        mask[0, 3] = 1

        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            # 使用随机非结构化剪枝方法对权重进行剪枝
            prune.random_unstructured(m, name="weight", amount=0.9)

        # 计算带有剪枝掩码的前向传播结果
        yhat = m(input_)
        # 断言计算的输出与剪枝前的权重和偏置计算得到的结果一致
        self.assertEqual(yhat[0, 0], m.weight_orig[0, 3] + m.bias[0])
        self.assertEqual(yhat[0, 1], m.weight_orig[1, 0] + m.bias[1])

    def test_remove_pruning_forward(self):
        r"""Remove pruning and check forward is unchanged from previous
        pruned state.
        """
        # 创建输入张量，形状为 (1, 5)，所有元素初始化为1
        input_ = torch.ones(1, 5)
        # 创建线性层，输入维度为5，输出维度为2
        m = nn.Linear(5, 2)

        # 定义自定义掩码以用于模拟
        mask = torch.ones_like(m.weight)
        # 将掩码中指定位置的元素设为0
        mask[1, 0] = 0
        mask[0, 3] = 0

        # 检查掩码后的权重的梯度是否为零
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            # 使用随机非结构化剪枝方法对权重进行剪枝
            prune.random_unstructured(m, name="weight", amount=0.9)

        # 计算剪枝后的输出
        y_postpruning = m(input_)

        # 移除剪枝效果
        prune.remove(m, "weight")

        # 计算移除剪枝效果后的输出
        y_postremoval = m(input_)
        # 断言剪枝前后的输出结果一致
        self.assertEqual(y_postpruning, y_postremoval)
    def test_pruning_id_consistency(self):
        r"""Test that pruning doesn't change the id of the parameters, which
        would otherwise introduce issues with pre-existing optimizers that
        point to old parameters.
        """
        # 创建一个线性层模型，输入大小为5，输出大小为2，无偏置
        m = nn.Linear(5, 2, bias=False)

        # 获取模型第一个参数的内存地址作为标识
        tensor_id = id(next(iter(m.parameters())))

        # 对模型进行随机非结构化剪枝，剪枝名称为"weight"，剪枝比例为0.9
        prune.random_unstructured(m, name="weight", amount=0.9)
        # 断言剪枝前后参数的内存地址标识是否相同
        self.assertEqual(tensor_id, id(next(iter(m.parameters()))))

        # 移除模型的"weight"剪枝
        prune.remove(m, "weight")
        # 再次断言移除剪枝后参数的内存地址标识是否与最初相同
        self.assertEqual(tensor_id, id(next(iter(m.parameters()))))

    def test_random_pruning_pickle(self):
        # 创建线性层和三维卷积层的模型列表
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 定义需要剪枝的模型参数名称列表
        names = ["weight", "bias"]

        # 遍历模型列表
        for m in modules:
            # 遍历参数名称列表
            for name in names:
                with self.subTest(m=m, name=name):
                    # 对模型进行随机非结构化剪枝，剪枝名称为name，剪枝比例为0.1
                    prune.random_unstructured(m, name=name, amount=0.1)
                    # 通过pickle序列化和反序列化来创建一个新的模型副本
                    m_new = pickle.loads(pickle.dumps(m))
                    # 断言新创建的模型副本类型与原始模型类型相同
                    self.assertIsInstance(m_new, type(m))

    def test_multiple_pruning_calls(self):
        # 创建一个三维卷积层模型
        m = nn.Conv3d(2, 2, 2)
        # 对模型的"weight"参数进行L1非结构化剪枝，剪枝比例为0.1
        prune.l1_unstructured(m, name="weight", amount=0.1)
        # 保存剪枝后的权重掩码，用于后续的合理性检查
        weight_mask0 = m.weight_mask

        # 再次对模型进行ln结构化剪枝，剪枝名称为"weight"，剪枝比例为0.3，n=2，dim=0
        prune.ln_structured(m, name="weight", amount=0.3, n=2, dim=0)
        # 获取剪枝操作后的前向预处理钩子
        hook = next(iter(m._forward_pre_hooks.values()))
        # 断言钩子类型为PruningContainer
        self.assertIsInstance(hook, torch.nn.utils.prune.PruningContainer)
        # 检查容器中的_tensor_name属性是否正确设置为"weight"
        self.assertEqual(hook._tensor_name, "weight")

        # 检查剪枝容器的长度是否等于剪枝操作次数
        self.assertEqual(len(hook), 2)  # m.weight has been pruned twice

        # 检查剪枝容器中的条目类型和顺序是否符合预期
        self.assertIsInstance(hook[0], torch.nn.utils.prune.L1Unstructured)
        self.assertIsInstance(hook[1], torch.nn.utils.prune.LnStructured)

        # 检查所有在第一个掩码中为0的条目，在第二个掩码中是否也为0
        self.assertTrue(torch.all(m.weight_mask[weight_mask0 == 0] == 0))

        # 再次对模型进行ln结构化剪枝，剪枝名称为"weight"，剪枝比例为0.1，n为无穷大，dim=1
        prune.ln_structured(m, name="weight", amount=0.1, n=float("inf"), dim=1)
        # 再次获取剪枝操作后的前向预处理钩子
        hook = next(iter(m._forward_pre_hooks.values()))
        # 检查容器中的_tensor_name属性是否正确设置为"weight"
        self.assertEqual(hook._tensor_name, "weight")
    # 定义一个测试方法，用于测试 PruningContainer 类
    def test_pruning_container(self):
        # 创建一个空的 PruningContainer 实例
        container = prune.PruningContainer()
        # 设置实例的 _tensor_name 属性为 "test"
        container._tensor_name = "test"
        # 断言容器的长度为 0
        self.assertEqual(len(container), 0)

        # 创建一个 L1Unstructured 类型的剪枝方法实例，并设置其 _tensor_name 属性为 "test"
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = "test"

        # 测试将剪枝方法添加到容器中
        container.add_pruning_method(p)

        # 测试当剪枝方法的 _tensor_name 与容器的 _tensor_name 不匹配时是否会引发 ValueError 异常
        q = prune.L1Unstructured(amount=2)
        q._tensor_name = "another_test"
        with self.assertRaises(ValueError):
            container.add_pruning_method(q)

        # 测试尝试将非剪枝方法对象添加到剪枝容器中是否会引发 TypeError 异常
        with self.assertRaises(TypeError):
            container.add_pruning_method(10)
        with self.assertRaises(TypeError):
            container.add_pruning_method("ugh")
    def test_pruning_container_compute_mask(self):
        r"""Test `compute_mask` of pruning container with a known `t` and
        `default_mask`. Indirectly checks that Ln structured pruning is
        acting on the right axis.
        """
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = "test"

        # 1) test unstructured pruning
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = "test"
        # add the pruning method to the container
        container.add_pruning_method(p)

        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning
        q = prune.LnStructured(amount=1, n=2, dim=0)
        q._tensor_name = "test"
        container.add_pruning_method(q)
        # since we are pruning the lowest magnitude one of the two rows, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 3) test structured pruning, along another axis
        r = prune.LnStructured(amount=1, n=2, dim=1)
        r._tensor_name = "test"
        container.add_pruning_method(r)
        # since we are pruning the lowest magnitude of the four columns, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)


注释：

# 定义测试函数，用于测试修剪容器的 `compute_mask` 方法
def test_pruning_container_compute_mask(self):
    r"""Test `compute_mask` of pruning container with a known `t` and
    `default_mask`. Indirectly checks that Ln structured pruning is
    acting on the right axis.
    """

    # 创建一个空的修剪容器对象
    container = prune.PruningContainer()
    container._tensor_name = "test"

    # 1) 测试非结构化修剪
    # 创建一个新的修剪方法
    p = prune.L1Unstructured(amount=2)
    p._tensor_name = "test"
    # 将修剪方法添加到容器中
    container.add_pruning_method(p)

    # 创建要修剪的张量
    t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
    # 手动创建先前的掩码
    default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
    # 因为我们要修剪两个最低幅度的单元，计算的结果应该是这样的：
    expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
    computed_mask = container.compute_mask(t, default_mask)
    self.assertEqual(expected_mask, computed_mask)

    # 2) 测试结构化修剪
    q = prune.LnStructured(amount=1, n=2, dim=0)
    q._tensor_name = "test"
    container.add_pruning_method(q)
    # 因为我们要修剪两行中最低幅度的一个，计算的结果应该是这样的：
    expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]], dtype=torch.float32)
    computed_mask = container.compute_mask(t, default_mask)
    self.assertEqual(expected_mask, computed_mask)

    # 3) 测试沿另一个轴的结构化修剪
    r = prune.LnStructured(amount=1, n=2, dim=1)
    r._tensor_name = "test"
    container.add_pruning_method(r)
    # 因为我们要修剪四列中最低幅度的一个，计算的结果应该是这样的：
    expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    computed_mask = container.compute_mask(t, default_mask)
    self.assertEqual(expected_mask, computed_mask)
    # 定义测试函数，用于测试基于 L1 范数的非结构化剪枝的效果
    def test_l1_unstructured_pruning(self):
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries by l1 norm (by hand). It also checks that applying l1
        unstructured pruning more than once respects the previous mask.
        """
        # 创建一个具有输入和输出尺寸的线性层对象
        m = nn.Linear(4, 2)
        # 手动修改其权重矩阵
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32)
        )

        # 对权重矩阵应用 L1 非结构化剪枝，删除最小的两个条目
        prune.l1_unstructured(m, "weight", amount=2)
        # 预期的权重矩阵
        expected_weight = torch.tensor(
            [[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

        # 再次检查剪枝，确保下两个最小条目被移除
        prune.l1_unstructured(m, "weight", amount=2)
        # 更新预期的权重矩阵
        expected_weight = torch.tensor(
            [[0, 0, 3, 4], [-4, -3, 0, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

    # 测试基于重要性分数的 L1 非结构化剪枝效果
    def test_l1_unstructured_pruning_with_importance_scores(self):
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries of importance scores and not the parameter by l1 norm (by hand).
        It also checks that applying l1 unstructured pruning more than once
        respects the previous mask.
        """
        # 创建一个具有输入和输出尺寸的线性层对象
        m = nn.Linear(4, 2)
        # 手动修改其权重矩阵
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32)
        )
        # 指定重要性分数的张量
        importance_scores = torch.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32
        )

        # 对权重矩阵应用基于 L1 范数的非结构化剪枝，指定剪枝数量和重要性分数
        prune.l1_unstructured(
            m, "weight", amount=2, importance_scores=importance_scores
        )
        # 预期的权重矩阵
        expected_weight = torch.tensor(
            [[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

        # 再次检查剪枝，确保下两个最小的重要性分数对应的条目被移除
        prune.l1_unstructured(
            m, "weight", amount=2, importance_scores=importance_scores
        )
        # 更新预期的权重矩阵
        expected_weight = torch.tensor(
            [[1, 0, 0, 4], [-4, 0, 0, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)
    def test_unstructured_pruning_same_magnitude(self):
        r"""Since it may happen that the tensor to prune has entries with the
        same exact magnitude, it is important to check that pruning happens
        consistently based on the bottom % of weights, and not by threshold,
        which would instead kill off *all* units with magnitude = threshold.
        """
        # 设定要剪枝的比例
        AMOUNT = 0.2
        # 创建 L1Unstructured 剪枝器对象
        p = prune.L1Unstructured(amount=AMOUNT)
        # 创建一个随机张量，其中元素取值为{-2, 0, 2}
        t = 2 * torch.randint(low=-1, high=2, size=(10, 7))
        # 计算需要剪枝的参数数量
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.nelement())

        # 计算剪枝后的掩码
        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        # 统计被剪枝的参数数量
        nparams_pruned = torch.sum(computed_mask == 0)
        # 断言剪枝后的参数数量与预期相等
        self.assertEqual(nparams_toprune, nparams_pruned)

    def test_random_structured_pruning_amount(self):
        # 设定要剪枝的比例和剪枝的维度轴
        AMOUNT = 0.6
        AXIS = 2
        # 创建 RandomStructured 剪枝器对象
        p = prune.RandomStructured(amount=AMOUNT, dim=AXIS)
        # 创建一个随机张量，元素取值为{-2, 0, 2}，并转换为 float32 类型
        t = 2 * torch.randint(low=-1, high=2, size=(5, 4, 2)).to(dtype=torch.float32)
        # 计算需要剪枝的参数数量
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.shape[AXIS])

        # 计算剪枝后的掩码
        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        # 检查每一列是否只有一列完全被剪枝，其它列保持不变
        remaining_axes = [_ for _ in range(len(t.shape)) if _ != AXIS]
        per_column_sums = sorted(torch.sum(computed_mask == 0, axis=remaining_axes))
        assert per_column_sums == [0, 20]

    def test_ln_structured_pruning(self):
        r"""Check Ln structured pruning by hand."""
        # 创建一个卷积层对象，输入通道数为 3，输出通道数为 1，卷积核大小为 2x2
        m = nn.Conv2d(3, 1, 2)
        # 设定卷积核的权重张量，包含三个输入通道的三个 2x2 的卷积核
        m.weight.data = torch.tensor(
            [
                [
                    [[1.0, 2.0], [1.0, 2.5]],
                    [[0.5, 1.0], [0.1, 0.1]],
                    [[-3.0, -5.0], [0.1, -1.0]],
                ]
            ]
        )
        # 预期剪枝结果，通过 L2-norm 剪枝一个通道
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 1] = 0.0

        # 执行 L2-norm 结构化剪枝
        prune.ln_structured(m, "weight", amount=1, n=2, dim=1)
        # 断言剪枝后的掩码与预期相等
        self.assertEqual(expected_mask_axis1, m.weight_mask)

        # 预期剪枝结果，通过 L1-norm 剪枝一个通道的一个列
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 0] = 0.0

        # 执行 L1-norm 结构化剪枝
        prune.ln_structured(m, "weight", amount=1, n=1, dim=-1)
        # 断言剪枝后的掩码与预期相等
        self.assertEqual(expected_mask_axis3, m.weight_mask)
    def test_ln_structured_pruning_importance_scores(self):
        r"""Check Ln structured pruning by hand."""
        # 创建一个2通道的2维卷积层，输入通道为3
        m = nn.Conv2d(3, 1, 2)
        # 设置卷积核的权重数据
        m.weight.data = torch.tensor(
            [
                [
                    [[1.0, 2.0], [1.0, 2.5]],
                    [[0.5, 1.0], [0.1, 0.1]],
                    [[-3.0, -5.0], [0.1, -1.0]],
                ]
            ]
        )
        # 设置重要性分数的张量
        importance_scores = torch.tensor(
            [
                [
                    [[10.0, 1.0], [10.0, 1.0]],
                    [[30.0, 3.0], [30.0, 3.0]],
                    [[-20.0, -2.0], [-20.0, -2.0]],
                ]
            ]
        )
        # 预期的剪枝效果：通过L2范数剪枝掉1个通道，共3个通道
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 0] = 0.0

        # 执行Ln结构化剪枝，剪枝1个通道，维度为2，沿第1维度，使用重要性分数
        prune.ln_structured(
            m, "weight", amount=1, n=2, dim=1, importance_scores=importance_scores
        )
        # 断言：检查剪枝后的掩码是否符合预期
        self.assertEqual(expected_mask_axis1, m.weight_mask)

        # 预期的剪枝效果：通过L1范数剪枝掉1个列，共2列，沿着-1维度
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 1] = 0.0

        # 执行Ln结构化剪枝，剪枝1个列，维度为1，沿-1维度，使用重要性分数
        prune.ln_structured(
            m, "weight", amount=1, n=1, dim=-1, importance_scores=importance_scores
        )
        # 断言：检查剪枝后的掩码是否符合预期
        self.assertEqual(expected_mask_axis3, m.weight_mask)

    def test_remove_pruning(self):
        r"""`prune.remove` removes the hook and the reparametrization
        and makes the pruning final in the original parameter.
        """
        # 创建包含线性层和3维卷积层的模块列表
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 指定需要处理的参数名列表
        names = ["weight", "bias"]

        # 遍历每个模块
        for m in modules:
            # 遍历每个参数名
            for name in names:
                with self.subTest(m=m, name=name):
                    # 随机非结构化剪枝，剪枝比例为0.5
                    prune.random_unstructured(m, name, amount=0.5)
                    # 断言：检查剪枝后参数名是否存在并已备份，检查是否存在掩码
                    self.assertIn(name + "_orig", dict(m.named_parameters()))
                    self.assertIn(name + "_mask", dict(m.named_buffers()))
                    self.assertNotIn(name, dict(m.named_parameters()))
                    self.assertTrue(hasattr(m, name))
                    # 获取剪枝后的张量
                    pruned_t = getattr(m, name)

                    # 移除剪枝
                    prune.remove(m, name)
                    # 断言：检查剪枝是否移除，恢复原始参数，并检查张量是否恢复到原始状态
                    self.assertIn(name, dict(m.named_parameters()))
                    self.assertNotIn(name + "_orig", dict(m.named_parameters()))
                    self.assertNotIn(name + "_mask", dict(m.named_buffers()))
                    final_t = getattr(m, name)

                    # 断言：检查剪枝前后的张量是否相等
                    self.assertEqual(pruned_t, final_t)
    def test_remove_pruning_exception(self):
        r"""Removing from an unpruned tensor throws an assertion error"""
        # 创建两个包含神经网络模块的列表
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 定义需要操作的参数名称列表
        names = ["weight", "bias"]

        # 遍历每个模块
        for m in modules:
            # 遍历每个参数名称
            for name in names:
                # 使用子测试检查模块是否未经修剪
                with self.subTest(m=m, name=name):
                    # 断言模块未经修剪
                    self.assertFalse(prune.is_pruned(m))
                    # 由于模块未经修剪，尝试从中删除修剪操作应该引发 ValueError 异常
                    with self.assertRaises(ValueError):
                        prune.remove(m, name)

    def test_global_pruning(self):
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        # 创建两个线性层模块
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # 手动修改权重矩阵
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=torch.float32)
        )
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(dtype=torch.float32)
        )

        # 定义需要进行修剪的参数元组列表
        params_to_prune = (
            (m, "weight"),
            (n, "weight"),
        )

        # 使用全局 L1 非结构化修剪方法修剪每个参数中的最小权重，数量为 4
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=4
        )

        # 预期的 m 权重
        expected_mweight = torch.tensor(
            [[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_mweight, m.weight)

        # 预期的 n 权重
        expected_nweight = torch.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_nweight, n.weight)
    def test_global_pruning_importance_scores(self):
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        # 创建一个包含4个输入和2个输出的线性层模型m
        m = nn.Linear(4, 2)
        # 创建一个包含3个输入和1个输出的线性层模型n
        n = nn.Linear(3, 1)
        
        # 手动修改m的权重矩阵
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=torch.float32)
        )
        
        # 设置m的重要性分数
        m_importance_scores = torch.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32
        )
        
        # 手动修改n的权重矩阵
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(dtype=torch.float32)
        )
        
        # 设置n的重要性分数
        n_importance_scores = torch.tensor([[0, 10.0, -0.2]]).to(dtype=torch.float32)

        # 需要进行修剪的参数对
        params_to_prune = (
            (m, "weight"),
            (n, "weight"),
        )
        
        # 指定每个参数的重要性分数
        importance_scores = {
            (m, "weight"): m_importance_scores,
            (n, "weight"): n_importance_scores,
        }

        # 使用L1幅度进行全局非结构化修剪，修剪掉全局权重最小的4个值
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=4,
            importance_scores=importance_scores,
        )

        # 预期的m的权重
        expected_m_weight = torch.tensor(
            [[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype
        )
        # 断言m的权重是否符合预期
        self.assertEqual(expected_m_weight, m.weight)

        # 预期的n的权重
        expected_n_weight = torch.tensor([[0, 0.1, 0]]).to(dtype=n.weight.dtype)
        # 断言n的权重是否符合预期
        self.assertEqual(expected_n_weight, n.weight)

    def test_custom_from_mask_pruning(self):
        r"""Test that the CustomFromMask is capable of receiving
        as input at instantiation time a custom mask, and combining it with
        the previous default mask to generate the correct final mask.
        """
        # 新的掩码
        mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
        # 旧的默认掩码
        default_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

        # 一些张量（实际上并未使用）
        t = torch.rand_like(mask.to(dtype=torch.float32))

        # 创建一个自定义掩码的CustomFromMask对象p
        p = prune.CustomFromMask(mask=mask)

        # 计算得到的掩码，将自定义掩码与默认掩码组合以生成正确的最终掩码
        computed_mask = p.compute_mask(t, default_mask)
        
        # 预期的掩码
        expected_mask = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 1]], dtype=computed_mask.dtype
        )

        # 断言计算得到的掩码与预期的掩码是否相等
        self.assertEqual(computed_mask, expected_mask)
    def test_pruning_rollback(self):
        r"""Test that if something fails when the we try to compute the mask,
        then the model isn't left in some intermediate half-pruned state.
        The try/except statement in `apply` should handle rolling back
        to the previous state before pruning began.
        """
        # 定义模型的几个模块
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        # 要操作的参数名称列表
        names = ["weight", "bias"]

        # 遍历模块和参数名称
        for m in modules:
            for name in names:
                # 使用子测试，针对每个模块和参数名称进行测试
                with self.subTest(m=m, name=name):
                    # 使用 mock.patch 临时替换 compute_mask 方法，模拟异常情况
                    with mock.patch(
                        "torch.nn.utils.prune.L1Unstructured.compute_mask"
                    ) as compute_mask:
                        compute_mask.side_effect = Exception("HA!")
                        # 确保异常被抛出
                        with self.assertRaises(Exception):
                            prune.l1_unstructured(m, name=name, amount=0.9)

                        # 检查是否恢复到异常发生前的状态
                        self.assertTrue(name in dict(m.named_parameters()))
                        self.assertFalse(name + "_mask" in dict(m.named_buffers()))
                        self.assertFalse(name + "_orig" in dict(m.named_parameters()))

    def test_pruning_serialization_model(self):
        # 创建一个序列模型
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        # 在修剪之前检查模型状态是否正常
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # 对模型的一个参数进行修剪
        prune.l1_unstructured(module=model[0], name="weight", amount=0.9)

        # 检查原始权重和新的掩码是否存在
        self.assertIn("0.weight_orig", model.state_dict())
        self.assertIn("0.weight_mask", model.state_dict())
        self.assertNotIn("0.weight", model.state_dict())
        self.assertTrue(hasattr(model[0], "weight"))

        pruned_weight = model[0].weight

        # 使用临时文件名保存和加载模型
        with TemporaryFileName() as fname:
            torch.save(model, fname)
            new_model = torch.load(fname)

        # 检查原始权重和新的掩码是否存在
        self.assertIn("0.weight_orig", new_model.state_dict())
        self.assertIn("0.weight_mask", new_model.state_dict())
        self.assertNotIn("0.weight", new_model.state_dict())
        self.assertTrue(hasattr(new_model[0], "weight"))

        # 检查修剪后的权重是否与加载后的模型权重一致
        self.assertEqual(pruned_weight, new_model[0].weight)
    def test_pruning_serialization_state_dict(self):
        # 创建一个模型
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),  # 添加一个线性层，输入维度为10，输出维度为10
            torch.nn.ReLU(),  # ReLU激活函数层
            torch.nn.Linear(10, 1),  # 添加一个线性层，输入维度为10，输出维度为1
        )
        # 在修剪之前检查模型状态字典确保一切正常
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # 对其中一个参数进行修剪
        prune.l1_unstructured(module=model[0], name="weight", amount=0.9)

        # 检查修剪后的模型状态字典，确保原始权重和新的掩码存在
        self.assertIn("0.weight_orig", model.state_dict())
        self.assertIn("0.weight_mask", model.state_dict())
        self.assertNotIn("0.weight", model.state_dict())  # 原始权重不再存在
        self.assertTrue(hasattr(model[0], "weight"))

        pruned_weight = model[0].weight

        # 使修剪永久化，并按照基础架构恢复参数名
        prune.remove(module=model[0], name="weight")

        # 检查模型状态字典，确保原始权重和新的掩码不再存在
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # 将模型的状态字典保存并加载到new_model中
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)  # 保存模型状态字典到临时文件fname
            new_model.load_state_dict(torch.load(fname))  # 加载临时文件fname中的状态字典到new_model

        # 检查new_model中的状态字典，确保原始权重和新的掩码在其中也不存在
        self.assertNotIn("0.weight_orig", new_model.state_dict())
        self.assertNotIn("0.weight_mask", new_model.state_dict())
        self.assertIn("0.weight", new_model.state_dict())

        self.assertEqual(pruned_weight, new_model[0].weight)

    def test_prune(self):
        # 创建一个新的修剪方法
        p = prune.L1Unstructured(amount=2)
        # 创建待修剪的张量
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # 手动创建先前的掩码
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # 因为我们要修剪两个最小幅度的单元，所以计算的结果应该是这样的：
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask)  # 执行修剪操作
        self.assertEqual(t * expected_mask, pruned_tensor)
    def test_prune_importance_scores(self):
        # 创建一个新的剪枝方法实例
        p = prune.L1Unstructured(amount=2)
        # 创建待剪枝的张量
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # 创建重要性分数张量
        importance_scores = torch.tensor([[1, 2, 3, 4], [1.5, 1.6, 1.7, 1.8]]).to(
            dtype=torch.float32
        )
        # 手动创建先前的掩码
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # 因为我们要剪枝最小的两个单位，所以预期的结果应该是：
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]])
        # 执行剪枝操作，基于重要性分数
        pruned_tensor = p.prune(t, default_mask, importance_scores=importance_scores)
        # 断言剪枝后的张量是否符合预期掩码的乘积结果
        self.assertEqual(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores_mimic_default(self):
        # 创建一个新的剪枝方法实例
        p = prune.L1Unstructured(amount=2)
        # 创建待剪枝的张量
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # 手动创建先前的掩码
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # 因为我们要剪枝最小的两个单位，所以预期的结果应该是：
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        # 执行剪枝操作，不使用重要性分数
        pruned_tensor_without_importance_scores = p.prune(t, default_mask)
        # 执行剪枝操作，使用张量自身作为重要性分数
        pruned_tensor_with_importance_scores = p.prune(
            t, default_mask, importance_scores=t
        )
        # 断言两种剪枝方式得到的结果是否一致
        self.assertEqual(
            pruned_tensor_without_importance_scores,
            pruned_tensor_with_importance_scores,
        )
        # 断言剪枝后的张量是否符合预期掩码的乘积结果
        self.assertEqual(t * expected_mask, pruned_tensor_without_importance_scores)
    # 定义一个测试方法，用于测试循环神经网络 (RNN) 的剪枝功能
    def test_rnn_pruning(self):
        # 创建一个具有输入和隐藏状态大小为32的 LSTM 模型
        l = torch.nn.LSTM(32, 32)
        
        # 对其中一个参数进行 L1 非结构化剪枝，剪枝率为50%
        # 在此操作后，其中一个权重将变为一个普通张量
        prune.l1_unstructured(l, "weight_ih_l0", 0.5)
        
        # 断言：经剪枝后，模型的参数中有3个是 torch.nn.Parameter 类型的对象
        assert sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights) == 3
        
        # 移除剪枝的重新参数化，将权重恢复为 Parameter 类型
        prune.remove(l, "weight_ih_l0")
        
        # 断言：经移除重新参数化后，模型的参数中有4个是 torch.nn.Parameter 类型的对象
        assert sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights) == 4
        
        # 确保在移除重新参数化后，._parameters 和 .named_parameters 包含正确的参数
        # 具体来说，原始的权重 ('weight_ih_l0') 应该放回到参数中，
        # 而重新参数化的组件 ('weight_ih_l0_orig') 应该被移除
        assert "weight_ih_l0" in l._parameters
        assert l._parameters["weight_ih_l0"] is not None
        assert "weight_ih_l0_orig" not in l._parameters
        assert "weight_ih_l0" in dict(l.named_parameters())
        assert dict(l.named_parameters())["weight_ih_l0"] is not None
        assert "weight_ih_l0_orig" not in dict(l.named_parameters())
# 实例化参数化测试，使用 TestPruningNN 类
instantiate_parametrized_tests(TestPruningNN)

# 如果当前脚本作为主程序运行，执行测试函数
if __name__ == "__main__":
    run_tests()
```