# `.\pytorch\test\distributed\optim\test_named_optimizer.py`

```
# 导入unittest模块，用于编写和运行测试
import unittest

# 导入PyTorch库
import torch
import torch.nn as nn

# 导入分布式优化器的命名优化器模块
from torch.distributed.optim import _NamedOptimizer

# 定义一个函数用于模型训练
def _run_model_training(model_optim_lists):
    # 迭代两次，模拟训练过程
    for _ in range(2):
        # 生成一个5x8的随机张量作为输入数据x
        x = torch.rand(5, 8)
        # 遍历传入的模型-优化器列表
        for model_optim_list in model_optim_lists:
            # 获取模型和优化器列表
            model = model_optim_list[0]
            optim_list = model_optim_list[1]
            # 使用模型处理输入数据x，得到输出y
            y = model(x)
            # 对输出y进行反向传播
            y.sum().backward()
            # 遍历优化器列表，对每个优化器执行一步优化
            for optim in optim_list:
                optim.step()

# 定义一个测试用的虚拟模型类
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # 定义网络层net1，net2，net3和net4
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    # 前向传播函数，定义了模型的数据流向
    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

# 定义一个测试类，用于测试命名优化器
class NamedOptimizerTest(unittest.TestCase):
    # 比较状态字典的辅助函数，用于验证两个字典是否相等
    def _compare_state_dict_group(self, group, named_group, assert_equal=True):
        for key, val in group.items():
            if key != "params":
                # 断言key存在于命名优化器的状态字典中
                self.assertTrue(
                    key in named_group, f"{key} not in named optimizer state dict"
                )
                err_msg = (
                    f"{key} state not equal" if assert_equal else f"{key} state equal"
                )
                if isinstance(val, torch.Tensor):
                    # 如果值是张量，则使用torch.allclose验证是否相似
                    fn = self.assertTrue if assert_equal else self.assertFalse
                    fn(torch.allclose(val, named_group[key]), err_msg)
                else:
                    # 否则直接比较值
                    fn = self.assertEqual if assert_equal else self.assertNotEqual
                    fn(val, named_group[key], err_msg)

    # 比较参数组的辅助函数，用于验证两个参数组列表是否相等
    def _compare_param_groups(self, param_groups_1, param_groups_2):
        self.assertTrue(isinstance(param_groups_1, list))
        self.assertTrue(isinstance(param_groups_2, list))
        # 逐对比较参数组
        for groups in zip(param_groups_1, param_groups_2):
            self._compare_param_group(groups[0], groups[1])

    # 比较参数组的辅助函数，用于验证两个参数组是否相等
    def _compare_param_group(self, group_1, group_2):
        self.assertTrue(isinstance(group_1, dict))
        self.assertTrue(isinstance(group_2, dict))
        # 逐对比较参数组的键值对
        for key, val in group_1.items():
            self.assertTrue(key in group_2)
            if key != "params":
                # 如果不是params键，则直接比较值
                self.assertEqual(val, group_2[key])
            else:
                # 如果是params键，则逐对比较张量
                for tensors in zip(val, group_2[key]):
                    self.assertTrue(torch.allclose(tensors[0], tensors[1]))
    # 定义一个测试方法，用于验证 NamedOptimizer 是否暴露了预期的状态字典接口
    def test_state_dict(self):
        """Check that NamedOptimizer exposes the expected state dict
        interface."""
        
        # 创建一个测试用的 Dummy 模型实例
        m = TestDummyModel()
        # 创建另一个相同的 Dummy 模型实例，用于比较
        m_dup = TestDummyModel()
        
        # 使用 SGD 优化器来优化模型 m 的参数，设置学习率和动量
        optim = torch.optim.SGD(
            m.parameters(),
            lr=1e-2,
            momentum=0.9,
        )

        # 使用 _NamedOptimizer 包装模型 m_dup 的命名参数，使用相同的 SGD 优化器和参数
        named_optim = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )
        
        # 比较普通优化器 optim 和命名优化器 named_optim 的参数组
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        # 运行模型训练过程，分别使用 optim 和 named_optim 优化器
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        
        # 再次比较优化器的参数组，确保训练后状态一致
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        # 获取普通优化器 optim 的状态字典
        sd = optim.state_dict()
        # 获取命名优化器 named_optim 的状态字典
        named_sd = named_optim.state_dict()

        # 比较 optim 状态字典中 "state" 部分的内容，确保一致性
        self._compare_state_dict_group(
            sd["state"][0],
            named_sd["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][3],
            named_sd["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][4],
            named_sd["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][7],
            named_sd["state"]["net4.1.bias"],
            assert_equal=True,
        )
    def test_state_dict_multi_param_group(self):
        """
        Check that NamedOptimizer exposes the expected state dict
        interface when multiple param groups are specified.
        """
        # 创建测试用的模型实例
        m = TestDummyModel()
        m_dup = TestDummyModel()

        # 创建两个不同的优化器，每个优化器包含多个参数组
        optim_1 = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},  # 第一个参数组使用 m.net1 的参数
                {"params": m.net3.parameters(), "lr": 1e-3},  # 第二个参数组使用 m.net3 的参数，并设置不同的学习率
            ],
            lr=1e-2,  # 全局学习率设定为 0.01
            momentum=0.9,  # SGD 优化器的动量设定为 0.9
        )

        optim_2 = torch.optim.Adam(
            [
                {"params": m.net2.parameters()},  # 第一个参数组使用 m.net2 的参数
                {"params": m.net4.parameters(), "lr": 1e-5},  # 第二个参数组使用 m.net4 的参数，并设置不同的学习率
            ]
        )

        # 创建 NamedOptimizer 实例，与上面的优化器相似，但用于比较的模型参数是 m_dup 的
        named_optim_1 = _NamedOptimizer(
            m_dup.named_parameters(),  # 使用 m_dup 的命名参数
            torch.optim.SGD,  # 指定优化器类型为 SGD
            [
                {"params": m_dup.net1.parameters()},  # 第一个参数组使用 m_dup.net1 的参数
                {"params": m_dup.net3.parameters(), "lr": 1e-3},  # 第二个参数组使用 m_dup.net3 的参数，并设置不同的学习率
            ],
            lr=1e-2,  # 全局学习率设定为 0.01
            momentum=0.9,  # SGD 优化器的动量设定为 0.9
        )

        named_optim_2 = _NamedOptimizer(
            m_dup.named_parameters(),  # 使用 m_dup 的命名参数
            torch.optim.Adam,  # 指定优化器类型为 Adam
            [
                {"params": m_dup.net2.parameters()},  # 第一个参数组使用 m_dup.net2 的参数
                {"params": m_dup.net4.parameters(), "lr": 1e-5},  # 第二个参数组使用 m_dup.net4 的参数，并设置不同的学习率
            ],
        )

        # 使用自定义方法比较优化器的参数组
        self._compare_param_groups(optim_1.param_groups, named_optim_1.param_groups)
        self._compare_param_groups(optim_2.param_groups, named_optim_2.param_groups)

        # 运行模型训练，传入模型及其对应的优化器列表
        _run_model_training(
            [(m, [optim_1, optim_2]), (m_dup, [named_optim_1, named_optim_2])]
        )

        # 再次使用自定义方法比较优化器的参数组
        self._compare_param_groups(optim_1.param_groups, named_optim_1.param_groups)
        self._compare_param_groups(optim_2.param_groups, named_optim_2.param_groups)

        # 获取优化器的状态字典
        sd_1 = optim_1.state_dict()
        sd_2 = optim_2.state_dict()
        named_sd_1 = named_optim_1.state_dict()
        named_sd_2 = named_optim_2.state_dict()

        # 比较优化器状态字典中的 "state" 部分
        self._compare_state_dict_group(
            sd_1["state"][0],  # 对比 optim_1 的状态字典中第一个参数组的状态
            named_sd_1["state"]["net1.0.weight"],  # 对比 named_optim_1 中与 m_dup.net1.0.weight 对应的状态
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][1],  # 对比 optim_2 的状态字典中第二个参数组的状态
            named_sd_2["state"]["net2.0.bias"],  # 对比 named_optim_2 中与 m_dup.net2.0.bias 对应的状态
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_1["state"][2],  # 对比 optim_1 的状态字典中第三个参数组的状态
            named_sd_1["state"]["net3.weight"],  # 对比 named_optim_1 中与 m_dup.net3.weight 对应的状态
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][3],  # 对比 optim_2 的状态字典中第四个参数组的状态
            named_sd_2["state"]["net4.1.bias"],  # 对比 named_optim_2 中与 m_dup.net4.1.bias 对应的状态
            assert_equal=True,
        )

        # 比较优化器状态字典中的 "param_groups" 部分
        self._compare_state_dict_group(
            sd_1["param_groups"][0],  # 对比 optim_1 的状态字典中第一个参数组的参数组信息
            named_sd_1["param_groups"][0],  # 对比 named_optim_1 的状态字典中第一个参数组的参数组信息
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["param_groups"][1],  # 对比 optim_2 的状态字典中第二个参数组的参数组信息
            named_sd_2["param_groups"][1],  # 对比 named_optim_2 的状态字典中第二个参数组的参数组信息
            assert_equal=True
        )
    # 定义测试函数，用于验证 NamedOptimizer 的 load_state_dict 方法是否按预期工作
    def test_load_state_dict(self):
        """Check that NamedOptimizer's load_state_dict works as expected."""
        # 创建测试用的模型实例
        m = TestDummyModel()
        
        # 创建第一个 NamedOptimizer 实例
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        # 运行模型训练
        _run_model_training([(m, [named_optim_1])])
        
        # 获取第一个 NamedOptimizer 的状态字典
        state_dict_to_load = named_optim_1.state_dict()

        # 创建第二个 NamedOptimizer 实例
        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        # 再次运行模型训练
        _run_model_training([(m, [named_optim_2])])
        
        # 获取第二个 NamedOptimizer 的状态字典，作为加载前的状态字典
        state_dict_before_load = named_optim_2.state_dict()

        # 比较 state_dict_to_load 和 state_dict_before_load 中的各个部分
        # 比较 net1.0.weight 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_before_load["state"]["net1.0.weight"],
            assert_equal=False,
        )
        # 比较 net2.0.bias 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_before_load["state"]["net2.0.bias"],
            assert_equal=False,
        )
        # 比较 net3.weight 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_before_load["state"]["net3.weight"],
            assert_equal=False,
        )
        # 比较 net4.1.bias 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_before_load["state"]["net4.1.bias"],
            assert_equal=False,
        )

        # 使用 state_dict_to_load 加载 named_optim_2 的状态
        named_optim_2.load_state_dict(state_dict_to_load)
        
        # 获取加载后的 named_optim_2 的状态字典
        state_dict_after_load = named_optim_2.state_dict()

        # 再次比较 state_dict_to_load 和 state_dict_after_load 中的各个部分
        # 比较 net1.0.weight 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        # 比较 net2.0.bias 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_after_load["state"]["net2.0.bias"],
            assert_equal=True,
        )
        # 比较 net3.weight 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )
        # 比较 net4.1.bias 的状态
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_after_load["state"]["net4.1.bias"],
            assert_equal=True,
        )
    def test_load_state_dict_conditional_training(self):
        """Check that NamedOptimizer load_state_dict works under conditional training case."""
        # 创建测试用的模型实例
        m = TestDummyModel()
        # 创建第一个命名优化器实例
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),  # 使用模型的命名参数
            torch.optim.SGD,  # 使用 SGD 优化器
            [
                {"params": m.net1.parameters()},  # 设置 net1 的参数
                {"params": m.net3.parameters(), "lr": 1e-3},  # 设置 net3 的参数和学习率
            ],
            lr=1e-2,  # 设置全局学习率
            momentum=0.9,  # 设置动量
        )

        # 运行模型训练
        _run_model_training([(m, [named_optim_1])])
        # 获取第一个优化器的状态字典
        state_dict_to_load = named_optim_1.state_dict()

        # 创建第二个命名优化器实例
        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),  # 使用模型的命名参数
            torch.optim.SGD,  # 使用 SGD 优化器
            lr=1e-2,  # 设置全局学习率
            momentum=0.6,  # 设置动量
        )

        # 再次运行模型训练
        _run_model_training([(m, [named_optim_2])])
        # 加载第一个优化器的状态字典到第二个优化器中
        named_optim_2.load_state_dict(state_dict_to_load)
        # 获取加载后的第二个优化器的状态字典
        state_dict_after_load = named_optim_2.state_dict()

        # 比较优化器状态字典中的 "net1.0.weight" 部分
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        # 比较优化器状态字典中的 "net3.weight" 部分
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )

    def test_load_state_dict_error(self):
        # 创建测试用的模型实例
        m = TestDummyModel()
        # 创建第一个命名优化器实例
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),  # 使用模型的命名参数
            torch.optim.SGD,  # 使用 SGD 优化器
            lr=1e-2,  # 设置全局学习率
            momentum=0.9,  # 设置动量
        )

        # 运行模型训练
        _run_model_training([(m, [named_optim_1])])
        # 获取第一个优化器的状态字典
        state_dict_to_load = named_optim_1.state_dict()

        # 创建第二个命名优化器实例
        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),  # 使用模型的命名参数
            torch.optim.SGD,  # 使用 SGD 优化器
            lr=1e-2,  # 设置全局学习率
            momentum=0.6,  # 设置动量
        )

        # 定义预期的错误消息
        err_msg = (
            "Expects the optim to be initialized before load but found not initialized"
        )
        # 断言加载状态字典时抛出的特定错误消息
        with self.assertRaisesRegex(ValueError, err_msg):
            named_optim_2.load_state_dict(state_dict_to_load)
    # 定义一个测试函数，用于测试参数组的添加操作
    def test_add_param_group(self):
        # 创建一个测试用的模型实例
        m = TestDummyModel()
        # 创建另一个相同的测试用模型实例
        m_dup = TestDummyModel()
        
        # 使用 SGD 优化器初始化 `optim`，指定不同的学习率和动量参数
        optim = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},  # 添加模型 `m` 的 net1 层参数
                {"params": m.net3.parameters(), "lr": 1e-3},  # 添加模型 `m` 的 net3 层参数，并指定不同的学习率
            ],
            lr=1e-2,  # 设置全局学习率
            momentum=0.9,  # 设置动量参数
        )
        
        # 使用 `_NamedOptimizer` 初始化 `named_optim`，传入模型 `m_dup` 的命名参数，使用 SGD 优化器
        named_optim = _NamedOptimizer(
            m_dup.named_parameters(),  # 传入模型 `m_dup` 的命名参数
            torch.optim.SGD,  # 使用 SGD 优化器
            [
                {"params": m_dup.net1.parameters()},  # 添加模型 `m_dup` 的 net1 层参数
                {"params": m_dup.net3.parameters(), "lr": 1e-3},  # 添加模型 `m_dup` 的 net3 层参数，并指定不同的学习率
            ],
            lr=1e-2,  # 设置全局学习率
            momentum=0.9,  # 设置动量参数
        )

        # 运行模型训练，传入 [(模型m, [optim]), (模型m_dup, [named_optim])]
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        
        # 比较 `optim` 和 `named_optim` 的参数组是否相同
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        # 向 `optim` 添加一个新的参数组，包括模型 `m` 的 net2 层参数和指定的学习率
        optim.add_param_group({"params": m.net2.parameters(), "lr": 1e-5})
        
        # 向 `named_optim` 添加一个新的参数组，包括模型 `m_dup` 的 net2 层参数和指定的学习率
        named_optim.add_param_group({"params": m_dup.net2.parameters(), "lr": 1e-5})
        
        # 再次运行模型训练，传入 [(模型m, [optim]), (模型m_dup, [named_optim])]
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        
        # 再次比较 `optim` 和 `named_optim` 的参数组是否相同
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        # 向 `optim` 添加一个新的参数组，包括模型 `m` 的 net4[1].weight 参数和指定的学习率
        optim.add_param_group({"params": m.net4[1].weight, "lr": 1e-3})
        
        # 向 `named_optim` 添加一个新的参数组，包括模型 `m_dup` 的 net4[1].weight 参数和指定的学习率
        named_optim.add_param_group({"params": m_dup.net4[1].weight, "lr": 1e-3})
        
        # 再次运行模型训练，传入 [(模型m, [optim]), (模型m_dup, [named_optim])]
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        
        # 再次比较 `optim` 和 `named_optim` 的参数组是否相同
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

    # 定义一个测试函数，用于测试添加参数组时的错误处理
    def test_add_param_group_error(self):
        # 创建一个测试用的模型实例
        m = TestDummyModel()
        
        # 使用 `_NamedOptimizer` 初始化 `named_optim`，传入模型 `m` 的命名参数，使用 SGD 优化器
        named_optim = _NamedOptimizer(
            m.named_parameters(),  # 传入模型 `m` 的命名参数
            torch.optim.SGD,  # 使用 SGD 优化器
            [
                {"params": m.net1.parameters()},  # 添加模型 `m` 的 net1 层参数
                {"params": m.net3.parameters(), "lr": 1e-3},  # 添加模型 `m` 的 net3 层参数，并指定不同的学习率
            ],
            lr=1e-2,  # 设置全局学习率
            momentum=0.9,  # 设置动量参数
        )

        # 设置预期的错误消息
        err_msg = "some parameters are not in the module"
        
        # 使用 `assertRaisesRegex` 断言捕获 `ValueError` 异常，并检查错误消息是否匹配预期的消息
        with self.assertRaisesRegex(ValueError, err_msg):
            # 尝试向 `named_optim` 添加一个错误的参数组，包括一个形状为 (8, 1) 的张量和指定的学习率
            named_optim.add_param_group({"params": [torch.ones(8, 1)], "lr": 1e-5})
    # 定义测试类中的初始化状态测试方法
    def test_init_state(self):
        # 创建测试用的虚拟模型对象
        m = TestDummyModel()
        # 创建命名优化器对象，使用 SGD 优化器，传入模型的命名参数列表
        named_optim = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,  # 设置学习率为 0.01
            momentum=0.9,  # 设置动量为 0.9
        )
        # 获取命名优化器的状态字典
        named_sd = named_optim.state_dict()
        # 断言网络 net1 的第一个层的权重梯度为 None
        self.assertTrue(m.net1[0].weight.grad is None)
        # 断言状态字典中的状态部分为空
        self.assertTrue(len(named_sd["state"]) == 0)
        # 初始化命名优化器的状态
        named_optim.init_state()
        # 重新获取更新后的状态字典
        named_sd = named_optim.state_dict()
        # 断言网络 net1 的第一个层的权重梯度不为 None
        self.assertTrue(m.net1[0].weight.grad is not None)
        # 断言状态字典中 net1.0.weight 参数存在动量缓冲区
        self.assertTrue("momentum_buffer" in named_sd["state"]["net1.0.weight"])
        # 断言 net1.0.weight 的动量缓冲区中所有元素不全为 True
        self.assertFalse(
            torch.all(named_sd["state"]["net1.0.weight"]["momentum_buffer"]).item()
        )
        # 断言 net1.0.bias 的动量缓冲区中所有元素不全为 True
        self.assertFalse(
            torch.all(named_sd["state"]["net1.0.bias"]["momentum_buffer"]).item()
        )
        # 断言网络 net3 的偏置项梯度不为 None
        self.assertTrue(m.net3.bias.grad is not None)
        # 断言状态字典中 net3.bias 参数存在动量缓冲区
        self.assertTrue("momentum_buffer" in named_sd["state"]["net3.bias"])
        # 断言 net3.bias 的动量缓冲区中所有元素不全为 True
        self.assertFalse(
            torch.all(named_sd["state"]["net3.bias"]["momentum_buffer"]).item()
        )
        # 断言 net3.weight 的动量缓冲区中所有元素不全为 True
        self.assertFalse(
            torch.all(named_sd["state"]["net3.weight"]["momentum_buffer"]).item()
        )
```