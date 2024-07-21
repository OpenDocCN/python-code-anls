# `.\pytorch\test\ao\sparsity\test_sparsifier.py`

```
# Owner(s): ["module: unknown"]

import itertools  # 导入 itertools 模块，提供用于操作迭代器的函数
import logging  # 导入 logging 模块，用于记录日志信息
import re  # 导入 re 模块，提供正则表达式的支持

import torch  # 导入 PyTorch 模块
from torch import nn  # 从 torch 中导入 nn 模块，神经网络定义的核心类
from torch.ao.pruning import (  # 导入 PyTorch 剪枝相关模块
    BaseSparsifier,  # 基础稀疏化器的抽象基类
    FakeSparsity,  # 伪稀疏性模块
    NearlyDiagonalSparsifier,  # 近对角稀疏化器
    WeightNormSparsifier,  # 权重范数稀疏化器
)
from torch.nn.utils.parametrize import is_parametrized  # 导入参数化相关的工具函数
from torch.testing._internal.common_pruning import (  # 导入内部测试中的剪枝相关类和函数
    ImplementedSparsifier,  # 已实现的稀疏化器
    MockSparseLinear,  # 模拟稀疏线性层
    SimpleLinear,  # 简单的线性层模型
)

from torch.testing._internal.common_utils import TestCase  # 导入测试框架的 TestCase 类

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)  # 配置全局日志记录器的格式和日志级别为 INFO


class TestBaseSparsifier(TestCase):
    def test_constructor(self):
        # 测试不能实例化抽象基类的情况
        self.assertRaises(TypeError, BaseSparsifier)
        # 测试能够用没有配置的模型实例化稀疏化器
        model = SimpleLinear()
        sparsifier = ImplementedSparsifier(test=3)
        sparsifier.prepare(model, config=None)
        assert len(sparsifier.groups) == 5  # 断言稀疏化器的分组数量为 5
        sparsifier.step()  # 执行稀疏化器的一步操作
        # 测试能够用带有配置的模型实例化稀疏化器
        sparsifier = ImplementedSparsifier(test=3)
        sparsifier.prepare(model, [{"tensor_fqn": "linear1.weight"}])
        assert len(sparsifier.groups) == 1  # 断言稀疏化器的分组数量为 1
        assert sparsifier.groups[0]["tensor_fqn"] == "linear1.weight"  # 断言第一个分组的张量全限定名
        assert "test" in sparsifier.groups[0]  # 断言第一个分组包含 "test" 键
        assert sparsifier.groups[0]["test"] == 3  # 断言第一个分组的 "test" 值为 3

    def test_prepare_config(self):
        model = SimpleLinear()  # 创建一个简单线性层模型实例
        sparsifier = ImplementedSparsifier(test=3)  # 创建一个已实现稀疏化器实例
        # 在 `prepare` 方法调用之前，确保模型的各层没有被参数化
        assert not hasattr(model.seq[0], "parametrizations")
        assert not hasattr(model.linear1, "parametrizations")
        assert not hasattr(model.linear2, "parametrizations")
        # 调用 `prepare` 方法，配置模型的稀疏化组
        sparsifier.prepare(
            model,
            config=[
                {"tensor_fqn": "seq.0.weight", "test": 42},  # 针对序列第一个层的权重配置
                {"tensor_fqn": "linear2.weight"},  # 针对第二个线性层的权重配置
            ],
        )
        assert len(sparsifier.groups) == 2  # 断言稀疏化器的分组数量为 2
        # 检查默认参数是否被显式指定
        assert sparsifier.groups[0]["tensor_fqn"] == "seq.0.weight"
        assert sparsifier.groups[0]["test"] == 42
        # 检查 FQN 和模块是否指向相同位置
        assert sparsifier.groups[1]["tensor_fqn"] == "linear2.weight"
        assert sparsifier.groups[1]["module"] == model.linear2
        # 检查参数化是否已经附加到相应的模型层
        assert hasattr(model.seq[0], "parametrizations")
        assert not hasattr(model.linear1, "parametrizations")
        assert hasattr(model.linear2, "parametrizations")
    # 定义一个测试方法，用于测试稀疏化流程中的步骤
    def test_step(self):
        # 创建简单线性模型对象
        model = SimpleLinear()
        # 创建一个实现了稀疏化的对象，设定测试参数为3
        sparsifier = ImplementedSparsifier(test=3)
        # 启用掩码更新
        sparsifier.enable_mask_update = True
        # 准备稀疏化处理，指定模型和要处理的张量全名列表
        sparsifier.prepare(model, [{"tensor_fqn": "linear1.weight"}])
        # 执行一步稀疏化操作
        sparsifier.step()
        # 断言：检查模型的 linear1.weight 的第一个参数化对象的掩码的第一个元素是否为0
        assert torch.all(model.linear1.parametrizations.weight[0].mask[0] == 0)

    # 定义一个测试方法，用于测试状态字典的保存与加载
    def test_state_dict(self):
        # 设置步骤计数为3
        step_count = 3
        # 创建一个简单线性模型对象
        model0 = SimpleLinear()
        # 创建一个实现了稀疏化的对象，设定测试参数为3
        sparsifier0 = ImplementedSparsifier(test=3)
        # 准备稀疏化处理，指定模型和要处理的张量全名列表
        sparsifier0.prepare(model0, [{"tensor_fqn": "linear1.weight"}])
        # 获取模型的 linear1.weight 参数化对象的掩码
        mask = model0.linear1.parametrizations["weight"][0].mask
        # 设置掩码数据为按顺序排列的一维数组
        mask.data = torch.arange(mask.shape[0] * mask.shape[1]).reshape(mask.shape)
        # 对于每个步骤进行稀疏化处理
        for step in range(step_count):
            sparsifier0.step()
        # 获取稀疏化对象的状态字典
        state_dict = sparsifier0.state_dict()

        # 检查状态字典中预期的键是否存在
        assert "state" in state_dict
        assert "step_count" in state_dict["state"]["linear1.weight"]
        assert state_dict["state"]["linear1.weight"]["step_count"] == 3
        assert "groups" in state_dict
        assert "test" in state_dict["groups"][0]
        assert "tensor_fqn" in state_dict["groups"][0]
        assert state_dict["groups"][0]["tensor_fqn"] == "linear1.weight"

        # 断言：检查加载状态字典后创建的模型与原模型等效
        model1 = SimpleLinear()
        sparsifier1 = ImplementedSparsifier()
        sparsifier1.prepare(model1, None)

        assert sparsifier0.state != sparsifier1.state

        # 确保初始时掩码不同
        for mg in sparsifier0.groups:
            if mg["tensor_fqn"] == "linear1.weight":
                mask0 = mg["module"].parametrizations.weight[0].mask
        for mg in sparsifier1.groups:
            if mg["tensor_fqn"] == "linear1.weight":
                mask1 = mg["module"].parametrizations.weight[0].mask
        self.assertNotEqual(mask0, mask1)

        # 加载状态字典到 sparsifier1
        sparsifier1.load_state_dict(state_dict)

        # 确保加载后状态相同且正确
        assert sparsifier0.state == sparsifier1.state

        # 确保加载后掩码（及所有字典）相同
        assert len(sparsifier0.groups) == len(sparsifier1.groups)
        for idx in range(len(sparsifier0.groups)):
            mg0 = sparsifier0.groups[idx]
            mg1 = sparsifier1.groups[idx]
            for key in mg0.keys():
                assert key in mg1
                if key == "module":
                    # 由于模块不同，无法直接比较，但要确保参数的属性字典相同
                    param0 = mg0[key].parametrizations.weight[0]
                    param1 = mg1[key].parametrizations.weight[0]
                    assert hasattr(param0, "mask")
                    assert hasattr(param1, "mask")
                    self.assertEqual(param0.__dict__, param1.__dict__)
                else:
                    assert mg0[key] == mg1[key]
    # 定义测试方法，用于测试稀疏化转换功能
    def test_convert(self):
        # 创建简单线性模型实例
        model = SimpleLinear()
        # 创建实现了稀疏化接口的对象实例，传入测试参数
        sparsifier = ImplementedSparsifier(test=3)
        # 针对模型进行准备操作，指定线性层权重作为稀疏化目标
        sparsifier.prepare(model, [{"tensor_fqn": "linear1.weight"}])
        # 进行模型转换，将模型中的线性层替换为 MockSparseLinear 类型的对象
        new_model = sparsifier.convert(
            model, mapping={nn.Linear: MockSparseLinear}, inplace=False
        )

        # 断言新模型的属性类型符合预期
        assert isinstance(new_model.linear1, MockSparseLinear)
        assert isinstance(new_model.seq[0], nn.Linear)
        assert isinstance(new_model.linear2, nn.Linear)

    # 定义测试方法，用于测试掩码压缩功能
    def test_mask_squash(self):
        # 创建简单线性模型实例
        model = SimpleLinear()
        # 创建实现了稀疏化接口的对象实例，传入测试参数
        sparsifier = ImplementedSparsifier(test=3)
        # 针对模型进行准备操作，指定线性层权重作为稀疏化目标
        sparsifier.prepare(model, [{"tensor_fqn": "linear1.weight"}])
        # 断言模型的 linear1 属性具有 "mask" 成员
        assert hasattr(model.linear1.parametrizations.weight[0], "mask")
        # 断言模型的 linear1 的权重是否被稀疏化
        assert is_parametrized(model.linear1, "weight")
        # 断言模型的 seq 中的第一个层是否未被稀疏化
        assert not is_parametrized(model.seq[0], "weight")

        # 执行掩码压缩操作
        sparsifier.squash_mask()
        # 再次断言 seq 中的第一个层是否未被稀疏化
        assert not is_parametrized(model.seq[0], "weight")
        # 再次断言 linear1 的权重是否未被稀疏化
        assert not is_parametrized(model.linear1, "weight")

    # 定义测试方法，测试带参数的掩码压缩功能
    def test_mask_squash_with_params1(self):
        # 创建简单线性模型实例
        model = SimpleLinear()
        # 创建实现了稀疏化接口的对象实例，传入参数 foo=3, bar=2, baz=1
        sparsifier = ImplementedSparsifier(foo=3, bar=2, baz=1)
        # 针对模型进行准备操作，指定线性层和序列第一个权重作为稀疏化目标
        sparsifier.prepare(
            model, [{"tensor_fqn": "linear1.weight"}, {"tensor_fqn": "seq.0.weight"}]
        )
        # 执行带参数的掩码压缩操作，保留每层指定的参数
        sparsifier.squash_mask(
            params_to_keep_per_layer={"linear1": ("foo", "bar"), "seq.0": ("baz",)}
        )
        # 断言 seq 中的第一个层是否未被稀疏化
        assert not is_parametrized(model.seq[0], "weight")
        # 断言 linear1 的权重是否未被稀疏化
        assert not is_parametrized(model.linear1, "weight")
        # 断言 seq 中的第一个层是否具有 "sparse_params" 属性
        assert hasattr(model.seq[0], "sparse_params")
        # 断言 linear1 是否具有 "sparse_params" 属性
        assert hasattr(model.linear1, "sparse_params")
        # 断言 seq 中的第一个层的 sparse_params 属性 foo 值为 None
        assert model.seq[0].sparse_params.get("foo", None) is None
        # 断言 seq 中的第一个层的 sparse_params 属性 bar 值为 None
        assert model.seq[0].sparse_params.get("bar", None) is None
        # 断言 seq 中的第一个层的 sparse_params 属性 baz 值为 1
        assert model.seq[0].sparse_params.get("baz", None) == 1
        # 断言 linear1 的 sparse_params 属性 foo 值为 3
        assert model.linear1.sparse_params.get("foo", None) == 3
        # 断言 linear1 的 sparse_params 属性 bar 值为 2
        assert model.linear1.sparse_params.get("bar", None) == 2
        # 断言 linear1 的 sparse_params 属性 baz 值为 None
        assert model.linear1.sparse_params.get("baz", None) is None

    # 定义测试方法，测试带参数的掩码压缩功能
    def test_mask_squash_with_params2(self):
        # 创建简单线性模型实例
        model = SimpleLinear()
        # 创建实现了稀疏化接口的对象实例，传入参数 foo=3, bar=2, baz=1
        sparsifier = ImplementedSparsifier(foo=3, bar=2, baz=1)
        # 针对模型进行准备操作，指定线性层和序列第一个权重作为稀疏化目标
        sparsifier.prepare(
            model, [{"tensor_fqn": "linear1.weight"}, {"tensor_fqn": "seq.0.weight"}]
        )
        # 执行带参数的掩码压缩操作，保留指定的参数
        sparsifier.squash_mask(params_to_keep=("foo", "bar"))
        # 断言 seq 中的第一个层是否未被稀疏化
        assert not is_parametrized(model.seq[0], "weight")
        # 断言 linear1 的权重是否未被稀疏化
        assert not is_parametrized(model.linear1, "weight")
        # 断言 seq 中的第一个层是否具有 "sparse_params" 属性
        assert hasattr(model.seq[0], "sparse_params")
        # 断言 linear1 是否具有 "sparse_params" 属性
        assert hasattr(model.linear1, "sparse_params")
        # 断言 seq 中的第一个层的 sparse_params 属性 foo 值为 3
        assert model.seq[0].sparse_params.get("foo", None) == 3
        # 断言 seq 中的第一个层的 sparse_params 属性 bar 值为 2
        assert model.seq[0].sparse_params.get("bar", None) == 2
        # 断言 seq 中的第一个层的 sparse_params 属性 baz 值为 None
        assert model.seq[0].sparse_params.get("baz", None) is None
        # 断言 linear1 的 sparse_params 属性 foo 值为 3
        assert model.linear1.sparse_params.get("foo", None) == 3
        # 断言 linear1 的 sparse_params 属性 bar 值为 2
        assert model.linear1.sparse_params.get("bar", None) == 2
        # 断言 linear1 的 sparse_params 属性 baz 值为 None
        assert model.linear1.sparse_params.get("baz", None) is None
    `
        def test_mask_squash_with_params3(self):
            # 初始化一个 SimpleLinear 模型对象
            model = SimpleLinear()
            # 创建一个 ImplementedSparsifier 对象，传入初始化参数 foo=3, bar=2, baz=1
            sparsifier = ImplementedSparsifier(foo=3, bar=2, baz=1)
            # 准备 sparsifier，传入模型和需要处理的张量路径
            sparsifier.prepare(
                model, [{"tensor_fqn": "linear1.weight"}, {"tensor_fqn": "seq.0.weight"}]
            )
            # 调用 sparsifier 的 squash_mask 方法，指定需要保留的参数 foo 和 bar，以及每层 seq.0 的参数 baz
            sparsifier.squash_mask(
                params_to_keep=("foo", "bar"), params_to_keep_per_layer={"seq.0": ("baz",)}
            )
            # 断言 seq.0 层的 weight 参数没有被参数化
            assert not is_parametrized(model.seq[0], "weight")
            # 断言 linear1 层的 weight 参数没有被参数化
            assert not is_parametrized(model.linear1, "weight")
            # 断言 seq.0 层的模型对象有 sparse_params 属性
            assert hasattr(model.seq[0], "sparse_params")
            # 断言 linear1 层的模型对象有 sparse_params 属性
            assert hasattr(model.linear1, "sparse_params")
            # 断言 seq.0 层的 sparse_params 包含 foo 参数并且值为 3
            assert model.seq[0].sparse_params.get("foo", None) == 3
            # 断言 seq.0 层的 sparse_params 包含 bar 参数并且值为 2
            assert model.seq[0].sparse_params.get("bar", None) == 2
            # 断言 seq.0 层的 sparse_params 包含 baz 参数并且值为 1
            assert model.seq[0].sparse_params.get("baz", None) == 1
            # 断言 linear1 层的 sparse_params 包含 foo 参数并且值为 3
            assert model.linear1.sparse_params.get("foo", None) == 3
            # 断言 linear1 层的 sparse_params 包含 bar 参数并且值为 2
            assert model.linear1.sparse_params.get("bar", None) == 2
            # 断言 linear1 层的 sparse_params 不包含 baz 参数
            assert model.linear1.sparse_params.get("baz", None) is None
class TestWeightNormSparsifier(TestCase):
    def test_constructor(self):
        model = SimpleLinear()  # 创建一个简单的线性模型对象
        sparsifier = WeightNormSparsifier()  # 创建一个权重归一化稀疏化器对象
        sparsifier.prepare(model, config=None)  # 准备稀疏化器，使用给定的模型和配置（这里配置为空）
        for g in sparsifier.groups:
            assert isinstance(g["module"], nn.Linear)  # 断言每个组的模块是 nn.Linear 类型
            # The groups are unordered
            assert g["module_fqn"] in ("seq.0", "seq.1", "seq.2", "linear1", "linear2")  # 断言模块的全限定名属于指定的一组名称

    def test_step(self):
        model = SimpleLinear()  # 创建一个简单的线性模型对象
        sparsifier = WeightNormSparsifier(sparsity_level=0.5)  # 创建一个稀疏水平为0.5的权重归一化稀疏化器对象
        sparsifier.prepare(model, config=[{"tensor_fqn": "linear1.weight"}])  # 准备稀疏化器，使用给定的模型和配置（这里配置指定了linear1.weight）
        for g in sparsifier.groups:
            # Before step
            module = g["module"]  # 获取组中的模块
            assert (
                1.0 - module.parametrizations["weight"][0].mask.mean()
            ) == 0  # 检查稀疏化器中权重掩码的平均值是否为0，即检查稀疏水平是否为0
        sparsifier.enable_mask_update = True  # 启用掩码更新
        sparsifier.step()  # 执行稀疏化步骤
        self.assertAlmostEqual(
            model.linear1.parametrizations["weight"][0].mask.mean().item(),
            0.5,
            places=2,
        )  # 断言模型中linear1的权重掩码的平均值为0.5（保留两位小数）
        for g in sparsifier.groups:
            # After step
            module = g["module"]  # 获取组中的模块
            assert (
                1.0 - module.parametrizations["weight"][0].mask.mean()
            ) > 0  # 检查稀疏化器中权重掩码的平均值是否大于0，即检查稀疏水平是否增加了
        # Test if the mask collapses to all zeros if the weights are randomized
        iters_before_collapse = 1000  # 设置测试权重随机化前的迭代次数
        for _ in range(iters_before_collapse):
            model.linear1.weight.data = torch.randn(model.linear1.weight.shape)  # 随机化linear1的权重数据
            sparsifier.step()  # 执行稀疏化步骤
        for g in sparsifier.groups:
            # After step
            module = g["module"]  # 获取组中的模块
            assert (
                1.0 - module.parametrizations["weight"][0].mask.mean()
            ) > 0  # 检查稀疏化器中权重掩码的平均值是否仍大于0，即检查稀疏水平未发生塌缩

    def test_step_2_of_4(self):
        model = SimpleLinear()  # 创建一个简单的线性模型对象
        sparsifier = WeightNormSparsifier(
            sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
        )  # 创建一个稀疏水平为1.0，稀疏块形状为(1, 4)，每个块中零的数量为2的权重归一化稀疏化器对象
        sparsifier.prepare(model, config=[{"tensor_fqn": "linear1.weight"}])  # 准备稀疏化器，使用给定的模型和配置（这里配置指定了linear1.weight）
        sparsifier.step()  # 执行稀疏化步骤
        # make sure the sparsity level is approximately 50%
        mask = model.linear1.parametrizations["weight"][0].mask.to(
            torch.float
        )  # 将linear1的权重掩码转换为浮点数张量，以便计算均值
        self.assertAlmostEqual(mask.mean().item(), 0.5, places=2)  # 断言权重掩码的平均值约为0.5（保留两位小数）
        # Make sure each block has exactly 50% zeros
        module = sparsifier.groups[0]["module"]  # 获取组中的模块
        mask = module.parametrizations["weight"][0].mask  # 获取模块的权重掩码
        for row in mask:
            for idx in range(0, len(row), 4):
                block = row[idx : idx + 4]  # 获取权重掩码的一个块
                block, _ = block.sort()  # 将块排序
                assert (block[:2] == 0).all()  # 断言块的前两个元素全为0
                assert (block[2:] != 0).all()  # 断言块的后两个元素非全为0
    # 定义一个测试函数 test_prepare
    def test_prepare(self):
        # 创建一个简单线性模型对象
        model = SimpleLinear()
        # 创建一个权重归一化稀疏化器对象
        sparsifier = WeightNormSparsifier()
        # 准备模型，对权重进行稀疏化处理，配置参数为 None
        sparsifier.prepare(model, config=None)
        # 遍历稀疏化器对象的分组
        for g in sparsifier.groups:
            # 获取当前分组中的模块
            module = g["module"]
            # 断言：模块的参数化 weight[0] 中存在属性 mask
            assert hasattr(module.parametrizations["weight"][0], "mask")
            # 断言：检查参数化存在且正确
            assert is_parametrized(module, "weight")
            # 断言：模块的参数化 weight[0] 的类型为 FakeSparsity

    # 定义一个测试函数 test_mask_squash
    def test_mask_squash(self):
        # 创建一个简单线性模型对象
        model = SimpleLinear()
        # 创建一个权重归一化稀疏化器对象
        sparsifier = WeightNormSparsifier()
        # 准备模型，对权重进行稀疏化处理，配置参数为 None
        sparsifier.prepare(model, config=None)
        # 执行稀疏化器对象的 squash_mask 方法
        sparsifier.squash_mask()
        # 遍历稀疏化器对象的分组
        for g in sparsifier.groups:
            # 获取当前分组中的模块
            module = g["module"]
            # 断言：模块的 weight 参数未被参数化
            assert not is_parametrized(module, "weight")
            # 断言：模块不具有属性 mask

    # 定义一个测试函数 test_sparsity_levels
    def test_sparsity_levels(self):
        # 定义不同的稀疏度水平、稀疏块形状和每块中的零值数量的测试用例
        sparsity_levels = [-1.0, 0.0, 0.5, 1.0, 2.0]
        sparse_block_shapes = [(1, 1), (1, 4), (2, 2), (4, 1)]
        zeros_per_blocks = [0, 1, 2, 3, 4]

        # 使用 itertools.tee 对测试用例进行复制
        testcases = itertools.tee(
            itertools.product(sparsity_levels, sparse_block_shapes, zeros_per_blocks)
        )
        # 创建一个神经网络的顺序容器对象
        model = nn.Sequential()
        # 创建一个权重归一化稀疏化器对象
        sparsifier = WeightNormSparsifier()

        # 创建存储每层稀疏性配置的列表
        sparsity_per_layer_config = []
        # 编译用于名称规范化的正则表达式模式
        p = re.compile(r"[-\.\s]")
        # 遍历测试用例中的每个元组
        for sl, sbs, zpb in testcases[0]:
            # 确保每块中的零值数量不超过块内值的数量
            if zpb > sbs[0] * sbs[1]:
                continue
            # 构建层名称，使用正则表达式替换非法字符
            layer_name = f"{sl}_{sbs}_{zpb}"
            layer_name = p.sub("_", layer_name)

            # 创建一个线性层对象
            layer = nn.Linear(12, 12, bias=False)
            # 将权重参数设为全 1
            layer.weight = nn.Parameter(torch.ones(12, 12))
            # 将层添加到模型中，使用构建的层名称
            model.add_module(layer_name, layer)
            # 创建层的稀疏性配置字典
            config = {
                "tensor_fqn": layer_name + ".weight",
                "sparsity_level": sl,
                "sparse_block_shape": sbs,
                "zeros_per_block": zpb,
            }
            # 将配置字典添加到列表中
            sparsity_per_layer_config.append(config)

        # 准备模型，使用层稀疏性配置列表
        sparsifier.prepare(model, sparsity_per_layer_config)
        # 执行稀疏化器的步骤方法
        sparsifier.step()
        # 执行稀疏化器的 squash_mask 方法
        sparsifier.squash_mask()
        # 将模型设为评估模式
        model.eval()

        # 再次遍历测试用例中的每个元组
        for sl, sbs, zpb in testcases[1]:
            # 确保每块中的零值数量不超过块内值的数量
            if zpb > sbs[0] * sbs[1]:
                continue
            # 构建层名称，使用正则表达式替换非法字符
            layer_name = f"{sl}_{sbs}_{zpb}"
            layer_name = p.sub("_", layer_name)
            # 获取模型中对应名称的层对象
            layer = getattr(model, layer_name)

            # 断言：验证达到了所需的稀疏度水平
            sparse_mask = (layer.weight == 0).float()
            if zpb == 0:
                assert sparse_mask.mean() == 0
            else:
                # 计算期望的稀疏率
                true_sl = min(max(sl, 0.0), 1.0)
                true_sl = true_sl * zpb / sbs[0] / sbs[1]
                # 断言：验证稀疏掩码的平均值与期望值相等
                assert sparse_mask.mean() == true_sl
class TestNearlyDiagonalSparsifier(TestCase):
    # 测试 NearlyDiagonalSparsifier 类的单元测试
    def test_constructor(self):
        # 创建一个 SimpleLinear 模型实例
        model = SimpleLinear()
        # 创建 NearlyDiagonalSparsifier 实例，设置近似度参数为 1
        sparsifier = NearlyDiagonalSparsifier(nearliness=1)
        # 准备 sparsifier 实例，传入模型和配置为 None
        sparsifier.prepare(model, config=None)
        # 遍历 sparsifier 中的分组
        for g in sparsifier.groups:
            # 断言分组中的模块是 nn.Linear 类型
            assert isinstance(g["module"], nn.Linear)
            # 断言模块全限定名在给定的列表中，顺序不重要
            assert g["module_fqn"] in ("seq.0", "seq.1", "seq.2", "linear1", "linear2")

    # 测试 NearlyDiagonalSparsifier 类的单步测试
    def test_step(self):
        # 创建一个 SimpleLinear 模型实例
        model = SimpleLinear()
        # 创建 NearlyDiagonalSparsifier 实例，设置近似度参数为 1
        sparsifier = NearlyDiagonalSparsifier(nearliness=1)
        # 准备 sparsifier 实例，传入模型和指定的配置
        sparsifier.prepare(model, config=[{"tensor_fqn": "linear1.weight"}])

        # 遍历 sparsifier 中的分组
        for g in sparsifier.groups:
            # 在执行步骤之前
            module = g["module"]
            # 断言权重的稀疏程度为 0
            assert (1.0 - module.parametrizations["weight"][0].mask.mean()) == 0  # checking sparsity level is 0

        # 启用 mask 更新
        sparsifier.enable_mask_update = True
        # 执行 sparsifier 的步骤
        sparsifier.step()
        # 获取权重的 mask
        mask = module.parametrizations["weight"][0].mask
        height, width = mask.shape
        # 断言 mask 等于单位矩阵
        assert torch.all(mask == torch.eye(height, width))

        # 再次遍历 sparsifier 中的分组
        for g in sparsifier.groups:
            # 在执行步骤之后
            module = g["module"]
            # 断言权重的稀疏程度增加了
            assert (1.0 - module.parametrizations["weight"][0].mask.mean()) > 0  # checking sparsity level has increased

        # 如果权重被随机化，测试 mask 是否会全部收缩为零
        iters_before_collapse = 1000
        for _ in range(iters_before_collapse):
            # 将 linear1 的权重数据随机化
            model.linear1.weight.data = torch.randn(model.linear1.weight.shape)
            sparsifier.step()
        # 再次遍历 sparsifier 中的分组
        for g in sparsifier.groups:
            # 在执行步骤之后
            module = g["module"]
            # 断言权重的稀疏程度没有完全收缩为零
            assert (1.0 - module.parametrizations["weight"][0].mask.mean()) > 0  # checking sparsity level did not collapse

    # 测试 NearlyDiagonalSparsifier 类的准备方法
    def test_prepare(self):
        # 创建一个 SimpleLinear 模型实例
        model = SimpleLinear()
        # 创建 NearlyDiagonalSparsifier 实例，设置近似度参数为 1
        sparsifier = NearlyDiagonalSparsifier(nearliness=1)
        # 准备 sparsifier 实例，传入模型和配置为 None
        sparsifier.prepare(model, config=None)
        # 遍历 sparsifier 中的分组
        for g in sparsifier.groups:
            module = g["module"]
            # 断言模块具有 mask 属性
            assert hasattr(module.parametrizations["weight"][0], "mask")
            # 断言模块的权重参数被正确地参数化
            assert is_parametrized(module, "weight")
            assert type(module.parametrizations.weight[0]) == FakeSparsity
    # 测试函数，验证模型是否能正确进行掩码压缩
    def test_mask_squash(self):
        # 创建一个简单的线性模型
        model = SimpleLinear()
        # 创建一个近对角稀疏化对象，设置近对角距离为1
        sparsifier = NearlyDiagonalSparsifier(nearliness=1)
        # 对模型进行稀疏化准备，配置为None
        sparsifier.prepare(model, config=None)
        # 执行稀疏化步骤
        sparsifier.step()
        # 对掩码进行压缩
        sparsifier.squash_mask()
        # 遍历稀疏化器的分组
        for g in sparsifier.groups:
            # 获取当前组的模块
            module = g["module"]
            # 断言当前模块的权重参数没有被稀疏化
            assert not is_parametrized(module, "weight")
            # 断言当前模块没有名为"mask"的属性
            assert not hasattr(module, "mask")
            # 获取当前模块的权重
            weights = module.weight
            # 获取权重的高度和宽度
            height, width = weights.shape
            # 断言权重矩阵中只有对角线上的元素不为零
            assert torch.all(
                weights == torch.eye(height, width) * weights
            )  # 只有对角线上的元素应该存在

    # 测试函数，验证不同稀疏化级别的效果
    def test_sparsity_levels(self):
        # 创建一个包含多个层的神经网络模型
        nearliness_levels = list(range(-1, 100))
        model = nn.Sequential()

        # 正则表达式模式，用于处理层名称中的特殊字符
        p = re.compile(r"[-\.\s]")
        # 遍历各个稀疏化级别
        for nearliness in nearliness_levels:
            # 创建一个近对角稀疏化对象，设置近对角距离为1
            sparsifier = NearlyDiagonalSparsifier(nearliness=1)
            # 构造层名称
            layer_name = f"{nearliness}"
            # 使用正则表达式处理层名称，将特殊字符替换为下划线
            layer_name = p.sub("_", layer_name)

            # 创建一个线性层
            layer = nn.Linear(32, 32, bias=False)
            # 设置线性层的权重为全1
            layer.weight = nn.Parameter(torch.ones(32, 32))
            # 获取权重的宽度和高度
            width, height = layer.weight.shape
            # 将层添加到模型中，使用处理后的层名称作为标识
            model.add_module(layer_name, layer)
            # 构造配置字典，用于稀疏化准备阶段
            config = {"tensor_fqn": layer_name + ".weight", "nearliness": nearliness}

            # 执行模型稀疏化准备阶段
            sparsifier.prepare(model, [config])
            # 当稀疏化级别参数不合法时，应抛出 ValueError 异常
            if (nearliness > 0 and nearliness % 2 == 0) or (
                nearliness // 2 >= min(width, height)
            ):
                with self.assertRaises(ValueError):
                    sparsifier.step()
            else:
                # 执行稀疏化步骤
                sparsifier.step()
                # 对掩码进行压缩
                sparsifier.squash_mask()
                # 切换模型为评估模式
                model.eval()

                # 获取特定名称的层对象
                layer = getattr(model, layer_name)
                # 验证创建的掩码是否与稀疏化级别相对应
                self._verify_nearliness(layer.weight, nearliness)

    # 辅助函数，用于验证掩码的近对角稀疏性
    def _verify_nearliness(self, mask: torch.Tensor, nearliness: int):
        # 如果稀疏化级别小于等于0，则掩码应该全为零
        if nearliness <= 0:
            assert torch.all(mask == torch.zeros(mask.shape[0], mask.shape[1]))
        else:
            # 获取掩码的高度和宽度
            height, width = mask.shape
            # 计算到对角线的距离
            dist_to_diagonal = nearliness // 2
            # 遍历掩码的每一个元素
            for row in range(0, height):
                for col in range(0, width):
                    # 根据距离判断掩码元素应为1还是0
                    if abs(row - col) <= dist_to_diagonal:
                        assert mask[row, col] == 1
                    else:
                        assert mask[row, col] == 0
```