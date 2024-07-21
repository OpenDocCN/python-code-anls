# `.\pytorch\test\ao\sparsity\test_data_sparsifier.py`

```
# Owner(s): ["module: unknown"]

# 导入必要的库
import copy  # 导入深拷贝模块
import itertools  # 导入迭代工具模块
import logging  # 导入日志模块
import math  # 导入数学模块

from typing import Tuple  # 导入类型提示模块中的元组类型

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch中的神经网络模块

# 导入PyTorch中用于稀疏化数据的实验性功能
from torch.ao.pruning._experimental.data_sparsifier import (
    BaseDataSparsifier,
    DataNormSparsifier,
)
# 导入PyTorch中用于量化的实验性工具函数
from torch.ao.pruning._experimental.data_sparsifier.quantization_utils import (
    post_training_sparse_quantize,
)
# 导入PyTorch中的参数化检测工具函数
from torch.nn.utils.parametrize import is_parametrized
# 导入PyTorch中的测试工具类
from torch.testing._internal.common_utils import TestCase

# 配置日志格式和日志级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class ImplementedSparsifier(BaseDataSparsifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_mask(self, name, data, **kwargs):
        # 获取名为name的掩码，并将第一个元素置为0
        mask = self.get_mask(name)
        mask[0] = 0
        # 获取名为name的线性状态，并更新其步数计数器
        linear_state = self.state[name]
        linear_state["step_count"] = linear_state.get("step_count", 0) + 1


class _BaseDataSparsiferTestCase(TestCase):
    r"""This helper test class takes in any supported type of and runs some tests.
    The user is required to pass in the data that needs to sparsified and the
    runner will run some tests that needs to be passed in order for the data
    type to be supported.
    TODO: Change the structure by creating a separate test case class for each
          member function
    """

    def run_all_checks(self, data_list, data_with_config, defaults):
        # 依次运行各种检查函数，验证数据列表、配置数据和默认值
        self.check_constructor(data_list, data_with_config, defaults)
        self.check_squash_mask(data_list, data_with_config, defaults)
        self.check_add_data(data_list, data_with_config, defaults)
        self.check_step(data_list, data_with_config, defaults)
        self.check_state_dict(data_list, data_with_config, defaults)
        self.check_memory_reference(data_list, data_with_config, defaults)

    @staticmethod
    def _get_name_data_config(some_data, defaults=None):
        if isinstance(some_data, Tuple):
            # 如果some_data是元组，则处理数据列表
            name, data = some_data
            config = defaults
        else:
            # 如果some_data是字典，则处理配置数据
            name, data, config = (
                some_data["name"],
                some_data["data"],
                some_data["config"],
            )
        return name, data, config

    @staticmethod
    def _make_sparsifier(
        data_list,
        data_with_config,
        defaults,
        sparsifier_type=None,
        sparsifier_kwargs=None,
    # 创建稀疏化器对象的私有方法，根据给定的数据列表、配置字典和默认参数生成一个稀疏化器对象
    def _make_sparsifier(self, data_list, data_with_config, defaults, **kwargs):
        # 如果稀疏化器类型未指定，则使用默认的实现稀疏化器，并传入数据列表和默认参数
        if sparsifier_type is None:
            sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        else:
            # 如果指定了稀疏化器类型，则深拷贝默认参数，并更新为传入的稀疏化器关键字参数
            kwargs = copy.deepcopy(defaults)
            kwargs.update(sparsifier_kwargs)
            kwargs["data_list"] = data_list
            sparsifier = sparsifier_type(**kwargs)
        
        # 断言生成的稀疏化器对象的数据组数量与数据列表的长度相等
        assert len(sparsifier.data_groups) == len(data_list)
        
        # 遍历数据与配置的列表，每个元素包含名称、数据和配置，将其添加到稀疏化器中
        for data_config_dict in data_with_config:
            name, data, config = (
                data_config_dict["name"],
                data_config_dict["data"],
                data_config_dict["config"],
            )
            sparsifier.add_data(name=name, data=data, **config)
        
        # 返回生成的稀疏化器对象
        return sparsifier

    # 检查稀疏化器对象的构造函数是否正确设置，比较稀疏化器的数据组数量与输入数据列表与配置总和的长度
    def check_constructor(self, data_list, data_with_config, defaults, **kwargs):
        # 调用私有方法 _make_sparsifier 来生成稀疏化器对象
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        
        # 使用断言确保稀疏化器的数据组数量等于数据列表和数据配置总和的长度
        self.assertEqual(
            len(sparsifier.data_groups),
            len(data_list) + len(data_with_config),
            msg="Sparsifier data groups don't match the input "
            f"({len(sparsifier.data_groups)} vs. "
            f"{len(data_list) + len(data_with_config)}).",
        )

        # 将数据列表和数据配置合并为一个列表
        all_data = data_list + data_with_config

        # 遍历所有数据，获取名称、数据和配置，并确保每个名称存在于稀疏化器的数据组中，并且其配置与稀疏化器的配置匹配
        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults=defaults)
            self.assertIn(name, sparsifier.data_groups)
            self.assertEqual(sparsifier.data_groups[name], config)
    # 定义一个方法，用于检查稀疏化对象的执行步骤结果
    def check_step(self, data_list, data_with_config, defaults, **kwargs):
        # 创建稀疏化器对象，根据传入的数据列表、带配置的数据和默认参数
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        # 将数据列表和带配置的数据合并为一个列表
        all_data = data_list + data_with_config
        
        # 在执行步骤前检查数据和掩码
        for some_data in all_data:
            # 获取数据的名称、数据本身以及配置信息
            name, data, _ = self._get_name_data_config(some_data)
            # 从数据中提取权重，使用稀疏化器对象
            data = sparsifier._extract_weight(data)
            # 获取稀疏化后的数据，不返回原始数据
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            # 获取原始数据
            original_data = sparsifier.get_data(name=name, return_original=True)
            # 获取数据的掩码
            mask = sparsifier.get_mask(name=name)
            # 断言稀疏化后的数据与提取的权重数据相等
            self.assertEqual(sparsified_data, data)
            # 断言原始数据与提取的权重数据相等
            self.assertEqual(original_data, data)
            # 断言掩码的广播形状第一个元素为1
            self.assertEqualBroadcasting(mask[0], 1)

        # 设置步骤计数为3
        step_count = 3

        # 执行指定次数的步骤
        for _ in range(0, step_count):
            sparsifier.step()

        # 再次检查数据和掩码
        for some_data in all_data:
            name, data, _ = self._get_name_data_config(some_data)
            data = sparsifier._extract_weight(data)
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            original_data = sparsifier.get_data(name=name, return_original=True)
            mask = sparsifier.get_mask(name=name)
            # 断言稀疏化后的数据的第一个元素为0
            self.assertEqualBroadcasting(sparsified_data[0], 0)
            # 断言原始数据与提取的权重数据相等
            self.assertEqual(original_data, data)
            # 断言掩码的广播形状第一个元素为0
            self.assertEqualBroadcasting(mask[0], 0)
            # 确保在稀疏化状态对象中存在"step_count"键，并且其值为3
            assert "step_count" in sparsifier.state[name]
            assert sparsifier.state[name]["step_count"] == 3

    # 定义一个方法，用于检查压缩掩码操作的结果
    def check_squash_mask(self, data_list, data_with_config, defaults, **kwargs):
        # 创建稀疏化器对象，根据传入的数据列表、带配置的数据和默认参数
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        # 将数据列表和带配置的数据合并为一个列表
        all_data = data_list + data_with_config
        
        # 针对每个数据检查其名称是否存在于稀疏化容器中，并且是否已参数化
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            assert hasattr(sparsifier._container, name)
            assert is_parametrized(sparsifier._container, name)
        
        # 执行一次步骤
        sparsifier.step()
        # 压缩掩码
        sparsifier.squash_mask()

        # 针对每个数据检查其名称是否不存在于稀疏化容器中，已不再参数化
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            assert not is_parametrized(
                sparsifier._container, name
            )  # 不再参数化
            # 使用断言确保获取数据时会引发 ValueError 异常
            with self.assertRaises(ValueError):
                sparsifier.get_data(name, return_original=True)
    # 定义一个方法，用于检查并添加数据到稀疏化对象中
    def check_add_data(self, data_list, data_with_config, defaults, **kwargs):
        # 创建稀疏化对象，使用给定的数据列表、带配置的数据和默认参数
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        # 将所有数据列表与带配置的数据列表合并
        all_data = data_list + data_with_config
        # 遍历所有数据
        for some_data in all_data:
            # 获取数据的名称、数据本身和配置信息
            name1, data1, config = self._get_name_data_config(
                some_data, defaults=defaults
            )
            # 提取数据1的权重信息并更新
            data1 = sparsifier._extract_weight(data1)
            # 深拷贝数据1备份
            data1_old = copy.deepcopy(data1)
            # 断言数据1与稀疏化对象中对应名称的数据完全一致
            assert torch.all(data1 == sparsifier.get_data(name=name1))

            # 执行稀疏化对象的步进操作
            sparsifier.step()
            # 获取稀疏化对象中名称为name1的掩码
            mask = sparsifier.get_mask(name1)

            # 创建一个与原始数据1相同形状的随机数据data2
            data2 = torch.randn(
                data1.shape
            )  # add another data with the same shape as original data
            # 向稀疏化对象中添加数据2，并断言数据2与稀疏化对象中对应名称的数据完全一致
            sparsifier.add_data(name=name1, data=data2)
            assert torch.all(data2 == sparsifier.get_data(name=name1))

            # 断言稀疏化对象中名称为name1的掩码未发生变化
            assert torch.all(
                sparsifier.get_mask(name1) == mask
            )  # mask should not change
            # 断言数据1的备份与数据1本身未发生变化
            assert torch.all(data1_old == data1)

            # 断言稀疏化对象中名称为name1的数据组信息与配置信息相匹配
            assert (
                sparsifier.data_groups[name1] == config
            )  # if replaced old_config should match new config
    def check_state_dict(self, data_list, data_with_config, defaults, **kwargs):
        # 创建第一个稀疏化器对象，使用传入的数据列表、数据配置和默认参数
        sparsifier1 = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        # 创建第二个稀疏化器对象，仅使用第一个数据列表的第一个元素，并且不包含数据配置
        sparsifier2 = self._make_sparsifier(
            data_list=[data_list[0]], data_with_config=[], defaults=defaults, **kwargs
        )
        # 执行第一个稀疏化器的步骤
        sparsifier1.step()

        # 获取第一个稀疏化器的状态字典
        state_dict1 = sparsifier1.state_dict()

        # 断言第一个稀疏化器和第二个稀疏化器的状态不同
        assert sparsifier1.state != sparsifier2.state
        # 获取第一个数据列表的第一个元素的名称、数据和配置信息
        name, _, _ = self._get_name_data_config(data_list[0])
        # 断言第一个稀疏化器和第二个稀疏化器的指定掩码不相等
        self.assertNotEqual(sparsifier1.get_mask(name), sparsifier2.get_mask(name))

        # 将第一个稀疏化器的状态字典加载到第二个稀疏化器中
        sparsifier2.load_state_dict(state_dict1)
        # 断言两个稀疏化器的状态长度相等
        assert len(sparsifier1.state) == len(sparsifier2.state)
        # 断言两个稀疏化器的数据组长度相等
        assert len(sparsifier1.data_groups) == len(sparsifier2.data_groups)

        # 获取第一个稀疏化器状态字典中的"state"字段
        state1 = state_dict1["state"]
        for name in state1.keys():
            # 比较掩码
            assert name in sparsifier2.state
            assert "mask" in sparsifier2.state[name]
            assert "mask" in sparsifier1.state[name]
            mask1, mask2 = state1[name]["mask"], sparsifier2.state[name]["mask"]
            # 断言第一个掩码为稀疏 COO 格式，第二个掩码为密集格式
            assert mask1.is_sparse and not mask2.is_sparse
            assert torch.all(
                mask1.to_dense() == mask2
            )  # mask1 is stored as sparse coo now

            # 比较 data_groups
            dg1, dg2 = sparsifier1.data_groups, sparsifier2.data_groups
            assert name in dg1 and name in dg2
            assert dg1[name] == dg2[name]

            # 比较 container
            container1, container2 = sparsifier1._container, sparsifier2._container
            assert torch.all(getattr(container1, name) == getattr(container2, name))
            assert is_parametrized(container1, name) == is_parametrized(
                container2, name
            )
            if is_parametrized(container1, name):
                param1 = getattr(container1.parametrizations, name)[0]
                param2 = getattr(container2.parametrizations, name)[0]
                # 断言参数化对象具有"mask"属性
                assert hasattr(param1, "mask")
                assert hasattr(param2, "mask")
                # 断言参数化对象的属性字典相等
                self.assertEqual(param1.__dict__, param2.__dict__)
    # 定义一个方法，用于检查数据是否真正“附加”到稀疏化对象上。
    # 这意味着当数据在稀疏化对象之外改变时，这些改变必须在稀疏化对象内部的数据上反映出来。
    # 这确保了稀疏化对象持有数据的内存引用，而不是数据的副本。

    # 此测试修改数据，并断言稀疏化对象中的数据也发生了变化。
    def check_memory_reference(self, data_list, data_with_config, defaults, **kwargs):
        # 使用给定的数据列表、配置数据和默认参数创建稀疏化对象
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        # 将所有数据合并为一个列表
        all_data = data_list + data_with_config
        # 遍历所有数据项
        for some_data in all_data:
            # 获取数据项的名称、数据及其配置
            name, data, _ = self._get_name_data_config(some_data)
            # 从稀疏化对象中提取权重数据
            weight = sparsifier._extract_weight(data)
            # 修改权重数据，加上随机数以确保变化
            weight.data = weight + torch.randn(*weight.shape)
            # 获取稀疏化对象中指定名称的数据
            contained_data = sparsifier.get_data(name=name)
            # 断言权重数据的存储地址与稀疏化对象中对应数据的存储地址相同
            assert (
                weight.data.storage().data_ptr()
                == contained_data.data.storage().data_ptr()
            )
            # 断言稀疏化对象中的数据与权重数据一致
            assert torch.all(contained_data == weight)
class _NormDataSparsifierTestCase(_BaseDataSparsiferTestCase):
    r"""This helper test class takes in any supported type of and runs some tests.
    This inherits the TestBaseDataSparsifierRuner wherein some functions are
    over-ridden to take accomodate the specific sparsifier.
    TODO: Change the structure by creating a separate test case class for each
          member function
    """

    def run_all_checks(self, data_list, defaults, data_with_config, norm_type="L1"):
        # 检查规范化类型是否为支持的 L1 或 L2
        assert norm_type in ["L1", "L2"]
        # 准备关键字参数字典，用于构造函数测试
        kwargs = {
            "sparsifier_type": DataNormSparsifier,
            "sparsifier_kwargs": {"norm": norm_type},
        }
        # 调用基类方法，测试构造函数
        self.check_constructor(data_list, data_with_config, defaults, **kwargs)
        # 调用基类方法，测试压缩掩码函数
        self.check_squash_mask(data_list, data_with_config, defaults, **kwargs)
        # 调用基类方法，测试添加数据函数
        self.check_add_data(data_list, data_with_config, defaults, **kwargs)
        # 调用基类方法，测试状态字典函数
        self.check_state_dict(data_list, data_with_config, defaults, **kwargs)
        # 调用基类方法，测试步骤函数
        self.check_step(data_list, data_with_config, defaults, norm_type=norm_type)
        # 调用基类方法，测试步骤 2 中的一部分
        self.check_step_2_of_4(norm_type=norm_type)
        # 调用基类方法，测试稀疏级别函数
        self.check_sparsity_level(
            data_list, data_with_config, defaults, norm_type=norm_type
        )
        # 调用基类方法，测试内存引用函数
        self.check_memory_reference(data_list, data_with_config, defaults, **kwargs)

    @staticmethod
    def _get_bounds_on_actual_sparsity(config, tensor_shape):
        r"""This function gets the bounds on actual sparsity.
        Note::
            Although we specify the sparsity_level parameter, this does not mean that
            the actual sparsity obtained after sparsification is the same as sparsity_level.
            The actual sparsity depends largely on the shape and the data itself.
        """
        # 获取稀疏度配置参数
        sparsity_level = config["sparsity_level"]
        zeros_per_block = config["zeros_per_block"]
        sparse_block_shape = config["sparse_block_shape"]

        # 获取张量形状的高度和宽度
        height, width = tensor_shape[-2], tensor_shape[-1]
        block_height, block_width = sparse_block_shape

        # 计算块的数量和每个块的值
        number_blocks = math.ceil(height / block_height) * math.ceil(width / block_width)
        values_per_block = block_height * block_width

        if zeros_per_block == 0:
            # 如果每个块的零数为零，则返回最大稀疏度为 1.0
            return (1.0, 1.0)
        else:
            # 计算最小和最大稀疏值
            # 最小值假设每块零数为 1
            min_values_sparsified = round(number_blocks * sparsity_level)
            # 最大值假设实际的每块零数
            max_values_sparsified = min_values_sparsified * min(
                values_per_block, zeros_per_block
            )
            # 计算稀疏度的下限和上限
            lower_bound = min_values_sparsified / (height * width)
            upper_bound = min(1.0, max_values_sparsified / (height * width))

            # 四舍五入下限和上限的值，保留三位小数
            lower_bound, upper_bound = round(lower_bound, 3), round(upper_bound, 3)
            return lower_bound, upper_bound
    def check_step(self, data_list, data_with_config, defaults, norm_type="L1"):
        # 创建稀疏化器对象，用于数据规范化
        sparsifier = self._make_sparsifier(
            data_list,
            data_with_config,
            defaults,
            sparsifier_type=DataNormSparsifier,
            sparsifier_kwargs={"norm": norm_type},
        )
        # 将所有数据列表合并为一个列表
        all_data = data_list + data_with_config

        # 在执行 step() 方法前，验证 mask 不应该被稀疏化
        for some_data in all_data:
            # 获取数据名称及其配置
            name, _, _ = self._get_name_data_config(some_data)
            # 获取指定数据的稀疏化 mask
            mask = sparsifier.get_mask(name=name)
            # 断言稀疏化水平为 0，即全为稠密值
            assert (1.0 - mask.mean()) == 0  # checking sparsity level is 0

        # 执行一步稀疏化操作
        sparsifier.step()

        # 验证每个数据在执行 step() 后的稀疏化结果
        for some_data in all_data:
            # 获取数据名称及其配置
            name, _, _ = self._get_name_data_config(some_data)
            # 获取指定数据的稀疏化 mask
            mask = sparsifier.get_mask(name=name)
            # 获取数据的配置信息
            config = sparsifier.data_groups[name]
            # 获取实际稀疏化的下界和上界
            lb, ub = self._get_bounds_on_actual_sparsity(config, mask.shape)
            # 将 mask 转换为 float 类型
            mask = mask.to(torch.float)
            # 计算实际稀疏化水平并四舍五入到三位小数
            actual_sparsity = round(1 - mask.mean().item(), 3)
            # 断言实际稀疏化水平在预期的下界和上界之间
            assert actual_sparsity >= lb and actual_sparsity <= ub
            # 断言实际稀疏化水平大于 0
            assert (
                actual_sparsity > 0.0
            )  # exact sparsity level cannot be achieved due to size of tensor

        # 设置在发生折叠之前的迭代次数
        iters_before_collapse = 100

        # 创建测试用的稀疏化器对象
        test_sparsifier = DataNormSparsifier(
            sparsity_level=0.5,
            sparse_block_shape=(1, 4),
            zeros_per_block=4,
            norm=norm_type,
        )

        # 进行一定数量的迭代来测试稀疏化器
        for _ in range(iters_before_collapse):
            # 创建新的测试数据
            new_data = torch.randn(20, 20)
            # 向稀疏化器中添加数据并执行一步稀疏化操作
            test_sparsifier.add_data(name="test_data", data=new_data)
            test_sparsifier.step()
            # 获取稀疏化 mask
            mask = test_sparsifier.get_mask(name="test_data")
            # 将 mask 转换为 float 类型
            mask = mask.to(torch.float)
            # 断言稀疏化水平大于 0
            assert (1.0 - mask.mean().item()) > 0  # some sparsity achieved

    def check_step_2_of_4(self, norm_type):
        # 为测试目的覆盖默认配置
        default_config = {
            "sparsity_level": 1.0,
            "zeros_per_block": 2,
            "sparse_block_shape": (1, 4),
        }
        # 创建数据列表
        data_list = [("test_data", torch.randn(4, 4))]

        # 创建稀疏化器对象并执行一步稀疏化操作
        sparsifier = DataNormSparsifier(
            data_list=data_list, norm=norm_type, **default_config
        )
        sparsifier.step()

        # 验证每个数据在执行 step() 后的稀疏化结果
        for some_data in data_list:
            # 获取数据名称
            name, _ = some_data
            # 获取指定数据的稀疏化 mask
            mask = sparsifier.get_mask(name=name)
            # 将 mask 转换为 float 类型
            mask = mask.to(torch.float)
            # 断言稀疏化水平为 0.5，允许的误差为两位小数
            self.assertAlmostEqual(1.0 - mask.mean().item(), 0.5, places=2)
            # 遍历 mask 的每一行
            for row in mask:
                # 每四个元素为一个稀疏化块，验证前两个元素为 0，后两个元素不为 0
                for idx in range(0, len(row), 4):
                    block = row[idx : idx + 4]
                    block, _ = block.sort()
                    assert (block[:2] == 0).all()
                    assert (block[2:] != 0).all()

    def check_sparsity_level(
        self, data_list, data_with_config, defaults, norm_type="L1"
    ):
    ):
        # 不同的稀疏级别
        sparsity_levels = [-1.0, 0.0, 0.5, 1.0, 2.0]
        # 不同的稀疏块形状
        sparse_block_shapes = [(1, 1), (1, 4), (2, 2), (4, 1)]
        # 每个稀疏块中的零值数量
        zeros_per_blocks = [0, 1, 2, 3, 4]
        # 创建一个数据规范稀疏化器对象
        sparsifier = DataNormSparsifier(data_list=data_list, norm=norm_type)

        # 生成测试用例，包含不同的稀疏级别、稀疏块形状和零值数量组合
        testcases = itertools.tee(
            itertools.product(sparsity_levels, sparse_block_shapes, zeros_per_blocks)
        )

        # 断言数据配置列表不为空，并且第一个元素包含'name'和'data'字段
        assert (
            len(data_with_config) > 0
            and "name" in data_with_config[0]
            and "data" in data_with_config[0]
        )
        # 获取一些数据
        name, data = data_with_config[0]["name"], data_with_config[0]["data"]
        # 遍历测试用例
        for idx, (sl, sbs, zpb) in enumerate(testcases[0]):
            new_name = f"{name}_{idx}"
            # 如果零值数量大于稀疏块的乘积，则继续下一个循环
            if zpb > sbs[0] * sbs[1]:
                continue
            current_config = {
                "sparsity_level": sl,
                "sparse_block_shape": sbs,
                "zeros_per_block": zpb,
            }
            # 向稀疏化器中添加数据
            sparsifier.add_data(name=new_name, data=data, **current_config)
            # 如果零值数量大于稀疏块的乘积，则继续下一个循环
            if zpb > sbs[0] * sbs[1]:
                continue

        # 执行稀疏化器的步骤
        sparsifier.step()
        # 压缩稀疏化器的掩码
        sparsifier.squash_mask()
        # 再次遍历测试用例
        for idx, (sl, sbs, zpb) in enumerate(testcases[0]):
            new_name = f"{name}_{idx}"
            # 获取经过稀疏化处理的数据
            sparsified_data = sparsifier.get_data(name=new_name, original=False)
            # 计算稀疏掩码
            sparse_mask = (sparsified_data == 0).float()
            # 如果零值数量为0，则断言稀疏掩码的均值为0
            if zpb == 0:
                assert sparse_mask.mean() == 0
            else:
                # 计算在张量中个别零值的比例
                true_sl = min(max(sl, 0.0), 1.0)
                true_sl = true_sl * zpb / sbs[0] / sbs[1]
                # 断言稀疏掩码的均值等于计算得到的真实稀疏水平
                assert sparse_mask.mean() == true_sl
class TestBaseDataSparsifier(_BaseDataSparsiferTestCase):
    """To add unit tests to support new data types for the BaseDataSparsifier, create the following
        data_list: List of tuples of name, data to be added to the constructor
        defaults: default config for the above data in data_list
        data_with_config: list of dictionaries defining name, data and config (look test_tensors())

    Once the above is done, create an instance of TestBaseDataSparsifierType and call all the run_tests()
    """

    def test_tensors(self):
        # 定义几个随机生成的张量 tensor1, tensor2, tensor3
        tensor1, tensor2, tensor3 = (
            torch.randn(3, 3),
            torch.randn(4, 4),
            torch.randn(5, 5),
        )
        # 再定义两个随机张量 tensor4, tensor5
        tensor4, tensor5 = torch.randn(1, 1), torch.randn(4, 4)
        # 构建包含张量名和张量数据的列表 data_list
        data_list = [("tensor1", tensor1), ("tensor2", tensor2), ("tensor3", tensor3)]
        # 设置默认配置 defaults
        defaults = {"test": 3}

        # 构建包含张量名、张量数据和配置的字典列表 data_with_config
        data_with_config = [
            {"name": "tensor4", "data": tensor4, "config": {"test": 7}},
            {"name": "tensor5", "data": tensor5, "config": {"test": 8}},
        ]
        # 调用父类方法 run_all_checks，传入数据列表和配置参数
        self.run_all_checks(
            data_list=data_list, defaults=defaults, data_with_config=data_with_config
        )

    def test_nn_parameters(self):
        # 定义几个随机生成的神经网络参数 param1, param2, param3
        param1, param2, param3 = (
            nn.Parameter(torch.randn(3, 3)),
            nn.Parameter(torch.randn(4, 4)),
            nn.Parameter(torch.randn(5, 5)),
        )
        # 再定义两个神经网络参数 param4, param5
        param4, param5 = nn.Parameter(torch.randn(1, 1)), nn.Parameter(
            torch.randn(4, 4)
        )
        # 构建包含参数名和参数数据的列表 data_list
        data_list = [("param1", param1), ("param2", param2), ("param3", param3)]
        # 设置默认配置 defaults
        defaults = {"test": 3}

        # 构建包含参数名、参数数据和配置的字典列表 data_with_config
        data_with_config = [
            {"name": "param4", "data": param4, "config": {"test": 7}},
            {"name": "param5", "data": param5, "config": {"test": 8}},
        ]
        # 调用父类方法 run_all_checks，传入数据列表和配置参数
        self.run_all_checks(
            data_list=data_list, defaults=defaults, data_with_config=data_with_config
        )

    def test_nn_embeddings(self):
        # 定义几个嵌入层对象 emb1, emb2
        emb1, emb2 = nn.Embedding(
            10, 3
        ), nn.Embedding(20, 3)
        # 定义两个嵌入袋对象 emb1_bag, emb2_bag
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 3), nn.EmbeddingBag(20, 3)

        # 再定义一个嵌入层对象 emb3 和一个嵌入袋对象 emb3_bag
        emb3, emb3_bag = nn.Embedding(15, 3), nn.EmbeddingBag(20, 3)
        # 构建包含嵌入层和嵌入袋名称及对象的列表 data_list
        data_list = [
            ("emb1", emb1),
            ("emb1_bag", emb1_bag),
            ("emb2", emb2),
            ("emb2_bag", emb2_bag),
        ]
        # 设置默认配置 defaults
        defaults = {"test": 3}

        # 构建包含嵌入层和嵌入袋名称、对象及配置的字典列表 data_with_config
        data_with_config = [
            {"name": "emb3", "data": emb3, "config": {"test": 7}},
            {"name": "emb3_bag", "data": emb3_bag, "config": {"test": 8}},
        ]
        # 调用父类方法 run_all_checks，传入数据列表和配置参数
        self.run_all_checks(
            data_list=data_list, defaults=defaults, data_with_config=data_with_config
        )


class TestNormDataSparsifiers(_NormDataSparsifierTestCase):
    """To add unit tests to support new data types for the NormDataSparsifier, create the following
    data_list: List of tuples of name, data to be added to the constructor
    defaults: default config for the above data in data_list
    data_with_config: list of dictionaries defining name, data and config (look test_tensors())

    Once the above is done, create an instance of _NormDataSparsifierTestRunner and call run_tests()
    """

    # 定义测试方法 test_tensors
    def test_tensors(self):
        # 定义多个随机张量 tensor1 到 tensor5
        tensor1, tensor2, tensor3 = (
            torch.randn(1, 10),    # 生成形状为 (1, 10) 的随机张量 tensor1
            torch.randn(4, 4),     # 生成形状为 (4, 4) 的随机张量 tensor2
            torch.randn(1, 5),     # 生成形状为 (1, 5) 的随机张量 tensor3
        )
        tensor4, tensor5 = torch.randn(1, 2), torch.randn(4, 4)  # 生成形状分别为 (1, 2) 和 (4, 4) 的随机张量 tensor4 和 tensor5
        
        # 定义数据列表 data_list，包含张量名称和张量数据的元组
        data_list = [("tensor1", tensor1), ("tensor2", tensor2), ("tensor3", tensor3)]
        
        # 定义默认配置 defaults，包含稀疏度、稀疏块形状和每块零元素个数
        defaults = {
            "sparsity_level": 0.5,              # 稀疏度
            "sparse_block_shape": (1, 4),        # 稀疏块形状
            "zeros_per_block": 4,               # 每块零元素个数
        }

        # 定义数据与配置列表 data_with_config，每个元素是一个字典，包含张量名称、张量数据和配置字典
        data_with_config = [
            {
                "name": "tensor4",              # 张量名称
                "data": tensor4,                # 张量数据
                "config": {
                    "sparsity_level": 0.7,      # 稀疏度
                    "sparse_block_shape": (2, 3),  # 稀疏块形状
                    "zeros_per_block": 6,       # 每块零元素个数
                },
            },
            {
                "name": "tensor5",              # 张量名称
                "data": tensor5,                # 张量数据
                "config": {
                    "sparsity_level": 0.3,      # 稀疏度
                    "sparse_block_shape": (2, 3),  # 稀疏块形状
                    "zeros_per_block": 6,       # 每块零元素个数
                },
            },
        ]
        
        # 调用 self 对象的 run_all_checks 方法，分别传入 data_list、defaults、data_with_config 和 norm_type="L1"
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L1",
        )
        
        # 再次调用 self 对象的 run_all_checks 方法，传入相同的参数，但 norm_type 改为 "L2"
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L2",
        )
    # 定义一个测试神经网络参数的方法，这是一个单元测试方法
    def test_nn_parameters(self):
        # 初始化多个神经网络参数，每个参数都是一个 nn.Parameter 对象
        param1, param2, param3 = (
            nn.Parameter(torch.randn(1, 8)),  # 创建一个参数，大小为 1x8
            nn.Parameter(torch.randn(4, 4)),  # 创建一个参数，大小为 4x4
            nn.Parameter(torch.randn(5, 5)),  # 创建一个参数，大小为 5x5
        )
        # 初始化两个额外的神经网络参数，每个参数也是 nn.Parameter 对象
        param4, param5 = nn.Parameter(torch.randn(10, 10)), nn.Parameter(
            torch.randn(4, 4)
        )
        # 创建一个包含参数名称和对应 nn.Parameter 对象的列表
        data_list = [("param1", param1), ("param2", param2), ("param3", param3)]
        # 创建一个包含默认配置的字典
        defaults = {
            "sparsity_level": 0.5,            # 稀疏水平为 0.5
            "sparse_block_shape": (1, 4),     # 稀疏块形状为 (1, 4)
            "zeros_per_block": 4,              # 每个块内的零值个数为 4
        }

        # 创建一个包含参数名称、数据和配置的字典列表
        data_with_config = [
            {
                "name": "param4",               # 参数名称为 "param4"
                "data": param4,                 # 参数数据为 param4
                "config": {
                    "sparsity_level": 0.7,       # 稀疏水平为 0.7
                    "sparse_block_shape": (2, 3),# 稀疏块形状为 (2, 3)
                    "zeros_per_block": 6,        # 每个块内的零值个数为 6
                },
            },
            {
                "name": "param5",               # 参数名称为 "param5"
                "data": param5,                 # 参数数据为 param5
                "config": {
                    "sparsity_level": 0.3,       # 稀疏水平为 0.3
                    "sparse_block_shape": (2, 3),# 稀疏块形状为 (2, 3)
                    "zeros_per_block": 6,        # 每个块内的零值个数为 6
                },
            },
        ]
        
        # 运行自定义的检查方法，对神经网络参数进行各种检查
        self.run_all_checks(
            data_list=data_list,                # 使用上面定义的 data_list
            defaults=defaults,                  # 使用上面定义的 defaults
            data_with_config=data_with_config,  # 使用上面定义的 data_with_config
            norm_type="L1",                     # 指定归一化类型为 "L1"
        )
        # 再次运行自定义的检查方法，对神经网络参数进行另一种类型的检查
        self.run_all_checks(
            data_list=data_list,                # 使用上面定义的 data_list
            defaults=defaults,                  # 使用上面定义的 defaults
            data_with_config=data_with_config,  # 使用上面定义的 data_with_config
            norm_type="L2",                     # 指定归一化类型为 "L2"
        )
    # 定义一个测试方法，用于测试神经网络嵌入层的功能
    def test_nn_embeddings(self):
        (
            emb1,  # 创建第一个 Embedding 层，词汇表大小为10，向量维度为3
            emb2,
        ) = nn.Embedding(
            10, 3  # 创建第二个 Embedding 层，词汇表大小为20，向量维度为3
        ), nn.Embedding(20, 3)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 3), nn.EmbeddingBag(20, 3)  # 创建两个 EmbeddingBag 层

        emb3, emb3_bag = nn.Embedding(15, 3), nn.EmbeddingBag(20, 3)  # 创建第三个 Embedding 层和第二个 EmbeddingBag 层
        data_list = [  # 创建一个包含不同嵌入层的数据列表
            ("emb1", emb1),
            ("emb1_bag", emb1_bag),
            ("emb2", emb2),
            ("emb2_bag", emb2_bag),
        ]
        defaults = {  # 默认配置字典
            "sparsity_level": 0.5,  # 稀疏度
            "sparse_block_shape": (1, 4),  # 稀疏块形状
            "zeros_per_block": 4,  # 每个块中的零值数
        }

        data_with_config = [  # 包含配置信息的数据列表
            {
                "name": "emb3",  # 第三个嵌入层的名称
                "data": emb3,  # 第三个嵌入层的数据
                "config": {
                    "sparsity_level": 0.7,  # 第三个嵌入层的稀疏度
                    "sparse_block_shape": (2, 3),  # 第三个嵌入层的稀疏块形状
                    "zeros_per_block": 6,  # 第三个嵌入层每个块中的零值数
                },
            },
            {
                "name": "emb3_bag",  # 第二个 EmbeddingBag 层的名称
                "data": emb3_bag,  # 第二个 EmbeddingBag 层的数据
                "config": {
                    "sparsity_level": 0.3,  # 第二个 EmbeddingBag 层的稀疏度
                    "sparse_block_shape": (2, 3),  # 第二个 EmbeddingBag 层的稀疏块形状
                    "zeros_per_block": 6,  # 第二个 EmbeddingBag 层每个块中的零值数
                },
            },
        ]
        self.run_all_checks(  # 运行所有的检查方法，验证嵌入层的数据和配置
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L1",  # 使用 L1 范数进行归一化
        )

        self.run_all_checks(  # 再次运行所有的检查方法，验证嵌入层的数据和配置
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L2",  # 使用 L2 范数进行归一化
        )
class Model(nn.Module):
    # 定义一个继承自 nn.Module 的模型类
    def __init__(self):
        # 初始化函数，定义模型结构
        super().__init__()
        # 创建一个维度为 100x3 的嵌入层
        self.emb1 = nn.Embedding(100, 3)
        # 创建一个维度为 200x32 的嵌入袋层
        self.embbag1 = nn.EmbeddingBag(200, 32)
        # 创建一个包含两个嵌入层的顺序容器：维度为 150x3 和 100x3
        self.emb_seq = nn.Sequential(nn.Embedding(150, 3), nn.EmbeddingBag(100, 3))
        # 创建一个线性层，输入维度为 32，输出维度为 32
        self.linear1 = nn.Linear(32, 32)
        # 创建一个线性层，输入维度为 16，输出维度为 16
        self.linear2 = nn.Linear(16, 16)


class TestQuantizationUtils(TestCase):
    # 测试量化工具的测试用例类
    def test_ptq_sparsify_first(self):
        """The expectation is post_training_sparse_quantize function
        1. Takes in a model
        2. Sparsifies the embeddings
        3. Quantize the embeddings

        This unit test checks that
        1. Embeddings and EmbeddingBags are sparsified to the right sparsity levels
        2. Embeddings and EmbeddingBags are quantized
        3. Linear modules are not quantized
        """
        # 创建一个 Model 类的实例
        model = Model()

        # 设置稀疏化配置参数
        sparse_config = {"sparsity_level": 0.80, "sparse_block_shape": (1, 1)}
        # 选择要稀疏化的嵌入层和嵌入袋层
        select_embeddings = [model.embbag1, model.emb1]
        # 调用 post_training_sparse_quantize 函数对模型进行后训练稀疏量化
        post_training_sparse_quantize(
            model,
            data_sparsifier_class=DataNormSparsifier,
            sparsify_first=True,
            select_embeddings=select_embeddings,
            **sparse_config,
        )

        # 断言嵌入层和嵌入袋层已经被量化
        assert type(model.emb1) == torch.ao.nn.quantized.modules.embedding_ops.Embedding
        assert (
            type(model.embbag1)
            == torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag
        )
        # 断言嵌入层序列中的第一个是普通嵌入层
        assert type(model.emb_seq[0]) == nn.Embedding
        # 断言嵌入层序列中的第二个是嵌入袋层
        assert type(model.emb_seq[1]) == nn.EmbeddingBag
        # 断言线性层没有被量化
        assert type(model.linear1) == nn.Linear
        assert type(model.linear2) == nn.Linear

        # 对嵌入层进行反量化处理
        dequant_emb1 = torch.dequantize(model.emb1.weight())
        dequant_embbag1 = torch.dequantize(model.embbag1.weight())

        # 设置阈值
        threshold = 1e-2

        # 计算嵌入层的稀疏级别
        sl_emb1 = (torch.abs(dequant_emb1) < threshold).float().mean()
        sl_embbag1 = (torch.abs(dequant_embbag1) < threshold).float().mean()

        # 断言稀疏级别在预期范围内（+- 5%）
        assert abs(sl_emb1 - 0.80) <= 0.05
        assert abs(sl_embbag1 - 0.80) <= 0.05
    # 定义一个名为 test_ptq_quantize_first 的测试方法，用于验证 post_training_sparse_quantize 函数的行为
    """The expectation is post_training_sparse_quantize function
    1. Takes in a model
    2. Quantize the embeddings
    3. Sparsifies the embeddings

    This unit test checks that
    1. Embeddings and EmbeddingBags are sparsified to the right sparsity levels
    2. Embeddings and EmbeddingBags are quantized
    3. Linear modules are not quantized
    """
    # 创建一个 Model 实例
    model = Model()

    # 定义稀疏化配置参数
    sparse_config = {"sparsity_level": 0.8, "sparse_block_shape": (1, 1)}
    # 调用 post_training_sparse_quantize 函数，对模型进行稀疏化量化处理，不优先进行稀疏化
    post_training_sparse_quantize(
        model, DataNormSparsifier, sparsify_first=False, **sparse_config
    )

    # 断言各部分模型组件的类型是否正确进行了量化
    assert type(model.emb1) == torch.ao.nn.quantized.modules.embedding_ops.Embedding
    assert (
        type(model.embbag1)
        == torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag
    )
    assert type(
        model.emb_seq[0] == torch.ao.nn.quantized.modules.embedding_ops.Embedding
    )
    assert type(
        model.emb_seq[1] == torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag
    )
    assert type(model.linear1) == nn.Linear  # 线性模块未进行量化
    assert type(model.linear2) == nn.Linear  # 线性模块未进行量化

    # 对量化后的 embedding 权重进行反量化操作，计算稀疏级别
    dequant_emb1 = torch.dequantize(model.emb1.weight())
    dequant_embbag1 = torch.dequantize(model.embbag1.weight())
    dequant_emb_seq_0 = torch.dequantize(model.emb_seq[0].weight())
    dequant_emb_seq_1 = torch.dequantize(model.emb_seq[1].weight())

    # 设置阈值，用于确定稀疏级别的计算
    threshold = (
        1  # zero points seem to have higher magnitude with sparsity occuring after
    )

    # 计算各 embedding 的稀疏级别
    sl_emb1 = (torch.abs(dequant_emb1) < threshold).float().mean()
    sl_embbag1 = (torch.abs(dequant_embbag1) < threshold).float().mean()
    sl_emb_seq_0 = (torch.abs(dequant_emb_seq_0) < threshold).float().mean()
    sl_emb_seq_1 = (torch.abs(dequant_emb_seq_1) < threshold).float().mean()

    # 断言计算出的稀疏级别是否在预期范围内，给定了+-5%的允许误差
    assert abs(sl_emb1 - 0.80) <= 0.05  # +- 5% leeway
    assert abs(sl_embbag1 - 0.80) <= 0.05  # +- 5% leeway
    assert abs(sl_emb_seq_0 - 0.80) <= 0.05  # +- 5% leeway
    assert abs(sl_emb_seq_1 - 0.80) <= 0.05  # +- 5% leeway
```