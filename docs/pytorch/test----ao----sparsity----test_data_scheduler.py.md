# `.\pytorch\test\ao\sparsity\test_data_scheduler.py`

```
# Owner(s): ["module: unknown"]

# 导入必要的库和模块
import copy  # 导入复制模块
import logging  # 导入日志模块
import warnings  # 导入警告模块
from typing import Tuple  # 导入类型提示中的元组类型

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch中的神经网络模块
from torch.ao.pruning._experimental.data_scheduler import BaseDataScheduler  # 导入基础数据调度器
from torch.ao.pruning._experimental.data_sparsifier import DataNormSparsifier  # 导入数据规范化稀疏化器
from torch.testing._internal.common_utils import TestCase  # 导入PyTorch内部测试工具中的测试用例类

# 配置日志记录格式和级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class ImplementedDataScheduler(BaseDataScheduler):
    def __init__(self, sparsifier, sparsifier_hyperparam, last_epoch=-1, verbose=False):
        super().__init__(sparsifier, sparsifier_hyperparam, last_epoch, verbose)

    def get_schedule_param(self):
        # 如果最后一个epoch大于0，则返回每个数据组的稀疏程度的一半
        if self.last_epoch > 0:
            return {
                name: config["sparsity_level"] * 0.5
                for name, config in self.data_sparsifier.data_groups.items()
            }
        else:
            return self.base_param  # 否则返回基础参数


class TestBaseDataScheduler(TestCase):
    def _get_data(self):
        # 定义一些测试数据：tensor1, param1, emb1
        tensor1, param1, emb1 = (
            torch.randn(5, 5),
            nn.Parameter(torch.randn(10, 10)),
            nn.Embedding(50, 5),
        )
        # 将数据打包成列表，每个元素包含名称和对应的数据
        data_list = [("tensor1", tensor1), ("param1", param1), ("emb1", emb1)]
        # 默认参数字典
        defaults = {
            "sparsity_level": 0.7,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 2,
        }
        # 配置了特定参数的数据列表
        data_with_config = [
            {
                "name": "tensor2",
                "data": torch.randn(4, 4),
                "config": {"sparsity_level": 0.3},
            }
        ]
        return data_list, data_with_config, defaults

    def _get_sparsifier(self, data_list, data_with_config, defaults):
        # 创建数据规范化稀疏化器对象，并使用默认参数初始化
        sparsifier = DataNormSparsifier(data_list, **defaults)
        # 为稀疏化器对象添加具有特定配置的数据
        for data_config_dict in data_with_config:
            name, data, config = (
                data_config_dict["name"],
                data_config_dict["data"],
                data_config_dict["config"],
            )
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def _get_scheduler(self, sparsifier, schedule_param):
        # 创建实现的数据调度器对象，使用给定的稀疏化器和调度参数
        scheduler = ImplementedDataScheduler(sparsifier, schedule_param)
        return scheduler

    def _get_schedule_param(self):
        return "sparsity_level"  # 返回调度参数的名称

    def _get_name_data_config(self, some_data, defaults):
        config = copy.deepcopy(defaults)
        # 深拷贝默认配置
        if isinstance(some_data, Tuple):
            # 如果some_data是元组，则处理data_list
            name, data = some_data
        else:
            # 如果some_data是字典，则处理data_with_config
            name, data, new_config = (
                some_data["name"],
                some_data["data"],
                some_data["config"],
            )
            config.update(new_config)  # 更新配置
        return name, data, config
    # 测试构造函数，验证调度器步骤在稀疏化步骤之前被调用时是否会抛出警告
    def test_constructor(self):
        # 获取测试数据列表、带配置的数据、默认参数
        data_list, data_with_config, defaults = self._get_data()
        # 获取稀疏化器对象
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        # 获取调度参数
        schedule_param = self._get_schedule_param()
        # 获取调度器对象
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        # 断言调度器的数据稀疏化器与预期一致
        assert scheduler.data_sparsifier == sparsifier
        # 断言调度器的步数为1
        assert scheduler._step_count == 1

        # 遍历数据组，验证调度器基本参数与配置中的调度参数匹配
        for name, config in sparsifier.data_groups.items():
            assert scheduler.base_param[name] == config.get(schedule_param, None)

    # 测试步骤的顺序
    def test_order_of_steps(self):
        # 获取测试数据列表、带配置的数据、默认参数
        data_list, data_with_config, defaults = self._get_data()
        # 获取稀疏化器对象
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        # 获取调度参数
        schedule_param = self._get_schedule_param()
        # 获取调度器对象
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        # 断言未调用稀疏化器步骤时会抛出警告
        with self.assertWarns(UserWarning):
            scheduler.step()

        # 验证正确的步骤顺序没有警告
        # 注意：如果有其他警告存在，这里会触发
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # 确保没有与 base_data_scheduler 相关的警告
            for warning in w:
                fname = warning.filename
                fname = "/".join(fname.split("/")[-5:])
                assert (
                    fname
                    != "torch/ao/sparsity/experimental/scheduler/data_scheduler/base_data_scheduler.py"
                )

    # 测试单步调度器
    def test_step(self):
        # 获取测试数据列表、带配置的数据、默认参数
        data_list, data_with_config, defaults = self._get_data()
        # 获取稀疏化器对象
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        # 获取调度参数
        schedule_param = self._get_schedule_param()
        # 获取调度器对象
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        # 获取所有数据
        all_data = data_list + data_with_config

        # 遍历所有数据，验证稀疏化器数据组中的调度参数与配置中的一致性
        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults)
            assert (
                sparsifier.data_groups[name][schedule_param] == config[schedule_param]
            )

        # 执行稀疏化器的步骤
        sparsifier.step()
        # 执行调度器的步骤
        scheduler.step()

        # 再次遍历所有数据，验证稀疏化器数据组中的调度参数是否减半
        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults)
            assert (
                sparsifier.data_groups[name][schedule_param]
                == config[schedule_param] * 0.5
            )

        # 验证步数计数
        step_cnt = 5
        for _ in range(0, step_cnt):
            sparsifier.step()
            scheduler.step()

        # 断言调度器的步数是否为 step_cnt + 2
        assert (
            scheduler._step_count == step_cnt + 2
        )  # step_cnt + step above + 1 step in constructor
    # 定义一个测试函数，用于测试状态字典的加载和保存
    def test_state_dict(self):
        # 调用辅助函数获取测试数据
        data_list, data_with_config, defaults = self._get_data()
        # 调用辅助函数创建一个稀疏化器对象
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        # 调用辅助函数获取调度参数
        schedule_param = self._get_schedule_param()
        # 调用辅助函数创建第一个调度器对象
        scheduler1 = self._get_scheduler(sparsifier, schedule_param)

        # 调用第一个调度器对象的步进方法
        sparsifier.step()
        scheduler1.step()

        # 使用相同的稀疏化器和调度参数创建第二个调度器对象
        scheduler2 = self._get_scheduler(sparsifier, schedule_param)
        # 将所有数据合并为一个列表
        all_data = data_list + data_with_config
        # 遍历所有数据
        for some_data in all_data:
            # 调用辅助函数获取数据的名称、数据和配置
            name, _, _ = self._get_name_data_config(some_data, defaults)
            # 断言两个调度器对象的基础参数不相等
            assert scheduler1.base_param[name] != scheduler2.base_param[name]
            # 断言第一个调度器对象的最后参数等于第二个调度器对象的基础参数
            assert scheduler1._last_param[name] == scheduler2.base_param[name]

        # 获取第一个调度器对象的状态字典
        scheduler1_state = scheduler1.state_dict()
        # 加载第一个调度器对象的状态字典到第二个调度器对象
        scheduler2.load_state_dict(scheduler1_state)

        # 再次遍历所有数据
        for some_data in all_data:
            # 调用辅助函数获取数据的名称、数据和配置
            name, _, _ = self._get_name_data_config(some_data, defaults)
            # 断言两个调度器对象的基础参数相等
            assert scheduler1.base_param[name] == scheduler2.base_param[name]
            # 断言两个调度器对象的最后参数相等
            assert scheduler1._last_param[name] == scheduler2._last_param[name]
```