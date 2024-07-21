# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\lightning\tests\test_callbacks.py`

```py
# mypy: allow-untyped-defs
# 引入需要的库和模块
from torch.ao.pruning._experimental.data_sparsifier.data_norm_sparsifier import DataNormSparsifier
from torch.ao.pruning._experimental.data_scheduler.base_data_scheduler import BaseDataScheduler
import torch
import torch.nn as nn
from typing import List
from torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks.data_sparsity import (
    PostTrainingDataSparsity,
    TrainingAwareDataSparsity
)
from torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks._data_sparstity_utils import _get_valid_name
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests
import importlib
import unittest
import warnings
import math
from torch.nn.utils.parametrize import is_parametrized


class DummyModel(nn.Module):
    def __init__(self, iC: int, oC: List[int]):
        super().__init__()
        # 初始化模型的线性层序列
        self.linears = nn.Sequential()
        i = iC
        # 根据输出通道数列表构建线性层和ReLU激活函数
        for idx, c in enumerate(oC):
            self.linears.append(nn.Linear(i, c, bias=False))
            if idx < len(oC) - 1:
                self.linears.append(nn.ReLU())
            i = c


def _make_lightning_module(iC: int, oC: List[int]):
    import pytorch_lightning as pl  # type: ignore[import]

    class DummyLightningModule(pl.LightningModule):
        def __init__(self, ic: int, oC: List[int]):
            super().__init__()
            # 创建包含线性层的模型
            self.model = DummyModel(iC, oC)

        def forward(self):
            pass

    return DummyLightningModule(iC, oC)


class StepSLScheduler(BaseDataScheduler):
    """The sparsity param of each data group is multiplied by gamma every step_size epochs.
    """
    def __init__(self, data_sparsifier, schedule_param='sparsity_level',
                 step_size=1, gamma=2, last_epoch=-1, verbose=False):
        # 初始化 StepSLScheduler 类的实例
        self.gamma = gamma
        self.step_size = step_size
        # 调用父类的初始化方法
        super().__init__(data_sparsifier, schedule_param, last_epoch, verbose)

    def get_schedule_param(self):
        if not self._get_sp_called_within_step:
            # 如果在步骤中没有调用 _get_sp_called_within_step，则发出警告
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        # 获取数据稀疏化器的数据组
        data_groups = self.data_sparsifier.data_groups
        # 如果是第一轮迭代或者当前轮次不是调度步长的整数倍，则返回原始的参数字典
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return {name: config[self.schedule_param] for name, config in data_groups.items()}

        # 否则，返回经过 gamma 放大后的参数字典
        return {name: config[self.schedule_param] * self.gamma for name, config in data_groups.items()}


class TestPostTrainingCallback(TestCase):
    # 测试后训练回调函数
    def _check_on_fit_end(self, pl_module, callback, sparsifier_args):
        """Makes sure that each component of is working as expected while calling the
        post-training callback.
        Specifically, check the following -
            1. sparsifier config is the same as input config
            2. data sparsifier is correctly attached to the model
            3. sparsity is achieved after .step()
            4. non-sparsified values are the same as original values
        """
        # 调用回调函数的 on_fit_end 方法，传入虚拟值 42 和 pl_module 实例
        callback.on_fit_end(42, pl_module)  # 42 is a dummy value

        # 检查稀疏化参数配置是否与输入的配置相同
        for key, value in sparsifier_args.items():
            assert callback.data_sparsifier.defaults[key] == value

        # 断言模型已正确连接到稀疏化对象
        for name, param in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            # 如果参数类型不在支持的类型列表中，则跳过检查
            if type(param) not in SUPPORTED_TYPES:
                assert valid_name not in callback.data_sparsifier.state
                assert valid_name not in callback.data_sparsifier.data_groups
                continue
            # 断言数据组在稀疏化对象的数据组列表中
            assert valid_name in callback.data_sparsifier.data_groups
            # 断言数据组在稀疏化对象的状态列表中
            assert valid_name in callback.data_sparsifier.state

            # 获取数据组的掩码
            mask = callback.data_sparsifier.get_mask(name=valid_name)

            # 断言已达到一定程度的稀疏性
            assert (1.0 - mask.float().mean()) > 0.0

            # 确保在 squash mask 后非零值的数据与原始值相等
            sparsified_data = callback.data_sparsifier.get_data(name=valid_name, return_original=False)
            assert torch.all(sparsified_data[sparsified_data != 0] == param[sparsified_data != 0])

    @unittest.skipIf(not importlib.util.find_spec("pytorch_lightning"), "No pytorch_lightning")
    def test_post_training_callback(self):
        # 定义稀疏化参数
        sparsifier_args = {
            'sparsity_level': 0.5,
            'sparse_block_shape': (1, 4),
            'zeros_per_block': 4
        }
        # 创建后训练数据稀疏性回调对象
        callback = PostTrainingDataSparsity(DataNormSparsifier, sparsifier_args)
        # 创建 PyTorch Lightning 模型
        pl_module = _make_lightning_module(100, [128, 256, 16])

        # 调用 _check_on_fit_end 方法进行测试
        self._check_on_fit_end(pl_module, callback, sparsifier_args)
class TestTrainingAwareCallback(TestCase):
    """Class to test in-training version of lightning callback
    Simulates model training and makes sure that each hook is doing what is expected
    """

    def _check_on_train_start(self, pl_module, callback, sparsifier_args, scheduler_args):
        """Makes sure that the data_sparsifier and data_scheduler objects are being created
        correctly.
        Basically, confirms that the input args and sparsifier/scheduler args are in-line.
        """

        # 调用 on_train_start 方法，模拟训练开始，传入的参数 42 是虚拟值
        callback.on_train_start(42, pl_module)

        # 断言确保 data_scheduler 和 data_sparsifier 对象已经被实例化
        assert callback.data_scheduler is not None and callback.data_sparsifier is not None

        # 验证 data_sparsifier 的参数是否正确
        for key, value in sparsifier_args.items():
            assert callback.data_sparsifier.defaults[key] == value

        # 验证 data_scheduler 的参数是否正确
        for key, value in scheduler_args.items():
            assert getattr(callback.data_scheduler, key) == value

    def _simulate_update_param_model(self, pl_module):
        """This function might not be needed as the model is being copied
        during train_epoch_end() but good to have if things change in the future
        """

        # 遍历模型的命名参数，模拟更新参数（这可能在未来的 train_epoch_end() 中用到）
        for _, param in pl_module.model.named_parameters():
            param.data = param + 1
    def _check_on_train_epoch_start(self, pl_module, callback):
        """Ensure sparsifier's state is correctly restored at the start of each training epoch.

        The state_dict() comparison ensures that the sparsifier's state is consistent across epochs.

        """
        # 调用回调函数的on_train_epoch_start方法，开始训练周期
        callback.on_train_epoch_start(42, pl_module)
        # 如果回调对象的data_sparsifier_state_dict为空，直接返回
        if callback.data_sparsifier_state_dict is None:
            return

        # 获取当前回调对象中data_sparsifier的状态字典
        data_sparsifier_state_dict = callback.data_sparsifier.state_dict()

        # 比较容器对象
        container_obj1 = data_sparsifier_state_dict['_container']
        container_obj2 = callback.data_sparsifier_state_dict['_container']
        assert len(container_obj1) == len(container_obj2)
        for key, value in container_obj2.items():
            assert key in container_obj1
            assert torch.all(value == container_obj1[key])

        # 比较状态对象
        state_obj1 = data_sparsifier_state_dict['state']
        state_obj2 = callback.data_sparsifier_state_dict['state']
        assert len(state_obj1) == len(state_obj2)
        for key, value in state_obj2.items():
            assert key in state_obj1
            assert 'mask' in value and 'mask' in state_obj1[key]
            assert torch.all(value['mask'] == state_obj1[key]['mask'])

        # 比较data_groups字典
        data_grp1 = data_sparsifier_state_dict['data_groups']
        data_grp2 = callback.data_sparsifier_state_dict['data_groups']
        assert len(data_grp1) == len(data_grp2)
        for key, value in data_grp2.items():
            assert key in data_grp1
            assert value == data_grp1[key]
    def _check_on_train_epoch_end(self, pl_module, callback):
        """Checks the following -
        1. sparsity is correctly being achieved after .step()
        2. scheduler and data_sparsifier sparsity levels are in-line
        """
        # 调用回调函数的 on_train_epoch_end 方法，传入 epoch 数和模块对象
        callback.on_train_epoch_end(42, pl_module)
        # 获取回调对象中的 data_scheduler
        data_scheduler = callback.data_scheduler
        # 获取 data_scheduler 中的 base_param
        base_sl = data_scheduler.base_param

        # 遍历模型的命名参数
        for name, _ in pl_module.model.named_parameters():
            # 获取有效的参数名
            valid_name = _get_valid_name(name)
            # 根据有效参数名从 data_sparsifier 中获取掩码
            mask = callback.data_sparsifier.get_mask(name=valid_name)

            # 检查稀疏水平是否达到要求
            assert (1.0 - mask.float().mean()) > 0  # 某种稀疏水平已达到

            # 获取 data_scheduler 中最后的参数和最后的 epoch
            last_sl = data_scheduler.get_last_param()
            last_epoch = data_scheduler.last_epoch

            # 检查调度器的稀疏水平
            log_last_sl = math.log(last_sl[valid_name])
            log_actual_sl = math.log(base_sl[valid_name] * (data_scheduler.gamma ** last_epoch))
            assert log_last_sl == log_actual_sl

    def _check_on_train_end(self, pl_module, callback):
        """Confirms that the mask is squashed after the training ends
        This is achieved by making sure that each parameter in the internal container
        are not parametrized.
        """
        # 调用回调函数的 on_train_end 方法，传入 epoch 数和模块对象
        callback.on_train_end(42, pl_module)

        # 检查所有参数的掩码是否已经被压缩
        for name, _ in pl_module.model.named_parameters():
            # 获取有效的参数名
            valid_name = _get_valid_name(name)
            # 确保 data_sparsifier 的内部容器中的每个参数都不再被参数化
            assert not is_parametrized(callback.data_sparsifier._continer, valid_name)

    @unittest.skipIf(not importlib.util.find_spec("pytorch_lightning"), "No pytorch_lightning")
    def test_train_aware_callback(self):
        sparsifier_args = {
            'sparsity_level': 0.5,
            'sparse_block_shape': (1, 4),
            'zeros_per_block': 4
        }
        scheduler_args = {
            'gamma': 2,
            'step_size': 1
        }

        # 创建 TrainingAwareDataSparsity 回调对象，传入参数
        callback = TrainingAwareDataSparsity(
            data_sparsifier_class=DataNormSparsifier,
            data_sparsifier_args=sparsifier_args,
            data_scheduler_class=StepSLScheduler,
            data_scheduler_args=scheduler_args
        )

        # 创建一个 PyTorch Lightning 模块
        pl_module = _make_lightning_module(100, [128, 256, 16])

        # 模拟训练过程并检查所有步骤
        self._check_on_train_start(pl_module, callback, sparsifier_args, scheduler_args)

        num_epochs = 5
        for _ in range(0, num_epochs):
            self._check_on_train_epoch_start(pl_module, callback)
            self._simulate_update_param_model(pl_module)
            self._check_on_train_epoch_end(pl_module, callback)
# 如果这个脚本被直接运行（而不是作为模块被导入），那么执行下面的代码块
if __name__ == "__main__":
    # 调用名为 run_tests() 的函数来运行测试
    run_tests()
```