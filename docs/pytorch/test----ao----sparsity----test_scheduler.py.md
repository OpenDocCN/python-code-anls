# `.\pytorch\test\ao\sparsity\test_scheduler.py`

```py
# Owner(s): ["module: unknown"]

# 导入警告模块
import warnings

# 导入 PyTorch 中的神经网络模块
from torch import nn

# 导入稀疏性调度相关模块
from torch.ao.pruning import BaseScheduler, CubicSL, LambdaSL, WeightNormSparsifier

# 导入测试工具类
from torch.testing._internal.common_utils import TestCase


# 自定义的调度器类，继承自 BaseScheduler
class ImplementedScheduler(BaseScheduler):
    # 获取稀疏性水平列表的方法
    def get_sl(self):
        # 如果上一个周期的 epoch 大于 0，则返回每个组的稀疏性水平的一半
        if self.last_epoch > 0:
            return [group["sparsity_level"] * 0.5 for group in self.sparsifier.groups]
        else:
            # 否则返回基础稀疏性水平列表
            return list(self.base_sl)


# 测试调度器的测试类
class TestScheduler(TestCase):
    # 测试构造函数
    def test_constructor(self):
        # 创建一个简单的神经网络模型
        model = nn.Sequential(nn.Linear(16, 16))
        # 创建一个权重归一化稀疏化器
        sparsifier = WeightNormSparsifier()
        # 准备稀疏化器，传入模型和配置为 None
        sparsifier.prepare(model, config=None)
        # 创建一个自定义调度器对象，传入稀疏化器
        scheduler = ImplementedScheduler(sparsifier)

        # 断言调度器的稀疏化器与初始传入的稀疏化器相同
        assert scheduler.sparsifier is sparsifier
        # 断言调度器的步数计数为 1
        assert scheduler._step_count == 1
        # 断言调度器的基础稀疏性水平与稀疏化器第一个组的稀疏性水平相同
        assert scheduler.base_sl == [sparsifier.groups[0]["sparsity_level"]]

    # 测试步骤顺序的测试方法
    def test_order_of_steps(self):
        """检查调度器在调用稀疏化器步骤之前是否会发出警告"""

        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        # 确保在未调用稀疏化器步骤时会发出警告
        with self.assertWarns(UserWarning):
            scheduler.step()

        # 正确的顺序不会有警告
        # 注意：如果存在其他警告，这将会触发。
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # 确保没有与 base_scheduler 相关的警告
            for warning in w:
                fname = warning.filename
                fname = "/".join(fname.split("/")[-5:])
                assert fname != "torch/ao/sparsity/scheduler/base_scheduler.py"

    # 测试步骤方法的测试方法
    def test_step(self):
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.groups[0]["sparsity_level"] == 0.5
        scheduler = ImplementedScheduler(sparsifier)
        assert sparsifier.groups[0]["sparsity_level"] == 0.5

        # 执行稀疏化器步骤和调度器步骤，然后断言稀疏化器组的稀疏性水平变为 0.25
        sparsifier.step()
        scheduler.step()
        assert sparsifier.groups[0]["sparsity_level"] == 0.25

    # 测试 Lambda 调度器的测试方法
    def test_lambda_scheduler(self):
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.groups[0]["sparsity_level"] == 0.5
        # 使用 LambdaSL 创建一个调度器对象，传入稀疏化器和 lambda 函数
        scheduler = LambdaSL(sparsifier, lambda epoch: epoch * 10)
        # 断言稀疏化器第一个组的稀疏性水平为 0.0（第 0 个 epoch）
        assert sparsifier.groups[0]["sparsity_level"] == 0.0
        # 调度器执行一步
        scheduler.step()
        # 断言稀疏化器第一个组的稀疏性水平为 5.0（第 1 个 epoch）
        assert sparsifier.groups[0]["sparsity_level"] == 5.0


class TestCubicScheduler(TestCase):
    def setUp(self):
        # 设置测试的初始配置，包括稀疏模型的配置列表和排序后的稀疏级别列表
        self.model_sparse_config = [
            {"tensor_fqn": "0.weight", "sparsity_level": 0.8},
            {"tensor_fqn": "2.weight", "sparsity_level": 0.4},
        ]
        self.sorted_sparse_levels = [
            conf["sparsity_level"] for conf in self.model_sparse_config
        ]
        self.initial_sparsity = 0.1  # 初始稀疏度
        self.initial_step = 3  # 初始步数

    def _make_model(self, **kwargs):
        # 创建一个简单的神经网络模型
        model = nn.Sequential(
            nn.Linear(13, 17),
            nn.Dropout(0.5),
            nn.Linear(17, 3),
        )
        return model

    def _make_scheduler(self, model, **kwargs):
        # 创建一个调度器对象，并准备稀疏化模型
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=self.model_sparse_config)

        # 设置调度器的参数，包括初始稀疏度和初始步数
        scheduler_args = {
            "init_sl": self.initial_sparsity,
            "init_t": self.initial_step,
        }
        scheduler_args.update(kwargs)

        # 创建一个 CubicSL 调度器对象
        scheduler = CubicSL(sparsifier, **scheduler_args)
        return sparsifier, scheduler

    @staticmethod
    def _get_sparsity_levels(sparsifier, precision=32):
        r"""获取稀疏化器中当前的稀疏水平列表。"""
        return [
            round(group["sparsity_level"], precision) for group in sparsifier.groups
        ]

    def test_constructor(self):
        # 测试构造函数的正确性
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=True)
        # 断言：确保调度器正确地附加到稀疏化器上
        self.assertIs(
            scheduler.sparsifier, sparsifier, msg="Sparsifier is not properly attached"
        )
        # 断言：确保调度器的初始步数为1
        self.assertEqual(
            scheduler._step_count,
            1,
            msg="Scheduler is initialized with incorrect step count",
        )
        # 断言：确保调度器正确存储了目标稀疏水平列表
        self.assertEqual(
            scheduler.base_sl,
            self.sorted_sparse_levels,
            msg="Scheduler did not store the target sparsity levels correctly",
        )

        # 断言：在 t_0 之前的值为 0.0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(0.0),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler",
        )

        # 断言：在 t_0 之前的值为 s_0
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=False)
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler",
        )
    # 定义一个测试方法 test_step，用于测试调度器的步骤功能
    def test_step(self):
        # 创建模型对象
        model = self._make_model()
        # 创建稀疏化器和调度器对象，设置初始步数为3，步长为2，总步数为5
        sparsifier, scheduler = self._make_scheduler(
            model=model, initially_zero=True, init_t=3, delta_t=2, total_t=5
        )

        # 调度器执行一步
        scheduler.step()
        # 断言调度器的步数为3，验证步数是否正确增加
        self.assertEqual(
            scheduler._step_count,
            3,
            msg="Scheduler step_count is expected to increment",
        )

        # 断言在 t_0 之前的值应为0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(0.0),
            msg="Scheduler step updating the sparsity level before t_0",
        )

        # 调度器再执行一步，步数为3时稀疏度应重置为初始稀疏度
        scheduler.step()  # Step = 3  =>  sparsity = initial_sparsity
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset to initial sparsity at the first step",
        )

        # 调度器再执行一步，步数为4时稀疏度应设置为[0.3, 0.2]
        scheduler.step()  # Step = 4  =>  sparsity ~ [0.3, 0.2]
        self.assertEqual(
            self._get_sparsity_levels(sparsifier, 1),
            [0.3, 0.2],
            msg="Sparsity level is not set correctly after the first step",
        )

        # 计算当前步数到目标步数之间还需要多少步
        current_step = scheduler._step_count - scheduler.init_t[0] - 1
        more_steps_needed = scheduler.delta_t[0] * scheduler.total_t[0] - current_step
        # 循环执行剩余步数，直到达到目标稀疏度
        for _ in range(more_steps_needed):  # More steps needed to final sparsity level
            scheduler.step()
        # 断言稀疏度是否达到预期的排序后的稀疏度水平
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            self.sorted_sparse_levels,
            msg="Sparsity level is not reaching the target level afer delta_t * n steps ",
        )
```