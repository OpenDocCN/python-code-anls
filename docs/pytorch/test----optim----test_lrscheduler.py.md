# `.\pytorch\test\optim\test_lrscheduler.py`

```
# Owner(s): ["module: optimizer", "module: LrScheduler" ]

import copy  # 导入用于复制对象的标准库
import math  # 导入数学计算函数的标准库
import pickle  # 导入用于序列化和反序列化 Python 对象的标准库
import tempfile  # 导入用于创建临时文件和目录的标准库
import types  # 导入支持动态类型创建和操作的标准库
import warnings  # 导入用于处理警告的标准库
from functools import partial  # 导入用于创建偏函数的标准库中的partial函数

import torch  # 导入 PyTorch 深度学习框架
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口模块
from torch.nn import Parameter  # 导入 PyTorch 中的参数类
from torch.optim import Adam, Rprop, SGD  # 导入 PyTorch 中的优化器类：Adam, Rprop, SGD
from torch.optim.lr_scheduler import (  # 导入 PyTorch 中的学习率调度器类
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    EPOCH_DEPRECATION_WARNING,  # 警告信息：学习率调度器中的epoch参数已弃用
    ExponentialLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.optim.swa_utils import SWALR  # 导入 PyTorch 中的 SWA（Stochastic Weight Averaging）学习率类
from torch.testing._internal.common_utils import (  # 导入 PyTorch 内部测试工具类和函数
    instantiate_parametrized_tests,
    load_tests,
    parametrize,
    skipIfTorchDynamo,
    TestCase,
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # 设置 load_tests 函数来自 common_utils，用于在 sandcastle 上进行测试分片，忽略 flake 警告


class TestLRScheduler(TestCase):
    class SchedulerTestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 1, 1)  # 定义第一个卷积层
            self.conv2 = torch.nn.Conv2d(1, 1, 1)  # 定义第二个卷积层

        def forward(self, x):
            return self.conv2(F.relu(self.conv1(x)))  # 网络的前向传播

    class LambdaLRTestObject:
        def __init__(self, value):
            self.value = value

        def __call__(self, epoch):
            return self.value * epoch

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.__dict__ == other.__dict__
            else:
                return False

    exact_dtype = True  # 设置精确的数据类型

    def setUp(self):
        super().setUp()  # 调用父类的setUp方法
        self.net = self.SchedulerTestNet()  # 创建测试用的网络模型
        self.opt = SGD(  # 创建随机梯度下降优化器对象
            [
                {"params": self.net.conv1.parameters()},  # 添加第一个卷积层的参数到优化器
                {"params": self.net.conv2.parameters(), "lr": 0.5},  # 添加第二个卷积层的参数到优化器，设置学习率为0.5
            ],
            lr=0.05,  # 设置全局学习率为0.05
        )

    def _check_warning_is_epoch_deprecation_warning(self, w, *, num_warnings: int = 1):
        """This function swallows the epoch deprecation warning which is produced when we
        call `scheduler.step(epoch)` with some not `None` value of `epoch`.
        this is deprecated, and this function will need to be removed/updated when
        the schedulers no longer accept the parameter at all.
        """
        self.assertEqual(len(w), num_warnings)  # 断言警告数量与预期数量相等
        for warning in w:
            self.assertEqual(len(warning.message.args), 1)  # 断言警告消息参数长度为1
            self.assertEqual(warning.message.args[0], EPOCH_DEPRECATION_WARNING)  # 断言警告消息内容为学习率调度器epoch参数的弃用警告
    def test_error_when_getlr_has_epoch(self):
        # 定义一个测试方法，用于测试当 getlr 包含 epoch 时是否会出错
        class MultiStepLR(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, gamma, milestones, last_epoch=-1):
                # 初始化学习率列表为每个参数组的初始学习率
                self.init_lr = [group["lr"] for group in optimizer.param_groups]
                self.gamma = gamma  # 设定衰减因子
                self.milestones = milestones  # 里程碑列表
                super().__init__(optimizer, last_epoch)  # 调用父类初始化方法

            def get_lr(self, step):
                # 获取当前步数对应的全局步数
                global_step = self.last_epoch
                # 计算衰减幂次
                gamma_power = (
                    [0]
                    + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m]
                )[-1]
                # 返回每个参数组的衰减后的学习率列表
                return [
                    init_lr * (self.gamma**gamma_power) for init_lr in self.init_lr
                ]

        optimizer = SGD([torch.rand(1)], lr=1)

        # 测试是否会抛出 TypeError 异常
        with self.assertRaises(TypeError):
            scheduler = MultiStepLR(optimizer, gamma=1, milestones=[10, 20])

    @skipIfTorchDynamo(
        "Torchdynamo keeps references to optim in the guards and the stack of the graph break frames"
    )
    def test_no_cyclic_references(self):
        # 导入 gc 模块，用于垃圾回收操作
        import gc

        param = Parameter(torch.empty(10))
        optim = SGD([param], lr=0.5)
        scheduler = LambdaLR(optim, lambda epoch: 1.0)
        del scheduler  # 删除 scheduler 对象

        # 断言优化器对象不包含循环引用
        self.assertTrue(
            len(gc.get_referrers(optim)) == 0,
            "Optimizer should contain no cyclic references",
        )

        gc.collect()  # 执行垃圾回收
        del optim  # 删除优化器对象
        self.assertEqual(
            gc.collect(), 0, msg="Optimizer should be garbage-collected on __del__"
        )

    @skipIfTorchDynamo(
        "Torchdynamo keeps references to optim in the guards and the stack of the graph break frames"
    )
    def test_no_cyclic_references_in_step(self):
        # 导入 gc 和 weakref 模块
        import gc
        import weakref

        def run():
            param = torch.empty(10, requires_grad=True)
            optim = SGD(params=[param], lr=0.5)
            scheduler = LambdaLR(optim, lambda epoch: 1.0)
            param.sum().backward()  # 反向传播
            optim.step()  # 执行优化器步骤
            scheduler.step()  # 执行学习率调度器步骤

            return weakref.ref(scheduler)  # 返回 scheduler 的弱引用

        # 禁用垃圾回收，以确保 scheduler 中没有循环引用
        gc.disable()
        ref = run()

        assert ref() is None  # 断言 scheduler 已被垃圾回收
        gc.enable()  # 恢复垃圾回收

    def test_old_pattern_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 允许记录任何警告
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()  # 执行学习率调度器步骤
                self.opt.step()  # 执行优化器步骤

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)
    # 定义一个测试方法，测试在带参数的情况下是否会收到旧模式警告
    def test_old_pattern_warning_with_arg(self):
        epochs = 35  # 设定训练周期数为35
        # 使用警告捕获机制记录所有警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 设置为始终引发警告
            # 创建 StepLR 调度器对象，设置学习率衰减参数
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            # 断言没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 定义一个模拟旧模式调用的函数
        def old_pattern2():
            # 执行设定的训练周期数次调度器步进和优化器步进操作
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        # 断言会触发 UserWarning 并且警告信息包含指定的文本
        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    # 定义一个测试方法，测试在恢复模式下是否会收到旧模式警告
    def test_old_pattern_warning_resuming(self):
        epochs = 35  # 设定训练周期数为35
        # 对优化器参数组进行设置，设置初始学习率
        for i, group in enumerate(self.opt.param_groups):
            group["initial_lr"] = 0.01

        # 使用警告捕获机制记录所有警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 设置为始终引发警告
            # 创建 StepLR 调度器对象，设置学习率衰减参数和上次周期数
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            # 断言没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 定义一个模拟旧模式调用的函数
        def old_pattern():
            # 执行设定的训练周期数次调度器步进和优化器步进操作
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        # 断言会触发 UserWarning 并且警告信息包含指定的文本
        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    # 定义一个测试方法，测试在恢复模式下是否会收到旧模式警告（带参数版本）
    def test_old_pattern_warning_resuming_with_arg(self):
        epochs = 35  # 设定训练周期数为35
        # 对优化器参数组进行设置，设置初始学习率
        for i, group in enumerate(self.opt.param_groups):
            group["initial_lr"] = 0.01

        # 使用警告捕获机制记录所有警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 设置为始终引发警告
            # 创建 StepLR 调度器对象，设置学习率衰减参数和上次周期数
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            # 断言没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 定义一个模拟旧模式调用的函数
        def old_pattern2():
            # 执行设定的训练周期数次调度器步进和优化器步进操作
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        # 断言会触发 UserWarning 并且警告信息包含指定的文本
        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    # 定义一个测试方法，测试在重写优化器步进方法的情况下是否会收到旧模式警告
    def test_old_pattern_warning_with_overridden_optim_step(self):
        epochs = 35  # 设定训练周期数为35
        # 对优化器参数组进行设置，设置初始学习率
        for i, group in enumerate(self.opt.param_groups):
            group["initial_lr"] = 0.01

        # 使用警告捕获机制记录所有警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 设置为始终引发警告
            # 创建 StepLR 调度器对象，设置学习率衰减参数和上次周期数
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            # 断言没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 模拟一个情况，重写了优化器的 step 方法
        import types
        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        # 定义一个模拟旧模式调用的函数
        def old_pattern2():
            # 执行设定的训练周期数次调度器步进和优化器步进操作
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        # 断言会触发 UserWarning 并且警告信息包含指定的文本
        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)
    # 定义一个测试方法，用于验证新模式下不会触发警告
    def test_new_pattern_no_warning(self):
        # 设定训练周期数为35
        epochs = 35
        # 使用 warnings.catch_warnings 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 允许记录所有警告
            # 创建 StepLR 调度器对象，设定 gamma 为 0.1，步长为 3
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            # 断言当前没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 再次使用 warnings.catch_warnings 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 允许记录所有警告
            # 执行多个周期的优化步骤和调度步骤
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            # 断言当前没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    # 定义一个测试方法，用于验证带参数的新模式下不会触发警告
    def test_new_pattern_no_warning_with_arg(self):
        # 设定训练周期数为35
        epochs = 35
        # 使用 warnings.catch_warnings 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 允许记录所有警告
            # 创建 StepLR 调度器对象，设定 gamma 为 0.1，步长为 3
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            # 断言当前没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 再次使用 warnings.catch_warnings 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 允许记录所有警告
            # 执行多个周期的优化步骤和调度步骤
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            # 断言当前没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    # 定义一个测试方法，用于验证重写 optimizer.step 后的新模式是否触发警告
    def test_new_pattern_no_warning_with_overridden_optim_step(self):
        # 设定训练周期数为35
        epochs = 35
        # 使用 warnings.catch_warnings 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 允许记录所有警告
            # 创建 StepLR 调度器对象，设定 gamma 为 0.1，步长为 3
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            # 断言当前没有任何警告被触发
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # 模拟 optimizer.step 被重写的使用情况
        import types
        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        # 定义一个新的模式函数
        def new_pattern():
            # 执行多个周期的优化步骤和调度步骤
            for e in range(epochs):
                self.opt.step()
                scheduler.step()

        # 断言新模式函数会触发 UserWarning 警告，且警告信息包含指定内容
        self.assertWarnsRegex(
            UserWarning, r"`optimizer.step\(\)` has been overridden", new_pattern
        )

    # 定义一个内部方法，用于验证在恒定周期内学习率保持不变的情况
    def _test_lr_is_constant_for_constant_epoch(self, scheduler):
        # 初始化空列表
        l = []
        
        # 执行10次循环
        for _ in range(10):
            # 执行一次优化步骤
            scheduler.optimizer.step()
            # 使用 warnings.catch_warnings 捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                # 执行调度步骤，步长为2
                scheduler.step(2)
                # 检查警告是否为“epoch deprecation”警告
                self._check_warning_is_epoch_deprecation_warning(w)

            # 将当前学习率添加到列表中
            l.append(self.opt.param_groups[0]["lr"])
        
        # 断言列表中的最小值等于最大值，即学习率保持不变
        self.assertEqual(min(l), max(l))

    # 定义一个测试方法，用于验证 StepLR 调度器在恒定周期内学习率保持不变的情况
    def test_step_lr_is_constant_for_constant_epoch(self):
        # 创建 StepLR 调度器对象，步长为2
        scheduler = StepLR(self.opt, 2)
        # 调用内部方法验证学习率是否保持恒定
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    # 定义一个测试方法，用于验证 ExponentialLR 调度器在恒定周期内学习率保持不变的情况
    def test_exponential_lr_is_constant_for_constant_epoch(self):
        # 创建 ExponentialLR 调度器对象，设定 gamma 为 0.9
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        # 调用内部方法验证学习率是否保持恒定
        self._test_lr_is_constant_for_constant_epoch(scheduler)
    # 测试使用 ConstantLR 调度器时，学习率在固定的 epoch 下保持不变
    def test_constantlr_is_constant_for_constant_epoch(self):
        # 创建 ConstantLR 调度器对象
        scheduler = ConstantLR(self.opt)
        # 调用通用测试函数，验证学习率在固定的 epoch 下保持不变
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    # 测试使用 LinearLR 调度器时，学习率在固定的 epoch 下保持不变
    def test_linear_linearlr_is_constant_for_constant_epoch(self):
        # 创建 LinearLR 调度器对象
        scheduler = LinearLR(self.opt)
        # 调用通用测试函数，验证学习率在固定的 epoch 下保持不变
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    # 测试使用 PolynomialLR 调度器时，学习率在固定的 epoch 下保持不变
    def test_polynomial_lr_is_constant_for_constant_epoch(self):
        # 创建 PolynomialLR 调度器对象，使用指数 0.9
        scheduler = PolynomialLR(self.opt, power=0.9)
        # 调用通用测试函数，验证学习率在固定的 epoch 下保持不变
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    # 测试使用 StepLR 调度器
    def test_step_lr(self):
        # 定义总共的 epoch 数量
        epochs = 10
        # 定义每个 epoch 对应的目标学习率列表
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3
        # 将每个 epoch 的目标学习率列表扩展到两倍，用于测试
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 StepLR 调度器对象，设置衰减率 gamma 为 0.1，步长为 3
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        # 调用通用测试函数，验证学习率的变化是否符合预期
        self._test(scheduler, targets, epochs)

    # 测试获取 StepLR 调度器的最后一个学习率
    def test_get_last_lr_step_lr(self):
        # 导入需要的库
        from torch.nn import Parameter
        # 定义总共的 epoch 数量
        epochs = 10
        # 创建一个 SGD 优化器对象，用于测试
        optimizer = SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        # 定义每个 epoch 对应的目标学习率列表
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        # 创建 StepLR 调度器对象，设置衰减率 gamma 为 0.1，步长为 3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        # 调用通用测试函数，验证获取的最后一个学习率是否符合预期
        self._test_get_last_lr(scheduler, targets, epochs)

    # 测试使用 MultiStepLR 调度器
    def test_get_last_lr_multi_step_lr(self):
        # 定义总共的 epoch 数量
        epochs = 10
        # 定义每个 epoch 对应的目标学习率列表
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 1
        # 将每个 epoch 的目标学习率列表扩展到两倍，用于测试
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 MultiStepLR 调度器对象，设置衰减率 gamma 为 0.1，里程碑为 [2, 5, 9]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 调用通用测试函数，验证学习率的变化是否符合预期
        self._test_get_last_lr(scheduler, targets, epochs)

    # 测试使用 MultiStepLR 调度器
    def test_multi_step_lr(self):
        # 定义总共的 epoch 数量
        epochs = 10
        # 定义每个 epoch 对应的目标学习率列表
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        # 将每个 epoch 的目标学习率列表扩展到两倍，用于测试
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 MultiStepLR 调度器对象，设置衰减率 gamma 为 0.1，里程碑为 [2, 5, 9]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 调用通用测试函数，验证学习率的变化是否符合预期
        self._test(scheduler, targets, epochs)

    # 测试使用 MultiStepLR 调度器，并包含 epoch 信息
    def test_multi_step_lr_with_epoch(self):
        # 定义总共的 epoch 数量
        epochs = 10
        # 定义每个 epoch 对应的目标学习率列表
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        # 将每个 epoch 的目标学习率列表扩展到两倍，用于测试
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 MultiStepLR 调度器对象，设置衰减率 gamma 为 0.1，里程碑为 [2, 5, 9]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 调用通用测试函数，验证学习率的变化是否符合预期，并包含 epoch 信息
        self._test_with_epoch(scheduler, targets, epochs)
    # 定义测试方法，用于测试 ConstantLR 调度器的行为
    def test_get_last_lr_constantlr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 根据训练周期数创建学习率变化列表
        single_targets = [0.025] * 5 + [0.05] * 5
        # 创建包含两个学习率变化列表的列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 ConstantLR 调度器对象
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        # 调用测试方法，验证 ConstantLR 调度器的最终学习率
        self._test_get_last_lr(scheduler, targets, epochs)

    # 定义测试方法，用于测试 LinearLR 调度器的行为
    def test_get_last_lr_linearlr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 设置学习率插值的起始和结束因子
        start_factor = 1.0 / 4
        end_factor = 3.0 / 5
        # 设置插值的总步数
        iters = 4
        # 创建线性插值列表
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        # 创建单一目标学习率列表
        single_targets = [x * 0.05 for x in interpolation] + [0.05 * end_factor] * (
            epochs - iters
        )
        # 创建包含两个学习率变化列表的列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 LinearLR 调度器对象
        scheduler = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        # 调用测试方法，验证 LinearLR 调度器的最终学习率
        self._test_get_last_lr(scheduler, targets, epochs)

    # 定义测试方法，用于测试 ConstantLR 调度器的行为
    def test_constantlr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 根据训练周期数创建学习率变化列表
        single_targets = [0.025] * 5 + [0.05] * 5
        # 创建包含两个学习率变化列表的列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 ConstantLR 调度器对象
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        # 调用测试方法，验证 ConstantLR 调度器的行为
        self._test(scheduler, targets, epochs)

    # 定义测试方法，用于测试 LinearLR 调度器的行为
    def test_linearlr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 设置线性插值的起始因子
        start_factor = 1.0 / 2
        # 设置线性插值的总步数
        iters = 4
        # 创建线性插值列表
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        # 创建单一目标学习率列表
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        # 创建包含两个学习率变化列表的列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建 LinearLR 调度器对象
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        # 调用测试方法，验证 LinearLR 调度器的行为
        self._test(scheduler, targets, epochs)

    # 定义测试方法，验证 LinearLR 调度器的起始因子限制
    def test_linearlr_start_factor_limits1(self):
        # 设置非法的起始因子
        start_factor = 0.0
        iters = 4
        # 检查是否抛出预期的 ValueError 异常
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    # 定义测试方法，验证 LinearLR 调度器的起始因子限制
    def test_linearlr_start_factor_limits2(self):
        # 设置非法的起始因子
        start_factor = 1.1
        iters = 4
        # 检查是否抛出预期的 ValueError 异常
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
    def test_constantlr_with_epoch(self):
        # 设置总共的训练轮数为10
        epochs = 10
        # 按照规定生成每个epoch的学习率目标值列表
        single_targets = [0.025] * 5 + [0.05] * 5
        # 将目标值列表组成两个嵌套列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 使用 ConstantLR 调度器创建对象，设置学习率因子为1/2，总迭代次数为5
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        # 调用 _test_with_epoch 方法进行测试
        self._test_with_epoch(scheduler, targets, epochs)

    def test_linearlr_with_epoch(self):
        # 设置总共的训练轮数为10
        epochs = 10
        # 定义线性变化的起始和结束因子，以及迭代次数
        start_factor = 1.0 / 2
        end_factor = 1.0
        iters = 4
        # 计算线性插值列表
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        # 按照线性插值生成每个epoch的学习率目标值列表
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        # 将目标值列表组成两个嵌套列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 使用 LinearLR 调度器创建对象，设置起始因子和总迭代次数
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        # 调用 _test_with_epoch 方法进行测试
        self._test_with_epoch(scheduler, targets, epochs)

    def test_exp_lr(self):
        # 设置总共的训练轮数为10
        epochs = 10
        # 按照指数衰减生成每个epoch的学习率目标值列表
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        # 将目标值列表组成两个嵌套列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 使用 ExponentialLR 调度器创建对象，设置衰减因子 gamma=0.9
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        # 调用 _test 方法进行测试
        self._test(scheduler, targets, epochs)

    def test_poly_lr(self):
        # 设置总共的训练轮数为10，定义多项式衰减的幂次和总迭代次数
        epochs = 10
        power = 0.9
        total_iters = 5
        # 按照多项式衰减生成每个epoch的学习率目标值列表
        single_targets = [
            (1.0 - x / total_iters) ** power * 0.05 for x in range(total_iters)
        ] + [0.0] * (epochs - total_iters)
        # 将目标值列表组成两个嵌套列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 使用 PolynomialLR 调度器创建对象，设置幂次和总迭代次数
        scheduler = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        # 调用 _test 方法进行测试
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        # 设置总共的训练轮数为10，定义余弦退火的最小学习率
        epochs = 10
        eta_min = 1e-10
        # 按照余弦退火生成每个epoch的学习率目标值列表
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        # 将目标值列表组成两个嵌套列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 使用 CosineAnnealingLR 调度器创建对象，设置最大周期数和最小学习率
        scheduler = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        # 调用 _test 方法进行测试
        self._test(scheduler, targets, epochs)

    def test_closed_form_step_lr(self):
        # 使用 StepLR 调度器创建对象，设置衰减因子 gamma=0.1 和步长 step_size=3
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        # 创建另一个 StepLR 调度器对象作为对照
        closed_form_scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        # 调用 _test_against_closed_form 方法进行测试，设置总训练次数为20
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_linearlr(self):
        # 使用 LinearLR 调度器创建对象，设置起始因子和结束因子，总迭代次数为4
        scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        # 创建另一个 LinearLR 调度器对象作为对照
        closed_form_scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        # 调用 _test_against_closed_form 方法进行测试，设置总训练次数为20
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)
    # 测试使用 ConstantLR 调度器的闭式解
    def test_closed_form_constantlr(self):
        # 创建 ConstantLR 调度器对象，设定学习率衰减因子和总迭代次数
        scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        # 创建闭式解的 ConstantLR 调度器对象，参数与上述相同
        closed_form_scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        # 调用 _test_against_closed_form 方法，比较两个调度器的输出
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    # 测试使用 MultiStepLR 调度器的闭式解
    def test_closed_form_multi_step_lr(self):
        # 创建 MultiStepLR 调度器对象，设定学习率衰减率和里程碑（迭代次数）
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 创建闭式解的 MultiStepLR 调度器对象，参数与上述相同
        closed_form_scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 调用 _test_against_closed_form 方法，比较两个调度器的输出
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    # 测试使用 ExponentialLR 调度器的闭式解
    def test_closed_form_exp_lr(self):
        # 创建 ExponentialLR 调度器对象，设定学习率衰减率
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        # 创建闭式解的 ExponentialLR 调度器对象，参数与上述相同
        closed_form_scheduler = ExponentialLR(self.opt, gamma=0.9)
        # 调用 _test_against_closed_form 方法，比较两个调度器的输出
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    # 测试使用 PolynomialLR 调度器的闭式解
    def test_closed_form_poly_lr(self):
        # 创建 PolynomialLR 调度器对象，设定多项式衰减的幂次
        scheduler = PolynomialLR(self.opt, power=0.9)
        # 创建闭式解的 PolynomialLR 调度器对象，参数与上述相同
        closed_form_scheduler = PolynomialLR(self.opt, power=0.9)
        # 调用 _test_against_closed_form 方法，比较两个调度器的输出
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    # 测试使用 CosineAnnealingLR 调度器的闭式解
    def test_closed_form_cos_anneal_lr(self):
        eta_min = 1e-10
        epochs = 20
        T_max = 5
        # 创建 CosineAnnealingLR 调度器对象，设定周期 T_max 和最小学习率 eta_min
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        # 创建闭式解的 CosineAnnealingLR 调度器对象，参数与上述相同
        closed_form_scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        # 调用 _test_against_closed_form 方法，比较两个调度器的输出
        self._test_against_closed_form(scheduler, closed_form_scheduler, epochs)

    # 测试使用 CosineAnnealingLR 调度器的继续使用
    def test_cos_anneal_lr_continue(self):
        eta_min = 0.1
        T_max = 5
        # 创建 CosineAnnealingLR 调度器对象，设定周期 T_max 和最小学习率 eta_min
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        # 优化器执行一步优化
        self.opt.step()
        # 调度器执行一步学习率更新
        scheduler.step()
        # 获取原始学习率列表
        original_lrs = scheduler._last_lr
        # 创建新的 CosineAnnealingLR 调度器对象，设定周期 T_max 和最小学习率 eta_min，并指定起始迭代数为 0
        new_scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min, last_epoch=0)
        # 获取新的学习率列表
        new_lrs = new_scheduler._last_lr
        # 使用测试工具断言，比较原始学习率列表和新学习率列表的近似性
        torch.testing.assert_close(original_lrs, new_lrs, rtol=1e-4, atol=1e-5)

    # 测试使用 ReduceLROnPlateau 调度器的第一种情况
    def test_reduce_lr_on_plateau1(self):
        epochs = 10
        # 将优化器中所有参数组的学习率设置为 0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 设定预期的目标学习率列表
        targets = [[0.5] * 20]
        # 设定指标列表，用于模拟训练中的指标变化
        metrics = [10 - i * 0.0167 for i in range(20)]
        # 创建 ReduceLROnPlateau 调度器对象，设定阈值模式、优化模式、阈值、耐心和冷却时间
        scheduler = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        # 调用 _test_reduce_lr_on_plateau 方法，比较调度器在给定情况下的输出
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    # 测试使用 ReduceLROnPlateau 调度器的第二种情况
    def test_reduce_lr_on_plateau2(self):
        epochs = 22
        # 将优化器中所有参数组的学习率设置为 0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 设定预期的目标学习率列表
        targets = [[0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2]
        # 设定指标列表，用于模拟训练中的指标变化
        metrics = [10 - i * 0.0165 for i in range(22)]
        # 创建 ReduceLROnPlateau 调度器对象，设定耐心、冷却时间、阈值模式、优化模式和阈值
        scheduler = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        # 调用 _test_reduce_lr_on_plateau 方法，比较调度器在给定情况下的输出
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)
    def test_reduce_lr_on_plateau3(self):
        epochs = 22
        # 遍历优化器中的参数组，并将学习率设为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义目标值列表
        targets = [[0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4]
        # 定义指标值列表
        metrics = [-0.8] * 2 + [-0.234] * 20
        # 创建 ReduceLROnPlateau 调度器对象，设置模式为最大化，耐心为5，冷却时间为5，阈值模式为绝对值
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        # 调用私有方法 _test_reduce_lr_on_plateau 进行测试
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau4(self):
        epochs = 20
        # 遍历优化器中的参数组，并将学习率设为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义目标值列表
        targets = [[0.5] * 20]
        # 定义指标值列表，使用指数增长的方式生成
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        # 创建 ReduceLROnPlateau 调度器对象，设置模式为最大化，耐心为3，阈值模式为相对值，阈值为0.1
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        # 调用私有方法 _test_reduce_lr_on_plateau 进行测试
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau5(self):
        epochs = 20
        # 遍历优化器中的参数组，并将学习率设为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义目标值列表
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        # 定义指标值列表，使用指数增长的方式生成
        metrics = [1.5 * (1.005**i) for i in range(20)]
        # 创建 ReduceLROnPlateau 调度器对象，设置模式为最大化，阈值模式为相对值，阈值为0.1，耐心为5，冷却时间为5
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        # 调用私有方法 _test_reduce_lr_on_plateau 进行测试
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau6(self):
        epochs = 20
        # 遍历优化器中的参数组，并将学习率设为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义目标值列表
        targets = [[0.5] * 20]
        # 定义指标值列表，使用指数衰减的方式生成
        metrics = [1.5 * (0.85**i) for i in range(20)]
        # 创建 ReduceLROnPlateau 调度器对象，设置模式为最小化，阈值模式为相对值，阈值为0.1
        scheduler = ReduceLROnPlateau(
            self.opt, mode="min", threshold_mode="rel", threshold=0.1
        )
        # 调用私有方法 _test_reduce_lr_on_plateau 进行测试
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau7(self):
        epochs = 20
        # 遍历优化器中的参数组，并将学习率设为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义目标值列表
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        # 定义指标值列表，具有两个阶段，第一阶段指标为1，第二阶段逐渐减小到0.5
        metrics = [1] * 7 + [0.6] + [0.5] * 12
        # 创建 ReduceLROnPlateau 调度器对象，设置模式为最小化，阈值模式为相对值，阈值为0.1，耐心为5，冷却时间为5
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="min",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        # 调用私有方法 _test_reduce_lr_on_plateau 进行测试
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)
    def test_reduce_lr_on_plateau8(self):
        epochs = 20
        # 设置优化器中所有参数组的学习率为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义两个目标列表
        targets = [[0.5] * 6 + [0.4] * 14, [0.5] * 6 + [0.3] * 14]
        # 定义指标列表，采用指数增长
        metrics = [1.5 * (1.005**i) for i in range(20)]
        # 创建ReduceLROnPlateau学习率调度器对象
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",  # 模式为最大值
            threshold_mode="rel",  # 阈值模式为相对值
            min_lr=[0.4, 0.3],  # 最小学习率
            threshold=0.1,  # 阈值
            patience=5,  # 忍耐周期
            cooldown=5,  # 冷却周期
        )
        # 调用测试函数，验证ReduceLROnPlateau学习率调度器的效果
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau_get_last_lr_before_step(self):
        # 设置优化器中所有参数组的学习率为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 创建ReduceLROnPlateau学习率调度器对象
        scheduler = ReduceLROnPlateau(
            self.opt,
        )
        # 断言验证获取最后学习率是否与设置一致
        self.assertEqual(
            scheduler.get_last_lr(), [0.5 for param_group in self.opt.param_groups]
        )

    def test_sequentiallr1(self):
        epochs = 19
        # 初始化调度器列表
        schedulers = [None] * 2
        # 定义目标列表
        targets = [
            [0.05, 0.04, 0.032]
            + [0.05 for x in range(4)]
            + [0.05 * 0.1 for x in range(4)]
            + [0.05 * 0.01 for x in range(4)]
            + [0.05 * 0.001 for x in range(4)]
        ]
        # 定义里程碑
        milestones = [3]
        # 初始化ExponentialLR学习率调度器对象
        schedulers[0] = ExponentialLR(self.opt, gamma=0.8)
        # 初始化StepLR学习率调度器对象
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=4)
        # 创建SequentialLR学习率调度器对象
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        # 调用测试函数，验证SequentialLR学习率调度器的效果
        self._test(scheduler, targets, epochs)

    def test_sequentiallr2(self):
        epochs = 13
        # 初始化调度器列表
        schedulers = [None] * 2
        # 定义目标列表
        targets = [[0.005, 0.005, 0.005] + [0.05 * 0.9**x for x in range(10)]]
        # 定义里程碑
        milestones = [3]
        # 初始化ConstantLR学习率调度器对象
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        # 初始化ExponentialLR学习率调度器对象
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        # 创建SequentialLR学习率调度器对象
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        # 调用测试函数，验证SequentialLR学习率调度器的效果
        self._test(scheduler, targets, epochs)

    def test_sequentiallr3(self):
        epochs = 12
        # 初始化调度器列表
        schedulers = [None] * 3
        # 定义目标列表
        targets = [
            [0.005, 0.005, 0.005]
            + [0.05, 0.04, 0.032]
            + [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        ]
        # 定义里程碑
        milestones = [3, 6]
        # 初始化ConstantLR学习率调度器对象
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        # 初始化ExponentialLR学习率调度器对象
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        # 初始化StepLR学习率调度器对象
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        # 创建SequentialLR学习率调度器对象
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        # 调用测试函数，验证SequentialLR学习率调度器的效果
        self._test(scheduler, targets, epochs)
    def test_sequentiallr4(self):
        optimizer = SGD([torch.tensor(0.5)], lr=0.1)
        prev_lr = optimizer.param_groups[0]["lr"]

        schedulers = [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1),
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=[10]
        )

        new_lr = optimizer.param_groups[0]["lr"]

        # Ensure that multiple schedulers does not affect the initial learning rate
        self.assertEqual(prev_lr, new_lr)

    def test_get_last_lr_sequentiallr(self):
        epochs = 12
        milestones = [3, 6]
        schedulers = [None] * 3
        
        # 创建三个不同的学习率调度器对象
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        
        # 创建一个串行的学习率调度器对象
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        
        # 预期的学习率列表
        constant_lr_target = [0.005] * 3
        exponential_lr_target = [0.05, 0.04, 0.032]
        step_lr_target = [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        single_targets = constant_lr_target + exponential_lr_target + step_lr_target
        targets = [single_targets, [x * 10 for x in single_targets]]
        
        # 调用测试函数，验证最终的学习率列表
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_chained_lr2_get_last_lr_before_step(self):
        schedulers = [
            LinearLR(self.opt, start_factor=0.4, total_iters=3),
            MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1),
        ]
        
        # 创建一个链式调度器对象
        scheduler = ChainedScheduler(schedulers)
        
        # 验证链式调度器对象的最终学习率是否与最后一个调度器的最终学习率相同
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr1(self):
        epochs = 10
        schedulers = [None] * 1
        
        # 创建一个阶梯调度器对象
        targets = [[0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        
        # 创建一个链式调度器对象
        scheduler = ChainedScheduler(schedulers)
        
        # 调用测试函数，验证最终的学习率列表
        self._test([scheduler], targets, epochs)
        
        # 验证链式调度器对象的最终学习率是否与最后一个调度器的最终学习率相同
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr2(self):
        epochs = 10
        schedulers = [None] * 1
        
        # 创建一个线性调度器对象
        targets = [[0.02, 0.03, 0.04] + [0.05] * 9]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        
        # 创建一个链式调度器对象
        scheduler = ChainedScheduler(schedulers)
        
        # 调用测试函数，验证最终的学习率列表
        self._test([scheduler], targets, epochs)
        
        # 验证链式调度器对象的最终学习率是否与最后一个调度器的最终学习率相同
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())
    def test_chained_lr3(self):
        # 设定训练周期数
        epochs = 10
        # 初始化调度器列表
        schedulers = [None] * 2
        # 设定目标学习率列表
        targets = [
            [0.02, 0.03, 0.04, 0.05] + [0.005] * 4 + [0.0005] * 3 + [0.00005] * 3
        ]
        # 创建线性学习率调度器并添加到调度器列表中
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        # 创建多步学习率调度器并添加到调度器列表中
        schedulers[1] = MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1)
        # 创建链式调度器对象
        scheduler = ChainedScheduler(schedulers)
        # 运行测试方法，验证结果与目标学习率列表是否一致
        self._test([scheduler], targets, epochs)
        # 断言最后一个调度器的学习率是否正确
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr4(self):
        # 设定训练周期数
        epochs = 9
        # 初始化调度器列表
        schedulers = [None] * 3
        # 设定目标学习率列表
        targets = [
            [0.05 * 0.2 * 0.9**x for x in range(3)]
            + [0.05 * 0.2 * 0.9**3 * 0.1]
            + [0.05 * 0.9**x * 0.1 for x in range(4, 6)]
            + [0.05 * 0.9**x * 0.01 for x in range(6, 9)]
        ]
        # 创建指数衰减学习率调度器并添加到调度器列表中
        schedulers[0] = ExponentialLR(self.opt, gamma=0.9)
        # 创建恒定学习率调度器并添加到调度器列表中
        schedulers[1] = ConstantLR(self.opt, factor=0.2, total_iters=4)
        # 创建步长学习率调度器并添加到调度器列表中
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=3)
        # 创建链式调度器对象
        scheduler = ChainedScheduler(schedulers)
        # 运行测试方法，验证结果与目标学习率列表是否一致
        self._test([scheduler], targets, epochs)
        # 断言最后一个调度器的学习率是否正确
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr5(self):
        # 定义多项式学习率函数
        def poly_lr(lr: float):
            return [
                (lr * ((1.0 - x / total_iters) ** power)) for x in range(total_iters)
            ] + [0.0] * (epochs - total_iters)

        # 初始化调度器列表
        schedulers = [None] * 2
        # 设定训练周期数
        epochs = 10
        # 设定多项式学习率参数
        power = 0.9
        total_iters = 5
        const_factor = 0.1
        # 计算单个多项式学习率目标值
        single_targets = [x * const_factor for x in poly_lr(lr=0.05)]
        # 计算多个多项式学习率目标值
        targets = [single_targets, [x * const_factor for x in poly_lr(0.5)]]
        # 创建多项式衰减学习率调度器并添加到调度器列表中
        schedulers[0] = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        # 创建恒定学习率调度器并添加到调度器列表中
        schedulers[1] = ConstantLR(self.opt, factor=const_factor)
        # 创建链式调度器对象
        scheduler = ChainedScheduler(schedulers)
        # 运行测试方法，验证结果与目标学习率列表是否一致
        self._test([scheduler], targets, epochs)
        # 断言最后一个调度器的学习率是否正确
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_compound_step_and_multistep_lr(self):
        # 设定训练周期数
        epochs = 10
        # 初始化调度器列表
        schedulers = [None] * 2
        # 创建步长学习率调度器并添加到调度器列表中
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        # 创建多步学习率调度器并添加到调度器列表中
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 设定目标学习率列表
        targets = [[0.05] * 2 + [0.005] * 1 + [5e-4] * 2 + [5e-5] + [5e-6] * 3 + [5e-8]]
        # 运行测试方法，验证结果与目标学习率列表是否一致
        self._test(schedulers, targets, epochs)
    # 定义测试方法，测试复合步长和指数学习率调度器
    def test_compound_step_and_exp_lr(self):
        # 设定总共的训练周期数
        epochs = 10
        # 初始化调度器列表，长度为2
        schedulers = [None] * 2
        # 计算单一目标学习率列表，分为多个范围，每个范围使用不同的衰减率
        single_targets = [0.05 * (0.9**x) for x in range(3)]
        single_targets += [0.005 * (0.9**x) for x in range(3, 6)]
        single_targets += [0.0005 * (0.9**x) for x in range(6, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 12)]
        # 构建目标列表，包含单一目标列表及其乘以训练周期数的结果列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建步长调度器对象，设置衰减率和步长
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        # 创建指数调度器对象，设置衰减率
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        # 调用测试方法，传入调度器、目标列表和训练周期数
        self._test(schedulers, targets, epochs)

    # 定义测试方法，测试复合指数和多步长学习率调度器
    def test_compound_exp_and_multistep_lr(self):
        # 设定总共的训练周期数
        epochs = 10
        # 初始化调度器列表，长度为2
        schedulers = [None] * 2
        # 计算单一目标学习率列表，分为多个范围，每个范围使用不同的衰减率
        single_targets = [0.05 * (0.9**x) for x in range(2)]
        single_targets += [0.005 * (0.9**x) for x in range(2, 5)]
        single_targets += [0.0005 * (0.9**x) for x in range(5, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 11)]
        # 构建目标列表，包含单一目标列表及其乘以训练周期数的结果列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建多步长调度器对象，设置衰减率和里程碑步数
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 创建指数调度器对象，设置衰减率
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        # 调用测试方法，传入调度器、目标列表和训练周期数
        self._test(schedulers, targets, epochs)

    # 定义测试方法，测试复合指数和线性学习率调度器
    def test_compound_exp_and_linearlr(self):
        # 设定总共的训练周期数
        epochs = 10
        # 设定迭代次数
        iters = 4
        # 设定起始和结束因子
        start_factor = 0.4
        end_factor = 0.9
        # 初始化调度器列表，长度为2
        schedulers = [None] * 2
        # 计算单一目标学习率列表，每个范围使用不同的衰减率，根据迭代次数调整学习率
        single_targets = [0.05 * (0.9**x) for x in range(11)]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (end_factor - start_factor)
        for i in range(iters, 11):
            single_targets[i] *= end_factor
        # 构建目标列表，包含单一目标列表及其乘以训练周期数的结果列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建线性调度器对象，设置起始因子、结束因子和总迭代次数
        schedulers[0] = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        # 创建指数调度器对象，设置衰减率
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        # 调用测试方法，传入调度器、目标列表和训练周期数
        self._test(schedulers, targets, epochs)

    # 定义测试方法，测试复合步长和常数学习率调度器
    def test_compound_step_and_constantlr(self):
        # 设定总共的训练周期数
        epochs = 10
        # 设定迭代次数
        iters = 4
        # 设定常数因子
        factor = 0.4
        # 初始化调度器列表，长度为2
        schedulers = [None] * 2
        # 计算单一目标学习率列表，根据迭代次数调整学习率
        single_targets = (
            [0.05 * 0.4] * 3
            + [0.005 * 0.4]
            + [0.005] * 2
            + [0.0005] * 3
            + [0.00005] * 3
        )
        # 构建目标列表，包含单一目标列表及其乘以训练周期数的结果列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 创建步长调度器对象，设置衰减率和步长
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        # 创建常数学习率调度器对象，设置常数因子和总迭代次数
        schedulers[1] = ConstantLR(self.opt, factor=0.4, total_iters=4)
        # 调用测试方法，传入调度器、目标列表和训练周期数
        self._test(schedulers, targets, epochs)
    def test_compound_linearlr_and_multistep_lr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 设置迭代次数
        iters = 4
        # 设置初始因子
        start_factor = 0.4
        # 初始化调度器列表
        schedulers = [None] * 2
        # 单个目标学习率列表，根据训练周期的不同阶段逐步减小
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 2
        # 根据迭代次数调整每个目标学习率
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        # 设置目标学习率列表，分别对应单步和多步调度器
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 初始化多步和线性调度器
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        # 执行测试
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_step_lr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 设置最小的学习率
        eta_min = 1e-10
        # 根据余弦退火计算单个目标学习率列表
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        # 根据阶段性因子调整目标学习率
        single_targets = [x * 0.1 ** (i // 3) for i, x in enumerate(single_targets)]
        # 设置目标学习率列表，分别对应余弦退火和阶段调度器
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 初始化余弦退火和阶段调度器
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        # 执行测试
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_multistep_lr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 设置最小的学习率
        eta_min = 1e-10
        # 根据余弦退火计算单个目标学习率列表
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        # 设置多步调度器的阶段性倍增因子
        multipliers = [1] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001]
        # 根据倍增因子调整目标学习率
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        # 设置目标学习率列表，分别对应余弦退火和多步调度器
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 初始化余弦退火和多步调度器
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        # 执行测试
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_linearlr(self):
        # 设置总共的训练周期数
        epochs = 10
        # 设置迭代次数
        iters = 4
        # 设置初始因子
        start_factor = 0.4
        # 设置最小的学习率
        eta_min = 1e-10
        # 初始化调度器列表
        schedulers = [None] * 2
        # 根据余弦退火计算单个目标学习率列表
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        # 根据迭代次数调整每个目标学习率
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        # 设置目标学习率列表，分别对应线性和余弦退火调度器
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 初始化线性和余弦退火调度器
        schedulers[0] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        schedulers[1] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        # 执行测试
        self._test(schedulers, targets, epochs)
    # 定义一个测试函数，用于测试复合学习率调度器的效果：CosineAnnealingLR 和 ExponentialLR
    def test_compound_cosanneal_and_exp_lr(self):
        # 设置总的训练周期数
        epochs = 10
        # 设定最小学习率
        eta_min = 1e-10
        # 根据 CosineAnnealing 调度器的公式计算单一目标学习率列表
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        # 根据指数衰减调度器的公式计算倍乘因子列表
        multipliers = [0.1**i for i in range(epochs)]
        # 将单一目标学习率与倍乘因子相乘得到最终的单一目标学习率列表
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        # 组装成两个目标学习率列表
        targets = [single_targets, [x * epochs for x in single_targets]]
        # 初始化调度器列表
        schedulers = [None] * 2
        # 创建 CosineAnnealingLR 调度器对象，绑定到优化器 self.opt 上
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        # 创建 ExponentialLR 调度器对象，绑定到优化器 self.opt 上
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        # 调用测试函数 _test，传入调度器、目标学习率列表和总周期数进行测试
        self._test(schedulers, targets, epochs)

    # 定义一个测试函数，用于测试复合学习率调度器 ReduceLROnPlateau 和 StepLR 的效果
    def test_compound_reduce_lr_on_plateau1(self):
        # 设置总的训练周期数
        epochs = 10
        # 将优化器 self.opt 中的所有参数组的学习率设为 0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 设置单一目标学习率列表为 20 个 0.5
        single_targets = [0.5] * 20
        # 根据索引将倍乘因子列表设置为递减的 20 个元素
        multipliers = [0.1 ** (i // 3) for i in range(20)]
        # 将单一目标学习率与倍乘因子相乘得到最终的单一目标学习率列表
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        # 设置目标学习率列表，删除第一个元素
        targets = [single_targets]
        # 初始化指标列表，包含 20 个元素，每个元素为从 10 到 0 递减的数列
        metrics = [10 - i * 0.0167 for i in range(20)]
        # 初始化调度器列表
        schedulers = [None, None]
        # 创建 ReduceLROnPlateau 调度器对象，绑定到优化器 self.opt 上，设定参数
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        # 创建 StepLR 调度器对象，绑定到优化器 self.opt 上，设定参数
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        # 调用测试函数 _test_reduce_lr_on_plateau，传入调度器、目标学习率列表、指标列表和总周期数进行测试
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    # 定义一个测试函数，用于测试复合学习率调度器 ReduceLROnPlateau 和 MultiStepLR 的效果
    def test_compound_reduce_lr_on_plateau2(self):
        # 设置总的训练周期数
        epochs = 22
        # 将优化器 self.opt 中的所有参数组的学习率设为 0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 设置单一目标学习率列表，分段递减设置为 6 个 0.5、7 个 0.05、7 个 0.005、2 个 0.0005
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        # 设置倍乘因子列表，分段设置为 3 个 1、5 个 0.1、4 个 0.01、10 个 0.001
        multipliers = [1] * 3 + [0.1] * 5 + [0.01] * 4 + [0.001] * 10
        # 将单一目标学习率与倍乘因子相乘得到最终的单一目标学习率列表
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        # 设置目标学习率列表，删除第一个元素
        targets = [single_targets]
        # 初始化指标列表，包含 22 个元素，每个元素为从 10 到 0 递减的数列
        metrics = [10 - i * 0.0165 for i in range(22)]
        # 初始化调度器列表
        schedulers = [None] * 2
        # 创建 ReduceLROnPlateau 调度器对象，绑定到优化器 self.opt 上，设定参数
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        # 创建 MultiStepLR 调度器对象，绑定到优化器 self.opt 上，设定参数
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[3, 8, 12])
        # 调用测试函数 _test_reduce_lr_on_plateau，传入调度器、目标学习率列表、指标列表和总周期数进行测试
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)
    def test_compound_reduce_lr_on_plateau3(self):
        epochs = 22
        # 遍历优化器中的参数组，将学习率设置为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义单一目标列表，包含不同阶段的学习率目标值
        single_targets = [0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4
        # 计算每个阶段的目标值乘以对应的倍数
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        # 将目标列表放入嵌套列表中
        targets = [single_targets]
        # 去除第一个元素，用于在检查学习率之前运行步骤
        targets = targets[1:]
        # 定义指标列表，表示每个 epoch 的度量结果
        metrics = [-0.8] * 2 + [-0.234] * 20
        # 创建调度器列表并初始化为 None
        schedulers = [None, None]
        # 使用 ReduceLROnPlateau 调度器，设置参数
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        # 使用 ExponentialLR 调度器，设置参数
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        # 调用测试函数，传入调度器、目标、度量和 epochs 参数
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau4(self):
        epochs = 20
        # 遍历优化器中的参数组，将学习率设置为0.05
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.05
        epochs = 10
        eta_min = 1e-10
        # 使用余弦函数计算单一目标列表的值
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        # 将目标列表放入嵌套列表中
        targets = [single_targets]
        # 去除第一个元素，用于在检查学习率之前运行步骤
        targets = targets[1:]
        # 定义指标列表，表示每个 epoch 的度量结果
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        # 创建调度器列表并初始化为 None
        schedulers = [None, None]
        # 使用 ReduceLROnPlateau 调度器，设置参数
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        # 使用 CosineAnnealingLR 调度器，设置参数
        schedulers[1] = CosineAnnealingLR(self.opt, epochs, eta_min)
        # 调用测试函数，传入调度器、目标、度量和 epochs 参数
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau5(self):
        iters = 4
        start_factor = 0.4
        epochs = 22
        # 遍历优化器中的参数组，将学习率设置为0.5
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        # 定义单一目标列表，包含不同阶段的学习率目标值，根据倍数调整
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 22
        # 根据迭代次数调整每个阶段的目标值
        for i in range(iters):
            multipliers[i] *= start_factor + i / iters * (1 - start_factor)
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        # 将目标列表放入嵌套列表中
        targets = [single_targets]
        # 去除第一个元素，用于在检查学习率之前运行步骤
        targets = targets[1:]
        # 定义指标列表，表示每个 epoch 的度量结果
        metrics = [10 - i * 0.0165 for i in range(22)]
        # 创建调度器列表并初始化为 None
        schedulers = [None] * 2
        # 使用 ReduceLROnPlateau 调度器，设置参数
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        # 使用 LinearLR 调度器，设置参数
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        # 调用测试函数，传入调度器、目标、度量和 epochs 参数
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_cycle_lr_invalid_mode(self):
        # 使用无效的模式参数创建 CyclicLR 调度器，预期引发 ValueError 异常
        with self.assertRaises(ValueError):
            scheduler = CyclicLR(self.opt, base_lr=0, max_lr=0, mode="CATS")
    # 定义测试函数 test_cycle_lr_triangular_mode_one_lr，测试带有一个学习率的三角形周期调度器
    def test_cycle_lr_triangular_mode_one_lr(self):
        # 预期的学习率变化列表
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        # 预期的动量变化列表
        momentum_target = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        # 将学习率变化列表复制两次，用于测试
        lr_targets = [lr_target, lr_target]
        # 将动量变化列表复制两次，用于测试
        momentum_targets = [momentum_target, momentum_target]
        # 创建一个三角形周期调度器对象，设置学习率和动量相关参数
        scheduler = CyclicLR(
            self.opt,  # 优化器对象
            base_lr=1,  # 基础学习率
            max_lr=5,   # 最大学习率
            step_size_up=4,  # 上升步数
            cycle_momentum=True,  # 是否循环动量
            base_momentum=1,  # 基础动量
            max_momentum=5,   # 最大动量
            mode="triangular",  # 模式设为三角形
        )
        # 调用辅助函数 _test_cycle_lr 进行测试
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    # 定义测试函数 test_cycle_lr_triangular_mode_one_lr_no_momentum，测试带有一个学习率且无动量的三角形周期调度器
    def test_cycle_lr_triangular_mode_one_lr_no_momentum(self):
        # 预期的学习率变化列表
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        # 将学习率变化列表复制两次，用于测试
        lr_targets = [lr_target, lr_target]
        # 获取优化器默认的动量值，创建与学习率变化列表等长的动量变化列表
        momentum_target = [self.opt.defaults["momentum"]] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        # 创建一个三角形周期调度器对象，设置学习率相关参数，不使用动量
        scheduler = CyclicLR(
            self.opt,  # 优化器对象
            base_lr=1,  # 基础学习率
            max_lr=5,   # 最大学习率
            step_size_up=4,  # 上升步数
            cycle_momentum=False,  # 不使用动量
            mode="triangular",  # 模式设为三角形
        )
        # 调用辅助函数 _test_cycle_lr 进行测试
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    # 定义测试函数 test_cycle_lr_triangular2_mode_one_lr，测试带有一个学习率的三角形2周期调度器
    def test_cycle_lr_triangular2_mode_one_lr(self):
        # 预期的学习率变化列表
        lr_target = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        # 预期的动量变化列表
        momentum_target = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        # 将学习率变化列表复制两次，用于测试
        lr_targets = [lr_target, lr_target]
        # 将动量变化列表复制两次，用于测试
        momentum_targets = [momentum_target, momentum_target]
        # 创建一个三角形2周期调度器对象，设置学习率和动量相关参数
        scheduler = CyclicLR(
            self.opt,  # 优化器对象
            base_lr=1,  # 基础学习率
            max_lr=5,   # 最大学习率
            step_size_up=4,  # 上升步数
            cycle_momentum=True,  # 是否循环动量
            base_momentum=1,  # 基础动量
            max_momentum=5,   # 最大动量
            mode="triangular2",  # 模式设为三角形2
        )
        # 调用辅助函数 _test_cycle_lr 进行测试
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))
    # 定义一个测试函数，测试使用 exp_range 模式的 CyclicLR 调度器
    def test_cycle_lr_exp_range_mode_one_lr(self):
        # 设置基础学习率和最大学习率
        base_lr, max_lr = 1, 5
        # 计算学习率差异
        diff_lr = max_lr - base_lr
        # 设置 gamma 值
        gamma = 0.9
        # 定义一个 x 值序列
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        # 计算目标学习率列表
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        # 计算目标动量列表
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        # 将学习率目标列表重复两次，构成二维列表
        lr_targets = [lr_target, lr_target]
        # 将动量目标列表重复两次，构成二维列表
        momentum_targets = [momentum_target, momentum_target]
        # 创建一个 CyclicLR 调度器对象，使用 exp_range 模式
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode="exp_range",
            gamma=gamma,
        )
        # 调用测试函数 _test_cycle_lr，验证 CyclicLR 调度器的输出是否符合预期
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    # 定义一个测试函数，测试使用 triangular 模式的 CyclicLR 调度器
    def test_cycle_lr_triangular_mode(self):
        # 设置第一个学习率目标列表
        lr_target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        # 根据第一个学习率目标列表计算第二个学习率目标列表
        lr_target_2 = [x + 1 for x in lr_target_1]
        # 将两个学习率目标列表构成二维列表
        lr_targets = [lr_target_1, lr_target_2]
        # 设置第一个动量目标列表
        momentum_target_1 = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        # 根据第一个动量目标列表计算第二个动量目标列表
        momentum_target_2 = [x + 1 for x in momentum_target_1]
        # 将两个动量目标列表构成二维列表
        momentum_targets = [momentum_target_1, momentum_target_2]
        # 创建一个 CyclicLR 调度器对象，使用 triangular 模式
        scheduler = CyclicLR(
            self.opt,
            base_lr=[1, 2],
            max_lr=[5, 6],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 2],
            max_momentum=[5, 6],
            mode="triangular",
        )
        # 调用测试函数 _test_cycle_lr，验证 CyclicLR 调度器的输出是否符合预期
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))
    def test_cycle_lr_triangular2_mode(self):
        # 定义第一个学习率目标列表
        lr_target_1 = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        # 根据第一个学习率目标列表计算第二个学习率目标列表
        lr_target_2 = [x + 2 for x in lr_target_1]
        # 将两个学习率目标列表组成列表
        lr_targets = [lr_target_1, lr_target_2]

        # 定义第一个动量目标列表
        momentum_target_1 = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        # 根据第一个动量目标列表计算第二个动量目标列表
        momentum_target_2 = [x + 2 for x in momentum_target_1]
        # 将两个动量目标列表组成列表
        momentum_targets = [momentum_target_1, momentum_target_2]

        # 创建一个周期学习率调度器对象，使用三角形2模式
        scheduler = CyclicLR(
            self.opt,
            base_lr=[1, 3],  # 基础学习率列表
            max_lr=[5, 7],  # 最大学习率列表
            step_size_up=4,  # 上升步长
            cycle_momentum=True,  # 启用动量循环
            base_momentum=[1, 3],  # 基础动量列表
            max_momentum=[5, 7],  # 最大动量列表
            mode="triangular2",  # 使用三角形2模式
        )
        # 执行周期学习率测试函数，传入学习率目标、动量目标和目标长度
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_exp_range_mode(self):
        # 定义两个不同基础学习率和最大学习率
        base_lr_1, max_lr_1 = 1, 5
        base_lr_2, max_lr_2 = 5, 12

        # 计算学习率差异
        diff_lr_1 = max_lr_1 - base_lr_1
        diff_lr_2 = max_lr_2 - base_lr_2

        # 设置指数衰减的伽马值和衰减因子列表
        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        # 根据给定的参数计算第一个学习率目标列表
        lr_target_1 = [base_lr_1 + x * diff_lr_1 * gamma**i for i, x in enumerate(xs)]
        # 根据给定的参数计算第二个学习率目标列表
        lr_target_2 = [base_lr_2 + x * diff_lr_2 * gamma**i for i, x in enumerate(xs)]
        # 将两个学习率目标列表组成列表
        lr_targets = [lr_target_1, lr_target_2]

        # 根据给定的参数计算第一个动量目标列表
        momentum_target_1 = [
            max_lr_1 - x * diff_lr_1 * gamma**i for i, x in enumerate(xs)
        ]
        # 根据给定的参数计算第二个动量目标列表
        momentum_target_2 = [
            max_lr_2 - x * diff_lr_2 * gamma**i for i, x in enumerate(xs)
        ]
        # 将两个动量目标列表组成列表
        momentum_targets = [momentum_target_1, momentum_target_2]

        # 创建一个周期学习率调度器对象，使用指数范围模式
        scheduler = CyclicLR(
            self.opt,
            base_lr=[base_lr_1, base_lr_2],  # 基础学习率列表
            max_lr=[max_lr_1, max_lr_2],  # 最大学习率列表
            step_size_up=4,  # 上升步长
            cycle_momentum=True,  # 启用动量循环
            base_momentum=[base_lr_1, base_lr_2],  # 基础动量列表
            max_momentum=[max_lr_1, max_lr_2],  # 最大动量列表
            mode="exp_range",  # 使用指数范围模式
            gamma=gamma,  # 指数衰减因子
        )
        # 执行周期学习率测试函数，传入学习率目标、动量目标和目标长度
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))
    def test_cycle_lr_triangular_mode_step_size_up_down(self):
        lr_target = [
            1.0,                                    # 目标学习率列表的起始值
            2.0,
            3.0,
            4.0,
            5.0,
            13.0 / 3,                               # 以及中间和结束的学习率值
            11.0 / 3,
            9.0 / 3,
            7.0 / 3,
            5.0 / 3,
            1.0,
        ]
        lr_targets = [lr_target, lr_target]          # 重复的学习率目标列表
        momentum_target = [
            5.0,                                    # 动量的目标列表，从高到低再到高
            4.0,
            3.0,
            2.0,
            1.0,
            5.0 / 3,
            7.0 / 3,
            3.0,
            11.0 / 3,
            13.0 / 3,
            5.0,
        ]
        momentum_targets = [momentum_target, momentum_target]  # 重复的动量目标列表

        scheduler = CyclicLR(                       # 创建循环学习率调度器对象
            self.opt,                               # 使用给定的优化器
            base_lr=1,                              # 初始学习率
            max_lr=5,                               # 最大学习率
            step_size_up=4,                         # 上升步长
            step_size_down=6,                       # 下降步长
            cycle_momentum=True,                    # 是否循环动量
            base_momentum=1,                        # 初始动量
            max_momentum=5,                         # 最大动量
            mode="triangular",                      # 循环学习率模式为三角形
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))  # 执行循环学习率测试

    def test_cycle_lr_triangular2_mode_step_size_up_down(self):
        lr_base_target = [
            1.0,                                    # 目标学习率列表的起始值
            3.0,
            5.0,
            13.0 / 3,
            11.0 / 3,
            9.0 / 3,
            7.0 / 3,
            5.0 / 3,
            1.0,
            2.0,
            3.0,
            8.0 / 3,
            7.0 / 3,
            6.0 / 3,
            5.0 / 3,
            4.0 / 3,
            1.0,
            3.0 / 2,
            2.0,
            11.0 / 6,
            10.0 / 6,
            9.0 / 6,
            8.0 / 6,
            7.0 / 6,
        ]
        momentum_base_target = [
            5.0,                                    # 动量的目标列表，从高到低再到高
            3.0,
            1.0,
            5.0 / 3,
            7.0 / 3,
            3.0,
            11.0 / 3,
            13.0 / 3,
            5.0,
            4.0,
            3.0,
            10.0 / 3,
            11.0 / 3,
            4.0,
            13.0 / 3,
            14.0 / 3,
            5.0,
            4.5,
            4.0,
            25.0 / 6,
            13.0 / 3,
            4.5,
            14.0 / 3,
            29.0 / 6,
        ]
        deltas = [2 * i for i in range(0, 2)]        # 生成两倍数的增量列表
        base_lrs = [1 + delta for delta in deltas]   # 基础学习率列表，加上每个增量
        max_lrs = [5 + delta for delta in deltas]    # 最大学习率列表，加上每个增量
        lr_targets = [[x + delta for x in lr_base_target] for delta in deltas]  # 生成两个学习率目标列表
        momentum_targets = [
            [x + delta for x in momentum_base_target] for delta in deltas
        ]                                          # 生成两个动量目标列表
        scheduler = CyclicLR(                       # 创建循环学习率调度器对象
            self.opt,                               # 使用给定的优化器
            base_lr=base_lrs,                       # 初始学习率列表
            max_lr=max_lrs,                         # 最大学习率列表
            step_size_up=2,                         # 上升步长
            step_size_down=6,                       # 下降步长
            cycle_momentum=True,                    # 是否循环动量
            base_momentum=base_lrs,                 # 初始动量列表
            max_momentum=max_lrs,                   # 最大动量列表
            mode="triangular2",                     # 循环学习率模式为二次三角形
        )
        self._test_cycle_lr(
            scheduler, lr_targets, momentum_targets, len(lr_base_target)
        )                                          # 执行循环学习率测试
    # 测试循环学习率调度器的指数范围模式下的步长变化
    def test_cycle_lr_exp_range_mode_step_size_up_down(self):
        base_lr, max_lr = 1, 5
        # 计算基础学习率和最大学习率的差异
        diff_lr = max_lr - base_lr
        gamma = 0.9
        # 定义学习率变化的时间步 xs
        xs = [
            0.0,
            0.5,
            1.0,
            5.0 / 6,
            4.0 / 6,
            3.0 / 6,
            2.0 / 6,
            1.0 / 6,
            0.0,
            0.5,
            1.0,
            5.0 / 6,
            4.0 / 6,
        ]
        # 计算目标学习率列表
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        # 计算动量目标列表
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_targets = [momentum_target, momentum_target]
        # 创建循环学习率调度器对象
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode="exp_range",
            gamma=gamma,
        )
        # 调用测试函数，验证循环学习率调度器的行为是否符合预期
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    # 测试在没有动量的优化器上使用循环学习率调度器
    def test_cycle_lr_with_momentumless_optimizer():
        # 注意 [临时将优化器设置为 Adam]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TestLRScheduler 对象携带了一个 SGD 优化器，以避免为每个测试实例化一个新的优化器。
        # 在我们需要使用 Adam（或者任何不使用动量的优化器）的特定情况下，这会造成干扰，
        # 因为我们需要测试 CyclicLR 中的动量错误修复（错误的详细信息在 https://github.com/pytorch/pytorch/issues/19003 中有描述）。
        old_opt = self.opt
        # 将当前优化器设置为 Adam
        self.opt = Adam(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        # 创建循环学习率调度器对象，不使用动量
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode="triangular",
        )
        # 调用测试函数，验证循环学习率调度器的行为是否符合预期
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

        self.opt = old_opt  # 恢复原来的优化器为 SGD

    # 测试在没有动量的优化器上使用循环学习率调度器中的动量循环失败
    def test_cycle_lr_cycle_momentum_fail_with_momentumless_optimizer(self):
        # 使用 Rprop 优化器来创建循环学习率调度器，预期会引发 ValueError 异常
        with self.assertRaises(ValueError):
            rprop_opt = Rprop(self.net.parameters())
            scheduler = CyclicLR(rprop_opt, base_lr=1, max_lr=5, cycle_momentum=True)

    # 测试在具有 beta1 优化器的情况下使用循环学习率调度器中的动量循环
    def test_cycle_lr_cycle_momentum_with_beta1_optimizer(self):
        # 使用 Adam 优化器来创建循环学习率调度器，开启动量循环
        adam_opt = Adam(self.net.parameters())
        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=True)
    def test_cycle_lr_removed_after_out_of_scope(self):
        # 导入必要的模块：垃圾回收模块(gc)和弱引用模块(weakref)
        import gc
        import weakref

        # 禁用垃圾回收器，以便在测试期间对象不会被回收
        gc.disable()

        # 定义内部函数test，用于测试周期学习率调度器对象的生命周期
        def test():
            # 创建Adam优化器对象，传入神经网络的参数
            adam_opt = Adam(self.net.parameters())
            # 创建循环学习率调度器对象（CyclicLR），设定基础学习率、最大学习率和是否循环动量
            scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
            # 返回调度器对象的弱引用
            return weakref.ref(scheduler)

        # 调用test函数，获取其返回的调度器对象的弱引用
        ref = test()
        # 断言调度器对象的弱引用为None，即测试周期学习率调度器对象是否被正确释放
        assert ref() is None
        # 启用垃圾回收器，以便后续的测试可以正常进行
        gc.enable()

    def test_cycle_lr_state_dict_picklable(self):
        # 创建Adam优化器对象，传入神经网络的参数
        adam_opt = Adam(self.net.parameters())

        # Case 1: 内置模式
        # 创建循环学习率调度器对象（CyclicLR），设定基础学习率、最大学习率和是否循环动量
        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
        # 断言调度器对象的_scale_fn_ref属性类型为FunctionType
        self.assertIsInstance(scheduler._scale_fn_ref, types.FunctionType)
        # 获取调度器对象的状态字典
        state = scheduler.state_dict()
        # 断言状态字典中不包含"_scale_fn_ref"键，以验证该属性是否从状态字典中排除
        self.assertNotIn("_scale_fn_ref", state)
        # 断言状态字典中"_scale_fn_custom"键的值为None，以验证自定义缩放函数是否为None
        self.assertIs(state["_scale_fn_custom"], None)
        # 序列化调度器对象的状态字典，以验证其是否可以正确序列化

        pickle.dumps(state)

        # Case 2: 自定义`scale_fn`，一个函数对象
        # 定义自定义缩放函数`scale_fn`
        def scale_fn(_):
            return 0.5

        # 创建循环学习率调度器对象（CyclicLR），传入自定义缩放函数`scale_fn`，并设定其他参数
        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        # 获取调度器对象的状态字典
        state = scheduler.state_dict()
        # 断言状态字典中不包含"_scale_fn_ref"键，以验证该属性是否从状态字典中排除
        self.assertNotIn("_scale_fn_ref", state)
        # 断言状态字典中"_scale_fn_custom"键的值为None，以验证自定义缩放函数是否为None
        self.assertIs(state["_scale_fn_custom"], None)
        # 序列化调度器对象的状态字典，以验证其是否可以正确序列化
        pickle.dumps(state)

        # Case 3: 自定义`scale_fn`，一个可调用的类
        # 定义可调用类`ScaleFn`
        class ScaleFn:
            def __init__(self):
                self.x = 0.5

            def __call__(self, _):
                return self.x

        # 创建`ScaleFn`类的实例`scale_fn`
        scale_fn = ScaleFn()

        # 创建循环学习率调度器对象（CyclicLR），传入自定义缩放函数`scale_fn`，并设定其他参数
        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        # 获取调度器对象的状态字典
        state = scheduler.state_dict()
        # 断言状态字典中不包含"_scale_fn_ref"键，以验证该属性是否从状态字典中排除
        self.assertNotIn("_scale_fn_ref", state)
        # 断言状态字典中"_scale_fn_custom"键的值等于`scale_fn`对象的字典形式，以验证自定义缩放函数是否正确存储
        self.assertEqual(state["_scale_fn_custom"], scale_fn.__dict__)
        # 序列化调度器对象的状态字典，以验证其是否可以正确序列化
        pickle.dumps(state)
    def test_cycle_lr_scale_fn_restored_from_state_dict(self):
        adam_opt = Adam(self.net.parameters())

        # Case 1: Built-in mode
        # 创建循环学习率调度器，使用内置模式 "triangular2"
        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, mode="triangular2"
        )
        # 创建另一个循环学习率调度器，加载已保存的状态
        restored_scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False
        )
        restored_scheduler.load_state_dict(scheduler.state_dict())
        # 检查恢复后的调度器模式与原调度器模式是否一致
        self.assertTrue(restored_scheduler.mode == scheduler.mode == "triangular2")
        # 检查是否成功引用了比例函数 `_scale_fn_ref`
        self.assertIsNotNone(restored_scheduler._scale_fn_ref) and self.assertIsNotNone(
            scheduler._scale_fn_ref
        )
        # 检查自定义比例函数是否为 None
        self.assertIs(restored_scheduler._scale_fn_custom, None)
        self.assertIs(scheduler._scale_fn_custom, None)

        # Case 2: Custom `scale_fn`
        # 定义自定义比例函数
        def scale_fn(_):
            return 0.5

        # 创建循环学习率调度器，使用自定义比例函数
        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        # 创建另一个循环学习率调度器，加载已保存的状态
        restored_scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        restored_scheduler.load_state_dict(scheduler.state_dict())
        # 检查是否成功引用了自定义比例函数 `_scale_fn_custom`
        self.assertIs(scheduler._scale_fn_custom, scale_fn)
        self.assertIs(restored_scheduler._scale_fn_custom, scale_fn)

    def test_onecycle_lr_invalid_anneal_strategy(self):
        # 测试无效的退火策略参数
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(
                self.opt, max_lr=1e-3, total_steps=10, anneal_strategy="CATS"
            )

    def test_onecycle_lr_invalid_pct_start(self):
        # 测试无效的起始百分比参数
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3, total_steps=10, pct_start=1.1)

    def test_onecycle_lr_cannot_calculate_total_steps(self):
        # 测试无法计算总步数的情况
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3)

    def test_onecycle_lr_linear_annealing(self):
        # 测试线性退火策略的单周期学习率调度器
        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        # 创建单周期学习率调度器，使用线性退火策略
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
        )
        # 执行单周期学习率调度器的测试
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)
    def test_onecycle_lr_linear_annealing_three_phases(self):
        # 定义目标学习率和动量列表
        lr_target = [1, 9, 17, 25, 17, 9, 1, 0.75, 0.5, 0.25]
        momentum_target = [22, 15, 8, 1, 8, 15, 22, 22, 22, 22]
        
        # 将学习率和动量列表组成列表，用于测试
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        
        # 创建 OneCycleLR 调度器对象，配置相关参数
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            div_factor=25,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
            pct_start=0.4,
            final_div_factor=4,
            three_phase=True,
        )
        
        # 使用测试函数测试 OneCycleLR 调度器的行为
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_cosine_annealing(self):
        # 定义余弦退火函数
        def annealing_cos(start, end, pct):
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out
        
        # 定义目标学习率和动量列表，其中学习率包含余弦退火函数的计算
        lr_target = [
            1,
            13,
            25,
            annealing_cos(25, 0.5, 1 / 7.0),
            annealing_cos(25, 0.5, 2 / 7.0),
            annealing_cos(25, 0.5, 3 / 7.0),
            annealing_cos(25, 0.5, 4 / 7.0),
            annealing_cos(25, 0.5, 5 / 7.0),
            annealing_cos(25, 0.5, 6 / 7.0),
            0.5,
        ]
        
        # 定义目标动量列表，其中动量也包含余弦退火函数的计算
        momentum_target = [
            22,
            11.5,
            1,
            annealing_cos(1, 22, 1 / 7.0),
            annealing_cos(1, 22, 2 / 7.0),
            annealing_cos(1, 22, 3 / 7.0),
            annealing_cos(1, 22, 4 / 7.0),
            annealing_cos(1, 22, 5 / 7.0),
            annealing_cos(1, 22, 6 / 7.0),
            22,
        ]
        
        # 将学习率和动量列表组成列表，用于测试
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        
        # 创建 OneCycleLR 调度器对象，配置相关参数
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
        )
        
        # 使用测试函数测试 OneCycleLR 调度器的行为
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)
    # 测试 OneCycleLR 类的 legacy state_dict 方法
    def test_onecycle_lr_legacy_state_dict(self):
        # 创建 OneCycleLR 调度器对象，设置各种参数
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="cos",
        )
        # 删除 scheduler 对象的 _anneal_func_type 属性
        delattr(scheduler, "_anneal_func_type")
        # 获取调度器对象的状态字典
        state_dict = scheduler.state_dict()
        # 确保状态字典中不包含 "anneal_func_type" 键
        self.assertNotIn("anneal_func_type", state_dict)
        # 向状态字典中添加 "anneal_func" 键，值为 OneCycleLR._annealing_cos 函数
        state_dict["anneal_func"] = OneCycleLR._annealing_cos
        # 使用修改后的状态字典恢复调度器对象的状态
        scheduler.load_state_dict(state_dict)

        # 定义一个余弦退火函数
        def annealing_cos(start, end, pct):
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out

        # 定义预期的学习率目标列表
        lr_target = [
            1,
            13,
            25,
            annealing_cos(25, 0.5, 1 / 7.0),
            annealing_cos(25, 0.5, 2 / 7.0),
            annealing_cos(25, 0.5, 3 / 7.0),
            annealing_cos(25, 0.5, 4 / 7.0),
            annealing_cos(25, 0.5, 5 / 7.0),
            annealing_cos(25, 0.5, 6 / 7.0),
            0.5,
        ]
        # 定义预期的动量目标列表
        momentum_target = [
            22,
            11.5,
            1,
            annealing_cos(1, 22, 1 / 7.0),
            annealing_cos(1, 22, 2 / 7.0),
            annealing_cos(1, 22, 3 / 7.0),
            annealing_cos(1, 22, 4 / 7.0),
            annealing_cos(1, 22, 5 / 7.0),
            annealing_cos(1, 22, 6 / 7.0),
            22,
        ]
        # 创建两个 lr_target 列表的列表
        lr_targets = [lr_target, lr_target]
        # 创建两个 momentum_target 列表的列表
        momentum_targets = [momentum_target, momentum_target]
        # 调用内部测试方法 _test_cycle_lr，验证 OneCycleLR 对象的行为
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    # 测试带有 Adam 优化器的 OneCycleLR 类
    def test_cycle_lr_with_adam(self):
        # 保存旧的优化器对象
        old_opt = self.opt
        # 使用 Adam 优化器，设置不同的学习率和参数组
        self.opt = Adam(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

        # 定义预期的学习率目标列表
        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        # 定义预期的动量目标列表
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        # 创建两个 lr_target 列表的列表
        lr_targets = [lr_target, lr_target]
        # 创建两个 momentum_target 列表的列表
        momentum_targets = [momentum_target, momentum_target]

        # 创建新的 OneCycleLR 对象，使用线性退火策略
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
        )
        # 调用内部测试方法 _test_cycle_lr，验证 OneCycleLR 对象的行为，使用 beta1 参数为 True
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10, use_beta1=True)
        # 将优化器对象恢复为旧的 SGD 优化器
        self.opt = old_opt  # set optimizer back to SGD

    # 测试 LambdaLR 类的功能
    def test_lambda_lr(self):
        # 定义总共的 epochs 数量
        epochs = 10
        # 设置第一个参数组的学习率为 0.05
        self.opt.param_groups[0]["lr"] = 0.05
        # 设置第二个参数组的学习率为 0.4
        self.opt.param_groups[1]["lr"] = 0.4
        # 定义预期的学习率目标列表
        targets = [
            [0.05 * (0.9**x) for x in range(epochs)],
            [0.4 * (0.8**x) for x in range(epochs)],
        ]
        # 创建 LambdaLR 对象，使用 lambda 函数来指定学习率的更新方式
        scheduler = LambdaLR(
            self.opt, lr_lambda=[lambda x1: 0.9**x1, lambda x2: 0.8**x2]
        )
        # 调用内部测试方法 _test，验证 LambdaLR 对象的行为
        self._test(scheduler, targets, epochs)
    # 定义一个测试方法，用于测试多重乘法学习率调度器的功能
    def test_multiplicative_lr(self):
        # 设置训练周期数为10
        epochs = 10
        # 设置第一个参数组的学习率为0.05
        self.opt.param_groups[0]["lr"] = 0.05
        # 设置第二个参数组的学习率为0.4
        self.opt.param_groups[1]["lr"] = 0.4
        # 计算两个目标列表，分别对应于两个参数组的学习率按指数递减的方式
        targets = [
            [0.05 * (0.9**x) for x in range(epochs)],
            [0.4 * (0.8**x) for x in range(epochs)],
        ]
        # 创建一个多重乘法学习率调度器对象
        scheduler = MultiplicativeLR(
            self.opt, lr_lambda=[lambda x1: 0.9, lambda x2: 0.8]
        )
        # 调用测试方法来验证调度器的行为是否符合预期
        self._test(scheduler, targets, epochs)

    # 使用@parametrize装饰器定义一个参数化测试方法，用于测试余弦退火热重启学习率调度器的第一种情况
    @parametrize("T_mult", [1, 2, 4])
    def test_CosineAnnealingWarmRestarts_lr1(self, T_mult):
        # 设置迭代次数为100
        iters = 100
        # 设置最小学习率为1e-10
        eta_min = 1e-10
        # 设置初始周期为10
        T_i = 10
        # 初始当前周期为0
        T_cur = 0
        # 初始化目标列表，用于存储两个参数组的学习率变化情况
        targets = [[0.05], [0.5]]
        # 创建一个余弦退火热重启学习率调度器对象
        scheduler = CosineAnnealingWarmRestarts(
            self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min
        )
        # 循环执行迭代次数次数的计算和调度操作
        for _ in range(1, iters, 1):
            # 当前周期数自增1
            T_cur += 1
            # 如果当前周期数超过了初始周期数T_i
            if T_cur >= T_i:
                # 更新当前周期数为T_cur减去T_i
                T_cur = T_cur - T_i
                # 更新初始周期数T_i为T_mult倍数的T_i
                T_i = int(T_mult) * T_i
            # 计算并更新第一个目标列表的学习率变化情况
            targets[0] += [
                eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            ]
            # 计算并更新第二个目标列表的学习率变化情况
            targets[1] += [
                eta_min + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            ]
        # 调用测试方法来验证调度器的行为是否符合预期
        self._test(scheduler, targets, iters)

    # 定义一个测试方法，用于测试余弦退火热重启学习率调度器的第二种情况
    def test_CosineAnnealingWarmRestarts_lr2(self):
        # 设置迭代次数为30
        iters = 30
        # 设置最小学习率为1e-10
        eta_min = 1e-10
        # 定义T_mults列表，包含[1, 2, 4]三个元素
        T_mults = [1, 2, 4]
        # 遍历T_mults列表中的每一个T_mult值
        for T_mult in T_mults:
            # 设置初始周期数为10
            T_i = 10
            # 初始当前周期数为0
            T_cur = 0
            # 初始化目标列表，用于存储两个参数组的学习率变化情况
            targets = [[0.05], [0.5]]
            # 创建一个余弦退火热重启学习率调度器对象
            scheduler = CosineAnnealingWarmRestarts(
                self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min
            )
            # 使用torch.arange在0.1到iters之间以0.1为步长迭代计算
            for _ in torch.arange(0.1, iters, 0.1):
                # 当前周期数按0.1自增
                T_cur = round(T_cur + 0.1, 1)
                # 如果当前周期数超过了初始周期数T_i
                if T_cur >= T_i:
                    # 更新当前周期数为T_cur减去T_i
                    T_cur = T_cur - T_i
                    # 更新初始周期数T_i为T_mult倍数的T_i
                    T_i = int(T_mult) * T_i
                # 计算并更新第一个目标列表的学习率变化情况
                targets[0] += [
                    eta_min
                    + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
                # 计算并更新第二个目标列表的学习率变化情况
                targets[1] += [
                    eta_min
                    + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
            # 调用测试方法来验证调度器的行为是否符合预期
            self._test_CosineAnnealingWarmRestarts(scheduler, targets, iters)
    # 测试用例：测试 CosineAnnealingWarmRestarts 调度器的学习率变化
    def test_CosineAnnealingWarmRestarts_lr3(self):
        # 不同的 epochs 配置，用于测试多次 T_mult 的效果
        epochs_for_T_mults = [
            [0, 1, 2, 3, 4, 5, 12, 27, 3, 4, 5, 6, 13],
            [0, 1, 2, 3, 4, 5, 25, 32, 33, 34, 80, 81, 3],
            [0, 0.1, 0.2, 0.3, 1.3, 2.3, 17.5, 18.5, 19.5, 29.5, 30.5, 31.5, 50],
        ]
        # 不同的 T_cur 配置，用于测试不同 T_mult 下的当前 T 值效果
        T_curs_for_T_mults = [
            [1, 2, 3, 4, 5, 2, 7, 3, 4, 5, 6, 3],
            [1, 2, 3, 4, 5, 15, 2, 3, 4, 10, 11, 3],
            [0.1, 0.2, 0.3, 1.3, 2.3, 7.5, 8.5, 9.5, 19.5, 20.5, 21.5, 10],
        ]
        # 不同的 T_i 配置，用于测试不同 T_mult 下的总 T 值效果
        T_is_for_T_mults = [
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 20, 40, 40, 40, 80, 80, 10],
            [10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 90],
        ]
        # 最小学习率设置
        eta_min = 1e-10
        # 不同的 T_mult 配置，用于测试多次 T_mult 的效果
        T_mults = [1, 2, 3]
        # 循环遍历不同的 epochs, T_mult, T_cur, T_i 组合
        for epochs, T_mult, T_curs, T_is in zip(
            epochs_for_T_mults, T_mults, T_curs_for_T_mults, T_is_for_T_mults
        ):
            # 初始 lr 配置
            initial_lrs = [group["lr"] for group in self.opt.param_groups]
            # 目标 lr 配置，用于验证调度器的效果
            targets = [
                [lr] * (swa_start + 1) + [swa_lr] * (epochs - swa_start - 1)
                for lr in initial_lrs
            ]
            # 创建 CosineAnnealingWarmRestarts 调度器对象
            scheduler = CosineAnnealingWarmRestarts(
                self.opt, T_0=10, T_mult=T_mult, eta_min=eta_min
            )
            # 遍历 T_cur 和 T_i，计算目标 lr 值
            for T_cur, T_i in zip(T_curs, T_is):
                targets[0] += [
                    eta_min
                    + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
                targets[1] += [
                    eta_min
                    + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
            # 调用测试函数验证 CosineAnnealingWarmRestarts 调度器的效果
            self._test_interleaved_CosineAnnealingWarmRestarts(
                scheduler, targets, epochs
            )

    # 测试用例：测试 SWALR 调度器在没有退火时的学习率变化
    def test_swalr_no_anneal(self):
        # epochs, swa_start, swa_lr 初始化设置
        epochs, swa_start, swa_lr = 10, 5, 0.01
        # 获取初始 lr 配置
        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        # 目标 lr 配置，用于验证 SWALR 调度器的效果
        targets = [
            [lr] * (swa_start + 1) + [swa_lr] * (epochs - swa_start - 1)
            for lr in initial_lrs
        ]
        # 创建 SWALR 调度器对象
        swa_scheduler = SWALR(self.opt, anneal_epochs=1, swa_lr=swa_lr)
        # 调用测试函数验证 SWALR 调度器的效果
        self._test_swalr(swa_scheduler, None, targets, swa_start, epochs)
    def test_swalr_cosine_anneal_after_multiplicative(self):
        # 定义一个测试函数，用于验证 SWA LR 在乘法调度后的余弦退火效果
        # 设置训练周期数、SWA 开始周期、SWA 学习率、退火周期数
        epochs, swa_start, swa_lr, anneal_epochs = 15, 5, 0.01, 5
        mult_factor = 0.9
        # 创建一个乘法调度器，基于 self.opt 的参数组进行调整学习率
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        # 创建一个 SWA LR 调度器，设定退火周期和 SWA 学习率
        swa_scheduler = SWALR(self.opt, anneal_epochs=anneal_epochs, swa_lr=swa_lr)

        def anneal_coef(t):
            # 定义一个退火系数函数，根据当前周期 t 进行计算
            if t + 1 >= anneal_epochs:
                return 0.0
            return (1 + math.cos(math.pi * (t + 1) / anneal_epochs)) / 2

        # 获取初始学习率列表
        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        # 计算 SWA 开始前的目标学习率列表
        targets_before_swa = [
            [lr * mult_factor**i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        # 计算 SWA 结束后的周期数
        swa_epochs = epochs - swa_start - 1
        # 计算每个参数组的目标学习率列表
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs in targets_before_swa
        ]

        # 执行 SWALR 测试函数，验证结果
        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def test_swalr_linear_anneal_after_multiplicative(self):
        # 定义一个测试函数，用于验证 SWA LR 在乘法调度后的线性退火效果
        # 设置训练周期数、SWA 开始周期、各参数组的 SWA 学习率列表、线性退火周期数
        epochs, swa_start, swa_lrs, anneal_epochs = 15, 5, [0.01, 0.02], 4
        mult_factor = 0.9
        # 创建一个乘法调度器，基于 self.opt 的参数组进行调整学习率
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        # 创建一个 SWA LR 调度器，设定退火策略为线性，指定各参数组的 SWA 学习率列表
        swa_scheduler = SWALR(
            self.opt,
            anneal_epochs=anneal_epochs,
            anneal_strategy="linear",
            swa_lr=swa_lrs,
        )

        def anneal_coef(t):
            # 定义一个退火系数函数，根据当前周期 t 进行计算
            if t + 1 >= anneal_epochs:
                return 0.0
            return 1 - (t + 1) / anneal_epochs

        # 获取初始学习率列表
        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        # 计算 SWA 开始前的目标学习率列表
        targets_before_swa = [
            [lr * mult_factor**i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        # 计算 SWA 结束后的周期数
        swa_epochs = epochs - swa_start - 1
        # 计算每个参数组的目标学习率列表
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs, swa_lr in zip(targets_before_swa, swa_lrs)
        ]

        # 执行 SWALR 测试函数，验证结果
        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)
    def _test_swalr(self, swa_scheduler, scheduler, targets, swa_start, epochs):
        # 遍历每个 epoch
        for epoch in range(epochs):
            # 遍历优化器参数组和目标列表
            for param_group, target in zip(self.opt.param_groups, targets):
                # 断言当前 epoch 的学习率与目标学习率相等
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )
            # 如果当前 epoch 大于等于 SWA 开始的 epoch
            if epoch >= swa_start:
                # 执行优化步骤和 SWA 调度器的步进
                self.opt.step()
                swa_scheduler.step()
            # 否则，如果有普通调度器，则执行优化步骤和普通调度器的步进
            elif scheduler is not None:
                self.opt.step()
                scheduler.step()

    def test_swalr_hypers(self):
        # 测试 SWALR 对于不正确的超参数是否引发错误
        with self.assertRaisesRegex(ValueError, "anneal_strategy must"):
            # 尝试创建 SWALR 调度器，期望引发 "anneal_strategy must" 的值错误
            swa_scheduler = SWALR(self.opt, anneal_strategy="exponential", swa_lr=1.0)

        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            # 尝试创建 SWALR 调度器，期望引发 "anneal_epochs must" 的值错误
            swa_scheduler = SWALR(self.opt, anneal_epochs=-1, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            # 尝试创建 SWALR 调度器，期望引发 "anneal_epochs must" 的值错误
            swa_scheduler = SWALR(self.opt, anneal_epochs=1.7, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "swa_lr must"):
            # 尝试创建 SWALR 调度器，期望引发 "swa_lr must" 的值错误
            swa_scheduler = SWALR(self.opt, swa_lr=[1.0, 0.1, 0.01])

    def test_step_lr_state_dict(self):
        # 测试 StepLR 调度器状态字典的正确性
        self._check_scheduler_state_dict(
            lambda: StepLR(self.opt, gamma=0.1, step_size=3),
            lambda: StepLR(self.opt, gamma=0.01 / 2, step_size=1),
        )

    def test_multi_step_lr_state_dict(self):
        # 测试 MultiStepLR 调度器状态字典的正确性
        self._check_scheduler_state_dict(
            lambda: MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9]),
            lambda: MultiStepLR(self.opt, gamma=0.01, milestones=[1, 4, 6]),
        )

    def test_exp_step_lr_state_dict(self):
        # 测试 ExponentialLR 调度器状态字典的正确性
        self._check_scheduler_state_dict(
            lambda: ExponentialLR(self.opt, gamma=0.1),
            lambda: ExponentialLR(self.opt, gamma=0.01),
        )

    def test_cosine_lr_state_dict(self):
        # 测试 CosineAnnealingLR 调度器状态字典的正确性
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min),
            lambda: CosineAnnealingLR(self.opt, T_max=epochs // 2, eta_min=eta_min / 2),
            epochs=epochs,
        )
    # 测试 ReduceLROnPlateau 调度器的状态字典是否正确
    def test_reduce_lr_on_plateau_state_dict(self):
        # 创建 ReduceLROnPlateau 调度器对象，设定模式为最小化，学习率衰减因子为 0.1，容忍度为 2
        scheduler = ReduceLROnPlateau(self.opt, mode="min", factor=0.1, patience=2)
        
        # 模拟多次训练得分，并调用调度器的步进方法
        for score in [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 3.0, 2.0, 1.0]:
            scheduler.step(score)
        
        # 创建一个新的 ReduceLROnPlateau 调度器对象，加载之前调度器的状态字典
        scheduler_copy = ReduceLROnPlateau(
            self.opt, mode="max", factor=0.5, patience=10
        )
        scheduler_copy.load_state_dict(scheduler.state_dict())
        
        # 检查两个调度器对象除了 optimizer 和 is_better 外的所有属性是否一致
        for key in scheduler.__dict__.keys():
            if key not in {"optimizer", "is_better"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    # 测试 LambdaLR 调度器的状态字典是否正确
    def test_lambda_lr_state_dict_fn(self):
        # 创建 LambdaLR 调度器对象，使用 lambda 函数 lr_lambda=lambda x: x
        scheduler = LambdaLR(self.opt, lr_lambda=lambda x: x)
        state = scheduler.state_dict()
        # 验证状态字典中 lr_lambdas 的第一个元素是否为 None
        self.assertIsNone(state["lr_lambdas"][0])

        # 创建一个新的 LambdaLR 调度器对象，加载之前调度器的状态字典
        scheduler_copy = LambdaLR(self.opt, lr_lambda=lambda x: x)
        scheduler_copy.load_state_dict(state)
        
        # 检查两个调度器对象除了 optimizer 和 lr_lambdas 外的所有属性是否一致
        for key in scheduler.__dict__.keys():
            if key not in {"optimizer", "lr_lambdas"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    # 测试 LambdaLR 调度器使用对象作为 lr_lambda 的状态字典是否正确
    def test_lambda_lr_state_dict_obj(self):
        # 创建 LambdaLR 调度器对象，使用对象作为 lr_lambda
        scheduler = LambdaLR(self.opt, lr_lambda=self.LambdaLRTestObject(10))
        state = scheduler.state_dict()
        # 验证状态字典中 lr_lambdas 的第一个元素是否不为 None
        self.assertIsNotNone(state["lr_lambdas"][0])

        # 创建一个新的 LambdaLR 调度器对象，加载之前调度器的状态字典
        scheduler_copy = LambdaLR(self.opt, lr_lambda=self.LambdaLRTestObject(-1))
        scheduler_copy.load_state_dict(state)
        
        # 检查两个调度器对象除了 optimizer 外的所有属性是否一致
        for key in scheduler.__dict__.keys():
            if key not in {"optimizer"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    # 测试 CosineAnnealingWarmRestarts 调度器的状态字典是否正确
    def test_CosineAnnealingWarmRestarts_lr_state_dict(self):
        # 使用 _check_scheduler_state_dict 方法验证状态字典的正确性
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=2),
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=100),
        )

    # 测试 SWALR 调度器的状态字典是否正确
    def test_swa_lr_state_dict(self):
        # 使用 _check_scheduler_state_dict 方法验证状态字典的正确性
        self._check_scheduler_state_dict(
            lambda: SWALR(self.opt, anneal_epochs=3, swa_lr=0.5),
            lambda: SWALR(
                self.opt, anneal_epochs=10, anneal_strategy="linear", swa_lr=5.0
            ),
        )

    # 验证调度器状态字典的一般方法，确保在多次迭代后的一致性
    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        scheduler = constr()
        for _ in range(epochs):
            scheduler.optimizer.step()
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        
        # 检查两个调度器对象除了 optimizer 外的所有属性是否一致
        for key in scheduler.__dict__.keys():
            if key != "optimizer":
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])
        
        # 检查最后学习率是否一致
        self.assertEqual(scheduler.get_last_lr(), scheduler_copy.get_last_lr())
    # 测试获取最后学习率的方法
    def _test_get_last_lr(self, schedulers, targets, epochs=10):
        # 如果 schedulers 是单个 LRScheduler 对象，则转换为列表
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        # 收集所有 schedulers 对应的 optimizer 对象
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        # 迭代执行指定次数的 epochs
        for epoch in range(epochs):
            # 获取每个 scheduler 的最后学习率
            result = [scheduler.get_last_lr() for scheduler in schedulers]
            # 对每个 optimizer 执行一次优化步骤
            [optimizer.step() for optimizer in optimizers]
            # 对每个 scheduler 执行一次调度步骤
            [scheduler.step() for scheduler in schedulers]
            # 构建目标学习率列表，用于后续断言比较
            target = [[t[epoch] for t in targets]] * len(schedulers)
            # 逐一断言每个 scheduler 的最后学习率与目标学习率的一致性
            for t, r in zip(target, result):
                self.assertEqual(
                    t,
                    r,
                    msg=f"LR is wrong in epoch {epoch}: expected {t}, got {r}",
                    atol=1e-5,
                    rtol=0,
                )

    # 测试带有 epoch 参数的方法
    def _test_with_epoch(self, schedulers, targets, epochs=10):
        # 如果 schedulers 是单个 LRScheduler 对象，则转换为列表
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        # 收集所有 schedulers 对应的 optimizer 对象
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        # 迭代执行指定次数的 epochs
        for epoch in range(epochs):
            # 对每个 optimizer 执行一次优化步骤
            [optimizer.step() for optimizer in optimizers]
            # 捕获 epoch 警告，并记录下来
            with warnings.catch_warnings(record=True) as w:
                # 对每个 scheduler 执行一次调度步骤，传入当前 epoch 数
                [
                    scheduler.step(epoch) for scheduler in schedulers
                ]  # step before assert: skip initial lr
                # 检查是否触发了 epoch 弃用警告
                self._check_warning_is_epoch_deprecation_warning(
                    w, num_warnings=len(schedulers)
                )
            # 逐一断言每个参数组的学习率与目标学习率的一致性
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    # 标准测试方法
    def _test(self, schedulers, targets, epochs=10):
        # 如果 schedulers 是单个 LRScheduler 对象，则转换为列表
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        # 迭代执行指定次数的 epochs
        for epoch in range(epochs):
            # 逐一断言每个参数组的学习率与目标学习率的一致性
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )
            # 对每个 scheduler 执行一次调度步骤
            [scheduler.step() for scheduler in schedulers]
    # 测试 CosineAnnealingWarmRestarts 调度器的功能
    def _test_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs=10):
        # 使用 torch.arange 生成指定区间内的浮点数序列，以 0.1 为步长
        for index, epoch in enumerate(torch.arange(0, epochs, 0.1)):
            # 将浮点数保留一位小数并转换为 Python float 类型
            epoch = round(epoch.item(), 1)
            # 更新调度器到指定的 epoch
            scheduler.step(epoch)
            # 遍历优化器的参数组和目标学习率列表，验证每个参数组的学习率是否正确
            for param_group, target in zip(self.opt.param_groups, targets):
                # 使用断言验证学习率的正确性，并给出详细的错误消息
                self.assertEqual(
                    target[index],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[index], param_group["lr"]
                    ),
                    atol=1e-5,  # 允许的绝对误差
                    rtol=0,     # 允许的相对误差
                )

    # 测试 interleaved_CosineAnnealingWarmRestarts 调度器的功能
    def _test_interleaved_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs):
        # 遍历给定的 epochs，逐个更新调度器
        for index, epoch in enumerate(epochs):
            scheduler.step(epoch)
            # 遍历优化器的参数组和目标学习率列表，验证每个参数组的学习率是否正确
            for param_group, target in zip(self.opt.param_groups, targets):
                # 使用断言验证学习率的正确性，并给出详细的错误消息
                self.assertEqual(
                    target[index],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[index], param_group["lr"]
                    ),
                    atol=1e-5,  # 允许的绝对误差
                    rtol=0,     # 允许的相对误差
                )

    # 测试调度器与闭式解形式的比较功能
    def _test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        # 初始化测试环境
        self.setUp()
        targets = []
        # 遍历指定的 epochs，使用闭式解形式调度器更新优化器的学习率
        for epoch in range(epochs):
            closed_form_scheduler.optimizer.step()
            # 捕获 epoch 警告，并验证警告类型
            with warnings.catch_warnings(record=True) as w:
                closed_form_scheduler.step(epoch)
                self._check_warning_is_epoch_deprecation_warning(w)
            # 记录每个 epoch 各参数组的学习率
            targets.append([group["lr"] for group in self.opt.param_groups])
        # 重新设置测试环境
        self.setUp()
        # 再次遍历指定的 epochs，使用常规调度器更新优化器的学习率
        for epoch in range(epochs):
            self.opt.step()
            scheduler.step()
            # 遍历优化器的参数组，验证每个参数组的学习率是否正确
            for i, param_group in enumerate(self.opt.param_groups):
                # 使用断言验证学习率的正确性，并给出详细的错误消息
                self.assertEqual(
                    targets[epoch][i],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, targets[epoch][i], param_group["lr"]
                    ),
                    atol=1e-5,  # 允许的绝对误差
                    rtol=0,     # 允许的相对误差
                )

    # 测试 reduce_lr_on_plateau 调度器的功能
    def _test_reduce_lr_on_plateau(
        self, schedulers, targets, metrics, epochs=10, verbose=False
    ):
        # 如果 schedulers 是单个学习率调度器（LRScheduler 或 ReduceLROnPlateau），转换为列表形式以便统一处理
        if isinstance(schedulers, (LRScheduler, ReduceLROnPlateau)):
            schedulers = [schedulers]
        # 循环执行指定的 epochs 次数
        for epoch in range(epochs):
            # 执行优化器的一步操作
            self.opt.step()
            # 遍历所有的学习率调度器
            for scheduler in schedulers:
                # 如果调度器是 ReduceLROnPlateau 类型，根据当前 epoch 的指标值调整学习率
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics[epoch])
                else:
                    # 否则，按照预设的步长调整学习率
                    scheduler.step()
            # 如果设定了输出详细信息，打印当前 epoch 的学习率
            if verbose:
                print("epoch{}:\tlr={}".format(epoch, self.opt.param_groups[0]["lr"]))
            # 对每个参数组和其对应的目标值进行断言，验证学习率的正确性
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,  # 允许的绝对误差
                    rtol=0,     # 相对误差为零
                )

    def _test_cycle_lr(
        self,
        scheduler,
        lr_targets,
        momentum_targets,
        batch_iterations,
        verbose=False,
        use_beta1=False,
    ):
        # 遍历指定次数的批次迭代
        for batch_num in range(batch_iterations):
            # 如果设置了输出详细信息
            if verbose:
                # 如果优化器参数组中包含动量信息
                if "momentum" in self.opt.param_groups[0].keys():
                    # 打印当前批次的学习率和动量值
                    print(
                        "batch{}:\tlr={},momentum={}".format(
                            batch_num,
                            self.opt.param_groups[0]["lr"],
                            self.opt.param_groups[0]["momentum"],
                        )
                    )
                # 如果使用了 beta1 并且优化器参数组中包含 betas 信息
                elif use_beta1 and "betas" in self.opt.param_groups[0].keys():
                    # 打印当前批次的学习率和 beta1 值
                    print(
                        "batch{}:\tlr={},beta1={}".format(
                            batch_num,
                            self.opt.param_groups[0]["lr"],
                            self.opt.param_groups[0]["betas"][0],
                        )
                    )
                else:
                    # 打印当前批次的学习率
                    print(
                        "batch{}:\tlr={}".format(
                            batch_num, self.opt.param_groups[0]["lr"]
                        )
                    )

            # 遍历优化器的参数组、学习率目标值和动量目标值
            for param_group, lr_target, momentum_target in zip(
                self.opt.param_groups, lr_targets, momentum_targets
            ):
                # 断言当前批次的学习率与目标学习率一致
                self.assertEqual(
                    lr_target[batch_num],
                    param_group["lr"],
                    msg="LR is wrong in batch_num {}: expected {}, got {}".format(
                        batch_num, lr_target[batch_num], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

                # 如果使用了 beta1 并且参数组中包含 betas 信息
                if use_beta1 and "betas" in param_group.keys():
                    # 断言当前批次的 beta1 与目标值一致
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group["betas"][0],
                        msg="Beta1 is wrong in batch_num {}: expected {}, got {}".format(
                            batch_num,
                            momentum_target[batch_num],
                            param_group["betas"][0],
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
                # 如果参数组中包含动量信息
                elif "momentum" in param_group.keys():
                    # 断言当前批次的动量与目标动量一致
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group["momentum"],
                        msg="Momentum is wrong in batch_num {}: expected {}, got {}".format(
                            batch_num,
                            momentum_target[batch_num],
                            param_group["momentum"],
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
            # 执行优化器的一步更新
            self.opt.step()
            # 调度器执行一步调度
            scheduler.step()
    def test_cosine_then_cyclic(self):
        # https://github.com/pytorch/pytorch/issues/21965
        # 定义最大学习率、基础学习率和优化器学习率
        max_lr = 0.3
        base_lr = 0.1
        optim_lr = 0.5

        # 创建一个线性模型和一个随机梯度下降优化器
        model = torch.nn.Linear(2, 1)
        optimizer = SGD(model.parameters(), lr=optim_lr)

        # 创建余弦退火学习率调度器和循环学习率调度器
        lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=0.1
        )
        lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=1, step_size_down=3
        )

        # 进行40次优化步骤
        for i in range(40):
            optimizer.step()
            # 根据当前步数决定使用哪个学习率调度器
            if i <= lr_scheduler_1.T_max:
                lr_scheduler_1.step()
            else:
                lr_scheduler_2.step()
            # 获取当前优化器的学习率
            last_lr = optimizer.param_groups[0]["lr"]

        # 断言最终学习率不超过最大学习率
        self.assertLessEqual(last_lr, max_lr)

    @parametrize(
        "LRClass",
        [
            # 下面列出的是不同的学习率调度器类及其参数
            partial(LambdaLR, lr_lambda=lambda e: e // 10),
            partial(MultiplicativeLR, lr_lambda=lambda: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            partial(ExponentialLR, gamma=0.9),
            lambda opt, **kwargs: SequentialLR(
                opt,
                schedulers=[ConstantLR(opt), ConstantLR(opt)],
                milestones=[2],
                **kwargs,
            ),
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            ReduceLROnPlateau,
            partial(CyclicLR, base_lr=0.01, max_lr=0.1),
            partial(CosineAnnealingWarmRestarts, T_0=20),
            partial(OneCycleLR, max_lr=0.01, total_steps=10),
        ],
    )
    def test_lr_scheduler_verbose_deprecation_warning(self, LRClass):
        """Check that a deprecating warning with verbose parameter."""
        # 检查带有 verbose 参数的废弃警告
        with self.assertWarnsOnceRegex(
            UserWarning, "The verbose parameter is deprecated"
        ):
            LRClass(self.opt, verbose=True)

        with self.assertWarnsOnceRegex(
            UserWarning, "The verbose parameter is deprecated"
        ):
            LRClass(self.opt, verbose=False)

        # 当 verbose 参数为默认值时，不应该有警告被触发
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LRClass(self.opt)
    @parametrize(
        "LRClass",
        [
            partial(LambdaLR, lr_lambda=lambda e: e // 10),
            partial(MultiplicativeLR, lr_lambda=lambda: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            partial(ExponentialLR, gamma=0.9),
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            lambda opt, **kwargs: ChainedScheduler(
                schedulers=[ConstantLR(opt), ConstantLR(opt)], **kwargs
            ),
            lambda opt, **kwargs: SequentialLR(
                opt,
                schedulers=[ConstantLR(opt), ConstantLR(opt)],
                milestones=[2],
                **kwargs,
            ),
            ReduceLROnPlateau,
            partial(CyclicLR, base_lr=0.01, max_lr=0.1),
            partial(OneCycleLR, max_lr=0.01, total_steps=10, anneal_strategy="linear"),
            partial(CosineAnnealingWarmRestarts, T_0=20),
        ],
    )
    @parametrize("weights_only", [True, False])
    # 定义测试函数，用于测试学习率调度器的状态字典加载
    def test_lr_scheduler_state_dict_load(self, LRClass, weights_only):
        # 创建指定类型的学习率调度器对象
        scheduler = LRClass(self.opt)
        # 获取调度器当前状态的字典表示
        state_dict = scheduler.state_dict()

        # 使用临时文件保存状态字典，并读取出来以进行比较
        with tempfile.TemporaryFile() as f:
            torch.save(state_dict, f)
            f.seek(0)
            state_dict_loaded = torch.load(f, weights_only=weights_only)
            # 断言保存前后状态字典内容一致
            self.assertEqual(state_dict, state_dict_loaded)
            # 确保状态字典可以成功加载到新创建的调度器对象上
            scheduler2 = LRClass(self.opt)
            scheduler2.load_state_dict(state_dict_loaded)
            self.assertEqual(scheduler2.state_dict(), state_dict)

    @parametrize(
        "LRClass",
        [
            partial(LambdaLR, lr_lambda=lambda e: e // 10),
            partial(MultiplicativeLR, lr_lambda=lambda e: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            partial(ExponentialLR, gamma=0.9),
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            partial(CosineAnnealingWarmRestarts, T_0=20),
        ],
    )
    # 测试初始学习率为常数的情况
    def test_constant_initial_lr(self, LRClass):
        # 设置初始学习率为0.1，并创建一个优化器对象
        lr = torch.as_tensor(0.1)
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        # 使用指定的学习率调度器类创建调度器对象
        sch = LRClass(opt)

        # 复制原始的参数组信息
        ori_param_groups = copy.deepcopy(opt.param_groups)

        # 进行两次优化步骤，每次步骤后学习率减少为原来的十分之一
        for i in range(2):
            opt.step()
            sch.step(i)
            lr.multiply_(0.1)
            # 检查每个参数组的初始学习率是否与原始一致，以及调度器的基础学习率列表是否正确
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(sch.base_lrs, [0.1])
    def test_constant_initial_params_cyclelr(self):
        # Test that the initial learning rate is constant

        # 初始化学习率为常数
        lr = torch.as_tensor(0.1)
        # 最大学习率为常数
        max_lr = torch.as_tensor(0.2)
        # 基础动量为常数
        base_momentum = torch.as_tensor(0.8)
        # 最大动量为常数
        max_momentum = torch.as_tensor(0.9)
        
        # 使用随机参数创建 SGD 优化器
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        # 创建 CyclicLR 学习率调度器
        sch = CyclicLR(
            opt,
            base_lr=lr,
            max_lr=max_lr,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )
        # 复制原始参数组
        ori_param_groups = copy.deepcopy(opt.param_groups)

        # 循环两次
        for i in range(2):
            # 学习率减半
            lr.multiply_(0.5)
            # 最大学习率减半
            max_lr.multiply_(0.5)
            # 基础动量减半
            base_momentum.multiply_(0.5)
            # 最大动量减半
            max_momentum.multiply_(0.5)
            # 执行优化步骤
            opt.step()
            # 调度器前进一步
            sch.step(i)
            # 检查每个参数组与原始参数组的初始学习率和动量是否相等
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(group["max_momentum"], ori_group["max_momentum"])
                self.assertEqual(group["base_momentum"], ori_group["base_momentum"])
                # 检查调度器的基础学习率列表是否为 [0.1]
                self.assertEqual(sch.base_lrs, [0.1])
                # 检查调度器的最大学习率列表是否为 [0.2]
                self.assertEqual(sch.max_lrs, [0.2])
                # 检查参数组的最大动量是否为 0.9
                self.assertEqual(group["max_momentum"], 0.9)
                # 检查参数组的基础动量是否为 0.8
                self.assertEqual(group["base_momentum"], 0.8)

    def test_constant_initial_params_onecyclelr(self):
        # Test that the initial learning rate is constant

        # 初始化学习率为常数
        lr = torch.as_tensor(0.1)
        # 基础动量为常数
        base_momentum = torch.as_tensor(0.85)
        # 最大动量为常数
        max_momentum = torch.as_tensor(0.95)
        
        # 使用随机参数创建 SGD 优化器
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        # 创建 OneCycleLR 学习率调度器
        sch = OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=10,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )
        # 复制原始参数组
        ori_param_groups = copy.deepcopy(opt.param_groups)

        # 循环两次
        for i in range(2):
            # 学习率减半
            lr.multiply_(0.5)
            # 基础动量减半
            base_momentum.multiply_(0.5)
            # 最大动量减半
            max_momentum.multiply_(0.5)
            # 执行优化步骤
            opt.step()
            # 调度器前进一步
            sch.step(i)

            # 检查每个参数组与原始参数组的初始学习率、最大学习率、最小学习率和动量是否相等
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(group["max_lr"], ori_group["max_lr"])
                self.assertEqual(group["min_lr"], ori_group["min_lr"])
                self.assertEqual(group["max_momentum"], ori_group["max_momentum"])
                self.assertEqual(group["base_momentum"], ori_group["base_momentum"])
                # 检查参数组的最大动量是否为 0.95
                self.assertEqual(group["max_momentum"], 0.95)
                # 检查参数组的基础动量是否为 0.85
                self.assertEqual(group["base_momentum"], 0.85)
    def test_constant_initial_params_swalr(self):
        # 定义一个测试函数，验证初始学习率是否保持不变
        lr = torch.as_tensor(0.1)  # 创建一个张量，表示初始学习率为0.1
        swa_lr = torch.as_tensor(0.05)  # 创建一个张量，表示SWA学习率为0.05
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)  # 使用SGD优化器，设置初始学习率为lr
        sch = SWALR(opt, swa_lr=swa_lr)  # 创建一个SWALR学习率调度器，设置SWA学习率为swa_lr
        ori_param_groups = copy.deepcopy(opt.param_groups)  # 复制优化器的参数组信息

        for i in range(2):
            lr.multiply_(0.5)  # 将当前学习率lr减半
            swa_lr.multiply_(0.5)  # 将SWA学习率swa_lr减半
            opt.step()  # 执行优化器的一步更新
            sch.step()  # 执行SWALR学习率调度器的一步更新
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])  # 断言：当前学习率保持与初始学习率一致
                self.assertEqual(group["swa_lr"], ori_group["swa_lr"])  # 断言：SWA学习率与初始SWA学习率一致
                self.assertEqual(group["swa_lr"], 0.05)  # 断言：SWA学习率应为0.05
                self.assertEqual(sch.base_lrs, [0.1])  # 断言：SWALR学习率调度器的基础学习率应为[0.1]
# 使用给定的参数化测试类实例化参数化测试
instantiate_parametrized_tests(TestLRScheduler)

# 如果当前脚本被作为主程序执行，则打印警告信息
if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
```