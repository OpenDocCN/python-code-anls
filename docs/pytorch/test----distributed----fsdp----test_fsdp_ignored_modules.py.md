# `.\pytorch\test\distributed\fsdp\test_fsdp_ignored_modules.py`

```
# Owner(s): ["oncall: distributed"]

import functools  # 导入 functools 模块，用于高阶函数操作
import math  # 导入 math 模块，提供数学函数支持
import sys  # 导入 sys 模块，提供与 Python 解释器交互的功能

import torch  # 导入 PyTorch 深度学习框架
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入分布式训练相关的工具模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch import distributed as dist  # 导入 PyTorch 分布式包
from torch.distributed._composable import fully_shard  # 导入 PyTorch 分布式计算相关函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 PyTorch FSDP 模块
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state  # 导入 FSDP 模块的通用函数
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy  # 导入 FSDP 模块的包装策略
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试相关的分布式功能
from torch.testing._internal.common_fsdp import (  # 导入测试相关的 FSDP 初始化和配置
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (  # 导入测试相关的通用工具函数
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if not dist.is_available():  # 检查当前环境是否支持分布式训练
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)  # 如果不支持，则退出测试

if TEST_WITH_DEV_DBG_ASAN:  # 如果使用了 dev-asan 调试模式
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)  # 由于已知问题，跳过 dev-asan 调试模式

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(3, 5)  # 输入维度为 3，输出维度为 5 的线性层
        layer1_modules = [
            torch.nn.Linear(5, 4),  # 输入维度为 5，输出维度为 4 的线性层
            torch.nn.Linear(4, 4),  # 输入维度为 4，输出维度为 4 的线性层
            torch.nn.Linear(4, 4),  # 输入维度为 4，输出维度为 4 的线性层
        ]
        self.layer1 = torch.nn.Sequential(*layer1_modules)  # 将以上线性层作为序列连接起来
        self.layer2 = torch.nn.Linear(4, 2)  # 输入维度为 4，输出维度为 2 的线性层
        self.layer3 = torch.nn.Linear(2, 2)  # 输入维度为 2，输出维度为 2 的线性层
        self.relu = torch.nn.ReLU()  # ReLU 激活函数

    def forward(self, x):
        z = self.relu(self.layer0(x))  # 第一层线性变换后经过 ReLU 激活函数
        z = self.relu(self.layer1(z))  # 第二层线性变换后经过 ReLU 激活函数
        z = self.relu(self.layer2(z))  # 第三层线性变换后经过 ReLU 激活函数
        z = self.relu(self.layer3(z))  # 第四层线性变换后经过 ReLU 激活函数
        return z  # 返回输出结果

    def get_input(self, device):
        return (torch.randn((8, 3)).to(device),)  # 生成随机输入数据，大小为 (8, 3)，放到指定设备上

    def get_loss(self, input, output):
        return output.sum()  # 计算输出的和作为损失函数值

    def run_backward(self, loss):
        loss.backward()  # 执行反向传播计算梯度


class IgnoredModule(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))  # 随机初始化权重参数

    def forward(self, x):
        return x @ self.weight  # 矩阵乘法操作


class ModelWithIgnoredModules(Model):
    """Adds a variable number of :class:`IgnoredModule` to ``self.layer1``."""

    def __init__(self, num_ignored: int) -> None:
        assert num_ignored >= 0
        super().__init__()
        layer1_modules = (
            [torch.nn.Linear(5, 4), torch.nn.Linear(4, 4)]  # 固定的线性层
            + [IgnoredModule(4, 4) for _ in range(num_ignored)]  # 可变数量的 IgnoredModule 模块
            + [torch.nn.Linear(4, 4)]  # 固定的线性层
        )
        self.layer1 = torch.nn.Sequential(*layer1_modules)  # 构建包含可变数量 IgnoredModule 的序列层


class TestFSDPIgnoredModules(FSDPTest):
    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 2)  # 返回当前环境的 GPU 数量与 2 中较小的值
    # 定义一个私有方法，用于训练模型
    def _train_model(self, model, optim, num_iters, device=torch.device("cuda")):
        # 循环执行给定次数的迭代
        for _ in range(num_iters):
            # 如果模型是 FSDP 类型的实例，则获取其 module 属性作为真正的模型
            module = model.module if isinstance(model, FSDP) else model
            # 获取模型的输入数据，根据设备选择 GPU 或 CPU
            inp = module.get_input(device)
            # 使用模型进行前向计算得到输出
            output = model(*inp)
            # 计算模型的损失并将其移动到指定的设备上
            loss = module.get_loss(inp, output).to(device)
            # 执行反向传播优化模型参数
            module.run_backward(loss)
            # 执行优化器的步骤，更新模型参数
            optim.step()

    # 装饰器函数，如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer(self):
        """Tests that ignored modules' parameters are not flattened for a
        transformer model with shared parameters."""
        # 运行子测试，测试忽略模块参数对于具有共享参数的 transformer 模型是否正确处理
        self.run_subtests(
            {
                "use_orig_params": [False, True],  # 使用原始参数或者转换后的参数
                "ignore_modules": [True, False],  # 是否忽略模块参数（相对于忽略状态）
                "use_auto_wrap": [False, True],  # 是否使用自动包装
                "composable": [False],  # 是否是可组合的
            },
            self._test_ignored_modules_transformer,  # 执行具体测试方法
        )

    # 装饰器函数，如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer_composable(self):
        """Tests that ignored modules' parameters are not flattened for a
        transformer model with shared parameters."""
        # 运行子测试，测试忽略模块参数对于具有共享参数的 transformer 模型是否正确处理（可组合情况）
        self.run_subtests(
            {
                "use_orig_params": [True],  # 使用原始参数
                "ignore_modules": [True, False],  # 是否忽略模块参数（相对于忽略状态）
                "use_auto_wrap": [False, True],  # 是否使用自动包装
                "composable": [True],  # 是否是可组合的
            },
            self._test_ignored_modules_transformer,  # 执行具体测试方法
        )

    # 定义一个私有方法，用于测试忽略模块参数对于具有共享参数的 transformer 模型是否正确处理
    def _test_ignored_modules_transformer(
        self,
        use_orig_params: bool,
        ignore_modules: bool,  # 是否忽略模块参数（相对于忽略状态）
        use_auto_wrap: bool,
        composable: bool,
        ```
    # 当条件满足时，初始化一个使用FSDP封装的Transformer模型，并配置FSDP忽略`nn.Transformer`模块的参数
    model: nn.Module = TransformerWithSharedParams.init(
        self.process_group,
        FSDPInitMode.NO_FSDP,
        CUDAInitMode.CUDA_BEFORE,
        deterministic=True,
    )
    # 设置FSDP的关键字参数
    fsdp_kwargs = {"process_group": self.process_group}
    
    if use_auto_wrap:
        # 如果需要自动包装，则取消共享输出投影权重和嵌入权重，以便正确地自动包装每个线性层
        model.output_proj.weight = nn.Parameter(model.output_proj.weight.clone())
        fsdp_kwargs[
            "policy" if composable else "auto_wrap_policy"
        ] = ModuleWrapPolicy({nn.Linear})
    
    if ignore_modules:
        # 如果需要忽略某些模块，则将transformer模块添加到被忽略的模块列表中
        fsdp_kwargs["ignored_modules"] = [model.transformer]
    else:
        # 否则，将transformer模块的参数添加到被忽略的状态列表中
        fsdp_kwargs["ignored_states"] = list(model.transformer.parameters())
    
    # 根据是否可组合选择封装器类
    wrapper_cls = fully_shard if composable else FSDP
    # 使用指定的FSDP关键字参数创建封装后的模型
    wrapped_model = wrapper_cls(model, **fsdp_kwargs)
    
    # 初始化一个未使用FSDP封装的Transformer模型，以便后续比较参数数量
    nonwrapped_model: nn.Module = TransformerWithSharedParams.init(
        self.process_group,
        FSDPInitMode.NO_FSDP,
        CUDAInitMode.CUDA_BEFORE,
        deterministic=True,
    )
    
    if use_auto_wrap:
        # 如果需要自动包装，则取消共享输出投影权重，以便正确地自动包装每个线性层
        nonwrapped_model.output_proj.weight = nn.Parameter(
            nonwrapped_model.output_proj.weight.clone()
        )
    
    # 计算未封装模型的总参数数量
    total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
    # 计算transformer模块的参数数量
    ignored_numel = sum(
        p.numel() for p in nonwrapped_model.transformer.parameters()
    )
    # 计算未忽略的参数数量
    nonignored_numel = total_numel - ignored_numel
    
    fsdp_managed_numel = 0
    # 使用FSDP管理所有参数，并计算FSDP管理的参数数量
    with FSDP.summon_full_params(wrapped_model):
        for handle in traversal_utils._get_fsdp_handles(wrapped_model):
            flat_param = handle.flat_param
            flat_param_numel = flat_param.numel()
            if composable or use_orig_params:
                # 减去由于对齐填充而贡献的数量
                padding_numel = sum(
                    numel
                    for (numel, is_padding) in zip(
                        flat_param._numels_with_padding, flat_param._is_padding_mask
                    )
                    if is_padding
                )
                flat_param_numel -= padding_numel
            fsdp_managed_numel += flat_param_numel
    
    # 断言FSDP管理的参数数量等于未忽略的参数数量
    self.assertEqual(fsdp_managed_numel, nonignored_numel)
    
    # 检查封装后的模型是否可以运行几次迭代
    optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
    self._train_model(wrapped_model, optim, 3)
    # 定义一个测试方法，用于测试传递具有嵌套FSDP模块的模块，确保不报错，并且仍然忽略非FSDP模块的参数。
    def test_ignored_modules_nested(self):
        """Tests that passing a module with nested FSDP modules does not
        error and still ignores non-FSDP modules' parameters."""
        # 运行子测试，传递参数字典和测试方法 _test_ignored_modules_nested
        self.run_subtests(
            {
                "use_orig_params": [False, True],
                "ignore_modules": [True, False],
                "composable": [False],
            },
            self._test_ignored_modules_nested,
        )

    # 如果GPU数小于2，则跳过此测试方法
    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_nested_composable(self):
        """Tests that passing a module with nested FSDP modules does not
        error and still ignores non-FSDP modules' parameters."""
        # 运行子测试，传递参数字典和测试方法 _test_ignored_modules_nested
        self.run_subtests(
            {
                "use_orig_params": [True],
                "ignore_modules": [True, False],
                "composable": [True],
            },
            self._test_ignored_modules_nested,
        )

    # 定义一个私有方法，用于测试传递具有嵌套FSDP模块的模块，确保不报错，并且仍然忽略非FSDP模块的参数。
    def _test_ignored_modules_nested(
        self, use_orig_params: bool, ignore_modules: bool, composable: bool
    ):
        # Initialize an FSDP-wrapped nested model that first wraps the nested
        # sequential's second linear layer (`layer1[1]`) and then wraps the
        # overall model while ignoring the nested sequential (`layer1`)
        # 初始化一个使用FSDP包装的嵌套模型，首先包装嵌套顺序模型的第二个线性层（`layer1[1]`），
        # 然后包装整个模型，同时忽略嵌套顺序模型（`layer1`）
        model = Model().cuda()
        fsdp_fn = (
            fully_shard
            if composable
            else functools.partial(FSDP, use_orig_params=use_orig_params)
        )
        # 将模型的第二个线性层（`layer1[1]`）使用fsdp_fn进行包装
        model.layer1[1] = fsdp_fn(model.layer1[1])
        if ignore_modules:
            # 如果忽略模块被设置为True，则对整个模型应用fsdp_fn，忽略`model.layer1`
            wrapped_model = fsdp_fn(model, ignored_modules=[model.layer1])
        else:
            # 否则，对模型应用fsdp_fn，忽略`model.layer1`的参数状态
            wrapped_model = fsdp_fn(
                model, ignored_states=list(model.layer1.parameters())
            )
        # 检查包装后的模型的扁平参数是否不包括被忽略的嵌套顺序模型的参数
        nonwrapped_model = Model()
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(p.numel() for p in nonwrapped_model.layer1.parameters())
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            # 获取扁平参数
            flat_param = (
                wrapped_model.params[0]
                if not composable
                else _get_module_fsdp_state(wrapped_model).params[0]
            )
            flat_param_numel = flat_param.numel()
            if composable or use_orig_params:
                # 如果是可组合或使用原始参数，则减去由对齐填充贡献的数量
                padding_numel = sum(
                    numel
                    for (numel, is_padding) in zip(
                        flat_param._numels_with_padding, flat_param._is_padding_mask
                    )
                    if is_padding
                )
                flat_param_numel -= padding_numel
                self.assertEqual(flat_param_numel, nonignored_numel)
            # 断言扁平参数的数量等于非被忽略的参数数量
            self.assertEqual(flat_param_numel, nonignored_numel)
        # 检查是否可以运行几次迭代
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    def test_ignored_states_auto_wrap(self):
        transformer_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={nn.Sequential}
        )
        # 运行子测试，测试自动包装忽略状态
        self.run_subtests(
            {
                "policy": [transformer_policy, ModuleWrapPolicy((nn.Sequential,))],
                "ignore_bias": [True, False],
            },
            self._test_ignored_states_auto_wrap,
        )
    # 定义一个测试方法，用于测试自动包装时忽略的状态
    def _test_ignored_states_auto_wrap(self, policy, ignore_bias: bool):
        # 创建一个 CUDA 模型对象
        model = Model().cuda()
        # 定义被忽略的状态，初始为模型第一层第二个元素的权重
        ignored_states = [model.layer1[1].weight]
        # 如果需要忽略偏置，则将模型第一层第二个元素的偏置也加入被忽略的状态列表
        if ignore_bias:
            ignored_states.append(model.layer1[1].bias)
        # 使用 FSDP 封装模型
        fsdp_model = FSDP(
            model,
            # 设置为 False，避免内部平坦参数填充的复杂性
            use_orig_params=False,
            auto_wrap_policy=policy,
            ignored_states=ignored_states,
        )
        # 创建一个参考模型
        ref_model = Model()
        # 计算预期的未分片的 layer1 的元素数量
        expected_layer1_unsharded_numel = (
            sum(p.numel() for p in ref_model.layer1.parameters())
            - ref_model.layer1[1].weight.numel()
        )
        # 如果忽略偏置，则从预期未分片数量中减去偏置的元素数量
        if ignore_bias:
            expected_layer1_unsharded_numel -= ref_model.layer1[1].bias.numel()
        # 计算预期的未分片的整个模型的元素数量
        expected_model_unsharded_numel = sum(
            p.numel() for p in ref_model.parameters()
        ) - sum(p.numel() for p in ref_model.layer1.parameters())
        # 计算预期的分片后的 layer1 的元素数量
        expected_layer1_sharded_numel = math.ceil(
            expected_layer1_unsharded_numel / self.world_size
        )
        # 计算预期的分片后的整个模型的元素数量
        expected_model_sharded_numel = math.ceil(
            expected_model_unsharded_numel / self.world_size
        )
        # 断言确保 fsdp_model 的 layer1 模块的平坦参数数量不超过预期的分片后的数量
        self.assertLessEqual(
            fsdp_model.layer1.module._flat_param.numel(), expected_layer1_sharded_numel
        )
        # 断言确保 fsdp_model 整个模型的平坦参数数量不超过预期的分片后的数量
        self.assertLessEqual(
            fsdp_model.module._flat_param.numel(), expected_model_sharded_numel
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("composable", [True, False])
    # 定义一个测试方法，用于测试不合法的忽略模块
    def test_ignored_modules_invalid(self, composable):
        """Tests that passing an FSDP module as an ignored module or the
        top-level module itself errors."""
        # 创建一个 CUDA 模型对象
        model = Model().cuda()
        # 根据是否可组合设置包装类
        wrap_cls = FSDP if composable else fully_shard
        # 使用 wrap_cls 包装模型的 layer1 层
        model.layer1 = wrap_cls(model.layer1)
        # 断言传递一个 FSDP 模块作为被忽略模块会引发 ValueError 异常
        with self.assertRaises(
            ValueError,
            msg="`ignored_modules` should not include FSDP modules",
        ):
            wrap_cls(model, ignored_modules=[model.layer1])
        # 断言传递顶层模块自身作为被忽略模块会引发 UserWarning 警告
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Trying to ignore the top-level module passed into "
            "the FSDP constructor itself will result in all parameters being "
            "ignored",
        ):
            # fully_shard 不允许对同一模型进行两次包装，因此在此处创建一个新的局部模型。
            new_model = Model().cuda()
            wrap_cls(new_model, ignored_modules=[new_model])

    @skip_if_lt_x_gpu(2)
    def test_diff_ignored_modules_across_ranks(self):
        """
        Tests ignoring different modules across ranks.

        Args:
            pass_ignored_modules_to_root (bool): If ``False``, does not pass
                any ignored modules (including those already ignored in child
                FSDP instances) to the root FSDP instance; if ``True``, passes
                all ignored modules (representing a superset of the children's
                ignored modules) to the root FSDP instance.
        """
        # 运行子测试，传入不同的参数组合
        self.run_subtests(
            {
                "pass_ignored_modules_to_root": [False, True],
                "ignore_modules": [True, False],
                "composable": [True, False],
            },
            self._test_diff_ignored_modules_across_ranks,
        )

    def _test_diff_ignored_modules_across_ranks(
        self,
        pass_ignored_modules_to_root: bool,
        ignore_modules: bool,
        composable: bool,
    ):
        # 为了测试在不同的等级中忽略不同的 `FlatParameter` 枚举，
        # 我们将 `layer3` 包装在 FSDP 中，其中 `layer3` 被注册为一个模块，
        # 在 `layer1` 之后，`layer1` 中有多个被忽略的模块
        wrap_cls = FSDP if composable else fully_shard
        # 创建一个带有被忽略模块数量为 `self.rank + 1` 的 ModelWithIgnoredModules 模型，并放在 CUDA 设备上
        model = ModelWithIgnoredModules(num_ignored=self.rank + 1).cuda()
        # 找出 `layer1` 中所有的 IgnoredModule 模块
        layer1_ignored_modules = [
            m for m in model.layer1.modules() if isinstance(m, IgnoredModule)
        ]
        # 如果 `ignore_modules` 为 True，构建忽略模块的参数字典；否则构建忽略状态的参数字典
        ignore_kwargs = (
            {"ignored_modules": layer1_ignored_modules}
            if ignore_modules
            else {
                "ignored_states": (
                    p for m in layer1_ignored_modules for p in m.parameters()
                )
            }
        )
        # 使用 wrap_cls 封装 `model.layer1`，并传入忽略参数
        model.layer1 = wrap_cls(model.layer1, **ignore_kwargs)
        # 使用 wrap_cls 封装 `model.layer3`
        model.layer3 = wrap_cls(model.layer3)
        # 如果 `pass_ignored_modules_to_root` 为 True，则找出所有的被忽略模块，并传递给根 FSDP 实例
        model_ignored_modules = (
            [m for m in model.modules() if isinstance(m, IgnoredModule)]
            if pass_ignored_modules_to_root
            else []
        )
        # 如果 `ignore_modules` 为 True，构建忽略模块的参数字典；否则构建忽略状态的参数字典
        ignore_kwargs_top = (
            {"ignored_modules": model_ignored_modules}
            if ignore_modules
            else {
                "ignored_states": {
                    p for m in model_ignored_modules for p in m.parameters()
                }
            }
        )
        # 使用 wrap_cls 封装整个 `model`，并传入顶层的忽略参数
        wrapped_model = wrap_cls(model, **ignore_kwargs_top)
        # 使用 Adam 优化器优化被封装后模型的参数
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        # 调用 _train_model 方法来训练封装后的模型，训练 3 次
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    @parametrize("ignore_modules", [True, False])
    @parametrize("composable", [True, False])
    def test_ignored_modules_not_under_wrapped_root(
        self, ignore_modules: bool, composable: bool
    ):
        # 测试被包装根模块下是否不包含被忽略模块
    ):
        # 创建一个新的模型对象，并将其放置在 GPU 上
        model = Model().cuda()
        # 获取 model.layer1 中除了第一个子模块外的其余子模块
        ignored_modules = list(model.layer1.children())[1:]

        # 根据 ignore_modules 的值确定 ignore_kwargs 的内容
        ignore_kwargs = (
            {"ignored_modules": ignored_modules}
            if ignore_modules
            else {
                # 构建一个包含所有被忽略参数的集合
                "ignored_states": {p for m in ignored_modules for p in m.parameters()}
            }
        )

        # 根据 composable 的值选择合适的 wrapper 类
        wrap_cls = FSDP if composable else fully_shard

        # 使用 wrap_cls 对 model.layer1 进行重新封装，传入 ignore_kwargs 中的参数
        model.layer1 = wrap_cls(
            model.layer1,
            **ignore_kwargs,
        )
        # 使用 wrap_cls 对 model.layer3 进行重新封装，传入 ignore_kwargs 中的参数
        model.layer3 = wrap_cls(
            model.layer3,
            # ignored modules/parameters contains submodule under model.layer1, which
            # is out of the local root model.layer3.
            **ignore_kwargs,
        )

        # 使用 Adam 优化器来优化模型参数，设置学习率为 1e-3
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        # 调用 self._train_model 方法来训练模型，训练周期为 3
        self._train_model(model, optim, 3)

    @skip_if_lt_x_gpu(1)
    def test_ignored_states_check(self):
        """
        Tests that passing invalid ``ignored_modules`` or ``ignored_states``
        raises an appropriate error.
        """
        # 运行一系列子测试，传入 ignore_modules 参数为 [True, False]，以及一个测试方法的引用
        self.run_subtests(
            {"ignore_modules": [True, False]},
            self._test_ignored_states_check,
        )

    def _test_ignored_states_check(self, ignore_modules: bool):
        # 创建一个新的模型对象，并将其放置在 GPU 上
        model = Model().cuda()
        # 获取 model.layer1 中除了第一个子模块外的其余子模块
        ignored_modules = list(model.layer1.children())[1:]
        # 创建一个集合，包含所有被忽略参数
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        # 创建一个集合，包含所有被忽略状态（参数和子模块）
        ignored_states = ignored_params.union(set(ignored_modules))
        if ignore_modules:
            # 检查传入的 `ignored_modules` 不是 `nn.Module` 类型列表时是否引发错误
            with self.assertRaisesRegex(
                ValueError,
                "ignored_modules expects nn.Module list elements but got types "
                r"\[<class 'torch.nn.parameter.Parameter'>\]",
            ):
                # 尝试使用 FSDP 对象并传入 ignored_params 参数
                FSDP(model, ignored_modules=ignored_params)
            # 检查同时传入 `ignored_modules` 和 `ignored_states` 是否引发错误（仅在 `ignore_modules=True` 时执行）
            with self.assertRaisesRegex(
                ValueError,
                "Cannot pass both ignored_modules and ignored_states at the same time",
            ):
                # 尝试使用 FSDP 对象并同时传入 ignored_modules 和 ignored_params 参数
                FSDP(
                    model,
                    ignored_modules=ignored_modules,
                    ignored_states=ignored_params,
                )
        else:
            # 检查传入的 `ignored_states` 不是全部为 `nn.Parameter` 或全部为 `nn.Module` 类型时是否引发错误
            with self.assertRaisesRegex(
                ValueError,
                "ignored_states expects all nn.Parameter or all nn.Module list "
                r"elements but got types \[<class 'torch.nn.modules.linear.Linear'>, "
                r"<class 'torch.nn.parameter.Parameter'>\]",
            ):
                # 尝试使用 FSDP 对象并传入 ignored_states 参数
                FSDP(model, ignored_states=ignored_states)
# 实例化一个带参数的测试对象 `TestFSDPIgnoredModules`
instantiate_parametrized_tests(TestFSDPIgnoredModules)

# 如果当前脚本作为主程序运行，调用 `run_tests()` 函数执行测试
if __name__ == "__main__":
    run_tests()
```