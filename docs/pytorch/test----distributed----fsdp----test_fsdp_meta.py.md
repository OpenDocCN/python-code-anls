# `.\pytorch\test\distributed\fsdp\test_fsdp_meta.py`

```
# Owner(s): ["oncall: distributed"]

import itertools  # 导入 itertools 库，用于生成迭代器的函数
import sys  # 导入 sys 库，提供对解释器的访问

from typing import Union  # 导入 Union 类型提示，用于指定多种类型的参数

import torch  # 导入 PyTorch 深度学习框架
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision  # 导入 FSDP 和 MixedPrecision 类
from torch.distributed.fsdp.wrap import (
    always_wrap_policy as always_wrap,  # 导入 always_wrap 策略函数，并重命名为 always_wrap
    enable_wrap,  # 导入 enable_wrap 函数，用于启用包装
    ModuleWrapPolicy,  # 导入 ModuleWrapPolicy 类，定义模块包装策略
    wrap,  # 导入 wrap 函数，用于包装模块
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试的 GPU 数量检查函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入 FSDP 测试相关的类和函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入函数，用于实例化带参数化的测试用例
    parametrize,  # 导入 parametrize 装饰器，用于参数化测试用例
    run_tests,  # 导入函数，用于运行测试
    skip_but_pass_in_sandcastle_if,  # 导入函数，用于在 Sandcastle 中跳过测试但仍通过
    TEST_WITH_DEV_DBG_ASAN,  # 导入测试标志，用于指示是否使用开发时地址污染检查器
)

_TORCHDISTX_AVAIL = True  # 设置变量标志，指示是否安装了 torchdistx 扩展库
try:
    from torchdistx import deferred_init  # 尝试导入 deferred_init 函数
except ImportError:
    _TORCHDISTX_AVAIL = False  # 如果导入失败，则标志为 False，表示未安装 torchdistx 扩展库


if not dist.is_available():  # 检查分布式通信是否可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出提示信息到标准错误流
    sys.exit(0)  # 退出程序，返回状态码 0 表示正常退出

if TEST_WITH_DEV_DBG_ASAN:  # 检查是否使用开发时地址污染检查器
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 输出相关问题的提示信息到标准错误流
    )
    sys.exit(0)  # 退出程序，返回状态码 0 表示正常退出


def _reset_params_if_meta(is_meta: bool, model: nn.Module):
    # 根据 is_meta 的值判断是否进行模型参数重置
    # 对于 torchdistX 初始化，我们不需要调用 reset_params，因为
    # deferred_init(model).materialize() 等效于 model()。
    if is_meta:  # 如果是元模型
        for module in model.modules():  # 遍历模型的所有子模块
            # 假设一个模块具有 `reset_parameters()` 当且仅当它直接管理参数或缓冲区
            if hasattr(module, "reset_parameters"):  # 如果模块有 reset_parameters 方法
                module.reset_parameters()  # 调用模块的 reset_parameters 方法重置参数


class MyLinear(nn.Linear):
    """
    Linear layer with deterministic reset_parameters for testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类的构造方法初始化线性层

    def reset_parameters(self, *args, **kwargs):
        torch.manual_seed(42)  # 设置随机种子为 42
        with torch.no_grad():
            # 使用依赖于形状的初始化方法
            torch.nn.init.xavier_uniform_(self.weight, 1.0)  # 使用 Xavier 初始化方法初始化权重


class MyBuffer(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()  # 调用父类的构造方法初始化模块
        self.register_buffer("buf", torch.empty((3, 3), device=device))  # 注册一个设备相关的缓冲区

    def reset_parameters(self, *args, **kwargs):
        torch.manual_seed(42)  # 设置随机种子为 42
        # 使用依赖于形状的初始化方法
        torch.nn.init.xavier_uniform_(self.buf, 0.5)  # 使用 Xavier 初始化方法初始化缓冲区


class MyModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()  # 调用父类的构造方法初始化模块
        self.lin1 = MyLinear(2, 2, bias=False, device=device)  # 创建 MyLinear 实例 lin1
        self.lin2 = MyLinear(2, 2, bias=False, device=device)  # 创建 MyLinear 实例 lin2
        self.buf_mod = MyBuffer(device)  # 创建 MyBuffer 实例 buf_mod

    def forward(self, x):
        return self.lin2(self.lin1(x))  # 模型的前向传播逻辑


class NestedModel(nn.Module):
    # 初始化函数，用于初始化类的实例
    def __init__(self, device):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个自定义的线性层实例 lin1，输入维度为 2，输出维度为 2，无偏置项，使用给定的设备
        self.lin1 = MyLinear(2, 2, bias=False, device=device)
        # 对 lin1 进行包装（wrap）处理
        self.lin1 = wrap(self.lin1)
        # 创建另一个自定义的线性层实例 lin2，输入维度为 2，输出维度为 2，无偏置项，使用给定的设备
        self.lin2 = MyLinear(2, 2, bias=False, device=device)
        # 创建一个自定义模型实例 l3，使用给定的设备
        self.l3 = MyModel(device=device)
        # 对 l3 进行包装（wrap）处理
        self.l3 = wrap(self.l3)

    # 前向传播函数，接收输入 x，并返回计算结果
    def forward(self, x):
        # 执行嵌套调用：先对 x 应用 lin1，再对结果应用 lin2，最后对结果应用 l3
        return self.l3(self.lin2(self.lin1(x)))
# 定义一个函数用于初始化带有重置参数功能的模块，以及使用设备"meta"初始化的示例
def _init_with_reset_params(module: nn.Module):
    # 检查模块中是否存在元数据状态的参数或缓冲区
    has_meta_states = any(
        t.is_meta
        for t in itertools.chain(
            module.parameters(recurse=False), module.buffers(recurse=False)
        )
    )
    # 如果模块包含元数据状态
    if has_meta_states:
        # 将设备设置为当前 CUDA 设备
        device = torch.device("cuda", torch.cuda.current_device())
        # 使用设备"meta"调用模块的to_empty方法，不递归
        module.to_empty(device=device, recurse=False)
        # 重置模块的参数
        module.reset_parameters()


# 定义一个函数用于基于 torchdistX 的延迟模块初始化示例，使用materialize_module函数
def _init_with_torchdistX(module: nn.Module):
    # 断言 torchdistX 是否可用
    assert _TORCHDISTX_AVAIL

    # 定义一个检查函数，用于判断是否不是 FSDP 类型
    def check_fn(k):
        return not isinstance(k, FSDP)

    # 使用deferred_init模块的materialize_module函数来实例化模块
    deferred_init.materialize_module(module, check_fn=check_fn)


# 定义一个测试类，继承自FSDPTest，用于测试带有元设备的FSDP
class TestFSDPWithMetaDevice(FSDPTest):

    # 定义一个属性，返回世界大小为2
    @property
    def world_size(self):
        return 2

    # 定义一个属性，返回进程组，默认使用c10d库获取默认组
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    # 定义一个方法，用于比较两个FSDP对象
    def _compare_fsdp(self, fsdp1, fsdp2):
        # 使用FSDP类的summon_full_params上下文管理器，对fsdp1和fsdp2的参数进行全参数召唤
        with FSDP.summon_full_params(fsdp1):
            with FSDP.summon_full_params(fsdp2):
                # 遍历两个FSDP对象的参数，并检查它们是否在数值上接近
                for p1, p2 in zip(fsdp1.parameters(), fsdp2.parameters()):
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")
    # 定义一个测试方法，用于在元设备上测试简单模型的行为，支持传入元模块和初始化函数
    def _test_simple_model_with_meta_device(self, meta_module_fn, init_fn=None):
        # 在元设备上创建模型并使用FSDP进行包装
        model = meta_module_fn()
        # 检查模型的第一个参数是否为元参数
        is_meta = next(model.parameters()).is_meta
        # 使用FSDP包装模型，设置自动包装策略为always_wrap，并可选地初始化参数
        fsdp_meta = FSDP(
            model,
            auto_wrap_policy=always_wrap,
            param_init_fn=init_fn,
        )

        # 使用SGD优化器对元模型的参数进行优化
        meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)

        # 测试确保与常规FSDP方法中相同的模型参数
        regular = MyModel(device="cuda")
        _reset_params_if_meta(is_meta, regular)
        # 使用FSDP包装常规模型，并使用SGD优化器对其参数进行优化
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)
        regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)

        # 比较元FSDP和常规FSDP的行为
        self._compare_fsdp(fsdp_meta, fsdp_regular)

        # 创建输入张量，放置在cuda设备上
        inp = torch.randn(10, 2, device="cuda")
        # 对元FSDP模型进行前向传播、计算损失、反向传播，并执行优化步骤
        fsdp_meta(inp).sum().backward()
        # 对常规FSDP模型进行类似的操作
        fsdp_regular(inp).sum().backward()
        # 执行元优化器的优化步骤
        meta_opt.step()
        # 执行常规优化器的优化步骤
        regular_opt.step()
        # 再次比较元FSDP和常规FSDP的行为
        self._compare_fsdp(fsdp_meta, fsdp_regular)

        # 测试如果所有子模块都包含在单个FSDP单元中，则元初始化是否起作用
        model = meta_module_fn()
        # 使用FSDP包装模型，并使用指定的初始化函数初始化参数
        fsdp_meta = FSDP(model, param_init_fn=init_fn)
        meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)
        # 创建常规模型
        regular = MyModel(device="cuda")
        _reset_params_if_meta(is_meta, regular)
        # 使用FSDP包装常规模型
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)
        regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)

        # 执行一次前向传播、反向传播和优化步骤
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        # 再次比较元FSDP和常规FSDP的行为
        self._compare_fsdp(fsdp_meta, fsdp_regular)

    # 标记为跳过测试，如果GPU数少于2，则跳过
    @skip_if_lt_x_gpu(2)
    def test_simple_model_with_meta_device_reset_params(self):
        # 定义一个返回元设备上模型的函数
        def meta_module_fn():
            return MyModel(device="meta")

        # 调用_test_simple_model_with_meta_device方法，传入元模块函数和初始化函数
        self._test_simple_model_with_meta_device(
            meta_module_fn, _init_with_reset_params
        )

    # 标记为跳过测试，如果GPU数少于2，则跳过
    @skip_if_lt_x_gpu(2)
    def test_simple_model_with_meta_device_default_init(self):
        # 定义一个返回元设备上模型的函数
        def meta_module_fn():
            return MyModel(device="meta")

        # 调用_test_simple_model_with_meta_device方法，传入元模块函数
        self._test_simple_model_with_meta_device(meta_module_fn)

    # 标记为跳过测试，如果GPU数少于2，并且_TORCHDISTX_AVAIL为假，则跳过
    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    def test_simple_model_with_torchdistX_default_init(self):
        # 定义一个返回在cuda设备上延迟初始化后的模型的函数
        def meta_module_fn():
            return deferred_init.deferred_init(MyModel, device="cuda")

        # 调用_test_simple_model_with_meta_device方法，传入延迟初始化函数
        self._test_simple_model_with_meta_device(meta_module_fn)

    # 标记为跳过测试，如果GPU数少于2，并且_TORCHDISTX_AVAIL为假，则跳过
    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    # 定义一个测试方法，测试带有 torchdistX 初始化函数的简单模型
    def test_simple_model_with_torchdistX_init_fn(self):
        # 定义一个元模块函数，返回通过 deferred_init 初始化后的 MyModel 对象，使用 CUDA 设备
        def meta_module_fn():
            return deferred_init.deferred_init(MyModel, device="cuda")

        # 调用通用的测试方法，测试带有元设备的简单模型
        self._test_simple_model_with_meta_device(
            meta_module_fn, init_fn=_init_with_torchdistX
        )

    # 定义一个测试嵌套模型带有元设备的方法
    def _test_nested_model_with_meta_device(
        self, auto_wrap, meta_module_fn, init_fn=None
    ):
        # 如果需要自动包装
        if auto_wrap:
            # 通过元模块函数获取模块对象
            module = meta_module_fn()
            # 判断模块是否是元模块，并获取其缓冲区是否是元缓冲区
            is_meta = (
                next(module.parameters()).is_meta or next(module.buffers()).is_meta
            )
            # 使用 FSDP 包装模块，设置自动包装策略为 always_wrap，并应用参数初始化函数
            fsdp_meta = FSDP(
                module,
                auto_wrap_policy=always_wrap,
                param_init_fn=init_fn,
            )
            # 使用 SGD 优化器优化元设备的参数
            meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)
            # 创建一个常规 NestedModel 对象，使用 CUDA 设备
            module_regular = NestedModel(device="cuda")
            # 如果模块是元模块，则重置参数
            _reset_params_if_meta(is_meta, module_regular)
            # 使用 FSDP 包装常规模块，设置自动包装策略为 always_wrap
            fsdp_regular = FSDP(
                module_regular,
                auto_wrap_policy=always_wrap,
            )
            # 使用 SGD 优化器优化常规设备的参数
            regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)
        else:
            # 否则，使用 enable_wrap 包装模块
            with enable_wrap(
                wrapper_cls=FSDP,
                param_init_fn=init_fn,
            ):
                # 通过元模块函数获取模块对象
                module = meta_module_fn()
                # 判断模块是否是元模块
                is_meta = next(module.parameters()).is_meta
                # 尽管非 FSDP 模块也会被初始化，因为它们将成为较大 FSDP 单元的一部分
                fsdp_meta = wrap(module)
                # 使用 SGD 优化器优化元设备的参数
                meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)

            # 初始化并重置参数，以便在包装之前，重置参数与元设备的初始化相匹配
            module_regular = NestedModel(device="cuda")
            _reset_params_if_meta(is_meta, module_regular)
            # 使用 enable_wrap 包装模块
            with enable_wrap(wrapper_cls=FSDP):
                # 对特定的线性层进行包装
                module_regular.lin1 = wrap(module_regular.lin1)
                module_regular.l3 = wrap(module_regular.l3)
                # 使用 wrap 包装常规模块
                fsdp_regular = wrap(module_regular)
                # 使用 SGD 优化器优化常规设备的参数
                regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)

        # 在训练之前比较两个 FSDP 对象
        self._compare_fsdp(fsdp_meta, fsdp_regular)
        # 创建输入张量，使用 CUDA 设备
        inp = torch.randn(10, 2, device="cuda")
        # 对元设备执行前向传播和反向传播
        fsdp_meta(inp).sum().backward()
        # 对常规设备执行前向传播和反向传播
        fsdp_regular(inp).sum().backward()
        # 元设备优化器执行一步优化
        meta_opt.step()
        # 常规设备优化器执行一步优化
        regular_opt.step()
        # 再次比较两个 FSDP 对象
        self._compare_fsdp(fsdp_meta, fsdp_regular)

    # 用于测试嵌套模型带有元设备重置参数的方法
    @skip_if_lt_x_gpu(2)
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_meta_device_reset_params(self, auto_wrap):
        # 定义一个元模块函数，返回使用 meta 设备的 NestedModel 对象
        def meta_module_fn():
            return NestedModel(device="meta")

        # 调用测试嵌套模型带有元设备的方法
        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap,
            meta_module_fn=meta_module_fn,
            init_fn=_init_with_reset_params,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("auto_wrap", [True, False])
    # 定义测试函数，用于测试带有默认初始化和元数据设备的嵌套模型
    def test_nested_model_with_meta_device_default_init(self, auto_wrap):
        # 定义一个函数，返回一个使用 "meta" 设备的 NestedModel 对象
        def meta_module_fn():
            return NestedModel(device="meta")

        # 调用测试函数 _test_nested_model_with_meta_device，传入自动包装参数和 meta_module_fn 函数
        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap,
            meta_module_fn=meta_module_fn,
        )

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 在 Sandcastle 环境中，如果 torchdistX 不可用，则跳过测试，并附带说明信息
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    # 参数化测试函数，测试嵌套模型与 torchdistX 默认初始化
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_torchdistX_default_init(self, auto_wrap):
        # 定义一个函数，返回一个使用 "cuda" 设备的 NestedModel 对象，使用延迟初始化
        def meta_module_fn():
            return deferred_init.deferred_init(NestedModel, device="cuda")

        # 调用测试函数 _test_nested_model_with_meta_device，传入自动包装参数和 meta_module_fn 函数
        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap, meta_module_fn=meta_module_fn
        )

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 在 Sandcastle 环境中，如果 torchdistX 不可用，则跳过测试，并附带说明信息
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    # 参数化测试函数，测试嵌套模型与使用 torchdistX 初始化函数
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_torchdistX_init_fn(self, auto_wrap):
        # 定义一个函数，返回一个使用 "cuda" 设备的 NestedModel 对象，使用延迟初始化
        def meta_module_fn():
            return deferred_init.deferred_init(NestedModel, device="cuda")

        # 调用测试函数 _test_nested_model_with_meta_device，传入自动包装参数、meta_module_fn 函数和初始化函数 _init_with_torchdistX
        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap,
            meta_module_fn=meta_module_fn,
            init_fn=_init_with_torchdistX,
        )

    # 定义测试函数，测试传入错误参数的情况
    def _test_bad_arg(self, meta_module_fn):
        # 使用 meta_module_fn 函数创建模型对象
        mod = meta_module_fn()
        # 断言抛出 ValueError 异常，异常信息包含 "to be callable"
        with self.assertRaisesRegex(ValueError, "to be callable"):
            # 尝试使用参数初始化函数为模型 mod 提供参数 42，应当引发异常
            FSDP(mod, param_init_fn=42)

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 在 Sandcastle 环境中，如果 torchdistX 不可用，则跳过测试，并附带说明信息
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    # 测试函数，测试在使用 torchdistX 时传入错误参数的情况
    def test_bad_arg_torchdistx(self):
        # 定义一个函数，返回一个在 "cuda" 设备上延迟初始化的 NestedModel 对象
        def meta_module_fn():
            return deferred_init.deferred_init(NestedModel, "cuda")

        # 调用测试函数 _test_bad_arg，传入 meta_module_fn 函数
        self._test_bad_arg(meta_module_fn)

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试函数，测试在使用 meta 设备时传入错误参数的情况
    def test_bad_arg_meta(self):
        # 定义一个函数，返回一个使用 "meta" 设备的 NestedModel 对象
        def meta_module_fn():
            return NestedModel(device="meta")

        # 调用测试函数 _test_bad_arg，传入 meta_module_fn 函数
        self._test_bad_arg(meta_module_fn)

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_meta_device_with_mixed_precision(self):
        """
        Tests meta device initialization with a ``param_init_fn`` when
        specifying mixed precision with ``param_dtype=torch.float32``.
        """

        class FakeLinear(nn.Module):
            def __init__(
                self, in_dim: int, out_dim: int, device: Union[torch.device, str]
            ) -> None:
                super().__init__()
                # 初始化一个参数为随机张量的权重参数
                self.weight = nn.Parameter(
                    torch.randn((in_dim, out_dim), device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 前向传播函数，计算输入张量 x 与权重参数的矩阵乘积
                return x @ self.weight

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 初始化两个线性层，指定设备为 "meta"
                self.lin1 = nn.Linear(5, 5, device="meta")
                self.lin2 = FakeLinear(5, 5, device="meta")
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 模型的前向传播，先通过 lin1 -> relu -> lin2 的顺序计算
                return self.lin2(self.relu(self.lin1(x)))

            def _module_init_fn(self, module: nn.Module):
                if isinstance(module, nn.Linear):
                    # 对线性层的参数进行初始化：权重正态分布，偏置为零
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

        def _param_init_fn(module: nn.Module) -> None:
            # 设置模块的设备为 "cuda"，并应用模型的初始化函数 _module_init_fn
            # 这里的 `model` 变量是外部作用域中的 Model 类的实例
            module.to_empty(device=torch.device("cuda"))
            module.apply(model._module_init_fn)

        # 创建一个 Model 类的实例
        model = Model()
        
        # 使用 FSDP 包装模型，用于混合精度训练
        FSDP(
            model,
            # 指定自动包装策略，对 nn.Linear 类型的模块进行包装
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            # 指定混合精度训练的参数数据类型和减少数据类型
            mixed_precision=MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float16
            ),
            # 指定参数初始化函数为 _param_init_fn
            param_init_fn=_param_init_fn,
            # 指定设备 id 为当前 CUDA 设备的 id
            device_id=torch.cuda.current_device(),
        )
# 实例化带有元设备的参数化测试，使用 TestFSDPWithMetaDevice 类进行测试
instantiate_parametrized_tests(TestFSDPWithMetaDevice)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```