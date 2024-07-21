# `.\pytorch\test\lazy\test_ts_opinfo.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块和库
import functools  # 提供函数式编程支持
import itertools  # 提供高效的迭代工具
import os  # 提供与操作系统交互的功能
from pathlib import Path  # 提供处理路径的功能
from typing import Sequence  # 提供类型提示支持
from unittest import skip  # 提供测试跳过功能

import yaml  # 提供YAML文件的读取和解析支持

import torch  # PyTorch核心库
import torch._lazy  # PyTorch内部的懒加载模块
import torch._lazy.config  # 懒加载模块的配置
import torch._lazy.ir_cache  # 懒加载模块的IR缓存
import torch._lazy.metrics  # 懒加载模块的指标
import torch._lazy.ts_backend  # 懒加载模块的时间序列后端
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,  # 实例化设备类型的测试
    ops,  # 操作函数
)
from torch.testing._internal.common_methods_invocations import op_db  # 操作数据库
from torch.testing._internal.common_utils import run_tests, TestCase  # 运行测试和测试用例
from torch.testing._internal.jit_utils import JitTestCase  # JIT测试用例


torch._lazy.ts_backend.init()  # 初始化懒加载模块的时间序列后端


def get_test_device():
    # 根据环境变量选择测试设备，若有环境变量"LTC_TS_CUDA"则选择CUDA，否则选择CPU
    return "cuda" if "LTC_TS_CUDA" in os.environ else "cpu"


def remove_suffixes(l):
    # 移除列表中每个元素的后缀，后缀以"."分隔
    return [x.split(".")[0] for x in l]


def init_lists():
    # 获取当前脚本所在的绝对路径
    path_to_script = Path(os.path.abspath(os.path.dirname(__file__)))
    # 构建时间序列原生函数定义文件的路径
    TS_NATIVE_FUNCTIONS_PATH = (
        path_to_script.parent.parent / "aten/src/ATen/native/ts_native_functions.yaml"
    )
    # 使用安全加载器打开并解析YAML文件
    with open(TS_NATIVE_FUNCTIONS_PATH) as f:
        yaml_ts = yaml.load(f, yaml.SafeLoader)
    # 获取支持的时间序列操作的列表，并去除文件扩展名
    LAZY_OPS_LIST = set(
        remove_suffixes(
            itertools.chain(
                yaml_ts["full_codegen"], yaml_ts["supported"], yaml_ts["autograd"]
            )
        )
    )
    # 检查是否支持符号整数后缀
    HAS_SYMINT_SUFFIX = yaml_ts["symint"]
    # 回退操作列表，用于在必要时进行降级处理
    FALLBACK_LIST = {"clamp"}
    # 跳过运行时错误的操作列表
    SKIP_RUNTIME_ERROR_LIST = {
        "index_select",  # 空输出大小不支持
        "clone",  # clone分解？
        # 与生成布尔值相关的一般ASAN失败。
        # https://github.com/pytorch/pytorch/issues/74519
        # https://github.com/pytorch/pytorch/issues/63034
        "nonzero",  # ASAN失败（粘贴：P501906539）
        "all",  # ASAN失败
        "any",  # ASAN失败
        "logdet",  # ASAN失败
    }
    # 跳过不正确结果的操作列表
    SKIP_INCORRECT_RESULTS_LIST = {
        "squeeze",  # 值超出范围
        "t",  # 值超出范围
        "transpose",  # 值超出范围
        "bernoulli",  # 不正确的结果
        "pow",  # 不正确的结果
        "addcdiv",  # 不正确的结果（在CI上而非本地？）
    }
    # 这些操作直接出现在ts_native_functions.yaml中，
    # 但在核心中运行功能化版本的复合内核。
    # 这意味着我们不希望这些操作直接显示在LTC指标中。
    FUNCTIONAL_DECOMPOSE_LIST = {
        "diag_embed",
        "block_diag",
        "new_empty_strided",
        "narrow_copy",
        "pixel_shuffle",
        "pixel_unshuffle",
        "select_backward",
        "_trilinear",
        "linalg_inv_ex",
        "linalg_pinv.atol_rtol_tensor",
        "logsumexp",
    }
    # 对于某些操作，我们不支持所有变体。在这里我们使用formatted_name
    # 来唯一标识变体。
    SKIP_VARIANT_LIST = {"norm_nuc", "min_reduction_with_dim"}
    # 返回包含以下常量的元组：
    # LAZY_OPS_LIST: 懒操作列表
    # FALLBACK_LIST: 回退列表
    # SKIP_RUNTIME_ERROR_LIST: 跳过运行时错误列表
    # SKIP_INCORRECT_RESULTS_LIST: 跳过不正确结果列表
    # FUNCTIONAL_DECOMPOSE_LIST: 函数分解列表
    # HAS_SYMINT_SUFFIX: 具有符号整数后缀
    # SKIP_VARIANT_LIST: 跳过变体列表
    return (
        LAZY_OPS_LIST,
        FALLBACK_LIST,
        SKIP_RUNTIME_ERROR_LIST,
        SKIP_INCORRECT_RESULTS_LIST,
        FUNCTIONAL_DECOMPOSE_LIST,
        HAS_SYMINT_SUFFIX,
        SKIP_VARIANT_LIST,
    )
(
    # 初始化各种列表，并将返回的元组解包给对应的变量
    LAZY_OPS_LIST,
    FALLBACK_LIST,
    SKIP_RUNTIME_ERROR_LIST,
    SKIP_INCORRECT_RESULTS_LIST,
    FUNCTIONAL_DECOMPOSE_LIST,
    HAS_SYMINT_SUFFIX,
    SKIP_VARIANT_LIST,
) = init_lists()

# 设置随机数种子为42，确保结果的可重复性
torch.manual_seed(42)


def clone_move(t):
    # 设定设备为"lazy"
    dev = "lazy"
    # 对输入张量进行深拷贝，并移动到指定设备上，保留梯度信息
    copy_t = t.detach().clone().requires_grad_(True).to(device=dev)
    return copy_t


class TestLazyTensor(JitTestCase):
    @skip("Disable until autograd supports symints")
    def testConvolutionBackward(self):
        # 获取测试设备
        test_device = get_test_device()
        # 创建需要梯度的随机输入张量
        inp = torch.rand(1, 3, 128, 128, device=test_device, requires_grad=True)
        # 对输入张量进行深拷贝并移动到指定设备上
        inp_copy = clone_move(inp)
        # 创建不需要梯度的随机梯度张量
        grad = torch.rand(1, 32, 121, 121, device=test_device)  # no requires_grad
        # 对梯度张量进行深拷贝并移动到指定设备上
        grad_copy = clone_move(grad)
        # 创建需要梯度的随机权重张量
        weight = torch.rand(32, 3, 8, 8, device=test_device, requires_grad=True)
        # 对权重张量进行深拷贝并移动到指定设备上
        weight_copy = clone_move(weight)
        # 创建需要梯度的随机偏置张量
        bias = torch.rand(32, device=test_device, requires_grad=True)
        # 对偏置张量进行深拷贝并移动到指定设备上
        bias_copy = clone_move(bias)

        # 运行即时模式下的卷积操作
        conv_out = torch.nn.functional.conv2d(inp, weight, bias)
        # 计算即时模式下的梯度
        (inp_grad, weight_grad, bias_grad) = torch.autograd.grad(
            [conv_out], [inp, weight, bias], [grad]
        )

        # 运行惰性模式下的卷积操作
        conv_copy_out = torch.nn.functional.conv2d(inp_copy, weight_copy, bias_copy)
        # 计算惰性模式下的梯度
        (inp_copy_grad, weight_copy_grad, bias_copy_grad) = torch.autograd.grad(
            [conv_copy_out], [inp_copy, weight_copy, bias_copy], [grad_copy]
        )

        # 检查数值的一致性
        torch.testing.assert_close(bias_copy_grad.cpu(), bias_grad.cpu())

        torch.testing.assert_close(weight_copy_grad.cpu(), weight_grad.cpu())
        torch.testing.assert_close(inp_copy_grad.cpu(), inp_grad.cpu())

    def test_view_mark_step_preserved(self):
        # 获取测试设备
        test_device = get_test_device()
        # 创建张量输入
        inp = torch.rand(4, device=test_device)
        # 对输入张量进行深拷贝并移动到指定设备上
        inp_lazy = clone_move(inp)

        def foo(x, *, mark_step):
            # 对输入张量进行视图操作
            y = x.view(2, 2)
            # 对张量进行加法操作
            y.add_(1)
            # 对输入张量进行加法操作
            z = x + x

            # 如果 mark_step 为真，则调用惰性操作标记函数
            if mark_step:
                torch._lazy.mark_step()

            # 在 mark_step 调用后，y 和 x 应继续保持别名关系
            y.add_(1)
            return x

        # 调用 foo 函数，不进行 mark_step 操作
        out_ref = foo(inp, mark_step=False)
        # 调用 foo 函数，进行 mark_step 操作
        out = foo(inp_lazy, mark_step=True)
        # 对 out 进行 CPU 同步以处理待处理的突变
        torch.testing.assert_close(out_ref.cpu(), out.cpu())

    def test_tensor_ctr(self):
        # 获取测试设备
        test_device = get_test_device()
        # 创建张量输入
        inp = torch.tensor([[1, 2, 3, 4, 5]], device=test_device)
        # 创建懒惰设备上的张量输入
        inp_lazy = torch.tensor([[1, 2, 3, 4, 5]], device="lazy")

        def foo(x):
            # 调用视图操作以确保发生功能化包装
            return x.view(-1)

        # 调用 foo 函数，对正常设备上的输入进行操作
        out_ref = foo(inp)
        # 调用 foo 函数，对懒惰设备上的输入进行操作
        out = foo(inp_lazy)
        # 对 out 进行 CPU 同步以处理待处理的突变
        torch.testing.assert_close(out_ref.cpu(), out.cpu())


class TestLazyOpInfo(TestCase):
    # 定义一个装饰器 ops，用于标记测试方法，该方法用于测试延迟执行的操作
    @ops(
        [
            op
            for op in op_db
            if op.name in LAZY_OPS_LIST
            and op.name not in SKIP_RUNTIME_ERROR_LIST
            and op.name not in FUNCTIONAL_DECOMPOSE_LIST
            and op.formatted_name not in SKIP_VARIANT_LIST
        ],
        allowed_dtypes=(torch.float,),
    )
    # 定义测试方法 test_dispatched_to_lazy，接受 device, dtype, op 作为参数
    def test_dispatched_to_lazy(self, device, dtype, op):
        # 定义函数 get_name，返回操作的完整名称，包括变体测试名称（如果有）
        def get_name(op):
            l = [op.name]
            if op.variant_test_name != "":
                l.append(op.variant_test_name)
            return ".".join(l)

        # 设置全局变量
        global HAS_SYMINT_SUFFIX, FALLBACK_LIST
        # 从操作中获取样本输入，用于 lazy 模式的测试，不需要梯度
        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        sample = next(iter(samples))
        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs
        # 标记一个 lazy 操作的步骤
        torch._lazy.mark_step()
        # 等待设备上的操作完成
        torch._lazy.wait_device_ops()
        # 重置 lazy 操作的指标
        torch._lazy.metrics.reset()

        # 执行操作，获取结果
        r = op(*args, **kwargs)
        # 再次标记一个 lazy 操作的步骤
        torch._lazy.mark_step()
        # 再次等待设备上的操作完成
        torch._lazy.wait_device_ops()
        # 根据操作是否在 FALLBACK_LIST 中选择前缀
        prefix = "aten" if op.name in FALLBACK_LIST else "lazy"
        # 如果操作在 HAS_SYMINT_SUFFIX 中，则添加 "_symint" 后缀
        symint_suffix = "_symint" if op.name in HAS_SYMINT_SUFFIX else ""
        # 检查操作的完整名称是否在去除后缀的指标名称中
        found = f"{prefix}::{op.name}{symint_suffix}" in remove_suffixes(
            torch._lazy.metrics.counter_names()
        )
        # 如果未找到，则检查操作的别名
        if not found:
            for alias in op.aliases:
                alias_found = (
                    f"{prefix}::{alias.name}{symint_suffix}"
                    in remove_suffixes(torch._lazy.metrics.counter_names())
                )
                found = found or alias_found
                if found:
                    break
        # 断言操作是否被正确调度执行
        self.assertTrue(found)

    @ops(
        [
            op
            for op in op_db
            if op.name in LAZY_OPS_LIST
            and op.name not in SKIP_RUNTIME_ERROR_LIST | SKIP_INCORRECT_RESULTS_LIST
        ],
        allowed_dtypes=(torch.float,),
    )  # noqa: B950
    # 定义一个测试方法，用于验证操作的正确性，接受设备、数据类型和操作作为参数
    def test_correctness(self, device, dtype, op):
        # 获取测试设备
        test_device = get_test_device()

        # 定义一个函数，将输入对象克隆到指定设备上
        def clone_to_device(input, dev):
            # 如果是 Torch 张量，则进行分离、克隆并移动到指定设备
            if isinstance(input, torch.Tensor):
                return input.detach().clone().to(device=dev)
            # 如果是序列且不是字符串，则递归地克隆序列中的每个元素到指定设备
            if isinstance(input, Sequence) and not isinstance(input, str):
                return tuple(map(functools.partial(clone_to_device, dev=dev), input))
            # 其他情况直接返回输入对象
            return input

        # 定义一个递归函数，用于断言两个对象是否近似相等
        def assert_allclose_rec(t):
            a, b = t
            self.assertEqual(type(a), type(b))
            # 如果对象是 Torch 张量，则断言其在指定设备上的克隆版本和期望值近似相等
            if isinstance(a, torch.Tensor):
                self.assertTrue(
                    torch.allclose(clone_to_device(a, test_device), b, atol=1e-4)
                )

            # 如果对象是序列，则递归地对序列中的每一对元素调用 assert_allclose_rec
            if isinstance(a, Sequence):
                map(assert_allclose_rec, zip(a, b))

        # 从操作对象中获取样本输入
        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        for sample in samples:
            # 需要运行 mark_step 方法以确保所有随机操作按正确顺序计算
            torch._lazy.mark_step()

            # 构造参数列表和关键字参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 将参数列表克隆到测试设备上
            copy_args = clone_to_device(args, test_device)

            # 计算操作的实际结果和期望结果
            r_exp = op(*copy_args, **kwargs)
            r_actual = op(*args, **kwargs)

            # 再次运行 mark_step 方法以确保所有随机操作按正确顺序计算
            torch._lazy.mark_step()
            # 对实际结果和期望结果进行近似相等断言
            assert_allclose_rec((r_actual, r_exp))

    # 使用装饰器 ops 对包含在 op_db 中的操作进行测试，限定仅对 LAZY_OPS_LIST 中的操作进行测试，并排除在 SKIP_RUNTIME_ERROR_LIST 和 SKIP_INCORRECT_RESULTS_LIST 中的操作
    @ops(
        [
            op
            for op in op_db
            if op.name in LAZY_OPS_LIST
            and op.name not in SKIP_RUNTIME_ERROR_LIST | SKIP_INCORRECT_RESULTS_LIST
        ],
        allowed_dtypes=(torch.float,),
    )  # noqa: B950
    # 定义测试函数，用于检验在启用重用 IR 情况下的正确性
    def test_correctness_with_reusing_ir(self, device, dtype, op):
        # 设置 Torch 懒加载模块以重用 IR
        torch._lazy.config.set_reuse_ir(True)
        # 获取测试设备
        test_device = get_test_device()

        # 定义一个函数，用于将输入对象克隆到指定设备上
        def clone_to_device(input, dev):
            # 如果输入是 Torch 张量，则进行去梯度、克隆和设备转换操作
            if isinstance(input, torch.Tensor):
                return input.detach().clone().to(device=dev)
            # 如果输入是序列但不是字符串，则递归地对序列中的每个元素应用克隆到设备函数
            if isinstance(input, Sequence) and not isinstance(input, str):
                return tuple(map(functools.partial(clone_to_device, dev=dev), input))
            # 其他情况直接返回输入
            return input

        # 定义一个递归函数，用于断言两个对象的内容近似相等
        def assert_allclose_rec(t):
            a, b = t
            # 断言 a 和 b 的类型相同
            self.assertEqual(type(a), type(b))
            # 如果 a 是 Torch 张量，则断言经过克隆到指定设备后与 b 的内容在指定的误差范围内近似相等
            if isinstance(a, torch.Tensor):
                self.assertTrue(
                    torch.allclose(clone_to_device(a, test_device), b, atol=1e-4)
                )

            # 如果 a 是序列，则对 a 和 b 的元素逐对应用递归断言函数
            if isinstance(a, Sequence):
                map(assert_allclose_rec, zip(a, b))

        # 从操作对象获取样本输入数据
        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        # 遍历每个样本
        for sample in samples:
            # 需要运行标记步骤，以确保所有随机操作按正确顺序计算
            torch._lazy.mark_step()

            # 准备参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 对参数进行克隆到指定设备操作
            copy_args = clone_to_device(args, test_device)

            # 计算期望的操作结果和实际的操作结果
            r_exp = op(*copy_args, **kwargs)
            r_actual = op(*args, **kwargs)

            # 再次运行标记步骤
            torch._lazy.mark_step()
            # 断言实际结果和期望结果的近似性
            assert_allclose_rec((r_actual, r_exp))

        # 重置 IR 缓存
        torch._lazy.ir_cache.reset()
        # 关闭 Torch 懒加载模块的 IR 重用
        torch._lazy.config.set_reuse_ir(False)
# 在我们迁移到主分支后，将 Lazy 作为新设备添加到这里：
# https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_device_type.py#L532
# 实例化 DeviceType 测试，将 TestLazyOpInfo 作为参数传递给 instantiate_device_type_tests 函数，
# 全局范围内生效，仅适用于 CPU。
instantiate_device_type_tests(TestLazyOpInfo, globals(), only_for="cpu")


class TestLazyDynamicOps(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 设置动态形状模式
        cls.old_ssa_mode = torch._C._lazy._get_symbolic_shape_mode()
        torch._C._lazy._set_symbolic_shape_mode(True)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        # 恢复之前的符号形状模式
        torch._C._lazy._set_symbolic_shape_mode(cls.old_ssa_mode)
        return super().tearDownClass()

    def test_nonzero_dynamic(self):
        # 测试在启用符号形状模式时，nonzero 函数给出上界大小
        test_device = get_test_device()
        x1 = torch.tensor(
            [[0, 1.0, 2.0], [3.0, 0, 0]], device=test_device, requires_grad=True
        )
        # 创建 x1 的惰性版本
        x1_lazy = clone_move(x1)
        # 使用惰性张量计算非零元素的索引
        x2_lazy = torch.nonzero(x1_lazy)

        # FIXME: 添加绑定以获取上界
        # self.assertEqual(tuple(x2_lazy.size()), (6, 2))

        # 我们仍然应该能够实例化它并得到实际的结果
        # 将惰性张量移动到 CPU 并验证其形状
        x2_eager = x2_lazy.cpu()
        self.assertEqual(tuple(x2_eager.size()), (3, 2))

    def test_adaptiveavgpool3d_dynamic(self):
        # 测试在惰性后端使用时，adaptive_avg_pool3d 函数能够给出正确的形状
        img_cpu = torch.zeros([2, 3, 4, 5, 6], device="cpu")
        # 在 CPU 上执行 adaptive_avg_pool3d 操作
        out_cpu = torch.nn.AdaptiveAvgPool3d(2).to(device="cpu")(img_cpu)

        test_device = get_test_device()
        img_lazy = torch.zeros([2, 3, 4, 5, 6], device=test_device)
        # 在惰性设备上执行 adaptive_avg_pool3d 操作
        out_lazy = torch.nn.AdaptiveAvgPool3d(2).to(test_device)(img_lazy)

        # 验证 CPU 和惰性设备上的输出形状是否一致
        self.assertEqual(out_cpu.shape, out_lazy.shape)


if __name__ == "__main__":
    # 运行测试
    run_tests()
```