# `.\pytorch\test\inductor\test_aot_inductor.py`

```py
# 导入必要的模块和库
import copy  # 导入copy模块，用于复制对象
import itertools  # 导入itertools模块，用于高效的迭代工具
import os  # 导入os模块，提供了与操作系统交互的功能
import sys  # 导入sys模块，提供了对Python解释器的访问
import tempfile  # 导入tempfile模块，用于处理临时文件和目录
import types  # 导入types模块，用于操作Python类型和对象
import unittest  # 导入unittest模块，用于编写和运行单元测试
from typing import Dict, Tuple  # 导入类型提示相关的功能

import torch  # 导入PyTorch深度学习框架
import torch._export  # 导入PyTorch的导出模块
import torch._inductor  # 导入PyTorch的inductor模块
import torch._inductor.config  # 导入PyTorch的inductor配置模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch._dynamo.testing import rand_strided, same  # 导入测试相关的工具函数
from torch._dynamo.utils import counters  # 导入计数器相关的工具函数
from torch._inductor import config  # 导入inductor的配置模块
from torch._inductor.exc import CppWrapperCodeGenError  # 导入C++封装代码生成错误异常
from torch._inductor.runtime.runtime_utils import cache_dir  # 导入运行时工具函数
from torch._inductor.test_case import TestCase  # 导入inductor测试用例

from torch.export import Dim, export  # 导入导出相关的模块和类
from torch.testing import FileCheck  # 导入文件检查工具
from torch.testing._internal import common_utils  # 导入内部测试相关的通用工具
from torch.testing._internal.common_cuda import SM80OrLater, SM90OrLater  # 导入CUDA相关的工具
from torch.testing._internal.common_quantization import (
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)  # 导入量化相关的工具和装饰器
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_CI,
    IS_FBCODE,
    IS_WINDOWS,
    skipIfRocm,
    TEST_WITH_ROCM,
)  # 导入通用的测试相关工具和条件判断函数

from torch.testing._internal.triton_utils import HAS_CUDA, requires_cuda  # 导入Triton相关的工具和装饰器
from torch.utils import _pytree as pytree  # 导入_pytree别名

# 如果有CUDA支持，则进一步导入Triton相关的功能
if HAS_CUDA:
    import triton
    from torch.testing._internal.triton_utils import (
        add_kernel,
        add_kernel_2d_autotuned,
        add_kernel_autotuned,
        add_kernel_with_optional_param,
        add_kernel_with_scaling,
    )

# 如果是Windows且是CI环境，则输出相关警告信息并且跳过测试
if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    # 尝试导入测试依赖的模块和类
    try:
        from .test_aot_inductor_utils import AOTIRunnerUtil
        from .test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from .test_torchinductor import copy_tests, requires_multigpu, TestFailure
    except ImportError:
        from test_aot_inductor_utils import AOTIRunnerUtil
        from test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from test_torchinductor import copy_tests, requires_multigpu, TestFailure
except (unittest.SkipTest, ImportError) as e:
    # 如果导入失败，则在测试环境下退出，否则抛出异常
    if __name__ == "__main__":
        sys.exit(0)
    raise

def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    disable_constraint_solver=False,
    atol=None,
    rtol=None,
):
    # 检查模型在给定输入下的输出是否正确，可选参数用于配置检查行为
    with torch.no_grad(), config.patch(
        {
            "abi_compatible": self.abi_compatible,  # 设定ABI兼容性
            "allow_stack_allocation": self.allow_stack_allocation,  # 允许堆栈分配
            "use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,  # 使用最小的arrayref接口
        }
        # 设置随机种子为0，以确保可重复性
        torch.manual_seed(0)
        # 如果传入的 model 不是函数类型，则将其移到当前对象的设备上
        if not isinstance(model, types.FunctionType):
            model = model.to(self.device)
        # 深度复制模型和示例输入，用于后续参考
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        # 使用参考模型和输入计算预期输出
        expected = ref_model(*ref_inputs)

        # 再次设置随机种子为0，确保 AOTIRunnerUtil.run 的结果可重复
        torch.manual_seed(0)
        # 运行 AOTIRunnerUtil 工具的主函数，获取实际输出
        actual = AOTIRunnerUtil.run(
            self.device,
            model,
            example_inputs,
            options,
            dynamic_shapes,
            disable_constraint_solver,
        )

    # 使用断言比较实际输出和预期输出是否相等，设置允许的数值误差
    self.assertEqual(actual, expected, atol=atol, rtol=rtol)
# 定义一个方法，用于测试模型在多个输入下的行为
def check_model_with_multiple_inputs(
    self: TestCase,
    model,  # 模型对象，用于进行测试
    list_example_inputs,  # 包含多个示例输入的列表
    options=None,  # 可选参数，配置选项
    dynamic_shapes=None,  # 可选参数，动态形状的配置
):
    # 禁用梯度追踪，在此期间修改配置
    with torch.no_grad(), config.patch(
        {
            "abi_compatible": self.abi_compatible,  # 设置 ABI 兼容性
            "allow_stack_allocation": self.allow_stack_allocation,  # 设置允许堆栈分配
        }
    ):
        # 设定随机种子为 0
        torch.manual_seed(0)
        # 将模型移至指定设备
        model = model.to(self.device)
        # 复制模型以备参考
        ref_model = copy.deepcopy(model)
        # 复制示例输入列表以备参考
        ref_inputs = copy.deepcopy(list_example_inputs)
        # 创建预期输出列表，调用参考模型的 forward 方法
        list_expected = [ref_model(*inputs) for inputs in ref_inputs]

        # 再次设定随机种子为 0
        torch.manual_seed(0)
        # 运行 AOTIRunnerUtil 的多输入方法，获取实际输出列表
        list_actual = AOTIRunnerUtil.run_multiple(
            self.device, model, list_example_inputs, options, dynamic_shapes
        )

    # 断言实际输出与预期输出是否一致
    self.assertTrue(same(list_actual, list_expected))


# 定义一个方法，用于检查编译模型的代码中特定字符串出现的次数
def code_check_count(
    self: TestCase,
    model,  # 模型对象，用于进行编译和检查
    example_inputs,  # 示例输入，用于编译模型
    target_str: str,  # 目标字符串，要在生成的源码中检查的字符串
    target_count: int,  # 目标计数，期望目标字符串在源码中出现的次数
):
    # 编译模型并获取生成的源文件路径
    so_path = torch._export.aot_compile(model, example_inputs)
    # 打开对应的生成的 C++ 源码文件
    with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
        src_code = cpp.read()
        # 创建 FileCheck 对象，检查目标字符串是否符合预期次数出现
        FileCheck().check_count(
            target_str,
            target_count,
            exactly=True,
        ).run(src_code)


# 定义一个模板类，用于执行 AOTInductor 的测试
class AOTInductorTestsTemplate:
    # 测试简单模型
    def test_simple(self):
        # 定义一个简单的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        # 示例输入为两个随机张量
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 调用 check_model 方法，测试模型行为
        self.check_model(Model(), example_inputs)

    # 测试小型常量模型
    def test_small_constant(self):
        # 定义一个小型常量模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        # 示例输入为一个随机张量
        example_inputs = (torch.randn(4, 4, device=self.device),)
        # 使用配置选项，调用 check_model 方法，测试模型行为
        with config.patch({"always_keep_tensor_constants": True}):
            self.check_model(Model().to(self.device), example_inputs)

    # 测试输出路径设置
    def test_output_path_1(self):
        # 定义一个简单的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        # 示例输入为两个随机张量
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 使用配置选项，调用 check_model 方法，测试模型行为
        with config.patch("aot_inductor.output_path", "tmp_output_"):
            self.check_model(Model(), example_inputs)
    # 定义一个测试方法，用于验证模型编译后的输出路径是否符合预期
    def test_output_path_2(self):
        # 定义一个简单的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # 模型的前向传播：返回输入 x 和线性层对输入 y 的加权和
                return x + self.linear(y)

        # 创建一个模型实例，并将其移到指定设备上
        model = Model().to(device=self.device)
        # 准备模型输入的示例数据
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 生成预期的输出路径，将编译后的模型保存为一个动态链接库
        expected_path = os.path.join(tempfile.mkdtemp(dir=cache_dir()), "model.so")
        # 编译模型并获取实际的输出路径
        actual_path = AOTIRunnerUtil.compile(
            model, example_inputs, options={"aot_inductor.output_path": expected_path}
        )
        # 断言实际输出路径与预期路径相等
        self.assertTrue(actual_path == expected_path)

    # 测试模型常量折叠的功能
    def test_constant_folding(self):
        # 定义一个包含常量折叠操作的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                # 执行常量折叠：转置、ReLU 激活函数、加权和
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        # 准备模型输入的示例数据
        example_inputs = (torch.randn(4, 4, device=self.device),)
        # 在配置中启用运行时常量折叠
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            # 检查模型的常量折叠功能是否正常工作
            self.check_model(Model(self.device), example_inputs)

    # 测试多个常量折叠的功能
    @requires_cuda
    def test_duplicate_constant_folding(self):
        # 定义一个包含多个常量折叠操作的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w1 = torch.randn(4, 4, device=device)
                self.w2 = torch.randn(4, 4, device=device)
                self.w3 = torch.randn(4, 4, device=device)
                self.w4 = torch.randn(4, 4, device=device)

            def forward(self, x):
                # 执行多个常量折叠：连接多个权重张量
                w_concat = torch.cat((self.w1, self.w2, self.w3, self.w4))
                return torch.cat((x, w_concat))

        # 准备模型输入的示例数据
        example_inputs = (torch.randn(4, 4, device=self.device),)
        # 在配置中启用运行时常量折叠
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            # 检查模型的多个常量折叠功能是否正常工作
            self.check_model(Model(self.device), example_inputs)

    # 测试模型在多设备上的功能
    @requires_cuda
    def test_multi_device(self):
        # 定义一个模型，展示在不同设备间切换的操作
        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = x.cpu()
                x = x + 2
                x = x.cuda()
                return x

        # 准备模型输入的示例数据
        example_inputs = (torch.randn(32, 64, device=self.device),)
        # 检查模型在多设备上的运行情况
        self.check_model(Model(), example_inputs)
    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )


# 如果运行在 FBCODE 环境中，则跳过该测试用例，因为新生成的 model.so 可能与旧版本的 PyTorch 不兼容
@unittest.skipIf(
    IS_FBCODE,
    "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
)



        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2048, 262144)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 262144, device=self.device),
            torch.randn(1, 2048, device=self.device),
        )

        # We only test compilation since we often get OOM running in CI.
        model = Model()
        model = model.to(self.device)
        AOTIRunnerUtil.compile(model, example_inputs)


# 定义一个模型类 Model，继承自 torch.nn.Module
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入大小为 2048，输出大小为 262144
        self.linear = torch.nn.Linear(2048, 262144)

    def forward(self, x, y):
        # 模型的前向传播，返回 x 加上 linear 层对 y 的线性变换结果
        return x + self.linear(y)

# 示例输入，包含两个张量，分别为大小为 (1, 262144) 和 (1, 2048)，存储在指定的设备上
example_inputs = (
    torch.randn(1, 262144, device=self.device),
    torch.randn(1, 2048, device=self.device),
)

# 只测试模型的编译，因为在 CI 中经常因为内存溢出而无法运行
model = Model()
model = model.to(self.device)
# 使用 AOTIRunnerUtil.compile 方法编译模型
AOTIRunnerUtil.compile(model, example_inputs)



    def test_large_mmaped_weights(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch({"aot_inductor.force_mmap_weights": True}):
            self.check_model(Model(), example_inputs)


# 测试模型使用大内存映射权重
def test_large_mmaped_weights(self):
    # 定义一个模型类 Model，继承自 torch.nn.Module
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 定义一个线性层，输入大小为 512，输出大小为 250112
            self.linear = torch.nn.Linear(512, 250112)

        def forward(self, x, y):
            # 模型的前向传播，返回 x 加上 linear 层对 y 的线性变换结果
            return x + self.linear(y)

    # 示例输入，包含两个张量，分别为大小为 (1, 250112) 和 (1, 512)，存储在指定的设备上
    example_inputs = (
        torch.randn(1, 250112, device=self.device),
        torch.randn(1, 512, device=self.device),
    )

    # 使用 config.patch 方法设置权重强制使用内存映射
    with config.patch({"aot_inductor.force_mmap_weights": True}):
        # 调用 self.check_model 方法检查模型
        self.check_model(Model(), example_inputs)



    def test_with_offset(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 创建一个原始张量，形状为 (2, 15, 10)，然后选择子张量，形状为 (10, 10)，存储在指定的设备上
                self.orig_tensor = torch.randn(2, 15, 10, device=device)[0]
                self.tensor = self.orig_tensor[5:, :]

            def forward(self, x, y):
                # 模型的前向传播，返回 x 加上 y 与原始张量的线性变换结果再加上子张量
                return (
                    x
                    + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                    + self.tensor
                )

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 调用 self.check_model 方法检查模型
        self.check_model(Model(self.device), example_inputs)


# 测试带有偏移的模型
def test_with_offset(self):
    # 定义一个模型类 Model，继承自 torch.nn.Module
    class Model(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            # 创建一个原始张量，形状为 (2, 15, 10)，然后选择子张量，形状为 (10, 10)，存储在指定的设备上
            self.orig_tensor = torch.randn(2, 15, 10, device=device)[0]
            self.tensor = self.orig_tensor[5:, :]

        def forward(self, x, y):
            # 模型的前向传播，返回 x 加上 y 与原始张量的线性变换结果再加上子张量
            return (
                x
                + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                + self.tensor
            )

    # 示例输入，包含两个张量，形状为 (10, 10)，存储在指定的设备上
    example_inputs = (
        torch.randn(10, 10, device=self.device),
        torch.randn(10, 10, device=self.device),
    )
    # 调用 self.check_model 方法检查模型
    self.check_model(Model(self.device), example_inputs)
    # 定义名为 test_conv_freezing 的测试方法
    def test_conv_freezing(self):
        # 使用 itertools.product 生成 dtype 和 groups 的所有组合
        for dtype, groups in itertools.product([torch.bfloat16, torch.float], [1, 2]):
            # 设置输入通道数和输出通道数
            iC = 2
            oC = 3

            # 定义名为 Model 的内部类，继承自 torch.nn.Module
            class Model(torch.nn.Module):
                # 构造函数，初始化权重为随机值张量
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(oC * groups, iC, 3, 3, device=device).to(
                        dtype
                    )

                # 前向传播方法，使用 torch.nn.functional.conv2d 进行卷积操作
                def forward(self, y):
                    return torch.nn.functional.conv2d(y, self.weight, groups=groups)

            # 生成示例输入 example_inputs
            example_inputs = (
                torch.randn(2, iC * groups, 10, 10, device=self.device).to(dtype),
            )

            # 使用 config.patch 设置 freezing 参数为 True，并调用 self.check_model 进行模型检查
            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    # 跳过测试条件：若 IS_FBCODE 为真，则不执行该测试方法
    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    # 定义名为 test_deconv_freezing 的测试方法
    def test_deconv_freezing(self):
        # 根据条件选择 dtypes 和 groups 的组合
        dtypes = [torch.float]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype, groups in itertools.product(dtypes, [2, 1]):
            # 设置输入通道数和输出通道数
            iC = 4
            oC = 2

            # 定义名为 Model 的内部类，继承自 torch.nn.Module
            class Model(torch.nn.Module):
                # 构造函数，初始化权重为随机值张量
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(iC, oC * groups, 2, 2, device=device).to(
                        dtype
                    )

                # 前向传播方法，使用 torch.nn.functional.conv_transpose2d 进行反卷积操作
                def forward(self, y):
                    return torch.nn.functional.conv_transpose2d(
                        y, self.weight, groups=groups
                    )

            # 生成示例输入 example_inputs
            example_inputs = (torch.randn(1, iC, 3, 3, device=self.device).to(dtype),)
            
            # 使用 config.patch 设置 freezing 参数为 True，并调用 self.check_model 进行模型检查
            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    # 跳过测试条件：若 IS_FBCODE 为真，则不执行该测试方法
    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    # 定义名为 test_linear_freezing 的测试方法
    def test_linear_freezing(self):
        # 遍历指定的 dtype 列表
        for dtype in [torch.float32, torch.bfloat16]:

            # 定义名为 LinearModel 的内部类，继承自 torch.nn.Module
            class LinearModel(torch.nn.Module):
                # 构造函数，初始化权重和偏置为随机值张量
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(10, 10, device=device).to(dtype)
                    self.bias = torch.randn(10, device=device).to(dtype)

                # 前向传播方法，使用 torch.nn.functional.linear 进行线性操作
                def forward(self, y):
                    return torch.nn.functional.linear(y, self.weight, self.bias)

            # 生成示例输入 example_inputs
            example_inputs = (torch.randn(10, 10, device=self.device).to(dtype),)

            # 使用 config.patch 设置 freezing 参数为 True，并调用 self.check_model 进行模型检查
            with config.patch({"freezing": True}):
                self.check_model(LinearModel(self.device), example_inputs)
    # 使用 @torch._inductor.config.patch 装饰器，配置预梯度融合选项和后梯度融合选项为空字典
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "remove_split_with_size_one_pass": {},
            "merge_getitem_cat_pass": {},
            "merge_stack_tahn_unbind_pass": {},
            "merge_splits_pass": {},
            "mutate_cat_pass": {},
            "split_cat_pass": {},
            "unbind_stack_pass": {},
        },
        post_grad_fusion_options={},
    )
    # 定义一个测试方法 test_simple_split
    def test_simple_split(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 实现模型的前向传播方法
            def forward(self, x):
                # 将输入张量 x 沿着 dim=1 切分为长度为 4 的张量列表，然后沿着 dim=-2 连接起来
                return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        # 生成一个例子输入 example_inputs，包含一个形状为 (2, 8) 的随机张量，使用 self.device 指定设备
        example_inputs = (torch.randn(2, 8, device=self.device),)
        # 清空计数器 counters
        counters.clear()
        # 检查模型 Model 在例子输入 example_inputs 上的表现
        self.check_model(Model(), example_inputs)
        # 断言计数器中的指定值，检查预期的融合操作是否执行了
        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_split_sections_removed"], 1)

    # 定义一个测试方法 test_amp_fallback_random
    def test_amp_fallback_random(self):
        # 定义一个函数 fn，执行 torch 的线性变换操作
        def fn(x, w):
            return torch.functional.F.linear(x, w)

        # 生成例子输入 example_inputs，包含两个形状为 (10, 10) 的随机张量，使用 self.device 指定设备
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 根据 self.device 的值选择上下文 ctx，用于自动混合精度计算
        if self.device == "cuda":
            ctx = torch.cuda.amp.autocast
        elif self.device == "cpu":
            ctx = torch.cpu.amp.autocast
        else:
            raise AssertionError("Unsupported device")

        # 使用 config.patch({"fallback_random": True}) 设置配置，测试自动混合精度计算 ctx 下的模型 fn
        with config.patch({"fallback_random": True}):
            with ctx():
                self.check_model(fn, example_inputs)

    # 定义一个测试方法 test_missing_output
    def test_missing_output(self):
        # 定义一个模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 实现模型的前向传播方法
            def forward(self, x, y):
                # 计算 x 的正弦值
                a = torch.sin(x)
                # 计算 a 与 y 的矩阵乘积
                b = torch.mm(a, y)
                # 计算 b 的余弦值
                c = torch.cos(b)
                return c

        # 生成例子输入 example_inputs，包含两个形状为 (10, 10) 的随机张量，使用 self.device 指定设备
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 检查模型 Model 在例子输入 example_inputs 上的表现
        self.check_model(Model(), example_inputs)

    # 定义一个测试方法 test_output_misaligned
    def test_output_misaligned(self):
        # 定义一个模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 实现模型的前向传播方法
            def forward(self, x, y):
                # 在维度 0 上对 x 和 y 进行 unsqueeze 操作
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                # 在维度 0 上将 x_unsqueeze 和 y_unsqueeze 进行拼接
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                # 获取 cat 中的第 0 个索引处的张量
                x_getitem = cat[0]
                # 获取 cat 中的第 1 个索引处的张量
                y_getitem = cat[1]
                # 对 x_getitem 执行 sigmoid 操作
                x_sigmoid = torch.sigmoid(x_getitem)
                # 返回 x_sigmoid 和 y_getitem
                return x_sigmoid, y_getitem

        # 生成例子输入 example_inputs，包含两个形状为 (10, 10) 的随机张量，使用 self.device 指定设备
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 检查模型 Model 在例子输入 example_inputs 上的表现
        self.check_model(Model(), example_inputs)

    # 使用 @skip 装饰器标记的测试方法，说明该测试已经不总是失败了
    @skip("Test was marked as expected failure, but does not fail always anymore.")
    def test_dynamic_smem_above_default_limit(self):
        # 定义一个简单的神经网络模型
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        # 创建模型实例并将其移动到指定的设备上
        model = Model().to(self.device)
        
        # 对于 A100 设备，下面这个矩阵乘法的 Triton 核心生成的内核
        # 需要55296字节的动态共享内存（SMEM），超过了 A100 默认的49152字节的限制
        example_inputs = (
            torch.randn(10285, 96, device=self.device),
            torch.randn(96, 1, device=self.device),
        )
        
        # 检查模型在给定输入上的表现
        self.check_model(
            model,
            example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_seq(self):
        # 创建一个包含层归一化和ReLU激活函数的神经网络序列
        layernorm = torch.nn.LayerNorm(10)
        net = torch.nn.Sequential(
            layernorm,
            torch.nn.ReLU(),
            layernorm,
            torch.nn.ReLU(),
        )

        example_inputs = (torch.randn(10, device=self.device),)
        
        # 检查在评估模式下，神经网络序列在给定输入上的表现
        self.check_model(net.eval(), example_inputs)

    def test_addmm(self):
        # 定义一个包含线性层操作的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        # 创建模型实例并传入参数
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)
        
        # 检查模型在给定输入上的表现
        self.check_model(model, example_inputs)

    def test_aliased_buffer_reuse(self):
        # 定义一个使用缓冲重用的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = 2 * x
                y = 2 * y
                c = torch.cat([x, y], dim=-1)
                d = 1 + c
                m = torch.mm(d, d)
                return m[:, :2] + x

        example_inputs = (
            torch.randn(4, 2, device=self.device),
            torch.randn(4, 2, device=self.device),
        )
        
        # 检查模型在给定输入上的表现
        self.check_model(Model(), example_inputs)

    def test_buffer_reuse(self):
        # 定义一个使用缓冲重用的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.cos(y)
                c = torch.mm(a, b)
                d = torch.relu(c)
                e = torch.sigmoid(d)
                f = torch.mm(x, y)
                g = e + f
                return g

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        
        # 检查模型在给定输入上的表现
        self.check_model(Model(), example_inputs)
    # 定义一个名为 test_duplicated_params 的测试方法
    def test_duplicated_params(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个包含随机值的参数 p
                self.p = torch.nn.Parameter(torch.rand(6))
                # 将参数 q 设置为 p 的引用
                self.q = self.p

            # 前向传播方法
            def forward(self, x):
                # 返回 p 与 q 的加权和
                return self.p * x + self.q

        # 创建一个示例输入，包含一个随机张量，使用当前设备
        example_inputs = (torch.rand(6, device=self.device),)
        # 调用 self.check_model 方法来检查 Model 类的行为
        self.check_model(Model(), example_inputs)

    # 使用 unittest.skip 注解跳过当前测试，并附带说明
    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    # 定义一个名为 test_inf 的测试方法
    def test_inf(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层
                self.linear = torch.nn.Linear(10, 10)

            # 前向传播方法，接受两个输入 x 和 y
            def forward(self, x, y):
                # 返回 x 与线性层作用于 y 的结果
                return x + self.linear(y)

        # 创建一个包含随机张量的 x，其中第一个元素设置为正无穷
        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("Inf")
        # 创建示例输入，包含 x 和另一个随机张量，使用当前设备
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        # 调用 self.check_model 方法来检查 Model 类的行为，并传入 debug 参数选项
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    # 使用 unittest.skip 注解跳过当前测试，并附带说明
    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    # 定义一个名为 test_nan 的测试方法
    def test_nan(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层
                self.linear = torch.nn.Linear(10, 10)

            # 前向传播方法，接受两个输入 x 和 y
            def forward(self, x, y):
                # 返回 x 与线性层作用于 y 的结果
                return x + self.linear(y)

        # 创建一个包含随机张量的 x，其中第一个元素设置为 NaN
        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("nan")
        # 创建示例输入，包含 x 和另一个随机张量，使用当前设备
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        # 调用 self.check_model 方法来检查 Model 类的行为，并传入 debug 参数选项
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    # 定义一个名为 test_assert_async 的测试方法
    def test_assert_async(self):
        # 如果当前设备不是 CUDA，则抛出 unittest.SkipTest 异常
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，接受一个输入 x
            def forward(self, x):
                # 获取 x 的第一个元素并将其命名为 u0
                u0 = x.item()
                # 检查 u0 是否大于 3
                torch._check(u0 > 3)
                # 返回一个包含单个元素的张量，其值为 1.0
                return torch.ones(u0)[0]

        # 创建一个包含整数 23 的张量 x，使用当前设备
        x = torch.tensor(23, device=self.device)
        # 创建示例输入，包含 x
        example_inputs = (x,)
        # 调用 self.check_model 方法来检查 Model 类的行为
        self.check_model(Model(), example_inputs)

    # 定义一个名为 test_simple_dynamic 的测试方法
    def test_simple_dynamic(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，接受两个输入 x 和 y
            def forward(self, x, y):
                # 将 x 和 y 相加得到 add_0
                add_0 = x + y
                # 对 add_0 应用 ReLU 激活函数，并返回结果
                return torch.nn.functional.relu(input=add_0, inplace=False)

        # 创建两个包含随机张量 x 和 y，使用当前设备
        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        # 创建动态形状字典，指定 x 和 y 的维度范围
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        # 创建示例输入，包含 x 和 y
        example_inputs = (x, y)
        # 调用 self.check_model 方法来检查 Model 类的行为，并传入 dynamic_shapes 参数选项
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "FP8 is only supported on H100+",
    )
    @skipIfRocm  # _scaled_mm_out_cuda  is not compiled for ROCm platform
    def test_fp8(self):
        # 定义一个测试方法，用于测试使用 FP8 数据类型的模型计算
        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.out_dtype = dtype

            def forward(self, x, weight, bias, scale_a, scale_b):
                # 将权重转换为 torch.float8_e4m3fn 类型
                weight = weight.to(torch.float8_e4m3fn)
                # 使用 torch._scaled_mm 执行矩阵乘法操作，并返回结果和更新后的最大值
                output, updated_amax = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,  # 使用给定的偏置
                    out_dtype=self.out_dtype,  # 输出的数据类型
                    scale_a=scale_a,  # 缩放因子 A
                    scale_b=scale_b,  # 缩放因子 B
                )
                return output  # 返回计算结果

        dtype = torch.float16  # 指定模型使用的数据类型为 torch.float16

        a_scale = torch.Tensor([1.0]).to(device="cuda")  # 缩放因子 A
        b_scale = torch.Tensor([1.0]).to(device="cuda")  # 缩放因子 B
        input_bias = torch.rand(32, device="cuda", dtype=dtype)  # 输入数据的偏置项
        weight_shape = (32, 16)  # 权重张量的形状
        weight = torch.rand(*weight_shape, device="cuda", dtype=dtype).T  # 随机生成并转置的权重张量
        a_inverse_scale = 1 / a_scale  # 缩放因子 A 的倒数
        b_inverse_scale = 1 / b_scale  # 缩放因子 B 的倒数

        x_shape = (16, 16)  # 输入张量的形状
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(torch.float8_e4m3fn)  # 随机生成并转换为 FP8 数据类型的输入张量
        dim0_x = Dim("dim0_x", min=1, max=2048)  # 定义维度参数
        dynamic_shapes = ({0: dim0_x}, None, None, None, None)  # 动态形状参数

        # 调用 check_model 方法验证模型
        self.check_model(
            Model(dtype),  # 使用指定数据类型的模型
            (x, weight, input_bias, a_inverse_scale, b_inverse_scale),  # 输入参数
            dynamic_shapes=dynamic_shapes,  # 动态形状参数
        )

    def test_poi_multiple_dynamic(self):
        # 定义一个测试方法，用于测试具有多个动态输入的模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y  # 执行输入张量 x 和 y 的加法操作
                return torch.nn.functional.relu(input=add_0, inplace=False)  # 返回加法结果的 ReLU 激活函数值

        x = torch.randn(128, 2048, device=self.device)  # 随机生成输入张量 x
        y = torch.randn(128, 2048, device=self.device)  # 随机生成输入张量 y
        dim0_x = Dim("dim0_x", min=1, max=2048)  # 定义维度参数
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}  # 指定动态形状参数

        # 创建包含不同输入对的示例输入列表
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),  # 随机生成不同形状的输入张量
                torch.randn(64, 2048, device=self.device),  # 随机生成不同形状的输入张量
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),  # 随机生成不同形状的输入张量
                torch.randn(211, 2048, device=self.device),  # 随机生成不同形状的输入张量
            ),
        )

        # 调用 check_model_with_multiple_inputs 方法验证模型
        self.check_model_with_multiple_inputs(
            Model(),  # 使用定义的模型
            list_example_inputs,  # 示例输入列表
            dynamic_shapes=dynamic_shapes  # 动态形状参数
        )
    # 定义一个测试方法，用于测试支持多个动态输入的 addmm 操作
    def test_addmm_multiple_dynamic(self):
        # 定义一个简单的神经网络模型类，包含权重和偏置参数
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                # 随机初始化权重矩阵和偏置向量
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # 使用线性函数计算输出
                return torch.nn.functional.linear(a, self.weight, self.bias)

        # 设置模型参数的维度
        M = 8
        N = 6
        K = 16
        # 创建模型实例并指定计算设备
        model = Model(N, K, self.device)
        # 生成随机输入张量 a，形状为 (batch, M, K)，使用指定的设备
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        # 定义动态维度规格，这里设置了张量 a 的第一个维度的动态范围
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}}
        # 构建示例输入列表，包含一个 (a,) 的元组
        list_example_inputs = [(a,)]
        # 扩展示例输入列表，添加更多的示例输入张量，每个张量都有不同的 batch 大小
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        # 调用自定义方法检查模型对多个输入的处理情况
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    # 定义一个测试方法，用于测试支持多个动态输入的 bmm 操作
    def test_bmm_multiple_dynamic(self):
        # 定义一个简单的神经网络模型类，执行矩阵乘法 bmm 操作
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                # 执行批量矩阵乘法操作
                return torch.bmm(a, b)

        # 设置模型参数的维度
        M = 8
        N = 6
        K = 16
        # 创建模型实例
        model = Model()
        # 生成随机输入张量 a 和 b，形状分别为 (batch, M, K) 和 (batch, K, N)，使用指定的设备
        batch = 1024
        a = torch.randn(batch, M, K, device=self.device)
        b = torch.randn(batch, K, N, device=self.device)
        # 定义动态维度规格，这里设置了张量 a 和 b 的第一个维度的动态范围
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_a}}
        # 构建示例输入列表，包含一个 (a, b) 的元组
        list_example_inputs = [(a, b)]
        # 扩展示例输入列表，添加更多的示例输入张量对，每个张量对都有不同的 batch 大小
        batch = 2048
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        batch = 128
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        # 调用自定义方法检查模型对多个输入的处理情况
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            dynamic_shapes=dynamic_shapes,
        )
    # 定义一个测试方法，用于测试模型在多个动态输入下的表现
    def test_foreach_multiple_dynamic(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 对输入张量x和y进行维度扩展，增加维度0
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                # 在维度0上连接x和y，形成一个新的张量
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                return cat

        # 创建一个Model类的实例
        model = Model()
        # 生成两个形状为(128, 2048)的随机张量x和y，使用self.device指定设备
        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        # 定义一个Dim对象dim0_x，设置其最小值为1，最大值为2048
        dim0_x = Dim("dim0_x", min=1, max=2048)
        # 定义一个动态形状字典dynamic_shapes，指定x和y在维度0上使用dim0_x
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        # 创建一个示例输入列表list_example_inputs，包含(x, y)作为第一个元素
        list_example_inputs = [(x, y)]
        # 向list_example_inputs添加第二个元素，包含形状为(64, 2048)的两个随机张量
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        # 向list_example_inputs添加第三个元素，包含形状为(211, 2048)的两个随机张量
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        # 调用self.check_model_with_multiple_inputs方法，测试模型在多个输入下的表现
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    # scaled_dot_product_flash_attention
    # 如果在Facebook环境中，跳过此测试；如果不支持SM80或更高版本，则跳过测试
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(not SM80OrLater, "bfloat16 only supported in sm80+")
    def test_sdpa(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法，使用scaled_dot_product_attention函数
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)[0]

        # 创建一个示例输入example_inputs，包含三个形状为(1, 48, 64, 64)的bfloat16类型张量
        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        # 调用self.check_model方法，测试模型在示例输入下的表现
        self.check_model(Model(), example_inputs)

    # 如果在Facebook环境中，跳过此测试；如果不支持SM80或更高版本，则跳过测试
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(not SM80OrLater, "bfloat16 only supported in sm80+")
    def test_sdpa_2(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法，使用scaled_dot_product_attention函数，设置is_causal=True
            def forward(self, q, k, v, x):
                t = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )[0]
                # 返回x与t的和作为模型的输出
                return x + t

        # 创建一个示例输入example_inputs，包含四个形状为(1, 48, 64, 64)的bfloat16类型张量
        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        # 调用self.check_model方法，测试模型在示例输入下的表现
        self.check_model(Model(), example_inputs)

    # 如果不支持FBGEMM，跳过测试
    @skipIfNoFBGEMM
    def test_quantized_linear(self):
        # 定义一个测试函数，用于测试量化线性模型
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 初始化模型参数：权重和偏置，使用给定的设备
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)

            def forward(self, x):
                # 调用量化线性操作的动态 FP16 解压缩权重版本
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, self.bias
                )

        # 准备示例输入数据
        example_inputs = (torch.randn(10, 10, device=self.device),)
        # 使用特定的配置上下文，执行模型检查
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_zero_grid_with_unbacked_symbols(self):
        # 定义一个测试函数，用于测试零网格与未支持的符号
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # 计算 x 中非零元素的索引
                nz = torch.nonzero(x)
                # 创建与 nz 张量相同形状和数据类型的全 1 张量 b
                b = torch.ones_like(nz, dtype=torch.float16)
                # 创建与 nz 张量相同形状和数据类型的全 0 张量 c
                c = torch.zeros_like(nz, dtype=torch.float16)
                # 计算 b 和 c 的和，并与 y 矩阵相乘
                d = (b + c) @ y
                # 返回结果 d 的和
                return d.sum()

        # 准备示例输入数据
        example_inputs = (
            torch.tensor([1, 1, 1], device=self.device),
            torch.randn((1, 32), dtype=torch.float16, device=self.device),
        )
        # 执行模型检查
        self.check_model(Repro(), example_inputs)

    def test_large_grid(self):
        # 如果设备不是 CUDA，跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个测试函数，用于测试大网格操作
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, primals_5):
                # 将 primals_5 张量重新形状为 [-1, 2, 4] 的视图
                view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
                # 清空 primals_5 张量
                primals_5 = None
                # 将视图张量按指定的维度重新排列
                permute = torch.ops.aten.permute.default(view, [0, 2, 1])
                # 克隆排列后的张量，并使用连续内存格式
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                # 返回克隆的张量
                return clone

        # 设置示例输入尺寸
        s0 = 16777472
        s1 = 8
        example_inputs = (torch.rand(s0, s1, device=self.device),)
        # 执行模型检查
        self.check_model(Model(), example_inputs)

    def test_cond_simple(self):
        # 准备输入数据
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 定义维度对象 dim0_ab
        dim0_ab = Dim("s0", min=2, max=1024)
        # 定义动态形状字典
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        # 使用多个输入执行带有动态形状的简单条件模型检查
        self.check_model_with_multiple_inputs(
            CondModels.Simple(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )
    # 测试条件嵌套模型的方法
    def test_cond_nested(self):
        # 定义输入数据，三个张量随机生成，设备为测试设备
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 创建维度对象 dim0_abc，表示维度名称为 "s0"，最小值为 2，最大值为 1024
        dim0_abc = Dim("s0", min=2, max=1024)
        # 动态形状字典，包含预测输入的动态形状信息
        dynamic_shapes = {
            "p0": {},
            "p1": {},
            "p2": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        # 使用多输入检查模型的方法，传入 CondModels.Nested() 模型实例、预测输入、动态形状信息
        self.check_model_with_multiple_inputs(
            CondModels.Nested(),
            prepend_predicates(inputs, num_predicates=3),
            dynamic_shapes=dynamic_shapes,
        )

    # 测试带参数的条件模型方法
    def test_cond_with_parameters(self):
        # 定义输入数据，一个张量随机生成，设备为测试设备
        inputs = (torch.randn((10, 20), device=self.device),)
        # 创建维度对象 dim0_abc，表示维度名称为 "s0"，最小值为 2，最大值为 1024
        dim0_abc = Dim("s0", min=2, max=1024)
        # 动态形状字典，包含预测输入的动态形状信息
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
        }
        # 使用多输入检查模型的方法，传入 CondModels.Parameters() 模型实例、预测输入、动态形状信息
        self.check_model_with_multiple_inputs(
            CondModels.Parameters(self.device),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # 测试重新解释视图输入输出的条件模型方法
    def test_cond_with_reinterpret_view_inputs_outputs(self):
        # 定义输入数据，两个张量随机生成，设备为测试设备
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 创建维度对象 dim0_ab，表示维度名称为 "s0"，最小值为 3，最大值为 1024
        dim0_ab = Dim("s0", min=3, max=1024)
        # 动态形状字典，包含预测输入的动态形状信息
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        # 使用多输入检查模型的方法，传入 CondModels.ReinterpretView() 模型实例、预测输入、动态形状信息
        self.check_model_with_multiple_inputs(
            CondModels.ReinterpretView(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # 测试带多个输出的条件模型方法
    def test_cond_with_multiple_outputs(self):
        # 定义输入数据，三个张量随机生成，设备为测试设备
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((30, 40), device=self.device),
        )
        # 创建维度对象 dim0_ab 和 dim0_c，分别表示维度名称为 "s0" 和 "s1"，最小值和最大值不同
        dim0_ab = Dim("s0", min=2, max=1024)
        dim0_c = Dim("s1", min=2, max=1024)
        # 动态形状字典，包含预测输入的动态形状信息
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
            "c": {0: dim0_c, 1: None},
        }
        # 使用多输入检查模型的方法，传入 CondModels.MultipleOutputs() 模型实例、预测输入、动态形状信息
        self.check_model_with_multiple_inputs(
            CondModels.MultipleOutputs(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # 测试外部代码前后的条件模型方法
    def test_cond_with_outer_code_before_after(self):
        # 定义输入数据，两个张量随机生成，设备为测试设备
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 创建维度对象 dim0_ab，表示维度名称为 "s0"，最小值为 2，最大值为 1024
        dim0_ab = Dim("s0", min=2, max=1024)
        # 动态形状字典，包含预测输入的动态形状信息
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        # 使用多输入检查模型的方法，传入 CondModels.OuterCode() 模型实例、预测输入、动态形状信息
        self.check_model_with_multiple_inputs(
            CondModels.OuterCode(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )
    # 定义一个测试方法，用于测试在使用外部作用域缓冲区的条件下的模型行为
    def test_cond_use_buffers_from_outer_scope(self):
        # 准备输入数据，包括三个张量矩阵，位于指定的设备上
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 创建一个维度对象，表示维度"s0"的范围在2到1024之间
        dim0_abc = Dim("s0", min=2, max=1024)
        # 定义动态形状的字典，其中"a", "b", "c"是键，每个键对应一个字典，表示动态形状信息
        dynamic_shapes = {
            "p": {},  # 空字典，表示无动态形状信息
            "a": {0: dim0_abc, 1: None},  # "a"键的动态形状信息，第一个位置有dim0_abc对象，第二个位置为None
            "b": {0: dim0_abc, 1: None},  # "b"键的动态形状信息，第一个位置有dim0_abc对象，第二个位置为None
            "c": {0: dim0_abc, 1: None},  # "c"键的动态形状信息，第一个位置有dim0_abc对象，第二个位置为None
        }
        # 调用方法，检查带有多个输入的模型行为，使用OuterBuffers条件模型
        self.check_model_with_multiple_inputs(
            CondModels.OuterBuffers(),  # 使用OuterBuffers条件模型
            prepend_predicates(inputs),  # 在输入数据前添加谓词
            dynamic_shapes=dynamic_shapes,  # 设置动态形状信息
        )

    # 使用参数化装饰器进行多次测试，其中dynamic参数为False或True
    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, dynamic):
        # 准备两组输入数据，每组包括两个张量矩阵，位于指定的设备上
        inputs1 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
        )
        inputs2 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((5, 20), device=self.device),
        )
        inputs = (inputs1,)  # inputs变量初始化为包含inputs1的元组
        dynamic_shapes = None  # 初始化动态形状信息为None
        if dynamic:
            inputs = (inputs1, inputs2)  # 如果dynamic为True，则inputs包含inputs1和inputs2
            dim0_a = Dim("s0", min=2, max=1024)  # 创建维度对象，表示维度"s0"的范围在2到1024之间
            dim0_b = Dim("s1", min=2, max=1024)  # 创建维度对象，表示维度"s1"的范围在2到1024之间
            # 定义动态形状的字典，其中"a", "b"是键，每个键对应一个字典，表示动态形状信息
            dynamic_shapes = {
                "a": {0: dim0_a, 1: None},  # "a"键的动态形状信息，第一个位置有dim0_a对象，第二个位置为None
                "b": {0: dim0_b, 1: None},  # "b"键的动态形状信息，第一个位置有dim0_b对象，第二个位置为None
            }
        # 调用方法，检查带有多个输入的模型行为，使用WithNonTensorPredicate条件模型
        self.check_model_with_multiple_inputs(
            CondModels.WithNonTensorPredicate(),  # 使用WithNonTensorPredicate条件模型
            inputs,  # 输入数据
            dynamic_shapes=dynamic_shapes,  # 设置动态形状信息
        )

    # 定义一个测试方法，用于测试简单的while循环模型行为
    def test_while_loop_simple(self):
        # 准备输入数据，包括两个张量矩阵，位于指定的设备上
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 创建一个维度对象，表示维度"s0"的范围在2到1024之间
        dim0_ab = Dim("s0", min=2, max=1024)
        # 定义动态形状的字典，其中"ci", "a", "b"是键，每个键对应一个字典，表示动态形状信息
        dynamic_shapes = {
            "ci": {},  # 空字典，表示无动态形状信息
            "a": {0: dim0_ab, 1: None},  # "a"键的动态形状信息，第一个位置有dim0_ab对象，第二个位置为None
            "b": {0: dim0_ab, 1: None},  # "b"键的动态形状信息，第一个位置有dim0_ab对象，第二个位置为None
        }
        # 调用方法，检查带有多个输入的模型行为，使用Simple条件模型
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Simple(),  # 使用Simple条件模型
            prepend_counters(inputs),  # 在输入数据前添加计数器
            dynamic_shapes=dynamic_shapes,  # 设置动态形状信息
        )

    # 定义一个测试方法，用于测试嵌套的while循环模型行为
    def test_while_loop_nested(self):
        # 准备输入数据，包括两个张量矩阵，位于指定的设备上
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 创建一个维度对象，表示维度"s0"的范围在2到1024之间
        dim0_ab = Dim("s0", min=2, max=1024)
        # 定义动态形状的字典，其中"ci", "cj", "a", "b"是键，每个键对应一个字典，表示动态形状信息
        dynamic_shapes = {
            "ci": {},  # 空字典，表示无动态形状信息
            "cj": {},  # 空字典，表示无动态形状信息
            "a": {0: dim0_ab, 1: None},  # "a"键的动态形状信息，第一个位置有dim0_ab对象，第二个位置为None
            "b": {0: dim0_ab, 1: None},  # "b"键的动态形状信息，第一个位置有dim0_ab对象，第二个位置为None
        }
        # 调用方法，检查带有多个输入的模型行为，使用Nested条件模型
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Nested(),  # 使用Nested条件模型
            prepend_counters(inputs, num_counters=2),  # 在输入数据前添加计数器，计数器数量为2
            dynamic_shapes=dynamic_shapes,  # 设置动态形状信息
        )
    # 定义一个测试方法，用于测试带有外部代码的 While 循环模型
    def test_while_loop_with_outer_code(self):
        # 创建输入数据，包括两个大小为 (10, 20) 的随机张量，使用设备 self.device
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 定义维度 "s0"，最小为 2，最大为 1024 的维度对象 dim0_ab
        dim0_ab = Dim("s0", min=2, max=1024)
        # 定义动态形状 dynamic_shapes，包括 "c"、"a"、"b" 三个键值对
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        # 调用 self.check_model_with_multiple_inputs 方法，传入 WhileLoopModels.OuterCode() 模型对象，
        # prepend_counters(inputs) 数据作为输入，以及 dynamic_shapes 作为动态形状参数
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterCode(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # 定义一个测试方法，用于测试带有参数的 While 循环模型
    def test_while_loop_with_parameters(self):
        # 创建输入数据，包括一个大小为 (10, 20) 的随机张量，使用设备 self.device
        inputs = (torch.randn((10, 20), device=self.device),)
        # 定义维度 "s0"，最小为 2，最大为 1024 的维度对象 dim0_a
        dim0_a = Dim("s0", min=2, max=1024)
        # 定义动态形状 dynamic_shapes，包括 "c"、"a" 两个键值对
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_a, 1: None},
        }
        # 调用 self.check_model_with_multiple_inputs 方法，传入 WhileLoopModels.Parameters(self.device) 模型对象，
        # prepend_counters(inputs) 数据作为输入，以及 dynamic_shapes 作为动态形状参数
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Parameters(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # 定义一个测试方法，用于测试带有外部缓冲区的 While 循环模型
    def test_while_loop_with_outer_buffers(self):
        # 创建输入数据，包括两个大小为 (10, 20) 的随机张量，使用设备 self.device
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # 禁用动态形状 dynamic_shapes，由于存在问题 https://github.com/pytorch/pytorch/issues/123596
        dynamic_shapes = None
        # 调用 self.check_model_with_multiple_inputs 方法，传入 WhileLoopModels.OuterBuffers() 模型对象，
        # prepend_counters(inputs) 数据作为输入，以及 dynamic_shapes 作为动态形状参数
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterBuffers(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # 使用 config.patch 方法将 is_predispatch 设置为 True，定义一个测试常量的方法
    @config.patch({"is_predispatch": True})
    def test_constant(self):
        # 定义一个名为 M 的内部类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, x):
                # 创建一个张量 t，其值为 x 的最后一个维度的大小，使用设备 self.device 和 torch.float 类型
                t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
                # 对张量 t 进行开平方和乘法操作
                t = torch.sqrt(t * 3)
                # 返回 x 与 t 的乘积
                return x * t

        # 调用 self.check_model 方法，传入 M(self.device) 模型对象和一个大小为 (5, 5) 的随机张量作为输入
        self.check_model(M(self.device), (torch.randn(5, 5, device=self.device),))
    def test_zero_grid_with_backed_symbols(self):
        # 定义一个名为 `Repro` 的内部类，继承自 `torch.nn.Module`，用于示例模型
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模型的前向传播函数，将输入 `x` 和 `b` 相加并返回
            def forward(self, x, b):
                return x + b

        # 创建示例输入 `example_inputs`
        example_inputs = (
            # 初始化一个形状为 (3, 2) 的张量 `x`，在当前设备上
            x := torch.randn((3, 2), device=self.device),
            # 初始化一个形状为 (1, 2) 的张量，也在当前设备上
            torch.randn((1, 2), device=self.device),
        )
        # 标记张量 `x` 的第 0 维为动态维度，以便后续的编译处理
        torch._dynamo.mark_dynamic(x, index=0)  # Create dynamic symbol

        # 编译模型并运行，其中动态维度大小大于 0
        so_path: str = AOTIRunnerUtil.compile(
            Repro(),
            example_inputs,
        )
        # 加载编译后的 AOT 模块
        aot_inductor_module = AOTIRunnerUtil.load("cuda", so_path)
        # 使用示例输入 `example_inputs` 运行加载的 AOT 模块

        # 再次运行，此时动态维度大小为 0
        example_inputs = (
            # 初始化一个形状为 (0, 2) 的张量 `x`，在当前设备上
            torch.randn((0, 2), device=self.device),
            # 保持之前的形状 (1, 2) 的张量 `b`
            torch.randn((1, 2), device=self.device),
        )
        # 获取 AOT 模块返回的实际结果
        actual = aot_inductor_module(*example_inputs)
        # 获取预期结果，使用 `Repro` 模型计算 `example_inputs` 的输出
        expected = Repro()(*example_inputs)
        # 使用测试框架断言实际结果与预期结果的接近程度
        torch.testing.assert_close(actual, expected)

    def test_repeat_interleave(self):
        # 定义一个名为 `Repro` 的内部类，继承自 `torch.nn.Module`，用于示例模型
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模型的前向传播函数，使用 `torch.ops.aten.repeat_interleave.Tensor` 对输入 `x` 进行重复插值，输出大小为 12
            def forward(self, x):
                return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        # 创建示例输入 `example_inputs`
        example_inputs = (torch.ones((1,), dtype=torch.int32, device=self.device) * 12,)
        # 使用 `self.check_model` 方法检查 `Repro` 模型在给定输入下的输出
        self.check_model(Repro(), example_inputs)

    def test_dynamic_cat(self):
        # 定义一个名为 `Model` 的内部类，继承自 `torch.nn.Module`，用于示例模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模型的前向传播函数，沿指定维度 `dim=0` 对输入张量 `a` 和 `b` 进行拼接
            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        # 初始化形状为 (2, 4) 和 (3, 4) 的张量 `a` 和 `b`，在当前设备上
        a = torch.randn(2, 4, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        # 创建动态形状字典 `dynamic_shapes`，指定张量 `a` 和 `b` 的动态维度范围
        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        # 创建示例输入 `example_inputs`
        example_inputs = (a, b)
        # 使用 `self.check_model` 方法检查 `Model` 模型在给定输入下的输出，并使用动态形状字典进行检查
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_buffer_mutation_1(self):
        # 定义一个名为 `Model` 的内部类，继承自 `torch.nn.Module`，用于示例模型
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 注册名为 `foo` 的缓冲区，形状为 (4, 4)，内容为随机生成的张量，在指定设备上
                self.register_buffer("foo", torch.randn(4, 4, device=device))

            # 模型的前向传播函数，对缓冲区 `foo` 进行就地加法操作，并将结果与输入 `x` 相加后返回
            def forward(self, x):
                self.foo.add_(1)
                return self.foo + x

        # 创建示例输入 `example_inputs`
        example_inputs = (torch.rand(4, 4, device=self.device),)
        # 使用 `self.check_model` 方法检查 `Model` 模型在给定输入下的输出
        self.check_model(Model(self.device), example_inputs)
    # 测试处理非张量输入的情况
    def test_non_tensor_input(self):
        
        # 定义一个函数 fn，接受两个张量 a 和 b，可选参数 alpha，默认为 1.0
        def fn(a, b, alpha=1.0):
            # 使用 torch.add 函数对张量 a 和 b 进行加法操作，乘以 alpha
            return torch.add(a, b, alpha=alpha)

        # 生成一个形状为 (10,) 的随机张量 a，使用 self.device 指定设备
        a = torch.randn(10, device=self.device)
        # 生成一个形状为 (10,) 的随机张量 b，使用 self.device 指定设备
        b = torch.randn(10, device=self.device)
        
        # 断言在下列情况下会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 使用 torch._export.aot_compile 函数编译函数 fn，并传入参数 a, b，以及参数字典 {"alpha": 2.0}
            torch._export.aot_compile(fn, args=(a, b), kwargs={"alpha": 2.0})

        # 遍历 simdlen 可能的取值 [0, None]
        for simdlen in [0, None]:
            # 使用 torch._inductor.config.patch 方法设置配置项 {"cpp.simdlen": simdlen}
            with torch._inductor.config.patch({"cpp.simdlen": simdlen}):
                # 编译 torch.ops.aten.add 操作符，传入参数 a, b，并使用参数字典 {"alpha": 2.0}，不要求签名相同
                so_path = torch._export.aot_compile(
                    torch.ops.aten.add,
                    args=(a, b),
                    kwargs={"alpha": 2.0},
                    same_signature=False,
                )
                # 使用 AOTIRunnerUtil.load_runner 方法加载生成的动态链接库，指定设备 self.device 和路径 so_path
                kernel_runner = AOTIRunnerUtil.load_runner(self.device, so_path)
                # 运行加载的内核运行器 kernel_runner，传入张量列表 [a, b]
                res = kernel_runner.run([a, b])
                # 断言结果 res 是列表类型
                self.assertTrue(isinstance(res, list))
                # 断言结果 res 的长度为 1
                self.assertTrue(len(res) == 1)
                # 断言调用函数 fn 对 a, b 执行加法操作并乘以 alpha=2.0，与 res[0] 相等
                self.assertEqual(fn(a, b, alpha=2.0), res[0])

    # 测试缓冲区变异情况2
    def test_buffer_mutation_2(self):
        
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 注册一个名为 "foo" 的缓冲区，内容为设备上的 [0, 1, 2, ..., 9]
                self.register_buffer("foo", torch.arange(10, device=device))
                # 注册一个名为 "bar" 的缓冲区，内容为设备上的 [0, 1, 2, ..., 9]
                self.register_buffer("bar", torch.arange(10, device=device))

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 在 bar 缓冲区上执行就地乘法运算，乘以 2
                self.bar.mul_(2)
                # 将 foo 缓冲区中索引为 5 的元素设置为 bar 缓冲区中索引为 0 的元素的值
                self.foo[5] = self.bar[0]
                # 返回 x 加上 bar 缓冲区的值，以及 x 乘以 foo 缓冲区的值
                return x + self.bar, x * self.foo

        # 创建一个示例输入元组 example_inputs，包含一个形状为 (10,) 的随机张量，使用 self.device 指定设备
        example_inputs = (torch.randn(10, device=self.device),)
        # 使用 self.check_model 方法检查模型 Model 在 example_inputs 上的行为
        self.check_model(Model(self.device), example_inputs)
    def test_buffer_mutation_4(self):
        # 如果设备不是 CUDA，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 "_tensor_constant0" 的缓冲区，其中包含随机整数张量
                self.register_buffer(
                    "_tensor_constant0",
                    torch.randint(1, size=[38], dtype=torch.int64, device="cpu"),
                )

            def forward(self, x):
                # 返回输入张量 x 和缓冲区 "_tensor_constant0" 的和，将其转移到 CUDA 设备上
                return x + self._tensor_constant0.to(torch.device(type="cuda", index=0))

        # 定义一个在 CUDA 设备上执行 AOT 编译的示例输入
        example_inputs = (
            torch.randint(1, size=[38], dtype=torch.int64, device="cuda"),
        )
        # 使用 torch._export.aot_compile 对模型 Model 进行 AOT 编译
        torch._export.aot_compile(Model(), example_inputs)

    @requires_multigpu()
    # 测试在设备上复制模型
    def test_replicate_on_devices(self):
        # 如果设备不是 CUDA，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self, w1, w2):
                super().__init__()
                self.w1 = w1
                self.w2 = w2

            def forward(self, x, y):
                # 前向传播函数，计算加权和
                a = x * self.w1
                b = y * self.w2
                return a + b

        # 生成随机权重 w1 和 w2
        w1 = torch.randn(10, 10)
        w2 = torch.randn(10, 10)
        # 生成输入数据
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        # 在 CPU 上运行模型，计算结果
        result_cpu = Model(w1, w2)(*inputs)

        # 使用 AOTInductor 编译模型
        with torch.cuda.device(0), config.patch("abi_compatible", self.abi_compatible):
            # 将模型和输入数据编译为 AOT 文件
            so_path = AOTIRunnerUtil.compile(
                model=Model(w1.cuda(0), w2.cuda(0)),
                example_inputs=tuple(t.cuda(0) for t in inputs),
            )

        # 在多个 CUDA 设备上运行模型
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                # 将输入数据移到当前设备
                example_inputs = tuple(t.cuda(i) for t in inputs)
                # 加载优化后的 AOT 模型
                optimized = AOTIRunnerUtil.load("cuda", so_path)
                # 在当前设备上运行优化后的模型
                result_cuda = optimized(*example_inputs)
            # 断言 CPU 和 CUDA 计算结果相同
            self.assertTrue(same(result_cpu, result_cuda.cpu()))

    # 测试接受 PyTree 输入的模型
    def test_pytree_inputs(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: Dict[str, torch.Tensor]):
                # 初始化加法和乘法的结果
                add_ = torch.zeros(5)
                mul_ = torch.ones(5)
                # 遍历输入字典中的张量，累加和累乘操作
                for v in x.values():
                    add_ += v
                    mul_ *= v

                return [add_, mul_]

        # 检查模型 M 的运行结果
        self.check_model(M(), ({"x": torch.ones(5), "y": torch.ones(5)},))

    # 测试在非默认 CUDA 设备上运行模型
    @requires_multigpu()
    def test_non_default_cuda_device(self):
        # 如果设备不是 CUDA，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                # 前向传播函数，计算线性层的输出
                return x + torch.nn.functional.linear(y, self.weight)

        # 生成随机权重 weight 和输入数据 inputs
        weight = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        # 在 CPU 上运行模型，计算结果
        result_cpu = Model(weight)(*inputs)

        # 在 CUDA 设备 0 上运行优化后的模型
        with torch.cuda.device(0), torch.no_grad(), config.patch(
            "abi_compatible", self.abi_compatible
        ):
            result_cuda_0 = AOTIRunnerUtil.run(
                "cuda", Model(weight.cuda(0)), tuple(t.cuda(0) for t in inputs)
            )

        # 在 CUDA 设备 1 上运行优化后的模型
        with torch.cuda.device(1), torch.no_grad(), config.patch(
            "abi_compatible", self.abi_compatible
        ):
            result_cuda_1 = AOTIRunnerUtil.run(
                "cuda", Model(weight.cuda(1)), tuple(t.cuda(1) for t in inputs)
            )

        # 断言 CPU 和 CUDA 计算结果相同
        self.assertTrue(same(result_cpu, result_cuda_0.cpu()))
        self.assertTrue(same(result_cpu, result_cuda_1.cpu()))
    def test_reuse_kernel(self):
        # 定义一个内部类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()

            # 前向传播函数，接受输入参数 x 和 y
            def forward(self, x, y):
                # 计算 x 的正弦值
                a = torch.sin(x)
                # 计算 a 和 y 的矩阵乘积
                b = torch.mm(a, y)
                # 计算 b 的正弦值
                c = torch.sin(b)
                # 计算 b 和 c 的矩阵乘积
                d = torch.mm(b, c)
                # 返回计算结果 d
                return d

        # 示例输入，包括两个随机张量，使用当前设备 self.device
        example_inputs = (
            torch.randn(87, 87, device=self.device),
            torch.randn(87, 87, device=self.device),
        )
        # 创建 Model 实例
        model = Model()
        # 调用 self.check_model 方法，检查模型行为是否符合预期，设置容差为 1e-4
        self.check_model(
            model, example_inputs, atol=1e-4, rtol=1e-4
        )  # 1e-4 is the tol value used in pytorch/torch/_dynamo/utils.py

        # 如果当前设备是 "cuda"
        if self.device == "cuda":
            # 调用 self.code_check_count 方法，检查模型的代码行为
            self.code_check_count(
                model, example_inputs, "triton_poi_fused_sin_0 = loadKernel(", 1
            )
    def test_reuse_kernel_dynamic(self):
        # 定义一个测试方法，用于测试动态重用内核的情况

        class Model(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的模型类
            def __init__(self, device):
                # 模型初始化方法
                super().__init__()
                # 初始化一个大小为 48 的随机张量，设备为给定设备，数据类型为浮点型
                self.cst = torch.randn(48, device=device, dtype=torch.float)
                # 初始化一个大小为 [6, 48, 48] 的随机张量，设备为给定设备，数据类型为浮点型
                self.weights = torch.randn(6, 48, 48, device=device, dtype=torch.float)
                # 初始化一个大小为 48 的随机张量，设备为给定设备，数据类型为浮点型
                self.cst_1 = torch.randn(48, device=device, dtype=torch.float)
                # 初始化一个大小为 [6, 48, 48] 的随机张量，设备为给定设备，数据类型为浮点型
                self.weights_1 = torch.randn(
                    6, 48, 48, device=device, dtype=torch.float
                )

            def forward(self, x, y, z):
                # 定义模型的前向传播方法
                dim0 = x.size(1)
                # 计算 z + z
                add_0 = z + z
                # 将 add_0 在最后一个维度上扩展为大小为 48
                expand_2 = add_0.expand(-1, -1, 48)
                # 计算 add_0 与 expand_2 的逐元素乘积
                mul_3 = add_0 * expand_2
                # 将 mul_3 的维度进行转置，顺序为 (1, 0, 2)
                permute_4 = torch.permute(mul_3, (1, 0, 2))
                # 使用 self.weights 执行批矩阵乘操作
                bmm_5 = torch.bmm(permute_4, self.weights)
                # 将 bmm_5 与 self.cst 相加
                add_6 = bmm_5 + self.cst
                # 将 add_6 重塑为 [6, dim0 * 6, 8] 的张量
                reshape_7 = torch.reshape(add_6, [6, dim0 * 6, 8])
                # 将 reshape_7 的维度进行转置，顺序为 (1, 0, 2)
                permute_8 = torch.permute(reshape_7, (1, 0, 2))
                # 将 permute_8 与标量 0.123 执行逐元素乘法
                mul_9 = permute_8 * 0.123
                # 将 y 重塑为 [8, dim0 * 6, 4] 的张量
                reshape_10 = torch.reshape(y, [8, dim0 * 6, 4])
                # 将 reshape_10 的维度进行转置，顺序为 (1, 0, 2)
                permute_11 = torch.permute(reshape_10, (1, 0, 2))
                # 使用 mul_9 与 permute_11 执行批矩阵乘操作
                bmm_12 = torch.bmm(mul_9, permute_11)

                # 以下是与上述过程类似的操作，但是使用了另一组权重和偏置
                add_0_1 = z + z
                expand_2_1 = add_0_1.expand(-1, -1, 48)
                mul_3_1 = add_0_1 * expand_2_1
                permute_4_1 = torch.permute(mul_3_1, (1, 0, 2))
                bmm_5_1 = torch.bmm(permute_4_1, self.weights_1)
                add_6_1 = bmm_5_1 + self.cst_1
                reshape_7_1 = torch.reshape(add_6_1, [6, dim0 * 6, 8])
                permute_8_1 = torch.permute(reshape_7_1, (1, 0, 2))
                mul_9_1 = permute_8_1 * 0.123
                reshape_10_1 = torch.reshape(y, [8, dim0 * 6, 4])
                permute_11_1 = torch.permute(reshape_10_1, (1, 0, 2))
                bmm_12_1 = torch.bmm(mul_9_1, permute_11_1)
                # 返回两次计算结果的和
                return bmm_12 + bmm_12_1

        # 生成随机张量作为模型输入的示例
        x = torch.randn(6, 2, 48, device=self.device, dtype=torch.float)
        y = torch.randn(48, 2, 4, device=self.device, dtype=torch.float)
        z = torch.randn(2, 6, 1, device=self.device, dtype=torch.float)
        # 定义一个动态维度对象，包含 x、y、z 的动态维度限制
        dim0 = Dim("dim0", min=1, max=2048)
        dynamic_shapes = {
            "x": {1: dim0},
            "y": {1: dim0},
            "z": {0: dim0},
        }
        # 将输入数据打包为元组
        example_inputs = (x, y, z)
        # 创建模型实例，并将其转换为浮点数类型
        m = Model(self.device).to(dtype=torch.float)
        # 调用检查模型方法，验证模型行为是否符合预期
        self.check_model(m, example_inputs, dynamic_shapes=dynamic_shapes)
    def test_fake_tensor_device_validation(self):
        # 检查当前设备是否为 CUDA，若不是则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个简单的模型类，用于测试
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        # 在 CPU 上导出模型
        exported_program = export(Model(), example_inputs)

        # 将导出的模型编译到 CUDA 上
        gm = exported_program.graph_module.to(self.device)
        # 使用带有自定义错误信息的断言检查编译时可能出现的设备不匹配错误
        with self.assertRaisesRegex(ValueError, "Device mismatch between fake input"):
            torch._inductor.aot_compile(
                gm, tuple(i.to(self.device) for i in example_inputs)
            )

    @unittest.mock.patch("torch._inductor.graph.supported_dtype_of_cpp_wrapper")
    def test_unsupported_input_dtype(self, supported_dtype_of_cpp_wrapper_mock):
        supported_dtype_of_cpp_wrapper_mock.return_value = False

        # 定义一个简单的模型类，用于测试
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(10, 10).to(self.device),
            torch.randn(10, 10).to(self.device),
        )
        # 使用带有自定义错误信息的断言检查编译时可能出现的不支持的数据类型错误
        with self.assertRaisesRegex(
            CppWrapperCodeGenError, "Unsupported input dtype torch.float32"
        ):
            torch._export.aot_compile(Model(), example_inputs)

        # 验证模拟函数被调用一次，检查输入数据类型是否被支持
        supported_dtype_of_cpp_wrapper_mock.assert_called_once_with(
            torch.float32, self.device == "cuda"
        )

    def test_consecutive_compiles(self):
        """Test that compilation behaves correctly with cache hits"""
        # 定义一个测试模型，用于检查编译是否正确缓存
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        mod = TestModule()
        inp = torch.rand(1)
        mod(inp)
        mod2 = torch.fx.symbolic_trace(mod, concrete_args=[inp])
        # 编译模型并断言是否成功
        so = torch._export.aot_compile(mod2, (inp,))
        assert so is not None
        # 第二次编译应有缓存命中，断言是否成功
        so = torch._export.aot_compile(mod2, (inp,))
        assert so is not None

    def test_normal_functional(self):
        # 定义一个模型类，测试正常的函数操作
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.normal_functional.default(x)

        self.check_model(Model(), (torch.empty(4, 1, 4, 4),))

    def test_empty_graph(self):
        # 定义一个模型类，测试空图的情况
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        example_inputs = (torch.randn(8, 4, 4, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    # 定义一个测试方法，用于测试未备份符号声明的情况
    def test_dup_unbacked_sym_decl(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模型的前向传播方法
            def forward(self, x):
                # 计算输入张量x的绝对值
                abs_1 = torch.ops.aten.abs.default(x)
                # 判断abs_1中小于0.001的元素，并返回一个布尔张量
                lt = torch.ops.aten.lt.Scalar(abs_1, 0.001)
                # 判断lt中等于0的元素，并返回一个布尔张量
                eq = torch.ops.aten.eq.Scalar(lt, 0)
                # 使用eq作为索引条件，从输入张量x中选择符合条件的元素
                index_1 = torch.ops.aten.index.Tensor(x, [eq])
                # 对index_1中的元素求正弦
                sin = torch.ops.aten.sin.default(index_1)
                # 再次使用eq作为索引条件，从输入张量x中选择符合条件的元素
                index_2 = torch.ops.aten.index.Tensor(x, [eq])
                # 将sin张量中的元素按照index_2中的元素进行除法运算
                div_3 = torch.ops.aten.div.Tensor(sin, index_2)
                # 返回除法运算结果
                return div_3

        # 创建一个例子输入，包含一个随机张量，并将模型和输入传入self.check_model方法中进行测试
        example_inputs = (torch.randn(4, 4, 4, 4).to(self.device),)
        self.check_model(Model(), example_inputs)

    # 该测试方法用于在ShapeEnv中测试_eliminate_unbacked路径
    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_dup_unbacked_sym_decl_with_refinement(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模型的前向传播方法
            def forward(self, x):
                # 计算输入张量x的绝对值
                abs_1 = torch.ops.aten.abs.default(x)
                # 判断abs_1中小于0.001的元素，并返回一个布尔张量
                lt = torch.ops.aten.lt.Scalar(abs_1, 0.001)
                # 判断lt中等于0的元素，并返回一个布尔张量
                eq = torch.ops.aten.eq.Scalar(lt, 0)
                # 使用eq作为索引条件，从输入张量x中选择符合条件的元素
                index_1 = torch.ops.aten.index.Tensor(x, [eq])
                # 检查index_1的大小是否等于4的4次方
                torch._check(index_1.size(0) == 4**4)
                # 对index_1中的元素求正弦
                sin = torch.ops.aten.sin.default(index_1)
                # 再次使用eq作为索引条件，从输入张量x中选择符合条件的元素
                index_2 = torch.ops.aten.index.Tensor(x, [eq])
                # 将sin张量中的元素按照index_2中的元素进行除法运算
                div_3 = torch.ops.aten.div.Tensor(sin, index_2)
                # 返回除法运算结果
                return div_3

        # 创建一个例子输入，包含一个全为1的张量，并将模型和输入传入self.check_model方法中进行测试
        example_inputs = (torch.ones(4, 4, 4, 4).to(self.device),)
        self.check_model(Model(), example_inputs)

    # 该测试方法用于测试在梯度启用状态下运行模型
    def test_run_with_grad_enabled(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 模型的前向传播方法，执行矩阵相乘加法操作
            def forward(self, x, weight, bias):
                return torch.ops.aten.addmm(bias, weight, x)

        # 创建一个Model实例并将其移动到设备self.device
        m = Model().to(device=self.device)
        # 创建三个张量：x、weight、bias，并将其设置为需要梯度的张量
        x = torch.rand(8, 8, device=self.device, requires_grad=True)
        weight = torch.rand(8, 8, device=self.device, requires_grad=True)
        bias = torch.rand(8, device=self.device, requires_grad=True)
        example_inputs = (x, weight, bias)

        # 计算预期输出，并将其展平成一维张量
        expected = m(*example_inputs)
        expected = pytree.tree_leaves(expected)

        # 在没有梯度的情况下编译模型，并获取动态链接库路径
        with torch.no_grad():
            so_path = AOTIRunnerUtil.compile(m, example_inputs)

        # 断言梯度是否被启用
        self.assertTrue(torch.is_grad_enabled())

        # 加载优化后的模型，并执行前向传播计算实际输出，再将其展平成一维张量
        optimized = AOTIRunnerUtil.load(self.device, so_path)
        actual = optimized(*example_inputs)
        actual = pytree.tree_leaves(actual)

        # 断言实际输出与预期输出是否相同
        self.assertTrue(same(actual, expected))
    # 定义一个测试方法，测试模型返回常量的情况
    def test_return_constant(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 初始化一个大小为 5x5 的张量，内容为设备上随机生成的数据
                self.cst = torch.randn(5, 5, device=device)

            def forward(self, x):
                # 对常量张量进行克隆操作
                a = self.cst.clone()
                return (x, a)

        # 生成一个大小为 5 的随机张量，使用 self.device 设备
        x = torch.randn(5, device=self.device)
        # 使用自定义的检查方法检查模型输出
        self.check_model(Model(self.device), (x,))

    # 定义一个测试方法，测试模型返回转置后的常量的情况
    def test_return_view_constant(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 初始化一个大小为 5x5 的张量，内容为设备上随机生成的数据
                self.cst = torch.randn(5, 5, device=device)

            def forward(self, x):
                # 返回常量张量的转置
                a = torch.transpose(self.cst, 0, 1)
                return (x, a)

        # 生成一个大小为 5 的随机张量，使用 self.device 设备
        x = torch.randn(5, device=self.device)
        # 使用自定义的检查方法检查模型输出
        self.check_model(Model(self.device), (x,))

    # 定义一个测试方法，测试带有性能分析器的模型执行情况
    def test_with_profiler(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个大小为 10x10 的线性层
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # 返回输入 x 与线性层作用在 y 上的结果
                return x + self.linear(y)

        # 生成两个大小为 10x10 的随机张量，使用 self.device 设备
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        # 使用给定配置对性能分析器进行设置，然后检查模型输出
        with config.patch({"profile_bandwidth": "1", "profile_bandwidth_regex": ""}):
            self.check_model(Model(), example_inputs)

    # 定义一个测试方法，测试不带 Triton 性能分析器的模型执行情况
    def test_with_no_triton_profiler(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 返回输入张量的转置
                return torch.permute(x, (1, 0))

        # 生成一个大小为 10x10 的随机张量，使用 self.device 设备
        example_inputs = (torch.randn(10, 10, device=self.device),)
        # 使用给定配置对性能分析器进行设置，然后检查模型输出
        with config.patch({"profile_bandwidth": "1", "profile_bandwidth_regex": ""}):
            self.check_model(Model(), example_inputs)

    # 定义一个测试方法，测试模型返回重复输出的情况
    def test_repeat_output(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 对输入张量执行正弦函数操作
                y = torch.sin(x)
                # 返回两次相同的输出
                return y, y

        # 生成一个大小为 3x10 的随机张量，使用 self.device 设备
        example_inputs = (torch.randn(3, 10, device=self.device),)
        # 使用自定义的检查方法检查模型输出
        self.check_model(Model(), example_inputs)

    # 定义一个测试方法，测试模型返回不同视图大小的输出情况
    def test_view_outputs(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            def forward(self, x):
                # 对输入张量执行正弦函数操作
                y = torch.sin(x)
                # 返回原始大小、相同大小和不同大小视图的输出
                y_same_size = y.view(*y.shape)
                y_diff_size = y.view(1, *y.shape)
                return y, y_same_size, y_diff_size

        # 生成一个大小为 3x10 的随机张量，使用 self.device 设备
        example_inputs = (torch.randn(3, 10, device=self.device),)
        # 使用自定义的检查方法检查模型输出
        self.check_model(Model(), example_inputs)

    # 跳过没有 TorchVision 的测试用例
    @skip_if_no_torchvision
    # 定义一个测试方法，用于测试缺失的 cubin 文件情况
    def test_missing_cubin(self):
        # 导入 torchvision 库中的 Bottleneck 和 ResNet 模型类
        from torchvision.models.resnet import Bottleneck, ResNet

        # 定义一个继承自 ResNet 的 Model 类
        class Model(ResNet):
            def __init__(self):
                # 调用 ResNet 类的初始化方法，配置网络结构
                super().__init__(
                    block=Bottleneck,  # 使用 Bottleneck 作为基本块
                    layers=[3, 4, 6, 3],  # 每个阶段的块数量
                    replace_stride_with_dilation=[False, False, True],  # 是否替换步长为扩展
                    norm_layer=None,  # 不使用标准化层
                )

            def forward(self, x):
                # 网络的前向传播过程
                x = self.conv1(x)  # 第一层卷积操作
                x = self.bn1(x)  # 第一层批量归一化操作
                x = self.relu(x)  # 第一层激活函数 ReLU
                f1 = x  # 第一特征图
                x = self.maxpool(x)  # 最大池化操作
                x = self.layer1(x)  # 第一阶段的 ResNet 模块
                f2 = x  # 第二特征图
                x = self.layer2(x)  # 第二阶段的 ResNet 模块
                f3 = x  # 第三特征图
                x = self.layer3(x)  # 第三阶段的 ResNet 模块
                x = self.layer4(x)  # 第四阶段的 ResNet 模块
                f4 = x  # 第四特征图
                return [f1, f2, f3, f4]  # 返回所有阶段的特征图列表

        # 调用 Model 类的构造方法，将模型移到指定设备上，并设置数据类型为 float64，并设为评估模式
        model = Model().to(device=self.device, dtype=torch.float64).eval()
        # 创建一个示例输入，形状为 (4, 3, 64, 64)，放在指定设备上，数据类型为 float64
        example_inputs = (
            torch.randn(4, 3, 64, 64, device=self.device, dtype=torch.float64),
        )
        # 调用 self.check_model 方法来验证模型在示例输入上的输出
        self.check_model(model, example_inputs)

    # 使用 common_utils.parametrize 装饰器，对下面的四个参数进行参数化组合
    @common_utils.parametrize("grid_type", [1, 2, 3])
    @common_utils.parametrize("num_dims", [1, 2])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotune", [False, True])
    def test_triton_kernel(self, grid_type, num_dims, dynamic, autotune):
        # 检查是否使用 CUDA 设备，否则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个内部模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # 创建一个和 x 相同大小的零张量 output
                output = torch.zeros_like(x)
                # 如果开启自动调优并且维度为 2
                if autotune and num_dims == 2:
                    # 获取 x 和 y 的元素个数
                    x_elements = output.size()[0]
                    y_elements = output.size()[1]
                else:
                    # 否则获取 output 的元素总数
                    n_elements = output.numel()

                # 选择网格
                if autotune and num_dims == 2:
                    # 如果是 2 维并且 grid_type 为 1
                    if grid_type == 1:
                        grid = (x_elements, y_elements)
                    # 如果 grid_type 为 2
                    elif grid_type == 2:
                        grid = lambda meta: (
                            triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                            triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                        )
                    else:
                        # 否则定义一个 grid_fn 函数并赋给 grid
                        def grid_fn(meta):
                            return (
                                triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                                triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                            )

                        grid = grid_fn
                else:
                    # 如果不是 2 维或者未开启自动调优
                    if grid_type == 1:
                        grid = (n_elements,)
                    elif grid_type == 2:
                        grid = lambda meta: (
                            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                        )
                    else:
                        # 否则定义一个 grid_fn 函数并赋给 grid
                        def grid_fn(meta):
                            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                        grid = grid_fn

                # 选择内核
                if autotune:
                    # 如果开启自动调优
                    if num_dims == 1:
                        # 如果维度为 1，则调用 add_kernel_autotuned 内核
                        add_kernel_autotuned[grid](x, y, output, n_elements)
                    else:
                        # 否则调用 add_kernel_2d_autotuned 内核
                        add_kernel_2d_autotuned[grid](
                            x, y, output, x_elements, y_elements
                        )
                else:
                    # 如果未开启自动调优，则调用普通 add_kernel 内核
                    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                # 返回输出张量 output
                return output

        # 创建一个维度为 num_dims 的列表 dims，并初始化 x 和 y
        dims = [10] * num_dims
        x = torch.randn(*dims, device=self.device)
        y = torch.randn(*dims, device=self.device)
        dynamic_shapes = []
        # 如果 dynamic 参数为 True
        if dynamic:
            # 创建名为 dim0_x 和 dim0_y 的维度对象，并放入 dynamic_shapes 字典
            dim0_x = Dim("dim0_x", min=1, max=10)
            dim0_y = Dim("dim0_y", min=1, max=10)
            dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        # 使用 check_model 方法检查 Model 的输出结果
        self.check_model(Model(), (x, y), dynamic_shapes=dynamic_shapes)
    def test_triton_kernel_dynamic_shape_with_div(self):
        # 检查设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个使用 Triton JIT 编译的内核函数 pass_kernel
        @triton.jit
        def pass_kernel(x, num):
            pass

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 根据输入张量 x 的元素数量计算 num
                num = x.numel() // 4

                # 使用 lambda 函数定义 grid 函数，计算并返回 grid 元组
                grid = lambda meta: (triton.cdiv(num, 16),)  # noqa: E731
                # 调用 JIT 编译的 pass_kernel 函数，传入参数 x 和 num
                pass_kernel[grid](x, num)
                return x

        # 创建一个形状为 (10,) 的随机张量 x，指定设备为 self.device
        x = torch.randn(10, device=self.device)
        # 创建一个名为 dim0_x 的动态维度对象，限制在 1 到 10 之间
        dim0_x = Dim("dim0_x", min=1, max=10)
        # 创建 dynamic_shapes 字典，指定键为 'x'，值为包含动态维度 dim0_x 的字典
        dynamic_shapes = {"x": {0: dim0_x}}
        # 调用 self.check_model 方法，传入 Model 实例、输入张量 x 和 dynamic_shapes
        self.check_model(Model(), (x,), dynamic_shapes=dynamic_shapes)

    def test_triton_kernel_reinterpret_view(self):
        # 检查设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个使用 Triton JIT 编译的内核函数 pass_kernel
        @triton.jit
        def pass_kernel(x, y):
            pass

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 根据输入张量 x 创建与其同样形状的全零张量 out
                out = torch.zeros_like(x[:, 4:])
                # 下面的切片操作创建了两个 ReinterpretView 实例，分别为 offset=3 和 offset=4
                add_kernel[(10,)](
                    in_ptr0=x[:, 3:-1],
                    in_ptr1=x[:, 4:],
                    out_ptr=out,
                    n_elements=160,
                    BLOCK_SIZE=16,
                )
                return out

        # 创建一个形状为 (10, 20) 的随机张量 example_inputs[0]，指定设备为 self.device
        example_inputs = (torch.randn(10, 20, device=self.device),)
        # 调用 self.check_model 方法，传入 Model 实例和 example_inputs
        self.check_model(Model(), example_inputs)

    def test_triton_kernel_sympy_expr_arg(self):
        # 检查设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def forward(self, x, e):
                # 计算 sympy_expr，取 e 的最大值并转换为标量
                sympy_expr = max(1, e.item())
                # 创建与输入张量 x 形状相同的全零张量 out
                out = torch.zeros_like(x)
                # 调用 add_kernel 函数，传入参数指针和其他参数
                add_kernel[(1,)](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=sympy_expr,
                    BLOCK_SIZE=1,
                )
                return out

        # 定义 NUMEL 为 64
        NUMEL = 64
        # 创建两个输入张量 inputs，分别是形状为 (NUMEL,) 的随机张量和包含标量 NUMEL 的张量
        inputs = (
            torch.randn(NUMEL, device=self.device),
            torch.tensor(NUMEL, device=self.device),
        )
        # 调用 self.check_model 方法，传入 Model 实例和 inputs
        self.check_model(Model(), inputs)
    def test_triton_kernel_with_none_input(self):
        # 检查当前设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 获取输入张量 x 的元素数量
                n_elements = x.size()[0]
                # 定义一个块大小常量
                BLOCK_SIZE = 1024

                # 创建一个与输入张量 x 同样大小的空张量 output_wo_y
                output_wo_y = torch.empty_like(x)
                # 创建一个与输入张量 x 同样大小的空张量 output_with_y
                output_with_y = torch.empty_like(x)

                # 调用带有可选参数的 add_kernel 函数，处理无 y 输入的情况
                wo_kernel = add_kernel_with_optional_param[(1,)](
                    x,
                    None,
                    output_wo_y,
                    n_elements,
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                # 调用带有可选参数的 add_kernel 函数，处理有 y 输入的情况
                with_kernel = add_kernel_with_optional_param[(1,)](
                    x,
                    y,
                    output_with_y,
                    n_elements,
                    ARGS_PASSED="two",
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                # 返回两个结果的加权和
                return 2.71 * output_wo_y + 3.14 * output_with_y

        # 定义一个示例输入元组
        example_inputs = (
            torch.randn(1023, device=self.device),
            torch.randn(1023, device=self.device),
        )

        # 调用测试函数检查模型输出
        self.check_model(Model(), example_inputs)

    def test_triton_kernel_equal_to_1_arg(self):
        # 检查当前设备是否为 CUDA，如果不是，则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 创建一个与输入张量 x 同样大小的空张量 out
                out = torch.empty_like(x)
                # 获取输入张量 x 的元素数量
                n_elements = x.numel()
                # 调用带有可选参数的 add_kernel 函数，处理单个输入的情况
                add_kernel[(n_elements,)](x, y, out, n_elements, BLOCK_SIZE=16)
                # 返回计算结果张量 out
                return out

        # 定义一个示例输入元组
        example_inputs = (
            torch.randn(1, device=self.device),
            torch.randn(1, device=self.device),
        )

        # 调用测试函数检查模型输出
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("dynamic", [False, True])
    # 定义一个测试方法，用于测试 Triton 内核是否等于 1，并接受一个动态参数
    def test_triton_kernel_equal_to_1_float_arg(self, dynamic):
        # 检查当前设备是否为 CUDA，如果不是则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 根据输入 x 创建一个相同形状的空张量 out
                out = torch.empty_like(x)
                # 计算 x 中元素的总数
                n_elements = x.numel()
                # 计算缩放因子，此处应为 0 次方计算错误，应该是 1 次方
                scaling_factor = (n_elements**0) / 1.0
                # 调用 add_kernel_with_scaling 函数来执行特定的计算操作
                add_kernel_with_scaling[(n_elements,)](
                    x,
                    y,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                # 返回计算结果 out
                return out

        # 初始化动态形状为 None
        dynamic_shapes = None
        # 如果 dynamic 参数为 True，则创建动态形状字典
        if dynamic:
            # 创建维度限制对象 dim0_xy，限制在 2 到 1024 之间
            dim0_xy = Dim("s0", min=2, max=1024)
            # 设置动态形状为 x 和 y 的字典映射
            dynamic_shapes = {
                "x": {0: dim0_xy, 1: None},
                "y": {0: dim0_xy, 1: None},
            }
        
        # 创建示例输入数据，包括两个随机张量，使用当前设备
        example_inputs = (
            torch.randn(2, device=self.device),
            torch.randn(2, device=self.device),
        )
        # 调用 self.check_model 方法来测试 Model 类的功能
        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    # 定义一个测试方法，用于测试偏移约束的范围
    def test_shifted_constraint_ranges(self):
        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 定义模型的初始化方法
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法，接受两个张量参数 x 和 y
            def forward(
                self,
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                # 检查 y 的大小是否等于 x 的大小加一
                torch._check(y.size(0) == x.size(0) + 1)
                # 返回 x 和 y 按第 0 维度求和的结果
                return x.sum(0) + y.sum(0)

        # 创建两个随机张量 a 和 b，使用当前设备
        a = torch.randn((4, 5), device=self.device)
        b = torch.randn((5, 5), device=self.device)
        # 创建维度对象 dim0_x，限制在 2 到 1024 之间
        dim0_x = Dim("dim0_x", min=2, max=1024)
        # 计算动态形状中 y 的维度为 dim0_x + 1
        dim0_y = dim0_x + 1
        # 创建动态形状字典，指定 x 和 y 的动态形状
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        # 调用 self.check_model 方法来测试 Model 类的功能
        self.check_model(
            Model(),
            (a, b),
            dynamic_shapes=dynamic_shapes,
        )

    # 定义一个测试方法，用于测试散布操作的后备方案
    def test_scatter_fallback(self):
        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 定义模型的初始化方法
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法，接受三个张量参数 inp、index 和 src
            def forward(
                self,
                inp: torch.Tensor,
                index: torch.Tensor,
                src: torch.Tensor,
            ):
                # 执行 torch 中的散布操作，返回结果
                return torch.scatter(inp, 1, index, src)

        # 创建三个示例输入张量，使用当前设备和数据类型
        inputs = (
            torch.ones((3, 5), device=self.device, dtype=torch.int64),
            torch.tensor([[0, 1, 2, 0]], device=self.device, dtype=torch.int64),
            torch.zeros((2, 5), device=self.device, dtype=torch.int64),
        )

        # 调用 self.check_model 方法来测试 Model 类的功能
        self.check_model(Model(), inputs)
    def test_scatter_reduce_fallback(self):
        # 定义一个继承自torch.nn.Module的模型类Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法
            def forward(
                self,
                inp: torch.Tensor,  # 输入张量
                index: torch.Tensor,  # 索引张量
                src: torch.Tensor,  # 源张量
            ):
                # 使用torch.scatter_reduce函数对输入张量进行操作，求和
                return torch.scatter_reduce(inp, 0, index, src, reduce="sum")

        # 定义测试用例的输入数据
        inputs = (
            torch.tensor([1, 10, 100, 1000], device=self.device, dtype=torch.int64),  # 输入张量
            torch.tensor([0, 1, 0, 1, 2, 1], device=self.device, dtype=torch.int64),  # 索引张量
            torch.tensor([1, 2, 3, 4, 5, 6], device=self.device, dtype=torch.int64),  # 源张量
        )

        # 调用测试方法检查模型
        self.check_model(Model(), inputs)

    def test_index_put_fallback(self):
        # 进入确定性模式下的index_put方法
        with DeterministicGuard(True):

            # 定义一个继承自torch.nn.Module的模型类Model
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                # 定义模型的前向传播方法
                def forward(
                    self,
                    self_tensor: torch.Tensor,  # 输入张量
                    indices: Tuple[torch.Tensor],  # 索引元组
                    values: torch.Tensor,  # 值张量
                ):
                    # 使用torch.index_put函数对输入张量进行操作，累加更新
                    return torch.index_put(
                        self_tensor, indices, values, accumulate=True
                    )

            # 定义测试用例的输入数据
            inputs = (
                torch.ones(4, device=self.device, dtype=torch.int64),  # 输入张量
                (torch.tensor([1, 1, 2, 2], device=self.device, dtype=torch.bool),),  # 索引元组
                torch.ones(4, device=self.device, dtype=torch.int64),  # 值张量
            )

            # 调用测试方法检查模型
            self.check_model(Model(), inputs)

    def test_convolution(self):
        # 定义一个继承自torch.nn.Module的模型类Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模型的前向传播方法
            def forward(self, x, w, b):
                # 使用torch.ops.aten.convolution函数进行卷积操作
                return torch.ops.aten.convolution(x, w, b, [4], [0], [1], True, [0], 1)

        # 定义示例输入数据
        example_inputs = (
            torch.randn([2, 32, 90], device=self.device),  # 输入张量x
            torch.randn([32, 16, 8], device=self.device),  # 权重张量w
            torch.randn([16], device=self.device),  # 偏置张量b
        )

        # 使用配置上下文修改参数配置
        with config.patch(
            {
                "max_autotune": True,  # 开启自动调优
                "max_autotune_gemm_backends": "Triton",  # 指定自动调优的后端为Triton
            }
        ):
            # 调用测试方法检查模型
            self.check_model(Model(), example_inputs)
    # 定义一个测试方法，用于测试模型在权重为零的情况下的行为
    def test_zero_size_weight(self):
        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self, channel, r=8):
                super().__init__()
                # 创建一个自适应平均池化层，输出大小为1x1
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                # 创建一个顺序网络，包含两个线性层和两个激活函数
                self.net = torch.nn.Sequential(
                    # 第一个线性层，将输入特征通道数减少到原来的1/8，没有偏置项
                    torch.nn.Linear(channel, channel // r, bias=False),
                    torch.nn.ReLU(inplace=True),  # 使用 ReLU 激活函数，inplace=True 表示原地操作
                    # 第二个线性层，将特征通道数恢复到原始大小，没有偏置项
                    torch.nn.Linear(channel // r, channel, bias=False),
                    torch.nn.Sigmoid(),  # 使用 Sigmoid 激活函数
                )

            def forward(self, inp):
                b, c, _, _ = inp.shape  # 获取输入张量的形状信息
                x = self.pool(inp).view(b, c)  # 经过池化层后，将结果展平成一维张量
                x = self.net(x).view(b, c, 1, 1)  # 经过顺序网络，然后恢复形状
                x = inp * x  # 将原始输入张量与计算结果相乘
                return x  # 返回处理后的张量作为输出

        inputs = (torch.rand(4, 4, 4, 4, device=self.device),)  # 创建随机输入张量元组
        self.check_model(Model(4), inputs)  # 调用检查模型方法，检查模型行为

    # 定义一个测试方法，用于测试没有参数的模型行为
    def test_no_args(self):
        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self, m, n):
                super().__init__()
                # 创建一个模型参数，包含随机初始化的权重矩阵
                self.weight = torch.nn.Parameter(
                    torch.randn(m, n),
                )
                # 创建一个模型参数，包含随机初始化的 alpha 矩阵
                self.alpha = torch.nn.Parameter(torch.randn(m, n))

            def forward(self):
                return self.weight * self.alpha  # 返回权重和 alpha 矩阵的乘积作为输出

        self.check_model(Model(6, 4), ())  # 调用检查模型方法，检查模型行为

    # 定义一个测试方法，用于测试动态标量的模型行为
    def test_dynamic_scalar(self):
        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个交叉熵损失函数对象，设定 reduction="none" 参数
                self.criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")

            def forward(self, inputs, targets, split_index=None):
                statistics = {}
                total_loss = self.criterion_ce(inputs, targets).sum()  # 计算交叉熵损失的总和
                statistics["dl"] = total_loss.item()  # 将总损失存储在 statistics 字典中
                return total_loss, statistics  # 返回总损失和统计信息作为输出

        inputs = (
            torch.rand(4, 4, 4, 4, device=self.device),  # 创建随机输入张量
            torch.rand(4, 4, 4, 4, device=self.device),  # 创建随机目标张量
        )
        self.check_model(Model(), inputs)  # 调用检查模型方法，检查模型行为
    # 定义一个测试方法，用于测试模块的常量原始全限定名和数据类型
    def test_constant_original_fqn_and_dtype(self):
        # 定义一个继承自 torch.nn.Module 的类 FooBarModule
        class FooBarModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 "0" 的参数，值为 3x4 的随机张量
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                # 注册一个名为 "test_buf" 的缓冲区，值为 3x4 的随机张量
                self.register_buffer("test_buf", torch.randn(3, 4))
                # 注册一个名为 "test_param" 的参数，值为 3x4 的随机张量
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            # 前向传播方法
            def forward(self, x):
                # 返回计算结果：((输入张量 x + self.test_buf) * self.0) / self.test_param
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        # 定义一个继承自 torch.nn.Module 的类 TestModule
        class TestModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 FooBarModule 实例
                self.foo_bar = FooBarModule()
                # 注册一个名为 "test_param" 的参数，值为 3x4 的随机张量
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                # 注册一个名为 "test_buf" 的缓冲区，值为 3x4 的随机张量
                self.register_buffer("test_buf", torch.randn(3, 4))

            # 前向传播方法
            def forward(self, x):
                # 返回计算结果：(self.foo_bar(x) + self.test_param) * self.test_buf
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        # 进入 torch.no_grad 上下文
        with torch.no_grad():
            # 编译 TestModule 的模型到指定设备上，并获取共享对象文件的路径
            so_path = AOTIRunnerUtil.compile(
                model=TestModule().to(device=self.device),
                example_inputs=(torch.rand(3, 4, device=self.device),),
            )

        # 根据设备和共享对象路径加载运行时对象
        runner = AOTIRunnerUtil.load_runner(self.device, so_path)

        # 预期的常量原始全限定名字典
        expected_original_fqns = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "getattr_L__self___foo_bar___0__": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        # 断言获取的常量原始全限定名字典与预期值相等
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        # 预期的常量数据类型字典
        expected_dtypes = {
            "L__self___test_param": 6,
            "L__self___test_buf": 6,
            "getattr_L__self___foo_bar___0__": 6,
            "L__self___foo_bar_test_param": 6,
            "L__self___foo_bar_test_buf": 6,
        }
        # 断言获取的常量数据类型字典与预期值相等
        self.assertEqual(expected_dtypes, runner.get_constant_names_to_dtypes())
    def test_fqn(self):
        # 定义嵌套的子类 NestedChild，继承自 torch.nn.Module
        class NestedChild(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 nestedchild3buffer 的缓冲区，初始值为 3 的全 1 矩阵
                self.register_buffer("nestedchild3buffer", torch.ones(2, 3) * 3)

            # 前向传播函数，对输入 x 进行处理
            def forward(self, x):
                # 返回 x 除以 self.nestedchild3buffer 的结果
                return x / self.nestedchild3buffer

        # 定义子类 Child1，继承自 torch.nn.Module
        class Child1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 NestedChild 实例
                self.nested = NestedChild()
                # 注册一个名为 child1param 的参数，初始值为 1 的全 1 矩阵
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 前向传播函数，对输入 x 进行处理
            def forward(self, x):
                # 使用 NestedChild 对象处理输入 x
                x = self.nested(x)
                # 返回 x 加上 self.child1param 的结果
                return x + self.child1param

        # 定义子类 Child2，继承自 torch.nn.Module
        class Child2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 child2buffer 的缓冲区，初始值为 2 的全 1 矩阵
                self.register_buffer("child2buffer", torch.ones(2, 3) * 2)

            # 前向传播函数，对输入 x 进行处理
            def forward(self, x):
                # 返回 x 减去 self.child2buffer 的结果
                return x - self.child2buffer

        # 定义主模块类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 Child1 实例并赋值给属性 foo
                self.foo = Child1()
                # 创建 Child2 实例并赋值给属性 bar
                self.bar = Child2()
                # 注册一个名为 rootparam 的参数，初始值为 4 的全 1 矩阵
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3) * 4)
                )

            # 前向传播函数，对输入 x 进行处理
            def forward(self, x):
                # x 乘以 self.rootparam
                x = x * self.rootparam
                # 使用 Child1 对象处理 x
                x = self.foo(x)
                # 使用 Child2 对象处理 x
                x = self.bar(x)
                # 返回处理后的 x
                return x

        # 创建原始的 MyModule 实例
        orig_eager = MyModule()

        # 使用 self.check_model 方法检查新创建的 MyModule 实例
        self.check_model(MyModule(), (torch.randn(2, 3, device=self.device),))

    def test_model_modified_weights(self):
        # 定义模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                # 创建权重张量，形状为 (n, k)，存储在指定的设备上
                self.weight = torch.randn(n, k, device=device)
                # 创建偏置张量，形状为 (n)，存储在指定的设备上
                self.bias = torch.randn(n, device=device)

            # 前向传播函数，对输入 a 进行线性变换
            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 16
        N = 10
        K = 128
        batch = 8
        # 创建示例输入 example_inputs，形状为 (2, M, K)，存储在 self.device 上
        example_inputs = (torch.randn(2, M, K, device=self.device),)
        # 创建 Model 实例 model
        model = Model(N, K, self.device)
        # 使用 self.check_model 方法检查 Model 实例
        self.check_model(model, example_inputs)
        # 更新模型权重，此后 AOTInductor 应重新生成 model.so
        # 如果权重存储在 model.so 中
        model.weight += 1
        # 再次使用 self.check_model 方法检查更新后的 Model 实例
        self.check_model(model, example_inputs)

    def test_triton_kernel_extern_kernel_arg(self):
        # 如果设备不是 CUDA，抛出跳过当前测试的异常
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 前向传播函数，对输入 x, y 进行处理
            def forward(self, x, y):
                # 创建全零张量 out，形状与 x 相同
                out = torch.zeros_like(x)
                # 调用外部核函数 add_kernel[(4,)] 处理输入 x, torch.mm(x, y)，并将结果存储到 out 中
                add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
                # 返回处理后的 out
                return out

        # 创建示例输入 example_inputs，形状为 (4, 4)，存储在 CUDA 设备上
        example_inputs = (
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        )

        # 使用 self.check_model 方法检查 Model 实例
        self.check_model(Model(), example_inputs)
    def test_triton_kernel_multi_output_arg(self):
        # 检查当前设备是否为 CUDA，如果不是则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def forward(self, x, y):
                # 创建一个与 x 相同大小的零张量 out
                out = torch.zeros_like(x)
                # 使用自定义的 Triton Kernel 进行加法操作，处理 x 和已排序的 y
                add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
                return out

        # 定义一个示例输入 example_inputs，包含两个随机张量，设备为 CUDA
        example_inputs = (
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        )

        # 调用测试方法 check_model，检查模型 Model 在给定输入下的行为
        self.check_model(Model(), example_inputs)

    @config.patch({"abi_compatible": True})
    def test_triton_kernel_reinterpret_view_mem_leak(self):
        # 检查在使用用户定义的 Triton Kernel 和 AOTI 时是否存在内存泄漏问题
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # 创建一个与 x 相同大小的零张量 out
                out = torch.zeros_like(x)
                # 对 y 进行平方操作
                yy = y * y
                # 使用自定义的 Triton Kernel 进行加法操作，处理 x 和经过 reshape 的 yy
                add_kernel[(4,)](x, yy.reshape_as(x), out, 4, 16)
                return out

        # 定义一个示例输入 example_inputs，包含一个大小为 (4, 4) 和一个大小为 (1, 16) 的随机张量，设备为 CUDA
        example_inputs = (
            torch.randn(4, 4, device="cuda"),
            torch.randn(1, 16, device="cuda"),
        )

        # 使用 AOTIRunnerUtil 编译模型 Model，获取动态链接库路径 so_path
        so_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
        )
        # 使用 AOTIRunnerUtil 加载 CUDA 设备上的动态链接库 so_path
        aot_inductor_module = AOTIRunnerUtil.load("cuda", so_path)

        # 获取当前 CUDA 设备的编号
        device: int = torch.cuda.current_device()
        # 记录 GPU 内存分配情况的起始值
        mem_before = torch.cuda.memory_allocated(device)
        # 调用 AOTI 模块两次，执行模型推理操作
        aot_inductor_module(*example_inputs)
        aot_inductor_module(*example_inputs)
        # 记录 GPU 内存分配情况的结束值
        mem_after = torch.cuda.memory_allocated(device)
        # 断言两次操作后 GPU 内存分配是否一致
        self.assertEqual(mem_before, mem_after)

        # 调用 AOTI 模块，获取其推理结果
        actual = aot_inductor_module(*example_inputs)
        # 创建一个预期的模型实例，并获取其推理结果
        expected = Model()(*example_inputs)
        # 使用 torch.testing.assert_close 断言实际输出与预期输出的接近程度
        torch.testing.assert_close(actual, expected)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotuning", [False, True])
    def test_triton_kernel_unbacked_symint_in_grid(self, dynamic, autotuning):
        # 检查当前设备是否为 CUDA，如果不是则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 前向传播函数
            def forward(self, x, y, n_elements_tensor):
                # 初始化一个与 x 相同大小的全零张量作为输出
                output = torch.zeros_like(x)
                # 获取 n_elements_tensor 的数值部分
                n_elements_symint = n_elements_tensor.item()
                # 获取张量 x 的元素总数
                n_elements = x.numel()

                # 定义内部函数 grid，根据 meta 参数返回一个元组
                def grid(meta):
                    return (triton.cdiv(n_elements_symint, meta["BLOCK_SIZE"]),)

                # 根据 autotuning 的值选择使用自动调优的 add_kernel_autotuned 函数或普通的 add_kernel 函数
                if autotuning:
                    add_kernel_autotuned[grid](
                        x,
                        y,
                        output,
                        n_elements,
                    )
                else:
                    add_kernel[grid](
                        x,
                        y,
                        output,
                        n_elements,
                        BLOCK_SIZE=16,
                    )

                # 返回前向传播的输出张量
                return output

        # 定义一个示例输入元组
        example_inputs = (
            torch.randn(123, device="cuda"),
            torch.randn(123, device="cuda"),
            torch.tensor(123),
        )

        # 如果 dynamic 为 True，则定义动态形状
        dynamic_shapes = None
        if dynamic:
            dim0 = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "x": {0: dim0},
                "y": {0: dim0},
                "n_elements_tensor": {},
            }

        # 调用 self.check_model 方法检查 Model 类的输出
        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfRocm  # USE_MEM_EFF_ATTENTION was not enabled for build.
    def test_scaled_dot_product_efficient_attention(self):
        # 检查当前设备是否为 CUDA，如果不是则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 前向传播函数，调用 torch.ops.aten._scaled_dot_product_efficient_attention 完成计算
            def forward(self, q, k, v, attn_bias):
                return torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, attn_bias, False
                )[0]

        # 定义一个示例输入元组
        example_inputs = (
            torch.randn(4, 4, 36, 36, device="cuda"),
            torch.randn(4, 4, 36, 36, device="cuda"),
            torch.randn(4, 4, 36, 36, device="cuda"),
            torch.randn(4, 4, 36, 36, device="cuda"),
        )
        # 调用 self.check_model 方法检查 Model 类的输出
        self.check_model(Model(), example_inputs)
    def test_index_put_with_none_index(self):
        # 定义一个测试方法，测试在空索引的情况下的 index_put 行为

        # 使用 DeterministicGuard 来确保 index_put 的行为是确定性的
        with DeterministicGuard(True):

            # 定义一个继承自 torch.nn.Module 的模型类
            class Model(torch.nn.Module):
                # 模型的前向传播方法
                def forward(self, x, i1, i2, y):
                    # 调用 torch.ops.aten.index_put 函数，执行索引操作
                    return torch.ops.aten.index_put(
                        x,
                        (None, None, i1, i2.transpose(0, 1)),
                        y,
                        accumulate=True,
                    )

            # 准备示例输入数据
            example_inputs = (
                torch.rand(8, 192, 30, 30, device=self.device),
                torch.zeros(3, 14, 1, 1, dtype=torch.int64, device=self.device),
                torch.ones(14, 3, dtype=torch.int64, device=self.device),
                torch.randn(8, 192, 3, 14, 3, 14, device=self.device),
            )
            # 调用 self.check_model 方法，验证模型在示例输入上的行为
            self.check_model(Model(), example_inputs)
    # 定义一个测试方法，用于运行时检查
    def test_runtime_checks(self):
        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 前向传播方法，返回所有输入的元组
            def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
                return (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        # 准备不同数据类型的输入张量列表
        inputs = []
        for dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            inputs.append(torch.ones(4, 8, 10, dtype=dtype, device=self.device))
        
        # 定义多个维度参数对象
        dim0 = Dim("s0", min=2, max=1024)
        dim1 = Dim("s1", min=2, max=512)
        dim2 = Dim("s2", min=2, max=128)
        
        # 定义动态形状字典，用于描述不同输入张量的维度约束
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
            "x2": {0: dim0},
            "x3": {1: dim1},
            "x4": {1: dim1},
            "x5": {1: dim1},
            "x6": {},
            "x7": {2: dim2},
            "x8": {2: dim2},
            "x9": {2: dim2},
        }
        
        # 创建模型实例
        m = Model()
        
        # 将输入张量列表转换为元组
        inputs = tuple(inputs)
        
        # 使用torch.no_grad()上下文管理器，配置测试运行时的环境
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
                "aot_inductor.debug_compile": True,
            }
        ):
            # 编译模型为AOT（Ahead-Of-Time）可执行对象，并获取生成的共享库路径
            so_path = AOTIRunnerUtil.compile(m, inputs, dynamic_shapes=dynamic_shapes)
        
        # 打开生成的.cpp文件，读取其内容
        with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            
            # 在源代码中进行多次匹配检查，以确保生成的代码符合预期
            FileCheck().check_count(
                "unmatched dtype",
                10,
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched dim value at",
                21,  # 生成了9个动态维度的不同检查
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "dim value is too",
                18,  # 为9个动态维度生成了两次检查
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched stride value at",
                21,  # 为9个符号步幅未生成检查
                exactly=True,
            ).run(src_code)
        
        # 加载优化后的共享库到设备上
        optimized = AOTIRunnerUtil.load(self.device, so_path)
        
        # 执行优化后的模型，获取实际输出
        actual = optimized(*inputs)
        
        # 使用原始模型，获取预期输出
        expected = m(*inputs)
        
        # 使用torch.testing.assert_close()方法比较实际输出和预期输出的近似程度
        torch.testing.assert_close(actual, expected)

    # 根据测试环境跳过某些测试条件的装饰器
    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    # 定义一个测试函数，用于运行 fp8 数据类型的运行时检查
    def test_runtime_checks_fp8(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，接受四个输入参数并将它们转换为 float 类型后相加
            def forward(self, x0, x1, x2, x3):
                t = (
                    x0.to(torch.float)
                    + x1.to(torch.float)
                    + x2.to(torch.float)
                    + x3.to(torch.float)
                )
                return t

        # 创建空列表，用于存储不同数据类型的输入
        inputs = []
        # 循环遍历不同的 torch float8 数据类型，并创建对应的全为 1 的张量
        for dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ):
            inputs.append(torch.ones(8, 8, 8, dtype=dtype, device=self.device))
        
        # 定义维度为 "s0"，范围从 2 到 1024 的维度对象
        dim0 = Dim("s0", min=2, max=1024)
        # 定义动态形状字典，指定各输入张量的动态维度
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
            "x2": {0: dim0},
            "x3": {0: dim0},
        }
        
        # 在无梯度计算环境下，应用配置补丁来调用模型的检查方法
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
                "aot_inductor.debug_compile": True,
            }
        ):
            # 调用检查模型方法，传入模型实例、输入张量元组和动态形状字典
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    # 定义一个测试函数，用于运行复杂数据类型的运行时检查
    def test_runtime_checks_complex(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，直接返回输入的三个复杂张量
            def forward(self, x0, x1, x2):
                return (x0, x1, x2)

        # 创建空列表，用于存储复杂数据类型的输入张量
        inputs = []
        # 创建复杂32位浮点型张量 x0
        x0 = torch.tensor([1, -1], dtype=torch.complex32, device=self.device)
        # 创建复杂64位浮点型张量 x1
        x1 = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
            dtype=torch.complex64,
            device=self.device,
        )
        # 创建复杂128位浮点型张量 x2
        x2 = torch.tensor(128, dtype=torch.complex128, device=self.device)
        # 将创建的三个张量依次添加到输入列表中
        inputs.append(x0)
        inputs.append(x1)
        inputs.append(x2)
        
        # 定义维度为 "s0"，范围从 2 到 1024 的维度对象
        dim0 = Dim("s0", min=2, max=1024)
        # 定义动态形状字典，指定部分输入张量的动态维度
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {},
            "x2": {},
        }
        
        # 在无梯度计算环境下，应用配置补丁来调用模型的检查方法
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
                "aot_inductor.debug_compile": True,
            }
        ):
            # 调用检查模型方法，传入模型实例、输入张量元组和动态形状字典
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    # 如果运行环境是 FBCODE，则跳过当前测试用例
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    # 测试运行时检查 dtype 失败的情况
    def test_runtime_checks_dtype_failed(self):
        # 定义一个简单的 PyTorch 模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 将输入 x 转换为 float 类型
                y = x.type(torch.float)
                return y

        # 创建一个随机张量 x，指定 dtype 为 torch.float16，在指定设备上
        x = torch.randn(1, 4, dtype=torch.float16, device=self.device)
        model = Model()  # 初始化模型
        # 使用 torch.no_grad() 上下文管理器，同时配置一些参数
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
                "aot_inductor.debug_compile": True,
            }
        ):
            # 编译模型，并获取编译后的动态链接库路径
            so_path: str = AOTIRunnerUtil.compile(
                model,
                (x,),
            )
        # 加载编译后的动态链接库模块到指定设备上
        aot_inductor_module = AOTIRunnerUtil.load(self.device, so_path)
        # 将输入张量 x 转换为 float 类型
        x_casted = x.float()
        # 断言在模块中对 x_casted 的调用会抛出异常
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(x_casted)

    # 测试非连续输出别名的情况
    def test_non_contiguous_output_alias(self):
        # 定义一个简单的 PyTorch 模型
        class Model(torch.nn.Module):
            def forward(self, x):
                # 计算输入张量 x 的平方
                squared = x * x
                # 对平方后的张量进行转置，产生非连续的张量
                transposed = squared.t()  # non-contiguous
                # 将非连续的张量转换为连续的张量
                contig = transposed.contiguous()
                return transposed, contig

        # 创建一个随机张量 x，指定 dtype 为 torch.float16，在指定设备上
        x = torch.randn(3, 4, dtype=torch.float16, device=self.device)
        model = Model()  # 初始化模型
        # 使用 torch.no_grad() 上下文管理器，同时配置一些参数
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
            }
        ):
            # 运行模型，获取结果
            result = AOTIRunnerUtil.run(
                self.device,
                model,
                (x,),
            )
        # 计算模型的实际输出
        actual = model(x)
        # 断言运行结果与实际输出相同
        self.assertTrue(same(result, actual))

        # 连续化操作 contig() 应创建一个新的张量
        self.assertTrue(result[0].data_ptr() != result[1].data_ptr())

    # 测试多个输出别名的情况
    def test_multiple_output_alias(self):
        # 测试当多个输出别名同一个张量时的情况
        class Model(torch.nn.Module):
            def forward(self, x):
                # 计算输入张量 x 的平方
                squared = x * x
                # 将平方张量转换为连续张量，产生别名
                contig = squared.contiguous()  # alias
                # 将平方张量重塑为其形状，产生别名
                reshaped = squared.reshape(squared.shape)  # alias
                # 计算平方张量与输入张量 x 的乘积
                cubed = squared * x
                return squared, contig, reshaped, cubed

        # 创建一个随机张量 x，指定 dtype 为 torch.float32，在指定设备上
        x = torch.randn(3, 4, dtype=torch.float32, device=self.device)
        model = Model()  # 初始化模型

        # 使用 torch.no_grad() 上下文管理器，同时配置一些参数
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
            }
        ):
            # 运行模型，获取结果
            result = AOTIRunnerUtil.run(
                self.device,
                model,
                (x,),
            )
        # 计算模型的实际输出
        actual = model(x)
        # 断言运行结果与实际输出相同
        self.assertTrue(same(result, actual))

        # squared、contig 和 reshaped 三者别名同一个张量
        self.assertTrue(result[0].data_ptr() == result[1].data_ptr())
        self.assertTrue(result[0].data_ptr() == result[2].data_ptr())
        # cubed 不应该是别名
        self.assertTrue(result[0].data_ptr() != result[3].data_ptr())
    # 定义一个测试函数，用于检查模型在运行时维度检查失败的情况
    def test_runtime_checks_shape_failed(self):
        # 定义一个简单的 PyTorch 模型，仅将输入直接返回
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        # 创建不同维度的张量作为输入数据
        x = torch.randn(4, 4, 4, dtype=torch.float16, device=self.device)
        y0 = torch.randn(8, 4, 4, dtype=torch.float16, device=self.device)
        y1 = torch.randn(4, 8, 4, dtype=torch.float16, device=self.device)
        y2 = rand_strided(
            (4, 4, 4), (16, 1, 4), dtype=torch.float16, device=self.device
        )
        # batch size 超出允许范围
        y3 = torch.randn(2048, 3, 4, dtype=torch.float16, device=self.device)
        y4 = torch.randn(2048, 4, 4, dtype=torch.float16, device=self.device)
        # 创建维度对象 dim0，并将其放入 dynamic_shapes 字典中
        dim0 = Dim("s0", min=4, max=1024)
        dynamic_shapes = {
            "x": {0: dim0},
        }
        # 实例化模型
        model = Model()
        # 使用 torch.no_grad() 禁用梯度计算，并且通过 config.patch 设置一些配置项
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
                "aot_inductor.debug_compile": True,
            }
        ):
            # 调用 AOTIRunnerUtil.compile 编译模型并获取生成的动态链接库路径
            so_path: str = AOTIRunnerUtil.compile(
                model, (x,), dynamic_shapes=dynamic_shapes
            )
        # 使用 AOTIRunnerUtil.load 加载编译好的模型至设备中
        aot_inductor_module = AOTIRunnerUtil.load(self.device, so_path)
        # 对不同的输入数据进行模型调用并期望引发异常
        # dynamic dim 正常工作
        _ = aot_inductor_module(y0)
        # 期望引发异常，但异常信息为空字符串
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y1)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y2)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y3)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y4)

    # 定义一个测试函数，用于测试复数相加
    def test_add_complex(self):
        # 定义一个简单的 PyTorch 模型，实现复数相加操作
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        # 创建包含复数的张量作为输入数据
        x = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1], device=self.device
        )
        y = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1], device=self.device
        )
        # 使用 self.check_model 检查模型在给定输入下的表现
        self.check_model(Model(), (x, y))

    # 定义一个测试函数，用于测试 embedding_bag 操作
    def test_embedding_bag(self):
        # 定义一个 PyTorch 模型，调用内置的 _embedding_bag 操作
        class Model(torch.nn.Module):
            def forward(self, w, i, o):
                return torch.ops.aten._embedding_bag(w, i, o, False, 0, False, None)

        # 创建示例输入数据 example_inputs
        example_inputs = (
            torch.randn([10, 4], device=self.device),
            torch.randint(10, [8], device=self.device),
            torch.tensor([0, 2, 6], device=self.device),
        )
        # 使用 self.check_model 检查模型在给定输入下的表现
        self.check_model(Model(), example_inputs)

    # 定义一个测试函数，用于测试复数的傅里叶变换
    def test_fft_c2c(self):
        # 定义一个 PyTorch 模型，实现输入的复数的傅里叶变换
        class Model(torch.nn.Module):
            def forward(self, x):
                # 返回输入张量的傅里叶变换及其实部
                return torch.fft.fftn(x), torch.fft.fftn(x).real

        # 创建示例输入数据 example_inputs
        example_inputs = (torch.randn(16, 16, 16, device=self.device),)
        # 使用 self.check_model 检查模型在给定输入下的表现
        self.check_model(Model(), example_inputs)
    def test_nested_tensor_from_jagged(self):
        # 定义一个内部模型类 Model，继承自 nn.Module
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的 MLP 部分，包括线性层、ReLU 激活函数和线性层
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )

            def forward(self, values, offsets):
                # 使用 torch.nested.nested_tensor_from_jagged 方法创建嵌套张量
                nt = torch.nested.nested_tensor_from_jagged(values, offsets)
                # 将嵌套张量输入到 MLP 中进行前向计算
                res = self.mlp(nt)
                # 返回 MLP 输出的值部分
                return res.values()

        # 创建 Model 类的实例，并将其移动到设备上（由 self.device 指定）
        model = Model().to(device=self.device)

        # 定义多个示例输入，每个输入包含随机张量和偏移张量
        example_inputs_1 = (
            torch.randn((15, 128), device=self.device),
            torch.tensor([0, 3, 4, 10, 15], device=self.device),
        )

        # 第二个示例输入，不同的 "NT batch size"，不同的数据量
        example_inputs_2 = (
            torch.randn((31, 128), device=self.device),
            torch.tensor([0, 1, 20, 25, 31], device=self.device),
        )

        # 第三个示例输入，相同的数据量，不同的 "NT batch size"
        example_inputs_3 = (
            torch.randn((15, 128), device=self.device),
            torch.tensor([0, 3, 10, 15], device=self.device),
        )

        # 第四个示例输入，不同的 "NT batch size"
        example_inputs_4 = (
            torch.randn((37, 128), device=self.device),
            torch.tensor([0, 5, 16, 25, 29, 37], device=self.device),
        )

        # 定义维度限制对象 dim0_values 和 dim0_offsets
        dim0_values = Dim("dim0_values", min=1, max=128)
        dim0_offsets = Dim("dim0_offsets", min=1, max=9)
        # 定义动态形状字典，描述 values 和 offsets 的维度
        dynamic_shapes = {"values": {0: dim0_values}, "offsets": {0: dim0_offsets}}
        # 将所有示例输入放入列表中
        example_inputs_list = [
            example_inputs_1,
            example_inputs_2,
            example_inputs_3,
            example_inputs_4,
        ]

        # 使用 self.check_model_with_multiple_inputs 方法检查模型在多个输入下的表现
        self.check_model_with_multiple_inputs(
            model, example_inputs_list, dynamic_shapes=dynamic_shapes
        )

    def test_misc_1(self):
        # 定义一个内部模型类 Model，继承自 nn.Module
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的 MLP 部分，包括线性层、ReLU 激活函数和线性层
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )
                # 定义模型的 EmbeddingBag 层
                self.emb = nn.EmbeddingBag(num_embeddings=128, embedding_dim=32)
                # 定义模型的 over_arch 部分，包括线性层、ReLU 激活函数和线性层
                self.over_arch = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32), nn.Sigmoid()
                )

            def forward(self, x, y):
                # 对输入 x 应用 MLP，并获取其输出
                mlp_output = self.mlp(x)
                # 对输入 y 应用 EmbeddingBag，并获取其输出
                emb_output = self.emb(y)
                # 将 MLP 输出和 EmbeddingBag 输出在维度 1 上进行连接
                return self.over_arch(torch.concat([mlp_output, emb_output], dim=1))

        # 创建一个示例输入，包含两个张量：大小为 (16, 128) 的随机张量和大小为 (16, 10) 的随机整数张量
        example_inputs = (
            torch.randn(16, 128, device=self.device),
            torch.randint(0, 128, (16, 10), device=self.device),
        )
        # 使用 self.check_model 方法检查模型在示例输入下的表现
        self.check_model(Model(), example_inputs)
# 使用 common_utils 模块中的函数实例化带参数的测试模板 AOTInductorTestsTemplate
common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)

# 定义 AOTInductorTestABICompatibleCpu 类，继承自 TestCase 类
class AOTInductorTestABICompatibleCpu(TestCase):
    # 设定设备为 "cpu"
    device = "cpu"
    # 设置 abi_compatible 为 True
    abi_compatible = True
    # 将 check_model 函数赋给 check_model 属性
    check_model = check_model
    # 将 check_model_with_multiple_inputs 函数赋给 check_model_with_multiple_inputs 属性
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    # 将 code_check_count 变量赋给 code_check_count 属性
    code_check_count = code_check_count
    # 禁止堆栈分配，设定 allow_stack_allocation 为 False
    allow_stack_allocation = False
    # 禁用最小的 arrayref 接口，设定 use_minimal_arrayref_interface 为 False
    use_minimal_arrayref_interface = False

# 定义函数 fail_with_and_without_stack_allocation，返回 TestFailure 对象
def fail_with_and_without_stack_allocation(is_skip=False):
    return TestFailure(
        (
            "abi_compatible_cpu",
            "abi_compatible_cpu_with_stack_allocation",
            "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",
        ),
        is_skip=is_skip,
    )

# 定义函数 fail_stack_allocation，返回 TestFailure 对象
def fail_stack_allocation(is_skip=False):
    return TestFailure(
        (
            "abi_compatible_cpu_with_stack_allocation",
            "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",
        ),
        is_skip=is_skip,
    )

# 定义函数 fail_minimal_arrayref_interface，返回 TestFailure 对象
def fail_minimal_arrayref_interface(is_skip=False):
    return TestFailure(
        ("abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",),
        is_skip=is_skip,
    )

# 定义函数 fail_cuda，返回 TestFailure 对象
def fail_cuda(is_skip=False):
    return TestFailure(
        ("abi_compatible_cuda", "non_abi_compatible_cuda"),
        is_skip=is_skip,
    )

# 定义函数 fail_abi_compatible_cuda，返回 TestFailure 对象
def fail_abi_compatible_cuda(is_skip=False):
    return TestFailure(
        ("abi_compatible_cuda",),
        is_skip=is_skip,
    )

# 定义函数 fail_non_abi_compatible_cuda，返回 TestFailure 对象
def fail_non_abi_compatible_cuda(is_skip=False):
    return TestFailure(
        ("non_abi_compatible_cuda",),
        is_skip=is_skip,
    )

# 定义 CPU_TEST_FAILURES 字典，包含多个测试失败的情况
CPU_TEST_FAILURES = {
    # "test_add_complex" 测试失败，跳过该测试
    "test_add_complex": fail_stack_allocation(is_skip=True),
    # "test_conv_freezing_abi_compatible_cpu" 测试失败，跳过该测试
    #   原因：可选输出不支持，抛出 AssertionError: None
    "test_conv_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # "test_deconv_freezing_abi_compatible_cpu" 测试失败，跳过该测试
    #   原因：可选输出不支持，抛出 AssertionError: None
    "test_deconv_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # "test_duplicate_constant_folding" 测试失败，跳过该测试
    #   原因：在退出 Python 运行时时发生 Segfault
    "test_duplicate_constant_folding": fail_with_and_without_stack_allocation(
        is_skip=True
    ),
    # "test_dup_unbacked_sym_decl" 测试失败，跳过该测试
    #   原因：使用最小的 arrayref 接口失败
    "test_dup_unbacked_sym_decl": fail_minimal_arrayref_interface(is_skip=True),
    # "test_dup_unbacked_sym_decl_with_refinement" 测试失败，跳过该测试
    #   原因：使用最小的 arrayref 接口失败
    "test_dup_unbacked_sym_decl_with_refinement": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # "test_dynamic_cat" 测试未标记为失败，默认不跳过
    "test_dynamic_cat": fail_minimal_arrayref_interface(),
    # "test_dynamic_scalar" 测试失败，跳过该测试
    #   原因：堆栈分配问题
    "test_dynamic_scalar": fail_stack_allocation(is_skip=True),
    # "test_fft_c2c" 测试失败，跳过该测试
    #   原因：堆栈分配问题
    "test_fft_c2c": fail_stack_allocation(is_skip=True),
    # "test_freezing_abi_compatible_cpu" 测试失败，跳过该测试
    #   原因：可选输出不支持，抛出 AssertionError: None
    "test_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # 将测试函数 "test_freezing" 标记为跳过，因为它在有和没有堆栈分配时均失败

    "test_linear_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # TODO: 测试 "test_linear_freezing" 失败，ABI 兼容的 CPU，断言错误：None，即不支持可选输出

    "test_missing_cubin": fail_with_and_without_stack_allocation(is_skip=True),
    # FIXME: 在退出 Python 运行时时出现 Segfault 错误

    "test_multi_device": fail_minimal_arrayref_interface(is_skip=True),
    # 最小的 arrayref 接口仅适用于 CPU；测试会崩溃

    "test_normal_functional": fail_with_and_without_stack_allocation(is_skip=True),
    # "test_normal_functional" 失败，有和没有堆栈分配时均失败

    "test_non_contiguous_output_alias": fail_with_and_without_stack_allocation(is_skip=True),
    # 出现未定义符号：_Z16aoti_torch_dtypeIN3c104HalfEEiv

    "test_return_view_constant": fail_minimal_arrayref_interface(is_skip=True),
    # 与 https://github.com/pytorch/pytorch/issues/122978 相同的问题

    "test_reuse_kernel_dynamic": fail_minimal_arrayref_interface(is_skip=True),
    # 测试会导致段错误

    "test_repeat_output": fail_stack_allocation(is_skip=True),
    # 测试会导致段错误

    "test_view_outputs": fail_stack_allocation(is_skip=True),
    # 测试会导致段错误

    "test_multiple_output_alias": fail_with_and_without_stack_allocation(is_skip=True),
    # "test_multiple_output_alias" 失败，有和没有堆栈分配时均失败

    "test_buffer_mutation_1": fail_stack_allocation(is_skip=True),
    # 测试会导致段错误

    "test_buffer_mutation_2": fail_stack_allocation(is_skip=True),
    # 测试会导致段错误

    "test_buffer_mutation_3": fail_stack_allocation(is_skip=True),
    # 测试会导致段错误

    "test_scatter_fallback": fail_stack_allocation(is_skip=True),
    # FIXME: 在退出 Python 运行时时出现 Segfault 错误

    "test_scatter_reduce_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # 与 https://github.com/pytorch/pytorch/issues/122978 相同的问题

    "test_index_put_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # 与 https://github.com/pytorch/pytorch/issues/122978 相同的问题

    "test_index_put_with_none_index": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122984

    "test_constant": fail_stack_allocation(is_skip=True),
    # FIXME: 在退出 Python 运行时时出现 Segfault 错误

    "test_sdpa": fail_with_and_without_stack_allocation(is_skip=True),
    # C++ 编译错误，需要 "aoti_torch___scaled_dot_product_flash_attention_for_cpu"
    # https://github.com/pytorch/pytorch/issues/122986

    "test_sdpa_2": fail_with_and_without_stack_allocation(is_skip=True),
    # 与 https://github.com/pytorch/pytorch/issues/122986 相同的问题

    "test_shifted_constraint_ranges": fail_with_and_without_stack_allocation(is_skip=True),
    # 与 https://github.com/pytorch/pytorch/issues/122978 相同的问题

    "test_amp_fallback_random": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/123691
    # 调用 fail_minimal_arrayref_interface 函数执行单元测试 "test_simple_dynamic"，期望其失败
    "test_simple_dynamic": fail_minimal_arrayref_interface(),

    # 调用 fail_minimal_arrayref_interface 函数执行单元测试 "test_zero_grid_with_unbacked_symbols"，传入 is_skip=True 参数，期望其失败
    # 这个测试在 MacOS 上失败
    "test_zero_grid_with_unbacked_symbols": fail_minimal_arrayref_interface(is_skip=True),

    # 调用 fail_with_and_without_stack_allocation 函数执行单元测试 "test_zero_grid_with_backed_symbols"，传入 is_skip=True 参数，期望其失败
    # 这个测试在 MacOS 上失败
    "test_zero_grid_with_backed_symbols": fail_with_and_without_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_cond_non_tensor_predicates_dynamic_False"，传入 is_skip=True 参数，期望其失败
    # 这个测试与 https://github.com/pytorch/pytorch/issues/122990 类似
    "test_cond_non_tensor_predicates_dynamic_False": fail_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_cond_non_tensor_predicates_dynamic_True"，传入 is_skip=True 参数，期望其失败
    # 这个测试与 https://github.com/pytorch/pytorch/issues/122990 类似
    "test_cond_non_tensor_predicates_dynamic_True": fail_stack_allocation(is_skip=True),

    # 调用 fail_with_and_without_stack_allocation 函数执行单元测试 "test_runtime_checks_complex"，传入 is_skip=True 参数，期望其失败
    # 这个测试与 https://github.com/pytorch/pytorch/issues/122991 类似
    "test_runtime_checks_complex": fail_with_and_without_stack_allocation(is_skip=True),

    # 调用 fail_with_and_without_stack_allocation 函数执行单元测试 "test_runtime_checks_fp8"，传入 is_skip=True 参数，期望其失败
    # 这个测试与 https://github.com/pytorch/pytorch/issues/122991 类似
    "test_runtime_checks_fp8": fail_with_and_without_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_while_loop_simple"，传入 is_skip=True 参数，期望其失败
    "test_while_loop_simple": fail_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_while_loop_nested"，传入 is_skip=True 参数，期望其失败
    "test_while_loop_nested": fail_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_while_loop_with_outer_code"，传入 is_skip=True 参数，期望其失败
    "test_while_loop_with_outer_code": fail_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_while_loop_with_parameters"，传入 is_skip=True 参数，期望其失败
    "test_while_loop_with_parameters": fail_stack_allocation(is_skip=True),

    # 调用 fail_stack_allocation 函数执行单元测试 "test_while_loop_with_outer_buffers"，传入 is_skip=True 参数，期望其失败
    "test_while_loop_with_outer_buffers": fail_stack_allocation(is_skip=True),
}

CUDA_TEST_FAILURES = {
    # 定义 CUDA 测试失败情况的字典
    # test_failures 默认为 xfail，设置 is_skip=True 以跳过测试
    "test_normal_functional": fail_abi_compatible_cuda(is_skip=True),
    # 对于 non_abi_compatible 模式，没有运行时检查
    "test_runtime_checks": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_complex": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_fp8": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_dtype_failed": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_shape_failed": fail_non_abi_compatible_cuda(is_skip=True),
    # GPU 不支持量化
    "test_quantized_linear": fail_cuda(is_skip=True),
}

# 如果不是在 FBCODE 环境下
if not IS_FBCODE:
    # 下列测试在 pytest 和 unittest 中看起来通过（xml 和终端输出显示通过），
    # 但会导致段错误。这种情况仅在 OSS CI 中出现，在内部是可以接受的。
    CPU_TEST_FAILURES.update(
        {
            # 更新 CPU_TEST_FAILURES 字典，添加测试名称和对应的失败信息
            "test_duplicated_params": fail_stack_allocation(is_skip=True),
            "test_embedding_bag": fail_stack_allocation(is_skip=True),
            "test_fqn": fail_stack_allocation(is_skip=True),
            "test_no_args": fail_stack_allocation(is_skip=True),
            "test_output_misaligned": fail_stack_allocation(is_skip=True),
            "test_pytree_inputs": fail_stack_allocation(is_skip=True),
            "test_seq": fail_stack_allocation(is_skip=True),
            "test_simple_split": fail_stack_allocation(is_skip=True),
            "test_addmm": fail_minimal_arrayref_interface(is_skip=True),
            "test_aliased_buffer_reuse": fail_minimal_arrayref_interface(is_skip=True),
            "test_buffer_reuse": fail_minimal_arrayref_interface(is_skip=True),
            "test_constant_folding": fail_minimal_arrayref_interface(is_skip=True),
            "test_convolution": fail_minimal_arrayref_interface(is_skip=True),
            "test_empty_graph": fail_minimal_arrayref_interface(is_skip=True),
            "test_large_weight": fail_minimal_arrayref_interface(is_skip=True),
            "test_large_mmaped_weights": fail_minimal_arrayref_interface(is_skip=True),
            "test_normal_functional": fail_minimal_arrayref_interface(is_skip=True),
            "test_misc_1": fail_minimal_arrayref_interface(is_skip=True),
            "test_missing_output": fail_minimal_arrayref_interface(is_skip=True),
            "test_model_modified_weights": fail_minimal_arrayref_interface(
                is_skip=True
            ),
            "test_output_path_1": fail_minimal_arrayref_interface(is_skip=True),
            "test_quantized_linear": fail_minimal_arrayref_interface(is_skip=True),
            "test_repeat_interleave": fail_minimal_arrayref_interface(is_skip=True),
            "test_return_constant": fail_minimal_arrayref_interface(is_skip=True),
            "test_reuse_kernel": fail_minimal_arrayref_interface(is_skip=True),
            "test_simple": fail_minimal_arrayref_interface(is_skip=True),
            "test_small_constant": fail_minimal_arrayref_interface(is_skip=True),
            "test_with_no_triton_profiler": fail_minimal_arrayref_interface(
                is_skip=True
            ),
            "test_with_offset": fail_minimal_arrayref_interface(is_skip=True),
            "test_with_profiler": fail_minimal_arrayref_interface(is_skip=True),
            "test_zero_size_weight": fail_minimal_arrayref_interface(is_skip=True),
        }
    )
copy_tests(
    AOTInductorTestsTemplate,  # 复制测试用例模板 AOTInductorTestsTemplate
    AOTInductorTestABICompatibleCpu,  # 复制 ABI 兼容的 CPU 测试用例类 AOTInductorTestABICompatibleCpu
    "abi_compatible_cpu",  # 将复制的测试用例命名为 "abi_compatible_cpu"
    CPU_TEST_FAILURES,  # 使用 CPU 测试失败集合 CPU_TEST_FAILURES
)


class AOTInductorTestABICompatibleCpuWithStackAllocation(TestCase):
    device = "cpu"  # 设备类型为 CPU
    abi_compatible = True  # 兼容 ABI
    check_model = check_model  # 使用 check_model 进行模型检查
    check_model_with_multiple_inputs = check_model_with_multiple_inputs  # 使用多输入的模型检查
    code_check_count = code_check_count  # 代码检查计数
    allow_stack_allocation = True  # 允许堆栈分配
    use_minimal_arrayref_interface = False  # 不使用最小化的数组引用接口


copy_tests(
    AOTInductorTestsTemplate,  # 复制测试用例模板 AOTInductorTestsTemplate
    AOTInductorTestABICompatibleCpuWithStackAllocation,  # 复制带有堆栈分配的 ABI 兼容的 CPU 测试用例类 AOTInductorTestABICompatibleCpuWithStackAllocation
    "abi_compatible_cpu_with_stack_allocation",  # 将复制的测试用例命名为 "abi_compatible_cpu_with_stack_allocation"
    CPU_TEST_FAILURES,  # 使用 CPU 测试失败集合 CPU_TEST_FAILURES
)


class AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface(
    TestCase
):
    device = "cpu"  # 设备类型为 CPU
    abi_compatible = True  # 兼容 ABI
    check_model = check_model  # 使用 check_model 进行模型检查
    check_model_with_multiple_inputs = check_model_with_multiple_inputs  # 使用多输入的模型检查
    allow_stack_allocation = True  # 允许堆栈分配
    use_minimal_arrayref_interface = True  # 使用最小化的数组引用接口


copy_tests(
    AOTInductorTestsTemplate,  # 复制测试用例模板 AOTInductorTestsTemplate
    AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface,  # 复制带有堆栈分配和最小数组引用接口的 ABI 兼容的 CPU 测试用例类 AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface
    "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",  # 将复制的测试用例命名为 "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface"
    CPU_TEST_FAILURES,  # 使用 CPU 测试失败集合 CPU_TEST_FAILURES
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleCuda(TestCase):
    device = "cuda"  # 设备类型为 CUDA
    abi_compatible = True  # 兼容 ABI
    check_model = check_model  # 使用 check_model 进行模型检查
    check_model_with_multiple_inputs = check_model_with_multiple_inputs  # 使用多输入的模型检查
    code_check_count = code_check_count  # 代码检查计数
    allow_stack_allocation = False  # 不允许堆栈分配
    use_minimal_arrayref_interface = False  # 不使用最小化的数组引用接口


copy_tests(
    AOTInductorTestsTemplate,  # 复制测试用例模板 AOTInductorTestsTemplate
    AOTInductorTestABICompatibleCuda,  # 复制 ABI 兼容的 CUDA 测试用例类 AOTInductorTestABICompatibleCuda
    "abi_compatible_cuda",  # 将复制的测试用例命名为 "abi_compatible_cuda"
    CUDA_TEST_FAILURES,  # 使用 CUDA 测试失败集合 CUDA_TEST_FAILURES
)


@unittest.skipIf(
    IS_FBCODE or sys.platform == "darwin",
    "NonABI mode should not be used in fbcode nor on MacOS",
)
class AOTInductorTestNonABICompatibleCpu(TestCase):
    device = "cpu"  # 设备类型为 CPU
    abi_compatible = False  # 不兼容 ABI
    check_model = check_model  # 使用 check_model 进行模型检查
    check_model_with_multiple_inputs = check_model_with_multiple_inputs  # 使用多输入的模型检查
    code_check_count = code_check_count  # 代码检查计数
    allow_stack_allocation = False  # 不允许堆栈分配
    use_minimal_arrayref_interface = False  # 不使用最小化的数组引用接口


copy_tests(
    AOTInductorTestsTemplate,  # 复制测试用例模板 AOTInductorTestsTemplate
    AOTInductorTestNonABICompatibleCpu,  # 复制不兼容 ABI 的 CPU 测试用例类 AOTInductorTestNonABICompatibleCpu
    "non_abi_compatible_cpu",  # 将复制的测试用例命名为 "non_abi_compatible_cpu"
    {
        "test_duplicate_constant_folding": TestFailure(
            ("non_abi_compatible_cpu",), is_skip=True
        ),
        # 对于非 ABI 兼容模式，不跑重复常量折叠测试
        "test_runtime_checks": TestFailure(("non_abi_compatible_cpu",), is_skip=True),
        "test_runtime_checks_dtype_failed": TestFailure(
            ("non_abi_compatible_cpu",), is_skip=True
        ),
        "test_runtime_checks_shape_failed": TestFailure(
            ("non_abi_compatible_cpu",), is_skip=True
        ),
    },  # 使用自定义的测试失败集合来指定跳过的测试用例
)


@unittest.skipIf(
    IS_FBCODE or sys.platform == "darwin",
    "NonABI mode should not be used in fbcode nor on MacOS",
)
# 定义一个测试类 AOTInductorTestNonABICompatibleCuda，用于测试非ABI兼容的CUDA功能
class AOTInductorTestNonABICompatibleCuda(TestCase):
    # 设定测试设备为 CUDA
    device = "cuda"
    # 设置为非ABI兼容
    abi_compatible = False
    # 导入的函数和变量用于检查模型
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    # 代码检查计数
    code_check_count = code_check_count
    # 不允许堆栈分配
    allow_stack_allocation = False
    # 不使用最小的数组引用接口
    use_minimal_arrayref_interface = False

# 复制测试用例，从 AOTInductorTestsTemplate 类复制测试到 AOTInductorTestNonABICompatibleCuda 类中
copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCuda,
    "non_abi_compatible_cuda",
    CUDA_TEST_FAILURES,
)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 导入 torch._inductor.test_case 模块中的 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 在 fbcode 环境中，cpp_extension 不适用
    # 如果有 CUDA 或者运行平台是 macOS
    if HAS_CUDA or sys.platform == "darwin":
        # 运行测试，指定需要 filelock
        run_tests(needs="filelock")
```