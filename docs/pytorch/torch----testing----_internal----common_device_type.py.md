# `.\pytorch\torch\testing\_internal\common_device_type.py`

```py
# 忽略类型检查错误的标志，用于禁止 MyPy 对导入的模块进行类型检查
# 进行必要的导入
import copy  # 导入复制模块
import gc  # 导入垃圾回收模块
import inspect  # 导入检查模块
import os  # 导入操作系统接口模块
import runpy  # 导入运行 Python 文件模块
import sys  # 导入系统模块
import threading  # 导入多线程模块
import unittest  # 导入单元测试框架模块
from collections import namedtuple  # 导入命名元组模块
from enum import Enum  # 导入枚举模块
from functools import partial, wraps  # 导入偏函数和装饰器模块
from typing import (  # 导入类型提示模块中的各种类型
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# 导入 PyTorch 模块
import torch
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,  # 导入获取 Torch CUDA 版本的函数
    _get_torch_rocm_version,  # 导入获取 Torch ROCm 版本的函数
    TEST_CUSPARSE_GENERIC,  # 导入测试 CuSparse 通用性的标志
    TEST_HIPSPARSE_GENERIC,  # 导入测试 HipSparse 通用性的标志
)
from torch.testing._internal.common_dtype import get_all_dtypes  # 导入获取所有数据类型的函数
from torch.testing._internal.common_utils import (
    _TestParametrizer,  # 导入测试参数化的类
    clear_tracked_input,  # 导入清除已跟踪输入的函数
    compose_parametrize_fns,  # 导入组合参数化函数的函数
    dtype_name,  # 导入获取数据类型名称的函数
    get_tracked_input,  # 导入获取已跟踪输入的函数
    IS_FBCODE,  # 导入 FBCode 环境标志
    IS_REMOTE_GPU,  # 导入远程 GPU 标志
    IS_SANDCASTLE,  # 导入沙堡环境标志
    IS_WINDOWS,  # 导入 Windows 环境标志
    NATIVE_DEVICES,  # 导入本地设备列表
    PRINT_REPRO_ON_FAILURE,  # 导入测试失败时打印复现信息的标志
    skipCUDANonDefaultStreamIf,  # 导入如果不使用默认 CUDA 流则跳过的装饰器
    skipIfTorchDynamo,  # 导入如果是 Torch Dynamo 则跳过的装饰器
    TEST_HPU,  # 导入测试 HPU 的标志
    TEST_MKL,  # 导入测试 MKL 的标志
    TEST_MPS,  # 导入测试 MPS 的标志
    TEST_WITH_ASAN,  # 导入测试 ASAN 的标志
    TEST_WITH_MIOPEN_SUGGEST_NHWC,  # 导入测试 MIOpen 推荐 NHWC 的标志
    TEST_WITH_ROCM,  # 导入测试 ROCm 的标志
    TEST_WITH_TORCHINDUCTOR,  # 导入测试 Torch Inductor 的标志
    TEST_WITH_TSAN,  # 导入测试 TSAN 的标志
    TEST_WITH_UBSAN,  # 导入测试 UBSAN 的标志
    TEST_XPU,  # 导入测试 XPU 的标志
    TestCase,  # 导入单元测试基类
)

try:
    import psutil  # 尝试导入 psutil 模块，用于系统进程和系统利用率的检查
    HAS_PSUTIL = True  # 设置 psutil 模块可用的标志
except ModuleNotFoundError:
    HAS_PSUTIL = False  # 设置 psutil 模块不可用的标志
    psutil = None  # 如果 psutil 模块不可用，将其设置为 None

# Note [Writing Test Templates]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# PyTorch has its own framework for instantiating test templates. That is, for
#   taking test classes that look similar to unittest or pytest
#   compatible test classes and optionally doing the following:
#
#     - instantiating a version of the test class for each available device type
#         (often the CPU, CUDA, and META device types)
#     - further instantiating a version of each test that's always specialized
#         on the test class's device type, and optionally specialized further
#         on datatypes or operators
#
# This functionality is similar to pytest's parametrize functionality
#   (see https://docs.pytest.org/en/6.2.x/parametrize.html), but with considerable
#   additional logic that specializes the instantiated test classes for their
#   device types (see CPUTestBase and CUDATestBase below), supports a variety
#   of composable decorators that allow for test filtering and setting
#   tolerances, and allows tests parametrized by operators to instantiate
#   only the subset of device type x dtype that operator supports.
#
# This framework was built to make it easier to write tests that run on
#   multiple device types, multiple datatypes (dtypes), and for multiple
#   operators. It's also useful for controlling which tests are run. For example,
#   only tests that use a CUDA device can be run on platforms with CUDA.
#   Let's dive in with an example to get an idea for how it works:
#
# --------------------------------------------------------
# A template class (looks like a regular unittest TestCase)
# class TestClassFoo(TestCase):

# A template test that can be specialized with a device
# NOTE: this test case is not runnable by unittest or pytest because it
#   accepts an extra positional argument, "device", that they do not understand
# def test_bar(self, device):
#   pass

# Function that instantiates a template class and its tests
# instantiate_device_type_tests(TestCommon, globals())
# --------------------------------------------------------

# In the above code example we see a template class and a single test template
#   that can be instantiated with a device. The function
#   instantiate_device_type_tests(), called at file scope, instantiates
#   new test classes, one per available device type, and new tests in those
#   classes from these templates. It actually does this by removing
#   the class TestClassFoo and replacing it with classes like TestClassFooCPU
#   and TestClassFooCUDA, instantiated test classes that inherit from CPUTestBase
#   and CUDATestBase respectively. Additional device types, like XLA,
#   (see https://github.com/pytorch/xla) can further extend the set of
#   instantiated test classes to create classes like TestClassFooXLA.

# The test template, test_bar(), is also instantiated. In this case the template
#   is only specialized on a device, so (depending on the available device
#   types) it might become test_bar_cpu() in TestClassFooCPU and test_bar_cuda()
#   in TestClassFooCUDA. We can think of the instantiated test classes as
#   looking like this:
# --------------------------------------------------------
# # An instantiated test class for the CPU device type
# class TestClassFooCPU(CPUTestBase):

#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cpu(self):
#     test_bar(self, 'cpu')

# # An instantiated test class for the CUDA device type
# class TestClassFooCUDA(CUDATestBase):

#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cuda(self):
#     test_bar(self, 'cuda:0')
# --------------------------------------------------------

# These instantiated test classes ARE discoverable and runnable by both
#   unittest and pytest. One thing that may be confusing, however, is that
#   attempting to run "test_bar" will not work, despite it appearing in the
#   original template code. This is because "test_bar" is no longer discoverable
#   after instantiate_device_type_tests() runs, as the above snippet shows.
#   Instead "test_bar_cpu" and "test_bar_cuda" may be run directly, or both
#   can be run with the option "-k test_bar".

# Removing the template class and adding the instantiated classes requires
#   passing "globals()" to instantiate_device_type_tests(), because it
#   edits the file's Python objects.
# 作为说明，测试可以通过数据类型或运算符进行额外参数化。数据类型的参数化使用 @dtypes 装饰器，并且需要像下面这样的测试模板：

# --------------------------------------------------------
# # 可以根据设备和数据类型（dtype）特化的模板测试
# @dtypes(torch.float32, torch.int64)
# def test_car(self, device, dtype)
#   pass
# --------------------------------------------------------

# 如果CPU和CUDA设备类型都可用，则此测试将被实例化为4个测试，覆盖这两种数据类型和两种设备类型的交叉组合：
#   - test_car_cpu_float32
#   - test_car_cpu_int64
#   - test_car_cuda_float32
#   - test_car_cuda_int64

# 数据类型通过 torch.dtype 对象传递。

# 参数化运算符（实际上是 OpInfos，稍后会详细讨论...）使用 @ops 装饰器，并且需要像下面这样的测试模板：
# --------------------------------------------------------
# # 可以根据设备、数据类型和 OpInfo 特化的模板测试
# @ops(op_db)
# def test_car(self, device, dtype, op)
#   pass
# --------------------------------------------------------

# 有关如何使用 @ops 装饰器的详细信息，请参阅下面的文档，并查看 common_methods_invocations.py 中关于 OpInfos 的注释 [OpInfos]。

# 在整个 "op_db" 上参数化的测试通常会有数百甚至数千个实例。测试将在设备类型、运算符和该设备类型上运算符支持的数据类型的交叉组合上实例化。实例化的测试将具有如下名称：
#   - test_car_add_cpu_float32
#   - test_car_sub_cuda_int64

# 第一个实例化的测试将以 torch.add 的 OpInfo 作为其 "op" 参数调用原始的 test_car() 函数，以字符串 'cpu' 作为其 "device" 参数，以 torch.float32 作为其 "dtype" 参数。第二个实例化的测试将以 torch.sub 的 OpInfo 作为其 "op" 参数调用 test_car() 函数，以类似 'cuda:0' 或 'cuda:1' 的 CUDA 设备字符串作为其 "device" 参数，以 torch.int64 作为其 "dtype" 参数。

# 除了通过 OpInfos 对设备、数据类型和运算符进行参数化外，还支持 @parametrize 装饰器进行任意参数化：
# --------------------------------------------------------
# # 可以根据设备、数据类型和 x 值特化的模板测试
# @parametrize("x", range(5))
# def test_car(self, device, dtype, x)
#   pass
# --------------------------------------------------------

# 有关 @parametrize 的详细信息，请参阅 common_utils.py 中的文档。注意，instantiate_device_type_tests() 函数将处理此类参数化；无需额外调用 instantiate_parametrized_tests()。
#
# Clever test filtering can be very useful when working with parametrized
# tests. "-k test_car" would run every instantiated variant of the test_car()
# test template, and "-k test_car_add" runs every variant instantiated with
# torch.add.
#
# It is important to use the passed device and dtype as appropriate. Use
# helper functions like make_tensor() that require explicitly specifying
# the device and dtype so they're not forgotten.
#
# Test templates can use a variety of composable decorators to specify
# additional options and requirements, some are listed here:
#
#   - @deviceCountAtLeast(<minimum number of devices to run test with>)
#       Passes a list of strings representing all available devices of
#       the test class's device type as the test template's "device" argument.
#       If there are fewer devices than the value passed to the decorator
#       the test is skipped.
#   - @dtypes(<list of tuples of dtypes>)
#       In addition to accepting multiple dtypes, the @dtypes decorator
#       can accept a sequence of tuple pairs of dtypes. The test template
#       will be called with each tuple for its "dtype" argument.
#   - @onlyNativeDeviceTypes
#       Skips the test if the device is not a native device type (currently CPU, CUDA, Meta)
#   - @onlyCPU
#       Skips the test if the device is not a CPU device
#   - @onlyCUDA
#       Skips the test if the device is not a CUDA device
#   - @onlyMPS
#       Skips the test if the device is not a MPS device
#   - @skipCPUIfNoLapack
#       Skips the test if the device is a CPU device and LAPACK is not installed
#   - @skipCPUIfNoMkl
#       Skips the test if the device is a CPU device and MKL is not installed
#   - @skipCUDAIfNoMagma
#       Skips the test if the device is a CUDA device and MAGMA is not installed
#   - @skipCUDAIfRocm
#       Skips the test if the device is a CUDA device and ROCm is being used


# Note [Adding a Device Type]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To add a device type:
#
# (1) Create a new "TestBase" extending DeviceTypeTestBase.
#     See CPUTestBase and CUDATestBase below.
# (2) Define the "device_type" attribute of the base to be the
#     appropriate string.
# (3) Add logic to this file that appends your base class to
#     device_type_test_bases when your device type is available.
# (4) (Optional) Write setUpClass/tearDownClass class methods that
#     instantiate dependencies (see MAGMA in CUDATestBase).
# (5) (Optional) Override the "instantiate_test" method for total
#     control over how your class creates tests.
#
# setUpClass is called AFTER tests have been created and BEFORE and ONLY IF
# they are run. This makes it useful for initializing devices and dependencies.


# Note [Overriding methods in generic tests]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Device generic tests look a lot like normal test classes, but they differ
# 定义一个基类 DeviceTypeTestBase，继承自 TestCase，用于设备类型测试
class DeviceTypeTestBase(TestCase):
    # 设备类型，默认为 "generic_device_type"
    device_type: str = "generic_device_type"

    # 用于标识是否因为不可恢复的错误（如 CUDA 错误）而提前禁用测试套件
    _stop_test_suite = False

    # 精度是线程局部设置，因为可以在每个测试中进行重写
    _tls = threading.local()
    _tls.precision = TestCase._precision  # 使用 TestCase 类的默认精度
    _tls.rel_tol = TestCase._rel_tol      # 使用 TestCase 类的默认相对容差

    @property
    def precision(self):
        return self._tls.precision  # 返回当前线程的精度设置

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec  # 设置当前线程的精度设置为给定值

    @property
    def rel_tol(self):
        return self._tls.rel_tol  # 返回当前线程的相对容差设置

    @rel_tol.setter
    def rel_tol(self, prec):
        self._tls.rel_tol = prec  # 设置当前线程的相对容差设置为给定值

    # 返回一个表示单设备测试应该使用的设备的字符串
    # 注意：单设备测试将专门使用此设备
    @classmethod
    def get_primary_device(cls):
        return cls.device_type  # 返回类变量 device_type 的值

    @classmethod
    # Attempts to retrieve the primary device for the class, handling exceptions if not available.
    def _init_and_get_primary_device(cls):
        try:
            return cls.get_primary_device()
        except Exception:
            # 对于 CUDATestBase、XLATestBase 等，可能主设备在 setUpClass() 设置之前不可用。
            # 如果需要的话，在这里手动调用 setUpClass()。
            if hasattr(cls, "setUpClass"):
                cls.setUpClass()
            return cls.get_primary_device()

    # 返回一个字符串列表，表示此设备类型的所有可用设备。
    # 主设备必须是列表中的第一个字符串，列表不得包含重复项。
    # 注意：不稳定的 API。一旦 PyTorch 有获取所有可用设备的通用机制，将替换此方法。
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    # 返回测试所请求的数据类型。
    # 优先使用特定设备的数据类型规范。
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, "dtypes"):
            return None

        # 获取默认的数据类型列表
        default_dtypes = test.dtypes.get("all")
        msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
        assert default_dtypes is not None, msg

        return test.dtypes.get(cls.device_type, default_dtypes)

    # 根据测试和数据类型获取精度覆盖设置。
    def _get_precision_override(self, test, dtype):
        if not hasattr(test, "precision_overrides"):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    # 根据测试和数据类型获取容差覆盖设置。
    def _get_tolerance_override(self, test, dtype):
        if not hasattr(test, "tolerance_overrides"):
            return self.precision, self.rel_tol
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    # 为测试应用精度覆盖设置。
    def _apply_precision_override_for_test(self, test, param_kwargs):
        # 获取参数中的数据类型
        dtype = param_kwargs["dtype"] if "dtype" in param_kwargs else None
        dtype = param_kwargs["dtypes"] if "dtypes" in param_kwargs else dtype
        if dtype:
            # 设置当前精度和相对容差
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    # 创建特定设备的测试。
    @classmethod
    def run(self, result=None):
        # 调用父类的运行方法
        super().run(result=result)
        # 如果 _stop_test_suite 被设置，则提前终止测试套件。
        if self._stop_test_suite:
            result.stop()
# 继承自设备类型测试基类，用于CPU测试
class CPUTestBase(DeviceTypeTestBase):
    # 设备类型为CPU
    device_type = "cpu"

    # 判断是否应该停止测试套件，CPU测试中不会因为关键错误停止
    def _should_stop_test_suite(self):
        return False


# 继承自设备类型测试基类，用于CUDA测试
class CUDATestBase(DeviceTypeTestBase):
    # 设备类型为CUDA
    device_type = "cuda"
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    # 判断是否有cuDNN库可用
    def has_cudnn(self):
        return not self.no_cudnn

    # 获取主要的CUDA设备
    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    # 获取所有CUDA设备
    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = "cuda:{0}"
        # 生成除主要设备以外的所有CUDA设备字符串列表
        non_primary_devices = [
            cuda_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    # 在类初始化时设置环境，检查是否支持cuDNN和版本
    @classmethod
    def setUpClass(cls):
        # 在初始化CUDA张量后，检查是否有magma库
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        # 检测cuDNN是否可用以及其版本
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # 获取当前设备作为主要（测试）设备
        cls.primary_device = f"cuda:{torch.cuda.current_device()}"


# 懒加载张量测试的设备类型基类
lazy_ts_backend_init = False


class LazyTestBase(DeviceTypeTestBase):
    # 设备类型为lazy
    device_type = "lazy"

    # 判断是否应该停止测试套件，lazy测试中不会因为关键错误停止
    def _should_stop_test_suite(self):
        return False

    # 在类初始化时设置环境，初始化lazy相关后端
    @classmethod
    def setUpClass(cls):
        import torch._lazy
        import torch._lazy.metrics
        import torch._lazy.ts_backend

        global lazy_ts_backend_init
        if not lazy_ts_backend_init:
            # 在运行测试之前，需要将TS后端连接到lazy key
            torch._lazy.ts_backend.init()
            lazy_ts_backend_init = True


# 继承自设备类型测试基类，用于MPS测试
class MPSTestBase(DeviceTypeTestBase):
    # 设备类型为MPS
    device_type = "mps"
    primary_device: ClassVar[str]

    # 获取主要的MPS设备
    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    # 获取所有MPS设备，当前只支持一个设备
    @classmethod
    def get_all_devices(cls):
        prim_device = cls.get_primary_device()
        return [prim_device]

    # 在类初始化时设置环境
    @classmethod
    def setUpClass(cls):
        cls.primary_device = "mps:0"

    # 判断是否应该停止测试套件，MPS测试中不会因为关键错误停止
    def _should_stop_test_suite(self):
        return False


# 继承自设备类型测试基类，用于XPU测试
class XPUTestBase(DeviceTypeTestBase):
    # 设备类型为XPU
    device_type = "xpu"
    primary_device: ClassVar[str]

    # 获取主要的XPU设备
    @classmethod
    def get_primary_device(cls):
        return cls.primary_device
    # 获取所有设备的方法，这里的类方法(cls)只返回主设备
    def get_all_devices(cls):
        # 获取主设备的方法调用
        prim_device = cls.get_primary_device()
        # 返回包含主设备的列表
        return [prim_device]

    # 类方法setUpClass，设置类的主设备为"xpu:0"
    @classmethod
    def setUpClass(cls):
        cls.primary_device = "xpu:0"

    # 私有方法_should_stop_test_suite，始终返回False，用于判断测试套件是否应该停止
    def _should_stop_test_suite(self):
        return False
class HPUTestBase(DeviceTypeTestBase):
    # 设备类型为 "hpu"
    device_type = "hpu"
    # 主设备类变量
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        # 返回主设备
        return cls.primary_device

    @classmethod
    def setUpClass(cls):
        # 设置主设备为 "hpu:0"
        cls.primary_device = "hpu:0"


class PrivateUse1TestBase(DeviceTypeTestBase):
    # 主设备类变量
    primary_device: ClassVar[str]
    # 设备模块为空
    device_mod = None
    # 设备类型为 "privateuse1"
    device_type = "privateuse1"

    @classmethod
    def get_primary_device(cls):
        # 返回主设备
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # 获取主设备索引
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        # 获取设备数量
        num_devices = cls.device_mod.device_count()
        # 获取主设备
        prim_device = cls.get_primary_device()
        # 设备字符串格式
        device_str = f"{cls.device_type}:{{0}}"
        # 非主设备列表
        non_primary_devices = [
            device_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        # 返回所有设备列表，包括主设备和非主设备
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # 获取私有设备类型名称
        cls.device_type = torch._C._get_privateuse1_backend_name()
        # 获取设备模块
        cls.device_mod = getattr(torch, cls.device_type, None)
        # 断言设备模块不为空
        assert (
            cls.device_mod is not None
        ), f"""torch has no module of `{cls.device_type}`, you should register
                                            a module by `torch._register_device_module`."""
        # 设置主设备为 "privateuse1:<当前设备>"
        cls.primary_device = f"{cls.device_type}:{cls.device_mod.current_device()}"

# 添加可用的设备类型特定测试基类
def get_device_type_test_bases():
    # 由于 mypy 列表-联合问题，将类型设置为 List[Any]
    test_bases: List[Any] = list()

    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            # 如果未启用 sanitizer，则添加 CUDA 测试基类
            if not TEST_WITH_ASAN and not TEST_WITH_TSAN and not TEST_WITH_UBSAN:
                test_bases.append(CUDATestBase)
        else:
            # 添加 CPU 测试基类
            test_bases.append(CPUTestBase)
    else:
        # 添加 CPU 测试基类
        test_bases.append(CPUTestBase)
        # 如果 CUDA 可用，则添加 CUDA 测试基类
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)

        # 获取私有设备类型名称
        device_type = torch._C._get_privateuse1_backend_name()
        # 获取设备模块
        device_mod = getattr(torch, device_type, None)
        # 如果设备模块具有可用属性且可用，则添加 PrivateUse1TestBase 测试基类
        if hasattr(device_mod, "is_available") and device_mod.is_available():
            test_bases.append(PrivateUse1TestBase)
        # 暂时在通用设备测试中禁用 MPS 测试，因为我们正在增加支持
        # elif torch.backends.mps.is_available():
        #   test_bases.append(MPSTestBase)

    # 返回设备类型测试基类列表
    return test_bases


# 获取设备类型测试基类列表
device_type_test_bases = get_device_type_test_bases()


def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):
    # 设备类型不能同时出现在 except_for 和 only_for 中
    intersect = set(except_for if except_for else []) & set(
        only_for if only_for else []
    )
    # 断言不存在交集
    assert (
        not intersect
    ), "Device type cannot appear in both 'except_for' and 'only_for'"
    ), f"device ({intersect}) appeared in both except_for and only_for"

这行代码是一个字符串格式化表达式，用于生成一个错误消息，如果交集 `intersect` 出现在 `except_for` 和 `only_for` 中，则抛出异常。


    if except_for:

如果 `except_for` 非空（即有元素），执行下面的代码块。


        device_type_test_bases = filter(
            lambda x: x.device_type not in except_for, device_type_test_bases
        )

使用 `filter` 函数，过滤掉 `device_type_test_bases` 中 `device_type` 属性在 `except_for` 中出现的元素。


    if only_for:

如果 `only_for` 非空（即有元素），执行下面的代码块。


        device_type_test_bases = filter(
            lambda x: x.device_type in only_for, device_type_test_bases
        )

使用 `filter` 函数，过滤掉 `device_type_test_bases` 中 `device_type` 属性不在 `only_for` 中出现的元素。


    return list(device_type_test_bases)

将 `device_type_test_bases` 转换为列表并返回。这是最终返回的结果，经过过滤后的 `device_type_test_bases` 列表。
# 获取环境变量 TORCH_TEST_DEVICES 的值，如果不存在则为 None
_TORCH_TEST_DEVICES = os.environ.get("TORCH_TEST_DEVICES", None)

# 如果环境变量 _TORCH_TEST_DEVICES 存在
if _TORCH_TEST_DEVICES:
    # 使用 ':' 分割路径列表
    for path in _TORCH_TEST_DEVICES.split(":"):
        # 使用 runpy.run_path 运行指定路径的 Python 文件，并将其模块导入当前全局命名空间
        mod = runpy.run_path(path, init_globals=globals())  # type: ignore[func-returns-value]
        # 将模块中的 TEST_CLASS 变量加入 device_type_test_bases 列表中
        device_type_test_bases.append(mod["TEST_CLASS"])

# 检查是否启用了 PYTORCH_CUDA_MEMCHECK，将其转换为布尔值
PYTORCH_CUDA_MEMCHECK = os.getenv("PYTORCH_CUDA_MEMCHECK", "0") == "1"

# 定义用于设备测试的环境变量键名
PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = "PYTORCH_TESTING_DEVICE_ONLY_FOR"
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = "PYTORCH_TESTING_DEVICE_EXCEPT_FOR"
PYTORCH_TESTING_DEVICE_FOR_CUSTOM_KEY = "PYTORCH_TESTING_DEVICE_FOR_CUSTOM"


def get_desired_device_type_test_bases(
    except_for=None, only_for=None, include_lazy=False, allow_mps=False, allow_xpu=False
):
    # 复制 device_type_test_bases 列表，以便修改副本而不影响原始列表
    test_bases = device_type_test_bases.copy()

    # 如果允许测试 MPS 并且 TEST_MPS 存在且 MPSTestBase 不在 test_bases 中，则将其加入列表
    if allow_mps and TEST_MPS and MPSTestBase not in test_bases:
        test_bases.append(MPSTestBase)

    # 如果允许测试 XPU 并且 TEST_XPU 存在且 XPUTestBase 不在 test_bases 中，则将其加入列表
    if allow_xpu and TEST_XPU and XPUTestBase not in test_bases:
        test_bases.append(XPUTestBase)

    # 如果 TEST_HPU 存在且 HPUTestBase 不在 test_bases 中，则将其加入列表
    if TEST_HPU and HPUTestBase not in test_bases:
        test_bases.append(HPUTestBase)

    # 根据用户输入过滤设备类型测试基类
    desired_device_type_test_bases = filter_desired_device_types(
        test_bases, except_for, only_for
    )
    # 如果 include_lazy 为真，则进行以下操作
    if include_lazy:
        # 注意 [在设备无关测试中使用延迟张量]
        # 目前，test_view_ops.py 使用 LazyTensor 运行。
        # 我们不希望所有设备无关的测试都使用延迟设备，
        # 因为其中许多测试将会失败。
        # 因此，唯一一种方式将特定的设备无关测试文件选择为使用延迟张量测试是通过 include_lazy=True。
        if IS_FBCODE:
            print(
                "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds",
                file=sys.stderr,
            )
        else:
            # 将 LazyTestBase 添加到 desired_device_type_test_bases 列表中
            desired_device_type_test_bases.append(LazyTestBase)

    # 定义一个函数 split_if_not_empty，用于根据逗号分隔字符串 x，并返回列表
    def split_if_not_empty(x: str):
        return x.split(",") if x else []

    # 如果存在自定义环境变量 PYTORCH_TESTING_DEVICE_FOR_CUSTOM，则根据其值获取列表
    env_custom_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_FOR_CUSTOM_KEY, "")
    )
    if env_custom_only_for:
        # 将 test_bases 中设备类型在 env_custom_only_for 中的测试添加到 desired_device_type_test_bases 列表中
        desired_device_type_test_bases += filter(
            lambda x: x.device_type in env_custom_only_for, test_bases
        )
        # 去重 desired_device_type_test_bases 列表中的元素
        desired_device_type_test_bases = list(set(desired_device_type_test_bases))

    # 根据环境变量 PYTORCH_TESTING_DEVICE_ONLY_FOR 过滤设备类型
    env_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, "")
    )
    # 根据环境变量 PYTORCH_TESTING_DEVICE_EXCEPT_FOR 过滤掉设备类型
    env_except_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, "")
    )

    # 返回根据环境变量过滤后的设备类型测试列表
    return filter_desired_device_types(
        desired_device_type_test_bases, env_except_for, env_only_for
    )
# 添加“实例化”的特定设备测试用例到给定的作用域中。
# 这些测试用例是从通用测试类 generic_test_class 中派生出来的。
# 如果测试类包含设备特定的测试，应该使用这个函数，而不是使用 instantiate_parametrized_tests()。
# （注意：这支持额外的 @parametrize 使用）。

# 参见“编写测试模板”部分的说明。
# TODO: 在 Interl GPU 支持此函数实例化的所有测试用例之后，移除 "allow_xpu" 选项。

def instantiate_device_type_tests(
    generic_test_class,
    scope,
    except_for=None,
    only_for=None,
    include_lazy=False,
    allow_mps=False,
    allow_xpu=False,
):
    # 从其包含范围中移除通用测试类，以便其测试不会被发现。
    del scope[generic_test_class.__name__]

    # 创建通用测试类的一个“空”版本
    # 注意：我们不直接从 generic_test_class 继承，因为这样会将其测试添加到我们的测试类中，
    #   虽然这些测试不能运行。继承的方法也不能后续删除，而且我们不能依赖于 load_tests，
    #   因为 pytest（截至目前）不支持它。
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})

    # 获取通用测试类的成员名称
    # 参见“覆盖通用测试中的方法”部分的说明
    generic_members = set(generic_test_class.__dict__.keys()) - set(
        empty_class.__dict__.keys()
    )
    generic_tests = [x for x in generic_members if x.startswith("test")]

    # 创建特定设备类型的测试用例
    for base in get_desired_device_type_test_bases(
        except_for, only_for, include_lazy, allow_mps, allow_xpu
    ):
        # 待续，未完整注释
        ):
            # 构造测试类的名称，由通用测试类名和设备类型大写组成
            class_name = generic_test_class.__name__ + base.device_type.upper()

            # 定义一个类型为 Any 的设备类型测试类，由通用测试类、空类组成
            # 由于运行时类不支持，此处类型设为 Any 并进行抑制:
            # https://github.com/python/mypy/wiki/Unsupported-Python-Features
            device_type_test_class: Any = type(class_name, (base, empty_class), {})

            # 遍历通用成员列表
            for name in generic_members:
                if name in generic_tests:  # 实例化测试成员
                    test = getattr(generic_test_class, name)

                    # 获取设备类型测试类中 instantiate_test 方法的签名
                    sig = inspect.signature(device_type_test_class.instantiate_test)
                    if len(sig.parameters) == 3:
                        # 实例化设备特定测试
                        device_type_test_class.instantiate_test(
                            name, copy.deepcopy(test), generic_cls=generic_test_class
                        )
                    else:
                        device_type_test_class.instantiate_test(name, copy.deepcopy(test))
                else:  # 处理非测试成员
                    assert (
                        name not in device_type_test_class.__dict__
                    ), f"Redefinition of directly defined member {name}"
                    # 获取通用测试类中的非测试成员
                    nontest = getattr(generic_test_class, name)
                    # 将非测试成员添加到设备类型测试类中
                    setattr(device_type_test_class, name, nontest)

            # 设备类型测试类从测试模板类和空类继承
            # 设置 setUpClass 和 tearDownClass 方法
            @classmethod
            def _setUpClass(cls):
                base.setUpClass()
                empty_class.setUpClass()

            @classmethod
            def _tearDownClass(cls):
                empty_class.tearDownClass()
                base.tearDownClass()

            device_type_test_class.setUpClass = _setUpClass
            device_type_test_class.tearDownClass = _tearDownClass

            # 模拟在调用方文件中定义实例化类
            # 将其模块设置为通用测试类的模块，并将其添加到给定的作用域中
            # 这使得 unittest 可以发现实例化的类
            device_type_test_class.__module__ = generic_test_class.__module__
            scope[class_name] = device_type_test_class
# 定义 OpDTypes 枚举类，用于指定操作的数据类型测试类别
class OpDTypes(Enum):
    supported = 0  # 测试所有支持的数据类型（默认）
    unsupported = 1  # 仅测试不支持的数据类型
    supported_backward = 2  # 测试所有支持的反向传播数据类型
    unsupported_backward = 3  # 仅测试不支持的反向传播数据类型
    any_one = 4  # 测试一个操作支持的数据类型
    none = 5  # 对于不依赖于数据类型的测试，不需要传递数据类型参数
    any_common_cpu_cuda_one = (
        6  # 测试一个同时支持在 CUDA 和 CPU 上常见的数据类型
    )


# 定义了一个任意顺序的数据类型元组
ANY_DTYPE_ORDER = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.long,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
)


def _serialize_sample(sample_input):
    # 如果 sample_input 具有 summary 方法，则调用 summary 方法打印其摘要信息
    if getattr(sample_input, "summary", None) is not None:
        return sample_input.summary()
    # 否则将 sample_input 转换为字符串并返回
    return str(sample_input)


# ops 装饰器定义了测试模板应该实例化的 OpInfos
#
# 例如用法:
#
# @ops(unary_ufuncs)
# def test_numerics(self, device, dtype, op):
#   <test_code>
#
# 这将为每个给定的 OpInfo，在操作符支持的每个设备上，以及操作符支持的每种数据类型上，
# 实例化 test_numerics 的变体。关于数据类型规则，下面有几个注意事项。
#
# ops 装饰器可以接受两个额外的参数，"dtypes" 和 "allowed_dtypes"。
# 如果指定了 "dtypes"，则不论操作符支持哪些数据类型，测试变体都会为这些数据类型实例化。
# 如果指定了 "allowed_dtypes"，则测试变体仅为允许的数据类型列表与原本应实例化的数据类型列表的交集实例化。
# 即，allowed_dtypes 与上面和下面列出的选项相结合。
#
# "dtypes" 参数还可以接受其他值（参见上面的 OpDTypes 枚举类）:
#   OpDTypes.supported - 实例化操作符支持的所有数据类型的测试
#   OpDTypes.unsupported - 实例化操作符不支持的所有数据类型的测试
# Defines various options for instantiating tests based on supported and unsupported dtypes.
# OpDTypes.supported_backward - test for dtypes supported by the operator's gradient formula
# OpDTypes.unsupported_backward - test for dtypes not supported by the operator's gradient formula
# OpDTypes.any_one - test for any one supported dtype, allowing forward and backward if possible
# OpDTypes.none - test instantiated without any specific dtype, hence no dtype kwarg in the test signature.
#
# These options provide flexibility in controlling the dtypes for which tests are instantiated.

class ops(_TestParametrizer):
    def __init__(
        self,
        op_list,
        *,
        dtypes: Union[OpDTypes, Sequence[torch.dtype]] = OpDTypes.supported,
        allowed_dtypes: Optional[Sequence[torch.dtype]] = None,
        skip_if_dynamo=True,
    ):
        # Initializes with a list of operators and dtype options for the tests.
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = (
            set(allowed_dtypes) if allowed_dtypes is not None else None
        )
        self.skip_if_dynamo = skip_if_dynamo

# Decorator that skips a test based on a given condition.
# Notes:
#   (1) Skip conditions can stack.
#   (2) Conditions can be bools or strings. If a string, the corresponding attribute
#       in the test base class must be defined and set to False for the test to run.
#       If a string argument is used, consider defining a dedicated decorator instead.
#   (3) Prefer existing decorators over defining the 'device_type' kwarg directly.

class skipIf:
    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):
        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            # Skips the test if the device type matches or if the skip condition is met.
            if self.device_type is None or self.device_type == slf.device_type:
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (
                    isinstance(self.dep, bool) and self.dep
                ):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, *args, **kwargs)

        return dep_fn

# Skips a test on CPU if the condition is true.
class skipCPUIf(skipIf):
    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type="cpu")

# Skips a test on CUDA if the condition is true.
class skipCUDAIf(skipIf):
    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type="cuda")

# Skips a test on Lazy if the condition is true.
class skipLazyIf(skipIf):
    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type="lazy")

# Skips a test on Meta if the condition is true.
class skipMetaIf(skipIf):
    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type="meta")

# Skips a test on MPS if the condition is true.
class skipMPSIf(skipIf):
    # 定义构造函数，初始化对象
    def __init__(self, dep, reason):
        # 调用父类（超类）的构造函数，传入参数 dep, reason 和 device_type="mps"
        super().__init__(dep, reason, device_type="mps")
class skipHPUIf(skipIf):
    def __init__(self, dep, reason):
        # 调用父类构造函数，设置依赖项和跳过的原因，并指定设备类型为"hpu"
        super().__init__(dep, reason, device_type="hpu")


class skipXLAIf(skipIf):
    def __init__(self, dep, reason):
        # 调用父类构造函数，设置依赖项和跳过的原因，并指定设备类型为"xla"
        super().__init__(dep, reason, device_type="xla")


class skipPRIVATEUSE1If(skipIf):
    def __init__(self, dep, reason):
        # 获取私有使用1后端的名称，调用父类构造函数，设置依赖项和跳过的原因，并指定设备类型为私有后端名称
        device_type = torch._C._get_privateuse1_backend_name()
        super().__init__(dep, reason, device_type=device_type)


def _has_sufficient_memory(device, size):
    # 如果设备是cuda类型
    if torch.device(device).type == "cuda":
        # 如果CUDA可用性为假，则返回假
        if not torch.cuda.is_available():
            return False
        # 执行垃圾收集和清空CUDA缓存
        gc.collect()
        torch.cuda.empty_cache()
        # 获取GPU的内存信息，返回一个元组（空闲内存，总内存）
        if device == "cuda":
            device = "cuda:0"
        return torch.cuda.memory.mem_get_info(device)[0] >= size

    # 如果设备是xla类型，抛出unittest.SkipTest异常，指示需要为XLA进行内存可用性检查
    if device == "xla":
        raise unittest.SkipTest("TODO: Memory availability checks for XLA?")

    # 如果设备是xpu类型，抛出unittest.SkipTest异常，指示需要为Intel GPU进行内存可用性检查
    if device == "xpu":
        raise unittest.SkipTest("TODO: Memory availability checks for Intel GPU?")

    # 如果设备不是cpu类型，则抛出unittest.SkipTest异常，指示设备类型未知
    if device != "cpu":
        raise unittest.SkipTest("Unknown device type")

    # 如果没有psutil，则抛出unittest.SkipTest异常，指示需要psutil来确定内存是否足够
    if not HAS_PSUTIL:
        raise unittest.SkipTest("Need psutil to determine if memory is sufficient")

    # 如果使用ASAN、TSAN或UBSAN，则有效大小增加到size的10倍
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size

    # 如果可用的虚拟内存小于有效大小，进行垃圾收集
    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size


def largeTensorTest(size, device=None):
    """Skip test if the device has insufficient memory to run the test

    size may be a number of bytes, a string of the form "N GB", or a callable

    If the test is a device generic test, available memory on the primary device will be checked.
    It can also be overriden by the optional `device=` argument.
    In other tests, the `device=` argument needs to be specified.
    """
    # 如果size是字符串类型且以"GB"结尾，则将其转换为字节数
    if isinstance(size, str):
        assert size.endswith(("GB", "gb")), "only bytes or GB supported"
        size = 1024**3 * int(size[:-2])

    def inner(fn):
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            # 如果size是可调用对象，则调用它以获取实际的大小值
            size_bytes = size(self, *args, **kwargs) if callable(size) else size
            # 如果未指定设备，则获取主设备
            _device = device if device is not None else self.get_primary_device()
            # 如果设备内存不足，抛出unittest.SkipTest异常，指示内存不足
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest(f"Insufficient {_device} memory")

            return fn(self, *args, **kwargs)

        return dep_fn

    return inner


class expectedFailure:
    def __init__(self, device_type):
        # 初始化预期失败的设备类型
        self.device_type = device_type
    def __call__(self, fn):
        # 定义一个装饰器函数，接受一个函数 `fn` 作为参数
        @wraps(fn)
        # 使用 functools.wraps 装饰器保留原始函数的元数据
        def efail_fn(slf, *args, **kwargs):
            # 定义内部函数 `efail_fn`，用于处理函数调用
            # 检查是否没有 `device_type` 属性，
            # 但有 `device` 属性且 `device` 是字符串类型
            if (
                not hasattr(slf, "device_type")
                and hasattr(slf, "device")
                and isinstance(slf.device, str)
            ):
                # 如果满足条件，则将 `slf.device` 赋值给 `target_device_type`
                target_device_type = slf.device
            else:
                # 否则将 `slf.device_type` 赋值给 `target_device_type`
                target_device_type = slf.device_type

            # 检查 `self.device_type` 是否为 None 或与 `target_device_type` 相同
            if self.device_type is None or self.device_type == target_device_type:
                try:
                    # 尝试执行传入的函数 `fn`
                    fn(slf, *args, **kwargs)
                except Exception:
                    # 如果抛出异常则捕获，不做任何处理
                    return
                else:
                    # 否则调用 `slf.fail` 方法，指示预期测试失败但实际通过了
                    slf.fail("expected test to fail, but it passed")

            # 返回函数 `fn` 的执行结果
            return fn(slf, *args, **kwargs)

        # 返回经装饰后的内部函数 `efail_fn`
        return efail_fn
# 定义一个装饰器类 onlyOn，用于检查测试函数是否只能在特定设备类型上运行
class onlyOn:
    def __init__(self, device_type):
        self.device_type = device_type

    # 装饰器的实现，将被装饰的函数进行包装
    def __call__(self, fn):
        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            # 检查当前对象的设备类型是否与指定的设备类型一致，否则跳过测试
            if self.device_type != slf.device_type:
                reason = f"Only runs on {self.device_type}"
                raise unittest.SkipTest(reason)

            return fn(slf, *args, **kwargs)

        return only_fn


# Decorator that provides all available devices of the device type to the test
# as a list of strings instead of providing a single device string.
# Skips the test if the number of available devices of the variant's device
# type is less than the 'num_required_devices' arg.
# 定义一个装饰器类 deviceCountAtLeast，用于检查测试函数是否至少需要指定数量的设备
class deviceCountAtLeast:
    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    # 装饰器的实现，将被装饰的函数进行包装
    def __call__(self, fn):
        # 确保被装饰的函数没有重定义 num_required_devices 属性
        assert not hasattr(
            fn, "num_required_devices"
        ), f"deviceCountAtLeast redefinition for {fn.__name__}"
        # 设置被装饰函数的 num_required_devices 属性
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            # 如果提供的设备数量少于要求的数量，则跳过测试
            if len(devices) < self.num_required_devices:
                reason = f"fewer than {self.num_required_devices} devices detected"
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn


# Only runs the test on the native device type (currently CPU, CUDA, Meta and PRIVATEUSE1)
# 定义一个装饰器函数 onlyNativeDeviceTypes，用于检查测试函数是否只能在本机设备类型上运行
def onlyNativeDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        # 如果当前设备类型不在预定义的本机设备类型列表 NATIVE_DEVICES 中，则跳过测试
        if self.device_type not in NATIVE_DEVICES:
            reason = f"onlyNativeDeviceTypes: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn


# Specifies per-dtype precision overrides.
# Ex.
#
# @precisionOverride({torch.half : 1e-2, torch.float : 1e-4})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's precision will be set to the
# corresponding override, if it exists.
# self.precision can be accessed directly, and it also controls the behavior of
# functions like self.assertEqual().
#
# Note that self.precision is a scalar value, so if you require multiple
# precisions (or are working with multiple dtypes) they should be specified
# explicitly and computed using self.precision (e.g.
# self.precision *2, max(1, self.precision)).
# 定义一个装饰器类 precisionOverride，用于指定每个数据类型的精度覆盖值
class precisionOverride:
    def __init__(self, d):
        # 确保传入的参数 d 是一个字典，且其中的键是有效的 torch 数据类型
        assert isinstance(
            d, dict
        ), "precisionOverride not given a dtype : precision dict!"
        for dtype in d.keys():
            assert isinstance(
                dtype, torch.dtype
            ), f"precisionOverride given unknown dtype {dtype}"

        self.d = d

    # 装饰器的实现，将被装饰的函数进行包装
    def __call__(self, fn):
        # 将精度覆盖字典 d 设置为被装饰函数的 precision_overrides 属性
        fn.precision_overrides = self.d
        return fn
# precisionOverride 命名元组，用于存储绝对容差和相对容差
tol = namedtuple("tol", ["atol", "rtol"])

# toleranceOverride 类，用于设置特定数据类型的容差覆盖值
class toleranceOverride:
    def __init__(self, d):
        # 确保参数 d 是字典类型，包含 torch.dtype 到 tol 对象的映射
        assert isinstance(d, dict), "toleranceOverride not given a dtype : tol dict!"
        # 检查每个键和值的类型是否正确
        for dtype, prec in d.items():
            assert isinstance(
                dtype, torch.dtype
            ), f"toleranceOverride given unknown dtype {dtype}"
            assert isinstance(
                prec, tol
            ), "toleranceOverride not given a dtype : tol dict!"

        self.d = d

    # 将 toleranceOverride 实例作为装饰器使用
    def __call__(self, fn):
        # 将容差覆盖值设置为被装饰函数的属性 tolerance_overrides
        fn.tolerance_overrides = self.d
        return fn


# dtypes 类，用于指定测试函数接受的数据类型
class dtypes:
    def __init__(self, *args, device_type="all"):
        # 如果参数的第一个元素是列表或元组，则验证每个元素都是列表或元组，并且每个元素都是已知的 torch.dtype
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), (
                    "When one dtype variant is a tuple or list, "
                    "all dtype variants must be. "
                    f"Received non-list non-tuple dtype {str(arg)}"
                )
                assert all(
                    isinstance(dtype, torch.dtype) for dtype in arg
                ), f"Unknown dtype in {str(arg)}"
        else:
            # 否则，验证所有参数都是已知的 torch.dtype
            assert all(
                isinstance(arg, torch.dtype) for arg in args
            ), f"Unknown dtype in {str(args)}"

        self.args = args
        self.device_type = device_type

    # 将 dtypes 实例作为装饰器使用
    def __call__(self, fn):
        # 获取现有的 dtypes 属性或创建新的空字典
        d = getattr(fn, "dtypes", {})
        # 确保在同一设备类型下不会重定义 dtypes
        assert self.device_type not in d, f"dtypes redefinition for {self.device_type}"
        # 将设备类型及其对应的数据类型参数存储在函数的 dtypes 属性中
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn


# dtypesIfCPU 类，继承自 dtypes，用于在 CPU 上覆盖指定的数据类型
class dtypesIfCPU(dtypes):
    def __init__(self, *args):
        super().__init__(*args, device_type="cpu")


# dtypesIfCUDA 类，继承自 dtypes，用于在 CUDA 上覆盖指定的数据类型
class dtypesIfCUDA(dtypes):
    def __init__(self, *args):
        super().__init__(*args, device_type="cuda")


# dtypesIfMPS 类，继承自 dtypes，用于在 MPS 上覆盖指定的数据类型
class dtypesIfMPS(dtypes):
    # 定义一个构造函数，初始化对象
    def __init__(self, *args):
        # 调用父类的构造函数，传入可变参数 *args，并指定设备类型为 "mps"
        super().__init__(*args, device_type="mps")
# 定义一个名为 dtypesIfPRIVATEUSE1 的类，继承自 dtypes 类
class dtypesIfPRIVATEUSE1(dtypes):
    # 初始化方法，接受任意数量的参数，并调用父类的初始化方法
    def __init__(self, *args):
        # 调用父类 dtypes 的初始化方法，同时设定 device_type 为私有使用1的后端名称
        super().__init__(*args, device_type=torch._C._get_privateuse1_backend_name())


# 定义一个装饰器函数 onlyCPU，接受一个函数 fn 作为参数，返回一个仅在 CPU 上运行的函数
def onlyCPU(fn):
    return onlyOn("cpu")(fn)


# 定义一个装饰器函数 onlyCUDA，接受一个函数 fn 作为参数，返回一个仅在 CUDA 上运行的函数
def onlyCUDA(fn):
    return onlyOn("cuda")(fn)


# 定义一个装饰器函数 onlyMPS，接受一个函数 fn 作为参数，返回一个仅在 MPS 上运行的函数
def onlyMPS(fn):
    return onlyOn("mps")(fn)


# 定义一个装饰器函数 onlyXPU，接受一个函数 fn 作为参数，返回一个仅在 XPU 上运行的函数
def onlyXPU(fn):
    return onlyOn("xpu")(fn)


# 定义一个装饰器函数 onlyHPU，接受一个函数 fn 作为参数，返回一个仅在 HPU 上运行的函数
def onlyHPU(fn):
    return onlyOn("hpu")(fn)


# 定义一个装饰器函数 onlyPRIVATEUSE1，接受一个函数 fn 作为参数，根据私有使用1后端的可用性选择是否跳过测试
def onlyPRIVATEUSE1(fn):
    # 获取私有使用1后端的名称
    device_type = torch._C._get_privateuse1_backend_name()
    # 获取与该名称对应的 torch 模块
    device_mod = getattr(torch, device_type, None)
    # 如果找不到对应的模块，返回一个跳过测试的装饰器
    if device_mod is None:
        reason = f"Skip as torch has no module of {device_type}"
        return unittest.skip(reason)(fn)
    # 否则返回一个仅在私有使用1后端上运行的函数
    return onlyOn(device_type)(fn)


# 定义一个装饰器函数 onlyCUDAAndPRIVATEUSE1，接受一个函数 fn 作为参数，在 CUDA 和私有使用1后端上运行
def onlyCUDAAndPRIVATEUSE1(fn):
    # 包装函数，检查当前设备类型是否是 CUDA 或私有使用1后端，否则抛出跳过测试的异常
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ("cuda", torch._C._get_privateuse1_backend_name()):
            reason = f"onlyCUDAAndPRIVATEUSE1: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)
        return fn(self, *args, **kwargs)

    return only_fn


# 定义一个装饰器函数 disablecuDNN，接受一个函数 fn 作为参数，在 CUDA 并且存在 cuDNN 的情况下禁用 cuDNN
def disablecuDNN(fn):
    # 包装函数，如果当前设备类型是 CUDA 并且有 cuDNN，设置 cuDNN 标志为禁用，然后执行原函数
    @wraps(fn)
    def disable_cudnn(self, *args, **kwargs):
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        # 如果不是 CUDA 或者没有 cuDNN，直接执行原函数
        return fn(self, *args, **kwargs)

    return disable_cudnn


# 定义一个装饰器函数 disableMkldnn，接受一个函数 fn 作为参数，在存在 MKL-DNN 的情况下禁用 MKL-DNN
def disableMkldnn(fn):
    # 包装函数，如果当前环境支持 MKL-DNN，设置 MKL-DNN 标志为禁用，然后执行原函数
    @wraps(fn)
    def disable_mkldnn(self, *args, **kwargs):
        if torch.backends.mkldnn.is_available():
            with torch.backends.mkldnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        # 如果不支持 MKL-DNN，直接执行原函数
        return fn(self, *args, **kwargs)

    return disable_mkldnn


# 定义一个装饰器函数 expectedFailureCPU，接受一个函数 fn 作为参数，返回一个预期在 CPU 上失败的测试
def expectedFailureCPU(fn):
    return expectedFailure("cpu")(fn)


# 定义一个装饰器函数 expectedFailureCUDA，接受一个函数 fn 作为参数，返回一个预期在 CUDA 上失败的测试
def expectedFailureCUDA(fn):
    return expectedFailure("cuda")(fn)


# 定义一个装饰器函数 expectedFailureXPU，接受一个函数 fn 作为参数，返回一个预期在 XPU 上失败的测试
def expectedFailureXPU(fn):
    return expectedFailure("xpu")(fn)


# 定义一个装饰器函数 expectedFailureMeta，接受一个函数 fn 作为参数，在 Torch Dynamo 环境下预期失败的测试
def expectedFailureMeta(fn):
    return skipIfTorchDynamo()(expectedFailure("meta")(fn))


# 定义一个装饰器函数 expectedFailureXLA，接受一个函数 fn 作为参数，返回一个预期在 XLA 上失败的测试
def expectedFailureXLA(fn):
    return expectedFailure("xla")(fn)


# 定义一个装饰器函数 expectedFailureHPU，接受一个函数 fn 作为参数，返回一个预期在 HPU 上失败的测试
def expectedFailureHPU(fn):
    return expectedFailure("hpu")(fn)


# 定义一个装饰器函数 skipCPUIfNoLapack，接受一个函数 fn 作为参数，如果没有 LAPACK 则跳过在 CPU 上的测试
def skipCPUIfNoLapack(fn):
    return skipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# 定义一个装饰器函数 skipCPUIfNoFFT，接受一个函数 fn 作为参数，如果没有 FFT 则跳过在 CPU 上的测试
def skipCPUIfNoFFT(fn):
    return skipCPUIf(not torch._C.has_spectral, "PyTorch is built without FFT support")(fn)


# 定义一个装饰器函数 skipCPUIfNoMkl，接受一个函数 fn 作为参数，如果没有 MKL 则跳过在 CPU 上的测试
def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


# 定义一个装饰器函数 skipCPUIfNoMklSparse，接受一个函数 fn 作为参数，如果没有 MKL Sparse 则跳过在 CPU 上的测试
def skipCPUIfNoMklSparse(fn):
    return skipCPUIf(
        IS_WINDOWS or not TEST_MKL, "PyTorch is built without MKL support"
    )(fn)


# 定义一个装饰器函数 skipCPUIfNoMkldnn，接受一个函数 fn 作为参数，如果没有 MKL-DNN 则跳过在 CPU 上的测试
def skipCPUIfNoMkldnn(fn):
    # 尚未完整定义，后续代码可能继续定义这个函数
    # 如果当前环境下 PyTorch 没有 MKL-DNN 支持，则跳过 CPU 加速，否则执行指定的函数
    return skipCPUIf(
        # 检查当前 PyTorch 是否可用 MKL-DNN 加速
        not torch.backends.mkldnn.is_available(),
        # 如果没有 MKL-DNN 支持，则返回相应的提示消息
        "PyTorch is built without mkldnn support",
    )(fn)
# 如果未检测到 MAGMA 库，则跳过 CUDA 测试。
def skipCUDAIfNoMagma(fn):
    # 返回一个装饰器，如果未检测到 "no_magma" 标志，则跳过测试。
    return skipCUDAIf("no_magma", "no MAGMA library detected")(
        # 如果条件为真，则跳过非默认流测试，然后再应用给定的测试函数。
        skipCUDANonDefaultStreamIf(True)(fn)
    )


# 检查是否具有 cuSOLVER 库。
def has_cusolver():
    # 如果 TEST_WITH_ROCM 为假，则返回 True，表示具有 cuSOLVER。
    return not TEST_WITH_ROCM


# 检查是否具有 hipSOLVER 库。
def has_hipsolver():
    # 获取当前 ROCm 的版本信息。
    rocm_version = _get_torch_rocm_version()
    # 如果 ROCm 版本大于等于 (5, 3)，则 hipSOLVER 可用。
    # hipSOLVER 在 ROCm < 5.3 上被禁用。
    return rocm_version >= (5, 3)


# 如果 cuSOLVER 或 hipSOLVER 不可用，则跳过 CUDA/ROCM 测试。
def skipCUDAIfNoCusolver(fn):
    return skipCUDAIf(
        not has_cusolver() and not has_hipsolver(), "cuSOLVER not available"
    )(fn)


# 如果 cuSOLVER 不可用且 MAGMA 也不可用，则跳过测试。
def skipCUDAIfNoMagmaAndNoCusolver(fn):
    if has_cusolver():
        # 如果有 cuSOLVER，则不跳过测试。
        return fn
    else:
        # 否则，调用 skipCUDAIfNoMagma 函数来决定是否跳过测试。
        # cuSolver 在 CUDA < 10.1.243 上被禁用，测试依赖于 MAGMA。
        return skipCUDAIfNoMagma(fn)


# 如果 cuSOLVER/hipSOLVER 和 MAGMA 都不可用，则跳过测试。
def skipCUDAIfNoMagmaAndNoLinalgsolver(fn):
    if has_cusolver() or has_hipsolver():
        # 如果有 cuSOLVER 或 hipSOLVER，则不跳过测试。
        return fn
    else:
        # 否则，调用 skipCUDAIfNoMagma 函数来决定是否跳过测试。
        # cuSolver 在 CUDA < 10.1.243 上被禁用，测试依赖于 MAGMA。
        return skipCUDAIfNoMagma(fn)


# 在使用 ROCm 时，跳过 CUDA 测试。
def skipCUDAIfRocm(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    def dec_fn(fn):
        # 构造跳过测试的理由。
        reason = f"skipCUDAIfRocm: {msg}"
        # 如果 TEST_WITH_ROCM 为真，则跳过测试。
        return skipCUDAIf(TEST_WITH_ROCM, reason=reason)(fn)

    if func:
        return dec_fn(func)
    return dec_fn


# 在不使用 ROCm 时，跳过 CUDA 测试。
def skipCUDAIfNotRocm(fn):
    return skipCUDAIf(
        not TEST_WITH_ROCM, "test doesn't currently work on the CUDA stack"
    )(fn)


# 如果 ROCm 不可用或其版本低于请求的版本，则跳过 CUDA 测试。
def skipCUDAIfRocmVersionLessThan(version=None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == "cuda":
                if not TEST_WITH_ROCM:
                    # 如果 ROCm 不可用，则抛出跳过测试的异常。
                    reason = "ROCm not available"
                    raise unittest.SkipTest(reason)
                rocm_version_tuple = _get_torch_rocm_version()
                if (
                    rocm_version_tuple is None
                    or version is None
                    or rocm_version_tuple < tuple(version)
                ):
                    # 如果 ROCm 版本过低，则抛出跳过测试的异常。
                    reason = (
                        f"ROCm {rocm_version_tuple} is available but {version} required"
                    )
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# 如果没有使用 MIOpen NHWC 激活，则跳过 CUDA 测试。
def skipCUDAIfNotMiopenSuggestNHWC(fn):
    return skipCUDAIf(
        not TEST_WITH_MIOPEN_SUGGEST_NHWC,
        "test doesn't currently work without MIOpen NHWC activation",
    )(fn)


# 跳过指定 CUDA 版本的测试，版本以 [主版本号，次版本号] 列表的形式给出。
def skipCUDAVersionIn(versions: List[Tuple[int, int]] = None):
    # 定义一个装饰器函数 `dec_fn`，接受一个函数 `fn` 作为参数
    def dec_fn(fn):
        # 使用 functools 库的 `wraps` 装饰器，将 `wrap_fn` 函数包装成 `fn` 的修饰符
        @wraps(fn)
        # 定义修饰后的函数 `wrap_fn`，它接受 `self` 和任意位置和关键字参数 `args` 和 `kwargs`
        def wrap_fn(self, *args, **kwargs):
            # 调用 `_get_torch_cuda_version` 函数，获取当前的 Torch CUDA 版本
            version = _get_torch_cuda_version()
            # 如果 CUDA 版本为 (0, 0)，表示在 CPU 或 ROCm 上运行，直接调用原始函数 `fn` 并返回结果
            if version == (0, 0):  # cpu or rocm
                return fn(self, *args, **kwargs)
            # 如果当前 CUDA 版本在已定义的版本集合 `versions` 中，抛出跳过测试的异常
            if version in (versions or []):
                reason = f"test skipped for CUDA version {version}"
                raise unittest.SkipTest(reason)
            # 否则，调用原始函数 `fn` 并返回结果
            return fn(self, *args, **kwargs)

        # 返回修饰后的函数 `wrap_fn`
        return wrap_fn

    # 返回装饰器函数 `dec_fn` 本身
    return dec_fn
# 如果 CUDA 版本低于指定版本，则跳过测试
def skipCUDAIfVersionLessThan(versions: Tuple[int, int] = None):
    # 装饰器函数，用于装饰测试函数
    def dec_fn(fn):
        @wraps(fn)
        # 包装后的测试函数
        def wrap_fn(self, *args, **kwargs):
            # 获取当前系统的 CUDA 版本
            version = _get_torch_cuda_version()
            # 如果是 CPU 或者 ROCm，则不跳过测试
            if version == (0, 0):  # cpu or rocm
                return fn(self, *args, **kwargs)
            # 如果 CUDA 版本低于指定版本，则抛出 SkipTest 异常
            if version < versions:
                reason = f"test skipped for CUDA versions < {version}"
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# 如果 cuDNN 不可用或其版本低于请求的版本，则跳过测试
def skipCUDAIfCudnnVersionLessThan(version=0):
    # 装饰器函数，用于装饰测试函数
    def dec_fn(fn):
        @wraps(fn)
        # 包装后的测试函数
        def wrap_fn(self, *args, **kwargs):
            # 如果设备类型为 CUDA
            if self.device_type == "cuda":
                # 如果没有 cuDNN，则抛出 SkipTest 异常
                if self.no_cudnn:
                    reason = "cuDNN not available"
                    raise unittest.SkipTest(reason)
                # 如果 cuDNN 版本为 None 或者低于请求的版本，则抛出 SkipTest 异常
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = f"cuDNN version {self.cudnn_version} is available but {version} required"
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# 如果 cuSparse 通用 API 不可用，则跳过测试
def skipCUDAIfNoCusparseGeneric(fn):
    return skipCUDAIf(not TEST_CUSPARSE_GENERIC, "cuSparse Generic API not available")(fn)


# 如果 hipSparse 通用 API 不可用，则跳过测试
def skipCUDAIfNoHipsparseGeneric(fn):
    return skipCUDAIf(not TEST_HIPSPARSE_GENERIC, "hipSparse Generic API not available")(fn)


# 如果 Sparse 通用 API 不可用，则跳过测试
def skipCUDAIfNoSparseGeneric(fn):
    return skipCUDAIf(
        not (TEST_CUSPARSE_GENERIC or TEST_HIPSPARSE_GENERIC),
        "Sparse Generic API not available",
    )(fn)


# 如果没有 cuDNN，则跳过测试
def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)


# 如果使用的是 MIOpen，则标记为跳过测试
def skipCUDAIfMiopen(fn):
    return skipCUDAIf(torch.version.hip is not None, "Marked as skipped for MIOpen")(fn)


# 如果 MIOpen 不可用，则跳过测试
def skipCUDAIfNoMiopen(fn):
    return skipCUDAIf(torch.version.hip is None, "MIOpen is not available")(skipCUDAIfNoCudnn(fn))


# 标记为跳过 lazy tensors 的测试
def skipLazy(fn):
    return skipLazyIf(True, "test doesn't work with lazy tensors")(fn)


# 标记为跳过 meta tensors 的测试
def skipMeta(fn):
    return skipMetaIf(True, "test doesn't work with meta tensors")(fn)


# 标记为跳过 XLA 的测试
def skipXLA(fn):
    return skipXLAIf(True, "Marked as skipped for XLA")(fn)


# 标记为跳过 MPS backend 的测试
def skipMPS(fn):
    return skipMPSIf(True, "test doesn't work on MPS backend")(fn)


# 标记为跳过 HPU backend 的测试
def skipHPU(fn):
    return skipHPUIf(True, "test doesn't work on HPU backend")(fn)


# 标记为跳过 privateuse1 backend 的测试
def skipPRIVATEUSE1(fn):
    return skipPRIVATEUSE1If(True, "test doesn't work on privateuse1 backend")(fn)


# TODO: "all" 在名称中不再准确，因为现在还有 XLA 和 MPS 等，可能需要列出所有可用设备类型的测试基类。
# 定义一个函数，用于获取所有可用的设备类型列表
def get_all_device_types() -> List[str]:
    # 如果当前系统不支持 CUDA（即没有 GPU），则返回包含 "cpu" 的列表
    return ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
```