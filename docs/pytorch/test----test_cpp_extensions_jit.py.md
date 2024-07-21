# `.\pytorch\test\test_cpp_extensions_jit.py`

```
# Owner(s): ["module: cpp-extensions"]

# 导入必要的库和模块
import glob  # 文件通配符匹配
import os  # 系统操作接口
import re  # 正则表达式库
import shutil  # 文件操作工具
import subprocess  # 子进程管理工具
import sys  # 系统参数和函数
import tempfile  # 创建临时文件和目录
import unittest  # 单元测试框架
import warnings  # 警告控制

import torch  # PyTorch 深度学习库
import torch.backends.cudnn  # CuDNN 后端
import torch.multiprocessing as mp  # 多进程管理

import torch.testing._internal.common_utils as common  # 测试工具
import torch.utils.cpp_extension  # PyTorch C++ 扩展工具
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN  # CUDA 相关测试
from torch.testing._internal.common_utils import gradcheck  # 梯度检查
from torch.utils.cpp_extension import (
    _TORCH_PATH,  # PyTorch 路径
    check_compiler_is_gcc,  # 检查是否使用 GCC 编译器
    CUDA_HOME,  # CUDA 安装路径
    get_cxx_compiler,  # 获取 C++ 编译器
    remove_extension_h_precompiler_headers,  # 移除扩展头文件的预编译器头信息
    ROCM_HOME,  # ROCm 安装路径
)

# 根据条件设置是否测试 ROCm
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
# 根据条件设置是否测试 CUDA
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
# 检查是否支持 MPS（Multi-Process Service）
TEST_MPS = torch.backends.mps.is_available()
# 判断操作系统是否为 Windows
IS_WINDOWS = sys.platform == "win32"
# 判断操作系统是否为 Linux
IS_LINUX = sys.platform.startswith("linux")


def remove_build_path():
    # 获取默认的构建根目录
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    # 如果构建根目录存在，则删除之
    if os.path.exists(default_build_root):
        if IS_WINDOWS:
            # Windows 平台下，使用 subprocess.run 调用 rm 命令来删除目录
            # rmtree 在 Windows 下会出现权限错误，这是一个绕过方式
            subprocess.run(["rm", "-rf", default_build_root], stdout=subprocess.PIPE)
        else:
            # 在非 Windows 平台，使用 shutil.rmtree 完成目录删除操作
            shutil.rmtree(default_build_root)


# 仅有一个测试使用了 gradcheck，需手动以慢速模式运行
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionJIT(common.TestCase):
    """Tests just-in-time cpp extensions.
    Don't confuse this with the PyTorch JIT (aka TorchScript).
    """

    def setUp(self):
        super().setUp()
        # C++ 扩展使用相对路径，相对路径基于当前文件，因此暂时切换工作目录
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # 恢复工作目录到原始状态（参见 setUp）
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        # 在整个测试类开始时，调用 remove_build_path 函数删除构建路径
        remove_build_path()

    @classmethod
    def tearDownClass(cls):
        # 在整个测试类结束时，再次调用 remove_build_path 函数删除构建路径
        remove_build_path()
    # 定义一个测试用例，用于测试即时编译的 C++ 扩展模块
    def test_jit_compile_extension(self):
        # 使用 torch.utils.cpp_extension.load() 方法加载 JIT 编译的 C++ 扩展模块
        module = torch.utils.cpp_extension.load(
            name="jit_extension",
            # 指定源文件列表，包括 "cpp_extensions/jit_extension.cpp" 和 "cpp_extensions/jit_extension2.cpp"
            sources=[
                "cpp_extensions/jit_extension.cpp",
                "cpp_extensions/jit_extension2.cpp",
            ],
            # 指定额外的头文件路径列表，包括 "cpp_extensions"、"path / with spaces in it" 和 "path with quote'"
            extra_include_paths=[
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            # 添加额外的编译标志 "-g"
            extra_cflags=["-g"],
            # 开启详细输出模式
            verbose=True,
        )
        # 创建两个大小为 4x4 的随机张量 x 和 y
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # 调用 JIT 编译的模块中的 tanh_add 方法，计算 x.tanh() + y.tanh()
        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        # 检查是否能够调用在主 C++ 文件之外定义的方法
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())

        # 检查是否可以使用 JIT 编译的类 Doubler
        doubler = module.Doubler(2, 2)
        # 检查返回的张量是否没有梯度信息
        self.assertIsNone(doubler.get().grad)
        # 检查返回的张量所有元素的和是否为 4
        self.assertEqual(doubler.get().sum(), 4)
        # 检查类的 forward 方法返回的张量所有元素的和是否为 8
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    # 定义一个测试用例，用于测试 JIT 编译的 CUDA 扩展模块
    def test_jit_cuda_extension(self):
        # NOTE: 扩展模块的名称必须与模块的名称相同
        # 使用 torch.utils.cpp_extension.load() 方法加载 JIT 编译的 CUDA 扩展模块
        module = torch.utils.cpp_extension.load(
            name="torch_test_cuda_extension",
            # 指定源文件列表，包括 "cpp_extensions/cuda_extension.cpp" 和 "cpp_extensions/cuda_extension.cu"
            sources=[
                "cpp_extensions/cuda_extension.cpp",
                "cpp_extensions/cuda_extension.cu",
            ],
            # 指定额外的 CUDA 编译标志 "-O2"
            extra_cuda_cflags=["-O2"],
            # 开启详细输出模式
            verbose=True,
            # 禁止保留中间文件
            keep_intermediates=False,
        )

        # 创建一个在 CUDA 设备上的大小为 100x1 的零张量 x 和 y
        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        # 调用 JIT 编译的模块中的 sigmoid_add 方法，计算 x 和 y 的 sigmoid 函数之和，并将结果移动到 CPU
        z = module.sigmoid_add(x, y).cpu()

        # 断言 z 的所有元素是否都等于 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_MPS, "MPS not found")
    # 定义一个测试用例，用于测试 MPS 扩展模块
    def test_mps_extension(self):
        # 使用 torch.utils.cpp_extension.load() 方法加载 MPS 扩展模块
        module = torch.utils.cpp_extension.load(
            name="torch_test_mps_extension",
            # 指定源文件列表，包括 "cpp_extensions/mps_extension.mm"
            sources=[
                "cpp_extensions/mps_extension.mm",
            ],
            # 开启详细输出模式
            verbose=True,
            # 禁止保留中间文件
            keep_intermediates=False,
        )

        # 创建一个大小为 100000 的随机张量 x 和 y，数据类型为 torch.float32，设备为 CPU
        tensor_length = 100000
        x = torch.randn(tensor_length, device="cpu", dtype=torch.float32)
        y = torch.randn(tensor_length, device="cpu", dtype=torch.float32)

        # 调用 MPS 扩展模块中的 get_cpu_add_output 方法，计算 x 和 y 的加法结果
        cpu_output = module.get_cpu_add_output(x, y)
        # 调用 MPS 扩展模块中的 get_mps_add_output 方法，计算 x 和 y 的加法结果（使用 MPS 加速）
        mps_output = module.get_mps_add_output(x.to("mps"), y.to("mps"))

        # 断言 CPU 输出与 MPS 输出相等
        self.assertEqual(cpu_output, mps_output.to("cpu"))

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(TEST_ROCM, "disabled on rocm")
    def test_jit_cuda_archflags(self):
        # 测试多种组合情况:
        #   - 在当前测试的机器上的默认配置
        #   - 分隔符可以是 ';'（最常见）或 ' '
        #   - 架构名称
        #   - 带有/不带有 '+PTX'

        # 获取当前 CUDA 设备的数量
        n = torch.cuda.device_count()
        # 获取每个设备的计算能力信息，并将其存储为集合
        capabilities = {torch.cuda.get_device_capability(i) for i in range(n)}
        
        # 定义期望的架构标志字典
        # 如果不带 '+PTX'，则 PTX 列表为 None
        archflags = {
            "": (
                [f"{capability[0]}{capability[1]}" for capability in capabilities],
                None,
            ),
            "Maxwell+Tegra;6.1": (["53", "61"], None),
            "Volta": (["70"], ["70"]),
        }
        
        # 添加额外的架构标志组合并更新 archflags 字典
        archflags["7.5+PTX"] = (["75"], ["75"])
        archflags["5.0;6.0+PTX;7.0;7.5"] = (["50", "60", "70", "75"], ["60"])
        
        # 检查当前 CUDA 版本是否低于 12
        if int(torch.version.cuda.split(".")[0]) < 12:
            # CUDA 12 之后不支持计算能力低于 5.0 的设备
            archflags["Pascal 3.5"] = (["35", "60", "61"], None)

        # 遍历 archflags 字典中的每个架构标志
        for flags, expected in archflags.items():
            try:
                # 调用 _run_jit_cuda_archflags 方法，并传入当前标志及其期望的结果
                self._run_jit_cuda_archflags(flags, expected)
            except RuntimeError as e:
                # 如果使用设备的默认标志（空标志）可能会失败，这里捕获特定的 RuntimeError 错误信息
                # 如果错误信息中包含 "Error building"，则忽略此异常
                if not flags and "Error building" in str(e):
                    pass
                else:
                    raise
            try:
                # 同步 CUDA 设备以确保操作完成
                torch.cuda.synchronize()
            except RuntimeError:
                # 忽略任何错误，例如当前设备不支持的 PTX 代码，以避免错误影响其他测试
                pass

    @unittest.skipIf(not TEST_CUDNN, "CuDNN not found")
    @unittest.skipIf(TEST_ROCM, "Not supported on ROCm")
    def test_jit_cudnn_extension(self):
        # 实现 CuDNN ReLU

        # 根据操作系统选择额外的链接标志
        if IS_WINDOWS:
            extra_ldflags = ["cudnn.lib"]
        else:
            extra_ldflags = ["-lcudnn"]
        
        # 加载 CuDNN 扩展模块
        module = torch.utils.cpp_extension.load(
            name="torch_test_cudnn_extension",
            sources=["cpp_extensions/cudnn_extension.cpp"],
            extra_ldflags=extra_ldflags,
            verbose=True,
            with_cuda=True,
        )

        # 在 CUDA 设备上创建随机张量 x 和全零张量 y
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)
        
        # 调用 CuDNN 扩展模块中的 cudnn_relu 函数进行操作
        module.cudnn_relu(x, y)  # y=relu(x)
        
        # 使用 PyTorch 中的函数验证操作结果
        self.assertEqual(torch.nn.functional.relu(x), y)
        
        # 使用断言检测特定错误是否会被引发
        with self.assertRaisesRegex(RuntimeError, "same size"):
            y_incorrect = torch.zeros(20, device="cuda", dtype=torch.float32)
            module.cudnn_relu(x, y_incorrect)
    def test_inline_jit_compile_extension_with_functions_as_list(self):
        # 定义包含 C++ 函数的字符串
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        # 使用 torch.utils.cpp_extension.load_inline 函数加载内联编译的扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_with_functions_list",  # 模块名称
            cpp_sources=cpp_source,  # 源代码字符串列表
            functions="tanh_add",  # 指定要加载的函数列表
            verbose=True,  # 输出详细信息
        )

        # 检查生成的模块中 tanh_add 函数的文档字符串
        self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "tanh_add")

        # 创建随机张量 x 和 y
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # 调用模块中的 tanh_add 函数，并检查结果是否符合预期
        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_inline_jit_compile_extension_with_functions_as_dict(self):
        # 定义包含 C++ 函数的字符串
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        # 使用 torch.utils.cpp_extension.load_inline 函数加载内联编译的扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_with_functions_dict",  # 模块名称
            cpp_sources=cpp_source,  # 源代码字符串列表
            functions={"tanh_add": "Tanh and then sum :D"},  # 指定要加载的函数及其描述
            verbose=True,  # 输出详细信息
        )

        # 检查生成的模块中 tanh_add 函数的文档字符串
        self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "Tanh and then sum :D")

    def test_inline_jit_compile_extension_multiple_sources_and_no_functions(self):
        # 第一个源代码字符串定义一个 C++ 函数
        cpp_source1 = """
        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y) {
          return x.sin() + y.sin();
        }
        """

        # 第二个源代码字符串定义一个 Pybind11 绑定
        cpp_source2 = """
        #include <torch/extension.h>
        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
          m.def("sin_add", &sin_add, "sin(x) + sin(y)");
        }
        """

        # 使用 torch.utils.cpp_extension.load_inline 函数加载内联编译的扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension",  # 模块名称
            cpp_sources=[cpp_source1, cpp_source2],  # 源代码字符串列表
            verbose=True,  # 输出详细信息
        )

        # 创建随机张量 x 和 y
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # 调用模块中的 sin_add 函数，并检查结果是否符合预期
        z = module.sin_add(x, y)
        self.assertEqual(z, x.sin() + y.sin())

    @unittest.skip("Temporarily disabled")
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    # 定义一个测试函数，用于测试内联 JIT 编译的 CUDA 扩展
    def test_inline_jit_compile_extension_cuda(self):
        # CUDA 源码字符串，包含了一个 CUDA 核函数和一个调用该核函数的函数
        cuda_source = """
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
          auto output = torch::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
          return output;
        }
        """

        # 在这里，C++ 源码只需要声明函数的签名
        cpp_source = "torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);"

        # 使用 torch.utils.cpp_extension.load_inline 函数加载内联的 JIT 扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_cuda",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["cos_add"],
            verbose=True,
        )

        # 断言模块中 cos_add 函数的文档字符串的第三行是 "cos_add"
        self.assertEqual(module.cos_add.__doc__.split("\n")[2], "cos_add")

        # 在 CUDA 设备上创建随机张量 x 和 y
        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        # 调用模块中的 cos_add 函数，计算 x.cos() + y.cos() 并保存在 z 中
        z = module.cos_add(x, y)
        # 断言 z 等于 x.cos() + y.cos()
        self.assertEqual(z, x.cos() + y.cos())

    # 临时禁用此测试用例
    @unittest.skip("Temporarily disabled")
    # 如果未找到 CUDA 或 ROCm，则跳过此测试用例
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_inline_jit_compile_custom_op_cuda(self):
        # CUDA源码定义了一个包含__global__函数的字符串，实现了对输入张量x和y进行元素级cos加法操作
        cuda_source = """
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
          auto output = torch::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());
          return output;
        }
        """

        # 在此处，C++源码仅声明了函数签名
        cpp_source = """
           #include <torch/library.h>
           torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);

           TORCH_LIBRARY(inline_jit_extension_custom_op_cuda, m) {
             m.def("cos_add", cos_add);
           }
        """

        # 使用torch.utils.cpp_extension.load_inline加载内联的CUDA和C++源码，创建一个自定义操作的扩展模块
        torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_custom_op_cuda",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            verbose=True,
            is_python_module=False,
        )

        # 在CUDA设备上创建随机张量x和y
        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        # 调用加载的扩展模块中的cos_add函数，对张量x和y进行元素级cos加法，返回结果张量z
        z = torch.ops.inline_jit_extension_custom_op_cuda.cos_add(x, y)
        # 使用断言验证结果张量z是否等于x和y的元素级cos和
        self.assertEqual(z, x.cos() + y.cos())

    def test_inline_jit_compile_extension_throws_when_functions_is_bad(self):
        # 使用断言验证加载内联C++扩展时，如果指定了无效的functions参数，会抛出ValueError异常
        with self.assertRaises(ValueError):
            torch.utils.cpp_extension.load_inline(
                name="invalid_jit_extension", cpp_sources="", functions=5
            )

    def test_lenient_flag_handling_in_jit_extensions(self):
        # 定义一个包含tanh_add函数的C++源码字符串，实现对输入张量x和y进行元素级tanh加法操作
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        # 使用torch.utils.cpp_extension.load_inline加载内联C++源码，创建一个自定义操作的扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="lenient_flag_handling_extension",
            cpp_sources=cpp_source,
            functions="tanh_add",
            extra_cflags=["-g\n\n", "-O0 -Wall"],  # 额外的编译标志
            extra_include_paths=["       cpp_extensions\n"],  # 额外的包含路径
            verbose=True,
        )

        # 创建两个全零的张量x和y
        x = torch.zeros(100, dtype=torch.float32)
        y = torch.zeros(100, dtype=torch.float32)

        # 调用加载的扩展模块中的tanh_add函数，对张量x和y进行元素级tanh加法，返回结果张量z
        z = module.tanh_add(x, y).cpu()
        # 使用断言验证结果张量z是否等于x和y的元素级tanh和
        self.assertEqual(z, x.tanh() + y.tanh())

    @unittest.skip("Temporarily disabled")  # 暂时禁用此测试
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")  # 如果没有CUDA或ROCM，则跳过测试
    def test_half_support(self):
        """
        Checks for an issue with operator< ambiguity for half when certain
        THC headers are included.

        See https://github.com/pytorch/pytorch/pull/10301#issuecomment-416773333
        for the corresponding issue.
        """
        cuda_source = """
        template<typename T, typename U>
        __global__ void half_test_kernel(const T* input, U* output) {
            if (input[0] < input[1] || input[0] >= input[1]) {
                output[0] = 123;
            }
        }

        torch::Tensor half_test(torch::Tensor input) {
            auto output = torch::empty(1, input.options().dtype(torch::kFloat));
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "half_test", [&] {
                // 调用 CUDA kernel，比较输入张量的两个元素并根据条件设置输出值
                half_test_kernel<scalar_t><<<1, 1>>>(
                    input.data<scalar_t>(),
                    output.data<float>());
            });
            // 返回处理后的输出张量
            return output;
        }
        """

        # 使用 PyTorch 内联加载 CUDA 源码生成的扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="half_test_extension",
            cpp_sources="torch::Tensor half_test(torch::Tensor input);",
            cuda_sources=cuda_source,
            functions=["half_test"],
            verbose=True,
        )

        # 创建一个 CUDA 上的半精度张量
        x = torch.randn(3, device="cuda", dtype=torch.half)
        # 调用扩展模块中的 half_test 函数进行测试
        result = module.half_test(x)
        # 断言测试结果的第一个元素是否为 123
        self.assertEqual(result[0], 123)

    def test_reload_jit_extension(self):
        def compile(code):
            # 使用内联方式编译给定的 C++ 代码并加载为扩展模块
            return torch.utils.cpp_extension.load_inline(
                name="reloaded_jit_extension",
                cpp_sources=code,
                functions="f",
                verbose=True,
            )

        # 编译并加载第一段 C++ 代码作为扩展模块
        module = compile("int f() { return 123; }")
        # 断言调用模块中的 f 函数返回值是否为 123
        self.assertEqual(module.f(), 123)

        # 重新编译并加载第二段 C++ 代码作为扩展模块
        module = compile("int f() { return 456; }")
        # 断言调用模块中的 f 函数返回值是否为 456
        self.assertEqual(module.f(), 456)

        # 再次使用相同的代码编译并加载，确保模块的可重复性
        module = compile("int f() { return 456; }")
        # 断言调用模块中的 f 函数返回值是否为 456
        self.assertEqual(module.f(), 456)

        # 编译并加载另一段 C++ 代码作为扩展模块
        module = compile("int f() { return 789; }")
        # 断言调用模块中的 f 函数返回值是否为 789
        self.assertEqual(module.f(), 789)
    # 测试函数，验证 C++ 前端模块与 Python 实现在给定数据类型下的输出是否一致
    def test_cpp_frontend_module_has_same_output_as_python(self, dtype=torch.double):
        # 加载 C++ 扩展模块，指定模块名称和源文件路径，并打印详细信息
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        # 生成指定数据类型和形状的随机输入数据
        input = torch.randn(2, 5, dtype=dtype)
        
        # 使用 C++ 扩展模块创建线性层对象
        cpp_linear = extension.Net(5, 2)
        
        # 将 C++ 线性层对象转换为指定数据类型
        cpp_linear.to(dtype)
        
        # 创建 Python 实现的线性层对象，并转换为指定数据类型
        python_linear = torch.nn.Linear(5, 2).to(dtype)

        # 确保两个线性层对象具有相同的参数
        cpp_parameters = dict(cpp_linear.named_parameters())
        with torch.no_grad():
            # 将 C++ 线性层的权重和偏置复制给 Python 线性层
            python_linear.weight.copy_(cpp_parameters["fc.weight"])
            python_linear.bias.copy_(cpp_parameters["fc.bias"])

        # 分别计算 C++ 和 Python 线性层在输入上的输出
        cpp_output = cpp_linear.forward(input)
        python_output = python_linear(input)
        
        # 断言两者的输出结果应该一致
        self.assertEqual(cpp_output, python_output)

        # 对两个模型的输出结果进行反向传播求导数
        cpp_output.sum().backward()
        python_output.sum().backward()

        # 确保 C++ 线性层的所有参数都有梯度
        for p in cpp_linear.parameters():
            self.assertFalse(p.grad is None)

        # 断言 C++ 线性层权重和偏置的梯度与 Python 线性层相同
        self.assertEqual(cpp_parameters["fc.weight"].grad, python_linear.weight.grad)
        self.assertEqual(cpp_parameters["fc.bias"].grad, python_linear.bias.grad)


    # 测试函数，验证 C++ 前端模块的属性是否是最新状态
    def test_cpp_frontend_module_has_up_to_date_attributes(self):
        # 加载 C++ 扩展模块，指定模块名称和源文件路径，并打印详细信息
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        # 使用 C++ 扩展模块创建网络对象
        net = extension.Net(5, 2)

        # 断言网络对象没有任何参数
        self.assertEqual(len(net._parameters), 0)
        
        # 向网络对象添加新的参数 "foo"，并断言参数数量为 1
        net.add_new_parameter("foo", torch.eye(5))
        self.assertEqual(len(net._parameters), 1)

        # 断言网络对象有一个缓冲区
        self.assertEqual(len(net._buffers), 1)
        
        # 向网络对象添加新的缓冲区 "bar"，并断言缓冲区数量为 2
        net.add_new_buffer("bar", torch.eye(5))
        self.assertEqual(len(net._buffers), 2)

        # 断言网络对象有一个子模块
        self.assertEqual(len(net._modules), 1)
        
        # 向网络对象添加新的子模块 "fc2"，并断言子模块数量为 2
        net.add_new_submodule("fc2")
        self.assertEqual(len(net._modules), 2)

    # 如果 CUDA 或 ROCm 不可用，则跳过此单元测试
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    # 测试 CPP 前端模块与 CUDA 的 Python 互操作性
    def test_cpp_frontend_module_python_inter_op_with_cuda(self):
        # 加载名为 "cpp_frontend_extension" 的 Torch C++ 扩展模块
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        # 使用加载的扩展模块创建一个名为 Net 的对象，参数为 5 和 2
        net = extension.Net(5, 2)

        # 检查网络中所有参数的设备是否为 CPU
        for p in net.parameters():
            self.assertTrue(p.device.type == "cpu")
        # 复制当前所有 CPU 参数
        cpu_parameters = [p.clone() for p in net.parameters()]

        # 将网络对象转移到 CUDA 设备上
        device = torch.device("cuda", 0)
        net.to(device)

        # 检查网络中所有参数的设备是否为 CUDA 设备 0
        for i, p in enumerate(net.parameters()):
            self.assertTrue(p.device.type == "cuda")
            self.assertTrue(p.device.index == 0)
            # 检查 CUDA 参数与之前复制的 CPU 参数是否相等
            self.assertEqual(cpu_parameters[i], p)

        # 将网络对象转回 CPU 设备
        net.cpu()

        # 添加新的名为 "a" 和 "b" 的参数，以及名为 "c" 和 "d" 的缓冲区
        net.add_new_parameter("a", torch.eye(5))
        net.add_new_parameter("b", torch.eye(5))
        net.add_new_buffer("c", torch.eye(5))
        net.add_new_buffer("d", torch.eye(5))

        # 添加名为 "fc2" 和 "fc3" 的新子模块
        net.add_new_submodule("fc2")
        net.add_new_submodule("fc3")

        # 再次检查网络中所有参数的设备是否为 CPU
        for p in net.parameters():
            self.assertTrue(p.device.type == "cpu")

        # 将网络对象转移到 CUDA 设备上
        net.cuda()

        # 最后确认网络中所有参数的设备是否为 CUDA 设备
        for p in net.parameters():
            self.assertTrue(p.device.type == "cuda")
    def test_compilation_error_formatting(self):
        # 测试缺少分号错误消息是否包含换行符。
        # 如果消息被合并成单行，则测试失败。
        # 由于每个编译器都有自己的错误格式化方式，很难写出更具体的说明。
        with self.assertRaises(RuntimeError) as e:
            # 调用内联加载函数，期望抛出 RuntimeError 异常
            torch.utils.cpp_extension.load_inline(
                name="test_compilation_error_formatting",
                cpp_sources="int main() { return 0 }",
            )
        pattern = r".*(\\n|\\r).*"
        # 确保异常消息中不包含换行符
        self.assertNotRegex(str(e), pattern)

    def test_autograd_from_cpp(self):
        source = """
        void run_back(at::Tensor x) {
            x.backward({});
        }

        void run_back_no_gil(at::Tensor x) {
            pybind11::gil_scoped_release no_gil;
            x.backward({});
        }
        """

        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gx):
                return gx

        # 使用内联加载函数加载 C++ 源码
        test_backward_deadlock = torch.utils.cpp_extension.load_inline(
            name="test_backward_deadlock",
            cpp_sources=[source],
            functions=["run_back", "run_back_no_gil"],
        )

        # 这段代码曾经会导致死锁
        inp = torch.rand(20, requires_grad=True)
        loss = MyFn.apply(inp).sum()
        with self.assertRaisesRegex(
            RuntimeError, "The autograd engine was called while holding the GIL."
        ):
            # 测试在持有 GIL 时调用自动求导函数是否会抛出 RuntimeError
            test_backward_deadlock.run_back(loss)

        inp = torch.rand(20, requires_grad=True)
        loss = MyFn.apply(inp).sum()
        # 测试在释放 GIL 的情况下调用自动求导函数是否正常运行
        test_backward_deadlock.run_back_no_gil(loss)

    def test_custom_compound_op_autograd(self):
        # 测试自定义复合操作（即只调用其他 ATen 操作的自定义操作）是否正确返回这些操作的梯度

        source = """
        #include <torch/library.h>
        torch::Tensor my_add(torch::Tensor x, torch::Tensor y) {
          return x + y;
        }
        TORCH_LIBRARY(my, m) {
            m.def("add", &my_add);
        }
        """

        # 使用内联加载函数加载 C++ 源码作为不是 Python 模块的库
        torch.utils.cpp_extension.load_inline(
            name="is_python_module",
            cpp_sources=source,
            verbose=True,
            is_python_module=False,
        )

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        for fast_mode in (True, False):
            # 使用 gradcheck 检查自定义操作的梯度
            gradcheck(torch.ops.my.add, [a, b], eps=1e-2, fast_mode=fast_mode)
    # 定义一个测试函数，用于测试在functorch转换下，自定义的C++函数是否会引发错误
    def test_custom_functorch_error(self):
        # 加载名为"identity"的自定义C++扩展模块，来自cpp_extensions/identity.cpp
        identity_m = torch.utils.cpp_extension.load(
            name="identity",
            sources=["cpp_extensions/identity.cpp"],
        )

        # 创建一个形状为(3,)的张量t，要求其梯度计算
        t = torch.randn(3, requires_grad=True)

        # 定义一个错误消息的正则表达式，用于检查是否抛出RuntimeError异常
        msg = r"cannot use C\+\+ torch::autograd::Function with functorch"

        # 断言在使用torch.func.vmap函数时抛出RuntimeError异常，并验证异常消息是否符合预期
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.func.vmap(identity_m.identity)(t)

        # 断言在使用torch.func.grad函数时抛出RuntimeError异常，并验证异常消息是否符合预期
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.func.grad(identity_m.identity)(t)

    # 定义一个测试函数，用于测试在Linux系统下生成扩展头文件预编译文件(.pch)和签名文件(.sign)
    def test_gen_extension_h_pch(self):
        # 如果不是Linux系统，则直接返回，不执行后续代码
        if not IS_LINUX:
            return

        # 定义一个包含C++源代码的字符串变量source，实现at::Tensor的sin_add函数
        source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
            return x.sin() + y.sin();
        }
        """

        # 定义预编译头文件的路径和签名文件的路径
        head_file_pch = os.path.join(_TORCH_PATH, "include", "torch", "extension.h.gch")
        head_file_signature = os.path.join(
            _TORCH_PATH, "include", "torch", "extension.h.sign"
        )

        # 移除扩展头文件的预编译器头文件
        remove_extension_h_precompiler_headers()

        # 检查预编译头文件和签名文件是否存在，并断言它们都应该不存在
        pch_exist = os.path.exists(head_file_pch)
        signature_exist = os.path.exists(head_file_signature)
        self.assertEqual(pch_exist, False)
        self.assertEqual(signature_exist, False)

        # 使用torch.utils.cpp_extension.load_inline加载内联扩展模块，并启用预编译头文件(use_pch=True)
        torch.utils.cpp_extension.load_inline(
            name="inline_extension_with_pch",
            cpp_sources=[source],
            functions=["sin_add"],
            verbose=True,
            use_pch=True,
        )

        # 再次检查预编译头文件和签名文件是否存在，并断言它们应该存在
        pch_exist = os.path.exists(head_file_pch)
        signature_exist = os.path.exists(head_file_signature)

        # 获取C++编译器，并根据编译器类型进行额外的断言验证
        compiler = get_cxx_compiler()
        if check_compiler_is_gcc(compiler):
            self.assertEqual(pch_exist, True)
            self.assertEqual(signature_exist, True)
# 如果当前脚本作为主程序运行，则执行common模块中的run_tests函数
if __name__ == "__main__":
    common.run_tests()
```