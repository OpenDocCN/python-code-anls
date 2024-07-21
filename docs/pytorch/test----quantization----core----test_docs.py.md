# `.\pytorch\test\quantization\core\test_docs.py`

```
# Owner(s): ["oncall: quantization"]

# 导入正则表达式模块
import re
# 导入上下文管理模块
import contextlib
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 导入 PyTorch 模块
import torch

# 从 torch.testing._internal.common_quantization 模块中导入相关测试类和模型
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    SingleLayerLinearModel,
)
# 从 torch.testing._internal.common_quantized 模块中导入覆盖量化引擎的方法
from torch.testing._internal.common_quantized import override_quantized_engine
# 从 torch.testing._internal.common_utils 模块中导入 IS_ARM64 常量
from torch.testing._internal.common_utils import IS_ARM64


class TestQuantizationDocs(QuantizationTestCase):
    r"""
    The tests in this section import code from the quantization docs and check that
    they actually run without errors. In cases where objects are undefined in the code snippet,
    they must be provided in the test. The imports seem to behave a bit inconsistently,
    they can be imported either in the test file or passed as a global input
    """

    def run(self, result=None):
        # 如果运行环境是 ARM64 架构，则使用 qnnpack 引擎覆盖量化引擎；否则使用空上下文
        with override_quantized_engine("qnnpack") if IS_ARM64 else contextlib.nullcontext():
            # 调用父类的 run 方法来执行测试
            super().run(result)

    def _get_code(
        self, path_from_pytorch, unique_identifier, offset=2, short_snippet=False
    def get_code_from_docs(unique_identifier, path_from_pytorch, offset=2, short_snippet=False):
        r"""
        This function reads in the code from the docs given a unique identifier.
        Most code snippets have a 2 space indentation, for other indentation levels,
        change the offset `arg`. the `short_snippet` arg can be set to allow for testing
        of smaller snippets, the check that this arg controls is used to make sure that
        we are not accidentally only importing a blank line or something.
        """
    
        def get_correct_path(path_from_pytorch):
            r"""
            Current working directory when CI is running test seems to vary, this function
            looks for docs relative to this test file.
            """
            # 获取当前文件的父目录作为核心目录
            core_dir = Path(__file__).parent
            # 断言核心目录匹配特定模式，用于检查文件位置是否正确
            assert core_dir.match("test/quantization/core/"), (
                "test_docs.py is in an unexpected location. If you've been "
                "moving files around, ensure that the test and build files have "
                "been updated to have the correct relative path between "
                "test_docs.py and the docs."
            )
            # 获取 PyTorch 根目录
            pytorch_root = core_dir.parent.parent.parent
            return pytorch_root / path_from_pytorch
    
        # 根据给定路径获取正确的文件路径
        path_to_file = get_correct_path(path_from_pytorch)
        if path_to_file:
            # 打开文件并读取所有行内容
            with open(path_to_file) as file:
                content = file.readlines()
    
            # 确保 unique_identifier 后有换行符
            if "\n" not in unique_identifier:
                unique_identifier += "\n"
    
            # 断言 unique_identifier 存在于内容中
            assert unique_identifier in content, f"could not find {unique_identifier} in {path_to_file}"
    
            # 获取代码块起始行号
            line_num_start = content.index(unique_identifier) + 1
    
            # 寻找代码块结束行号
            # 此正则表达式将匹配不以换行符或以 "  " + offset 个空格开头的行
            r = re.compile("^[^\n," + " " * offset + "]")
            # 找到第一个与正则表达式匹配的行
            line_after_code = next(filter(r.match, content[line_num_start:]))
            last_line_num = content.index(line_after_code)
    
            # 去除每行开头的 offset 个字符并组合成完整代码字符串
            code = "".join(
                [x[offset:] for x in content[line_num_start + 1 : last_line_num]]
            )
    
            # 确保获取的代码长度大于3行或者 short_snippet 为 True
            assert last_line_num - line_num_start > 3 or short_snippet, (
                f"The code in {path_to_file} identified by {unique_identifier} seems suspiciously short:"
                f"\n\n###code-start####\n{code}###code-end####"
            )
            return code
    
        return None
    # 定义一个方法 `_test_code`，用于运行给定的代码字符串 `code`，并可以使用全局输入 `global_inputs` 中的变量
    def _test_code(self, code, global_inputs=None):
        r"""
        This function runs `code` using any vars in `global_inputs`
        """
        # 如果代码不为 None，则编译代码字符串 `code` 成为表达式 `expr`
        if code is not None:
            expr = compile(code, "test", "exec")
            # 执行表达式 `expr`，在给定的全局输入 `global_inputs` 上下文中执行
            exec(expr, global_inputs)

    # 测试函数，用于测试文档中的量化部分的代码示例
    def test_quantization_doc_ptdq(self):
        # 指定 PyTorch 文档中的量化部分的路径
        path_from_pytorch = "docs/source/quantization.rst"
        # 设置唯一标识符，用于获取特定的代码示例
        unique_identifier = "PTDQ API Example::"
        # 从指定路径中获取标识符对应的代码
        code = self._get_code(path_from_pytorch, unique_identifier)
        # 使用 `_test_code` 方法执行获取到的代码
        self._test_code(code)

    # 测试函数，测试量化文档中的另一个 API 示例
    def test_quantization_doc_ptsq(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "PTSQ API Example::"
        code = self._get_code(path_from_pytorch, unique_identifier)
        self._test_code(code)

    # 测试函数，测试量化文档中的 QAT API 示例
    def test_quantization_doc_qat(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "QAT API Example::"

        # 定义一个虚拟的函数 `_dummy_func`，用作示例中的训练循环函数
        def _dummy_func(*args, **kwargs):
            return None

        # 生成一个随机的 FP32 输入
        input_fp32 = torch.randn(1, 1, 1, 1)
        # 设置全局输入，包括一个训练循环函数和输入数据
        global_inputs = {"training_loop": _dummy_func, "input_fp32": input_fp32}
        # 获取指定标识符的代码示例
        code = self._get_code(path_from_pytorch, unique_identifier)
        # 使用 `_test_code` 方法执行获取到的代码，使用全局输入 `global_inputs`
        self._test_code(code, global_inputs)

    # 测试函数，测试量化文档中的 FXPTQ API 示例
    def test_quantization_doc_fx(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "FXPTQ API Example::"

        # 获取单层线性模型的示例输入 FP32
        input_fp32 = SingleLayerLinearModel().get_example_inputs()
        # 设置全局输入，包括用户模型类和输入数据
        global_inputs = {"UserModel": SingleLayerLinearModel, "input_fp32": input_fp32}

        # 获取指定标识符的代码示例
        code = self._get_code(path_from_pytorch, unique_identifier)
        # 使用 `_test_code` 方法执行获取到的代码，使用全局输入 `global_inputs`
        self._test_code(code, global_inputs)

    # 测试函数，测试量化文档中的自定义 API 示例
    def test_quantization_doc_custom(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "Custom API Example::"

        # 设置全局输入，包括 PyTorch 的 AO 模块中的量化子模块
        global_inputs = {"nnq": torch.ao.nn.quantized}

        # 获取指定标识符的代码示例
        code = self._get_code(path_from_pytorch, unique_identifier)
        # 使用 `_test_code` 方法执行获取到的代码，使用全局输入 `global_inputs`
        self._test_code(code, global_inputs)
```