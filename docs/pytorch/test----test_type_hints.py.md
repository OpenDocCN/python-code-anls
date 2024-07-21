# `.\pytorch\test\test_type_hints.py`

```
# 指定允许未类型化的函数（由于 mypy: allow-untyped-defs 注释）
# 定义所有者信息，指定这个模块属于 typing 模块
# 导入 doctest 模块，用于解析文档字符串中的示例代码
import doctest
# 导入 inspect 模块，用于获取对象的文档字符串
import inspect
# 导入 os 模块，提供了与操作系统相关的功能
import os
# 导入 tempfile 模块，用于创建临时文件和目录
import tempfile
# 导入 unittest 模块，用于编写和运行单元测试
import unittest
# 导入 Path 类，用于操作路径名
from pathlib import Path

# 导入 torch 库，用于进行深度学习相关的计算
import torch
# 从 torch.testing._internal.common_utils 中导入 run_tests 和 set_cwd 函数，以及 TestCase 类
from torch.testing._internal.common_utils import run_tests, set_cwd, TestCase

try:
    # 尝试导入 mypy.api 模块，用于静态类型检查
    import mypy.api
    # 设置 HAVE_MYPY 为 True，表示成功导入 mypy
    HAVE_MYPY = True
except ImportError:
    # 若导入失败，则设置 HAVE_MYPY 为 False
    HAVE_MYPY = False

# 定义函数 get_examples_from_docstring，从文档字符串中提取可运行的 Python 代码示例
def get_examples_from_docstring(docstr):
    """
    Extracts all runnable python code from the examples
    in docstrings; returns a list of lines.
    """
    # 使用 doctest.DocTestParser 解析文档字符串中的示例代码
    examples = doctest.DocTestParser().get_examples(docstr)
    # 将每个示例代码的每行前面加上四个空格，以匹配其在文件中的缩进
    return [f"    {l}" for e in examples for l in e.source.splitlines()]

# 定义函数 get_all_examples，从 torch 模块的文档字符串中提取所有示例代码
def get_all_examples():
    """get_all_examples() -> str

    This function grabs (hopefully all) examples from the torch documentation
    strings and puts them in one nonsensical module returned as a string.
    """
    # 设置需要屏蔽的函数名列表
    blocklist = {
        "_np",
    }
    # 初始化示例代码字符串
    allexamples = ""

    # 初始化示例文件的起始代码行
    example_file_lines = [
        "import torch",
        "import torch.nn.functional as F",
        "import math",
        "import numpy",
        "import io",
        "import itertools",
        "",
        # for requires_grad_ example
        # NB: We are parsing this file as Python 2, so we must use
        # Python 2 type annotation syntax
        "def preprocess(inp):",
        "    # type: (torch.Tensor) -> torch.Tensor",
        "    return inp",
    ]

    # 遍历 torch 模块中的每个函数名
    for fname in dir(torch):
        # 获取函数对象
        fn = getattr(torch, fname)
        # 获取函数的文档字符串
        docstr = inspect.getdoc(fn)
        # 如果文档字符串存在且函数名不在 blocklist 中
        if docstr and fname not in blocklist:
            # 获取文档字符串中的示例代码
            e = get_examples_from_docstring(docstr)
            # 如果存在示例代码
            if e:
                # 添加示例函数的定义及其示例代码到示例文件的代码行中
                example_file_lines.append(f"\n\ndef example_torch_{fname}():")
                example_file_lines += e

    # 遍历 torch.Tensor 类中的每个函数名
    for fname in dir(torch.Tensor):
        # 获取函数对象
        fn = getattr(torch.Tensor, fname)
        # 获取函数的文档字符串
        docstr = inspect.getdoc(fn)
        # 如果文档字符串存在且函数名不在 blocklist 中
        if docstr and fname not in blocklist:
            # 获取文档字符串中的示例代码
            e = get_examples_from_docstring(docstr)
            # 如果存在示例代码
            if e:
                # 添加示例函数的定义及其示例代码到示例文件的代码行中
                example_file_lines.append(f"\n\ndef example_torch_tensor_{fname}():")
                example_file_lines += e

    # 将示例文件的代码行列表连接成一个字符串返回
    return "\n".join(example_file_lines)

# 定义 TestTypeHints 类，用于测试类型提示
class TestTypeHints(TestCase):
    # 使用 unittest 的装饰器 @unittest.skipIf 根据条件跳过测试
    @unittest.skipIf(not HAVE_MYPY, "need mypy")
    def test_doc_examples(self):
        """
        Run documentation examples through mypy.
        """
        # 定义文件名 fn，指向当前文件的父目录下的特定文件
        fn = Path(__file__).resolve().parent / "generated_type_hints_smoketest.py"
        
        # 以写模式打开文件 fn，准备写入文档例子
        with open(fn, "w") as f:
            print(get_all_examples(), file=f)

        # OK, so here's the deal.  mypy treats installed packages
        # and local modules differently: if a package is installed,
        # mypy will refuse to use modules from that package for type
        # checking unless the module explicitly says that it supports
        # type checking. (Reference:
        # https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
        # )
        #
        # Now, PyTorch doesn't support typechecking, and we shouldn't
        # claim that it supports typechecking (it doesn't.) However, not
        # claiming we support typechecking is bad for this test, which
        # wants to use the partial information we get from the bits of
        # PyTorch which are typed to check if it typechecks.  And
        # although mypy will work directly if you are working in source,
        # some of our tests involve installing PyTorch and then running
        # its tests.
        #
        # The guidance we got from Michael Sullivan and Joshua Oreman,
        # and also independently developed by Thomas Viehmann,
        # is that we should create a fake directory and add symlinks for
        # the packages that should typecheck.  So that is what we do
        # here.
        #
        # If you want to run mypy by hand, and you run from PyTorch
        # root directory, it should work fine to skip this step (since
        # mypy will preferentially pick up the local files first).  The
        # temporary directory here is purely needed for CI.  For this
        # reason, we also still drop the generated file in the test
        # source folder, for ease of inspection when there are failures.
        
        # 使用临时目录 tmp_dir，创建符号链接以模拟需要进行类型检查的包
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 创建指向 PyTorch 所在目录的符号链接
                os.symlink(
                    os.path.dirname(torch.__file__),
                    os.path.join(tmp_dir, "torch"),
                    target_is_directory=True,
                )
            except OSError:
                # 如果创建符号链接失败，跳过当前测试
                raise unittest.SkipTest("cannot symlink") from None
            
            # 获取仓库根目录路径
            repo_rootdir = Path(__file__).resolve().parent.parent
            
            # 设置当前工作目录为仓库根目录（这会影响整个进程）
            with set_cwd(str(repo_rootdir)):
                # 运行 mypy 检查文档中生成的文件，使用特定选项
                (stdout, stderr, result) = mypy.api.run(
                    [
                        "--cache-dir=.mypy_cache/doc",
                        "--no-strict-optional",  # 需要因为 torch.lu_unpack，参见 gh-36584
                        str(fn),
                    ]
                )
            
            # 如果 mypy 返回非零结果，测试失败并输出错误信息
            if result != 0:
                self.fail(f"mypy failed:\n{stderr}\n{stdout}")
# 如果当前脚本被作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```