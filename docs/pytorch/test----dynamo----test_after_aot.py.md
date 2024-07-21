# `.\pytorch\test\dynamo\test_after_aot.py`

```
# Owner(s): ["module: dynamo"]

import io  # 导入 io 模块，提供了各种 I/O 操作功能
import os  # 导入 os 模块，提供了操作系统相关的功能
import shutil  # 导入 shutil 模块，提供了高级文件操作功能
import sys  # 导入 sys 模块，提供了与 Python 解释器相关的功能
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch._dynamo.test_case  # 导入 torch._dynamo.test_case 模块中的测试用例

from torch._dynamo.repro.after_aot import InputReader, InputWriter, save_graph_repro  # 导入保存图重现所需的相关函数

from torch.fx.experimental.proxy_tensor import make_fx  # 导入创建代理张量的函数
from torch.testing._internal.common_utils import IS_FBCODE  # 导入是否为 FBCODE 环境的标志
from torch.utils._traceback import report_compile_source_on_error  # 导入用于在错误发生时报告编译源代码的函数


def strip_trailing_whitespace(r):
    return "\n".join([l.rstrip() for l in r.split("\n")])  # 去除字符串 r 中每行末尾的空白字符


class TestAfterAot(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(IS_FBCODE, "NotImplementedError")
    def test_save_graph_repro(self):
        # TODO: This triggers CUDA context initialization, even though
        # it is CPU only
        buf = io.StringIO()  # 创建一个内存缓冲区对象
        args = [torch.randn(4)]  # 创建一个包含随机张量的列表

        def f(x):
            return (x * x,)  # 定义一个函数 f，返回输入张量的平方

        gm = make_fx(f)(*args)  # 使用 make_fx 创建函数 f 的图模式表示
        with tempfile.TemporaryDirectory() as d:  # 创建一个临时目录 d
            save_graph_repro(buf, gm, args, "inductor_accuracy", save_dir=d)  # 将图的重现信息保存到 buf 中，并将结果存储在临时目录 d 中
            r = buf.getvalue()  # 获取 buf 中的内容作为字符串
            with report_compile_source_on_error():  # 在发生错误时报告编译源代码
                exec(r, {"__compile_source__": r})  # 执行字符串 r 中的代码，并传入额外的环境变量 "__compile_source__"

            shutil.rmtree(os.path.join(d, "storages"))  # 递归删除临时目录 d 中名为 "storages" 的子目录

            # 即使没有保存目录，应该仍然可以正常工作
            with report_compile_source_on_error():  # 再次在发生错误时报告编译源代码
                exec(r, {"__compile_source__": r})  # 再次执行字符串 r 中的代码，并传入额外的环境变量 "__compile_source__"

    @unittest.skipIf(sys.byteorder != "little", "checksum depends on endianness")
    def test_dump_tensor(self):
        def test(tensor, expected):
            with tempfile.TemporaryDirectory() as d:  # 创建一个临时目录 d
                writer = InputWriter(d, stable_hash=True)  # 创建一个输入写入器对象，使用稳定哈希
                writer.tensor("x", tensor)  # 将张量 tensor 写入到写入器中，并命名为 "x"
                self.assertExpectedInline("\n".join(writer._lines), expected, skip=1)  # 断言写入器中的行与预期的内容 expected 匹配，跳过第一行

                reader = InputReader(d)  # 创建一个输入读取器对象，用于读取临时目录 d 中的数据
                env = {"reader": reader, "torch": torch}  # 定义一个环境字典，包含 reader 对象和 torch 模块

                exec("\n".join(writer._lines), env)  # 在环境 env 中执行写入器中的行
                self.assertEqual(reader.args[0], tensor)  # 断言读取器中的第一个参数与张量 tensor 相等

        test(
            torch.zeros(3, 4),
            """\
buf0 = reader.storage('c17fd92682ca5b304ac71074b558dda9e8eb4d66', 48)
reader.tensor(buf0, (3, 4), is_leaf=True)  # x""",
        )
        test(
            torch.ones(3, 4, dtype=torch.int32),
            """\
buf0 = reader.storage('7c221e2da0c58c700cc2996644dd13d042bd552e', 48, dtype_hint=torch.int32)
reader.tensor(buf0, (3, 4), dtype=torch.int32, is_leaf=True)  # x""",
        )
        test(
            torch.empty((3, 4, 5, 6), memory_format=torch.channels_last).fill_(2),
            """\
buf0 = reader.storage('49ebab3961d6221e64c4c72b0aefd976bdd2afc4', 1440)
reader.tensor(buf0, (3, 4, 5, 6), (120, 1, 24, 4), is_leaf=True)  # x""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()  # 运行测试用例
```