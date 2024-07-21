# `.\pytorch\test\mobile\test_upgrader_codegen.py`

```py
# Owner(s): ["oncall: mobile"]

import os  # 导入操作系统相关的模块
import tempfile  # 导入处理临时文件和目录的模块
from pathlib import Path  # 导入处理路径的模块

from torch.jit.generate_bytecode import generate_upgraders_bytecode  # 导入生成字节码的函数
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关的模块和类

from torchgen.operator_versions.gen_mobile_upgraders import sort_upgrader, write_cpp  # 导入排序和写入 C++ 文件的函数

pytorch_caffe2_dir = Path(__file__).resolve().parents[2]  # 获取当前文件的父目录的父目录，即 PyTorch Caffe2 的根目录


class TestLiteScriptModule(TestCase):
    def test_generate_bytecode(self):
        upgrader_list = generate_upgraders_bytecode()  # 调用生成字节码的函数，获取升级器列表
        sorted_upgrader_list = sort_upgrader(upgrader_list)  # 对升级器列表进行排序
        upgrader_mobile_cpp_path = (
            pytorch_caffe2_dir
            / "torch"
            / "csrc"
            / "jit"
            / "mobile"
            / "upgrader_mobile.cpp"
        )  # 设置 upgrader_mobile.cpp 文件的路径

        with tempfile.TemporaryDirectory() as tmpdirname:  # 创建一个临时目录，使用 with 语句可以自动清理
            write_cpp(tmpdirname, sorted_upgrader_list)  # 调用函数将排序后的升级器列表写入临时目录中的 C++ 文件
            with open(os.path.join(tmpdirname, "upgrader_mobile.cpp")) as file_name:
                actual_output = [line.strip() for line in file_name if line]  # 读取临时目录中生成的 C++ 文件的内容，并去除空行
            with open(str(upgrader_mobile_cpp_path)) as file_name:
                expect_output = [line.strip() for line in file_name if line]  # 读取预期的 upgrader_mobile.cpp 文件的内容，并去除空行

            actual_output_filtered = list(
                filter(lambda token: len(token) != 0, actual_output)
            )  # 过滤实际输出中的空行
            expect_output_filtered = list(
                filter(lambda token: len(token) != 0, expect_output)
            )  # 过滤预期输出中的空行

            self.assertEqual(actual_output_filtered, expect_output_filtered)  # 断言实际输出和预期输出相等


if __name__ == "__main__":
    run_tests()  # 运行测试用例
```