# `.\pytorch\tools\jit\test\test_gen_unboxing.py`

```py
import tempfile  # 导入临时文件模块
import unittest  # 导入单元测试模块
from unittest.mock import NonCallableMock, patch  # 导入单元测试模块中的模拟对象和补丁功能

import tools.jit.gen_unboxing as gen_unboxing  # 导入待测试的模块 gen_unboxing


@patch("tools.jit.gen_unboxing.get_custom_build_selector")  # 使用 patch 装饰器模拟 get_custom_build_selector 函数
@patch("tools.jit.gen_unboxing.parse_native_yaml")  # 使用 patch 装饰器模拟 parse_native_yaml 函数
@patch("tools.jit.gen_unboxing.make_file_manager")  # 使用 patch 装饰器模拟 make_file_manager 函数
@patch("tools.jit.gen_unboxing.gen_unboxing")  # 使用 patch 装饰器模拟 gen_unboxing 函数
class TestGenUnboxing(unittest.TestCase):  # 定义单元测试类 TestGenUnboxing，继承自 unittest.TestCase

    def test_get_custom_build_selector_with_allowlist(
        self,
        mock_gen_unboxing: NonCallableMock,
        mock_make_file_manager: NonCallableMock,
        mock_parse_native_yaml: NonCallableMock,
        mock_get_custom_build_selector: NonCallableMock,
    ) -> None:
        # 准备测试参数
        args = ["--op-registration-allowlist=op1", "--op-selection-yaml-path=path2"]
        # 调用 gen_unboxing.main 函数
        gen_unboxing.main(args)
        # 断言 mock_get_custom_build_selector 被调用一次，带指定参数
        mock_get_custom_build_selector.assert_called_once_with(["op1"], "path2")

    def test_get_custom_build_selector_with_allowlist_yaml(
        self,
        mock_gen_unboxing: NonCallableMock,
        mock_make_file_manager: NonCallableMock,
        mock_parse_native_yaml: NonCallableMock,
        mock_get_custom_build_selector: NonCallableMock,
    ) -> None:
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile()
        # 向临时文件写入测试数据
        temp_file.write(b"- aten::add.Tensor")
        temp_file.seek(0)
        # 准备测试参数，使用临时文件路径
        args = [
            f"--TEST-ONLY-op-registration-allowlist-yaml-path={temp_file.name}",
            "--op-selection-yaml-path=path2",
        ]
        # 调用 gen_unboxing.main 函数
        gen_unboxing.main(args)
        # 断言 mock_get_custom_build_selector 被调用一次，带指定参数
        mock_get_custom_build_selector.assert_called_once_with(
            ["aten::add.Tensor"], "path2"
        )
        # 关闭临时文件
        temp_file.close()

    def test_get_custom_build_selector_with_both_allowlist_and_yaml(
        self,
        mock_gen_unboxing: NonCallableMock,
        mock_make_file_manager: NonCallableMock,
        mock_parse_native_yaml: NonCallableMock,
        mock_get_custom_build_selector: NonCallableMock,
    ) -> None:
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile()
        # 向临时文件写入测试数据
        temp_file.write(b"- aten::add.Tensor")
        temp_file.seek(0)
        # 准备测试参数，同时使用 allowlist 和 yaml 文件路径
        args = [
            "--op-registration-allowlist=op1",
            f"--TEST-ONLY-op-registration-allowlist-yaml-path={temp_file.name}",
            "--op-selection-yaml-path=path2",
        ]
        # 调用 gen_unboxing.main 函数
        gen_unboxing.main(args)
        # 断言 mock_get_custom_build_selector 被调用一次，带指定参数
        mock_get_custom_build_selector.assert_called_once_with(["op1"], "path2")
        # 关闭临时文件
        temp_file.close()


if __name__ == "__main__":
    unittest.main()  # 运行单元测试
```