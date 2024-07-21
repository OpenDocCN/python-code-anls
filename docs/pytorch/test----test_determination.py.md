# `.\pytorch\test\test_determination.py`

```
# Owner(s): ["module: ci"]  # 指定代码所有者为 CI 模块

import os  # 导入操作系统相关模块

import run_test  # 导入自定义的运行测试模块

from torch.testing._internal.common_utils import run_tests, TestCase  # 从 PyTorch 测试工具包中导入运行测试和测试用例类


class DummyOptions:
    verbose = False  # 创建一个虚拟选项类，设置默认的详细输出为假


class DeterminationTest(TestCase):
    # 在一个测试子集上进行确定性测试
    TESTS = [
        "test_nn",
        "test_jit_profiling",
        "test_jit",
        "test_torch",
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "test_utils",
        "test_determination",
        "test_quantization",
    ]

    @classmethod
    def determined_tests(cls, changed_files):
        # 标准化所有变更文件的路径名
        changed_files = [os.path.normpath(path) for path in changed_files]
        # 返回在给定变更文件上应该运行的测试列表
        return [
            test
            for test in cls.TESTS
            if run_test.should_run_test(
                run_test.TARGET_DET_LIST, test, changed_files, DummyOptions()
            )
        ]

    def test_target_det_list_is_sorted(self):
        # 我们保持 TARGET_DET_LIST 排序以最小化合并冲突，
        # 更重要的是可以通过注释指出测试的缺失。在列表旁边添加文件会变得非常困难。
        self.assertListEqual(run_test.TARGET_DET_LIST, sorted(run_test.TARGET_DET_LIST))

    def test_config_change_only(self):
        """CI configs trigger all tests"""
        # 测试仅配置更改是否触发所有测试
        self.assertEqual(self.determined_tests([".ci/pytorch/test.sh"]), self.TESTS)

    def test_run_test(self):
        """run_test.py is imported by determination tests"""
        # 测试运行测试是否由确定性测试导入
        self.assertEqual(
            self.determined_tests(["test/run_test.py"]), ["test_determination"]
        )

    def test_non_code_change(self):
        """Non-code changes don't trigger any tests"""
        # 测试非代码更改是否不会触发任何测试
        self.assertEqual(
            self.determined_tests(["CODEOWNERS", "README.md", "docs/doc.md"]), []
        )

    def test_cpp_file(self):
        """CPP files trigger all tests"""
        # 测试 CPP 文件是否触发所有测试
        self.assertEqual(
            self.determined_tests(["aten/src/ATen/native/cpu/Activation.cpp"]),
            self.TESTS,
        )

    def test_test_file(self):
        """Test files trigger themselves and dependent tests"""
        # 测试文件触发自身及其依赖的测试
        self.assertEqual(
            self.determined_tests(["test/test_jit.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(["test/jit/test_custom_operators.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(
                ["test/quantization/eager/test_quantize_eager_ptq.py"]
            ),
            ["test_quantization"],
        )
    # 测试函数，用于测试特定文件触发的依赖测试
    def test_test_internal_file(self):
        """testing/_internal files trigger dependent tests"""
        # 断言调用 self.determined_tests 方法，检查特定文件是否触发了期望的依赖测试
        self.assertEqual(
            self.determined_tests(["torch/testing/_internal/common_quantization.py"]),
            [
                "test_jit_profiling",
                "test_jit",
                "test_quantization",
            ],
        )

    # 测试函数，用于测试 Torch 文件触发的依赖测试
    def test_torch_file(self):
        """Torch files trigger dependent tests"""
        # 断言调用 self.determined_tests 方法，检查特定文件是否触发了期望的依赖测试
        self.assertEqual(
            # 由于项目布局的原因，许多文件被强制导入所有测试
            self.determined_tests(["torch/onnx/utils.py"]),
            self.TESTS,
        )
        # 断言调用 self.determined_tests 方法，检查多个文件是否触发了期望的依赖测试
        self.assertEqual(
            self.determined_tests(
                [
                    "torch/autograd/_functions/utils.py",
                    "torch/autograd/_functions/utils.pyi",
                ]
            ),
            ["test_utils"],
        )
        # 断言调用 self.determined_tests 方法，检查特定文件是否触发了期望的依赖测试
        self.assertEqual(
            self.determined_tests(["torch/utils/cpp_extension.py"]),
            [
                "test_cpp_extensions_aot_ninja",
                "test_cpp_extensions_aot_no_ninja",
                "test_utils",
                "test_determination",
            ],
        )

    # 测试函数，用于测试新的顶级 Python 文件夹触发的所有测试
    def test_new_folder(self):
        """New top-level Python folder triggers all tests"""
        # 断言调用 self.determined_tests 方法，检查特定文件是否触发了所有测试
        self.assertEqual(self.determined_tests(["new_module/file.py"]), self.TESTS)

    # 测试函数，用于测试新的测试脚本不触发任何测试（因为不在 run_tests.py 中）
    def test_new_test_script(self):
        """New test script triggers nothing (since it's not in run_tests.py)"""
        # 断言调用 self.determined_tests 方法，检查特定文件是否触发了期望的依赖测试（预期为空列表）
        self.assertEqual(self.determined_tests(["test/test_new_test_script.py"]), [])
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```