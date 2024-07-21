# `.\pytorch\test\onnx\test_symbolic_helper.py`

```py
# Owner(s): ["module: onnx"]
"""Unit tests on `torch.onnx.symbolic_helper`."""

import torch  # 导入 torch 库
from torch.onnx import symbolic_helper  # 导入 torch.onnx.symbolic_helper 模块
from torch.onnx._globals import GLOBALS  # 导入 torch.onnx._globals 模块中的 GLOBALS 对象
from torch.testing._internal import common_utils  # 导入 torch.testing._internal 中的 common_utils 模块


class TestHelperFunctions(common_utils.TestCase):
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        self._initial_training_mode = GLOBALS.training_mode  # 记录初始的 GLOBALS.training_mode 值

    def tearDown(self):
        GLOBALS.training_mode = self._initial_training_mode  # 在测试结束时恢复 GLOBALS.training_mode 到初始值

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.PRESERVE], name="export_mode_is_preserve"
            ),
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.EVAL],
                name="modes_match_op_train_mode_0_export_mode_eval",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.TRAINING],
                name="modes_match_op_train_mode_1_export_mode_training",
            ),
        ],
    )
    def test_check_training_mode_does_not_warn_when(
        self, op_train_mode: int, export_mode: torch.onnx.TrainingMode
    ):
        GLOBALS.training_mode = export_mode  # 设置 GLOBALS.training_mode 为 export_mode
        self.assertNotWarn(
            lambda: symbolic_helper.check_training_mode(op_train_mode, "testop")
        )  # 测试调用 symbolic_helper.check_training_mode 是否不会触发警告

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.TRAINING],
                name="modes_do_not_match_op_train_mode_0_export_mode_training",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.EVAL],
                name="modes_do_not_match_op_train_mode_1_export_mode_eval",
            ),
        ],
    )
    def test_check_training_mode_warns_when(
        self,
        op_train_mode: int,
        export_mode: torch.onnx.TrainingMode,
    ):
        with self.assertWarnsRegex(
            UserWarning, f"ONNX export mode is set to {export_mode}"
        ):
            GLOBALS.training_mode = export_mode  # 设置 GLOBALS.training_mode 为 export_mode
            symbolic_helper.check_training_mode(op_train_mode, "testop")  # 调用 symbolic_helper.check_training_mode，预期会触发警告


common_utils.instantiate_parametrized_tests(TestHelperFunctions)  # 实例化参数化测试

if __name__ == "__main__":
    common_utils.run_tests()  # 运行测试
```