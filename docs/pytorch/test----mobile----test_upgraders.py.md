# `.\pytorch\test\mobile\test_upgraders.py`

```
# Owner(s): ["oncall: mobile"]

# 导入必要的模块和函数
import io
from itertools import product
from pathlib import Path

import torch
import torch.utils.bundled_inputs

# 导入用于移动端的模型加载和保存函数
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase

# 获取当前文件的父目录路径作为 PyTorch 测试目录
pytorch_test_dir = Path(__file__).resolve().parents[1]


# 定义测试类 TestLiteScriptModule，继承自 TestCase
class TestLiteScriptModule(TestCase):

    # 函数：保存并加载移动端模块
    def _save_load_mobile_module(self, script_module: torch.jit.ScriptModule):
        # 将脚本模块保存为字节流
        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        # 从字节流中加载为移动端模块
        mobile_module = _load_for_lite_interpreter(buffer)
        return mobile_module

    # 函数：尝试执行一个函数，并捕获可能的异常
    def _try_fn(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    # 测试函数：测试版本化的除法张量操作
    def test_versioned_div_tensor(self):
        # 函数：定义版本化的除法张量操作
        def div_tensor_0_3(self, other):
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode="trunc")

        # 设置模型路径为特定位置
        model_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / "upgrader_models"
            / "test_versioned_div_tensor_v2.ptl"
        )
        # 加载指定路径的移动端模块版本2
        mobile_module_v2 = _load_for_lite_interpreter(str(model_path))
        # 加载指定路径的 JIT 模块版本2
        jit_module_v2 = torch.jit.load(str(model_path))
        # 保存并加载 JIT 模块版本2 为当前的移动端模块
        current_mobile_module = self._save_load_mobile_module(jit_module_v2)
        # 定义测试值
        vals = (2.0, 3.0, 2, 3)
        # 对每对测试值进行迭代
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            # 辅助函数：对比两个函数对同一输入的执行结果
            def _helper(m, fn):
                # 使用移动端模块执行函数 m，并尝试捕获异常
                m_results = self._try_fn(m, a, b)
                # 使用原始函数执行 fn，并尝试捕获异常
                fn_result = self._try_fn(fn, a, b)

                # 如果移动端模块执行结果是异常，则断言原始函数执行结果也是异常
                if isinstance(m_results, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    # 否则，对比每个结果是否相等
                    for result in m_results:
                        print("result: ", result)
                        print("fn_result: ", fn_result)
                        print(result == fn_result)
                        self.assertTrue(result.eq(fn_result))
                        # self.assertEqual(result, fn_result)

            # 比较旧操作符产生的结果与 torch.div 操作升级后的结果是否相同
            # _helper(mobile_module_v2, div_tensor_0_3)
            # 比较最新操作符产生的结果与 torch.div 操作的结果是否相同
            # _helper(current_mobile_module, torch.div)


# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```