# `.\pytorch\test\lazy\test_debug_util.py`

```
# Owner(s): ["oncall: jit"]

# 引入标准库和模块
import os
import re
import tempfile
import unittest

# 引入 Torch 的内部模块和类
import torch._lazy
import torch._lazy.ts_backend
import torch.nn as nn
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase

# 初始化 Torch 懒加载模块的时间序列后端
torch._lazy.ts_backend.init()

# 标记为 Windows 系统下的测试需要跳过
@unittest.skipIf(IS_WINDOWS, "To be fixed")
class DebugUtilTest(TestCase):
    # 运行线性模型测试
    def _run_linear(self):
        # 设备选择为"lazy"
        device = "lazy"
        # 创建并移到指定设备的线性模型
        model = nn.Linear(5, 5).to(device)
        # 在模型上执行随机输入，生成输出
        output = model(torch.randn(1, 5).to(device))
        # 在 Torch 懒加载模块中标记当前步骤
        torch._lazy.mark_step()

    # 测试获取 Python 堆栈帧
    def test_get_python_frames(self):
        # 我们只关心保存的图中第一个"Python Stacktrace"部分。
        # 但是，由于它依赖于很多东西，我们无法保存整个堆栈以进行比较。
        partial_graph = (
            r"Python Stacktrace:.*"
            r"mark_step \(.*/_lazy/__init__.py:[0-9]+\).*"
            r"_run_linear \(.*lazy/test_debug_util.py:[0-9]+\).*"
            r"test_get_python_frames \(.*lazy/test_debug_util.py:[0-9]+\)"
        )

        # 使用临时命名文件作为保存张量的文件
        with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8") as graph_file:
            # 设置环境变量以指定保存文件的路径
            os.environ["LTC_SAVE_TENSORS_FILE"] = graph_file.name
            # 执行线性模型运行
            self._run_linear()
            # 读取保存的图文件内容
            file = graph_file.read()
            # 如果无法在文件中找到部分图形的匹配，则输出文件内容并断言失败
            if re.search(partial_graph, file, re.DOTALL) is None:
                print(file)
                self.assertTrue(False)

# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```