# `.\pytorch\test\test_show_pickle.py`

```
# Owner(s): ["oncall: mobile"]

# 导入所需的模块
import io  # 导入用于处理文件流的模块
import tempfile  # 导入用于创建临时文件的模块
import unittest  # 导入用于编写和运行测试的模块

import torch  # 导入PyTorch深度学习框架
import torch.utils.show_pickle  # 导入用于显示pickle内容的工具函数

from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase  # 导入测试相关的工具和函数


class TestShowPickle(TestCase):
    @unittest.skipIf(IS_WINDOWS, "Can't re-open temp file on Windows")
    def test_scripted_model(self):
        # 定义一个简单的PyTorch模块
        class MyCoolModule(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return x * self.weight

        # 使用torch.jit.script方法将模块脚本化
        m = torch.jit.script(MyCoolModule(torch.tensor([2.0])))

        # 使用tempfile.NamedTemporaryFile创建一个临时文件，并使用torch.jit.save保存模型
        with tempfile.NamedTemporaryFile() as tmp:
            torch.jit.save(m, tmp)
            tmp.flush()  # 刷新文件缓冲区，确保内容已写入临时文件

            # 创建一个StringIO对象作为输出流缓冲区
            buf = io.StringIO()

            # 使用torch.utils.show_pickle.main显示保存的pickle文件内容
            torch.utils.show_pickle.main(
                ["", tmp.name + "@*/data.pkl"], output_stream=buf
            )

            output = buf.getvalue()  # 从StringIO对象中获取输出的字符串内容

            # 断言输出内容中包含模块的名称和权重信息
            self.assertRegex(output, "MyCoolModule")
            self.assertRegex(output, "weight")


if __name__ == "__main__":
    run_tests()  # 运行所有的测试用例
```