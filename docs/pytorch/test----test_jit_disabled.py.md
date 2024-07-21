# `.\pytorch\test\test_jit_disabled.py`

```
# Owner(s): ["oncall: jit"]

import sys  # 导入系统模块
import os   # 导入操作系统接口模块
import contextlib  # 上下文管理模块，用于创建上下文管理器
import subprocess  # 子进程管理模块
from torch.testing._internal.common_utils import TestCase, run_tests, TemporaryFileName  # 导入测试框架和临时文件名生成器


@contextlib.contextmanager
def _jit_disabled():
    cur_env = os.environ.get("PYTORCH_JIT", "1")  # 获取当前环境中的 PYTORCH_JIT 变量，若不存在则默认为 "1"
    os.environ["PYTORCH_JIT"] = "0"  # 设置 PYTORCH_JIT 环境变量为 "0"，禁用 JIT
    try:
        yield  # 执行 yield 之前的代码作为上下文管理器的进入操作
    finally:
        os.environ["PYTORCH_JIT"] = cur_env  # 恢复 PYTORCH_JIT 变量为之前的值


class TestJitDisabled(TestCase):
    """
    These tests are separate from the rest of the JIT tests because we need
    run a new subprocess and `import torch` with the correct environment
    variables set.
    """

    def compare_enabled_disabled(self, src):
        """
        Runs the script in `src` with PYTORCH_JIT enabled and disabled and
        compares their stdout for equality.
        """
        # Write `src` out to a temporary so our source inspection logic works
        # correctly.
        with TemporaryFileName() as fname:  # 使用临时文件名 fname
            with open(fname, 'w') as f:  # 打开临时文件 fname 进行写操作
                f.write(src)  # 将源码写入临时文件
                with _jit_disabled():  # 禁用 JIT 环境
                    out_disabled = subprocess.check_output([  # 运行子进程，获取禁用 JIT 后的输出结果
                        sys.executable,
                        fname])
                out_enabled = subprocess.check_output([  # 运行子进程，获取启用 JIT 后的输出结果
                    sys.executable,
                    fname])
                self.assertEqual(out_disabled, out_enabled)  # 断言禁用 JIT 和启用 JIT 后的输出结果是否相等

    def test_attribute(self):
        _program_string = """
import torch

class Foo(torch.jit.ScriptModule):
    def __init__(self, x):
        super().__init__()
        self.x = torch.jit.Attribute(x, torch.Tensor)

    def forward(self, input):
        return input

s = Foo(torch.ones(2, 3))
print(s.x)
"""
        self.compare_enabled_disabled(_program_string)

    def test_script_module_construction(self):
        _program_string = """
import torch

class AModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, input):
        pass

AModule()
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

    def test_recursive_script(self):
        _program_string = """
import torch

class AModule(torch.nn.Module):
    def forward(self, input):
        pass

sm = torch.jit.script(AModule())
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

if __name__ == '__main__':
    run_tests()  # 执行测试运行
```