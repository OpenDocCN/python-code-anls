# `.\pytorch\test\jit\test_fuser_common.py`

```py
# Owner(s): ["oncall: jit"]

import torch  # 导入 PyTorch 库
from torch.testing._internal.jit_utils import JitTestCase  # 导入测试工具类 JitTestCase


class TestFuserCommon(JitTestCase):
    def test_autodiff_fallback(self):
        for rq in [True, False]:  # 针对不同的 requires_grad 值进行循环测试

            @torch.jit.script
            def fn(x):
                return torch.max(x**2.0, x**3.0)  # 定义一个脚本化的函数，返回 x^2 和 x^3 中的较大值

            x = torch.randn(5, requires_grad=not rq)  # 根据 rq 的值生成一个张量 x
            # 引发优化的创建
            for i in range(5):  # 循环多次执行 fn(x)，以便优化被创建
                fn(x)
            # 当无法应用优化时测试回退情况
            y = fn(torch.randn(5, requires_grad=rq))  # 对另一个张量应用 fn 函数
            self.assertEqual(y.requires_grad, rq)  # 断言 y 的 requires_grad 是否等于 rq
```