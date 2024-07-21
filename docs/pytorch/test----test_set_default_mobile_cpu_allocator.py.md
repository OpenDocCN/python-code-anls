# `.\pytorch\test\test_set_default_mobile_cpu_allocator.py`

```
#`
# Owner(s): ["oncall: mobile"]

# 导入PyTorch库及相关测试工具
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

# 定义测试类，测试移动端CPU分配器的默认设置
class TestSetDefaultMobileCPUAllocator(TestCase):
    
    # 测试默认设置不会引发异常
    def test_no_exception(self):
        # 设置默认移动端CPU分配器
        torch._C._set_default_mobile_cpu_allocator()
        # 取消默认移动端CPU分配器设置
        torch._C._unset_default_mobile_cpu_allocator()

    # 测试设置过程中可能引发异常
    def test_exception(self):
        # 验证取消默认移动端CPU分配器设置时是否引发异常
        with self.assertRaises(Exception):
            torch._C._unset_default_mobile_cpu_allocator()

        # 验证设置默认移动端CPU分配器时重复调用是否引发异常
        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._set_default_mobile_cpu_allocator()

        # 必须重置为正常状态
        # 以进行下一个测试
        torch._C._unset_default_mobile_cpu_allocator()

        # 验证设置和取消默认移动端CPU分配器时可能引发的异常set_default_mobile_cpu_allocator()

        # 测试在设置和取消设置默认移动CPU分配器之间多次取消设置是否引发异常
        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()

# 如果脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    run_tests()
```