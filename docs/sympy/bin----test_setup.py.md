# `D:\src\scipysrc\sympy\bin\test_setup.py`

```
#!/usr/bin/env python
"""
Test that the installed modules in setup.py are up-to-date.

If this test fails, run

python bin/generate_test_list.py

and

python bin/generate_module_list.py

to generate the up-to-date test and modules list to put in setup.py.

"""

# 导入用于生成测试列表和模块列表的脚本
import generate_test_list
import generate_module_list

# 导入路径修正函数
from get_sympy import path_hack

# 执行路径修正，确保导入的模块路径正确
path_hack()

# 导入setup.py中的设置
import setup

# 生成当前的模块列表和测试列表
module_list = generate_module_list.generate_module_list()
test_list = generate_test_list.generate_test_list()

# 检查当前生成的模块列表是否与setup.py中的模块列表匹配
assert setup.modules == module_list, set(setup.modules).symmetric_difference(set(module_list))

# 检查当前生成的测试列表是否与setup.py中的测试列表匹配
assert setup.tests == test_list, set(setup.tests).symmetric_difference(set(test_list))

# 打印结果确认所有模块和测试列表与setup.py中的一致
print("setup.py modules and tests are OK")
```