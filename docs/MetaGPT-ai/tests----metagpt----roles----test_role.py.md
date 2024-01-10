# `MetaGPT\tests\metagpt\roles\test_role.py`

```

#!/usr/bin/env python
# 指定解释器为 python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
# @Desc   : unittest of Role
# 描述：Role 类的单元测试

import pytest
# 导入 pytest 模块

from metagpt.roles.role import Role
# 从 metagpt.roles.role 模块中导入 Role 类


def test_role_desc():
    # 测试 Role 类的 desc 属性
    role = Role(profile="Sales", desc="Best Seller")
    # 创建一个 Role 对象，profile 属性为 "Sales"，desc 属性为 "Best Seller"
    assert role.profile == "Sales"
    # 断言 role 的 profile 属性为 "Sales"
    assert role.desc == "Best Seller"
    # 断言 role 的 desc 属性为 "Best Seller"

if __name__ == "__main__":
    # 如果当前模块被直接执行
    pytest.main([__file__, "-s"])
    # 运行 pytest 进行单元测试，-s 参数表示输出所有的 print 语句

```