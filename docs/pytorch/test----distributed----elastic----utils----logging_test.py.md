# `.\pytorch\test\distributed\elastic\utils\logging_test.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入日志模块中的 logging
import torch.distributed.elastic.utils.logging as logging
# 导入测试相关的工具和类
from torch.testing._internal.common_utils import run_tests, TestCase

# 获取根日志记录器
log = logging.get_logger()

# 定义日志测试类 LoggingTest，继承自 TestCase
class LoggingTest(TestCase):
    # 设置测试环境
    def setUp(self):
        super().setUp()
        # 获取类级别的日志记录器
        self.clazz_log = logging.get_logger()

    # 测试日志记录器的名称
    def test_logger_name(self):
        # 获取本地日志记录器
        local_log = logging.get_logger()
        # 获取指定名称的日志记录器
        name_override_log = logging.get_logger("foobar")

        # 断言日志记录器的名称与当前模块的名称相同
        self.assertEqual(__name__, log.name)
        self.assertEqual(__name__, self.clazz_log.name)
        self.assertEqual(__name__, local_log.name)
        # 断言指定名称的日志记录器名称为 "foobar"
        self.assertEqual("foobar", name_override_log.name)

    # 测试根据深度获取模块名称的方法
    def test_derive_module_name(self):
        # 调用内部方法获取模块名称
        module_name = logging._derive_module_name(depth=1)
        # 断言获取的模块名称与当前模块的名称相同
        self.assertEqual(__name__, module_name)

# 当脚本作为主程序运行时，执行测试
if __name__ == "__main__":
    run_tests()
```