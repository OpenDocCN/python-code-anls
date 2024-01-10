# `MetaGPT\tests\metagpt\utils\test_session.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器来执行脚本
# _*_ coding: utf-8 _*_
# 指定文件编码格式为 UTF-8

import pytest
# 导入 pytest 模块

def test_nodeid(request):
    # 打印测试用例的节点 ID
    print(request.node.nodeid)
    # 断言节点 ID 存在
    assert request.node.nodeid

if __name__ == "__main__":
    # 执行 pytest 主程序，并传入当前文件名和 "-s" 参数
    pytest.main([__file__, "-s"])

```