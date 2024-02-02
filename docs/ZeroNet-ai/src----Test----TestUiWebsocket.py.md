# `ZeroNet\src\Test\TestUiWebsocket.py`

```py
# 导入 sys 模块
import sys
# 导入 pytest 模块

import pytest

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
# 定义 TestUiWebsocket 类
class TestUiWebsocket:
    # 定义 testPermission 方法，传入 ui_websocket 参数
    def testPermission(self, ui_websocket):
        # 调用 ui_websocket 的 testAction 方法，传入参数 "ping"，并将结果赋给 res
        res = ui_websocket.testAction("ping")
        # 断言 res 等于 "pong"
        assert res == "pong"

        # 调用 ui_websocket 的 testAction 方法，传入参数 "certList"，并将结果赋给 res
        res = ui_websocket.testAction("certList")
        # 断言 res["error"] 中包含字符串 "You don't have permission"
        assert "You don't have permission" in res["error"]
```