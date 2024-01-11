# `ZeroNet\plugins\disabled-Multiuser\Test\TestMultiuser.py`

```
# 导入 pytest 模块
import pytest
# 导入 json 模块
import json
# 从 Config 模块中导入 config 对象
from Config import config
# 从 User 模块中导入 UserManager 类
from User import UserManager

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
# 使用 resetTempSettings 修饰器来重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
# 定义 TestMultiuser 类
class TestMultiuser:
    # 定义 testMemorySave 方法，接受 user 参数
    def testMemorySave(self, user):
        # 应该不会将用户写入磁盘
        # 读取修改前的用户数据
        users_before = open("%s/users.json" % config.data_dir).read()
        # 创建一个新用户
        user = UserManager.user_manager.create()
        # 保存用户数据
        user.save()
        # 断言修改后的用户数据与修改前的用户数据相同
        assert open("%s/users.json" % config.data_dir).read() == users_before
```