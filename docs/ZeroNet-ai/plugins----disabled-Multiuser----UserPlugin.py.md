# `ZeroNet\plugins\disabled-Multiuser\UserPlugin.py`

```
# 从Config模块中导入config变量
from Config import config
# 从Plugin模块中导入PluginManager类
from Plugin import PluginManager

# 设置allow_reload变量为False
allow_reload = False

# 将UserManagerPlugin类注册到PluginManager中
@PluginManager.registerTo("UserManager")
class UserManagerPlugin(object):
    # 加载用户数据
    def load(self):
        # 如果不是多用户模式，则不加载用户数据
        if not config.multiuser_local:
            if not self.users:
                self.users = {}
            return self.users
        else:
            return super(UserManagerPlugin, self).load()

    # 通过主地址查找用户
    # 返回：User对象或None
    def get(self, master_address=None):
        # 获取用户列表
        users = self.list()
        # 如果主地址在用户列表中，则返回对应的用户对象，否则返回None
        if master_address in users:
            user = users[master_address]
        else:
            user = None
        return user

# 将UserPlugin类注册到PluginManager中
@PluginManager.registerTo("User")
class UserPlugin(object):
    # 在多用户模式下，用户数据仅存在于内存中，不写入data/user.json文件
    def save(self):
        if not config.multiuser_local:
            return False
        else:
            return super(UserPlugin, self).save()
```