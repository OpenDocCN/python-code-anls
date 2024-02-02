# `ZeroNet\src\User\UserManager.py`

```py
# 导入所需的模块
import json
import logging
import time

# 导入 ZeroNet 模块
from .User import User
from Plugin import PluginManager
from Config import config

# 声明 UserManager 类，并接受插件
@PluginManager.acceptPlugins
class UserManager(object):
    def __init__(self):
        self.users = {}  # 初始化用户字典
        self.log = logging.getLogger("UserManager")  # 获取日志记录器

    # 从 data/users.json 中加载所有用户
    def load(self):
        if not self.users:  # 如果用户字典为空
            self.users = {}  # 初始化用户字典

        user_found = []  # 存储找到的用户
        added = 0  # 记录添加的用户数量
        s = time.time()  # 记录开始时间
        # 加载新用户
        try:
            json_path = "%s/users.json" % config.data_dir  # 获取用户数据文件路径
            data = json.load(open(json_path))  # 加载用户数据
        except Exception as err:
            raise Exception("Unable to load %s: %s" % (json_path, err))  # 抛出加载异常

        for master_address, data in list(data.items()):  # 遍历用户数据
            if master_address not in self.users:  # 如果用户不在用户字典中
                user = User(master_address, data=data)  # 创建用户对象
                self.users[master_address] = user  # 将用户对象添加到用户字典
                added += 1  # 记录添加的用户数量
            user_found.append(master_address)  # 将找到的用户添加到列表中

        # 移除已删除的地址
        for master_address in list(self.users.keys()):  # 遍历用户字典中的地址
            if master_address not in user_found:  # 如果地址不在找到的用户列表中
                del(self.users[master_address])  # 从用户字典中删除用户
                self.log.debug("Removed user: %s" % master_address)  # 记录删除用户的日志

        if added:  # 如果有添加的用户
            self.log.debug("Added %s users in %.3fs" % (added, time.time() - s))  # 记录添加用户的日志

    # 创建新用户
    # 返回: User
    def create(self, master_address=None, master_seed=None):
        self.list()  # 如果尚未加载用户，则加载用户
        user = User(master_address, master_seed)  # 创建用户对象
        self.log.debug("Created user: %s" % user.master_address)  # 记录创建用户的日志
        if user.master_address:  # 如果成功创建用户
            self.users[user.master_address] = user  # 将用户添加到用户字典
            user.saveDelayed()  # 延迟保存用户数据
        return user  # 返回用户对象

    # 从 data/users.json 中列出所有用户
    # 返回: {"usermasteraddr": User}
    # 定义一个方法，用于返回用户列表
    def list(self):
        # 如果用户列表为空，表示还未加载，调用load方法加载用户列表
        if self.users == {}:  # Not loaded yet
            self.load()
        # 返回用户列表
        return self.users
    
    # 根据主地址获取用户
    # 返回：用户对象或空
    def get(self, master_address=None):
        # 获取用户列表
        users = self.list()
        # 如果用户列表不为空，返回第一个用户（单用户模式）
        if users:
            return list(users.values())[0]  # Single user mode, always return the first
        # 如果用户列表为空，返回空
        else:
            return None
# 创建一个用户管理器对象，这是一个单例对象，确保整个程序中只有一个实例
user_manager = UserManager()  # Singleton
```