# `.\PaddleOCR\PPOCRLabel\libs\settings.py`

```
# 版权声明，允许任何人免费获取和使用该软件，但需包含版权声明和许可声明
# 导入所需的模块
import pickle
import os
import sys

# 定义 Settings 类
class Settings(object):
    # 初始化方法
    def __init__(self):
        # 默认情况下，home 路径为 labelImg 所在文件夹
        home = os.path.expanduser("~")
        self.data = {}
        # 设置保存设置的文件路径
        # self.path = os.path.join(home, '.labelImgSettings.pkl')
        self.path = os.path.join(home, '.autoOCRSettings.pkl')

    # 设置键值对
    def __setitem__(self, key, value):
        self.data[key] = value

    # 获取键对应的值
    def __getitem__(self, key):
        return self.data[key]

    # 获取键对应的值，如果不存在则返回默认值
    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default
    # 保存数据到指定路径的 pickle 文件中
    def save(self):
        # 如果存在路径
        if self.path:
            # 以二进制写模式打开文件
            with open(self.path, 'wb') as f:
                # 使用 pickle 序列化数据到文件中，使用最高协议
                pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
                # 返回 True 表示保存成功
                return True
        # 返回 False 表示保存失败
        return False

    # 从指定路径的 pickle 文件中加载数据
    def load(self):
        try:
            # 如果路径存在
            if os.path.exists(self.path):
                # 以二进制读模式打开文件
                with open(self.path, 'rb') as f:
                    # 从文件中反序列化数据到 self.data
                    self.data = pickle.load(f)
                    # 返回 True 表示加载成功
                    return True
        except:
            # 加载失败时打印错误信息
            print('Loading setting failed')
        # 返回 False 表示加载失败
        return False

    # 重置数据和路径
    def reset(self):
        # 如果路径存在
        if os.path.exists(self.path):
            # 删除指定路径的文件
            os.remove(self.path)
            # 打印提示信息
            print('Remove setting pkl file ${0}'.format(self.path))
        # 重置数据为空字典
        self.data = {}
        # 重置路径为 None
        self.path = None
```