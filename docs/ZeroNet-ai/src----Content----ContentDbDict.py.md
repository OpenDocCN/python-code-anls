# `ZeroNet\src\Content\ContentDbDict.py`

```py
# 导入时间和操作系统模块
import time
import os

# 从当前目录下的 ContentDb 模块中导入 ContentDb 类
from . import ContentDb
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 Config 模块中导入 config 变量
from Config import config

# 创建 ContentDbDict 类，继承自 dict 类
class ContentDbDict(dict):
    # 初始化方法，接受 site 参数以及任意位置参数和关键字参数
    def __init__(self, site, *args, **kwargs):
        # 记录初始化开始时间
        s = time.time()
        # 将 site 参数赋值给 self.site
        self.site = site
        # 初始化缓存键列表
        self.cached_keys = []
        # 将 site 的日志记录器赋值给 self.log
        self.log = self.site.log
        # 获取 ContentDb 实例并赋值给 self.db
        self.db = ContentDb.getContentDb()
        # 获取当前站点的数据库 ID 并赋值给 self.db_id
        self.db_id = self.db.needSite(site)
        # 初始化已加载的条目数量
        self.num_loaded = 0
        # 调用父类 dict 的初始化方法，并将从数据库加载键的结果作为参数传入
        super(ContentDbDict, self).__init__(self.db.loadDbDict(site))  # Load keys from database
        # 记录初始化结束时间，并打印日志
        self.log.debug("ContentDb init: %.3fs, found files: %s, sites: %s" % (time.time() - s, len(self), len(self.db.site_ids)))

    # 加载指定键对应的条目
    def loadItem(self, key):
        try:
            # 增加已加载条目的数量
            self.num_loaded += 1
            # 每加载 100 个条目打印调试信息
            if self.num_loaded % 100 == 0:
                if config.verbose:
                    self.log.debug("Loaded json: %s (latest: %s) called by: %s" % (self.num_loaded, key, Debug.formatStack()))
                else:
                    self.log.debug("Loaded json: %s (latest: %s)" % (self.num_loaded, key))
            # 从站点存储中加载指定键对应的 JSON 数据
            content = self.site.storage.loadJson(key)
            # 将加载的内容添加到字典中
            dict.__setitem__(self, key, content)
        except IOError:
            # 如果发生 IOError，则删除对应的键值对并抛出 KeyError
            if dict.get(self, key):
                self.__delitem__(key)  # File not exists anymore
            raise KeyError(key)

        # 将键添加到缓存键列表中，并检查是否超出限制
        self.addCachedKey(key)
        self.checkLimit()

        return content

    # 获取指定键对应的条目大小
    def getItemSize(self, key):
        return self.site.storage.getSize(key)

    # 仅保留内存中最近访问的 10 个 JSON 条目
    def checkLimit(self):
        if len(self.cached_keys) > 10:
            key_deleted = self.cached_keys.pop(0)
            dict.__setitem__(self, key_deleted, False)

    # 添加已缓存的键
    def addCachedKey(self, key):
        if key not in self.cached_keys and key != "content.json" and len(key) > 40:  # Always keep keys smaller than 40 char
            self.cached_keys.append(key)
    # 重载字典的获取值的方法，根据键获取值
    def __getitem__(self, key):
        # 获取键对应的值
        val = dict.get(self, key)
        # 如果值存在，则直接返回
        if val:  # Already loaded
            return val
        # 如果值为 None，则表示未知的键，抛出 KeyError 异常
        elif val is None:  # Unknown key
            raise KeyError(key)
        # 如果值为 False，则表示之前加载过但已从缓存中清除，重新加载并返回
        elif val is False:  # Loaded before, but purged from cache
            return self.loadItem(key)

    # 重载字典的设置值的方法，设置键值对
    def __setitem__(self, key, val):
        # 将键添加到缓存的键集合中
        self.addCachedKey(key)
        # 检查缓存大小是否超出限制
        self.checkLimit()
        # 获取值的大小
        size = self.getItemSize(key)
        # 将键值对存储到数据库中
        self.db.setContent(self.site, key, val, size)
        # 调用父类的设置值方法
        dict.__setitem__(self, key, val)

    # 重载字典的删除值的方法，根据键删除值
    def __delitem__(self, key):
        # 从数据库中删除键值对
        self.db.deleteContent(self.site, key)
        # 调用父类的删除值方法
        dict.__delitem__(self, key)
        # 尝试从缓存的键集合中移除键
        try:
            self.cached_keys.remove(key)
        except ValueError:
            pass

    # 返回一个迭代器，遍历字典的键值对
    def iteritems(self):
        for key in dict.keys(self):
            try:
                val = self[key]
            except Exception as err:
                self.log.warning("Error loading %s: %s" % (key, err))
                continue
            yield key, val

    # 返回字典的键值对列表
    def items(self):
        back = []
        for key in dict.keys(self):
            try:
                val = self[key]
            except Exception as err:
                self.log.warning("Error loading %s: %s" % (key, err))
                continue
            back.append((key, val))
        return back

    # 返回字典的值列表
    def values(self):
        back = []
        for key, val in dict.iteritems(self):
            # 如果值不存在，则重新加载
            if not val:
                try:
                    val = self.loadItem(key)
                except Exception:
                    continue
            back.append(val)
        return back

    # 根据键获取值，如果键不存在则返回默认值
    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
        except Exception as err:
            # 将出错的键记录到 bad_files 中
            self.site.bad_files[key] = self.site.bad_files.get(key, 1)
            # 从字典中删除出错的键值对
            dict.__delitem__(self, key)
            self.log.warning("Error loading %s: %s" % (key, err))
            return default
    # 定义一个方法，用于执行数据库查询操作
    def execute(self, query, params={}):
        # 将参数中的 site_id 设置为当前数据库的 ID
        params["site_id"] = self.db_id
        # 调用数据库对象的 execute 方法执行查询，并返回结果
        return self.db.execute(query, params)
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 导入 psutil 模块
    import psutil
    # 获取当前进程的进程号
    process = psutil.Process(os.getpid())
    # 获取当前进程的内存使用量，并转换为 MB
    s_mem = process.memory_info()[0] / float(2 ** 20)
    # 设置根目录路径
    root = "data-live/1MaiL5gfBM1cyb4a8e3iiL8L5gXmoAJu27"
    # 创建 ContentDbDict 对象，并传入参数
    contents = ContentDbDict("1MaiL5gfBM1cyb4a8e3iiL8L5gXmoAJu27", root)
    # 打印初始内容长度
    print("Init len", len(contents))

    # 记录开始时间
    s = time.time()
    # 遍历指定目录下的文件夹，最多遍历前 8000 个
    for dir_name in os.listdir(root + "/data/users/")[0:8000]:
        # 获取指定文件的内容
        contents["data/users/%s/content.json" % dir_name]
    # 打印加载时间
    print("Load: %.3fs" % (time.time() - s))

    # 记录开始时间
    s = time.time()
    # 初始化 found 变量
    found = 0
    # 遍历 contents 中的键值对
    for key, val in contents.items():
        # 增加 found 计数
        found += 1
        # 断言键值对的键不为空
        assert key
        # 断言键值对的值不为空
        assert val
    # 打印找到的键值对数量
    print("Found:", found)
    # 打印遍历键值对的时间
    print("Iteritem: %.3fs" % (time.time() - s))

    # 记录开始时间
    s = time.time()
    # 初始化 found 变量
    found = 0
    # 遍历 contents 的键的列表
    for key in list(contents.keys()):
        # 增加 found 计数
        found += 1
        # 断言键在 contents 中
        assert key in contents
    # 打印遍历键的时间
    print("In: %.3fs" % (time.time() - s))

    # 打印内容值的长度和键的长度
    print("Len:", len(list(contents.values())), len(list(contents.keys())))

    # 打印内存使用量的变化
    print("Mem: +", process.memory_info()[0] / float(2 ** 20) - s_mem)
```