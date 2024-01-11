# `ZeroNet\plugins\ContentFilter\ContentFilterStorage.py`

```
import os  # 导入操作系统模块
import json  # 导入 JSON 模块
import logging  # 导入日志模块
import collections  # 导入 collections 模块
import time  # 导入时间模块
import hashlib  # 导入哈希模块

from Debug import Debug  # 从 Debug 模块导入 Debug 类
from Plugin import PluginManager  # 从 Plugin 模块导入 PluginManager 类
from Config import config  # 从 Config 模块导入 config 变量
from util import helper  # 从 util 模块导入 helper 函数


class ContentFilterStorage(object):
    def __init__(self, site_manager):
        self.log = logging.getLogger("ContentFilterStorage")  # 获取名为 "ContentFilterStorage" 的日志记录器
        self.file_path = "%s/filters.json" % config.data_dir  # 设置文件路径为 config.data_dir 下的 filters.json
        self.site_manager = site_manager  # 初始化 site_manager
        self.file_content = self.load()  # 调用 load 方法加载文件内容

        # Set default values for filters.json
        if not self.file_content:  # 如果文件内容为空
            self.file_content = {}  # 设置文件内容为空字典

        # Site blacklist renamed to site blocks
        if "site_blacklist" in self.file_content:  # 如果文件内容中包含 "site_blacklist"
            self.file_content["siteblocks"] = self.file_content["site_blacklist"]  # 将 "site_blacklist" 重命名为 "siteblocks"
            del self.file_content["site_blacklist"]  # 删除 "site_blacklist" 键

        for key in ["mutes", "siteblocks", "includes"]:  # 遍历列表中的键
            if key not in self.file_content:  # 如果键不在文件内容中
                self.file_content[key] = {}  # 将键添加到文件内容中，并设置为空字典

        self.include_filters = collections.defaultdict(set)  # 创建默认值为集合的 include_filters 字典，用于存储所有包含的静音和黑名单列表
        self.includeUpdateAll(update_site_dbs=False)  # 调用 includeUpdateAll 方法，更新所有包含的站点数据库

    def load(self):
        # Rename previously used mutes.json -> filters.json
        if os.path.isfile("%s/mutes.json" % config.data_dir):  # 如果 mutes.json 文件存在
            self.log.info("Renaming mutes.json to filters.json...")  # 记录日志信息
            os.rename("%s/mutes.json" % config.data_dir, self.file_path)  # 将 mutes.json 重命名为 filters.json
        if os.path.isfile(self.file_path):  # 如果 filters.json 文件存在
            try:
                return json.load(open(self.file_path))  # 加载并返回文件内容
            except Exception as err:  # 捕获异常
                self.log.error("Error loading filters.json: %s" % err)  # 记录错误日志信息
                return None  # 返回空
        else:
            return None  # 返回空
    # 定义一个方法，用于更新所有包含的内容，并可以选择是否更新站点数据库
    def includeUpdateAll(self, update_site_dbs=True):
        # 记录当前时间
        s = time.time()
        # 创建一个默认值为集合的字典，用于存储新的包含过滤器
        new_include_filters = collections.defaultdict(set)

        # 将所有包含文件的数据加载到一个合并的集合中
        for include_path in self.file_content["includes"]:
            # 将包含路径分割成地址和内部路径
            address, inner_path = include_path.split("/", 1)
            try:
                # 从站点管理器中获取地址对应的内容，并加载指定内部路径的 JSON 数据
                content = self.site_manager.get(address).storage.loadJson(inner_path)
            except Exception as err:
                # 如果加载出错，则记录警告信息并继续下一个包含文件的处理
                self.log.warning(
                    "Error loading include %s: %s" %
                    (include_path, Debug.formatException(err))
                )
                continue

            # 遍历内容字典，将值为字典类型的项合并到新的包含过滤器字典中
            for key, val in content.items():
                if type(val) is not dict:
                    continue
                new_include_filters[key].update(val.keys())

        # 计算新添加的静音项和移除的静音项
        mutes_added = new_include_filters["mutes"].difference(self.include_filters["mutes"])
        mutes_removed = self.include_filters["mutes"].difference(new_include_filters["mutes"])

        # 更新当前的包含过滤器为新的包含过滤器
        self.include_filters = new_include_filters

        # 如果需要更新站点数据库
        if update_site_dbs:
            # 对于每个新添加的静音项，将其从数据库中移除
            for auth_address in mutes_added:
                self.changeDbs(auth_address, "remove")

            # 对于每个移除的静音项，如果其不在静音列表中，则加载到数据库中
            for auth_address in mutes_removed:
                if not self.isMuted(auth_address):
                    self.changeDbs(auth_address, "load")

        # 计算静音项和被屏蔽站点的数量，并记录调试信息
        num_mutes = len(self.include_filters["mutes"])
        num_siteblocks = len(self.include_filters["siteblocks"])
        self.log.debug(
            "Loaded %s mutes, %s blocked sites from %s includes in %.3fs" %
            (num_mutes, num_siteblocks, len(self.file_content["includes"]), time.time() - s)
        )
    # 添加一个包含文件的地址和内部路径到文件内容的 includes 字典中
    def includeAdd(self, address, inner_path, description=None):
        self.file_content["includes"]["%s/%s" % (address, inner_path)] = {
            "date_added": time.time(),
            "address": address,
            "description": description,
            "inner_path": inner_path
        }
        # 更新所有包含文件
        self.includeUpdateAll()
        # 保存文件内容
        self.save()
    
    # 从文件内容的 includes 字典中移除指定地址和内部路径的包含文件
    def includeRemove(self, address, inner_path):
        del self.file_content["includes"]["%s/%s" % (address, inner_path)]
        # 更新所有包含文件
        self.includeUpdateAll()
        # 保存文件内容
        self.save()
    
    # 保存文件内容到文件中
    def save(self):
        s = time.time()
        # 使用原子写入方式将文件内容以 JSON 格式写入文件
        helper.atomicWrite(self.file_path, json.dumps(self.file_content, indent=2, sort_keys=True).encode("utf8"))
        # 记录保存操作所花费的时间
        self.log.debug("Saved in %.3fs" % (time.time() - s))
    
    # 检查指定的地址是否被静音
    def isMuted(self, auth_address):
        if auth_address in self.file_content["mutes"] or auth_address in self.include_filters["mutes"]:
            return True
        else:
            return False
    
    # 获取经过哈希处理的站点地址
    def getSiteAddressHashed(self, address):
        return "0x" + hashlib.sha256(address.encode("ascii")).hexdigest()
    
    # 检查指定的地址是否被阻止访问
    def isSiteblocked(self, address):
        if address in self.file_content["siteblocks"] or address in self.include_filters["siteblocks"]:
            return True
        return False
    # 获取指定地址的站点块详情
    def getSiteblockDetails(self, address):
        # 从文件内容中获取指定地址的站点块详情
        details = self.file_content["siteblocks"].get(address)
        # 如果没有找到指定地址的站点块详情
        if not details:
            # 计算地址的 SHA256 哈希值
            address_sha256 = self.getSiteAddressHashed(address)
            # 从文件内容中获取计算得到的地址的站点块详情
            details = self.file_content["siteblocks"].get(address_sha256)

        # 如果还是没有找到站点块详情
        if not details:
            # 获取所有包含的地址
            includes = self.file_content.get("includes", {}).values()
            # 遍历每个包含的地址
            for include in includes:
                # 获取包含地址对应的站点
                include_site = self.site_manager.get(include["address"])
                # 如果没有找到包含的站点
                if not include_site:
                    # 继续下一个循环
                    continue
                # 加载包含站点的指定路径的 JSON 内容
                content = include_site.storage.loadJson(include["inner_path"])
                # 获取指定地址的站点块详情
                details = content.get("siteblocks", {}).get(address)
                # 如果找到了站点块详情
                if details:
                    # 将包含信息添加到站点块详情中
                    details["include"] = include
                    # 退出循环
                    break

        # 返回站点块详情
        return details

    # 搜索并删除或重新添加用户的文件
    def changeDbs(self, auth_address, action):
        # 记录调试信息
        self.log.debug("Mute action %s on user %s" % (action, auth_address))
        # 执行 SQL 查询，获取符合条件的内容
        res = list(self.site_manager.list().values())[0].content_manager.contents.db.execute(
            "SELECT * FROM content LEFT JOIN site USING (site_id) WHERE inner_path LIKE :inner_path",
            {"inner_path": "%%/%s/%%" % auth_address}
        )
        # 遍历查询结果
        for row in res:
            # 获取地址对应的站点
            site = self.site_manager.sites.get(row["address"])
            # 如果没有找到站点
            if not site:
                # 继续下一个循环
                continue
            # 获取文件所在目录的路径
            dir_inner_path = helper.getDirname(row["inner_path"])
            # 遍历文件所在目录下的所有文件
            for file_name in site.storage.walk(dir_inner_path):
                # 如果执行的操作是删除
                if action == "remove":
                    # 标记文件为已删除
                    site.storage.onUpdated(dir_inner_path + file_name, False)
                else:
                    # 标记文件为已更新
                    site.storage.onUpdated(dir_inner_path + file_name)
                # 标记文件处理完成
                site.onFileDone(dir_inner_path + file_name)
```