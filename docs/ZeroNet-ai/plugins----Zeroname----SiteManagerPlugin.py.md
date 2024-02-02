# `ZeroNet\plugins\Zeroname\SiteManagerPlugin.py`

```py
# 导入 logging 模块
import logging
# 导入 re 模块
import re
# 导入 time 模块
import time

# 从 Config 模块中导入 config 对象
from Config import config
# 从 Plugin 模块中导入 PluginManager 对象
from Plugin import PluginManager

# 设置变量 allow_reload 为 False，表示不支持重新加载
allow_reload = False

# 获取名为 "ZeronamePlugin" 的 logger 对象
log = logging.getLogger("ZeronamePlugin")

# 将 SiteManagerPlugin 类注册到 PluginManager 的 SiteManager 中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # 初始化类属性
    site_zeroname = None
    db_domains = {}
    db_domains_modified = None

    # 加载方法
    def load(self, *args, **kwargs):
        # 调用父类的 load 方法
        super(SiteManagerPlugin, self).load(*args, **kwargs)
        # 如果没有获取到 config.bit_resolver，则需要获取
        if not self.get(config.bit_resolver):
            self.need(config.bit_resolver)  # 需要 ZeroName 站点

    # 判断地址是否为 .bit 域名，返回 True 或 False
    def isBitDomain(self, address):
        return re.match(r"(.*?)([A-Za-z0-9_-]+\.bit)$", address)

    # 解析 .bit 域名，返回地址或 None
    def resolveBitDomain(self, domain):
        domain = domain.lower()
        # 如果 site_zeroname 为空，则获取 config.bit_resolver
        if not self.site_zeroname:
            self.site_zeroname = self.need(config.bit_resolver)

        # 获取 site_zeroname 的 content.json 文件的修改时间
        site_zeroname_modified = self.site_zeroname.content_manager.contents.get("content.json", {}).get("modified", 0)
        # 如果 db_domains 为空或者 db_domains_modified 不等于 site_zeroname_modified
        if not self.db_domains or self.db_domains_modified != site_zeroname_modified:
            # 需要获取 site_zeroname 的 data/names.json 文件，优先级为 10
            self.site_zeroname.needFile("data/names.json", priority=10)
            s = time.time()
            try:
                # 尝试加载 data/names.json 文件到 db_domains
                self.db_domains = self.site_zeroname.storage.loadJson("data/names.json")
            except Exception as err:
                log.error("Error loading names.json: %s" % err)

            # 记录日志，显示加载了多少条记录，以及加载时间和修改时间
            log.debug(
                "Domain db with %s entries loaded in %.3fs (modification: %s -> %s)" %
                (len(self.db_domains), time.time() - s, self.db_domains_modified, site_zeroname_modified)
            )
            # 更新 db_domains_modified 为 site_zeroname_modified
            self.db_domains_modified = site_zeroname_modified
        # 返回 domain 对应的地址
        return self.db_domains.get(domain)

    # 将域名解析为地址
    def resolveDomain(self, domain):
        # 如果 resolveBitDomain 返回 None，则调用父类的 resolveDomain 方法
        return self.resolveBitDomain(domain) or super(SiteManagerPlugin, self).resolveDomain(domain)

    # 判断地址是否为域名
    # 检查地址是否为域名，调用 isBitDomain 方法和父类的 isDomain 方法，只要其中一个返回 True 就返回 True
    def isDomain(self, address):
        return self.isBitDomain(address) or super(SiteManagerPlugin, self).isDomain(address)
# 将 ConfigPlugin 类注册到 PluginManager 中的 ConfigPlugin 插件
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 添加参数组 "Zeroname plugin"
        group = self.parser.add_argument_group("Zeroname plugin")
        # 添加参数 "--bit_resolver"，用于解析 .bit 域名的 ZeroNet 站点
        group.add_argument(
            "--bit_resolver", help="ZeroNet site to resolve .bit domains",
            default="1Name2NXVi1RDPDgf5617UoW7xA6YrhM9F", metavar="address"
        )
        # 调用父类的 createArguments 方法
        return super(ConfigPlugin, self).createArguments()
```