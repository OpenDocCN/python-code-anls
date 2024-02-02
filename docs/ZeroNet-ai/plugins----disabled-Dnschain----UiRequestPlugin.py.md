# `ZeroNet\plugins\disabled-Dnschain\UiRequestPlugin.py`

```py
# 导入正则表达式模块
import re
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 初始化方法，接受一个 server 参数，默认为 None
    def __init__(self, server = None):
        # 从 Site 模块中导入 SiteManager 类
        from Site import SiteManager
        # 实例化 SiteManager 类，赋值给 self.site_manager
        self.site_manager = SiteManager.site_manager
        # 调用父类的初始化方法，传入 server 参数
        super(UiRequestPlugin, self).__init__(server)

    # 媒体请求方法
    def actionSiteMedia(self, path):
        # 使用正则表达式匹配路径中的媒体地址和内部路径
        match = re.match(r"/media/(?P<address>[A-Za-z0-9-]+\.[A-Za-z0-9\.-]+)(?P<inner_path>/.*|$)", path)
        # 如果匹配成功
        if match:
            # 获取域名
            domain = match.group("address")
            # 解析域名，获取地址
            address = self.site_manager.resolveDomain(domain)
            # 如果地址存在
            if address:
                # 重新构建路径
                path = "/media/"+address+match.group("inner_path")
        # 调用父类的 actionSiteMedia 方法，传入路径参数，获取包装框架输出
        return super(UiRequestPlugin, self).actionSiteMedia(path)

    # 检查媒体请求是否允许
    def isMediaRequestAllowed(self, site_address, referer):
        # 从引用中提取路径
        referer_path = re.sub("http[s]{0,1}://.*?/", "/", referer).replace("/media", "") # Remove site address
        # 从引用路径中提取站点地址
        referer_site_address = re.match(r"/(?P<address>[A-Za-z0-9\.-]+)(?P<inner_path>/.*|$)", referer_path).group("address")

        # 如果引用站点地址等于站点地址
        if referer_site_address == site_address:
            # 返回 True
            return True
        # 如果引用站点地址解析后等于站点地址
        elif self.site_manager.resolveDomain(referer_site_address) == site_address:
            # 返回 True
            return True
        # 其他情况，返回 False
        else:
            return False
```